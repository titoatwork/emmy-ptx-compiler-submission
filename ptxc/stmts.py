"""Statement lowering: the walk, control flow, cross-thread combines, async copies."""

from __future__ import annotations

from emmy.compiler.ir.expr import Literal, Var

from .common import *
from .ops import INT_BINARY, elem_bytes, norm_dtype, ptx_type, reg_class


def _single_trip(st, step_val) -> bool:
    """Whether the strided loop runs at most once for every start >= 0.

    The copy loops of a staged matmul start at a per-thread coordinate and
    stride the whole tile, so when the bound is within one step every thread
    copies once or not at all.  Emitting that as a guarded body drops the
    increment and the back edge without changing which threads run it.
    """
    if step_val is None or step_val <= 0:
        return False
    start_val = _const_int(st.start)
    if start_val is not None and start_val < 0:
        return False
    end_val = _const_int(st.end if st.end is not None else st.axis.extent)
    if end_val is None:
        return False
    return 0 <= end_val <= step_val



def _lane_fanout(stmts) -> int:
    """How many async copies share the most common lane index expression."""
    lanes: dict[str, int] = {}
    for st in stmts:
        if stmt_name(st) == "CpAsyncCopy":
            lane = getattr(st, "lane_index", None)
            if lane is not None:
                key = repr(tuple(lane))
                lanes[key] = lanes.get(key, 0) + 1
    return max(lanes.values(), default=0)


class StmtMixin:
    def walk(self, stmts) -> None:
        self.load_cse = {}
        # Addresses are only reusable inside one basic block: a nested loop
        # or branch can change the registers the runtime part reads.
        self.addr_cse = {}
        # Folding a lane row/column pair is an improvement when only a couple of
        # copies share it.  With four or more copies ptxas already
        # common-subexpresses the divide and remainder, and the shorter
        # form costs scheduling quality instead.
        saved_lane = getattr(self, "recombine_lane", False)
        self.recombine_lane = _lane_fanout(stmts) <= 2
        try:
            for st in stmts:
                self.stmt(st)
        finally:
            self.recombine_lane = saved_lane


    def stmt(self, st) -> None:
        name = stmt_name(st)
        if getattr(self, "deferred_mode", None) == "load" and name != "Load":
            return
        handler = getattr(self, "stmt_" + name, None)
        if handler is None:
            raise CompileError(f"unsupported statement: {name}")
        self.icache = {}
        handler(st)


    def stmt_Load(self, st) -> None:
        plan = getattr(self, "hoist_plan", None)
        if plan is not None and id(st) in plan:
            copy = getattr(self, "copy_index", None)
            if copy is None or copy >= len(plan[id(st)]):
                raise CompileError("hoisted load used outside its unrolled loop")
            for name, value in plan[id(st)][copy]:
                self.values[name] = value
            return
        mode = getattr(self, "deferred_mode", None)
        if mode == "use":
            for name, value in self.deferred[id(st)]:
                self.values[name] = value
            return
        pairs = getattr(self, "load_pairs", None) or {}
        pair = pairs.get(id(st))
        index = getattr(self, "copy_index", 0)
        if mode == "load":
            if pair is not None and pair[0] == "same_run":
                others = pair[1]
                regs = self._emit_load_impl(st, extras=[o.names[0] for o in others])
                self._deferred_current[id(st)] = [(st.names[0], (regs[0], "f32"))] + [
                    (o.names[0], (regs[j + 1], "f32")) for j, o in enumerate(others)]
                for j, o in enumerate(others):
                    self._deferred_current[id(o)] = [(o.names[0], (regs[j + 1], "f32"))]
                return
            if pair is not None and pair[0] == "same_rest":
                return
            if pair is not None and pair[0] == "same_first":
                regs = self._emit_load_impl(st, extra=pair[1].names[0])
                self._deferred_current[id(st)] = [(st.names[0], (regs[0], "f32"))]
                self._deferred_current[id(pair[1])] = [(pair[1].names[0], (regs[1], "f32"))]
                return
            if pair is not None and pair[0] == "same_second":
                return
            if pair is not None and pair[0] == "cross":
                width = pair[1]
                if index % width == 0:
                    rest = self._emit_cross_load(st, width)
                    for slot, bindings in enumerate(rest):
                        self._pair_queue[slot][id(st)] = bindings
                    self._deferred_current[id(st)] = [(st.names[0], self.values[st.names[0]])]
                return
            self._emit_load_impl(st)
            self._deferred_current[id(st)] = [(n, self.values[n]) for n in st.names]
            return
        if pair is not None and pair[0] == "same_run":
            self._emit_load_impl(st, extras=[o.names[0] for o in pair[1]])
            return
        if pair is not None and pair[0] == "same_rest":
            return
        if pair is not None and pair[0] == "same_first":
            self._emit_load_impl(st, extra=pair[1].names[0])
            return
        if pair is not None and pair[0] == "same_second":
            return
        if pair is not None and pair[0] == "cross":
            width = pair[1]
            if index % width == 0:
                rest = self._emit_cross_load(st, width)
                for slot, bindings in enumerate(rest):
                    self._pair_queue[slot][id(st)] = bindings
            return
        key = self._load_key(st)
        if key is not None:
            hit = self.load_cse.get(key)
            if hit is not None:
                for name, value in zip(st.names, hit):
                    self.values[name] = value
                return
        self._emit_load_impl(st)
        if key is not None:
            self.load_cse[key] = [self.values[n] for n in st.names]


    def _load_key(self, st):
        if len(st.names) > 4:
            return None
        dt = norm_dtype(st.dtype) if st.dtype else self.buffer_dtype(st.input)
        if dt not in FLOAT_KINDS and dt not in ("f16", "f64", "f32", "bool"):
            return None
        return (st.input, str(tuple(st.index)), len(st.names))


    def _apply_pending(self) -> None:
        pending = self._pair_queue.pop(0)
        self._pair_queue.append({})
        for bindings in pending.values():
            for name, value in bindings:
                self.values[name] = value


    def _vector_load_regs(self, addr, width: int, dt: str, esize: int, shared: bool) -> list[str]:
        prefix = "ld.shared" if shared else getattr(self, "_global_load_prefix", "ld.global")
        regs = [self.em.f() for _ in range(width)]
        self.em.emit(f"{prefix}.v{width}.f32 {{{', '.join(regs)}}}, [{addr}];")
        return regs


    def _cached_view(self, st):
        info = self.cache_info.get(st.input)
        if info is None or st.index is None or len(st.index) != 2:
            return None
        from dataclasses import replace
        index = (st.index[1], st.index[0]) if info["swap"] else tuple(st.index)
        return replace(st, input=info["name"], index=index)


    def _resolve_load_addr(self, st):
        info = self.cache_info.get(st.input)
        if info is not None and info.get("slice"):
            dt = norm_dtype(st.dtype) if st.dtype else self.buffer_dtype(st.input)
            esize = elem_bytes(dt)
            cached = self._chunk_address(info["name"], st, st.index)
            if cached is not None:
                return cached, dt, esize, True
            flat = self.flat_index(st.index, self.shape_of(st.input))
            block = info.get("block_base")
            if block is not None:
                flat = self._s32_bin("sub.s32", flat, block)
            addr = self._shared_addr(self._smem_base(info["name"]), flat, esize)
            self._remember_chunk_address(info["name"], st.index, addr)
            return addr, dt, esize, True
        cached = self._cached_view(st)
        if cached is not None:
            st = cached
        src = st.input
        self._global_load_prefix = (
            "ld.global.nc" if not self._is_written(src) else "ld.global"
        )
        cached = self._chunk_address(src, st, st.index)
        if cached is not None:
            dt = norm_dtype(st.dtype) if st.dtype else self.buffer_dtype(src)
            return cached, dt, elem_bytes(dt), src in self.smem
        if src in self.scalars:
            return None
        dt = norm_dtype(st.dtype) if st.dtype else self.buffer_dtype(src)
        _check_dtype(dt, f"Load into {st.names}")
        esize = elem_bytes(dt)
        shape = self.shape_of(src)
        if src in self.smem:
            flat = self.flat_index(st.index, shape)
            base = self._smem_base(src)
            addr = self._shared_addr(base, flat, esize)
            self._remember_chunk_address(src, st.index, addr)
            return addr, dt, esize, True
        ptr = self.params.get(src)
        if ptr is None:
            raise CompileError(f"load from unknown buffer {src!r}")
        addr = self._cse_global_addr(src, ptr, st.index, shape, esize)
        self._remember_chunk_address(src, st.index, addr)
        return addr, dt, esize, False


    def _emit_load_impl(self, st, extra=None, extras=()):
        names = list(st.names) + ([extra] if extra is not None else []) + list(extras)
        resolved = self._resolve_load_addr(st)
        if resolved is None:
            reg, kind = self.scalars[st.input]
            for out in names:
                self.values[out] = (reg, kind)
            return None
        addr, dt, esize, shared = resolved
        if extra is not None or extras:
            if len(names) not in (2, 4):
                raise CompileError(f"unsupported fused load width {len(names)}")
            regs = self._vector_load_regs(addr, len(names), dt, esize, shared)
            for name, reg in zip(names, regs):
                self.values[name] = (reg, "f32")
            return regs
        self._load_consecutive(st.names, addr, dt, esize, len(st.names),
                               shared=shared, shape=self.shape_of(st.input),
                               index=st.index)
        return None


    def _emit_cross_load(self, st, width: int):
        resolved = self._resolve_load_addr(st)
        if resolved is None:
            raise CompileError("cannot vectorize a scalar buffer load")
        addr, dt, esize, shared = resolved
        regs = self._vector_load_regs(addr, width, dt, esize, shared)
        if len(st.names) != 1:
            raise CompileError("cross-copy vector load expects one name")
        self.values[st.names[0]] = (regs[0], "f32")
        return [[(st.names[0], (regs[i], "f32"))] for i in range(1, width)]


    def _plan_pairs(self, st, stride: int, n: int) -> dict:
        if not self._pair_safe(st.body):
            return {}
        body = st.body
        loads = [s for s in body
                 if stmt_name(s) == "Load" and len(s.names) == 1]
        plan: dict = {}
        loop_start = 0
        if stmt_name(st) == "StridedLoop":
            start = _const_int(getattr(st, "start", None))
            if start is None or start % 2:
                return plan
            loop_start = start
        axis = st.axis.name
        for i, a in enumerate(loads):
            if id(a) in plan:
                continue
            src = a.input
            if src in self.scalars:
                continue
            dt = norm_dtype(a.dtype) if a.dtype else self.buffer_dtype(src)
            if dt != "f32":
                continue
            info = self.cache_info.get(src)
            if info is not None and info.get("slice") and a.index is not None:
                shape = self.shape_of(src)
                index = a.index
                width_cap = 4
            elif info is not None and a.index is not None and len(a.index) == 2:
                shape = tuple(info["shape"])
                index = (a.index[1], a.index[0]) if info["swap"] else tuple(a.index)
                width_cap = 4
            else:
                shape = self.shape_of(src)
                index = a.index
                width_cap = 2
            esize = elem_bytes(dt)

            def view(load):
                info_b = self.cache_info.get(load.input)
                if info_b is not None and not info_b.get("slice") and load.index is not None \
                        and len(load.index) == 2:
                    return (load.index[1], load.index[0]) if info_b["swap"] else tuple(load.index)
                return load.index

            run = []
            previous = index
            for b in loads[i + 1:]:
                if b.input != src or id(b) in plan:
                    continue
                bindex = view(b)
                if not self._adjacent_index(previous, bindex):
                    continue
                run.append(b)
                previous = bindex
                if len(run) == 3:
                    break
            width = 4 if len(run) >= 3 else (2 if run else 0)
            if width == 4 and src in self.smem and \
                    int(self.smem[src].get("align", 0)) < width * esize:
                width = 2
            if width and self._aligned_tolerant(index, shape, width):
                plan[id(a)] = ("same_run", tuple(run[:width - 1]))
                for other in run[:width - 1]:
                    plan[id(other)] = ("same_rest", None)
                continue
            coef = self._axis_coef_tolerant(index, shape, axis)
            if coef is None or coef * stride != 1:
                continue
            width = 0
            for candidate in (4, 2):
                if candidate > width_cap:
                    break
                if n % candidate or loop_start % candidate:
                    continue
                if not self._cross_aligned(index, shape, esize, candidate, axis):
                    continue
                width = candidate
                break
            if width:
                plan[id(a)] = ("cross", width)
        return plan


    def _pair_safe(self, body) -> bool:
        """A pair may be fused only when no top-level statement can change memory.

        The planner looks at the loads of one body in order.  A store, barrier,
        shared-memory tree or async copy between two adjacent loads means the
        fused read would observe a different value than the original order.
        """
        unsafe = {"Write", "Sync", "TreeHalve", "CpAsyncCopy", "ZeroPrologue"}
        return not any(stmt_name(s) in unsafe for s in body)


    def _adjacent_index(self, first, second) -> bool:
        if first is None or second is None or len(first) != len(second):
            return False
        fa = [_affine(c) for c in first]
        fb = [_affine(c) for c in second]
        def strip(aff):
            return (aff[0], {k: v for k, v in aff[1].items() if v})

        for i in range(len(first) - 1):
            if fa[i] is None or fb[i] is None:
                if str(first[i]) != str(second[i]):
                    return False
            elif strip(fa[i]) != strip(fb[i]):
                return False
        if fa[-1] is None or fb[-1] is None:
            # The last coordinate is not affine (it carries the tile's div/mod),
            # so adjacency cannot be proven; leave the loads scalar.
            return False
        ca, coa = fa[-1]
        cb, cob = fb[-1]
        return strip((0, coa)) == strip((0, cob)) and cb == ca + 1


    def _cross_aligned(self, index, shape, esize, width, axis) -> bool:
        coords = []
        found = False
        for coord in index:
            if isinstance(coord, Var) and coord.name == axis:
                coords.append(Literal(0))
                found = True
            else:
                coords.append(coord)
        if not found:
            return False
        return self._aligned_tolerant(tuple(coords), shape, width)


    def _axis_coef_tolerant(self, index, shape, axis: str):
        if index is None or len(index) != len(shape):
            return None
        strides = _strides(shape, len(index))
        total = 0
        for coord, stride in zip(index, strides):
            if axis not in coord.free_vars():
                continue
            coef = _axis_coefficient(coord, axis)
            if coef is None:
                return None
            total += coef * stride
        return total


    def _aligned_tolerant(self, index, shape, need: int) -> bool:
        """Divisibility by `need` in elements, allowing opaque strides.

        Address expressions in a staged matmul contain integer division and
        modulo (the stage rotation), so _affine gives up.  A non-affine term
        multiplied by a stride that is itself a multiple of `need` is still
        provably aligned; that is enough to vectorize the inner loop.
        """
        if index is None or len(index) != len(shape):
            return False
        strides = _strides(shape, len(index))
        const = 0
        for coord, stride in zip(index, strides):
            aff = _affine(coord)
            if aff is None:
                if stride % need:
                    return False
                continue
            c, coeffs = aff
            const += c * stride
            for v in coeffs.values():
                if (v * stride) % need:
                    return False
        return const % need == 0


    def stmt_Write(self, st) -> None:
        self.load_cse = {}
        dst = st.output
        cached = self._chunk_address(dst, st, st.index)
        # The destination buffer decides the store type.  ``value_dtype`` is
        # the type of the value being stored and may differ (a half max is
        # written into a float scratch slot in the f16 softmax schedule).
        dt = norm_dtype(self.buffer_dtype(dst))
        _check_dtype(dt, f"Write to {dst!r}")
        esize = elem_bytes(dt)
        shape = self.shape_of(dst)
        missing = [name for name in st.values if name not in self.values]
        if missing:
            raise CompileError(
                f"write to {dst!r} uses undefined value(s): {', '.join(missing)}")
        values = [self._cast_reg(self.values[v][0], self.values[v][1], dt)
                  for v in st.values]

        if cached is not None:
            addr = cached
            if dst in self.smem:
                self._store_values(addr, values, dt, esize, len(values), True, shape, st.index)
            elif st.atomic:
                cur = addr
                for i, reg in enumerate(values):
                    if i:
                        cur = self._next_addr(cur, esize)
                    self.em.emit(f"red.global.add.{ptx_type(dt)} [{cur}], {reg};")
            else:
                self._store_values(addr, values, dt, esize, len(values), False, shape, st.index)
            return
        if dst in self.smem:
            flat = self.flat_index(st.index, shape)
            if getattr(st, "swizzle", "NONE") != "NONE":
                flat = self._swizzle(flat, st.swizzle)
            base = self._smem_base(dst)
            addr = self._shared_addr(base, flat, esize)
            self._store_values(addr, values, dt, esize, len(values), True,
                               shape, st.index)
            return
        ptr = self.params.get(dst)
        if ptr is None:
            raise CompileError(f"write to unknown buffer {dst!r}")
        addr = self._cse_global_addr(dst, ptr, st.index, shape, esize)
        self._remember_chunk_address(dst, st.index, addr)
        if st.atomic:
            cur = addr
            for i, reg in enumerate(values):
                if i:
                    cur = self._next_addr(cur, esize)
                self.em.emit(f"red.global.add.{ptx_type(dt)} [{cur}], {reg};")
            return
        self._store_values(addr, values, dt, esize, len(values), False,
                           shape, st.index)


    def stmt_Let(self, st) -> None:
        if st.name in getattr(self, "fused_lets", ()):
            return
        dt = norm_dtype(st.dtype)
        _check_dtype(dt, f"Let {st.name!r}")
        value = st.value
        if dt in FLOAT_KINDS:
            reg = self.fexpr(value)
            self.values[st.name] = (reg, "f32")
        else:
            reg = self.iexpr(value)
            self.values[st.name] = (reg, "i64")


    def stmt_Init(self, st) -> None:
        dt = norm_dtype(st.dtype)
        _check_dtype(dt, f"Init {st.name!r}")
        self.values[st.name] = self._seed_value(dt, st.identity)


    def stmt_Assign(self, st) -> None:
        if st.name in getattr(self, "fused_lets", ()):
            return
        opname = _op_name(st.op)
        dt = norm_dtype(st.dtype) if st.dtype else self._infer_kind(st.args)
        _check_dtype(dt, f"Assign {st.name!r}")
        args = list(st.args)

        if dt == "bool" or opname in COMPARE_OPS or opname in BOOL_BINARY:
            self.values[st.name] = (self._bool_assign(opname, args), "pred")
            return
        if opname == "where":
            self._assign_where(st, args)
            return
        if dt == "f32" and opname in ("multiply", "add", "subtract"):
            self.defs[st.name] = (opname, tuple(args))

        if dt == "f16":
            f32_args = [self.fexpr_operand(a) for a in args]
            tmp = self.em.f()
            self._emit_float_op(opname, f32_args, tmp, "f32")
            dst = self.em.h()
            self.em.emit(f"cvt.rn.f16.f32 {dst}, {tmp};")
            self.values[st.name] = (dst, "f16")
        elif dt == "f64":
            f64_args = [self.f64_operand(a) for a in args]
            dst = self.em.d()
            self._emit_float_op(opname, f64_args, dst, "f64")
            self.values[st.name] = (dst, "f64")
        elif dt in FLOAT_KINDS:
            regs = [self.fexpr_operand(a) for a in args]
            dst = self.em.f()
            self._emit_float_op(opname, regs, dst, dt)
            self.values[st.name] = (dst, "f32")
        elif opname in ("copy", "pad", "bitcast"):
            src, kind = self.value(args[0])
            dst = self.em.reg(reg_class(kind))
            self.em.emit(f"mov.{ptx_type(kind)} {dst}, {src};")
            self.values[st.name] = (dst, kind)
        else:
            regs = [self.iexpr_operand(a) for a in args]
            if opname in INT_BINARY:
                if opname in ("divide", "floor_divide"):
                    dst = self._div(regs[0], regs[1])
                elif opname in ("remainder", "mod"):
                    dst = self._rem(regs[0], regs[1])
                else:
                    dst = self.em.rd()
                    self.em.emit(f"{INT_BINARY[opname]} {dst}, {regs[0]}, {regs[1]};")
            else:
                raise CompileError(f"unsupported integer op {opname!r}")
            self.values[st.name] = (dst, "i64")


    def stmt_Accum(self, st) -> None:
        """Fold into a loop-carried register.

        The loop body is emitted once, so the accumulator must stay in the
        same register across iterations: the fold writes back in place.
        """
        opname = _op_name(st.op)
        dt = norm_dtype(getattr(st, "dtype", None) or "f32")
        _check_dtype(dt, f"Accum {st.name!r}")
        if dt not in FLOAT_KINDS and opname not in INT_BINARY:
            raise CompileError(
                f"unsupported integer op {opname!r} in Accum {st.name!r}")
        if st.name not in self.values:
            self._seed_accumulator(st)
        if dt == "f32" and self._try_fused_accumulate(st, opname):
            return
        if dt in FLOAT_KINDS and dt != "f32":
            # The accumulator register keeps the dtype Emmy declared.  Half
            # precision is computed in f32 and converted back into the same
            # half register; a double accumulator stays in f64.
            acc, kind = self.values[st.name]
            if kind == "f64":
                left = self.f64_operand(st.base) if st.base is not None else acc
                right = self.f64_operand(st.value)
                self._emit_float_op(opname, [left, right], acc, "f64")
            elif kind == "f16":
                left = self.fexpr_operand(st.base) if st.base is not None else self.fexpr_operand(st.name)
                right = self.fexpr_operand(st.value)
                tmp = self.em.f()
                self._emit_float_op(opname, [left, right], tmp, "f32")
                self.em.emit(f"cvt.rn.f16.f32 {acc}, {tmp};")
            else:
                left = self.fexpr_operand(st.base) if st.base is not None else acc
                right = self.fexpr_operand(st.value)
                self._emit_float_op(opname, [left, right], acc, "f32")
            self.values[st.name] = (acc, kind)
        elif dt in FLOAT_KINDS:
            acc, kind = self.values[st.name]
            left = self.fexpr_operand(st.base) if st.base is not None else acc
            right = self.fexpr_operand(st.value)
            self._emit_float_op(opname, [left, right], acc, dt)
            self.values[st.name] = (acc, "f32")
        else:
            acc, kind = self.values[st.name]
            left = self.iexpr_operand(st.base) if st.base is not None else acc
            right = self.iexpr_operand(st.value)
            self.em.emit(f"{INT_BINARY[opname]} {acc}, {left}, {right};")
            self.values[st.name] = (acc, "i64")


    def _try_fused_accumulate(self, st, opname: str) -> bool:
        """Fold ``acc += a * b`` as one IEEE FMA when the product has no other reader.

        The multiply statement still exists (and ptxas removes it as dead
        code), so this is a pure improvement over a separate mul + add, and it keeps
        ``rtol=atol=0`` pointwise chains exact because they never use Accum.
        """
        if opname != "add" or st.base is not None:
            return False
        prod = self.defs.get(st.value)
        if not prod or prod[0] != "multiply" or len(prod[1]) != 2:
            return False
        if self.uses.get(st.value, 0) != 1:
            return False
        acc = self.values[st.name][0]
        a = self.fexpr_operand(prod[1][0])
        b = self.fexpr_operand(prod[1][1])
        self.em.emit(f"fma.rn.f32 {acc}, {a}, {b}, {acc};")
        return True


    def stmt_Cond(self, st) -> None:
        saved_ctx = self.unroll_ctx
        self.unroll_ctx = None
        self.load_cse = {}
        self.divergent += 1
        try:
            self._emit_cond_inner(st)
        finally:
            self.divergent -= 1
            self.unroll_ctx = saved_ctx
            # A load bound under the predicate must not be reused outside it.
            self.load_cse = {}

    def _emit_cond_inner(self, st) -> None:
        p = self.pexpr(st.cond)
        end = self.em.new_label()
        self.em.emit(f"@!{p} bra {end};")
        self.walk(st.body)
        if st.else_body:
            done = self.em.new_label()
            self.em.emit(f"bra {done};")
            self.em.place(end)
            self.walk(st.else_body)
            self.em.place(done)
        else:
            self.em.place(end)


    def stmt_Loop(self, st) -> None:
        saved_ctx = self.unroll_ctx
        self.unroll_ctx = None
        axis = st.axis.name
        saved_axis = self.axis_values.pop(axis, None)
        saved_binding = self.axes.pop(axis, None)
        saved_extent = self.axis_extent.pop(axis, None)
        try:
            self._emit_loop_inner(st)
        finally:
            self.unroll_ctx = saved_ctx
            self.load_cse = {}
            if saved_axis is not None:
                self.axis_values[axis] = saved_axis
            if saved_binding is not None:
                self.axes[axis] = saved_binding
            if saved_extent is not None:
                self.axis_extent[axis] = saved_extent

    def _emit_loop_inner(self, st) -> None:
        if self._try_warp_dot(st):
            return
        self._emit_loop_seeds(st)
        extent = int(st.axis.extent.value)
        var = self.em.r()
        self.axes[st.axis.name] = var
        if _should_unroll(st, set(self.smem)):
            cap = 64
            if self._batched_unroll(
                    st, lambda i: self.em.emit(f"mov.s32 {var}, {i};"), extent, 1, cap,
                    value_of=lambda i: i):
                return
            # Constant trip count, shared-memory body: emit the body once per
            # value with the axis bound to a literal.  ptxas folds every
            # address into a [base + immediate] operand.
            bases: dict = {}
            saved_pairs = self._begin_pairs()
            self.load_pairs = self._plan_pairs(st, 1, extent)
            for i in range(extent):
                self.em.emit(f"mov.s32 {var}, {i};")
                self.icache = {}
                self.copy_index = i
                self._apply_pending()
                self.unroll_ctx = {"axis": st.axis.name, "k": i, "bases": bases, "axis_step": 1}
                self.walk(st.body)
            self._end_pairs(saved_pairs)
            return
        factor = _partial_unroll_factor(st, set(self.smem))
        if factor > 1:
            # Global-memory body: unroll by a small factor inside a runtime
            # chunk loop.  The copies carry constant offsets, so addresses
            # fold, without the register pressure of a full unroll.
            self._loop_partial(st, var, extent, factor)
            return
        self.em.emit(f"mov.u32 {var}, 0;")
        saved_plan = getattr(self, "hoist_plan", None)
        self._emit_hoisted_loads(st)
        prev = self._begin_hoist(st, st.axis.name)
        try:
            self._loop(var, extent, 1, st.body)
        finally:
            self.hoisted = prev
            self.hoist_plan = saved_plan


    def stmt_StridedLoop(self, st) -> None:
        saved_ctx = self.unroll_ctx
        self.unroll_ctx = None
        axis = st.axis.name
        saved_axis = self.axis_values.pop(axis, None)
        saved_binding = self.axes.pop(axis, None)
        saved_extent = self.axis_extent.pop(axis, None)
        try:
            self._emit_strided_inner(st)
        finally:
            self.unroll_ctx = saved_ctx
            self.load_cse = {}
            if saved_axis is not None:
                self.axis_values[axis] = saved_axis
            if saved_binding is not None:
                self.axes[axis] = saved_binding
            if saved_extent is not None:
                self.axis_extent[axis] = saved_extent

    def _emit_strided_inner(self, st) -> None:
        self._emit_loop_seeds(st)
        var = self.em.r()
        self.em.emit(f"mov.u32 {var}, {self.idx32(st.start)};")
        self.axes[st.axis.name] = var
        end = st.end if st.end is not None else st.axis.extent
        limit = self.idx32(end)
        step = st.step
        step_val = int(step.value) if hasattr(step, "value") else None
        prev = self._begin_hoist(st, st.axis.name)
        plan = self._strided_plan(st, step_val)
        if plan is not None:
            trips, factor = plan
            self._strided_unrolled(st, var, step_val, trips, factor,
                                   _const_int(st.start))
        elif _single_trip(st, step_val):
            self._guarded_body(var, limit, st.body)
        else:
            bounded = self._strided_bounded_plan(st, step_val)
            if bounded is not None:
                trips, guard_from = bounded
                self._strided_unrolled(
                    st, var, step_val, trips, trips, _const_int(st.start),
                    limit=limit, guard_from=guard_from)
            else:
                self._loop(var, limit, step_val, st.body)
        self.hoisted = prev

    def _guarded_body(self, var, limit, body) -> None:
        em = self.em
        done = em.new_label()
        p = em.p()
        em.emit(f"setp.ge.u32 {p}, {var}, {limit};")
        em.emit(f"@{p} bra {done};")
        self.walk(body)
        em.place(done)

    def _invariant_inner_loads(self, st):
        """Read-only global loads inside an inner loop that do not vary with st's axis.

        Attention's score loop reloads the same 32-element query row for every
        key.  With one warp per block there are no other warps to hide those
        loads, so they dominate.  Cache them in registers for the whole outer
        loop and rebind per unrolled copy.
        """
        outer = st.axis.name
        body = st.body
        writes = _written_buffers(body)
        found = {}
        for inner in body:
            if stmt_name(inner) != "Loop":
                continue
            try:
                extent = int(inner.axis.extent.value)
            except (AttributeError, TypeError, ValueError):
                continue
            if not 2 <= extent <= 32:
                continue
            if not (_should_unroll(inner, set(self.smem))
                    or _partial_unroll_factor(inner, set(self.smem)) >= extent):
                continue
            defined = _defined_in(inner.body)
            for child in inner.body:
                if stmt_name(child) != "Load":
                    continue
                src = child.input
                if src in self.smem or src in self.scalars or src in writes:
                    continue
                if len(child.names) > 2:
                    continue
                if self.flat_axis_coefficient(child.index, self.shape_of(src), outer) != 0:
                    continue
                free = set()
                for expr in _index_exprs([child]):
                    free |= expr.free_vars()
                if free & defined:
                    continue
                found.setdefault(id(inner), (inner, []))[1].append(child)
        return found


    def _emit_hoisted_loads(self, st) -> None:
        found = self._invariant_inner_loads(st)
        plan: dict = {}
        saved_axes = dict(self.axes)
        saved_ctx = self.unroll_ctx
        self.unroll_ctx = None
        total = 0
        try:
            for inner, loads in found.values():
                extent = int(inner.axis.extent.value)
                regs = [[None] * len(loads) for _ in range(extent)]
                for i in range(extent):
                    self.axes[inner.axis.name] = self.em.mov_s32(i)
                    self.icache = {}
                    # Rebinding the inner axis changes the registers every
                    # symbolic address key reads.  The symbolic key alone
                    # cannot tell the copies apart, so each copy starts with
                    # an empty address cache.
                    self.addr_cse = {}
                    for j, child in enumerate(loads):
                        self._emit_load_impl(child)
                        regs[i][j] = [(n, self.values[n]) for n in child.names]
                        total += len(child.names)
                        if total > 64:
                            raise CompileError("too many hoisted registers")
                for j, child in enumerate(loads):
                    plan[id(child)] = [regs[i][j] for i in range(extent)]
        except CompileError:
            plan = {}
        finally:
            self.axes = saved_axes
            self.unroll_ctx = saved_ctx
        if plan:
            merged = dict(getattr(self, "hoist_plan", None) or {})
            merged.update(plan)
            self.hoist_plan = merged


    def _begin_pairs(self):
        saved = (getattr(self, "load_pairs", None),
                 getattr(self, "_pair_queue", None))
        self.load_pairs = {}
        self._pair_queue = [{}, {}, {}]
        return saved


    def _end_pairs(self, saved) -> None:
        self.load_pairs, self._pair_queue = saved


    def _batchable(self, body) -> bool:
        names = {stmt_name(s) for s in body}
        if not names or not names <= {"Load", "Let", "Assign", "Accum"}:
            return False
        if "Load" not in names:
            return False
        defined = _defined_in(body)
        for st in body:
            if stmt_name(st) != "Load":
                continue
            for expr in _index_exprs([st]):
                if expr.free_vars() & defined:
                    return False
        return True


    def _batched_unroll(self, st, set_axis, n: int, stride: int, cap: int,
                        value_of=None) -> bool:
        """Emit every load of a group of copies before the arithmetic that uses it.

        A single-warp block (attention) cannot hide global-load latency with
        other warps, and an unrolled load/use/load/use stream exposes it once
        per load.  Collecting a group's loads back to back turns N exposures
        into one.  Only pure bodies are eligible: no stores, syncs or branches.
        """
        body = st.body
        per_copy = sum(len(s.names) for s in body if stmt_name(s) == "Load")
        if per_copy == 0 or not self._batchable(body):
            return False
        group = min(n, cap // per_copy)
        if group < 2:
            return False
        saved_pairs = self._begin_pairs()
        self.load_pairs = self._plan_pairs(st, stride, n)
        need = 0
        for kind, *rest in self.load_pairs.values():
            if kind == "cross":
                need = max(need, rest[0])
        if need:
            group = (group // need) * need
            if group < need:
                self._end_pairs(saved_pairs)
                return False
        axis = st.axis.name
        prev_axis = self.axis_values.get(axis)
        bases: dict = {}
        for start in range(0, n, group):
            count = min(group, n - start)
            recorded = []
            for i in range(start, start + count):
                set_axis(i)
                if value_of is not None:
                    self.axis_values[axis] = value_of(i)
                else:
                    self.axis_values.pop(axis, None)
                self.icache = {}
                self.copy_index = i
                self.unroll_ctx = {"axis": st.axis.name, "k": i, "bases": bases,
                                   "axis_step": stride}
                self._deferred_current = dict(self._pair_queue.pop(0))
                self._pair_queue.append({})
                self.deferred_mode = "load"
                self.walk(body)
                recorded.append(self._deferred_current)
            for i in range(start, start + count):
                set_axis(i)
                if value_of is not None:
                    self.axis_values[axis] = value_of(i)
                else:
                    self.axis_values.pop(axis, None)
                self.icache = {}
                self.copy_index = i
                self.unroll_ctx = {"axis": st.axis.name, "k": i, "bases": bases,
                                   "axis_step": stride}
                self.deferred = recorded[i - start]
                self.deferred_mode = "use"
                self.walk(body)
        self.deferred_mode = None
        self.deferred = None
        self._end_pairs(saved_pairs)
        if prev_axis is None:
            self.axis_values.pop(axis, None)
        else:
            self.axis_values[axis] = prev_axis
        return True


    def _chunk_address(self, buffer: str, stmt, index):
        """Address for an unrolled copy, reusing the chunk base and a constant offset."""
        ctx = self.unroll_ctx
        if ctx is None or index is None:
            return None
        key = (buffer, tuple(index))
        k = ctx["k"]
        # The address stride is the buffer element size, never the type of
        # the value flowing through the statement.
        dt = norm_dtype(self.buffer_dtype(buffer))
        esize = elem_bytes(dt)
        if k == 0:
            ctx["bases"].pop(key, None)
            return None
        if key not in ctx["bases"]:
            return None
        base_addr, per_copy = ctx["bases"][key]
        return _addr_add(base_addr, k * per_copy * esize)

    def _remember_chunk_address(self, buffer: str, index, addr) -> None:
        ctx = self.unroll_ctx
        if ctx is None or index is None or ctx["k"] != 0:
            return
        coef = self.flat_axis_coefficient(index, self.shape_of(buffer), ctx["axis"])
        if coef is None:
            return
        ctx["bases"][(buffer, tuple(index))] = (addr, coef * ctx.get("axis_step", 1))

    def _loop_partial(self, st, var: str, extent: int, factor: int) -> None:
        saved_plan = getattr(self, "hoist_plan", None)
        self._emit_hoisted_loads(st)
        try:
            self._loop_partial_inner(st, var, extent, factor)
        finally:
            self.hoist_plan = saved_plan


    def _loop_partial_inner(self, st, var: str, extent: int, factor: int) -> None:
        em = self.em
        cap = 64
        per_copy = sum(len(s.names) for s in st.body if stmt_name(s) == "Load")
        if per_copy and extent * per_copy <= cap:
            # The whole loop fits in one register-resident load batch.
            if self._batched_unroll(
                    st, lambda i: em.emit(f"mov.s32 {var}, {i};"), extent, 1, cap,
                    value_of=lambda i: i):
                return
        if factor >= extent:
            bases: dict = {}
            saved_pairs = self._begin_pairs()
            self.load_pairs = self._plan_pairs(st, 1, extent)
            prev_axis = self.axis_values.get(st.axis.name)
            for i in range(extent):
                em.emit(f"mov.s32 {var}, {i};")
                self.icache = {}
                self.copy_index = i
                self.axis_values[st.axis.name] = i
                self._apply_pending()
                self.unroll_ctx = {"axis": st.axis.name, "k": i, "bases": bases,
                                   "axis_step": 1}
                self.walk(st.body)
            self._end_pairs(saved_pairs)
            if prev_axis is None:
                self.axis_values.pop(st.axis.name, None)
            else:
                self.axis_values[st.axis.name] = prev_axis
            return
        chunks = extent // factor
        remainder = extent % factor
        if chunks:
            base = em.r()
            count = em.r()
            em.emit(f"mov.u32 {base}, 0;")
            em.emit(f"mov.u32 {count}, {chunks};")
            top = em.new_label()
            done = em.new_label()
            em.place(top)
            p = em.p()
            em.emit(f"setp.eq.u32 {p}, {count}, 0;")
            em.emit(f"@{p} bra {done};")
            if per_copy and factor * per_copy <= cap and self._batchable(st.body):
                self._batched_unroll(
                    st, lambda i: em.emit(f"add.s32 {var}, {base}, {i};"),
                    factor, 1, cap, value_of=None)
            else:
                bases: dict = {}
                saved_pairs = self._begin_pairs()
                self.load_pairs = self._plan_pairs(st, 1, factor)
                for k in range(factor):
                    em.emit(f"add.s32 {var}, {base}, {k};")
                    self.icache = {}
                    self.copy_index = k
                    self._apply_pending()
                    self.unroll_ctx = {"axis": st.axis.name, "k": k,
                                       "bases": bases, "axis_step": 1}
                    self.walk(st.body)
                self._end_pairs(saved_pairs)
            em.emit(f"add.u32 {base}, {base}, {factor};")
            em.emit(f"sub.u32 {count}, {count}, 1;")
            em.emit(f"bra {top};")
            em.place(done)
        remainder_bases: dict = {}
        saved_pairs = self._begin_pairs()
        self.load_pairs = self._plan_pairs(st, 1, remainder)
        prev_axis = self.axis_values.get(st.axis.name)
        for k in range(remainder):
            em.emit(f"mov.s32 {var}, {chunks * factor + k};")
            self.icache = {}
            self.copy_index = k
            self.axis_values[st.axis.name] = chunks * factor + k
            self._apply_pending()
            self.unroll_ctx = {"axis": st.axis.name, "k": 0, "bases": remainder_bases, "axis_step": 1}
            self.walk(st.body)
        self._end_pairs(saved_pairs)
        if prev_axis is None:
            self.axis_values.pop(st.axis.name, None)
        else:
            self.axis_values[st.axis.name] = prev_axis

    def _strided_plan(self, st, step_val):
        """(trip count, unroll factor) when the trip count is compile-time constant.

        A cooperating strided loop such as ``for i in tid..65536:128`` always
        runs ``end/step`` times when the axis range fits in one step and the
        bound is a multiple of it.  The reference unrolls those 16x and turns
        every load into a [base+immediate] access; a rolled loop leaves the
        reduction exposed.  The budget keeps the expanded body bounded.
        """
        if step_val is None or step_val <= 0:
            return None
        from emmy.compiler.dim import Dim
        from emmy.compiler.ir.expr import Var

        end = st.end if st.end is not None else st.axis.extent
        if isinstance(end, Dim):
            end_val = int(end.value)
        else:
            end_val = _const_int(end)
        if end_val is None or end_val % step_val:
            return None
        start = st.start
        if isinstance(start, Var):
            extent = self.axis_extent.get(start.name)
            if extent is None or extent > step_val:
                return None
            trips = end_val // step_val
        else:
            start_val = _const_int(start)
            if start_val is None or (end_val - start_val) % step_val:
                return None
            trips = (end_val - start_val) // step_val
        if trips <= 1:
            return None
        weight = _expanded_weight(st.body, set(self.smem))
        loads = _loop_load_buffers(st.body)
        shared_only = not (loads - set(self.smem))
        if shared_only and trips * weight <= 8000:
            # A staged matmul's k loop: every load is shared memory, so a full
            # unroll folds all addresses and removes the loop entirely.  This is
            # a conventional staged-GEMM shape.
            return trips, trips
        if trips * weight > 4000:
            return None
        # largest factor that divides the trip count, capped for code size
        for factor in (16, 8, 4, 2):
            if trips % factor == 0 and factor * weight <= 512:
                return trips, factor
        return trips, 1

    def _strided_bounded_plan(self, st, step_val):
        """Unroll a strided loop whose start is a bounded runtime expression.

        Copy loops such as ``for fa in start..640:256`` run three times for
        small starts and twice for large ones.  The trip count is not constant,
        so the divisible-case planner declines, but the maximum is: unroll to
        that maximum, run the copies every thread is sure to reach unguarded,
        and guard the rest with the same comparison the loop used.
        """
        if step_val is None or step_val <= 0:
            return None
        end_val = _const_int(st.end if st.end is not None else st.axis.extent)
        if end_val is None or end_val <= 0:
            return None
        start = st.start
        if start is None:
            low = high = 0
        else:
            aff = _affine(start)
            if aff is None:
                return None
            low, coeffs = aff
            high = low
            for name, coef in coeffs.items():
                if coef <= 0:
                    return None
                extent = self.axis_extent.get(name)
                if extent is None:
                    return None
                high += coef * (extent - 1)
        if low < 0 or high >= end_val:
            return None
        trips = (end_val - low + step_val - 1) // step_val
        if trips < 2 or trips > 8:
            return None
        guard_from = 0
        while guard_from < trips and high + guard_from * step_val < end_val:
            guard_from += 1
        weight = _expanded_weight(st.body, set(self.smem))
        if trips * weight > 512:
            return None
        return trips, guard_from

    def _strided_unrolled(self, st, var: str, step: int, trips: int, factor: int,
                          start_val=None, limit=None, guard_from=None) -> None:
        em = self.em
        base = em.r()
        em.emit(f"mov.u32 {base}, {var};")
        chunks = trips // factor
        if chunks > 1:
            count = em.r()
            em.emit(f"mov.u32 {count}, {chunks};")
            top = em.new_label()
            done = em.new_label()
            em.place(top)
            p = em.p()
            em.emit(f"setp.eq.u32 {p}, {count}, 0;")
            em.emit(f"@{p} bra {done};")
        if factor == trips and guard_from is None:
            cap = 64
            value_of = None
            if start_val is not None:
                value_of = lambda i: start_val + i * step
            if self._batched_unroll(
                    st, lambda i: em.emit(f"add.s32 {var}, {base}, {i * step};"),
                    trips, step, cap, value_of=value_of):
                return
        strided_bases: dict = {}
        saved_pairs = self._begin_pairs()
        self.load_pairs = self._plan_pairs(st, step, factor)
        axis = st.axis.name
        prev_axis = self.axis_values.get(axis)
        for k in range(factor):
            em.emit(f"add.s32 {var}, {base}, {k * step};")
            self.icache = {}
            self.copy_index = k
            if chunks == 1 and start_val is not None:
                self.axis_values[axis] = start_val + k * step
            else:
                self.axis_values.pop(axis, None)
            skip = None
            if guard_from is not None and k >= guard_from:
                # Some threads run one copy fewer than the unrolled maximum.
                # The guard replicates the loop bound they used to test.
                skip = em.new_label()
                p = em.p()
                em.emit(f"setp.ge.u32 {p}, {var}, {limit};")
                em.emit(f"@{p} bra {skip};")
            self._apply_pending()
            self.unroll_ctx = {"axis": axis, "k": k, "bases": strided_bases,
                               "axis_step": step, "trips": trips}
            self.walk(st.body)
            if skip is not None:
                em.place(skip)
        self._end_pairs(saved_pairs)
        if prev_axis is None:
            self.axis_values.pop(axis, None)
        else:
            self.axis_values[axis] = prev_axis
        if chunks > 1:
            em.emit(f"add.u32 {base}, {base}, {factor * step};")
            em.emit(f"sub.u32 {count}, {count}, 1;")
            em.emit(f"bra {top};")
            em.place(done)


    def _begin_hoist(self, st, axis_name: str):
        """Hoist index subexpressions that cannot change inside this loop.

        ptxas does not do loop-invariant code motion, and recomputing a stage
        base or a row offset for every shared-memory load dominated the matmul
        inner loop.  Candidates are maximal integer subexpressions whose free
        variables are bound outside this loop and are not redefined inside it.
        """
        body = getattr(st, "body", ())
        shadowed = _axes_defined_in(body)
        safe = {name for name in self.axes if name != axis_name and name not in shadowed}
        rebound = set(self.accumulators) | _defined_in(body)
        for name in self.values:
            if name not in rebound:
                safe.add(name)
        previous = self.hoisted
        self.hoisted = dict(previous)
        saved_cache = self.icache
        self.icache = {}
        for expr in _index_exprs(body):
            self._hoist_expr(expr, safe)
        self.icache = saved_cache
        return previous


    def _hoist_expr(self, e, safe: set) -> None:
        from emmy.compiler.ir.expr import Expr

        if not isinstance(e, Expr):
            return
        if self._is_int_expr(e) and e.free_vars() <= safe and _is_expensive(e):
            if e not in self.hoisted:
                self.hoisted[e] = self.idx32(e)
            return
        for child in _expr_children(e):
            self._hoist_expr(child, safe)


    def _is_int_expr(self, e) -> bool:
        from emmy.compiler.ir.expr import (
            BinaryExpr,
            Builtin,
            CastExpr,
            FlatIndex,
            Literal,
            TernaryExpr,
            Var,
        )

        if isinstance(e, Literal):
            return not _literal_is_float(e)
        if isinstance(e, Var):
            if e.name in self.axes:
                return True
            try:
                return self.value(e.name)[1] in INT_KINDS
            except CompileError:
                return False
        if isinstance(e, BinaryExpr):
            if e.op in COMPARE_OPS or e.op in ("&&", "||", "and", "or", "!", "not"):
                return False
            return self._is_int_expr(e.left) and self._is_int_expr(e.right)
        if isinstance(e, TernaryExpr):
            return self._is_int_expr(e.if_true) and self._is_int_expr(e.if_false)
        if isinstance(e, FlatIndex) or isinstance(e, Builtin):
            return True
        if isinstance(e, CastExpr):
            return norm_dtype(e.dtype) not in FLOAT_KINDS
        return False


    def _try_warp_dot(self, st) -> bool:
        """Split a warp-uniform reduction across lanes and reduce with shuffles.

        In a one-row-per-warp schedule every lane of the warp computes the same
        dot product.  Each lane can take 1/32 of the reduction axis instead,
        then a butterfly shuffle sums the partials into every lane.  This turns
        H loads and H multiplies per lane into H/32 of each plus five shuffles.
        """
        if (not getattr(self, "warp_axes", ()) or not getattr(self, "full_tile", False)
                or getattr(self, "divergent", 0)):
            return False
        try:
            ext = int(st.axis.extent.value)
        except (AttributeError, TypeError, ValueError):
            return False
        if ext < 32 or ext % 32:
            return False
        body = st.body
        if len(body) != 4:
            return False
        kinds = [stmt_name(x) for x in body]
        if kinds != ["Load", "Load", "Assign", "Accum"]:
            return False
        load_a, load_b, prod, acc = body
        if _op_name(prod.op) != "multiply" or len(prod.args) != 2:
            return False
        if _op_name(acc.op) != "add" or acc.base is not None:
            return False
        if acc.value != prod.name or len(load_a.names) != 1 or len(load_b.names) != 1:
            return False
        if set(prod.args) != {load_a.names[0], load_b.names[0]}:
            return False
        axis = st.axis.name
        varying: dict[str, int] = {}
        for load in (load_a, load_b):
            fv = set()
            for expr in _index_exprs([load]):
                fv |= expr.free_vars()
            if axis not in fv:
                return False
            for name in fv & self.warp_axes:
                varying[name] = self.axis_suffix.get(name, 1)
        group = min(varying.values()) if varying else 32
        if group < 2 or group & (group - 1) or ext % group:
            # A shuffle tree needs a power-of-two width that divides the axis.
            return False
        self._emit_loop_seeds(st)
        acc_reg, kind = self.values[acc.name]
        em = self.em
        tid = em.r()
        em.emit(f"mov.u32 {tid}, %tid.x;")
        lane = em.r()
        em.emit(f"and.b32 {lane}, {tid}, {group - 1};")
        saved_hoist = getattr(self, "hoist_plan", None)
        self.hoist_plan = None
        saved_pairs = self._begin_pairs()
        saved_axis = self.axes.get(axis)
        try:
            for j in range(ext // group):
                if j:
                    ax = em.r()
                    em.emit(f"add.s32 {ax}, {lane}, {j * group};")
                else:
                    ax = lane
                self.axes[axis] = ax
                self.icache = {}
                self.walk(body)
        finally:
            self.hoist_plan = saved_hoist
            self._end_pairs(saved_pairs)
            if saved_axis is not None:
                self.axes[axis] = saved_axis
            else:
                self.axes.pop(axis, None)
        off = group // 2
        while off:
            tmp = em.f()
            em.emit(f"shfl.sync.bfly.b32 {tmp}, {acc_reg}, {off}, 0x1f, 0xffffffff;")
            em.emit(f"add.f32 {acc_reg}, {acc_reg}, {tmp};")
            off //= 2
        self.values[acc.name] = (acc_reg, "f32")
        return True


    def _emit_loop_seeds(self, st) -> None:
        """Seed the accumulators this loop instance declares.

        Emmy emits each accumulator's seed where its carrying loop is entered
        (see the reference CUDA: two sibling loops over the same axis each own
        their own dot-product accumulator), so the association is with the loop
        *statement*, not with the axis name.  Accumulators with an explicit
        ``Init`` are seeded by that statement instead.
        """
        axis = st.axis.name
        for acc in _accums_for_loop(st, axis, self.accumulator_inits):
            self._seed_accumulator(acc)


    def _loop(self, var, limit, step, body) -> None:
        em = self.em
        top = em.new_label()
        end = em.new_label()
        em.place(top)
        p = em.p()
        em.emit(f"setp.ge.u32 {p}, {var}, {limit};")
        em.emit(f"@{p} bra {end};")
        self.walk(body)
        if isinstance(step, int):
            em.emit(f"add.u32 {var}, {var}, {step};")
        else:
            em.emit(f"add.u32 {var}, {var}, {self.idx32(step)};")
        em.emit(f"bra {top};")
        em.place(end)


    def stmt_Sync(self, st) -> None:
        self.load_cse = {}
        if getattr(st, "warp", False):
            self.em.emit("bar.warp.sync 0xffffffff;")
        elif getattr(st, "count", None):
            self.em.emit(f"bar.sync {int(st.barrier_id)}, {int(st.count)};")
        else:
            self.em.emit(f"bar.sync {int(st.barrier_id)};")


    def stmt_Smem(self, st) -> None:
        # declarations are hoisted; nothing to emit at the use site
        return


    def stmt_WarpShuffle(self, st) -> None:
        length = int(st.length)
        offset = length // 2
        while offset >= 1:
            for state, state_b in zip(st.state, st.state_b):
                src, kind = self.values[state]
                shuffled = self._shuffle(src, offset, kind)
                self.values[state_b] = (shuffled, kind)
            self.walk(st.combine_states)
            offset //= 2


    def _shuffle(self, src: str, offset: int, kind: str) -> str:
        if kind == "f16":
            up = self.em.f()
            self.em.emit(f"cvt.f32.f16 {up}, {src};")
            out32 = self._shuffle(up, offset, "f32")
            back = self.em.h()
            self.em.emit(f"cvt.rn.f16.f32 {back}, {out32};")
            return back
        if kind == "f64":
            lo = self.em.r()
            hi = self.em.r()
            self.em.emit(f"mov.b64 {{{lo}, {hi}}}, {src};")
            slo = self.em.r()
            shi = self.em.r()
            self.em.emit(f"shfl.sync.bfly.b32 {slo}, {lo}, {offset}, 0x1f, 0xffffffff;")
            self.em.emit(f"shfl.sync.bfly.b32 {shi}, {hi}, {offset}, 0x1f, 0xffffffff;")
            out = self.em.d()
            self.em.emit(f"mov.b64 {out}, {{{slo}, {shi}}};")
            return out
        if kind == "f32":
            # shfl is a .b32 instruction; PTX accepts an .f32 register as a
            # .b32 operand, so the round trip through an integer register is
            # dead code.
            reg = self.em.f()
            self.em.emit(f"shfl.sync.bfly.b32 {reg}, {src}, {offset}, 0x1f, 0xffffffff;")
            return reg
        if kind in ("i64", "u64"):
            lo = self.em.r()
            hi = self.em.r()
            self.em.emit(f"mov.b64 {{{lo}, {hi}}}, {src};")
            slo = self.em.r()
            shi = self.em.r()
            self.em.emit(f"shfl.sync.bfly.b32 {slo}, {lo}, {offset}, 0x1f, 0xffffffff;")
            self.em.emit(f"shfl.sync.bfly.b32 {shi}, {hi}, {offset}, 0x1f, 0xffffffff;")
            out = self.em.rd()
            self.em.emit(f"mov.b64 {out}, {{{slo}, {shi}}};")
            return out
        out = self.em.r()
        self.em.emit(f"shfl.sync.bfly.b32 {out}, {src}, {offset}, 0x1f, 0xffffffff;")
        return out


    def stmt_TreeHalve(self, st) -> None:
        """Shared-memory halving tree over one or more coupled states.

        Each level halves the active thread range: thread `t` folds in `t+s`
        from every state's buffer, writes the result back, then the block
        synchronizes.  The loop's final value is broadcast from slot 0.
        """
        self.load_cse = {}
        length = int(st.length)
        barrier = int(st.barrier_id)
        count = getattr(st, "barrier_count", None)
        tid = self._int_var(st.tid_var)
        inner = getattr(st, "inner", None)
        inner_reg = self._int_var(inner[0]) if inner else None
        scale = int(inner[1]) if inner else 0
        s = length // 2
        while s >= 1:
            p = self.em.p()
            self.em.emit(f"setp.lt.u32 {p}, {tid}, {s};")
            skip = self.em.new_label()
            self.em.emit(f"@!{p} bra {skip};")
            self._tree_level(st, tid, s, inner_reg, scale)
            self.em.place(skip)
            if count:
                self.em.emit(f"bar.sync {barrier}, {int(count)};")
            else:
                self.em.emit(f"bar.sync {barrier};")
            s //= 2

        for state, buf in zip(st.state, st.bufs):
            dt = self.smem[buf]["dtype"]
            base = self._smem_base(buf)
            kind = _kind_of_dtype(dt)
            reg = self.em.preg(dt)
            addr = base
            if inner_reg is not None:
                off = self._scaled_offset(inner_reg, elem_bytes(dt))
                addr = self.em.r()
                self.em.emit(f"add.u32 {addr}, {base}, {off};")
            self.em.emit(f"ld.shared.{ptx_type(dt)} {reg}, [{addr}];")
            self.values[state] = (reg, kind)


    def _tree_level(self, st, tid: str, s: int, inner_reg=None, scale: int = 0) -> None:
        """Load state / state+s from each buffer, run the combine, store back."""
        slots = []
        for state, state_b, buf in zip(st.state, st.state_b, st.bufs):
            info = self.smem[buf]
            dt = info["dtype"]
            esize = elem_bytes(dt)
            kind = _kind_of_dtype(dt)
            base = self._smem_base(buf)
            if inner_reg is not None:
                off0 = self._tree_inner_offset(tid, inner_reg, scale, 0, esize)
                off1 = self._tree_inner_offset(tid, inner_reg, scale, s, esize)
            else:
                off0 = self._scaled_offset(tid, esize)
                off1 = self._scaled_offset_imm(tid, s, esize)
            a0 = self.em.r()
            self.em.emit(f"add.u32 {a0}, {base}, {off0};")
            a1 = self.em.r()
            self.em.emit(f"add.u32 {a1}, {base}, {off1};")
            r0 = self.em.preg(dt)
            self.em.emit(f"ld.shared.{ptx_type(dt)} {r0}, [{a0}];")
            self.values[state] = (r0, kind)
            r1 = self.em.preg(dt)
            self.em.emit(f"ld.shared.{ptx_type(dt)} {r1}, [{a1}];")
            self.values[state_b] = (r1, kind)
            slots.append((state, buf, base, a0, dt))
        self.walk(st.combine_states)
        for state, buf, base, addr, dt in slots:
            reg, kind = self.values[state]
            reg = self._cast_reg(reg, kind, dt)
            self.em.emit(f"st.shared.{ptx_type(dt)} [{addr}], {reg};")


    def _tree_inner_offset(self, tid: str, inner_reg: str, scale: int, add: int, esize: int) -> str:
        t = self.em.r()
        if add:
            v = self.em.r()
            self.em.emit(f"add.u32 {v}, {tid}, {add};")
            base = v
        else:
            base = tid
        if scale != 1:
            self.em.emit(f"mul.lo.u32 {t}, {base}, {scale};")
        else:
            self.em.emit(f"mov.u32 {t}, {base};")
        out = self.em.r()
        self.em.emit(f"add.u32 {out}, {t}, {inner_reg};")
        if esize != 1:
            self.em.emit(f"shl.b32 {out}, {out}, {_log2(esize)};")
        return out


    def stmt_CpAsyncCopy(self, st) -> None:
        self.load_cse = {}
        info = self.smem[st.smem]
        esize = elem_bytes(info["dtype"])
        lane_index = getattr(st, "lane_index", None)
        ctx = self.unroll_ctx
        reuse = lane_index is None and ctx is not None and ctx.get("trips", 0) >= 4
        cached = self._chunk_address(st.smem, st, st.smem_index) if reuse else None
        if cached is not None:
            saddr = cached
        else:
            flat = self.flat_index(st.smem_index, tuple(info["extents"]),
                                   recombine=reuse)
            if lane_index is not None:
                cols = int(info["extents"][-1])
                lane_flat = self.flat_index(lane_index, (0, cols),
                                            recombine=self.recombine_lane)
                flat = self._s32_bin("add.s32", flat, lane_flat)
                if getattr(st, "swizzle", "NONE") != "NONE":
                    flat = self._swizzle(flat, st.swizzle)
            base = self._smem_base(st.smem)
            saddr = self._shared_addr(base, flat, esize)
            if reuse:
                self._remember_chunk_address(st.smem, st.smem_index, saddr)
        ptr = self.params.get(st.src)
        if ptr is None:
            raise CompileError(f"cp.async from unknown buffer {st.src!r}")
        gflat = self.flat_index(st.src_index, self.shape_of(st.src))
        gaddr = self._global_addr(ptr, gflat, esize)
        nbytes = int(st.nbytes)
        if nbytes == 16:
            self.em.emit(f"cp.async.cg.shared.global [{saddr}], [{gaddr}], 16;")
        else:
            self.em.emit(f"cp.async.ca.shared.global [{saddr}], [{gaddr}], {nbytes};")


    def stmt_ZeroPrologue(self, st) -> None:
        self.load_cse = {}
        """Delegated zero-init: every grid thread sweeps a stride of 32-bit words."""
        em = self.em
        ptr = self.params.get(st.dst)
        if ptr is None:
            raise CompileError(f"zero prologue for unknown buffer {st.dst!r}")
        words = int(st.words)
        tx, ty, tz = em.r(), em.r(), em.r()
        em.emit(f"mov.u32 {tx}, %tid.x;")
        em.emit(f"mov.u32 {ty}, %tid.y;")
        em.emit(f"mov.u32 {tz}, %tid.z;")
        bx, by, bz = em.r(), em.r(), em.r()
        em.emit(f"mov.u32 {bx}, %ctaid.x;")
        em.emit(f"mov.u32 {by}, %ctaid.y;")
        em.emit(f"mov.u32 {bz}, %ctaid.z;")
        bdx, bdy, bdz = em.r(), em.r(), em.r()
        em.emit(f"mov.u32 {bdx}, %ntid.x;")
        em.emit(f"mov.u32 {bdy}, %ntid.y;")
        em.emit(f"mov.u32 {bdz}, %ntid.z;")
        gdx, gdy = em.r(), em.r()
        em.emit(f"mov.u32 {gdx}, %nctaid.x;")
        em.emit(f"mov.u32 {gdy}, %nctaid.y;")
        # block/thread linearization identical to emmy's renderer
        blk = em.r()
        em.emit(f"mad.lo.u32 {blk}, {bz}, {gdy}, {by};")
        em.emit(f"mad.lo.u32 {blk}, {blk}, {gdx}, {bx};")
        thr = em.r()
        em.emit(f"mad.lo.u32 {thr}, {tz}, {bdy}, {ty};")
        em.emit(f"mad.lo.u32 {thr}, {thr}, {bdx}, {tx};")
        nthreads = em.r()
        em.emit(f"mul.lo.u32 {nthreads}, {bdx}, {bdy};")
        em.emit(f"mul.lo.u32 {nthreads}, {nthreads}, {bdz};")
        grid = em.r()
        em.emit(f"mul.lo.u32 {grid}, {gdx}, {gdy};")
        em.emit(f"mov.u32 {gdx}, %nctaid.z;")
        em.emit(f"mul.lo.u32 {grid}, {grid}, {gdx};")
        em.emit(f"mul.lo.u32 {grid}, {grid}, {nthreads};")
        idx = em.r()
        em.emit(f"mad.lo.u32 {idx}, {blk}, {nthreads}, {thr};")
        zero = em.mov_u32(0)
        top = em.new_label()
        done = em.new_label()
        em.place(top)
        p = em.p()
        em.emit(f"setp.ge.u32 {p}, {idx}, {words};")
        em.emit(f"@{p} bra {done};")
        off = em.r()
        em.emit(f"shl.b32 {off}, {idx}, 2;")
        addr = em.rd()
        em.emit(f"cvt.u64.u32 {addr}, {off};")
        em.emit(f"add.s64 {addr}, {ptr}, {addr};")
        em.emit(f"st.global.u32 [{addr}], {zero};")
        em.emit(f"add.u32 {idx}, {idx}, {grid};")
        em.emit(f"bra {top};")
        em.place(done)


    def stmt_CpAsyncCommit(self, st) -> None:
        self.em.emit("cp.async.commit_group;")


    def stmt_CpAsyncWait(self, st) -> None:
        self.em.emit(f"cp.async.wait_group {int(st.group)};")
