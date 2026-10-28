"""Lowering driver: build the kernel prologue, decode the Tile, assemble the module.

Emmy's Kernel IR is a tree of typed statements over a small expression language.
This package walks it once and prints PTX.  Three facts shape the design:

1. **The launch is the contract.**  ``meta.json`` gives the parameter order,
   grid/block and shared-memory budget.  Code generation reads thread and block
   ids at runtime and decodes each Tile axis exactly as Emmy's renderer does:
   ``idx_i = (gid / suffix_i) % extent_i`` over one flattened work id.  That
   makes a single ``gid < N`` guard sufficient for over-provisioned launches.

2. **Accumulators are loop-carried.**  A loop body is emitted once, so a fold
   writes back into the same register.  Each accumulator is seeded where Emmy
   declares it: at the entry of the loop instance it reduces over.  Sibling
   loops over the same axis own separate accumulators.

3. **State used under a predicate must be defined unconditionally.**  The
   lane/warp helpers and every shared-memory base are hoisted into the prologue;
   materializing one at first use can leave it undefined in the other branch.

Index arithmetic is 32-bit (buffers are capped at 2^26 elements), vector
accesses are gated on a static alignment proof, and small constant-trip loops
over shared memory are unrolled so ptxas folds addresses into immediate fields.
"""

from __future__ import annotations

from .common import *
from .emitter import Emitter
from .expr import ExprMixin
from .memory import MemoryMixin
from .ops import IDENTITY, elem_bytes, f32_bits, norm_dtype, ptx_type, reg_class
from .stmts import StmtMixin


class Lowerer(StmtMixin, ExprMixin, MemoryMixin):
    def __init__(self, kernel_op, meta_kernel) -> None:
        try:
            self._init(kernel_op, meta_kernel)
        except CompileError:
            raise
        except (TypeError, AttributeError, KeyError, IndexError, ValueError, OverflowError) as exc:
            raise CompileError(f"invalid IR: {exc}") from exc


    def _init(self, kernel_op, meta_kernel) -> None:
        self.op = kernel_op
        self.name = kernel_op.name
        self.meta = meta_kernel if meta_kernel is not None else {}
        self.em = Emitter()

        self.shapes: dict[str, tuple[int, ...]] = {}
        self.dtypes: dict[str, str] = {}
        self.scalars: dict[str, tuple[str, str]] = {}   # name -> (reg, kind)
        self.params: dict[str, str] = {}                # buffer name -> global ptr reg
        self.values: dict[str, tuple[str, str]] = {}    # SSA name -> (reg, kind)
        self.axes: dict[str, str] = {}                  # axis name -> s32 reg
        self.axis_extent: dict[str, int] = {}           # axis name -> static extent
        self.smem: dict[str, dict] = {}
        self.smem_dynamic = False
        self.smem_total = 0
        self.smem_pool_base: str | None = None

        self.divergent = 0
        self._read_meta()
        self._scan_smem(kernel_op.body)
        self._layout_smem()
        self._scan_accumulators(kernel_op.body)
        self.written_globals = {
            st.output for st in _iter_stmts(kernel_op.body)
            if stmt_name(st) == "Write" and st.output not in self.smem
        } | {
            st.dst for st in _iter_stmts(kernel_op.body) if stmt_name(st) == "ZeroPrologue"
        }
        self.cache_info: dict[str, dict] = {}
        self.warp_axes: set[str] = set()
        self.axis_suffix: dict[str, int] = {}
        self.full_tile = False
        self._select_caches(kernel_op.body)
        budget = int(self.meta.get("smem_bytes", 0) or 0)
        if self.smem_dynamic and self.smem_total > budget:
            raise CompileError(
                f"shared layout needs {self.smem_total} bytes but meta.json "
                f"reserves {budget}")
        self.uses: dict[str, int] = {}
        self.defs: dict[str, tuple[str, tuple]] = {}
        self.icache: dict = {}
        self.axis_values: dict[str, int] = {}
        self.load_cse: dict = {}
        self.addr_cse: dict = {}
        self.cse_enabled = True
        self.recombine_lane = False
        self.hoisted: dict = {}
        self.unroll_ctx = None
        _count_refs(kernel_op.body, self.uses)
        for st in _iter_stmts(kernel_op.body):
            if stmt_name(st) == "Assign" and norm_dtype(st.dtype) == "f32":
                opname = _op_name(st.op)
                if opname in ("multiply", "add", "subtract"):
                    self.defs[st.name] = (opname, tuple(st.args))
        self.fused_lets: set[str] = set()
        for st in _iter_stmts(kernel_op.body):
            if stmt_name(st) != "Accum" or _op_name(st.op) != "add" or st.base is not None:
                continue
            prod = self.defs.get(st.value)
            if (prod and prod[0] == "multiply" and len(prod[1]) == 2
                    and self.uses.get(st.value, 0) == 1):
                self.fused_lets.add(st.value)


    def _read_volume(self, stmts, name: str, mult: int = 1) -> int:
        total = 0
        for st in stmts:
            kind = stmt_name(st)
            if kind == "Load" and st.input == name:
                total += mult * len(st.names)
            elif kind == "Loop":
                total += self._read_volume(st.body, name, mult * int(st.axis.extent.value))
            elif kind == "StridedLoop":
                trips = _const_int(st.step)
                end = _const_int(st.end if st.end is not None else st.axis.extent)
                start = _const_int(st.start)
                if trips and end is not None and start is not None and (end - start) % trips == 0:
                    total += self._read_volume(st.body, name, mult * ((end - start) // trips))
            for attr in ("body", "else_body"):
                child = getattr(st, attr, None)
                if kind == "Loop" or kind == "StridedLoop":
                    continue
                if isinstance(child, (tuple, list)) and child:
                    total += self._read_volume(child, name, mult)
        return total


    def _select_caches(self, body) -> None:
        """Cache a small read-only global tile in shared memory.

        Two shapes are possible.  The whole buffer fits (attention's key tile),
        or only the block's row slice fits (a large query).  A slice is
        contiguous because blocks partition the first dimension only.
        """
        if self.smem_dynamic or self.smem:
            return
        block = [int(x) for x in self.meta["block"]][:3]
        grid = [int(x) for x in self.meta["grid"]][:3]
        if block[1] != 1 or block[2] != 1 or grid[1] != 1 or grid[2] != 1:
            # The cooperative fill is emitted for a one-dimensional launch;
            # selecting a cache the fill cannot populate would emit reads of
            # uninitialized shared memory.
            return
        threads = max(1, block[0])
        tile = next((st for st in _iter_stmts(body) if stmt_name(st) == "Tile"), None)
        if tile is None:
            return
        axes = [(a.name, int(a.extent.value)) for a in tile.axes]
        suffix = [1] * len(axes)
        for i in range(len(axes) - 2, -1, -1):
            suffix[i] = suffix[i + 1] * axes[i + 1][1]
        block_axes = {name: ext for (name, ext), suf in zip(axes, suffix)
                      if suf >= threads and suf % threads == 0}

        candidates = []
        for arg in self.meta.get("args", []):
            name = arg["name"]
            shape = tuple(int(x) for x in arg.get("shape", ()))
            if len(shape) != 2 or any(x <= 0 for x in shape):
                continue
            dt = norm_dtype(arg.get("dtype", "f32"))
            if dt != "f32" or name in self.written_globals:
                continue
            loads = [st for st in _iter_stmts(body)
                     if stmt_name(st) == "Load" and st.input == name]
            if not loads or any(st.index is None or len(st.index) != 2 for st in loads):
                continue
            reads = self._read_volume(body, name)
            motion = self._load_motion(body, name)
            if motion is None:
                continue
            coef, extent = motion
            n_full = shape[0] * shape[1]
            b0 = b1 = 1
            terms: dict[str, int] = {}
            consistent = True
            for axis, ext in block_axes.items():
                totals = set()
                for st in loads:
                    c0 = _axis_coefficient(st.index[0], axis) or 0
                    c1 = _axis_coefficient(st.index[1], axis) or 0
                    totals.add(c0 * shape[1] + c1)
                if len(totals) != 1:
                    consistent = False
                    break
                total = totals.pop()
                c0 = _axis_coefficient(loads[0].index[0], axis) or 0
                c1 = _axis_coefficient(loads[0].index[1], axis) or 0
                if c0:
                    b0 *= ext
                if c1:
                    b1 *= ext
                if total:
                    terms[axis] = total
            if not consistent:
                continue
            if b0 > 1 and b1 == 1 and shape[0] % b0 == 0:
                if coef != 1:
                    continue
                slice_rows = shape[0] // b0
                cache_shape = (slice_rows, shape[1])
                n = slice_rows * shape[1]
                is_slice = True
                swap = False
            else:
                cache_shape = (shape[1], shape[0]) if coef == shape[1] else shape
                n = n_full
                is_slice = False
                swap = coef == shape[1]
            if swap and any(len(st.names) > 1 for st in loads):
                # A transposed cache turns consecutive source rows into
                # consecutive shared elements.  An IR load that reads several
                # adjacent elements in the source's last dimension would read
                # across rows after the swap, so leave that shape uncached.
                continue
            size = n * 4
            if size > 32 * 1024:
                continue
            if reads < n:
                continue
            if 0 < self._hoist_volume(body, name) <= 64:
                continue
            vec = 4 if extent and extent % 4 == 0 else 1
            saved = reads - reads // vec
            fill_ops = (2 * n) // threads
            if saved <= 2 * fill_ops:
                continue
            candidates.append((saved, name, dt, cache_shape, swap, shape, n, is_slice,
                               terms, size))

        candidates.sort(key=lambda c: -c[0])
        used = 0
        for saved, name, dt, cache_shape, swap, src_shape, n, is_slice, terms, size in candidates:
            if used + size > 40 * 1024:
                continue
            cache_name = "__cache_" + name
            self.smem[cache_name] = {
                "extents": cache_shape, "dtype": dt, "align": 16, "offset": 0,
            }
            self.cache_info[name] = {
                "name": cache_name, "shape": cache_shape, "swap": swap,
                "src_shape": src_shape, "n": n, "slice": is_slice,
                "block_terms": terms,
            }
            used += size
        if self.cache_info:
            self._layout_smem()


    def _load_motion(self, stmts, name: str, axis: str | None = None, extent: int = 1):
        for st in stmts:
            kind = stmt_name(st)
            if (kind == "Load" and st.input == name and axis is not None
                    and st.index is not None and len(st.index) == 2):
                coef = self.flat_axis_coefficient(st.index, self.shapes.get(name, ()), axis)
                if coef is not None:
                    return coef, extent
            if kind in ("Loop", "StridedLoop"):
                try:
                    ext = int(st.axis.extent.value)
                except (AttributeError, TypeError, ValueError):
                    ext = 0
                found = self._load_motion(st.body, name, st.axis.name, ext)
                if found is not None:
                    return found
                continue
            for attr in ("body", "else_body"):
                child = getattr(st, attr, None)
                if isinstance(child, (tuple, list)) and child:
                    found = self._load_motion(child, name, axis, extent)
                    if found is not None:
                        return found
        return None


    def _hoist_volume(self, body, name: str) -> int:
        total = 0
        for outer in _iter_stmts(body):
            if stmt_name(outer) != "Loop":
                continue
            axis = outer.axis.name

            def walk(stmts, mult):
                nonlocal total
                for st in stmts:
                    kind = stmt_name(st)
                    if kind == "Load" and st.input == name and st.index is not None \
                            and len(st.index) == 2:
                        if self.flat_axis_coefficient(st.index, self.shapes[name], axis) == 0:
                            total += mult * len(st.names)
                        continue
                    if kind == "Loop":
                        walk(st.body, mult * int(st.axis.extent.value))
                    elif kind == "StridedLoop":
                        step = _const_int(st.step)
                        end = _const_int(st.end if st.end is not None else st.axis.extent)
                        if step and end is not None:
                            walk(st.body, mult * (end // step))
                    else:
                        for attr in ("body", "else_body"):
                            child = getattr(st, attr, None)
                            if isinstance(child, (tuple, list)) and child:
                                walk(child, mult)

            walk(getattr(outer, "body", ()), 1)
        return total


    def _layout_smem(self) -> None:
        offset = 0
        for info in self.smem.values():
            size = elem_bytes(info["dtype"])
            for e in info["extents"]:
                size *= e
            align = info["align"]
            offset = (offset + align - 1) // align * align
            info["offset"] = offset
            offset += size
        self.smem_total = offset


    def _emit_cache_fills(self) -> None:
        em = self.em
        if not self.cache_info:
            return
        grid = [int(x) for x in self.meta["grid"]][:3]
        block = [int(x) for x in self.meta["block"]][:3]
        if grid[1] != 1 or grid[2] != 1 or block[1] != 1 or block[2] != 1:
            raise CompileError("cache fills require a 1D launch")
        threads = max(1, block[0])
        for src, info in self.cache_info.items():
            ptr = self.params.get(src)
            if ptr is None:
                raise CompileError(f"cache of unknown buffer {src!r}")
            base = self._smem_base(info["name"])
            a2 = int(info["src_shape"][0])
            keys = int(info["src_shape"][1])
            n = info["n"]
            block_base = None
            if info["slice"]:
                for axis, term in info["block_terms"].items():
                    reg = self.axes.get(axis)
                    if reg is None:
                        raise CompileError(f"slice cache lost axis {axis!r}")
                    part = self._s32_bin("mul.lo.s32", reg, self.em.mov_s32(term))
                    block_base = part if block_base is None else self._s32_bin("add.s32", block_base, part)
            info["block_base"] = block_base
            tid = em.r()
            em.emit(f"mov.u32 {tid}, %tid.x;")
            top = em.new_label()
            done = em.new_label()
            p = em.p()
            aligned = (not info["slice"]
                       or all(int(term) % 4 == 0 for term in info["block_terms"].values()))
            vector = n % 4 == 0 and (not info["swap"] or keys % 4 == 0) and aligned
            if vector:
                i = em.r()
                em.emit(f"mul.lo.u32 {i}, {tid}, 4;")
                em.place(top)
                em.emit(f"setp.ge.u32 {p}, {i}, {n};")
                em.emit(f"@{p} bra {done};")
                gflat = i if block_base is None else self._s32_bin("add.s32", block_base, i)
                gaddr = self._global_addr(ptr, gflat, 4)
                regs = [em.f() for _ in range(4)]
                em.emit(f"ld.global.nc.v4.f32 {{{', '.join(regs)}}}, [{gaddr}];")
                if info["swap"]:
                    col = em.r()
                    row = em.r()
                    if keys & (keys - 1) == 0:
                        em.emit(f"shr.u32 {col}, {i}, {_log2(keys)};")
                        em.emit(f"and.b32 {row}, {i}, {keys - 1};")
                    else:
                        em.emit(f"div.u32 {col}, {i}, {keys};")
                        em.emit(f"rem.u32 {row}, {i}, {keys};")
                    flat = em.r()
                    em.emit(f"mad.lo.s32 {flat}, {row}, {a2}, {col};")
                    for j, reg in enumerate(regs):
                        offset = flat if j == 0 else self._s32_bin(
                            "add.s32", flat, self.em.mov_s32(j * a2))
                        saddr = self._shared_addr(base, offset, 4)
                        em.emit(f"st.shared.f32 [{saddr}], {reg};")
                else:
                    saddr = self._shared_addr(base, i, 4)
                    em.emit(f"st.shared.v4.f32 [{saddr}], {{{', '.join(regs)}}};")
                step = threads * 4
                em.emit(f"add.u32 {i}, {i}, {step};")
                em.emit(f"bra {top};")
                em.place(done)
            else:
                i = em.r()
                em.emit(f"mov.u32 {i}, {tid};")
                em.place(top)
                em.emit(f"setp.ge.u32 {p}, {i}, {n};")
                em.emit(f"@{p} bra {done};")
                row = em.r()
                col = em.r()
                if keys & (keys - 1) == 0:
                    em.emit(f"shr.u32 {col}, {i}, {_log2(keys)};")
                    em.emit(f"and.b32 {row}, {i}, {keys - 1};")
                else:
                    em.emit(f"div.u32 {col}, {i}, {keys};")
                    em.emit(f"rem.u32 {row}, {i}, {keys};")
                gflat = i if block_base is None else self._s32_bin("add.s32", block_base, i)
                gaddr = self._global_addr(ptr, gflat, 4)
                reg = em.f()
                em.emit(f"ld.global.nc.f32 {reg}, [{gaddr}];")
                if info["swap"]:
                    flat = em.r()
                    em.emit(f"mad.lo.s32 {flat}, {row}, {a2}, {col};")
                else:
                    flat = i
                saddr = self._shared_addr(base, flat, 4)
                em.emit(f"st.shared.f32 [{saddr}], {reg};")
                em.emit(f"add.u32 {i}, {i}, {threads};")
                em.emit(f"bra {top};")
                em.place(done)
        em.emit("bar.sync 0;")

    def _read_meta(self) -> None:
        if not isinstance(self.meta, dict):
            raise CompileError("meta.json must be a JSON object")
        for key in ("grid", "block"):
            if key not in self.meta:
                raise CompileError(f"meta.json is missing {key}")
            raw = self.meta[key]
            if not isinstance(raw, (list, tuple)) or len(raw) != 3:
                raise CompileError(
                    f"meta.json {key} must be a list of 3 integers")
            if any(isinstance(x, bool) or not isinstance(x, int) for x in raw):
                raise CompileError(
                    f"meta.json {key} must contain integers: {raw!r}")
            if any(x <= 0 for x in raw):
                raise CompileError(
                    f"meta.json {key} must contain positive integers: {raw!r}")
        raw_smem = self.meta.get("smem_bytes", 0)
        if raw_smem is None:
            raw_smem = 0
        if (isinstance(raw_smem, bool) or not isinstance(raw_smem, int)
                or raw_smem < 0):
            raise CompileError("meta.json smem_bytes must be a non-negative integer")
        smem_bytes = raw_smem
        arch = self.meta.get("arch")
        if arch is not None:
            digits = arch[3:] if isinstance(arch, str) and arch.startswith("sm_") else ""
            if (not isinstance(arch, str) or not arch.startswith("sm_")
                    or not digits or any(ch < "0" or ch > "9" for ch in digits)):
                raise CompileError(
                    f"meta.json arch must look like 'sm_89': {arch!r}")
        if "args" not in self.meta:
            raise CompileError("meta.json is missing args")
        args = self.meta["args"]
        if not isinstance(args, list):
            raise CompileError("meta.json args must be a list")
        seen = set()
        for arg in args:
            if not isinstance(arg, dict):
                raise CompileError("each meta.json argument must be an object")
            name = arg.get("name")
            if not isinstance(name, str) or not name:
                raise CompileError("each meta.json argument needs a non-empty name")
            if name in seen:
                raise CompileError(f"meta.json has duplicate argument {name!r}")
            seen.add(name)
            if "shape" not in arg:
                raise CompileError(
                    f"meta.json argument {name!r} is missing shape")
            raw_shape = arg["shape"]
            if not isinstance(raw_shape, (list, tuple)):
                raise CompileError(
                    f"meta.json argument {name!r} shape must be a list")
            if any(isinstance(x, bool) or not isinstance(x, int) for x in raw_shape):
                raise CompileError(
                    f"meta.json argument {name!r} shape must contain integers: "
                    f"{raw_shape!r}")
            if any(x < 0 for x in raw_shape):
                raise CompileError(
                    f"meta.json argument {name!r} shape must be non-negative: "
                    f"{raw_shape!r}")
            shape = tuple(raw_shape)
            if "dtype" not in arg:
                raise CompileError(
                    f"meta.json argument {name!r} is missing dtype")
            dt = norm_dtype(arg["dtype"])
            _check_dtype(dt, f"argument {name!r}")
            if "scalar" in arg and not isinstance(arg["scalar"], bool):
                raise CompileError(
                    f"meta.json argument {name!r} scalar must be a boolean")
            self.shapes[name] = shape
            self.dtypes[name] = dt
        if "arg_order" not in self.meta:
            raise CompileError("meta.json is missing arg_order")
        order = self.meta["arg_order"]
        if not isinstance(order, list) or not all(isinstance(x, str) for x in order):
            raise CompileError(
                "meta.json arg_order must be a list of argument names")
        duplicates = sorted({name for name in order if order.count(name) > 1})
        if duplicates:
            raise CompileError(
                "meta.json arg_order has duplicate entries: "
                + ", ".join(duplicates))
        if seen and not order:
            raise CompileError(
                "meta.json arg_order must list every argument")
        missing = sorted(seen - set(order))
        extra = sorted(set(order) - seen)
        if missing or extra:
            detail = []
            if missing:
                detail.append("missing " + ", ".join(missing))
            if extra:
                detail.append("unknown " + ", ".join(extra))
            raise CompileError(
                "meta.json arg_order must name every argument ("
                + "; ".join(detail) + ")")
        self.smem_dynamic = smem_bytes > 0


    def _scan_accumulators(self, stmts, accs=None, inits=None) -> None:
        if accs is None:
            accs, inits = {}, set()
            self._scan_accumulators(self.op.body, accs, inits)
            self.accumulators = accs
            self.accumulator_inits = inits
            self.top_seeds: list = []
            for name, acc in accs.items():
                if name in inits:
                    continue
                if not (getattr(acc, "axes", ()) or ()):
                    self.top_seeds.append(acc)
            return
        for st in stmts:
            n = stmt_name(st)
            if n == "Accum":
                accs.setdefault(st.name, st)
            elif n == "Init":
                inits.add(st.name)
            for attr in ("body", "else_body"):
                child = getattr(st, attr, None)
                if isinstance(child, (tuple, list)) and child:
                    self._scan_accumulators(child, accs, inits)


    def compile(self) -> str:
        try:
            return self._compile()
        except CompileError:
            raise
        except (TypeError, AttributeError, KeyError, IndexError, ValueError, OverflowError) as exc:
            raise CompileError(f"invalid IR: {exc}") from exc


    def _compile(self) -> str:
        self._emit_params()
        self._emit_smem_bases()
        self._emit_accumulator_seeds()

        body = self.op.body
        tile = None
        prelude = []
        trailing = []
        for st in body:
            if stmt_name(st) == "Tile" and tile is None:
                tile = st
            elif tile is None:
                prelude.append(st)
            else:
                trailing.append(st)
        if tile is None:
            raise CompileError("kernel body has no Tile statement")
        if trailing:
            names = ", ".join(stmt_name(st) for st in trailing)
            raise CompileError(
                f"kernel body has statements after the Tile, which would be dropped: {names}")

        if prelude:
            self.walk(prelude)
        self._emit_tile(tile)
        self.em.emit("ret;")

        out = [".version 8.0", f".target {self._target()}", ".address_size 64", ""]
        out += self._smem_declarations()
        out.append("")
        out.append(self._signature())
        out.append("{")
        out += self.em.declarations()
        out += self.em.literal_block()
        out += self.em.body
        out.append("}")
        return "\n".join(out) + "\n"


    def _target(self) -> str:
        arch = self.meta.get("arch") or "sm_89"
        return arch


    def _ordered_args(self) -> list[dict]:
        """Arguments in launch order (``arg_order`` is the contract)."""
        by_name = {arg["name"]: arg for arg in self.meta.get("args", [])}
        order = self.meta.get("arg_order")
        if not order:
            return list(self.meta.get("args", []))
        missing = [name for name in order if name not in by_name]
        if missing:
            raise CompileError(
                f"meta.json arg_order names unknown argument(s): {', '.join(missing)}")
        return [by_name[name] for name in order]


    def _signature(self) -> str:
        params = []
        for arg in self._ordered_args():
            pname = "param_" + _ident(arg["name"])
            if arg.get("scalar"):
                params.append(f"        .param .{ptx_type(arg.get('dtype', 'f32'))} {pname}")
            else:
                params.append(f"        .param .u64 {pname}")
        block = [int(x) for x in self.meta["block"]][:3]
        while len(block) < 3:
            block.append(1)
        launch = f"\n.maxntid {block[0]}, {block[1]}, {block[2]}"
        if not params:
            return f".visible .entry {self.name}()" + launch
        return f".visible .entry {self.name}(\n" + ",\n".join(params) + "\n)" + launch


    def _emit_params(self) -> None:
        for arg in self._ordered_args():
            name = arg["name"]
            pname = "param_" + _ident(name)
            dt = norm_dtype(arg.get("dtype", "f32"))
            if arg.get("scalar"):
                cls = reg_class(dt)
                reg = self.em.reg(cls)
                self.em.emit(f"ld.param.{ptx_type(dt)} {reg}, [{pname}];")
                self.scalars[name] = (reg, dt)
            else:
                raw = self.em.rd()
                self.em.emit(f"ld.param.u64 {raw}, [{pname}];")
                ptr = self.em.rd()
                self.em.emit(f"cvta.to.global.u64 {ptr}, {raw};")
                self.params[name] = ptr


    def _emit_accumulator_seeds(self) -> None:
        for acc in self.top_seeds:
            self._seed_accumulator(acc)


    def _seed_accumulator(self, acc) -> None:
        dt = norm_dtype(getattr(acc, "dtype", None) or "f32")
        ident = IDENTITY.get(_op_name(acc.op), 0.0)
        self.values[acc.name] = self._seed_value(dt, ident)


    def _seed_value(self, dt: str, value):
        _check_dtype(dt, "accumulator seed")
        if dt == "f16":
            tmp = self.em.f()
            self.em.emit(f"mov.f32 {tmp}, {f32_bits(float(value))};")
            reg = self.em.h()
            self.em.emit(f"cvt.rn.f16.f32 {reg}, {tmp};")
            return (reg, "f16")
        if dt == "f64":
            import struct
            reg = self.em.d()
            self.em.emit(f"mov.f64 {reg}, 0d{struct.unpack('<Q', struct.pack('<d', float(value)))[0]:016X};")
            return (reg, "f64")
        if dt in FLOAT_KINDS:
            return (self.em.mov_f32(float(value)), "f32")
        return (self.em.mov_s64(int(value)), "i64")


    def _emit_thread_helpers(self) -> None:
        """Hoist lane / warp out of any branch: Emmy declares them at kernel scope."""
        em = self.em
        want_tid = any(self.uses.get(n, 0) for n in ("tid", "thread_id"))
        want_lane = any(self.uses.get(n, 0) for n in ("lane", "lane_id"))
        want_warp = any(self.uses.get(n, 0) for n in ("warp", "warp_id"))
        if not (want_tid or want_lane or want_warp):
            return
        tid = em.r()
        em.emit(f"mov.u32 {tid}, %tid.x;")
        self.values.setdefault("tid", (tid, "u32"))
        self.values.setdefault("thread_id", (tid, "u32"))
        if want_lane:
            lane = em.r()
            em.emit(f"and.b32 {lane}, {tid}, 31;")
            self.values.setdefault("lane", (lane, "u32"))
            self.values.setdefault("lane_id", (lane, "u32"))
        if want_warp:
            warp = em.r()
            em.emit(f"shr.u32 {warp}, {tid}, 5;")
            self.values.setdefault("warp", (warp, "u32"))
            self.values.setdefault("warp_id", (warp, "u32"))


    def _has_block_collective(self, stmts) -> bool:
        """Whether a subtree executes a block-wide combine or barrier."""
        for st in stmts:
            name = stmt_name(st)
            if name == "TreeHalve":
                return True
            if name == "Sync" and not getattr(st, "warp", False):
                return True
            for attr in ("body", "else_body"):
                child = getattr(st, attr, None)
                if isinstance(child, (tuple, list)) and child:
                    if self._has_block_collective(child):
                        return True
        return False


    def _has_warp_collective(self, stmts) -> bool:
        """Whether a subtree executes a warp-wide shuffle or warp barrier."""
        for st in stmts:
            name = stmt_name(st)
            if name == "WarpShuffle":
                return True
            if name == "Sync" and getattr(st, "warp", False):
                return True
            for attr in ("body", "else_body"):
                child = getattr(st, attr, None)
                if isinstance(child, (tuple, list)) and child:
                    if self._has_warp_collective(child):
                        return True
        return False


    def _emit_tile(self, tile) -> None:
        self._emit_thread_helpers()
        axes = []
        for axis in tile.axes:
            try:
                ext = int(axis.extent.value)
            except (AttributeError, TypeError, ValueError):
                raise CompileError(
                    f"Tile axis {axis.name!r} has no constant extent; static shapes only")
            axes.append((axis.name, ext))

        gid = self._gid()
        n = 1
        for _, ext in axes:
            n *= ext

        done = self.em.new_label()
        p = self.em.p()

        suffix = [1] * len(axes)
        for i in range(len(axes) - 2, -1, -1):
            suffix[i] = suffix[i + 1] * axes[i + 1][1]
        for name, ext in axes:
            self.axis_extent[name] = ext
        threads = 1
        for b in self.meta["block"]:
            threads *= int(b)
        self.warp_axes = {name for (name, ext), suf in zip(axes, suffix)
                          if ext > 1 and (suf < 32 or suf % 32)}
        self.axis_suffix = {name: suf for (name, ext), suf in zip(axes, suffix)}
        grid = [int(x) for x in self.meta["grid"]]
        block = [int(x) for x in self.meta["block"]]
        launch = threads * grid[0] * grid[1] * grid[2]
        # On very small tile grids the address cache perturbs ptxas scheduling
        # more than the saved arithmetic helps; larger grids amortize it.
        self.cse_enabled = n >= 512
        if n < launch:
            # The guard skips part of a block only when the partial block is
            # the one containing tile n.  Extra fully inactive blocks are safe.
            if n % threads and self._has_block_collective(tile.body):
                raise CompileError(
                    f"block collective under a partial block: {n} tiles, "
                    f"{threads} threads per block, {launch} launched. "
                    "The gid guard would skip part of the barrier.")
            if n % 32 and self._has_warp_collective(tile.body):
                raise CompileError(
                    f"warp collective under a partial warp: {n} tiles, "
                    f"{launch} launched. The full mask would include inactive lanes.")
        self.full_tile = (n == grid[0] * grid[1] * grid[2] * threads
                          and int(block[0]) % 32 == 0)
        single = len(axes) == 1 and axes[0][1] > 1
        for i, (name, ext) in enumerate(axes):
            if single:
                self.axes[name] = gid
                continue
            stride = suffix[i]
            cur = gid
            # The shift/mask form pays off when the tile grid is large: the
            # decode then runs for many threads.  Below this count the shorter
            # form perturbs ptxas scheduling on small attention blocks, so the
            # divide and remainder stay.
            short_decode = n >= 1024
            if stride != 1:
                cur = self.em.r()
                if stride & (stride - 1) or not short_decode:
                    self.em.emit(f"div.u32 {cur}, {gid}, {stride};")
                else:
                    self.em.emit(f"shr.u32 {cur}, {gid}, {_log2(stride)};")
            if ext == 1:
                self.axes[name] = self.em.mov_u32(0)
            else:
                nxt = self.em.r()
                if ext & (ext - 1) or not short_decode:
                    self.em.emit(f"rem.u32 {nxt}, {cur}, {ext};")
                else:
                    self.em.emit(f"and.b32 {nxt}, {cur}, {ext - 1};")
                self.axes[name] = nxt

        self._emit_cache_fills()
        # When the launch covers the tile grid exactly, every thread owns a
        # tile and the work-id comparison is provably true.  Skipping it drops
        # a predicate and a branch from every kernel with an exact launch.
        guarded = n < launch
        if guarded:
            self.em.emit(f"setp.ge.u32 {p}, {gid}, {n};")
            self.em.emit(f"@{p} bra {done};")

        self.walk(tile.body)
        if guarded:
            self.em.place(done)


    def _gid(self) -> str:
        em = self.em
        grid = [int(x) for x in self.meta["grid"]][:3]
        block = [int(x) for x in self.meta["block"]][:3]
        while len(grid) < 3:
            grid.append(1)
        while len(block) < 3:
            block.append(1)
        if grid[1] == 1 and grid[2] == 1 and block[1] == 1 and block[2] == 1:
            # Every launch in this task is one-dimensional; the flattened work
            # id is blockIdx.x * blockDim.x + threadIdx.x and the other eight
            # special-register reads are dead weight in the prologue.
            tx = em.r()
            em.emit(f"mov.u32 {tx}, %tid.x;")
            bx = em.r()
            em.emit(f"mov.u32 {bx}, %ctaid.x;")
            bdx = em.r()
            em.emit(f"mov.u32 {bdx}, %ntid.x;")
            gid = em.r()
            em.emit(f"mad.lo.u32 {gid}, {bx}, {bdx}, {tx};")
            return gid
        tx, ty, tz = em.r(), em.r(), em.r()
        em.emit(f"mov.u32 {tx}, %tid.x;")
        em.emit(f"mov.u32 {ty}, %tid.y;")
        em.emit(f"mov.u32 {tz}, %tid.z;")
        bx, by, bz = em.r(), em.r(), em.r()
        em.emit(f"mov.u32 {bx}, %ctaid.x;")
        em.emit(f"mov.u32 {by}, %ctaid.y;")
        em.emit(f"mov.u32 {bz}, %ctaid.z;")
        bdx, bdy = em.r(), em.r()
        em.emit(f"mov.u32 {bdx}, %ntid.x;")
        em.emit(f"mov.u32 {bdy}, %ntid.y;")
        gdx, gdy = em.r(), em.r()
        em.emit(f"mov.u32 {gdx}, %nctaid.x;")
        em.emit(f"mov.u32 {gdy}, %nctaid.y;")

        tflat = em.r()
        em.emit(f"mad.lo.u32 {tflat}, {tz}, {bdy}, {ty};")
        em.emit(f"mad.lo.u32 {tflat}, {tflat}, {bdx}, {tx};")

        bflat = em.r()
        em.emit(f"mad.lo.u32 {bflat}, {bz}, {gdy}, {by};")
        em.emit(f"mad.lo.u32 {bflat}, {bflat}, {gdx}, {bx};")

        gid = em.r()
        em.emit(f"mad.lo.u32 {gid}, {bflat}, {bdx}, {tflat};")
        # recover the block's thread count in x*y*z: bdx*bdy*(blockDim.z)
        bdz = em.r()
        em.emit(f"mov.u32 {bdz}, %ntid.z;")
        prod = em.r()
        em.emit(f"mul.lo.u32 {prod}, {bdx}, {bdy};")
        em.emit(f"mul.lo.u32 {prod}, {prod}, {bdz};")
        out = em.r()
        em.emit(f"mad.lo.u32 {out}, {bflat}, {prod}, {tflat};")
        return out
