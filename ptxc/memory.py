"""Memory layout and addressing: shared slabs, flat indices, addresses, alignment."""

from __future__ import annotations

from .common import *
from .ops import elem_bytes, norm_dtype, ptx_type



class MemoryMixin:
    def _is_written(self, name: str) -> bool:
        return name in self.written_globals

    def buffer_dtype(self, name: str) -> str:
        if name in self.smem:
            return self.smem[name]["dtype"]
        return self.dtypes.get(name, "f32")


    def shape_of(self, name: str) -> tuple[int, ...]:
        if name in self.smem:
            return tuple(self.smem[name]["extents"])
        return self.shapes.get(name, ())


    def _scan_smem(self, stmts) -> None:
        for st in stmts:
            if stmt_name(st) == "Smem":
                dt = norm_dtype(st.dtype)
                _check_dtype(dt, f"Smem {st.name!r}")
                extents = tuple(int(d.value if hasattr(d, "value") else d) for d in st.extents)
                natural = elem_bytes(dt)
                align = max(int(st.align or 0), natural)
                self.smem[st.name] = {
                    "extents": extents,
                    "dtype": dt,
                    "align": align,
                    "offset": 0,
                }
                continue
            for attr in ("body", "else_body"):
                child = getattr(st, attr, None)
                if isinstance(child, (tuple, list)) and child:
                    self._scan_smem(child)


    def _smem_base(self, name: str) -> str:
        return self.smem[name]["base"]


    def _emit_smem_bases(self) -> None:
        """Materialize every shared-array base in the prologue.

        Caching them at first use is a trap: the first use can sit inside a
        conditional or a short-running loop, leaving the register undefined for
        the threads that skipped it.
        """
        if not self.smem:
            return
        if self.smem_dynamic:
            self.smem_pool_base = self.em.r()
            self.em.emit(f"mov.u32 {self.smem_pool_base}, __emmy_smem;")
        for name, info in self.smem.items():
            base = self.em.r()
            if self.smem_dynamic:
                if info["offset"]:
                    self.em.emit(f"add.u32 {base}, {self.smem_pool_base}, {info['offset']};")
                else:
                    self.em.emit(f"mov.u32 {base}, {self.smem_pool_base};")
            else:
                self.em.emit(f"mov.u32 {base}, {name};")
            info["base"] = base


    def _smem_declarations(self) -> list[str]:
        if not self.smem or self.smem_dynamic:
            if self.smem_dynamic:
                return [".extern .shared .align 16 .b8 __emmy_smem[];"]
            return []
        return [self._static_smem_line(name, info) for name, info in self.smem.items()]


    def _static_smem_line(self, name: str, info: dict) -> str:
        size = elem_bytes(info["dtype"])
        for e in info["extents"]:
            size *= e
        return f".shared .align {info['align']} .b8 {name}[{size}];"


    def _load_consecutive(self, names, addr, dt, esize, count, *, shared, shape, index) -> None:
        prefix = "ld.shared" if shared else getattr(self, "_global_load_prefix", "ld.global")
        if count in (2, 4) and dt == "f32" and self._aligned(index, shape, esize, count):
            regs = [self.em.f() for _ in range(count)]
            self.em.emit(f"{prefix}.v{count}.f32 {{{', '.join(regs)}}}, [{addr}];")
            for name, reg in zip(names, regs):
                self.values[name] = (reg, "f32")
            return
        for i, name in enumerate(names):
            if i:
                addr = self._next_addr(addr, esize, shared)
            reg = self.em.preg(dt)
            self.em.emit(f"{prefix}.{ptx_type(dt)} {reg}, [{addr}];")
            self.values[name] = (reg, dt)


    def _store_values(self, addr, values, dt, esize, count, shared, shape, index) -> None:
        prefix = "st.shared" if shared else "st.global"
        if count in (2, 4) and dt == "f32" and self._aligned(index, shape, esize, count):
            self.em.emit(f"{prefix}.v{count}.f32 [{addr}], {{{', '.join(values)}}};")
            return
        cur = addr
        for i, reg in enumerate(values):
            if i:
                cur = self._next_addr(cur, esize, shared)
            self.em.emit(f"{prefix}.{ptx_type(dt)} [{cur}], {reg};")


    def flat_axis_coefficient(self, index, shape, axis_name: str):
        """Constant coefficient of one loop axis in the flat index, or None.

        Used to turn an unrolled copy's address into a chunk base plus an
        immediate offset: ``addr + k * coef * esize``.
        """
        if index is None or len(index) == 0:
            return 0
        if len(index) == 1:
            return _axis_coefficient(index[0], axis_name)
        strides = _strides(shape, len(index))
        if strides[-2] > 1:
            row = _div_operand(index[-2], strides[-2])
            col = _mod_operand(index[-1], strides[-2])
            if row is not None and col is not None and (row is col or row == col):
                total = _axis_coefficient(row, axis_name)
                if total is None:
                    return None
                for coord, stride in zip(index[:-2], strides[:-2]):
                    coef = _axis_coefficient(coord, axis_name)
                    if coef is None:
                        return None
                    total += coef * stride
                return total
        total = 0
        for coord, stride in zip(index, strides):
            coef = _axis_coefficient(coord, axis_name)
            if coef is None:
                return None
            total += coef * stride
        return total

    def flat_index(self, index, shape, recombine: bool = False) -> str:
        """Row-major flat element index in 32-bit arithmetic.

        ``recombine`` folds a (x / cols, x % cols) row/column pair back into
        x.  Callers enable it only where the result feeds a chunk-address base,
        because the shorter form can perturb ptxas scheduling elsewhere.
        """
        if index is None or len(index) == 0:
            return self.em.mov_s32(0)
        if len(index) == 1:
            # emmy treats a single coordinate as the flat address, whatever
            # the buffer's declared rank.
            return self.idx32(index[0])
        strides = _strides(shape, len(index))
        if recombine and strides[-2] > 1:
            row = _div_operand(index[-2], strides[-2])
            col = _mod_operand(index[-1], strides[-2])
            if row is not None and col is not None and \
                    (row is col or row == col):
                acc = self.idx32(row)
                for coord, stride in zip(index[:-2], strides[:-2]):
                    reg = self.idx32(coord)
                    if stride != 1:
                        reg = self._s32_bin("mul.lo.s32", reg, self.em.mov_s32(stride))
                    acc = self._s32_bin("add.s32", acc, reg)
                return acc
        terms = []
        strides = _strides(shape, len(index))
        for coord, stride in zip(index, strides):
            reg = self.idx32(coord)
            if stride != 1:
                reg = self._s32_bin("mul.lo.s32", reg, self.em.mov_s32(stride))
            terms.append(reg)
        acc = terms[0]
        for reg in terms[1:]:
            acc = self._s32_bin("add.s32", acc, reg)
        return acc


    def _global_addr(self, ptr: str, flat: str, esize: int) -> str:
        off = self.em.rd()
        if esize == 1:
            self.em.emit(f"cvt.s64.s32 {off}, {flat};")
        else:
            self.em.emit(f"mul.wide.s32 {off}, {flat}, {esize};")
        out = self.em.rd()
        self.em.emit(f"add.s64 {out}, {ptr}, {off};")
        return out

    def _cse_global_addr(self, buffer, ptr, index, shape, esize):
        """A global address, reusing a base when only the constant offset differs.

        An unrolled matmul writes a tile of results from one row and column
        base, and every store rebuilt the whole address.  The key is the
        runtime part of the flat index plus the element size; the first access
        computes the address, later ones add an immediate.
        """
        if not getattr(self, "cse_enabled", True):
            return self._global_addr(ptr, self.flat_index(index, shape), esize)
        split = _split_flat(index, shape)
        if split is not None:
            key, const = split
            hit = self.addr_cse.get((buffer, esize, key))
            if hit is not None:
                delta = (const - hit[1]) * esize
                if delta >= 0:
                    return _addr_add(hit[0], delta)
                # ptxas only accepts [base+imm], so a lower address becomes
                # the new base below instead of a negative offset.
        flat = self.flat_index(index, shape)
        addr = self._global_addr(ptr, flat, esize)
        if split is not None and not any(
                name in self.accumulators for terms, _ in key for name, _ in terms):
            self.addr_cse[(buffer, esize, key)] = (addr, const)
        return addr


    def _shared_addr(self, base: str, flat: str, esize: int) -> str:
        out = self.em.r()
        if esize == 1:
            self.em.emit(f"add.u32 {out}, {base}, {flat};")
            return out
        scaled = self.em.r()
        if esize & (esize - 1) == 0:
            self.em.emit(f"shl.b32 {scaled}, {flat}, {_log2(esize)};")
        else:
            self.em.emit(f"mul.lo.s32 {scaled}, {flat}, {esize};")
        self.em.emit(f"add.u32 {out}, {base}, {scaled};")
        return out


    def _swizzle(self, flat: str, mode: str) -> str:
        """emmy's software smem XOR permutation: e ^ (((e >> shift) & mask) << 3)."""
        base = mode.partition("@")[0]
        shift, mask = {"B32": (1, 1), "B64": (1, 3), "B128": (1, 7)}.get(base, (0, 0))
        if "@" in mode:
            shift = int(mode.partition("@")[2])
        if not mask:
            return flat
        t = self.em.r()
        self.em.emit(f"shr.s32 {t}, {flat}, {shift};")
        self.em.emit(f"and.b32 {t}, {t}, {mask};")
        self.em.emit(f"shl.b32 {t}, {t}, 3;")
        out = self.em.r()
        self.em.emit(f"xor.b32 {out}, {flat}, {t};")
        return out


    def _scaled_offset(self, reg: str, esize: int) -> str:
        if esize == 1:
            return reg
        out = self.em.r()
        if esize & (esize - 1) == 0:
            self.em.emit(f"shl.b32 {out}, {reg}, {_log2(esize)};")
        else:
            self.em.emit(f"mul.lo.s32 {out}, {reg}, {esize};")
        return out


    def _scaled_offset_imm(self, reg: str, add: int, esize: int) -> str:
        out = self.em.r()
        self.em.emit(f"add.u32 {out}, {reg}, {add};")
        if esize != 1:
            self.em.emit(f"shl.b32 {out}, {out}, {_log2(esize)};")
        return out


    def _next_addr(self, addr: str, esize: int, shared: bool = False) -> str:
        if "+" in addr or "-" in addr[1:]:
            return _addr_add(addr, esize)
        if shared:
            # Shared-memory addresses are 32-bit; add.s64 with a 32-bit base
            # is rejected by ptxas.
            out = self.em.r()
            self.em.emit(f"add.u32 {out}, {addr}, {esize};")
            return out
        out = self.em.rd()
        self.em.emit(f"add.s64 {out}, {addr}, {esize};")
        return out


    def _aligned(self, index, shape, esize, width) -> bool:
        stride = esize * width
        need = max(1, stride // esize)
        terms = []
        strides = _strides(shape, len(index))
        for coord, s in zip(index, strides):
            aff = _affine(coord)
            if aff is None:
                return False
            terms.append((aff, s))
        const = 0
        coeffs: dict[str, int] = {}
        for (c, co), s in terms:
            const += c * s
            for name, v in co.items():
                coeffs[name] = coeffs.get(name, 0) + v * s
        if const % need:
            return False
        return all(v % need == 0 for v in coeffs.values())


def _div_operand(e, divisor):
    """The dividend when ``e`` is ``x / divisor``, else None."""
    from emmy.compiler.ir.expr import BinaryExpr

    if not isinstance(e, BinaryExpr) or e.op not in ("/", "//", "floor_divide"):
        return None
    if _const_int(e.right) != divisor:
        return None
    return e.left


def _mod_operand(e, divisor):
    """The dividend when ``e`` is ``x % divisor``, else None."""
    from emmy.compiler.ir.expr import BinaryExpr

    if not isinstance(e, BinaryExpr) or e.op not in ("%", "mod", "remainder"):
        return None
    if _const_int(e.right) != divisor:
        return None
    return e.left


def _split_flat(index, shape):
    """(runtime key, constant) for a flat index that is affine in every coord.

    Two accesses whose keys match address the same base; only the constant
    part differs, so the second can reuse the first address with an immediate
    offset.  Returns None when any coordinate is not affine, which keeps
    div/mod and opaque expressions out of the cache.
    """
    if index is None:
        return None
    strides = _strides(shape, len(index))
    const = 0
    key = []
    for coord, stride in zip(index, strides):
        aff = _affine(coord)
        if aff is None:
            return None
        c, vars_ = aff
        terms = tuple(sorted((name, coef) for name, coef in vars_.items() if coef))
        const += c * stride
        key.append((terms, stride))
    return tuple(key), const
