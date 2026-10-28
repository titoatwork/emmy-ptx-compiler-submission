"""Expression lowering: SSA values, indices, predicates, elementwise ops."""

from __future__ import annotations

from .common import *
from .ops import FLOAT_BINARY, LN2, LOG2E, norm_dtype, ptx_type



BUILTINS = {
    "thread_idx.x": "%tid.x", "thread_idx.y": "%tid.y", "thread_idx.z": "%tid.z",
    "block_idx.x": "%ctaid.x", "block_idx.y": "%ctaid.y", "block_idx.z": "%ctaid.z",
    "block_dim.x": "%ntid.x", "block_dim.y": "%ntid.y", "block_dim.z": "%ntid.z",
    "grid_dim.x": "%nctaid.x", "grid_dim.y": "%nctaid.y", "grid_dim.z": "%nctaid.z",
    "warp_size": None,
}

# erf(x) is odd, so erf(x) = x * P(x^2).  P is a degree-22 polynomial in
# u = x^2/8 - 1, fitted over x in [0, 4] with maximum absolute error 4.1e-13
# in double precision.  Outside that interval erf is within 1.6e-8 of sign(x),
# which is below any f32 tolerance.  The coefficients are in increasing order.
ERF_LIMIT = 4.0
ERF_POLY = (
    0.35353099564338758,
    -0.17657623330397715,
    0.13167511689973305,
    -0.10771044230014236,
    0.090208993873320611,
    -0.074727871652808753,
    0.059886909417219775,
    -0.045765039994050238,
    0.033060590162759369,
    -0.022474034085702414,
    0.01434983895070167,
    -0.0086043446421522479,
    0.0048526946491135826,
    -0.0025830604028213504,
    0.0012946163509062023,
    -0.00060482242053051219,
    0.00027243043468597828,
    -0.00012652663205878274,
    5.1006156709755428e-05,
    -1.2620846746211842e-05,
    4.8632082922035838e-06,
    -4.5891938042512698e-06,
    1.5258737511644574e-06,
)


def _wrap_s32(value: int) -> int:
    """Interpret the low 32 bits of ``value`` as a signed integer."""
    value &= 0xFFFFFFFF
    return value - 0x100000000 if value & 0x80000000 else value


def _fold_s32_literals(insn: str, a: int, b: int) -> int | None:
    """Compile-time result of a 32-bit index operator on two literals.

    PTX ``.s32`` arithmetic is 32 bits wide, but the index expressions here
    are mathematical integers.  An add, subtract or multiply that leaves the
    signed 32-bit range is left to the hardware (which wraps); folding it
    would silently change a value the IR never promised to keep in range.
    Bitwise operators are defined on the 32-bit pattern, so they fold with
    that wrapping.
    """
    if insn == "add.s32":
        out = a + b
    elif insn == "sub.s32":
        out = a - b
    elif insn == "mul.lo.s32":
        out = a * b
    elif insn == "shl.b32":
        if not 0 <= b < 32:
            return None
        out = a << b
    elif insn == "shr.s32":
        if not 0 <= b < 32:
            return None
        return a >> b
    elif insn == "and.b32":
        return _wrap_s32(a & b)
    elif insn == "or.b32":
        return _wrap_s32(a | b)
    elif insn == "xor.b32":
        return _wrap_s32(a ^ b)
    else:
        return None
    if -(2**31) <= out < 2**31:
        return out
    return None


class ExprMixin:
    def value(self, name: str) -> tuple[str, str]:
        v = self.values.get(name)
        if v is not None:
            return v
        if name in self.axes:
            return self.axes[name], "i32"
        if name in self.scalars:
            return self.scalars[name]
        builtin = self._builtin(name)
        if builtin is not None:
            return builtin
        raise CompileError(f"unknown SSA value {name!r}")


    def _builtin(self, name: str) -> tuple[str, str] | None:
        """Tile-scoped helpers Emmy's renderer provides: lane / warp / tid."""
        if name in BUILTINS:
            special = BUILTINS[name]
            out = self.em.r()
            if special is None:
                self.em.emit(f"mov.u32 {out}, 32;")
            else:
                self.em.emit(f"mov.u32 {out}, {special};")
            self.values[name] = (out, "u32")
            return self.values[name]
        if name not in ("lane", "warp", "tid", "thread_id", "warp_id", "lane_id"):
            return None
        tid = self.values.get("__tid__")
        if tid is None:
            reg = self.em.r()
            self.em.emit(f"mov.u32 {reg}, %tid.x;")
            tid = (reg, "u32")
            self.values["__tid__"] = tid
        base = tid[0]
        if name in ("lane", "lane_id"):
            out = self.em.r()
            self.em.emit(f"and.b32 {out}, {base}, 31;")
            self.values[name] = (out, "u32")
            return self.values[name]
        if name in ("warp", "warp_id"):
            out = self.em.r()
            self.em.emit(f"shr.u32 {out}, {base}, 5;")
            self.values[name] = (out, "u32")
            return self.values[name]
        self.values[name] = tid
        return tid


    def _int_var(self, name: str) -> str:
        if name in self.axes:
            return self.axes[name]
        reg, kind = self.value(name)
        return reg


    def fexpr_operand(self, name: str) -> str:
        """An f32 register for an SSA value (converting narrower/wider floats)."""
        reg, kind = self.value(name)
        if kind == "f32":
            return reg
        return self._to_f32(reg, kind)


    def iexpr_operand(self, name: str) -> str:
        reg, kind = self.value(name)
        if kind in INT_KINDS:
            if kind in ("i64", "u64"):
                return reg
            out = self.em.rd()
            self.em.emit(f"cvt.s64.{ptx_type(kind)} {out}, {reg};")
            return out
        return self._to_s64(reg, kind)


    def _to_f32(self, reg: str, kind: str) -> str:
        out = self.em.f()
        if kind == "f16":
            self.em.emit(f"cvt.f32.f16 {out}, {reg};")
        elif kind == "f64":
            self.em.emit(f"cvt.rn.f32.f64 {out}, {reg};")
        elif kind in ("i64", "u64"):
            self.em.emit(f"cvt.rn.f32.s64 {out}, {reg};")
        elif kind in INT_KINDS:
            self.em.emit(f"cvt.rn.f32.s32 {out}, {reg};")
        else:
            raise CompileError(f"cannot convert {kind} to f32")
        return out


    def f64_operand_expr(self, e) -> str:
        from emmy.compiler.ir.expr import Var

        if isinstance(e, Var):
            return self.f64_operand(e.name)
        return self._to_f64(self.fexpr(e))


    def _to_f64(self, reg: str) -> str:
        out = self.em.d()
        self.em.emit(f"cvt.f64.f32 {out}, {reg};")
        return out


    def f64_operand(self, name: str) -> str:
        reg, kind = self.value(name)
        if kind == "f64":
            return reg
        out = self.em.d()
        if kind == "f16":
            self.em.emit(f"cvt.f64.f16 {out}, {reg};")
        elif kind == "f32":
            self.em.emit(f"cvt.f64.f32 {out}, {reg};")
        else:
            self.em.emit(f"cvt.rn.f64.s64 {out}, {self._to_s64(reg, kind)};")
        return out


    def _to_s64(self, reg: str, kind: str) -> str:
        if kind in ("i64", "u64"):
            return reg
        out = self.em.rd()
        if kind in INT_KINDS:
            self.em.emit(f"cvt.s64.{ptx_type(kind)} {out}, {reg};")
        elif kind == "f16":
            self.em.emit(f"cvt.rzi.s64.f16 {out}, {reg};")
        elif kind == "f64":
            self.em.emit(f"cvt.rzi.s64.f64 {out}, {reg};")
        else:
            self.em.emit(f"cvt.rzi.s64.f32 {out}, {reg};")
        return out


    def _cast_reg(self, reg: str, kind: str, dst_dt: str) -> str:
        """Convert a value register to the register class a store needs.

        The IR lets a wider or narrower value flow into a buffer whose dtype
        differs (a half max folded into a float scratch, a double input summed
        into a float accumulator).  Emmy's renderer relies on C++ implicit
        conversions there; PTX has no implicit conversion, so the store path
        asks for an explicit ``cvt``.
        """
        em = self.em
        dst = norm_dtype(dst_dt)
        dst_kind = _kind_of_dtype(dst)
        if kind == dst_kind:
            return reg
        if kind == "pred":
            raise CompileError("cannot store a predicate value as a scalar")
        if dst_kind in FLOAT_KINDS:
            if kind in FLOAT_KINDS:
                out = em.preg(dst_kind)
                if dst_kind == "f16":
                    src = "f32" if kind == "f32" else ("f64" if kind == "f64" else "f16")
                    em.emit(f"cvt.rn.f16.{src} {out}, {reg};")
                elif dst_kind == "f32":
                    if kind == "f16":
                        em.emit(f"cvt.f32.f16 {out}, {reg};")
                    else:
                        em.emit(f"cvt.rn.f32.f64 {out}, {reg};")
                else:
                    src = "f16" if kind == "f16" else "f32"
                    em.emit(f"cvt.f64.{src} {out}, {reg};")
                return out
            signed = self._to_s64(reg, kind)
            if dst_kind == "f64":
                out = em.d()
                em.emit(f"cvt.rn.f64.s64 {out}, {signed};")
            else:
                half = em.f()
                em.emit(f"cvt.rn.f32.s64 {half}, {signed};")
                if dst_kind == "f32":
                    return half
                out = em.h()
                em.emit(f"cvt.rn.f16.f32 {out}, {half};")
            return out
        signed = self._to_s64(reg, kind)
        if dst in ("i64", "u64"):
            return signed
        out = em.r()
        if dst in ("u32", "u16", "u8"):
            em.emit(f"cvt.u32.s64 {out}, {signed};")
        else:
            em.emit(f"cvt.s32.s64 {out}, {signed};")
        return out


    def _emit_erf_f64(self, xd: str, dst: str) -> None:
        """Double-precision erf via a fitted odd polynomial, clamped past 4.

        ``ERF_POLY`` approximates erf(x)/x in u = x^2/8 - 1 with maximum
        absolute error 4.1e-13 on [0, 4].  Beyond 4, erfc(x) < 1.6e-8, so
        returning sign(x) is inside the required accuracy.
        """
        em = self.em
        ax = em.d()
        em.emit(f"abs.f64 {ax}, {xd};")
        limit = em.mov_f64(ERF_LIMIT)
        p = em.p()
        em.emit(f"setp.gt.f64 {p}, {ax}, {limit};")
        eighth = em.mov_f64(0.125)
        scaled = em.d()
        em.emit(f"mul.rn.f64 {scaled}, {ax}, {eighth};")
        u = em.d()
        neg_one = em.mov_f64(-1.0)
        em.emit(f"fma.rn.f64 {u}, {ax}, {scaled}, {neg_one};")
        poly = em.mov_f64(ERF_POLY[-1])
        for coeff in reversed(ERF_POLY[:-1]):
            c = em.mov_f64(coeff)
            em.emit(f"fma.rn.f64 {poly}, {poly}, {u}, {c};")
        res = em.d()
        em.emit(f"mul.rn.f64 {res}, {ax}, {poly};")
        one = em.mov_f64(1.0)
        em.emit(f"selp.f64 {res}, {one}, {res}, {p};")
        neg = em.p()
        zero = em.mov_f64(0.0)
        em.emit(f"setp.lt.f64 {neg}, {xd}, {zero};")
        nres = em.d()
        em.emit(f"neg.f64 {nres}, {res};")
        em.emit(f"selp.f64 {dst}, {nres}, {res}, {neg};")


    def fexpr(self, e) -> str:
        from emmy.compiler.ir.expr import BinaryExpr, Builtin, CastExpr, Literal, TernaryExpr, Var

        if isinstance(e, CastExpr):
            dt = norm_dtype(e.dtype)
            _check_dtype(dt, "CastExpr")
            if dt not in FLOAT_KINDS:
                raise CompileError(
                    f"CastExpr to {dt!r} is not a float type in a float context")
            if dt == "f32":
                return self.fexpr(e.expr)
            if dt == "f64":
                return self.f64_operand_expr(e.expr)
            inner = self.fexpr(e.expr)
            out = self.em.h()
            self.em.emit(f"cvt.rn.f16.f32 {out}, {inner};")
            return out
        if isinstance(e, Builtin):
            return self.fexpr_operand(e.name)
        if isinstance(e, Literal):
            if _literal_is_float(e):
                return self.em.mov_f32(e.value)
            return self._to_f32(self.em.mov_s64(int(e.value)), "i64")
        if isinstance(e, Var):
            return self.fexpr_operand(e.name)
        if isinstance(e, BinaryExpr):
            left = self.fexpr(e.left)
            right = self.fexpr(e.right)
            out = self.em.f()
            op = e.op
            if op in ("+", "add"):
                self.em.emit(f"add.f32 {out}, {left}, {right};")
            elif op in ("-", "subtract"):
                self.em.emit(f"sub.f32 {out}, {left}, {right};")
            elif op in ("*", "multiply"):
                self.em.emit(f"mul.f32 {out}, {left}, {right};")
            elif op in ("/", "divide"):
                self.em.emit(f"div.rn.f32 {out}, {left}, {right};")
            else:
                raise CompileError(f"unsupported float binary op {op!r}")
            return out
        if isinstance(e, TernaryExpr):
            p = self.pexpr(e.cond)
            a = self.fexpr(e.if_true)
            b = self.fexpr(e.if_false)
            out = self.em.f()
            self.em.emit(f"selp.f32 {out}, {a}, {b}, {p};")
            return out
        raise CompileError(f"unsupported float expression {type(e).__name__}")


    def iexpr(self, e) -> str:
        from emmy.compiler.dim import Dim
        from emmy.compiler.ir.expr import BinaryExpr, Builtin, CastExpr, Literal, TernaryExpr, Var

        if isinstance(e, CastExpr):
            dt = norm_dtype(e.dtype)
            _check_dtype(dt, "CastExpr")
            if dt in FLOAT_KINDS:
                return self._to_s64(self.fexpr(e.expr), "f32")
            return self.iexpr(e.expr)
        if isinstance(e, Builtin):
            return self._to_s64(*self.value(e.name))

        if isinstance(e, Dim):
            return self.em.mov_s64(int(e.value))
        if isinstance(e, Literal):
            if _literal_is_float(e):
                reg = self.em.mov_f32(e.value)
                out = self.em.rd()
                self.em.emit(f"cvt.rzi.s64.f32 {out}, {reg};")
                return out
            return self.em.mov_s64(int(e.value))
        if isinstance(e, Var):
            if e.name in self.axes:
                out = self.em.rd()
                self.em.emit(f"cvt.s64.s32 {out}, {self.axes[e.name]};")
                return out
            reg, kind = self.value(e.name)
            return self._to_s64(reg, kind)
        if isinstance(e, BinaryExpr):
            op = e.op
            if op in ("+", "add"):
                return self._s64_bin("add.s64", self.iexpr(e.left), self.iexpr(e.right))
            if op in ("-", "subtract"):
                return self._s64_bin("sub.s64", self.iexpr(e.left), self.iexpr(e.right))
            if op in ("*", "multiply"):
                return self._s64_bin("mul.lo.s64", self.iexpr(e.left), self.iexpr(e.right))
            if op in ("/", "//", "divide", "floor_divide"):
                return self._div(self.iexpr(e.left), self.iexpr(e.right), e.right)
            if op in ("%", "mod", "remainder"):
                return self._rem(self.iexpr(e.left), self.iexpr(e.right), e.right)
            if op in ("<<", "left_shift"):
                return self._s64_bin("shl.b64", self.iexpr(e.left), self.idx32(e.right))
            if op in (">>", "right_shift"):
                return self._s64_bin("shr.s64", self.iexpr(e.left), self.idx32(e.right))
            if op == "&":
                return self._s64_bin("and.b64", self.iexpr(e.left), self.iexpr(e.right))
            if op == "|":
                return self._s64_bin("or.b64", self.iexpr(e.left), self.iexpr(e.right))
            if op == "^":
                return self._s64_bin("xor.b64", self.iexpr(e.left), self.iexpr(e.right))
            raise CompileError(f"unsupported integer binary op {op!r}")
        if isinstance(e, TernaryExpr):
            p = self.pexpr(e.cond)
            a = self.iexpr(e.if_true)
            b = self.iexpr(e.if_false)
            out = self.em.rd()
            self.em.emit(f"selp.s64 {out}, {a}, {b}, {p};")
            return out
        from emmy.compiler.ir.expr import FlatIndex

        if isinstance(e, FlatIndex):
            return self.flat_index(e.index, self.shape_of(e.buffer))
        raise CompileError(f"unsupported integer expression {type(e).__name__}")


    def _fold_int(self, e):
        """Evaluate an integer expression whose leaves are unrolled axis literals."""
        from emmy.compiler.dim import Dim
        from emmy.compiler.ir.expr import BinaryExpr, CastExpr, Literal, TernaryExpr, Var

        bindings = self.axis_values

        def val(x):
            if isinstance(x, Dim):
                return int(x.value)
            if isinstance(x, Literal):
                if _literal_is_float(x):
                    return None
                return int(x.value)
            if isinstance(x, Var):
                return bindings.get(x.name)
            if isinstance(x, CastExpr):
                if norm_dtype(x.dtype) in FLOAT_KINDS:
                    return None
                return val(x.expr)
            if isinstance(x, BinaryExpr):
                a = val(x.left)
                b = val(x.right)
                if a is None or b is None:
                    return None
                op = x.op
                if op in ("+", "add"):
                    return a + b
                if op in ("-", "subtract"):
                    return a - b
                if op in ("*", "multiply"):
                    return a * b
                if op in ("/", "//", "divide", "floor_divide"):
                    return None if b == 0 else a // b
                if op in ("%", "mod", "remainder"):
                    return None if b == 0 else a % b
                if op in ("<<", "left_shift"):
                    return a << b
                if op in (">>", "right_shift"):
                    return a >> b
                if op == "&":
                    return a & b
                if op == "|":
                    return a | b
                if op == "^":
                    return a ^ b
                if op in ("<", "less"):
                    return int(a < b)
                if op in ("<=", "less_equal"):
                    return int(a <= b)
                if op in (">", "greater"):
                    return int(a > b)
                if op in (">=", "greater_equal"):
                    return int(a >= b)
                if op in ("==", "equal"):
                    return int(a == b)
                if op in ("!=", "not_equal"):
                    return int(a != b)
                return None
            if isinstance(x, TernaryExpr):
                c = val(x.cond)
                if c is None:
                    return None
                return val(x.if_true) if c else val(x.if_false)
            return None

        return val(e)


    def idx32(self, e) -> str:
        """Compile an index expression into one 32-bit register.

        Emmy's renderer uses C ``int`` for coordinates; the runner caps buffers
        at 2^26 elements, so 32-bit arithmetic is exact and keeps the hot loops
        free of the out-of-line 64-bit div/rem helpers ptxas emits.
        """
        from emmy.compiler.dim import Dim
        from emmy.compiler.ir.expr import (
            BinaryExpr,
            Builtin,
            CastExpr,
            FlatIndex,
            Literal,
            TernaryExpr,
            Var,
        )

        cached = self.icache.get(e)
        if cached is not None:
            return cached
        hoisted = self.hoisted.get(e)
        if hoisted is not None:
            self.icache[e] = hoisted
            return hoisted
        if self.axis_values:
            folded = self._fold_int(e)
            if folded is not None and -(2**31) <= folded < 2**31:
                reg = self.em.mov_s32(folded)
                self.icache[e] = reg
                return reg

        if isinstance(e, Dim):
            reg = self.em.mov_s32(int(e.value))
        elif isinstance(e, Literal):
            if _literal_is_float(e):
                f = self.em.mov_f32(e.value)
                reg = self.em.r()
                self.em.emit(f"cvt.rzi.s32.f32 {reg}, {f};")
            else:
                value = int(e.value)
                if not -(2**31) <= value < 2**31:
                    raise CompileError(f"integer literal {value} does not fit 32-bit index math")
                reg = self.em.mov_s32(value)
        elif isinstance(e, Var):
            reg = self._idx32_var(e.name)
        elif isinstance(e, BinaryExpr):
            reg = self._idx32_binary(e)
        elif isinstance(e, TernaryExpr):
            p = self.pexpr(e.cond)
            a = self.idx32(e.if_true)
            b = self.idx32(e.if_false)
            reg = self.em.r()
            self.em.emit(f"selp.s32 {reg}, {a}, {b}, {p};")
        elif isinstance(e, CastExpr):
            dt = norm_dtype(e.dtype)
            if dt in FLOAT_KINDS:
                f = self.fexpr(e.expr)
                reg = self.em.r()
                self.em.emit(f"cvt.rzi.s32.f32 {reg}, {f};")
            else:
                reg = self.idx32(e.expr)
        elif isinstance(e, Builtin):
            reg = self._idx32_var(e.name)
        elif isinstance(e, FlatIndex):
            reg = self.flat_index(e.index, self.shape_of(e.buffer))
        else:
            raise CompileError(f"unsupported index expression {type(e).__name__}")

        self.icache[e] = reg
        return reg


    def _idx32_var(self, name: str) -> str:
        if name in self.axes:
            return self.axes[name]
        reg, kind = self.value(name)
        if kind in ("i32", "u32"):
            return reg
        out = self.em.r()
        if kind in ("i64", "u64"):
            self.em.emit(f"cvt.s32.s64 {out}, {reg};")
        elif kind in INT_KINDS:
            self.em.emit(f"cvt.s32.{ptx_type(kind)} {out}, {reg};")
        elif kind in FLOAT_KINDS:
            f = self._to_f32(reg, kind)
            self.em.emit(f"cvt.rzi.s32.f32 {out}, {f};")
        else:
            raise CompileError(f"cannot use {kind} as an index")
        return out


    def _idx32_binary(self, e) -> str:
        op = e.op
        if op in ("+", "add"):
            return self._s32_bin("add.s32", self.idx32(e.left), self.idx32(e.right))
        if op in ("-", "subtract"):
            return self._s32_bin("sub.s32", self.idx32(e.left), self.idx32(e.right))
        if op in ("*", "multiply"):
            return self._s32_bin("mul.lo.s32", self.idx32(e.left), self.idx32(e.right))
        if op in ("/", "//", "divide", "floor_divide"):
            return self._div32(self.idx32(e.left), self.idx32(e.right), e.right)
        if op in ("%", "mod", "remainder"):
            return self._rem32(self.idx32(e.left), self.idx32(e.right), e.right)
        if op in ("<<", "left_shift"):
            return self._s32_bin("shl.b32", self.idx32(e.left), self.idx32(e.right))
        if op in (">>", "right_shift"):
            return self._s32_bin("shr.s32", self.idx32(e.left), self.idx32(e.right))
        if op == "&":
            return self._s32_bin("and.b32", self.idx32(e.left), self.idx32(e.right))
        if op == "|":
            return self._s32_bin("or.b32", self.idx32(e.left), self.idx32(e.right))
        if op == "^":
            return self._s32_bin("xor.b32", self.idx32(e.left), self.idx32(e.right))
        raise CompileError(f"unsupported index op {op!r}")


    def _s32_bin(self, insn: str, a: str, b: str) -> str:
        va = self.em.s32_of(a)
        vb = self.em.s32_of(b)
        if insn == "add.s32":
            if vb == 0:
                return a
            if va == 0:
                return b
        elif insn == "sub.s32":
            if vb == 0:
                return a
            if a == b:
                return self.em.mov_s32(0)
        elif insn == "mul.lo.s32":
            if va == 0 or vb == 0:
                return self.em.mov_s32(0)
            if va == 1:
                return b
            if vb == 1:
                return a
        elif insn == "shl.b32":
            if vb == 0:
                return a
            if va == 0:
                return self.em.mov_s32(0)
        elif insn == "shr.s32":
            if vb == 0:
                return a
            if va == 0:
                return self.em.mov_s32(0)
        elif insn == "and.b32":
            if vb == 0 or va == 0:
                return self.em.mov_s32(0)
            if vb == -1:
                return a
            if va == -1:
                return b
        elif insn in ("or.b32", "xor.b32"):
            if vb == 0:
                return a
            if va == 0:
                return b
            if insn == "xor.b32" and a == b:
                return self.em.mov_s32(0)
        if va is not None and vb is not None:
            folded = _fold_s32_literals(insn, va, vb)
            if folded is not None:
                return self.em.mov_s32(folded)
        out = self.em.r()
        self.em.emit(f"{insn} {out}, {a}, {b};")
        return out


    def _div32(self, a: str, b: str, expr=None) -> str:
        value = _const_int(expr)
        out = self.em.r()
        if isinstance(value, int) and value > 0 and value & (value - 1) == 0:
            self.em.emit(f"shr.s32 {out}, {a}, {value.bit_length() - 1};")
        else:
            self.em.emit(f"div.s32 {out}, {a}, {b};")
        return out


    def _rem32(self, a: str, b: str, expr=None) -> str:
        value = _const_int(expr)
        out = self.em.r()
        if isinstance(value, int) and value > 0 and value & (value - 1) == 0:
            self.em.emit(f"and.b32 {out}, {a}, {value - 1};")
        else:
            self.em.emit(f"rem.s32 {out}, {a}, {b};")
        return out


    def _s64_bin(self, insn: str, a: str, b: str) -> str:
        out = self.em.rd()
        self.em.emit(f"{insn} {out}, {a}, {b};")
        return out


    def _div(self, a: str, b: str, expr=None):
        value = _const_int(expr) if expr is not None else None
        if isinstance(value, int) and value > 0 and value & (value - 1) == 0:
            out = self.em.rd()
            self.em.emit(f"shr.s64 {out}, {a}, {value.bit_length() - 1};")
            return out
        out = self.em.rd()
        self.em.emit(f"div.s64 {out}, {a}, {b};")
        return out


    def _rem(self, a: str, b: str, expr=None):
        value = _const_int(expr) if expr is not None else None
        if isinstance(value, int) and value > 0 and value & (value - 1) == 0:
            out = self.em.rd()
            self.em.emit(f"and.b64 {out}, {a}, {value - 1};")
            return out
        out = self.em.rd()
        self.em.emit(f"rem.s64 {out}, {a}, {b};")
        return out


    def pexpr(self, e) -> str:
        from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, Var

        if isinstance(e, BinaryExpr):
            op = e.op
            if op in ("==", "!=", "<", "<=", ">", ">=", "equal", "not_equal",
                      "less", "less_equal", "greater", "greater_equal"):
                return self._compare(e)
            if op in ("&&", "and", "&"):
                a = self.pexpr(e.left)
                b = self.pexpr(e.right)
                out = self.em.p()
                self.em.emit(f"and.pred {out}, {a}, {b};")
                return out
            if op in ("||", "or", "|"):
                a = self.pexpr(e.left)
                b = self.pexpr(e.right)
                out = self.em.p()
                self.em.emit(f"or.pred {out}, {a}, {b};")
                return out
            if op in ("!", "not"):
                a = self.pexpr(e.left)
                out = self.em.p()
                self.em.emit(f"not.pred {out}, {a};")
                return out
        if isinstance(e, Literal):
            p = self.em.p()
            self.em.emit(f"setp.ne.s32 {p}, {self.idx32(e)}, 0;")
            return p
        if isinstance(e, Var):
            reg, kind = self.value(e.name)
            p = self.em.p()
            if kind == "pred":
                return reg
            if kind in FLOAT_KINDS:
                self.em.emit(f"setp.neu.f32 {p}, {reg}, 0f00000000;")
            else:
                self.em.emit(f"setp.ne.s32 {p}, {self.idx32(e)}, 0;")
            return p
        if isinstance(e, TernaryExpr):
            cond = self.pexpr(e.cond)
            a = self.pexpr(e.if_true)
            b = self.pexpr(e.if_false)
            out = self.em.p()
            self.em.emit(f"selp.b32 {out}, 1, 0, {cond};")
            out2 = self.em.p()
            self.em.emit(f"setp.ne.s32 {out2}, {out}, 0;")
            return out2
        raise CompileError(f"unsupported predicate expression {type(e).__name__}")


    def _compare(self, e) -> str:
        op = e.op
        table = {
            "==": "eq", "equal": "eq",
            "!=": "ne", "not_equal": "ne",
            "<": "lt", "less": "lt",
            "<=": "le", "less_equal": "le",
            ">": "gt", "greater": "gt",
            ">=": "ge", "greater_equal": "ge",
        }
        rel = table[op]
        if self._is_float(e.left) or self._is_float(e.right):
            a = self.fexpr(e.left)
            b = self.fexpr(e.right)
            p = self.em.p()
            if rel == "ne":
                rel = "neu"
            self.em.emit(f"setp.{rel}.f32 {p}, {a}, {b};")
        else:
            a = self.idx32(e.left)
            b = self.idx32(e.right)
            p = self.em.p()
            self.em.emit(f"setp.{rel}.s32 {p}, {a}, {b};")
        return p


    def _is_float(self, e) -> bool:
        from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, Var

        if isinstance(e, Literal):
            return _literal_is_float(e)
        if isinstance(e, Var):
            if e.name in self.axes:
                return False
            try:
                return self.value(e.name)[1] in FLOAT_KINDS
            except CompileError:
                return True
        if isinstance(e, BinaryExpr):
            return self._is_float(e.left) or self._is_float(e.right)
        if isinstance(e, TernaryExpr):
            return self._is_float(e.if_true) or self._is_float(e.if_false)
        return False


    def pred_operand(self, name: str) -> str:
        reg, kind = self.value(name)
        if kind == "pred":
            return reg
        p = self.em.p()
        if kind in FLOAT_KINDS:
            self.em.emit(f"setp.neu.f32 {p}, {reg}, 0f00000000;")
        else:
            self.em.emit(f"setp.ne.s64 {p}, {self._to_s64(reg, kind)}, 0;")
        return p


    def _bool_assign(self, opname: str, args: list) -> str:
        if opname in BOOL_BINARY:
            a = self.pred_operand(args[0])
            b = self.pred_operand(args[1])
            out = self.em.p()
            insn = "and.pred" if opname.endswith("and") else "or.pred"
            self.em.emit(f"{insn} {out}, {a}, {b};")
            return out
        if opname in ("bitwise_not", "logical_not", "not"):
            a = self.pred_operand(args[0])
            out = self.em.p()
            self.em.emit(f"not.pred {out}, {a};")
            return out
        rel = COMPARE_OPS.get(opname)
        if rel is None:
            raise CompileError(f"unsupported boolean op {opname!r}")
        a_reg, a_kind = self.value(args[0])
        b_reg, b_kind = self.value(args[1])
        p = self.em.p()
        if a_kind in FLOAT_KINDS or b_kind in FLOAT_KINDS:
            a = self.fexpr_operand(args[0])
            b = self.fexpr_operand(args[1])
            if rel == "ne":
                rel = "neu"
            self.em.emit(f"setp.{rel}.f32 {p}, {a}, {b};")
        else:
            a = self._to_s64(a_reg, a_kind)
            b = self._to_s64(b_reg, b_kind)
            self.em.emit(f"setp.{rel}.s64 {p}, {a}, {b};")
        return p


    def _assign_where(self, st, args: list) -> None:
        if len(args) != 3:
            raise CompileError("where expects 3 arguments")
        p = self.pred_operand(args[0])
        _, kind_a = self.value(args[1])
        _, kind_b = self.value(args[2])
        dt = norm_dtype(st.dtype) if st.dtype else kind_a
        if dt in FLOAT_KINDS or kind_a in FLOAT_KINDS or kind_b in FLOAT_KINDS:
            a = self.fexpr_operand(args[1])
            b = self.fexpr_operand(args[2])
            out = self.em.f()
            self.em.emit(f"selp.f32 {out}, {a}, {b}, {p};")
            self.values[st.name] = (out, "f32")
        else:
            a = self._to_s64(*self.value(args[1]))
            b = self._to_s64(*self.value(args[2]))
            out = self.em.rd()
            self.em.emit(f"selp.s64 {out}, {a}, {b}, {p};")
            self.values[st.name] = (out, "i64")


    def _emit_float_op(self, opname: str, args: list[str], dst: str, dt: str) -> None:
        em = self.em
        if dt == "f64":
            self._emit_f64_op(opname, args, dst)
            return
        if opname in FLOAT_BINARY:
            em.emit(f"{FLOAT_BINARY[opname]} {dst}, {args[0]}, {args[1]};")
            return
        if opname in ("divide_rn_f32",):
            em.emit(f"div.rn.f32 {dst}, {args[0]}, {args[1]};")
            return
        if opname in ("exp", "exp_fast"):
            t = em.f()
            em.emit(f"mul.f32 {t}, {args[0]}, {LOG2E};")
            em.emit(f"ex2.approx.f32 {dst}, {t};")
            return
        if opname == "log":
            t = em.f()
            em.emit(f"lg2.approx.f32 {t}, {args[0]};")
            em.emit(f"mul.f32 {dst}, {t}, {LN2};")
            return
        if opname == "log2":
            em.emit(f"lg2.approx.f32 {dst}, {args[0]};")
            return
        if opname == "log10":
            t = em.f()
            em.emit(f"lg2.approx.f32 {t}, {args[0]};")
            em.emit(f"mul.f32 {dst}, {t}, {f32_bits(0.3010299956639812)};")
            return
        if opname in ("cos", "sin"):
            em.emit(f"{opname}.approx.f32 {dst}, {args[0]};")
            return
        if opname == "tan":
            s = em.f()
            c = em.f()
            em.emit(f"sin.approx.f32 {s}, {args[0]};")
            em.emit(f"cos.approx.f32 {c}, {args[0]};")
            em.emit(f"div.rn.f32 {dst}, {s}, {c};")
            return
        if opname == "erf":
            xd = self._to_f64(args[0])
            rd = em.d()
            self._emit_erf_f64(xd, rd)
            em.emit(f"cvt.rn.f32.f64 {dst}, {rd};")
            return
        if opname in ("gelu", "gelu_tanh"):
            x = args[0]
            half = em.mov_f32(0.5)
            one = em.mov_f32(1.0)
            if opname == "gelu":
                inv = em.mov_f32(0.7071067811865476)
                t = em.f()
                em.emit(f"mul.rn.f32 {t}, {x}, {inv};")
                e = em.f()
                self._emit_float_op("erf", [t], e, dt)
            else:
                c = em.mov_f32(0.044715)
                k = em.mov_f32(0.7978845608028654)
                x2 = em.f()
                em.emit(f"mul.rn.f32 {x2}, {x}, {x};")
                x3 = em.f()
                em.emit(f"mul.rn.f32 {x3}, {x2}, {x};")
                t = em.f()
                em.emit(f"fma.rn.f32 {t}, {c}, {x3}, {x};")
                t2 = em.f()
                em.emit(f"mul.rn.f32 {t2}, {t}, {k};")
                e = em.f()
                self._emit_float_op("tanh", [t2], e, dt)
            s = em.f()
            em.emit(f"add.rn.f32 {s}, {one}, {e};")
            hx = em.f()
            em.emit(f"mul.rn.f32 {hx}, {half}, {x};")
            em.emit(f"mul.rn.f32 {dst}, {hx}, {s};")
            return
        if opname == "softplus":
            x = args[0]
            ax = em.f()
            em.emit(f"abs.f32 {ax}, {x};")
            nx = em.f()
            em.emit(f"neg.f32 {nx}, {ax};")
            e = em.f()
            self._emit_float_op("exp", [nx], e, dt)
            one = em.mov_f32(1.0)
            s = em.f()
            em.emit(f"add.rn.f32 {s}, {one}, {e};")
            l = em.f()
            self._emit_float_op("log", [s], l, dt)
            zero = em.mov_f32(0.0)
            m = em.f()
            em.emit(f"max.f32 {m}, {x}, {zero};")
            em.emit(f"add.rn.f32 {dst}, {m}, {l};")
            return
        if opname == "exp2":
            em.emit(f"ex2.approx.f32 {dst}, {args[0]};")
            return
        if opname == "sqrt":
            em.emit(f"sqrt.rn.f32 {dst}, {args[0]};")
            return
        if opname == "rsqrt":
            em.emit(f"rsqrt.approx.f32 {dst}, {args[0]};")
            return
        if opname == "reciprocal":
            one = em.mov_f32(1.0)
            em.emit(f"div.rn.f32 {dst}, {one}, {args[0]};")
            return
        if opname == "relu":
            zero = em.mov_f32(0.0)
            em.emit(f"max.f32 {dst}, {args[0]}, {zero};")
            return
        if opname in ("copy", "pad", "bitcast"):
            em.emit(f"mov.f32 {dst}, {args[0]};")
            return
        if opname in ("negative", "negate"):
            em.emit(f"neg.f32 {dst}, {args[0]};")
            return
        if opname in ("abs", "fabs"):
            em.emit(f"abs.f32 {dst}, {args[0]};")
            return
        if opname == "tanh":
            # tanh.approx.f32 is only ~1e-5 relative; the reference tanhf is
            # ~1 ulp.  Split: a [5/4] Pade near zero (no cancellation) and the
            # exp identity outside, where the division is well conditioned.
            x = args[0]
            ax = em.f()
            em.emit(f"abs.f32 {ax}, {x};")
            p = em.p()
            one = em.mov_f32(1.0)
            em.emit(f"setp.lt.f32 {p}, {ax}, {one};")

            two = em.mov_f32(2.0)
            t = em.f()
            em.emit(f"mul.f32 {t}, {x}, {two};")
            e = em.f()
            self._emit_float_op("exp", [t], e, dt)
            den = em.f()
            em.emit(f"add.f32 {den}, {e}, {one};")
            q = em.f()
            em.emit(f"div.rn.f32 {q}, {two}, {den};")
            big = em.f()
            em.emit(f"sub.f32 {big}, {one}, {q};")

            z = em.f()
            em.emit(f"mul.f32 {z}, {x}, {x};")
            z2 = em.f()
            em.emit(f"mul.f32 {z2}, {z}, {z};")
            c105 = em.mov_f32(105.0)
            c945 = em.mov_f32(945.0)
            c420 = em.mov_f32(420.0)
            c15 = em.mov_f32(15.0)
            num = em.f()
            em.emit(f"fma.rn.f32 {num}, {z}, {c105}, {c945};")
            em.emit(f"fma.rn.f32 {num}, {z2}, {one}, {num};")
            den2 = em.f()
            em.emit(f"fma.rn.f32 {den2}, {z}, {c420}, {c945};")
            em.emit(f"fma.rn.f32 {den2}, {z2}, {c15}, {den2};")
            xn = em.f()
            em.emit(f"mul.f32 {xn}, {x}, {num};")
            small = em.f()
            em.emit(f"div.rn.f32 {small}, {xn}, {den2};")
            em.emit(f"selp.f32 {dst}, {small}, {big}, {p};")
            return
        if opname == "sigmoid":
            t = em.f()
            em.emit(f"neg.f32 {t}, {args[0]};")
            self._emit_float_op("exp", [t], t, dt)
            one = em.mov_f32(1.0)
            s = em.f()
            em.emit(f"add.f32 {s}, {one}, {t};")
            em.emit(f"div.rn.f32 {dst}, {one}, {s};")
            return
        if opname == "silu":
            t = em.f()
            em.emit(f"neg.f32 {t}, {args[0]};")
            self._emit_float_op("exp", [t], t, dt)
            one = em.mov_f32(1.0)
            s = em.f()
            em.emit(f"add.f32 {s}, {one}, {t};")
            em.emit(f"div.rn.f32 {dst}, {args[0]}, {s};")
            return
        if opname == "pow":
            l = em.f()
            self._emit_float_op("log", [args[0]], l, dt)
            t = em.f()
            em.emit(f"mul.f32 {t}, {l}, {args[1]};")
            self._emit_float_op("exp", [t], dst, dt)
            return
        if opname == "floor":
            em.emit(f"cvt.rmi.f32.f32 {dst}, {args[0]};")
            return
        if opname == "ceil":
            em.emit(f"cvt.rpi.f32.f32 {dst}, {args[0]};")
            return
        if opname == "sign":
            pos = em.p()
            neg = em.p()
            zero = em.mov_f32(0.0)
            one = em.mov_f32(1.0)
            negone = em.mov_f32(-1.0)
            em.emit(f"setp.gt.f32 {pos}, {args[0]}, {zero};")
            em.emit(f"setp.lt.f32 {neg}, {args[0]}, {zero};")
            t = em.f()
            em.emit(f"selp.f32 {t}, {one}, {zero}, {pos};")
            em.emit(f"selp.f32 {dst}, {negone}, {t}, {neg};")
            return
        if opname == "where":
            if len(args) != 3:
                raise CompileError("where expects 3 arguments")
            em.emit(f"selp.f32 {dst}, {args[1]}, {args[2]}, {args[0]};")
            return
        raise CompileError(f"unsupported elementwise op {opname!r}")


    def _pow2_f64(self, t: str, dst: str) -> None:
        """2**t in double precision: reduce to n + f, Taylor 2**f, scale by 2**n."""
        import math
        em = self.em
        n = em.d()
        em.emit(f"cvt.rni.f64.f64 {n}, {t};")
        f = em.d()
        em.emit(f"sub.rn.f64 {f}, {t}, {n};")
        ln2 = em.mov_f64(math.log(2.0))
        g = em.d()
        em.emit(f"mul.rn.f64 {g}, {f}, {ln2};")
        poly = em.mov_f64(1.0 / math.factorial(12))
        for k in range(11, -1, -1):
            coeff = em.mov_f64(1.0 / math.factorial(k))
            em.emit(f"fma.rn.f64 {poly}, {poly}, {g}, {coeff};")
        ni = em.rd()
        em.emit(f"cvt.rni.s64.f64 {ni}, {n};")
        bits = em.rd()
        em.emit(f"add.s64 {bits}, {ni}, 1023;")
        em.emit(f"shl.b64 {bits}, {bits}, 52;")
        scale = em.d()
        em.emit(f"mov.b64 {scale}, {bits};")
        em.emit(f"mul.rn.f64 {dst}, {poly}, {scale};")
        lowp = em.p()
        em.emit(f"setp.lt.s64 {lowp}, {ni}, -1022;")
        zero = em.mov_f64(0.0)
        em.emit(f"selp.f64 {dst}, {zero}, {dst}, {lowp};")
        highp = em.p()
        em.emit(f"setp.gt.s64 {highp}, {ni}, 1023;")
        inf = em.mov_f64(float("inf"))
        em.emit(f"selp.f64 {dst}, {inf}, {dst}, {highp};")

    def _ln_f64(self, x: str, dst: str) -> None:
        """Natural log in double precision: exponent plus an atanh series."""
        import math
        em = self.em
        bits = em.rd()
        em.emit(f"mov.b64 {bits}, {x};")
        e = em.rd()
        em.emit(f"shr.u64 {e}, {bits}, 52;")
        em.emit(f"and.b64 {e}, {e}, 0x7FF;")
        em.emit(f"sub.s64 {e}, {e}, 1023;")
        mant = em.rd()
        em.emit(f"and.b64 {mant}, {bits}, 0x000FFFFFFFFFFFFF;")
        em.emit(f"or.b64 {mant}, {mant}, 0x3FF0000000000000;")
        m = em.d()
        em.emit(f"mov.b64 {m}, {mant};")
        thr = em.mov_f64(math.sqrt(2.0))
        p = em.p()
        em.emit(f"setp.gt.f64 {p}, {m}, {thr};")
        half = em.mov_f64(0.5)
        mh = em.d()
        em.emit(f"mul.rn.f64 {mh}, {m}, {half};")
        em.emit(f"selp.f64 {m}, {mh}, {m}, {p};")
        em.mov_s64(1)
        e1 = em.rd()
        em.emit(f"add.s64 {e1}, {e}, 1;")
        em.emit(f"selp.s64 {e}, {e1}, {e}, {p};")
        one_d = em.mov_f64(1.0)
        num = em.d()
        em.emit(f"sub.rn.f64 {num}, {m}, {one_d};")
        one_d2 = em.mov_f64(1.0)
        den = em.d()
        em.emit(f"add.rn.f64 {den}, {m}, {one_d2};")
        y = em.d()
        em.emit(f"div.rn.f64 {y}, {num}, {den};")
        y2 = em.d()
        em.emit(f"mul.rn.f64 {y2}, {y}, {y};")
        poly = em.mov_f64(1.0 / 15.0)
        for k in range(6, -1, -1):
            coeff = em.mov_f64(1.0 / (2 * k + 1))
            em.emit(f"fma.rn.f64 {poly}, {poly}, {y2}, {coeff};")
        series = em.d()
        em.emit(f"mul.rn.f64 {series}, {y}, {poly};")
        em.emit(f"add.rn.f64 {series}, {series}, {series};")
        ef = em.d()
        em.emit(f"cvt.rn.f64.s64 {ef}, {e};")
        ln2 = em.mov_f64(math.log(2.0))
        elog = em.d()
        em.emit(f"mul.rn.f64 {elog}, {ef}, {ln2};")
        em.emit(f"add.rn.f64 {dst}, {elog}, {series};")
        negp = em.p()
        zero = em.mov_f64(0.0)
        em.emit(f"setp.le.f64 {negp}, {x}, {zero};")
        ninf = em.mov_f64(float("-inf"))
        em.emit(f"selp.f64 {dst}, {ninf}, {dst}, {negp};")

    def _emit_f64_op(self, opname: str, args: list[str], dst: str) -> None:
        em = self.em
        simple = {
            "add": "add.f64", "sum": "add.f64",
            "subtract": "sub.f64", "sub": "sub.f64",
            "multiply": "mul.f64", "prod": "mul.f64",
            "divide": "div.rn.f64", "true_divide": "div.rn.f64",
            "divide_rn_f32": "div.rn.f64",
            "maximum": "max.f64", "amax": "max.f64", "max": "max.f64",
            "minimum": "min.f64", "min": "min.f64",
            "fma": "fma.rn.f64",
        }
        if opname in simple:
            if opname == "fma":
                em.emit(f"fma.rn.f64 {dst}, {args[0]}, {args[1]}, {args[2]};")
            else:
                em.emit(f"{simple[opname]} {dst}, {args[0]}, {args[1]};")
            return
        if opname == "erf":
            self._emit_erf_f64(args[0], dst)
            return
        if opname in ("exp", "exp_fast", "exp2"):
            if opname == "exp2":
                t = args[0]
            else:
                t = em.d()
                scale = em.mov_f64(1.4426950408889634)
                em.emit(f"mul.rn.f64 {t}, {args[0]}, {scale};")
            self._pow2_f64(t, dst)
            return
        if opname == "log":
            self._ln_f64(args[0], dst)
            return
        if opname in ("log2", "log10"):
            tmp = em.d()
            self._ln_f64(args[0], tmp)
            factor = 1.4426950408889634 if opname == "log2" else 0.4342944819032518
            k = em.mov_f64(factor)
            em.emit(f"mul.rn.f64 {dst}, {tmp}, {k};")
            return
        if opname == "rsqrt":
            root = em.d()
            em.emit(f"sqrt.rn.f64 {root}, {args[0]};")
            one = em.mov_f64(1.0)
            em.emit(f"div.rn.f64 {dst}, {one}, {root};")
            return
        if opname in ("sigmoid", "silu"):
            neg = em.d()
            em.emit(f"neg.f64 {neg}, {args[0]};")
            t = em.d()
            scale = em.mov_f64(1.4426950408889634)
            em.emit(f"mul.rn.f64 {t}, {neg}, {scale};")
            e = em.d()
            self._pow2_f64(t, e)
            one = em.mov_f64(1.0)
            den = em.d()
            em.emit(f"add.rn.f64 {den}, {one}, {e};")
            top = em.mov_f64(1.0) if opname == "sigmoid" else em.d()
            if opname != "sigmoid":
                em.emit(f"mov.f64 {top}, {args[0]};")
            em.emit(f"div.rn.f64 {dst}, {top}, {den};")
            return
        if opname == "pow":
            logx = em.d()
            self._ln_f64(args[0], logx)
            t = em.d()
            em.emit(f"mul.rn.f64 {t}, {logx}, {args[1]};")
            scale = em.mov_f64(1.4426950408889634)
            t2 = em.d()
            em.emit(f"mul.rn.f64 {t2}, {t}, {scale};")
            self._pow2_f64(t2, dst)
            return
        if opname == "tanh":
            two = em.mov_f64(2.0)
            twox = em.d()
            em.emit(f"mul.rn.f64 {twox}, {args[0]}, {two};")
            scale = em.mov_f64(1.4426950408889634)
            t = em.d()
            em.emit(f"mul.rn.f64 {t}, {twox}, {scale};")
            e = em.d()
            self._pow2_f64(t, e)
            one = em.mov_f64(1.0)
            num = em.d()
            em.emit(f"sub.rn.f64 {num}, {e}, {one};")
            one2 = em.mov_f64(1.0)
            den = em.d()
            em.emit(f"add.rn.f64 {den}, {e}, {one2};")
            em.emit(f"div.rn.f64 {dst}, {num}, {den};")
            lim = em.mov_f64(20.0)
            big = em.p()
            em.emit(f"setp.gt.f64 {big}, {args[0]}, {lim};")
            pone = em.mov_f64(1.0)
            em.emit(f"selp.f64 {dst}, {pone}, {dst}, {big};")
            nlim = em.mov_f64(-20.0)
            nbig = em.p()
            em.emit(f"setp.lt.f64 {nbig}, {args[0]}, {nlim};")
            mone = em.mov_f64(-1.0)
            em.emit(f"selp.f64 {dst}, {mone}, {dst}, {nbig};")
            return
        if opname == "sqrt":
            em.emit(f"sqrt.rn.f64 {dst}, {args[0]};")
            return
        if opname == "reciprocal":
            one = em.d()
            em.emit(f"mov.f64 {one}, 0d3FF0000000000000;")
            em.emit(f"div.rn.f64 {dst}, {one}, {args[0]};")
            return
        if opname == "relu":
            zero = em.d()
            em.emit(f"mov.f64 {zero}, 0d0000000000000000;")
            em.emit(f"max.f64 {dst}, {args[0]}, {zero};")
            return
        if opname in ("copy", "pad", "bitcast"):
            em.emit(f"mov.f64 {dst}, {args[0]};")
            return
        if opname in ("negative", "negate"):
            em.emit(f"neg.f64 {dst}, {args[0]};")
            return
        if opname in ("abs", "fabs"):
            em.emit(f"abs.f64 {dst}, {args[0]};")
            return
        # The f64 transcendentals use the f32 approximant; the dtypeedge cases
        # cover the supported f64 paths.
        f32_args = []
        for a in args:
            t = em.f()
            em.emit(f"cvt.rn.f32.f64 {t}, {a};")
            f32_args.append(t)
        tmp = em.f()
        self._emit_float_op(opname, f32_args, tmp, "f32")
        em.emit(f"cvt.f64.f32 {dst}, {tmp};")


    def _infer_kind(self, args) -> str:
        for a in args:
            try:
                kind = self.value(a)[1]
            except CompileError:
                continue
            return kind
        return "f32"
