"""Scalar tables and small IR-tree utilities shared by the lowering mixins."""

from __future__ import annotations

from .ops import DTYPES, norm_dtype


FLOAT_KINDS = {"f16", "f32", "f64"}

COMPARE_OPS = {
    "equal": "eq", "eq": "eq",
    "not_equal": "ne", "ne": "ne",
    "greater": "gt", "gt": "gt",
    "less": "lt", "lt": "lt",
    "greater_equal": "ge", "ge": "ge",
    "less_equal": "le", "le": "le",
}

BOOL_BINARY = {
    "bitwise_and": "and", "logical_and": "and", "and": "and",
    "bitwise_or": "or", "logical_or": "or", "or": "or",
}
INT_KINDS = {"i8", "u8", "i16", "u16", "i32", "u32", "i64", "u64"}



def stmt_name(stmt) -> str:
    return type(stmt).__name__


# ---------------------------------------------------------------------------
# helpers


def _count_refs(stmts, uses: dict[str, int]) -> None:
    """Count every read of an SSA name (used only to prove single-use products)."""
    import dataclasses

    from emmy.compiler.ir.expr import Var

    def bump(name):
        uses[name] = uses.get(name, 0) + 1

    def expr(e):
        if isinstance(e, Var):
            bump(e.name)
        if dataclasses.is_dataclass(e):
            for f in dataclasses.fields(e):
                v = getattr(e, f.name)
                if hasattr(v, "__dataclass_fields__"):
                    expr(v)
                elif isinstance(v, (tuple, list)):
                    for x in v:
                        if hasattr(x, "__dataclass_fields__"):
                            expr(x)

    for s in stmts:
        name = stmt_name(s)
        if name in ("Assign", "Accum"):
            if name == "Accum":
                for a in (s.value, s.base):
                    if a is not None:
                        bump(a)
            else:
                for a in s.args:
                    if isinstance(a, str):
                        bump(a)
        elif name == "Write":
            for v in s.values:
                bump(v)
            for e in s.index:
                expr(e)
        elif name in ("Loop", "StridedLoop"):
            for e in (getattr(s, "start", None), getattr(s, "step", None), getattr(s, "end", None)):
                if e is not None:
                    expr(e)
        elif name == "Cond":
            expr(s.cond)
        elif name == "Let":
            expr(s.value)
        for attr in ("body", "else_body", "combine_states"):
            child = getattr(s, attr, None)
            if isinstance(child, (tuple, list)) and child:
                _count_refs(child, uses)


def _accums_for_loop(loop_stmt, axis: str, inits) -> list:
    """Accum statements inside a loop's subtree that fold over this loop's axis.

    A nested loop may rebind the same axis name.  Its accumulators belong to
    that inner loop instance, not to the outer one, so the walk stops
    descending once the name is shadowed.
    """
    found: list = []

    def walk(stmts, shadowed=False):
        for s in stmts:
            name = stmt_name(s)
            if name == "Accum" and not shadowed:
                axes = tuple(getattr(s, "axes", ()) or ())
                if axes and axes[0] == axis and s.name not in inits:
                    found.append(s)
            child_shadowed = shadowed or (
                name in ("Loop", "StridedLoop") and getattr(s.axis, "name", None) == axis
            )
            for attr in ("body", "else_body"):
                child = getattr(s, attr, None)
                if isinstance(child, (tuple, list)) and child:
                    walk(child, child_shadowed)

    walk(getattr(loop_stmt, "body", ()))
    return found


class CompileError(Exception):
    pass


def _op_name(op) -> str:
    return getattr(op, "name", str(op))


def _ident(name: str) -> str:
    out = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
    if not out or out[0].isdigit():
        out = "_" + out
    return out


def _literal_is_float(e) -> bool:
    return str(getattr(e, "dtype", "")).lower() in ("float", "f32", "f64", "f16")


def _const_int(e):
    if e is None:
        return None
    value = getattr(e, "value", None)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _addr_add(addr: str, offset: int) -> str:
    """Add a constant to a PTX address operand, keeping it as [base+imm]."""
    head, base = addr, 0
    for sign in ("+", "-"):
        cut = addr.rfind(sign, 1)
        if cut > 0 and addr[cut + 1:].isdigit():
            head, base = addr[:cut], int(addr[cut:])
            break
    total = base + offset
    if total == 0:
        return head
    return f"{head}+{total}" if total > 0 else f"{head}{total}"


def _log2(n: int) -> int:
    return n.bit_length() - 1




UNROLL_MAX_EXTENT = 64
UNROLL_MAX_EXPANDED = 5000


def _expanded_weight(stmts, smem=()) -> int:
    total = 0
    for st in stmts:
        kind = stmt_name(st)
        if kind == "Loop":
            body = getattr(st, "body", ())
            inner = _expanded_weight(body, smem)
            extent = int(st.axis.extent.value)
            total += extent * inner if _should_unroll(st, smem) else inner + 1
        elif kind in ("StridedLoop", "Cond"):
            for attr in ("body", "else_body"):
                child = getattr(st, attr, None)
                if isinstance(child, (tuple, list)) and child:
                    total += _expanded_weight(child, smem) + 1
        else:
            total += 1
    return total


def _should_unroll(st, smem=()) -> bool:
    """Unroll a constant-trip `Loop` when the expanded body stays manageable.

    Only shared-memory bodies qualify.  A shared-memory load is low latency and
    the unrolled constant offsets fold into the instruction's immediate field;
    a global-memory body needs many loads in flight, and forcing it straight
    line regressed the attention kernel: the register budget and the
    scheduler's loop-level software pipelining both changed.
    """
    extent = int(st.axis.extent.value)
    if extent <= 1:
        return True
    if extent > UNROLL_MAX_EXTENT:
        return False
    body = getattr(st, "body", ())
    if _loaded_buffers(body) - set(smem):
        return False
    return extent * max(1, _expanded_weight(body, smem)) <= UNROLL_MAX_EXPANDED

def _iter_stmts(stmts):
    for st in stmts:
        yield st
        for attr in ("body", "else_body", "combine_states"):
            child = getattr(st, attr, None)
            if isinstance(child, (tuple, list)) and child:
                yield from _iter_stmts(child)




DIVISION_OPS = frozenset({"/", "//", "divide", "floor_divide", "%", "mod", "remainder"})


def _expr_size(e, limit: int = 40) -> int:
    from emmy.compiler.ir.expr import Expr

    if not isinstance(e, Expr):
        return 0
    total = 1
    for child in _expr_children(e):
        total += _expr_size(child, limit)
        if total >= limit:
            return limit
    return total


def _is_expensive(e) -> bool:
    """Whether recomputing this subtree at every use is worth a live register.

    Division and remainder lower to long multiply-high sequences, and a deep
    tree costs enough instructions to matter.  Cheap two-or-three-node offsets
    are deliberately left inline: sharing them shortens live ranges and made
    the attention kernel slower by cutting memory-level parallelism.
    """
    from emmy.compiler.ir.expr import BinaryExpr

    if isinstance(e, BinaryExpr) and e.op in DIVISION_OPS:
        return True
    return _expr_size(e) >= 10

def _loop_load_buffers(stmts) -> set[str]:
    """Buffers read by Load statements (cp.async sources are not loads)."""
    names = set()
    for st in _iter_stmts(stmts):
        if stmt_name(st) == "Load":
            names.add(st.input)
    return names


def _partial_unroll_factor(st, smem) -> int:
    """Unroll a constant-trip global loop by a small factor, never fully.

    A global load's address folds into an immediate only when the unrolled copy
    carries a constant offset; doing that by 8 keeps the register pressure of a
    rolled loop while removing seven of every eight loop branches.  Full
    unrolling a global body was measured slower: it changed the ptxas register
    allocation and lost memory-level parallelism.
    """
    extent = int(st.axis.extent.value)
    if extent <= 1:
        return 1
    body = getattr(st, "body", ())
    weight = _expanded_weight(body, smem)
    if weight * extent > 8000:
        return 1
    loads = _loop_load_buffers(body)
    if loads & set(smem):
        # A long loop whose loads are all shared has low latency per iteration
        # and no global address to keep in flight.  Unrolling it by four cuts
        # the loop branches without the register pressure of a full unroll.
        if loads and not (loads - set(smem)) and extent > UNROLL_MAX_EXTENT:
            if 16 * weight <= 1600:
                return 16
        return 1
    if extent <= 32 and extent * weight <= 800:
        # A small odd trip count (matmul with K=17) gains more from removing the
        # loop and folding every address than from keeping loads in flight.
        return extent
    if extent < 4 or extent % 2:
        return 1
    for factor in (64, 32, 16, 8, 4, 2):
        if extent % factor and factor != 1:
            continue
        if factor * weight > 1600:
            continue
        if factor > 8 and weight > 8:
            continue
        return factor
    return 1


def _loaded_buffers(stmts) -> set[str]:
    names = set()
    for st in _iter_stmts(stmts):
        if stmt_name(st) in ("Load", "CpAsyncCopy"):
            names.add(st.input if stmt_name(st) == "Load" else st.src)
    return names


def _written_buffers(stmts) -> set[str]:
    names = set()
    for st in _iter_stmts(stmts):
        if stmt_name(st) == "Write":
            names.add(st.output)
    return names


def _axes_defined_in(stmts) -> set[str]:
    """Axis names rebound by loops inside a subtree.

    A sibling loop can reuse an axis name (matmul uses ``_fa`` for both the
    prologue fill and the in-loop fill). Hoisting an expression that reads the
    outer binding and then entering the rebinding loop would freeze the stale
    counter, so those names can never be treated as invariant.
    """
    names: set[str] = set()
    for st in _iter_stmts(stmts):
        if stmt_name(st) in ("Loop", "StridedLoop"):
            names.add(st.axis.name)
    return names


def _defined_in(stmts) -> set[str]:
    """SSA names assigned anywhere inside a loop subtree."""
    names: set[str] = set()
    for st in _iter_stmts(stmts):
        kind = stmt_name(st)
        if kind == "Load":
            names.update(st.names)
        elif kind in ("Assign", "Let", "Init", "Accum"):
            names.add(st.name)
        elif kind in ("WarpShuffle", "TreeHalve"):
            names.update(getattr(st, "state", ()))
            names.update(getattr(st, "state_b", ()))
    return names


def _index_exprs(stmts):
    """Every expression reachable from the statements in a loop subtree."""
    import dataclasses

    from emmy.compiler.ir.expr import Expr

    out = []
    for st in _iter_stmts(stmts):
        for field in dataclasses.fields(st):
            value = getattr(st, field.name)
            if isinstance(value, Expr):
                out.append(value)
            elif isinstance(value, (tuple, list)):
                out.extend(x for x in value if isinstance(x, Expr))
    return out


def _expr_children(e):
    import dataclasses

    from emmy.compiler.ir.expr import Expr

    if not dataclasses.is_dataclass(e):
        return
    for field in dataclasses.fields(e):
        value = getattr(e, field.name)
        if isinstance(value, Expr):
            yield value
        elif isinstance(value, (tuple, list)):
            for x in value:
                if isinstance(x, Expr):
                    yield x


def _check_dtype(dt: str, where: str) -> None:
    """Reject dtypes outside the implemented set instead of guessing f32."""
    if dt == "bool":
        return
    if dt not in DTYPES:
        raise CompileError(
            f"unsupported dtype {dt!r} in {where}; implemented: {', '.join(sorted(DTYPES))}"
        )


def _kind_of_dtype(dt: str) -> str:
    dt = norm_dtype(dt)
    if dt in FLOAT_KINDS:
        return "f32" if dt == "f32" else dt
    if dt in INT_KINDS:
        return "i64"
    return "f32"


def _strides(shape, ncoords: int) -> list[int]:
    """Row-major element strides for ``ncoords`` coordinates over ``shape``.

    A lone coordinate is a flat address in emmy's renderer, so its stride is 1
    regardless of the buffer's rank.
    """
    shape = tuple(int(s) for s in shape)
    if ncoords <= 1:
        return [1] * ncoords
    if not shape:
        return [1] * ncoords
    strides = []
    for i in range(ncoords):
        stride = 1
        for dim in shape[i + 1:]:
            stride *= dim
        strides.append(stride)
    return strides


def _axis_coefficient(e, axis: str):
    """Coefficient of one loop axis in an expression, treating the rest as const.

    Unlike ``_affine`` this tolerates non-affine *invariant* subtrees such as a
    staged index ``((_ks / 64) % 3) * 32``, which is a constant while iterating
    ``_ki``.  A None means the dependence on ``axis`` is not a constant stride.
    """
    from emmy.compiler.ir.expr import BinaryExpr, Literal, Var

    try:
        if axis not in e.free_vars():
            return 0
    except AttributeError:
        return None
    if isinstance(e, Var):
        return 1 if e.name == axis else 0
    if isinstance(e, Literal):
        return 0
    if isinstance(e, BinaryExpr):
        if e.op in ("+", "add", "-", "subtract"):
            left = _axis_coefficient(e.left, axis)
            right = _axis_coefficient(e.right, axis)
            if left is None or right is None:
                return None
            return left - right if e.op in ("-", "subtract") else left + right
        if e.op in ("*", "multiply"):
            left = _axis_coefficient(e.left, axis)
            right = _axis_coefficient(e.right, axis)
            if left is None or right is None:
                return None
            if left and right:
                return None
            if not left:
                c = _const_int(e.left)
                return None if c is None else c * right
            c = _const_int(e.right)
            return None if c is None else left * c
    return None


def _affine(e):
    """(constant, coefficients) if the expression is affine, else None.

    Only used for the static alignment proof; being conservative is fine.
    """
    from emmy.compiler.ir.expr import BinaryExpr, Literal, Var

    if isinstance(e, Literal):
        value = _const_int(e)
        if value is None:
            return None
        return (value, {})
    if isinstance(e, Var):
        return (0, {e.name: 1})
    if isinstance(e, BinaryExpr):
        left = _affine(e.left)
        right = _affine(e.right)
        if left is None or right is None:
            return None
        lc, lv = left
        rc, rv = right
        if e.op in ("+", "add"):
            merged = dict(lv)
            for k, v in rv.items():
                merged[k] = merged.get(k, 0) + v
            return (lc + rc, merged)
        if e.op in ("-", "subtract"):
            merged = dict(lv)
            for k, v in rv.items():
                merged[k] = merged.get(k, 0) - v
            return (lc - rc, merged)
        if e.op in ("*", "multiply"):
            if not rv:
                return (lc * rc, {k: v * rc for k, v in lv.items()})
            if not lv:
                return (lc * rc, {k: v * lc for k, v in rv.items()})
            return None
        if e.op in ("/", "//", "floor_divide") and not rv:
            if rc <= 0:
                return None
            if any((lc % rc) or (v % rc) for v in lv.values()):
                return None
            return (lc // rc, {k: v // rc for k, v in lv.items()})
        if e.op in ("%", "mod", "remainder") and not rv:
            if rc <= 0:
                return None
            if any((lc % rc) or (v % rc) for v in lv.values()):
                return None
            return (0, {})
    return None


__all__ = [name for name in globals() if not name.startswith("__") and name != "dataclasses"]
