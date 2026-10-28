"""Scalar elementwise operations: Emmy op names to PTX instruction spellings.

The IR names come from emmy.compiler.ir.elementwise; the PTX spellings are
chosen for IEEE-ish behaviour first (correctness gates, speed is a tiebreak).
"""

from __future__ import annotations

import struct

# emmy dtype name -> (ptx register class, byte width, ptx load/store type)
DTYPES: dict[str, tuple[str, int, str]] = {
    "f32": ("f", 4, "f32"),
    "f16": ("h", 2, "b16"),
    "f64": ("d", 8, "f64"),
    "i8": ("r", 1, "s8"),
    "u8": ("r", 1, "u8"),
    "i16": ("r", 2, "s16"),
    "u16": ("r", 2, "u16"),
    "i32": ("r", 4, "s32"),
    "u32": ("r", 4, "u32"),
    "i64": ("rd", 8, "s64"),
    "u64": ("rd", 8, "u64"),
    "bool": ("p", 1, "pred"),
}

# C spellings that appear in Smem.dtype
C_DTYPES = {
    "float": "f32",
    "double": "f64",
    "half": "f16",
    "__half": "f16",
    "int": "i32",
    "unsigned": "u32",
    "long": "i64",
    "short": "i16",
    "char": "i8",
}


def norm_dtype(name) -> str:
    if name is None:
        return "f32"
    name = getattr(name, "name", name)
    name = str(name)
    return C_DTYPES.get(name, name)


def f32_bits(value: float) -> str:
    """A PTX f32 immediate: 0fXXXXXXXX (bit pattern)."""
    return "0f%08X" % struct.unpack("<I", struct.pack("<f", float(value)))[0]


def reg_class(dtype: str) -> str:
    return DTYPES.get(norm_dtype(dtype), ("f", 4, "f32"))[0]


def elem_bytes(dtype: str) -> int:
    return DTYPES.get(norm_dtype(dtype), ("f", 4, "f32"))[1]


def ptx_type(dtype: str) -> str:
    return DTYPES.get(norm_dtype(dtype), ("f", 4, "f32"))[2]


# float identity seeds, matching emmy's ElementwiseImpl._IDENTITY
IDENTITY = {
    "add": 0.0,
    "sum": 0.0,
    "multiply": 1.0,
    "prod": 1.0,
    "maximum": -1e30,
    "amax": -1e30,
    "minimum": 1e30,
}

# Explicit round-to-nearest spellings: PTX only allows ptxas to contract an
# unrounded mul and add into an FMA, and a contracted chain differs from the
# IEEE result the reference numpy expression computes. Pointwise problems are
# graded at rtol=atol=0, so elementwise arithmetic must not be contracted.
FLOAT_BINARY = {
    "add": "add.rn.f32",
    "sum": "add.rn.f32",
    "subtract": "sub.rn.f32",
    "sub": "sub.rn.f32",
    "multiply": "mul.rn.f32",
    "prod": "mul.rn.f32",
    "divide": "div.rn.f32",
    "true_divide": "div.rn.f32",
    "maximum": "max.f32",
    "amax": "max.f32",
    "max": "max.f32",
    "minimum": "min.f32",
    "min": "min.f32",
    "fma": "fma.rn.f32",
}

INT_BINARY = {
    "add": "add.s64",
    "sum": "add.s64",
    "subtract": "sub.s64",
    "sub": "sub.s64",
    "multiply": "mul.lo.s64",
    "prod": "mul.lo.s64",
    "divide": "div.s64",
    "floor_divide": "div.s64",
    "true_divide": "div.s64",
    "remainder": "rem.s64",
    "mod": "rem.s64",
    "left_shift": "shl.b64",
    "right_shift": "shr.s64",
    "bitwise_and": "and.b64",
    "bitwise_or": "or.b64",
    "bitwise_xor": "xor.b64",
    "maximum": "max.s64",
    "amax": "max.s64",
    "minimum": "min.s64",
}

# f32 exp: ex2(x * log2(e)).
LOG2E = f32_bits(1.4426950408889634)
LN2 = f32_bits(0.6931471805599453)
