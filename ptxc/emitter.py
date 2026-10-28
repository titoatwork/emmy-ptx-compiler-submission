"""PTX text emission: virtual registers, labels, typed helper instructions."""

from __future__ import annotations

import re
import struct

from .ops import f32_bits, reg_class


# First written operand of a 32-bit instruction, e.g. "add.s32 %r7, ...".
# Predicate and 64-bit destinations are intentionally not matched: the literal
# cache only tracks %r registers.
_WRITE_R32 = re.compile(r"^\s*[a-z][a-z0-9_.]*\s+(%r\d+)\s*,")


class Emitter:
    """Buffers the kernel body and tracks virtual-register high-water marks.

    ptxas does the real allocation, so the only job here is handing out
    distinct %-names and declaring enough of them.
    """

    PREFIX = {"f": "%f", "d": "%d", "h": "%h", "r": "%r", "rd": "%rd", "p": "%p"}
    DECL = {"f": ".f32", "d": ".f64", "h": ".b16", "r": ".b32", "rd": ".b64", "p": ".pred"}

    def __init__(self) -> None:
        self.body: list[str] = []
        self.used = {k: 0 for k in self.PREFIX}
        self._label = 0
        self._indent = 1
        # One register per integer literal, plus the literal each register
        # holds.  Index arithmetic materializes the same constants (0, 1, a
        # stride) many times per kernel; sharing them removes both the moves
        # and the register pressure they cause.
        #
        # The moves live in a separate block that is emitted once, before any
        # kernel code, so a literal first seen inside a conditional arm still
        # dominates every later use.  Materializing at first use is what made
        # the first version of this cache unsound.
        self._s32_literals: dict[int, str] = {}
        self._s32_values: dict[str, int] = {}
        self._literal_block: list[str] = []

    # -- registers ---------------------------------------------------------
    def reg(self, cls: str) -> str:
        self.used[cls] += 1
        return f"{self.PREFIX[cls]}{self.used[cls]}"

    def f(self) -> str:
        return self.reg("f")

    def r(self) -> str:
        return self.reg("r")

    def rd(self) -> str:
        return self.reg("rd")

    def p(self) -> str:
        return self.reg("p")

    def d(self) -> str:
        return self.reg("d")

    def h(self) -> str:
        return self.reg("h")

    def preg(self, dtype: str) -> str:
        return self.reg(reg_class(dtype))

    # -- text --------------------------------------------------------------
    def emit(self, text: str) -> None:
        match = _WRITE_R32.match(text)
        if match is not None:
            # A register that is written no longer holds the literal the
            # cache recorded for it.  Dropping both directions keeps
            # ``s32_of`` honest for every caller.
            reg = match.group(1)
            value = self._s32_values.pop(reg, None)
            if value is not None and self._s32_literals.get(value) == reg:
                del self._s32_literals[value]
        self.body.append("        " * self._indent + text)

    def new_label(self) -> str:
        self._label += 1
        return f"$L{self._label}"

    def place(self, label: str) -> None:
        self.body.append(f"{label}:")

    # -- common instructions ----------------------------------------------
    def mov_f32(self, value: float) -> str:
        r = self.f()
        self.emit(f"mov.f32 {r}, {f32_bits(value)};")
        return r

    def mov_f64(self, value: float) -> str:
        r = self.d()
        bits = struct.unpack("<Q", struct.pack("<d", float(value)))[0]
        self.emit(f"mov.f64 {r}, 0d{bits:016X};")
        return r

    def mov_s64(self, value: int) -> str:
        r = self.rd()
        self.emit(f"mov.s64 {r}, {int(value)};")
        return r

    def mov_s32(self, value: int) -> str:
        value = int(value)
        cached = self._s32_literals.get(value)
        if cached is not None:
            return cached
        r = self.r()
        self._literal_block.append("        " * self._indent + f"mov.s32 {r}, {value};")
        self._s32_literals[value] = r
        self._s32_values[r] = value
        return r

    def s32_of(self, reg: str):
        """The literal a 32-bit register holds, or None."""
        return self._s32_values.get(reg)

    def literal_block(self) -> list[str]:
        """Literal moves, placed ahead of the kernel body so they dominate it."""
        return list(self._literal_block)

    def mov_u32(self, value: int) -> str:
        r = self.r()
        self.emit(f"mov.u32 {r}, {int(value)};")
        return r

    def declarations(self) -> list[str]:
        lines = []
        for cls, count in self.used.items():
            if count:
                lines.append(f"        .reg {self.DECL[cls]} {self.PREFIX[cls]}<{count + 1}>;")
        return lines
