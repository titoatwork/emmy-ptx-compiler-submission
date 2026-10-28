"""Resolve a kernel IR file by its exact entry name.

A problem package names each IR ``<entry>.ir.json``.  A prefix glob is not
safe: entries such as ``k_sum_1`` and ``k_sum_1__partial`` share a prefix, and
``next(glob(...))`` would silently test the wrong kernel.
"""

from __future__ import annotations

from pathlib import Path


class IRError(Exception):
    pass


def resolve_ir(problem_dir, entry: str) -> Path:
    pdir = Path(problem_dir)
    exact = pdir / f"{entry}.ir.json"
    if exact.is_file():
        return exact
    matches = sorted(pdir.glob("*.ir.json"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise IRError(f"{pdir}: no .ir.json files")
    names = ", ".join(path.name for path in matches)
    raise IRError(f"{pdir}: no IR for entry {entry!r}; found {names}")
