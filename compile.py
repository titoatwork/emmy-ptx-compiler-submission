#!/usr/bin/env python3
"""Kernel IR -> PTX compiler.

Usage:
    python compile.py path/to/<kernel>.ir.json > out.ptx

Reads one Emmy Kernel IR document and writes PTX for the kernel it holds.
The launch contract (parameter order, grid/block, shared-memory size) lives
in the ``meta.json`` shipped beside the IR; the kernel itself never guesses
it.
"""

from __future__ import annotations

import json
import os
import sys

sys.dont_write_bytecode = True

from ptxc import CompileError, Lowerer


def find_kernel(graph):
    kernels = [node.op for node in graph.nodes.values()
               if type(node.op).__name__ == "KernelOp"]
    if not kernels:
        raise CompileError("graph contains no KernelOp")
    if len(kernels) > 1:
        raise CompileError(
            f"graph contains {len(kernels)} KernelOp nodes; expected exactly one")
    return kernels[0]


def find_meta(ir_path: str, entry: str):
    """The launch descriptor next to the IR, for this kernel.

    Hidden problems ship the same meta.json layout as the open set, so the
    lookup is by directory plus entry name (a problem may hold several
    kernels, each with its own IR file).
    """
    path = os.path.join(os.path.dirname(os.path.abspath(ir_path)), "meta.json")
    if os.path.exists(path):
        try:
            with open(path) as fh:
                meta = json.load(fh)
        except (OSError, ValueError, RecursionError) as exc:
            raise CompileError(f"cannot read {path}: {exc}") from exc
        if not isinstance(meta, dict):
            raise CompileError(f"{path}: meta.json must be a JSON object")
        kernels = meta.get("kernels") or []
        if not isinstance(kernels, list):
            raise CompileError(f"{path}: meta.json kernels must be a list")
        if not kernels:
            raise CompileError(f"{path}: meta.json contains no kernels")
        for kernel in kernels:
            if not isinstance(kernel, dict):
                raise CompileError(f"{path}: each meta.json kernel must be an object")
        for kernel in kernels:
            if kernel.get("entry") == entry:
                return kernel
        if kernels:
            names = ", ".join(str(k.get("entry")) for k in kernels)
            raise CompileError(
                f"meta.json has no kernel named {entry!r}; entries: {names}")
        return None
    return None


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python compile.py <kernel.ir.json>", file=sys.stderr)
        return 2
    ir_path = argv[1]

    from emmy.compiler.graph import Graph

    try:
        try:
            with open(ir_path) as fh:
                graph = Graph.from_dict(json.load(fh))
        except OSError as exc:
            raise CompileError(f"cannot read {ir_path}: {exc}") from exc
        except (json.JSONDecodeError, ValueError, TypeError, KeyError, AttributeError,
                NameError, SyntaxError, RecursionError) as exc:
            raise CompileError(f"invalid IR {ir_path}: {exc}") from exc

        kernel_op = find_kernel(graph)
        meta = find_meta(ir_path, kernel_op.name)
        if meta is None:
            raise CompileError(
                f"meta.json not found next to {ir_path}; the launch descriptor is required")

        ptx = Lowerer(kernel_op, meta).compile()
        if not ptx.isascii():
            raise CompileError("PTX must be ASCII")
        sys.stdout.write(ptx)
    except CompileError as exc:
        print(f"compile.py: {exc}", file=sys.stderr)
        return 1
    except (TypeError, AttributeError, KeyError, IndexError, ValueError, OverflowError,
            RecursionError) as exc:
        print(f"compile.py: invalid input: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
