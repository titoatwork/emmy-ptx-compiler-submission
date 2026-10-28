#!/usr/bin/env python3
"""Lowering tests for machinery the open set does not exercise.

The hidden evaluation only uses constructs marked in the task reference, but a
compiler should still fail loudly or lower correctly for the rest of the
documented IR.  These build small variants by rewriting a real problem's
statement tree, then assert on the emitted PTX (and assemble it with ptxas
when available).
"""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import dataclasses
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from emmy.compiler.graph import Graph  # noqa: E402

from ptxc.lower import CompileError, Lowerer  # noqa: E402


def _task_dir() -> Path | None:
    for name in ("emmy-hackathon-2026", "task"):
        for base in (ROOT.parent, ROOT):
            candidate = base / name
            if (candidate / "problems").exists():
                return candidate
    return None


def _load(problem: str):
    task = _task_dir()
    if task is None:
        raise unittest.SkipTest("task repository not checked out next to this one")
    ir = next((task / "problems" / problem).glob("*.ir.json"))
    graph = Graph.from_dict(json.loads(ir.read_text()))
    kernel = next(n.op for n in graph.nodes.values() if type(n.op).__name__ == "KernelOp")
    meta = json.loads(next((task / "problems" / problem).glob("meta.json")).read_text())["kernels"][0]
    return kernel, meta


def _load_variant(name: str):
    pdir = ROOT / "tests" / "variants" / name
    ir = next(pdir.glob("*.ir.json"))
    graph = Graph.from_dict(json.loads(ir.read_text()))
    kernel = next(n.op for n in graph.nodes.values() if type(n.op).__name__ == "KernelOp")
    meta = json.loads((pdir / "meta.json").read_text())["kernels"][0]
    return kernel, meta


def _rewrite(stmts, predicate, transform):
    """Rebuild a statement tuple, replacing the first match found recursively."""
    out = []
    done = [False]
    for st in stmts:
        if not done[0] and predicate(st):
            out.append(transform(st))
            done[0] = True
            continue
        new = st
        for attr in ("body", "else_body"):
            child = getattr(st, attr, None)
            if isinstance(child, (tuple, list)) and child:
                rewritten = _rewrite(child, predicate, transform)
                if rewritten is not child:
                    new = dataclasses.replace(new, **{attr: rewritten})
        out.append(new)
    return tuple(out)


def _compile(kernel, meta) -> str:
    return Lowerer(kernel, meta).compile()


def _find_ptxas() -> str | None:
    for candidate in ("ptxas", "/usr/local/cuda/bin/ptxas"):
        try:
            subprocess.run([candidate, "--version"], capture_output=True, check=True)
            return candidate
        except (OSError, subprocess.CalledProcessError):
            continue
    return None


def setUpModule():
    if _find_ptxas() is None:
        print("note: ptxas not found; PTX assembly checks are skipped")


def _assemble(ptx: str) -> str | None:
    exe = _find_ptxas()
    if exe is None:
        return None
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "k.ptx"
        src.write_text(ptx)
        r = subprocess.run([exe, "-arch=sm_89", str(src), "-o", str(src.with_suffix(".cubin"))],
                           capture_output=True, text=True)
        return None if r.returncode == 0 else r.stderr


class MachineryTests(unittest.TestCase):
    def check(self, ptx: str, *needles: str) -> None:
        for needle in needles:
            self.assertIn(needle, ptx)
        err = _assemble(ptx)
        if err:
            self.fail(f"ptxas rejected the lowered kernel:\n{err}")

    def _strip(self, stmts, drop):
        out = []
        for st in stmts:
            if type(st).__name__ in drop:
                continue
            new = st
            for attr in ("body", "else_body"):
                child = getattr(st, attr, None)
                if isinstance(child, (tuple, list)) and child:
                    new = dataclasses.replace(new, **{attr: self._strip(child, drop)})
            out.append(new)
        return tuple(out)

    def test_nested_loop_does_not_leak_axis_binding(self):
        # A nested loop that reuses the Tile's axis name must not redirect a
        # later statement to the loop counter.  An empty two-trip loop makes
        # the last counter write easy to find: nothing after it may use that
        # register once the outer binding is restored.
        kernel, meta = _load("p00_scalar_add")
        from emmy.compiler.dim import Dim
        from emmy.compiler.ir.kernel import Axis
        from emmy.compiler.ir.stmt import Loop

        tile = next(s for s in kernel.body if type(s).__name__ == "Tile")
        body = list(tile.body)
        shadow = Loop(Axis("a0", Dim(2)), body=())
        new_tile = dataclasses.replace(tile, body=tuple(body[:-1]) + (shadow, body[-1]))
        ptx = _compile(dataclasses.replace(kernel, body=(new_tile,)), meta)
        self.check(ptx)
        lines = ptx.splitlines()
        reg = None
        at = 0
        for i, line in enumerate(lines):
            found = re.search(r"mov\.s32 (%r\d+), 1;", line)
            if found:
                reg, at = found.group(1), i
        self.assertIsNotNone(reg)
        self.assertNotIn(reg, "\n".join(lines[at + 1:]))

    def test_conditional_load_is_not_reused_outside(self):
        # A load bound under a predicate is undefined for the other lanes and
        # must not satisfy a later unconditional load with the same index.
        kernel, meta = _load("p00_scalar_add")
        from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
        from emmy.compiler.ir.stmt import Cond, Load

        tile = next(s for s in kernel.body if type(s).__name__ == "Tile")
        body = list(tile.body)
        dtype = body[-1].value_dtype
        cond = Cond(BinaryExpr("==", Var("a0"), Literal(0, dtype="int")),
                    body=(Load(names=("cond_load",), input="x0",
                               index=body[0].index, dtype=dtype),))
        new_tile = dataclasses.replace(tile, body=(cond,) + tuple(body))
        ptx = _compile(dataclasses.replace(kernel, body=(new_tile,)), meta)
        self.check(ptx)
        base = _compile(kernel, meta)
        self.assertEqual(ptx.count("ld.global.nc.f32"),
                         base.count("ld.global.nc.f32") + 1)

    def test_cache_requires_one_dimensional_grid(self):
        # The cooperative fill is only emitted for a one-dimensional launch,
        # so cache selection must refuse a multi-dimensional grid.
        kernel, meta = _load("a01_attention")
        meta = dict(meta, grid=[2, 2, 1])
        ptx = _compile(kernel, meta)
        self.assertNotIn("__cache_", ptx)

    def test_warp_sync(self):
        kernel, meta = _load("r01_array_sum")
        kernel = dataclasses.replace(
            kernel,
            body=_rewrite(kernel.body, lambda s: type(s).__name__ == "Sync",
                          lambda s: dataclasses.replace(s, warp=True)),
        )
        self.check(_compile(kernel, meta), "bar.warp.sync 0xffffffff;")

    def test_named_barrier_with_count(self):
        kernel, meta = _load("r01_array_sum")
        kernel = dataclasses.replace(
            kernel,
            body=_rewrite(kernel.body, lambda s: type(s).__name__ == "Sync",
                          lambda s: dataclasses.replace(s, barrier_id=1, count=64)),
        )
        self.check(_compile(kernel, meta), "bar.sync 1, 64;")

    def test_named_barrier_without_count(self):
        kernel, meta = _load("r01_array_sum")
        kernel = dataclasses.replace(
            kernel,
            body=_rewrite(kernel.body, lambda s: type(s).__name__ == "Sync",
                          lambda s: dataclasses.replace(s, barrier_id=3)),
        )
        self.check(_compile(kernel, meta), "bar.sync 3;")

    def test_cache_refused_for_2d_launch(self):
        # Cache selection refuses the cache for a 2D launch, so the fill-level
        # guard is a safety net here.  The useful assertion is that no cache is
        # declared and the kernel still compiles.
        kernel, meta = _load("a01_attention")
        meta = dict(meta)
        meta["grid"] = [meta["grid"][0], 2, 1]
        ptx = _compile(kernel, meta)
        self.check(ptx)
        self.assertNotIn("__cache_", ptx)

    def test_cp_async_8_bytes(self):
        kernel, meta = _load("m02_matmul_rect")
        kernel = dataclasses.replace(
            kernel,
            body=_rewrite(kernel.body, lambda s: type(s).__name__ == "CpAsyncCopy",
                          lambda s: dataclasses.replace(s, nbytes=8)),
        )
        self.check(_compile(kernel, meta), "cp.async.ca.shared.global", ", 8;")

    def test_smem_write_swizzle(self):
        kernel, meta = _load("r01_array_sum")
        kernel = dataclasses.replace(
            kernel,
            body=_rewrite(kernel.body,
                          lambda s: type(s).__name__ == "Write" and s.output.endswith("_smem"),
                          lambda s: dataclasses.replace(s, swizzle="B128")),
        )
        self.check(_compile(kernel, meta), "xor.b32")

    def test_tree_halve_inner(self):
        kernel, meta = _load("r01_array_sum")
        kernel = dataclasses.replace(
            kernel,
            body=_rewrite(kernel.body, lambda s: type(s).__name__ == "TreeHalve",
                          lambda s: dataclasses.replace(s, inner=("warp", 4))),
        )
        self.check(_compile(kernel, meta), "mul.lo.u32")

    def test_zero_prologue(self):
        kernel, meta = _load("p00_scalar_add")
        meta = dict(meta)
        meta["args"] = list(meta["args"]) + [
            {"name": "zero_buf", "dtype": "f32", "shape": [64]},
        ]
        meta["arg_order"] = list(meta["arg_order"]) + ["zero_buf"]
        from emmy.compiler.ir.stmt.leaves import ZeroPrologue

        tile = kernel.body[0]
        zero = ZeroPrologue(dst="zero_buf", words=64)
        tile = dataclasses.replace(tile, body=(zero,) + tuple(tile.body))
        kernel = dataclasses.replace(kernel, body=(tile,) + tuple(kernel.body[1:]))
        self.check(_compile(kernel, meta), "st.global.u32", "setp.ge.u32")

    def test_over_provisioned_extra_blocks_compile(self):
        # N is a multiple of the block size, so the extra blocks are fully
        # inactive and the block barrier in the active blocks is safe.
        kernel, meta = _load("r01_array_sum")
        meta = dict(meta)
        meta["grid"] = [10000, 1, 1]
        self.check(_compile(kernel, meta))

    def test_partial_block_barrier_fails_loudly(self):
        kernel, meta = _load("r01_array_sum")
        meta = dict(meta)
        meta["block"] = [256, 1, 1]
        meta["grid"] = [1, 1, 1]
        with self.assertRaises(CompileError):
            _compile(kernel, meta)

    def test_full_warp_collective_compiles(self):
        # 128 active threads of a 256-thread block are four full warps.  The
        # block is partial, but the warp shuffle has no inactive lanes.
        kernel, meta = _load("r01_array_sum")
        meta = dict(meta)
        meta["block"] = [256, 1, 1]
        meta["grid"] = [1, 1, 1]
        tile = next(s for s in kernel.body if type(s).__name__ == "Tile")
        body = self._strip(tile.body, {"Sync", "TreeHalve"})
        new_tile = dataclasses.replace(tile, body=body)
        kernel = dataclasses.replace(kernel, body=(new_tile,))
        self.check(_compile(kernel, meta))

    def test_partial_warp_shuffle_fails_loudly(self):
        from emmy.compiler.dim import Dim

        kernel, meta = _load("r01_array_sum")
        meta = dict(meta)
        meta["block"] = [256, 1, 1]
        meta["grid"] = [1, 1, 1]
        tile = next(s for s in kernel.body if type(s).__name__ == "Tile")
        body = self._strip(tile.body, {"Sync", "TreeHalve"})
        axes = tuple(dataclasses.replace(a, extent=Dim(100))
                     if i == len(tile.axes) - 1 else a
                     for i, a in enumerate(tile.axes))
        new_tile = dataclasses.replace(tile, body=body, axes=axes)
        kernel = dataclasses.replace(kernel, body=(new_tile,))
        with self.assertRaises(CompileError):
            _compile(kernel, meta)

    def test_partial_warp_sync_fails_loudly(self):
        from emmy.compiler.dim import Dim

        kernel, meta = _load("r01_array_sum")
        meta = dict(meta)
        meta["block"] = [256, 1, 1]
        meta["grid"] = [1, 1, 1]
        tile = next(s for s in kernel.body if type(s).__name__ == "Tile")
        body = self._strip(tile.body, {"TreeHalve"})
        body = tuple(dataclasses.replace(s, warp=True)
                     if type(s).__name__ == "Sync" else s for s in body)
        axes = tuple(dataclasses.replace(a, extent=Dim(100))
                     if i == len(tile.axes) - 1 else a
                     for i, a in enumerate(tile.axes))
        new_tile = dataclasses.replace(tile, body=body, axes=axes)
        kernel = dataclasses.replace(kernel, body=(new_tile,))
        with self.assertRaises(CompileError):
            _compile(kernel, meta)

    def test_pointwise_over_provisioned_compiles(self):
        kernel, meta = _load("p00_scalar_add")
        meta = dict(meta)
        meta["grid"] = [10000, 1, 1]
        self.check(_compile(kernel, meta))

    def test_float_ne_compare_uses_neu(self):
        from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
        from emmy.compiler.ir.stmt import Cond

        kernel, meta = _load("p00_scalar_add")
        tile = next(s for s in kernel.body if type(s).__name__ == "Tile")
        body = list(tile.body)
        cond = Cond(BinaryExpr("!=", Var("in0__u0"), Literal(0, dtype="int")),
                    body=(body[-1],))
        new_tile = dataclasses.replace(tile, body=tuple(body[:-1]) + (cond,))
        self.check(_compile(dataclasses.replace(kernel, body=(new_tile,)), meta),
                   "setp.neu.f32")

    def test_float_truthiness_uses_neu(self):
        from emmy.compiler.ir.expr import Var
        from emmy.compiler.ir.stmt import Cond

        kernel, meta = _load("p00_scalar_add")
        tile = next(s for s in kernel.body if type(s).__name__ == "Tile")
        body = list(tile.body)
        cond = Cond(Var("in0__u0"), body=(body[-1],))
        new_tile = dataclasses.replace(tile, body=tuple(body[:-1]) + (cond,))
        self.check(_compile(dataclasses.replace(kernel, body=(new_tile,)), meta),
                   "setp.neu.f32")

    def test_float_where_condition_uses_neu(self):
        from emmy.compiler.ir.elementwise import ElementwiseImpl
        from emmy.compiler.ir.stmt import Assign

        kernel, meta = _load("p00_scalar_add")
        tile = next(s for s in kernel.body if type(s).__name__ == "Tile")
        body = list(tile.body)
        f32 = body[-1].value_dtype
        where = Assign("v_where", ElementwiseImpl("where"),
                       ("in0__u0", "v0__u0", "v0__u1"), dtype=f32)
        write = dataclasses.replace(body[-1], values=("v_where", "v_where"))
        new_tile = dataclasses.replace(tile, body=tuple(body[:-1]) + (where, write))
        self.check(_compile(dataclasses.replace(kernel, body=(new_tile,)), meta),
                   "setp.neu.f32")

    def test_scalar_parameter(self):
        kernel, meta = _load("p00_scalar_add")
        meta = dict(meta)
        meta["args"] = list(meta["args"]) + [
            {"name": "alpha", "dtype": "f32", "shape": [1], "scalar": True},
        ]
        meta["arg_order"] = list(meta["arg_order"]) + ["alpha"]
        self.check(_compile(kernel, meta), ".param .f32 param_alpha", "ld.param.f32")

    def test_unknown_dtype_fails_loudly(self):
        kernel, meta = _load("p00_scalar_add")
        meta = dict(meta)
        meta["args"] = [dict(a) for a in meta["args"]]
        meta["args"][0]["dtype"] = "f16x2"
        with self.assertRaises(CompileError) as ctx:
            _compile(kernel, meta)
        self.assertIn("f16x2", str(ctx.exception))

    def test_unknown_statement_fails_loudly(self):
        kernel, meta = _load("p00_scalar_add")

        class Mystery:
            pass

        lowerer = Lowerer(kernel, meta)
        with self.assertRaises(CompileError) as ctx:
            lowerer.stmt(Mystery())
        self.assertIn("Mystery", str(ctx.exception))

    def test_slice_cache_unaligned_base_uses_scalar_fill(self):
        # The vector fill is only legal when the slice base is 16-byte
        # aligned.  Force an odd block term and check that the fill falls
        # back to scalar loads instead of emitting a misaligned v4 access.
        kernel, meta = _load("a01_attention")
        self.assertIn("ld.global.nc.v4.f32", _compile(kernel, meta))
        lowerer = Lowerer(kernel, meta)
        slices = [info for info in lowerer.cache_info.values() if info["slice"]]
        self.assertTrue(slices)
        for info in lowerer.cache_info.values():
            if info["slice"]:
                info["block_terms"] = {axis: 2 for axis in info["block_terms"]}
        for name in [name for name, info in lowerer.cache_info.items() if not info["slice"]]:
            del lowerer.cache_info[name]
        ptx = lowerer.compile()
        self.assertNotIn("ld.global.nc.v4.f32", ptx)

    def test_accumulator_ownership_is_per_loop_instance(self):
        # A nested loop that rebinds the axis name owns its accumulator; a
        # sibling loop over the same axis owns a different one.  The outer
        # loop must seed only the accumulator declared directly in it.
        from ptxc.common import _accums_for_loop
        from emmy.compiler.dim import Dim
        from emmy.compiler.ir.kernel import Axis
        from emmy.compiler.ir.elementwise import ElementwiseImpl
        from emmy.compiler.ir.stmt import Loop
        from emmy.compiler.ir.stmt.leaves import Accum

        def acc(name, axis):
            return Accum(name, "v", ElementwiseImpl("add"), dtype="f32", axes=(axis,))

        outer = Loop(Axis("a2", Dim(2)), body=(
            acc("outer", "a2"),
            Loop(Axis("a3", Dim(2)), body=(acc("inner", "a3"),)),
            Loop(Axis("a2", Dim(2)), body=(acc("shadow", "a2"),)),
        ))
        self.assertEqual([a.name for a in _accums_for_loop(outer, "a2", set())], ["outer"])
        self.assertEqual([a.name for a in _accums_for_loop(outer, "a2", {"outer"})], [])

    def test_store_converts_to_f16_buffer(self):
        kernel, meta = _load_variant("w_p_f16")
        self.check(_compile(kernel, meta), "cvt.rn.f16.f32", "st.global.b16")

    def test_store_converts_f32_value_to_f16_buffer(self):
        kernel, meta = _load("p00_scalar_add")
        meta = dict(meta)
        meta["args"] = [dict(a) for a in meta["args"]]
        meta["args"][-1]["dtype"] = "f16"
        ptx = _compile(kernel, meta)
        self.assertIn("cvt.rn.f16.f32", ptx)
        self.assertIn("st.global.b16", ptx)
        self.check(ptx)

    def test_store_converts_f16_value_to_f32_buffer(self):
        kernel, meta = _load_variant("w_p_f16")
        meta = dict(meta)
        meta["args"] = [dict(a) for a in meta["args"]]
        meta["args"][-1]["dtype"] = "f32"
        ptx = _compile(kernel, meta)
        self.assertIn("cvt.f32.f16", ptx)
        self.assertIn("st.global.v2.f32", ptx)
        self.check(ptx)

    def test_erf_f64_emits_fma_chain(self):
        from emmy.compiler.ir.elementwise import ElementwiseImpl
        from emmy.compiler.ir.stmt import Assign

        kernel, meta = _load_variant("w_p_f64")
        tile = next(s for s in kernel.body if type(s).__name__ == "Tile")
        body = list(tile.body)
        assign = next(s for s in body if type(s).__name__ == "Assign")
        write = next(s for s in body if type(s).__name__ == "Write")
        erf = Assign("v_erf", ElementwiseImpl("erf"), (assign.args[0],), dtype=assign.dtype)
        new_body = []
        for st in body:
            if st is assign:
                new_body.append(erf)
            elif st is write:
                new_body.append(dataclasses.replace(st, values=("v_erf",)))
            else:
                new_body.append(st)
        new_tile = dataclasses.replace(tile, body=tuple(new_body))
        ptx = _compile(dataclasses.replace(kernel, body=(new_tile,)), meta)
        import struct
        from ptxc.expr import ERF_POLY

        self.assertIn("abs.f64", ptx)
        self.assertGreaterEqual(ptx.count("fma.rn.f64"), len(ERF_POLY))
        expected = [
            "0d%016X" % struct.unpack("<Q", struct.pack("<d", float(value)))[0]
            for value in (ERF_POLY[-1], *reversed(ERF_POLY[:-1]))
        ]
        seen = re.findall(r"mov\.f64 %[A-Za-z0-9_]+, (0d[0-9A-F]{16});", ptx)
        self.assertTrue(
            any(seen[i:i + len(expected)] == expected
                for i in range(len(seen) - len(expected) + 1)),
            "erf coefficient sequence does not match ERF_POLY")
        self.check(ptx)

    def test_multi_kernel_meta_selects_each_entry(self):
        import shutil

        cases = (("v_p_div", "k_div_pointwise"), ("v_p_exp", "k_exp_pointwise"))
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            kernels = []
            orders = {}
            for name, entry in cases:
                src = ROOT / "tests" / "variants" / name
                meta = json.loads((src / "meta.json").read_text())["kernels"][0]
                kernels.append(meta)
                orders[entry] = meta["arg_order"]
                shutil.copy(next(src.glob("*.ir.json")), root / f"{entry}.ir.json")
            (root / "meta.json").write_text(json.dumps({"kernels": kernels}))
            for name, entry in cases:
                r = subprocess.run(
                    [sys.executable, str(ROOT / "compile.py"), str(root / f"{entry}.ir.json")],
                    capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertIn(f".visible .entry {entry}(", r.stdout)
                params = r.stdout.split(f".visible .entry {entry}(", 1)[1].split(")", 1)[0]
                names = [line.split()[-1].rstrip(",")[len("param_"):]
                         for line in params.splitlines() if ".param" in line]
                self.assertEqual(names, orders[entry])

    def test_let_unknown_dtype_fails_loudly(self):
        from emmy.compiler.ir.expr import Literal
        from emmy.compiler.ir.stmt.leaves import DataType, Let

        kernel, meta = _load("p00_scalar_add")
        tile = next(s for s in kernel.body if type(s).__name__ == "Tile")
        body = list(tile.body)
        bad = DataType(name="foo", np=body[-1].value_dtype.np, nbytes=4)
        let = Let("vbad", Literal(1, dtype="int"), dtype=bad)
        new_tile = dataclasses.replace(tile, body=tuple(body[:-1]) + (let, body[-1]))
        with self.assertRaises(CompileError) as ctx:
            _compile(dataclasses.replace(kernel, body=(new_tile,)), meta)
        self.assertIn("foo", str(ctx.exception))

    def test_cast_unknown_dtype_fails_loudly(self):
        from emmy.compiler.ir.expr import CastExpr, Literal
        from emmy.compiler.ir.stmt.leaves import Let

        kernel, meta = _load("p00_scalar_add")
        tile = next(s for s in kernel.body if type(s).__name__ == "Tile")
        body = list(tile.body)
        cast = CastExpr(expr=Literal(1, dtype="int"), dtype="foo")
        let = Let("vcast", cast, dtype=body[-1].value_dtype)
        new_tile = dataclasses.replace(tile, body=tuple(body[:-1]) + (let, body[-1]))
        with self.assertRaises(CompileError) as ctx:
            _compile(dataclasses.replace(kernel, body=(new_tile,)), meta)
        self.assertIn("foo", str(ctx.exception))

    def test_integer_accum_unknown_op_fails_loudly(self):
        from emmy.compiler.ir.elementwise import ElementwiseImpl
        from emmy.compiler.ir.stmt.leaves import Accum, DataType

        kernel, meta = _load("p00_scalar_add")
        tile = next(s for s in kernel.body if type(s).__name__ == "Tile")
        body = list(tile.body)
        i32 = DataType(name="i32", np=body[-1].value_dtype.np, nbytes=4)
        acc = Accum("accx", body[-1].values[0], ElementwiseImpl("relu"),
                    dtype=i32, axes=("a0",))
        new_tile = dataclasses.replace(tile, body=tuple(body[:-1]) + (acc, body[-1]))
        with self.assertRaises(CompileError) as ctx:
            _compile(dataclasses.replace(kernel, body=(new_tile,)), meta)
        self.assertIn("relu", str(ctx.exception))

    def test_write_undefined_value_fails_loudly(self):
        kernel, meta = _load("p00_scalar_add")
        tile = next(s for s in kernel.body if type(s).__name__ == "Tile")
        body = list(tile.body)
        write = dataclasses.replace(body[-1], values=("nosuch_ssa",))
        new_tile = dataclasses.replace(tile, body=tuple(body[:-1]) + (write,))
        with self.assertRaises(CompileError) as ctx:
            _compile(dataclasses.replace(kernel, body=(new_tile,)), meta)
        self.assertIn("nosuch_ssa", str(ctx.exception))

    def test_malformed_meta_fails_loudly(self):
        kernel, _ = _load("p00_scalar_add")
        cases = (
            [],
            {"args": [{"dtype": "f32", "shape": [1]}]},
            {"block": "oops", "args": []},
            {"block": [2.0, 1, 1], "args": []},
            {"block": [False, 1, 1], "args": []},
            {"block": [0, 1, 1], "args": []},
            {"block": [-256, 1, 1], "args": []},
            {"grid": [1.5, 1, 1], "args": []},
            {"grid": [True, 1, 1], "args": []},
            {"grid": [0, 1, 1], "args": []},
            {"grid": [-1, 1, 1], "args": []},
            {"args": [{"name": "x", "dtype": "f32", "shape": [-1]}]},
            {"args": [], "smem_bytes": 1.5},
            {"args": [], "smem_bytes": True},
            {"args": [], "smem_bytes": -1},
            {"args": [{"name": "x", "dtype": "f32", "shape": [1]}],
             "arg_order": ["x", "x"]},
        )
        for meta in cases:
            with self.assertRaises(CompileError):
                Lowerer(kernel, meta)

    def test_multiple_kernel_ops_fail_loudly(self):
        from types import SimpleNamespace
        from compile import find_kernel

        kernel_op = type("KernelOp", (), {})
        graph = SimpleNamespace(nodes={
            "a": SimpleNamespace(op=kernel_op()),
            "b": SimpleNamespace(op=kernel_op()),
        })
        with self.assertRaises(CompileError):
            find_kernel(graph)

    def test_missing_ir_fails_without_traceback(self):
        r = subprocess.run(
            [sys.executable, str(ROOT / "compile.py"), str(ROOT / "no-such.ir.json")],
            capture_output=True, text=True)
        self.assertEqual(r.returncode, 1)
        self.assertIn("compile.py:", r.stderr)
        self.assertNotIn("Traceback", r.stderr)


    def test_empty_meta_fails_loudly(self):
        import shutil
        task = _task_dir()
        if task is None:
            self.skipTest("task repository not checked out")
        src = task / "problems" / "p00_scalar_add"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            shutil.copy(next(src.glob("*.ir.json")), root / "k.ir.json")
            (root / "meta.json").write_text("{\"kernels\": []}")
            r = subprocess.run(
                [sys.executable, str(ROOT / "compile.py"), str(root / "k.ir.json")],
                capture_output=True, text=True)
            self.assertEqual(r.returncode, 1)
            self.assertIn("no kernels", r.stderr)
            self.assertNotIn("Traceback", r.stderr)

    def test_bad_shape_fails_loudly(self):
        import shutil
        task = _task_dir()
        if task is None:
            self.skipTest("task repository not checked out")
        src = task / "problems" / "p00_scalar_add"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            shutil.copy(next(src.glob("*.ir.json")), root / "k.ir.json")
            for bad_shape in ("128", [1.5], [True]):
                meta = json.loads((src / "meta.json").read_text())
                meta["kernels"][0]["args"][0]["shape"] = bad_shape
                (root / "meta.json").write_text(json.dumps(meta))
                r = subprocess.run(
                    [sys.executable, str(ROOT / "compile.py"), str(root / "k.ir.json")],
                    capture_output=True, text=True)
                self.assertEqual(r.returncode, 1)
                self.assertIn("shape", r.stderr)
                self.assertNotIn("Traceback", r.stderr)

    def test_invalid_utf8_meta_fails_loudly(self):
        import shutil
        task = _task_dir()
        if task is None:
            self.skipTest("task repository not checked out")
        src = task / "problems" / "p00_scalar_add"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            shutil.copy(next(src.glob("*.ir.json")), root / "k.ir.json")
            (root / "meta.json").write_bytes(b"\xff")
            r = subprocess.run(
                [sys.executable, str(ROOT / "compile.py"), str(root / "k.ir.json")],
                capture_output=True, text=True)
            self.assertEqual(r.returncode, 1)
            self.assertIn("cannot read", r.stderr)
            self.assertNotIn("Traceback", r.stderr)

    def test_unknown_statement_name_in_ir_fails_loudly(self):
        import shutil
        task = _task_dir()
        if task is None:
            self.skipTest("task repository not checked out")
        src = task / "problems" / "p00_scalar_add"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ir = next(src.glob("*.ir.json"))
            text = ir.read_text().replace("Assign(", "Mystery(")
            (root / "k.ir.json").write_text(text)
            shutil.copy(src / "meta.json", root / "meta.json")
            r = subprocess.run(
                [sys.executable, str(ROOT / "compile.py"), str(root / "k.ir.json")],
                capture_output=True, text=True)
            self.assertEqual(r.returncode, 1)
            self.assertIn("invalid IR", r.stderr)
            self.assertNotIn("Traceback", r.stderr)


    def test_invalid_ir_syntax_fails_loudly(self):
        import shutil
        task = _task_dir()
        if task is None:
            self.skipTest("task repository not checked out")
        src = task / "problems" / "p00_scalar_add"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ir = next(src.glob("*.ir.json"))
            text = ir.read_text().replace("Assign(", "Mystery((")
            (root / "k.ir.json").write_text(text)
            shutil.copy(src / "meta.json", root / "meta.json")
            r = subprocess.run(
                [sys.executable, str(ROOT / "compile.py"), str(root / "k.ir.json")],
                capture_output=True, text=True)
            self.assertEqual(r.returncode, 1)
            self.assertIn("invalid IR", r.stderr)
            self.assertNotIn("Traceback", r.stderr)


    def test_bad_arch_fails_loudly(self):
        kernel, meta = _load("p00_scalar_add")
        for arch in ({"x": 1}, "sm_\u0668\u0669", "sm_", "sm_89x"):
            bad = dict(meta)
            bad["arch"] = arch
            with self.assertRaises(CompileError) as ctx:
                Lowerer(kernel, bad)
            self.assertIn("arch", str(ctx.exception))


    def test_malformed_structural_field_fails_loudly(self):
        kernel, meta = _load("p00_scalar_add")
        tile = next(s for s in kernel.body if type(s).__name__ == "Tile")
        broken = dataclasses.replace(tile, axes=None)
        kernel = dataclasses.replace(kernel, body=(broken,))
        with self.assertRaises(CompileError) as ctx:
            Lowerer(kernel, meta).compile()
        self.assertIn("invalid IR", str(ctx.exception))



if __name__ == "__main__":
    unittest.main(verbosity=2)
