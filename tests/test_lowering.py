#!/usr/bin/env python3
"""Lowering tests that do not need a GPU.

Covers the failure modes that caused real miscompiles, plus the index rules
that the PTX path depends on. Run with:

    python tests/test_lowering.py
"""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from irfiles import resolve_ir  # noqa: E402
from ptxarch import arch_number, retarget_ptx  # noqa: E402
from ptxc.lower import _affine, _strides  # noqa: E402
from ptxc.stmts import _single_trip  # noqa: E402


class StrideTests(unittest.TestCase):
    def test_single_index_is_flat(self):
        # emmy's renderer treats one coordinate as the flat address: a 2D
        # buffer indexed with one coordinate does not get row stride 64.
        self.assertEqual(_strides((64, 128), 1), [1])

    def test_row_major_strides(self):
        self.assertEqual(_strides((64, 128), 2), [128, 1])
        self.assertEqual(_strides((8, 32, 256), 3), [8192, 256, 1])


class AffineTests(unittest.TestCase):
    """The alignment proof gates vector loads and stores."""

    def _var(self, name):
        from emmy.compiler.ir.expr import Var

        return Var(name=name)

    def _lit(self, value):
        from emmy.compiler.ir.expr import Literal

        return Literal(value=value, dtype="int")

    def _bin(self, op, left, right):
        from emmy.compiler.ir.expr import BinaryExpr

        return BinaryExpr(op=op, left=left, right=right)

    def test_axis_times_four_is_affine(self):
        expr = self._bin("*", self._var("a0"), self._lit(4))
        self.assertEqual(_affine(expr), (0, {"a0": 4}))

    def test_axis_times_four_plus_offset(self):
        expr = self._bin("+", self._bin("*", self._var("a0"), self._lit(4)), self._lit(3))
        self.assertEqual(_affine(expr), (3, {"a0": 4}))

    def test_division_by_power_of_two_divides_every_term(self):
        expr = self._bin("//", self._bin("*", self._var("a0"), self._lit(4)), self._lit(2))
        self.assertEqual(_affine(expr), (0, {"a0": 2}))

    def test_division_that_would_lose_information_is_rejected(self):
        # (4*a0)//128 is not affine in general (it depends on a0's low bits).
        expr = self._bin("//", self._bin("*", self._var("a0"), self._lit(4)), self._lit(128))
        self.assertIsNone(_affine(expr))

    def test_product_of_two_variables_is_rejected(self):
        self.assertIsNone(_affine(self._bin("*", self._var("a0"), self._var("a1"))))


class SingleTripTests(unittest.TestCase):
    """A strided loop whose bound fits in one step is a guarded body, not a loop."""

    def _stmt(self, start, end, extent=64):
        from types import SimpleNamespace

        return SimpleNamespace(start=start, end=end,
                               axis=SimpleNamespace(extent=extent))

    def _var(self, name):
        from emmy.compiler.ir.expr import Var

        return Var(name=name)

    def _lit(self, value):
        from emmy.compiler.ir.expr import Literal

        return Literal(value=value, dtype="int")

    def test_bound_within_one_step_is_single_trip(self):
        st = self._stmt(self._var("t"), self._lit(32))
        self.assertTrue(_single_trip(st, 256))

    def test_bound_equal_to_step_is_single_trip(self):
        st = self._stmt(self._var("t"), self._lit(256))
        self.assertTrue(_single_trip(st, 256))

    def test_longer_range_is_not_single_trip(self):
        st = self._stmt(self._var("t"), self._lit(512))
        self.assertFalse(_single_trip(st, 256))

    def test_unknown_bound_is_not_single_trip(self):
        st = self._stmt(self._var("t"), self._var("n"))
        self.assertFalse(_single_trip(st, 256))

    def test_negative_start_is_not_single_trip(self):
        st = self._stmt(self._lit(-1), self._lit(8))
        self.assertFalse(_single_trip(st, 8))

    def test_non_positive_step_is_not_single_trip(self):
        st = self._stmt(self._var("t"), self._lit(8))
        self.assertFalse(_single_trip(st, 0))

    def test_axis_extent_is_used_when_end_is_missing(self):
        st = self._stmt(self._var("t"), None, extent=self._lit(16))
        self.assertTrue(_single_trip(st, 32))


class SmallTileDecodeTests(unittest.TestCase):
    """Small tile grids keep the divide and remainder form."""

    def test_small_attention_tile_keeps_div_rem(self):
        pdir = Path("../attnadv/x_attn_s64_h48")
        if not (pdir / "meta.json").exists():
            self.skipTest("attnadv corpus not generated")
        import json
        import subprocess

        kernel = json.loads((pdir / "meta.json").read_text())["kernels"][0]
        ir = next(pdir.glob("*.ir.json"))
        r = subprocess.run([sys.executable, str(ROOT / "compile.py"), str(ir)],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        # N is 256, below the large-grid threshold, so the work-id decode
        # keeps the divide and remainder form.
        self.assertIn("div.u32", r.stdout)
        self.assertIn("rem.u32", r.stdout)


class BoundedUnrollTests(unittest.TestCase):
    """A bounded runtime start unrolls to its maximum with guards on the tail."""

    def _host(self):
        from ptxc.emitter import Emitter
        from ptxc.stmts import StmtMixin

        class Host(StmtMixin):
            def __init__(self):
                self.em = Emitter()
                self.axis_extent = {"a0_u": 8, "a1_u": 32}
                self.smem = {}

        return Host()

    def _stmt(self, start, end):
        from types import SimpleNamespace

        return SimpleNamespace(start=start, end=end,
                               axis=SimpleNamespace(name="fa", extent=end))

    def _var(self, name):
        from emmy.compiler.ir.expr import Var

        return Var(name=name)

    def _lit(self, value):
        from emmy.compiler.ir.expr import Literal

        return Literal(value=value, dtype="int")

    def _bin(self, op, left, right):
        from emmy.compiler.ir.expr import BinaryExpr

        return BinaryExpr(op=op, left=left, right=right)

    def _body(self):
        from types import SimpleNamespace

        return (SimpleNamespace(names=("x",), input="buf", index=(self._lit(0),)),
                SimpleNamespace(names=("y",), input="buf", index=(self._lit(1),)))

    def test_variable_start_unrolls_to_the_maximum(self):
        h = self._host()
        start = self._bin("+", self._bin("*", self._var("a0_u"), self._lit(32)),
                          self._var("a1_u"))
        st = self._stmt(start, self._lit(640))
        st.body = self._body()
        self.assertEqual(h._strided_bounded_plan(st, 256), (3, 2))

    def test_constant_start_is_uniform(self):
        h = self._host()
        st = self._stmt(self._lit(0), self._lit(768))
        st.body = self._body()
        self.assertEqual(h._strided_bounded_plan(st, 256), (3, 3))

    def test_unknown_bound_declines(self):
        h = self._host()
        st = self._stmt(self._var("a1_u"), self._var("n"))
        st.body = self._body()
        self.assertIsNone(h._strided_bounded_plan(st, 256))

    def test_negative_start_declines(self):
        h = self._host()
        st = self._stmt(self._lit(-1), self._lit(640))
        st.body = self._body()
        self.assertIsNone(h._strided_bounded_plan(st, 256))

    def test_start_beyond_bound_declines(self):
        h = self._host()
        st = self._stmt(self._lit(640), self._lit(640))
        st.body = self._body()
        self.assertIsNone(h._strided_bounded_plan(st, 256))


class PTXInvariantTests(unittest.TestCase):
    """Every open IR must produce ascii-only PTX with the promised signature."""

    @classmethod
    def setUpClass(cls):
        cls.task = None
        for name in ("emmy-hackathon-2026", "task"):
            for base in (ROOT.parent, ROOT):
                candidate = base / name
                if (candidate / "problems").exists():
                    cls.task = candidate
                    break
            if cls.task:
                break
        cls.problems = []
        if cls.task is not None:
            cls.problems = sorted(p for p in (cls.task / "problems").iterdir()
                                  if (p / "meta.json").exists())

    def test_open_set_contract(self):
        if not self.problems:
            self.skipTest("task repository not checked out next to this one")
        import json
        import subprocess

        for pdir in self.problems:
            meta = json.loads((pdir / "meta.json").read_text())
            for kernel in meta["kernels"]:
                ir = resolve_ir(pdir, kernel["entry"])
                r = subprocess.run([sys.executable, str(ROOT / "compile.py"), str(ir)],
                                   capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertTrue(all(ord(c) < 128 for c in r.stdout))
                self.assertIn(f".visible .entry {kernel['entry']}(", r.stdout)
                params = r.stdout.split(f".visible .entry {kernel['entry']}(", 1)[1].split(")", 1)[0]
                names = [line.split()[-1].rstrip(",")[len("param_"):]
                         for line in params.splitlines() if ".param" in line]
                self.assertEqual(names, kernel["arg_order"])

    def test_power_of_two_tile_decode_has_no_divide(self):
        if not self.problems:
            self.skipTest("task repository not checked out next to this one")
        import json
        import subprocess

        pdir = next((p for p in self.problems if p.name == "m02_matmul_rect"), None)
        if pdir is None:
            self.skipTest("m02_matmul_rect not present")
        kernel = json.loads((pdir / "meta.json").read_text())["kernels"][0]
        ir = resolve_ir(pdir, kernel["entry"])
        r = subprocess.run([sys.executable, str(ROOT / "compile.py"), str(ir)],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        # Every tile axis in this launch is a power-of-two extent, so the
        # decode is shifts and masks, not div/rem.
        self.assertNotIn("div.u32", r.stdout)
        self.assertNotIn("rem.u32", r.stdout)


class SharedLoopUnrollTests(unittest.TestCase):
    def test_long_all_shared_loop_unrolls_by_sixteen(self):
        from types import SimpleNamespace

        from ptxc.common import _partial_unroll_factor

        class Load:
            def __init__(self, name):
                self.input = name

        st = SimpleNamespace(
            axis=SimpleNamespace(extent=SimpleNamespace(value=128)),
            body=(Load("smem_a"), Load("smem_b")),
        )
        self.assertEqual(_partial_unroll_factor(st, {"smem_a", "smem_b"}), 16)

    def test_mixed_shared_and_global_loop_is_not_unrolled(self):
        from types import SimpleNamespace

        from ptxc.common import _partial_unroll_factor

        class Load:
            def __init__(self, name):
                self.input = name

        st = SimpleNamespace(
            axis=SimpleNamespace(extent=SimpleNamespace(value=128)),
            body=(Load("smem_a"), Load("global_a")),
        )
        self.assertEqual(_partial_unroll_factor(st, {"smem_a"}), 1)


class LaneFanoutTests(unittest.TestCase):
    def test_fanout_counts_shared_lane_expressions(self):
        from types import SimpleNamespace

        from ptxc.stmts import _lane_fanout

        class CpAsyncCopy:
            def __init__(self, lane):
                self.lane_index = lane

        def copy(lane):
            return CpAsyncCopy(lane)

        lane = (SimpleNamespace(name="u"), SimpleNamespace(name="v"))
        self.assertEqual(_lane_fanout([copy(lane), copy(lane)]), 2)
        self.assertEqual(_lane_fanout([copy(lane)] * 4), 4)
        self.assertEqual(_lane_fanout([copy(None), copy(lane)]), 1)
        self.assertEqual(_lane_fanout([]), 0)


class DivModRecombineTests(unittest.TestCase):
    """(x / cols, x % cols) is x when cols is the row stride."""

    def _var(self, name):
        from emmy.compiler.ir.expr import Var

        return Var(name=name)

    def _lit(self, value):
        from emmy.compiler.ir.expr import Literal

        return Literal(value=value, dtype="int")

    def _bin(self, op, left, right):
        from emmy.compiler.ir.expr import BinaryExpr

        return BinaryExpr(op=op, left=left, right=right)

    def _host(self):
        from ptxc.emitter import Emitter
        from ptxc.expr import ExprMixin
        from ptxc.memory import MemoryMixin

        class Host(MemoryMixin, ExprMixin):
            def __init__(self):
                self.em = Emitter()
                self.axes = {"u": self.em.r()}
                self.axis_values = {}
                self.values = {}
                self.scalars = {}
                self.hoisted = {}
                self.icache = {}

        return Host()

    def test_div_and_mod_recombine_to_the_dividend(self):
        h = self._host()
        e = self._bin("*", self._var("u"), self._lit(4))
        index = (self._bin("/", e, self._lit(64)), self._bin("%", e, self._lit(64)))
        h.flat_index(index, (224, 64))
        text = "\n".join(h.em.body)
        self.assertNotIn("div.", text)
        self.assertNotIn("rem.", text)
        self.assertIn("mul.lo.s32", text)

    def test_mismatched_divisor_does_not_recombine(self):
        from ptxc.memory import _div_operand, _mod_operand

        e = self._var("u")
        self.assertEqual(_div_operand(self._bin("/", e, self._lit(64)), 64), e)
        self.assertIsNone(_div_operand(self._bin("/", e, self._lit(32)), 64))
        self.assertEqual(_mod_operand(self._bin("%", e, self._lit(64)), 64), e)
        self.assertIsNone(_mod_operand(self._bin("%", e, self._lit(32)), 64))


class FusedLoadTests(unittest.TestCase):
    """Four adjacent shared loads become one vector load; f16 does not."""

    def _compile_path(self, path):
        import subprocess

        r = subprocess.run([sys.executable, str(ROOT / "compile.py"), str(path)],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout

    def test_adjacent_shared_loads_fuse_to_v4(self):
        pdir = Path("../stress/s_mm_2048")
        if not (pdir / "meta.json").exists():
            self.skipTest("stress corpus not generated")
        ir = next(pdir.glob("*.ir.json"))
        ptx = self._compile_path(ir)
        self.assertIn("ld.shared.v4.f32", ptx)

    def test_multi_name_f16_load_is_not_forced_to_v4(self):
        pdir = Path("../f16/h16_exp")
        if not (pdir / "meta.json").exists():
            self.skipTest("f16 corpus not generated")
        ir = next(pdir.glob("*.ir.json"))
        ptx = self._compile_path(ir)
        self.assertNotIn("v4.f32", ptx)
        self.assertIn(".b16", ptx)


class ErfApproximationTests(unittest.TestCase):
    def test_polynomial_matches_reference(self):
        # The same coefficients the emitter ships, evaluated in Python.  The
        # PTX path uses f64 fma chains; this test pins the numerical promise.
        from ptxc.expr import ERF_LIMIT, ERF_POLY

        def approx(x):
            u = x * x / 8.0 - 1.0
            acc = ERF_POLY[-1]
            for coeff in reversed(ERF_POLY[:-1]):
                acc = acc * u + coeff
            return x * acc

        worst = 0.0
        for i in range(4001):
            x = ERF_LIMIT * i / 4000.0
            worst = max(worst, abs(approx(x) - math.erf(x)))
        self.assertLess(worst, 1e-9)


class PTXRetargetTests(unittest.TestCase):
    def test_older_arch_rewrites_target(self):
        ptx = ".version 8.0\n.target sm_89\n"
        self.assertEqual(retarget_ptx(ptx, "sm_86"), ".version 8.0\n.target sm_86\n")
        self.assertEqual(retarget_ptx(ptx, "sm_80"), ".version 8.0\n.target sm_80\n")

    def test_same_or_newer_arch_keeps_sm89(self):
        ptx = ".version 8.0\n.target sm_89\n"
        for arch in ("sm_89", "sm_90", "sm_120", "sm_121"):
            self.assertEqual(retarget_ptx(ptx, arch), ptx)

    def test_arch_number(self):
        self.assertEqual(arch_number("sm_86"), 86)
        self.assertEqual(arch_number("sm_120"), 120)
        self.assertEqual(arch_number("sm_90a"), 90)


class IRResolutionTests(unittest.TestCase):
    def test_exact_entry_beats_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdir = Path(tmp)
            (pdir / "k_sum_1.ir.json").write_text("{}")
            (pdir / "k_sum_1__partial.ir.json").write_text("{}")
            self.assertEqual(resolve_ir(pdir, "k_sum_1").name, "k_sum_1.ir.json")
            self.assertEqual(resolve_ir(pdir, "k_sum_1__partial").name,
                             "k_sum_1__partial.ir.json")

    def test_missing_entry_is_loud(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdir = Path(tmp)
            (pdir / "a.ir.json").write_text("{}")
            (pdir / "b.ir.json").write_text("{}")
            with self.assertRaises(Exception):
                resolve_ir(pdir, "c")


if __name__ == "__main__":
    unittest.main(verbosity=2)
