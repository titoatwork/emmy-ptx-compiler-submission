#!/usr/bin/env python3
"""CLI contract tests: usage, loud failures, and a working compile."""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VARIANT = ROOT / "tests/variants/v_p_div/k_div_pointwise.ir.json"


def run(*args, cwd=None, env=None):
    return subprocess.run([sys.executable, str(ROOT / "compile.py"), *args],
                          capture_output=True, text=True, cwd=cwd, env=env)


class CliTests(unittest.TestCase):
    def test_no_argument_is_a_usage_error(self):
        r = run()
        self.assertEqual(r.returncode, 2)
        self.assertIn("usage:", r.stdout + r.stderr)

    def test_missing_file_fails_without_a_traceback(self):
        r = run(str(ROOT / "tests/does_not_exist.ir.json"))
        self.assertEqual(r.returncode, 1)
        self.assertIn("compile.py:", r.stderr)
        self.assertNotIn("Traceback", r.stderr)

    def test_malformed_json_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.ir.json"
            path.write_text("{not json")
            r = run(str(path))
        self.assertEqual(r.returncode, 1)
        self.assertIn("compile.py:", r.stderr)
        self.assertNotIn("Traceback", r.stderr)

    def test_valid_variant_emits_the_contract_header(self):
        r = run(str(VARIANT))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(".version 8.0", r.stdout)
        self.assertIn(".target sm_89", r.stdout)
        self.assertIn(".address_size 64", r.stdout)
        meta = json.loads((VARIANT.parent / "meta.json").read_text())
        kernel = meta["kernels"][0]
        self.assertIn(f".visible .entry {kernel['entry']}(", r.stdout)
        for name in kernel["arg_order"]:
            self.assertIn(f"param_{name}", r.stdout)
        self.assertTrue(all(ord(c) < 128 for c in r.stdout))

    def test_missing_meta_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "k.ir.json"
            path.write_text(VARIANT.read_text())
            r = run(str(path))
        self.assertEqual(r.returncode, 1)
        self.assertIn("meta.json", r.stderr)
        self.assertNotIn("Traceback", r.stderr)
        self.assertNotIn(".visible .entry", r.stdout)

    def test_strict_argument_and_launch_fields(self):
        meta = json.loads((VARIANT.parent / "meta.json").read_text())

        def check(mutate, needle):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "k.ir.json").write_text(VARIANT.read_text())
                broken = json.loads(json.dumps(meta))
                mutate(broken)
                (root / "meta.json").write_text(json.dumps(broken))
                r = run(str(root / "k.ir.json"))
            self.assertEqual(r.returncode, 1)
            self.assertIn(needle, r.stderr)
            self.assertNotIn("Traceback", r.stderr)
            self.assertEqual(r.stdout, "")

        for field in ("shape", "dtype"):
            check(lambda m, f=field: m["kernels"][0]["args"][0].pop(f), field)
        check(lambda m: m["kernels"][0].__setitem__("arg_order", []), "arg_order")
        check(lambda m: m["kernels"][0].__setitem__("grid", [1, 1, 1, 1]), "3 integers")
        check(lambda m: m["kernels"][0].__setitem__("block", [256, 1]), "3 integers")
        check(lambda m: m["kernels"][0]["args"][0].__setitem__("scalar", "false"), "scalar")
        check(lambda m: m["kernels"][0].__setitem__("entry", "k_wrong"), "no kernel named")

    def test_non_ascii_entry_fails_loudly(self):
        meta = json.loads((VARIANT.parent / "meta.json").read_text())
        ir = json.loads(VARIANT.read_text())
        for node in ir["nodes"].values():
            if node.get("op") == "KernelOp":
                node["op_fields"]["name"] = "k_caf\u00e9"
        meta["kernels"][0]["entry"] = "k_caf\u00e9"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "k.ir.json").write_text(json.dumps(ir))
            (root / "meta.json").write_text(json.dumps(meta))
            r = run(str(root / "k.ir.json"))
        self.assertEqual(r.returncode, 1)
        self.assertIn("ASCII", r.stderr)
        self.assertNotIn("Traceback", r.stderr)
        self.assertEqual(r.stdout, "")

    def test_deep_meta_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "k.ir.json").write_text(VARIANT.read_text())
            (root / "meta.json").write_text("[" * 20000 + "]" * 20000)
            r = run(str(root / "k.ir.json"))
        self.assertEqual(r.returncode, 1)
        self.assertIn("meta.json", r.stderr)
        self.assertNotIn("Traceback", r.stderr)
        self.assertEqual(r.stdout, "")

    def test_missing_launch_field_fails_loudly(self):
        meta = json.loads((VARIANT.parent / "meta.json").read_text())
        for field in ("grid", "block", "arg_order", "args"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "k.ir.json").write_text(VARIANT.read_text())
                broken = json.loads(json.dumps(meta))
                broken["kernels"][0].pop(field, None)
                (root / "meta.json").write_text(json.dumps(broken))
                r = run(str(root / "k.ir.json"))
                self.assertEqual(r.returncode, 1)
                self.assertIn(field, r.stderr)
                self.assertNotIn("Traceback", r.stderr)
                self.assertNotIn(".visible .entry", r.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
