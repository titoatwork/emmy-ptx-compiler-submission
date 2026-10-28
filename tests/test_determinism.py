#!/usr/bin/env python3
"""The same IR must produce byte-identical PTX across runs and working directories."""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VARIANT = ROOT / "tests/variants/v_m_rect_small/k_matmul.ir.json"


def compile_once(cwd, hashseed):
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(hashseed)
    r = subprocess.run([sys.executable, str(ROOT / "compile.py"), str(VARIANT)],
                       capture_output=True, text=True, cwd=cwd, env=env)
    if r.returncode:
        raise AssertionError(r.stderr)
    return r.stdout


class DeterminismTests(unittest.TestCase):
    def test_two_runs_are_identical(self):
        self.assertEqual(compile_once(ROOT, 0), compile_once(ROOT, 0))

    def test_working_directory_does_not_change_the_output(self):
        self.assertEqual(compile_once(ROOT, 1), compile_once("/tmp", 1))

    def test_hash_seed_does_not_change_the_output(self):
        self.assertEqual(compile_once(ROOT, 2), compile_once(ROOT, 3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
