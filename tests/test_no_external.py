#!/usr/bin/env python3
"""The compile path must not call any external compiler or renderer."""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN = (
    "import subprocess",
    "os.system(",
    "render_kernelop",
    "nvrtc",
    "emmy compile",
)


class NoExternalToolTests(unittest.TestCase):
    def test_compile_path_is_original(self):
        sources = [ROOT / "compile.py"] + sorted((ROOT / "ptxc").glob("*.py"))
        for path in sources:
            text = path.read_text()
            for token in FORBIDDEN:
                self.assertNotIn(token, text,
                                 f"{path.relative_to(ROOT)} contains {token!r}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
