#!/usr/bin/env python3
"""Compile every open-set IR and verify the PTX contract.

Runs without a GPU or network: needs only emmy-ml (to read the IR) and,
optionally, ptxas (to assemble).  With --submit it also uses the hosted
runner through the task repository's submit.py.

    python tests/check_open_set.py --task-dir ../emmy-hackathon-2026
    python tests/check_open_set.py --task-dir ../emmy-hackathon-2026 --submit
"""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from irfiles import resolve_ir  # noqa: E402


def check_contract(ptx: str, entry: str, arg_order: list[str]) -> list[str]:
    errors = []
    if any(ord(ch) > 127 for ch in ptx):
        errors.append("PTX contains non-ASCII characters")
    head = [line.strip() for line in ptx.splitlines() if line.strip()][:4]
    if not head or not head[0].startswith(".version"):
        errors.append("missing .version directive")
    if not any(line.startswith(".target") for line in head):
        errors.append("missing .target directive")
    if not any(line.startswith(".address_size") for line in head):
        errors.append("missing .address_size directive")
    if f".visible .entry {entry}(" not in ptx:
        errors.append(f"entry point {entry!r} not declared")
    params = ptx.split(f".visible .entry {entry}(", 1)[-1].split(")", 1)[0]
    found = [line.split()[-1].rstrip(",").removeprefix("param_")
             for line in params.splitlines() if ".param" in line]
    if found != arg_order:
        errors.append(f"parameters {found} do not match arg_order {arg_order}")
    return errors


def find_ptxas() -> str | None:
    for candidate in ("ptxas", "/usr/local/cuda/bin/ptxas"):
        try:
            subprocess.run([candidate, "--version"], capture_output=True, check=True)
            return candidate
        except (OSError, subprocess.CalledProcessError):
            continue
    return None


def ptxas_check(ptx: str) -> str | None:
    exe = find_ptxas()
    if exe is None:
        return None
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "k.ptx"
        src.write_text(ptx)
        r = subprocess.run([exe, "-arch=sm_89", str(src), "-o", str(src.with_suffix(".cubin"))],
                           capture_output=True, text=True)
    return None if r.returncode == 0 else r.stderr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-dir", required=True, help="path to emmy-hackathon-2026")
    ap.add_argument("--submit", action="store_true", help="also run each problem on the runner")
    args = ap.parse_args()

    task = Path(args.task_dir)
    problem_root = task / "problems"
    if not problem_root.is_dir():
        print(f"no problems directory at {problem_root}", file=sys.stderr)
        return 1
    problems = sorted(p for p in problem_root.iterdir() if (p / "meta.json").exists())
    if not problems:
        print(f"no problem packages under {problem_root}", file=sys.stderr)
        return 1
    if find_ptxas() is None:
        print("note: ptxas not found; PTX assembly check skipped")
    failures = 0
    for pdir in problems:
        meta = json.loads((pdir / "meta.json").read_text())
        kernels = meta.get("kernels") or []
        if not kernels:
            print(f"FAIL {pdir.name}: meta.json has no kernels")
            failures += 1
            continue
        for kernel in kernels:
            entry = kernel["entry"]
            ir = resolve_ir(pdir, entry)
            r = subprocess.run([sys.executable, str(ROOT / "compile.py"), str(ir)],
                               capture_output=True, text=True)
            if r.returncode != 0:
                print(f"FAIL {pdir.name}/{entry}: compiler exited {r.returncode}: {r.stderr.strip()[:300]}")
                failures += 1
                continue
            errors = check_contract(r.stdout, entry, kernel["arg_order"])
            ptx_err = ptxas_check(r.stdout)
            if ptx_err:
                errors.append("ptxas: " + ptx_err.strip().splitlines()[0])
            if errors:
                print(f"FAIL {pdir.name}/{entry}: " + "; ".join(errors))
                failures += 1
            else:
                print(f"OK   {pdir.name}/{entry}")
        if args.submit:
            cmd = [sys.executable, str(task / "submit.py"), pdir.name,
                   "--compiler", str(ROOT / "compile.py")]
            if os.environ.get("RUNNER_URL"):
                cmd += ["--server", os.environ["RUNNER_URL"]]
            r = subprocess.run(cmd, capture_output=True, text=True)
            status = "PASS" if r.returncode == 0 else "FAIL"
            print(f"     runner: {status}")
            if r.returncode != 0:
                failures += 1
    print(f"\n{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
