#!/usr/bin/env python3
"""Measure one problem on the hosted runner.

Cold mode (the harness metric):   tools/prof.py a02_attention_128 --warmup 0 --iters 1
Warm mode (code quality):         tools/prof.py a02_attention_128

The runner is chosen with RUNNER_URL, which the organizers provide. The
compiler defaults to this repository's compile.py; the problem can be a name
under $TASK/problems or a path to any problem directory with meta.json.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent.parent
RUNNER = os.environ.get("RUNNER_URL", "").rstrip("/")
TASK = Path(os.environ.get("TASK", HERE.parent / "task"))


def post(path: str, body: dict) -> dict:
    if not RUNNER:
        sys.exit("set RUNNER_URL to the event-provided runner address")
    r = requests.post(f"{RUNNER}{path}", json=body, timeout=(10, 300))
    r.raise_for_status()
    return r.json()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("problem", help="problem id or a directory with meta.json")
    ap.add_argument("--compiler", default=str(HERE / "compile.py"))
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--repeat", type=int, default=1)
    a = ap.parse_args()

    pdir = Path(a.problem)
    if not (pdir / "meta.json").exists():
        pdir = TASK / "problems" / a.problem
    if not (pdir / "meta.json").exists():
        sys.exit(f"no meta.json at {pdir}")

    meta = json.loads((pdir / "meta.json").read_text())
    env = dict(os.environ)
    produced: dict[str, list] = {}
    for k in meta["kernels"]:
        ir = next(pdir.glob(f"{k['entry']}.ir.json"))
        r = subprocess.run([sys.executable, a.compiler, str(ir)],
                           capture_output=True, text=True, env=env)
        if r.returncode:
            sys.exit(r.stderr[-2000:] or "compiler failed")
        c = post("/compile", {"ptx": r.stdout})
        if not c.get("ok"):
            sys.exit(c.get("stderr", "")[-2000:])
        regs = (c.get("registers") or {}).get(k["entry"])
        run_args = []
        for arg in k["args"]:
            arg = dict(arg)
            if arg.pop("intermediate", False):
                arg["data"] = produced[arg["name"]]
            run_args.append(arg)
        need = max((math.prod(x["shape"]) for x in k["args"] if x.get("output")), default=1)
        times = []
        for _ in range(a.repeat):
            r = post("/run", {
                "cubin_id": c["cubin_id"], "entry": k["entry"],
                "grid": k["grid"], "block": k["block"], "smem_bytes": k["smem_bytes"],
                "zero_outputs": k["zero_outputs"], "args": run_args,
                "warmup": a.warmup, "iters": a.iters,
                "max_return": max(4096, need),
            })
            if not r.get("ok"):
                sys.exit(r.get("stderr", "")[-2000:])
            times.append(r.get("median_us"))
            for name, out in r["outputs"].items():
                produced[name] = out["values"]
        print(f"{a.problem} {k['entry']} regs={regs} "
              f"times={[round(t, 3) for t in times]} best={min(times):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
