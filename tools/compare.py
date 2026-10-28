#!/usr/bin/env python3
"""Interleaved A/B benchmark for two compiler checkouts.

Both compilers are run on the same problems, in alternating order, and the
result is the median of the per-round medians. The minimum is printed for
context; it is not the result.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import re
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPEN = ["p00_scalar_add", "p01_vector_add", "p02_scale_shift", "p03_relu_mul",
        "r01_array_sum", "r02_big_sum", "r03_mean", "r04_row_sum",
        "m01_matmul_512", "m02_matmul_rect", "f01_rmsnorm", "f02_row_softmax",
        "f03_l2norm", "a01_attention", "a02_attention_128"]


def measure(compiler: str, problem: str, mode: str) -> float | None:
    args = [sys.executable, str(HERE / "prof.py"), problem, "--compiler", compiler,
            "--repeat", "1"]
    if mode == "cold":
        args += ["--warmup", "0", "--iters", "1"]
    else:
        args += ["--warmup", "10", "--iters", "100"]
    r = subprocess.run(args, capture_output=True, text=True)
    m = re.search(r"best=([0-9.]+)", r.stdout)
    if not m:
        print(r.stdout[-300:], r.stderr[-300:], file=sys.stderr)
        return None
    return float(m.group(1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--mode", choices=("cold", "warm"), default="warm")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--problems", default=",".join(OPEN))
    args = ap.parse_args()

    problems = [p for p in args.problems.split(",") if p]
    print(f"mode={args.mode} rounds={args.rounds}")
    print(f"{'problem':22s} {'A med':>9s} {'B med':>9s} {'A/B':>6s} "
          f"{'A min':>9s} {'B min':>9s}")
    rows = []
    failed = False
    for p in problems:
        ta, tb = [], []
        for rnd in range(args.rounds):
            # alternate the order each round, and also swap inside each round
            order = (("a", args.a), ("b", args.b)) if rnd % 2 == 0 else \
                    (("b", args.b), ("a", args.a))
            for tag, comp in order:
                t = measure(comp, p, args.mode)
                if t is not None:
                    (ta if tag == "a" else tb).append(t)
        if not ta or not tb:
            failed = True
            print(f"{p:22s} {'?':>9s} {'?':>9s} {'?':>6s} {'?':>9s} {'?':>9s}")
            rows.append((p, None, None, None))
            continue
        a = statistics.median(ta)
        b = statistics.median(tb)
        ratio = a / b
        rows.append((p, a, b, ratio))
        print(f"{p:22s} {a:9.3f} {b:9.3f} {ratio:6.3f} "
              f"{min(ta):9.3f} {min(tb):9.3f}")
    good = [r for r in rows if r[3]]
    if good:
        import math
        gm = math.exp(sum(math.log(r[3]) for r in good) / len(good))
        print(f"{'geometric mean A/B':22s} {gm:.3f}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
