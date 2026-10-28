# Verification record

This page records what was run, on which revision, and what came back. It is the static evidence
behind the status line in the README. The benchmark matrix has its own method page in
[BENCHMARKS.md](BENCHMARKS.md).

## Revision

| item | value |
|---|---|
| verified release | `v1.0.0` |
| verification date | 2026-10-28 |
| Python | 3.12 |
| local GPU | NVIDIA RTX 3050 6 GB, CUDA 12.6, ptxas 12.6 |
| hosted runner | NVIDIA RTX 5090, sm_120, CUDA 12.9 |
| compiler dependencies | `emmy-ml==0.3.11` and the Python standard library |

The release tag `v1.0.0` identifies the published tree. The launch descriptor is strict:
`meta.json`, `grid`, `block`, `arg_order` and `args` are required; `grid` and `block` are three
positive integers; every argument has a name, dtype and shape; `scalar` is boolean; `arg_order`
names every argument exactly, and `meta.json` names the IR kernel. A violation exits 1 with no PTX
instead of inventing a launch. The
CLI contract suite covers each case.

## Tier 1: offline

No GPU or network is required. The counts marked with the task repository need the organizer
checkout and the generated corpora; without them those cases skip.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt -r requirements-tools.txt
tools/verify.sh --quick
ruff check --select E9,F63,F7,F82 .
```

Recorded result:

| check | result |
|---|---|
| Python syntax | pass |
| lowering suite | 35 tests pass with the task repository (30 run, 5 skip without it) |
| machinery suite | 44 tests pass with the task repository (7 run, 37 skip without it) |
| contract suite | 13 tests pass (CLI, determinism, no external tool) |
| dash policy | 0 violations in 154 files |
| ruff E9,F63,F7,F82 | clean |
| quick gate | 6/6 steps with the task repository (5 passed, 1 skipped without it) |

## Tier 2: local GPU

The 46 variant packages and the 86 generated corpus IRs are executed on the local GPU. Each
variant package has `meta.json`, one `.ir.json` and `expected.npz`; the two-kernel corpus package
has one `meta.json`, two IRs and one `expected.npz`. `tools/localrun.py` compiles with the local
ptxas, launches under the `meta.json` geometry, and compares within the package tolerance.

```bash
tools/checkvariants.sh
TASK=../task tools/verify.sh --corpora
```

Recorded result:

| suite | result |
|---|---|
| unit and contract suites | 92/92 pass |
| checked-in variants | 46/46 pass |
| generated corpora (13 directories) | 86/86 IRs pass in 85 packages |
| local IR compile sweep | 147/147 compile |
| quick plus corpora ladder | 20/20 steps, 0 failed |

## Tier 3: hosted runner

The 15 open problems are compiled by `compile.py`, assembled by the runner's ptxas, launched
under each problem's `meta.json`, and compared with the expected output. The run was repeated
twice on the verified revision.

```bash
TASK=../task RUNNER_URL="$RUNNER_URL" tools/testall.sh
```

Recorded result on `v1.0.0`, 2026-10-28:

| run | result |
|---|---|
| first | 15 passed, 0 failed |
| second | 15 passed, 0 failed |

The runner is shared and its clock state moves between sessions. The open-set pass/fail result
does not depend on the clock state; the timing matrix does, which is why the S1 comparisons in
[BENCHMARKS.md](BENCHMARKS.md) are interleaved same-session A/B runs; S2 and S3 are labeled
rechecks against the S1 baseline.

## Compile time and output size

| kernel | compile time | PTX virtual registers (declared) | SASS registers | spills |
|---|---:|---:|---:|---:|
| m02_matmul_rect | 0.45 s | 130 | 33 | 0 |
| 2048x2048x2048 | 0.50 s | 1614 | 128 | 0 |
| split-K 512x4096 | 0.46 s | 1620 | 120 | 0 |
| 128x128x128 holdout | 0.42 s | 719 | 72 | 0 |

PTX virtual registers are the sum of the `.reg` declaration counts in the emitted PTX; SASS
registers are from `ptxas -arch=sm_89 -v`, the target the compiler emits.

Compile time is measured as wall time for one `compile.py` invocation on the local machine. The
runner's time limit is generous; the largest kernel is under a second.

## Known measurement caveats

- The runner has two stable clock states for short kernels. The same binary measured 3.30 and
  5.02 us on m02 in two sessions while p00 and 2048 stayed fixed. Absolute numbers are not
  portable between sessions; ratios and same-session medians are.
- Warm numbers are quantized to 32 ns. A difference at or below one tick is noise.
- Cold medians move with the launch floor and the clock state. A cold difference is noise when
  the ranges overlap and the medians are within 1.5 percent.
- The 46 variant packages are the checked-in regression set. The 86 corpus IRs are generated
  outside the repository from the specs in `tools/specs`; they are not committed.
