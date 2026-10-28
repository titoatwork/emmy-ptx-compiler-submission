# Local tools

Everything here talks to the same hosted runner the judge uses, or to a local CUDA GPU for fast
debugging, or checks the repository itself. The runner endpoint is provided to participants;
export `RUNNER_URL` before using the runner tools. `compare.py` takes two compiler checkouts;
the public tree contains one compiler.

Install the tool dependencies once:

```bash
python3.12 -m venv ../venv
../venv/bin/python -m pip install -r requirements-tools.txt
```

Activate it (`source ../venv/bin/activate`) or prefix Python commands with
`../venv/bin/python`. The gate and test scripts auto-detect `../venv` when `PYTHON` is unset.

Expected layout (the scripts use these defaults; the parent directory name is arbitrary):

```
task/          # organizer repo: problems/, submit.py, tools/generate.py
emmy-ptx-compiler-submission/    # this repository
stress/ blocks/ fixed/ ln/ small/ f16/ fuzz/      # generated corpora
attngen/ attnedge/ attnadv/                        # attention shape sweeps
ops/ dtypeedge/ adv/                               # operator, dtype and adversarial corpora
holdout/                                           # generated holdout packages
```

Override any default with environment variables: `TASK`, `RUNNER_URL`, `PYTHON`, `TORCH_PYTHON`,
`CUDA_LIB`, `PTXAS`, `LOCAL_ARCH`, `VARIANTS`, `COMPILER`, `EMMY`, `OUT`.

## Scripts

| script | purpose |
|---|---|
| `doctor.py` | Report Python, `emmy`, tool dependencies, `ptxas`, CUDA, GPU, task repository, corpora, variants, dash check and optionally the runner. Prints suggested fixes. |
| `bootstrap.sh` | Print or execute the fresh-machine setup: venv, tool dependencies, task clone. `--run` executes, `--with-torch` adds torch. |
| `verify.sh` | One-command gate. Default is local and network-free; `--corpora` adds the variants and generated corpora on the local GPU; `--runner` adds the 15 open problems; `--full` adds both. |
| `prof.py <problem>` | Compile with this repository's compiler and measure on the runner. Default warm (10 warmup, 100 iters); add `--warmup 0 --iters 1` for the cold harness metric. Prints registers and samples. |
| `compare.py --a A --b B` | Interleaved A/B benchmark over the open set. `--mode warm` for code quality, `--mode cold` for the score. Alternates the order per round, reports the median of the per-round medians as the result and the minimum for context. Pass `--problems` with package paths to measure generated or holdout shapes. |
| `localrun.py <problem_dir> <compiler.py>` | Compile and run locally on a CUDA GPU, compare against `expected.npz`. Auto-detects `libcuda`, `ptxas` and the device architecture. |
| `checkvariants.sh` | Run the 46-case regression in `tests/variants` on the local GPU. `VARIANTS=<dir>` runs another corpus. |
| `testall.sh` | Run all 15 open problems through the hosted runner. Exits non-zero on any failure. |
| `check_dashes.py` | Reject em dash (U+2014) and en dash (U+2013) in text files. Run manually or from your own CI. |
| `generate_corpus.sh` | Regenerate the corpora from `tools/specs` with the task's generator. Prefers the project venv CLI when it can import torch. |
| `emmy-torch` | Wrapper that runs the `emmy` CLI with a torch-enabled Python when torch lives in a separate venv. |

## Specs

`tools/specs/*.py` are the expressions used to generate the corpora. Each file exports `OPEN`, a
list of dicts with `id`, `code`, `ref`, `rtol`, `atol` (see the task's `tools/generate.py`). The
attention sweeps (`attn_extra`, `attn_edge`, `attn_adv`) cover sequence 16 to 512 and head 16 to
256, including odd and non-power-of-two shapes.

## Typical loops

```bash
# environment and fast gate
python tools/doctor.py --runner
tools/verify.sh --quick

# correctness on the local GPU and the hosted runner
tools/checkvariants.sh
TASK=../task tools/testall.sh

# every local variant and generated corpus
tools/verify.sh --corpora

# one warm number and one cold number
../venv/bin/python tools/prof.py a01_attention
../venv/bin/python tools/prof.py a01_attention --warmup 0 --iters 1

```

Read `docs/BENCHMARKS.md` before trusting any number: cold small-kernel measurements move with
GPU clock state, and single samples are not evidence. `docs/VERIFICATION.md` records the gates.
