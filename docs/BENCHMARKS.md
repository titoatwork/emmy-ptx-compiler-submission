# Benchmarks and methodology

Two numbers exist for every kernel and they answer different questions.

* **Cold (harness metric)**: one launch, `warmup=0, iters=1`, as `submit.py` reports it. This is
  what the score sees. Launch overhead dominates the small kernels, and the number moves with the
  GPU's clock state.
* **Warm (code quality)**: `warmup=10, iters=100`, via `tools/prof.py`. The runner reports a
  stable median (often quantized to 32 ns). Use this to judge a codegen change.

Never compare a cold number for one compiler against a warm number for another.

## Final matrix

Measured 2026-10-27 on the hosted RTX 5090 runner, interleaved A/B with the order rotating every
round. Every row is our median of the stated number of samples. `cleanup baseline` is the frozen
pre-optimization measurement of this compiler from the same session; the baseline compiler is not
part of this release. A dash means the baseline was not measured in that mode. The 44 shapes (88 cells) are 15 open problems, 16
generated attention shapes, 5 matmul variants and 8 holdout shapes generated with the task
generator outside the repository.

Rows marked `S2`/`S3` in [docs/benchmarks/final/matrix.csv](benchmarks/final/matrix.csv) are deep rechecks; the baseline column is from the main session and is an ablation, never the headline. The runner has two clock states for short kernels. In one session m02 warm measured 3.30 us and
in another 5.02 us, while p00 stayed at 2.96 us and 2048 at about 315 us; the same PTX moved in
both directions. The CSV states the session. In the tables below, `x128h96` and `h_mm_96` warm and `h_mm_128` warm/cold, plus `p03`, `m01`
and `f01` cold, are S2/S3 rechecks; every other row is S1. Absolute medians are session-specific; the S1 medians are same-session and S2/S3 are
labeled rechecks.

### Warm

| cell | ours | cleanup baseline | n |
|---|---:|---:|---:|
| p00_scalar_add | 3.040 | 3.040 | 5 |
| p01_vector_add | 3.040 | 3.040 | 5 |
| p02_scale_shift | 3.104 | 3.152 | 5 |
| p03_relu_mul | 3.040 | 3.040 | 5 |
| r01_array_sum | 3.008 | 3.040 | 5 |
| r02_big_sum | 8.704 | 8.704 | 5 |
| r03_mean | 6.688 | 6.688 | 5 |
| r04_row_sum | 3.040 | 3.040 | 5 |
| m01_matmul_512 | 13.872 | 14.880 | 5 |
| m02_matmul_rect | 3.296 | 4.896 | 5 |
| f01_rmsnorm | 4.640 | 4.672 | 5 |
| f02_row_softmax | 3.040 | 3.040 | 5 |
| f03_l2norm | 3.040 | 3.008 | 5 |
| a01_attention | 27.136 | 27.024 | 5 |
| a02_attention_128 | 10.816 | 10.816 | 5 |
| attn 16x64 | 8.704 | 8.704 | 5 |
| attn 256x16 | 8.752 | 8.768 | 5 |
| attn 32x32 | 6.688 | 6.656 | 5 |
| attn 512x16 | 147.984 | 147.968 | 5 |
| attn 64x256 | 35.360 | 35.360 | 5 |
| attn 128x128 | 195.104 | 195.136 | 5 |
| attn 128x64 | 12.832 | 12.832 | 5 |
| attn 256x32 | 45.600 | 45.600 | 5 |
| attn 256x64 | 113.184 | 113.184 | 5 |
| attn 64x128 | 61.984 | 62.016 | 5 |
| attn 100x32 | 23.104 | 23.072 | 5 |
| attn 128x96 | 125.424 | 123.424 | 5 |
| attn 33x64 | 16.912 | 16.928 | 5 |
| attn 48x96 | 16.928 | 16.960 | 5 |
| attn 64x48 | 39.424 | 39.440 | 5 |
| attn 256x128 | 256.480 | 259.312 | 5 |
| mm 2048x2048x2048 | 302.512 | 322.880 | 5 |
| mm split-K 512x4096 @ 4096x512 | 115.424 | 123.552 | 5 |
| mm 64x17 @ 17x96 | 4.640 | 6.688 | 5 |
| mm 64x8 @ 8x64 | 3.136 | 3.200 | 5 |
| mm 128x64 @ 64x1 | 3.072 | 3.072 | 5 |
| holdout mm 96x80x48 | 3.200 | 4.864 | 15 |
| holdout mm 1024^3 | 70.208 | 70.208 | 5 |
| holdout mm 64^3 | 3.072 | 3.072 | 5 |
| holdout mm 128^3 | 4.672 | 6.688 | 7 |
| holdout bmm 4x32x64 @ 4x64x32 | 3.136 | 3.136 | 5 |
| holdout mm 97x83 @ 83x59 | 8.736 | 8.736 | 5 |
| holdout attn 48x16 | 8.704 | 8.704 | 5 |
| holdout attn 96x32 | 10.784 | 10.816 | 5 |

### Cold

| cell | ours | cleanup baseline | n |
|---|---:|---:|---:|
| p00_scalar_add | 23.632 | - | 10 |
| p01_vector_add | 23.440 | - | 10 |
| p02_scale_shift | 16.832 | - | 10 |
| p03_relu_mul | 16.560 | - | 10 |
| r01_array_sum | 23.312 | - | 10 |
| r02_big_sum | 28.224 | - | 10 |
| r03_mean | 23.424 | - | 10 |
| r04_row_sum | 24.336 | - | 10 |
| m01_matmul_512 | 26.656 | 27.872 | 10 |
| m02_matmul_rect | 16.624 | 16.544 | 10 |
| f01_rmsnorm | 17.392 | - | 20 |
| f02_row_softmax | 16.544 | - | 10 |
| f03_l2norm | 16.624 | - | 10 |
| a01_attention | 37.344 | - | 5 |
| a02_attention_128 | 22.688 | - | 5 |
| attn 16x64 | 26.864 | - | 10 |
| attn 256x16 | 20.512 | - | 10 |
| attn 32x32 | 23.680 | - | 10 |
| attn 512x16 | 159.808 | - | 5 |
| attn 64x256 | 46.816 | - | 5 |
| attn 128x128 | 206.720 | - | 5 |
| attn 128x64 | 23.808 | - | 5 |
| attn 256x32 | 57.376 | - | 5 |
| attn 256x64 | 124.960 | - | 5 |
| attn 64x128 | 74.176 | - | 5 |
| attn 100x32 | 34.752 | - | 5 |
| attn 128x96 | 137.152 | - | 5 |
| attn 33x64 | 28.864 | - | 5 |
| attn 48x96 | 28.608 | - | 5 |
| attn 64x48 | 49.696 | - | 5 |
| attn 256x128 | 267.776 | - | 5 |
| mm 2048x2048x2048 | 338.560 | 358.048 | 5 |
| mm split-K 512x4096 @ 4096x512 | 126.944 | 134.880 | 5 |
| mm 64x17 @ 17x96 | 16.688 | 16.608 | 10 |
| mm 64x8 @ 8x64 | 16.368 | - | 10 |
| mm 128x64 @ 64x1 | 23.600 | 23.440 | 10 |
| holdout mm 96x80x48 | 16.352 | - | 10 |
| holdout mm 1024^3 | 81.568 | - | 5 |
| holdout mm 64^3 | 16.848 | - | 10 |
| holdout mm 128^3 | 16.608 | 18.192 | 10 |
| holdout bmm 4x32x64 @ 4x64x32 | 16.352 | - | 10 |
| holdout mm 97x83 @ 83x59 | 20.608 | - | 10 |
| holdout attn 48x16 | 27.152 | - | 10 |
| holdout attn 96x32 | 22.144 | - | 5 |


## Selected residual cells

The full 88-row absolute matrix is in the curated CSV. The 1024 holdout is the slowest cell; the
others show the 3 us floor and the S3 128 holdout.

| cell | warm | cold | note |
|---|---:|---:|---|
| holdout mm 1024^3 | 70.208 | 81.568 | the slowest holdout; hosted sm_120 SASS 1268 on CUDA 12.9 (local sm_89 1240), 88 blocks of 512; scheduling residual |
| holdout mm 128^3 | 4.672 | 16.608 | the long all-shared loop unroll |
| attn 256x16 | 8.752 | 20.512 | the smallest cold median; launch-bound |
| m02_matmul_rect | 3.296 | 16.624 | warm at the 3 us floor; cold within 0.7 percent of the cleanup baseline |
| n1 128x64 @ 64x1 | 3.072 | 23.600 | the smallest matmul |

The 1024x1024x1024 holdout warm is the only holdout cell above 70 us. Its hosted sm_120 SASS is
1268 instructions on the runner CUDA 12.9 (local sm_89 ptxas 12.6 gives 1240). The holdout package is generated from the
public spec `tools/specs/holdout.py`; its meta launches 88 blocks of 512 threads on a 170-SM GPU,
so it runs less than one block per SM and is latency-bound. The next work is the ptxas schedule, not a missing
mechanism. Cold is 81.568.

## Attention sweep

The 16 generated attention shapes run from 6.7 us to 256.5 us warm. The smallest cold median,
256x16, is launch-bound at 8.752 warm and 20.512 cold. a01 is 27.136 warm and 37.344 cold; a02 is 10.816
warm and 22.688 cold. The largest shape, 256x128, is 256.480 warm. All 16 are correct on the
local GPU and the runner.

## How to reproduce

```bash
# cold, hosted runner, one problem (the task repository is a sibling checkout)
cd ../task
../venv/bin/python submit.py a02_attention_128 \
  --compiler ../emmy-ptx-compiler-submission/compile.py --server "$RUNNER_URL"

# warm steady state, same runner (from this repository)
cd ../emmy-ptx-compiler-submission
../venv/bin/python tools/prof.py a02_attention_128

```

Set `RUNNER_URL` to the event-provided runner. The task's `submit.py` uses its own default when
`--server` is omitted. `tools/prof.py` prints the register count, the per-repeat medians and the
best value; it does not check correctness.

## Measurement traps

* **Two runner states.** The same binary can move by 50 percent on short kernels between
  sessions. Interleave A/B and report the session.
* **Quantization.** 32 ns boundaries. A one-tick difference on a 3 us kernel is noise.
* **Cold variance.** Ranges overlap. A 1.5 percent median difference with overlapping ranges is
  noise; a single cold sample is not evidence.
* **Dead-code elimination.** Ablations can delete dependent math. Use them for attribution only.
* **Stale cubins and session drift.** Recompile every run and record the order.
