# Benchmark evidence

`matrix.csv` is the curated 88-cell matrix: 44 shapes in warm and cold mode. It is the source
for the `ours` values in the README and the tables in [../../BENCHMARKS.md](../../BENCHMARKS.md);
the cleanup baseline is recorded in that page.

## Columns

| column | meaning |
|---|---|
| `cell` | short cell id used in the CSV and BENCHMARKS |
| `family` | `open`, `attention`, `matmul` or `holdout` |
| `mode` | `warm` or `cold` |
| `session` | `S1` main matrix, `S2` deep recheck, `S3` 128 holdout unroll |
| `ours_n` | independent rounds for this compiler |
| `ours_median`, `ours_min`, `ours_p10`, `ours_p90`, `ours_max` | round statistics, in microseconds |
| `note` | edges, unstable cells and session caveats |

## Reading the rows

- Warm: a tie at the 3 us floor is one 32 ns tick; the note marks it.
- Cold: small-kernel ranges overlap across the runner's two clock states; the note marks it.
- A cell with two same-code sessions that disagree is marked unstable in the note.
- The minimum is context only. The cleanup baseline is in
  [../../BENCHMARKS.md](../../BENCHMARKS.md); this CSV is our absolute evidence.

## Sessions

| session | contents |
|---|---|
| `S1` | the main warm and cold matrix (43 shapes in this CSV; `h_mm_128` is S3), 5 warm rounds and 5 to 10 cold rounds |
| `S2` | deep rechecks: m01 cold 10 rounds, p03 cold 10, f01 cold 20, h_mm_96 warm 15, x128h96 warm 5 |
| `S3` | the 128x128x128 all-shared unroll confirmation, warm 7 and cold 10 |

The runner has two clock states for short kernels. `manifest.json` records the dates and the
commands; `BENCHMARKS.md` records the caveats. The curated CSV is the redistributable evidence;
raw per-round JSON is not part of this release.

## Holdout

The 8-shape holdout was generated outside the repository with the task generator from
`tools/specs/holdout.py`:

```bash
TASK=../task
../venv/bin/python "$TASK/tools/generate.py" --emmy tools/emmy-torch --out ../holdout \
  tools/specs/holdout.py
```

The generated packages are not committed. The spec is, so the holdout can be rebuilt.
