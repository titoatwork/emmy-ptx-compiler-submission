# From PTX text to machine code

The optimization work changed the generated machine code. Each accepted change was one
conceptual change, measured with an interleaved A/B on the same runner, and kept only if the
target cell moved and no other cell regressed beyond noise. The final medians are in
[BENCHMARKS.md](BENCHMARKS.md).

## What changed

| mechanism | file | effect |
|---|---|---|
| integer literals materialized in a dominating block | `ptxc/emitter.py`, `ptxc/expr.py` | shorter index expressions, no stale cache |
| affine global address cache, cleared per block and per hoisted copy | `ptxc/memory.py`, `ptxc/stmts.py` | fewer global address computations, gated by tile count |
| single-trip strided loops as guarded bodies | `ptxc/stmts.py` | fewer instructions and branches |
| bounded copy unroll with tail guards | `ptxc/stmts.py` | fewer loop-carried dependencies |
| four-wide shared load fusion with alignment guards | `ptxc/stmts.py` | fewer shared load instructions |
| divide and remainder recombination for copy destinations | `ptxc/memory.py` | shorter copy addresses |
| lane folding at low fan-out | `ptxc/stmts.py` | fewer shuffle instructions on split-K |
| shift and mask tile decode on large grids only | `ptxc/lower.py` | cheaper tile decode on large grids |
| exact-launch guard elision | `ptxc/lower.py` | one predicate and branch removed on exact launches |
| float shuffle without the b32 round trip | `ptxc/stmts.py` | fewer instructions on reductions |

## What was rejected

- An unsafe literal cache that materialized a value at first use inside a conditional arm and
  reused it after the join. ptxas -O3 hid it; ptxas -O0 produced a wrong result. The safe
  version materializes ahead of the body.
- A global address cache without invalidation when a hoisted loop rebinds an axis. The symbolic
  key alone does not identify the register binding. Clearing the cache per hoisted copy fixed
  the three failing attention cases.
- An unconditional width fallback for shared loads. It was SASS neutral and regressed 2048.
- An unconditional lane fold. It helped split-K and regressed 2048 through ptxas scheduling;
  the fold is now limited to low fan-out.
- A fully general shared-address cache. It cut registers but regressed 2048. Reverted.
- Dropping the address cache to protect attention. It restored attention but gave back k17. The
  final gate is a tile-count threshold, which keeps both.

## Results

Warm, RTX 5090, same session, medians. `cleanup baseline` is our own frozen pre-optimization
measurement; the baseline compiler is not part of this release.

| cell | cleanup baseline | ours | note |
|---|---:|---:|---|
| 2048x2048x2048 | 322.9 us | 302.5 us | 6.3 percent lower; cold 338.6 against 358.0 |
| split-K 512x4096 @ 4096x512 | 123.6 us | 115.4 us | 6.6 percent lower; cold 126.9 against 134.9 |
| 64x17 @ 17x96 | 6.69 us | 4.64 us | 31 percent lower; cold 16.69 against 16.61 |
| 128x64 @ 64x1 | 3.07 us | 3.07 us | at the measurement floor; cold 23.60 |
| 128x256 @ 256x64 | 4.90 us | 3.30 us | 33 percent lower; warm at the 3 us floor |

Cold, same method: 2048 is 338.6 us against our 358.0 cleanup baseline (5.4 percent lower);
split-K is 126.9 against 134.9 (5.9 percent lower). m02 is 16.62 against 16.54 (within 0.7
percent) and k17 is 16.69 against 16.61 (within 0.5 percent); both are at the session noise
floor and are reported as edges in [BENCHMARKS.md](BENCHMARKS.md). The 2048 and split-K cold
differences are outside that floor.

Attention remains the strongest absolute family. a01 is 27.136 warm and 37.344 cold; a02 is
10.816 and 22.688. The 16 generated attention shapes run from 6.688 to 256.480 warm, all
correct; 256x16 has the smallest cold median and is launch-bound at 8.752 warm and 20.512 cold. The full sweep,
including the three cases that the unsafe address cache broke, is green on the local GPU and
the runner.

## Measurement rule

Every change has a hypothesis, a SASS count before and after, an interleaved A/B, and a
decision. Warm differences at or below one 32 ns tick are noise; cold differences need
non-overlapping ranges or a median gap above 1.5 percent. The runner has two clock states for
short kernels, so absolute numbers are reported with their session and the cleanup baseline.
