# Design notes

This document explains how the compiler turns Emmy's Kernel IR into PTX, what is covered, how it
was tested, and where the remaining gaps are. The README is the quick start; this is the
engineering record.

## Pipeline

```
Kernel IR json --> Graph.from_dict --> KernelOp body --+
                                                       +--> Lowerer --> PTX text
meta.json (arg_order, grid, block, smem) --------------+
```

`compile.py` loads the IR through Emmy (the required deserializer), finds the `KernelOp`, looks up
the launch descriptor next to the IR (`meta.json`, matched by entry name), and hands both to
`Lowerer`. The whole backend is a single walk over the typed statement tree; there is no
intermediate IR and no C++ step.

| module | responsibility |
|---|---|
| `ptxc/lower.py` | driver: parameter prologue, shared bases, Tile decode and guard, module assembly |
| `ptxc/stmts.py` | statement walk: Load/Write, Assign/Accum, Cond, loops, Sync, WarpShuffle, TreeHalve, cp.async, ZeroPrologue |
| `ptxc/expr.py` | SSA and expression lowering: 32-bit indices, float/64-bit expressions, predicates, `where`, elementwise ops |
| `ptxc/memory.py` | shared-slab layout, flat indices, global/shared addresses, swizzle, alignment proof |
| `ptxc/common.py` | dtype tables, reduction identities, IR-tree utilities (free vars, defined names, unroll heuristics) |
| `ptxc/ops.py` | emmy op name to PTX spelling, dtype widths |
| `ptxc/emitter.py` | virtual registers, labels, PTX text |

## Three invariants

**1. The launch is the contract.** `meta.json` gives the parameter order, grid, block and shared
budget, so code generation never guesses them. The Tile axes are decoded exactly the way Emmy's
CUDA renderer does it: one flattened work id

```
gid = flatten(blockIdx, blockDim, threadIdx)
idx_i = (gid / suffix_i) % extent_i
```

That makes a single `gid < N` guard sufficient for any over-provisioned launch (`p00` runs 64
cells on 256 threads). Skipping the final modulo was the first real bug: every thread in a row
reduction started at its global id and only the first rows were written. Skipping the guard was
the second.

**2. Accumulators are loop-carried, and seeds belong to a loop instance.** The loop body is
emitted once, so a fold writes back into the same virtual register. Emmy declares an accumulator
where its carrying loop is entered, which is not necessarily kernel scope: `a01` has four dot
products over `a2` inside one `a1` loop. Seeding by axis name resets the siblings when the second
loop opens and silently turns three of four attention rows into `sum(v)/n`. The seeder therefore
associates the seed with the loop *statement*. Accumulators with an explicit `Init` are seeded by
that statement instead (`m02`).

**3. State read under a predicate must be defined unconditionally.** `lane`, `warp` and every
shared-memory base are materialized in the prologue. The first version materialized a base at its
first use, which in the reduction sits inside `if (lane == 0)`; every other thread then used an
undefined address and the runner reported an illegal memory access. The same class of bug appears
with axis shadowing: a sibling loop reusing an axis name (`_fa` in matmul) froze the stale
counter inside a hoisted expression, which produced out-of-bounds `cp.async` destinations.

## Statement lowering

| IR | lowering |
|---|---|
| `Tile` | `gid` decode (direct `blockIdx*blockDim+tid` for one-dimensional launches), per-axis modulo (skipped for a single axis), `gid < N` guard |
| `Load` / `Write` | flat index; vector `v2`/`v4` when the IR groups values; adjacent column pairs and k-adjacent pairs across unrolled copies are fused into `v2`/`v4`; identical loads are reused; read-only global tiles and block row slices are redirected to shared-memory caches |
| `Write(atomic)` | `red.global.add.<ty>` |
| `Assign` / `Accum` | elementwise op table; `Accum` folds in place; `acc += a*b` with a single-use product becomes one IEEE `fma.rn` |
| `Cond` | `setp` + predicated branch, optional else |
| `Loop` / `StridedLoop` | counter loop; constant-trip loops with small bodies are fully unrolled with literal axes (compiler-side constant folding of the axis arithmetic); larger global loops are partially unrolled in chunks; pure bodies batch their loads before the arithmetic; a warp-uniform dot becomes a lane-strided loop plus a butterfly shuffle reduction |
| `WarpShuffle` | XOR butterfly `shfl.sync.bfly.b32` over `length/2 ... 1`, then the combine statements |
| `TreeHalve` | shared-memory halving tree over every coupled state, one barrier per level, broadcast from slot 0; `inner` slot addressing supported |
| `Smem` | static arrays below 48 KiB, `.extern .shared` pool at a matching offset when the launch reserves dynamic memory |
| `Sync` | `bar.sync id` / `bar.sync id, count` / `bar.warp.sync` |
| `CpAsyncCopy` | `cp.async.cg` (16 B) or `.ca` (4/8 B); lane offset `row*cols + col`; XOR swizzle |
| `CpAsyncCommit` / `CpAsyncWait` | `commit_group` / `wait_group N` |
| `ZeroPrologue` | every grid thread sweeps a stride of 32-bit words |
| `Let` / `Init` | literal or index expression into a register; dtype-correct seeds |

## Coverage

Implemented and exercised by the open set: every unmarked statement in the task reference.

Implemented and exercised by generated IRs: the same, plus 64/128/256/512-thread reductions
(`TreeHalve` lengths 2/4/8/16), cross-block atomic reductions (atomic `Write`),
split-K matmuls, batched and rectangular matmuls, layer-norm (three coupled accumulators),
short rows (7, 32), long rows (4096), f16 and f64 pointwise, f16 row sums and row maxima,
f16 softmax and layer norm, f64 reductions, and the erf/gelu/gelu_tanh/softplus and
sin/cos elementwise family.

Implemented with unit tests but not generated by the current task tooling: warp-level `Sync`,
named barriers with and without a count, 8-byte `cp.async`, smem write swizzles, `TreeHalve.inner`,
scalar parameters. Unknown dtypes, ops and statements raise `CompileError` instead of emitting a
guess; the tests assert the failure text and exit code.

Not implemented: the `*` constructs in the task reference (tensor-core `mma`/`wgmma`, `Ldmatrix*`,
`RegStore`, TMA/mbarrier, `Select`/`Pre`/`Carry`, `FuncCallExpr`, `Builtin` casts beyond the
integer builtins). They appear in no open or hidden problem by the task's own statement.

## Performance

The final matrix is in [docs/BENCHMARKS.md](docs/BENCHMARKS.md). It covers 44 shapes in warm
and cold mode, with medians and sample counts. The headline: a01 is 27.136 warm and 37.344
cold; a02 is 10.816 and 22.688; 2048 is 302.512 and 338.560; split-K is 115.424 and 126.944; the
16 attention shapes run from 6.688 to 256.480 warm. The slowest holdout matmul is the
1024x1024x1024 cell at 70.208 warm and 81.568 cold; its hosted sm_120 SASS is 1268 instructions on
the runner CUDA 12.9 (local sm_89 ptxas 12.6 gives 1240) and the holdout meta launches 88 blocks of 512 threads on 170 SMs,
so the next work is scheduling. The old tables that used mixed
sessions and best-of-N minima were removed because they distorted the comparison.

### What the optimization pass found

Each change below was selected from the runner's disassembly and measured; the final medians are
in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

1. **Launch bounds.** The GEMM and attention kernels are latency-bound with one block per SM.
   Without `.maxntid`, ptxas reserves registers for the 1024-thread worst case. Declaring the
   real block size raised the attention register budget and cut the time: more loads in flight,
   not fewer instructions.
2. **Load batching.** An unrolled `load/use/load/use` stream exposes one global latency per load
   in a one-warp block. Issuing a group's loads first and the arithmetic second turns N exposures
   into one.
3. **Register hoisting with per-copy rebinding.** The attention query row is invariant across
   the key loop. It is loaded once into registers and rebound to the IR name at each unrolled
   copy, which needs no renaming pass.
4. **Vector fusion in shared memory.** The matmul inner loop reads two adjacent columns and two
   k-adjacent values per copy. Fusing both into `LDS.64` removed every scalar shared load.
   Widening the k-pair to `LDS.128` was tested and rejected: ptxas scheduled it worse even
   though it removed instructions.
5. **Literal-axis folding.** A fully unrolled copy knows its axis value, so stage math and
   selects evaluate at compile time. The first version folded `chunk_base + k` as `k`; the
   layer-norm corpus caught it, and the fix is guarded by both a shadowing rule and the corpus.
6. **Rejected: fusing attention's four dot loops.** It removes duplicate key-column loads, but
   ptxas responds by dropping the register budget and the kernel gets slower. Redundant loads
   are cheaper than the scheduling freedom they buy.
7. **Rejected: splitting cp.async addresses into base plus immediate.** It produced the intended
   code shape and changed nothing measurable, so it was removed.
8. **Read-only tile caches.** A block of attention re-reads the whole key tile for every key.
   The tile fits in shared memory, so one cooperative fill, transposed when the dot axis is its
   first coordinate, replaces thousands of global loads with 128-bit shared loads. For a query
   too large to cache whole, the block's row slice is cached instead (blocks partition the first
   dimension, so the slice is contiguous). This is the largest attention change.
9. **Warp-cooperative dot products.** In a one-row-per-warp schedule every lane computes the
   same score. The loop is emitted lane-strided, each lane sums head/32 terms, and a butterfly
   shuffle adds the partials into every lane. The tree width is the warp subgroup that actually
   shares inputs (32 for a02, 16 for a01's two-rows-per-warp layout).
10. **Rejected: nothing here removes latency.** `a01` did not improve when its dot became
    cooperative; the online-softmax dependency chain and the shuffle tree are its bound. The
    remaining a01 work is scheduling, not another missing transformation.

### Remaining gaps

The final matrix is in [docs/BENCHMARKS.md](docs/BENCHMARKS.md): 88 absolute cells with medians
and sample counts. The slowest holdout matmul is the 1024x1024x1024 cell warm at 70.208, cold
81.568; the hosted sm_120 SASS is 1268 instructions on the runner CUDA 12.9 (local sm_89 ptxas 12.6 gives 1240) and the holdout meta
launches 88 blocks of 512 threads on 170 SMs. The next work is
ptxas scheduling on that shape, not a missing transformation. Further micro-optimization there
has low expected value and real regression risk, which is why the docs and the corpus take
priority.

### Measured gating

The affine address cache improves the large matmuls but regresses a 256-tile attention shape.
The lane fold helps split-K but regresses the 2048 matmul, and the full shared-address cache also
regresses 2048. A static tile-count and fan-out gate removes the measured regressions while
keeping most of the large-shape gain. We report this as a measured instance of phase ordering
and a guardrail, not as a new optimization or a general law.

### Exactness and precision

Elementwise float arithmetic uses explicit `.rn` spellings. PTX only permits ptxas to contract an
*unrounded* multiply and add into an FMA, and a contracted chain differs from the IEEE result the
reference NumPy expression computes. Pointwise problems are graded tightly, and several use `rtol=atol=0`, so they must not contract;
accumulations still use `fma.rn` because they are reductions with loose tolerances (and `m01`
keeps its FMA inner loop).

Transcendentals were checked against the nvcc reference over dense input ranges:

| op | ours max abs / rel | nvcc reference max abs / rel |
|---|---|---|
| exp, x in [-10, 10] | 9.8e-3 / 7.4e-7 | 6.8e-3 / 5.2e-7 |
| sigmoid, x in [-20, 20] | 6.0e-8 / 1.6e-6 | 6.0e-8 / 1.0e-6 |
| silu, x in [-10, 10] | 9.5e-7 / 7.1e-7 | 9.5e-7 / 5.0e-7 |
| tanh, x in [-6, 6] | 1.8e-7 / 2.1e-7 | 1.2e-7 / 2.0e-7 |

The first `tanh` implementation used `tanh.approx.f32`, which was less accurate than the
reference (7.3e-6 absolute). It is now a branch-free split: a [5/4] Pade approximant for |x| < 1,
and `1 - 2/(exp(2x) + 1)` outside, selected with `selp`. Nothing uses relaxed-precision flags.

## Testing

- **Open set through the hosted runner**: all 15 problems, all five categories.
- **Generated corpus**: 86 IRs from 13 generated directories, plus 46 checked-in variants and
  the 15 open problems, 147 local IRs in total, covering shapes, block sizes, axes, dtypes,
  matmul tiles and attention head dimensions. Run locally against `expected.npz`.
- **Attention shapes beyond the open set**: sixteen generated variants with sequence 16 to 512 and
  head 16 to 256, including odd sequence lengths (33, 100) and non-power-of-two heads (48, 96).
  All correct; the 16 shapes run from 6.688 to 256.480 warm, and the 256x16 head-16 shape is
  launch-bound at 8.752 warm and 20.512 cold. This sweep
  found the only compile-time crash of the project: a non-affine adjacent index made the pair
  planner unpack `None`. The fix leaves such loads scalar instead of crashing.
- **Unit tests** (`tests/test_lowering.py`): stride rules (one index is flat), the affine
  alignment proof, PTX contract invariants over every open IR.
- **Machinery tests** (`tests/test_machinery.py`): constructs the open set does not exercise, plus
  loud-failure tests for unknown dtypes and statements. `ZeroPrologue` is unit-test only.

## Known gaps and next steps

- **The 1024 holdout is the remaining scheduling residual.** Its hosted sm_120 SASS is 1268
  instructions on the runner CUDA 12.9 (local sm_89 ptxas 12.6 gives 1240); the holdout meta launches 88 blocks of 512 threads
  on 170 SMs, so the gap is ptxas order, not a missing mechanism. m01 is
  13.872 warm and 26.656 cold; a01 is 27.136 and 37.344.
- **Cost model for unrolling** is a size heuristic, not a search. The
  literal-axis folding and chunk batching are conservative by design.
- **Address strength reduction** (induction pointers) was attempted for
  cp.async and removed after it changed nothing; it may still help a different
  pipeline shape.
- **Occupancy tuning** is now explicit through `.maxntid`; no `.maxnreg`.
- **Serial axes** (`KernelOp.serial`) are unimplemented; the task states they
  are empty in every problem.
- **Multi-kernel problems** are covered by the two-kernel `adv_mm_bias_relu`
  case and by a meta-selection regression test.

The function-level map is in [docs/CODEMAP.md](docs/CODEMAP.md).


## Codegen mechanisms

The matmul gap was structural, not a failure of the IR lowering. Four mechanisms carry the
codegen gains; the rest are supporting changes.

### Dominating integer literals

`ptxc/emitter.py` materializes each distinct `s32` literal once and places the move in a block
emitted ahead of the kernel body. A literal first seen in a conditional arm therefore dominates
every later use, which is the soundness property the first implementation lacked.
`Emitter.emit` parses the destination of a 32-bit write and invalidates the literal record, so
`s32_of` cannot report a stale value. `ptxc/expr.py` `_s32_bin` folds identities and two-literal
add/subtract/multiply/shift/bitwise expressions. Signed 32-bit overflow is not folded; the PTX
operation is left in place so the hardware wrapping is preserved.

### Affine global addresses

`ptxc/memory.py` `_cse_global_addr` splits an affine index into a symbolic runtime key and a
constant. The first access computes the address; later accesses with the same key reuse it with
an immediate offset, or rebase when the offset would be negative. The cache is cleared at every
basic block, every hoisted copy, and every axis rebind. It is disabled below 512 tiles because
on tiny grids the shorter address chains perturb ptxas scheduling more than they save.

### Copy loops and copy destinations

`ptxc/stmts.py` `_single_trip` turns a strided loop whose bound fits in one step into a guarded
straight-line body. `_strided_bounded_plan` handles a variable start (`fa in start..640:256`):
it unrolls to the maximum, leaves the copies every thread reaches unguarded, and guards only the
tail. `_strided_unrolled` reuses a destination base across the unrolled copies when the run has
at least four copies; the copy destination pairs `(x / cols, x % cols)` are folded back to `x`
for this path. Async-copy lane offsets are folded only when at most two copies share them; with
four or more ptxas already common-subexpresses the divide and remainder. `ptxc/common.py`
`_partial_unroll_factor` unrolls a loop longer than 64 iterations by 16 when every load is from
shared memory; the low latency per iteration makes the branch removal pay, and the expanded-body
cap keeps it from growing without bound.

### Tile decode and shared widths

`ptxc/lower.py` `_emit_tile` decodes power-of-two strides and extents with shifts and masks on
tile grids of 1024 or more. Below that the divide and remainder form is kept: the shorter form
changed ptxas scheduling on a small attention block. Exact launches skip the
`gid < n` guard. `ptxc/stmts.py` `_plan_pairs` fuses runs of up to four consecutive single-name
loads into one vector load, gated on the index alignment proof and, for shared buffers, the
declaration alignment. An eight-byte declaration never receives a sixteen-byte load.
`_shuffle` emits `shfl` directly on the float register; the `.b32` instruction accepts it.

### Invariants

The mechanisms preserve the two failure modes that have caused real bugs: state read under a
predicate is defined unconditionally, and a cached value is invalidated when the registers it
reads are rebound. The literal block is unconditional; the address caches are per basic block and per
hoisted copy; the copy destination and lane caches are per unrolled run.

## Related work

[LLVM's NVPTX backend](https://llvm.org/docs/NVPTXUsage.html) and
[libNVVM](https://docs.nvidia.com/cuda/libnvvm-api/index.html) lower LLVM IR to PTX; this project
lowers Emmy's Kernel IR, a different input, and makes no priority claim.
[TVM](https://arxiv.org/abs/1802.04799) and [IREE](https://github.com/iree-org/iree) are tensor
compilers with their own scheduling and codegen; the delta here is a small, readable backend with
measured guards for this IR. [Triton](https://github.com/triton-lang/triton) and
[tinygrad](https://github.com/tinygrad/tinygrad) compile Python-level tensor programs; this
project starts at Emmy's Kernel IR. [FlashAttention](https://arxiv.org/abs/2205.14135) and
[CUTLASS](https://github.com/NVIDIA/cutlass) establish tiled attention, online softmax and
`cp.async` GEMM staging; [CUB](https://github.com/NVIDIA/cub) establishes warp reductions. The
individual mechanisms here are standard: SSA and phi construction, loop rotation and unrolling,
scalar evolution and alignment proofs, divmod recombination, `cp.async` staging and tiled online
softmax. They are re-derived for Emmy's IR, not claimed as new. Phase ordering and optimization
gating are long-studied ([Whitfield and
Soffa](https://doi.org/10.1145/99163.99179)). GPU latency hiding is also established
([Volkov](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-143.html)). The contribution
here is a measured instance and a static guard, not a new principle.
