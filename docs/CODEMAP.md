# Code map and vocabulary

This is the navigation document. It explains what each module owns, which function to change for
a given behavior, and the words used in the rest of the documentation. The architecture and the
three correctness invariants are in [DESIGN.md](../DESIGN.md); this file is the index into the
code.

## Vocabulary

| term | meaning |
|---|---|
| **Kernel IR** | Emmy's typed statement tree for one kernel, shipped as `<entry>.ir.json`. |
| **KernelOp** | The IR node that holds the kernel body. `compile.py` looks it up by class name. |
| **meta.json** | The launch descriptor next to the IR: `arg_order`, grid, block, `smem_bytes`, `zero_outputs`, and argument shapes. |
| **Tile** | The IR statement that gives the kernel's iteration space. Its axes are decoded from one flattened work id. |
| **axis** | One Tile dimension. Its **extent** is the constant size. |
| **suffix** | The product of the extents after an axis. `idx = (gid / suffix) % extent`. |
| **gid** | Flattened global work id: `blockIdx * blockDim + threadIdx`, collapsed over the launch axes. |
| **guard** | The single `gid < N` predicate that turns an over-provisioned launch into a no-op for extra threads. |
| **cache** | A read-only global buffer copied into shared memory for repeated reads. It is a whole tile or the block's row slice. |
| **fill** | The cooperative loop that copies a global buffer into a cache, followed by one unconditional `bar.sync 0`. |
| **transposed fill** | A fill that stores the buffer with its coordinates swapped, so the dot axis becomes contiguous in shared memory. Also called a swap cache. |
| **slice cache** | A cache of only the rows a block reads. Valid because blocks partition the first dimension. |
| **block terms** | The per-block offset added to a slice-cache address, derived from the decoded block-level axes. |
| **hoisted load** | An invariant load moved into registers before a loop and rebound to its IR name at each unrolled copy. |
| **literal-axis folding** | Evaluating axis arithmetic at compile time inside a fully unrolled copy. The runtime chunk base must still be added. |
| **partial unroll / chunk** | Emitting a loop body several times with constant offsets while keeping a runtime chunk base. |
| **cross-copy pair** | Two loads from adjacent unrolled copies fused into one vector load (for example `v2` across `k`). |
| **pending queue** | The pair planner's queue that carries a vector load's register bindings into the next copy. |
| **warp dot** | A dot product split across lanes with a butterfly `shfl.sync.bfly` reduction. |
| **group** | The width of the warp subgroup that shares a dot's inputs. It is 32 for a full warp, 16 for two rows per warp. |
| **full_tile** | All lanes of the launch are active and the block is a multiple of 32, so the shuffle mask is safe. |
| **warp_axes** | Tile axes considered warp-varying; used to decide whether a dot can be split across lanes. |
| **fused let** | A single-use multiply that is folded directly into an accumulating `fma`. |
| **CSE** | Common subexpression elimination. Loads use `load_cse`; index subtrees use the hoist cache. |
| **seed** | The identity value written into an accumulator when its loop is entered. |
| **top seed** | An accumulator seeded at kernel scope rather than inside a loop. |
| **coupled state** | Several accumulators reduced together by one `TreeHalve` (layer-norm has three). |
| **dynamic smem** | Shared memory reserved by `meta.json` `smem_bytes`; the compiler lays its own pool out at a matching offset. |
| **cold** | The harness metric: one launch, `warmup=0, iters=1`. |
| **warm** | Steady state: `warmup=10, iters=100`. It measures code, not launch overhead. |

## Pipeline

```
Kernel IR json
    |
    v
compile.py
    Graph.from_dict          emmy deserializer only
    find_kernel              first KernelOp
    find_meta                sibling meta.json, matched by entry
    |
    v
Lowerer(kernel_op, meta)
    __init__                 meta, shared memory, accumulators, cache plan, use counts
    compile()
        _emit_params         ld.param and cvta.to.global for each ordered argument
        _emit_smem_bases     static declarations or the dynamic pool base
        _emit_accumulator_seeds
        _emit_tile           axis decode, cache fills, gid guard
            walk             statement tree
        ret
    |
    v
Emitter                     virtual registers, labels, declarations, PTX text
    |
    v
PTX on stdout
```

The compiler is a single walk. There is no intermediate IR and no C++ step.

## Entry point

| location | role |
|---|---|
| `compile.py:main` | Loads the IR, finds the kernel and meta, runs `Lowerer.compile`, prints PTX. |
| `compile.py:find_kernel` | Returns the single `KernelOp` in the graph; raises if there are none or more than one. |
| `compile.py:find_meta` | Reads `meta.json` beside the IR, selects the kernel whose `entry` matches, and raises when `meta.json` is absent or no entry matches. |
| `ptxc/lower.py:Lowerer.compile` | Assembles the module: header, shared declarations, signature, declarations, body. |
| `ptxc/lower.py:Lowerer.__init__` | Reads meta, scans shared memory and accumulators, selects caches, counts uses, records single-use products for `fma` fusion. |

## Modules

### `ptxc/lower.py`: driver and tile schedule

| function | responsibility |
|---|---|
| `_read_volume` | Static count of reads of a buffer, multiplied through loop extents. Used to decide whether a cache pays. |
| `_select_caches` | Chooses read-only f32 2D buffers. Whole tile if it fits, otherwise the block's row slice. Requires a 1D block, no existing shared memory, at most 32 KiB per buffer and 40 KiB total. Skips buffers already covered by register hoisting and candidates whose fill cost exceeds the saving. |
| `_load_motion` | Finds the loop axis and extent that move a buffer's load index. A coefficient of 1 selects a whole-tile cache; a row coefficient selects a slice cache. |
| `_hoist_volume` | Counts loads of a buffer that are invariant to an outer loop. At or below 64, the register hoist already handles the buffer; above that, the cache is considered. |
| `_layout_smem` | Assigns 16-byte-aligned offsets and computes `smem_total`. |
| `_emit_cache_fills` | Emits each cooperative fill, the transposed store when needed, the slice base, and one `bar.sync 0` after all fills. It runs before the `gid` guard so every thread reaches the barrier. |
| `_read_meta` | Stores shapes and dtypes, and whether the launch reserves dynamic shared memory. |
| `_scan_accumulators` | Maps each accumulator name to its first `Accum`, records explicit `Init` names, and collects top-level seeds. |
| `_signature` | Emits `.visible .entry`, the `.param` list in `arg_order`, and `.maxntid` from the launch block. |
| `_emit_params` | Loads scalar params and converts buffer params to global addresses with `cvta.to.global`. |
| `_seed_accumulator`, `_seed_value` | Writes the reduction identity with the correct dtype handling, including f16 and f64. |
| `_emit_thread_helpers` | Hoists `tid`, `lane` and `warp` when the IR uses them. |
| `_emit_tile` | Decodes axes, sets `warp_axes` and `full_tile`, emits cache fills, the guard, then the tile body. |
| `_gid` | Flattens the launch ids. A one-dimensional launch uses `ctaid.x * ntid.x + tid.x` directly. |

### `ptxc/stmts.py`: statement walk

| function | responsibility |
|---|---|
| `walk`, `stmt` | Iterate statements and dispatch to `stmt_<ClassName>`. Unknown statements raise `CompileError`. |
| `stmt_Load` | Cache redirection, load CSE, pair planning, deferred batching, vector loads. |
| `_load_key` | CSE key for a load: buffer, index expression text, width. |
| `_vector_load_regs` | Emits `ld.shared.vN` or `ld.global.vN`. |
| `_cached_view` | Rewrites a global load to a cache load, swapping coordinates for a transposed cache. |
| `_resolve_load_addr` | Chooses the address path, including the slice-cache offset subtraction. |
| `_emit_load_impl` | Emits scalar or vector loads and binds the IR names. |
| `_emit_cross_load`, `_plan_pairs`, `_adjacent_index`, `_cross_aligned` | Fuse same-copy and cross-copy adjacent loads into `v2` or `v4`. Non-affine indices stay scalar. |
| `stmt_Write` | Scalar, vector and atomic stores. |
| `stmt_Let`, `stmt_Init`, `stmt_Assign`, `stmt_Accum` | SSA definitions, literals, elementwise ops and folds. |
| `_try_fused_accumulate` | Turns `acc += a * b` with a single-use product into `fma.rn`. |
| `stmt_Cond`, `_emit_cond_inner` | `setp` plus a predicated branch, with an optional else body. |
| `stmt_Loop`, `_emit_loop_inner` | Full unroll for small shared loops, partial unroll for global loops, load batching, hoisting, seeds and warp-dot dispatch. |
| `stmt_StridedLoop`, `_emit_strided_inner` | Cooperating strided loops, chunk unrolling and the same batching rules. |
| `_invariant_inner_loads`, `_emit_hoisted_loads` | Load invariant values once per loop and rebind them per unrolled copy. |
| `_begin_pairs`, `_end_pairs`, `_batchable`, `_batched_unroll` | Issue a group's loads before its arithmetic. |
| `_chunk_address`, `_remember_chunk_address` | Remember the runtime base of a partial-unroll chunk for literal folding. |
| `_loop_partial`, `_loop_partial_inner` | Partial unroll of constant-trip global loops. |
| `_strided_plan`, `_strided_unrolled` | Chunked strided loop with constant offsets. |
| `_begin_hoist`, `_hoist_expr`, `_is_int_expr` | Selective hoisting of expensive index subtrees. |
| `_try_warp_dot` | Lane-strided dot plus butterfly shuffle. Requires a pure four-statement body, a group that divides the head extent, and `full_tile`. |
| `stmt_Sync`, `stmt_Smem` | Barriers and shared-memory declarations. |
| `stmt_WarpShuffle`, `_shuffle` | XOR butterfly over `length/2 ... 1`. |
| `stmt_TreeHalve`, `_tree_level`, `_tree_inner_offset` | Shared-memory halving tree over every coupled state. |
| `stmt_CpAsyncCopy`, `stmt_CpAsyncCommit`, `stmt_CpAsyncWait` | Async copies, commit groups and waits. |
| `stmt_ZeroPrologue` | Strided zeroing of an output before atomic reduction. |

### `ptxc/expr.py`: expressions and indices

| function | responsibility |
|---|---|
| `value` | Entry point for evaluating an IR expression into a register and kind. |
| `_builtin` | Builtin casts and math spellings. |
| `_int_var` | Integer variable lookup and materialization. |
| `fexpr`, `fexpr_operand` | f32 expression lowering and operand coercion. |
| `f64_operand_expr`, `f64_operand`, `_to_f64` | f64 expressions. |
| `_to_f32`, `_to_s64` | Coercions used by the operator tables. |
| `iexpr`, `iexpr_operand` | Integer expression lowering. |
| `idx32`, `_idx32_var`, `_idx32_binary` | 32-bit index arithmetic. |
| `_s32_bin`, `_div32`, `_rem32` | 32-bit binary, division and remainder helpers. |
| `_s64_bin`, `_div`, `_rem` | 64-bit fallbacks. |
| `_fold_int` | Compile-time evaluation of axis arithmetic, using the `Lowerer.axis_values` binding map. |
| `pexpr`, `_compare`, `_is_float`, `pred_operand` | Predicates and comparisons. |
| `_bool_assign`, `_assign_where` | Boolean assignment and `where` selection. |
| `_emit_float_op`, `_emit_f64_op` | Elementwise float operations, including accurate transcendentals. |
| `_infer_kind` | Result kind of an expression. |
| `_cast_reg` | Explicit store-time conversion between f16/f32/f64 and integer widths. |
| `_emit_erf_f64` | Double-precision erf polynomial, clamped past 4. |

### `ptxc/memory.py`: addresses and shared layout

| function | responsibility |
|---|---|
| `flat_index` | One index is already flat; two or more are row-major flattened against the shape. |
| `flat_axis_coefficient` | Affine coefficient of an axis in a flattened index. |
| `shape_of` | Buffer shape metadata. |
| `_scan_smem`, `_smem_base`, `_emit_smem_bases`, `_smem_declarations`, `_static_smem_line` | Shared-memory discovery, bases, declarations and the dynamic pool. |
| `_global_addr`, `_shared_addr` | Address materialization for global and shared memory. |
| `_scaled_offset`, `_scaled_offset_imm`, `_next_addr` | Scaled offsets, immediate forms and address chaining. |
| `_aligned` | Static alignment proof used to permit vector accesses. |
| `_swizzle` | Emmy's XOR swizzle for shared-memory copies. |
| `_load_consecutive`, `_store_values` | Grouped value access. |
| `_is_written` | Distinguishes read-only buffers from outputs. |
| `buffer_dtype` | Buffer element type from `meta.json` args or the `Smem` declaration. |

### `ptxc/common.py`: shared rules

| function | responsibility |
|---|---|
| `CompileError` | The only expected compiler failure. It must carry a useful message. |
| `stmt_name` | Uses the statement class name for dispatch and metadata scans. |
| `_iter_stmts` | Depth-first walk over `body` and `else_body`. Extend it if a new statement stores children elsewhere. |
| `_count_refs`, `_accums_for_loop` | Use counts and accumulator ownership. |
| `_should_unroll` | Full-unroll rule for constant-trip loops over shared memory. |
| `_partial_unroll_factor` | Partial-unroll factor for global loops. |
| `_expanded_weight`, `_expr_size`, `_is_expensive` | Body size and expression cost estimates. |
| `_loaded_buffers`, `_written_buffers`, `_loop_load_buffers` | Buffer classification for unrolling and cache decisions. |
| `_axes_defined_in` | Axis names rebound by loops inside a subtree. |
| `_defined_in` | SSA names assigned anywhere inside a loop subtree. |
| `_index_exprs`, `_expr_children` | Expression tree traversal. |
| `_check_dtype`, `_kind_of_dtype` | Loud dtype validation. |
| `_strides`, `_axis_coefficient`, `_affine` | Row-major strides and the affine analysis behind alignment and caches. |
| `_op_name`, `_ident`, `_const_int`, `_literal_is_float`, `_addr_add`, `_log2` | Small shared helpers. |

### `ptxc/ops.py` and `ptxc/emitter.py`

| location | responsibility |
|---|---|
| `ops.norm_dtype`, `ops.reg_class`, `ops.elem_bytes`, `ops.ptx_type`, `ops.f32_bits` | dtype normalization and PTX spellings. |
| `ops.IDENTITY`, `ops.INT_BINARY`, `common.FLOAT_KINDS` | Reduction identities, operator tables and float dtypes. |
| `Emitter.reg`, `Emitter.f`, `Emitter.r`, `Emitter.rd`, `Emitter.p`, `Emitter.d`, `Emitter.h` | Virtual registers by class. |
| `Emitter.emit`, `Emitter.place` | Body emission and label placement. |
| `Emitter.new_label` | Label allocation. |
| `Emitter.declarations` | Register declarations at the top of the function. |

## Where to change what

| behavior | first place to look |
|---|---|
| CLI, meta discovery, launch descriptor | `compile.py` |
| Parameter order, scalar vs buffer, entry signature, `.maxntid` | `lower._ordered_args`, `lower._emit_params`, `lower._signature` |
| Tile axis decode, guard, warp schedule flags | `lower._emit_tile`, `lower._gid` |
| Cache selection, budget, slice logic | `lower._select_caches`, `lower._layout_smem`, `lower._emit_cache_fills` |
| Cache reads and transposed coordinates | `stmts._cached_view`, `stmts._resolve_load_addr` |
| Load CSE, batching, vector fusion | `stmts._load_key`, `stmts._batched_unroll`, `stmts._plan_pairs` |
| Unroll decisions | `common._should_unroll`, `common._partial_unroll_factor`, `stmts._loop_partial`, `stmts._strided_plan` |
| Register hoisting | `stmts._invariant_inner_loads`, `stmts._emit_hoisted_loads` |
| Warp dot and its safety conditions | `stmts._try_warp_dot`, `lower._emit_tile` |
| Accumulator ownership and seeds | `lower._scan_accumulators`, `lower._emit_accumulator_seeds`, `stmts._emit_loop_seeds` |
| Index arithmetic and folding | `expr.idx32`, `expr._fold_int`, `Lowerer.axis_values`, `common._affine` |
| Vector alignment | `memory._aligned`, `stmts._cross_aligned` |
| Shared layout, swizzle, addresses | `memory` |
| Elementwise ops and dtype spellings | `ops` tables, `expr._emit_float_op`, `expr._emit_f64_op` |
| New statement | `stmts.stmt_<Name>`, `common._iter_stmts`, `common._expr_children` |
| Store-time dtype conversion | `expr._cast_reg`, `stmts.stmt_Write`, `stmts._tree_level` |
| Elementwise ops, erf, trig | `expr._emit_float_op`, `expr._emit_erf_f64`, `ops` tables |
| Exact multi-kernel IR selection | `tools/irfiles.resolve_ir` |
| Error message or unsupported construct | `common.CompileError`, `common._check_dtype` |

## Recipes

### Add an elementwise operation

1. Find the Emmy op name (`_op_name`) and add its PTX spelling and result rules in
   `ptxc/ops.py`.
2. Route the op in `ptxc/expr.py` (`_emit_float_op`, `_emit_f64_op` or the integer path).
3. Add a unit test for the spelling and a generated corpus case for the numeric result.
4. Run `tools/verify.sh --quick`, then the corpus on a local GPU. Check exactness rules in
   [DESIGN.md](../DESIGN.md): pointwise float arithmetic stays `.rn` unless the reference
   permits contraction.

### Add a statement

1. Implement `stmt_<ClassName>` in `ptxc/stmts.py`. The dispatcher uses the class name, so the
   method name is the contract.
2. If the statement has children or buffers, extend `common._iter_stmts`, `_expr_children`,
   `_loaded_buffers` or `_written_buffers` as needed.
3. If it affects accumulator ownership, extend `lower._scan_accumulators`.
4. Add a machinery test in `tests/test_machinery.py` that asserts the emitted PTX, and a
   variant if the construct can run end to end.
5. An unknown statement must still raise `CompileError`, never emit a guess.

### Change the cache heuristic

1. Read `_select_caches`, `_layout_smem` and `_emit_cache_fills` together. The budget, the
   slice base and the fill stride are one design.
2. Run the attention shapes in `tools/specs/attn_*.py`, not only the open set. Large tiles
   exercise the slice path; odd sequence lengths exercise the scalar fill.
3. Test an over-provisioned launch (`N < grid * block`) so the fill runs before the guard.
4. Check total shared memory: at most 32 KiB per buffer and 40 KiB total today, and static
   shared memory must stay legal when `smem_bytes=0`.
5. Measure warm and cold. A cache that improves warm but regresses cold on a small kernel is not a net improvement
   for the score.

### Change unrolling or folding

1. The rules live in `common._should_unroll` and `common._partial_unroll_factor`; the emitters
   are `stmts._loop_partial_inner`, `stmts._strided_unrolled` and `stmts._batched_unroll`.
2. A partial copy must bind `chunk_base + k`, not `k`. A nested loop that reuses an axis name
   hides the outer literal.
3. Test K=17, layer-norm (three coupled accumulators), odd row lengths and attention. The
   layer-norm corpus caught the last folding bug.
4. Measure instruction count and register count, not only time. A shorter kernel can schedule
   worse.

## Invariants and where they are enforced

| invariant | enforcement |
|---|---|
| Entry name and parameter order match the contract | `lower._signature`, `tests/check_open_set.py` |
| One `gid < N` guard covers over-provisioned launches | `lower._emit_tile` |
| Accumulator seeds belong to the loop instance, not the axis name | `lower._scan_accumulators`, `stmts._emit_loop_seeds` |
| Predicate-shared state is defined in the prologue | `lower._emit_thread_helpers`, `lower._emit_smem_bases` |
| Cache fills and barriers are unconditional | `lower._emit_cache_fills` before the guard |
| Vector accesses are statically aligned | `memory._aligned`, `stmts._cross_aligned` |
| Unknown dtype, op or statement fails loudly | `common._check_dtype`, `ops` tables, `stmts.stmt` |
| PTX is ASCII and deterministic | `compile.py` ASCII guard, `tests/check_open_set.py`, determinism tests |

## Tests by layer

| layer | command | needs |
|---|---|---|
| Dash and syntax | `tools/verify.sh --quick` | Python only |
| Unit rules | `tests/test_lowering.py` | Python, task repository for 2 cases; generated corpora for 3 |
| Machinery | `tests/test_machinery.py` | Python, task repository for most cases, optional ptxas |
| Contract over the open set | `tests/check_open_set.py --task-dir ../task` | task repository, optional ptxas |
| Variants | `tools/checkvariants.sh` | local CUDA GPU, ptxas |
| Generated corpora | `VARIANTS=../stress tools/checkvariants.sh` | local CUDA GPU, ptxas |
| Hosted runner | `TASK=../task tools/testall.sh` | network |
| Everything local | `tools/verify.sh --corpora` | local CUDA GPU, ptxas |


## Codegen additions

| file | symbol | responsibility |
|---|---|---|
| `ptxc/emitter.py` | `_WRITE_R32` | recognizes the destination of a 32-bit write |
| `ptxc/emitter.py` | `mov_s32`, `literal_block` | one literal register per value, emitted before the body |
| `ptxc/expr.py` | `_fold_s32_literals` | exact add/sub/mul/shift/bitwise folds with signed range checks |
| `ptxc/expr.py` | `_s32_bin` | identity and literal folding for index arithmetic |
| `ptxc/memory.py` | `_cse_global_addr` | affine global address cache, positive offsets and rebase |
| `ptxc/memory.py` | `_split_flat`, `_div_operand`, `_mod_operand` | symbolic key and row/column recombination |
| `ptxc/memory.py` | `flat_index`, `flat_axis_coefficient` | optional recombination for copy destinations |
| `ptxc/memory.py` | `_next_addr` | negative-offset address stepping |
| `ptxc/stmts.py` | `_single_trip`, `_guarded_body` | one-trip strided loop as a guarded body |
| `ptxc/stmts.py` | `_lane_fanout` | counts copies sharing one lane expression |
| `ptxc/stmts.py` | `_strided_bounded_plan`, `_strided_unrolled` | bounded unroll with tail guards and base reuse |
| `ptxc/stmts.py` | `_plan_pairs`, `_emit_load_impl` | up to four-wide shared load fusion with alignment |
| `ptxc/stmts.py` | `stmt_CpAsyncCopy` | copy destination base reuse on long runs |
| `ptxc/stmts.py` | `_emit_hoisted_loads` | clears the address caches per hoisted copy |
| `ptxc/stmts.py` | `_shuffle` | float shuffle without the b32 round trip |
| `ptxc/lower.py` | `_emit_tile` | tile decode threshold, exact-launch guard, tile-count gates |
| `ptxc/lower.py` | `_compile` | literal block placed ahead of the body |
| `ptxc/common.py` | `_partial_unroll_factor` | long all-shared loops unroll by 16, mixed loops stay rolled |
