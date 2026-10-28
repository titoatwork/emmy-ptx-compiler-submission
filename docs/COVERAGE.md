# Coverage matrix

What the compiler supports, and the case that proves it. The counts are 15 open problems, 46
checked-in variants and 86 generated corpus IRs across 13 directories. "Generated" means the
case comes from the task generator and a spec under `tools/specs`; "variant" means a checked-in
package under `tests/variants`.

## Category 1: pointwise

| construct | evidence |
|---|---|
| 1D, 2D and 3D pointwise tiles | open set, `stress`, `fuzz` (`z_3d_point`), `blocks`, variants |
| unrolled bodies, vector stores | p00-p03, variants |
| constants as kernel args | p02, generated corpora |
| where, pow, relu | `fuzz`, variants, open p03 |
| NaN float `!=` and `==` | `adv` |
| abs | `ops` (softplus decomposition) |
| exp, log, sqrt, rsqrt, reciprocal | open set, `fuzz`, `ops`, `fixed` |
| tanh, sigmoid, silu | `fixed`, `fuzz`, `ops`, `dtypeedge` |
| erf, gelu, gelu_tanh, softplus | `ops` |
| sin, cos | `ops` |
| f16 pointwise | `f16`, `dtypeedge`, `ops` |
| f64 pointwise | `dtypeedge` f64 cases |
| odd lengths, grouped writes | open set, `fuzz`, `small` |
| tiny over-provisioned | `adv` |

`sign`, `tan`, `log2` and `log10` are implemented defensively. They are not emitted by the
current corpus.

## Category 2: reductions

| construct | evidence |
|---|---|
| 1D sum, mean | open set, `stress` |
| f32 row max | `fuzz` (`z_rowmax32`) |
| 2D row sums, middle axes | open set, `fuzz`, `adv` |
| long rows (4096) | `stress` |
| short rows (7, 32) | `small` |
| over-provisioned pointwise and tiny split reductions | p00, `adv` |
| split reductions with atomics | `stress` |
| 64/128/256/512-thread trees | `blocks`, `fuzz`, variant `w_r_t512` |
| f16 row sum, f16 row max | `dtypeedge` |
| f64 reduction | `dtypeedge` |
| `TreeHalve.inner` | unit test only |

## Category 3: matmul

| construct | evidence |
|---|---|
| square 512, 2048 | open m01, `stress` |
| rectangular, tall, wide | open m02, variants (`v_m_rect_small`, `v_m_tall`, `v_m_wide`) |
| batched | variant `w_m_batch` |
| matvec | `fuzz` (`z_mm_n1`) |
| matmul + gelu epilogue | `adv` |
| bias and relu epilogue, two kernels | `adv` (`adv_mm_bias_relu`) |
| BMM | `adv` (`adv_bmm`) |
| split-K, odd K, small K | `stress`, `fuzz` |
| cp.async staging, dynamic smem | m01, `stress` |

## Category 4: fused

| construct | evidence |
|---|---|
| rmsnorm | open f01 |
| layernorm, affine layernorm | `ln`, `adv` |
| 2D row softmax | open f02, `fuzz`, `small`, `blocks` |
| 4D softmax | `fuzz` (`z_sm_4d`) |
| l2norm | open f03 |
| f16 softmax, f16 layer norm | `dtypeedge` |
| f64 softmax | `adv` |
| three coupled accumulators | `ln` |

## Category 5: attention-like

| construct | evidence |
|---|---|
| heads 16, 32, 64, 96, 128, 256 | `attnedge`, `attnadv`, `attngen` |
| sequence 16 to 512, odd lengths 33, 100 | `attnedge`, `attnadv` |
| scaled and unscaled | open a01/a02, `adv` (`adv_attn_noscale`) |
| cache whole tile and slice | open a01/a02, large generated shapes |
| warp-cooperative dot | a01, a02, generated shapes |

## Infrastructure

| construct | evidence |
|---|---|
| static and dynamic shared memory | open matmul, `stress` |
| cache fill alignment (slice base) | code plus open attention slices |
| 2D cache refusal | unit test `test_cache_refused_for_2d_launch` |
| partial-block barrier guard | unit test `test_partial_block_barrier_fails_loudly` |
| partial-warp collective guard | unit tests for shuffle and warp sync |
| scalar parameters | unit test |
| warp sync, named barriers | unit test |
| 8-byte cp.async | unit test |
| shared write swizzle | unit test (store side only) |
| `ZeroPrologue` | unit test only |
| multi-kernel chain and exact IR selection | `adv` (`adv_mm_bias_relu`, `k_relu__partial` -> `k_relu`) and the resolver unit test |
| unknown dtype, op, statement | machinery tests, loud `CompileError` |

## Known gaps

- `Select`, `Pre`, `Carry`, `FuncCallExpr` and the tensor-core and TMA star constructs are not
  implemented. The task marks them absent from every open and hidden problem.
- Shared read swizzle is not implemented; stores and cp.async apply the swizzle.
- The f32 `max`/`min` `.NaN` semantics are not adopted. The reference uses `fmaxf`, and
  `np.allclose` rejects NaN expected values. Float compares use `setp.neu`.
- `max.NaN.f64` and `min.NaN.f64` are rejected by ptxas 12.6 and the runner's CUDA, so f64
  max/min/relu keep the non-NaN behavior.
- Count barriers in a partial block are conservatively rejected; no generated case exercises
  them.
