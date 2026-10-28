# Feature tour

Every example below is self-contained: the IR, `meta.json` and `expected.npz` live in the
package, so the compile command works from a fresh clone with only `requirements.txt` installed.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Pointwise: division

```bash
python compile.py tests/variants/v_p_div/k_div_pointwise.ir.json > /tmp/p.ptx
head -8 /tmp/p.ptx
```

The entry name and the parameter order come from `meta.json`. The body uses vector stores where
the alignment proof allows them.

## Matmul: rectangular with a shared tile

```bash
python compile.py tests/variants/v_m_rect_small/k_matmul.ir.json > /tmp/m.ptx
```

This is a 64x32 output tile from a 64x128 by 128x32 pair. It exercises `cp.async`, a static
shared tile, the accumulator ownership rule and the exact-launch guard elision.

## Fused: rmsnorm

```bash
python compile.py tests/variants/v_f_rms_512/k_mean_50e2d6.ir.json > /tmp/f.ptx
```

One kernel, a reduction and an elementwise pass, 128 rows of 512.

## Attention: 32x256 with a cached key tile

```bash
python compile.py tests/variants/v_a_32x256/k_matmul_softmax_reduce_20729c.ir.json > /tmp/a.ptx
```

The key tile is cached in shared memory, the dot is warp cooperative, and the softmax is fused
into the same kernel.

## Run one example on a local GPU

```bash
tools/localrun.py tests/variants/v_m_rect_small ./compile.py
```

`localrun.py` compiles with the local ptxas, launches under the case's `meta.json` geometry and
compares against `expected.npz`. For a machine without a CUDA device, the compile step still
works and the PTX can be assembled with any compatible ptxas.

## More cases

`tests/variants` holds 46 packages covering odd shapes, block sizes 4 to 1024, f16 and f64,
reductions, matmuls and attention tiles. The generated corpora and the open set add split-K,
BMM, batched matmul, 4D softmax, two-kernel chains and attention heads 16 to 256. The coverage
matrix in [../docs/COVERAGE.md](../docs/COVERAGE.md) maps every construct to a case.
