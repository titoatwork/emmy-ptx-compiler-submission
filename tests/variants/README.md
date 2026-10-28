# Variant regression (46 cases)

Each directory is a problem package: `meta.json`, one `.ir.json`, and `expected.npz`.
They were generated for this submission to cover shapes, block sizes, reduction axes, coupled state,
dtypes and attention head dimensions that the open set does not exercise.

Run them on a local CUDA GPU:

```bash
tools/checkvariants.sh
# or one case
tools/localrun.py tests/variants/v_a_128 ./compile.py
```

`localrun.py` compiles the PTX with the local `ptxas` for the detected device architecture
(`LOCAL_ARCH` overrides) and compares the kernel output against `expected.npz` with the
tolerances in `meta.json`. The open set is also checked on the hosted runner in
[docs/VERIFICATION.md](../../docs/VERIFICATION.md); these variants run on the local GPU.

These packages are derived from the Emmy Hackathon 2026 task materials and are
carved out of the repository's MIT license. See [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md).
