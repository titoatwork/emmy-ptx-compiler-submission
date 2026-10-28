"""Performance holdout specs.

Generate outside the repository with the task generator:

    python $TASK/tools/generate.py --emmy tools/emmy-torch --out ../holdout \
        tools/specs/holdout.py

The generated packages are not committed. The spec is, so the holdout can be rebuilt.
"""
import numpy as np

def _softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)

OPEN = [
 dict(id="h_mm_96x80x48", code="torch.randn(96, 80) @ torch.randn(80, 48)",
      ref=lambda a, b: [(a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)],
      rtol=1e-3, atol=1e-2),
 dict(id="h_mm_1024", code="torch.randn(1024, 1024) @ torch.randn(1024, 1024)",
      ref=lambda a, b: [(a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)],
      rtol=1e-3, atol=1e-2),
 dict(id="h_mm_64", code="torch.randn(64, 64) @ torch.randn(64, 64)",
      ref=lambda a, b: [(a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)],
      rtol=1e-3, atol=1e-3),
 dict(id="h_mm_128", code="torch.randn(128, 128) @ torch.randn(128, 128)",
      ref=lambda a, b: [(a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)],
      rtol=1e-3, atol=1e-3),
 dict(id="h_bmm", code="torch.bmm(torch.randn(4, 32, 64), torch.randn(4, 64, 32))",
      ref=lambda a, b: [np.einsum("bij,bjk->bik", a.astype(np.float64), b.astype(np.float64)).astype(np.float32)],
      rtol=1e-3, atol=1e-2),
 dict(id="h_mm_odd", code="torch.randn(97, 83) @ torch.randn(83, 59)",
      ref=lambda a, b: [(a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)],
      rtol=1e-3, atol=1e-2),
 dict(id="h_attn_48x16",
      code="q = torch.randn(48, 16); kt = torch.randn(16, 48); v = torch.randn(48, 16); torch.softmax(q @ kt / 4.0, dim=-1) @ v",
      ref=lambda q, kt, v: [(_softmax((q.astype(np.float64) @ kt.astype(np.float64)) / 4.0) @ v.astype(np.float64)).astype(np.float32)],
      rtol=1e-3, atol=1e-4),
 dict(id="h_attn_96x32",
      code="q = torch.randn(96, 32); kt = torch.randn(32, 96); v = torch.randn(96, 32); torch.softmax(q @ kt / 5.656854, dim=-1) @ v",
      ref=lambda q, kt, v: [(_softmax((q.astype(np.float64) @ kt.astype(np.float64)) / 5.656854) @ v.astype(np.float64)).astype(np.float32)],
      rtol=1e-3, atol=1e-4),
]
