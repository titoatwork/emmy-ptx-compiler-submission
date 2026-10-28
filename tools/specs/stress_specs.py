import numpy as np
def _softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True); e = np.exp(x - m); return e / e.sum(axis=axis, keepdims=True)
OPEN = [
 dict(id="s_sum_1m", code="torch.randn(1000000).sum()",
      ref=lambda x: [x.astype(np.float64).sum().astype(np.float32)], rtol=1e-3, atol=1e-2),
 dict(id="s_sumdim0", code="torch.randn(128, 512).sum(0)",
      ref=lambda x: [x.astype(np.float64).sum(0).astype(np.float32)], rtol=1e-3, atol=1e-3),
 dict(id="s_mean_rows", code="torch.randn(512, 4096).mean(-1)",
      ref=lambda x: [x.astype(np.float64).mean(-1).astype(np.float32)], rtol=1e-3, atol=1e-3),
 dict(id="s_mm_2048", code="torch.randn(2048, 2048) @ torch.randn(2048, 2048)",
      ref=lambda a, b: [(a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)], rtol=1e-3, atol=1e-2),
 dict(id="s_mm_splitk", code="torch.randn(512, 4096) @ torch.randn(4096, 512)",
      ref=lambda a, b: [(a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)], rtol=1e-3, atol=1e-2),
 dict(id="s_softmax_big", code="torch.softmax(torch.randn(64, 4096), dim=-1)",
      ref=lambda x: [_softmax(x.astype(np.float64)).astype(np.float32)], rtol=1e-3, atol=1e-5),
 dict(id="s_attn_big", code="q = torch.randn(256, 128); kt = torch.randn(128, 256); v = torch.randn(256, 128); torch.softmax(q @ kt / 11.313708, dim=-1) @ v",
      ref=lambda q, kt, v: [(_softmax((q.astype(np.float64) @ kt.astype(np.float64)) / 11.313708) @ v.astype(np.float64)).astype(np.float32)], rtol=1e-3, atol=1e-4),
]
