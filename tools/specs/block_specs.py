import numpy as np
def _softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True); e = np.exp(x - m); return e / e.sum(axis=axis, keepdims=True)
OPEN = [
 dict(id="b_sum_t64", code="torch.randn(32768).sum()",
      ref=lambda x: [x.astype(np.float64).sum().astype(np.float32)], rtol=1e-3, atol=1e-3, knobs="WORK=t64,REDUCE=coop"),
 dict(id="b_sum_t256", code="torch.randn(65536).mean()",
      ref=lambda x: [x.astype(np.float64).mean().astype(np.float32)], rtol=1e-4, atol=1e-4, knobs="WORK=t256,REDUCE=coop"),
 dict(id="b_softmax_1024_t256", code="torch.softmax(torch.randn(64, 1024), dim=-1)",
      ref=lambda x: [_softmax(x.astype(np.float64)).astype(np.float32)], rtol=1e-3, atol=1e-5),
 dict(id="b_rms_t256", code="x = torch.randn(64, 2048); w = torch.randn(2048); "
                           "x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-5) * w",
      ref=lambda x, w: [(x / np.sqrt((x.astype(np.float64) ** 2).mean(-1, keepdims=True) + 1e-5) * w).astype(np.float32)],
      rtol=1e-3, atol=1e-4),
]
