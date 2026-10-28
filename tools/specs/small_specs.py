import numpy as np
def _softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True); e = np.exp(x - m); return e / e.sum(axis=axis, keepdims=True)
OPEN = [
 dict(id="n_row32_sum", code="torch.randn(64, 32).sum(-1)",
      ref=lambda x: [x.astype(np.float64).sum(-1).astype(np.float32)], rtol=1e-4, atol=1e-4, knobs="WORK=t128,REDUCE=coop"),
 dict(id="n_row32_mean", code="torch.randn(64, 32).mean(-1)",
      ref=lambda x: [x.astype(np.float64).mean(-1).astype(np.float32)], rtol=1e-4, atol=1e-4, knobs="WORK=t128,REDUCE=coop"),
 dict(id="n_softmax32", code="torch.softmax(torch.randn(64, 32), dim=-1)",
      ref=lambda x: [_softmax(x.astype(np.float64)).astype(np.float32)], rtol=1e-3, atol=1e-5),
 dict(id="n_row7", code="torch.randn(33, 7).sum(-1)",
      ref=lambda x: [x.astype(np.float64).sum(-1).astype(np.float32)], rtol=1e-4, atol=1e-4, knobs="WORK=t128,REDUCE=coop"),
]
