import numpy as np
def _ln(x):
    m = x.mean(-1, keepdims=True)
    d = x - m
    v = (d * d).mean(-1, keepdims=True)
    return (d / np.sqrt(v + 1e-5)).astype(np.float32)
OPEN = [
 dict(id="ln_512", code="x = torch.randn(64, 512); (x - x.mean(-1, keepdim=True)) / torch.sqrt(((x - x.mean(-1, keepdim=True)) * (x - x.mean(-1, keepdim=True))).mean(-1, keepdim=True) + 1e-5)",
      ref=lambda x: [_ln(x.astype(np.float64))], rtol=1e-3, atol=1e-4),
 dict(id="ln_1024", code="x = torch.randn(32, 1024); (x - x.mean(-1, keepdim=True)) / torch.sqrt(((x - x.mean(-1, keepdim=True)) * (x - x.mean(-1, keepdim=True))).mean(-1, keepdim=True) + 1e-5)",
      ref=lambda x: [_ln(x.astype(np.float64))], rtol=1e-3, atol=1e-4),
]
