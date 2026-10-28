import numpy as np

def _softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)

def _attn(q, kt, v, scale):
    return (_softmax((q.astype(np.float64) @ kt.astype(np.float64)) / scale) @ v.astype(np.float64)).astype(np.float32)

SPECS = [
 dict(id="x_attn_s48_h96", code="q = torch.randn(48, 96); kt = torch.randn(96, 48); v = torch.randn(48, 96); torch.softmax(q @ kt / 9.797959, dim=-1) @ v",
      ref=lambda q, kt, v: [_attn(q, kt, v, 9.797959)], rtol=1e-3, atol=1e-4),
 dict(id="x_attn_s100_h32", code="q = torch.randn(100, 32); kt = torch.randn(32, 100); v = torch.randn(100, 32); torch.softmax(q @ kt / 5.656854, dim=-1) @ v",
      ref=lambda q, kt, v: [_attn(q, kt, v, 5.656854)], rtol=1e-3, atol=1e-4),
 dict(id="x_attn_s33_h64", code="q = torch.randn(33, 64); kt = torch.randn(64, 33); v = torch.randn(33, 64); torch.softmax(q @ kt / 8.0, dim=-1) @ v",
      ref=lambda q, kt, v: [_attn(q, kt, v, 8.0)], rtol=1e-3, atol=1e-4),
 dict(id="x_attn_s128_h96", code="q = torch.randn(128, 96); kt = torch.randn(96, 128); v = torch.randn(128, 96); torch.softmax(q @ kt / 9.797959, dim=-1) @ v",
      ref=lambda q, kt, v: [_attn(q, kt, v, 9.797959)], rtol=1e-3, atol=1e-4),
 dict(id="x_attn_s64_h48", code="q = torch.randn(64, 48); kt = torch.randn(48, 64); v = torch.randn(64, 48); torch.softmax(q @ kt / 6.928203, dim=-1) @ v",
      ref=lambda q, kt, v: [_attn(q, kt, v, 6.928203)], rtol=1e-3, atol=1e-4),
]
OPEN = SPECS
