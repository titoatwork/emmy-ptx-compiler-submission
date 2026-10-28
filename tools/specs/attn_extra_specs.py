import numpy as np

def _softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)

def _attn(q, kt, v, scale):
    return (_softmax((q.astype(np.float64) @ kt.astype(np.float64)) / scale) @ v.astype(np.float64)).astype(np.float32)

SPECS = [
 dict(id="g_attn_s128_h64", code="q = torch.randn(128, 64); kt = torch.randn(64, 128); v = torch.randn(128, 64); torch.softmax(q @ kt / 8.0, dim=-1) @ v",
      ref=lambda q, kt, v: [_attn(q, kt, v, 8.0)], rtol=1e-3, atol=1e-4),
 dict(id="g_attn_s256_h32", code="q = torch.randn(256, 32); kt = torch.randn(32, 256); v = torch.randn(256, 32); torch.softmax(q @ kt / 5.656854, dim=-1) @ v",
      ref=lambda q, kt, v: [_attn(q, kt, v, 5.656854)], rtol=1e-3, atol=1e-4),
 dict(id="g_attn_s128_h128", code="q = torch.randn(128, 128); kt = torch.randn(128, 128); v = torch.randn(128, 128); torch.softmax(q @ kt / 11.313708, dim=-1) @ v",
      ref=lambda q, kt, v: [_attn(q, kt, v, 11.313708)], rtol=1e-3, atol=1e-4),
 dict(id="g_attn_s64_h128", code="q = torch.randn(64, 128); kt = torch.randn(128, 64); v = torch.randn(64, 128); torch.softmax(q @ kt / 11.313708, dim=-1) @ v",
      ref=lambda q, kt, v: [_attn(q, kt, v, 11.313708)], rtol=1e-3, atol=1e-4),
 dict(id="g_attn_s256_h64", code="q = torch.randn(256, 64); kt = torch.randn(64, 256); v = torch.randn(256, 64); torch.softmax(q @ kt / 8.0, dim=-1) @ v",
      ref=lambda q, kt, v: [_attn(q, kt, v, 8.0)], rtol=1e-3, atol=1e-4),
]
OPEN = SPECS
