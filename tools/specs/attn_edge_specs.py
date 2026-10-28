import numpy as np

def _softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)

def _attn(q, kt, v, scale):
    return (_softmax((q.astype(np.float64) @ kt.astype(np.float64)) / scale) @ v.astype(np.float64)).astype(np.float32)

SPECS = [
 dict(id="e_attn_s32_h32", code="q = torch.randn(32, 32); kt = torch.randn(32, 32); v = torch.randn(32, 32); torch.softmax(q @ kt / 5.656854, dim=-1) @ v",
      ref=lambda q, kt, v: [_attn(q, kt, v, 5.656854)], rtol=1e-3, atol=1e-4),
 dict(id="e_attn_s64_h256", code="q = torch.randn(64, 256); kt = torch.randn(256, 64); v = torch.randn(64, 256); torch.softmax(q @ kt / 16.0, dim=-1) @ v",
      ref=lambda q, kt, v: [_attn(q, kt, v, 16.0)], rtol=1e-3, atol=1e-4),
 dict(id="e_attn_s512_h16", code="q = torch.randn(512, 16); kt = torch.randn(16, 512); v = torch.randn(512, 16); torch.softmax(q @ kt / 4.0, dim=-1) @ v",
      ref=lambda q, kt, v: [_attn(q, kt, v, 4.0)], rtol=1e-3, atol=1e-4),
 dict(id="e_attn_s16_h64", code="q = torch.randn(16, 64); kt = torch.randn(64, 16); v = torch.randn(16, 64); torch.softmax(q @ kt / 8.0, dim=-1) @ v",
      ref=lambda q, kt, v: [_attn(q, kt, v, 8.0)], rtol=1e-3, atol=1e-4),
 dict(id="e_attn_s256_h16", code="q = torch.randn(256, 16); kt = torch.randn(16, 256); v = torch.randn(256, 16); torch.softmax(q @ kt / 4.0, dim=-1) @ v",
      ref=lambda q, kt, v: [_attn(q, kt, v, 4.0)], rtol=1e-3, atol=1e-4),
]
OPEN = SPECS
