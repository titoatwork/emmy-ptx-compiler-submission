import numpy as np


def _softmax(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


OPEN = [
    dict(id="adv_layernorm_affine",
         code="x = torch.randn(64, 512); w = torch.randn(512); b = torch.randn(512); torch.nn.functional.layer_norm(x, (512,), w, b, 1e-5)",
         ref=lambda x, w, b: [((x.astype(np.float64) - x.astype(np.float64).mean(-1, keepdims=True)) / np.sqrt(x.astype(np.float64).var(-1, keepdims=True) + 1e-5) * w.astype(np.float64) + b.astype(np.float64))],
         rtol=1e-3, atol=1e-4),
    dict(id="adv_mm_gelu",
         code="torch.nn.functional.gelu(torch.randn(64, 64) @ torch.randn(64, 64))",
         ref=lambda a, b: [0.5 * (a.astype(np.float64) @ b.astype(np.float64)) * (1.0 + np.vectorize(__import__('math').erf)((a.astype(np.float64) @ b.astype(np.float64)) / np.sqrt(2.0)))],
         rtol=1e-4, atol=1e-4),
    dict(id="adv_tiny_add", code="torch.randn(3) + torch.randn(3)",
         ref=lambda a, b: [a + b], rtol=0, atol=0),
    dict(id="adv_tiny_sum", code="torch.randn(4).sum()",
         ref=lambda x: [x.astype(np.float64).sum().astype(np.float32)], rtol=1e-6, atol=1e-6),
    dict(id="adv_red3d_mid", code="torch.randn(4, 128, 64).sum(1)",
         ref=lambda x: [x.astype(np.float64).sum(1).astype(np.float32)], rtol=1e-3, atol=1e-4),
    dict(id="adv_where_pow", code="x = torch.randn(4096); torch.where(x > 0, x.pow(2), -x)",
         ref=lambda x: [np.where(x > 0, x.astype(np.float64) ** 2, -x.astype(np.float64)).astype(np.float32)],
         rtol=1e-5, atol=1e-6),
    dict(id="adv_softmax64", code="torch.softmax(torch.randn(64, 256, dtype=torch.float64), dim=-1)",
         ref=lambda x: [_softmax(x.astype(np.float64))], rtol=1e-6, atol=1e-7),
    dict(id="adv_attn_noscale",
         code="q = torch.randn(32, 64); kt = torch.randn(64, 32); v = torch.randn(32, 64); torch.softmax(q @ kt, dim=-1) @ v",
         ref=lambda q, kt, v: [(_softmax(q.astype(np.float64) @ kt.astype(np.float64)) @ v.astype(np.float64))],
         rtol=1e-3, atol=1e-4),
    dict(id="adv_nan_ne",
         code="x = torch.randn(1024); torch.where(x != float('nan'), 1.0, 2.0)",
         ref=lambda x: [np.where(x != np.float32('nan'), np.float32(1.0), np.float32(2.0))],
         rtol=0, atol=0),
    dict(id="adv_nan_eq",
         code="x = torch.randn(1024); torch.where(x == float('nan'), 1.0, 2.0)",
         ref=lambda x: [np.where(x == np.float32('nan'), np.float32(1.0), np.float32(2.0))],
         rtol=0, atol=0),
    dict(id="adv_mm_bias_relu",
         code="torch.relu(torch.randn(64, 64) @ torch.randn(64, 64) + torch.randn(64))",
         ref=lambda a, b, c: [np.maximum(a.astype(np.float64) @ b.astype(np.float64)
                                         + c.astype(np.float64), 0).astype(np.float32)],
         rtol=1e-4, atol=1e-4),
    dict(id="adv_bmm",
         code="torch.bmm(torch.randn(4, 8, 16), torch.randn(4, 16, 8))",
         ref=lambda a, b: [np.matmul(a.astype(np.float64), b.astype(np.float64)).astype(np.float32)],
         rtol=1e-4, atol=1e-4),
]
