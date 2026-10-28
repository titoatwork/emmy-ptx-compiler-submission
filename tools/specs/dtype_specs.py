import numpy as np


def _softmax(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def _layer(x):
    mean = np.mean(x.astype(np.float64), axis=-1, keepdims=True)
    var = np.var(x.astype(np.float64), axis=-1, keepdims=True)
    return (x.astype(np.float64) - mean) / np.sqrt(var + 1e-5)


OPEN = [
    dict(id="dt_rowsum16", code="torch.randn(64, 512, dtype=torch.float16).sum(-1)",
         ref=lambda x: [x.astype(np.float64).sum(-1).astype(np.float16)],
         rtol=3e-2, atol=3e-2),
    dict(id="dt_rowmax16", code="torch.randn(64, 512, dtype=torch.float16).max(-1).values",
         ref=lambda x: [x.astype(np.float64).max(-1).astype(np.float16)],
         rtol=2e-2, atol=2e-2),
    dict(id="dt_softmax16", code="torch.softmax(torch.randn(64, 512, dtype=torch.float16), dim=-1)",
         ref=lambda x: [_softmax(x.astype(np.float64)).astype(np.float16)],
         rtol=3e-2, atol=3e-3),
    dict(id="dt_layernorm16",
         code="torch.nn.functional.layer_norm(torch.randn(64, 512, dtype=torch.float16), (512,), eps=1e-5)",
         ref=lambda x: [_layer(x).astype(np.float16)], rtol=4e-2, atol=4e-2),
    dict(id="dt_sum64", code="torch.randn(4096, dtype=torch.float64).sum()",
         ref=lambda x: [x.astype(np.float64).sum().astype(np.float32)],
         rtol=1e-5, atol=1e-5),
    dict(id="dt_add64",
         code="torch.randn(2048, dtype=torch.float64) + torch.randn(2048, dtype=torch.float64)",
         ref=lambda a, b: [a + b], rtol=1e-10, atol=1e-12),
    dict(id="dt_exp64", code="torch.exp(torch.randn(2048, dtype=torch.float64) * 3)",
         ref=lambda x: [np.exp(3.0 * x)], rtol=1e-14, atol=1e-14),
    dict(id="dt_log64", code="torch.log(torch.abs(torch.randn(2048, dtype=torch.float64)) + 0.5)",
         ref=lambda x: [np.log(np.abs(x) + 0.5)], rtol=1e-12, atol=1e-13),
    dict(id="dt_pow64", code="torch.pow(torch.abs(torch.randn(1024, dtype=torch.float64)) + 0.5, 1.7)",
         ref=lambda x: [np.power(np.abs(x) + 0.5, 1.7)], rtol=1e-12, atol=1e-13),
    dict(id="dt_tanh64", code="torch.tanh(torch.randn(2048, dtype=torch.float64) * 2)",
         ref=lambda x: [np.tanh(2.0 * x)], rtol=1e-14, atol=1e-15),
    dict(id="dt_mul16_64",
         code="torch.randn(2048, dtype=torch.float16) * torch.randn(2048, dtype=torch.float64)",
         ref=lambda a, b: [(a.astype(np.float64) * b).astype(np.float16)],
         rtol=2e-2, atol=2e-2),
]
