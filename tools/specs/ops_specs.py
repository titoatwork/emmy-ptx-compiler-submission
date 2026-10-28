import numpy as np


def _erf(x):
    return np.vectorize(__import__("math").erf)(x.astype(np.float64))


def _gelu(x):
    x = x.astype(np.float64)
    return 0.5 * x * (1.0 + _erf(x / np.sqrt(2.0)))


def _gelu_tanh(x):
    x = x.astype(np.float64)
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def _softplus(x):
    return np.log1p(np.exp(x.astype(np.float64)))


OPEN = [
    dict(id="op_gelu", code="torch.nn.functional.gelu(torch.randn(4096) * 3)",
         ref=lambda x: [_gelu(3.0 * x)], rtol=1e-5, atol=1e-6),
    dict(id="op_gelu_tanh", code="torch.nn.functional.gelu(torch.randn(4096) * 3, approximate='tanh')",
         ref=lambda x: [_gelu_tanh(3.0 * x)], rtol=1e-4, atol=1e-5),
    dict(id="op_gelu16", code="torch.nn.functional.gelu(torch.randn(4096, dtype=torch.float16) * 3)",
         ref=lambda x: [_gelu(3.0 * x.astype(np.float64)).astype(np.float16)],
         rtol=2e-2, atol=2e-2),
    dict(id="op_softplus", code="torch.nn.functional.softplus(torch.randn(4096) * 3)",
         ref=lambda x: [_softplus(3.0 * x)], rtol=1e-4, atol=1e-5),
    dict(id="op_cos_sin", code="torch.cos(torch.randn(2048) * 4) + torch.sin(torch.randn(2048) * 4)",
         ref=lambda a, b: [np.cos(4.0 * a.astype(np.float64)) + np.sin(4.0 * b.astype(np.float64))],
         rtol=2e-5, atol=2e-6),
    dict(id="op_erf64", code="torch.erf(torch.randn(2048, dtype=torch.float64) * 2)",
         ref=lambda x: [np.vectorize(__import__("math").erf)(2.0 * x)], rtol=1e-6, atol=2e-8),
]
