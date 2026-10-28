import numpy as np
OPEN = [
 dict(id="x_p_log", code="x = torch.randn(512); torch.log(x * x + 10.0)",
      ref=lambda x: [np.log(x * x + 10.0).astype(np.float32)], rtol=1e-6, atol=1e-6),
 dict(id="x_p_recip", code="x = torch.randn(1024); 1.0 / (x * x + 5.0)",
      ref=lambda x: [(1.0 / (x * x + 5.0)).astype(np.float32)], rtol=1e-6, atol=1e-6),
 dict(id="x_p_rsqrt", code="x = torch.randn(1024); torch.rsqrt(x * x + 1.0)",
      ref=lambda x: [(1.0 / np.sqrt(x * x + 1.0)).astype(np.float32)], rtol=1e-5, atol=1e-6),
 dict(id="x_p_sigmoid", code="torch.sigmoid(torch.randn(1024))",
      ref=lambda x: [(1.0 / (1.0 + np.exp(-x))).astype(np.float32)], rtol=1e-6, atol=1e-6),
 dict(id="x_p_silu", code="x = torch.randn(1024); x * torch.sigmoid(x)",
      ref=lambda x: [(x * (1.0 / (1.0 + np.exp(-x)))).astype(np.float32)], rtol=1e-6, atol=1e-6),
]
