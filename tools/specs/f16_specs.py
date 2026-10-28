import numpy as np
OPEN = [
 dict(id="h16_add4k", code="x = torch.randn(4096, dtype=torch.float16); y = torch.randn(4096, dtype=torch.float16); x + y",
      ref=lambda a, b: [(a + b)], rtol=1e-2, atol=1e-2),
 dict(id="h16_exp", code="torch.exp(torch.randn(2048, dtype=torch.float16))",
      ref=lambda x: [np.exp(x.astype(np.float32)).astype(np.float16)], rtol=1e-2, atol=1e-2),
]
