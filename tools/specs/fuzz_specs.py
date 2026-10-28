import numpy as np
def _softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True); e = np.exp(x - m); return e / e.sum(axis=axis, keepdims=True)
OPEN = [
 # block-size extremes
 dict(id="z_t32_sum", code="torch.randn(4096).sum()",
      ref=lambda x: [x.astype(np.float64).sum().astype(np.float32)], rtol=1e-3, atol=1e-3, knobs="WORK=t32,REDUCE=coop"),
 dict(id="z_t1024_sum", code="torch.randn(262144).sum()",
      ref=lambda x: [x.astype(np.float64).sum().astype(np.float32)], rtol=1e-3, atol=1e-2, knobs="WORK=t1024,REDUCE=coop"),
 # unusual pointwise sizes and shapes
 dict(id="z_prime", code="torch.randn(1009) * 2.5 + 0.5",
      ref=lambda x: [(x * np.float32(2.5) + np.float32(0.5))], rtol=1e-6, atol=1e-6),
 dict(id="z_3d_point", code="torch.randn(4, 8, 32) * 3.0",
      ref=lambda x: [(x * np.float32(3.0))], rtol=0, atol=0),
 dict(id="z_big_point", code="torch.randn(262144) + 1.0",
      ref=lambda x: [(x + np.float32(1.0))], rtol=0, atol=0),
 # reductions over odd dimensions
 dict(id="z_red_mid3", code="torch.randn(8, 30, 64).sum(1)",
      ref=lambda x: [x.astype(np.float64).sum(1).astype(np.float32)], rtol=1e-4, atol=1e-4, knobs="WORK=t128,REDUCE=coop"),
 dict(id="z_red_row1", code="torch.randn(256, 1).sum(-1)",
      ref=lambda x: [x.astype(np.float64).sum(-1).astype(np.float32)], rtol=1e-4, atol=1e-4, knobs="WORK=t128,REDUCE=coop"),
 dict(id="z_mean_col", code="torch.randn(32, 128).mean(0)",
      ref=lambda x: [x.astype(np.float64).mean(0).astype(np.float32)], rtol=1e-4, atol=1e-4, knobs="WORK=t128,REDUCE=coop"),
 # matmul edge K values
 dict(id="z_mm_k8", code="torch.randn(64, 8) @ torch.randn(8, 64)",
      ref=lambda a, b: [(a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)], rtol=1e-3, atol=1e-3),
 dict(id="z_mm_k17", code="torch.randn(64, 17) @ torch.randn(17, 96)",
      ref=lambda a, b: [(a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)], rtol=1e-3, atol=1e-3),
 dict(id="z_mm_n1", code="torch.randn(128, 64) @ torch.randn(64, 1)",
      ref=lambda a, b: [(a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)], rtol=1e-3, atol=1e-3),
 # softmax/norm edge shapes
 dict(id="z_sm_row1", code="torch.softmax(torch.randn(64, 1), dim=-1)",
      ref=lambda x: [_softmax(x.astype(np.float64)).astype(np.float32)], rtol=1e-3, atol=1e-6),
 dict(id="z_sm_row128", code="torch.softmax(torch.randn(128, 128), dim=-1)",
      ref=lambda x: [_softmax(x.astype(np.float64)).astype(np.float32)], rtol=1e-3, atol=1e-5),
 dict(id="z_rms_col", code="x = torch.randn(32, 512); w = torch.randn(512); x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-5) * w",
      ref=lambda x, w: [(x / np.sqrt((x.astype(np.float64) ** 2).mean(-1, keepdims=True) + 1e-5) * w).astype(np.float32)],
      rtol=1e-3, atol=1e-4),
 # attention head dim 16, seq 64
 dict(id="z_attn_h16", code="q = torch.randn(64, 16); kt = torch.randn(16, 64); v = torch.randn(64, 16); torch.softmax(q @ kt / 4.0, dim=-1) @ v",
      ref=lambda q, kt, v: [(_softmax((q.astype(np.float64) @ kt.astype(np.float64)) / 4.0) @ v.astype(np.float64)).astype(np.float32)],
      rtol=1e-3, atol=1e-4),
    dict(id="z_sm_4d", code="torch.softmax(torch.randn(2, 3, 16, 256), dim=-1)",
         ref=lambda x: [_softmax(x.astype(np.float64)).astype(np.float32)], rtol=1e-4, atol=1e-5),
    dict(id="z_rowmax32", code="torch.randn(64, 512).max(-1).values",
         ref=lambda x: [x.max(-1)], rtol=1e-6, atol=1e-6),
]
