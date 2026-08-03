"""Correctness sweep over shapes that straddle the 3d bq/v tile boundaries."""

from __future__ import annotations

import torch

from exhaustive_benchmark_suite import (
    DEEPSEEK_V3,
    mla_attention_official,
    mla_attention_official_cuda,
)

torch.backends.cuda.matmul.allow_tf32 = False
cfg = DEEPSEEK_V3
H, R, D, V = cfg.n_heads, cfg.kv_lora_rank, cfg.qk_rope_head_dim, cfg.v_head_dim

# First group: bq_total = B*Sq around the 64-wide bq tile of 3d (under, exact, +1, over).
# Second group: low flat count + long Sk, which is what trips 3c into split-K, including
# Sk values that leave a ragged tail inside the last slice.
shapes = [
    (1, 1, 32), (3, 7, 40), (1, 64, 64), (1, 65, 48), (2, 33, 96), (5, 13, 71),
    (1, 1, 4096), (1, 1, 4999), (2, 3, 1000), (1, 1, 16384), (1, 2, 8191),
]

print(f"{'B':>3} {'Sq':>4} {'Sk':>4} {'bq':>5}  {'max abs':>10} {'max rel':>10}  ref-noise")
for B, Sq, Sk in shapes:
    g = torch.Generator(device="cuda").manual_seed(0)
    qa = torch.randn(B, Sq, H, R, device="cuda", generator=g)
    qr = torch.randn(B, Sq, H, D, device="cuda", generator=g)
    ckv = torch.randn(B, Sk, R, device="cuda", generator=g)
    pe = torch.randn(B, Sk, D, device="cuda", generator=g)
    wuv = torch.randn(H, V, R, device="cuda", generator=g)

    cuda_out = mla_attention_official_cuda(qa, qr, ckv, pe, wuv, cfg)
    ref32 = mla_attention_official(qa, qr, ckv, pe, wuv, cfg)
    ref64 = mla_attention_official(
        qa.double(), qr.double(), ckv.double(), pe.double(), wuv.double(), cfg
    )

    scale = ref64.abs().max()
    err = (cuda_out.double() - ref64).abs().max()
    noise = (ref32.double() - ref64).abs().max()  # fp32 cuBLAS vs fp64, same op
    print(f"{B:3d} {Sq:4d} {Sk:4d} {B * Sq:5d}  {err:10.3e} {err / scale:10.3e}  "
          f"{noise:.3e}  {'OK' if err < 5 * max(noise, 1e-6) else 'SUSPECT'}")
