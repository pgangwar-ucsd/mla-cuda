"""Exercise the CUDA attention in every supported dtype, on both 3c paths.

Sk=64 keeps split == 1 (ctx stays in the input dtype); Sk=16384 with one flat tile
forces split-K, where ctx is promoted to fp32 for the atomics.
"""

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

print(f"{'dtype':>8} {'Sk':>6} {'path':>9}  {'out dtype':>10} {'max rel':>10}")
# tolerances are per-dtype so fp64 cannot silently pass at fp32 accuracy
for dtype, tol in [(torch.float32, 1e-5), (torch.float64, 1e-13), (torch.float16, 3e-2)]:
    for Sk, path in [(64, "plain"), (16384, "split-K")]:
        B = Sq = 1
        g = torch.Generator(device="cuda").manual_seed(0)
        mk = lambda *s: torch.randn(*s, device="cuda", generator=g).to(dtype)
        qa, qr = mk(B, Sq, H, R), mk(B, Sq, H, D)
        ckv, pe = mk(B, Sk, R), mk(B, Sk, D)
        wuv = mk(H, V, R)

        out = mla_attention_official_cuda(qa, qr, ckv, pe, wuv, cfg)
        ref = mla_attention_official(
            qa.double(), qr.double(), ckv.double(), pe.double(), wuv.double(), cfg
        )
        rel = ((out.double() - ref).abs().max() / ref.abs().max()).item()
        ok = "OK" if out.dtype == dtype and rel < tol else "BAD"
        print(f"{str(dtype):>8} {Sk:6d} {path:>9}  {str(out.dtype):>10} {rel:10.2e}  {ok}")
