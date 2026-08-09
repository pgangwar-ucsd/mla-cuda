"""Per-kernel breakdown of the CUDA attention path (3a/3b/3c/3d)."""

from __future__ import annotations

import torch
from torch.profiler import ProfilerActivity, profile

from exhaustive_benchmark_suite import (
    DEEPSEEK_V3,
    PRODUCTION_SCENARIOS,
    make_mla_tensors,
    mla_attention_official_cuda,
    prepare_q_mla_absorbed_cuda,
    split_wkv_b,
)


def run(scenario, cfg):
    tensors = make_mla_tensors(scenario, cfg)
    x, c_kv, pe_cache, W_q, wkv_b, q_positions = tensors
    W_uk, W_uv = split_wkv_b(wkv_b, cfg)
    W_uv = W_uv.contiguous()
    qa, qr = prepare_q_mla_absorbed_cuda(x, W_q, W_uk, q_positions, cfg)

    for _ in range(3):
        mla_attention_official_cuda(qa, qr, c_kv, pe_cache, W_uv, cfg)
    torch.cuda.synchronize()

    iters = 5
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            mla_attention_official_cuda(qa, qr, c_kv, pe_cache, W_uv, cfg)
        torch.cuda.synchronize()

    print(f"\n{'=' * 78}\n{scenario.name}  B={scenario.batch} Sq={scenario.sq} "
          f"Sk={scenario.sk}\n{'=' * 78}")
    rows = []
    for evt in prof.key_averages():
        if evt.self_cuda_time_total <= 0:
            continue
        rows.append((evt.self_cuda_time_total / 1e3 / iters, evt.key))
    rows.sort(reverse=True)
    total = sum(r[0] for r in rows)
    for ms, name in rows:
        print(f"  {ms:9.3f} ms  {100 * ms / total:5.1f}%  {name[:70]}")
    print(f"  {total:9.3f} ms  100.0%  TOTAL")

    del tensors, x, c_kv, pe_cache, W_q, wkv_b, qa, qr, W_uv
    torch.cuda.empty_cache()


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = DEEPSEEK_V3
    want = {"decode_serving_avg_ctx", "decode_single_user_long_ctx", "prefill_chat_batch"}
    for scenario in PRODUCTION_SCENARIOS:
        if scenario.name in want:
            run(scenario, cfg)


if __name__ == "__main__":
    main()
