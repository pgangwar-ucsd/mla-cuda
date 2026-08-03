"""Per-stage profiler for the fully-fused MLA path.

Breaks the three lines of cuda_fully_fused_mla_attention into measurable pieces:

    W_uk, W_uv = split_wkv_b(...)                      # weight prep + .contiguous()
    q_absorbed, q_rope = prepare_q_mla_absorbed_cuda(...)   # fused kernel 2
    return mla_attention_official_cuda(...)                 # stage 3 (3a-3d)
"""

from __future__ import annotations

import torch

import mla_custom_cuda
from exhaustive_benchmark_suite import (
    DEEPSEEK_V3,
    PRODUCTION_SCENARIOS,
    make_mla_tensors,
    mla_attention_official,
    mla_attention_official_cuda,
    prepare_q_mla_absorbed,
    prepare_q_mla_absorbed_cuda,
    split_wkv_b,
)

# RTX A6000: 84 SM x 128 fp32 cores x 2 flop x 1.8 GHz boost, GDDR6 384-bit @ 16 Gbps
PEAK_FP32_TFLOPS = 38.7
PEAK_HBM_GBPS = 768.0


def time_ms(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def profile(scenario, cfg):
    torch.cuda.empty_cache()
    tensors = make_mla_tensors(scenario, cfg)
    x, c_kv, pe_cache, W_q, wkv_b, q_positions = tensors
    B, Sq, Sk, H = scenario.batch, scenario.sq, scenario.sk, cfg.n_heads
    R, D, V = cfg.kv_lora_rank, cfg.qk_rope_head_dim, cfg.v_head_dim
    nope = cfg.qk_nope_head_dim

    W_uk, W_uv = split_wkv_b(wkv_b, cfg)
    qa, qr = prepare_q_mla_absorbed_cuda(x, W_q, W_uk, q_positions, cfg)

    print(f"\n{'=' * 78}\n{scenario.name}  B={B} Sq={Sq} Sk={Sk}\n{'=' * 78}")

    # ---- weight prep: the .contiguous() copies hidden inside the two callees ----
    t_split = time_ms(lambda: split_wkv_b(wkv_b, cfg))
    t_contig = time_ms(lambda: (W_uk.contiguous(), W_uv.contiguous()))
    wt_mb = 2 * H * nope * R * 4 / 1e6
    print(f"  split_wkv_b (views only)        {t_split:9.4f} ms")
    print(f"  .contiguous() on W_uk + W_uv    {t_contig:9.4f} ms   "
          f"({wt_mb:.0f} MB copied per call)")

    # ---- stage 2: Q-prep ----
    t_q_cuda = time_ms(lambda: prepare_q_mla_absorbed_cuda(x, W_q, W_uk, q_positions, cfg))
    t_q_torch = time_ms(lambda: prepare_q_mla_absorbed(x, W_q, W_uk, q_positions, cfg))

    print(f"  prepare_q_cuda (fused)          {t_q_cuda:9.4f} ms")
    print(f"  prepare_q_torch (baseline)      {t_q_torch:9.4f} ms")

    # ---- stage 3: attention ----
    t_attn_cuda = time_ms(
        lambda: mla_attention_official_cuda(qa, qr, c_kv, pe_cache, W_uv, cfg),
        iters=scenario.num_iters,
    )
    t_attn_torch = time_ms(
        lambda: mla_attention_official(qa, qr, c_kv, pe_cache, W_uv, cfg),
        iters=scenario.num_iters,
    )
    print(f"  attention CUDA (fused)          {t_attn_cuda:9.4f} ms")
    print(f"  attention PyTorch (baseline)    {t_attn_torch:9.4f} ms")

    # ---- roofline for a fused stage-3 kernel ----
    flop_scores = 2 * B * Sq * H * Sk * (R + D)
    flop_ctx = 2 * B * Sq * H * Sk * R
    flop_out = 2 * B * Sq * H * R * V
    flop_attn = flop_scores + flop_ctx + flop_out
    flop_q = 2 * B * Sq * cfg.hidden_dim * H * cfg.qk_head_dim + 2 * B * Sq * H * nope * R

    # fused stage 3 only has to touch inputs + output once (scores never leave SMEM)
    bytes_fused = 4 * (B * Sq * H * (R + D) + B * Sk * (R + D) + B * Sq * H * V)
    bytes_scores_materialized = 4 * B * Sq * H * Sk * 4  # 3a write + 3b r/w + 3c read

    print(f"  --- roofline ---")
    print(f"  attention GFLOP                 {flop_attn / 1e9:9.1f}"
          f"   -> {flop_attn / 1e9 / PEAK_FP32_TFLOPS:7.3f} ms at fp32 peak")
    print(f"  Q-prep GFLOP                    {flop_q / 1e9:9.1f}"
          f"   -> {flop_q / 1e9 / PEAK_FP32_TFLOPS:7.3f} ms at fp32 peak")
    print(f"  fused stage-3 HBM (GB)          {bytes_fused / 1e9:9.3f}"
          f"   -> {bytes_fused / 1e9 / PEAK_HBM_GBPS * 1e3:7.3f} ms at peak BW")
    print(f"  scores round-trip HBM (GB)      {bytes_scores_materialized / 1e9:9.3f}"
          f"   -> {bytes_scores_materialized / 1e9 / PEAK_HBM_GBPS * 1e3:7.3f} ms  (fusion deletes this)")

    del tensors, x, c_kv, pe_cache, W_q, wkv_b, qa, qr
    torch.cuda.empty_cache()


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = DEEPSEEK_V3
    want = {"decode_production_avg_cache", "decode_single_64k_cache", "prefill_short_prompt"}
    for scenario in PRODUCTION_SCENARIOS:
        if scenario.name in want:
            profile(scenario, cfg)


if __name__ == "__main__":
    main()
