"""
DeepSeek V3 MLA end-to-end attention benchmark suite.

Dimension ratios from the DeepSeek-V3 technical report (arXiv:2412.19437).
Layout follows the official absorbed inference path (DeepSeek-V3 inference/model.py):
  kv_cache c_kv [B,Sk,512], pe_cache [B,Sk,64], wkv_b [H,nope+v,512].

Scenario shapes are drawn from public production / kernel benchmarks:
  - DeepSeek inference overview (avg KV length ~4,989 tokens)
  - NVIDIA DGX Cloud benchmarking (ISL=8150, decode batch 128–256)
  - FlashMLA / ThunderMLA decode sweeps (batch × cache × query tokens)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import Callable, List, NamedTuple, Optional

import torch
import mla_custom_cuda


# ---------------------------------------------------------------------------
# DeepSeek V3 model constants (official ratios)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeepSeekV3Config:
    """Official MLA hyper-parameters from DeepSeek-V3 (671B)."""

    hidden_dim: int = 7168
    n_heads: int = 128
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def q_absorbed_dim(self) -> int:
        """Q after absorption: kv_lora_rank + qk_rope_head_dim = 576."""
        return self.kv_lora_rank + self.qk_rope_head_dim


DEEPSEEK_V3 = DeepSeekV3Config()


# ---------------------------------------------------------------------------
# Benchmark scenarios (production-relevant shapes)
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkScenario:
    name: str
    batch: int
    sq: int
    sk: int
    phase: str
    source: str
    description: str
    num_iters: int = 30
    tags: List[str] = field(default_factory=list)


# fmt: off
PRODUCTION_SCENARIOS: List[BenchmarkScenario] = [
    # --- quick / CI ---
    BenchmarkScenario(
        name="smoke",
        batch=2, sq=128, sk=128,
        phase="mixed",
        source="local CI",
        description="Fast sanity check with full V3 ratios",
        num_iters=20,
        tags=["quick"],
    ),

    # --- decode (memory-bandwidth bound; sq=1 or small sq) ---
    BenchmarkScenario(
        name="decode_single_64k_cache",
        batch=1, sq=1, sk=65536,
        phase="decode",
        source="ThunderMLA / FlashMLA (B=1, Seq=64k, Q=1)",
        description="Single stream, very long KV cache — classic FlashMLA decode point",
        num_iters=10,
        tags=["decode", "flashmla"],
    ),
    BenchmarkScenario(
        name="decode_production_avg_cache",
        batch=128, sq=1, sk=4989,
        phase="decode",
        source="DeepSeek inference overview (avg KV length/output token)",
        description="High-concurrency decode at DeepSeek production average cache depth",
        num_iters=15,
        tags=["decode", "production"],
    ),
    BenchmarkScenario(
        name="decode_nvidia_h100_throughput",
        batch=128, sq=1, sk=8192,
        phase="decode",
        source="NVIDIA DGXC DeepSeek-R1 benchmark (decode batch=128, long context)",
        description="H100-class max-throughput decode configuration",
        num_iters=15,
        tags=["decode", "nvidia"],
    ),
    BenchmarkScenario(
        name="decode_nvidia_b200_throughput",
        batch=256, sq=1, sk=2048,
        phase="decode",
        source="NVIDIA DGXC DeepSeek-R1 benchmark (decode batch=256)",
        description="Blackwell-class high-batch decode",
        num_iters=15,
        tags=["decode", "nvidia"],
    ),
    BenchmarkScenario(
        name="decode_mtp_2tok",
        batch=64, sq=2, sk=512,
        phase="decode",
        source="ThunderMLA (B=64, Seq=512, Q=2)",
        description="Multi-token / speculative decode with moderate cache",
        num_iters=20,
        tags=["decode", "mtp"],
    ),
    BenchmarkScenario(
        name="decode_mtp_4tok",
        batch=64, sq=4, sk=768,
        phase="decode",
        source="ThunderMLA (B=64, random Seq 256–1024, Q=4)",
        description="Multi-query decode; sk set to mid-range of random-seq sweep",
        num_iters=20,
        tags=["decode", "mtp"],
    ),
    BenchmarkScenario(
        name="decode_high_concurrency",
        batch=132, sq=4, sk=4096,
        phase="decode",
        source="ThunderMLA (B=132, Seq=4k, Q=4)",
        description="Large batch with 4k cache and 4 query tokens per sequence",
        num_iters=10,
        tags=["decode", "flashmla"],
    ),

    # --- prefill (compute bound; sq >> 1, typically sq == sk for new prompt) ---
    BenchmarkScenario(
        name="prefill_short_prompt",
        batch=8, sq=512, sk=512,
        phase="prefill",
        source="Typical chat / RAG prompt length",
        description="Small batched prefill for interactive workloads",
        num_iters=20,
        tags=["prefill"],
    ),
    BenchmarkScenario(
        name="prefill_single_2k",
        batch=1, sq=2048, sk=2048,
        phase="prefill",
        source="NVIDIA DGXC (H100 max_num_tokens=2048 chunked prefill chunk)",
        description="Single-sequence chunked-prefill step on H100",
        num_iters=10,
        tags=["prefill", "nvidia"],
    ),
    BenchmarkScenario(
        name="prefill_single_4k",
        batch=1, sq=4096, sk=4096,
        phase="prefill",
        source="NVIDIA DGXC (GB200/B200 max_num_tokens=4096)",
        description="Single-sequence chunked-prefill step on GB200/B200",
        num_iters=10,
        tags=["prefill", "nvidia"],
    ),
    BenchmarkScenario(
        name="prefill_nvidia_reference",
        batch=1, sq=8192, sk=8192,
        phase="prefill",
        source="NVIDIA DGXC (ISL=8150 reference, requires chunked prefill at scale)",
        description="Full reference input length; may OOM on smaller GPUs",
        num_iters=5,
        tags=["prefill", "nvidia", "large"],
    ),
    BenchmarkScenario(
        name="prefill_batched_rag",
        batch=16, sq=2048, sk=2048,
        phase="prefill",
        source="Batched document ingestion / RAG",
        description="Moderate batch prefill for offline indexing pipelines",
        num_iters=10,
        tags=["prefill"],
    ),
]
# fmt: on


QUICK_SCENARIOS = [s for s in PRODUCTION_SCENARIOS if "quick" in s.tags]
DEFAULT_SCENARIOS = [s for s in PRODUCTION_SCENARIOS if "large" not in s.tags]

# Subset for long-cache decode tuning (see --decode-focus)
DECODE_FOCUS_NAMES = ("decode_single_64k_cache", "decode_production_avg_cache")
DECODE_FOCUS_SCENARIOS = [s for s in PRODUCTION_SCENARIOS if s.name in DECODE_FOCUS_NAMES]


# ---------------------------------------------------------------------------
# MLA forward pass — official DeepSeek-V3 absorbed inference layout
#
# Weights (see deepseek-ai/DeepSeek-V3 inference/model.py):
#   wkv_a : x → [c_kv, k_pe_raw]           (cache write; not used in attn read)
#   wkv_b : c_kv → [K_nope, V] per head     shape [H, nope+v, kv_rank]
#           W_uk = wkv_b[:, :nope, :]       Q absorption
#           W_uv = wkv_b[:, -v_dim:, :]     V reconstruction after attn
#
# Cache at inference (absorbed path):
#   c_kv      [B, Sk, kv_rank]   compressed KV latent (kv_cache)
#   pe_cache  [B, Sk, rope_dim]  K_rope AFTER RoPE — shared across heads
#
# Attention:
#   scores = Q_absorbed @ c_kv^T + Q_rope @ pe_cache^T
#   ctx    = softmax(scores) @ c_kv
#   out    = ctx @ W_uv^T
# ---------------------------------------------------------------------------


# DeepSeek-V3 reference dims (production): hidden=7168, H=128, kv_rank=512,
# qk_nope=128, qk_rope=64, v=128, qk_head=192 (128+64), wkv_b row=256 (128+128).
# B, Sq, Sk vary per scenario; numbers below use V3 literals where fixed.


def apply_rope(x, positions, rope_dim):
    # in:  x [..., rope_dim=64]
    # pair-split: [..., 64] → even/odd [..., 32] each (adjacent pairs for RoPE)
    # Angles in the kernel's acc dtype (fp64 in stays fp64), not an unconditional fp32:
    # the angle is position * inv_freq, so at Sq=256 an fp32 angle costs ~6e-6 relative
    # and the fp64 reference would be less accurate than the kernel it is checking.
    acc = torch.float64 if x.dtype == torch.float64 else torch.float32
    freqs = 1.0 / (
        10000
        ** (torch.arange(0, rope_dim, 2, device=x.device, dtype=acc) / rope_dim)
    )
    pos = positions.to(device=x.device, dtype=acc)
    angles = torch.outer(pos, freqs)  # [Sq, 32]
    cos = torch.cos(angles).to(dtype=x.dtype).view(1, -1, 1, rope_dim // 2)
    sin = torch.sin(angles).to(dtype=x.dtype).view(1, -1, 1, rope_dim // 2)

    x_fp = x.to(acc)
    x_even = x_fp[..., 0::2]   # [..., 32]
    x_odd = x_fp[..., 1::2]    # [..., 32]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    # merge pairs back: [..., 32, 2] → flatten → [..., 64]
    return torch.stack([out_even, out_odd], dim=-1).flatten(-2).type_as(x)


class QProjection(NamedTuple):
    """DeepSeek-V3 computes Q as wq_b(q_norm(wq_a(x))), factorising the 7168 → 24576
    projection through a q_lora_rank=1536 latent instead of one dense matrix. That is
    48.7M weights rather than 176M — 195 MB against 704 MB in fp32 — which is what
    single-stream decode is bound by, since there Q-prep is a GEMV that touches every
    weight exactly once. The RMSNorm is load-bearing: without it the two matmuls collapse
    into a single rank-1536 matrix.
    """

    a: torch.Tensor       # [hidden, q_lora]
    norm_w: torch.Tensor  # [q_lora]  RMSNorm gain
    b: torch.Tensor       # [q_lora, H*qk_head_dim]


RMS_NORM_EPS = 1e-6


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = RMS_NORM_EPS):
    # Accumulate in the kernel's acc dtype, not an unconditional .float(): fp64 in must
    # stay fp64 all the way through, or the reference is less accurate than the kernel
    # it is checking.
    acc = torch.float64 if x.dtype == torch.float64 else torch.float32
    var = x.to(acc).pow(2).mean(-1, keepdim=True)
    return (x.to(acc) * torch.rsqrt(var + eps)).to(x.dtype) * weight


def split_wkv_b(wkv_b: torch.Tensor, cfg: DeepSeekV3Config):
    # split wkv_b [H=128, nope+v=256, kv_rank=512] along dim 1:
    W_uk = wkv_b[:, : cfg.qk_nope_head_dim, :]   # [128, 128, 512]  K_nope / Q-absorb
    W_uv = wkv_b[:, cfg.qk_nope_head_dim :, :]   # [128, 128, 512]  V reconstruct
    return W_uk, W_uv


def prepare_q_mla_absorbed(x, W_q: QProjection, W_uk, q_positions, cfg: DeepSeekV3Config):
    batch, sq, _ = x.shape
    n_heads = cfg.n_heads

    # Q projection, factorised: [B,Sq,7168] @ [7168,1536] → RMSNorm → @ [1536,24576]
    q_lat = torch.matmul(x, W_q.a)                        # [B,Sq,1536]
    q_lat = rmsnorm(q_lat, W_q.norm_w)                    # [B,Sq,1536]
    q_raw = torch.matmul(q_lat, W_q.b)                    # [B,Sq,24576]
    # reshape: [B,Sq,24576] → [B,Sq,128,192]
    q_raw = q_raw.view(batch, sq, n_heads, cfg.qk_head_dim)
    # split head dim: 192 → nope | rope
    q_nope = q_raw[..., : cfg.qk_nope_head_dim]          # [B,Sq,128,128]
    q_rope = q_raw[..., cfg.qk_nope_head_dim :]          # [B,Sq,128,64]

    # absorb: [B,Sq,128,128] × [128,128,512] → [B,Sq,128,512]
    q_absorbed = torch.einsum("bsni,nir->bsnr", q_nope, W_uk)
    # RoPE on rope slice: [B,Sq,128,64] → [B,Sq,128,64]
    q_rope_rot = apply_rope(q_rope, q_positions, cfg.qk_rope_head_dim)
    return q_absorbed, q_rope_rot


def mla_attention_official(q_absorbed, q_rope, c_kv, pe_cache, W_uv, cfg: DeepSeekV3Config):
    """
    Official absorbed MLA attention (DeepSeek inference/model.py absorbed path).
    pe_cache: [B, Sk, rope_dim] — RoPE-applied K rope, NOT per-head.
    """
    # nope scores: [B,Sq,128,512] @ [B,Sk,512]^T → [B,Sq,128,Sk]
    scores_nope = torch.einsum("bqhr,bsr->bqhs", q_absorbed, c_kv)
    # rope scores: [B,Sq,128,64] @ [B,Sk,64]^T → [B,Sq,128,Sk]  (pe_cache head-shared)
    scores_rope = torch.einsum("bqhd,bsd->bqhs", q_rope, pe_cache)
    scale = cfg.qk_head_dim ** 0.5  # sqrt(192)
    scores = (scores_nope + scores_rope) / scale  # [B,Sq,128,Sk]
    attn = torch.softmax(scores, dim=-1)          # [B,Sq,128,Sk]
    # ctx: [B,Sq,128,Sk] @ [B,Sk,512] → [B,Sq,128,512]
    ctx = torch.einsum("bqhs,bsr->bqhr", attn, c_kv)
    # V reconstruct: [B,Sq,128,512] × [128,128,512] → [B,Sq,128,128]
    return torch.einsum("bqhr,hdr->bqhd", ctx, W_uv)


# ---------------------------------------------------------------------------
# CUDA fused versions of Q-prep and attention
# ---------------------------------------------------------------------------

def prepare_q_mla_absorbed_cuda(x, W_q: QProjection, W_uk, q_positions, cfg: DeepSeekV3Config):
    """Kernel 2: x[B,Sq,7168] → Q_absorbed[B,Sq,128,512], Q_rope[B,Sq,128,64].

    q_positions [Sq] holds the absolute position of each query and is required: in
    decode Sq == 1 while the query sits at the end of the KV cache, so the kernel
    cannot derive it from the shape.
    """
    q_absorbed, q_rope = mla_custom_cuda.q_absorbed(
        x, W_q.a.contiguous(), W_q.norm_w.contiguous(), W_q.b.contiguous(),
        W_uk.contiguous(), q_positions,
        cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, RMS_NORM_EPS,
    )
    return q_absorbed, q_rope  # [B,Sq,128,512], [B,Sq,128,64]


def mla_attention_official_cuda(q_absorbed, q_rope, c_kv, pe_cache, W_uv, cfg: DeepSeekV3Config):
    """Kernel 3: scores → softmax → ctx@c_kv → @W_uv; returns [B,Sq,128,128]."""
    scale = float(cfg.qk_head_dim) ** 0.5  # sqrt(192)
    return mla_custom_cuda.mla_attention(
        q_absorbed.contiguous(),
        q_rope.contiguous(),
        c_kv.contiguous(),
        pe_cache.contiguous(),
        W_uv.contiguous(),
        scale,
    )


def make_mla_tensors(scenario: BenchmarkScenario, cfg: DeepSeekV3Config, dtype=torch.float32):
    device = "cuda"
    batch, sq, sk = scenario.batch, scenario.sq, scenario.sk

    # Xavier-ish scales so 7168-wide dots stay O(1). Raw N(0,1) weights make
    # scores ~1e3–1e4, which overflows the fp32 softmax path in kernel 3 and
    # exaggerates reduction-order differences in fused Q-prep.
    x = torch.randn(batch, sq, cfg.hidden_dim, device=device, dtype=dtype)
    x = x * (1.0 / cfg.hidden_dim ** 0.5)

    c_kv = torch.randn(batch, sk, cfg.kv_lora_rank, device=device, dtype=dtype)
    c_kv = c_kv * (1.0 / cfg.kv_lora_rank ** 0.5)
    pe_cache = torch.randn(batch, sk, cfg.qk_rope_head_dim, device=device, dtype=dtype)
    pe_cache = pe_cache * (1.0 / cfg.qk_rope_head_dim ** 0.5)

    # Q projection, factorised the way DeepSeek-V3 does it: 7168 → 1536 → 24576 with an
    # RMSNorm on the latent. 1/sqrt(fan_in) on both keeps the chain O(1): the norm pins
    # the latent to unit RMS, so q_raw lands at ~1 and scores at ~1/sqrt(192).
    W_q_a = torch.randn(cfg.hidden_dim, cfg.q_lora_rank, device=device, dtype=dtype)
    W_q_a = W_q_a * (1.0 / cfg.hidden_dim ** 0.5)
    W_q_b = torch.randn(
        cfg.q_lora_rank, cfg.n_heads * cfg.qk_head_dim, device=device, dtype=dtype
    )
    W_q_b = W_q_b * (1.0 / cfg.q_lora_rank ** 0.5)
    # Learned gain, initialised at 1 in the real model; jitter it so a kernel that drops
    # the gain entirely still fails the correctness check.
    q_norm_w = 1.0 + 0.1 * torch.randn(cfg.q_lora_rank, device=device, dtype=dtype)
    W_q = QProjection(W_q_a, q_norm_w, W_q_b)

    # wkv_b [H, nope+v, kv_rank] — official reshape of Linear(kv_rank → H*(128+128))
    wkv_b = torch.randn(
        cfg.n_heads,
        cfg.qk_nope_head_dim + cfg.v_head_dim,
        cfg.kv_lora_rank,
        device=device,
        dtype=dtype,
    )
    wkv_b = wkv_b * (1.0 / cfg.qk_nope_head_dim ** 0.5)

    # The query tokens are the *newest* Sq entries of a cache holding Sk of them, so
    # they sit at absolute positions [Sk-Sq, Sk). For prefill (Sq == Sk) this is the
    # usual arange(Sq); for decode it is what makes the scenario real. arange(Sq) would
    # place a decode query at position 0, where the RoPE angle is 0 and the rotation
    # degenerates to the identity — the decode rows would never exercise RoPE at all.
    q_positions = torch.arange(sk - sq, sk, device=device)
    return x, c_kv, pe_cache, W_q, wkv_b, q_positions


def baseline_official_mla_attention(x, c_kv, pe_cache, W_q, wkv_b, q_positions, cfg):
    W_uk, W_uv = split_wkv_b(wkv_b, cfg)
    q_absorbed, q_rope = prepare_q_mla_absorbed(x, W_q, W_uk, q_positions, cfg)
    return mla_attention_official(q_absorbed, q_rope, c_kv, pe_cache, W_uv, cfg)


def cuda_fully_fused_mla_attention(x, c_kv, pe_cache, W_q, wkv_b, q_positions, cfg):
    """Fused CUDA Q-prep (kernel 2) + CUDA attention (kernel 3)."""
    W_uk, W_uv = split_wkv_b(wkv_b, cfg)
    q_absorbed, q_rope = prepare_q_mla_absorbed_cuda(x, W_q, W_uk, q_positions, cfg)
    return mla_attention_official_cuda(q_absorbed, q_rope, c_kv, pe_cache, W_uv, cfg)


def benchmark(func: Callable, *args, num_iters: int = 30) -> float:
    for _ in range(min(10, num_iters)):
        func(*args)
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

    for i in range(num_iters):
        cache = torch.empty(int(256e6), dtype=torch.int8, device="cuda")
        cache.zero_()
        start_events[i].record()
        func(*args)
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return sum(times) / len(times)


@dataclass
class BenchmarkResult:
    scenario: str
    phase: str
    batch: int
    sq: int
    sk: int
    baseline_ms: float
    fused_ms: float    # CUDA fully-fused (Q-prep + attention)
    speedup: float     # baseline / fused_ms
    passed: bool
    skipped: bool = False
    skip_reason: str = ""


def run_scenario(
    scenario: BenchmarkScenario,
    cfg: DeepSeekV3Config,
    dtype=torch.float32,
) -> BenchmarkResult:
    header = (
        f"\n{'-' * 78}\n"
        f"[{scenario.phase.upper()}] {scenario.name}\n"
        f"  {scenario.description}\n"
        f"  Source: {scenario.source}\n"
        f"  Shapes: batch={scenario.batch}  sq={scenario.sq}  sk={scenario.sk}  "
        f"hidden={cfg.hidden_dim}  heads={cfg.n_heads}  "
        f"kv_rank={cfg.kv_lora_rank}  Q_final={cfg.q_absorbed_dim}\n"
        f"{'-' * 78}"
    )
    print(header)

    try:
        tensors = make_mla_tensors(scenario, cfg, dtype)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        reason = "OOM while allocating tensors"
        print(f"  SKIPPED: {reason}")
        return BenchmarkResult(
            scenario.name, scenario.phase, scenario.batch, scenario.sq, scenario.sk,
            0, 0, 0, False, skipped=True, skip_reason=reason,
        )

    x, c_kv, pe_cache, W_q, wkv_b, q_positions = tensors
    common = (x, c_kv, pe_cache, W_q, wkv_b, q_positions, cfg)

    try:
        print("  Correctness check...")
        base_out = baseline_official_mla_attention(*common)
        fused_out = cuda_fully_fused_mla_attention(*common)

        tol = dict(rtol=3e-2, atol=3e-2)
        # Long Sk: hand-written fp32 reductions disagree with cuBLAS einsum by more
        # than a flat 3e-2 abs tol (same order as fp32 vs fp64 for this op).
        if scenario.sk > 2048:
            tol = dict(rtol=5e-2, atol=max(3e-2, 3e-5 * scenario.sk))
        fused_ok = torch.allclose(base_out, fused_out, **tol)
        diff_fused = torch.max(torch.abs(base_out - fused_out)).item()

        print(f"  CUDA fully-fused vs baseline:  {'PASS' if fused_ok else 'FAIL'} (max diff {diff_fused:.6f})")

        if not fused_ok:
            return BenchmarkResult(
                scenario.name, scenario.phase, scenario.batch, scenario.sq, scenario.sk,
                0, 0, 0, False)

        print("  Benchmarking end-to-end MLA attention...")
        base_time = benchmark(baseline_official_mla_attention, *common, num_iters=scenario.num_iters)
        fused_time = benchmark(cuda_fully_fused_mla_attention, *common, num_iters=scenario.num_iters)

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        reason = "OOM during forward/benchmark"
        print(f"  SKIPPED: {reason}")
        return BenchmarkResult(
            scenario.name, scenario.phase, scenario.batch, scenario.sq, scenario.sk,
            0, 0, 0, False, skipped=True, skip_reason=reason,
        )

    speedup = base_time / fused_time
    print(f"  PyTorch baseline:  {base_time:.4f} ms")
    print(f"  CUDA fully-fused:  {fused_time:.4f} ms  ({speedup:.2f}x)")

    return BenchmarkResult(
        scenario.name, scenario.phase, scenario.batch, scenario.sq, scenario.sk,
        base_time, fused_time, speedup, True,
    )


def print_summary(results: List[BenchmarkResult], cfg: DeepSeekV3Config) -> None:
    print("\n" + "=" * 110)
    print("  BENCHMARK SUMMARY")
    print("=" * 110)
    print(
        f"  Model ratios: DeepSeek-V3  hidden={cfg.hidden_dim}  heads={cfg.n_heads}  "
        f"kv_rank={cfg.kv_lora_rank}  q_lora={cfg.q_lora_rank}  "
        f"qk=[{cfg.qk_nope_head_dim}+{cfg.qk_rope_head_dim}]  v={cfg.v_head_dim}  "
        f"Q_final={cfg.q_absorbed_dim}"
    )
    print("  Fused ms = fully-fused CUDA (Q-prep + attention)")
    print("-" * 78)
    print(
        f"{'Scenario':<32} {'Phase':<8} {'B':>4} {'Sq':>6} {'Sk':>7} "
        f"{'Base ms':>9} {'Fused ms':>9} {'Speedup':>9}"
    )
    print("-" * 78)

    for r in results:
        if r.skipped:
            reason = f" ({r.skip_reason})" if r.skip_reason else ""
            print(
                f"{r.scenario:<32} {r.phase:<8} {r.batch:>4} {r.sq:>6} {r.sk:>7} "
                f"{'SKIP':>9} {'SKIP':>9} {'—':>9}{reason}"
            )
            continue
        status = "ok" if r.passed else "FAIL"
        print(
            f"{r.scenario:<32} {r.phase:<8} {r.batch:>4} {r.sq:>6} {r.sk:>7} "
            f"{r.baseline_ms:>9.2f} {r.fused_ms:>9.2f} {r.speedup:>8.2f}x  [{status}]"
        )
    print("=" * 78)


def _resolve_scenario_names(args: argparse.Namespace) -> List[str]:
    names: List[str] = []
    if args.scenarios:
        for part in args.scenarios:
            names.extend(n.strip() for n in part.split(",") if n.strip())
    if args.scenario:
        names.append(args.scenario)
    # dedupe, preserve order
    seen: set = set()
    ordered: List[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    return ordered


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepSeek V3 MLA attention benchmark suite")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run smoke test only (fast CI)",
    )
    parser.add_argument(
        "--decode-focus",
        action="store_true",
        help=(
            "Run decode_single_64k_cache and decode_production_avg_cache only "
            "(long-cache decode targets)"
        ),
    )
    parser.add_argument(
        "--include-large",
        action="store_true",
        help="Include large prefill scenarios (e.g. sq=sk=8192) that may OOM",
    )
    parser.add_argument(
        "--phase",
        choices=["all", "decode", "prefill", "mixed"],
        default="all",
        help="Filter scenarios by workload phase",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Run one scenario by name (e.g. decode_production_avg_cache)",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        metavar="NAME",
        help=(
            "Run only these scenarios (space- or comma-separated), e.g. "
            "--scenarios decode_single_64k_cache decode_production_avg_cache"
        ),
    )
    return parser.parse_args(argv)


def select_scenarios(args: argparse.Namespace) -> List[BenchmarkScenario]:
    if args.decode_focus:
        return list(DECODE_FOCUS_SCENARIOS)

    scenario_names = _resolve_scenario_names(args)
    if scenario_names:
        by_name = {s.name: s for s in PRODUCTION_SCENARIOS}
        missing = [n for n in scenario_names if n not in by_name]
        if missing:
            names = ", ".join(s.name for s in PRODUCTION_SCENARIOS)
            print(f"Unknown scenario(s) {missing}. Available: {names}", file=sys.stderr)
            sys.exit(1)
        return [by_name[n] for n in scenario_names]

    if args.quick:
        return QUICK_SCENARIOS

    scenarios = PRODUCTION_SCENARIOS if args.include_large else DEFAULT_SCENARIOS

    if args.phase != "all":
        scenarios = [s for s in scenarios if s.phase == args.phase]

    return scenarios


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    # Match fp32 matmul numerics between PyTorch baseline and extension Q-prep.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    print("=" * 78)
    print("  DEEPSEEK V3 MLA ATTENTION BENCHMARK SUITE")
    print("=" * 78)
    cfg = DEEPSEEK_V3
    print(
        f"  Config: hidden={cfg.hidden_dim}  heads={cfg.n_heads}  "
        f"kv_lora_rank={cfg.kv_lora_rank}  q_lora_rank={cfg.q_lora_rank}  "
        f"qk_nope={cfg.qk_nope_head_dim}  qk_rope={cfg.qk_rope_head_dim}  "
        f"v_head={cfg.v_head_dim}  Q_final={cfg.q_absorbed_dim}"
    )

    scenarios = select_scenarios(args)
    print(f"  Running {len(scenarios)} scenario(s)...")

    results: List[BenchmarkResult] = []
    for scenario in scenarios:
        results.append(run_scenario(scenario, cfg))

    print_summary(results, cfg)


if __name__ == "__main__":
    main()
