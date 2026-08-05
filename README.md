# Multi-head Latent Attention CUDA Kernel

Hand-written CUDA kernels for DeepSeek-V3's **Multi-head Latent Attention (MLA)**,
absorbed-inference path, benchmarked against a PyTorch reference implementation.

Dimension ratios and the absorbed-path layout follow the DeepSeek-V3 technical
report (arXiv:2412.19437) and the official `inference/model.py`:

```
kv_cache c_kv   [B, Sk, 512]
pe_cache        [B, Sk, 64]
wkv_b           [H, nope+v, 512]
```

## What's in here

The forward pass is split into two fused CUDA kernels that replace the
equivalent PyTorch/cuBLAS ops:

| Stage | Kernel | Computes |
|---|---|---|
| 2 | `q_absorbed` | `W_q_b(RMSNorm(x @ W_q_a))` → split nope/rope → absorb `@ W_uk` → RoPE on `Q_rope` |
| 3 | `mla_attention` | scores (`Q_absorbed @ c_kv^T + Q_rope @ pe_cache^T`) → softmax → `ctx = attn @ c_kv` → `out = ctx @ W_uv` |

Kernel 2 runs 2a1 (`x @ W_q_a`) → 2a-norm → 2a2 (`@ W_q_b`) → 2b (RoPE) →
2c (absorb). The Q projection is factorised through the `q_lora_rank = 1536`
latent the way DeepSeek-V3 does it rather than as one dense `[7168, 24576]`
matrix — 48.7M weights instead of 176M, or 195 MB against 704 MB in fp32.
That dominates single-stream decode, where Q-prep is a GEMV that touches every
weight exactly once. The RMSNorm is load-bearing: without it the two matmuls
collapse into a single rank-1536 matrix.

Kernel 3 internally runs sub-stages 3a (scores) → 3b (softmax) → 3c (ctx) →
3d (output projection); see the comment block above `KERNEL 3` in
[mla_kernels.cu](mla_kernels.cu) for the tiling scheme. Kernels support
fp16/fp32/fp64 inputs, with fp32 accumulation for fp16 (and fp64 for fp64).

### Repo layout

| File | Purpose |
|---|---|
| `mla_kernels.cu` | CUDA kernel source + pybind11 bindings (`mla_custom_cuda` extension) |
| `setup.py` | Builds the `mla_custom_cuda` PyTorch CUDA extension |
| `run_eval.sh` | Rebuilds the extension and runs the benchmark suite |
| `exhaustive_benchmark_suite.py` | PyTorch baseline vs. fully-fused CUDA correctness + speed benchmarks across production-shaped scenarios (decode/prefill) |
| `stage_profile.py` | Per-stage (weight-prep / Q-prep / attention) timing + roofline breakdown of the fused path |
| `kernel_profile.py` | `torch.profiler`-based breakdown of kernel 3's sub-stages (3a-3d) |
| `check_dtypes.py` | Correctness sweep of `mla_attention` across fp16/fp32/fp64, plain and split-K paths |
| `check_3d.py` | Correctness sweep over shapes straddling kernel 3's tile boundaries |

## Requirements

- NVIDIA GPU, compute capability 8.0+ (Ampere/Ada/Hopper — prebuilt for
  `sm_80`, `sm_89`, `sm_90`; other Ampere/Ada/Hopper GPUs, e.g. `sm_86`, run
  the same binary via forward compatibility)
- CUDA toolkit 12.x (`nvcc` on `PATH`)
- conda (miniconda/anaconda)

## Setting up the conda environment

This was built and verified against **Python 3.9 + PyTorch 2.3.0 (cu121)**.
`run_eval.sh` looks for a conda env named `ece285` by default (override with
`MLA_CONDA_ENV`), under `~/miniconda3` by default (override with
`CONDA_ROOT`).

```bash
conda create -n ece285 python=3.9 -y
conda activate ece285

# PyTorch built against CUDA 12.1 (matches this repo's setup.py gencode flags)
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121

# optional but speeds up `pip install -e .` rebuilds
pip install ninja
```

Verify the GPU is visible:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Build and run

`run_eval.sh` activates the conda env (if found on `CONDA_ROOT`), rebuilds
the `mla_custom_cuda` extension, and runs the benchmark suite:

```bash
bash run_eval.sh --quick              # smoke test only (1 scenario)
bash run_eval.sh                      # default set: 12 scenarios (everything except --include-large)
bash run_eval.sh --phase decode       # ALL 7 decode scenarios
bash run_eval.sh --phase prefill      # ALL 4 prefill scenarios
bash run_eval.sh --include-large      # adds prefill_nvidia_reference (sq=sk=8192, may OOM)

# Named subsets — note --decode-focus is NOT "all decode": it is a hardcoded
# 2-scenario tuning subset (decode_single_64k_cache, decode_production_avg_cache).
# Use --phase decode to run all 7.
bash run_eval.sh --decode-focus
bash run_eval.sh --scenario decode_production_avg_cache
bash run_eval.sh --scenarios decode_single_64k_cache decode_production_avg_cache
```

Skip the rebuild step (if the extension is already built) with:

```bash
MLA_SKIP_BUILD=1 bash run_eval.sh --decode-focus
```

To build/install manually instead of via `run_eval.sh`:

```bash
pip install -e .
python exhaustive_benchmark_suite.py --quick
```

Each scenario prints a correctness check (PyTorch baseline vs. fully-fused
CUDA, `torch.allclose` within scenario-appropriate tolerance) followed by
timing, then a summary table:

```
Scenario                         Phase       B     Sq      Sk   Base ms  Fused ms   Speedup
decode_production_avg_cache      decode    128      1    4989      1.42      0.31    4.58x
...
```

### Other scripts

```bash
python stage_profile.py     # per-stage timing breakdown + roofline
python kernel_profile.py    # torch.profiler breakdown of kernel 3 sub-stages
python check_dtypes.py      # dtype correctness sweep (fp16/fp32/fp64)
python check_3d.py          # tile-boundary correctness sweep
```
