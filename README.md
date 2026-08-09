# Multi-head Latent Attention — Hand-Written CUDA Kernels

Hand-written CUDA implementation of DeepSeek-V3's **Multi-head Latent Attention
(MLA)**, absorbed-inference path, benchmarked end to end against a
PyTorch/cuBLAS reference on an RTX A6000.

```
hidden = 7168     H (heads)  = 128     kv_lora_rank  = 512     q_lora_rank = 1536
nope   = 128      rope       = 64      qk_head_dim   = 192     v_head_dim  = 128
                                       Q_absorbed    = 512 + 64 = 576
```

**Result up front:** correctness passes on all 10 production-shaped scenarios,
and the fused path runs at a **geometric-mean 0.77× of the PyTorch/cuBLAS
baseline** — about 1.3× slower, tightly grouped between 0.71× and 0.83×.
See [§3](#3-benchmarks).

## Contents

1. [Why MLA](#1-why-mla)
2. [The Algorithm, Step by Step](#2-the-algorithm-step-by-step)
3. [Benchmarks](#3-benchmarks)
4. [Build and Run](#4-build-and-run)

---

# 1. Why MLA

![Why MLA](docs/fig0_why_mla.png)

Two consequences shape every kernel here:

- **The cache carries no head index.** All 128 heads of a query read the *same*
  cache rows, so a block that owns all 128 heads at once stages the cache once
  and reuses it 128 times.
- **Decode is memory-bound.** With `Sq = 1` there is no arithmetic intensity to
  find in attention itself; the wins come from not moving bytes and from keeping
  all 84 SMs busy.

---

# 2. The Algorithm, Step by Step

![The 9 steps](docs/fig0_pipeline.png)

Shapes throughout are for **batch-128 decode**: `B=128, Sq=1, H=128`, so
`bq = B·Sq = 128` rows and `flat = B·Sq·H = 16,384` rows, at DeepSeek's reported
production-average cache depth `Sk = 4,989`.

### The Shared GEMM Recipe

Five of the eight kernels (2a1, 2a2, 2c, 3a, 3c, 3d) are the same GEMM shape and
use one recipe, so the per-step notes below only list what is *specific* to that
step:

- **128×128 output tile per block**, `16×16 = 256` threads, **8×8 outputs per
  thread**. An M×N register tile costs M+N shared-memory loads per M·N FMAs, so
  8×8 gives 0.25 loads/FMA — sm_86 issues 128 FP32 lanes against 32 shared-memory
  words per clock, i.e. it wants exactly 4 FMAs per loaded word.
- **Reduction contracted in shared-memory tiles of 32**, staged cooperatively by
  all 256 threads.
- **Coalescing:** `threadIdx.x` indexes the contiguous output dimension, so a
  warp writes runs of adjacent addresses — 4.00 store sectors per request (the
  ideal) against 23.99 if the row index came from `threadIdx.x`.
- **Bank-conflict-free shared memory** by `+1` row padding rather than an XOR
  swizzle, which keeps the address linear in the reduction index.
- **Occupancy traded for ILP:** 8×8 costs ~128 registers → 2 blocks/SM. The
  inner loop is deliberately *not* unrolled; any spill loses more than the
  occupancy is worth.
- **Adaptive `bq` tile** (16/32/64/128 rows) so a single-stream decode does not
  spend 127/128 of its FMAs on rows the epilogue discards.

---

## Step 2a1 — Project the Hidden State into the Latent

![Step 2a1](docs/step_2a1.png)

**Techniques Specific to This Step**

- **Low-rank factorization (algorithmic).** This step exists only because the Q
  projection is factored through `q_lora_rank = 1536` rather than applied as one
  dense `[7168, 24576]` matrix. Together with 2a2 that is 48.8 M weights instead
  of 176 M — **195 MB instead of 704 MB**. At single-stream decode this is a
  GEMV, so runtime is weight bytes ÷ bandwidth and nothing else.
- **Split-K over the 7,168 reduction.** The output is only 1536 columns = 12
  block tiles, so without it this step runs 12 blocks on an 84-SM GPU.

## Step 2a-norm — RMSNorm on the Latent

![Step 2a-norm](docs/step_2an.png)

**Techniques Specific to This Step**

- **Single-kernel block reduction.** One block per row, warp shuffles then one
  pass over the per-warp partials — no second launch and no global round-trip
  for the mean.
- **Accumulate wide, hand off narrow.** Reads 2a1's fp32 accumulator and writes
  the input dtype, which is what lets 2a2 reuse the same GEMM kernel unchanged.
- The norm is **load-bearing, not incidental**: with nothing between them,
  `W_q_a @ W_q_b` collapses into a single rank-1536 matrix and the factorization
  can be undone.

## Step 2a2 — Expand the Latent into the Per-Head Query

![Step 2a2](docs/step_2a2.png)

**Techniques Specific to This Step**

- **Kernel reuse.** Same templated GEMM as 2a1; only the shapes and the split-K
  plan differ.
- **Split-K sized against wave quantization.** 24576/128 = 192 blocks against
  84 SMs × 2 resident = 168 slots is **1.14 waves**: one full wave, then a tail
  wave leaving 144 slots idle — 43% of the machine. The planner targets ~8 waves,
  where the tail costs ~11%.
- **Blocking over `bq`, not one block per (row, head)**, which would re-read this
  weight once per row and turn 151 MB of traffic into 19.3 GB.

## Step 2b — RoPE on the Rope Half

![Step 2b](docs/step_2b.png)

**Techniques Specific to This Step**

- **Grid-stride loop capped at 32 × SM**, so the launch never scales with cache
  depth.
- **Absolute positions are a required argument** — in decode `Sq == 1` while the
  query sits at position `Sk`, so it cannot be recovered from the shape.
  `arange(Sq)` would place the query at position 0, where the rotation is the
  identity.

## Step 2c — Absorb `W_uk` into the Query

![Step 2c](docs/step_2c.png)

**Techniques Specific to This Step**

- **One head per block** (`grid.z = h`), so `W_uk[h]` (262 KB) is staged into
  shared memory once and amortized over the whole row tile instead of being
  re-read for every `(row, head)` pair.
- **In-place slicing.** The nope half is read straight out of `q_raw` at stride
  192; no separate slice tensor is materialized.

## Step 3a — Scores, then the Local Softmax Numerator

![Step 3a](docs/step_3a.png)

**Techniques Specific to This Step**

- **One merged 576-deep reduction.** `q_absorbed·c_kv` (512) and
  `q_rope·pe_cache` (64) run as a single loop that switches source operand at the
  boundary — no mid-kernel re-stage of aliased shared memory, no extra barrier
  pair, one row width throughout.
- **Cache reuse across all 128 heads.** The block tile is 128 rows = exactly `H`,
  so a block's rows are the 128 heads of one query and the cache tile it stages
  serves all of them.
- **Softmax exponentials moved *here*.** Step 3c stages each score once per rank
  tile — 4 times — so any per-score work there is done 4× over. 3a sees each score
  exactly once, in registers, so it exponentiates in the epilogue and gets the
  row sum for free.
- **Per-block partials instead of atomics.** Emitting `(m_j, l_j)` for step 3b to
  reconcile removes a float `atomicMax` (which CUDA has no native instruction
  for) and ~39 atomics per row.
- **Output row-stride padded to 128 B lines.** A row stride that is not a whole
  cache line makes every row start at a different offset; padding took store
  sectors per request from **5.75 → 4.00**.

## Step 3b — Reconcile the Per-Block Softmax Maxima

![Step 3b](docs/step_3b.png)

**Techniques Specific to This Step**

- **The flash-softmax combine, done as a separate pass.** Exact because
  `exp(s − m_j) · exp(m_j − M) = exp(s − M)` holds termwise for *any* `m_j`.
- **The kernel boundary is the grid-wide barrier**, for free. This cannot be done
  with atomics inside 3a: `(M, L)` is a *coupled* two-word update — `L`'s
  correction depends on `M`'s new value — which would need a lock or a 128-bit CAS
  under heavy contention.
- **`alpha` does double duty**: it is the weight in this kernel's denominator sum
  *and* exactly the factor step 3c needs, so one array serves both.

## Step 3c — Context

![Step 3c](docs/step_3c.png)

**Techniques Specific to This Step**

- **The correction rides the staging load.** Multiplying by `alpha_j` as the value
  lands means shared memory already holds `exp(s − M)` — the softmax numerator,
  ready to reduce. The Sk loop is *stage → barrier → reduce → barrier* with
  nothing in between: no exponential, no max scan, no row-sum pass.
- **Split-K over Sk.** `ctx` has no `Sk` axis, so at `B=Sq=1` the output alone
  yields 4 blocks and ~95% of the SMs would idle. Every slice stages values
  already on the global scale `M`, so the partials are addable *by construction*.
- **`alpha` read straight from global**, not staged: only 128 distinct values are
  live per tile and the threads sharing a row hit the same address, so it stays
  L1-resident and costs no barrier.

## Step 3d — Reconstruct V and Project Out

![Step 3d](docs/step_3d.png)

**Techniques Specific to This Step**

- **One head per block**, same reuse argument as 2c.
- **The deferred softmax divide.** Step 3c emitted the unnormalized numerator;
  this kernel loads one reciprocal per row and folds the multiply into a staging
  load it was already doing. Deferring it is what keeps 3c's split-K partials
  addable.

---

# 3. Benchmarks

## What the Baseline Is

`baseline_official_mla_attention` in
[exhaustive_benchmark_suite.py](exhaustive_benchmark_suite.py) — the **same
algorithm** in idiomatic PyTorch on cuBLAS, transcribed from DeepSeek-V3's
official `inference/model.py` absorbed path: two GEMMs for the Q projection, a
batched GEMM for the absorb, two einsums for scores, `torch.softmax`, then two
more einsums for context and output.

What makes it a fair comparison rather than a strawman:

- **Identical mathematics and shapes.** Both sides consume the same factorized
  `W_q_a / q_norm_w / W_q_b`, the same `wkv_b`, the same absolute positions.
- **TF32 disabled on both sides.** The baseline runs true FP32 cuBLAS; leaving
  TF32 on would have handed it tensor cores while the kernels stayed on CUDA
  cores.
- **Both timed end to end**, Q-prep through output projection — the fused number
  includes all nine launches plus two `.contiguous()` copies.
- **It is the real production path**, not a naive loop.

Method: FP32, RTX A6000, idle GPU, `torch.cuda.Event` timing, 10 warm-up
iterations, **L2 flushed with a 256 MB buffer between every timed iteration** so
both sides measure the cold-cache behaviour a real decode step sees.

## Results

![Results](docs/fig_results.png)

| Scenario | Phase | B | Sq | Sk | Base ms | Fused ms | Speedup |
|---|---|--:|--:|--:|--:|--:|--:|
| `decode_single_user_long_ctx` | decode | 1 | 1 | 65536 | 1.92 | 2.57 | 0.75× |
| `decode_serving_avg_ctx` | decode | 128 | 1 | 4989 | 14.14 | 18.11 | 0.78× |
| `decode_serving_long_ctx` | decode | 128 | 1 | 8192 | 23.43 | 28.40 | 0.83× |
| `decode_serving_high_batch` | decode | 256 | 1 | 2048 | 12.79 | 16.11 | 0.79× |
| `decode_speculative_2tok` | decode | 64 | 2 | 512 | 2.59 | 3.65 | 0.71× |
| `decode_speculative_4tok` | decode | 64 | 4 | 768 | 6.10 | 8.07 | 0.76× |
| `decode_speculative_long_ctx` | decode | 132 | 4 | 4096 | 46.83 | 58.41 | 0.80× |
| `prefill_chat_batch` | prefill | 8 | 512 | 512 | 69.95 | 93.56 | 0.75× |
| `prefill_chunk_2k` | prefill | 1 | 2048 | 2048 | 92.57 | 119.79 | 0.77× |
| `prefill_chunk_4k` | prefill | 1 | 4096 | 4096 | 351.10 | 437.59 | 0.80× |

**Average: 0.77× geometric mean** (0.77× arithmetic, 0.79× aggregate
Σbase/Σfused), and the same 0.77× for decode and prefill taken separately.

## Where the Time Goes

`torch.profiler`, summed over the seven decode scenarios:

| step | sum ms | share |
|---|--:|--:|
| **3a** scores | 60.68 | **44.8%** |
| **3c** ctx | 53.97 | **39.8%** |
| **2a1 + 2a2** Q projection | 13.28 | 9.8% |
| 3d output | 2.75 | 2.0% |
| 2c absorb | 2.46 | 1.8% |
| ATen `.contiguous()` copies | 1.40 | 1.0% |
| 3b combine | 0.67 | 0.5% |
| 2b RoPE + 2a-norm | 0.19 | 0.1% |
| **TOTAL** | **135.54** | 100% |

**3a + 3c are 85% of decode**, and they are the two steps that never got
`cp.async` or vectorized shared-memory loads — that is where the 0.77× lives.
2a's share tracks `1/Sk` (4.1% at Sk=8192, 31.3% at Sk=512), which is why the
factorization was worth doing first.

---

# 4. Build and Run

Requires an NVIDIA GPU of compute capability 8.0+ (prebuilt for `sm_80`, `sm_89`,
`sm_90`), CUDA 12.x, and conda. Built and verified against **Python 3.9 +
PyTorch 2.3.0 (cu121)**.

```bash
conda create -n ece285 python=3.9 -y
conda activate ece285
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
pip install ninja          # optional, speeds up rebuilds
```

`run_eval.sh` activates the env (override with `MLA_CONDA_ENV` / `CONDA_ROOT`),
rebuilds the extension, and runs the suite:

```bash
bash run_eval.sh                     # all 10 scenarios
bash run_eval.sh --quick             # smoke test (1 scenario)
bash run_eval.sh --phase decode      # the 7 decode scenarios
bash run_eval.sh --phase prefill     # the 3 prefill scenarios
bash run_eval.sh --scenario decode_serving_avg_ctx

MLA_SKIP_BUILD=1 bash run_eval.sh --decode-focus   # skip the rebuild
```

`--decode-focus` is **not** "all decode" — it is a hardcoded 2-scenario tuning
subset. Use `--phase decode` for all seven.

## Repo Layout

| File | Purpose |
|---|---|
| `mla_kernels.cu` | All eight CUDA kernels + pybind11 bindings; carries the full design rationale in comments |
| `setup.py` | Builds the PyTorch CUDA extension |
| `run_eval.sh` | Rebuild + run the benchmark suite |
| `exhaustive_benchmark_suite.py` | PyTorch baseline, fused path, correctness + timing across production shapes |
| `stage_profile.py` | Per-stage timing + roofline |
| `kernel_profile.py` | `torch.profiler` breakdown of kernel 3's sub-stages |
| `check_dtypes.py` | Correctness sweep across fp16/fp32/fp64, plain and split-K paths |
| `check_3d.py` | Correctness sweep over shapes straddling tile boundaries |
| `docs/make_figures.py` | Regenerates every figure above and `docs/mla_figures.pdf` |
