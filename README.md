# Multi-head Latent Attention — Hand-Written CUDA Kernels

Hand-written CUDA implementation of DeepSeek-V3's **Multi-head Latent Attention
(MLA)**, benchmarked end to end against a
PyTorch/cuBLAS reference on an RTX A6000.

```
hidden = 7168     H (heads)  = 128     kv_lora_rank  = 512     q_lora_rank = 1536
nope   = 128      rope       = 64      qk_head_dim   = 192     v_head_dim  = 128
Q_absorbed  = 512 + 64 = 576
```

**Results:** CUDA kernel runs at a **geometric-mean 0.83× of the PyTorch/cuBLAS
baseline**. See [Benchmarks](#3-benchmarks).

## Contents

1. [Why MLA](#1-why-mla)
2. [Algorithm](#2-algorithm)
3. [Benchmarks](#3-benchmarks)
4. [Build and Run](#4-build-and-run)

---

# 1. Why MLA

![Why MLA](docs/fig0_why_mla.png)

Two consequences shape every kernel here:

- **The KV cache is reused across all heads.** The cache is the pair
  `c_kv [B, Sk, 512]` and `pe_cache [B, Sk, 64]`; a *cache row* is one key's
  `c_kv[b, k, :]` (512 latent *ranks*) plus `pe_cache[b, k, :]` (64 rope dims).
  A **rank** is one index along `kv_lora_rank = 512`: the cache never stores K or
  V. Neither tensor has an `H` axis, so all 128 heads of a query score against the
  identical `k` rows — unlike standard MHA. Step 3a exploits this directly: its 
  128-row thread block tile is exactly `H`, so a block's rows are the 128 heads 
  of one query, and each `[128 keys × 32 dims]` slab it stages into shared memory 
  — one step of the 576-deep reduction, taken from `c_kv`'s 512 ranks and then 
  `pe_cache`'s 64 rope dims — is read by all 128 of them.
- **The projection weights are reused across a batch.** `W_uk [H, 128 nope, 512]` and
  `W_uv [H, 128 v_dim, 512]` carry a head axis, which is the same panel for every 
  query in a batch. The reuse is therefore over **`bq` rows**: the `B·Sq` 
  (batch, query-position) pairs. Step 2c gives one head to each thread block
  (`grid.z = h`); a block stages only its own head's slice of `W_uk`, but stages it 
  once for its whole tile of `bq` rows (16–128 rows). So, a block per `(bq, h)` would 
  re-read head `h`'s panel once per row instead. 3d does the same with `W_uv[h]`.
- **Decode is memory-bound.** With `Sq = 1` there is no arithmetic intensity in the attention calculation; the wins come from not moving bytes and from keeping all 84 SMs busy.

---

# 2. Algorithm

![The 9 steps](docs/fig0_pipeline.png)

Shapes throughout are for **batch-128 decode**: `B=128, Sq=1, H=128`, so
`bq = B·Sq = 128` rows and `flat = B·Sq·H = 16,384` rows, at DeepSeek's reported
production-average cache depth `Sk = 4,989`.

### High-Throughput GEMM Recipe

Five kernels across six of the nine steps (2a1, 2a2, 2c, 3a, 3c, 3d) are the same GEMM shape and
use one recipe, so the per-step notes below only list what is *specific* to that
step:

- **Coalescing** - All operand traffic is contiguous and line-aligned. Three
  things enable this: `threadIdx.x` indexes the contiguous axis on the staging loads 
  as well as the stores; each thread's 8 outputs are strided by `blockDim.x`, so one store instruction has the warp's lanes writing *adjacent* floats; and row strides are kept a 
  multiple of 32 floats, so every row starts on a 128 B line (padded for `scores`,
  whose width is `Sk` and not always line-aligned). 
- **Bank-conflict-free shared memory** - Row padding by `+1` rather than an XOR
  swizzle. Padding keeps the address *linear* in the reduction
  index, so addressing is a base pointer plus immediate offsets instead of an XOR
  and an add per operand per step. This reduces the ALU pressure,
  and it frees registers — a swizzle keeps its mask and permuted index live across
  the *entire* reduction, every such value is one the scheduler cannot spend on 
  prefetching the next operand.
- **64x ILP per thread** - `16×16 = 256` threads per block, **8×8 
  outputs per thread**, resulting in `128×128 output tile per thread block`. An M×N register tile costs M+N shared-memory loads per M·N FMAs, 
  so 8×8 gives 0.25 loads/FMA — sm_86 issues 128 FP32 lanes per SM against 32 shared-memory
  words (4 Bytes) per clock, i.e. it wants exactly 4 FMAs per loaded word.
- **Occupancy traded for ILP** - The 8×8 tile requiress 64 accumulator registers, with 
  operands, addressing, bounds and loop state accounting consuming to **107 to 128**
  registers per thread. `__launch_bounds__(256, 2)` caps a thread at
  `65,536 / 512 = 128 registers`, so all kernels fit without spilling.
- **Split-K when the output cannot fill the GPU** - A 128×128 output tile
  means a small output yields few blocks: 2a1's 1536 columns are 12 tiles, and the
  `bq` dimension collapses to a single tile for any `bq_total ≤ 128` — most of
  decode — so the launch would occupy 12 of 84 SMs. Split-K adds a grid dimension
  over the *reduction* instead: each slice takes a disjoint span of it, accumulates
  its own partial, and `atomicAdd`s into a pre-zeroed output.
- **Tiling of contracted dimension** - A block's 128×128 output
  tile is a dot product over 32-deep chunks: all 256 threads cooperatively stage a 
  `[128 × 32]` slab of each operand into shared memory, barrier, reduce that slab into 
  the register tiles, barrier, repeat. 32 is the largest chunk that keeps **2 blocks resident per SM**
  — `(128 + 128) × 33 × 4 B = 33.8 KB` per block, so 67.6 KB of the SM's 100 KB,
  and 32 floats is exactly a 128 B cache line, so each staged row is one aligned transaction.
- **Staging loops strength-reduced to one multiply per element** - Every staging
  loop walks `idx += 256` and splits it with `idx / D` and `idx % D`, for `D` of
  32 (reduction-tile loops) or 128 (column-tile loops). 256 is a multiple of
  both, so `idx % D` never changes across passes: that index and the
  `< tile_len` predicate built on it are loop-invariant and hoist out of the loop
  entirely, the *other* index then advances by a constant so its bound becomes a
  trip count rather than a per-element compare, and every block-invariant offset
  folds into a base pointer once. What is left in the body is one row multiply,
  one load and one store — against a shift, a mask, two compares, a source select
  and a runtime-stride `IMAD` before. Worth **0.77× → 0.83×** end to end. Note
  what it is *not*: register pressure barely moved, and unrolling the reduction
  strip (which does cut registers, 127 → 110 on 3a) buys nothing. The strip is
  already at the shared-memory bandwidth ceiling, so rescheduling the same work
  cannot help — only issuing less of it.
- **Bounds check during results write-out** - A block always computes its whole 
  128×128 tile, even when the real matrix has fewer rows or columns than that. 
  The threads holding the leftover slots read whatever happens to be in shared 
  memory, do their multiply-adds anyway, and the write-out step at the end simply 
  skips them. The price is that a tile bigger than the data does real work for 
  nothing, which is what the adaptive `bq` tile below avoids.
- **Adaptive `bq` tile** - Kernels with row axis `bq` pick their tile height at
  launch. The thread block is always 16×16 threads and `threadIdx.y` walks rows, so 
  if each of those 16 columns owns `kBqReg` rows the tile is `16 × kBqReg` = 16, 32, 
  64 or 128 rows, and a thread's accumulator block is `kBqReg × 8`. A single-stream decode has exactly one real row, so a 128-row tile would do 128 rows of work to keep 1. `bq_reg_for()` therefore picks the **smallest tile that still covers `bq_total`**: 16 at `bq = 1` (the floor, since `blockDim.y = 16`), 32 at `bq = 20`, etc., freeing up registers to increase occupancy.

---

## Step 2a1 — Project the Hidden State into the Latent

![Step 2a1](docs/step_2a1.png)

**Techniques Specific to This Step**

- **Low-rank factorization (algorithmic).** This step exists only because the Q
  projection is factored through `q_lora_rank = 1536` rather than applied as one
  dense `[7168, 24576]` matrix in the next step.
- **Split-K over the 7,168 reduction.** The output is only 1536 columns = 12
  block tiles, so without Split-K this step runs 12 blocks on an 84-SM GPU.

## Step 2a-norm — RMSNorm on the Latent

![Step 2a-norm](docs/step_2an.png)

**Techniques Specific to This Step**

- **Normalization to break matrices (algorithmic).** Matmul is associative, so with
  nothing between them `(x @ W_q_a) @ W_q_b` equals `x @ (W_q_a @ W_q_b)` — one
  precomputable `[7168, 24576]` matrix. What you would lose is *meaning*: the pair 
  would carry no more expressive power than a single rank-1536 linear layer, and the
  latent between them would have no fixed identity, since for any invertible
  `G [1536, 1536]` the pair `(W_q_a G, G⁻¹ W_q_b)` computes the identical function.
  RMSNorm removes that freedom: its scale is `rsqrt(mean(q_lat²))`, taken from the
  row's own values, so it acts in one specific basis and cannot be absorbed into
  either weight. The learnable `gain` *is* linear and could be folded into `W_q_b`;
  only the data-dependent factor does this work. 
- **Single-kernel block reduction.** One block owns a whole row — 1,536 values
  across 256 threads, 6 each — so the RMS never leaves the block. Each thread sums
  its own squares, `__shfl_down` collapses each warp of 32 down to one number
  (8 warps, so 8 partial sums), those 8 go through shared memory, and warp 0 runs
  one more shuffle round over them to finish the total. Two barriers in all, no
  shared-memory reduction tree, no second kernel launch, and the mean never
  round-trips through global memory.

## Step 2a2 — Expand the Latent into the Per-Head Query

![Step 2a2](docs/step_2a2.png)

**Techniques Specific to This Step**

- **Split-K sized against wave quantization.** 24576/128 = 192 blocks against
  84 SMs × 2 resident = 168 slots is **1.14 waves**: one full wave, then a tail
  wave leaving 144 slots idle — 43% of the machine. The planner targets ~8 waves,
  where the tail costs ~11%.
- **Blocking over `bq`.** A block owns 128 output columns
  for *all* 128 `bq` rows, so the grid is 192 column tiles by one row tile. These
  192 blocks all read the same `q_lat`, which stays in L2, and take disjoint column
  panels of `W_q_b`, so the weight is fetched only once across the grid.

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
- **The merged reduction costs nothing at the staging loop.** Because both
  operand pairs are contiguous in `r` and the rope/nope choice is uniform across
  the block, the shared recipe's base-pointer folding resolves that two-way
  source select *once per reduction tile* rather than per staged element. Both
  of this kernel's staging loops also split by the same 32, so they share one
  invariant column index and one hoisted predicate. 3a ends at 107 registers.

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
| `decode_single_user_long_ctx` | decode | 1 | 1 | 65536 | 1.91 | 2.39 | 0.80× |
| `decode_serving_avg_ctx` | decode | 128 | 1 | 4989 | 14.27 | 16.92 | 0.84× |
| `decode_serving_long_ctx` | decode | 128 | 1 | 8192 | 23.61 | 26.55 | 0.89× |
| `decode_serving_high_batch` | decode | 256 | 1 | 2048 | 12.91 | 15.06 | 0.86× |
| `decode_speculative_2tok` | decode | 64 | 2 | 512 | 2.59 | 3.38 | 0.77× |
| `decode_speculative_4tok` | decode | 64 | 4 | 768 | 6.16 | 7.53 | 0.82× |
| `decode_speculative_long_ctx` | decode | 132 | 4 | 4096 | 47.19 | 54.90 | 0.86× |
| `prefill_chat_batch` | prefill | 8 | 512 | 512 | 70.55 | 87.32 | 0.81× |
| `prefill_chunk_2k` | prefill | 1 | 2048 | 2048 | 94.15 | 113.11 | 0.83× |
| `prefill_chunk_4k` | prefill | 1 | 4096 | 4096 | 355.46 | 417.61 | 0.85× |

**Average: 0.83× geometric mean** (0.83× arithmetic, 0.84× aggregate
Σbase/Σfused); 0.83× for the seven decode shapes and for the three prefill shapes
alike.

Absolute milliseconds on this box drift by up to ~15% with GPU thermal state,
and both columns drift together — so the ratios are the stable quantity and the
table is reported from a single warm run rather than a best-of. The **first**
scenario of a cold run is not usable: the cuBLAS side is still on a ramping clock
and reads ~50% high, which is why the suite is run twice and the second run
taken.

## Where the Time Goes

`torch.profiler`, summed over the seven decode scenarios:

| step | sum ms | share |
|---|--:|--:|
| **3a** scores | 55.73 | **44.5%** |
| **3c** ctx | 50.52 | **40.3%** |
| **2a1 + 2a2** Q projection | 12.58 | 10.0% |
| 3d output | 2.49 | 2.0% |
| 2c absorb | 2.17 | 1.7% |
| ATen `.contiguous()` copies and fills | 1.57 | 1.3% |
| 2b RoPE + 2a-norm | 0.19 | 0.2% |
| 3b combine | 0.12 | 0.1% |
| **TOTAL** | **125.36** | 100% |

**3a + 3c are 85% of decode.** Both are limited by shared-memory bandwidth, not
by occupancy or instruction scheduling: at 0.25 loads/FMA the 8×8 register tile
asks for 32 shared-memory words per clock against the 32 that sm_86 can deliver,
so the reduction strip has no headroom to schedule into. Measured against that
model — unrolling the strip by 2, 4 or 8 compiles without spilling and drops 3a
from 127 to 110 registers, yet moves the benchmark by less than 1%, because
extra ILP has nothing to hide behind. What *did* move it was removing issued
work from the staging loops (see the GEMM recipe): measured as a matched
before/after pair on one GPU in one session, that is 0.77× → 0.83× geometric
mean, −7.3% fused time averaged over the ten scenarios and faster on every one
of them. Applied to 3a and 3c first (−5.7%), then 2a1/2a2 and 2c (−1.3%), then
3d (−0.5%) — the gains track each step's share of the total, and land hardest on
the shapes where a narrow `bq` tile leaves the reduction strip with the least
work to hide the staging behind.

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
