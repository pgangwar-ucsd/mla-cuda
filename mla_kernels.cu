#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <limits>

// DeepSeek-V3 reference dims: hidden=7168, H=128, kv_rank=512, nope=128,
// rope=64, v=128, qk_head=192. B, Sq, Sk vary per launch.

// All five tiled kernels share one shape: 8×8 outputs per thread on a 16×16 block,
// reducing in SMEM tiles of 32. Three properties matter, each paid for in measurement:
//
//  1. threadIdx.x indexes the *contiguous* output dimension. A warp is 32 consecutive
//     linear threads = two threadIdx.y rows of 16, so it stores two runs of 8 adjacent
//     elements per row — fully-used sectors. Driving the row index from threadIdx.x
//     instead scatters a warp across 16 rows: 23.99 store sectors/request versus 4.00.
//
//  2. 8×8 rather than 4×4. An M×N register tile needs M+N SMEM loads per M×N FMAs, so
//     8×8 is 0.25 loads/FMA against 4×4's 0.50. This GPU wants 4 FMAs per loaded word
//     (128 FP32 lanes/SM/clk against 32 SMEM words/clk), which 8×8 hits exactly. It
//     costs ~128 registers, hence 256 threads and 2 blocks/SM (33% occupancy) — the
//     halved l1tex traffic still beats the halved occupancy. Measured -19% on 3a.
//
//  3. The inner loop must NOT be unrolled. 64 FMAs per iteration is already ample ILP;
//     `#pragma unroll 8` on top of it asks for hundreds of live values and spills.
//     A 312-byte spill made 3a 6x slower, and even a 32-byte spill cost 54%. Any spill
//     loses — take the occupancy hit instead. That is also why blocks/SM is 2 and not
//     3: 3 caps registers at 65536/(256*3) = 85, below what 64 accumulators need.
//
// SMEM rows are padded by one element rather than XOR-swizzled. Both avoid bank
// conflicts, but padding keeps the address *linear* in the reduction index, so the
// loop becomes a base pointer plus immediate offsets instead of an XOR and an add per
// operand per step. That took the ALU pipe from 45% to 18% of peak (it had been
// running hotter than the FMA pipe it feeds) and was worth -21% across every kernel.
// Row stride PAD = TILE + 1; since 33 = 1 (mod 32) a warp reading row r at column c
// hits bank (r + c) % 32 — all 32 distinct. Only the row stride changes: staging loop
// extents stay TILE, with a short trailing tile handled by predicating the store.
//
//   kernel                output tiles                        reduction tile
//   2a q_raw_gemm      bq tile    x kQWqColTile   (128 cols)   kQHiddenTile  (hidden,
//                                                                then q_lora)
//   2c q_absorb        bq tile    x kQRankTile    (128 ranks)  kQNopeTile    (nope)
//   3a mla_scores      kScoreFlatTile x kScoreSkTile           kScoreRankTile(kv_rank
//                                                                + rope, merged)
//   3c mla_ctx         kCtxFlatTile   x kCtxRankTile           kCtxSkTile    (Sk)
//   3d mla_output      bq tile    x kOutVTile     (128 v)      kOutRankTile  (kv_rank)
//
// The reduction tile doubles as the SMEM row width, so it is the constant every
// staging extent and row stride must use.
static constexpr int kTileThreadsX   = 16;  // 16×8 = 128 along the contiguous output dim
static constexpr int kTileThreadsY   = 16;  // 16×8 = 128 along the row dim
static constexpr int kTileReg        = 8;   // 8×8 outputs per thread
static constexpr int kTileThreads    = kTileThreadsX * kTileThreadsY;  // 256
static constexpr int kTileBlocksPerSM = 2;  // 3 caps regs at 85 and spills
static constexpr int kTileRed        = 32;  // reduction tile == SMEM row width
static constexpr int kTileRedPad     = kTileRed + 1;

// --- 3a / 3c ---------------------------------------------------------------
static constexpr int kScoreFlatTile  = kTileThreadsY * kTileReg;  // 128 flats
static constexpr int kScoreSkTile    = kTileThreadsX * kTileReg;  // 128 keys out
static constexpr int kScoreRankTile  = kTileRed;                  // reduce over kv_rank
static constexpr int kScoreRankPad   = kTileRedPad;
static constexpr int kCtxFlatTile    = kTileThreadsY * kTileReg;  // 128 flats
static constexpr int kCtxRankTile    = kTileThreadsX * kTileReg;  // 128 kv_rank out
static constexpr int kCtxSkTile      = kTileRed;                  // reduce over Sk
static constexpr int kCtxSkPad       = kTileRedPad;

// --- Kernel 2 (Q-prep) ------------------------------------------------------
// Blocking over bq (not one block per (b,s,h)) is what keeps W_q traffic bounded: a
// (b,s,h)-per-block layout re-reads head h's 5.5 MB W_q panel once per (b,s), i.e.
// 90 GB at B=128. kBqReg is chosen per launch — see bq_reg_for().
static constexpr int kQWqColTile  = kTileThreadsX * kTileReg;  // 128 Q-projection cols out
static constexpr int kQRankTile   = kTileThreadsX * kTileReg;  // 128 kv_rank out
static constexpr int kQHiddenTile = kTileRed;                  // reduce over hidden / q_lora
static constexpr int kQHiddenPad  = kTileRedPad;
static constexpr int kQNopeTile   = kTileRed;                  // reduce over nope_dim
static constexpr int kQNopePad    = kTileRedPad;
// 2a split-K target: how many waves of blocks the grid should cover before the tail wave
// stops mattering. A W-wave grid loses at most 1/(W+1) of the machine to quantisation,
// against the 43% that the un-split 1.14 waves loses. 8 also puts the common 192-block
// case on exactly 8.0 waves; measured 2% faster on 2a than a target of 6.
static constexpr int kQSplitTargetWaves = 8;
static constexpr int kRopeThreads = 256;
// 2a-norm: one block per bq row over the q_lora latent (1536), so a single block
// reduction covers the row and 256 threads is 6 elements each.
static constexpr int kNormThreads = 256;

// --- 3d ---------------------------------------------------------------------
static constexpr int kOutVTile    = kTileThreadsX * kTileReg;  // 128 v_dim out
static constexpr int kOutRankTile = kTileRed;                  // reduce over kv_rank
static constexpr int kOutRankPad  = kTileRedPad;

// Widest bq register depth whose tile does not overshoot bq_total by more than one
// step. A wider tile raises arithmetic intensity *and* cuts passes over the streamed
// weight (W_q_b is 151 MB, re-read once per bq tile), but at bq_total=1 a 128-row tile
// would discard 127/128 of its FMAs — the bounds are only checked at the store.
static inline int bq_reg_for(int bq_total) {
    if (bq_total <= 1 * kTileThreadsY) return 1;
    if (bq_total <= 2 * kTileThreadsY) return 2;
    if (bq_total <= 4 * kTileThreadsY) return 4;
    return kTileReg;
}

// Precision stage 3 stages and accumulates in. fp64 in means fp64 all the way through;
// fp32 stays fp32; fp16 promotes to fp32, because summing a 512- (kv_rank) or 65536-long
// (Sk) reduction in half would throw away far more than the staging saves.
template <typename T> struct AccType         { using type = float;  };
template <>           struct AccType<double> { using type = double; };
template <typename T> using acc_t_of = typename AccType<T>::type;

// A function template cannot declare `extern __shared__ acc_t[]` directly: every
// instantiation would redeclare the one dynamic-SMEM symbol with a different type.
// Declare it as raw bytes once and reinterpret, which is the standard workaround.
template <typename T>
__device__ __forceinline__ T* dynamic_smem() {
    extern __shared__ __align__(alignof(double)) unsigned char mla_smem_raw[];
    return reinterpret_cast<T*>(mla_smem_raw);
}

// Dynamic SMEM above 48 KB (which fp64 tiles need) is opt-in per kernel. The status
// is checked: silently ignoring it turns "tile does not fit this device" into an
// "invalid argument" at launch time, several frames away from the cause.
template <typename KernelFn>
static void enable_large_smem(KernelFn kernel, size_t smem) {
    if (smem <= 48u * 1024u) return;
    const cudaError_t st =
        cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel),
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem));
    if (st != cudaSuccess) {
        cudaGetLastError();  // clear so the next real launch is not blamed for this
        TORCH_CHECK(false, "MLA: kernel needs ", smem,
                    " B of dynamic shared memory, which exceeds this device's "
                    "per-block limit (", cudaGetErrorString(st), ")");
    }
}

// SM count of the active device, cached — the 3c split-K heuristic needs it on every
// launch and cudaDeviceGetAttribute is a driver round-trip.
static int sm_count() {
    constexpr int kMaxDevices = 16;
    static int cached[kMaxDevices] = {0};
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= kMaxDevices) return 84;  // fall back to an A6000-ish default
    if (cached[dev] == 0)
        cudaDeviceGetAttribute(&cached[dev], cudaDevAttrMultiProcessorCount, dev);
    return cached[dev];
}

// Split-K plan for one q_raw_gemm launch. Both Q projections use the same kernel, but
// they have very different grids — [bq, 7168] @ [7168, 1536] is only 12 column tiles
// while [bq, 1536] @ [1536, 24576] is 192 — so each needs its own plan.
struct QGemmSplit {
    int  split;    // grid.z slices over the reduction
    int  k_chunk;  // reduction elements per slice (a whole number of SMEM tiles)
    bool atomic;   // > 1 slice, so the epilogue accumulates into a pre-zeroed output
};

static QGemmSplit plan_q_gemm_split(int bq_total, int cols, int k_total) {
    const int bq_tile  = bq_reg_for(bq_total) * kTileThreadsY;
    const int n_tiles  = (cols     + kQWqColTile  - 1) / kQWqColTile;
    const int bq_tiles = (bq_total + bq_tile      - 1) / bq_tile;
    const int k_tiles  = (k_total  + kQHiddenTile - 1) / kQHiddenTile;

    int split = (kQSplitTargetWaves * sm_count() * kTileBlocksPerSM)
              / (n_tiles * bq_tiles);
    if (split < 1)       split = 1;
    if (split > k_tiles) split = k_tiles;

    // Give each slice a whole number of SMEM tiles, then recompute the split from that so
    // no slice ends up empty (a block on nothing) and only the last one can be ragged —
    // keeping every other slice on the unrolled full-tile path.
    const int chunk_tiles = (k_tiles + split - 1) / split;
    split = (k_tiles + chunk_tiles - 1) / chunk_tiles;
    return QGemmSplit{split, chunk_tiles * kQHiddenTile, split > 1};
}


// ============================================================================
// KERNEL 2: Q absorbed path — 2a1 → 2a-norm → 2a2 → 2b (RoPE) → 2c (absorb @ W_uk)
//
//  2a1. q_lat = x     @ W_q_a     [B*Sq, hidden] @ [hidden, 1536]  → [B*Sq, 1536]
//  2a-n. RMSNorm over the q_lora latent           → q_lat      [B*Sq, 1536]
//  2a2. q_raw = q_lat @ W_q_b     [B*Sq, 1536]   @ [1536, H*192]  → [B*Sq, H*192]
//  2b.  RoPE on the rope half of q_raw            → Q_rope     [B,Sq,H,64]
//  2c.  absorb the nope half against W_uk[h]      → Q_absorbed [B,Sq,H,512]
//
// DeepSeek-V3 factorises the Q projection as wq_b(q_norm(wq_a(x))) through a
// q_lora_rank=1536 latent rather than applying one dense [7168, 24576] matrix. That is
// 48.7M weights instead of 176M — 195 MB against 704 MB in fp32 — which dominates
// single-stream decode, where 2a is a GEMV and reads every weight exactly once.
//

// --- 2a. out = in @ W: kBqReg×8 tile/thread; grid (col tiles, bq tiles, split) ---
//   Serves both halves of the factorised Q projection — 2a1 is [bq, 7168] @ [7168, 1536]
//   and 2a2 is [bq, 1536] @ [1536, 24576]. Only the shapes and the split-K plan differ,
//   so `hidden_dim` below is whichever reduction extent the call is contracting.
//   grid.x: n ∈ [0, n_total)         — 128 columns per block (contiguous in the output)
//   grid.y: bq = b*Sq + s            — kTileThreadsY*kBqReg rows per block
//   grid.z: split-K slice over the reduction — see below
//   block: 16×16 → each thread 8 cols × kBqReg bq; reduction in SMEM tiles of 32
//   SMEM [lb][k] / [ln][k], rows padded to kQHiddenPad; k staged fast so the HBM
//   reads coalesce
//
//   kBqReg is the only knob, picked by bq_reg_for(): 8 gives the full 8×8 shape at
//   0.25 SMEM loads per FMA. Single-stream decode has bq_total == 1, where a 128-row
//   tile would spend 127/128 of its FMAs on rows the epilogue discards — the bounds
//   are only checked at the store — so it drops to a 16-row tile.
//
//   Split-K: the column count is fixed by the model shape (2a2 is pinned at
//   H*qk_head_dim/128 = 192 tiles, 2a1 at just 12), and grid.y collapses to 1 for every
//   bq_total <= 128 — which is most of decode. 192 blocks against 84 SMs x 2 resident is
//   1.14 waves: one full wave, then a tail wave holding 24 blocks with 144 slots idle.
//   ncu confirms it (waves_per_multiprocessor 1.14, sm__throughput 47% elapsed-normalised
//   against l1tex 66% active-normalised — the gap is exactly the idle tail). grid.z chops
//   the reduction into `k_chunk`-sized slices; each accumulates a partial and atomically
//   adds it into the output, which the host pre-zeroes. Slices are disjoint spans of the
//   reduction, so no traffic is duplicated — only the epilogue is paid `split` times. The
//   host sets gridDim.z == 1 once the output dims alone fill the GPU, and `use_atomic`
//   then selects plain stores so that path pays neither the memset nor the atomics.
template <typename scalar_t, int kBqReg>
__global__ void __launch_bounds__(kTileThreads, kTileBlocksPerSM)
q_raw_gemm_kernel(
    const scalar_t* __restrict__ x,      // [B*Sq, hidden_dim]
    const scalar_t* __restrict__ W_q,    // [hidden_dim, n_total]
    acc_t_of<scalar_t>* __restrict__ q_raw,  // [B*Sq, n_total]
    int bq_total,
    int hidden_dim,  // reduction extent: 7168 for 2a1, q_lora for 2a2
    int n_total,
    int k_chunk,    // reduction elements per grid.z slice (multiple of kQHiddenTile)
    bool use_atomic)
{
    using acc_t = acc_t_of<scalar_t>;
    constexpr int kBqTile = kTileThreadsY * kBqReg;  // bq rows this block owns

    const int n_base  = blockIdx.x * kQWqColTile;
    const int bq_base = blockIdx.y * kBqTile;

    // this block's slice of the hidden reduction; k_chunk == hidden when gridDim.z == 1
    const int k_begin = blockIdx.z * k_chunk;
    const int k_end   = min(k_begin + k_chunk, hidden_dim);

    acc_t* smem   = dynamic_smem<acc_t>();
    acc_t* x_smem = smem;                            // [kBqTile][kQHiddenPad]
    acc_t* w_smem = x_smem + kBqTile * kQHiddenPad;  // [kQWqColTile][kQHiddenPad]

    const int coop_stride = blockDim.x * blockDim.y;

    acc_t s[kBqReg][kTileReg];
    #pragma unroll
    for (int db = 0; db < kBqReg; ++db)
        #pragma unroll
        for (int dn = 0; dn < kTileReg; ++dn)
            s[db][dn] = acc_t(0);

    // threadIdx.y → bq (the row), threadIdx.x → n (contiguous in q_raw), so a warp
    // stores two runs of 8 adjacent q_raw columns rather than scattering across rows.
    // SMEM row index bases, not arrays — see the note in 3a.
    const int lb0 = threadIdx.y * kBqReg;
    const int ln0 = threadIdx.x;

    for (int k0 = k_begin; k0 < k_end; k0 += kQHiddenTile) {
        const int tile_len =
            (k0 + kQHiddenTile <= k_end) ? kQHiddenTile : (k_end - k0);

        // stage x[bq_base:+kBqTile, k0:+tile) — k contiguous so warps coalesce.
        // Full-tile extent, not tile_len, so the div/mod fold to shift and mask
        // rather than a runtime integer division; a short trailing tile is handled by
        // predicating the store.
        const int x_elems = kBqTile * kQHiddenTile;
        for (int idx = threadIdx.x + threadIdx.y * blockDim.x;
             idx < x_elems; idx += coop_stride) {
            const int lb_s    = idx / kQHiddenTile;
            const int k_local = idx % kQHiddenTile;
            const int bq_s    = bq_base + lb_s;
            if (bq_s < bq_total && k_local < tile_len)
                x_smem[lb_s * kQHiddenPad + k_local] =
                    static_cast<acc_t>(x[bq_s * hidden_dim + k0 + k_local]);
        }

        // stage W_q[k0:+tile, n_base:+128) — n contiguous → ln_s = idx % kQWqColTile
        const int w_elems = tile_len * kQWqColTile;
        for (int idx = threadIdx.x + threadIdx.y * blockDim.x;
             idx < w_elems; idx += coop_stride) {
            const int k_local = idx / kQWqColTile;
            const int ln_s    = idx % kQWqColTile;
            const int n_s     = n_base + ln_s;
            if (k_local < tile_len && n_s < n_total)
                w_smem[ln_s * kQHiddenPad + k_local] =
                    static_cast<acc_t>(W_q[(k0 + k_local) * n_total + n_s]);
        }
        __syncthreads();

        // k-outer register strip. Lanes past bq_total / n_total read unwritten SMEM;
        // their accumulators are dropped at the store, so no inner predicate.
        if (tile_len == kQHiddenTile) {
            #pragma unroll 1
            for (int k_local = 0; k_local < kQHiddenTile; ++k_local) {
                acc_t x_reg[kBqReg], w_reg[kTileReg];
                #pragma unroll
                for (int db = 0; db < kBqReg; ++db)
                    x_reg[db] = x_smem[(lb0 + db) * kQHiddenPad + k_local];
                #pragma unroll
                for (int dn = 0; dn < kTileReg; ++dn)
                    w_reg[dn] = w_smem[(ln0 + dn * kTileThreadsX) * kQHiddenPad + k_local];
                #pragma unroll
                for (int db = 0; db < kBqReg; ++db)
                    #pragma unroll
                    for (int dn = 0; dn < kTileReg; ++dn)
                        s[db][dn] += x_reg[db] * w_reg[dn];
            }
        } else {
            // ragged final hidden tile: same body, runtime trip count
            for (int k_local = 0; k_local < tile_len; ++k_local) {
                acc_t x_reg[kBqReg], w_reg[kTileReg];
                #pragma unroll
                for (int db = 0; db < kBqReg; ++db)
                    x_reg[db] = x_smem[(lb0 + db) * kQHiddenPad + k_local];
                #pragma unroll
                for (int dn = 0; dn < kTileReg; ++dn)
                    w_reg[dn] = w_smem[(ln0 + dn * kTileThreadsX) * kQHiddenPad + k_local];
                #pragma unroll
                for (int db = 0; db < kBqReg; ++db)
                    #pragma unroll
                    for (int dn = 0; dn < kTileReg; ++dn)
                        s[db][dn] += x_reg[db] * w_reg[dn];
            }
        }
        __syncthreads();  // all reads done before the next tile overwrites SMEM
    }

    // Each grid.z slice owns a disjoint span of hidden, so the partials are a plain sum.
    // Lanes of a warp differ in `n`, so these hit consecutive addresses; the only
    // contention is between the gridDim.z blocks sharing an output tile, and each of
    // them contributes exactly one atomic per element after its whole reduction.
    #pragma unroll
    for (int db = 0; db < kBqReg; ++db) {
        const int bq_i = bq_base + (lb0 + db);
        if (bq_i >= bq_total) continue;
        const int row_base = bq_i * n_total;
        #pragma unroll
        for (int dn = 0; dn < kTileReg; ++dn) {
            const int n_i = n_base + (ln0 + dn * kTileThreadsX);
            if (n_i >= n_total) continue;
            if (use_atomic) atomicAdd(&q_raw[row_base + n_i], s[db][dn]);
            else            q_raw[row_base + n_i] = s[db][dn];
        }
    }
}

// --- 2a-norm. RMSNorm over the q_lora latent, between the two Q projections ---
// out[i] = in[i] * rsqrt(mean(in^2) + eps) * gain[i], one block per bq row.
//
// The norm is what makes the factorisation more than a low-rank matrix: with nothing
// between them, W_q_a @ W_q_b collapses into a single rank-1536 [7168, 24576] matrix.
//
// Reads 2a1's acc_t output and writes scalar_t, which is what lets 2a2 reuse
// q_raw_gemm_kernel unchanged — that kernel takes a scalar_t left operand. Narrowing
// here matches the existing policy for q_absorbed: accumulate wide, hand off narrow.
template <typename scalar_t>
__global__ void __launch_bounds__(kNormThreads)
q_lat_norm_kernel(
    const acc_t_of<scalar_t>* __restrict__ q_lat_acc,  // [B*Sq, q_lora]
    const scalar_t* __restrict__ gain,                 // [q_lora]
    scalar_t* __restrict__ q_lat,                      // [B*Sq, q_lora]
    int q_lora,
    acc_t_of<scalar_t> eps)
{
    using acc_t = acc_t_of<scalar_t>;
    const size_t row = static_cast<size_t>(blockIdx.x) * q_lora;

    acc_t ss = acc_t(0);
    for (int i = threadIdx.x; i < q_lora; i += blockDim.x) {
        const acc_t v = q_lat_acc[row + i];
        ss += v * v;
    }

    // block reduction: warp shuffles, then one pass over the warp partials
    #pragma unroll
    for (int off = warpSize / 2; off > 0; off >>= 1)
        ss += __shfl_down_sync(0xffffffffu, ss, off);

    __shared__ acc_t warp_ss[32];  // <= 1024 threads → <= 32 warps
    const int lane  = threadIdx.x & (warpSize - 1);
    const int warp  = threadIdx.x / warpSize;
    const int warps = (blockDim.x + warpSize - 1) / warpSize;
    if (lane == 0) warp_ss[warp] = ss;
    __syncthreads();

    if (warp == 0) {
        ss = (lane < warps) ? warp_ss[lane] : acc_t(0);
        #pragma unroll
        for (int off = warpSize / 2; off > 0; off >>= 1)
            ss += __shfl_down_sync(0xffffffffu, ss, off);
        if (lane == 0) warp_ss[0] = rsqrt(ss / acc_t(q_lora) + eps);
    }
    __syncthreads();

    const acc_t inv_rms = warp_ss[0];
    for (int i = threadIdx.x; i < q_lora; i += blockDim.x)
        q_lat[row + i] = static_cast<scalar_t>(
            q_lat_acc[row + i] * inv_rms * static_cast<acc_t>(gain[i]));
}

// --- 2b. RoPE on the rope half of q_raw → Q_rope [B, Sq, H, rope_dim] ---
// One thread per (bq, h, adjacent pair).
// `positions[s]` is the *absolute* position of query s, supplied by the caller.
template <typename scalar_t>
__global__ void __launch_bounds__(kRopeThreads)
q_rope_kernel(
    const acc_t_of<scalar_t>* __restrict__ q_raw,  // [B*Sq, H*qk_head_dim]
    const int64_t* __restrict__ positions,         // [sq] absolute query positions
    scalar_t* __restrict__ q_rope,                 // [B, Sq, H, rope_dim]
    int bq_total,
    int sq,
    int num_heads,
    int nope_dim,
    int rope_dim,
    int qk_head_dim)
{
    using acc_t = acc_t_of<scalar_t>;

    const int halves = rope_dim / 2;
    const int total  = bq_total * num_heads * halves;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += gridDim.x * blockDim.x) {
        const int half  = idx % halves;
        const int rest  = idx / halves;
        const int h     = rest % num_heads;
        const int bq    = rest / num_heads;
        const int s     = bq % sq;          // index within the sequence
        const int d     = half * 2;

        const int src = bq * (num_heads * qk_head_dim) + h * qk_head_dim + nope_dim + d;
        const acc_t even = q_raw[src];
        const acc_t odd  = q_raw[src + 1];

        const acc_t inv_freq = acc_t(1) / pow(acc_t(10000),
                                   static_cast<acc_t>(d) / static_cast<acc_t>(rope_dim));
        const acc_t angle = static_cast<acc_t>(positions[s]) * inv_freq;
        const acc_t c  = cos(angle);
        const acc_t sn = sin(angle);

        const int dst = (bq * num_heads + h) * rope_dim + d;
        q_rope[dst]     = static_cast<scalar_t>(even * c  - odd * sn);
        q_rope[dst + 1] = static_cast<scalar_t>(even * sn + odd * c);
    }
}

// --- 2c. Q_absorbed = q_nope @ W_uk[h]: kBqReg×8 tile/thread; grid (bq, r, head) ---
//   One head per block-column so W_uk[h] is staged once and amortised over the whole
//   bq tile. The nope half is read straight out of q_raw at stride qk_head_dim.
//   grid.x: bq tiles   grid.y: r tiles (128 ranks/block)   grid.z: h
//   block: 16×16 → each thread 8 ranks × kBqReg bq; nope in SMEM tiles of 32
//   kBqReg follows bq_reg_for(), same as 2a
template <typename scalar_t, int kBqReg>
__global__ void __launch_bounds__(kTileThreads, kTileBlocksPerSM)
q_absorb_kernel(
    const acc_t_of<scalar_t>* __restrict__ q_raw,  // [B*Sq, H*qk_head_dim]
    const scalar_t* __restrict__ W_uk,             // [H, nope_dim, kv_rank]
    scalar_t* __restrict__ q_absorbed,             // [B, Sq, H, kv_rank]
    int bq_total,
    int num_heads,
    int nope_dim,
    int kv_rank,
    int qk_head_dim)
{
    using acc_t = acc_t_of<scalar_t>;

    constexpr int kBqTile = kTileThreadsY * kBqReg;  // bq rows this block owns

    const int bq_base  = blockIdx.x * kBqTile;
    const int r_base   = blockIdx.y * kQRankTile;
    const int head_idx = blockIdx.z;

    acc_t* smem    = dynamic_smem<acc_t>();
    acc_t* qn_smem = smem;                           // [kBqTile][kQNopeTile]
    acc_t* w_smem  = qn_smem + kBqTile * kQNopePad;  // [kQRankTile][kQNopePad]

    const int qn_head_base = head_idx * qk_head_dim;        // nope half starts at +0
    const int w_head_base  = head_idx * nope_dim * kv_rank;
    const int coop_stride  = blockDim.x * blockDim.y;

    acc_t s[kBqReg][kTileReg];
    #pragma unroll
    for (int db = 0; db < kBqReg; ++db)
        #pragma unroll
        for (int dr = 0; dr < kTileReg; ++dr)
            s[db][dr] = acc_t(0);

    // Only the SMEM row indices live across the reduction (see 2a / 3a).
    // SMEM row index bases, not arrays — see the note in 3a.
    const int lb0 = threadIdx.y * kBqReg;
    const int lr0 = threadIdx.x;


    for (int n0 = 0; n0 < nope_dim; n0 += kQNopeTile) {
        const int tile_len =
            (n0 + kQNopeTile <= nope_dim) ? kQNopeTile : (nope_dim - n0);

        // stage q_raw[bq, h, n0:+tile) — n contiguous so warps coalesce.
        // Full-tile extent, not tile_len, so the div/mod fold to shift and mask
        // rather than a runtime integer division; a short trailing tile is handled by
        // predicating the store. 
        const int qn_elems = kBqTile * kQNopeTile;
        for (int idx = threadIdx.x + threadIdx.y * blockDim.x;
             idx < qn_elems; idx += coop_stride) {
            const int lb_s    = idx / kQNopeTile;
            const int n_local = idx % kQNopeTile;
            const int bq_s    = bq_base + lb_s;
            if (bq_s < bq_total && n_local < tile_len)
                qn_smem[lb_s * kQNopePad + n_local] =
                    q_raw[bq_s * (num_heads * qk_head_dim) + qn_head_base + n0 + n_local];
        }

        // stage W_uk[h, n0:+tile, r_base:+128) — r contiguous → lr_s = idx % kQRankTile
        const int w_elems = tile_len * kQRankTile;
        for (int idx = threadIdx.x + threadIdx.y * blockDim.x;
             idx < w_elems; idx += coop_stride) {
            const int n_local = idx / kQRankTile;
            const int lr_s    = idx % kQRankTile;
            const int r_s     = r_base + lr_s;
            if (n_local < tile_len && r_s < kv_rank)
                w_smem[lr_s * kQNopePad + n_local] =
                    static_cast<acc_t>(
                        W_uk[w_head_base + (n0 + n_local) * kv_rank + r_s]);
        }
        __syncthreads();

        if (tile_len == kQNopeTile) {
            #pragma unroll 1
            for (int n_local = 0; n_local < kQNopeTile; ++n_local) {
                acc_t qn_reg[kBqReg], w_reg[kTileReg];
                #pragma unroll
                for (int db = 0; db < kBqReg; ++db)
                    qn_reg[db] = qn_smem[(lb0 + db) * kQNopePad + n_local];
                #pragma unroll
                for (int dr = 0; dr < kTileReg; ++dr)
                    w_reg[dr] = w_smem[(lr0 + dr * kTileThreadsX) * kQNopePad + n_local];
                #pragma unroll
                for (int db = 0; db < kBqReg; ++db)
                    #pragma unroll
                    for (int dr = 0; dr < kTileReg; ++dr)
                        s[db][dr] += qn_reg[db] * w_reg[dr];
            }
        } else {
            // ragged final nope tile: same body, runtime trip count
            for (int n_local = 0; n_local < tile_len; ++n_local) {
                acc_t qn_reg[kBqReg], w_reg[kTileReg];
                #pragma unroll
                for (int db = 0; db < kBqReg; ++db)
                    qn_reg[db] = qn_smem[(lb0 + db) * kQNopePad + n_local];
                #pragma unroll
                for (int dr = 0; dr < kTileReg; ++dr)
                    w_reg[dr] = w_smem[(lr0 + dr * kTileThreadsX) * kQNopePad + n_local];
                #pragma unroll
                for (int db = 0; db < kBqReg; ++db)
                    #pragma unroll
                    for (int dr = 0; dr < kTileReg; ++dr)
                        s[db][dr] += qn_reg[db] * w_reg[dr];
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int db = 0; db < kBqReg; ++db) {
        const int bq_i = bq_base + (lb0 + db);
        if (bq_i >= bq_total) continue;
        const int out_base = (bq_i * num_heads + head_idx) * kv_rank;
        #pragma unroll
        for (int dr = 0; dr < kTileReg; ++dr) {
            const int r_i = r_base + (lr0 + dr * kTileThreadsX);
            if (r_i >= kv_rank) continue;
            q_absorbed[out_base + r_i] = static_cast<scalar_t>(s[db][dr]);
        }
    }
}

// Host launcher: CUDA Q-prep → {q_absorbed [B,Sq,H,kv_rank], q_rope [B,Sq,H,rope]}
std::vector<torch::Tensor> launch_q_absorbed(
    torch::Tensor x,          // [B, Sq, hidden]
    torch::Tensor W_q_a,      // [hidden, q_lora]
    torch::Tensor q_norm_w,   // [q_lora]           RMSNorm gain on the latent
    torch::Tensor W_q_b,      // [q_lora, H*(nope+rope)]
    torch::Tensor W_uk,       // [H, nope, kv_rank]
    torch::Tensor positions,  // [Sq] int64 absolute position of each query
    int nope_dim,
    int rope_dim,
    double eps)
{
    auto xc    = x.contiguous();
    auto W_qac = W_q_a.contiguous();
    auto gc    = q_norm_w.contiguous();
    auto W_qbc = W_q_b.contiguous();
    auto W_ukc = W_uk.contiguous();
    auto posc  = positions.to(torch::kLong).contiguous();

    const int batch_size  = static_cast<int>(xc.size(0));
    const int sq          = static_cast<int>(xc.size(1));
    const int hidden_dim  = static_cast<int>(xc.size(2));
    const int q_lora      = static_cast<int>(W_qac.size(1));
    const int num_heads   = static_cast<int>(W_ukc.size(0));
    const int kv_rank     = static_cast<int>(W_ukc.size(2));
    const int qk_head_dim = nope_dim + rope_dim;
    const int bq_total    = batch_size * sq;
    const int n_total     = num_heads * qk_head_dim;

    TORCH_CHECK(W_qac.size(0) == hidden_dim, "W_q_a must be [hidden, q_lora]");
    TORCH_CHECK(W_qbc.size(0) == q_lora && W_qbc.size(1) == n_total,
                "W_q_b must be [q_lora, H*(nope+rope)]");
    TORCH_CHECK(gc.dim() == 1 && gc.size(0) == q_lora,
                "q_norm_w must be a 1-D tensor of length q_lora (got ", gc.sizes(), ")");
    // Required, not defaulted: a wrong RoPE angle is silent, and the value cannot be
    // recovered here — during decode sq == 1 while the query sits at position Sk.
    TORCH_CHECK(posc.dim() == 1 && posc.size(0) == sq,
                "positions must be a 1-D tensor of length Sq (got ", posc.sizes(), ")");

    // Q-prep accumulates and stages in the accumulate dtype: fp64 in stays fp64,
    // fp16 promotes to fp32 (a 7168-long dot in half would lose far more than the
    // narrower staging saves). Outputs are always the input dtype.
    const bool is_f64 = xc.scalar_type() == torch::kFloat64;
    const auto acc_dtype = is_f64 ? torch::kFloat64 : torch::kFloat32;
    const size_t acc_size = is_f64 ? sizeof(double) : sizeof(float);

    // --- split-K decisions (needed up front: they pick how each output is initialised) ---
    // The column count of each projection is fixed by the model shape and grid.y collapses
    // to 1 for every bq_total <= 128, so 2a2 launches 192 blocks against 84*2 = 168
    // resident slots: 1.14 waves, i.e. one full wave then a tail wave holding 24 blocks
    // while 144 slots sit idle. 2a1 is far worse at just 12 column tiles. Split the
    // reduction until the grid spans enough waves that the tail is a small fraction of
    // them; shapes whose output dims already fill the GPU (large batch, prefill) land on
    // split == 1 and pay neither the pre-zero nor the atomics.
    const QGemmSplit lat_split = plan_q_gemm_split(bq_total, q_lora,  hidden_dim);
    const QGemmSplit raw_split = plan_q_gemm_split(bq_total, n_total, q_lora);

    // atomicAdd accumulates, so the split-K partials need a zeroed destination
    auto gemm_out = [&](int cols, bool atomic) {
        const auto opts = xc.options().dtype(acc_dtype);
        return atomic ? torch::zeros({bq_total, cols}, opts)
                      : torch::empty({bq_total, cols}, opts);
    };
    auto q_lat_acc  = gemm_out(q_lora,  lat_split.atomic);
    auto q_raw      = gemm_out(n_total, raw_split.atomic);
    auto q_lat      = torch::empty({bq_total, q_lora}, xc.options());
    auto q_absorbed = torch::empty({batch_size, sq, num_heads, kv_rank}, xc.options());
    auto q_rope_out = torch::empty({batch_size, sq, num_heads, rope_dim}, xc.options());

    auto stream = at::cuda::getCurrentCUDAStream();

    // 2a1 → 2a-norm → 2a2. Both projections are [bq, K] @ [K, N] and share one kernel;
    // only the shapes and the split-K plan differ.
    // grid.x=col tiles, grid.y=bq tiles, grid.z=split-K slice; 16×16 → bq tile × 128 cols
    {
        dim3 threads(kTileThreadsX, kTileThreadsY);  // 16×16 = 256 threads
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(xc.scalar_type(), "q_raw_gemm", ([&] {
            using acc_t = acc_t_of<scalar_t>;
            auto run = [&](const scalar_t* in, const scalar_t* w, acc_t* out,
                           int k_total, int cols, const QGemmSplit& sp) {
                auto launch = [&](auto kernel, int bq_tile) {
                    dim3 blocks((cols     + kQWqColTile - 1) / kQWqColTile,
                                (bq_total + bq_tile     - 1) / bq_tile,
                                sp.split);
                    // SMEM: in[bq_tile][32] + w[128][32] acc_t, rows padded to 33
                    size_t smem = static_cast<size_t>(
                        bq_tile + kQWqColTile) * kQHiddenPad * acc_size;
                    enable_large_smem(kernel, smem);  // fp64 tiles exceed the default 48 KB
                    kernel<<<blocks, threads, smem, stream>>>(
                        in, w, out, bq_total, k_total, cols, sp.k_chunk, sp.atomic);
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                };
                // Pick the narrowest bq tile that does not add a pass over the streamed
                // weight. Every bq tile re-streams the whole weight, so halving the tile
                // only pays while it leaves ceil(bq_total / tile) unchanged — true up to
                // 32 rows, not past it. At bq_total=33 a 32-row tile would compute the
                // same 64 rows over two passes instead of one, trading no arithmetic for
                // extra traffic. 2c and 3d below follow the same rule.
                switch (bq_reg_for(bq_total)) {
                  case 1:  launch(q_raw_gemm_kernel<scalar_t, 1>, 1 * kTileThreadsY); break;
                  case 2:  launch(q_raw_gemm_kernel<scalar_t, 2>, 2 * kTileThreadsY); break;
                  case 4:  launch(q_raw_gemm_kernel<scalar_t, 4>, 4 * kTileThreadsY); break;
                  default: launch(q_raw_gemm_kernel<scalar_t, kTileReg>,
                                  kTileReg * kTileThreadsY);
                }
            };

            // 2a1: x @ W_q_a → the q_lora latent
            run(xc.data_ptr<scalar_t>(), W_qac.data_ptr<scalar_t>(),
                q_lat_acc.data_ptr<acc_t>(), hidden_dim, q_lora, lat_split);

            // 2a-norm: RMSNorm the latent, narrowing to the input dtype for 2a2
            q_lat_norm_kernel<scalar_t><<<max(bq_total, 1), kNormThreads, 0, stream>>>(
                q_lat_acc.data_ptr<acc_t>(), gc.data_ptr<scalar_t>(),
                q_lat.data_ptr<scalar_t>(), q_lora, static_cast<acc_t>(eps));
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            // 2a2: latent @ W_q_b → q_raw
            run(q_lat.data_ptr<scalar_t>(), W_qbc.data_ptr<scalar_t>(),
                q_raw.data_ptr<acc_t>(), q_lora, n_total, raw_split);
        }));
    }

    // 2b: RoPE on the rope half of q_raw → q_rope [B,Sq,H,rope]
    {
        const int total  = bq_total * num_heads * (rope_dim / 2);
        const int wanted = (total + kRopeThreads - 1) / kRopeThreads;
        const int grid   = min(wanted, 32 * sm_count());  // grid-stride past that
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(xc.scalar_type(), "q_rope", ([&] {
            using acc_t = acc_t_of<scalar_t>;
            q_rope_kernel<scalar_t><<<max(grid, 1), kRopeThreads, 0, stream>>>(
                q_raw.data_ptr<acc_t>(), posc.data_ptr<int64_t>(),
                q_rope_out.data_ptr<scalar_t>(),
                bq_total, sq, num_heads, nope_dim, rope_dim, qk_head_dim);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }));
    }

    // 2c: grid.x=bq tiles, grid.y=kv_rank tiles, grid.z=head; 16×16 → bq tile × 128 r
    {
        dim3 threads(kTileThreadsX, kTileThreadsY);  // 16×16 = 256 threads
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(xc.scalar_type(), "q_absorb", ([&] {
            using acc_t = acc_t_of<scalar_t>;
            auto launch = [&](auto kernel, int bq_tile) {
                dim3 blocks((bq_total + bq_tile   - 1) / bq_tile,
                            (kv_rank  + kQRankTile - 1) / kQRankTile,
                            num_heads);
                // SMEM: qn[bq_tile][32] + w[128][32] acc_t, rows padded to 33
                size_t smem = static_cast<size_t>(
                    bq_tile + kQRankTile) * kQNopePad * acc_size;
                enable_large_smem(kernel, smem);
                kernel<<<blocks, threads, smem, stream>>>(
                    q_raw.data_ptr<acc_t>(), W_ukc.data_ptr<scalar_t>(),
                    q_absorbed.data_ptr<scalar_t>(),
                    bq_total, num_heads, nope_dim, kv_rank, qk_head_dim);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            };
            switch (bq_reg_for(bq_total)) {
              case 1:  launch(q_absorb_kernel<scalar_t, 1>, 1 * kTileThreadsY); break;
              case 2:  launch(q_absorb_kernel<scalar_t, 2>, 2 * kTileThreadsY); break;
              case 4:  launch(q_absorb_kernel<scalar_t, 4>, 4 * kTileThreadsY); break;
              default: launch(q_absorb_kernel<scalar_t, kTileReg>, kTileReg * kTileThreadsY);
            }
        }));
    }

    // No cudaDeviceSynchronize: every launch above is on the caller's stream, so
    // ordering against the consuming PyTorch ops is already guaranteed.
    return {q_absorbed, q_rope_out};
}


// ============================================================================
// KERNEL 3: Official absorbed MLA attention
//
//  3a. scores — grid.x over flat(B,Sq,128), grid.y over Sk
//      16×16 threads/block → 128×128 scores/block; each thread → 8×8 tile
//      one (kv_rank + rope_dim) = 576-deep reduction in SMEM tiles of 32:
//        r <  512 draws from q_absorbed [B,Sq,H,512] and c_kv     [B,Sk,512]
//        r >= 512 draws from q_rope     [B,Sq,H,64]  and pe_cache [B,Sk,64]
//      then /sqrt(192); the epilogue also reduces each row's maximum into row_max
//  3c. ctx, with the softmax folded in — same 16×16 / 8×8 shape as 3a
//      grid.x=flat, grid.y=kv_rank, grid.z=split-K over Sk
//      128 flats × 128 ranks/block; Sk contracted in SMEM tiles of kCtxSkTile=32
//      exp(scores - row_max) @ c_kv[B,Sk,512] → *unnormalised* ctx[B,Sq,H,512],
//      plus the denominator row_sum[B*Sq*H]. There is no separate softmax pass: the
//      numerator is formed on the 128×32 tile already sitting in SMEM, so the
//      normalised attention matrix is never written to memory at all.
//  3d. out — 16×16 threads, 8 v_dims × kBqReg bq/thread; grid.x=bq, .y=v, .z=head
//      one head per block so W_uv[h] is staged once and reused across the bq tile
//      (ctx[B,Sq,H,512] / row_sum) × W_uv[H,128,512] → out[B,Sq,H,128]
//      the deferred softmax divide rides along on a staging load 3d already does
// ============================================================================

// Atomic max on a float/double, which CUDA provides no native instruction for.
// IEEE-754 is sign-magnitude, so for a non-negative value the bit pattern read as a
// signed integer orders identically to the float, and atomicMax on int does the job.
// For a negative value the ordering inverts, but reading the pattern as *unsigned*
// reverses it back — a larger float is a smaller unsigned — so atomicMin gets it.
// Mixed signs work in both branches: positives have the high bit clear and so are
// smaller unsigned than any negative, and larger signed than any negative.
__device__ __forceinline__ void atomic_max_acc(float* addr, float v) {
    if (v >= 0.0f) atomicMax(reinterpret_cast<int*>(addr), __float_as_int(v));
    else           atomicMin(reinterpret_cast<unsigned int*>(addr), __float_as_uint(v));
}
__device__ __forceinline__ void atomic_max_acc(double* addr, double v) {
    if (v >= 0.0)
        atomicMax(reinterpret_cast<long long*>(addr), __double_as_longlong(v));
    else
        atomicMin(reinterpret_cast<unsigned long long*>(addr),
                  static_cast<unsigned long long>(__double_as_longlong(v)));
}

// --- 3a. scores: 8×8 output tile per thread; grid (flat B*Sq*H, Sk) ---
//   grid.x: flat = b*(Sq*128) + q*128 + h     ∈ [0, B*Sq*128)  — 128 flats/block
//   grid.y: k ∈ [0, Sk)                                        — 128 keys/block
//   block: 16×16 threads → each thread 8 keys × 8 flats (128×128 / block)
//   threadIdx.x owns k (contiguous in `scores`), threadIdx.y owns flat: a warp
//   writes 32 adjacent keys of one score row, so the epilogue is one 128 B store
//   SMEM [lf][r] / [lk][r], rows padded to kScoreRankPad (no swizzle — see header)
//   Staging: r is idx% — matches the HBM contiguous dim of all four sources
//   Reduction is r-outer: 8 qa + 8 ck loads → 64 FMAs, so 0.25 SMEM loads per FMA
//   rope_dim is read at runtime; nothing here is specialised to its value
template <typename scalar_t>
__global__ void __launch_bounds__(kTileThreads, kTileBlocksPerSM)
mla_scores_kernel(
    const scalar_t* __restrict__ q_absorbed, // [B, Sq, H, kv_rank=512]
    const scalar_t* __restrict__ q_rope,     // [B, Sq, H, rope_dim=64]
    const scalar_t* __restrict__ c_kv,       // [B, Sk, kv_rank=512]
    const scalar_t* __restrict__ pe_cache,   // [B, Sk, rope_dim=64]  head-shared
    acc_t_of<scalar_t>* __restrict__ scores, // [B, Sq, H, Sk]  in the accumulate dtype
    acc_t_of<scalar_t>* __restrict__ row_max, // [B*Sq*H] running row max, or nullptr
    int batch_size,
    int sq,
    int sk,
    int num_heads,
    int kv_rank,
    int rope_dim,
    acc_t_of<scalar_t> scale)
{
    using acc_t = acc_t_of<scalar_t>;

    const int flat_total = batch_size * sq * num_heads;  // B*Sq*128
    const int flat_base  = blockIdx.x * kScoreFlatTile;
    const int k0         = blockIdx.y * kScoreSkTile;

    // qa_smem[kScoreFlatTile][kScoreRankPad]; ck_smem[kScoreSkTile][kScoreRankPad]
    // — the rope tile reuses them as the trailing slice of the same reduction
    acc_t* smem    = dynamic_smem<acc_t>();
    acc_t* qa_smem = smem;
    acc_t* ck_smem = qa_smem + kScoreFlatTile * kScoreRankPad;

    // batch for c_kv / pe_cache rows (the flat tile stays inside one batch as long
    // as Sq*H is a multiple of it, which holds for H=128 and a 128-wide tile)
    const int batch_idx = min(flat_base, max(flat_total - 1, 0)) / (sq * num_heads);
    const int ck_batch_stride = batch_idx * sk * kv_rank;
    const int pe_batch_stride = batch_idx * sk * rope_dim;

    const int coop_stride = blockDim.x * blockDim.y;
    const acc_t inv_scale = acc_t(1) / scale;

    // Per-thread output tile; rope later accumulates into the same regs
    acc_t s[kTileReg][kTileReg];
    #pragma unroll
    for (int di = 0; di < kTileReg; ++di)
        #pragma unroll
        for (int dk = 0; dk < kTileReg; ++dk)
            s[di][dk] = acc_t(0);

    // SMEM row indices, as bases only — the per-element index is recomputed at each
    // use. lf(i) = lf0 + i and lk(j) = lk0 + j*kTileThreadsX are affine, so every SMEM
    // access folds to a base register plus a compile-time immediate. Materialising them
    // as arrays costs no registers (the compiler rematerialises them either way) but
    // does make the epilogue emit redundant 64-bit address arithmetic: -264 SASS
    // instructions here, though it measured inside noise since the epilogue is ~0.5% of
    // dynamic instructions.
    //
    // Deliberately *not* accompanied by precomputed flat[]/k_idx[]/valid_*[] arrays:
    // those are epilogue-only, and this kernel sits at exactly the 128-register budget
    // that 256 threads x 2 blocks/SM allows, so every value kept alive across the
    // reduction is a register the scheduler cannot use to prefetch the next operand.
    const int lf0 = threadIdx.y * kTileReg;
    const int lk0 = threadIdx.x;


    // ---- One reduction over the full qk depth: kv_rank then rope_dim ----
    // scores = q_absorbed·c_kv + q_rope·pe_cache is a single (kv_rank + rope_dim)-deep
    // dot product that happens to draw its operands from two pairs of tensors. Running
    // it as one loop rather than two phases means the rope contribution is just the
    // trailing tile: no mid-kernel re-stage of aliased SMEM, no extra __syncthreads
    // pair, and one SMEM row width (kScoreRankPad) throughout.
    for (int r0 = 0; r0 < kv_rank + rope_dim; r0 += kScoreRankTile) {
        // Uniform across the block (r0 is a loop counter), so no warp divergence.
        const bool rope     = r0 >= kv_rank;
        const int  depth    = rope ? rope_dim : kv_rank;  // extent of the source operand
        const int  r_base   = rope ? (r0 - kv_rank) : r0; // offset within that operand
        const int  tile_len = min(kScoreRankTile, depth - r_base);

        // Cooperative staging. The loop extent is the *full* tile width, not tile_len,
        // so the div/mod fold to a shift and a mask instead of an integer division by a
        // runtime value; a short trailing tile is handled by predicating the store. 
        const int stage_elems_q = kScoreFlatTile * kScoreRankTile;
        const int stage_elems_k = kScoreSkTile * kScoreRankTile;

        // stage the q side → qa_smem[lf][r^lf]
        // both sources are contiguous in r → r_local must be idx% so warps coalesce
        for (int idx = threadIdx.x + threadIdx.y * blockDim.x;
             idx < stage_elems_q; idx += coop_stride) {
            const int lf_s    = idx / kScoreRankTile;
            const int r_local = idx % kScoreRankTile;
            const int flat_s  = flat_base + lf_s;
            if (flat_s < flat_total && r_local < tile_len)
                qa_smem[lf_s * kScoreRankPad + r_local] =
                    static_cast<acc_t>(
                        rope ? q_rope[flat_s * rope_dim + r_base + r_local]
                             : q_absorbed[flat_s * kv_rank + r_base + r_local]);
        }

        // stage the k side → ck_smem[lk][r^lk] (contiguous in r either way)
        for (int idx = threadIdx.x + threadIdx.y * blockDim.x;
             idx < stage_elems_k; idx += coop_stride) {
            const int lk_s    = idx / kScoreRankTile;
            const int r_local = idx % kScoreRankTile;
            const int k_s     = k0 + lk_s;
            if (k_s < sk && r_local < tile_len)
                ck_smem[lk_s * kScoreRankPad + r_local] =
                    static_cast<acc_t>(
                        rope ? pe_cache[pe_batch_stride + k_s * rope_dim + r_base + r_local]
                             : c_kv[ck_batch_stride + k_s * kv_rank + r_base + r_local]);
        }
        __syncthreads();  // loads done before any thread reads this tile

        // r-outer register strip: 8 qa + 8 ck SMEM loads feed 64 FMAs (0.25 loads/FMA).
        // Lanes past flat_total / sk read unwritten SMEM; their accumulators are dropped
        // at the store, so no per-iteration bounds predicate is needed here.
        // Capped at 8, matching 3c/3d: a full 64-wide unroll asks for more live values
        // than the 64-register budget holds and spills ~1.2 KB per thread to local.
        if (tile_len == kScoreRankTile) {
            #pragma unroll 1
            for (int r_local = 0; r_local < kScoreRankTile; ++r_local) {
                acc_t qa_reg[kTileReg], ck_reg[kTileReg];
                #pragma unroll
                for (int di = 0; di < kTileReg; ++di)
                    qa_reg[di] = qa_smem[(lf0 + di) * kScoreRankPad + r_local];
                #pragma unroll
                for (int dk = 0; dk < kTileReg; ++dk)
                    ck_reg[dk] = ck_smem[(lk0 + dk * kTileThreadsX) * kScoreRankPad + r_local];
                #pragma unroll
                for (int di = 0; di < kTileReg; ++di)
                    #pragma unroll
                    for (int dk = 0; dk < kTileReg; ++dk)
                        s[di][dk] += qa_reg[di] * ck_reg[dk];
            }
        } else {
            // ragged final kv_rank tile: same body, runtime trip count
            for (int r_local = 0; r_local < tile_len; ++r_local) {
                acc_t qa_reg[kTileReg], ck_reg[kTileReg];
                #pragma unroll
                for (int di = 0; di < kTileReg; ++di)
                    qa_reg[di] = qa_smem[(lf0 + di) * kScoreRankPad + r_local];
                #pragma unroll
                for (int dk = 0; dk < kTileReg; ++dk)
                    ck_reg[dk] = ck_smem[(lk0 + dk * kTileThreadsX) * kScoreRankPad + r_local];
                #pragma unroll
                for (int di = 0; di < kTileReg; ++di)
                    #pragma unroll
                    for (int dk = 0; dk < kTileReg; ++dk)
                        s[di][dk] += qa_reg[di] * ck_reg[dk];
            }
        }
        __syncthreads();  // all reads done before the next tile overwrites SMEM
    }

    // Bounds are recomputed here rather than carried in registers through the loops.
    #pragma unroll
    for (int di = 0; di < kTileReg; ++di) {
        const int flat_i = flat_base + (lf0 + di);
        if (flat_i >= flat_total) continue;
        #pragma unroll
        for (int dk = 0; dk < kTileReg; ++dk) {
            const int k_i = k0 + (lk0 + dk * kTileThreadsX);
            if (k_i >= sk) continue;
            scores[flat_i * sk + k_i] = s[di][dk] * inv_scale;
        }
    }

    // Row maxima for 3c, computed here because the values are already in registers — a
    // separate pass would have to re-read the whole scores tensor. Each
    // thread holds 8 keys of 8 flats, so reduce over its keys, then across the 16
    // threadIdx.x lanes sharing those flats, then one atomic per flat per block.
    // inv_scale > 0, so scaling after the max is the same as scaling before it.
    if (row_max != nullptr) {
        #pragma unroll
        for (int di = 0; di < kTileReg; ++di) {
            acc_t m = -INFINITY;
            #pragma unroll
            for (int dk = 0; dk < kTileReg; ++dk) {
                const int k_i = k0 + (lk0 + dk * kTileThreadsX);
                if (k_i < sk) m = fmax(m, s[di][dk]);
            }
            // threadIdx.x is the low 4 bits of the lane id (blockDim.x == 16), so an XOR
            // butterfly up to 8 stays inside the 16 lanes that share these flats.
            #pragma unroll
            for (int off = 1; off < kTileThreadsX; off <<= 1)
                m = fmax(m, __shfl_xor_sync(0xffffffffu, m, off));
            const int flat_i = flat_base + (lf0 + di);
            if (threadIdx.x == 0 && flat_i < flat_total && m != -INFINITY)
                atomic_max_acc(&row_max[flat_i], m * inv_scale);
        }
    }
}

// Write one ctx element. The generic template keeps ctx in the input dtype and only ever
// stores, which is all the split == 1 path needs. Split-K accumulates partials with
// atomicAdd, and for that case the host promotes ctx to the accumulate dtype — so only
// the exact-match overloads below have to handle use_atomic. Non-template overloads win
// overload resolution when ctx_t and acc_t coincide, which is exactly the split-K case.
template <typename ctx_t, typename acc_t>
__device__ __forceinline__ void ctx_store(ctx_t* dst, acc_t v, bool /*use_atomic*/) {
    *dst = static_cast<ctx_t>(v);
}
__device__ __forceinline__ void ctx_store(float* dst, float v, bool use_atomic) {
    if (use_atomic) atomicAdd(dst, v);
    else            *dst = v;
}
__device__ __forceinline__ void ctx_store(double* dst, double v, bool use_atomic) {
    if (use_atomic) atomicAdd(dst, v);  // fp64 atomicAdd is native from sm_60
    else            *dst = v;
}

// --- 3c (+3b fused). ctx = softmax(scores) @ c_kv — flash-style online softmax ---
//
// WHAT THIS COMPUTES.  For every row flat = (b, q, h), of which there are
// flat_total = B*Sq*H:
//
//     s   = scores[flat, 0:Sk]          raw logits from 3a, already scaled by 1/sqrt(192)
//     p   = exp(s - max(s))             softmax numerator, max subtracted for stability
//     ctx = (p @ c_kv[b]) / sum(p)      attention output for that row
//
// The divide is *not* done here — this kernel emits the unnormalised numerator
// `p @ c_kv` plus the scalar denominator sum(p) per row, and 3d applies the divide while
// it stages ctx. That is free there (one multiply by a precomputed reciprocal) and it
// keeps the split-K partials addable.
//
// MATRIX DIMENSIONS.  Whole-tensor shapes, and the tile each block contracts:
//
//   in   scores   [B, Sq, H, Sk]         tile [kCtxFlatTile=128 flats][kCtxSkTile=32 keys]
//   in   c_kv     [B, Sk, kv_rank=512]   tile [kCtxRankTile=128 ranks][kCtxSkTile=32 keys]
//   out  ctx      [B, Sq, H, kv_rank]    tile [128 flats][128 ranks], held in registers
//   out  row_sum  [B*Sq*H]               the sum(p) denominator, consumed by 3d
//   in   row_max  [B*Sq*H]               per-row max of the logits, from 3a
//
// So one block evaluates  [128 x Sk] @ [Sk x 128] -> [128 x 128], walking Sk in 32-wide
// tiles and contracting 128 x 128 x 32 per tile. Per thread that is an 8x8 output tile
// fed by 8 + 8 SMEM loads per 64 FMAs = 0.25 loads/FMA, the sm_86 balance point.
//   grid.x: flat = b*(Sq*H) + q*H + h     — 128 flats/block
//   grid.y: r ∈ [0, kv_rank)              — 128 ranks/block
//   grid.z: split-K slice over Sk — see below
//   block: 16×16 threads; threadIdx.x owns r (contiguous in `ctx`), threadIdx.y owns
//   flat — same as 3a, so the epilogue is one 128 B transaction
//   SMEM [lf][k] / [lr][k] padded to kCtxSkPad; scores staged k-fast, c_kv r-fast
//
// ONLINE (FLASH) SOFTMAX.  The true row maximum is not known until all of Sk has been
// read, so each tile folds its own maximum into a running m and rescales the
// accumulator by exp(m_old - m_new) whenever m grows; the denominator l is rescaled the
// same way. Only the ratio ctx/l ever leaves this kernel, and that ratio is invariant to
// the choice of m, so no renormalisation pass is needed at the end.
//
//   Split-K: the grid only parallelises the output dims (flat × kv_rank), so a
//   single-stream decode (B=Sq=1 → 128 flats) yields just 1×4 = 4 blocks and leaves
//   ~90% of the SMs idle while each block walks all of Sk serially. grid.z chops the
//   Sk reduction into `sk_chunk`-sized slices; each slice accumulates a partial dot
//   product and atomically adds it into ctx, which the host pre-zeroes. The host sets
//   gridDim.z == 1 for shapes that already fill the GPU, and `use_atomic` then selects
//   plain stores so the common path pays neither the memset nor the atomics.
//   ctx_t is the input dtype on the plain path and the accumulate dtype on the split-K
//   path (see ctx_store): read-modify-write accumulation never narrows.
//
//   row_max and the softmax: 3a hands this kernel the maximum of each row, always,
//   and that serves two distinct purposes.
//
//   It is *required* whenever gridDim.z > 1. A slice sees only its own span of Sk, so a
//   maximum discovered online would differ between slices, their numerators would be on
//   different scales, and the partials could not simply be added. A shared maximum is
//   what lets the atomicAdd epilogue above stay as it is, with no per-slice buffers and
//   no combine pass.
//
//   It is *profitable* on every path, including gridDim.z == 1. Nothing can raise a
//   maximum that is already the maximum over all of Sk, so the per-tile scan and the
//   accumulator rescale below are provably dead work and are branched around, taking
//   one of the four per-tile barriers with them. Measured 3c -2.7% against 3a +0.7%.
//
//   The online path (row_max == nullptr) is kept because it is the formulation that
//   makes this kernel correct on its own, without 3a's cooperation.
template <typename scalar_t, typename ctx_t>
__global__ void __launch_bounds__(kTileThreads, kTileBlocksPerSM)
mla_ctx_kernel(
    const acc_t_of<scalar_t>* __restrict__ scores,  // [B, Sq, H, Sk] raw logits from 3a
    const scalar_t* __restrict__ c_kv,  // [B, Sk, kv_rank]
    ctx_t*          __restrict__ ctx,   // [B, Sq, H, kv_rank]  unnormalised numerator
    const acc_t_of<scalar_t>* __restrict__ row_max,  // [B*Sq*H] from 3a, or nullptr
    acc_t_of<scalar_t>* __restrict__ row_sum,        // [B*Sq*H] softmax denominator out
    int batch_size,
    int sq,
    int sk,
    int num_heads,
    int kv_rank,
    int sk_chunk,   // Sk elements owned by one grid.z slice (multiple of kCtxSkTile)
    bool use_atomic)
{
    using acc_t = acc_t_of<scalar_t>;

    const int flat_total = batch_size * sq * num_heads;
    const int flat_base  = blockIdx.x * kCtxFlatTile;
    const int r0         = blockIdx.y * kCtxRankTile;

    // this block's slice of the Sk reduction; sk_chunk == sk when gridDim.z == 1
    const int k_begin = blockIdx.z * sk_chunk;
    const int k_end   = min(k_begin + sk_chunk, sk);

    acc_t* smem      = dynamic_smem<acc_t>();
    // holds the raw logits after staging, then exp(s - m) in place after pass B
    acc_t* p_smem  = smem;                                // [kCtxFlatTile][kCtxSkTile]
    acc_t* ck_smem = p_smem + kCtxFlatTile * kCtxSkPad;   // [kCtxRankTile][kCtxSkPad]
    // Softmax state, one entry per flat row of the tile. Kept in SMEM rather than in
    // registers because the thread that scans a row here is not the thread that owns it
    // in the 8x8 compute tile, and registers are already at the 128 budget.
    acc_t* m_smem    = ck_smem + kCtxRankTile * kCtxSkPad;  // running row maximum
    acc_t* l_smem    = m_smem + kCtxFlatTile;               // running row denominator
    acc_t* resc_smem = l_smem + kCtxFlatTile;               // exp(m_old - m_new) this tile

    // one batch per block when flat tile does not cross Sq*H (true for H=128, tile=64)
    const int batch_idx = min(flat_base, max(flat_total - 1, 0)) / (sq * num_heads);
    const int ck_batch_stride = batch_idx * sk * kv_rank;

    const int coop_stride = blockDim.x * blockDim.y;

    acc_t s[kTileReg][kTileReg];
    #pragma unroll
    for (int di = 0; di < kTileReg; ++di)
        #pragma unroll
        for (int dr = 0; dr < kTileReg; ++dr)
            s[di][dr] = acc_t(0);

    // SMEM row index bases; see the note in 3a for why these are not arrays and why
    // epilogue-only values are recomputed at the bottom rather than kept alive.
    const int lf0 = threadIdx.y * kTileReg;
    const int lr0 = threadIdx.x;

    const int tid = threadIdx.x + threadIdx.y * blockDim.x;

    // Seed the softmax state from the maximum 3a already reduced. Every slice then
    // starts on the same scale, and no tile can raise it, so the loop below skips the
    // scan and the rescale entirely. Passing nullptr instead falls back to discovering
    // the maximum online, which is the textbook flash formulation and is kept because
    // it is what makes this kernel correct without 3a's cooperation.
    for (int f = tid; f < kCtxFlatTile; f += coop_stride) {
        const int flat_s = flat_base + f;
        m_smem[f] = (row_max != nullptr && flat_s < flat_total)
                        ? row_max[flat_s] : -INFINITY;
        l_smem[f] = acc_t(0);
        resc_smem[f] = acc_t(1);   // stays 1 whenever the maximum is precomputed
    }
    __syncthreads();

    for (int k0 = k_begin; k0 < k_end; k0 += kCtxSkTile) {
        const int tile_len =
            (k0 + kCtxSkTile <= k_end) ? kCtxSkTile : (k_end - k0);

        // stage the raw logits scores[flat, k0:k0+tile) — k contiguous so warps
        // coalesce. Pass B below overwrites them in place with exp(s - m).
        // Full-tile extent, not tile_len, so the div/mod fold to shift and mask
        // rather than a runtime integer division; a short trailing tile is handled by
        // predicating the store.
        const int stage_elems = kCtxFlatTile * kCtxSkTile;
        for (int idx = threadIdx.x + threadIdx.y * blockDim.x;
             idx < stage_elems; idx += coop_stride) {
            const int lf_s    = idx / kCtxSkTile;
            const int k_local = idx % kCtxSkTile;
            const int flat_s  = flat_base + lf_s;
            if (flat_s < flat_total && k_local < tile_len)
                p_smem[lf_s * kCtxSkPad + k_local] =
                    scores[flat_s * sk + k0 + k_local];
        }

        // stage c_kv[b, k0:k0+tile, r0:r0+128) — r contiguous → lr = idx % kRs
        const int ck_elems = tile_len * kCtxRankTile;
        for (int idx = threadIdx.x + threadIdx.y * blockDim.x;
             idx < ck_elems; idx += coop_stride) {
            const int k_local = idx / kCtxRankTile;
            const int lr_s    = idx % kCtxRankTile;
            const int r_s     = r0 + lr_s;
            if (k_local < tile_len && r_s < kv_rank)
                ck_smem[lr_s * kCtxSkPad + k_local] =
                    static_cast<acc_t>(
                        c_kv[ck_batch_stride + (k0 + k_local) * kv_rank + r_s]);
        }
        __syncthreads();

        // ---- online softmax over this Sk tile -------------------------------------
        // A precomputed row_max is already the maximum over all of Sk, so no tile can
        // raise it: every rescale would be exactly 1 and Pass A's scan would be
        // discarded. Skip both, and one of the barriers with them. Uniform across the
        // block (row_max is a kernel argument), so the __syncthreads inside is safe.
        if (row_max == nullptr) {
        // Pass A: this tile's row maxima, folded into the running maxima. One thread per
        // flat row; the row is 32 wide and rows are kCtxSkPad apart, so consecutive
        // threads land on consecutive banks.
        for (int f = tid; f < kCtxFlatTile; f += coop_stride) {
            acc_t tmax = -INFINITY;
            for (int k = 0; k < tile_len; ++k)
                tmax = fmax(tmax, p_smem[f * kCtxSkPad + k]);
            const acc_t m_old = m_smem[f];
            const acc_t m_new = fmax(m_old, tmax);
            // m_old == m_new covers both "maximum did not grow" and the all-empty case,
            // where -inf - -inf would be NaN. Otherwise m_new is finite and a -inf m_old
            // gives exp(-inf) == 0, correctly zeroing an accumulator that holds nothing.
            resc_smem[f] = (m_old == m_new) ? acc_t(1) : exp(m_old - m_new);
            m_smem[f]    = m_new;
        }
        __syncthreads();

        // Rescale the running accumulator onto the new maximum.
        #pragma unroll
        for (int di = 0; di < kTileReg; ++di) {
            const acc_t rs = resc_smem[lf0 + di];
            #pragma unroll
            for (int dr = 0; dr < kTileReg; ++dr)
                s[di][dr] *= rs;
        }
        }  // row_max == nullptr

        // Pass B: exponentiate the tile in place so the reduction below is unchanged,
        // and fold this tile's row sums into the running denominator.
        for (int f = tid; f < kCtxFlatTile; f += coop_stride) {
            const acc_t m_new = m_smem[f];
            acc_t sum = acc_t(0);
            for (int k = 0; k < tile_len; ++k) {
                const acc_t p = exp(p_smem[f * kCtxSkPad + k] - m_new);
                p_smem[f * kCtxSkPad + k] = p;
                sum += p;
            }
            l_smem[f] = l_smem[f] * resc_smem[f] + sum;
        }
        __syncthreads();
        // ---------------------------------------------------------------------------

        // k-outer register strip: 8 p + 8 ck SMEM loads feed 64 FMAs (0.25 loads/FMA).
        // Lanes past flat_total / kv_rank read unwritten SMEM; dropped at the store.
        // Not unrolled: 64 FMAs per step is already ample ILP and a wider unroll spills.
        if (tile_len == kCtxSkTile) {
            #pragma unroll 1
            for (int k_local = 0; k_local < kCtxSkTile; ++k_local) {
                acc_t p_reg[kTileReg], ck_reg[kTileReg];
                #pragma unroll
                for (int di = 0; di < kTileReg; ++di)
                    p_reg[di] = p_smem[(lf0 + di) * kCtxSkPad + k_local];
                #pragma unroll
                for (int dr = 0; dr < kTileReg; ++dr)
                    ck_reg[dr] = ck_smem[(lr0 + dr * kTileThreadsX) * kCtxSkPad + k_local];
                #pragma unroll
                for (int di = 0; di < kTileReg; ++di)
                    #pragma unroll
                    for (int dr = 0; dr < kTileReg; ++dr)
                        s[di][dr] += p_reg[di] * ck_reg[dr];
            }
        } else {
            // ragged final Sk tile: same body, runtime trip count
            for (int k_local = 0; k_local < tile_len; ++k_local) {
                acc_t p_reg[kTileReg], ck_reg[kTileReg];
                #pragma unroll
                for (int di = 0; di < kTileReg; ++di)
                    p_reg[di] = p_smem[(lf0 + di) * kCtxSkPad + k_local];
                #pragma unroll
                for (int dr = 0; dr < kTileReg; ++dr)
                    ck_reg[dr] = ck_smem[(lr0 + dr * kTileThreadsX) * kCtxSkPad + k_local];
                #pragma unroll
                for (int di = 0; di < kTileReg; ++di)
                    #pragma unroll
                    for (int dr = 0; dr < kTileReg; ++dr)
                        s[di][dr] += p_reg[di] * ck_reg[dr];
            }
        }
        __syncthreads();
    }

    // Each grid.z slice owns a disjoint span of Sk, so the partials are a plain sum —
    // they share a common maximum via row_max, so the numerators are on the same scale.
    // Lanes of a warp differ in `flat`, so these hit distinct cache lines; the only
    // contention is between the gridDim.z blocks sharing an output tile, and each of
    // them contributes exactly one atomic per element after its whole reduction.
    #pragma unroll
    for (int di = 0; di < kTileReg; ++di) {
        const int flat_i = flat_base + (lf0 + di);
        if (flat_i >= flat_total) continue;
        #pragma unroll
        for (int dr = 0; dr < kTileReg; ++dr) {
            const int r_i = r0 + (lr0 + dr * kTileThreadsX);
            if (r_i >= kv_rank) continue;
            ctx_store(&ctx[flat_i * kv_rank + r_i], s[di][dr], use_atomic);
        }
    }

    // The denominator depends only on (flat, Sk slice), not on the rank tile, so all
    // gridDim.y blocks of a slice compute the same value — only the first writes it.
    if (blockIdx.y == 0) {
        for (int f = tid; f < kCtxFlatTile; f += coop_stride) {
            const int flat_s = flat_base + f;
            if (flat_s >= flat_total) continue;
            if (use_atomic) atomicAdd(&row_sum[flat_s], l_smem[f]);
            else            row_sum[flat_s] = l_smem[f];
        }
    }
}

// --- 3d. out = ctx @ W_uv: kBqReg×8 tile/thread; grid (bq tiles, v tiles, head) ---
//   One head per block: W_uv[h] is staged into SMEM once and amortised over the
//   whole bq tile, instead of being re-read from HBM for every single (bq, h).
//   grid.x: bq = b*Sq + q                     — kTileThreadsY*kBqReg bq/block
//   grid.y: d ∈ [0, v_dim)                    — 128 v_dims/block
//   grid.z: h ∈ [0, H)                        — the shared W_uv slice
//   block: 16×16 → each thread 8 v_dims × kBqReg bq; kv_rank in SMEM tiles of 32
//   x indexes v_dim so a warp stores runs of 8 contiguous floats of `out`
//   SMEM [lb][r] / [ld][r] padded to kOutRankPad (no swizzle — see header), r fast
//   Reduction is r-outer: kBqReg ctx + 8 w loads → 8*kBqReg FMAs; kBqReg from
//   bq_reg_for(), same as 2a/2c
template <typename scalar_t, typename ctx_t, int kBqReg>
__global__ void __launch_bounds__(kTileThreads, kTileBlocksPerSM)
mla_output_smem_kernel(
    const ctx_t*    __restrict__ ctx,   // [B, Sq, H, kv_rank]  unnormalised, from 3c
    const scalar_t* __restrict__ W_uv,  // [H, v_dim, kv_rank]
    scalar_t*       __restrict__ out,   // [B, Sq, H, v_dim]
    const acc_t_of<scalar_t>* __restrict__ row_sum,  // [B*Sq*H] softmax denominator
    int bq_total,
    int num_heads,
    int kv_rank,
    int v_dim)
{
    using acc_t = acc_t_of<scalar_t>;

    constexpr int kBqTile = kTileThreadsY * kBqReg;  // bq rows this block owns

    const int bq_base   = blockIdx.x * kBqTile;
    const int d_base    = blockIdx.y * kOutVTile;
    const int head_idx  = blockIdx.z;

    acc_t* smem      = dynamic_smem<acc_t>();
    acc_t* ctx_smem  = smem;                              // [kBqTile][kOutRankTile]
    acc_t* w_smem    = ctx_smem + kBqTile * kOutRankPad;  // [kOutVTile][kOutRankPad]
    // 1/sum(p) for each bq row of this block. 3c emitted the unnormalised numerator, so
    // the softmax divide lands here — as one reciprocal per row, computed once, then a
    // multiply folded into the staging load that was happening anyway.
    acc_t* inv_l_smem = w_smem + kOutVTile * kOutRankPad;  // [kBqTile]

    // ctx[bq, h, r] and W_uv[h, d, r] are both contiguous in r
    const int ctx_head_base = head_idx * kv_rank;
    const int w_head_base   = head_idx * v_dim * kv_rank;
    const int coop_stride   = blockDim.x * blockDim.y;

    acc_t s[kBqReg][kTileReg];
    #pragma unroll
    for (int db = 0; db < kBqReg; ++db)
        #pragma unroll
        for (int dd = 0; dd < kTileReg; ++dd)
            s[db][dd] = acc_t(0);

    // Only the SMEM row indices live across the reduction (see the note in 3a).
    // SMEM row index bases, not arrays — see the note in 3a.
    const int lb0 = threadIdx.y * kBqReg;
    const int ld0 = threadIdx.x;

    // flat = bq*H + h, matching 3c's row indexing
    for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < kBqTile; i += coop_stride) {
        const int bq_s = bq_base + i;
        inv_l_smem[i] = (bq_s < bq_total)
            ? acc_t(1) / row_sum[bq_s * num_heads + head_idx]
            : acc_t(0);
    }
    __syncthreads();

    for (int r0 = 0; r0 < kv_rank; r0 += kOutRankTile) {
        const int tile_len =
            (r0 + kOutRankTile <= kv_rank) ? kOutRankTile : (kv_rank - r0);

        // stage ctx[bq_base:+kBqTile, h, r0:+tile) — r contiguous so warps coalesce.
        // Full-tile extent, not tile_len, so the div/mod fold to a shift and a mask
        // rather than a runtime integer division; a short trailing tile is handled by
        // predicating the store. Measured 26% off mla_scores_kernel.
        const int ctx_elems = kBqTile * kOutRankTile;
        for (int idx = threadIdx.x + threadIdx.y * blockDim.x;
             idx < ctx_elems; idx += coop_stride) {
            const int lb_s    = idx / kOutRankTile;
            const int r_local = idx % kOutRankTile;
            const int bq_s    = bq_base + lb_s;
            if (bq_s < bq_total && r_local < tile_len)
                ctx_smem[lb_s * kOutRankPad + r_local] =
                    static_cast<acc_t>(
                        ctx[(bq_s * num_heads) * kv_rank + ctx_head_base + r0 + r_local])
                    * inv_l_smem[lb_s];   // the deferred softmax divide
        }

        // stage W_uv[h, d_base:+128, r0:+tile) — reused by every bq in this block
        const int w_elems = kOutVTile * kOutRankTile;
        for (int idx = threadIdx.x + threadIdx.y * blockDim.x;
             idx < w_elems; idx += coop_stride) {
            const int ld_s    = idx / kOutRankTile;
            const int r_local = idx % kOutRankTile;
            const int d_s     = d_base + ld_s;
            if (d_s < v_dim && r_local < tile_len)
                w_smem[ld_s * kOutRankPad + r_local] =
                    static_cast<acc_t>(
                        W_uv[w_head_base + d_s * kv_rank + r0 + r_local]);
        }
        __syncthreads();

        // r-outer register strip; out-of-range lanes read unwritten SMEM and are
        // dropped at the store, so no per-iteration bounds predicate is needed
        if (tile_len == kOutRankTile) {
            #pragma unroll 1
            for (int r_local = 0; r_local < kOutRankTile; ++r_local) {
                acc_t ctx_reg[kBqReg], w_reg[kTileReg];
                #pragma unroll
                for (int db = 0; db < kBqReg; ++db)
                    ctx_reg[db] = ctx_smem[(lb0 + db) * kOutRankPad + r_local];
                #pragma unroll
                for (int dd = 0; dd < kTileReg; ++dd)
                    w_reg[dd] = w_smem[(ld0 + dd * kTileThreadsX) * kOutRankPad + r_local];
                #pragma unroll
                for (int db = 0; db < kBqReg; ++db)
                    #pragma unroll
                    for (int dd = 0; dd < kTileReg; ++dd)
                        s[db][dd] += ctx_reg[db] * w_reg[dd];
            }
        } else {
            // ragged final kv_rank tile: same body, runtime trip count
            for (int r_local = 0; r_local < tile_len; ++r_local) {
                acc_t ctx_reg[kBqReg], w_reg[kTileReg];
                #pragma unroll
                for (int db = 0; db < kBqReg; ++db)
                    ctx_reg[db] = ctx_smem[(lb0 + db) * kOutRankPad + r_local];
                #pragma unroll
                for (int dd = 0; dd < kTileReg; ++dd)
                    w_reg[dd] = w_smem[(ld0 + dd * kTileThreadsX) * kOutRankPad + r_local];
                #pragma unroll
                for (int db = 0; db < kBqReg; ++db)
                    #pragma unroll
                    for (int dd = 0; dd < kTileReg; ++dd)
                        s[db][dd] += ctx_reg[db] * w_reg[dd];
            }
        }
        __syncthreads();  // all reads done before the next tile overwrites SMEM
    }

    #pragma unroll
    for (int db = 0; db < kBqReg; ++db) {
        const int bq_i = bq_base + (lb0 + db);
        if (bq_i >= bq_total) continue;
        const int out_base = (bq_i * num_heads + head_idx) * v_dim;
        #pragma unroll
        for (int dd = 0; dd < kTileReg; ++dd) {
            const int d_i = d_base + (ld0 + dd * kTileThreadsX);
            if (d_i >= v_dim) continue;
            out[out_base + d_i] = static_cast<scalar_t>(s[db][dd]);
        }
    }
}

// Host launcher: official absorbed MLA attention
torch::Tensor launch_mla_attention(
    torch::Tensor q_absorbed,  // [B, Sq, H, kv_rank]
    torch::Tensor q_rope,      // [B, Sq, H, rope_dim]
    torch::Tensor c_kv,        // [B, Sk, kv_rank]
    torch::Tensor pe_cache,    // [B, Sk, rope_dim]
    torch::Tensor W_uv,        // [H, v_dim, kv_rank]
    double scale)              // double so an fp64 request keeps an fp64 scale
{
    int batch_size = q_absorbed.size(0), sq = q_absorbed.size(1);
    int num_heads  = q_absorbed.size(2), kv_rank = q_absorbed.size(3);
    int sk         = c_kv.size(1);
    int rope_dim   = q_rope.size(3);
    int v_dim      = W_uv.size(1);

    const int flat_total = batch_size * sq * num_heads;

    // Stage 3 stages, accumulates and stores intermediates in the accumulate dtype:
    // fp64 in stays fp64 end to end, fp16 promotes to fp32 (see AccType).
    const bool is_f64 = q_absorbed.scalar_type() == torch::kFloat64;
    const auto acc_dtype = is_f64 ? torch::kFloat64 : torch::kFloat32;
    const size_t acc_size = is_f64 ? sizeof(double) : sizeof(float);

    // --- 3c split-K decision (needed up front: it picks the ctx dtype) ---------
    // The 3c grid only parallelises the output dims (flat × kv_rank), so a
    // single-stream decode (B=Sq=1 → 128 flats) yields just 2×4 = 8 blocks and leaves
    // most SMs idle. Split Sk only when the output dims alone cannot fill the GPU;
    // batched decode and prefill have flat_tiles in the hundreds or thousands and stay
    // on split == 1, paying neither the pre-zero nor the atomics.
    const int ctx_flat_tiles = (flat_total + kCtxFlatTile - 1) / kCtxFlatTile;
    const int ctx_r_tiles    = (kv_rank    + kCtxRankTile    - 1) / kCtxRankTile;
    const int ctx_sk_tiles   = (sk         + kCtxSkTile           - 1) / kCtxSkTile;

    int ctx_split = (4 * sm_count()) / (ctx_flat_tiles * ctx_r_tiles);  // a few waves/SM
    if (ctx_split < 1)            ctx_split = 1;
    if (ctx_split > ctx_sk_tiles) ctx_split = ctx_sk_tiles;

    // Give each slice a whole number of SMEM tiles, then recompute the split from that
    // so no slice ends up empty (which would waste a block on nothing).
    const int ctx_chunk_tiles = (ctx_sk_tiles + ctx_split - 1) / ctx_split;
    ctx_split = (ctx_sk_tiles + ctx_chunk_tiles - 1) / ctx_chunk_tiles;
    const int  ctx_sk_chunk = ctx_chunk_tiles * kCtxSkTile;
    const bool ctx_atomic   = ctx_split > 1;

    auto scores = torch::empty({batch_size, sq, num_heads, sk},
                               q_absorbed.options().dtype(acc_dtype));  // [B,Sq,128,Sk]
    // ctx keeps the input dtype; only split-K promotes it to the accumulate dtype, since
    // combining the partials means read-modify-write via atomicAdd.
    auto ctx = torch::empty({batch_size, sq, num_heads, kv_rank},
                            ctx_atomic ? q_absorbed.options().dtype(acc_dtype)
                                       : q_absorbed.options());                        // [B,Sq,128,512]
    auto out = torch::empty({batch_size, sq, num_heads, v_dim}, q_absorbed.options());     // [B,Sq,128,128]
    // Softmax denominator, one scalar per (b,q,h) row: 3c accumulates it, 3d divides by
    // it. Split-K adds partials with atomicAdd, so that path needs a zeroed start.
    auto row_sum = ctx_atomic
        ? torch::zeros({flat_total}, q_absorbed.options().dtype(acc_dtype))
        : torch::empty({flat_total}, q_absorbed.options().dtype(acc_dtype));
    // Row maxima, folded into 3a's epilogue rather than costing a pass of their own.
    // Needed for correctness only when 3c splits Sk (partials must share a scale), but
    // supplied always: knowing the maximum up front lets 3c skip its per-tile max scan,
    // its accumulator rescale, and one of its four barriers. Measured 3c -2.7% against
    // 3a +0.7%.
    // 3a folds the row maxima in as it stores, so this starts at -inf rather than being
    // written wholesale by a second pass over `scores`.
    auto row_max = torch::full({flat_total},
                               -std::numeric_limits<double>::infinity(),
                               q_absorbed.options().dtype(acc_dtype));

    auto stream = at::cuda::getCurrentCUDAStream();

    // 3a: grid.x=flat(B*Sq*H), grid.y=Sk; 16×16 threads → 128×128 scores/block;
    //     576-deep (kv_rank + rope) reduction in SMEM tiles of kScoreRankTile=32
    {
        dim3 threads(kTileThreadsX, kTileThreadsY);  // 16×16 = 256 threads
        dim3 blocks(
            (flat_total + kScoreFlatTile - 1) / kScoreFlatTile,
            (sk         + kScoreSkTile    - 1) / kScoreSkTile);
        // SMEM: qa[128][32] + ck[128][32] acc_t, rows padded to 33 (rope reuses)
        size_t smem = static_cast<size_t>(
            kScoreFlatTile + kScoreSkTile) * kScoreRankPad * acc_size;
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(q_absorbed.scalar_type(), "mla_scores", ([&] {
            using acc_t = acc_t_of<scalar_t>;
            auto kernel = mla_scores_kernel<scalar_t>;
            enable_large_smem(kernel, smem);  // fp64 tiles exceed the default 48 KB
            kernel<<<blocks, threads, smem, stream>>>(
                q_absorbed.data_ptr<scalar_t>(), q_rope.data_ptr<scalar_t>(),
                c_kv.data_ptr<scalar_t>(),       pe_cache.data_ptr<scalar_t>(),
                scores.data_ptr<acc_t>(),
                row_max.data_ptr<acc_t>(),
                batch_size, sq, sk, num_heads, kv_rank, rope_dim,
                static_cast<acc_t>(scale));
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }));
    }

    // 3c: grid.x=flat(B*Sq*H), grid.y=kv_rank, grid.z=split-K slice over Sk;
    //     16×16 → 128×128 ctx/block; Sk reduced in SMEM tiles of kCtxSkTile=32
    {
        // atomicAdd accumulates, so the partials need a zeroed destination
        if (ctx_atomic) ctx.zero_();

        dim3 threads(kTileThreadsX, kTileThreadsY);  // 16×16 = 256 threads
        dim3 blocks(ctx_flat_tiles, ctx_r_tiles, ctx_split);
        // SMEM: p[128][32] + ck[128][32] acc_t rows padded to 33, plus the running
        // (max, sum, rescale) triple, one entry per flat row of the tile
        size_t smem = (static_cast<size_t>(kCtxFlatTile + kCtxRankTile) * kCtxSkPad
                       + 3 * static_cast<size_t>(kCtxFlatTile)) * acc_size;
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(q_absorbed.scalar_type(), "mla_ctx", ([&] {
            using acc_t = acc_t_of<scalar_t>;
            if (ctx_atomic) {
                auto kernel = mla_ctx_kernel<scalar_t, acc_t>;
                enable_large_smem(kernel, smem);
                kernel<<<blocks, threads, smem, stream>>>(
                    scores.data_ptr<acc_t>(), c_kv.data_ptr<scalar_t>(),
                    ctx.data_ptr<acc_t>(),
                    row_max.data_ptr<acc_t>(), row_sum.data_ptr<acc_t>(),
                    batch_size, sq, sk, num_heads, kv_rank, ctx_sk_chunk, true);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            } else {
                auto kernel = mla_ctx_kernel<scalar_t, scalar_t>;
                enable_large_smem(kernel, smem);
                kernel<<<blocks, threads, smem, stream>>>(
                    scores.data_ptr<acc_t>(), c_kv.data_ptr<scalar_t>(),
                    ctx.data_ptr<scalar_t>(),
                    row_max.data_ptr<acc_t>(), row_sum.data_ptr<acc_t>(),
                    batch_size, sq, sk, num_heads, kv_rank, ctx_sk_chunk, false);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
        }));
    }

    // 3d: grid.x=bq tiles, grid.y=v_dim tiles, grid.z=head; 16×16 → bq tile × 128 v/block
    //     kv_rank reduction in SMEM tiles of kOutRankTile=32; W_uv[h] staged per block
    {
        const int bq_total = batch_size * sq;
        dim3 threads(kTileThreadsX, kTileThreadsY);  // 16×16 = 256 threads
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(q_absorbed.scalar_type(), "mla_output", ([&] {
            using acc_t = acc_t_of<scalar_t>;
            auto launch = [&](auto kernel, auto* ctx_ptr, int bq_tile) {
                dim3 blocks((bq_total + bq_tile     - 1) / bq_tile,
                            (v_dim    + kOutVTile   - 1) / kOutVTile,
                            num_heads);
                // SMEM: ctx[bq_tile][32] + w[128][32] acc_t rows padded to 33, plus one
                // reciprocal denominator per bq row
                size_t smem = (static_cast<size_t>(bq_tile + kOutVTile) * kOutRankPad
                               + static_cast<size_t>(bq_tile)) * acc_size;
                enable_large_smem(kernel, smem);
                kernel<<<blocks, threads, smem, stream>>>(
                    ctx_ptr, W_uv.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),
                    row_sum.data_ptr<acc_t>(),
                    bq_total, num_heads, kv_rank, v_dim);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            };
            // Same bq-tile rule as 2a/2c, here against W_uv. Spelled out per ctx dtype
            // rather than macro'd: a #define cannot live inside an AT_DISPATCH argument.
            if (ctx_atomic) {  // 3c promoted ctx to the accumulate dtype for the atomics
                acc_t* p = ctx.data_ptr<acc_t>();
                switch (bq_reg_for(bq_total)) {
                  case 1:  launch(mla_output_smem_kernel<scalar_t, acc_t, 1>, p, 1 * kTileThreadsY); break;
                  case 2:  launch(mla_output_smem_kernel<scalar_t, acc_t, 2>, p, 2 * kTileThreadsY); break;
                  case 4:  launch(mla_output_smem_kernel<scalar_t, acc_t, 4>, p, 4 * kTileThreadsY); break;
                  default: launch(mla_output_smem_kernel<scalar_t, acc_t, kTileReg>, p, kTileReg * kTileThreadsY);
                }
            } else {
                scalar_t* p = ctx.data_ptr<scalar_t>();
                switch (bq_reg_for(bq_total)) {
                  case 1:  launch(mla_output_smem_kernel<scalar_t, scalar_t, 1>, p, 1 * kTileThreadsY); break;
                  case 2:  launch(mla_output_smem_kernel<scalar_t, scalar_t, 2>, p, 2 * kTileThreadsY); break;
                  case 4:  launch(mla_output_smem_kernel<scalar_t, scalar_t, 4>, p, 4 * kTileThreadsY); break;
                  default: launch(mla_output_smem_kernel<scalar_t, scalar_t, kTileReg>, p, kTileReg * kTileThreadsY);
                }
            }
        }));
    }

    // No cudaDeviceSynchronize: all four stages are ordered on the caller's stream.
    return out;  // [B, Sq, 128, 128]
}


// ============================================================================
// Python bindings
// ============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q_absorbed", &launch_q_absorbed,
          "Kernel 2 — fused Q path: wq_b(RMSNorm(x@W_q_a)) → absorb@W_uk + RoPE(Q_rope)");
    m.def("mla_attention", &launch_mla_attention,
          "Kernel 3 — Official absorbed MLA: scores → softmax → ctx@c_kv → @W_uv");
}