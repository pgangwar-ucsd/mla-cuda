#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// DeepSeek-V3 reference dims: hidden=7168, H=128, kv_rank=512, nope=128,
// rope=64, v=128, qk_head=192. B, Sq, Sk vary per launch.

// Every tiled kernel here uses the same shape: 4×4 outputs per thread on a 32×16
// block, with threadIdx.x indexing the *contiguous* output dimension. That last
// point is what makes each epilogue store a single 128 B transaction — a warp is
// 32 threadIdx.x lanes at one threadIdx.y, so it writes 32 adjacent elements of
// one output row. Driving the row index from threadIdx.x instead scatters a warp
// across 16 rows and costs ~6x the store sectors (measured 23.99 vs 4.00).
//
// Tile-constant naming: each kernel is a matmul, so it declares three tiles named
// k<Stage><Axis>Tile after the tensor axis they cut — two output tiles (the dims the
// grid parallelises) and one *reduction* tile (the contracted dim). The reduction
// tile is also the SMEM row width, so it is the constant every XOR swizzle mask must
// use: the swizzle permutes within a row, which needs mask < row width. Masking a
// row index by anything else only works while the two tiles happen to be equal.
//
//   kernel                 output tiles                     reduction tile
//   2ac q_raw_gemm         kQBqTile   x kQWqColTile          kQHiddenTile
//   2b  q_absorb           kQBqTile   x kQRankTile           kQNopeTile
//   3a  mla_scores         kFlatTile  x kScoreSkTile         kScoreRankTile
//   3c  mla_ctx            kFlatTile  x kCtxRankTile         kCtxSkTile
//   3d  mla_output         kOutBqTile x kOutVTile            kOutRankTile
static constexpr int kRegTile        = 4;   // 4×4 outputs per thread, every kernel
static constexpr int kScoreThreadsX  = 32;  // 32×4 = 128 keys (3a) or ranks (3c)
static constexpr int kScoreThreadsY  = 16;  // 16×4 = 64 flats per block
static constexpr int kFlatTile       = kScoreThreadsY * kRegTile;  // 64 flats (3a, 3c)
static constexpr int kScoreSkTile    = kScoreThreadsX * kRegTile;  // 128 keys out (3a)
static constexpr int kCtxRankTile    = kScoreThreadsX * kRegTile;  // 128 kv_rank out (3c)
static constexpr int kScoreRankTile  = 64;  // reduce over kv_rank (3a)
static constexpr int kCtxSkTile      = 64;  // reduce over Sk      (3c)
static constexpr int kRopeDim        = 64;  // qk_rope_head_dim (compile-time for unroll)

// Kernel 2 (Q-prep): same 32×16 / 4×4 shape. Blocking over bq (not one block per
// (b,s,h)) is what keeps W_q traffic bounded: a (b,s,h)-per-block layout re-reads
// head h's 5.5 MB W_q panel once per (b,s), i.e. 90 GB at B=128.
static constexpr int kQThreadsX   = 32;  // 32×4 = 128 outputs along the contiguous dim
static constexpr int kQThreadsY   = 16;  // 16×4 = 64 bq per block
static constexpr int kQBqTile     = kQThreadsY * kRegTile;  // 64 bq out (2ac and 2b)
static constexpr int kQWqColTile  = kQThreadsX * kRegTile;  // 128 W_q columns out (2ac)
static constexpr int kQRankTile   = kQThreadsX * kRegTile;  // 128 kv_rank out (2b)
static constexpr int kQHiddenTile = 64;  // reduce over hidden   (2ac)
static constexpr int kQNopeTile   = 64;  // reduce over nope_dim (2b)
static constexpr int kRopeThreads = 256;
// 3b: many rows → small blocks maximise SMs busy; few rows → wide blocks so a
// single-stream decode (B=Sq=1 gives only H=128 rows) still fills the machine.
static constexpr int kSoftmaxThreadsSmall = 256;
static constexpr int kSoftmaxThreadsLarge = 1024;
// Kernel 3d: one head per block so W_uv[h] is staged once and reused across bq.
// x indexes v_dim (contiguous in `out`) so the epilogue stores are coalesced.
static constexpr int kOutThreadsX    = 32;  // 32×4 = 128 v_dims per block
static constexpr int kOutThreadsY    = 16;  // 16×4 = 64  bq per block
static constexpr int kOutVTile       = kOutThreadsX * kRegTile;  // 128 v_dim out
static constexpr int kOutBqTile      = kOutThreadsY * kRegTile;  // 64 bq out
static constexpr int kOutRankTile    = 64;  // reduce over kv_rank (3d)
// 512 threads × 48 KB SMEM: without a cap nvcc spends 160+ regs on the 4×4 register
// strip, and 512×160 overflows the 64K register file ("too many resources"). Asking
// for 2 resident blocks pins it at 64 regs, which also fits 2 blocks of SMEM per SM.
static constexpr int kTileBlockThreads = 512;
static constexpr int kTileMinBlocksPerSM = 2;

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


// ============================================================================
// KERNEL 2: Q absorbed path — 2a (x @ W_q) → 2b (RoPE) → 2c (absorb @ W_uk)
//
//  2a. q_raw = x @ W_q            [B*Sq, hidden] @ [hidden, H*192] → [B*Sq, H*192]
//  2b.  RoPE on the rope half of q_raw            → Q_rope     [B,Sq,H,64]
//  2c.  absorb the nope half against W_uk[h]      → Q_absorbed [B,Sq,H,512]
//

// --- 2a. q_raw = x @ W_q: 4×4 tile/thread; grid (W_q col tiles, bq tiles) ---
//   grid.x: n ∈ [0, H*qk_head_dim)   — 128 columns per block (contiguous in q_raw)
//   grid.y: bq = b*Sq + s            — 64 bq per block
//   block: 32×16 → each thread 4 cols × 4 bq; hidden reduced in SMEM tiles of 64
//   SMEM [lb][k^lb] / [ln][k^ln] with col XOR swizzle; k staged fast for coalescing
//   Reduction is k-outer: 4 x + 4 w loads → 16 FMAs, so 0.5 SMEM loads per FMA
template <typename scalar_t>
__global__ void __launch_bounds__(kTileBlockThreads, kTileMinBlocksPerSM)
q_raw_gemm_kernel(
    const scalar_t* __restrict__ x,      // [B*Sq, hidden]
    const scalar_t* __restrict__ W_q,    // [hidden, H*qk_head_dim]
    acc_t_of<scalar_t>* __restrict__ q_raw,  // [B*Sq, H*qk_head_dim]
    int bq_total,
    int hidden_dim,
    int n_total)
{
    using acc_t = acc_t_of<scalar_t>;

    const int n_base  = blockIdx.x * kQWqColTile;
    const int bq_base = blockIdx.y * kQBqTile;

    acc_t* smem   = dynamic_smem<acc_t>();
    acc_t* x_smem = smem;                             // [kQBqTile][kQHiddenTile]
    acc_t* w_smem = x_smem + kQBqTile * kQHiddenTile; // [kQWqColTile][kQHiddenTile]

    const int coop_stride = blockDim.x * blockDim.y;

    acc_t s[kRegTile][kRegTile];
    #pragma unroll
    for (int db = 0; db < kRegTile; ++db)
        #pragma unroll
        for (int dn = 0; dn < kRegTile; ++dn)
            s[db][dn] = acc_t(0);

    // threadIdx.y → bq (the row), threadIdx.x → n (contiguous in q_raw): a warp
    int lb[kRegTile], ln[kRegTile];
    #pragma unroll
    for (int db = 0; db < kRegTile; ++db)
        lb[db] = threadIdx.y * kRegTile + db;
    #pragma unroll
    for (int dn = 0; dn < kRegTile; ++dn)
        ln[dn] = threadIdx.x + dn * blockDim.x;

    // XOR swizzle masks. Both are masked by the *row width* of the array they index —
    // kQHiddenTile for x_smem and w_smem alike, since that is the inner extent of both. The
    // swizzle has to permute within one row, which requires mask < row width. Do not
    // simplify the bq side to a bare lb[]: that is only equivalent while
    // kQBqTile <= kQHiddenTile, and would silently spill into the next row if the two tiles
    // were ever retuned apart.
    // Bitwise ANDing a number with 63 is a lightning-fast way of calculating modulo 64
    int lb_m[kRegTile], ln_m[kRegTile];
    #pragma unroll
    for (int db = 0; db < kRegTile; ++db)
        lb_m[db] = lb[db] & (kQHiddenTile - 1);
    #pragma unroll
    for (int dn = 0; dn < kRegTile; ++dn)
        ln_m[dn] = ln[dn] & (kQHiddenTile - 1);

    for (int k0 = 0; k0 < hidden_dim; k0 += kQHiddenTile) {
        const int tile_len =
            (k0 + kQHiddenTile <= hidden_dim) ? kQHiddenTile : (hidden_dim - k0);

        // stage x[bq_base:+64, k0:+tile) — k contiguous → k_local = idx % tile_len
        const int x_elems = kQBqTile * tile_len;
        for (int idx = threadIdx.x + threadIdx.y * blockDim.x;
             idx < x_elems; idx += coop_stride) {
            const int lb_s    = idx / tile_len;
            const int k_local = idx % tile_len;
            const int bq_s    = bq_base + lb_s;
            if (bq_s < bq_total)
                x_smem[lb_s * kQHiddenTile + (k_local ^ (lb_s & (kQHiddenTile - 1)))] =
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
                w_smem[ln_s * kQHiddenTile + (k_local ^ (ln_s & (kQHiddenTile - 1)))] =
                    static_cast<acc_t>(W_q[(k0 + k_local) * n_total + n_s]);
        }
        __syncthreads();

        // k-outer register strip. Lanes past bq_total / n_total read unwritten SMEM;
        // their accumulators are dropped at the store, so no inner predicate.
        if (tile_len == kQHiddenTile) {
            #pragma unroll 8
            for (int k_local = 0; k_local < kQHiddenTile; ++k_local) {
                acc_t x_reg[kRegTile], w_reg[kRegTile];
                #pragma unroll
                for (int db = 0; db < kRegTile; ++db)
                    x_reg[db] = x_smem[lb[db] * kQHiddenTile + (k_local ^ lb_m[db])];
                #pragma unroll
                for (int dn = 0; dn < kRegTile; ++dn)
                    w_reg[dn] = w_smem[ln[dn] * kQHiddenTile + (k_local ^ ln_m[dn])];
                #pragma unroll
                for (int db = 0; db < kRegTile; ++db)
                    #pragma unroll
                    for (int dn = 0; dn < kRegTile; ++dn)
                        s[db][dn] += x_reg[db] * w_reg[dn];
            }
        } else {
            // ragged final hidden tile: same body, runtime trip count
            for (int k_local = 0; k_local < tile_len; ++k_local) {
                acc_t x_reg[kRegTile], w_reg[kRegTile];
                #pragma unroll
                for (int db = 0; db < kRegTile; ++db)
                    x_reg[db] = x_smem[lb[db] * kQHiddenTile + (k_local ^ lb_m[db])];
                #pragma unroll
                for (int dn = 0; dn < kRegTile; ++dn)
                    w_reg[dn] = w_smem[ln[dn] * kQHiddenTile + (k_local ^ ln_m[dn])];
                #pragma unroll
                for (int db = 0; db < kRegTile; ++db)
                    #pragma unroll
                    for (int dn = 0; dn < kRegTile; ++dn)
                        s[db][dn] += x_reg[db] * w_reg[dn];
            }
        }
        __syncthreads();  // all reads done before the next tile overwrites SMEM
    }

    #pragma unroll
    for (int db = 0; db < kRegTile; ++db) {
        const int bq_i = bq_base + lb[db];
        if (bq_i >= bq_total) continue;
        const int row_base = bq_i * n_total;
        #pragma unroll
        for (int dn = 0; dn < kRegTile; ++dn) {
            const int n_i = n_base + ln[dn];
            if (n_i >= n_total) continue;
            q_raw[row_base + n_i] = s[db][dn];
        }
    }
}

// --- 2b. RoPE on the rope half of q_raw → Q_rope [B, Sq, H, rope_dim] ---
// One thread per (bq, h, adjacent pair). The angle is built in the accumulate dtype,
// so an fp64 request gets genuinely fp64 trig. (The PyTorch reference
// exhaustive_benchmark_suite.apply_rope always builds freqs/angles in fp32 and only
// casts cos/sin down afterwards, so an fp64 comparison against *it* bottoms out at
// ~1e-8 — that is the reference's precision, not this kernel's.)
template <typename scalar_t>
__global__ void __launch_bounds__(kRopeThreads)
q_rope_kernel(
    const acc_t_of<scalar_t>* __restrict__ q_raw,  // [B*Sq, H*qk_head_dim]
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
        const int s     = bq % sq;          // position within the sequence
        const int d     = half * 2;

        const int src = bq * (num_heads * qk_head_dim) + h * qk_head_dim + nope_dim + d;
        const acc_t even = q_raw[src];
        const acc_t odd  = q_raw[src + 1];

        const acc_t inv_freq = acc_t(1) / pow(acc_t(10000),
                                   static_cast<acc_t>(d) / static_cast<acc_t>(rope_dim));
        const acc_t angle = static_cast<acc_t>(s) * inv_freq;
        const acc_t c  = cos(angle);
        const acc_t sn = sin(angle);

        const int dst = (bq * num_heads + h) * rope_dim + d;
        q_rope[dst]     = static_cast<scalar_t>(even * c  - odd * sn);
        q_rope[dst + 1] = static_cast<scalar_t>(even * sn + odd * c);
    }
}

// --- 2c. Q_absorbed = q_nope @ W_uk[h]: 4×4 tile/thread; grid (bq, r, head) ---
//   One head per block-column so W_uk[h] is staged once and amortised over 64 bq,
//   exactly like 3d does with W_uv. The nope half is read straight out of q_raw at
//   stride qk_head_dim — no repacking pass.
//   grid.x: bq tiles (64 bq/block)   grid.y: r tiles (128 ranks/block)   grid.z: h
//   block: 32×16 → each thread 4 ranks × 4 bq; nope reduced in SMEM tiles of 64
template <typename scalar_t>
__global__ void __launch_bounds__(kTileBlockThreads, kTileMinBlocksPerSM)
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

    const int bq_base  = blockIdx.x * kQBqTile;
    const int r_base   = blockIdx.y * kQRankTile;
    const int head_idx = blockIdx.z;

    acc_t* smem    = dynamic_smem<acc_t>();
    acc_t* qn_smem = smem;                            // [kQBqTile][kQNopeTile]
    acc_t* w_smem  = qn_smem + kQBqTile * kQNopeTile; // [kQRankTile][kQNopeTile]

    const int qn_head_base = head_idx * qk_head_dim;        // nope half starts at +0
    const int w_head_base  = head_idx * nope_dim * kv_rank;
    const int coop_stride  = blockDim.x * blockDim.y;

    acc_t s[kRegTile][kRegTile];
    #pragma unroll
    for (int db = 0; db < kRegTile; ++db)
        #pragma unroll
        for (int dr = 0; dr < kRegTile; ++dr)
            s[db][dr] = acc_t(0);

    // Only the SMEM row indices live across the reduction (see 2a / 3a).
    int lb[kRegTile], lr[kRegTile];
    #pragma unroll
    for (int db = 0; db < kRegTile; ++db)
        lb[db] = threadIdx.y * kRegTile + db;
    #pragma unroll
    for (int dr = 0; dr < kRegTile; ++dr)
        lr[dr] = threadIdx.x + dr * blockDim.x;

    // Both masked by kQNopeTile — the row width of qn_smem and w_smem alike (see 2ac).
    int lb_m[kRegTile], lr_m[kRegTile];
    #pragma unroll
    for (int db = 0; db < kRegTile; ++db)
        lb_m[db] = lb[db] & (kQNopeTile - 1);
    #pragma unroll
    for (int dr = 0; dr < kRegTile; ++dr)
        lr_m[dr] = lr[dr] & (kQNopeTile - 1);

    for (int n0 = 0; n0 < nope_dim; n0 += kQNopeTile) {
        const int tile_len =
            (n0 + kQNopeTile <= nope_dim) ? kQNopeTile : (nope_dim - n0);

        // stage q_raw[bq, h, n0:+tile) — n contiguous → n_local = idx % tile_len
        const int qn_elems = kQBqTile * tile_len;
        for (int idx = threadIdx.x + threadIdx.y * blockDim.x;
             idx < qn_elems; idx += coop_stride) {
            const int lb_s    = idx / tile_len;
            const int n_local = idx % tile_len;
            const int bq_s    = bq_base + lb_s;
            if (bq_s < bq_total)
                qn_smem[lb_s * kQNopeTile + (n_local ^ (lb_s & (kQNopeTile - 1)))] =
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
                w_smem[lr_s * kQNopeTile + (n_local ^ (lr_s & (kQNopeTile - 1)))] =
                    static_cast<acc_t>(
                        W_uk[w_head_base + (n0 + n_local) * kv_rank + r_s]);
        }
        __syncthreads();

        if (tile_len == kQNopeTile) {
            #pragma unroll 8
            for (int n_local = 0; n_local < kQNopeTile; ++n_local) {
                acc_t qn_reg[kRegTile], w_reg[kRegTile];
                #pragma unroll
                for (int db = 0; db < kRegTile; ++db)
                    qn_reg[db] = qn_smem[lb[db] * kQNopeTile + (n_local ^ lb_m[db])];
                #pragma unroll
                for (int dr = 0; dr < kRegTile; ++dr)
                    w_reg[dr] = w_smem[lr[dr] * kQNopeTile + (n_local ^ lr_m[dr])];
                #pragma unroll
                for (int db = 0; db < kRegTile; ++db)
                    #pragma unroll
                    for (int dr = 0; dr < kRegTile; ++dr)
                        s[db][dr] += qn_reg[db] * w_reg[dr];
            }
        } else {
            // ragged final nope tile: same body, runtime trip count
            for (int n_local = 0; n_local < tile_len; ++n_local) {
                acc_t qn_reg[kRegTile], w_reg[kRegTile];
                #pragma unroll
                for (int db = 0; db < kRegTile; ++db)
                    qn_reg[db] = qn_smem[lb[db] * kQNopeTile + (n_local ^ lb_m[db])];
                #pragma unroll
                for (int dr = 0; dr < kRegTile; ++dr)
                    w_reg[dr] = w_smem[lr[dr] * kQNopeTile + (n_local ^ lr_m[dr])];
                #pragma unroll
                for (int db = 0; db < kRegTile; ++db)
                    #pragma unroll
                    for (int dr = 0; dr < kRegTile; ++dr)
                        s[db][dr] += qn_reg[db] * w_reg[dr];
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int db = 0; db < kRegTile; ++db) {
        const int bq_i = bq_base + lb[db];
        if (bq_i >= bq_total) continue;
        const int out_base = (bq_i * num_heads + head_idx) * kv_rank;
        #pragma unroll
        for (int dr = 0; dr < kRegTile; ++dr) {
            const int r_i = r_base + lr[dr];
            if (r_i >= kv_rank) continue;
            q_absorbed[out_base + r_i] = static_cast<scalar_t>(s[db][dr]);
        }
    }
}

// Host launcher: CUDA Q-prep → {q_absorbed [B,Sq,H,kv_rank], q_rope [B,Sq,H,rope]}
std::vector<torch::Tensor> launch_q_absorbed(
    torch::Tensor x,     // [B, Sq, hidden]
    torch::Tensor W_q,   // [hidden, H*(nope+rope)]
    torch::Tensor W_uk,  // [H, nope, kv_rank]
    int nope_dim,
    int rope_dim)
{
    auto xc    = x.contiguous();
    auto W_qc  = W_q.contiguous();
    auto W_ukc = W_uk.contiguous();

    const int batch_size  = static_cast<int>(xc.size(0));
    const int sq          = static_cast<int>(xc.size(1));
    const int hidden_dim  = static_cast<int>(xc.size(2));
    const int num_heads   = static_cast<int>(W_ukc.size(0));
    const int kv_rank     = static_cast<int>(W_ukc.size(2));
    const int qk_head_dim = nope_dim + rope_dim;
    const int bq_total    = batch_size * sq;
    const int n_total     = num_heads * qk_head_dim;

    TORCH_CHECK(W_qc.size(0) == hidden_dim && W_qc.size(1) == n_total,
                "W_q must be [hidden, H*(nope+rope)]");

    // Q-prep accumulates and stages in the accumulate dtype: fp64 in stays fp64,
    // fp16 promotes to fp32 (a 7168-long dot in half would lose far more than the
    // narrower staging saves). Outputs are always the input dtype.
    const bool is_f64 = xc.scalar_type() == torch::kFloat64;
    const auto acc_dtype = is_f64 ? torch::kFloat64 : torch::kFloat32;
    const size_t acc_size = is_f64 ? sizeof(double) : sizeof(float);

    auto q_raw      = torch::empty({bq_total, n_total}, xc.options().dtype(acc_dtype));
    auto q_absorbed = torch::empty({batch_size, sq, num_heads, kv_rank}, xc.options());
    auto q_rope_out = torch::empty({batch_size, sq, num_heads, rope_dim}, xc.options());

    auto stream = at::cuda::getCurrentCUDAStream();

    // 2a: grid.x=W_q col tiles, grid.y=bq tiles; 32×16 threads → 64 bq × 128 cols
    {
        dim3 threads(kQThreadsX, kQThreadsY);  // 32×16 = 512 threads
        dim3 blocks((n_total  + kQWqColTile  - 1) / kQWqColTile,
                    (bq_total + kQBqTile - 1) / kQBqTile);
        // SMEM: x[64][64] + w[128][64] acc_t with XOR col swizzle
        size_t smem = static_cast<size_t>(
            kQBqTile * kQHiddenTile + kQWqColTile * kQHiddenTile) * acc_size;
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(xc.scalar_type(), "q_raw_gemm", ([&] {
            using acc_t = acc_t_of<scalar_t>;
            auto kernel = q_raw_gemm_kernel<scalar_t>;
            enable_large_smem(kernel, smem);  // fp64 tiles exceed the default 48 KB
            kernel<<<blocks, threads, smem, stream>>>(
                xc.data_ptr<scalar_t>(), W_qc.data_ptr<scalar_t>(),
                q_raw.data_ptr<acc_t>(),
                bq_total, hidden_dim, n_total);
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
                q_raw.data_ptr<acc_t>(), q_rope_out.data_ptr<scalar_t>(),
                bq_total, sq, num_heads, nope_dim, rope_dim, qk_head_dim);
        }));
    }

    // 2c: grid.x=bq tiles, grid.y=kv_rank tiles, grid.z=head; 32×16 → 64 bq × 128 r
    {
        dim3 threads(kQThreadsX, kQThreadsY);  // 32×16 = 512 threads
        dim3 blocks((bq_total + kQBqTile - 1) / kQBqTile,
                    (kv_rank  + kQRankTile  - 1) / kQRankTile,
                    num_heads);
        // SMEM: qn[64][64] + w[128][64] acc_t with XOR col swizzle
        size_t smem = static_cast<size_t>(
            kQBqTile * kQNopeTile + kQRankTile * kQNopeTile) * acc_size;
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(xc.scalar_type(), "q_absorb", ([&] {
            using acc_t = acc_t_of<scalar_t>;
            auto kernel = q_absorb_kernel<scalar_t>;
            enable_large_smem(kernel, smem);
            kernel<<<blocks, threads, smem, stream>>>(
                q_raw.data_ptr<acc_t>(), W_ukc.data_ptr<scalar_t>(),
                q_absorbed.data_ptr<scalar_t>(),
                bq_total, num_heads, nope_dim, kv_rank, qk_head_dim);
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
//      16×32 threads/block → 64×128 scores/block; each thread → 4×4 tile (q × c_kv)
//      kv_rank=512 contracted in SMEM tiles of kScoreRankTile=64 (block-level reduction)
//      scores_nope: [B,Sq,H,512] @ [B,Sk,512]^T → [B,Sq,H,Sk]
//      scores_rope: [B,Sq,H,64]  @ [B,Sk,64]^T  → [B,Sq,H,Sk]
//      merge + /sqrt(192)
//  3b. softmax → attn [B,Sq,H,Sk]   (one block per (B*Sq, H) row)
//  3c. ctx — same 16×32 / 4×4 tiling as 3a; grid.x=flat, grid.y=kv_rank
//      64 flats × 128 ranks/block; Sk contracted in SMEM tiles of kCtxSkTile=64
//      attn[B,Sq,H,Sk] @ c_kv[B,Sk,512] → ctx[B,Sq,H,512]
//  3d. out — 32×16 / 4×4 tiling; grid.x=bq tiles, grid.y=v tiles, grid.z=head
//      one head per block so W_uv[h] is staged once and reused across 64 bq
//      ctx[B,Sq,H,512] × W_uv[H,128,512] → out[B,Sq,H,128]
// ============================================================================

// --- 3a. scores: 4×4 output tile per thread; grid (flat B*Sq*H, Sk) ---
//   grid.x: flat = b*(Sq*128) + q*128 + h     ∈ [0, B*Sq*128)  — 64 flats/block
//   grid.y: k ∈ [0, Sk)                                        — 128 keys/block
//   block: 32×16 threads → each thread 4 keys × 4 flats (64×128 / block)
//   threadIdx.x owns k (contiguous in `scores`), threadIdx.y owns flat: a warp
//   writes 32 adjacent keys of one score row, so the epilogue is one 128 B store
//   SMEM [lf][r] / [lk][r] with col XOR swizzle (no padding): bank = (r^row)%32
//   Staging: r (or rope d) is idx% — matches HBM contiguous dim for coalescing
//   Reduction is r-outer: 4 qa + 4 ck loads → 16 FMAs, so 0.5 SMEM loads per FMA
template <typename scalar_t>
__global__ void __launch_bounds__(kTileBlockThreads, kTileMinBlocksPerSM)
mla_scores_kernel(
    const scalar_t* __restrict__ q_absorbed, // [B, Sq, H, kv_rank=512]
    const scalar_t* __restrict__ q_rope,     // [B, Sq, H, rope_dim=64]
    const scalar_t* __restrict__ c_kv,       // [B, Sk, kv_rank=512]
    const scalar_t* __restrict__ pe_cache,   // [B, Sk, rope_dim=64]  head-shared
    acc_t_of<scalar_t>* __restrict__ scores, // [B, Sq, H, Sk]  in the accumulate dtype
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
    const int flat_base  = blockIdx.x * kFlatTile;
    const int k0         = blockIdx.y * kScoreSkTile;

    // qa_smem[kFlatTile][kScoreRankTile]; ck_smem[kScoreSkTile][kScoreRankTile]
    // — both reused as qr_smem/pe_smem after phase 1
    acc_t* smem    = dynamic_smem<acc_t>();
    acc_t* qa_smem = smem;
    acc_t* ck_smem = qa_smem + kFlatTile * kScoreRankTile;

    // batch for c_kv / pe_cache rows (64 flats/block ⊂ one batch when block < Sq*H)
    const int batch_idx = min(flat_base, max(flat_total - 1, 0)) / (sq * num_heads);
    const int ck_batch_stride = batch_idx * sk * kv_rank;
    const int pe_batch_stride = batch_idx * sk * rope_dim;

    const int coop_stride = blockDim.x * blockDim.y;
    const acc_t inv_scale = acc_t(1) / scale;

    // Per-thread output tile; rope later accumulates into the same regs
    acc_t s[kRegTile][kRegTile];
    #pragma unroll
    for (int di = 0; di < kRegTile; ++di)
        #pragma unroll
        for (int dk = 0; dk < kRegTile; ++dk)
            s[di][dk] = acc_t(0);

    // SMEM row indices. Deliberately *not* accompanied by precomputed flat[]/k_idx[]/
    // valid_*[] arrays: those are epilogue-only, and with __launch_bounds__(512, 2)
    // pinning this kernel at exactly 64 registers (512 threads x 64 x 2 blocks fills
    // the whole 64K register file), every value kept alive across the reduction is a
    // register the scheduler cannot use to prefetch the next SMEM operand.
    int lf[kRegTile], lk[kRegTile];
    #pragma unroll
    for (int di = 0; di < kRegTile; ++di)
        lf[di] = threadIdx.y * kRegTile + di;
    #pragma unroll
    for (int dk = 0; dk < kRegTile; ++dk)
        lk[dk] = threadIdx.x + dk * blockDim.x;

    // XOR swizzle masks for the nope phase, both taken against kScoreRankTile — the row
    // width of qa_smem and ck_smem alike. The swizzle must permute within a row, so the
    // mask has to be (row width - 1); masking the flat side by anything wider (or not at
    // all) only happens to work while kFlatTile <= kScoreRankTile.
    int lf_m[kRegTile], lk_m[kRegTile];
    #pragma unroll
    for (int di = 0; di < kRegTile; ++di)
        lf_m[di] = lf[di] & (kScoreRankTile - 1);
    #pragma unroll
    for (int dk = 0; dk < kRegTile; ++dk)
        lk_m[dk] = lk[dk] & (kScoreRankTile - 1);

    // ---- Phase 1: scores_nope — fuse qa+ck SMEM loads, one sync per kv_rank tile ----
    for (int r0 = 0; r0 < kv_rank; r0 += kScoreRankTile) {
        const int tile_len =
            (r0 + kScoreRankTile <= kv_rank) ? kScoreRankTile : (kv_rank - r0);

        // cooperative: stage Q_absorbed → qa_smem[lf][r^lf]
        // q_absorbed[flat, r] is contiguous in r → r_local must be idx% so warps coalesce
        const int qa_elems = kFlatTile * tile_len;
        for (int idx = threadIdx.x + threadIdx.y * blockDim.x;
             idx < qa_elems; idx += coop_stride) {
            const int lf_s    = idx / tile_len;
            const int r_local = idx % tile_len;
            const int flat_s  = flat_base + lf_s;
            if (flat_s < flat_total)
                qa_smem[lf_s * kScoreRankTile + (r_local ^ (lf_s & (kScoreRankTile - 1)))] =
                    static_cast<acc_t>(q_absorbed[flat_s * kv_rank + r0 + r_local]);
        }

        // cooperative: stage c_kv → ck_smem[lk][r^lk]  (r contiguous in c_kv[k, r])
        const int ck_elems = kScoreSkTile * tile_len;
        for (int idx = threadIdx.x + threadIdx.y * blockDim.x;
             idx < ck_elems; idx += coop_stride) {
            const int lk_s    = idx / tile_len;
            const int r_local = idx % tile_len;
            const int k_s     = k0 + lk_s;
            if (k_s < sk)
                ck_smem[lk_s * kScoreRankTile + (r_local ^ (lk_s & (kScoreRankTile - 1)))] =
                    static_cast<acc_t>(c_kv[ck_batch_stride + k_s * kv_rank + r0 + r_local]);
        }
        __syncthreads();  // loads done before any thread reads this tile

        // r-outer register strip: 4 qa + 4 ck SMEM loads feed 16 FMAs (0.5 loads/FMA).
        // Lanes past flat_total / sk read unwritten SMEM; their accumulators are dropped
        // at the store, so no per-iteration bounds predicate is needed here.
        // Capped at 8, matching 3c/3d: a full 64-wide unroll asks for more live values
        // than the 64-register budget holds and spills ~1.2 KB per thread to local.
        if (tile_len == kScoreRankTile) {
            #pragma unroll 8
            for (int r_local = 0; r_local < kScoreRankTile; ++r_local) {
                acc_t qa_reg[kRegTile], ck_reg[kRegTile];
                #pragma unroll
                for (int di = 0; di < kRegTile; ++di)
                    qa_reg[di] = qa_smem[lf[di] * kScoreRankTile + (r_local ^ lf_m[di])];
                #pragma unroll
                for (int dk = 0; dk < kRegTile; ++dk)
                    ck_reg[dk] = ck_smem[lk[dk] * kScoreRankTile + (r_local ^ lk_m[dk])];
                #pragma unroll
                for (int di = 0; di < kRegTile; ++di)
                    #pragma unroll
                    for (int dk = 0; dk < kRegTile; ++dk)
                        s[di][dk] += qa_reg[di] * ck_reg[dk];
            }
        } else {
            // ragged final kv_rank tile: same body, runtime trip count
            for (int r_local = 0; r_local < tile_len; ++r_local) {
                acc_t qa_reg[kRegTile], ck_reg[kRegTile];
                #pragma unroll
                for (int di = 0; di < kRegTile; ++di)
                    qa_reg[di] = qa_smem[lf[di] * kScoreRankTile + (r_local ^ lf_m[di])];
                #pragma unroll
                for (int dk = 0; dk < kRegTile; ++dk)
                    ck_reg[dk] = ck_smem[lk[dk] * kScoreRankTile + (r_local ^ lk_m[dk])];
                #pragma unroll
                for (int di = 0; di < kRegTile; ++di)
                    #pragma unroll
                    for (int dk = 0; dk < kRegTile; ++dk)
                        s[di][dk] += qa_reg[di] * ck_reg[dk];
            }
        }
        __syncthreads();  // all reads done before next tile (or rope) overwrites SMEM
    }

    // ---- Phase 2: scores_rope — accumulate into same s[][] (no second tile regs) ----
    // The SMEM side is addressed with the compile-time kRopeDim throughout, matching the
    // reduction loop below; `rope_dim` is used only for the HBM strides. The launcher
    // enforces rope_dim == kRopeDim, so the two agree — but they must be *written* the
    // same way, or a mismatch would stage with one row stride and read back with another.
    acc_t* qr_smem = qa_smem;  // [64][kRopeDim] with XOR swizzle
    acc_t* pe_smem = ck_smem;  // [128][kRopeDim] with XOR swizzle

    const int qr_elems = kFlatTile * kRopeDim;
    for (int idx = threadIdx.x + threadIdx.y * blockDim.x;
         idx < qr_elems; idx += coop_stride) {
        // q_rope[flat, d] contiguous in d
        const int lf_s = idx / kRopeDim;
        const int d    = idx % kRopeDim;
        const int flat_s = flat_base + lf_s;
        if (flat_s < flat_total)
            qr_smem[lf_s * kRopeDim + (d ^ (lf_s & (kRopeDim - 1)))] =
                static_cast<acc_t>(q_rope[flat_s * rope_dim + d]);
    }

    const int pe_elems = kScoreSkTile * kRopeDim;
    for (int idx = threadIdx.x + threadIdx.y * blockDim.x;
         idx < pe_elems; idx += coop_stride) {
        // pe_cache[k, d] contiguous in d
        const int lk_s = idx / kRopeDim;
        const int d    = idx % kRopeDim;
        const int k_s  = k0 + lk_s;
        if (k_s < sk)
            pe_smem[lk_s * kRopeDim + (d ^ (lk_s & (kRopeDim - 1)))] =
                static_cast<acc_t>(pe_cache[pe_batch_stride + k_s * rope_dim + d]);
    }
    __syncthreads();

    // qr_smem / pe_smem have row width kRopeDim, not kScoreRankTile, so the nope-phase
    // masks do not carry over — they are only interchangeable while the two constants
    // happen to be equal.
    int lf_rm[kRegTile], lk_rm[kRegTile];
    #pragma unroll
    for (int di = 0; di < kRegTile; ++di)
        lf_rm[di] = lf[di] & (kRopeDim - 1);
    #pragma unroll
    for (int dk = 0; dk < kRegTile; ++dk)
        lk_rm[dk] = lk[dk] & (kRopeDim - 1);

    // d-outer register strip; rope_dim is a full compile-time tile, so no ragged path
    #pragma unroll 8
    for (int d = 0; d < kRopeDim; ++d) {
        acc_t qr_reg[kRegTile], pe_reg[kRegTile];
        #pragma unroll
        for (int di = 0; di < kRegTile; ++di)
            qr_reg[di] = qr_smem[lf[di] * kRopeDim + (d ^ lf_rm[di])];
        #pragma unroll
        for (int dk = 0; dk < kRegTile; ++dk)
            pe_reg[dk] = pe_smem[lk[dk] * kRopeDim + (d ^ lk_rm[dk])];
        #pragma unroll
        for (int di = 0; di < kRegTile; ++di)
            #pragma unroll
            for (int dk = 0; dk < kRegTile; ++dk)
                s[di][dk] += qr_reg[di] * pe_reg[dk];
    }

    // Bounds are recomputed here rather than carried in registers through the loops.
    #pragma unroll
    for (int di = 0; di < kRegTile; ++di) {
        const int flat_i = flat_base + lf[di];
        if (flat_i >= flat_total) continue;
        #pragma unroll
        for (int dk = 0; dk < kRegTile; ++dk) {
            const int k_i = k0 + lk[dk];
            if (k_i >= sk) continue;
            scores[flat_i * sk + k_i] = s[di][dk] * inv_scale;
        }
    }
}

// --- 3b. Softmax over the Sk dimension, in-place on scores ---
// One block per (b, q, h) row. Online (flash-style) formulation: pass 1 carries a
// running (max, sum) so the row is read once instead of twice, and pass 2 writes
// the normalised value directly. That is 3 x Sk of HBM traffic instead of the 5 x Sk
// a separate max / exp+sum / normalize triple costs, on a kernel that already runs
// at ~75% of peak DRAM throughput.
//
// Combining two (max, sum) partials is the standard rescale:
//     M = max(m1, m2);  L = l1*exp(m1 - M) + l2*exp(m2 - M)
// A partial that saw no elements is (-inf, 0) — that is why the M == -inf case is
// branched around: -inf - -inf is NaN, and NaN*0 would poison the reduction.

// Merge partial (m, l) with (om, ol), in place.
template <typename acc_t>
__device__ __forceinline__ void softmax_merge(acc_t& m, acc_t& l, acc_t om, acc_t ol) {
    const acc_t nm = fmax(m, om);
    if (nm == -INFINITY) {
        l = acc_t(0);          // both partials empty; leave m at -inf
    } else {
        // At most one side is -inf here, and its l is 0, so exp(-inf)*0 == 0.
        l = l * exp(m - nm) + ol * exp(om - nm);
    }
    m = nm;
}

template <typename acc_t>
__global__ void mla_softmax_kernel(
    acc_t* __restrict__ scores,  // [B, Sq, H, Sk]  — modified in place
    int sk)
{
    acc_t* row = scores + static_cast<size_t>(blockIdx.x) * sk;  // one (b,q,h) row

    // ---- pass 1: running max + running sum in a single read of the row ----
    acc_t m = -INFINITY, l = acc_t(0);
    for (int k = threadIdx.x; k < sk; k += blockDim.x) {
        const acc_t v = row[k];
        if (v > m) {
            l *= exp(m - v);   // m may be -inf on the first element: exp(-inf) == 0
            m = v;
        }
        l += exp(v - m);
    }

    // ---- block reduction of (m, l): warp shuffles, then one pass over warps ----
    #pragma unroll
    for (int off = warpSize / 2; off > 0; off >>= 1) {
        const acc_t om = __shfl_down_sync(0xffffffffu, m, off);
        const acc_t ol = __shfl_down_sync(0xffffffffu, l, off);
        softmax_merge(m, l, om, ol);
    }

    __shared__ acc_t m_smem[32], l_smem[32];  // <= 1024 threads → <= 32 warps
    const int lane = threadIdx.x & (warpSize - 1);
    const int warp = threadIdx.x / warpSize;
    const int warps = (blockDim.x + warpSize - 1) / warpSize;
    if (lane == 0) { m_smem[warp] = m; l_smem[warp] = l; }
    __syncthreads();

    if (warp == 0) {
        m = (lane < warps) ? m_smem[lane] : -INFINITY;
        l = (lane < warps) ? l_smem[lane] : acc_t(0);
        #pragma unroll
        for (int off = warpSize / 2; off > 0; off >>= 1) {
            const acc_t om = __shfl_down_sync(0xffffffffu, m, off);
            const acc_t ol = __shfl_down_sync(0xffffffffu, l, off);
            softmax_merge(m, l, om, ol);
        }
        if (lane == 0) { m_smem[0] = m; l_smem[0] = l; }
    }
    __syncthreads();

    const acc_t row_max = m_smem[0];
    // reciprocal once, then multiply — a per-element divide is ~4x the latency
    const acc_t inv_sum = acc_t(1) / l_smem[0];

    // ---- pass 2: normalize → attn[b,q,h,:] ----
    for (int k = threadIdx.x; k < sk; k += blockDim.x)
        row[k] = exp(row[k] - row_max) * inv_sum;
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

// --- 3c. ctx = attn @ c_kv: 4×4 tile/thread; grid (flat B*Sq*H, kv_rank) ---
//   grid.x: flat = b*(Sq*H) + q*H + h     — 64 flats/block
//   grid.y: r ∈ [0, kv_rank)              — 128 ranks/block
//   block: 32×16 → each thread 4 ranks × 4 flats; Sk reduced in SMEM tiles of 64
//   threadIdx.x owns r (contiguous in `ctx`), threadIdx.y owns flat — same as 3a,
//   so the epilogue (plain store or split-K atomicAdd) is one 128 B transaction
//   grid.z: split-K slice over Sk — see below
//   SMEM [lf][k^lf] / [lr][k^lr]; attn staged k-fast, c_kv staged r-fast (HBM coalesce)
//   Reduction is k-outer: 4 attn + 4 ck loads → 16 FMAs, so 0.5 SMEM loads per FMA
//
//   Split-K: the grid only parallelises the output dims (flat × kv_rank), so a
//   single-stream decode (B=Sq=1 → 128 flats) yields just 2×4 = 8 blocks and leaves
//   ~90% of the SMs idle while each block walks all of Sk serially. grid.z chops the
//   Sk reduction into `sk_chunk`-sized slices; each slice accumulates a partial dot
//   product and atomically adds it into ctx, which the host pre-zeroes. The host sets
//   gridDim.z == 1 for shapes that already fill the GPU, and `use_atomic` then selects
//   plain stores so the common path pays neither the memset nor the atomics.
//   ctx_t is the input dtype on the plain path and the accumulate dtype on the split-K
//   path (see ctx_store): read-modify-write accumulation never narrows.
template <typename scalar_t, typename ctx_t>
__global__ void __launch_bounds__(kTileBlockThreads, kTileMinBlocksPerSM)
mla_ctx_kernel(
    const acc_t_of<scalar_t>* __restrict__ attn,  // [B, Sq, H, Sk]
    const scalar_t* __restrict__ c_kv,  // [B, Sk, kv_rank]
    ctx_t*          __restrict__ ctx,   // [B, Sq, H, kv_rank]
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
    const int flat_base  = blockIdx.x * kFlatTile;
    const int r0         = blockIdx.y * kCtxRankTile;

    // this block's slice of the Sk reduction; sk_chunk == sk when gridDim.z == 1
    const int k_begin = blockIdx.z * sk_chunk;
    const int k_end   = min(k_begin + sk_chunk, sk);

    acc_t* smem      = dynamic_smem<acc_t>();
    acc_t* attn_smem = smem;                                // [kFlatTile][kCtxSkTile]
    acc_t* ck_smem   = attn_smem + kFlatTile * kCtxSkTile;  // [kCtxRankTile][kCtxSkTile]

    // one batch per block when flat tile does not cross Sq*H (true for H=128, tile=64)
    const int batch_idx = min(flat_base, max(flat_total - 1, 0)) / (sq * num_heads);
    const int ck_batch_stride = batch_idx * sk * kv_rank;

    const int coop_stride = blockDim.x * blockDim.y;

    acc_t s[kRegTile][kRegTile];
    #pragma unroll
    for (int di = 0; di < kRegTile; ++di)
        #pragma unroll
        for (int dr = 0; dr < kRegTile; ++dr)
            s[di][dr] = acc_t(0);

    // Only the SMEM row indices live across the reduction — see the note in 3a: this
    // kernel is pinned at 64 registers too, so epilogue-only values are recomputed at
    // the bottom rather than held.
    int lf[kRegTile], lr[kRegTile];
    #pragma unroll
    for (int di = 0; di < kRegTile; ++di)
        lf[di] = threadIdx.y * kRegTile + di;
    #pragma unroll
    for (int dr = 0; dr < kRegTile; ++dr)
        lr[dr] = threadIdx.x + dr * blockDim.x;

    // Both masked by kCtxSkTile — the row width of attn_smem and ck_smem alike (see 3a).
    int lf_m[kRegTile], lr_m[kRegTile];
    #pragma unroll
    for (int di = 0; di < kRegTile; ++di)
        lf_m[di] = lf[di] & (kCtxSkTile - 1);
    #pragma unroll
    for (int dr = 0; dr < kRegTile; ++dr)
        lr_m[dr] = lr[dr] & (kCtxSkTile - 1);

    for (int k0 = k_begin; k0 < k_end; k0 += kCtxSkTile) {
        const int tile_len =
            (k0 + kCtxSkTile <= k_end) ? kCtxSkTile : (k_end - k0);

        // stage attn[flat, k0:k0+tile) — k contiguous → k_local = idx % tile_len
        const int attn_elems = kFlatTile * tile_len;
        for (int idx = threadIdx.x + threadIdx.y * blockDim.x;
             idx < attn_elems; idx += coop_stride) {
            const int lf_s    = idx / tile_len;
            const int k_local = idx % tile_len;
            const int flat_s  = flat_base + lf_s;
            if (flat_s < flat_total)
                attn_smem[lf_s * kCtxSkTile + (k_local ^ (lf_s & (kCtxSkTile - 1)))] =
                    attn[flat_s * sk + k0 + k_local];
        }

        // stage c_kv[b, k0:k0+tile, r0:r0+128) — r contiguous → lr = idx % kRs
        const int ck_elems = tile_len * kCtxRankTile;
        for (int idx = threadIdx.x + threadIdx.y * blockDim.x;
             idx < ck_elems; idx += coop_stride) {
            const int k_local = idx / kCtxRankTile;
            const int lr_s    = idx % kCtxRankTile;
            const int r_s     = r0 + lr_s;
            if (k_local < tile_len && r_s < kv_rank)
                ck_smem[lr_s * kCtxSkTile + (k_local ^ (lr_s & (kCtxSkTile - 1)))] =
                    static_cast<acc_t>(
                        c_kv[ck_batch_stride + (k0 + k_local) * kv_rank + r_s]);
        }
        __syncthreads();

        // k-outer register strip: 4 attn + 4 ck SMEM loads feed 16 FMAs (0.5 loads/FMA).
        // Lanes past flat_total / kv_rank read unwritten SMEM; dropped at the store.
        // Capped at 8: a full 64-wide unroll no longer fits the 64-register budget
        // once the split-K bounds are live, and spills cost ~8x.
        if (tile_len == kCtxSkTile) {
            #pragma unroll 8
            for (int k_local = 0; k_local < kCtxSkTile; ++k_local) {
                acc_t attn_reg[kRegTile], ck_reg[kRegTile];
                #pragma unroll
                for (int di = 0; di < kRegTile; ++di)
                    attn_reg[di] = attn_smem[lf[di] * kCtxSkTile + (k_local ^ lf_m[di])];
                #pragma unroll
                for (int dr = 0; dr < kRegTile; ++dr)
                    ck_reg[dr] = ck_smem[lr[dr] * kCtxSkTile + (k_local ^ lr_m[dr])];
                #pragma unroll
                for (int di = 0; di < kRegTile; ++di)
                    #pragma unroll
                    for (int dr = 0; dr < kRegTile; ++dr)
                        s[di][dr] += attn_reg[di] * ck_reg[dr];
            }
        } else {
            // ragged final Sk tile: same body, runtime trip count
            for (int k_local = 0; k_local < tile_len; ++k_local) {
                acc_t attn_reg[kRegTile], ck_reg[kRegTile];
                #pragma unroll
                for (int di = 0; di < kRegTile; ++di)
                    attn_reg[di] = attn_smem[lf[di] * kCtxSkTile + (k_local ^ lf_m[di])];
                #pragma unroll
                for (int dr = 0; dr < kRegTile; ++dr)
                    ck_reg[dr] = ck_smem[lr[dr] * kCtxSkTile + (k_local ^ lr_m[dr])];
                #pragma unroll
                for (int di = 0; di < kRegTile; ++di)
                    #pragma unroll
                    for (int dr = 0; dr < kRegTile; ++dr)
                        s[di][dr] += attn_reg[di] * ck_reg[dr];
            }
        }
        __syncthreads();
    }

    // Each grid.z slice owns a disjoint span of Sk, so the partials are a plain sum.
    // Lanes of a warp differ in `flat`, so these hit distinct cache lines; the only
    // contention is between the gridDim.z blocks sharing an output tile, and each of
    // them contributes exactly one atomic per element after its whole reduction.
    #pragma unroll
    for (int di = 0; di < kRegTile; ++di) {
        const int flat_i = flat_base + lf[di];
        if (flat_i >= flat_total) continue;
        #pragma unroll
        for (int dr = 0; dr < kRegTile; ++dr) {
            const int r_i = r0 + lr[dr];
            if (r_i >= kv_rank) continue;
            ctx_store(&ctx[flat_i * kv_rank + r_i], s[di][dr], use_atomic);
        }
    }
}

// --- 3d. out = ctx @ W_uv: 4×4 tile/thread; grid (bq tiles, v tiles, head) ---
//   One head per block: W_uv[h] is staged into SMEM once and amortised over the
//   whole bq tile, instead of being re-read from HBM for every single (bq, h).
//   grid.x: bq = b*Sq + q                     — 64 bq/block
//   grid.y: d ∈ [0, v_dim)                    — 128 v_dims/block
//   grid.z: h ∈ [0, H)                        — the shared W_uv slice
//   block: 32×16 → each thread 4 v_dims × 4 bq; kv_rank reduced in SMEM tiles of 64
//   x indexes v_dim so a warp stores 32 contiguous floats of `out`
//   SMEM [lb][r^lb] / [ld][r^ld] with col XOR swizzle (no padding), r staged fast
//   Reduction is r-outer: 4 ctx + 4 w loads → 16 FMAs, so 0.5 SMEM loads per FMA
template <typename scalar_t, typename ctx_t>
__global__ void __launch_bounds__(kTileBlockThreads, kTileMinBlocksPerSM)
mla_output_smem_kernel(
    const ctx_t*    __restrict__ ctx,   // [B, Sq, H, kv_rank]  dtype chosen by 3c
    const scalar_t* __restrict__ W_uv,  // [H, v_dim, kv_rank]
    scalar_t*       __restrict__ out,   // [B, Sq, H, v_dim]
    int bq_total,
    int num_heads,
    int kv_rank,
    int v_dim)
{
    using acc_t = acc_t_of<scalar_t>;

    const int bq_base   = blockIdx.x * kOutBqTile;
    const int d_base    = blockIdx.y * kOutVTile;
    const int head_idx  = blockIdx.z;

    acc_t* smem     = dynamic_smem<acc_t>();
    acc_t* ctx_smem = smem;                                 // [kOutBqTile][kOutRankTile]
    acc_t* w_smem   = ctx_smem + kOutBqTile * kOutRankTile; // [kOutVTile][kOutRankTile]

    // ctx[bq, h, r] and W_uv[h, d, r] are both contiguous in r
    const int ctx_head_base = head_idx * kv_rank;
    const int w_head_base   = head_idx * v_dim * kv_rank;
    const int coop_stride   = blockDim.x * blockDim.y;

    acc_t s[kRegTile][kRegTile];
    #pragma unroll
    for (int db = 0; db < kRegTile; ++db)
        #pragma unroll
        for (int dd = 0; dd < kRegTile; ++dd)
            s[db][dd] = acc_t(0);

    // Only the SMEM row indices live across the reduction (see the note in 3a).
    int lb[kRegTile], ld[kRegTile];
    #pragma unroll
    for (int db = 0; db < kRegTile; ++db)
        lb[db] = threadIdx.y * kRegTile + db;
    #pragma unroll
    for (int dd = 0; dd < kRegTile; ++dd)
        ld[dd] = threadIdx.x + dd * blockDim.x;

    // Both masked by kOutRankTile — the row width of ctx_smem and w_smem alike (see 3a).
    int lb_m[kRegTile], ld_m[kRegTile];
    #pragma unroll
    for (int db = 0; db < kRegTile; ++db)
        lb_m[db] = lb[db] & (kOutRankTile - 1);
    #pragma unroll
    for (int dd = 0; dd < kRegTile; ++dd)
        ld_m[dd] = ld[dd] & (kOutRankTile - 1);

    for (int r0 = 0; r0 < kv_rank; r0 += kOutRankTile) {
        const int tile_len =
            (r0 + kOutRankTile <= kv_rank) ? kOutRankTile : (kv_rank - r0);

        // stage ctx[bq_base:+64, h, r0:+tile) — r contiguous → r_local = idx % tile_len
        const int ctx_elems = kOutBqTile * tile_len;
        for (int idx = threadIdx.x + threadIdx.y * blockDim.x;
             idx < ctx_elems; idx += coop_stride) {
            const int lb_s    = idx / tile_len;
            const int r_local = idx % tile_len;
            const int bq_s    = bq_base + lb_s;
            if (bq_s < bq_total)
                ctx_smem[lb_s * kOutRankTile + (r_local ^ (lb_s & (kOutRankTile - 1)))] =
                    static_cast<acc_t>(
                        ctx[(bq_s * num_heads) * kv_rank + ctx_head_base + r0 + r_local]);
        }

        // stage W_uv[h, d_base:+128, r0:+tile) — reused by all 64 bq in this block
        const int w_elems = kOutVTile * tile_len;
        for (int idx = threadIdx.x + threadIdx.y * blockDim.x;
             idx < w_elems; idx += coop_stride) {
            const int ld_s    = idx / tile_len;
            const int r_local = idx % tile_len;
            const int d_s     = d_base + ld_s;
            if (d_s < v_dim)
                w_smem[ld_s * kOutRankTile + (r_local ^ (ld_s & (kOutRankTile - 1)))] =
                    static_cast<acc_t>(
                        W_uv[w_head_base + d_s * kv_rank + r0 + r_local]);
        }
        __syncthreads();

        // r-outer register strip; out-of-range lanes read unwritten SMEM and are
        // dropped at the store, so no per-iteration bounds predicate is needed
        if (tile_len == kOutRankTile) {
            #pragma unroll 8
            for (int r_local = 0; r_local < kOutRankTile; ++r_local) {
                acc_t ctx_reg[kRegTile], w_reg[kRegTile];
                #pragma unroll
                for (int db = 0; db < kRegTile; ++db)
                    ctx_reg[db] = ctx_smem[lb[db] * kOutRankTile + (r_local ^ lb_m[db])];
                #pragma unroll
                for (int dd = 0; dd < kRegTile; ++dd)
                    w_reg[dd] = w_smem[ld[dd] * kOutRankTile + (r_local ^ ld_m[dd])];
                #pragma unroll
                for (int db = 0; db < kRegTile; ++db)
                    #pragma unroll
                    for (int dd = 0; dd < kRegTile; ++dd)
                        s[db][dd] += ctx_reg[db] * w_reg[dd];
            }
        } else {
            // ragged final kv_rank tile: same body, runtime trip count
            for (int r_local = 0; r_local < tile_len; ++r_local) {
                acc_t ctx_reg[kRegTile], w_reg[kRegTile];
                #pragma unroll
                for (int db = 0; db < kRegTile; ++db)
                    ctx_reg[db] = ctx_smem[lb[db] * kOutRankTile + (r_local ^ lb_m[db])];
                #pragma unroll
                for (int dd = 0; dd < kRegTile; ++dd)
                    w_reg[dd] = w_smem[ld[dd] * kOutRankTile + (r_local ^ ld_m[dd])];
                #pragma unroll
                for (int db = 0; db < kRegTile; ++db)
                    #pragma unroll
                    for (int dd = 0; dd < kRegTile; ++dd)
                        s[db][dd] += ctx_reg[db] * w_reg[dd];
            }
        }
        __syncthreads();  // all reads done before the next tile overwrites SMEM
    }

    #pragma unroll
    for (int db = 0; db < kRegTile; ++db) {
        const int bq_i = bq_base + lb[db];
        if (bq_i >= bq_total) continue;
        const int out_base = (bq_i * num_heads + head_idx) * v_dim;
        #pragma unroll
        for (int dd = 0; dd < kRegTile; ++dd) {
            const int d_i = d_base + ld[dd];
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

    // 3a unrolls its rope reduction over the compile-time kRopeDim and reuses the
    // kScoreRankTile-wide nope tiles for the rope staging, so a different runtime rope_dim
    // would both mis-stride the SMEM and overrun those tiles. Reject it rather than
    // return quiet garbage.
    TORCH_CHECK(rope_dim == kRopeDim,
                "MLA: rope_dim must be ", kRopeDim, " (got ", rope_dim,
                "); kRopeDim is compile-time in mla_scores_kernel");

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
    const int ctx_flat_tiles = (flat_total + kFlatTile - 1) / kFlatTile;
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

    auto stream = at::cuda::getCurrentCUDAStream();

    // 3a: grid.x=flat(B*Sq*H), grid.y=Sk; 32×16 threads → 64×128 scores/block;
    //     kv_rank reduction iterated in SMEM tiles of kScoreRankTile=64
    {
        dim3 threads(kScoreThreadsX, kScoreThreadsY);  // 32×16 = 512 threads
        dim3 blocks(
            (flat_total + kFlatTile - 1) / kFlatTile,
            (sk         + kScoreSkTile    - 1) / kScoreSkTile);
        // SMEM: qa[64][64] + ck[128][64] acc_t with XOR col swizzle (qr/pe reuse)
        size_t smem = static_cast<size_t>(
            kFlatTile * kScoreRankTile + kScoreSkTile * kScoreRankTile) * acc_size;
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(q_absorbed.scalar_type(), "mla_scores", ([&] {
            using acc_t = acc_t_of<scalar_t>;
            auto kernel = mla_scores_kernel<scalar_t>;
            enable_large_smem(kernel, smem);  // fp64 tiles exceed the default 48 KB
            kernel<<<blocks, threads, smem, stream>>>(
                q_absorbed.data_ptr<scalar_t>(), q_rope.data_ptr<scalar_t>(),
                c_kv.data_ptr<scalar_t>(),       pe_cache.data_ptr<scalar_t>(),
                scores.data_ptr<acc_t>(),
                batch_size, sq, sk, num_heads, kv_rank, rope_dim,
                static_cast<acc_t>(scale));
        }));
    }

    // 3b: softmax in-place on scores → attn [B,Sq,128,Sk]; one block per row.
    //     With few rows (single-stream decode is only H=128 of them) a 256-thread
    //     block leaves most of the GPU idle, so widen the block instead; with many
    //     rows the narrow block packs more of them per SM.
    {
        const int rows = flat_total;
        const int threads = (rows >= 4 * sm_count()) ? kSoftmaxThreadsSmall
                                                     : kSoftmaxThreadsLarge;
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(q_absorbed.scalar_type(), "mla_softmax", ([&] {
            using acc_t = acc_t_of<scalar_t>;
            mla_softmax_kernel<acc_t><<<rows, threads, 0, stream>>>(
                scores.data_ptr<acc_t>(), sk);
        }));
    }

    // 3c: grid.x=flat(B*Sq*H), grid.y=kv_rank, grid.z=split-K slice over Sk;
    //     32×16 → 64×128 ctx/block; Sk reduced in SMEM tiles of kCtxSkTile=64
    {
        // atomicAdd accumulates, so the partials need a zeroed destination
        if (ctx_atomic) ctx.zero_();

        dim3 threads(kScoreThreadsX, kScoreThreadsY);  // 32×16 = 512 threads
        dim3 blocks(ctx_flat_tiles, ctx_r_tiles, ctx_split);
        // SMEM: attn[64][64] + ck[128][64] acc_t with XOR col swizzle
        size_t smem = static_cast<size_t>(
            kFlatTile * kCtxSkTile + kCtxRankTile * kCtxSkTile) * acc_size;
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(q_absorbed.scalar_type(), "mla_ctx", ([&] {
            using acc_t = acc_t_of<scalar_t>;
            if (ctx_atomic) {
                auto kernel = mla_ctx_kernel<scalar_t, acc_t>;
                enable_large_smem(kernel, smem);
                kernel<<<blocks, threads, smem, stream>>>(
                    scores.data_ptr<acc_t>(), c_kv.data_ptr<scalar_t>(),
                    ctx.data_ptr<acc_t>(),
                    batch_size, sq, sk, num_heads, kv_rank, ctx_sk_chunk, true);
            } else {
                auto kernel = mla_ctx_kernel<scalar_t, scalar_t>;
                enable_large_smem(kernel, smem);
                kernel<<<blocks, threads, smem, stream>>>(
                    scores.data_ptr<acc_t>(), c_kv.data_ptr<scalar_t>(),
                    ctx.data_ptr<scalar_t>(),
                    batch_size, sq, sk, num_heads, kv_rank, ctx_sk_chunk, false);
            }
        }));
    }

    // 3d: grid.x=bq tiles, grid.y=v_dim tiles, grid.z=head; 32×16 → 64 bq × 128 v/block
    //     kv_rank reduction in SMEM tiles of kOutRankTile=64; W_uv[h] staged per block
    {
        const int bq_total = batch_size * sq;
        dim3 threads(kOutThreadsX, kOutThreadsY);  // 32×16 = 512 threads
        dim3 blocks(
            (bq_total + kOutBqTile - 1) / kOutBqTile,
            (v_dim    + kOutVTile  - 1) / kOutVTile,
            num_heads);
        // SMEM: ctx[64][64] + w[128][64] acc_t with XOR col swizzle
        size_t smem = static_cast<size_t>(
            kOutBqTile * kOutRankTile + kOutVTile * kOutRankTile) * acc_size;
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(q_absorbed.scalar_type(), "mla_output", ([&] {
            using acc_t = acc_t_of<scalar_t>;
            if (ctx_atomic) {  // 3c promoted ctx to the accumulate dtype for the atomics
                auto kernel = mla_output_smem_kernel<scalar_t, acc_t>;
                enable_large_smem(kernel, smem);
                kernel<<<blocks, threads, smem, stream>>>(
                    ctx.data_ptr<acc_t>(), W_uv.data_ptr<scalar_t>(),
                    out.data_ptr<scalar_t>(),
                    bq_total, num_heads, kv_rank, v_dim);
            } else {
                auto kernel = mla_output_smem_kernel<scalar_t, scalar_t>;
                enable_large_smem(kernel, smem);
                kernel<<<blocks, threads, smem, stream>>>(
                    ctx.data_ptr<scalar_t>(), W_uv.data_ptr<scalar_t>(),
                    out.data_ptr<scalar_t>(),
                    bq_total, num_heads, kv_rank, v_dim);
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
          "Kernel 2 — fused Q path: x@W_q → absorb@W_uk + RoPE(Q_rope)");
    m.def("mla_attention", &launch_mla_attention,
          "Kernel 3 — Official absorbed MLA: scores → softmax → ctx@c_kv → @W_uv");
}