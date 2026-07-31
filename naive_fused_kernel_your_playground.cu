#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// DeepSeek-V3 reference dims: hidden=7168, H=128, kv_rank=512, nope=128,
// rope=64, v=128, qk_head=192. B, Sq, Sk vary per launch.

// ============================================================================
// KERNEL 1 (legacy / teaching): c_kv @ W_k_rope + RoPE
//
// NOT used in official DeepSeek MLA absorbed inference. In production, K_rope
// comes from wkv_a(x) → RoPE → pe_cache [B, Sk, rope_dim] (head-shared).
// Kept as a matmul+RoPE tiling exercise.
// ============================================================================
template <typename scalar_t>
__global__ void naive_decompress_and_rope_kernel(
    const scalar_t* __restrict__ c_kv,   // [B, Sk, kv_lora_rank]
    const scalar_t* __restrict__ W_k,    // [kv_lora_rank, H, rope_dim]
    scalar_t* __restrict__ out_k,        // [B, Sk, H, rope_dim]
    int batch_size,
    int seq_len,
    int latent_dim,
    int num_heads,
    int head_dim)
{
    // Grid: [B*Sk, H, rope_dim/2]
    int batch_seq_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int head_idx      = blockIdx.y * blockDim.y + threadIdx.y;
    int half_head_idx = blockIdx.z * blockDim.z + threadIdx.z;

    if (batch_seq_idx >= batch_size * seq_len ||
        head_idx      >= num_heads            ||
        half_head_idx >= head_dim / 2) return;

    int b = batch_seq_idx / seq_len;
    int s = batch_seq_idx % seq_len;
    int d = half_head_idx * 2;

    // matmul over kv_rank=512: c_kv[b,s,:512] · W_k[i,h,d:d+1] → sum_even/sum_odd
    // (legacy layout; official MLA uses pe_cache instead of this decompress path)
    float sum_even = 0.0f, sum_odd = 0.0f;
    for (int i = 0; i < latent_dim; ++i) {
        float c_val  = static_cast<float>(c_kv[(b * seq_len + s) * latent_dim + i]);
        float w_even = static_cast<float>(W_k[(i * num_heads + head_idx) * head_dim + d]);
        float w_odd  = static_cast<float>(W_k[(i * num_heads + head_idx) * head_dim + (d + 1)]);
        sum_even += c_val * w_even;
        sum_odd  += c_val * w_odd;
    }

    // RoPE on adjacent pair (d, d+1) within rope_dim=64; write [B,Sk,H,64]
    float inv_freq = 1.0f / powf(10000.0f,
                        static_cast<float>(d) / static_cast<float>(head_dim));
    float angle   = static_cast<float>(s) * inv_freq;
    float cos_val = cosf(angle), sin_val = sinf(angle);

    int base = ((b * seq_len + s) * num_heads + head_idx) * head_dim;
    out_k[base + d]     = static_cast<scalar_t>(sum_even * cos_val - sum_odd * sin_val);
    out_k[base + d + 1] = static_cast<scalar_t>(sum_even * sin_val + sum_odd * cos_val);
}

torch::Tensor launch_mla_fused_rope(torch::Tensor c_kv, torch::Tensor W_k) {
    int batch_size = c_kv.size(0), seq_len   = c_kv.size(1);
    int latent_dim = c_kv.size(2), num_heads = W_k.size(1), head_dim = W_k.size(2);

    auto out_k = torch::empty({batch_size, seq_len, num_heads, head_dim}, c_kv.options());

    dim3 threads(8, 8, 8);
    dim3 blocks(
        (batch_size * seq_len + threads.x - 1) / threads.x,
        (num_heads            + threads.y - 1) / threads.y,
        (head_dim / 2         + threads.z - 1) / threads.z);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(c_kv.scalar_type(), "mla_fused_rope", ([&] {
        naive_decompress_and_rope_kernel<scalar_t><<<blocks, threads>>>(
            c_kv.data_ptr<scalar_t>(), W_k.data_ptr<scalar_t>(),
            out_k.data_ptr<scalar_t>(),
            batch_size, seq_len, latent_dim, num_heads, head_dim);
    }));
    cudaDeviceSynchronize();
    return out_k;
}


// ============================================================================
// KERNEL 2: Q absorbed path — 2D grid (B*Sq, H) + shared memory
//
//  Grid: blockIdx.x = B*Sq, blockIdx.y = head (128). One block per (b,s,h).
//
//  2a. [B,Sq,7168] @ W_q[:,h,0:128]     → Q_nope      [B,Sq,128,128]
//  2b. [B,Sq,128,128] @ W_uk[128,128,512] → Q_absorbed [B,Sq,128,512]
//  2c. [B,Sq,7168] @ W_q[:,h,128:192] + RoPE → Q_rope  [B,Sq,128,64]
// ============================================================================

static constexpr int kBlockThreads = 256;
// Kernel 3a/3c: 4×4 output tile per thread; 16×32 block → 64×128 outputs/block
static constexpr int kScoreOutTile   = 4;   // 4 along flat × 4 along Sk (3a) or r (3c)
static constexpr int kScoreThreadsX  = 16;  // 16×4 = 64 flats per block
static constexpr int kScoreThreadsY  = 32;  // 32×4 = 128 keys (3a) or ranks (3c) per block
static constexpr int kFlatsPerThreadBlock  = kScoreThreadsX * kScoreOutTile;  // 64
static constexpr int kKsPerThreadBlock     = kScoreThreadsY * kScoreOutTile;  // 128
static constexpr int kRsPerThreadBlock     = kKsPerThreadBlock;               // 128 (3c r-tiles)
static constexpr int kKvRankTile     = 64;  // SMEM tile along kv_rank (3a reduction)
static constexpr int kRopeDim        = 64;  // qk_rope_head_dim (compile-time for unroll)
static constexpr int kCtxSkTile      = 64;  // SMEM tile along Sk (3c reduction)
// Kernel 3d: one head per block so W_uv[h] is staged once and reused across bq.
// x indexes v_dim (contiguous in `out`) so the epilogue stores are coalesced.
static constexpr int kOutThreadsX    = 32;  // 32×4 = 128 v_dims per block
static constexpr int kOutThreadsY    = 16;  // 16×4 = 64  bq per block
static constexpr int kOutVTile       = kOutThreadsX * kScoreOutTile;  // 128
static constexpr int kOutBqTile      = kOutThreadsY * kScoreOutTile;  // 64
static constexpr int kOutRTile       = 64;  // SMEM tile along kv_rank (3d reduction)
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

// Dynamic SMEM above 48 KB (which fp64 tiles need) is opt-in per kernel.
template <typename KernelFn>
static void enable_large_smem(KernelFn kernel, size_t smem) {
    if (smem > 48u * 1024u)
        cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel),
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem));
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

// --- 2a: x @ W_q[:,h,:nope] → Q_nope [B, Sq, H, nope] ---
template <typename scalar_t>
__global__ void q_nope_smem_kernel(
    const scalar_t* __restrict__ x,      // [B, Sq, hidden]
    const scalar_t* __restrict__ W_q,      // [hidden, H, qk_head_dim]
    scalar_t*       __restrict__ q_nope, // [B, Sq, H, nope_dim]
    int sq, int hidden_dim,
    int num_heads, int nope_dim, int qk_head_dim)
{
    extern __shared__ float x_smem[];

    int bs_idx   = blockIdx.x;   // b*sq + s  ∈ [0, B*Sq)
    int head_idx = blockIdx.y;   // h         ∈ [0, 128)
    int b = bs_idx / sq, s = bs_idx % sq;

    // stage x[b,s,:] → x_smem[7168]  (load [B,Sq,7168] row once per block)
    const int x_base = (b * sq + s) * hidden_dim;
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x)
        x_smem[i] = static_cast<float>(x[x_base + i]);
    __syncthreads();

    // matmul: x[b,s,0:7168] · W_q[0:7168,h,d] → Q_nope[b,s,h,d]
    // threads stride over d; output [B,Sq,128,128]
    for (int d_idx = threadIdx.x; d_idx < nope_dim; d_idx += blockDim.x) {
        double acc = 0.0;
        for (int i = 0; i < hidden_dim; ++i) {
            // W_q layout [7168, 128, 192]: W_q[i, h, d]
            double w_val = static_cast<double>(
                W_q[i * num_heads * qk_head_dim + head_idx * qk_head_dim + d_idx]);
            acc += static_cast<double>(x_smem[i]) * w_val;
        }
        q_nope[(bs_idx * num_heads + head_idx) * nope_dim + d_idx] =
            static_cast<scalar_t>(acc);  // [B,Sq,128,128]
    }
}

// --- 2b: Q_nope @ W_uk → Q_absorbed [B, Sq, H, kv_rank] ---
template <typename scalar_t>
__global__ void q_absorb_smem_kernel(
    const scalar_t* __restrict__ q_nope,     // [B, Sq, H, nope_dim]
    const scalar_t* __restrict__ W_uk,       // [H, nope_dim, kv_rank]
    scalar_t*       __restrict__ q_absorbed, // [B, Sq, H, kv_rank]
    int num_heads, int nope_dim, int kv_rank)
{
    extern __shared__ float qn_smem[];

    int bs_idx   = blockIdx.x;   // b*sq + s
    int head_idx = blockIdx.y;   // h
    const int qn_base = (bs_idx * num_heads + head_idx) * nope_dim;

    // stage Q_nope[b,s,h,:] → qn_smem[128]
    for (int d = threadIdx.x; d < nope_dim; d += blockDim.x)
        qn_smem[d] = static_cast<float>(q_nope[qn_base + d]);
    __syncthreads();

    // matmul: Q_nope[b,s,h,0:128] · W_uk[h,0:128,r] → Q_absorbed[b,s,h,r]
    // contract d=128; output [B,Sq,128,512]
    for (int r_idx = threadIdx.x; r_idx < kv_rank; r_idx += blockDim.x) {
        double acc = 0.0;
        for (int d = 0; d < nope_dim; ++d) {
            // W_uk [128, 128, 512]: W_uk[h, d, r]
            double wu = static_cast<double>(
                W_uk[head_idx * nope_dim * kv_rank + d * kv_rank + r_idx]);
            acc += static_cast<double>(qn_smem[d]) * wu;
        }
        q_absorbed[(bs_idx * num_heads + head_idx) * kv_rank + r_idx] =
            static_cast<scalar_t>(acc);  // [B,Sq,128,512]
    }
}

// --- 2c: x @ W_q[:,h,nope:] → Q_rope + RoPE [B, Sq, H, rope_dim] ---
template <typename scalar_t>
__global__ void q_rope_smem_kernel(
    const scalar_t* __restrict__ x,          // [B, Sq, hidden]
    const scalar_t* __restrict__ W_q,        // [hidden, H, qk_head_dim]
    scalar_t*       __restrict__ q_rope_out, // [B, Sq, H, rope_dim]
    int sq, int hidden_dim,
    int num_heads, int nope_dim, int rope_dim, int qk_head_dim)
{
    extern __shared__ float x_smem[];

    int bs_idx   = blockIdx.x;   // b*sq + s
    int head_idx = blockIdx.y;   // h
    int b = bs_idx / sq, s = bs_idx % sq;

    // stage x[b,s,:] → x_smem[7168]
    const int x_base = (b * sq + s) * hidden_dim;
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x)
        x_smem[i] = static_cast<float>(x[x_base + i]);
    __syncthreads();

    const int out_base = (bs_idx * num_heads + head_idx) * rope_dim;
    // threads over rope pairs (32 pairs); matmul cols 128:192 of W_q per head
    for (int half_rope_idx = threadIdx.x; half_rope_idx < rope_dim / 2;
         half_rope_idx += blockDim.x) {
        int d = half_rope_idx * 2;  // pair-split within rope_dim=64

        // matmul rope slice: x[b,s,0:7168] · W_q[0:7168,h,128+d:128+d+1] → even/odd
        double sum_even = 0.0, sum_odd = 0.0;
        for (int i = 0; i < hidden_dim; ++i) {
            double w_even = static_cast<double>(
                W_q[i * num_heads * qk_head_dim + head_idx * qk_head_dim + nope_dim + d]);
            double w_odd  = static_cast<double>(
                W_q[i * num_heads * qk_head_dim + head_idx * qk_head_dim + nope_dim + d + 1]);
            const double xv = static_cast<double>(x_smem[i]);
            sum_even += xv * w_even;
            sum_odd  += xv * w_odd;
        }

        // RoPE on pair (d,d+1); merge → Q_rope[b,s,h,d:d+1]  output [B,Sq,128,64]
        double inv_freq = 1.0 / pow(10000.0,
                            static_cast<double>(d) / static_cast<double>(rope_dim));
        double angle   = static_cast<double>(s) * inv_freq;
        double cos_val = cos(angle), sin_val = sin(angle);

        q_rope_out[out_base + d]     = static_cast<scalar_t>(
            static_cast<float>(sum_even * cos_val - sum_odd * sin_val));
        q_rope_out[out_base + d + 1] = static_cast<scalar_t>(
            static_cast<float>(sum_even * sin_val + sum_odd * cos_val));
    }
}

// RoPE helper — matches exhaustive_benchmark_suite.apply_rope (positions = arange(Sq))
static torch::Tensor apply_rope_aten(torch::Tensor x, int rope_dim) {
    const auto fp32 = x.options().dtype(torch::kFloat32);
    auto freqs = 1.0 / torch::pow(
        10000.0,
        torch::arange(0, rope_dim, 2, fp32) / static_cast<double>(rope_dim));
    const int sq = static_cast<int>(x.size(1));
    auto positions = torch::arange(sq, fp32);
    auto angles    = torch::outer(positions, freqs);
    auto cos_v = torch::cos(angles).to(x.scalar_type()).view({1, sq, 1, rope_dim / 2});
    auto sin_v = torch::sin(angles).to(x.scalar_type()).view({1, sq, 1, rope_dim / 2});

    auto x_fp = x.to(torch::kFloat32);
    const int64_t B = x.size(0), H = x.size(2);
    auto x_view = x_fp.view({B, sq, H, rope_dim / 2, 2});
    auto x_even = x_view.select(-1, 0);
    auto x_odd  = x_view.select(-1, 1);
    auto out_even = x_even * cos_v - x_odd * sin_v;
    auto out_odd  = x_even * sin_v + x_odd * cos_v;
    return torch::stack({out_even, out_odd}, -1).flatten(-2).to(x.scalar_type());
}

// Host launcher: returns {q_absorbed [B,Sq,H,kv_rank], q_rope [B,Sq,H,rope_dim]}
//
// Uses ATen matmul/einsum so Q matches PyTorch baseline bit-for-bit (same cuBLAS path).
// Kernels 2a–2c above are kept as SMEM tiling exercises; they differ slightly due to
// serial fp32 reduction order over hidden=7168.
std::vector<torch::Tensor> launch_q_absorbed(
    torch::Tensor x,     // [B, Sq, hidden]
    torch::Tensor W_q,   // [hidden, H*(nope+rope)]
    torch::Tensor W_uk,  // [H, nope, kv_rank]
    int nope_dim,
    int rope_dim)
{
    const int batch_size = static_cast<int>(x.size(0));
    const int sq         = static_cast<int>(x.size(1));
    const int num_heads  = static_cast<int>(W_uk.size(0));
    const int qk_head_dim = nope_dim + rope_dim;  // 128+64=192

    // 2a+2c equivalent: [B,Sq,7168] @ W_q → [B,Sq,128,192]
    auto q_raw = torch::matmul(x, W_q).view({batch_size, sq, num_heads, qk_head_dim});
    auto q_nope     = q_raw.narrow(-1, 0, nope_dim);          // [B,Sq,128,128]
    auto q_rope_raw = q_raw.narrow(-1, nope_dim, rope_dim);   // [B,Sq,128,64]

    // 2b: [B,Sq,128,128] × [128,128,512] → [B,Sq,128,512]
    auto q_absorbed = torch::einsum("bshd,hdr->bshr", {q_nope, W_uk});

    // RoPE on rope slice → [B,Sq,128,64]
    auto q_rope_out = apply_rope_aten(q_rope_raw, rope_dim);

    return {q_absorbed, q_rope_out};
}

#if 0  // reference CUDA Q-prep kernels (launch_q_absorbed_cuda_kernels)
std::vector<torch::Tensor> launch_q_absorbed_cuda_kernels(
    torch::Tensor x, torch::Tensor W_q, torch::Tensor W_uk,
    int nope_dim, int rope_dim)
{
    int batch_size = x.size(0), sq         = x.size(1), hidden_dim = x.size(2);
    int num_heads  = W_uk.size(0), kv_rank = W_uk.size(2);
    int qk_head_dim = nope_dim + rope_dim;

    auto q_nope     = torch::empty({batch_size, sq, num_heads, nope_dim}, x.options());
    auto q_absorbed = torch::empty({batch_size, sq, num_heads, kv_rank},  x.options());
    auto q_rope_out = torch::empty({batch_size, sq, num_heads, rope_dim}, x.options());

    // Kernel 2a: grid (B*Sq, 128); SMEM x[7168] → Q_nope [B,Sq,128,128]
    {
        dim3 blocks(batch_size * sq, num_heads);
        size_t smem = static_cast<size_t>(hidden_dim) * sizeof(float);  // 7168 floats
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "q_nope", ([&] {
            q_nope_smem_kernel<scalar_t><<<blocks, kBlockThreads, smem>>>(
                x.data_ptr<scalar_t>(), W_q.data_ptr<scalar_t>(),
                q_nope.data_ptr<scalar_t>(),
                sq, hidden_dim, num_heads, nope_dim, qk_head_dim);
        }));
        cudaDeviceSynchronize();
    }

    // Kernel 2b: grid (B*Sq, 128); SMEM Q_nope[128] → Q_absorbed [B,Sq,128,512]
    {
        dim3 blocks(batch_size * sq, num_heads);
        size_t smem = static_cast<size_t>(nope_dim) * sizeof(float);  // 128 floats
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "q_absorb", ([&] {
            q_absorb_smem_kernel<scalar_t><<<blocks, kBlockThreads, smem>>>(
                q_nope.data_ptr<scalar_t>(), W_uk.data_ptr<scalar_t>(),
                q_absorbed.data_ptr<scalar_t>(),
                num_heads, nope_dim, kv_rank);
        }));
        cudaDeviceSynchronize();
    }

    // Kernel 2c: grid (B*Sq, 128); SMEM x[7168] + RoPE → Q_rope [B,Sq,128,64]
    {
        dim3 blocks(batch_size * sq, num_heads);
        size_t smem = static_cast<size_t>(hidden_dim) * sizeof(float);  // 7168 floats
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "q_rope", ([&] {
            q_rope_smem_kernel<scalar_t><<<blocks, kBlockThreads, smem>>>(
                x.data_ptr<scalar_t>(), W_q.data_ptr<scalar_t>(),
                q_rope_out.data_ptr<scalar_t>(),
                sq, hidden_dim, num_heads, nope_dim, rope_dim, qk_head_dim);
        }));
        cudaDeviceSynchronize();
    }

    return {q_absorbed, q_rope_out};
}
#endif


// ============================================================================
// KERNEL 3: Official absorbed MLA attention
//
//  3a. scores — grid.x over flat(B,Sq,128), grid.y over Sk
//      16×32 threads/block → 64×128 scores/block; each thread → 4×4 tile (q × c_kv)
//      kv_rank=512 contracted in SMEM tiles of kKvRankTile=64 (block-level reduction)
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
//   block: 16×32 threads → each thread 4 flats × 4 keys (64×128 / block)
//   Flat ownership interleaved: lf = tx + di*blockDim.x
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
    const int flat_base  = blockIdx.x * kFlatsPerThreadBlock;
    const int k0         = blockIdx.y * kKsPerThreadBlock;

    // qa_smem[64][kKvRankTile]; ck_smem[128][kKvRankTile] — reused as qr/pe after phase 1
    acc_t* smem    = dynamic_smem<acc_t>();
    acc_t* qa_smem = smem;
    acc_t* ck_smem = qa_smem + kFlatsPerThreadBlock * kKvRankTile;

    // batch for c_kv / pe_cache rows (64 flats/block ⊂ one batch when block < Sq*H)
    const int batch_idx = min(flat_base, max(flat_total - 1, 0)) / (sq * num_heads);
    const int ck_batch_stride = batch_idx * sk * kv_rank;
    const int pe_batch_stride = batch_idx * sk * rope_dim;

    const int coop_stride = blockDim.x * blockDim.y;
    const acc_t inv_scale = acc_t(1) / scale;

    // Per-thread output tile; rope later accumulates into the same regs
    acc_t s[kScoreOutTile][kScoreOutTile];
    #pragma unroll
    for (int di = 0; di < kScoreOutTile; ++di)
        #pragma unroll
        for (int dk = 0; dk < kScoreOutTile; ++dk)
            s[di][dk] = acc_t(0);

    // Bounds once per thread (outside r / d reduction loops)
    int lf[kScoreOutTile], lk[kScoreOutTile];
    int flat[kScoreOutTile], k_idx[kScoreOutTile];
    bool valid_f[kScoreOutTile], valid_k[kScoreOutTile];
    #pragma unroll
    for (int di = 0; di < kScoreOutTile; ++di) {
        lf[di]      = threadIdx.x + di * blockDim.x;
        flat[di]    = flat_base + lf[di];
        valid_f[di] = flat[di] < flat_total;
    }
    #pragma unroll
    for (int dk = 0; dk < kScoreOutTile; ++dk) {
        lk[dk]      = threadIdx.y * kScoreOutTile + dk;
        k_idx[dk]   = k0 + lk[dk];
        valid_k[dk] = k_idx[dk] < sk;
    }

    // XOR swizzle masks — per-thread constants, hoisted out of the reduction loops
    int lf_m[kScoreOutTile], lk_m[kScoreOutTile];
    #pragma unroll
    for (int di = 0; di < kScoreOutTile; ++di)
        lf_m[di] = lf[di] & (kKvRankTile - 1);
    #pragma unroll
    for (int dk = 0; dk < kScoreOutTile; ++dk)
        lk_m[dk] = lk[dk] & (kKvRankTile - 1);

    // ---- Phase 1: scores_nope — fuse qa+ck SMEM loads, one sync per kv_rank tile ----
    for (int r0 = 0; r0 < kv_rank; r0 += kKvRankTile) {
        const int tile_len =
            (r0 + kKvRankTile <= kv_rank) ? kKvRankTile : (kv_rank - r0);

        // cooperative: stage Q_absorbed → qa_smem[lf][r^lf]
        // q_absorbed[flat, r] is contiguous in r → r_local must be idx% so warps coalesce
        const int qa_elems = kFlatsPerThreadBlock * tile_len;
        for (int idx = threadIdx.x + threadIdx.y * blockDim.x;
             idx < qa_elems; idx += coop_stride) {
            const int lf_s    = idx / tile_len;
            const int r_local = idx % tile_len;
            const int flat_s  = flat_base + lf_s;
            if (flat_s < flat_total)
                qa_smem[lf_s * kKvRankTile + (r_local ^ (lf_s & (kKvRankTile - 1)))] =
                    static_cast<acc_t>(q_absorbed[flat_s * kv_rank + r0 + r_local]);
        }

        // cooperative: stage c_kv → ck_smem[lk][r^lk]  (r contiguous in c_kv[k, r])
        const int ck_elems = kKsPerThreadBlock * tile_len;
        for (int idx = threadIdx.x + threadIdx.y * blockDim.x;
             idx < ck_elems; idx += coop_stride) {
            const int lk_s    = idx / tile_len;
            const int r_local = idx % tile_len;
            const int k_s     = k0 + lk_s;
            if (k_s < sk)
                ck_smem[lk_s * kKvRankTile + (r_local ^ (lk_s & (kKvRankTile - 1)))] =
                    static_cast<acc_t>(c_kv[ck_batch_stride + k_s * kv_rank + r0 + r_local]);
        }
        __syncthreads();  // loads done before any thread reads this tile

        // r-outer register strip: 4 qa + 4 ck SMEM loads feed 16 FMAs (0.5 loads/FMA).
        // Lanes past flat_total / sk read unwritten SMEM; their accumulators are dropped
        // at the store, so no per-iteration bounds predicate is needed here.
        if (tile_len == kKvRankTile) {
            #pragma unroll
            for (int r_local = 0; r_local < kKvRankTile; ++r_local) {
                acc_t qa_reg[kScoreOutTile], ck_reg[kScoreOutTile];
                #pragma unroll
                for (int di = 0; di < kScoreOutTile; ++di)
                    qa_reg[di] = qa_smem[lf[di] * kKvRankTile + (r_local ^ lf_m[di])];
                #pragma unroll
                for (int dk = 0; dk < kScoreOutTile; ++dk)
                    ck_reg[dk] = ck_smem[lk[dk] * kKvRankTile + (r_local ^ lk_m[dk])];
                #pragma unroll
                for (int di = 0; di < kScoreOutTile; ++di)
                    #pragma unroll
                    for (int dk = 0; dk < kScoreOutTile; ++dk)
                        s[di][dk] += qa_reg[di] * ck_reg[dk];
            }
        } else {
            // ragged final kv_rank tile: same body, runtime trip count
            for (int r_local = 0; r_local < tile_len; ++r_local) {
                acc_t qa_reg[kScoreOutTile], ck_reg[kScoreOutTile];
                #pragma unroll
                for (int di = 0; di < kScoreOutTile; ++di)
                    qa_reg[di] = qa_smem[lf[di] * kKvRankTile + (r_local ^ lf_m[di])];
                #pragma unroll
                for (int dk = 0; dk < kScoreOutTile; ++dk)
                    ck_reg[dk] = ck_smem[lk[dk] * kKvRankTile + (r_local ^ lk_m[dk])];
                #pragma unroll
                for (int di = 0; di < kScoreOutTile; ++di)
                    #pragma unroll
                    for (int dk = 0; dk < kScoreOutTile; ++dk)
                        s[di][dk] += qa_reg[di] * ck_reg[dk];
            }
        }
        __syncthreads();  // all reads done before next tile (or rope) overwrites SMEM
    }

    // ---- Phase 2: scores_rope — accumulate into same s[][] (no second tile regs) ----
    acc_t* qr_smem = qa_smem;  // [64][rope_dim] with XOR swizzle
    acc_t* pe_smem = ck_smem;  // [128][rope_dim] with XOR swizzle

    const int qr_elems = kFlatsPerThreadBlock * rope_dim;
    for (int idx = threadIdx.x + threadIdx.y * blockDim.x;
         idx < qr_elems; idx += coop_stride) {
        // q_rope[flat, d] contiguous in d
        const int lf_s = idx / rope_dim;
        const int d    = idx % rope_dim;
        const int flat_s = flat_base + lf_s;
        if (flat_s < flat_total)
            qr_smem[lf_s * rope_dim + (d ^ (lf_s & (rope_dim - 1)))] =
                static_cast<acc_t>(q_rope[flat_s * rope_dim + d]);
    }

    const int pe_elems = kKsPerThreadBlock * rope_dim;
    for (int idx = threadIdx.x + threadIdx.y * blockDim.x;
         idx < pe_elems; idx += coop_stride) {
        // pe_cache[k, d] contiguous in d
        const int lk_s = idx / rope_dim;
        const int d    = idx % rope_dim;
        const int k_s  = k0 + lk_s;
        if (k_s < sk)
            pe_smem[lk_s * rope_dim + (d ^ (lk_s & (rope_dim - 1)))] =
                static_cast<acc_t>(pe_cache[pe_batch_stride + k_s * rope_dim + d]);
    }
    __syncthreads();

    // rope row stride is kRopeDim, so the swizzle masks differ from the kv_rank phase
    int lf_rm[kScoreOutTile], lk_rm[kScoreOutTile];
    #pragma unroll
    for (int di = 0; di < kScoreOutTile; ++di)
        lf_rm[di] = lf[di] & (kRopeDim - 1);
    #pragma unroll
    for (int dk = 0; dk < kScoreOutTile; ++dk)
        lk_rm[dk] = lk[dk] & (kRopeDim - 1);

    // d-outer register strip; rope_dim is a full compile-time tile, so no ragged path
    #pragma unroll
    for (int d = 0; d < kRopeDim; ++d) {
        acc_t qr_reg[kScoreOutTile], pe_reg[kScoreOutTile];
        #pragma unroll
        for (int di = 0; di < kScoreOutTile; ++di)
            qr_reg[di] = qr_smem[lf[di] * kRopeDim + (d ^ lf_rm[di])];
        #pragma unroll
        for (int dk = 0; dk < kScoreOutTile; ++dk)
            pe_reg[dk] = pe_smem[lk[dk] * kRopeDim + (d ^ lk_rm[dk])];
        #pragma unroll
        for (int di = 0; di < kScoreOutTile; ++di)
            #pragma unroll
            for (int dk = 0; dk < kScoreOutTile; ++dk)
                s[di][dk] += qr_reg[di] * pe_reg[dk];
    }

    #pragma unroll
    for (int di = 0; di < kScoreOutTile; ++di) {
        if (!valid_f[di]) continue;
        #pragma unroll
        for (int dk = 0; dk < kScoreOutTile; ++dk) {
            if (!valid_k[dk]) continue;
            scores[flat[di] * sk + k_idx[dk]] = s[di][dk] * inv_scale;
        }
    }
}

// --- 3b. Softmax over the Sk dimension, in-place on scores ---
// One block per (b, q, h) row; threads reduce within the row.
template <typename acc_t>
__global__ void naive_softmax_kernel(
    acc_t* __restrict__ scores,  // [B, Sq, H, Sk]  — modified in place
    int batch_size,
    int sq,
    int num_heads,
    int sk)
{
    int bq_idx   = blockIdx.x;  // b*sq + q
    int head_idx = blockIdx.y;  // h

    if (bq_idx >= batch_size * sq || head_idx >= num_heads) return;

    acc_t* row = scores + (bq_idx * num_heads + head_idx) * sk;  // [Sk] of [B,Sq,128,Sk]

    // max over Sk (numerical stability); -inf so idle threads never win the reduction
    acc_t max_val = -INFINITY;
    for (int k = threadIdx.x; k < sk; k += blockDim.x)
        max_val = fmax(max_val, row[k]);
    // Block-wide reduction of max  (naive: shared memory reduce)
    __shared__ acc_t smem[256];
    smem[threadIdx.x] = max_val;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            smem[threadIdx.x] = fmax(smem[threadIdx.x], smem[threadIdx.x + stride]);
        __syncthreads();
    }
    max_val = smem[0];

    // exp and sum over Sk
    acc_t sum_exp = acc_t(0);
    for (int k = threadIdx.x; k < sk; k += blockDim.x) {
        acc_t e = exp(row[k] - max_val);
        row[k] = e;
        sum_exp += e;
    }
    smem[threadIdx.x] = sum_exp;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }
    sum_exp = smem[0];

    // normalize → attn[b,q,h,:]  [B,Sq,128,Sk]
    for (int k = threadIdx.x; k < sk; k += blockDim.x)
        row[k] /= sum_exp;
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
//   block: 16×32 → each thread 4 flats × 4 ranks; Sk reduced in SMEM tiles of 64
//   Flat ownership interleaved: lf = tx + di*Bx; lr = ty*4 + dr (same as 3a k-side)
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
    const int flat_base  = blockIdx.x * kFlatsPerThreadBlock;
    const int r0         = blockIdx.y * kRsPerThreadBlock;

    // this block's slice of the Sk reduction; sk_chunk == sk when gridDim.z == 1
    const int k_begin = blockIdx.z * sk_chunk;
    const int k_end   = min(k_begin + sk_chunk, sk);

    acc_t* smem      = dynamic_smem<acc_t>();
    acc_t* attn_smem = smem;  // [64][kCtxSkTile]
    acc_t* ck_smem   = attn_smem + kFlatsPerThreadBlock * kCtxSkTile;  // [128][kCtxSkTile]

    // one batch per block when flat tile does not cross Sq*H (true for H=128, tile=64)
    const int batch_idx = min(flat_base, max(flat_total - 1, 0)) / (sq * num_heads);
    const int ck_batch_stride = batch_idx * sk * kv_rank;

    const int coop_stride = blockDim.x * blockDim.y;

    acc_t s[kScoreOutTile][kScoreOutTile];
    #pragma unroll
    for (int di = 0; di < kScoreOutTile; ++di)
        #pragma unroll
        for (int dr = 0; dr < kScoreOutTile; ++dr)
            s[di][dr] = acc_t(0);

    int lf[kScoreOutTile], lr[kScoreOutTile];
    int flat[kScoreOutTile], r_idx[kScoreOutTile];
    bool valid_f[kScoreOutTile], valid_r[kScoreOutTile];
    #pragma unroll
    for (int di = 0; di < kScoreOutTile; ++di) {
        lf[di]      = threadIdx.x + di * blockDim.x;
        flat[di]    = flat_base + lf[di];
        valid_f[di] = flat[di] < flat_total;
    }
    #pragma unroll
    for (int dr = 0; dr < kScoreOutTile; ++dr) {
        lr[dr]      = threadIdx.y * kScoreOutTile + dr;
        r_idx[dr]   = r0 + lr[dr];
        valid_r[dr] = r_idx[dr] < kv_rank;
    }

    // XOR swizzle masks — per-thread constants, hoisted out of the Sk reduction
    int lf_m[kScoreOutTile], lr_m[kScoreOutTile];
    #pragma unroll
    for (int di = 0; di < kScoreOutTile; ++di)
        lf_m[di] = lf[di] & (kCtxSkTile - 1);
    #pragma unroll
    for (int dr = 0; dr < kScoreOutTile; ++dr)
        lr_m[dr] = lr[dr] & (kCtxSkTile - 1);

    for (int k0 = k_begin; k0 < k_end; k0 += kCtxSkTile) {
        const int tile_len =
            (k0 + kCtxSkTile <= k_end) ? kCtxSkTile : (k_end - k0);

        // stage attn[flat, k0:k0+tile) — k contiguous → k_local = idx % tile_len
        const int attn_elems = kFlatsPerThreadBlock * tile_len;
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
        const int ck_elems = tile_len * kRsPerThreadBlock;
        for (int idx = threadIdx.x + threadIdx.y * blockDim.x;
             idx < ck_elems; idx += coop_stride) {
            const int k_local = idx / kRsPerThreadBlock;
            const int lr_s    = idx % kRsPerThreadBlock;
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
                acc_t attn_reg[kScoreOutTile], ck_reg[kScoreOutTile];
                #pragma unroll
                for (int di = 0; di < kScoreOutTile; ++di)
                    attn_reg[di] = attn_smem[lf[di] * kCtxSkTile + (k_local ^ lf_m[di])];
                #pragma unroll
                for (int dr = 0; dr < kScoreOutTile; ++dr)
                    ck_reg[dr] = ck_smem[lr[dr] * kCtxSkTile + (k_local ^ lr_m[dr])];
                #pragma unroll
                for (int di = 0; di < kScoreOutTile; ++di)
                    #pragma unroll
                    for (int dr = 0; dr < kScoreOutTile; ++dr)
                        s[di][dr] += attn_reg[di] * ck_reg[dr];
            }
        } else {
            // ragged final Sk tile: same body, runtime trip count
            for (int k_local = 0; k_local < tile_len; ++k_local) {
                acc_t attn_reg[kScoreOutTile], ck_reg[kScoreOutTile];
                #pragma unroll
                for (int di = 0; di < kScoreOutTile; ++di)
                    attn_reg[di] = attn_smem[lf[di] * kCtxSkTile + (k_local ^ lf_m[di])];
                #pragma unroll
                for (int dr = 0; dr < kScoreOutTile; ++dr)
                    ck_reg[dr] = ck_smem[lr[dr] * kCtxSkTile + (k_local ^ lr_m[dr])];
                #pragma unroll
                for (int di = 0; di < kScoreOutTile; ++di)
                    #pragma unroll
                    for (int dr = 0; dr < kScoreOutTile; ++dr)
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
    for (int di = 0; di < kScoreOutTile; ++di) {
        if (!valid_f[di]) continue;
        #pragma unroll
        for (int dr = 0; dr < kScoreOutTile; ++dr) {
            if (!valid_r[dr]) continue;
            ctx_store(&ctx[flat[di] * kv_rank + r_idx[dr]], s[di][dr], use_atomic);
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
    acc_t* ctx_smem = smem;                                  // [64][kOutRTile]
    acc_t* w_smem   = ctx_smem + kOutBqTile * kOutRTile;     // [128][kOutRTile]

    // ctx[bq, h, r] and W_uv[h, d, r] are both contiguous in r
    const int ctx_head_base = head_idx * kv_rank;
    const int w_head_base   = head_idx * v_dim * kv_rank;
    const int coop_stride   = blockDim.x * blockDim.y;

    acc_t s[kScoreOutTile][kScoreOutTile];
    #pragma unroll
    for (int db = 0; db < kScoreOutTile; ++db)
        #pragma unroll
        for (int dd = 0; dd < kScoreOutTile; ++dd)
            s[db][dd] = acc_t(0);

    int lb[kScoreOutTile], ld[kScoreOutTile];
    int bq[kScoreOutTile], d_idx[kScoreOutTile];
    bool valid_b[kScoreOutTile], valid_d[kScoreOutTile];
    #pragma unroll
    for (int db = 0; db < kScoreOutTile; ++db) {
        lb[db]      = threadIdx.y * kScoreOutTile + db;
        bq[db]      = bq_base + lb[db];
        valid_b[db] = bq[db] < bq_total;
    }
    #pragma unroll
    for (int dd = 0; dd < kScoreOutTile; ++dd) {
        ld[dd]      = threadIdx.x + dd * blockDim.x;
        d_idx[dd]   = d_base + ld[dd];
        valid_d[dd] = d_idx[dd] < v_dim;
    }

    // XOR swizzle masks — per-thread constants, hoisted out of the reduction
    int lb_m[kScoreOutTile], ld_m[kScoreOutTile];
    #pragma unroll
    for (int db = 0; db < kScoreOutTile; ++db)
        lb_m[db] = lb[db] & (kOutRTile - 1);
    #pragma unroll
    for (int dd = 0; dd < kScoreOutTile; ++dd)
        ld_m[dd] = ld[dd] & (kOutRTile - 1);

    for (int r0 = 0; r0 < kv_rank; r0 += kOutRTile) {
        const int tile_len =
            (r0 + kOutRTile <= kv_rank) ? kOutRTile : (kv_rank - r0);

        // stage ctx[bq_base:+64, h, r0:+tile) — r contiguous → r_local = idx % tile_len
        const int ctx_elems = kOutBqTile * tile_len;
        for (int idx = threadIdx.x + threadIdx.y * blockDim.x;
             idx < ctx_elems; idx += coop_stride) {
            const int lb_s    = idx / tile_len;
            const int r_local = idx % tile_len;
            const int bq_s    = bq_base + lb_s;
            if (bq_s < bq_total)
                ctx_smem[lb_s * kOutRTile + (r_local ^ (lb_s & (kOutRTile - 1)))] =
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
                w_smem[ld_s * kOutRTile + (r_local ^ (ld_s & (kOutRTile - 1)))] =
                    static_cast<acc_t>(
                        W_uv[w_head_base + d_s * kv_rank + r0 + r_local]);
        }
        __syncthreads();

        // r-outer register strip; out-of-range lanes read unwritten SMEM and are
        // dropped at the store, so no per-iteration bounds predicate is needed
        if (tile_len == kOutRTile) {
            #pragma unroll 8
            for (int r_local = 0; r_local < kOutRTile; ++r_local) {
                acc_t ctx_reg[kScoreOutTile], w_reg[kScoreOutTile];
                #pragma unroll
                for (int db = 0; db < kScoreOutTile; ++db)
                    ctx_reg[db] = ctx_smem[lb[db] * kOutRTile + (r_local ^ lb_m[db])];
                #pragma unroll
                for (int dd = 0; dd < kScoreOutTile; ++dd)
                    w_reg[dd] = w_smem[ld[dd] * kOutRTile + (r_local ^ ld_m[dd])];
                #pragma unroll
                for (int db = 0; db < kScoreOutTile; ++db)
                    #pragma unroll
                    for (int dd = 0; dd < kScoreOutTile; ++dd)
                        s[db][dd] += ctx_reg[db] * w_reg[dd];
            }
        } else {
            // ragged final kv_rank tile: same body, runtime trip count
            for (int r_local = 0; r_local < tile_len; ++r_local) {
                acc_t ctx_reg[kScoreOutTile], w_reg[kScoreOutTile];
                #pragma unroll
                for (int db = 0; db < kScoreOutTile; ++db)
                    ctx_reg[db] = ctx_smem[lb[db] * kOutRTile + (r_local ^ lb_m[db])];
                #pragma unroll
                for (int dd = 0; dd < kScoreOutTile; ++dd)
                    w_reg[dd] = w_smem[ld[dd] * kOutRTile + (r_local ^ ld_m[dd])];
                #pragma unroll
                for (int db = 0; db < kScoreOutTile; ++db)
                    #pragma unroll
                    for (int dd = 0; dd < kScoreOutTile; ++dd)
                        s[db][dd] += ctx_reg[db] * w_reg[dd];
            }
        }
        __syncthreads();  // all reads done before the next tile overwrites SMEM
    }

    #pragma unroll
    for (int db = 0; db < kScoreOutTile; ++db) {
        if (!valid_b[db]) continue;
        const int out_base = (bq[db] * num_heads + head_idx) * v_dim;
        #pragma unroll
        for (int dd = 0; dd < kScoreOutTile; ++dd) {
            if (!valid_d[dd]) continue;
            out[out_base + d_idx[dd]] = static_cast<scalar_t>(s[db][dd]);
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
    const int ctx_flat_tiles = (flat_total + kFlatsPerThreadBlock - 1) / kFlatsPerThreadBlock;
    const int ctx_r_tiles    = (kv_rank    + kRsPerThreadBlock    - 1) / kRsPerThreadBlock;
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

    // 3a: grid.x=flat(B*Sq*H), grid.y=Sk; 16×32 threads → 64×128 scores/block;
    //     kv_rank reduction iterated in SMEM tiles of kKvRankTile=64
    {
        dim3 threads(kScoreThreadsX, kScoreThreadsY);  // 16×32 = 512 threads
        dim3 blocks(
            (flat_total + kFlatsPerThreadBlock - 1) / kFlatsPerThreadBlock,
            (sk         + kKsPerThreadBlock    - 1) / kKsPerThreadBlock);
        // SMEM: qa[64][64] + ck[128][64] acc_t with XOR col swizzle (qr/pe reuse)
        size_t smem = static_cast<size_t>(
            kFlatsPerThreadBlock * kKvRankTile + kKsPerThreadBlock * kKvRankTile) * acc_size;
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(q_absorbed.scalar_type(), "mla_scores", ([&] {
            using acc_t = acc_t_of<scalar_t>;
            auto kernel = mla_scores_kernel<scalar_t>;
            enable_large_smem(kernel, smem);  // fp64 tiles exceed the default 48 KB
            kernel<<<blocks, threads, smem>>>(
                q_absorbed.data_ptr<scalar_t>(), q_rope.data_ptr<scalar_t>(),
                c_kv.data_ptr<scalar_t>(),       pe_cache.data_ptr<scalar_t>(),
                scores.data_ptr<acc_t>(),
                batch_size, sq, sk, num_heads, kv_rank, rope_dim,
                static_cast<acc_t>(scale));
        }));
        cudaDeviceSynchronize();
    }

    // 3b: softmax in-place on scores → attn [B,Sq,128,Sk]
    {
        dim3 blocks(batch_size * sq, num_heads);
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(q_absorbed.scalar_type(), "mla_softmax", ([&] {
            using acc_t = acc_t_of<scalar_t>;
            naive_softmax_kernel<acc_t><<<blocks, kBlockThreads>>>(
                scores.data_ptr<acc_t>(), batch_size, sq, num_heads, sk);
        }));
        cudaDeviceSynchronize();
    }

    // 3c: grid.x=flat(B*Sq*H), grid.y=kv_rank, grid.z=split-K slice over Sk;
    //     16×32 → 64×128 ctx/block; Sk reduced in SMEM tiles of kCtxSkTile=64
    {
        // atomicAdd accumulates, so the partials need a zeroed destination
        if (ctx_atomic) ctx.zero_();

        dim3 threads(kScoreThreadsX, kScoreThreadsY);  // 16×32 = 512 threads
        dim3 blocks(ctx_flat_tiles, ctx_r_tiles, ctx_split);
        // SMEM: attn[64][64] + ck[128][64] acc_t with XOR col swizzle
        size_t smem = static_cast<size_t>(
            kFlatsPerThreadBlock * kCtxSkTile + kRsPerThreadBlock * kCtxSkTile) * acc_size;
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(q_absorbed.scalar_type(), "mla_ctx", ([&] {
            using acc_t = acc_t_of<scalar_t>;
            if (ctx_atomic) {
                auto kernel = mla_ctx_kernel<scalar_t, acc_t>;
                enable_large_smem(kernel, smem);
                kernel<<<blocks, threads, smem>>>(
                    scores.data_ptr<acc_t>(), c_kv.data_ptr<scalar_t>(),
                    ctx.data_ptr<acc_t>(),
                    batch_size, sq, sk, num_heads, kv_rank, ctx_sk_chunk, true);
            } else {
                auto kernel = mla_ctx_kernel<scalar_t, scalar_t>;
                enable_large_smem(kernel, smem);
                kernel<<<blocks, threads, smem>>>(
                    scores.data_ptr<acc_t>(), c_kv.data_ptr<scalar_t>(),
                    ctx.data_ptr<scalar_t>(),
                    batch_size, sq, sk, num_heads, kv_rank, ctx_sk_chunk, false);
            }
        }));
        cudaDeviceSynchronize();
    }

    // 3d: grid.x=bq tiles, grid.y=v_dim tiles, grid.z=head; 32×16 → 64 bq × 128 v/block
    //     kv_rank reduction in SMEM tiles of kOutRTile=64; W_uv[h] staged per block
    {
        const int bq_total = batch_size * sq;
        dim3 threads(kOutThreadsX, kOutThreadsY);  // 32×16 = 512 threads
        dim3 blocks(
            (bq_total + kOutBqTile - 1) / kOutBqTile,
            (v_dim    + kOutVTile  - 1) / kOutVTile,
            num_heads);
        // SMEM: ctx[64][64] + w[128][64] acc_t with XOR col swizzle
        size_t smem = static_cast<size_t>(
            kOutBqTile * kOutRTile + kOutVTile * kOutRTile) * acc_size;
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(q_absorbed.scalar_type(), "mla_output", ([&] {
            using acc_t = acc_t_of<scalar_t>;
            if (ctx_atomic) {  // 3c promoted ctx to the accumulate dtype for the atomics
                auto kernel = mla_output_smem_kernel<scalar_t, acc_t>;
                enable_large_smem(kernel, smem);
                kernel<<<blocks, threads, smem>>>(
                    ctx.data_ptr<acc_t>(), W_uv.data_ptr<scalar_t>(),
                    out.data_ptr<scalar_t>(),
                    bq_total, num_heads, kv_rank, v_dim);
            } else {
                auto kernel = mla_output_smem_kernel<scalar_t, scalar_t>;
                enable_large_smem(kernel, smem);
                kernel<<<blocks, threads, smem>>>(
                    ctx.data_ptr<scalar_t>(), W_uv.data_ptr<scalar_t>(),
                    out.data_ptr<scalar_t>(),
                    bq_total, num_heads, kv_rank, v_dim);
            }
        }));
        cudaDeviceSynchronize();
    }

    return out;  // [B, Sq, 128, 128]
}


// ============================================================================
// Python bindings
// ============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_decompress_rope", &launch_mla_fused_rope,
          "Legacy: c_kv @ W_k_rope + RoPE (not official MLA cache layout)");
    m.def("q_absorbed", &launch_q_absorbed,
          "Kernel 2 — Q path: x@W_q → Q_nope@W_uk + RoPE(Q_rope)");
    m.def("mla_attention", &launch_mla_attention,
          "Kernel 3 — Official absorbed MLA: scores → softmax → ctx@c_kv → @W_uv");
}