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
// Kernel 3a scores: each thread → 4×4 output tile over (flat q-side, k-side)
static constexpr int kScoreOutTile   = 4;   // 4 along flat (q) × 4 along Sk (c_kv)
static constexpr int kScoreThreadsX  = 16;  // 16×4 = 64 flats per block
static constexpr int kScoreThreadsY  = 32;  // 32×4 = 128 keys per block  (Sk ≫ other dims)
static constexpr int kFlatsPerThreadBlock  = kScoreThreadsX * kScoreOutTile;  // 64
static constexpr int kKsPerThreadBlock     = kScoreThreadsY * kScoreOutTile;  // 128
static constexpr int kKvRankTile     = 64;  // SMEM tile along kv_rank=512 (reduction dim)
static constexpr int kRopeDim        = 64;  // qk_rope_head_dim (compile-time for unroll)
static constexpr int kSkTile         = 4096; // SMEM tile along Sk for ctx kernel attn row

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
//      scores_nope: [B,Sq,128,512] @ [B,Sk,512]^T → [B,Sq,128,Sk]
//      scores_rope: [B,Sq,128,64]  @ [B,Sk,64]^T  → [B,Sq,128,Sk]
//      merge + /sqrt(192)
//  3b. softmax → attn [B,Sq,128,Sk]   (one block per (B*Sq, H) row)
//  3c. ctx:     [B,Sq,128,Sk] @ [B,Sk,512]    → [B,Sq,128,512]  (Sk tiled by kSkTile)
//  3d. out:     [B,Sq,128,512] × [128,128,512] → [B,Sq,128,128]
// ============================================================================

// --- 3a. scores: 4×4 output tile per thread; grid (flat B*Sq*H, Sk) ---
//   grid.x: flat = b*(Sq*128) + q*128 + h     ∈ [0, B*Sq*128)  — 64 flats/block
//   grid.y: k ∈ [0, Sk)                                        — 128 keys/block
//   block: 16×32 threads → each thread 4 flats × 4 keys (64×128 / block)
//   Flat ownership interleaved: lf = tx + di*blockDim.x
//   SMEM [lf][r] / [lk][r] with col XOR swizzle (no padding): bank = (r^row)%32
//   Staging: r (or rope d) is idx% — matches HBM contiguous dim for coalescing
template <typename scalar_t>
__global__ void mla_scores_smem_kernel(
    const scalar_t* __restrict__ q_absorbed, // [B, Sq, H, kv_rank=512]
    const scalar_t* __restrict__ q_rope,     // [B, Sq, H, rope_dim=64]
    const scalar_t* __restrict__ c_kv,       // [B, Sk, kv_rank=512]
    const scalar_t* __restrict__ pe_cache,   // [B, Sk, rope_dim=64]  head-shared
    float*          __restrict__ scores,     // [B, Sq, H, Sk]  fp32
    int batch_size,
    int sq,
    int sk,
    int num_heads,
    int kv_rank,
    int rope_dim,
    float scale)
{
    const int flat_total = batch_size * sq * num_heads;  // B*Sq*128
    const int flat_base  = blockIdx.x * kFlatsPerThreadBlock;
    const int k0         = blockIdx.y * kKsPerThreadBlock;

    // qa_smem[64][kKvRankTile]; ck_smem[128][kKvRankTile] — reused as qr/pe after phase 1
    extern __shared__ float smem[];
    float* qa_smem = smem;
    float* ck_smem = qa_smem + kFlatsPerThreadBlock * kKvRankTile;

    // batch for c_kv / pe_cache rows (64 flats/block ⊂ one batch when block < Sq*H)
    const int batch_idx = min(flat_base, max(flat_total - 1, 0)) / (sq * num_heads);
    const int ck_batch_stride = batch_idx * sk * kv_rank;
    const int pe_batch_stride = batch_idx * sk * rope_dim;

    const int coop_stride = blockDim.x * blockDim.y;
    const float inv_scale = 1.0f / scale;

    // Per-thread output tile; rope later accumulates into the same regs
    float s[kScoreOutTile][kScoreOutTile];
    #pragma unroll
    for (int di = 0; di < kScoreOutTile; ++di)
        #pragma unroll
        for (int dk = 0; dk < kScoreOutTile; ++dk)
            s[di][dk] = 0.0f;

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
                    static_cast<float>(q_absorbed[flat_s * kv_rank + r0 + r_local]);
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
                    static_cast<float>(c_kv[ck_batch_stride + k_s * kv_rank + r0 + r_local]);
        }
        __syncthreads();  // loads done before any thread reads this tile

        // partial dots; bounds already in valid_*
        #pragma unroll
        for (int dk = 0; dk < kScoreOutTile; ++dk) {
            if (!valid_k[dk]) continue;
            const int lk_s = lk[dk];
            const int lk_m = lk_s & (kKvRankTile - 1);
            #pragma unroll
            for (int di = 0; di < kScoreOutTile; ++di) {
                if (!valid_f[di]) continue;
                const int lf_s = lf[di];
                const int lf_m = lf_s & (kKvRankTile - 1);
                #pragma unroll
                for (int r_local = 0; r_local < kKvRankTile; ++r_local) {
                    if (r_local < tile_len)
                        s[di][dk] += qa_smem[lf_s * kKvRankTile + (r_local ^ lf_m)]
                                   * ck_smem[lk_s * kKvRankTile + (r_local ^ lk_m)];
                }
            }
        }
        __syncthreads();  // all reads done before next tile (or rope) overwrites SMEM
    }

    // ---- Phase 2: scores_rope — accumulate into same s[][] (no second tile regs) ----
    float* qr_smem = qa_smem;  // [64][rope_dim] with XOR swizzle
    float* pe_smem = ck_smem;  // [128][rope_dim] with XOR swizzle

    const int qr_elems = kFlatsPerThreadBlock * rope_dim;
    for (int idx = threadIdx.x + threadIdx.y * blockDim.x;
         idx < qr_elems; idx += coop_stride) {
        // q_rope[flat, d] contiguous in d
        const int lf_s = idx / rope_dim;
        const int d    = idx % rope_dim;
        const int flat_s = flat_base + lf_s;
        if (flat_s < flat_total)
            qr_smem[lf_s * rope_dim + (d ^ (lf_s & (rope_dim - 1)))] =
                static_cast<float>(q_rope[flat_s * rope_dim + d]);
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
                static_cast<float>(pe_cache[pe_batch_stride + k_s * rope_dim + d]);
    }
    __syncthreads();

    #pragma unroll
    for (int dk = 0; dk < kScoreOutTile; ++dk) {
        if (!valid_k[dk]) continue;
        const int lk_s = lk[dk];
        const int lk_m = lk_s & (kRopeDim - 1);
        #pragma unroll
        for (int di = 0; di < kScoreOutTile; ++di) {
            if (!valid_f[di]) continue;
            const int lf_s = lf[di];
            const int lf_m = lf_s & (kRopeDim - 1);
            #pragma unroll
            for (int d = 0; d < kRopeDim; ++d)
                s[di][dk] += qr_smem[lf_s * kRopeDim + (d ^ lf_m)]
                           * pe_smem[lk_s * kRopeDim + (d ^ lk_m)];
        }
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
__global__ void naive_softmax_kernel(
    float* __restrict__ scores,  // [B, Sq, H, Sk]  — modified in place
    int batch_size,
    int sq,
    int num_heads,
    int sk)
{
    int bq_idx   = blockIdx.x;  // b*sq + q
    int head_idx = blockIdx.y;  // h

    if (bq_idx >= batch_size * sq || head_idx >= num_heads) return;

    float* row = scores + (bq_idx * num_heads + head_idx) * sk;  // [Sk] of [B,Sq,128,Sk]

    // max over Sk (numerical stability)
    float max_val = -1e38f;
    for (int k = threadIdx.x; k < sk; k += blockDim.x)
        max_val = fmaxf(max_val, row[k]);
    // Block-wide reduction of max  (naive: shared memory reduce)
    __shared__ float smem[256];
    smem[threadIdx.x] = max_val;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
        __syncthreads();
    }
    max_val = smem[0];

    // exp and sum over Sk
    float sum_exp = 0.0f;
    for (int k = threadIdx.x; k < sk; k += blockDim.x) {
        float e = expf(row[k] - max_val);
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

// --- 3c. ctx = attn @ c_kv: attn row tiled along Sk in SMEM, threads over kv_rank=512 ---
template <typename scalar_t>
__global__ void mla_ctx_smem_kernel(
    const float*    __restrict__ attn,  // [B, Sq, H, Sk]
    const scalar_t* __restrict__ c_kv,  // [B, Sk, kv_rank]
    scalar_t*       __restrict__ ctx,   // [B, Sq, H, kv_rank]
    int sq,
    int sk,
    int num_heads,
    int kv_rank)
{
    // attn_smem[kSkTile] — tile along Sk (not full Sk; 64k would exceed SMEM limit)
    extern __shared__ float attn_smem[];

    int bq_idx   = blockIdx.x;   // b*sq + q
    int head_idx = blockIdx.y;   // h
    int b = bq_idx / sq;

    const float* attn_row = attn + (bq_idx * num_heads + head_idx) * sk;
    const int ck_batch_stride = b * sk * kv_rank;
    const int ctx_base = (bq_idx * num_heads + head_idx) * kv_rank;

    // Each thread owns up to two r indices (kv_rank=512, blockDim.x=256)
    const int r0 = threadIdx.x;
    const int r1 = threadIdx.x + blockDim.x;
    float acc0 = 0.0f, acc1 = 0.0f;

    // Tile Sk outermost so ALL threads hit the same __syncthreads (no sync inside
    // a divergent r-loop). Also loads each attn tile once, not once per r.
    for (int k0 = 0; k0 < sk; k0 += kSkTile) {
        const int tile_len =
            (k0 + kSkTile <= sk) ? kSkTile : (sk - k0);

        for (int kl = threadIdx.x; kl < tile_len; kl += blockDim.x)
            attn_smem[kl] = attn_row[k0 + kl];
        __syncthreads();

        for (int kl = 0; kl < tile_len; ++kl) {
            const int k = k0 + kl;
            const float a = attn_smem[kl];
            acc0 += a * static_cast<float>(
                c_kv[ck_batch_stride + k * kv_rank + r0]);
            if (r1 < kv_rank)
                acc1 += a * static_cast<float>(
                    c_kv[ck_batch_stride + k * kv_rank + r1]);
        }
        __syncthreads();
    }

    if (r0 < kv_rank)
        ctx[ctx_base + r0] = static_cast<scalar_t>(acc0);
    if (r1 < kv_rank)
        ctx[ctx_base + r1] = static_cast<scalar_t>(acc1);
}

// --- 3d. out = ctx @ W_uv: ctx[512] in SMEM, threads over v_dim=128 ---
template <typename scalar_t>
__global__ void mla_output_smem_kernel(
    const scalar_t* __restrict__ ctx,   // [B, Sq, H, kv_rank]
    const scalar_t* __restrict__ W_uv,  // [H, v_dim, kv_rank]
    scalar_t*       __restrict__ out,   // [B, Sq, H, v_dim]
    int num_heads,
    int kv_rank,
    int v_dim)
{
    extern __shared__ float ctx_smem[];

    int bq_idx   = blockIdx.x;   // b*sq + q
    int head_idx = blockIdx.y;   // h

    // stage ctx[b,q,h,:] → ctx_smem[512]
    const int ctx_base = (bq_idx * num_heads + head_idx) * kv_rank;
    for (int r = threadIdx.x; r < kv_rank; r += blockDim.x)
        ctx_smem[r] = static_cast<float>(ctx[ctx_base + r]);
    __syncthreads();

    // out[b,q,h,d] = ctx[b,q,h,0:512] · W_uv[h,d,0:512]  → [B,Sq,128,128]
    // contract r=512 (einsum bqhr,hdr→bqhd)
    const int out_base = (bq_idx * num_heads + head_idx) * v_dim;
    for (int d_idx = threadIdx.x; d_idx < v_dim; d_idx += blockDim.x) {
        float acc = 0.0f;
        for (int r = 0; r < kv_rank; ++r) {
            // W_uv [128, 128, 512]: W_uv[h, d, r]
            float w = static_cast<float>(
                W_uv[head_idx * v_dim * kv_rank + d_idx * kv_rank + r]);
            acc += ctx_smem[r] * w;
        }
        out[out_base + d_idx] = static_cast<scalar_t>(acc);
    }
}

// Host launcher: official absorbed MLA attention
torch::Tensor launch_mla_attention(
    torch::Tensor q_absorbed,  // [B, Sq, H, kv_rank]
    torch::Tensor q_rope,      // [B, Sq, H, rope_dim]
    torch::Tensor c_kv,        // [B, Sk, kv_rank]
    torch::Tensor pe_cache,    // [B, Sk, rope_dim]
    torch::Tensor W_uv,        // [H, v_dim, kv_rank]
    float scale)
{
    int batch_size = q_absorbed.size(0), sq = q_absorbed.size(1);
    int num_heads  = q_absorbed.size(2), kv_rank = q_absorbed.size(3);
    int sk         = c_kv.size(1);
    int rope_dim   = q_rope.size(3);
    int v_dim      = W_uv.size(1);

    auto scores = torch::empty({batch_size, sq, num_heads, sk},
                               torch::TensorOptions().dtype(torch::kFloat32)
                                                     .device(q_absorbed.device()));  // [B,Sq,128,Sk]
    auto ctx = torch::empty({batch_size, sq, num_heads, kv_rank}, q_absorbed.options());  // [B,Sq,128,512]
    auto out = torch::empty({batch_size, sq, num_heads, v_dim}, q_absorbed.options());     // [B,Sq,128,128]

    // 3a: grid.x=flat(B*Sq*H), grid.y=Sk; 16×32 threads → 64×128 scores/block;
    //     kv_rank reduction iterated in SMEM tiles of kKvRankTile=64
    {
        const int flat_total = batch_size * sq * num_heads;
        dim3 threads(kScoreThreadsX, kScoreThreadsY);  // 16×32 = 512 threads
        dim3 blocks(
            (flat_total + kFlatsPerThreadBlock - 1) / kFlatsPerThreadBlock,
            (sk         + kKsPerThreadBlock    - 1) / kKsPerThreadBlock);
        // SMEM: qa[64][64] + ck[128][64] floats with XOR col swizzle (qr/pe reuse)
        size_t smem = static_cast<size_t>(
            kFlatsPerThreadBlock * kKvRankTile + kKsPerThreadBlock * kKvRankTile) * sizeof(float);
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(q_absorbed.scalar_type(), "mla_scores", ([&] {
            mla_scores_smem_kernel<scalar_t><<<blocks, threads, smem>>>(
                q_absorbed.data_ptr<scalar_t>(), q_rope.data_ptr<scalar_t>(),
                c_kv.data_ptr<scalar_t>(),       pe_cache.data_ptr<scalar_t>(),
                scores.data_ptr<float>(),
                batch_size, sq, sk, num_heads, kv_rank, rope_dim, scale);
        }));
        cudaDeviceSynchronize();
    }

    // 3b: softmax in-place on scores → attn [B,Sq,128,Sk]
    {
        dim3 blocks(batch_size * sq, num_heads);
        naive_softmax_kernel<<<blocks, kBlockThreads>>>(
            scores.data_ptr<float>(), batch_size, sq, num_heads, sk);
        cudaDeviceSynchronize();
    }

    // 3c: grid (B*Sq, 128); SMEM attn[kSkTile] @ c_kv → ctx [B,Sq,128,512]
    {
        dim3 blocks(batch_size * sq, num_heads);
        size_t smem = static_cast<size_t>(kSkTile) * sizeof(float);  // 4096 floats
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(q_absorbed.scalar_type(), "mla_ctx", ([&] {
            mla_ctx_smem_kernel<scalar_t><<<blocks, kBlockThreads, smem>>>(
                scores.data_ptr<float>(), c_kv.data_ptr<scalar_t>(),
                ctx.data_ptr<scalar_t>(),
                sq, sk, num_heads, kv_rank);
        }));
        cudaDeviceSynchronize();
    }

    // 3d: grid (B*Sq, 128); SMEM ctx[512] × W_uv → out [B,Sq,128,128]
    {
        dim3 blocks(batch_size * sq, num_heads);
        size_t smem = static_cast<size_t>(kv_rank) * sizeof(float);  // 512 floats
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(q_absorbed.scalar_type(), "mla_output", ([&] {
            mla_output_smem_kernel<scalar_t><<<blocks, kBlockThreads, smem>>>(
                ctx.data_ptr<scalar_t>(), W_uv.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                num_heads, kv_rank, v_dim);
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