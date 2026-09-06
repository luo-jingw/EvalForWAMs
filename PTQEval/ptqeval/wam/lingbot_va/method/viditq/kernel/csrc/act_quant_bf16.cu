// Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
//
// Per-token symmetric BF16 -> INT8 activation quantization.
//   for each row n in x [N, K]:
//       max_n = max(|x[n, k]|) over k
//       scale_n = max_n / 127     (clamped to >= 1e-8 to avoid div-by-zero)
//       x_q[n, k] = clamp(round(x[n, k] / scale_n), -127, 127)
//   returns (x_q [N, K] int8, scale_x [N] bf16)
//
// One CUDA block per row. Two-phase: cooperative reduction of |x| max,
// then cooperative quantization. Block dim is configurable up to 1024.

#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "dispatch_utils.h"


namespace {

__device__ inline int32_t block_reduce_sum_int32(int32_t v, int32_t* warp_smem) {
    unsigned mask = 0xFFFFFFFFu;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset);
    }
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    if (lane == 0) warp_smem[warp_id] = v;
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = (blockDim.x + 31) >> 5;
        int32_t w = (lane < num_warps) ? warp_smem[lane] : 0;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            w += __shfl_down_sync(mask, w, offset);
        }
        if (lane == 0) warp_smem[0] = w;
    }
    __syncthreads();
    return warp_smem[0];
}

__device__ inline float block_reduce_max(float v, float* warp_smem) {
    unsigned mask = 0xFFFFFFFFu;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(mask, v, offset));
    }
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    if (lane == 0) warp_smem[warp_id] = v;
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = (blockDim.x + 31) >> 5;
        float w = (lane < num_warps) ? warp_smem[lane] : 0.f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            w = fmaxf(w, __shfl_down_sync(mask, w, offset));
        }
        if (lane == 0) warp_smem[0] = w;
    }
    __syncthreads();
    return warp_smem[0];
}

__global__ void act_quant_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    int8_t* __restrict__ x_q,
    __nv_bfloat16* __restrict__ scale_x,
    int N, int K
) {
    int n = blockIdx.x;
    if (n >= N) return;

    const __nv_bfloat16* x_row = x + n * K;
    int8_t* xq_row = x_q + n * K;

    __shared__ float warp_smem[32];   // up to 1024 threads / 32 lanes/warp
    __shared__ float row_inv_scale;

    // ---- Phase 1: reduce |x| max ----
    float local_max = 0.f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(__bfloat162float(x_row[k])));
    }
    float row_max = block_reduce_max(local_max, warp_smem);

    if (threadIdx.x == 0) {
        float scale = row_max / 127.0f;
        if (scale < 1e-8f) scale = 1e-8f;
        row_inv_scale = 1.0f / scale;
        scale_x[n] = __float2bfloat16(scale);
    }
    __syncthreads();

    // ---- Phase 2: quantize ----
    float inv_scale = row_inv_scale;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        float v = __bfloat162float(x_row[k]) * inv_scale;
        int q = __float2int_rn(v);
        q = q > 127 ? 127 : q;
        q = q < -127 ? -127 : q;
        xq_row[k] = (int8_t)q;
    }
}

// Phase 26a-1: per-row sym quant + fused post-quant sum_x reduction.
// Same algorithm as ViDiT-Q QuantKernel<__nv_bfloat16, _, kPostQuant>,
// implemented with the grid-stride structure of act_quant_bf16_kernel
// (above) so any hidden_size up to the 1024-thread block-reduce limit
// is supported (handles LingBot-VA down_proj K=14336 which the
// upstream ViDiT-Q kernel's 1-thread-per-chunk design caps at 8192).
//
// sum_x[n] = scale_x[n] * sum_k(x_int8[n, k]), cast bf16
// (equivalent to dividing sum_int by inv_scale = 127/amax in fp32 then
// casting, matching ViDiT-Q's __int2float_rn(sum) / tmp_scale path).
__global__ void act_quant_bf16_with_sum_kernel(
    const __nv_bfloat16* __restrict__ x,
    int8_t* __restrict__ x_q,
    __nv_bfloat16* __restrict__ scale_x,
    __nv_bfloat16* __restrict__ sum_x,
    int N, int K
) {
    int n = blockIdx.x;
    if (n >= N) return;

    const __nv_bfloat16* x_row = x + n * K;
    int8_t* xq_row = x_q + n * K;

    // Shared warp-scratch reused across Phase 1 (max, float) and Phase 3
    // (sum, int32). Phases are __syncthreads-separated, so the cast-aliased
    // reuse is race-free and saves smem.
    __shared__ float warp_smem[32];
    __shared__ float row_inv_scale;

    // ---- Phase 1: reduce |x| max ----
    float local_max = 0.f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(__bfloat162float(x_row[k])));
    }
    float row_max = block_reduce_max(local_max, warp_smem);

    if (threadIdx.x == 0) {
        float scale = row_max / 127.0f;
        if (scale < 1e-8f) scale = 1e-8f;
        row_inv_scale = 1.0f / scale;       // inv_scale = 127 / amax
        scale_x[n] = __float2bfloat16(scale);
    }
    __syncthreads();

    // ---- Phase 2: quantize + accumulate per-thread int sum ----
    float inv_scale = row_inv_scale;
    int32_t local_sum = 0;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        float v = __bfloat162float(x_row[k]) * inv_scale;
        int q = __float2int_rn(v);
        q = q > 127 ? 127 : q;
        q = q < -127 ? -127 : q;
        xq_row[k] = (int8_t)q;
        local_sum += q;
    }

    // ---- Phase 3: block-reduce int sum and write sum_x ----
    int32_t row_sum_int = block_reduce_sum_int32(
        local_sum, reinterpret_cast<int32_t*>(warp_smem));

    if (threadIdx.x == 0) {
        // sum_x = sum_int / inv_scale (= sum_int * scale), matches ViDiT-Q
        // QuantKernel kPostQuant epilogue: from_float_rn(__int2float_rn(sum)
        // / tmp_scale) where tmp_scale = 127/amax = inv_scale here.
        float sum_f = __int2float_rn(row_sum_int) / inv_scale;
        sum_x[n] = __float2bfloat16(sum_f);
    }
}

// ---------------------------------------------------------------------------
// Phase 42 step 2: per-token per-group symmetric INT4 quant (group=128 along K).
//
// Output convention (NATURAL layout; the Atom-permuted layout the W4A4 GEMM
// expects is applied by a separate downstream helper, not here, to keep
// concerns split — see plan.txt G5 + the Phase 42 wrapper):
//   x_int4_packed[n, k/2]  = (int4[2k+1] & 0xF) << 4 | (int4[2k] & 0xF)
//   scale_x[n, g]          = amax(|x[n, g*128 : (g+1)*128]|) / 7
//   int4 in [-8, 7], stored as raw 4-bit two's-complement.
//
// One CUDA block per (token, group). Warp = 32 threads, 4 elements per lane
// (4 * 32 = 128 = GROUP). Warp-shuffle amax reduce, then quantize + pack;
// each lane writes 2 packed bytes. Lane 0 writes the group scale.
template <int GROUP>
__global__ void act_quant_bf16_group_sym_kernel(
    const __nv_bfloat16* __restrict__ x,    // [N, K]
    uint8_t* __restrict__ x_int4_packed,    // [N, K/2]
    __nv_bfloat16* __restrict__ scale_x,    // [N, K/GROUP]
    int K
) {
    static_assert(GROUP == 128, "GROUP must be 128 (1 warp, 4 elems/lane)");
    constexpr int PER_THREAD = GROUP / 32;  // 4

    int n = blockIdx.x;
    int g = blockIdx.y;
    int K_groups = K / GROUP;

    const __nv_bfloat16* x_grp = x + n * K + g * GROUP;
    uint8_t* p_grp = x_int4_packed + n * (K / 2) + g * (GROUP / 2);

    int tid = threadIdx.x;

    float v[PER_THREAD];
    float local_max = 0.f;
    #pragma unroll
    for (int i = 0; i < PER_THREAD; ++i) {
        v[i] = __bfloat162float(x_grp[tid * PER_THREAD + i]);
        local_max = fmaxf(local_max, fabsf(v[i]));
    }

    unsigned mask = 0xFFFFFFFFu;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(mask, local_max, offset));
    }
    float amax = local_max;
    float scale = amax / 7.0f;
    float inv_scale = (amax > 0.f) ? (7.0f / amax) : 0.f;

    if (tid == 0) {
        scale_x[n * K_groups + g] = __float2bfloat16(scale);
    }

    int8_t q[PER_THREAD];
    #pragma unroll
    for (int i = 0; i < PER_THREAD; ++i) {
        int qi = __float2int_rn(v[i] * inv_scale);
        qi = qi < -8 ? -8 : (qi > 7 ? 7 : qi);
        q[i] = static_cast<int8_t>(qi);
    }
    uint8_t b0 = static_cast<uint8_t>((q[1] & 0xF) << 4) | static_cast<uint8_t>(q[0] & 0xF);
    uint8_t b1 = static_cast<uint8_t>((q[3] & 0xF) << 4) | static_cast<uint8_t>(q[2] & 0xF);
    int byte_off = tid * (PER_THREAD / 2);
    p_grp[byte_off + 0] = b0;
    p_grp[byte_off + 1] = b1;
}

}  // anonymous namespace


std::tuple<torch::Tensor, torch::Tensor> act_quant_bf16(torch::Tensor x_bf16) {
    TORCH_CHECK(x_bf16.is_cuda(), "x_bf16 must be on CUDA");
    TORCH_CHECK(x_bf16.dtype() == torch::kBFloat16, "x_bf16 must be bfloat16");
    TORCH_CHECK(x_bf16.is_contiguous(), "x_bf16 must be contiguous");
    TORCH_CHECK(x_bf16.dim() == 2, "x_bf16 must be 2D [N, K]");

    int N = static_cast<int>(x_bf16.size(0));
    int K = static_cast<int>(x_bf16.size(1));

    auto opts_int8 = torch::TensorOptions().dtype(torch::kInt8).device(x_bf16.device());
    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(x_bf16.device());

    auto x_q = torch::empty({N, K}, opts_int8);
    auto scale_x = torch::empty({N}, opts_bf16);

    int threads = 256;
    if (K < threads) {
        // Round up to a warp multiple, cap at 256.
        threads = 32 * ((K + 31) / 32);
        if (threads < 32) threads = 32;
    }

    act_quant_bf16_kernel<<<N, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_bf16.data_ptr()),
        x_q.data_ptr<int8_t>(),
        reinterpret_cast<__nv_bfloat16*>(scale_x.data_ptr()),
        N, K
    );
    QWAN_CUDA_CHECK(cudaGetLastError());
    return {x_q, scale_x};
}


// Phase 26a-1: as act_quant_bf16 but also returns sum_x = scale_x * sum_k(x_int8)
// per row, fused into the same kernel pass. Used by the W8A8 asym GEMM
// epilogue's `psums += a_sum * zp_b * b_scale` correction term.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
act_quant_bf16_with_sum(torch::Tensor x_bf16) {
    TORCH_CHECK(x_bf16.is_cuda(), "x_bf16 must be on CUDA");
    TORCH_CHECK(x_bf16.dtype() == torch::kBFloat16, "x_bf16 must be bfloat16");
    TORCH_CHECK(x_bf16.is_contiguous(), "x_bf16 must be contiguous");
    TORCH_CHECK(x_bf16.dim() == 2, "x_bf16 must be 2D [N, K]");

    int N = static_cast<int>(x_bf16.size(0));
    int K = static_cast<int>(x_bf16.size(1));

    auto opts_int8 = torch::TensorOptions().dtype(torch::kInt8).device(x_bf16.device());
    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(x_bf16.device());

    auto x_q = torch::empty({N, K}, opts_int8);
    auto scale_x = torch::empty({N}, opts_bf16);
    auto sum_x = torch::empty({N}, opts_bf16);

    int threads = 256;
    if (K < threads) {
        threads = 32 * ((K + 31) / 32);
        if (threads < 32) threads = 32;
    }

    act_quant_bf16_with_sum_kernel<<<N, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_bf16.data_ptr()),
        x_q.data_ptr<int8_t>(),
        reinterpret_cast<__nv_bfloat16*>(scale_x.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(sum_x.data_ptr()),
        N, K
    );
    QWAN_CUDA_CHECK(cudaGetLastError());
    return {x_q, scale_x, sum_x};
}


// Phase 42 step 2: per-token per-group symmetric INT4 quant launcher
// (group=128 along K, no zp, no sum_x per plan G5; output in NATURAL
// layout — the Atom-permuted layout the W4A4 GEMM expects is applied by
// a separate downstream helper).
std::tuple<torch::Tensor, torch::Tensor> act_quant_bf16_group128(
    torch::Tensor x_bf16
) {
    TORCH_CHECK(x_bf16.is_cuda(), "x_bf16 must be on CUDA");
    TORCH_CHECK(x_bf16.dtype() == torch::kBFloat16, "x_bf16 must be bfloat16");
    TORCH_CHECK(x_bf16.is_contiguous(), "x_bf16 must be contiguous");
    TORCH_CHECK(x_bf16.dim() == 2, "x_bf16 must be 2D [N, K]");

    constexpr int GROUP = 128;
    int N = static_cast<int>(x_bf16.size(0));
    int K = static_cast<int>(x_bf16.size(1));
    TORCH_CHECK(K % GROUP == 0, "K must be multiple of 128");
    int K_groups = K / GROUP;

    auto opts_u8   = torch::TensorOptions().dtype(torch::kUInt8).device(x_bf16.device());
    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(x_bf16.device());
    auto x_int4    = torch::empty({N, K / 2}, opts_u8);
    auto scale_x   = torch::empty({N, K_groups}, opts_bf16);

    dim3 grid(N, K_groups);
    dim3 block(32);
    act_quant_bf16_group_sym_kernel<GROUP>
        <<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(x_bf16.data_ptr()),
            reinterpret_cast<uint8_t*>(x_int4.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(scale_x.data_ptr()),
            K
        );
    QWAN_CUDA_CHECK(cudaGetLastError());
    return {x_int4, scale_x};
}
