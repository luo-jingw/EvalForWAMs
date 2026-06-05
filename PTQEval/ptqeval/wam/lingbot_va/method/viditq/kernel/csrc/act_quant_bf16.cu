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
