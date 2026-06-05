// Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
//
// W8A8 BF16-native GEMM. INT8 activation x INT8 weight -> BF16 output.
// Computation:
//   y[n, m] = sum_k (x[n, k] * w[m, k]) * scale_x[n] * scale_w[m] + bias[m]

#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "dispatch_utils.h"
#include "quant_gemm_common.cuh"


namespace {

constexpr int kTileM = 16;
constexpr int kTileN = 16;
constexpr int kTileK = 64;
constexpr int kThreadsPerBlock = kTileM * kTileN;  // 256

}  // anonymous namespace


torch::Tensor w8a8_gemm_bf16(
    torch::Tensor x_int8,            // [N, K]
    torch::Tensor scale_x_bf16,      // [N]
    torch::Tensor w_int8,            // [M, K]
    torch::Tensor scale_w_bf16,      // [M]
    c10::optional<torch::Tensor> bias_bf16   // [M] or None
) {
    TORCH_CHECK(x_int8.is_cuda() && w_int8.is_cuda(),
                "x_int8 and w_int8 must be on CUDA");
    TORCH_CHECK(x_int8.dtype() == torch::kInt8, "x_int8 must be int8");
    TORCH_CHECK(w_int8.dtype() == torch::kInt8, "w_int8 must be int8");
    TORCH_CHECK(scale_x_bf16.dtype() == torch::kBFloat16, "scale_x must be bf16");
    TORCH_CHECK(scale_w_bf16.dtype() == torch::kBFloat16, "scale_w must be bf16");
    TORCH_CHECK(x_int8.is_contiguous() && w_int8.is_contiguous(),
                "x_int8 and w_int8 must be contiguous");
    TORCH_CHECK(x_int8.dim() == 2 && w_int8.dim() == 2, "inputs must be 2D");

    int N = static_cast<int>(x_int8.size(0));
    int K = static_cast<int>(x_int8.size(1));
    int M = static_cast<int>(w_int8.size(0));
    TORCH_CHECK(w_int8.size(1) == K, "K dim mismatch: x.K=", K, " w.K=", w_int8.size(1));
    TORCH_CHECK(scale_x_bf16.size(0) == N, "scale_x.N mismatch");
    TORCH_CHECK(scale_w_bf16.size(0) == M, "scale_w.M mismatch");

    const __nv_bfloat16* bias_ptr = nullptr;
    if (bias_bf16.has_value()) {
        const auto& b = bias_bf16.value();
        TORCH_CHECK(b.is_cuda() && b.dtype() == torch::kBFloat16, "bias must be cuda bf16");
        TORCH_CHECK(b.is_contiguous() && b.dim() == 1 && b.size(0) == M, "bias shape mismatch");
        bias_ptr = reinterpret_cast<const __nv_bfloat16*>(b.data_ptr());
    }

    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(x_int8.device());
    auto y = torch::empty({N, M}, opts_bf16);

    dim3 grid((M + kTileM - 1) / kTileM, (N + kTileN - 1) / kTileN);
    dim3 block(kThreadsPerBlock);

    quant_gemm_bf16_kernel<8, kTileM, kTileN, kTileK><<<grid, block, 0,
        c10::cuda::getCurrentCUDAStream()>>>(
        x_int8.data_ptr<int8_t>(),
        w_int8.data_ptr<int8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(scale_x_bf16.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(scale_w_bf16.data_ptr()),
        bias_ptr,
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),
        N, M, K
    );
    QWAN_CUDA_CHECK(cudaGetLastError());
    return y;
}
