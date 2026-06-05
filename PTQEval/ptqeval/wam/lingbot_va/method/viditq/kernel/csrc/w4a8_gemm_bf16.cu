// Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
//
// W4A8 BF16-native GEMM. INT8 activation x packed INT4 weight -> BF16 output.
// Weight packing: w_int4_packed has shape [M, K/2] (int8 storage). Each
// byte holds two signed int4 values:
//   low nibble  (bits 0..3, sign-extended) -> column index 2*c+0
//   high nibble (bits 4..7, sign-extended) -> column index 2*c+1
//
// Activation quantization (per-token symmetric, bf16 scale) is shared with
// the W8A8 path (qwan_extension.act_quant_bf16).

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


torch::Tensor w4a8_gemm_bf16(
    torch::Tensor x_int8,            // [N, K]
    torch::Tensor scale_x_bf16,      // [N]
    torch::Tensor w_int4_packed,     // [M, K/2] int8 (two nibbles per byte)
    torch::Tensor scale_w_bf16,      // [M]
    c10::optional<torch::Tensor> bias_bf16   // [M] or None
) {
    TORCH_CHECK(x_int8.is_cuda() && w_int4_packed.is_cuda(),
                "x_int8 and w_int4_packed must be on CUDA");
    TORCH_CHECK(x_int8.dtype() == torch::kInt8, "x_int8 must be int8");
    TORCH_CHECK(w_int4_packed.dtype() == torch::kInt8,
                "w_int4_packed must be int8 (each byte holds two nibbles)");
    TORCH_CHECK(scale_x_bf16.dtype() == torch::kBFloat16, "scale_x must be bf16");
    TORCH_CHECK(scale_w_bf16.dtype() == torch::kBFloat16, "scale_w must be bf16");
    TORCH_CHECK(x_int8.is_contiguous() && w_int4_packed.is_contiguous(),
                "x_int8 and w_int4_packed must be contiguous");
    TORCH_CHECK(x_int8.dim() == 2 && w_int4_packed.dim() == 2, "inputs must be 2D");

    int N = static_cast<int>(x_int8.size(0));
    int K = static_cast<int>(x_int8.size(1));
    int M = static_cast<int>(w_int4_packed.size(0));
    TORCH_CHECK(K % 2 == 0, "K must be even for W4 packing, got K=", K);
    TORCH_CHECK(w_int4_packed.size(1) == K / 2,
                "packed K dim mismatch: x.K=", K, " w_packed.K=", w_int4_packed.size(1));
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

    quant_gemm_bf16_kernel<4, kTileM, kTileN, kTileK><<<grid, block, 0,
        c10::cuda::getCurrentCUDAStream()>>>(
        x_int8.data_ptr<int8_t>(),
        w_int4_packed.data_ptr<int8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(scale_x_bf16.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(scale_w_bf16.data_ptr()),
        bias_ptr,
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),
        N, M, K
    );
    QWAN_CUDA_CHECK(cudaGetLastError());
    return y;
}
