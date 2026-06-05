// Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
#include <torch/extension.h>


// act_quant_bf16.cu
std::tuple<torch::Tensor, torch::Tensor> act_quant_bf16(torch::Tensor x_bf16);

// w8a8_gemm_bf16.cu
torch::Tensor w8a8_gemm_bf16(
    torch::Tensor x_int8,
    torch::Tensor scale_x_bf16,
    torch::Tensor w_int8,
    torch::Tensor scale_w_bf16,
    c10::optional<torch::Tensor> bias_bf16
);

// w4a8_gemm_bf16.cu
torch::Tensor w4a8_gemm_bf16(
    torch::Tensor x_int8,
    torch::Tensor scale_x_bf16,
    torch::Tensor w_int4_packed,
    torch::Tensor scale_w_bf16,
    c10::optional<torch::Tensor> bias_bf16
);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "qwan_extension: BF16-native int8/int4 GEMM kernels for LingBot-VA.";

    m.def("act_quant_bf16",
          &act_quant_bf16,
          "Per-token symmetric BF16 -> INT8 activation quant.\n"
          "Input: x_bf16 [N, K]. Output: (x_int8 [N, K], scale_x_bf16 [N]).",
          py::arg("x_bf16"));

    m.def("w8a8_gemm_bf16",
          &w8a8_gemm_bf16,
          "W8A8 INT8 GEMM with BF16 scales, returns BF16.\n"
          "y[N, M] = sum_k(x_int8[n, k] * w_int8[m, k])"
          " * scale_x_bf16[n] * scale_w_bf16[m] + bias_bf16[m].",
          py::arg("x_int8"),
          py::arg("scale_x_bf16"),
          py::arg("w_int8"),
          py::arg("scale_w_bf16"),
          py::arg("bias_bf16") = c10::nullopt);

    m.def("w4a8_gemm_bf16",
          &w4a8_gemm_bf16,
          "W4A8 packed-INT4 weight x INT8 activation GEMM, returns BF16.\n"
          "w_int4_packed is [M, K/2] int8 with low nibble=col 2c, high "
          "nibble=col 2c+1, both sign-extended.",
          py::arg("x_int8"),
          py::arg("scale_x_bf16"),
          py::arg("w_int4_packed"),
          py::arg("scale_w_bf16"),
          py::arg("bias_bf16") = c10::nullopt);
}
