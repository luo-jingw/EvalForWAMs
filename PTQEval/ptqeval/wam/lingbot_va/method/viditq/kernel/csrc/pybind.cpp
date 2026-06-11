// Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
#include <torch/extension.h>


// act_quant_bf16.cu
std::tuple<torch::Tensor, torch::Tensor> act_quant_bf16(torch::Tensor x_bf16);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    act_quant_bf16_with_sum(torch::Tensor x_bf16);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    act_quant_bf16_with_sum_static(torch::Tensor x_bf16, torch::Tensor scale_in);

// toy_mma_int8.cu
void toy_mma_int8_gemm(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c
);

// w8a8/w8a8_gemm.cu (verbatim port of ViDiT-Q w8a8_gemm_cuda.cu)
torch::Tensor w8a8_of16_bias_weight_asym(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor scale_input,
    torch::Tensor scale_weight,
    torch::Tensor sum_input,
    torch::Tensor zp_weight
);

// Phase 25 bf16 instantiations of the same kernel via OutT template.
torch::Tensor w8a8_obf16_bias_weight_asym(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor scale_input,
    torch::Tensor scale_weight,
    torch::Tensor sum_input,
    torch::Tensor zp_weight
);
torch::Tensor w8a8_obf16_bias_weight_sym(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor scale_input,
    torch::Tensor scale_weight
);
torch::Tensor w8a8_obf16_nobias_weight_asym(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor scale_input,
    torch::Tensor scale_weight,
    torch::Tensor sum_input,
    torch::Tensor zp_weight
);
torch::Tensor w8a8_obf16_nobias_weight_sym(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor scale_input,
    torch::Tensor scale_weight
);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "qwan_extension: BF16-native int8 GEMM kernels for LingBot-VA.";

    m.def("act_quant_bf16",
          &act_quant_bf16,
          "Per-token symmetric BF16 -> INT8 activation quant.\n"
          "Input: x_bf16 [N, K]. Output: (x_int8 [N, K], scale_x_bf16 [N]).",
          py::arg("x_bf16"));

    m.def("act_quant_bf16_with_sum",
          &act_quant_bf16_with_sum,
          "Per-token sym quant + fused post-quant sum_x (Phase 26a-1).\n"
          "Input:  x_bf16 [N, K].\n"
          "Output: (x_int8 [N, K], scale_x_bf16 [N], sum_x_bf16 [N]).\n"
          "sum_x[n] = scale_x[n] * sum_k(x_int8[n, k]) cast bf16.\n"
          "Same algorithm as ViDiT-Q QuantKernel<bf16, _, kPostQuant>;\n"
          "grid-stride structure handles arbitrary K (no <=8192 cap).",
          py::arg("x_bf16"));

    m.def("act_quant_bf16_with_sum_static",
          &act_quant_bf16_with_sum_static,
          "Phase 33 static variant. Same as act_quant_bf16_with_sum but\n"
          "skips amax reduction; reads a pre-computed scalar scale from\n"
          "scale_in[0] (bf16). scale_x[N] is still filled with the scalar\n"
          "so downstream W8A8 GEMM consumes the same per-row scale[M]\n"
          "interface as the dynamic variant.\n"
          "Input:  x_bf16 [N, K], scale_in [1] bf16.\n"
          "Output: (x_int8 [N, K], scale_x_bf16 [N], sum_x_bf16 [N]).",
          py::arg("x_bf16"), py::arg("scale_in"));

    m.def("toy_mma_int8_gemm",
          &toy_mma_int8_gemm,
          "Phase 24a toy: single-CTA m16n8k32 s8s8s32 MMA (correctness only).\n"
          "a: int8 [16, 32], b: int8 [8, 32], c: int32 [16, 8]. Computes "
          "c = a @ b.T.",
          py::arg("a"),
          py::arg("b"),
          py::arg("c"));

    m.def("w8a8_of16_bias_weight_asym",
          &w8a8_of16_bias_weight_asym,
          "ViDiT-Q W8A8 GEMM (fp16 output, asymmetric weight, with bias).\n"
          "Computes y[m,n] = scale_input[m] * scale_weight[n] *\n"
          "  (sum_k(input_int8[m,k] * weight_int8[n,k])"
          " - zp_weight[n] * sum_k(input_int8[m,k])) + bias[n].\n"
          "Shapes: input [M,K] int8, weight [N,K] int8, bias [N] fp16,\n"
          "  scale_input [M] fp16, scale_weight [N] fp16,\n"
          "  sum_input [M] fp16 (= sum_k(input_int8[m,k]) cast to fp16),\n"
          "  zp_weight [N] int16. Output [M,N] fp16.",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("scale_input"),
          py::arg("scale_weight"),
          py::arg("sum_input"),
          py::arg("zp_weight"));

    // Phase 25: bf16-output W8A8 launchers. Same kernel, OutT=__nv_bfloat16.
    m.def("w8a8_obf16_bias_weight_asym",
          &w8a8_obf16_bias_weight_asym,
          "W8A8 GEMM with bf16 output, asym weight, with bias.\n"
          "Mirrors w8a8_of16_bias_weight_asym in bf16 domain.",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("scale_input"),
          py::arg("scale_weight"),
          py::arg("sum_input"),
          py::arg("zp_weight"));

    m.def("w8a8_obf16_bias_weight_sym",
          &w8a8_obf16_bias_weight_sym,
          "W8A8 GEMM with bf16 output, sym weight, with bias.",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("scale_input"),
          py::arg("scale_weight"));

    m.def("w8a8_obf16_nobias_weight_asym",
          &w8a8_obf16_nobias_weight_asym,
          "W8A8 GEMM with bf16 output, asym weight, no bias.",
          py::arg("input"),
          py::arg("weight"),
          py::arg("scale_input"),
          py::arg("scale_weight"),
          py::arg("sum_input"),
          py::arg("zp_weight"));

    m.def("w8a8_obf16_nobias_weight_sym",
          &w8a8_obf16_nobias_weight_sym,
          "W8A8 GEMM with bf16 output, sym weight, no bias.",
          py::arg("input"),
          py::arg("weight"),
          py::arg("scale_input"),
          py::arg("scale_weight"));
}
