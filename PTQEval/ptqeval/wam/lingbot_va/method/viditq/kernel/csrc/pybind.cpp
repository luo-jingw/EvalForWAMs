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

// w4a8/w4a8_gemm.cu (Phase 28: ViDiT-Q QServe W4A8 port + bf16).
// scale_weight is fp16/bf16 [N], szeros_weight is fp16/bf16 [N]
// (= scale_weight * zero_point_unsigned, precomputed at PTQ time).
// Packed weight: int8 [N, K/2] in QServe pre-permuted layout.
torch::Tensor w4a8_of16_nobias_weight_asym(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor scale_input,
    torch::Tensor scale_weight,
    torch::Tensor sum_input,
    torch::Tensor szeros_weight
);
torch::Tensor w4a8_obf16_nobias_weight_asym(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor scale_input,
    torch::Tensor scale_weight,
    torch::Tensor sum_input,
    torch::Tensor szeros_weight
);

// Phase 42 G4: W4A8 bias-fusion retrofit (has_bias=true epilogue).
torch::Tensor w4a8_of16_bias_weight_asym(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,           // fp16 [N]
    torch::Tensor scale_input,
    torch::Tensor scale_weight,
    torch::Tensor sum_input,
    torch::Tensor szeros_weight
);
torch::Tensor w4a8_obf16_bias_weight_asym(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,           // bf16 [N]
    torch::Tensor scale_input,
    torch::Tensor scale_weight,
    torch::Tensor sum_input,
    torch::Tensor szeros_weight
);

// w4a4/w4a4_gemm.cu (Phase 42 W4A4 mixed-precision GEMM family).
//   commit 1: port atom.cu, strip keeper (G2).
//   commit 2: OutT template + bf16 specialization (Phase 25 pattern).
//   commit 3: has_bias template + register-level FFMA in storeAccumulator
//             epilogue (G3 — bias-fusion). 4 launchers: {fp16,bf16} x
//             {with_bias, nobias}.
torch::Tensor w4a4_of16_nobias_weight_sym(
    torch::Tensor input,         // uint8 [M, K/2] packed int4 row-major
    torch::Tensor weight,        // uint8 [N, K/2] packed int4 row-major
    torch::Tensor scale_input,   // fp16 [M, K/128] Atom-permuted layout
    torch::Tensor scale_weight   // fp16 [K/128, N] Atom-permuted layout
);
torch::Tensor w4a4_obf16_nobias_weight_sym(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor scale_input,   // bf16 [M, K/128]
    torch::Tensor scale_weight   // bf16 [K/128, N]
);
torch::Tensor w4a4_of16_bias_weight_sym(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,          // fp16 [N]
    torch::Tensor scale_input,
    torch::Tensor scale_weight
);
torch::Tensor w4a4_obf16_bias_weight_sym(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,          // bf16 [N]
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

    m.def("w4a8_of16_nobias_weight_asym",
          &w4a8_of16_nobias_weight_asym,
          "ViDiT-Q/QServe W4A8 GEMM (fp16 output, asym weight, no bias).\n"
          "Computes y[m,n] = scale_input[m] * scale_weight[n] *\n"
          "  sum_k(input_int8[m,k] * weight_int4_unsigned[n,k])\n"
          "  - szeros_weight[n] * sum_input[m].\n"
          "Shapes: input [M,K] int8, weight [N,K/2] int8 (QServe-packed),\n"
          "  scale_input [M] fp16, scale_weight [N] fp16,\n"
          "  sum_input [M] fp16 (= scale_input * sum_k(input_int8)),\n"
          "  szeros_weight [N] fp16 (= scale_weight * zp_unsigned, precomputed).\n"
          "Output [M,N] fp16. No bias variant; caller adds bias post-GEMM.",
          py::arg("input"),
          py::arg("weight"),
          py::arg("scale_input"),
          py::arg("scale_weight"),
          py::arg("sum_input"),
          py::arg("szeros_weight"));

    m.def("w4a8_obf16_nobias_weight_asym",
          &w4a8_obf16_nobias_weight_asym,
          "W4A8 GEMM with bf16 output (Phase 28; mirrors w4a8_of16 in bf16).",
          py::arg("input"),
          py::arg("weight"),
          py::arg("scale_input"),
          py::arg("scale_weight"),
          py::arg("sum_input"),
          py::arg("szeros_weight"));

    m.def("w4a8_of16_bias_weight_asym",
          &w4a8_of16_bias_weight_asym,
          "Phase 42 G4: W4A8 GEMM with bias-fusion (fp16 out). Adds bias[n]\n"
          "inside the dequant epilogue (psums += Traits::to_float2(bias[col_wb/2]))\n"
          "after `psums = psums * wscale * ascale - w_sz * a_ssum`. bias\n"
          "must be fp16 [N].",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("scale_input"),
          py::arg("scale_weight"),
          py::arg("sum_input"),
          py::arg("szeros_weight"));

    m.def("w4a8_obf16_bias_weight_asym",
          &w4a8_obf16_bias_weight_asym,
          "Phase 42 G4: W4A8 GEMM with bias-fusion (bf16 out). bias must\n"
          "be bf16 [N]. Other args identical to fp16 variant.",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("scale_input"),
          py::arg("scale_weight"),
          py::arg("sum_input"),
          py::arg("szeros_weight"));

    m.def("w4a4_of16_nobias_weight_sym",
          &w4a4_of16_nobias_weight_sym,
          "Phase 42 W4A4 per-group symmetric INT4xINT4 GEMM (fp16 out).\n"
          "Ported from ViDiT-Q atom.cu with keeper code stripped (plan\n"
          "G2). Kernel is sym-only per plan G5; no zp, no sum_x, no bias.\n"
          "Computes y[m,n] = sum_g(scale_a[m,g] * scale_b[g,n] *\n"
          "  sum_{k in group g}(int4_a[m,k] * int4_b[n,k])).\n"
          "Shapes: input uint8 [M, K/2] packed nibbles row-major,\n"
          "  weight uint8 [N, K/2] packed nibbles (= [K,N] col-major),\n"
          "  scale_input fp16 [M, K/128] Atom-permuted layout,\n"
          "  scale_weight fp16 [K/128, N] Atom-permuted layout.\n"
          "Output [M, N] fp16. M/N/K must each be multiples of 128.",
          py::arg("input"),
          py::arg("weight"),
          py::arg("scale_input"),
          py::arg("scale_weight"));

    m.def("w4a4_obf16_nobias_weight_sym",
          &w4a4_obf16_nobias_weight_sym,
          "Phase 42 commit 2: bf16-output W4A4 GEMM. Mirrors\n"
          "w4a4_of16_nobias_weight_sym via the OutT template; scales\n"
          "must be bf16 (interpreted as __nv_bfloat162 in dequant).",
          py::arg("input"),
          py::arg("weight"),
          py::arg("scale_input"),
          py::arg("scale_weight"));

    m.def("w4a4_of16_bias_weight_sym",
          &w4a4_of16_bias_weight_sym,
          "Phase 42 commit 3 (G3): bias-fused W4A4 GEMM (fp16 out).\n"
          "Adds bias[n] inside the dequant epilogue via register-level\n"
          "FFMA (one add per output element); bias loaded warp-local\n"
          "at offset bi*BLOCK_N + wi*WARP_ROW_TILES*N. bias must be\n"
          "fp16 [N]. Other args identical to nobias variant.",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("scale_input"),
          py::arg("scale_weight"));

    m.def("w4a4_obf16_bias_weight_sym",
          &w4a4_obf16_bias_weight_sym,
          "Phase 42 commit 3 (G3): bias-fused W4A4 GEMM (bf16 out).\n"
          "bf16 specialization of w4a4_of16_bias_weight_sym; bias must\n"
          "be bf16 [N].",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("scale_input"),
          py::arg("scale_weight"));
}
