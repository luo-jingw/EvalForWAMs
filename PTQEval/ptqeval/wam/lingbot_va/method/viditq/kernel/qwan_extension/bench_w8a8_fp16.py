# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Phase 24b numerical bench for the verbatim-ported ViDiT-Q W8A8 fp16 kernel.

Mirrors ViDiT-Q kernels/bench/bench_gemm.py:
  - input/weight: int8 in [-80, 80]
  - scale_input:  0.01 * U(0,1) + 0.005   (per-row, fp16)
  - scale_weight: 0.10 * U(0,1) + 0.10    (per-col, fp16)
  - bias:         U(0,1) * 200            (per-col, fp16)
  - zp_weight:    int16 in [-10, 10]
  - input_sum:    (scale_input.view(-1,1) * input_fp32).sum(dim=1).to(fp16)
                  (per-row real-domain row-sum, fp16-truncated; the kernel
                   consumes this fp16 value directly per epilogue convention)

Reference (verbatim from ViDiT-Q bench_gemm.py:26-29):
    output_gt = (F.linear(input_fp32, weight_fp32) * scale_input * scale_weight
                 + input_sum * zp_weight * scale_weight
                 + bias).to(fp16)
ViDiT-Q evaluates this in fp32 on GPU; we follow them so our numerical
verification is a literal reproduction of theirs, not an alternative.

Three WAN-block shapes (M, N, K) cover the dominant Linear shapes in the
LingBot-VA transformer:
    (256,  3072,  3072)   self-attn qkv / out_proj
    (256,  3072, 14336)   ffn down_proj
    (256, 14336,  3072)   ffn up_proj

Acceptance: max relative error < 5e-3 across all shapes. ViDiT-Q's own
bench only prints abs error (no assert); we hard-assert relative for CI.
"""
import torch
import torch.nn.functional as F

from qwan_extension._C import w8a8_of16_bias_weight_asym


SHAPES = [
    (256,  3072,  3072),
    (256,  3072, 14336),
    (256, 14336,  3072),
]
REL_TOL = 5e-3


def make_inputs(M: int, N: int, K: int, seed: int = 0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    input_int  = torch.randint(-80, 80, (M, K), dtype=torch.int8, device="cuda", generator=g)
    weight_int = torch.randint(-80, 80, (N, K), dtype=torch.int8, device="cuda", generator=g)
    scale_input  = 0.01 * torch.rand(M, dtype=torch.float16, device="cuda", generator=g) + 0.005
    scale_weight = 0.10 * torch.rand(N, dtype=torch.float16, device="cuda", generator=g) + 0.10
    bias         = torch.rand(N,         dtype=torch.float16, device="cuda", generator=g) * 200
    zp_weight    = torch.randint(-10, 10, (N,), dtype=torch.int16, device="cuda", generator=g)
    # input_sum: per-row real-domain row-sum, fp16-truncated. Kernel epilogue
    # consumes this without any scale_input factor (see w8a8_gemm.cu L431-434).
    input_sum = (scale_input.view(-1, 1).to(torch.float32)
                 * input_int.to(torch.float32)).sum(dim=1).to(torch.float16)
    return input_int, weight_int, bias, scale_input, scale_weight, input_sum, zp_weight


def reference_viditq(input_int, weight_int, bias,
                     scale_input, scale_weight, input_sum, zp_weight) -> torch.Tensor:
    # Verbatim from ViDiT-Q bench_gemm.py:26-29. GPU fp32 throughout.
    input_fp32  = input_int.to(torch.float32)
    weight_fp32 = weight_int.to(torch.float32)
    y_fp32 = (F.linear(input_fp32, weight_fp32)
              * scale_input.view(-1, 1).to(torch.float32)
              * scale_weight.view(1, -1).to(torch.float32)
              + input_sum.view(-1, 1).to(torch.float32)
              * zp_weight.to(torch.float32).view(1, -1)
              * scale_weight.view(1, -1).to(torch.float32)
              + bias.to(torch.float32))
    return y_fp32.to(torch.float16)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required")

    print(f"{'shape (M, N, K)':<26}{'max_abs_err':>14}{'max_rel_err':>14}{'pass':>8}")
    print("-" * 62)

    all_pass = True
    for M, N, K in SHAPES:
        x, w, b, sx, sw, su, zp = make_inputs(M, N, K, seed=0)
        y_kernel = w8a8_of16_bias_weight_asym(x, w, b, sx, sw, su, zp)
        y_ref    = reference_viditq(x, w, b, sx, sw, su, zp)

        y_k_fp32 = y_kernel.to(torch.float32)
        y_r_fp32 = y_ref.to(torch.float32)
        abs_diff = (y_k_fp32 - y_r_fp32).abs()
        max_abs  = abs_diff.max().item()
        # Relative error guarded against tiny magnitudes (avoid div-by-zero
        # at outputs near zero where any nonzero noise yields huge rel err).
        denom    = y_r_fp32.abs().clamp_min(1.0)
        max_rel  = (abs_diff / denom).max().item()
        ok = max_rel < REL_TOL
        all_pass = all_pass and ok
        print(f"({M:>4}, {N:>5}, {K:>5}){max_abs:>16.4e}{max_rel:>14.4e}{('OK' if ok else 'FAIL'):>8}")

    if not all_pass:
        raise AssertionError(f"at least one shape exceeded max_rel_err < {REL_TOL}")
    print(f"\nbench_w8a8_fp16 OK (all shapes within max_rel_err < {REL_TOL})")


if __name__ == "__main__":
    main()
