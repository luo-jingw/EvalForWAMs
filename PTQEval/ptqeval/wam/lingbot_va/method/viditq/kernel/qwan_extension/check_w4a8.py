# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Phase 28 W4A8 numerical correctness check (bf16 + fp16 launchers).

Generates random unsigned int4 weights + per-channel scale/zp, packs
via the omniserve QServe layout (ptq._pack_int4_qserve), runs the
kernel, and compares against a fp32 PyTorch reference whose dequant
formula mirrors the kernel epilogue exactly:

    W_real    = scale_w * (W_int_unsigned - zp_unsigned)
    y_fp32    = (input_int.to(fp32) @ W_real.T) * scale_input
              = (input_int @ W_int).fp32 * scale_w * scale_x
                - sum_k(input_int * scale_x) * scale_w * zp

Test shape coverage (W4A8 kernel has 3 CTA_M paths -- 128 / 64 / 32):
    M=256 -> CTA_M=128 path
    M=128 -> CTA_M=64  path  (num_out_feats in [128, 256])
    M= 64 -> CTA_M=32  path  (num_out_feats < 128)
"""
import torch
import torch.nn.functional as F

from qwan_extension._C import (
    w4a8_obf16_nobias_weight_asym,
    w4a8_of16_nobias_weight_asym,
)

# Phase 28 ptq pack helper (top-level import would tug in heavy WAM deps; do
# it lazily inside main()).


SHAPES = [
    # (M, N, K)
    (256,   128,  128),   # small smoke
    (256,  3072, 3072),   # LingBot-VA self-attn projection size
    (128,  3072, 3072),   # CTA_M=64 path
    ( 64,  3072, 3072),   # CTA_M=32 path
    (256, 14336, 3072),   # ffn up_proj
    (256,  3072,14336),   # ffn down_proj (wide K)
]
# Tolerance: kernel introduces error at most a few ULPs of the
# OUTPUT DTYPE relative to the dynamic range (|y|.abs().max()).
# fp16 ULP at value V = V * 2^-10, so a kernel that only differs
# from the fp32 reference by the inevitable fp32->fp16 cast noise
# will have max_abs < 4 * 2^-10 * |y|.max() = ~4e-3 * dyn_range.
# bf16 ULP is 8x larger (mantissa 7 vs 10), so we allow 4 * 2^-7
# = ~3e-2 * dyn_range there.
ULP_BUDGET = 4  # tolerate up to 4 cast ULPs per output


def make_inputs(M: int, N: int, K: int, seed: int = 0, dtype: torch.dtype = torch.bfloat16):
    g = torch.Generator(device="cuda").manual_seed(seed)
    # Activation: int8 [M, K] with a per-token scale [M] bf16/fp16.
    input_int   = torch.randint(-80, 80, (M, K), dtype=torch.int8, device="cuda", generator=g)
    scale_input = 0.01 * torch.rand(M, dtype=dtype, device="cuda", generator=g) + 0.005
    # Weight: unsigned int4 [N, K] in [0, 15] with per-channel scale + unsigned zp.
    weight_uns  = torch.randint(0, 16, (N, K), dtype=torch.uint8, device="cuda", generator=g)
    scale_w     = 0.10 * torch.rand(N, dtype=dtype, device="cuda", generator=g) + 0.10
    zp_uns      = torch.randint(0, 16, (N,), dtype=torch.uint8, device="cuda", generator=g)
    # Precomputed scaled-zeros buffer the kernel epilogue actually reads.
    szeros = (
        scale_w.to(torch.float32) * zp_uns.to(torch.float32)
    ).to(dtype).contiguous()
    # sum_x = scale_x * sum_k(int8); cast to target dtype.
    sum_input = (
        scale_input.view(-1, 1).to(torch.float32)
        * input_int.to(torch.float32)
    ).sum(dim=1).to(dtype).contiguous()
    return input_int, weight_uns, scale_input, scale_w, szeros, sum_input, zp_uns


def reference_qserve(input_int, weight_uns, scale_input, scale_w, zp_uns,
                     out_dtype: torch.dtype) -> torch.Tensor:
    """fp32 reference implementing the QServe W4A8 dequant convention
    `w_real = scale * (w_int_unsigned - zp_unsigned)` then matmul +
    per-token activation scale."""
    input_fp32  = input_int.to(torch.float32)
    w_real_fp32 = (
        scale_w.view(-1, 1).to(torch.float32)
        * (weight_uns.to(torch.float32) - zp_uns.view(-1, 1).to(torch.float32))
    )
    y_fp32 = F.linear(input_fp32, w_real_fp32) * scale_input.view(-1, 1).to(torch.float32)
    return y_fp32.to(out_dtype)


def numerical_check(launcher, dtype, label: str) -> bool:
    """Reference must consume the SAME bf16/fp16 helper tensors the kernel
    sees (scale_input, scale_weight, szeros, sum_input). Using fp64-exact
    inputs would unfairly count the inherent precision loss of those
    intermediate buffers as 'kernel error'. The buffers are themselves the
    output of upstream fp16 ops (act_quant_bf16_with_sum at runtime; PTQ-
    time scale_w/szeros cast), so the kernel can at best reproduce
    `psums * sw * sa - szeros * sum_x` evaluated in fp32 with those exact
    fp16/bf16 operands, then cast to the output dtype. We replicate that
    formula here and tolerate ULP_BUDGET * 1 ULP per cell.
    """
    from ptqeval.wam.lingbot_va.method.viditq.ptq import _pack_int4_qserve

    ulp_unit = 2 ** -10 if dtype == torch.float16 else 2 ** -7
    print(f"\n{label}:")
    print(f"{'shape (M, N, K)':<26}{'max_abs':>12}{'dyn_range':>12}{'budget':>12}{'pass':>8}")
    print("-" * 70)
    all_pass = True
    for M, N, K in SHAPES:
        x, w_uns, sx, sw, szeros, sum_x, zp = make_inputs(M, N, K, seed=0, dtype=dtype)
        w_packed = _pack_int4_qserve(w_uns)
        y_kernel = launcher(x, w_packed, sx, sw, sum_x, szeros)
        # int32-exact dot product computed in fp64 (int values < 2^24 are
        # exact in fp64; this matches the kernel's int32 accumulator after
        # cast to fp32).
        psums = (x.to(torch.float64) @ w_uns.to(torch.float64).T)  # [M, N]
        # Epilogue replicated in fp32 with the SAME bf16/fp16 helper tensors
        # the kernel sees, then cast to output dtype.
        y_ref_fp32 = (
            psums.to(torch.float32)
            * sx.view(-1, 1).to(torch.float32)
            * sw.view(1, -1).to(torch.float32)
            - szeros.view(1, -1).to(torch.float32)
            * sum_x.view(-1, 1).to(torch.float32)
        )
        y_ref = y_ref_fp32.to(dtype)

        diff = (y_kernel.to(torch.float64) - y_ref.to(torch.float64)).abs()
        max_abs = diff.max().item()
        dyn_per_cell = y_ref.to(torch.float64).abs() * ulp_unit
        budget = (dyn_per_cell.clamp_min(ulp_unit) * ULP_BUDGET)
        over = diff > budget
        n_over = int(over.sum().item())
        ok = n_over == 0
        all_pass = all_pass and ok
        y_max = y_ref.to(torch.float32).abs().max().item()
        per_cell_ulp_at_max = y_max * ulp_unit
        print(f"({M:>4}, {N:>5}, {K:>5}){max_abs:>14.4e}{y_max:>12.2f}"
              f"{ULP_BUDGET * per_cell_ulp_at_max:>14.4e}{('OK' if ok else f'FAIL({n_over})'):>10}")
    return all_pass


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available, skipping check.")
        return 1
    bf16_ok = numerical_check(w4a8_obf16_nobias_weight_asym, torch.bfloat16,
                              "bf16 launcher (w4a8_obf16_nobias_weight_asym)")
    fp16_ok = numerical_check(w4a8_of16_nobias_weight_asym, torch.float16,
                              "fp16 launcher (w4a8_of16_nobias_weight_asym)")
    print()
    if bf16_ok and fp16_ok:
        print("ALL PASS")
        return 0
    print("FAIL")
    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
