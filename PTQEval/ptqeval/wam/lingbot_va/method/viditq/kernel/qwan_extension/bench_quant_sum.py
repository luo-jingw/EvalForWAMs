# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Phase 24c numerical bench for the verbatim-ported ViDiT-Q QuantKernel
with bf16 type extension.

Covers both dtype paths:
  quant_sum       -> QuantKernel<__half,        float2/float4, kPostQuant>
  quant_sum_bf16  -> QuantKernel<__nv_bfloat16, float2/float4, kPostQuant>

For each shape, the kernel produces (x_int8, scale, sum_output). The
Python reference computes:
    scale_ref[m]   = max_k(|x[m,k]|) / 127
    x_int8_ref[m,k]= round(x[m,k] / scale_ref[m]).clamp(-127, 127)
    sum_ref[m]     = sum_k(scale_ref[m] * x_int8_ref[m,k])   (in fp32)
                     cast to target dtype
All three are compared. The primary acceptance is on sum_output;
tolerance depends on dtype:
  fp16: max_rel < 1e-2
  bf16: max_rel < 1e-1
The looser bf16 bound is the joint floor of (i) bf16's 7-bit mantissa
(ULP ~ X / 128 at magnitude X) and (ii) --use_fast_math approximate
fp32 division (kept verbatim from ViDiT-Q upstream) which causes ~1
fp32 ULP divergence in tmp_scale = 127/amax. The two effects combine
to flip up to ~8 borderline int8 values per row in bf16; sum_rel
worst case ~ 8 / 128 = 6.25%, set bound to 1e-1 (~2x margin).

Kernel hidden_size constraint: <= 8192, %128 (or %256 if >4096). Shapes
chosen to exercise both <=4096 (float2 load) and >4096 (float4 load)
branches.
"""
import torch

from qwan_extension._C import quant_sum, quant_sum_bf16


# (num_tokens, hidden_size). hidden_size must satisfy the kernel asserts.
SHAPES = [
    ( 256,  3072),   # WAN model dim (float2 path)
    ( 256,  4096),   # boundary (float2 path)
    (  64,  8192),   # >4096 -> float4 path
]
SUM_REL_TOL = {
    torch.float16:  1e-2,
    torch.bfloat16: 1e-1,
}


def make_input(num_tokens: int, hidden_size: int, dtype: torch.dtype, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    # Per-row max magnitude varies across tokens; rescale to produce a
    # realistic post-norm activation distribution (mean ~1, range ~0-5).
    x = torch.randn((num_tokens, hidden_size), dtype=torch.float32,
                    device="cuda", generator=g)
    return x.to(dtype)


def reference(x: torch.Tensor, target_dtype: torch.dtype):
    # Bit-exact mirror of QuantKernel arithmetic to isolate any kernel
    # bug from incidental fp32-ordering noise. Key insight: the kernel
    # uses `tmp_scale = 127.0f / s_amax` then `x_int8 = float_to_int8_rn(
    # (float)x_val * tmp_scale)`. We must match the multiplication path
    # exactly; using `x / (amax/127)` is mathematically equivalent but
    # accumulates 1 fp32 ULP differently and flips borderline int8
    # values, propagating to sum as visible bench failure.
    x_fp32   = x.to(torch.float32)
    amax     = x_fp32.abs().amax(dim=1)              # [num_tokens], same as kernel s_amax
    tmp_scale = 127.0 / amax                          # [num_tokens], same as kernel tmp_scale
    # stored scale (kernel writes from_float_rn(amax/127) into scale[bidx]).
    scale_dtype = (amax / 127.0).to(target_dtype)
    # int8 quant: multiply by tmp_scale (matches kernel order).
    x_int8 = (x_fp32 * tmp_scale.view(-1, 1)).round().clamp(-127, 127).to(torch.int8)
    # sum_output: kernel computes (int32) sum_int, then
    #   sum_output = from_float_rn(__int2float_rn(sum_int) / tmp_scale)
    sum_int  = x_int8.to(torch.int32).sum(dim=1)
    sum_fp32 = sum_int.to(torch.float32) / tmp_scale
    sum_dtype = sum_fp32.to(target_dtype)
    return x_int8, scale_dtype, sum_dtype


def run_one(num_tokens: int, hidden_size: int, dtype: torch.dtype, name: str):
    x = make_input(num_tokens, hidden_size, dtype, seed=0)

    sum_out = torch.empty((num_tokens,), dtype=dtype, device="cuda")
    scaling = torch.empty((num_tokens,), dtype=dtype, device="cuda")
    if dtype == torch.float16:
        x_int8_k = quant_sum(x, sum_out, scaling)
    elif dtype == torch.bfloat16:
        x_int8_k = quant_sum_bf16(x, sum_out, scaling)
    else:
        raise ValueError(f"unsupported dtype {dtype}")

    x_int8_r, scaling_r, sum_r = reference(x, dtype)

    int8_diff = (x_int8_k.to(torch.int32) - x_int8_r.to(torch.int32)).abs().max().item()
    scale_abs = (scaling.to(torch.float32) - scaling_r.to(torch.float32)).abs().max().item()
    sum_abs   = (sum_out.to(torch.float32) - sum_r.to(torch.float32)).abs()
    denom     = sum_r.to(torch.float32).abs().clamp_min(1.0)
    sum_rel   = (sum_abs / denom).max().item()
    sum_abs_v = sum_abs.max().item()

    tol = SUM_REL_TOL[dtype]
    ok = sum_rel < tol
    print(f"{name:<6} ({num_tokens:>4}, {hidden_size:>5})"
          f"  int8_max_diff={int8_diff:>3d}"
          f"  scale_abs={scale_abs:.3e}"
          f"  sum_abs={sum_abs_v:.3e}"
          f"  sum_rel={sum_rel:.3e}"
          f"  tol={tol:.0e}"
          f"  {'OK' if ok else 'FAIL'}")
    return ok


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required")

    fp16_tol = SUM_REL_TOL[torch.float16]
    bf16_tol = SUM_REL_TOL[torch.bfloat16]
    print(f"Phase 24c bench: QuantKernel fp16/bf16 "
          f"(fp16 tol={fp16_tol:.0e}, bf16 tol={bf16_tol:.0e})")
    print("-" * 104)

    all_pass = True
    for nt, hs in SHAPES:
        all_pass &= run_one(nt, hs, torch.float16,  "fp16")
        all_pass &= run_one(nt, hs, torch.bfloat16, "bf16")

    if not all_pass:
        raise AssertionError("at least one (shape, dtype) exceeded its dtype-specific tol")
    print(f"\nbench_quant_sum OK (fp16 < {fp16_tol:.0e}, bf16 < {bf16_tol:.0e})")


if __name__ == "__main__":
    main()
