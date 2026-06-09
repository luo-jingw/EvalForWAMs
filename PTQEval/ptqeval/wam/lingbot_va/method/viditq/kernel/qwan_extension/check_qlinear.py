# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Phase 17 verify script.

For each of (W8A8, W4A8):
  - Build a torch.nn.Linear with bf16 weights on CUDA.
  - Construct the kernel-backed module via from_fp_linear.
  - Run a forward pass on a 3D random input [B, seq, in_features].
  - Check output shape, dtype, device, and finiteness.
  - Compare to a dequantize-then-matmul reference to confirm the wrapped
    forward path matches the lower-level kernel bench (Phase 15/16).

Pass condition: both wrappers produce bf16 outputs of the correct shape
with max abs error vs the reference below their respective tolerances.
"""
from __future__ import annotations

import sys

import torch
import torch.nn as nn

from qwan_extension.nn import QuantWanLinearW4A8, QuantWanLinearW8A8


def _make_ref_asym_w8a8(x: torch.Tensor, fp: nn.Linear) -> torch.Tensor:
    """W8A8 reference (Phase 26a-2 asym schema). Round-trips the weight
    through the same per-channel asymmetric quantization that
    QuantWanLinearW8A8.from_fp_linear performs, then runs the kernel
    epilogue formula in fp32 using the same act_quant_bf16_with_sum
    outputs (x_int8, scale_x, sum_x) the wrapper produces. Returns bf16.

    Formula matches w8a8_obf16_bias_weight_asym epilogue:
      y = scale_x * scale_w * x_int8 @ w_int.T
          + sum_x * zp_w * scale_w
          + bias
    where w_real = scale_w * (w_int + zp_w) is the asym dequant convention.
    """
    from qwan_extension import act_quant_bf16_with_sum

    w_f32 = fp.weight.detach().to(torch.float32)
    n_levels = 256
    x_max = w_f32.amax(dim=1).clamp_min(0.0)
    x_min = w_f32.amin(dim=1).clamp_max(0.0)
    delta = ((x_max - x_min) / (n_levels - 1)).clamp_min(1e-8)
    zero_point = torch.round(x_min / delta) + (n_levels / 2)
    scale_w_eff = delta.to(torch.bfloat16).to(torch.float32)
    w_int = (
        torch.round(w_f32 / scale_w_eff.unsqueeze(1))
        - zero_point.unsqueeze(1)
    ).clamp(-128, 127)

    x_bf16 = x.to(torch.bfloat16)
    x_2d = x_bf16.reshape(-1, fp.in_features).contiguous()
    x_int8, scale_x_bf16, sum_x_bf16 = act_quant_bf16_with_sum(x_2d)
    sx = scale_x_bf16.to(torch.float32)
    sa = sum_x_bf16.to(torch.float32)
    zp = zero_point.to(torch.float32)

    y_2d = (x_int8.to(torch.float32) @ w_int.T)
    y_2d = y_2d * sx.unsqueeze(1) * scale_w_eff.unsqueeze(0)
    y_2d = y_2d + sa.unsqueeze(1) * zp.unsqueeze(0) * scale_w_eff.unsqueeze(0)
    if fp.bias is not None:
        y_2d = y_2d + fp.bias.detach().to(torch.float32).unsqueeze(0)
    y = y_2d.reshape(*x.shape[:-1], fp.out_features)
    return y.to(torch.bfloat16)


def _make_ref_sym_w4a8(x: torch.Tensor, fp: nn.Linear) -> torch.Tensor:
    """W4A8 reference (scratch sym path; retained until Phase 26b)."""
    from qwan_extension import act_quant_bf16

    w_f32 = fp.weight.detach().to(torch.float32)
    qmax = 7
    row_max = w_f32.abs().amax(dim=1).clamp_min(1e-8)
    scale_w = (row_max / qmax).to(torch.bfloat16).to(torch.float32)
    w_int = torch.round(w_f32 / scale_w.unsqueeze(1)).clamp(-8, qmax)

    x_bf16 = x.to(torch.bfloat16)
    x_2d = x_bf16.reshape(-1, fp.in_features).contiguous()
    x_int8, scale_x_bf16 = act_quant_bf16(x_2d)
    sx = scale_x_bf16.to(torch.float32)

    y_2d = (x_int8.to(torch.float32) @ w_int.T)
    y_2d = y_2d * sx.unsqueeze(1) * scale_w.unsqueeze(0)
    if fp.bias is not None:
        y_2d = y_2d + fp.bias.detach().to(torch.float32).unsqueeze(0)
    y = y_2d.reshape(*x.shape[:-1], fp.out_features)
    return y.to(torch.bfloat16)


def _make_ref_dequant_matmul(x: torch.Tensor, fp: nn.Linear, weight_bits: int) -> torch.Tensor:
    if weight_bits == 8:
        return _make_ref_asym_w8a8(x, fp)
    elif weight_bits == 4:
        return _make_ref_sym_w4a8(x, fp)
    raise ValueError(f"unsupported weight_bits {weight_bits}")


def _check(name: str, wrapper_cls, in_features: int, out_features: int,
           batch: int, seq: int, has_bias: bool, tol: float,
           device: torch.device, seed: int = 0) -> bool:
    g = torch.Generator(device=device).manual_seed(seed)
    fp = nn.Linear(in_features, out_features, bias=has_bias).to(device).to(torch.bfloat16)
    with torch.no_grad():
        w_init = (torch.randn((out_features, in_features), device=device, generator=g) * 0.05).to(torch.bfloat16)
        fp.weight.copy_(w_init)
        if has_bias:
            b_init = (torch.randn((out_features,), device=device, generator=g) * 0.1).to(torch.bfloat16)
            fp.bias.copy_(b_init)

    mod = wrapper_cls.from_fp_linear(fp).to(device)

    x = (torch.randn((batch, seq, in_features), device=device, generator=g) * 0.5).to(torch.bfloat16)
    y = mod(x)

    shape_ok = tuple(y.shape) == (batch, seq, out_features)
    dtype_ok = y.dtype == torch.bfloat16
    device_ok = y.device == device
    finite_ok = torch.isfinite(y).all().item()

    y_ref = _make_ref_dequant_matmul(x, fp, wrapper_cls.WEIGHT_BITS)
    diff = (y.float() - y_ref.float()).abs()
    max_abs = diff.max().item()
    err_ok = max_abs < tol

    flag = "OK" if (shape_ok and dtype_ok and device_ok and finite_ok and err_ok) else "FAIL"
    print(f"{name:<40}  shape={tuple(y.shape)} dtype={y.dtype}  "
          f"max_abs={max_abs:.3e} (tol={tol})  {flag}")
    return shape_ok and dtype_ok and device_ok and finite_ok and err_ok


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA unavailable.", file=sys.stderr)
        return 2
    device = torch.device("cuda:0")

    # W8A8 tol bumped to 5e-2 (was 1e-2 for old sym scratch path):
    # asym epilogue + bf16 final cast + fast-math fp32 noise + small output
    # magnitudes (~1-10 for these random N(0, 0.05) weights) push the
    # per-element abs error into the 0.01-0.05 band. Same physics as the
    # max_rel < 1e-2 tol used in Phase 25 bench_w8a8_bf16 (this is an
    # abs-error variant). W4A8 stays at the scratch-sym 2e-2.
    # Phase 26a-2 covers an in=14336 down_proj-shaped case so any future
    # regression in the wrapper's handling of LingBot-VA shapes is caught.
    cases = [
        ("W8A8 attn  3072->3072   bias=True",  QuantWanLinearW8A8, 3072,  3072, 2, 256, True,  5e-2),
        ("W8A8 attn  3072->3072   bias=False", QuantWanLinearW8A8, 3072,  3072, 2, 256, False, 5e-2),
        ("W8A8 ffn   3072->14336  bias=True",  QuantWanLinearW8A8, 3072, 14336, 1, 128, True,  5e-2),
        ("W8A8 ffn   14336->3072  bias=True",  QuantWanLinearW8A8, 14336, 3072, 1, 128, True,  5e-2),
        ("W4A8 attn  3072->3072   bias=True",  QuantWanLinearW4A8, 3072,  3072, 2, 256, True,  2e-2),
        ("W4A8 attn  3072->3072   bias=False", QuantWanLinearW4A8, 3072,  3072, 2, 256, False, 2e-2),
        ("W4A8 ffn   3072->14336  bias=True",  QuantWanLinearW4A8, 3072, 14336, 1, 128, True,  2e-2),
    ]
    results = [_check(name, cls, ci, co, b, s, hb, tol, device)
               for (name, cls, ci, co, b, s, hb, tol) in cases]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
