# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""W8A8 wrapper numerical check.

Build a torch.nn.Linear with bf16 weights on CUDA, construct
QuantWanLinearW8A8 via from_fp_linear, run a forward pass on a 3D random
input [B, seq, in_features], and compare to a dequantize-then-matmul
reference. Pass condition: bf16 outputs of correct shape with max abs
error below the tolerance.

Phase 26b: W4A8 cases removed alongside the scratch W4A8 wrapper. Phase
28 reinstates W4A8 cases against the QServe port.
"""
from __future__ import annotations

import sys

import torch
import torch.nn as nn

from qwan_extension.nn import QuantWanLinearW8A8


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


def _check(name: str, in_features: int, out_features: int,
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

    mod = QuantWanLinearW8A8.from_fp_linear(fp).to(device)

    x = (torch.randn((batch, seq, in_features), device=device, generator=g) * 0.5).to(torch.bfloat16)
    y = mod(x)

    shape_ok = tuple(y.shape) == (batch, seq, out_features)
    dtype_ok = y.dtype == torch.bfloat16
    device_ok = y.device == device
    finite_ok = torch.isfinite(y).all().item()

    y_ref = _make_ref_asym_w8a8(x, fp)
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

    # tol = 5e-2: asym epilogue + bf16 final cast + fast-math fp32 noise +
    # small output magnitudes (~1-10 for N(0,0.05) weights). Same physics
    # as Phase 25 bench_w8a8_bf16's max_rel < 1e-2 (abs-error variant).
    # in=14336 case covers LingBot-VA down_proj shape.
    cases = [
        ("W8A8 attn  3072->3072   bias=True",  3072,  3072, 2, 256, True,  5e-2),
        ("W8A8 attn  3072->3072   bias=False", 3072,  3072, 2, 256, False, 5e-2),
        ("W8A8 ffn   3072->14336  bias=True",  3072, 14336, 1, 128, True,  5e-2),
        ("W8A8 ffn   14336->3072  bias=True",  14336, 3072, 1, 128, True,  5e-2),
    ]
    results = [_check(name, ci, co, b, s, hb, tol, device)
               for (name, ci, co, b, s, hb, tol) in cases]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
