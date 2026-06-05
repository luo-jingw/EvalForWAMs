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


def _make_ref_dequant_matmul(
    x: torch.Tensor,
    fp: nn.Linear,
    weight_bits: int,
) -> torch.Tensor:
    """Reference: round-trip the weight through the same per-channel symmetric
    quantization the wrapper performs, then do the matmul in fp32 with the
    same per-token activation scale produced by act_quant_bf16. Returns bf16."""
    from qwan_extension import act_quant_bf16

    w_f32 = fp.weight.detach().to(torch.float32)
    qmax = 127 if weight_bits == 8 else 7
    clamp_lo = -qmax if weight_bits == 8 else -8
    clamp_hi = qmax
    row_max = w_f32.abs().amax(dim=1).clamp_min(1e-8)
    scale_w = (row_max / qmax).to(torch.bfloat16).to(torch.float32)
    w_int = torch.round(w_f32 / scale_w.unsqueeze(1)).clamp(clamp_lo, clamp_hi)

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

    cases = [
        ("W8A8 attn  3072->3072  bias=True",  QuantWanLinearW8A8, 3072, 3072,  2, 256, True,  1e-2),
        ("W8A8 attn  3072->3072  bias=False", QuantWanLinearW8A8, 3072, 3072,  2, 256, False, 1e-2),
        ("W8A8 ffn   3072->14336 bias=True",  QuantWanLinearW8A8, 3072, 14336, 1, 128, True,  1e-2),
        ("W4A8 attn  3072->3072  bias=True",  QuantWanLinearW4A8, 3072, 3072,  2, 256, True,  2e-2),
        ("W4A8 attn  3072->3072  bias=False", QuantWanLinearW4A8, 3072, 3072,  2, 256, False, 2e-2),
        ("W4A8 ffn   3072->14336 bias=True",  QuantWanLinearW4A8, 3072, 14336, 1, 128, True,  2e-2),
    ]
    results = [_check(name, cls, ci, co, b, s, hb, tol, device)
               for (name, cls, ci, co, b, s, hb, tol) in cases]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
