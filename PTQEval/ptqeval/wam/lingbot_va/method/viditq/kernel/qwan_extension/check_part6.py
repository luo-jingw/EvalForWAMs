# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Part VI observational check: SmoothQuant + QuaRoT + ViDiT combined plumbing.

Per principle.txt L12: this script emits raw metrics only — no assert,
no PASS/FAIL judgement.  The user inspects the metric table and decides
whether the wrapper forward path correctly routes preprocessing buffers.

For each of four variants (baseline / smooth / quarot / viditq combined)
and each WAN shape, this measures the gap between
  (a) the W8A8 kernel wrapper output, with the matching preprocessing
      buffers installed and raw x handed in (base.py forward applies the
      preprocessing internally), versus
  (b) an fp32 dequant-then-matmul reference that applies the SAME
      preprocessing to (W, x) externally.

The gap is bf16-kernel noise + bf16 cast rounding when the wrapper code
path is correct.  Random Gaussian weights are deliberately the unfair
distribution for SmoothQuant and QuaRoT (which only help on outlier-
heavy real weights), so the vs-FP error is informational; the wrapper-
vs-ref error is the load-bearing one.
"""
from __future__ import annotations

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from qwan_extension import act_quant_bf16_with_sum
from qwan_extension.nn import QuantWanLinearW8A8

from ptqeval.wam.lingbot_va.method.viditq.quarot import (
    apply_input_rotation,
    random_sign_vector,
    rotate_weight,
)
from ptqeval.wam.lingbot_va.method.viditq.smooth_quant import compute_smooth_scale


def _per_channel_asym_quant_w(w_f32: torch.Tensor):
    x_max = w_f32.amax(dim=1).clamp_min(0.0)
    x_min = w_f32.amin(dim=1).clamp_max(0.0)
    delta = ((x_max - x_min) / 255.0).clamp_min(1e-8)
    zero_point = torch.round(x_min / delta) + 128.0
    scale_eff = delta.to(torch.bfloat16).to(torch.float32)
    w_int = (
        torch.round(w_f32 / scale_eff.unsqueeze(1)) - zero_point.unsqueeze(1)
    ).clamp(-128, 127)
    return w_int, scale_eff, zero_point


def _kernel_dequant_ref(
    x_preprocessed: torch.Tensor,
    w_preprocessed_f32: torch.Tensor,
    bias_bf16,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    """fp32 dequant-then-matmul reference for the preprocessed (W, x)
    pair.  Caller is responsible for applying the activation-side smooth /
    rotation BEFORE handing x_preprocessed in.  Runs through the dynamic
    act_quant launcher so the wrapper's per-token sym INT8 path is the
    one being verified."""
    w_int, scale_w_eff, zero_point = _per_channel_asym_quant_w(w_preprocessed_f32)

    x_bf16 = (
        x_preprocessed.to(torch.bfloat16)
        .reshape(-1, in_features)
        .contiguous()
    )
    x_int8, scale_x_bf16, sum_x_bf16 = act_quant_bf16_with_sum(x_bf16)
    sx = scale_x_bf16.to(torch.float32)
    sa = sum_x_bf16.to(torch.float32)
    zp = zero_point

    y_2d = x_int8.to(torch.float32) @ w_int.T
    y_2d = y_2d * sx.unsqueeze(1) * scale_w_eff.unsqueeze(0)
    y_2d = y_2d + sa.unsqueeze(1) * zp.unsqueeze(0) * scale_w_eff.unsqueeze(0)
    if bias_bf16 is not None:
        y_2d = y_2d + bias_bf16.to(torch.float32).unsqueeze(0)
    return y_2d.reshape(*x_preprocessed.shape[:-1], out_features).to(torch.bfloat16)


def _build_wrapper(
    W_preprocessed_f32: torch.Tensor,
    bias_bf16,
    device,
    smooth_mask=None,
    quarot_sign=None,
) -> QuantWanLinearW8A8:
    """Build a QuantWanLinearW8A8 by directly quantizing W_preprocessed_f32
    (fp32) -- matching ptq.py.  Going through nn.Linear.weight forces a
    bf16 cast that smooths over the fine-grained values introduced by
    SmoothQuant rescale / QuaRoT rotation, producing a different
    int_weight than ptq.py would emit.  Replicating ptq.py's code path
    here isolates the FORWARD plumbing, not the PTQ-time precision."""
    out_f, in_f = W_preprocessed_f32.shape
    has_bias = bias_bf16 is not None

    mod = QuantWanLinearW8A8(in_f, out_f, has_bias=has_bias).to(device)
    w_int, scale_eff, zp = _per_channel_asym_quant_w(W_preprocessed_f32)
    mod.int_weight = w_int.to(torch.int8).to(device).contiguous()
    mod.scale_weight = scale_eff.to(torch.bfloat16).to(device).contiguous()
    mod.zp_weight = zp.to(torch.int16).to(device).contiguous()
    if has_bias:
        mod.bias = bias_bf16.to(device).contiguous()
    if smooth_mask is not None:
        mod.act_channel_div = smooth_mask.to(device, torch.bfloat16).contiguous()
    if quarot_sign is not None:
        mod.quarot_sign = quarot_sign.to(device, torch.int8).contiguous()
    return mod


def _shape_metrics(
    variant: str,
    in_features: int,
    out_features: int,
    batch: int,
    seq: int,
    has_bias: bool,
    device: torch.device,
    seed: int = 0,
) -> dict:
    g = torch.Generator(device=device).manual_seed(seed)
    fp = nn.Linear(in_features, out_features, bias=has_bias).to(device).to(torch.bfloat16)
    with torch.no_grad():
        w_init = (
            torch.randn((out_features, in_features), device=device, generator=g) * 0.05
        ).to(torch.bfloat16)
        fp.weight.copy_(w_init)
        if has_bias:
            b_init = (
                torch.randn((out_features,), device=device, generator=g) * 0.1
            ).to(torch.bfloat16)
            fp.bias.copy_(b_init)
    x = (
        torch.randn((batch, seq, in_features), device=device, generator=g) * 0.5
    ).to(torch.bfloat16)

    bias_bf16 = fp.bias.detach().to(torch.bfloat16) if has_bias else None
    y_fp = F.linear(x, fp.weight, fp.bias).to(torch.bfloat16)

    use_smooth = "smooth" in variant or "viditq" in variant
    use_quarot = "quarot" in variant or "viditq" in variant
    smooth_mask = None
    quarot_sign = None
    if use_smooth:
        act_absmax = x.reshape(-1, in_features).abs().amax(dim=0).to(torch.float32)
        weight_absmax = fp.weight.detach().to(torch.float32).abs().amax(dim=0)
        smooth_mask = compute_smooth_scale(weight_absmax, act_absmax, alpha=0.99)
    if use_quarot:
        quarot_sign = random_sign_vector(in_features, seed=seed + 100).to(device)

    W_pre = fp.weight.detach().to(torch.float32)
    if smooth_mask is not None:
        W_pre = W_pre * smooth_mask.to(device, torch.float32).unsqueeze(0)
    if quarot_sign is not None:
        W_pre = rotate_weight(W_pre, quarot_sign)

    x_pre = x.to(torch.bfloat16)
    if smooth_mask is not None:
        x_pre = x_pre / smooth_mask.to(device, torch.bfloat16)
    if quarot_sign is not None:
        x_pre = apply_input_rotation(x_pre, quarot_sign).contiguous()

    y_ref = _kernel_dequant_ref(x_pre, W_pre, bias_bf16, out_features, in_features)
    mod = _build_wrapper(W_pre, bias_bf16, device, smooth_mask, quarot_sign)
    y_wrap = mod(x)

    err_vs_ref = (y_wrap.float() - y_ref.float()).abs().max().item()
    err_vs_fp = (y_wrap.float() - y_fp.float()).abs().max().item()
    out_mag = y_ref.float().abs().max().item()
    return {
        "err_vs_ref": err_vs_ref,
        "err_vs_fp": err_vs_fp,
        "out_mag": out_mag,
        "rel_vs_ref": err_vs_ref / max(out_mag, 1e-6),
    }


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA unavailable.", file=sys.stderr)
        return 2
    device = torch.device("cuda:0")

    shapes = [
        ("attn  3072->3072   b=T", 3072,  3072, 2, 256, True),
        ("attn  3072->3072   b=F", 3072,  3072, 2, 256, False),
        ("ffn   3072->14336  b=T", 3072, 14336, 1, 128, True),
        ("ffn   14336->3072  b=T", 14336, 3072, 1, 128, True),
    ]
    variants = ["baseline", "smooth", "quarot", "viditq"]

    header = (
        f"{'variant':<10} {'shape':<26} "
        f"{'err_vs_ref':>14} {'rel_vs_ref':>14} "
        f"{'err_vs_fp':>14} {'out_mag':>14}"
    )
    print(header)
    print("-" * len(header))
    for variant in variants:
        for name, in_f, out_f, batch, seq, has_bias in shapes:
            m = _shape_metrics(variant, in_f, out_f, batch, seq, has_bias, device)
            print(
                f"{variant:<10} {name:<26} "
                f"{m['err_vs_ref']:>14.3e} {m['rel_vs_ref']:>14.3e} "
                f"{m['err_vs_fp']:>14.3e} {m['out_mag']:>14.3e}"
            )
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
