# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Phase 36: SmoothQuant alpha-blending channel mask.

Direct port of ViDiT-Q/quant_utils/qdiff/smooth_quant/sq_quant_layer.py:24-33.

For each target Linear with weight W [C_out, C_in] and per-input-channel
activation absmax act_absmax [C_in] (from Phase 31 calibration):

    weight_absmax[c] = max_{r} |W[r, c]|
    channel_mask[c]  = weight_absmax[c]^alpha / act_absmax[c]^(1 - alpha)

The PTQ pass rescales weight  W_smooth = W * channel_mask[None, :]
The runtime forward divides    x_smooth = x / channel_mask[None, :]

Composition with per-channel asym weight quant: per-output-channel scales
absorb the |W|/|act| asymmetry that channel_mask redistributes, so the
per-channel quant error drops on outlier-heavy weight/activation
distributions. Unquantized round-trip is preserved exactly:
    (x / mask) @ (W * mask).T = x @ W.T

alpha = 0.99 is upstream's image-DiT default (ViDiT-Q pixart config).
"""
from __future__ import annotations

import torch


def compute_channel_mask(
    weight_absmax: torch.Tensor,
    act_absmax: torch.Tensor,
    alpha: float,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Compute the SmoothQuant channel_mask in fp32.

    weight_absmax: fp32 [C_in], = |W|.amax(dim=0).
    act_absmax:    fp32 [C_in], from Phase 31 calib_data.
    alpha:         float in (0, 1).
    eps:           clamp_min on both stats so layers with zero-magnitude
                   channels do not produce inf / nan.

    Returns: fp32 [C_in], positive, channel_mask.

    Negative numerator/denominator would raise nan on **alpha for
    non-integer exponents; both inputs are abs() upstream so this is
    guaranteed positive.
    """
    assert weight_absmax.dim() == 1
    assert act_absmax.dim() == 1
    assert weight_absmax.shape == act_absmax.shape, (
        f"weight_absmax {weight_absmax.shape} vs act_absmax {act_absmax.shape}"
    )
    assert 0.0 < alpha < 1.0, f"alpha must be in (0, 1), got {alpha}"
    w = weight_absmax.to(torch.float32).clamp_min(eps)
    a = act_absmax.to(torch.float32).clamp_min(eps)
    mask = w.pow(alpha) / a.pow(1.0 - alpha)
    assert torch.isfinite(mask).all(), "channel_mask contains nan/inf after compute"
    return mask
