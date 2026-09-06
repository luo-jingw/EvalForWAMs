# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Phase 36: SmoothQuant per-channel rescale factor (canonical form).

Mathematically equivalent to ViDiT-Q upstream
(quant_utils/qdiff/smooth_quant/sq_quant_layer.py:24-33) but returned in
the canonical SmoothQuant-paper convention (s, not 1/s), so the runtime
buffer semantics in base.py stay "per-channel DIVISOR for x":
    PTQ:     W_smooth = W * s   (per-input-channel rescale, broadcast over C_out)
    runtime: x_smooth = x / s   (matching inverse before quant)

Effect on the W8A8 quant pipeline (alpha=0.99 default, ViDiT-Q image-DiT):
  - Channels with large activation absmax get LARGER s, so x / s shrinks
    those columns -> per-token activation quant sees a tighter range ->
    less quant noise on outlier-heavy channels.
  - The matching W * s makes weights GROWS along the same columns; the
    per-channel weight scale absorbs the growth, so weight-side quant
    noise is unchanged in expectation.

Upstream notation cross-reference:
  upstream channel_mask = w^alpha / a^(1-alpha)        [sq_quant_layer.py:30]
  upstream applies as   W/channel_mask, x*channel_mask [:41, :55]
  ours       s          = a^(1-alpha) / w^alpha        [== 1 / channel_mask]
  ours applies as       W*s,            x/s
The two are bit-identical in the unquantized round-trip and produce the
same quantized output (modulo bf16 reciprocal precision) because s and
channel_mask satisfy s = 1/channel_mask exactly.

Round-trip identity (unquantized):
    y = x @ W.T = (x/s) @ (W*s).T = x @ W.T   for any positive s.
The benefit shows up only AFTER quantization (in Phase 24d's per-channel
asym quant of W*s and Phase 26a-1's per-token sym quant of x/s).
"""
from __future__ import annotations

import torch


def compute_smooth_scale(
    weight_absmax: torch.Tensor,
    act_absmax: torch.Tensor,
    alpha: float,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Compute the canonical SmoothQuant per-channel scale s in fp32.

    weight_absmax: fp32 [C_in], = |W|.amax(dim=0) per input channel.
    act_absmax:    fp32 [C_in], from Phase 31 calib_data (per-channel
                   running max of |x| across all calibration callsites).
    alpha:         float in (0, 1). alpha->1 weights act stats heavily
                   (canonical SmoothQuant uses 0.5 for LLMs; ViDiT-Q
                   image-DiT default is 0.99 because diffusion activations
                   have heavier outliers than text-token activations).
    eps:           clamp_min on both stats so layers with zero-magnitude
                   channels do not produce inf / nan.

    Returns: fp32 [C_in], strictly positive, s.

    Both inputs must be positive (abs() upstream guarantees this); the
    function asserts on non-finite output as a safety net.
    """
    assert weight_absmax.dim() == 1
    assert act_absmax.dim() == 1
    assert weight_absmax.shape == act_absmax.shape, (
        f"weight_absmax {weight_absmax.shape} vs act_absmax {act_absmax.shape}"
    )
    assert 0.0 < alpha < 1.0, f"alpha must be in (0, 1), got {alpha}"
    w = weight_absmax.to(torch.float32).clamp_min(eps)
    a = act_absmax.to(torch.float32).clamp_min(eps)
    s = a.pow(1.0 - alpha) / w.pow(alpha)
    assert torch.isfinite(s).all(), "smooth_scale contains nan/inf after compute"
    return s
