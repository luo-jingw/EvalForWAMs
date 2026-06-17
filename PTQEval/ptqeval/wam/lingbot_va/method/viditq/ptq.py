# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Phase 24d PTQ runner: FP WanTransformer3DModel -> int weights + scales + zp.

Per-channel ASYMMETRIC quantization of every nn.Linear whose full module
name does NOT match layer_config.remain_fp_regex. Matches the ViDiT-Q
W8A8 / W4A8 algorithm exactly (quant_utils/qdiff/base/base_quantizer.py
DynamicQuantizer asym path, lines 130-156). For weight_bits=4 the weight
is further packed two nibbles per byte (low nibble -> col 2c, high
nibble -> col 2c+1, both signed).

Phase 37 extension (--quarot): when layer_config.quarot is true, each
target Linear gets a per-Linear random sign vector s in {+-1}^C_in
(seed = quarot_seed_base + layer_index for determinism). The weight is
rotated W_rot = (W * s) @ H / sqrt(C_in) via the structured Hadamard
in quarot.py BEFORE per-channel asym quant. The sign vector is emitted
as quarot_sign in the state_dict so the runtime wrapper can apply the
matching rotation to its activation input. Math identity (orthogonal H):
   y = x @ W.T == (x_rot) @ (W_rot).T   (un-quantized round-trip),
so the rotation is loss-free when nothing is quantized; the benefit
shows up after quantization because the rotation decorrelates outliers
and lets per-channel scales fit a tighter range.

Per output channel c, with weight W[c, :] (length C_in):
    x_max = max(W[c, :], 0)                              # clamp_min to 0
    x_min = min(W[c, :], 0)                              # clamp_max to 0
    delta = (x_max - x_min) / (n_levels - 1)             # fp32 scale
    zp    = round(x_min / delta) + n_levels / 2          # fp32, integer-valued
    int_w[c, k] = round(W[c, k] / delta) - zp            # clamp to [-128, 127]
                                                         #   (or [-8, 7] for n_bits=4)
where n_levels = 256 (w8) or 16 (w4). This convention matches the W8A8
kernel epilogue w_real = scale_w * (w_int + zp_w) (Phase 24b kernel,
verified). Plan section 18.2's written formula has a typo; the formula
implemented here is from ViDiT-Q upstream directly and is the
authoritative one.

Output: flat torch state_dict via torch.save. Per quantized layer:
    <module_name>.int_weight     int8  [C_out, C_in] (W8A8) or
                                       [C_out, C_in/2] (W4A8, QServe layout)
    <module_name>.scale_weight   bf16  [C_out]
    <module_name>.zp_weight      int16 [C_out]   (W8A8 only)
    <module_name>.szeros_weight  bf16  [C_out]   (W4A8 only;
                                                  = scale_weight * zp_unsigned)
    <module_name>.bias           bf16  [C_out]   (omitted if absent)
    <module_name>.quarot_sign    int8  [C_in]    (only when --quarot)
    <module_name>.act_channel_div bf16 [C_in]    (only when --smooth_quant)
    <module_name>.act_scale_static bf16 [1]      (only when static_act)

CLI:
    python -m ptqeval.wam.lingbot_va.method.viditq.ptq \\
        --layer_config PTQEval/ptqeval/wam/lingbot_va/method/viditq/configs/w8a8.yaml \\
        --output       results/viditq_w8a8_kernel/calib/int_weights.pth
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import OmegaConf

# Triggers ptqeval.wam.lingbot_va package init, which puts lingbot-va/ on sys.path.
import ptqeval.wam.lingbot_va  # noqa: F401

from wan_va.modules.utils import load_transformer  # noqa: E402


logger = logging.getLogger("ptqeval.wam.lingbot_va.method.viditq.ptq")


@dataclass
class IntLayerEntry:
    int_weight: torch.Tensor      # int8 [C_out, C_in] (W8A8) or [C_out, C_in/2]
                                  # in QServe-packed layout (W4A8)
    scale_weight: torch.Tensor    # bf16 [C_out]
    bias: Optional[torch.Tensor]  # bf16 [C_out] or None
    # Exactly one of zp_weight (W8A8) and szeros_weight (W4A8) is set.
    # The W4A8 epilogue (`- szeros * a_sum`, ViDiT-Q/QServe convention)
    # uses scale_w * zp_unsigned precomputed at PTQ time; the W8A8
    # epilogue (`+ a_sum * zp * b_scale`, ViDiT-Q signed convention)
    # uses the raw int16 zero-point. Different kernels demand different
    # types; loader matches the buffer name to the active subclass.
    zp_weight: Optional[torch.Tensor] = None         # int16 [C_out] (W8A8 only)
    szeros_weight: Optional[torch.Tensor] = None     # bf16 [C_out]  (W4A8 only)
    quarot_sign: Optional[torch.Tensor] = None       # int8 [C_in] or None
    act_channel_div: Optional[torch.Tensor] = None   # bf16 [C_in] (SmoothQuant
                                                     #   channel_mask) or None
    act_scale_static: Optional[torch.Tensor] = None  # bf16 [1] (Phase 33
                                                     #   per-layer scalar
                                                     #   for static act
                                                     #   quant) or None


def _per_channel_asym_quant(
    w_f32: torch.Tensor,
    n_bits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Asymmetric per-channel quantization of W [C_out, C_in] in fp32.

    Returns:
      int_w  : int8  [C_out, C_in], values in [-2^(n_bits-1), 2^(n_bits-1) - 1]
      scale  : bf16  [C_out]
      zp     : int16 [C_out]

    n_bits is 8 or 4 (caller packs nibbles afterwards for n_bits == 4).
    """
    if n_bits not in (4, 8):
        raise ValueError(f"n_bits must be 8 or 4, got {n_bits}")
    n_levels = 2 ** n_bits             # 256 for w8, 16 for w4
    half = n_levels // 2               # 128  /  8
    int_min = -half                    # -128 / -8
    int_max = half - 1                 # 127  /  7
    # Match ViDiT-Q's DynamicQuantizer asym branch (base_quantizer.py:130-148):
    # the row-wise max is clamped non-negative, the row-wise min non-positive,
    # to keep zero inside the dynamic range when weights happen to be all
    # one-sided.
    x_max = w_f32.amax(dim=1).clamp_min(0.0)
    x_min = w_f32.amin(dim=1).clamp_max(0.0)
    delta = ((x_max - x_min) / (n_levels - 1)).clamp_min(1e-8)
    # zero_point: integer-valued fp32; range typically [0, n_levels] for
    # negative-skewed weights and around [0, 0] for symmetric distributions.
    zero_point = torch.round(x_min / delta) + (n_levels / 2)
    # The kernel sees scale as bf16, so quantize using the rounded scale to
    # avoid PTQ-vs-runtime divergence at borderline weight values.
    scale_bf16 = delta.to(torch.bfloat16)
    scale_eff = scale_bf16.to(torch.float32)
    int_w = (
        torch.round(w_f32 / scale_eff.unsqueeze(1)) - zero_point.unsqueeze(1)
    ).clamp(int_min, int_max).to(torch.int8)
    zp_int16 = zero_point.to(torch.int16)
    return int_w, scale_bf16, zp_int16


def _pack_int4_two_per_byte(w_int4: torch.Tensor) -> torch.Tensor:
    """w_int4: int8 [M, K] in [-8, 7], K even. Returns packed int8 [M, K/2]
    with low nibble = col 2c, high nibble = col 2c+1, both signed.

    Used by the W8A8-derived signed W4 path (Phase 24d ; SUPERSEDED).
    The kernel-active W4A8 path uses _pack_int4_qserve below."""
    M, K = w_int4.shape
    assert K % 2 == 0, f"K must be even for int4 packing, got {K}"
    w32 = w_int4.to(torch.int32)
    low = w32[:, 0::2] & 0xF
    high = w32[:, 1::2] & 0xF
    packed_u = ((high << 4) | low) & 0xFF
    return packed_u.to(torch.uint8).view(torch.int8).contiguous()


def _per_channel_asym_quant_unsigned(
    w_f32: torch.Tensor,
    n_bits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Asymmetric per-channel UNSIGNED quant of W [C_out, C_in] in fp32.

    Returns:
      int_w  : uint8 [C_out, C_in], values in [0, 2^n_bits - 1]
      scale  : bf16  [C_out]
      zp_uns : uint8 [C_out]   (unsigned zero-point, in [0, 2^n_bits - 1])

    Formula matches omniserve W4A8 from_linear per-channel branch +
    QServe kernel epilogue convention `w_real = scale * (w_int - zp)`:
        scale  = (W.amax(1) - W.amin(1)) / (n_levels - 1)
        zp_uns = round(-W.amin(1) / scale).clamp(0, n_levels - 1)
        W_int  = (round(W / scale) + zp_uns).clamp(0, n_levels - 1)
    The kernel sees scale as bf16, so quantize against the bf16-rounded
    scale to avoid PTQ-vs-runtime divergence at borderline values.

    Note: this is DIFFERENT from _per_channel_asym_quant above. That one
    produces signed int_w + signed zp matching the W8A8 kernel epilogue
    `+ a_sum * zp * scale` (Phase 24b). The W4A8 QServe kernel uses
    `- szeros * a_sum` with unsigned zp; the sign flips. Both formulas
    are mathematically valid asym quants -- they differ in which side
    of the dequant the zero-point is folded into.
    """
    if n_bits not in (4,):
        raise ValueError(
            f"_per_channel_asym_quant_unsigned: only n_bits=4 supported "
            f"(QServe W4A8); got {n_bits}"
        )
    n_levels = 2 ** n_bits  # 16
    max_int = n_levels - 1  # 15
    x_max = w_f32.amax(dim=1)
    x_min = w_f32.amin(dim=1)
    delta = ((x_max - x_min) / max_int).clamp_min(1e-8)
    scale_bf16 = delta.to(torch.bfloat16)
    scale_eff = scale_bf16.to(torch.float32)
    zp_uns_f = torch.round(-x_min / scale_eff).clamp(0, max_int)
    int_w = (
        torch.round(w_f32 / scale_eff.unsqueeze(1)) + zp_uns_f.unsqueeze(1)
    ).clamp(0, max_int).to(torch.uint8)
    zp_uns = zp_uns_f.to(torch.uint8)
    return int_w, scale_bf16, zp_uns


def _pack_int4_qserve(w_int_unsigned: torch.Tensor) -> torch.Tensor:
    """QServe W4 weight pack. Verbatim from omniserve W4A8 from_linear
    per-channel branch (lines 295-327 of
    omniserve/modeling/layers/quantized_linear/w4a8_linear.py).

    Layout: weight in_features-axis is reorganized into 32-channel groups
    with intra-group permutation that matches m16n8k32 ldmatrix access
    inside dense_kernel0::share_to_reg_one_stage_B.

    Input  : uint8 [N, K] with values in [0, 15], N % 32 == 0, K % 32 == 0
    Output : int8  [N, K/2] with packed nibbles ((hi<<4)|lo) interpreted
             by the kernel after its 8-level reshape unpack.

    Algorithm (verbatim from omniserve):
      reshape  [N, K] -> [N/32, 2, 2, 8, K/32, 2, 4, 4]
      permute  (0, 4, 3, 6, 1, 5, 2, 7)
      permute  (0, 1, 2, 3, 5, 6, 7, 4)
      pack     (X[..., 1] << 4) | X[..., 0]
      reshape  -> [N/32, K/32, 32, 16] -> [N, K/2]
    """
    N, K = w_int_unsigned.shape
    if N % 32 != 0:
        raise ValueError(f"QServe W4 pack: N must be a multiple of 32, got {N}")
    if K % 32 != 0:
        raise ValueError(f"QServe W4 pack: K must be a multiple of 32, got {K}")

    # Reshape into the 8-level layout that the kernel ldmatrix loads expect.
    lw = w_int_unsigned.to(torch.int32)  # use int32 for << / | arithmetic
    lw = lw.reshape(N // 32, 2, 2, 8, K // 32, 2, 4, 4)
    lw = lw.permute(0, 4, 3, 6, 1, 5, 2, 7).contiguous()
    lw = lw.permute(0, 1, 2, 3, 5, 6, 7, 4).contiguous()
    # lw last dim is 4 (the inner 'pair' axis after both permutes); pack
    # adjacent pairs into bytes.
    packed = ((lw[..., 1] << 4) | (lw[..., 0] & 0xF)) & 0xFF
    packed = packed.reshape(N // 32, K // 32, 32, 16).reshape(N, K // 2)
    return packed.to(torch.uint8).view(torch.int8).contiguous()


def _quantize_one(
    linear: nn.Linear,
    weight_bits: int,
    smooth_channel_mask: Optional[torch.Tensor] = None,
    quarot_sign: Optional[torch.Tensor] = None,
) -> IntLayerEntry:
    """Pure-tensor asym quantize of a single nn.Linear. Bias copied as bf16.

    Optional preprocessing stages, in order:
      1. SmoothQuant (Phase 36):  W <- W * channel_mask[None, :]
      2. QuaRoT      (Phase 37):  W <- (W * sign) @ H / sqrt(C_in)
    Then per-channel asym quant on the final W. The matching runtime
    inverses (x / channel_mask, then (x * sign) @ H / sqrt(C_in)) are
    applied in base.py forward.

    Both preprocessing stages preserve the unquantized result exactly:
      y = x @ W.T == (x / mask) @ (W * mask).T
                  == (x_rot)    @ (W_rot).T   [H orthogonal]
    The benefit shows up after quantization (post-Phase-24d): outliers
    are redistributed across channels, letting per-channel scales fit a
    tighter range.
    """
    from ptqeval.wam.lingbot_va.method.viditq.quarot import rotate_weight

    w_f32 = linear.weight.detach().to(torch.float32)
    if smooth_channel_mask is not None:
        # channel_mask multiplies the input axis (C_in is dim=1 of W).
        w_f32 = w_f32 * smooth_channel_mask.to(w_f32.device, torch.float32).unsqueeze(0)
    if quarot_sign is not None:
        w_f32 = rotate_weight(w_f32, quarot_sign.to(w_f32.device))

    bias = (linear.bias.detach().to(torch.bfloat16).contiguous()
            if linear.bias is not None else None)
    quarot_out = (quarot_sign.detach().to(torch.int8).contiguous()
                  if quarot_sign is not None else None)
    smooth_out = (smooth_channel_mask.detach().to(torch.bfloat16).contiguous()
                  if smooth_channel_mask is not None else None)

    if weight_bits == 4:
        # Phase 28: ViDiT-Q/QServe W4A8 path. Unsigned per-channel asym
        # quant + QServe weight pack + szeros = scale * zp_unsigned
        # precomputed for the kernel epilogue `- szeros * a_sum`.
        int_w_u, scale_w, zp_uns = _per_channel_asym_quant_unsigned(w_f32, n_bits=4)
        packed = _pack_int4_qserve(int_w_u)
        szeros = (
            scale_w.to(torch.float32) * zp_uns.to(torch.float32)
        ).to(torch.bfloat16).contiguous()
        return IntLayerEntry(
            int_weight=packed.contiguous(),
            scale_weight=scale_w.contiguous(),
            bias=bias,
            zp_weight=None,
            szeros_weight=szeros,
            quarot_sign=quarot_out,
            act_channel_div=smooth_out,
        )

    # W8A8: signed per-channel asym, kernel epilogue `+ a_sum * zp * scale`.
    int_w, scale_w, zp_w = _per_channel_asym_quant(w_f32, weight_bits)
    return IntLayerEntry(
        int_weight=int_w.contiguous(),
        scale_weight=scale_w.contiguous(),
        bias=bias,
        zp_weight=zp_w.contiguous(),
        szeros_weight=None,
        quarot_sign=quarot_out,
        act_channel_div=smooth_out,
    )


def compute_int_state_dict(
    model: nn.Module,
    remain_fp_regex: str,
    weight_bits: int,
    quarot_enabled: bool = False,
    quarot_seed_base: int = 0,
    smooth_quant_enabled: bool = False,
    smooth_alpha: float = 0.99,
    static_act_enabled: bool = False,
    calib_data: Optional[dict[str, torch.Tensor]] = None,
) -> dict[str, IntLayerEntry]:
    """Walk model. For every nn.Linear whose full name does NOT match
    remain_fp_regex, compute its IntLayerEntry.

    Phase 36: when smooth_quant_enabled, requires calib_data dict mapping
    full module name -> per-input-channel act_absmax tensor. Computes
    channel_mask per target Linear via smooth_quant.compute_channel_mask
    and applies the matching weight rescale.

    Phase 37: when quarot_enabled, each target Linear gets a per-layer
    Bernoulli ±1 sign vector (seed = quarot_seed_base + layer_index, in
    iteration order of named_modules) used to rotate its weight.

    Phase 38: setting BOTH flags reproduces the paper-namesake "ViDiT"
    method. Order matches viditq_quant_layer.py:47-48 exactly: smooth
    first, then rotate, then quant.

    Phase 33: when static_act_enabled, derive a per-layer scalar from
    calib_data and emit as act_scale_static buffer. base.py forward
    routes to act_quant_bf16_with_sum_static (skips runtime amax).
    When stacked with smooth_quant, the static scale is computed on the
    POST-smooth distribution (act_absmax / channel_mask) so it bounds
    what x_smooth actually sees at runtime.
    """
    from ptqeval.wam.lingbot_va.method.viditq.quarot import random_sign_vector
    from ptqeval.wam.lingbot_va.method.viditq.smooth_quant import compute_smooth_scale

    if (smooth_quant_enabled or static_act_enabled) and calib_data is None:
        raise ValueError(
            "smooth_quant_enabled / static_act_enabled require calib_data; "
            "load via Phase 31 calib_data.pth and pass dict[layer_name -> "
            "[C_in] absmax]."
        )

    pattern = re.compile(remain_fp_regex)
    entries: dict[str, IntLayerEntry] = {}
    layer_index = 0
    missing_calib: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if pattern.search(name):
            logger.debug(f"skip FP-kept {name}")
            continue

        smooth_mask = None
        if smooth_quant_enabled:
            if name not in calib_data:
                # PTQ quantizes more Linears than the kernel block swaps
                # (e.g. cross-attn). Calib data only covers the 180
                # block-swapped targets. Skip smooth_quant on layers
                # without calib stats; their weights still get the
                # standard per-channel asym quant.
                missing_calib.append(name)
            else:
                act_absmax = calib_data[name].to(module.weight.device, torch.float32)
                weight_absmax = (
                    module.weight.detach().to(torch.float32).abs().amax(dim=0)
                )
                # smooth_quant.compute_smooth_scale returns the canonical
                # SmoothQuant s in fp32 [C_in]. The runtime buffer in
                # base.py is named act_channel_div precisely because s
                # acts as a divisor on x. Bug history: an earlier draft
                # used the inverted upstream channel_mask formula here,
                # which preserves the unquantized round-trip but flips
                # the post-quant effect -- it AMPLIFIES outlier
                # activations instead of suppressing them. compute_
                # smooth_scale's name + docstring exist to make the
                # convention unambiguous.
                smooth_mask = compute_smooth_scale(
                    weight_absmax, act_absmax, smooth_alpha
                )

        quarot_sign = None
        if quarot_enabled:
            quarot_sign = random_sign_vector(
                module.in_features, seed=quarot_seed_base + layer_index
            )

        # Phase 33: per-layer static activation scale. Derive from calib
        # data, accounting for smooth_quant rescale if it ran first.
        # QuaRoT is orthogonal so its post-rotation max equals the pre-
        # rotation max (in expectation; rotation redistributes outliers
        # rather than amplifying them), so we don't compensate for it.
        act_scale_static = None
        if static_act_enabled and name in calib_data:
            act_absmax = calib_data[name].to(torch.float32)
            if smooth_mask is not None:
                # post-smooth distribution: x_smooth = x / channel_mask
                post = act_absmax / smooth_mask.to(act_absmax.device)
            else:
                post = act_absmax
            act_scale_static = (post.amax() / 127.0).to(torch.bfloat16).reshape(1)

        entries[name] = _quantize_one(
            module,
            weight_bits,
            smooth_channel_mask=smooth_mask,
            quarot_sign=quarot_sign,
        )
        entries[name].act_scale_static = act_scale_static
        layer_index += 1

    if missing_calib:
        logger.info(
            f"smooth_quant: {len(missing_calib)} layers had no calib entry "
            f"(quantized without SmoothQuant; e.g. cross-attn). First 3: "
            f"{missing_calib[:3]}"
        )
    return entries


def _flatten_to_state_dict(
    entries: dict[str, IntLayerEntry],
) -> dict[str, torch.Tensor]:
    sd: dict[str, torch.Tensor] = {}
    for name, e in entries.items():
        sd[f"{name}.int_weight"] = e.int_weight.cpu()
        sd[f"{name}.scale_weight"] = e.scale_weight.cpu()
        # Exactly one of (zp_weight, szeros_weight) is set per entry; W8A8
        # uses zp_weight int16, W4A8 uses szeros_weight bf16. Loader matches
        # the key name to the active subclass's registered buffer.
        if e.zp_weight is not None:
            sd[f"{name}.zp_weight"] = e.zp_weight.cpu()
        if e.szeros_weight is not None:
            sd[f"{name}.szeros_weight"] = e.szeros_weight.cpu()
        if e.bias is not None:
            sd[f"{name}.bias"] = e.bias.cpu()
        if e.quarot_sign is not None:
            sd[f"{name}.quarot_sign"] = e.quarot_sign.cpu()
        if e.act_channel_div is not None:
            sd[f"{name}.act_channel_div"] = e.act_channel_div.cpu()
        if e.act_scale_static is not None:
            sd[f"{name}.act_scale_static"] = e.act_scale_static.cpu()
    return sd


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute INT weights + per-channel scales for LingBot-VA "
        "WanTransformer3DModel. Output: torch state_dict at --output."
    )
    parser.add_argument("--model_path", type=str,
                        default="/home/arash/EvalForWAMs/models/lingbot-va-posttrain-robotwin/transformer",
                        help="Path to the diffusers transformer dir (the one with config.json).")
    parser.add_argument("--layer_config", type=str, required=True,
                        help="YAML with weight_bits and remain_fp_regex.")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for the flat state_dict (.pth).")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device used for the quant math.")
    parser.add_argument("--load_dtype", type=str, default="bf16",
                        choices=["bf16", "fp16", "fp32"],
                        help="Dtype to load the FP model in.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    cfg = OmegaConf.load(args.layer_config)
    weight_bits = int(cfg.weight_bits)
    remain_fp_regex = str(cfg.remain_fp_regex)
    if weight_bits not in (4, 8):
        raise ValueError(f"weight_bits must be 8 or 4, got {weight_bits}")
    # Sanity: Phase 24d implements asymmetric weight quant only. Configs
    # MUST declare weight_sym: false. Asserting fails loud if a future
    # config flips this without code support.
    weight_sym = bool(cfg.get("weight_sym", True))
    if weight_sym:
        raise ValueError(
            f"layer_config {args.layer_config} has weight_sym=true (or unset, "
            f"default true); Phase 24d only supports asymmetric quant. Set "
            f"weight_sym: false in the yaml."
        )
    # Phase 33/36/37/38: optional preprocessing. Config drives the variant
    # (one ptq.py invocation per variant); CLI does not override config.
    quarot_enabled = bool(cfg.get("quarot", False))
    quarot_seed_base = int(cfg.get("quarot_seed_base", 0))
    smooth_quant_enabled = bool(cfg.get("smooth_quant", False))
    smooth_alpha = float(cfg.get("smooth_alpha", 0.99))
    static_act_enabled = bool(cfg.get("static_act", False))
    calib_data_path = cfg.get("calib_data_path", None)
    calib_data = None
    if smooth_quant_enabled or static_act_enabled:
        if not calib_data_path:
            raise ValueError(
                f"layer_config {args.layer_config} requires calib_data_path "
                f"(smooth_quant={smooth_quant_enabled} static_act="
                f"{static_act_enabled}); run Phase 31 calibration first."
            )
        calib_data = torch.load(str(calib_data_path), weights_only=True)
        if not isinstance(calib_data, dict):
            raise ValueError(
                f"calib_data_path {calib_data_path} did not load a dict; "
                f"Phase 31 dump should be dict[layer_name -> [C_in] tensor]."
            )

    load_dtype = {"bf16": torch.bfloat16,
                  "fp16": torch.float16,
                  "fp32": torch.float32}[args.load_dtype]
    device = torch.device(args.device)
    logger.info(f"loading FP model from {args.model_path} (dtype={load_dtype}, device={device})")
    model = load_transformer(
        args.model_path,
        torch_dtype=load_dtype,
        torch_device=device,
        attn_mode="torch",
    )
    model.eval()

    n_linear_total = sum(1 for _, m in model.named_modules() if isinstance(m, nn.Linear))
    logger.info(f"weight_bits={weight_bits} weight_sym=False (asym per-channel) "
                f"remain_fp_regex={remain_fp_regex!r}")
    logger.info(
        f"smooth_quant={smooth_quant_enabled} alpha={smooth_alpha} "
        f"quarot={quarot_enabled} seed_base={quarot_seed_base} "
        f"static_act={static_act_enabled}"
    )
    if calib_data is not None:
        logger.info(f"calib_data: {calib_data_path} ({len(calib_data)} layers)")
    logger.info(f"total nn.Linear in model: {n_linear_total}")

    entries = compute_int_state_dict(
        model,
        remain_fp_regex,
        weight_bits,
        quarot_enabled=quarot_enabled,
        quarot_seed_base=quarot_seed_base,
        smooth_quant_enabled=smooth_quant_enabled,
        smooth_alpha=smooth_alpha,
        static_act_enabled=static_act_enabled,
        calib_data=calib_data,
    )
    n_quant = len(entries)
    n_kept_fp = n_linear_total - n_quant
    logger.info(f"quantized {n_quant} layers; kept {n_kept_fp} as FP")

    sd = _flatten_to_state_dict(entries)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(sd, args.output)

    size_mb = os.path.getsize(args.output) / (1024.0 * 1024.0)
    logger.info(f"wrote {args.output} ({size_mb:.1f} MB; {len(sd)} tensor keys)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
