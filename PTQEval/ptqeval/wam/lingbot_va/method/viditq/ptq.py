# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Phase 24d PTQ runner: FP WanTransformer3DModel -> int weights + scales + zp.

Per-channel ASYMMETRIC quantization of every nn.Linear whose full module
name does NOT match layer_config.remain_fp_regex. Matches the ViDiT-Q
W8A8 / W4A8 algorithm exactly (quant_utils/qdiff/base/base_quantizer.py
DynamicQuantizer asym path, lines 130-156). For weight_bits=4 the weight
is further packed two nibbles per byte (low nibble -> col 2c, high
nibble -> col 2c+1, both signed).

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

Output: flat torch state_dict via torch.save. 4 keys per quantized layer:
    <module_name>.int_weight     int8  [C_out, C_in] or [C_out, C_in/2]
    <module_name>.scale_weight   bf16  [C_out]
    <module_name>.zp_weight      int16 [C_out]
    <module_name>.bias           bf16  [C_out]   (omitted if absent)

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
    int_weight: torch.Tensor      # int8 [C_out, C_in] or [C_out, C_in/2]
    scale_weight: torch.Tensor    # bf16 [C_out]
    zp_weight: torch.Tensor       # int16 [C_out]
    bias: Optional[torch.Tensor]  # bf16 [C_out] or None


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
    with low nibble = col 2c, high nibble = col 2c+1, both signed."""
    M, K = w_int4.shape
    assert K % 2 == 0, f"K must be even for int4 packing, got {K}"
    w32 = w_int4.to(torch.int32)
    low = w32[:, 0::2] & 0xF
    high = w32[:, 1::2] & 0xF
    packed_u = ((high << 4) | low) & 0xFF
    return packed_u.to(torch.uint8).view(torch.int8).contiguous()


def _quantize_one(linear: nn.Linear, weight_bits: int) -> IntLayerEntry:
    """Pure-tensor asym quantize of a single nn.Linear. Bias copied as bf16."""
    w_f32 = linear.weight.detach().to(torch.float32)
    int_w, scale_w, zp_w = _per_channel_asym_quant(w_f32, weight_bits)
    if weight_bits == 4:
        int_w = _pack_int4_two_per_byte(int_w)
    bias = (linear.bias.detach().to(torch.bfloat16).contiguous()
            if linear.bias is not None else None)
    return IntLayerEntry(
        int_weight=int_w.contiguous(),
        scale_weight=scale_w.contiguous(),
        zp_weight=zp_w.contiguous(),
        bias=bias,
    )


def compute_int_state_dict(
    model: nn.Module,
    remain_fp_regex: str,
    weight_bits: int,
) -> dict[str, IntLayerEntry]:
    """Walk model. For every nn.Linear whose full name does NOT match
    remain_fp_regex, compute its IntLayerEntry."""
    pattern = re.compile(remain_fp_regex)
    entries: dict[str, IntLayerEntry] = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if pattern.search(name):
            logger.debug(f"skip FP-kept {name}")
            continue
        entries[name] = _quantize_one(module, weight_bits)
    return entries


def _flatten_to_state_dict(
    entries: dict[str, IntLayerEntry],
) -> dict[str, torch.Tensor]:
    sd: dict[str, torch.Tensor] = {}
    for name, e in entries.items():
        sd[f"{name}.int_weight"] = e.int_weight.cpu()
        sd[f"{name}.scale_weight"] = e.scale_weight.cpu()
        sd[f"{name}.zp_weight"] = e.zp_weight.cpu()
        if e.bias is not None:
            sd[f"{name}.bias"] = e.bias.cpu()
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
    logger.info(f"total nn.Linear in model: {n_linear_total}")

    entries = compute_int_state_dict(model, remain_fp_regex, weight_bits)
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
