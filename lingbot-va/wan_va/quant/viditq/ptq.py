# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Phase 18 PTQ runner: FP WanTransformer3DModel -> int weights + scales.

Per-channel symmetric quantization of every nn.Linear whose full module
name does NOT match layer_config.remain_fp_regex. For weight_bits=4 the
weight is further packed two nibbles per byte (low nibble -> col 2c,
high nibble -> col 2c+1, both signed).

Output: a flat torch state_dict written via torch.save. Keys are:
    <module_name>.int_weight     int8 [C_out, C_in_pack]
    <module_name>.scale_weight   bf16 [C_out]
    <module_name>.bias           bf16 [C_out]    (omitted if absent)

This matches the buffer names of QuantWanLinearBase (qwan_extension.nn),
so a kernel module can load its slice via mod.load_state_dict(filtered).

CLI:
    python -m wan_va.quant.viditq.ptq
        --layer_config wan_va/quant/viditq/configs/w8a8.yaml
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

# Allow `from wan_va.modules.utils import load_transformer` regardless of cwd.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_LINGBOT_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
if _LINGBOT_DIR not in sys.path:
    sys.path.insert(0, _LINGBOT_DIR)

from wan_va.modules.utils import load_transformer  # noqa: E402


logger = logging.getLogger("wan_va.quant.viditq.ptq")


@dataclass
class IntLayerEntry:
    int_weight: torch.Tensor      # int8 [C_out, C_in] or [C_out, C_in/2]
    scale_weight: torch.Tensor    # bf16 [C_out]
    bias: Optional[torch.Tensor]  # bf16 [C_out] or None


def _per_channel_sym_int8(w_f32: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """w_f32: [M, K] fp32. Returns (int8 [M, K] in [-127, 127], scale bf16 [M])."""
    row_max = w_f32.abs().amax(dim=1).clamp_min(1e-8)
    scale = row_max / 127.0
    scale_bf16 = scale.to(torch.bfloat16)
    scale_eff = scale_bf16.to(torch.float32)
    w_int = torch.round(w_f32 / scale_eff.unsqueeze(1)).clamp(-127, 127).to(torch.int8)
    return w_int, scale_bf16


def _per_channel_sym_int4_packed(
    w_f32: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """w_f32: [M, K] fp32, K even. Returns (packed int8 [M, K/2], scale bf16 [M])."""
    M, K = w_f32.shape
    assert K % 2 == 0, f"K must be even for int4 packing, got {K}"
    row_max = w_f32.abs().amax(dim=1).clamp_min(1e-8)
    scale = row_max / 7.0
    scale_bf16 = scale.to(torch.bfloat16)
    scale_eff = scale_bf16.to(torch.float32)
    w_dense = torch.round(w_f32 / scale_eff.unsqueeze(1)).clamp(-8, 7).to(torch.int8)
    w32 = w_dense.to(torch.int32)
    low = w32[:, 0::2] & 0xF
    high = w32[:, 1::2] & 0xF
    packed_u = ((high << 4) | low) & 0xFF
    packed = packed_u.to(torch.uint8).view(torch.int8).contiguous()
    return packed, scale_bf16


def _quantize_one(linear: nn.Linear, weight_bits: int) -> IntLayerEntry:
    """Pure-tensor quantize of a single nn.Linear. Bias copied as bf16."""
    w_f32 = linear.weight.detach().to(torch.float32)
    if weight_bits == 8:
        int_w, scale_w = _per_channel_sym_int8(w_f32)
    elif weight_bits == 4:
        int_w, scale_w = _per_channel_sym_int4_packed(w_f32)
    else:
        raise ValueError(f"weight_bits must be 8 or 4, got {weight_bits}")
    bias = (linear.bias.detach().to(torch.bfloat16).contiguous()
            if linear.bias is not None else None)
    return IntLayerEntry(
        int_weight=int_w.contiguous(),
        scale_weight=scale_w.contiguous(),
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
    logger.info(f"weight_bits={weight_bits} remain_fp_regex={remain_fp_regex!r}")
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
