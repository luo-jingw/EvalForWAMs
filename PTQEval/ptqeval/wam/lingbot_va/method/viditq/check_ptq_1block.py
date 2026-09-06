# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Phase 42 step 6b observational smoke: synthetic 1-block PTQ run.

Builds 6 nn.Linear instances under simulated block name `blocks.5.*`
matching the WAN target Linear shapes (attn1.to_{q,k,v,out[0]} +
ffn.net[0].proj + ffn.net[2]). Runs compute_int_state_dict with two
config dicts (W4A4-MP + W8A8 reproductions) and prints per-Linear
metrics. No assert / no PASS judgement (principle.txt L12).

Run:
    python -m ptqeval.wam.lingbot_va.method.viditq.check_ptq_1block
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ptqeval.wam.lingbot_va.method.viditq.ptq import (
    _parse_bit_alloc_key,
    _resolve_layer_qconfig,
    compute_int_state_dict,
)


# Matches WanTransformerBlock targets — 6 Linears per block, dim=3072, ffn=14336.
TARGET_SHAPES = [
    ("attn1.to_q",          3072,  3072,  False),  # (suffix, C_in, C_out, has_bias)
    ("attn1.to_k",          3072,  3072,  False),
    ("attn1.to_v",          3072,  3072,  False),
    ("attn1.to_out.0",      3072,  3072,  True),
    ("ffn.net.0.proj",      3072,  14336, True),
    ("ffn.net.2",          14336,  3072,  True),
]


class _SyntheticBlock(nn.Module):
    """Stand-in for one WanTransformerBlock; only the 6 target Linears."""

    def __init__(self, block_idx: int = 5):
        super().__init__()
        self.block_idx = block_idx
        for suffix, c_in, c_out, has_bias in TARGET_SHAPES:
            mod = nn.Linear(c_in, c_out, bias=has_bias).to(torch.bfloat16)
            with torch.no_grad():
                mod.weight.uniform_(-0.05, 0.05)
                if has_bias:
                    mod.bias.uniform_(-0.01, 0.01)
            # Register under the dotted suffix so named_modules() sees the
            # exact "blocks.5.attn1.to_q" style path that bit_alloc matches.
            self._register_path(suffix, mod)

    def _register_path(self, dotted: str, leaf: nn.Module):
        parts = dotted.split(".")
        parent = self
        for p in parts[:-1]:
            if not hasattr(parent, p):
                setattr(parent, p, nn.Module())
            parent = getattr(parent, p)
        setattr(parent, parts[-1], leaf)


def _scale_stats(t: torch.Tensor) -> str:
    f = t.float()
    return (f"min={f.min().item():.3e} max={f.max().item():.3e} "
            f"mean={f.mean().item():.3e} std={f.std().item():.3e}")


def _percentiles(flat: torch.Tensor) -> list[float]:
    # torch.quantile caps input at ~16M; subsample large tensors.
    if flat.numel() > 1_000_000:
        idx = torch.randperm(flat.numel(), device=flat.device)[:1_000_000]
        flat = flat[idx]
    qs = torch.quantile(
        flat.float(),
        torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=flat.device),
    )
    return [round(x.item(), 2) for x in qs]


def _int_weight_hist(t: torch.Tensor, dtype_kind: str) -> str:
    """For W4A4 (uint8 packed nibbles) compute distribution after unpack;
    for W8A8 / W4A8 just summarize the stored values."""
    if dtype_kind == "w4a4_packed":
        u = t.to(torch.int32)
        low = (u & 0xF)
        high = (u >> 4) & 0xF
        # Sign-extend 4-bit
        low = torch.where(low >= 8, low - 16, low)
        high = torch.where(high >= 8, high - 16, high)
        vals = torch.stack([low, high], dim=-1).flatten()
        return f"range[-8,7] unpacked p[0,25,50,75,100]={_percentiles(vals)}"
    return f"p[0,25,50,75,100]={_percentiles(t.flatten())}"


def _run_variant(name: str, model: nn.Module, cfg: dict):
    print()
    print("=" * 86)
    print(f"[{name}]  bit_alloc dispatch + per-Linear state_dict observation")
    print("=" * 86)

    # Parse bit_alloc keys the same way ptq.main does.
    bit_alloc_raw = cfg.get("bit_alloc")
    bit_alloc = (
        {_parse_bit_alloc_key(k): list(v) for k, v in bit_alloc_raw.items()}
        if bit_alloc_raw else None
    )
    default_w = cfg.get("weight_bits_default", cfg.get("weight_bits", 8))
    default_a = cfg.get("act_bits_default", cfg.get("act_bits", 8))

    # Predispatch — show what bit_alloc resolves to per Linear (before PTQ).
    print(f"{'layer name':40s} {'shape':>14s}   resolved (w, a)")
    for fullname, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        w, a = _resolve_layer_qconfig(fullname, bit_alloc, default_w, default_a)
        shape_s = f"{mod.in_features}->{mod.out_features}"
        print(f"{fullname:40s} {shape_s:>14s}   ({w}, {a})")

    # Run actual PTQ on the synthetic block.
    entries = compute_int_state_dict(
        model,
        remain_fp_regex=cfg["remain_fp_regex"],
        weight_bits=default_w,
        act_bits=default_a,
        quarot_enabled=cfg.get("quarot", False),
        quarot_seed_base=cfg.get("quarot_seed_base", 0),
        quarot_layer_regex=cfg.get("quarot_layer_regex", ".*"),
        smooth_quant_enabled=False,    # 1-block smoke: skip smooth (no calib)
        calib_data=None,
        bit_alloc=bit_alloc,
    )
    print()
    print(f"PTQ produced {len(entries)} IntLayerEntry instances.")
    print()
    print(f"{'layer name':40s} {'int_w dtype':>10s} {'int_w shape':>16s}  "
          f"{'scale shape':>14s}  zp/szeros  quarot_sign")
    for fullname, e in entries.items():
        zps = []
        if e.zp_weight is not None: zps.append(f"zp[{tuple(e.zp_weight.shape)}]")
        if e.szeros_weight is not None: zps.append(f"szeros[{tuple(e.szeros_weight.shape)}]")
        zp_s = "+".join(zps) if zps else "(none)"
        qr_s = "yes" if e.quarot_sign is not None else "no"
        print(f"{fullname:40s} {str(e.int_weight.dtype).split('.')[1]:>10s} "
              f"{str(tuple(e.int_weight.shape)):>16s}  "
              f"{str(tuple(e.scale_weight.shape)):>14s}  {zp_s:9s}  {qr_s}")

    print()
    print("Per-Linear scale_weight stats + int_weight distribution:")
    for fullname, e in entries.items():
        # Discriminate W4A4 (uint8 packed + 2-D scale) vs W8/W4A8 (int8 + 1-D scale)
        is_w4a4 = (e.int_weight.dtype == torch.uint8 and e.scale_weight.dim() == 2)
        dtype_kind = "w4a4_packed" if is_w4a4 else "int_stored"
        print(f"  {fullname}")
        print(f"    scale_weight: {_scale_stats(e.scale_weight)}")
        print(f"    int_weight  : {_int_weight_hist(e.int_weight, dtype_kind)}")


def main():
    print("Phase 42 step 6b observational smoke — 1-block synthetic PTQ")
    print(f"torch {torch.__version__}")
    torch.manual_seed(0)

    # Configuration A: replicate W4A4-MP (paper Sec D.4 yaml decode).
    # blocks.0 fully FP (regex), block 1-29 per-Linear: {to_q,k,v, ffn.net.2}
    # -> W4A4; {to_out, ffn.net.0.proj} -> W8A8.
    cfg_w4a4 = {
        "remain_fp_regex": "blocks.0",                   # block 0 stays FP
        "weight_bits_default": 8,
        "act_bits_default": 8,
        "bit_alloc": {
            "4_4": ["attn1.to_q", "attn1.to_k", "attn1.to_v", "ffn.net.2"],
            # W8A8 layers fall through to default (8, 8) via remainder match.
        },
        "quarot": True,
        "quarot_seed_base": 0,
        "quarot_layer_regex": "attn1.to_q|attn1.to_k|attn1.to_v|ffn.net",
    }

    # Configuration B: replicate Phase 38 W8A8 paper-namesake yaml on this
    # synthetic block (no bit_alloc, all 6 Linears -> (8, 8); quarot all-on).
    cfg_w8a8 = {
        "remain_fp_regex": "blocks.0",
        "weight_bits": 8,
        "act_bits": 8,
        "quarot": True,
        "quarot_seed_base": 0,
    }

    block = _SyntheticBlock(block_idx=5).cuda()
    # Wrap in a tiny container so named_modules emits "blocks.5.*" paths.
    container = nn.Module()
    container.add_module("blocks", nn.ModuleList())
    # blocks[0..4] placeholder + blocks[5] = our block, so name = blocks.5.*
    for i in range(5):
        container.blocks.append(nn.Module())
    container.blocks.append(block)

    _run_variant("W4A4-MP (Phase 42)", container, cfg_w4a4)
    _run_variant("W8A8 (Phase 38)",   container, cfg_w8a8)


if __name__ == "__main__":
    main()
