"""Observation check for the fastwam ViDiT-Q W4A4 path (Phase 52 + 53).

Emits metrics only; no assert / no PASS-FAIL (principle.txt L12). The reader
judges correctness from the printed numbers.

Covered observations:
  Phase 52 (PTQ selection): total nn.Linear, quantized count, per-expert count,
    per-tier (weight_bits, act_bits) histogram, sample IntLayerEntry buffer
    shapes / dtypes.
  Phase 53 (loader dispatch): class inferred from each tier's flattened
    state_dict buffers (_infer_cls_from_state_dict).
  Phase 53 (numerical): for one W4A4 and one W8A8 target Linear, the forward
    max_abs / mean_abs / rel_diff of the kernel-backed quant module vs bf16.

CLI:
    python -m ptqeval.wam.fastwam.method.viditq.check_w4a4 \\
        --ckpt models/robotwin2.0-fastwam/robotwin_uncond_3cam_384.pt \\
        --layer_config PTQEval/ptqeval/wam/fastwam/method/viditq/configs/w4a4.yaml
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from omegaconf import OmegaConf

import ptqeval.wam.fastwam  # noqa: F401
from ptqeval.wam.fastwam.method.viditq.loader import _infer_cls_from_state_dict
from ptqeval.wam.fastwam.method.viditq.ptq import load_fastwam_model
from ptqeval.wam.lingbot_va.method.viditq.ptq import (
    _flatten_to_state_dict,
    _parse_bit_alloc_key,
    _resolve_layer_qconfig,
    compute_int_state_dict,
)


def _expert_of(name: str) -> str:
    if "video_expert" in name:
        return "video"
    if "action_expert" in name:
        return "action"
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument(
        "--layer_config",
        default="PTQEval/ptqeval/wam/fastwam/method/viditq/configs/w4a4.yaml",
    )
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    dev = torch.device(args.device)
    cfg = OmegaConf.load(args.layer_config)
    bit_alloc = None
    if cfg.get("bit_alloc", None):
        bit_alloc = {
            _parse_bit_alloc_key(k): list(v)
            for k, v in OmegaConf.to_container(cfg.bit_alloc).items()
        }
    wd = int(cfg.get("weight_bits_default", cfg.get("weight_bits", 8)))
    ad = int(cfg.get("act_bits_default", cfg.get("act_bits", 8)))

    print(f"loading model {args.ckpt} ...", flush=True)
    model = load_fastwam_model(args.ckpt, device=dev, dtype=torch.bfloat16)
    lin_names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]

    # --- Phase 52: selection + tiers ---
    entries = compute_int_state_dict(
        model, str(cfg.remain_fp_regex), wd, act_bits=ad,
        quarot_enabled=bool(cfg.get("quarot", False)),
        quarot_layer_regex=str(cfg.get("quarot_layer_regex", ".*")),
        smooth_quant_enabled=False, bit_alloc=bit_alloc,
    )
    per_expert: dict[str, int] = {}
    per_tier: dict[tuple[int, int], int] = {}
    for name in entries:
        per_expert[_expert_of(name)] = per_expert.get(_expert_of(name), 0) + 1
        w, a = _resolve_layer_qconfig(name, bit_alloc, wd, ad)
        per_tier[(w, a)] = per_tier.get((w, a), 0) + 1

    print("\n===== Phase 52: PTQ selection =====")
    print(f"total nn.Linear      : {len(lin_names)}")
    print(f"quantized target     : {len(entries)}")
    print(f"by expert            : {per_expert}")
    print(f"by tier (w,a)        : {dict(sorted(per_tier.items()))}")

    # sample buffer shapes for one entry of each tier
    print("\nsample IntLayerEntry buffer shapes/dtypes by tier:")
    seen_tiers: set[tuple[int, int]] = set()
    sd = _flatten_to_state_dict(entries)
    for name, e in entries.items():
        w, a = _resolve_layer_qconfig(name, bit_alloc, wd, ad)
        if (w, a) in seen_tiers:
            continue
        seen_tiers.add((w, a))
        print(f"  ({w},{a}) {name}")
        print(f"      int_weight  {tuple(e.int_weight.shape)} {e.int_weight.dtype}")
        print(f"      scale_weight {tuple(e.scale_weight.shape)} {e.scale_weight.dtype}")
        print(f"      zp={None if e.zp_weight is None else tuple(e.zp_weight.shape)} "
              f"szeros={None if e.szeros_weight is None else tuple(e.szeros_weight.shape)} "
              f"quarot_sign={None if e.quarot_sign is None else tuple(e.quarot_sign.shape)}")
        # Phase 53 dispatch: class inferred from the flattened buffers.
        cls = _infer_cls_from_state_dict(name, set(sd.keys()), sd)
        print(f"      loader dispatch -> {cls.__name__ if cls else None}")

    # --- Phase 53: numerical forward diff for one W4A4 + one W8A8 target ---
    print("\n===== Phase 53: forward diff vs bf16 (from_fp_linear) =====")
    from qwan_extension.nn import QuantWanLinearW4A4, QuantWanLinearW8A8
    picks = {"W4A4": None, "W8A8": None}
    for name in entries:
        w, a = _resolve_layer_qconfig(name, bit_alloc, wd, ad)
        tag = "W4A4" if (w, a) == (4, 4) else ("W8A8" if (w, a) == (8, 8) else None)
        if tag and picks[tag] is None:
            picks[tag] = name
    for tag, cls in (("W4A4", QuantWanLinearW4A4), ("W8A8", QuantWanLinearW8A8)):
        name = picks[tag]
        if name is None:
            print(f"{tag}: no target of this tier in config")
            continue
        fp = model.get_submodule(name)
        x = torch.randn(1, 120, fp.in_features, device=dev, dtype=torch.bfloat16)
        with torch.no_grad():
            y_fp = fp(x).float()
            q = cls.from_fp_linear(fp).to(dev).eval()
            y_q = q(x).float()
        d = (y_q - y_fp).abs()
        print(f"{tag} {name}: in={fp.in_features} out={fp.out_features} "
              f"max_abs={d.max():.4e} mean_abs={d.mean():.4e} "
              f"rel={d.norm() / y_fp.norm():.4e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
