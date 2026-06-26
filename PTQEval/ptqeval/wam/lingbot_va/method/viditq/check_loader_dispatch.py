# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Phase 42 step 6d observational check: loader per-Linear dispatch.

Builds a SYNTHETIC int_weights state_dict whose buffer shapes match what
PTQ + the W4A4-MP / W4A8-MP / W8A8 yamls would produce, then exercises
loader._infer_cls_from_state_dict on every target Linear. Reports the
resolved class histogram + first/last block detail. No assert / no PASS
(principle.txt L12).

Avoids loading the 5 GB FP model + actual PTQ pipeline; that end-to-end
test is the user's step 8/9 (see Appendix A of plan.txt).

Run:
    python -m ptqeval.wam.lingbot_va.method.viditq.check_loader_dispatch
"""
from __future__ import annotations

import re
from collections import Counter

import torch
from omegaconf import OmegaConf

from ptqeval.wam.lingbot_va.method.viditq.loader import (
    _TARGET_SUFFIXES,
    _infer_cls_from_state_dict,
)
from ptqeval.wam.lingbot_va.method.viditq.ptq import (
    _parse_bit_alloc_key,
    _resolve_layer_qconfig,
)


N_BLOCKS = 30
DIM = 3072
FFN_DIM = 14336

# Per-suffix (C_in, C_out) shape table from WanTransformer3DModel.
_SUFFIX_SHAPE: dict[str, tuple[int, int]] = {
    "attn1.to_q":      (DIM, DIM),
    "attn1.to_k":      (DIM, DIM),
    "attn1.to_v":      (DIM, DIM),
    "attn1.to_out.0":  (DIM, DIM),
    "ffn.net.0.proj":  (DIM, FFN_DIM),
    "ffn.net.2":       (FFN_DIM, DIM),
}


def _synthesize_buffers(
    prefix: str,
    w_bits: int,
    a_bits: int,
    c_in: int,
    c_out: int,
    quarot_on: bool,
) -> dict[str, torch.Tensor]:
    """Mirror PTQ output shapes per the (w, a) tier."""
    sd: dict[str, torch.Tensor] = {}
    if w_bits == 4 and a_bits == 4:
        sd[f"{prefix}.int_weight"]   = torch.zeros(c_out, c_in // 2,  dtype=torch.uint8)
        sd[f"{prefix}.scale_weight"] = torch.zeros(c_out, c_in // 128, dtype=torch.bfloat16)
    elif w_bits == 4 and a_bits == 8:
        sd[f"{prefix}.int_weight"]    = torch.zeros(c_out, c_in // 2, dtype=torch.int8)
        sd[f"{prefix}.scale_weight"]  = torch.zeros(c_out, dtype=torch.bfloat16)
        sd[f"{prefix}.szeros_weight"] = torch.zeros(c_out, dtype=torch.bfloat16)
    elif w_bits == 8 and a_bits == 8:
        sd[f"{prefix}.int_weight"]   = torch.zeros(c_out, c_in,  dtype=torch.int8)
        sd[f"{prefix}.scale_weight"] = torch.zeros(c_out, dtype=torch.bfloat16)
        sd[f"{prefix}.zp_weight"]    = torch.zeros(c_out, dtype=torch.int16)
    else:
        raise ValueError(f"unsupported (w, a) tier = ({w_bits}, {a_bits})")
    if quarot_on:
        sd[f"{prefix}.quarot_sign"] = torch.zeros(c_in, dtype=torch.int8)
    return sd


def _synthesize_ckpt(cfg_path: str) -> tuple[dict[str, torch.Tensor], dict]:
    cfg = OmegaConf.load(cfg_path)
    remain_fp_regex = re.compile(str(cfg.remain_fp_regex))
    bit_alloc_raw = cfg.get("bit_alloc")
    bit_alloc = (
        {_parse_bit_alloc_key(k): list(v) for k, v in OmegaConf.to_container(bit_alloc_raw).items()}
        if bit_alloc_raw else None
    )
    default_w = int(cfg.get("weight_bits_default", cfg.get("weight_bits", 8)))
    default_a = int(cfg.get("act_bits_default", cfg.get("act_bits", 8)))
    quarot_on = bool(cfg.get("quarot", False))
    quarot_pat = (
        re.compile(str(cfg.get("quarot_layer_regex", ".*"))) if quarot_on else None
    )

    sd: dict[str, torch.Tensor] = {}
    for i in range(N_BLOCKS):
        for suf in _TARGET_SUFFIXES:
            full_name = f"blocks.{i}.{suf}"
            if remain_fp_regex.search(full_name):
                continue
            w, a = _resolve_layer_qconfig(full_name, bit_alloc, default_w, default_a)
            c_in, c_out = _SUFFIX_SHAPE[suf]
            q_on = bool(quarot_pat and quarot_pat.search(full_name))
            sd.update(_synthesize_buffers(full_name, w, a, c_in, c_out, q_on))
    return sd, {
        "default": (default_w, default_a),
        "bit_alloc": bit_alloc,
        "quarot": quarot_on,
        "remain_fp_regex": cfg.remain_fp_regex,
    }


def _exercise_loader_dispatch(label: str, cfg_path: str) -> None:
    print()
    print("=" * 88)
    print(f"[{label}]  config={cfg_path}")
    print("=" * 88)

    sd, info = _synthesize_ckpt(cfg_path)
    print(f"  default (w, a)         : {info['default']}")
    print(f"  bit_alloc              : {info['bit_alloc']}")
    print(f"  quarot                 : {info['quarot']}")
    print(f"  remain_fp_regex        : {info['remain_fp_regex']!r}")
    print(f"  synthetic ckpt entries : {len(sd)}")

    ckpt_keys = set(sd.keys())
    class_counter: Counter[str] = Counter()
    fp_block_count = 0
    first_block_view: dict[str, str] = {}
    last_block_view: dict[str, str] = {}
    for i in range(N_BLOCKS):
        block_view: dict[str, str] = {}
        for suf in _TARGET_SUFFIXES:
            cls = _infer_cls_from_state_dict(f"blocks.{i}.{suf}", ckpt_keys, sd)
            if cls is None:
                block_view[suf] = "FP"
            else:
                block_view[suf] = cls.__name__
                class_counter[cls.__name__] += 1
        if all(v == "FP" for v in block_view.values()):
            fp_block_count += 1
        if i == 0:
            first_block_view = block_view
        if i == N_BLOCKS - 1:
            last_block_view = block_view

    print(f"  per-Linear class histogram (target Linears): {dict(class_counter)}")
    print(f"  FP-kept blocks (all 6 targets FP)         : {fp_block_count}")
    print(f"  blocks.0  view: {first_block_view}")
    print(f"  blocks.{N_BLOCKS - 1} view: {last_block_view}")


def main() -> None:
    print("Phase 42 step 6d observational check — loader per-Linear class dispatch")
    cfg_root = "PTQEval/ptqeval/wam/lingbot_va/method/viditq/configs"
    _exercise_loader_dispatch("W4A4-MP (Phase 42)", f"{cfg_root}/w4a4.yaml")
    _exercise_loader_dispatch("W4A8-MP (Phase 40)", f"{cfg_root}/w4a8.yaml")
    _exercise_loader_dispatch("W8A8    (Phase 38)", f"{cfg_root}/w8a8.yaml")


if __name__ == "__main__":
    main()
