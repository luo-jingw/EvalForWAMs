# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""ViDiT-Q variant loader (kernel-only path, Phase 20 + Phase 42).

Implements the loader contract:

    load_quant_model(wan_model_path, variant_args, device, dtype) -> nn.Module

variant_args schema (yaml form, OmegaConf-loaded by the server then handed
in as a plain dict):

    layer_config:      path to a configs/<variant>.yaml — read only for
                       logging; per-Linear class dispatch is inferred from
                       the int_weights state_dict (single source of truth).
    int_weights_ckpt:  path to the PTQ-produced int_weights state_dict
                       (produced by ptqeval.wam.lingbot_va.method.viditq.ptq).

Class dispatch (Phase 42):
  For each potential target Linear "blocks.<i>.<suffix>" where suffix in
  the 6 target suffixes, the loader inspects the ckpt for matching
  buffers and decides the wrapper class:

    scale_weight 1-D + zp_weight        -> W8A8  (signed int8 + int16 zp)
    scale_weight 1-D + szeros_weight    -> W4A8  (QServe nibble pack + szeros)
    scale_weight 2-D + neither          -> W4A4  (per-group sym INT4, plan G5)
    no scale_weight at all              -> stay FP (e.g. blocks.0 in W4A4-MP)

  A block whose 6 target Linears all stay FP is left unwrapped: the
  original WanTransformerBlock instance keeps running its bf16 path.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn
from omegaconf import OmegaConf

# Triggers ptqeval.wam.lingbot_va package init, which puts lingbot-va/ on sys.path.
import ptqeval.wam.lingbot_va  # noqa: F401

from qwan_extension.nn import (  # noqa: E402
    QuantWanLinearBase,
    QuantWanLinearW4A4,
    QuantWanLinearW4A8,
    QuantWanLinearW8A8,
)
from wan_va.modules.utils import load_transformer  # noqa: E402

from ptqeval.wam.lingbot_va.method.viditq.block import (  # noqa: E402
    QuantWanTransformerBlockWithCudaKernel,
)


logger = logging.getLogger("ptqeval.wam.lingbot_va.method.viditq.loader")


_TARGET_SUFFIXES: tuple[str, ...] = (
    "attn1.to_q",
    "attn1.to_k",
    "attn1.to_v",
    "attn1.to_out.0",
    "ffn.net.0.proj",
    "ffn.net.2",
)


def _infer_cls_from_state_dict(
    prefix: str,
    ckpt_keys: set[str],
    ckpt: dict[str, torch.Tensor],
) -> Optional[type[QuantWanLinearBase]]:
    """Return wrapper class for the Linear at `<prefix>`, or None if absent.

    Discriminates W8A8 / W4A8 / W4A4 by buffer-shape pattern:
      scale_weight 2-D  -> W4A4
      scale_weight 1-D + zp_weight     -> W8A8
      scale_weight 1-D + szeros_weight -> W4A8
    """
    sw_key = f"{prefix}.scale_weight"
    if sw_key not in ckpt_keys:
        return None
    sw = ckpt[sw_key]
    if sw.dim() == 2:
        return QuantWanLinearW4A4
    if sw.dim() == 1:
        if f"{prefix}.zp_weight" in ckpt_keys:
            return QuantWanLinearW8A8
        if f"{prefix}.szeros_weight" in ckpt_keys:
            return QuantWanLinearW4A8
        raise RuntimeError(
            f"{prefix}.scale_weight is 1-D but neither zp_weight nor "
            f"szeros_weight present in ckpt — cannot infer W8A8 vs W4A8."
        )
    raise RuntimeError(
        f"{prefix}.scale_weight has unexpected dim={sw.dim()} (shape={tuple(sw.shape)})"
    )


def load_quant_model(
    wan_model_path: str,
    variant_args: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    layer_config_path = variant_args.get("layer_config")
    int_weights_ckpt = variant_args.get("int_weights_ckpt")
    if not int_weights_ckpt:
        raise ValueError("variant_args must include 'int_weights_ckpt'.")

    if layer_config_path:
        layer_cfg = OmegaConf.load(layer_config_path)
        logger.info(
            f"layer_config={layer_config_path} "
            f"(bit_alloc={layer_cfg.get('bit_alloc', None)}, "
            f"quarot={layer_cfg.get('quarot', False)}, "
            f"smooth_quant={layer_cfg.get('smooth_quant', False)})"
        )

    logger.info(
        f"loading FP transformer from {wan_model_path} "
        f"(dtype={dtype}, device={device})"
    )
    model = load_transformer(
        wan_model_path,
        torch_dtype=dtype,
        torch_device=device,
        attn_mode="torch",
    )
    model.eval()

    # Load ckpt early so we can drive class dispatch off its buffer shapes.
    logger.info(f"loading int weights from {int_weights_ckpt}")
    raw_sd: dict[str, torch.Tensor] = torch.load(
        int_weights_ckpt, map_location=device, weights_only=True
    )
    ckpt_keys = set(raw_sd.keys())

    # Per-block per-Linear class assignment from ckpt shapes.
    n_blocks = len(model.blocks)
    per_block_class_view: list[dict[str, str]] = []
    fp_block_count = 0
    for i in range(n_blocks):
        block_prefix = f"blocks.{i}"
        cls_map: dict[str, type[QuantWanLinearBase]] = {}
        for suf in _TARGET_SUFFIXES:
            cls = _infer_cls_from_state_dict(f"{block_prefix}.{suf}", ckpt_keys, raw_sd)
            if cls is not None:
                cls_map[suf] = cls
        if not cls_map:
            # All 6 targets absent from ckpt (e.g. W4A4-MP blocks.0).
            fp_block_count += 1
            per_block_class_view.append({"_kept_fp": "yes"})
            continue
        model.blocks[i] = QuantWanTransformerBlockWithCudaKernel(
            model.blocks[i], cls_map
        )
        per_block_class_view.append({s: c.__name__ for s, c in cls_map.items()})

    # Brief histogram (per-Linear class counts across all wrapped blocks).
    class_histogram: dict[str, int] = {}
    for view in per_block_class_view:
        for v in view.values():
            class_histogram[v] = class_histogram.get(v, 0) + 1
    logger.info(
        f"per-Linear class assignment: {class_histogram}; "
        f"{fp_block_count} blocks kept fully FP (no ckpt entry)."
    )

    torch.cuda.empty_cache()

    # ptq.py quantizes every nn.Linear that escapes remain_fp_regex; the
    # kernel block only swaps 6 target Linears per block. Filter ckpt to
    # keys that match an existing model buffer so extra ptq entries are
    # silently skipped.
    #
    # Phase 37+: PTQ output may include optional preprocessing tensors
    # (.quarot_sign, .act_channel_div) for target layers. These are NOT
    # in model_keys because base.py does not register placeholders for
    # them (would trigger size mismatch on load); instead loader installs
    # them as proper buffers post-load, only on layers that have them.
    model_keys = set(model.state_dict().keys())
    PREPROCESSING_SUFFIXES = (".quarot_sign", ".act_channel_div")

    main_sd = {}
    preprocessing_sd = {}
    skipped = 0
    for k, v in raw_sd.items():
        if any(k.endswith(suf) for suf in PREPROCESSING_SUFFIXES):
            module_name = k.rsplit(".", 1)[0]
            try:
                model.get_submodule(module_name)
                preprocessing_sd[k] = v
            except AttributeError:
                skipped += 1
            continue
        if k in model_keys:
            main_sd[k] = v
        else:
            skipped += 1

    _, unexpected = model.load_state_dict(main_sd, strict=False)
    if unexpected:
        raise RuntimeError(
            f"unexpected keys in int_weights state_dict after filtering: "
            f"{len(unexpected)} (first 5: {unexpected[:5]})"
        )

    # Register Phase 37+ preprocessing buffers on the matching Linears.
    # Done AFTER load_state_dict so the buffer's size is set from the ckpt
    # tensor and is consistent with the per-layer in_features. base.py
    # pre-seeds these names as `self.foo = None` so the forward path can
    # branch on `is not None`; delete that placeholder before
    # register_buffer to avoid "attribute already exists".
    for k, v in preprocessing_sd.items():
        module_name, _, buffer_name = k.rpartition(".")
        sub = model.get_submodule(module_name)
        if hasattr(sub, buffer_name) and buffer_name not in sub._buffers:
            delattr(sub, buffer_name)
        sub.register_buffer(buffer_name, v.to(device).contiguous(), persistent=True)

    logger.info(
        f"int_weights load: applied {len(main_sd)} core tensors, "
        f"{len(preprocessing_sd)} preprocessing tensors "
        f"(quarot_sign / act_channel_div), skipped {skipped} non-target entries."
    )

    model.to(device).eval().requires_grad_(False)
    torch.cuda.empty_cache()
    return model
