"""ViDiT-Q variant loader for Motus (kernel-only path).

In-place quantization of an already-instantiated Motus MoT model (3 experts:
video / und / action, 30 blocks each). The WAN/Motus blocks call each target
Linear as a plain submodule, so we replace them in place, matched by name suffix:

    video : video_model.wan_model.blocks.N.self_attn.{q,k,v,o} + ffn.{0,2}
    action: action_expert.blocks.N.wan_action_o + ffn.{0,2}
    und   : und_expert.blocks.N.wan_und_o + ffn.{0,2}

Scope note: action/und Q/K/V are fused nn.Parameter (wan_*_qkv, einsum) — not
nn.Linear — so they cannot be quantized; only their output proj + FFN are. Only
the video expert gets full attention QKV quant.

The wrapper class per Linear is inferred from the int_weights state_dict
buffer shapes (single source of truth), matching lingbot_va's discriminator:

    scale_weight 2-D                 -> W4A4  (per-group sym INT4)
    scale_weight 1-D + zp_weight     -> W8A8  (signed int8 + int16 zp)
    scale_weight 1-D + szeros_weight -> W4A8  (QServe nibble + szeros)
    no scale_weight                  -> stay FP

Contract (plan.txt §3.2), in-place, no return:

    load_quant_model(motus_model, int_weights_ckpt, layer_config=None,
                     device=None) -> None
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import OmegaConf

# Puts Motus/{,/bak,/inference/robotwin/Motus} on sys.path (models/wan/policy).
import ptqeval.wam.motus  # noqa: F401

from qwan_extension.nn import (  # noqa: E402
    QuantWanLinearBase,
    QuantWanLinearW4A4,
    QuantWanLinearW4A8,
    QuantWanLinearW8A8,
)

logger = logging.getLogger("ptqeval.wam.motus.method.viditq.loader")

# Per expert per block; matches ptq.py target set and the WAN/Motus module-call
# sites. Video expert (WanSelfAttention) has separate q/k/v/o Linears; the action
# and und experts fuse Q/K/V into an nn.Parameter (wan_{action,und}_qkv, applied
# by einsum -> NOT nn.Linear, cannot be quantized), so only their output proj
# (wan_{action,und}_o) + FFN are quantizable. FFN is nn.Sequential -> ffn.0/ffn.2.
_TARGET_SUFFIXES: tuple[str, ...] = (
    "self_attn.q",
    "self_attn.k",
    "self_attn.v",
    "self_attn.o",
    "wan_action_o",
    "wan_und_o",
    "ffn.0",
    "ffn.2",
)
PREPROCESSING_SUFFIXES: tuple[str, ...] = (".quarot_sign", ".act_channel_div")


def _infer_cls_from_state_dict(
    prefix: str,
    ckpt_keys: set[str],
    ckpt: dict[str, torch.Tensor],
) -> Optional[type[QuantWanLinearBase]]:
    """Return wrapper class for the Linear at `<prefix>`, or None if absent.

    Redefined here (rather than imported from lingbot_va.loader) because that
    module imports wan_va at class-definition time via its block subclass.
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
            f"szeros_weight present — cannot infer W8A8 vs W4A8."
        )
    raise RuntimeError(
        f"{prefix}.scale_weight has unexpected dim={sw.dim()} "
        f"(shape={tuple(sw.shape)})"
    )


def _set_leaf(parent: nn.Module, leaf: str, module: nn.Module) -> None:
    """Assign `module` at `leaf` on `parent` (int index for Sequential/ModuleList)."""
    if leaf.isdigit():
        parent[int(leaf)] = module
    else:
        setattr(parent, leaf, module)


def _get_leaf(parent: nn.Module, leaf: str) -> nn.Module:
    if leaf.isdigit():
        return parent[int(leaf)]
    return getattr(parent, leaf)


def load_quant_model(
    motus_model: nn.Module,
    int_weights_ckpt: str,
    layer_config: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> None:
    """In-place swap the two experts' target Linears to kernel-backed quant
    variants and load int weights. Mutates `motus_model`; returns None.
    """
    if device is None:
        device = next(motus_model.parameters()).device

    if layer_config:
        layer_cfg = OmegaConf.load(layer_config)
        logger.info(
            f"layer_config={layer_config} "
            f"(bit_alloc={layer_cfg.get('bit_alloc', None)}, "
            f"quarot={layer_cfg.get('quarot', False)}, "
            f"smooth_quant={layer_cfg.get('smooth_quant', False)})"
        )

    logger.info(f"loading int weights from {int_weights_ckpt}")
    raw_sd: dict[str, torch.Tensor] = torch.load(
        int_weights_ckpt, map_location=device, weights_only=True
    )
    ckpt_keys = set(raw_sd.keys())

    # Identify target Linears by module NAME suffix, using the model's own
    # named_modules() so loader and ptq agree on naming. Motus registers the
    # experts as top-level `video_model` / `action_expert` / `und_expert`
    # (the module_module wrappers re-hold the same objects; named_modules reports
    # the first-registered top-level name), so an in-place swap here is visible.
    def _is_target(name: str) -> bool:
        return any(name == s or name.endswith("." + s) for s in _TARGET_SUFFIXES)

    target_names = [
        n for n, m in motus_model.named_modules()
        if isinstance(m, nn.Linear) and _is_target(n)
    ]
    class_histogram: dict[str, int] = {}
    fp_linear_count = 0
    for name in target_names:
        cls = _infer_cls_from_state_dict(name, ckpt_keys, raw_sd)
        if cls is None:
            fp_linear_count += 1
            continue
        parent_path, _, leaf = name.rpartition(".")
        parent = motus_model.get_submodule(parent_path)
        fp_linear = _get_leaf(parent, leaf)
        _set_leaf(parent, leaf, cls.from_fp_linear(fp_linear))
        class_histogram[cls.__name__] = class_histogram.get(cls.__name__, 0) + 1

    logger.info(
        f"per-Linear class assignment: {class_histogram}; "
        f"{fp_linear_count} target Linears kept FP (no ckpt entry)."
    )
    # Guard against a silent no-op (e.g. module-naming drift): if the ckpt
    # carries quantized weights but nothing was swapped, the model would run
    # bf16 while reporting a quantized variant.
    if not class_histogram and any(k.endswith(".scale_weight") for k in ckpt_keys):
        raise RuntimeError(
            "int_weights ckpt contains quantized layers (*.scale_weight) but no "
            "target Linear was swapped — module naming mismatch between ptq and "
            "loader. Expected target names ending in one of "
            f"{_TARGET_SUFFIXES}."
        )
    torch.cuda.empty_cache()

    # Filter ckpt to keys matching an existing model buffer. Preprocessing
    # buffers (.quarot_sign / .act_channel_div) are installed post-load
    # because base.py registers them as `None` placeholders (registering an
    # empty buffer would size-mismatch on load_state_dict).
    model_keys = set(motus_model.state_dict().keys())
    main_sd: dict[str, torch.Tensor] = {}
    preprocessing_sd: dict[str, torch.Tensor] = {}
    skipped = 0
    for k, v in raw_sd.items():
        if any(k.endswith(suf) for suf in PREPROCESSING_SUFFIXES):
            module_name = k.rsplit(".", 1)[0]
            try:
                motus_model.get_submodule(module_name)
                preprocessing_sd[k] = v
            except AttributeError:
                skipped += 1
            continue
        if k in model_keys:
            main_sd[k] = v
        else:
            skipped += 1

    _, unexpected = motus_model.load_state_dict(main_sd, strict=False)
    if unexpected:
        raise RuntimeError(
            f"unexpected keys in int_weights state_dict after filtering: "
            f"{len(unexpected)} (first 5: {unexpected[:5]})"
        )

    for k, v in preprocessing_sd.items():
        module_name, _, buffer_name = k.rpartition(".")
        sub = motus_model.get_submodule(module_name)
        if hasattr(sub, buffer_name) and buffer_name not in sub._buffers:
            delattr(sub, buffer_name)
        sub.register_buffer(buffer_name, v.to(device).contiguous(), persistent=True)

    logger.info(
        f"int_weights load: applied {len(main_sd)} core tensors, "
        f"{len(preprocessing_sd)} preprocessing tensors, skipped {skipped} "
        f"non-target entries."
    )
    motus_model.to(device).eval().requires_grad_(False)
    torch.cuda.empty_cache()
