# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""ViDiT-Q variant loader (kernel-only path, Phase 20).

Implements the loader contract:

    load_quant_model(wan_model_path, variant_args, device, dtype) -> nn.Module

variant_args schema (yaml form, OmegaConf-loaded by the server then handed
in as a plain dict):

    layer_config:      path to a configs/w8a8.yaml or configs/w4a8.yaml
    int_weights_ckpt:  path to the PTQ-produced int_weights state_dict
                       (produced by ptqeval.wam.lingbot_va.method.viditq.ptq)

Behavior:
  1. Load the FP WanTransformer3DModel as a structural source (embeddings,
     norms, cross-attn, scale_shift_table all stay FP).
  2. Wrap every transformer block with
     QuantWanTransformerBlockWithCudaKernel(ref_block, quant_linear_cls).
     The wrapper mutates the ref block in place by swapping the 6 target
     Linears (4 attn projections + 2 ffn) for kernel-backed modules. The
     kernel modules are initialized from the ref FP weights via
     from_fp_linear.
  3. Load the offline-computed int weights from int_weights_ckpt and
     overwrite the kernel modules' buffers. This is the authoritative
     source. The from_fp_linear initialization in step 2 is redundant but
     keeps the wrapper construction self-contained.
"""
from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from omegaconf import OmegaConf

# Triggers ptqeval.wam.lingbot_va package init, which puts lingbot-va/ on sys.path.
import ptqeval.wam.lingbot_va  # noqa: F401

from qwan_extension.nn import (  # noqa: E402
    QuantWanLinearBase,
    QuantWanLinearW4A8,
    QuantWanLinearW8A8,
)
from wan_va.modules.utils import load_transformer  # noqa: E402

from ptqeval.wam.lingbot_va.method.viditq.block import (  # noqa: E402
    QuantWanTransformerBlockWithCudaKernel,
)


logger = logging.getLogger("ptqeval.wam.lingbot_va.method.viditq.loader")


_WEIGHT_BITS_TO_CLS: dict[int, type[QuantWanLinearBase]] = {
    8: QuantWanLinearW8A8,
    4: QuantWanLinearW4A8,  # Phase 28: ViDiT-Q/QServe W4A8 port.
}


def load_quant_model(
    wan_model_path: str,
    variant_args: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    layer_config_path = variant_args.get("layer_config")
    int_weights_ckpt = variant_args.get("int_weights_ckpt")
    if not layer_config_path:
        raise ValueError("variant_args must include 'layer_config'.")
    if not int_weights_ckpt:
        raise ValueError("variant_args must include 'int_weights_ckpt'.")

    layer_cfg = OmegaConf.load(layer_config_path)
    # Phase 40: mixed-precision configs declare `weight_bits_default` +
    # `bit_alloc`. Homogeneous (legacy) configs only set `weight_bits`.
    bit_alloc_raw = layer_cfg.get("bit_alloc", None)
    bit_alloc: dict[int, list[str]] = {}
    if bit_alloc_raw:
        bit_alloc = {int(k): list(v) for k, v in OmegaConf.to_container(bit_alloc_raw).items()}
        weight_bits_default = int(layer_cfg.get("weight_bits_default",
                                                layer_cfg.get("weight_bits", 8)))
    else:
        weight_bits_default = int(layer_cfg.weight_bits)

    if weight_bits_default not in _WEIGHT_BITS_TO_CLS:
        raise ValueError(
            f"weight_bits (default) ={weight_bits_default} not supported; "
            f"supported: {sorted(_WEIGHT_BITS_TO_CLS.keys())}."
        )
    for b in bit_alloc:
        if b not in _WEIGHT_BITS_TO_CLS:
            raise ValueError(
                f"bit_alloc bits {b} not supported; supported: "
                f"{sorted(_WEIGHT_BITS_TO_CLS.keys())}."
            )

    logger.info(
        f"loading FP transformer from {wan_model_path} "
        f"(dtype={dtype}, device={device}, weight_bits_default={weight_bits_default}, "
        f"bit_alloc={bit_alloc or 'homogeneous'})"
    )
    model = load_transformer(
        wan_model_path,
        torch_dtype=dtype,
        torch_device=device,
        attn_mode="torch",
    )
    model.eval()

    n_blocks = len(model.blocks)
    # Phase 40: per-block class dispatch. Resolve bits per block via
    # prefix match against bit_alloc ({4: ["blocks.13.", ...]}); the
    # whole block (all 6 swap targets) gets the same class. Matches
    # ViDiT-Q upstream w4a8_mixed_precision.yaml: bit policy is per-
    # block, not per-Linear within a block.
    block_classes: list[type[QuantWanLinearBase]] = []
    for i in range(n_blocks):
        block_name = f"blocks.{i}."
        bits = weight_bits_default
        for b, prefixes in bit_alloc.items():
            if any(p in block_name for p in prefixes):
                bits = b
                break
        block_classes.append(_WEIGHT_BITS_TO_CLS[bits])

    if bit_alloc:
        block_bits_view = [next(b for b, c in _WEIGHT_BITS_TO_CLS.items() if c is cls)
                           for cls in block_classes]
        logger.info(
            f"per-block bit assignment ({n_blocks} blocks): {block_bits_view}"
        )
    else:
        logger.info(f"wrapping {n_blocks} blocks with "
                    f"QuantWanTransformerBlockWithCudaKernel "
                    f"({block_classes[0].__name__}) homogeneous")

    for i in range(n_blocks):
        model.blocks[i] = QuantWanTransformerBlockWithCudaKernel(
            model.blocks[i], block_classes[i]
        )
    # Free CUDA fragments created by from_fp_linear scratch.
    torch.cuda.empty_cache()

    logger.info(f"loading int weights from {int_weights_ckpt}")
    raw_sd = torch.load(int_weights_ckpt, map_location=device, weights_only=True)

    # ptq.py quantizes every nn.Linear that escapes remain_fp_regex (~300
    # per WAN). The kernel block only swaps 6 target Linears per block
    # (4 self-attn + 2 ffn = 180 total); other quantized Linears
    # (e.g. cross-attn attn2) are not present as kernel buffers in the
    # model. Filter to keys that match an existing model buffer so the
    # extra ptq entries are silently skipped.
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
            # Buffer goes through install_preprocessing_buffer below; keep
            # only entries whose owning Linear module exists in our model
            # (drops e.g. cross-attn preprocessing entries that PTQ would
            # have emitted because the regex permits them but the kernel
            # block does not swap them).
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

    # Now register Phase 37+ preprocessing buffers on the matching Linears.
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
