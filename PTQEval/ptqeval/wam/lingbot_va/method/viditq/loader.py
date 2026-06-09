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
    4: QuantWanLinearW4A8,
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
    weight_bits = int(layer_cfg.weight_bits)
    # Phase 26a-2: W4A8 path is being rebuilt against the ViDiT-Q QServe
    # port; the old scratch W4A8 wrapper is incompatible with the
    # post-24d asym int_weights schema. Phase 28 restores W4A8.
    if weight_bits == 4:
        raise NotImplementedError(
            "W4A8 path is being rebuilt; not available between Phase 26a "
            "and Phase 28. Run a W8A8 config instead, or wait for "
            "Phase 28's w4a8_obf16_* launchers."
        )
    if weight_bits not in _WEIGHT_BITS_TO_CLS:
        raise ValueError(f"weight_bits must be 8 or 4, got {weight_bits}")
    quant_linear_cls = _WEIGHT_BITS_TO_CLS[weight_bits]

    logger.info(
        f"loading FP transformer from {wan_model_path} "
        f"(dtype={dtype}, device={device}, weight_bits={weight_bits})"
    )
    model = load_transformer(
        wan_model_path,
        torch_dtype=dtype,
        torch_device=device,
        attn_mode="torch",
    )
    model.eval()

    n_blocks = len(model.blocks)
    logger.info(f"wrapping {n_blocks} blocks with "
                f"QuantWanTransformerBlockWithCudaKernel ({quant_linear_cls.__name__})")
    for i in range(n_blocks):
        model.blocks[i] = QuantWanTransformerBlockWithCudaKernel(
            model.blocks[i], quant_linear_cls
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
    model_keys = set(model.state_dict().keys())
    filtered_sd = {k: v for k, v in raw_sd.items() if k in model_keys}
    skipped = len(raw_sd) - len(filtered_sd)

    _, unexpected = model.load_state_dict(filtered_sd, strict=False)
    if unexpected:
        raise RuntimeError(
            f"unexpected keys in int_weights state_dict after filtering: "
            f"{len(unexpected)} (first 5: {unexpected[:5]})"
        )
    logger.info(
        f"int_weights load: applied {len(filtered_sd)} tensors, "
        f"skipped {skipped} non-target quantized entries (e.g. cross-attn)."
    )

    model.to(device).eval().requires_grad_(False)
    torch.cuda.empty_cache()
    return model
