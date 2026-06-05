# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""ViDiT-Q variant loader (kernel-only path).

Implements the Section 15 loader protocol:

    load_quant_model(wan_model_path, variant_args, device, dtype) -> nn.Module

Body is intentionally a stub until Phase 20. Phase 14 only removes the old
algorithm-simulation imports (qdiff, QuantizedLinear) so the rest of the
codebase has no implicit dependency on the deleted modules.

variant_args schema (yaml form, parsed by OmegaConf in the server then handed
in as a plain dict):

    layer_config:        path to a configs/w8a8.yaml or configs/w4a8.yaml
    int_weights_ckpt:    path to the PTQ-produced int_weights state_dict
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def load_quant_model(
    wan_model_path: str,
    variant_args: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    raise NotImplementedError(
        "wan_va.quant.viditq.loader.load_quant_model is a Phase 14 stub. "
        "The kernel-only implementation is delivered in Phase 20."
    )
