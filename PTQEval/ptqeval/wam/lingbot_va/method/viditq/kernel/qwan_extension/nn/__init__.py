# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Kernel-backed quantized Linear modules for LingBot-VA WanTransformer."""
from qwan_extension.nn.base import QuantWanLinearBase
from qwan_extension.nn.qlinear_w4a8 import QuantWanLinearW4A8
from qwan_extension.nn.qlinear_w8a8 import QuantWanLinearW8A8

__all__ = [
    "QuantWanLinearBase",
    "QuantWanLinearW8A8",
    "QuantWanLinearW4A8",
]
