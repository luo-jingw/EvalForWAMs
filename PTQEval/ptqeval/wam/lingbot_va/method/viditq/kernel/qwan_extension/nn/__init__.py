# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Kernel-backed quantized Linear modules for LingBot-VA WanTransformer."""
from qwan_extension.nn.base import QuantWanLinearBase
from qwan_extension.nn.qlinear_w4a4 import QuantWanLinearW4A4
from qwan_extension.nn.qlinear_w4a8 import QuantWanLinearW4A8
from qwan_extension.nn.qlinear_w8a8 import QuantWanLinearW8A8

__all__ = [
    "QuantWanLinearBase",
    "QuantWanLinearW4A4",
    "QuantWanLinearW4A8",
    "QuantWanLinearW8A8",
]
