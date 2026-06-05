# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Common base for kernel-backed quantized Linear modules.

Subclasses (qlinear_w8a8, qlinear_w4a8) specify the weight bit width and
the GEMM kernel to call in forward. State (int_weight, scale_weight, bias)
is registered as buffers here so state_dict serialization is uniform.

state_dict keys (per spec, Section 10.5):
    int_weight   int8  [C_out, C_in_pack]   (C_in or C_in/2)
    scale_weight bf16  [C_out]
    bias         bf16  [C_out]               (omitted when has_bias=False)
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class QuantWanLinearBase(nn.Module, ABC):
    """Per-channel symmetric weight + per-token symmetric activation Linear,
    backed by qwan_extension CUDA kernels.

    Subclasses must define:
      WEIGHT_BITS:        class attribute, 8 or 4.
      _packed_in_features: shape of the in-feature axis after packing.
      _gemm:              kernel forward dispatch.
      from_fp_linear:     constructor from a torch.nn.Linear instance.
    """

    WEIGHT_BITS: int

    def __init__(self, in_features: int, out_features: int, has_bias: bool) -> None:
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.has_bias: bool = has_bias

        c_in_pack = self._packed_in_features(in_features)
        self.register_buffer(
            "int_weight",
            torch.empty(out_features, c_in_pack, dtype=torch.int8),
            persistent=True,
        )
        self.register_buffer(
            "scale_weight",
            torch.empty(out_features, dtype=torch.bfloat16),
            persistent=True,
        )
        if has_bias:
            self.register_buffer(
                "bias",
                torch.empty(out_features, dtype=torch.bfloat16),
                persistent=True,
            )
        else:
            # Register as None so state_dict does not contain a 'bias' key.
            self.bias = None

    @staticmethod
    @abstractmethod
    def _packed_in_features(in_features: int) -> int: ...

    @abstractmethod
    def _gemm(
        self,
        x_int8: torch.Tensor,
        scale_x_bf16: torch.Tensor,
    ) -> torch.Tensor: ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: bf16 with last-dim == in_features. Returns bf16 [..., out_features]."""
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features).contiguous()

        from qwan_extension import act_quant_bf16  # local import: avoids cycles
        x_int8, scale_x_bf16 = act_quant_bf16(x_2d)
        y_2d = self._gemm(x_int8, scale_x_bf16)
        return y_2d.reshape(*orig_shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.has_bias}, weight_bits={self.WEIGHT_BITS}"
        )
