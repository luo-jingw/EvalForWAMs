# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Common base for kernel-backed quantized Linear modules.

Subclasses (currently only qlinear_w8a8) specify the weight bit width
and the GEMM kernel to call in forward. State (int_weight, scale_weight,
zp_weight, bias) is registered as buffers here so state_dict serialization
is uniform. Phase 28 will add qlinear_w4a8 as a second subclass.

state_dict keys (Phase 26a-2, asym per-channel weight schema):
    int_weight   int8  [C_out, C_in_pack]   (C_in or C_in/2)
    scale_weight bf16  [C_out]
    zp_weight    int16 [C_out]
    bias         bf16  [C_out]               (omitted when has_bias=False)
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class QuantWanLinearBase(nn.Module, ABC):
    """Per-channel asymmetric weight + per-token symmetric activation Linear,
    backed by qwan_extension CUDA kernels.

    Subclasses must define:
      WEIGHT_BITS:        class attribute, 8 or 4.
      _packed_in_features: shape of the in-feature axis after packing.
      _gemm:              kernel forward dispatch (takes sum_x for asym).
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
        # Phase 26a-2: per-channel asymmetric weight quant; zp_weight is the
        # short2-aligned zero point consumed by the W8A8 kernel's asym
        # epilogue term `psums += a_sum * zp_b * b_scale`.
        self.register_buffer(
            "zp_weight",
            torch.empty(out_features, dtype=torch.int16),
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
        sum_x_bf16: torch.Tensor,
    ) -> torch.Tensor: ...

    # Phase 25 W8A8 GEMM tiles M in chunks of CTA_M = 128 (verbatim from
    # ViDiT-Q upstream). LingBot-VA activation num_tokens is not always
    # a multiple of 128, so the wrapper zero-pads the row axis before
    # the kernel call and slices the original M rows back. The padded
    # rows quantize to zero (after the 1e-8 scale clamp) and contribute
    # zero to the GEMM output, so the first M output rows are unaffected.
    _CTA_M: int = 128

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: bf16 with last-dim == in_features. Returns bf16 [..., out_features]."""
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features).contiguous()
        M = x_2d.shape[0]
        M_pad = (M + self._CTA_M - 1) // self._CTA_M * self._CTA_M
        if M_pad != M:
            x_2d = torch.nn.functional.pad(x_2d, (0, 0, 0, M_pad - M))

        from qwan_extension import act_quant_bf16_with_sum  # local import: avoids cycles
        x_int8, scale_x_bf16, sum_x_bf16 = act_quant_bf16_with_sum(x_2d)
        y_2d = self._gemm(x_int8, scale_x_bf16, sum_x_bf16)

        if M_pad != M:
            y_2d = y_2d[:M]
        return y_2d.reshape(*orig_shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.has_bias}, weight_bits={self.WEIGHT_BITS}"
        )
