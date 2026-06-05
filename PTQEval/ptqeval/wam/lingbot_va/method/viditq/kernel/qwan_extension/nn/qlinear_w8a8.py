# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""W8A8 kernel-backed Linear.

Weight: per-channel symmetric INT8 (qmax=127), stored dense as int_weight
[C_out, C_in].
Activation: per-token symmetric INT8 (qmax=127), computed online by
qwan_extension.act_quant_bf16.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from qwan_extension import w8a8_gemm_bf16
from qwan_extension.nn.base import QuantWanLinearBase


class QuantWanLinearW8A8(QuantWanLinearBase):
    WEIGHT_BITS: int = 8

    @staticmethod
    def _packed_in_features(in_features: int) -> int:
        return in_features

    def _gemm(
        self,
        x_int8: torch.Tensor,
        scale_x_bf16: torch.Tensor,
    ) -> torch.Tensor:
        return w8a8_gemm_bf16(
            x_int8,
            scale_x_bf16,
            self.int_weight,
            self.scale_weight,
            self.bias,
        )

    @classmethod
    def from_fp_linear(cls, fp: nn.Linear) -> "QuantWanLinearW8A8":
        """Build a W8A8 kernel module by per-channel symmetric INT8 quantizing
        the weight of an existing torch.nn.Linear. Bias is copied as bf16.
        Weight is placed on the same device as fp."""
        has_bias = fp.bias is not None
        mod = cls(fp.in_features, fp.out_features, has_bias=has_bias)

        w_f32 = fp.weight.detach().to(torch.float32)
        row_max = w_f32.abs().amax(dim=1).clamp_min(1e-8)
        scale = row_max / 127.0
        scale_bf16 = scale.to(torch.bfloat16)
        scale_eff = scale_bf16.to(torch.float32)
        w_int = torch.round(w_f32 / scale_eff.unsqueeze(1)).clamp(-127, 127).to(torch.int8)

        device = fp.weight.device
        mod.int_weight = w_int.to(device).contiguous()
        mod.scale_weight = scale_bf16.to(device).contiguous()
        if has_bias:
            mod.bias = fp.bias.detach().to(torch.bfloat16).to(device).contiguous()
        return mod
