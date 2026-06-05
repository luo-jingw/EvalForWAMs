# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""W4A8 kernel-backed Linear.

Weight: per-channel symmetric INT4 (qmax=7), packed two nibbles per int8
byte (low nibble -> col 2c, high nibble -> col 2c+1, both signed).
Activation: per-token symmetric INT8 (qmax=127), computed online by
qwan_extension.act_quant_bf16.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from qwan_extension import w4a8_gemm_bf16
from qwan_extension.nn.base import QuantWanLinearBase


def _pack_int4_dense_to_bytes(w_dense: torch.Tensor) -> torch.Tensor:
    """w_dense: [M, K] int8 with values in [-8, 7], K even.
    Returns packed [M, K/2] int8: byte[c] = (high<<4) | (low & 0xF) where
    low = w_dense[:, 2c], high = w_dense[:, 2c+1]."""
    assert w_dense.dim() == 2
    assert w_dense.dtype == torch.int8
    M, K = w_dense.shape
    assert K % 2 == 0, f"K must be even for int4 packing, got {K}"
    w32 = w_dense.to(torch.int32)
    low = w32[:, 0::2] & 0xF
    high = w32[:, 1::2] & 0xF
    packed_u = ((high << 4) | low) & 0xFF
    return packed_u.to(torch.uint8).view(torch.int8).contiguous()


class QuantWanLinearW4A8(QuantWanLinearBase):
    WEIGHT_BITS: int = 4

    @staticmethod
    def _packed_in_features(in_features: int) -> int:
        assert in_features % 2 == 0, (
            f"W4A8 requires even in_features for nibble packing, got {in_features}"
        )
        return in_features // 2

    def _gemm(
        self,
        x_int8: torch.Tensor,
        scale_x_bf16: torch.Tensor,
    ) -> torch.Tensor:
        return w4a8_gemm_bf16(
            x_int8,
            scale_x_bf16,
            self.int_weight,
            self.scale_weight,
            self.bias,
        )

    @classmethod
    def from_fp_linear(cls, fp: nn.Linear) -> "QuantWanLinearW4A8":
        """Build a W4A8 kernel module by per-channel symmetric INT4 quantizing
        the weight of an existing torch.nn.Linear, then packing 2 nibbles per
        byte. Bias is copied as bf16."""
        has_bias = fp.bias is not None
        mod = cls(fp.in_features, fp.out_features, has_bias=has_bias)

        w_f32 = fp.weight.detach().to(torch.float32)
        row_max = w_f32.abs().amax(dim=1).clamp_min(1e-8)
        scale = row_max / 7.0
        scale_bf16 = scale.to(torch.bfloat16)
        scale_eff = scale_bf16.to(torch.float32)
        w_dense = torch.round(w_f32 / scale_eff.unsqueeze(1)).clamp(-8, 7).to(torch.int8)
        w_packed = _pack_int4_dense_to_bytes(w_dense)

        device = fp.weight.device
        mod.int_weight = w_packed.to(device).contiguous()
        mod.scale_weight = scale_bf16.to(device).contiguous()
        if has_bias:
            mod.bias = fp.bias.detach().to(torch.bfloat16).to(device).contiguous()
        return mod
