# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""W8A8 kernel-backed Linear (Phase 26a-2 asym wiring).

Weight:     per-channel ASYMMETRIC INT8, stored dense as int_weight
            [C_out, C_in] + scale_weight [C_out] + zp_weight [C_out].
Activation: per-token symmetric INT8, computed online by
            qwan_extension.act_quant_bf16_with_sum (which also emits
            sum_x needed by the asym epilogue).

Dispatch: forward uses w8a8_obf16_bias_weight_asym when bias is present,
w8a8_obf16_nobias_weight_asym otherwise. Both are Phase 25 bf16
instantiations of the ViDiT-Q W8A8 GEMM (verbatim port + OutT template).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from qwan_extension import (
    w8a8_obf16_bias_weight_asym,
    w8a8_obf16_nobias_weight_asym,
)
from qwan_extension.nn.base import QuantWanLinearBase


class QuantWanLinearW8A8(QuantWanLinearBase):
    WEIGHT_BITS: int = 8

    def __init__(self, in_features: int, out_features: int, has_bias: bool) -> None:
        super().__init__(in_features, out_features, has_bias)
        # Phase 26a-2: per-channel asymmetric weight quant; zp_weight is the
        # short2-aligned zero point consumed by the W8A8 kernel's asym
        # epilogue term `psums += a_sum * zp_b * b_scale`.
        self.register_buffer(
            "zp_weight",
            torch.empty(out_features, dtype=torch.int16),
            persistent=True,
        )

    @staticmethod
    def _packed_in_features(in_features: int) -> int:
        return in_features

    def _gemm(
        self,
        x_int8: torch.Tensor,
        scale_x_bf16: torch.Tensor,
        sum_x_bf16: torch.Tensor,
    ) -> torch.Tensor:
        if self.has_bias:
            return w8a8_obf16_bias_weight_asym(
                x_int8,
                self.int_weight,
                self.bias,
                scale_x_bf16,
                self.scale_weight,
                sum_x_bf16,
                self.zp_weight,
            )
        return w8a8_obf16_nobias_weight_asym(
            x_int8,
            self.int_weight,
            scale_x_bf16,
            self.scale_weight,
            sum_x_bf16,
            self.zp_weight,
        )

    @classmethod
    def from_fp_linear(cls, fp: nn.Linear) -> "QuantWanLinearW8A8":
        """Build a W8A8 kernel module from an existing torch.nn.Linear via
        asymmetric per-channel quantization. Bias is copied as bf16. Weight,
        scale, zp are placed on the same device as fp.

        Formula matches Phase 24d ptq._per_channel_asym_quant (n_bits=8;
        kept inline to avoid a cross-package import cycle into ptq.py).
        The loader.load_state_dict overwrites these buffers with the
        offline PTQ output; from_fp_linear's quantization is the
        construction-time placeholder. Both formulas must agree so a
        from_fp_linear-only path (e.g. check_qlinear) also matches the
        kernel epilogue convention w_real = scale * (w_int + zp).
        """
        has_bias = fp.bias is not None
        mod = cls(fp.in_features, fp.out_features, has_bias=has_bias)

        w_f32 = fp.weight.detach().to(torch.float32)
        n_levels = 256
        x_max = w_f32.amax(dim=1).clamp_min(0.0)
        x_min = w_f32.amin(dim=1).clamp_max(0.0)
        delta = ((x_max - x_min) / (n_levels - 1)).clamp_min(1e-8)
        zero_point = torch.round(x_min / delta) + (n_levels / 2)
        scale_bf16 = delta.to(torch.bfloat16)
        scale_eff = scale_bf16.to(torch.float32)
        zp_int16 = zero_point.to(torch.int16)
        w_int = (
            torch.round(w_f32 / scale_eff.unsqueeze(1))
            - zero_point.unsqueeze(1)
        ).clamp(-128, 127).to(torch.int8)

        device = fp.weight.device
        mod.int_weight = w_int.to(device).contiguous()
        mod.scale_weight = scale_bf16.to(device).contiguous()
        mod.zp_weight = zp_int16.to(device).contiguous()
        if has_bias:
            mod.bias = fp.bias.detach().to(torch.bfloat16).to(device).contiguous()
        return mod
