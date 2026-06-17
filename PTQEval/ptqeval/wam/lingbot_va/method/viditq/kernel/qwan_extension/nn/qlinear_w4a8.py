# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""W4A8 kernel-backed Linear (Phase 28 ViDiT-Q/QServe port).

Weight:     per-channel ASYMMETRIC INT4 unsigned (range [0, 15]), stored
            in QServe pre-permuted layout as int_weight [C_out, C_in/2]
            (2 nibbles per byte; layout from omniserve W4A8 from_linear
            -- 8-level reshape + permute + (high<<4)|low pack). Companion
            buffers: scale_weight bf16 [C_out] and szeros_weight bf16 [C_out]
            (= scale_weight * zp_unsigned, precomputed at PTQ time).
Activation: per-token symmetric INT8, computed online by
            qwan_extension.act_quant_bf16_with_sum (also emits sum_x =
            scale_x * sum_k(x_int8), consumed by the W4A8 epilogue term
            `- szeros_weight * sum_x`).

Epilogue (QServe; matches ViDiT-Q upstream
w4a8_per_channel_gemm_cuda_qserve.cu):
    y = scale_x * scale_w * (acc_int32) - szeros_w * sum_x

i.e. dequant convention `w_real = scale_w * (w_int_unsigned - zp_unsigned)`.
This differs from the W8A8 convention `w_real = scale_w * (w_int_signed +
zp_signed)` -- the SIGN flips. PTQ output must use the unsigned formula;
see ptq._per_channel_asym_quant_unsigned.

Bias: kernel has no bias variant by design (mirrors upstream); wrapper
adds it post-GEMM (small elementwise add, ~microseconds vs ~1-3 ms GEMM).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from qwan_extension import w4a8_obf16_nobias_weight_asym
from qwan_extension.nn.base import QuantWanLinearBase


class QuantWanLinearW4A8(QuantWanLinearBase):
    WEIGHT_BITS: int = 4

    def __init__(self, in_features: int, out_features: int, has_bias: bool) -> None:
        super().__init__(in_features, out_features, has_bias)
        # Phase 28: szeros_weight = scale_weight * zp_unsigned, precomputed
        # at PTQ time so the kernel epilogue can do a single packed-bf162
        # load + multiply per warp tile instead of expanding zp to fp32 per
        # output. bf16 [C_out].
        self.register_buffer(
            "szeros_weight",
            torch.empty(out_features, dtype=torch.bfloat16),
            persistent=True,
        )

    @staticmethod
    def _packed_in_features(in_features: int) -> int:
        # 2 nibbles per int8 byte.
        return in_features // 2

    def _gemm(
        self,
        x_int8: torch.Tensor,
        scale_x_bf16: torch.Tensor,
        sum_x_bf16: torch.Tensor,
    ) -> torch.Tensor:
        y = w4a8_obf16_nobias_weight_asym(
            x_int8,
            self.int_weight,
            scale_x_bf16,
            self.scale_weight,
            sum_x_bf16,
            self.szeros_weight,
        )
        if self.has_bias:
            y = y + self.bias
        return y

    @classmethod
    def from_fp_linear(cls, fp: nn.Linear) -> "QuantWanLinearW4A8":
        """Build a W4A8 kernel module from a torch.nn.Linear via
        per-channel UNSIGNED int4 asymmetric quant + QServe weight pack.

        Quant formula (matches omniserve W4A8OF16LinearDynamicInputScale
        .from_linear per-channel branch + the kernel epilogue convention):
            scale_w  = (W.amax(1) - W.amin(1)) / 15
            zp_uns   = round(-W.amin(1) / scale_w).clamp(0, 15)        # uint8-valued
            W_int    = (round(W / scale_w) + zp_uns).clamp(0, 15)      # uint8
            szeros_w = scale_w * zp_uns                                # bf16, post-multiplied
        The loader.load_state_dict overwrites these buffers with the
        offline PTQ output; from_fp_linear is the construction-time
        placeholder so a from_fp_linear-only path also matches the
        kernel epilogue. Both implementations must agree.
        """
        from ptqeval.wam.lingbot_va.method.viditq.ptq import (
            _per_channel_asym_quant_unsigned,
            _pack_int4_qserve,
        )

        has_bias = fp.bias is not None
        mod = cls(fp.in_features, fp.out_features, has_bias=has_bias)

        w_f32 = fp.weight.detach().to(torch.float32)
        int_w, scale_w, zp_uns = _per_channel_asym_quant_unsigned(w_f32, n_bits=4)
        packed = _pack_int4_qserve(int_w)
        szeros = (scale_w.to(torch.float32) * zp_uns.to(torch.float32)).to(torch.bfloat16)

        device = fp.weight.device
        mod.int_weight = packed.to(device).contiguous()
        mod.scale_weight = scale_w.to(device).contiguous()
        mod.szeros_weight = szeros.to(device).contiguous()
        if has_bias:
            mod.bias = fp.bias.detach().to(torch.bfloat16).to(device).contiguous()
        return mod
