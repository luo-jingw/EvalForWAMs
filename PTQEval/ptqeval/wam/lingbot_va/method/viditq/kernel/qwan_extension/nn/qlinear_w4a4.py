# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""W4A4 kernel-backed Linear (Phase 42 ViDiT-Q Atom port).

Weight:     per-group SYMMETRIC INT4 (group=128 along K). Stored as
            int_weight uint8 [C_out, C_in/2] (2 nibbles per byte, signed
            two's complement) + scale_weight bf16 [K/128, C_out] held in
            Atom-permuted layout (= natural [C_out, K/128] transposed and
            made contiguous — see csrc/w4a4/scale_layout.cu). No zero-point
            buffer (kernel is sym-only per plan G5).
Activation: per-token per-group symmetric INT4 (group=128 along K),
            computed online by qwan_extension.act_quant_bf16_group128 + the
            output natural [N_tokens, K/128] scale packed via
            pack_atom_scale_a_bf16 to match the kernel's A_scale layout.

State_dict schema (KERNEL-AGNOSTIC):
    int_weight     uint8 [C_out, C_in/2]
    scale_weight   bf16  [C_out, C_in/128]   ← NATURAL layout on disk
    bias           bf16  [C_out]              (optional)
    quarot_sign    int8  [C_in]               (optional, Phase 37/42)
The wrapper overrides _load_from_state_dict to pack scale_weight at load
time, so the live buffer is the [K/128, C_out] Atom-permuted form and
runtime memory holds only ONE copy of the scale tensor.

Bias: kernel-fused (Phase 42 G3). The dequant epilogue does register-FFMA
psums += bias, eliminating the per-call Python post-add that the original
ViDiT-Q Atom kernel left out.

Preprocessing: SmoothQuant is NOT used by the W4A4-MP yaml (paper Sec D.4
omits the `smooth_quant` / `viditq` blocks); the base-class
`act_channel_div` buffer therefore stays None. QuaRoT is used on 5-of-6
target Linears per block (to_out excluded); buffer wiring matches the
W8A8 / W4A8 wrappers via the inherited optional `quarot_sign`.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from qwan_extension import (
    act_quant_bf16_group128,
    pack_atom_scale_a_bf16,
    pack_atom_scale_b_bf16,
    w4a4_obf16_bias_weight_sym,
    w4a4_obf16_nobias_weight_sym,
)
from qwan_extension.nn.base import QuantWanLinearBase


_GROUP_SIZE = 128


class QuantWanLinearW4A4(QuantWanLinearBase):
    WEIGHT_BITS: int = 4
    ACT_BITS: int = 4

    def __init__(self, in_features: int, out_features: int, has_bias: bool) -> None:
        if in_features % _GROUP_SIZE != 0:
            raise ValueError(
                f"W4A4 expects in_features % {_GROUP_SIZE} == 0; got {in_features}"
            )
        super().__init__(in_features, out_features, has_bias)
        # The base class registered int_weight as int8 [C_out, C_in/2] and
        # scale_weight as bf16 [C_out]. Atom W4A4 needs:
        #   int_weight   uint8 [C_out, C_in/2] (TORCH_CHECK kUInt8 in kernel)
        #   scale_weight bf16  [K/128, C_out]  (Atom-permuted layout)
        # Re-register both. register_buffer overwrites prior entries when
        # called with an existing name.
        n_groups = in_features // _GROUP_SIZE
        self.register_buffer(
            "int_weight",
            torch.empty(out_features, in_features // 2, dtype=torch.uint8),
            persistent=True,
        )
        self.register_buffer(
            "scale_weight",
            torch.empty(n_groups, out_features, dtype=torch.bfloat16),
            persistent=True,
        )

    @staticmethod
    def _packed_in_features(in_features: int) -> int:
        return in_features // 2

    def _gemm(self, *args, **kwargs):
        # The base class's _gemm contract assumes the W8/W4A8 ladder
        # (x_int8, scale_x, sum_x). W4A4 takes (packed_int4, packed_scale)
        # with no sum_x, so this subclass overrides forward() directly and
        # _gemm is unused. Implement-required by ABC; raises if hit.
        raise NotImplementedError(
            "QuantWanLinearW4A4 overrides forward(); _gemm is not used."
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Pack natural scale_weight [C_out, K/128] -> Atom layout
        [K/128, C_out] before the standard buffer-copy logic. Keeps the
        .pth schema kernel-agnostic while runtime memory holds only the
        kernel-ready packed form (no natural copy retained).
        """
        key = prefix + "scale_weight"
        natural_shape = (self.out_features, self.in_features // _GROUP_SIZE)
        if key in state_dict:
            t = state_dict[key]
            if tuple(t.shape) == natural_shape:
                t_bf16 = t.to(torch.bfloat16)
                if t_bf16.device.type != "cuda":
                    t_bf16 = t_bf16.cuda()
                state_dict[key] = pack_atom_scale_b_bf16(t_bf16)
            # else: already packed (e.g. mid-pipeline reload); let copy_ run.
        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: bf16 [..., in_features]. Returns bf16 [..., out_features].

        Pipeline:
          1. optional QuaRoT input rotation (matches PTQ-time weight rot)
          2. zero-pad M to CTA_M=128
          3. act_quant_bf16_group128 -> (packed INT4 [M, K/2],
                                          scale_x_natural [M, K/128] bf16)
          4. pack_atom_scale_a_bf16(scale_x_natural) -> Atom A_scale layout
          5. w4a4 GEMM (bias / nobias) with bias-fused epilogue
          6. slice back to original M, reshape to original batch dims
        """
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features).contiguous()

        if self.act_channel_div is not None:
            # W4A4-MP yaml omits SmoothQuant, but the buffer hook is kept
            # for parity with W8A8 / W4A8 in case a future config opts in.
            x_2d = x_2d / self.act_channel_div
        if self.quarot_sign is not None:
            from ptqeval.wam.lingbot_va.method.viditq.quarot import apply_input_rotation
            x_2d = apply_input_rotation(x_2d, self.quarot_sign).contiguous()

        M = x_2d.shape[0]
        M_pad = (M + self._CTA_M - 1) // self._CTA_M * self._CTA_M
        if M_pad != M:
            x_2d = torch.nn.functional.pad(x_2d, (0, 0, 0, M_pad - M))

        x_int4_packed, scale_x_natural = act_quant_bf16_group128(x_2d)
        scale_x_packed = pack_atom_scale_a_bf16(scale_x_natural.contiguous())

        if self.has_bias:
            y_2d = w4a4_obf16_bias_weight_sym(
                x_int4_packed,
                self.int_weight,
                self.bias,
                scale_x_packed,
                self.scale_weight,
            )
        else:
            y_2d = w4a4_obf16_nobias_weight_sym(
                x_int4_packed,
                self.int_weight,
                scale_x_packed,
                self.scale_weight,
            )

        if M_pad != M:
            y_2d = y_2d[:M]
        return y_2d.reshape(*orig_shape[:-1], self.out_features)

    @classmethod
    def from_fp_linear(cls, fp: nn.Linear) -> "QuantWanLinearW4A4":
        """Build a W4A4 kernel module from an existing torch.nn.Linear via
        per-group symmetric INT4 weight quant + Atom B-scale pack.

        Quant formula matches ptq._per_group_sym_quant_w4a4 (group_size=128
        along K, signed [-8, 7] nibble pack). Used by check scripts /
        ad-hoc tests; production load goes through state_dict + the
        _load_from_state_dict hook above.
        """
        from ptqeval.wam.lingbot_va.method.viditq.ptq import (
            _per_group_sym_quant_w4a4,
        )

        has_bias = fp.bias is not None
        mod = cls(fp.in_features, fp.out_features, has_bias=has_bias)

        w_f32 = fp.weight.detach().to(torch.float32)
        packed_w, scale_w_natural = _per_group_sym_quant_w4a4(
            w_f32, group_size=_GROUP_SIZE
        )

        device = fp.weight.device
        if device.type != "cuda":
            # The pack kernel requires a CUDA input; lift everything once.
            device = torch.device("cuda", torch.cuda.current_device())
            mod.to(device)
        scale_w_packed = pack_atom_scale_b_bf16(
            scale_w_natural.to(device, torch.bfloat16)
        )

        mod.int_weight = packed_w.to(device).contiguous()
        mod.scale_weight = scale_w_packed.contiguous()
        if has_bias:
            mod.bias = fp.bias.detach().to(torch.bfloat16).to(device).contiguous()
        return mod
