# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Common base for kernel-backed quantized Linear modules.

Subclasses (qlinear_w8a8, qlinear_w4a8) specify the weight bit width
and the GEMM kernel to call in forward. State that is common to both
(int_weight, scale_weight, bias) is registered as buffers here so
state_dict serialization is uniform. Each subclass registers its own
weight-zero-point representation in __init__ because the W8A8 and W4A8
kernel epilogues consume the zero-point in DIFFERENT shapes:

  W8A8: `zp_weight int16 [C_out]`         (kernel: `+ a_sum * zp_b * b_scale`)
  W4A8: `szeros_weight bf16 [C_out]`      (kernel: `- s_z * a_ssum`,
                                           s_z precomputed = scale_w * zp_unsigned)

state_dict keys common across both subclasses:
    int_weight   int8  [C_out, C_in_pack]   (C_in or C_in/2)
    scale_weight bf16  [C_out]
    bias         bf16  [C_out]               (omitted when has_bias=False)

Plus the subclass-specific zero-point buffer (see above).

Optional Part VI preprocessing buffers (default None -> no-op):
    quarot_sign      int8  [C_in]            (Phase 37 QuaRoT sign vector;
                                              triggers Hadamard rotation
                                              of x before quant)
    act_channel_div  bf16  [C_in]            (Phase 36 SmoothQuant
                                              channel_mask; triggers per-
                                              channel division of x
                                              before rotation/quant)
    act_scale_static bf16  [1]               (Phase 33 static activation
                                              scale; routes act quant to
                                              the static kernel variant
                                              which skips runtime amax)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

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
        # Subclass registers its own zero-point buffer (W8A8: zp_weight
        # int16, W4A8: szeros_weight bf16). See subclass __init__.
        if has_bias:
            self.register_buffer(
                "bias",
                torch.empty(out_features, dtype=torch.bfloat16),
                persistent=True,
            )
        else:
            # Register as None so state_dict does not contain a 'bias' key.
            self.bias = None

        # Part VI optional preprocessing tensors. Default: None -> forward
        # path skips them. Loader.install_preprocessing_buffer() registers
        # them as proper buffers after load_state_dict, ONLY for layers
        # whose ckpt actually contains the matching key. This keeps
        # state_dict size unchanged for Phase 27 baseline ckpts and avoids
        # the size-mismatch error that would arise from registering an
        # empty placeholder buffer (load_state_dict strict=False still
        # raises on shape mismatch, only ignores missing keys).
        self.act_channel_div: Optional[torch.Tensor] = None
        self.quarot_sign: Optional[torch.Tensor] = None
        self.act_scale_static: Optional[torch.Tensor] = None

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
        """x: bf16 with last-dim == in_features. Returns bf16 [..., out_features].

        Optional Part VI preprocessing runs in this order BEFORE quant:
          1. SmoothQuant: x <- x / act_channel_div  (per-channel division)
          2. QuaRoT:      x <- (x * quarot_sign) @ H / sqrt(C_in)
        Both no-op when the corresponding buffer is empty (Phase 27 ckpt).
        Order matches ViDiT-Q viditq_quant_layer.py:62-63 (smooth before
        rotation). M-padding for CTA_M alignment happens AFTER the
        preprocessing so the padded rows do not contaminate the
        preprocessing statistics.
        """
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features).contiguous()

        # Part VI preprocessing (no-op when buffers are absent).
        if self.act_channel_div is not None:
            x_2d = x_2d / self.act_channel_div
        if self.quarot_sign is not None:
            # Lazy import: quarot lives in ptqeval (the method package),
            # while base.py lives in qwan_extension (the kernel package).
            # Top-level import would create a cycle at module load time
            # because loader.py in ptqeval imports from qwan_extension.nn.
            from ptqeval.wam.lingbot_va.method.viditq.quarot import apply_input_rotation
            x_2d = apply_input_rotation(x_2d, self.quarot_sign).contiguous()

        M = x_2d.shape[0]
        M_pad = (M + self._CTA_M - 1) // self._CTA_M * self._CTA_M
        if M_pad != M:
            x_2d = torch.nn.functional.pad(x_2d, (0, 0, 0, M_pad - M))

        # Phase 33: static act quant when calibrated scale buffer is present;
        # else dynamic per-token amax. Both return the same (x_int8, scale_x,
        # sum_x) triple so the downstream W8A8 GEMM is unchanged.
        if self.act_scale_static is not None:
            from qwan_extension import act_quant_bf16_with_sum_static
            x_int8, scale_x_bf16, sum_x_bf16 = act_quant_bf16_with_sum_static(
                x_2d, self.act_scale_static
            )
        else:
            from qwan_extension import act_quant_bf16_with_sum  # local: avoids cycles
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
