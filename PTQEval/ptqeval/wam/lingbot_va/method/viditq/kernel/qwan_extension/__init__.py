# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""qwan_extension: BF16-native int8 GEMM kernels.

Re-exports the C++/CUDA entry points from the compiled extension so callers
can write `from qwan_extension import act_quant_bf16_with_sum,
w8a8_obf16_bias_weight_asym`.
"""
# Import torch first so libc10.so / libtorch.so are loaded into the
# process before the compiled extension dlopens against them.
import torch  # noqa: F401

from qwan_extension._C import (
    act_quant_bf16,
    act_quant_bf16_with_sum,
    w8a8_of16_bias_weight_asym,
    w8a8_obf16_bias_weight_asym,
    w8a8_obf16_bias_weight_sym,
    w8a8_obf16_nobias_weight_asym,
    w8a8_obf16_nobias_weight_sym,
)

__all__ = [
    "act_quant_bf16",
    "act_quant_bf16_with_sum",
    "w8a8_of16_bias_weight_asym",
    "w8a8_obf16_bias_weight_asym",
    "w8a8_obf16_bias_weight_sym",
    "w8a8_obf16_nobias_weight_asym",
    "w8a8_obf16_nobias_weight_sym",
]
