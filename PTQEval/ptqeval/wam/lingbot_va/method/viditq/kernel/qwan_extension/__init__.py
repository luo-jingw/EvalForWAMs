# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""qwan_extension: BF16-native quantized GEMM kernels.

Re-exports the C++/CUDA entry points from the compiled extension so callers
can write `from qwan_extension import act_quant_bf16, w8a8_gemm_bf16`.
"""
# Import torch first so libc10.so / libtorch.so are loaded into the
# process before the compiled extension dlopens against them.
import torch  # noqa: F401

from qwan_extension._C import (
    act_quant_bf16,
    act_quant_bf16_with_sum,
    w8a8_gemm_bf16,
    w4a8_gemm_bf16,
    quant_sum,
    quant_sum_bf16,
)

__all__ = [
    "act_quant_bf16",
    "act_quant_bf16_with_sum",
    "w8a8_gemm_bf16",
    "w4a8_gemm_bf16",
    "quant_sum",
    "quant_sum_bf16",
]
