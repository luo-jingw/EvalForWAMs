# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Block-level kernel-backed wrappers for WanTransformerBlock."""
from wan_va.quant.viditq.block.quant_wan_block import (
    QuantWanTransformerBlockWithCudaKernel,
)

__all__ = ["QuantWanTransformerBlockWithCudaKernel"]
