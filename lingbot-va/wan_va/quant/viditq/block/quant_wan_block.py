# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Phase 19: block-level kernel integration.

QuantWanTransformerBlockWithCudaKernel mutates a reference
WanTransformerBlock in place by swapping the 6 target Linears for
kernel-backed quantized variants, then delegates forward to the mutated
ref block.

Targets (per Section 9 / 10.6):
    self-attn (attn1):  to_q, to_k, to_v, to_out[0]
    feed-forward (ffn): net[0].proj (up), net[2] (down)
Untouched:
    cross-attn (attn2): all 4 Linears stay FP.
    Norms (FP32LayerNorm), RMSNorm in WanAttention, rope, scale_shift_table.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from qwan_extension.nn.base import QuantWanLinearBase
from wan_va.modules.model import WanTransformerBlock


class QuantWanTransformerBlockWithCudaKernel(nn.Module):

    def __init__(
        self,
        ref_block: WanTransformerBlock,
        quant_linear_cls: type[QuantWanLinearBase],
    ) -> None:
        super().__init__()
        assert isinstance(ref_block, WanTransformerBlock), (
            f"ref_block must be WanTransformerBlock, got {type(ref_block)}"
        )
        assert issubclass(quant_linear_cls, QuantWanLinearBase), (
            f"quant_linear_cls must subclass QuantWanLinearBase, got {quant_linear_cls}"
        )

        # 4 self-attention projections.
        ref_block.attn1.to_q = quant_linear_cls.from_fp_linear(ref_block.attn1.to_q)
        ref_block.attn1.to_k = quant_linear_cls.from_fp_linear(ref_block.attn1.to_k)
        ref_block.attn1.to_v = quant_linear_cls.from_fp_linear(ref_block.attn1.to_v)
        ref_block.attn1.to_out[0] = quant_linear_cls.from_fp_linear(
            ref_block.attn1.to_out[0]
        )

        # 2 feed-forward projections (diffusers FeedForward layout:
        #   net[0] is GEGLU/GELU wrapper exposing .proj; net[2] is the down Linear).
        ref_block.ffn.net[0].proj = quant_linear_cls.from_fp_linear(
            ref_block.ffn.net[0].proj
        )
        ref_block.ffn.net[2] = quant_linear_cls.from_fp_linear(ref_block.ffn.net[2])

        # Adopt the (now-mutated) reference block as our forward implementation.
        self._block: WanTransformerBlock = ref_block

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self._block(*args, **kwargs)
