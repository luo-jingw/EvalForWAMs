# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Phase 19: block-level kernel integration.

QuantWanTransformerBlockWithCudaKernel is a subclass of WanTransformerBlock
that adopts a reference block's submodules and swaps the 6 target Linears
for kernel-backed quantized variants. All other submodules (norms,
cross-attn, scale_shift_table) are reused as-is.

Structure matches WanTransformerBlock exactly so:
  - model.blocks[i].attn1 / attn2 / ffn are directly accessible (used by
    WanTransformer3DModel.clear_cache, create_empty_cache, etc).
  - state_dict keys mirror the FP block layout (no extra prefix), so
    ptq.py's int_weights .pth loads in place via model.load_state_dict.
  - forward is inherited from WanTransformerBlock; no delegation hop.

Targets:
  self-attn (attn1):  to_q, to_k, to_v, to_out[0]
  feed-forward (ffn): net[0].proj (up), net[2] (down)
Untouched:
  cross-attn (attn2): all 4 Linears stay FP.
  Norms (FP32LayerNorm), RMSNorm in WanAttention, rope, scale_shift_table.
"""
from __future__ import annotations

import torch.nn as nn

from qwan_extension.nn.base import QuantWanLinearBase
from wan_va.modules.model import WanTransformerBlock


class QuantWanTransformerBlockWithCudaKernel(WanTransformerBlock):

    def __init__(
        self,
        ref_block: WanTransformerBlock,
        quant_linear_cls: type[QuantWanLinearBase],
    ) -> None:
        assert isinstance(ref_block, WanTransformerBlock), (
            f"ref_block must be WanTransformerBlock, got {type(ref_block)}"
        )
        assert issubclass(quant_linear_cls, QuantWanLinearBase), (
            f"quant_linear_cls must subclass QuantWanLinearBase, got {quant_linear_cls}"
        )

        # Bypass WanTransformerBlock.__init__ (which would build a fresh
        # block from scratch). Initialize nn.Module's own bookkeeping then
        # adopt every submodule from the reference.
        nn.Module.__init__(self)
        self.attn_mode = ref_block.attn_mode
        self.norm1 = ref_block.norm1
        self.attn1 = ref_block.attn1
        self.attn2 = ref_block.attn2
        self.norm2 = ref_block.norm2
        self.ffn = ref_block.ffn
        self.norm3 = ref_block.norm3
        self.scale_shift_table = ref_block.scale_shift_table

        # Swap the 6 target Linears in place. self.attn1 / self.ffn are
        # the same Python objects as ref_block.attn1 / ref_block.ffn, so
        # this also mutates ref_block; the caller should treat ref_block
        # as consumed.
        self.attn1.to_q = quant_linear_cls.from_fp_linear(self.attn1.to_q)
        self.attn1.to_k = quant_linear_cls.from_fp_linear(self.attn1.to_k)
        self.attn1.to_v = quant_linear_cls.from_fp_linear(self.attn1.to_v)
        self.attn1.to_out[0] = quant_linear_cls.from_fp_linear(self.attn1.to_out[0])
        self.ffn.net[0].proj = quant_linear_cls.from_fp_linear(self.ffn.net[0].proj)
        self.ffn.net[2] = quant_linear_cls.from_fp_linear(self.ffn.net[2])

    # forward is inherited from WanTransformerBlock.
