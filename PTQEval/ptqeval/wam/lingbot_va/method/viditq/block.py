# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Block-level kernel integration (Phase 19 + Phase 42 per-Linear dispatch).

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

Phase 42: per-Linear class dispatch. The ctor accepts a
`quant_linear_cls_map: dict[suffix, cls]` keyed by the 6 target suffixes
("attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
"ffn.net.0.proj", "ffn.net.2"). Entries may map to different concrete
classes — required for the W4A4-MP variant which routes
{to_q, to_k, to_v, ffn.net.2} -> W4A4 and {to_out, ffn.net.0.proj} -> W8A8.
A missing suffix entry means "keep the original FP Linear in place"
(supports Phase 42's whole-block-FP case for blocks.0, although the
canonical handling for that is loader skipping the block wrap entirely).
Passing a SINGLE class (not a dict) is the Phase 19 / 26 homogeneous
shorthand kept for legacy single-bitwidth variants (W8A8, W4A8 pure).
"""
from __future__ import annotations

from typing import Union

import torch.nn as nn

from qwan_extension.nn.base import QuantWanLinearBase
from wan_va.modules.model import WanTransformerBlock


_TARGET_SUFFIXES: tuple[str, ...] = (
    "attn1.to_q",
    "attn1.to_k",
    "attn1.to_v",
    "attn1.to_out.0",
    "ffn.net.0.proj",
    "ffn.net.2",
)


class QuantWanTransformerBlockWithCudaKernel(WanTransformerBlock):

    def __init__(
        self,
        ref_block: WanTransformerBlock,
        quant_linear_cls_map: Union[
            type[QuantWanLinearBase],
            dict[str, type[QuantWanLinearBase]],
        ],
    ) -> None:
        assert isinstance(ref_block, WanTransformerBlock), (
            f"ref_block must be WanTransformerBlock, got {type(ref_block)}"
        )

        if isinstance(quant_linear_cls_map, type) and issubclass(
            quant_linear_cls_map, QuantWanLinearBase
        ):
            # Single-class shorthand: every target Linear gets the same wrapper.
            cls_map: dict[str, type[QuantWanLinearBase]] = {
                s: quant_linear_cls_map for s in _TARGET_SUFFIXES
            }
        elif isinstance(quant_linear_cls_map, dict):
            for s, c in quant_linear_cls_map.items():
                if s not in _TARGET_SUFFIXES:
                    raise ValueError(
                        f"unknown target suffix {s!r}; must be one of {_TARGET_SUFFIXES}"
                    )
                if not (isinstance(c, type) and issubclass(c, QuantWanLinearBase)):
                    raise ValueError(
                        f"quant_linear_cls_map[{s!r}] must subclass QuantWanLinearBase, "
                        f"got {c}"
                    )
            cls_map = dict(quant_linear_cls_map)
        else:
            raise TypeError(
                "quant_linear_cls_map must be a QuantWanLinearBase subclass "
                f"or dict[suffix, subclass]; got {type(quant_linear_cls_map)}"
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

        # Swap the target Linears in place. self.attn1 / self.ffn are the
        # same Python objects as ref_block.attn1 / ref_block.ffn, so this
        # also mutates ref_block; the caller should treat ref_block as
        # consumed. A suffix omitted from cls_map keeps the original FP
        # Linear (Phase 42 only — current callers always pass all 6).
        if "attn1.to_q" in cls_map:
            self.attn1.to_q = cls_map["attn1.to_q"].from_fp_linear(self.attn1.to_q)
        if "attn1.to_k" in cls_map:
            self.attn1.to_k = cls_map["attn1.to_k"].from_fp_linear(self.attn1.to_k)
        if "attn1.to_v" in cls_map:
            self.attn1.to_v = cls_map["attn1.to_v"].from_fp_linear(self.attn1.to_v)
        if "attn1.to_out.0" in cls_map:
            self.attn1.to_out[0] = cls_map["attn1.to_out.0"].from_fp_linear(
                self.attn1.to_out[0]
            )
        if "ffn.net.0.proj" in cls_map:
            self.ffn.net[0].proj = cls_map["ffn.net.0.proj"].from_fp_linear(
                self.ffn.net[0].proj
            )
        if "ffn.net.2" in cls_map:
            self.ffn.net[2] = cls_map["ffn.net.2"].from_fp_linear(self.ffn.net[2])

    # forward is inherited from WanTransformerBlock.
