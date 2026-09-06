# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Block-level OBSERVATIONAL check.

Per principle.txt L12: emits raw metrics only — no assert, no PASS/FAIL
judgement.  Builds a small WanTransformerBlock (dim/ffn small), clones,
wraps the clone with QuantWanTransformerBlockWithCudaKernel for W8A8,
runs forward on fake activations against the FP reference, and reports
shape / dtype / finite flag / max_abs / rel max-abs.  The user inspects
the table.
"""
from __future__ import annotations

import copy
import sys

import torch

from qwan_extension.nn import QuantWanLinearW8A8
from wan_va.modules.model import WanTransformerBlock

from ptqeval.wam.lingbot_va.method.viditq.block import (
    QuantWanTransformerBlockWithCudaKernel,
)


DIM = 128
FFN_DIM = 256
NUM_HEADS = 4
BATCH = 2
SEQ = 64
TEXT_SEQ = 16


def _build_block(device: torch.device, dtype: torch.dtype, seed: int) -> WanTransformerBlock:
    torch.manual_seed(seed)
    block = WanTransformerBlock(
        dim=DIM,
        ffn_dim=FFN_DIM,
        num_heads=NUM_HEADS,
        cross_attn_norm=True,
        attn_mode="torch",
    )
    return block.to(device).to(dtype)


def _make_inputs(device: torch.device, dtype: torch.dtype, seed: int):
    g = torch.Generator(device=device).manual_seed(seed)
    hidden = (torch.randn((BATCH, SEQ, DIM), device=device, generator=g) * 0.5).to(dtype)
    encoder = (torch.randn((BATCH, TEXT_SEQ, DIM), device=device, generator=g) * 0.5).to(dtype)
    # temb: scale_shift_table[None] has shape [1, 1, 6, dim]; temb is added,
    # so any broadcast-compatible shape works. Use [B, 1, 6, dim].
    temb = (torch.randn((BATCH, 1, 6, DIM), device=device, generator=g) * 0.1).to(torch.float32)
    return hidden, encoder, temb


def _shape_metrics(name: str, quant_cls,
                   device: torch.device, dtype: torch.dtype, seed: int = 0) -> None:
    block_fp = _build_block(device, dtype, seed)
    block_for_kernel = copy.deepcopy(block_fp)
    block_k = QuantWanTransformerBlockWithCudaKernel(block_for_kernel, quant_cls)

    hidden, encoder, temb = _make_inputs(device, dtype, seed + 1)
    with torch.no_grad():
        y_ref = block_fp(hidden, encoder, temb, rotary_emb=None)
        y_k = block_k(hidden, encoder, temb, rotary_emb=None)

    shape_match = tuple(y_k.shape) == tuple(y_ref.shape) == (BATCH, SEQ, DIM)
    dtype_match = y_k.dtype == dtype
    diff = (y_k.float() - y_ref.float()).abs()
    max_abs = diff.max().item()
    out_mag = y_ref.float().abs().max().item()
    rel = max_abs / max(out_mag, 1e-6)
    finite = bool(torch.isfinite(y_k).all().item())
    print(
        f"{name:<10}  shape_match={shape_match}  dtype_match={dtype_match}  "
        f"finite={finite}  max_abs={max_abs:.3e}  rel={rel:.3e}  "
        f"out_mag={out_mag:.3e}"
    )


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA unavailable.", file=sys.stderr)
        return 2
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    _shape_metrics("W8A8 block", QuantWanLinearW8A8, device, dtype)
    return 0


if __name__ == "__main__":
    sys.exit(main())
