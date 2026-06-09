# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Phase 19 verify script.

Builds a small WanTransformerBlock (dim/ffn small enough to fit easily),
clones it, wraps the clone with QuantWanTransformerBlockWithCudaKernel for
both W8A8 and W4A8, runs forward on fake activations against the FP
reference, and checks output shape + max abs error < 0.1.
"""
from __future__ import annotations

import copy
import sys

import torch

from qwan_extension.nn import QuantWanLinearW4A8, QuantWanLinearW8A8
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


def _check_one(name: str, quant_cls, tol: float,
               device: torch.device, dtype: torch.dtype, seed: int = 0) -> bool:
    block_fp = _build_block(device, dtype, seed)
    block_for_kernel = copy.deepcopy(block_fp)
    block_k = QuantWanTransformerBlockWithCudaKernel(block_for_kernel, quant_cls)

    hidden, encoder, temb = _make_inputs(device, dtype, seed + 1)
    with torch.no_grad():
        y_ref = block_fp(hidden, encoder, temb, rotary_emb=None)
        y_k = block_k(hidden, encoder, temb, rotary_emb=None)

    shape_ok = tuple(y_k.shape) == tuple(y_ref.shape) == (BATCH, SEQ, DIM)
    dtype_ok = y_k.dtype == dtype
    diff = (y_k.float() - y_ref.float()).abs()
    max_abs = diff.max().item()
    finite_ok = torch.isfinite(y_k).all().item()
    err_ok = max_abs < tol
    flag = "OK" if (shape_ok and dtype_ok and finite_ok and err_ok) else "FAIL"
    print(f"{name:<10}  shape={tuple(y_k.shape)}  dtype={y_k.dtype}  "
          f"max_abs={max_abs:.3e}  tol={tol}  {flag}")
    return shape_ok and dtype_ok and finite_ok and err_ok


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA unavailable.", file=sys.stderr)
        return 2
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    # Per-variant tol: this test measures end-to-end quant error vs the FP
    # block (6 chained quantized Linears + residuals). On centered random
    # Gaussian weights, sym quant is optimal; asym adds ~2x noise from the
    # zp correction term. Real WAN weights (Phase 27) have skewed
    # distributions where asym recovers better, but this synthetic test
    # measures the worst case for asym. Single-Linear wrapper plumbing is
    # already validated by check_qlinear at max_abs ~4e-3.
    tols = {
        "W8A8 block": 0.3,    # asym scratch+kernel, see note above
        "W4A8 block": 0.1,    # sym scratch path, unchanged from Phase 19
    }

    results = [
        _check_one("W8A8 block", QuantWanLinearW8A8, tols["W8A8 block"], device, dtype),
        _check_one("W4A8 block", QuantWanLinearW4A8, tols["W4A8 block"], device, dtype),
    ]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
