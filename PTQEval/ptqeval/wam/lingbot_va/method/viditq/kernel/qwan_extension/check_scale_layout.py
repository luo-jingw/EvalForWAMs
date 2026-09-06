# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Phase 42 step 6a: observational check for Atom W4A4 scale layout packers.

Emits metrics only; no assert / no PASS judgement (principle.txt L12). Inspect
the printed tables and decide pass/fail.

Run:
    cd PTQEval/ptqeval/wam/lingbot_va/method/viditq/kernel
    python -m qwan_extension.check_scale_layout

Sections:
  [A]  pack_atom_scale_a — kernel output vs pure-PyTorch reference using the
       same address formula derived from loadScaleReg (w4a4_gemm.cu L336).
       Shapes span single-block (M=128) up to multi-block multi-K (M=512,
       K=14336) for both fp16 and bf16.
  [As] sentinel — fill natural[m, g] with a unique scalar identifier,
       then read back the 4 replica slots predicted by the formula; print
       the values at those slots to verify replication semantics.
  [B]  pack_atom_scale_b — kernel output vs natural.transpose(0,1).contiguous().
"""
from __future__ import annotations

import torch

from qwan_extension import (
    pack_atom_scale_a_bf16,
    pack_atom_scale_a_fp16,
    pack_atom_scale_b_bf16,
    pack_atom_scale_b_fp16,
)


BLOCK_M = 128
GROUP_SIZE = 128


def _ref_pack_atom_scale_a(natural: torch.Tensor) -> torch.Tensor:
    """Pure-PyTorch reference. natural [M, G] -> packed [G, 4*M] same dtype."""
    M, G = natural.shape
    assert M % BLOCK_M == 0
    out = torch.zeros(G, 4 * M, dtype=natural.dtype, device=natural.device)
    m_idx = torch.arange(M, device=natural.device)
    block_m = m_idx // BLOCK_M
    local = m_idx % BLOCK_M
    wj = local // 64
    tile_i = (local % 64) // 16
    rir = local % 16
    is_low = rir // 8
    r = rir % 8
    base_b16 = block_m * 512 + wj * 256 + tile_i * 64 + r * 8
    # Build [M, 4] target column indices.
    even = torch.tensor([0, 2, 4, 6], device=natural.device)
    odd = torch.tensor([1, 3, 5, 7], device=natural.device)
    cols = torch.where(is_low.unsqueeze(1).bool(),
                       odd.unsqueeze(0).expand(M, 4),
                       even.unsqueeze(0).expand(M, 4))
    flat = base_b16.unsqueeze(1) + cols                            # [M, 4]
    flat_b = flat.unsqueeze(0).expand(G, M, 4).reshape(G, M * 4)   # [G, M*4]
    src = natural.transpose(0, 1).unsqueeze(-1).expand(G, M, 4).reshape(G, M * 4)
    out.scatter_(1, flat_b, src)
    return out


def _bytes_preview(t: torch.Tensor, n: int = 32) -> str:
    flat = t.view(torch.uint8).flatten()[:n].cpu().tolist()
    return " ".join(f"{b:02x}" for b in flat)


def _section_a():
    print("=" * 78)
    print("[A] pack_atom_scale_a — kernel vs PyTorch reference")
    print("=" * 78)
    print(f"{'M':>6} {'K':>6} {'G':>4} {'dtype':>6} {'shape_match':>11} "
          f"{'max_abs':>10} {'numel':>10}")
    cases = [
        (128, 3072, torch.bfloat16), (128, 3072, torch.float16),
        (128, 14336, torch.bfloat16),
        (256, 3072, torch.bfloat16),
        (512, 14336, torch.bfloat16),
        (1024, 14336, torch.bfloat16),
    ]
    for M, K, dtype in cases:
        G = K // GROUP_SIZE
        natural = torch.randn(M, G, dtype=dtype, device="cuda") * 0.1
        if dtype is torch.bfloat16:
            pkernel = pack_atom_scale_a_bf16(natural)
        else:
            pkernel = pack_atom_scale_a_fp16(natural)
        pref = _ref_pack_atom_scale_a(natural)
        shape_match = tuple(pkernel.shape) == tuple(pref.shape)
        diff = (pkernel.float() - pref.float()).abs().max().item()
        print(f"{M:>6} {K:>6} {G:>4} {str(dtype).split('.')[1]:>6} "
              f"{str(shape_match):>11} {diff:10.4e} {pkernel.numel():>10}")


def _section_a_sentinel():
    print()
    print("=" * 78)
    print("[As] sentinel — verify each natural scalar appears 4x at predicted offsets")
    print("=" * 78)
    M, K = 128, 256
    G = K // GROUP_SIZE
    dtype = torch.bfloat16
    natural = torch.arange(M * G, dtype=torch.float32, device="cuda").reshape(M, G).to(dtype)
    packed = pack_atom_scale_a_bf16(natural)
    print(f"natural shape       : {tuple(natural.shape)}  dtype {dtype}")
    print(f"packed  shape       : {tuple(packed.shape)} (expect [G={G}, 4*M={4*M}])")
    print(f"natural[0, 0] (=  0): probed 4 replicas (g=0):  ", end="")
    # m=0: wj=0, tile_i=0, rir=0, is_low=0, r=0 -> base=0, cols [0,2,4,6]
    print([packed[0, c].float().item() for c in (0, 2, 4, 6)])
    print(f"natural[8, 0] (=  16): probed 4 replicas (g=0): ", end="")
    # m=8: rir=8, is_low=1, r=0 -> base=0, cols [1,3,5,7]
    print([packed[0, c].float().item() for c in (1, 3, 5, 7)])
    print(f"natural[16, 0] (= 32): probed 4 replicas (g=0): ", end="")
    # m=16: tile_i=1, rir=0, base=64, cols [64,66,68,70]
    print([packed[0, c].float().item() for c in (64, 66, 68, 70)])
    print(f"natural[64, 0] (=128): probed 4 replicas (g=0): ", end="")
    # m=64: wj=1, tile_i=0, base=256, cols [256,258,260,262]
    print([packed[0, c].float().item() for c in (256, 258, 260, 262)])
    print(f"natural[127, 0] (=254): probed 4 replicas (g=0):", end="")
    # m=127: wj=1, tile_i=3, rir=15, is_low=1, r=7, base=256+192+56=504, cols [505,507,509,511]
    print([packed[0, c].float().item() for c in (505, 507, 509, 511)])
    print(f"natural[0, 1] (=  1): probed 4 replicas (g=1):  ", end="")
    print([packed[1, c].float().item() for c in (0, 2, 4, 6)])
    print(f"raw bytes packed[0, :32]: {_bytes_preview(packed[0, :16], n=32)}")


def _section_b():
    print()
    print("=" * 78)
    print("[B] pack_atom_scale_b — kernel vs natural.transpose(0,1).contiguous()")
    print("=" * 78)
    print(f"{'N':>6} {'K':>6} {'G':>4} {'dtype':>6} {'shape_match':>11} "
          f"{'max_abs':>10} {'is_contig':>10}")
    cases = [
        (3072, 3072, torch.bfloat16), (3072, 3072, torch.float16),
        (3072, 14336, torch.bfloat16),
        (14336, 3072, torch.bfloat16),
    ]
    for N, K, dtype in cases:
        G = K // GROUP_SIZE
        natural = torch.randn(N, G, dtype=dtype, device="cuda") * 0.1
        if dtype is torch.bfloat16:
            pkernel = pack_atom_scale_b_bf16(natural)
        else:
            pkernel = pack_atom_scale_b_fp16(natural)
        pref = natural.transpose(0, 1).contiguous()
        shape_match = tuple(pkernel.shape) == tuple(pref.shape)
        diff = (pkernel.float() - pref.float()).abs().max().item()
        print(f"{N:>6} {K:>6} {G:>4} {str(dtype).split('.')[1]:>6} "
              f"{str(shape_match):>11} {diff:10.4e} {str(pkernel.is_contiguous()):>10}")


def main():
    print("Phase 42 step 6a observational check — Atom W4A4 scale layout packers")
    print(f"torch {torch.__version__}, device {torch.cuda.get_device_name(0)}")
    _section_a()
    _section_a_sentinel()
    _section_b()


if __name__ == "__main__":
    main()
