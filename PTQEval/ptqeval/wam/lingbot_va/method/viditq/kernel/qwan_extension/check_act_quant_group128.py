# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Phase 42 step 2: OBSERVATIONAL check for act_quant_bf16_group128.

Per principle.txt L12: this script emits raw metrics only — no assert,
no PASS/FAIL judgement. The user inspects the metric table and decides
whether the launcher is correct relative to the torch fp32 reference.

For each shape (N, K):
  - Compares packed int4 byte stream against a torch fp32 reference.
  - Reports byte-level diff count + diff max (RNE rounding tie-breakers
    around half-step values can flip one nibble vs the kernel's
    __float2int_rn; small counts here are noise, not bugs).
  - Reports per-group scale absolute + relative diff (max).

Also emits dtype/shape sanity + scale value-range observation on one shape.
"""
from __future__ import annotations

import torch

from qwan_extension import act_quant_bf16_group128


GROUP = 128


def _reference_pack_and_scale(x_bf16: torch.Tensor):
    """Per-token per-group sym INT4 quant + pack (NATURAL layout)."""
    N, K = x_bf16.shape
    assert K % GROUP == 0
    Kg = K // GROUP
    x_f32 = x_bf16.float().reshape(N, Kg, GROUP)
    amax = x_f32.abs().amax(dim=-1)                        # [N, Kg]
    scale = amax / 7.0                                      # [N, Kg]
    inv_scale = torch.where(amax > 0, 7.0 / amax,
                            torch.zeros_like(amax))         # [N, Kg]
    qi = (x_f32 * inv_scale.unsqueeze(-1)).round().clamp(-8, 7).to(torch.int8)
    qi_flat = qi.reshape(N, K)
    lo = qi_flat[:, 0::2].to(torch.int32) & 0xF
    hi = qi_flat[:, 1::2].to(torch.int32) & 0xF
    packed = (hi << 4 | lo).to(torch.uint8)
    return packed, scale.to(torch.bfloat16)


def _shape_metrics(N: int, K: int, seed: int) -> dict:
    torch.manual_seed(seed)
    x = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 1.5
    x_int_k, scale_k = act_quant_bf16_group128(x)
    x_int_r, scale_r = _reference_pack_and_scale(x)
    torch.cuda.synchronize()

    diff_bytes = (x_int_k.to(torch.int32) - x_int_r.to(torch.int32)).abs()
    n_diff = int((diff_bytes != 0).sum().item())
    total = int(diff_bytes.numel())
    max_byte_diff = int(diff_bytes.max().item())

    scale_diff = (scale_k.float() - scale_r.float()).abs()
    scale_max_abs = float(scale_diff.max().item())
    scale_max_rel = float(
        (scale_diff / scale_r.float().clamp_min(1e-6)).max().item()
    )
    return {
        "n_diff_bytes": n_diff,
        "n_total_bytes": total,
        "byte_diff_max": max_byte_diff,
        "scale_max_abs": scale_max_abs,
        "scale_max_rel": scale_max_rel,
    }


def main() -> None:
    shapes = [
        (128, 128),
        (128, 3072),
        (128, 14336),
        (256, 3072),
        (256, 14336),
    ]
    header = (
        f"{'shape':<14} "
        f"{'n_diff_bytes':>14} "
        f"{'/total':>14} "
        f"{'byte_diff_max':>14} "
        f"{'scale_abs_max':>16} "
        f"{'scale_rel_max':>16}"
    )
    print(header)
    print("-" * len(header))
    for N, K in shapes:
        m = _shape_metrics(N, K, seed=0)
        print(
            f"({N},{K})".ljust(14)
            + f"{m['n_diff_bytes']:>14d}"
            + f"{m['n_total_bytes']:>14d}"
            + f"{m['byte_diff_max']:>14d}"
            + f"{m['scale_max_abs']:>16.4e}"
            + f"{m['scale_max_rel']:>16.4e}"
        )

    # Dtype / shape / value-range sanity (no PASS/FAIL judgement).
    torch.manual_seed(1)
    x = torch.randn(128, 3072, dtype=torch.bfloat16, device="cuda")
    x_int, scale = act_quant_bf16_group128(x)
    print()
    print(f"x_int.dtype = {x_int.dtype}, shape = {tuple(x_int.shape)}")
    print(f"scale.dtype = {scale.dtype}, shape = {tuple(scale.shape)}")
    print(
        f"x_int byte range = [{int(x_int.min().item())}, "
        f"{int(x_int.max().item())}]  (raw uint8 packing)"
    )
    print(
        f"scale value range = "
        f"[{float(scale.min().item()):.4e}, {float(scale.max().item()):.4e}]"
    )


if __name__ == "__main__":
    main()
