# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Phase 42 step 7 observational check — W4A4 GEMM kernel direct calls.

Direct kernel-level metric table for the 4 W4A4 launchers:
  w4a4_of16_nobias_weight_sym   (fp16 out, no bias)
  w4a4_obf16_nobias_weight_sym  (bf16 out, no bias)
  w4a4_of16_bias_weight_sym     (fp16 out, with bias — G3 fusion)
  w4a4_obf16_bias_weight_sym    (bf16 out, with bias — G3 fusion)

Two angles per launcher × shape:
  [Ref]   y_kernel vs fp32 dequant reference
            y_ref = (x_dq) @ (w_dq).T  (+ bias)
          isolates total W4A4 quant noise (act + weight + accum).
  [Bias]  bias-fused output vs (nobias output + python bias add)
          isolates G3 epilogue numerical equivalence (target ~1e-3 abs
          per plan L703; observational only — no assert).

Shapes mirror the 6 target WAN Linears: attn 3072x3072 / ffn up
3072x14336 / ffn down 14336x3072, plus M=128 (1 CTA) and M=256 (2 CTA)
batch dims so both single- and multi-block paths execute.

No assert / no PASS (principle.txt L12).

Run:
    cd PTQEval/ptqeval/wam/lingbot_va/method/viditq/kernel
    python -m qwan_extension.check_w4a4
"""
from __future__ import annotations

import torch

from qwan_extension import (
    act_quant_bf16_group128,
    pack_atom_scale_a_bf16,
    pack_atom_scale_a_fp16,
    pack_atom_scale_b_bf16,
    pack_atom_scale_b_fp16,
    w4a4_of16_bias_weight_sym,
    w4a4_of16_nobias_weight_sym,
    w4a4_obf16_bias_weight_sym,
    w4a4_obf16_nobias_weight_sym,
)


_GROUP_SIZE = 128


def _unpack_int4_signed(packed_u8: torch.Tensor) -> torch.Tensor:
    """uint8 [.., K/2] → int8 [.., K] signed [-8, 7]."""
    u = packed_u8.to(torch.int32)
    low = u & 0xF
    high = (u >> 4) & 0xF
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    out = torch.empty(*packed_u8.shape[:-1], packed_u8.shape[-1] * 2,
                      dtype=torch.int8, device=packed_u8.device)
    out[..., 0::2] = low.to(torch.int8)
    out[..., 1::2] = high.to(torch.int8)
    return out


def _per_group_sym_quant_w_test(w_f32: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Test-side replica of ptq._per_group_sym_quant_w4a4 — kept inline so the
    check has zero dependency on the ptq module beyond _GROUP_SIZE.
    Returns (uint8 [C_out, C_in/2] packed, bf16 [C_out, C_in/128])."""
    C_out, C_in = w_f32.shape
    n_groups = C_in // _GROUP_SIZE
    w_g = w_f32.view(C_out, n_groups, _GROUP_SIZE)
    delta = (w_g.abs().amax(dim=2) / 7).clamp_min(1e-8)
    scale_bf16 = delta.to(torch.bfloat16)
    int_w = (torch.round(w_g / scale_bf16.to(torch.float32).unsqueeze(2))
             .clamp(-8, 7).to(torch.int8))
    int_w_flat = int_w.view(C_out, C_in)
    w32 = int_w_flat.to(torch.int32)
    packed = (((w32[:, 1::2] & 0xF) << 4) | (w32[:, 0::2] & 0xF)) & 0xFF
    return packed.to(torch.uint8).contiguous(), scale_bf16.contiguous()


def _dequant_w(int_w_packed: torch.Tensor, scale_natural: torch.Tensor) -> torch.Tensor:
    int_w = _unpack_int4_signed(int_w_packed).to(torch.float32)
    C_out, C_in = int_w.shape
    n_groups = scale_natural.shape[1]
    return (int_w.view(C_out, n_groups, _GROUP_SIZE)
            * scale_natural.to(torch.float32).unsqueeze(2)).view(C_out, C_in)


def _dequant_x(x_int4_packed: torch.Tensor, scale_natural: torch.Tensor) -> torch.Tensor:
    """x_int4_packed uint8 [M, K/2] + scale [M, K/128] → fp32 [M, K]."""
    M, half = x_int4_packed.shape
    int_x = _unpack_int4_signed(x_int4_packed).to(torch.float32)
    n_groups = scale_natural.shape[1]
    return (int_x.view(M, n_groups, _GROUP_SIZE)
            * scale_natural.to(torch.float32).unsqueeze(2)).view(M, half * 2)


def _row(label, y_kernel, y_ref):
    diff = (y_kernel.float() - y_ref.float()).abs()
    mag = y_ref.float().abs().mean().item()
    print(f"  {label:48s} "
          f"shape_match={tuple(y_kernel.shape) == tuple(y_ref.shape)}  "
          f"max_abs={diff.max().item():.3e}  "
          f"mean_abs={diff.mean().item():.3e}  "
          f"ref_mag={mag:.3e}  "
          f"rel_max={diff.max().item() / (mag + 1e-12):.3e}")


def _case(M: int, K: int, N: int):
    print()
    print("-" * 110)
    print(f"M={M}  K={K}  N={N}")
    print("-" * 110)
    torch.manual_seed(M * 1000 + K + N)
    # bf16 random inputs / weights / bias scaled to typical WAN magnitudes.
    x_bf16 = (torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.05).contiguous()
    w_bf16 = (torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.05).contiguous()
    b_bf16 = (torch.randn(N, dtype=torch.bfloat16, device="cuda") * 0.01).contiguous()

    # PTQ-side pre-compute: packed int weights + per-group scale (natural).
    w_packed, scale_w_nat = _per_group_sym_quant_w_test(w_bf16.to(torch.float32))

    # Runtime-side: act quant + pack-on-the-fly.
    x_int4, scale_x_nat = act_quant_bf16_group128(x_bf16)

    # Pack scales for both dtypes (B side from natural [N, K/128] → [K/128, N]).
    scale_x_pack_bf16 = pack_atom_scale_a_bf16(scale_x_nat.contiguous())
    scale_x_pack_fp16 = pack_atom_scale_a_fp16(scale_x_nat.to(torch.float16).contiguous())
    scale_w_pack_bf16 = pack_atom_scale_b_bf16(scale_w_nat)
    scale_w_pack_fp16 = pack_atom_scale_b_fp16(scale_w_nat.to(torch.float16))

    # fp32 dequant reference (single source for both dtype/bias variants).
    x_dq = _dequant_x(x_int4, scale_x_nat)
    w_dq = _dequant_w(w_packed, scale_w_nat)
    y_ref_nobias = x_dq.float() @ w_dq.float().T

    # [Ref] vs fp32 dequant — 4 launchers
    y_bf16_nb = w4a4_obf16_nobias_weight_sym(
        x_int4, w_packed, scale_x_pack_bf16, scale_w_pack_bf16)
    _row("bf16 nobias  vs fp32 dequant ref", y_bf16_nb, y_ref_nobias)

    y_fp16_nb = w4a4_of16_nobias_weight_sym(
        x_int4, w_packed, scale_x_pack_fp16, scale_w_pack_fp16)
    _row("fp16 nobias  vs fp32 dequant ref", y_fp16_nb, y_ref_nobias)

    y_ref_bias = y_ref_nobias + b_bf16.float()
    y_bf16_b = w4a4_obf16_bias_weight_sym(
        x_int4, w_packed, b_bf16, scale_x_pack_bf16, scale_w_pack_bf16)
    _row("bf16 bias    vs fp32 dequant + bias", y_bf16_b, y_ref_bias)

    y_fp16_b = w4a4_of16_bias_weight_sym(
        x_int4, w_packed, b_bf16.to(torch.float16),
        scale_x_pack_fp16, scale_w_pack_fp16)
    _row("fp16 bias    vs fp32 dequant + bias", y_fp16_b, y_ref_bias)

    # [Bias] bias-fused vs (nobias kernel + Python bias add) — G3 equivalence
    # observational. Plan L703 target ~1e-3 abs.
    y_bf16_b_python = y_bf16_nb + b_bf16
    _row("bf16 bias    vs (nobias kernel + Py bias add)",
         y_bf16_b, y_bf16_b_python)
    y_fp16_b_python = y_fp16_nb + b_bf16.to(torch.float16)
    _row("fp16 bias    vs (nobias kernel + Py bias add)",
         y_fp16_b, y_fp16_b_python)


def main() -> None:
    print("Phase 42 step 7 — W4A4 kernel direct-call observational metrics")
    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}")
    cases = [
        (128, 3072, 3072),     # attn proj single-CTA
        (256, 3072, 3072),     # attn proj 2-CTA
        (128, 3072, 14336),    # ffn up
        (128, 14336, 3072),    # ffn down
        (256, 14336, 3072),    # ffn down 2-CTA
    ]
    for M, K, N in cases:
        _case(M, K, N)


if __name__ == "__main__":
    main()
