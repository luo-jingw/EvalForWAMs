# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Phase 42 step 7 observational check — W4A8 bias-fusion equivalence (G4).

The W4A8 QServe-port kernel originally shipped only no-bias launchers; the
G4 retrofit (commit 43bbba6 + defensive follow-up 37f5c04) added
has_bias=true template instantiations:
  w4a8_of16_bias_weight_asym   (fp16 out)
  w4a8_obf16_bias_weight_asym  (bf16 out)
with bias loaded inside the dequant epilogue (`psums += bias` in float2)
before the cast-to-OutT store, eliminating the per-Linear Memcpy DtoD
the Python post-add caused.

This check observes whether the fused path is numerically equivalent to
the legacy `nobias_kernel(...) + bias` Python path. Plan L703 target:
max_abs < 1e-3 (kept as a metric, not an assert, per principle.txt L12).

Shapes mirror WAN Linears with bias (to_out, ffn.net.0.proj, ffn.net.2).

Run:
    cd PTQEval/ptqeval/wam/lingbot_va/method/viditq/kernel
    python -m qwan_extension.check_w4a8_bias
"""
from __future__ import annotations

import torch

from qwan_extension import (
    act_quant_bf16_with_sum,
    w4a8_of16_bias_weight_asym,
    w4a8_of16_nobias_weight_asym,
    w4a8_obf16_bias_weight_asym,
    w4a8_obf16_nobias_weight_asym,
)


def _per_channel_w4_asym_unsigned(w_f32):
    """Inline replica of ptq._per_channel_asym_quant_unsigned for n_bits=4."""
    n_levels = 16
    max_int = n_levels - 1
    x_max = w_f32.amax(dim=1)
    x_min = w_f32.amin(dim=1)
    delta = ((x_max - x_min) / max_int).clamp_min(1e-8)
    scale_bf16 = delta.to(torch.bfloat16)
    scale_eff = scale_bf16.to(torch.float32)
    zp_uns_f = torch.round(-x_min / scale_eff).clamp(0, max_int)
    int_w = (torch.round(w_f32 / scale_eff.unsqueeze(1)) + zp_uns_f.unsqueeze(1)
             ).clamp(0, max_int).to(torch.uint8)
    return int_w, scale_bf16, zp_uns_f.to(torch.uint8)


def _pack_int4_qserve(w_int_unsigned):
    """Inline replica of ptq._pack_int4_qserve."""
    N, K = w_int_unsigned.shape
    lw = w_int_unsigned.to(torch.int32)
    lw = lw.reshape(N // 32, 2, 2, 8, K // 32, 2, 4, 4)
    lw = lw.permute(0, 4, 3, 6, 1, 5, 2, 7).contiguous()
    lw = lw.permute(0, 1, 2, 3, 5, 6, 7, 4).contiguous()
    packed = ((lw[..., 1] << 4) | (lw[..., 0] & 0xF)) & 0xFF
    packed = packed.reshape(N // 32, K // 32, 32, 16).reshape(N, K // 2)
    return packed.to(torch.uint8).view(torch.int8).contiguous()


def _row(label, y_fused, y_python):
    diff = (y_fused.float() - y_python.float()).abs()
    mag = y_python.float().abs().mean().item()
    print(f"  {label:48s} "
          f"shape_match={tuple(y_fused.shape) == tuple(y_python.shape)}  "
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
    x_bf16 = (torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.05).contiguous()
    w_bf16 = (torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.05).contiguous()
    b_bf16 = (torch.randn(N, dtype=torch.bfloat16, device="cuda") * 0.01).contiguous()

    # PTQ-side: per-channel asym unsigned int4 + QServe pack + szeros.
    int_w_u, scale_w, zp_uns = _per_channel_w4_asym_unsigned(w_bf16.to(torch.float32))
    packed_w = _pack_int4_qserve(int_w_u)
    szeros = (scale_w.to(torch.float32) * zp_uns.to(torch.float32)).to(torch.bfloat16)

    # Runtime: per-token sym INT8 act quant (W8A8/W4A8 path).
    x_int8, scale_x, sum_x = act_quant_bf16_with_sum(x_bf16)

    # bf16 path equivalence
    y_bf16_nb = w4a8_obf16_nobias_weight_asym(
        x_int8, packed_w, scale_x, scale_w, sum_x, szeros)
    y_bf16_python = y_bf16_nb + b_bf16
    y_bf16_fused = w4a8_obf16_bias_weight_asym(
        x_int8, packed_w, b_bf16, scale_x, scale_w, sum_x, szeros)
    _row("bf16  fused vs (nobias + Py bias add)", y_bf16_fused, y_bf16_python)

    # fp16 path equivalence (scales must be fp16 for the fp16 launcher).
    scale_x_fp16 = scale_x.to(torch.float16)
    scale_w_fp16 = scale_w.to(torch.float16)
    sum_x_fp16 = sum_x.to(torch.float16)
    szeros_fp16 = szeros.to(torch.float16)
    b_fp16 = b_bf16.to(torch.float16)
    y_fp16_nb = w4a8_of16_nobias_weight_asym(
        x_int8, packed_w, scale_x_fp16, scale_w_fp16, sum_x_fp16, szeros_fp16)
    y_fp16_python = y_fp16_nb + b_fp16
    y_fp16_fused = w4a8_of16_bias_weight_asym(
        x_int8, packed_w, b_fp16, scale_x_fp16, scale_w_fp16, sum_x_fp16, szeros_fp16)
    _row("fp16  fused vs (nobias + Py bias add)", y_fp16_fused, y_fp16_python)


def main() -> None:
    print("Phase 42 step 7 — W4A8 bias-fusion equivalence (G4) observational")
    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}")
    cases = [
        (128, 3072, 3072),
        (256, 3072, 3072),
        (128, 3072, 14336),
        (128, 14336, 3072),
        (256, 14336, 3072),
    ]
    for M, K, N in cases:
        _case(M, K, N)


if __name__ == "__main__":
    main()
