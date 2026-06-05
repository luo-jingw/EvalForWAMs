# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Phase 16 verify script for W4A8 BF16 kernel.

For the same WAN shapes used by W8A8, build random bf16 inputs, per-channel
quantize the weight to signed INT4 (qmax=7), pack 2 nibbles per int8 byte
(low nibble -> col 2c, high nibble -> col 2c+1), and compare the kernel
output against a naive PyTorch dequantize-then-matmul reference. Pass
condition: max abs error < 2e-2 (looser than W8A8 due to 4-bit weight
quantization noise).

Reference formula:
    x_int8, scale_x = act_quant_bf16(x_bf16)                 # per-token sym
    scale_w = max(|W|, axis=in) / 7                          # per-channel sym
    w_int4_dense = round(W / scale_w).clamp(-8, 7).to(int8)
    w_int4_packed = pack_nibbles(w_int4_dense)
    y_ref = (x_int8.float() @ w_int4_dense.float().T)
          * scale_x.unsqueeze(1) * scale_w.unsqueeze(0)
          + bias
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import torch

from qwan_extension import act_quant_bf16, w4a8_gemm_bf16


SHAPES: list[tuple[int, int, int]] = [
    (256, 3072, 3072),
    (256, 3072, 14336),
    (256, 14336, 3072),
]


@dataclass
class CaseResult:
    name: str
    n: int
    k: int
    m: int
    max_abs_err: float
    max_rel_err: float
    passed: bool


def _per_channel_sym_quantize_int4(
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """w: [M, K] fp32. K must be even.
    Returns (w_int_dense [M, K] int8 in [-8, 7],
             w_packed [M, K/2] int8 (two nibbles per byte),
             scale_w [M] bf16)."""
    assert w.dim() == 2
    M, K = w.shape
    assert K % 2 == 0, f"K must be even for int4 packing, got {K}"

    row_max = w.abs().amax(dim=1).clamp_min(1e-8)
    scale = row_max / 7.0
    scale_bf16 = scale.to(torch.bfloat16)
    scale_eff = scale_bf16.to(torch.float32)
    w_int = torch.round(w / scale_eff.unsqueeze(1)).clamp(-8, 7).to(torch.int8)

    # Pack low nibble (col 2c) and high nibble (col 2c+1) per byte.
    w32 = w_int.to(torch.int32)
    low = w32[:, 0::2] & 0xF
    high = w32[:, 1::2] & 0xF
    packed_u = ((high << 4) | low) & 0xFF
    packed = packed_u.to(torch.uint8).view(torch.int8).contiguous()
    return w_int, packed, scale_bf16


def _run_case(name: str, n: int, k: int, m: int,
              device: torch.device, tol: float, seed: int = 0) -> CaseResult:
    g = torch.Generator(device=device).manual_seed(seed)
    x_bf16 = (torch.randn((n, k), device=device, generator=g) * 0.5).to(torch.bfloat16)
    w_f32 = torch.randn((m, k), device=device, generator=g) * 0.05
    bias_bf16 = (torch.randn((m,), device=device, generator=g) * 0.1).to(torch.bfloat16)

    x_int8, scale_x_bf16 = act_quant_bf16(x_bf16)
    w_int_dense, w_packed, scale_w_bf16 = _per_channel_sym_quantize_int4(w_f32)

    y_kernel = w4a8_gemm_bf16(x_int8, scale_x_bf16, w_packed, scale_w_bf16, bias_bf16)

    sx = scale_x_bf16.to(torch.float32)
    sw = scale_w_bf16.to(torch.float32)
    bias_f = bias_bf16.to(torch.float32)
    y_ref_f32 = (x_int8.to(torch.float32) @ w_int_dense.to(torch.float32).T)
    y_ref_f32 = y_ref_f32 * sx.unsqueeze(1) * sw.unsqueeze(0) + bias_f.unsqueeze(0)
    y_ref_bf16 = y_ref_f32.to(torch.bfloat16)

    diff = (y_kernel.float() - y_ref_bf16.float()).abs()
    max_abs = diff.max().item()
    max_rel = (diff / (y_ref_bf16.float().abs() + 1e-6)).max().item()
    return CaseResult(
        name=name, n=n, k=k, m=m,
        max_abs_err=max_abs, max_rel_err=max_rel,
        passed=max_abs < tol,
    )


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA unavailable.", file=sys.stderr)
        return 2
    device = torch.device("cuda:0")
    tol = 2e-2

    results: list[CaseResult] = []
    for i, (n, k, m) in enumerate(SHAPES):
        name = f"shape{i}_{n}x{k}x{m}"
        try:
            results.append(_run_case(name, n, k, m, device, tol))
        except Exception as exc:
            print(f"[{name}] ERROR: {exc}")
            results.append(CaseResult(name=name, n=n, k=k, m=m,
                                       max_abs_err=float("inf"),
                                       max_rel_err=float("inf"),
                                       passed=False))

    print(f"\nW4A8 kernel bench: tol={tol}")
    print(f"{'case':<24} {'N':>5} {'K':>6} {'M':>6} {'max_abs':>12} {'max_rel':>12} {'pass':>6}")
    print("-" * 80)
    for r in results:
        flag = "OK" if r.passed else "FAIL"
        print(f"{r.name:<24} {r.n:>5} {r.k:>6} {r.m:>6} "
              f"{r.max_abs_err:>12.3e} {r.max_rel_err:>12.3e} {flag:>6}")

    all_passed = all(r.passed for r in results)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
