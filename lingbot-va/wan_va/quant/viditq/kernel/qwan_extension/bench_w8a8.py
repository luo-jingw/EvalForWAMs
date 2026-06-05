# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Phase 15 verify script for W8A8 BF16 kernel.

For a fixed set of representative WAN transformer shapes, build random
bf16 inputs, quantize via the in-package kernels, and compare against a
naive PyTorch dequantize-then-matmul reference. Pass condition:
max abs error < 1e-2.

Reference formula (matches the kernel mathematically):
    x_int8, scale_x = act_quant_bf16(x_bf16)            # per-token sym
    scale_w = max(|W|, axis=in) / 127                   # per-channel sym
    w_int8  = round(W / scale_w).clamp(-127, 127).to(int8)
    y_ref = (x_int8.float() @ w_int8.float().T)
          * scale_x.unsqueeze(1) * scale_w.unsqueeze(0)
          + bias
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import torch

from qwan_extension import act_quant_bf16, w8a8_gemm_bf16


# (N, K, M) shapes from WanTransformerBlock with dim=3072, ffn_dim=14336.
# N is chosen as a moderately large token count seen during real inference.
SHAPES: list[tuple[int, int, int]] = [
    (256, 3072, 3072),      # attention projection (qkv / out_proj)
    (256, 3072, 14336),     # FFN up
    (256, 14336, 3072),     # FFN down
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


def _per_channel_sym_quantize_int8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """w: [M, K] float32. Returns (w_int8 [M, K], scale_w [M] bfloat16)."""
    assert w.dim() == 2
    row_max = w.abs().amax(dim=1).clamp_min(1e-8)
    scale = row_max / 127.0
    scale_bf16 = scale.to(torch.bfloat16)
    # Round-trip through bf16 so the reference uses the same scale precision
    # the kernel will load from device memory.
    scale_eff = scale_bf16.to(torch.float32)
    w_int = torch.round(w / scale_eff.unsqueeze(1)).clamp(-127, 127).to(torch.int8)
    return w_int, scale_bf16


def _run_case(name: str, n: int, k: int, m: int,
              device: torch.device, tol: float, seed: int = 0) -> CaseResult:
    g = torch.Generator(device=device).manual_seed(seed)
    x_bf16 = (torch.randn((n, k), device=device, generator=g) * 0.5).to(torch.bfloat16)
    w_f32 = torch.randn((m, k), device=device, generator=g) * 0.05
    bias_bf16 = (torch.randn((m,), device=device, generator=g) * 0.1).to(torch.bfloat16)

    # Kernel quantization of activation.
    x_int8, scale_x_bf16 = act_quant_bf16(x_bf16)
    # Per-channel sym quantization of weight (Python).
    w_int8, scale_w_bf16 = _per_channel_sym_quantize_int8(w_f32)

    y_kernel = w8a8_gemm_bf16(x_int8, scale_x_bf16, w_int8, scale_w_bf16, bias_bf16)

    # Reference in fp32. Uses the same int8 buffers and bf16 scales as the kernel.
    sx = scale_x_bf16.to(torch.float32)         # [N]
    sw = scale_w_bf16.to(torch.float32)         # [M]
    bias_f = bias_bf16.to(torch.float32)        # [M]
    y_ref_f32 = (x_int8.to(torch.float32) @ w_int8.to(torch.float32).T)
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
    tol = 1e-2

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

    print(f"\nW8A8 kernel bench: tol={tol}")
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
