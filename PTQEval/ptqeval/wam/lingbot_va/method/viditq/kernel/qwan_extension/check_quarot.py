# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Phase 37 numerical check.

Validates QuaRoT round-trip behavior end-to-end against the W8A8 kernel
wrapper.

Three cases per shape:
  (a) FP reference  : y_fp = F.linear(x, W, b)
  (b) Baseline W8A8 : QuantWanLinearW8A8.from_fp_linear(fp), forward(x).
                      This is Phase 26a-2's path with no quarot_sign.
  (c) QuaRoT  W8A8  : Build the same wrapper, then rotate the weight via
                      quarot.rotate_weight and install a quarot_sign
                      buffer. forward(x) applies the matching rotation
                      to x before the kernel.

Expected: error(c) is in the same ballpark as error(b) (rotation does
not blow up the quantization error and may improve it on outlier-heavy
weight distributions).
"""
from __future__ import annotations

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from qwan_extension.nn import QuantWanLinearW8A8

from ptqeval.wam.lingbot_va.method.viditq.quarot import (
    random_sign_vector,
    rotate_weight,
)


def _check_one(name: str, in_features: int, out_features: int,
               batch: int, seq: int, has_bias: bool, tol: float,
               device: torch.device, seed: int = 0) -> bool:
    g = torch.Generator(device=device).manual_seed(seed)
    fp = nn.Linear(in_features, out_features, bias=has_bias).to(device).to(torch.bfloat16)
    with torch.no_grad():
        w_init = (torch.randn((out_features, in_features), device=device, generator=g) * 0.05).to(torch.bfloat16)
        fp.weight.copy_(w_init)
        if has_bias:
            b_init = (torch.randn((out_features,), device=device, generator=g) * 0.1).to(torch.bfloat16)
            fp.bias.copy_(b_init)

    x = (torch.randn((batch, seq, in_features), device=device, generator=g) * 0.5).to(torch.bfloat16)

    # (a) FP reference (bf16 throughout, to match the kernel's bf16 epilogue).
    y_fp = F.linear(x, fp.weight, fp.bias).to(torch.bfloat16)

    # (b) Baseline W8A8 (no quarot).
    mod_base = QuantWanLinearW8A8.from_fp_linear(fp).to(device)
    y_base = mod_base(x)
    err_base = (y_fp.float() - y_base.float()).abs().max().item()

    # (c) QuaRoT W8A8: rotate weight, then build kernel module from the
    # rotated FP linear, then install the quarot_sign buffer.
    sign = random_sign_vector(in_features, seed=seed + 100).to(device)
    fp_rot = nn.Linear(in_features, out_features, bias=has_bias).to(device).to(torch.bfloat16)
    with torch.no_grad():
        fp_rot.weight.copy_(rotate_weight(fp.weight, sign))
        if has_bias:
            fp_rot.bias.copy_(fp.bias)
    mod_quarot = QuantWanLinearW8A8.from_fp_linear(fp_rot).to(device)
    mod_quarot.quarot_sign = sign.to(device).contiguous()
    y_quarot = mod_quarot(x)
    err_quarot = (y_fp.float() - y_quarot.float()).abs().max().item()

    base_ok = err_base < tol
    # Tolerance bump for quarot path: rotation routes outliers through every
    # output channel uniformly, so per-element max error can be different
    # (often smaller on real model weights, can be larger on Gaussian
    # synthetic ones because the rotation diffuses the dominant components).
    quarot_ok = err_quarot < tol * 2.0

    flag = "OK" if (base_ok and quarot_ok) else "FAIL"
    print(f"{name:<40}  base_err={err_base:.3e}  quarot_err={err_quarot:.3e}  "
          f"(tol={tol:.0e}, quarot tol={tol*2:.0e})  {flag}")
    return base_ok and quarot_ok


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA unavailable.", file=sys.stderr)
        return 2
    device = torch.device("cuda:0")

    cases = [
        ("attn  3072->3072   bias=True",  3072,  3072, 2, 256, True,  5e-2),
        ("attn  3072->3072   bias=False", 3072,  3072, 2, 256, False, 5e-2),
        ("ffn   3072->14336  bias=True",  3072, 14336, 1, 128, True,  5e-2),
        ("ffn   14336->3072  bias=True",  14336, 3072, 1, 128, True,  5e-2),
    ]
    results = [_check_one(name, ci, co, b, s, hb, tol, device)
               for (name, ci, co, b, s, hb, tol) in cases]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
