# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Phase 26a-1 OBSERVATIONAL bench for act_quant_bf16_with_sum.

Per principle.txt L12: emits raw metrics only — no assert, no PASS/FAIL
judgement.  Reports kernel-vs-reference diff per shape; the user
inspects the table.

Reference: fp32-domain formula matching ViDiT-Q QuantKernel, recast for
the (x_int8, scale_x, sum_x) tuple.  Three shapes:
    (256,  3072)   WAN attn / up_proj activation
    ( 64,  8192)   ViDiT-Q upper-bound shape (regression vs Phase 24c)
    (256, 14336)   LingBot-VA ffn down_proj activation
"""
import torch

from qwan_extension._C import act_quant_bf16_with_sum


SHAPES = [
    (256,  3072),
    ( 64,  8192),
    (256, 14336),
]


def reference(x: torch.Tensor):
    x_fp32 = x.to(torch.float32)
    amax = x_fp32.abs().amax(dim=1)
    scale_fp32 = (amax / 127.0).clamp_min(1e-8)
    scale_bf16 = scale_fp32.to(torch.bfloat16)
    inv_scale = 127.0 / amax
    x_int8 = (x_fp32 * inv_scale.view(-1, 1)).round().clamp(-127, 127).to(torch.int8)
    sum_int = x_int8.to(torch.int32).sum(dim=1)
    sum_fp = sum_int.to(torch.float32) / inv_scale
    sum_bf16 = sum_fp.to(torch.bfloat16)
    return x_int8, scale_bf16, sum_bf16


def shape_metrics(N: int, K: int, seed: int) -> None:
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((N, K), dtype=torch.float32, device="cuda", generator=g).to(torch.bfloat16)

    x_int8_k, scale_k, sum_k = act_quant_bf16_with_sum(x)
    x_int8_r, scale_r, sum_r = reference(x)

    int8_diff = (x_int8_k.to(torch.int32) - x_int8_r.to(torch.int32)).abs().max().item()
    scale_abs = (scale_k.to(torch.float32) - scale_r.to(torch.float32)).abs().max().item()
    sum_abs   = (sum_k.to(torch.float32) - sum_r.to(torch.float32)).abs()
    denom     = sum_r.to(torch.float32).abs().clamp_min(1.0)
    sum_rel   = (sum_abs / denom).max().item()
    sum_abs_v = sum_abs.max().item()

    print(
        f"({N:>4}, {K:>5})  int8_max_diff={int8_diff:>3d}"
        f"  scale_abs={scale_abs:.3e}"
        f"  sum_abs={sum_abs_v:.3e}"
        f"  sum_rel={sum_rel:.3e}"
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required")

    print("Phase 26a-1 observational bench: act_quant_bf16_with_sum")
    print("-" * 80)
    for N, K in SHAPES:
        shape_metrics(N, K, seed=0)


if __name__ == "__main__":
    main()
