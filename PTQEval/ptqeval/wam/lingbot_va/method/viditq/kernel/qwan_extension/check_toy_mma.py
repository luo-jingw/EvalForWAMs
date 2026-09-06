# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Phase 24a verification: single-CTA m16n8k32 s8s8s32 MMA toy kernel.

Compares qwan_extension._C.toy_mma_int8_gemm against a torch.int32 reference.
Acceptance: zero difference (the MMA is exact integer arithmetic).
"""
import torch

from qwan_extension._C import toy_mma_int8_gemm


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for toy MMA check")

    generator = torch.Generator(device="cuda").manual_seed(0)
    a = torch.randint(
        -128, 127, (16, 32), dtype=torch.int8, device="cuda", generator=generator
    )
    b = torch.randint(
        -128, 127, (8, 32), dtype=torch.int8, device="cuda", generator=generator
    )
    c = torch.zeros((16, 8), dtype=torch.int32, device="cuda")

    toy_mma_int8_gemm(a, b, c)

    # torch CUDA does not implement int32 matmul; compute reference on CPU.
    ref_cpu = a.cpu().to(torch.int32) @ b.cpu().to(torch.int32).T
    ref = ref_cpu.to("cuda")
    if not torch.equal(c, ref):
        max_diff = (c - ref).abs().max().item()
        raise AssertionError(
            f"toy MMA mismatch: max_abs_diff={max_diff}"
        )

    print("toy MMA OK: zero diff vs int32 ref")


if __name__ == "__main__":
    main()
