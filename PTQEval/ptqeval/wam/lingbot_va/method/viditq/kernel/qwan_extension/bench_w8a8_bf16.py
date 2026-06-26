# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Phase 25 numerical bench for the bf16 W8A8 GEMM instantiation
(OutT = __nv_bfloat16). Mirrors bench_w8a8_fp16.py in bf16 domain.

Reference: same ViDiT-Q-style asym formula (bench_gemm.py:26-29), but
all fp16 scales/bias/sum_input become bf16.  OBSERVATIONAL per
principle.txt L12: emits {max_abs_err, max_rel_err} per shape +
wall-clock kernel-vs-torch ms / TFLOPS / speedup for one
representative shape; no PASS/FAIL judgement.
"""
import torch
import torch.nn.functional as F

from qwan_extension._C import w8a8_obf16_bias_weight_asym


SHAPES = [
    (256,  3072,  3072),
    (256,  3072, 14336),
    (256, 14336,  3072),
]
BENCH_SHAPE = (256, 3072, 3072)
WARMUP_ITERS = 20
TIMED_ITERS = 100


def make_inputs(M: int, N: int, K: int, seed: int = 0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    input_int  = torch.randint(-80, 80, (M, K), dtype=torch.int8, device="cuda", generator=g)
    weight_int = torch.randint(-80, 80, (N, K), dtype=torch.int8, device="cuda", generator=g)
    scale_input  = 0.01 * torch.rand(M, dtype=torch.bfloat16, device="cuda", generator=g) + 0.005
    scale_weight = 0.10 * torch.rand(N, dtype=torch.bfloat16, device="cuda", generator=g) + 0.10
    bias         = torch.rand(N,    dtype=torch.bfloat16, device="cuda", generator=g) * 200
    zp_weight    = torch.randint(-10, 10, (N,), dtype=torch.int16, device="cuda", generator=g)
    input_sum = (scale_input.view(-1, 1).to(torch.float32)
                 * input_int.to(torch.float32)).sum(dim=1).to(torch.bfloat16)
    return input_int, weight_int, bias, scale_input, scale_weight, input_sum, zp_weight


def reference_viditq(input_int, weight_int, bias,
                     scale_input, scale_weight, input_sum, zp_weight) -> torch.Tensor:
    # Same fp32-domain formula as bench_w8a8_fp16; only the final cast
    # changes (bf16 instead of fp16).
    input_fp32  = input_int.to(torch.float32)
    weight_fp32 = weight_int.to(torch.float32)
    y_fp32 = (F.linear(input_fp32, weight_fp32)
              * scale_input.view(-1, 1).to(torch.float32)
              * scale_weight.view(1, -1).to(torch.float32)
              + input_sum.view(-1, 1).to(torch.float32)
              * zp_weight.to(torch.float32).view(1, -1)
              * scale_weight.view(1, -1).to(torch.float32)
              + bias.to(torch.float32))
    return y_fp32.to(torch.bfloat16)


def numerical_check() -> None:
    print(f"{'shape (M, N, K)':<26}{'max_abs_err':>14}{'max_rel_err':>14}")
    print("-" * 54)
    for M, N, K in SHAPES:
        x, w, b, sx, sw, su, zp = make_inputs(M, N, K, seed=0)
        y_kernel = w8a8_obf16_bias_weight_asym(x, w, b, sx, sw, su, zp)
        y_ref    = reference_viditq(x, w, b, sx, sw, su, zp)

        y_k_fp32 = y_kernel.to(torch.float32)
        y_r_fp32 = y_ref.to(torch.float32)
        abs_diff = (y_k_fp32 - y_r_fp32).abs()
        max_abs  = abs_diff.max().item()
        denom    = y_r_fp32.abs().clamp_min(1.0)
        max_rel  = (abs_diff / denom).max().item()
        print(f"({M:>4}, {N:>5}, {K:>5}){max_abs:>16.4e}{max_rel:>14.4e}")


def wall_clock_compare() -> None:
    M, N, K = BENCH_SHAPE
    x, w, b, sx, sw, su, zp = make_inputs(M, N, K, seed=1)
    # torch reference: same M,N,K linear in bf16. Uses input_bf16 (not
    # int8) since torch's linear works on float dtypes; this is a fairness
    # baseline measuring (M,N,K) GEMM wall-clock at fp arithmetic.
    x_bf16 = x.to(torch.bfloat16)
    w_bf16 = w.to(torch.bfloat16)

    # Warmup.
    for _ in range(WARMUP_ITERS):
        _ = w8a8_obf16_bias_weight_asym(x, w, b, sx, sw, su, zp)
        _ = F.linear(x_bf16, w_bf16, bias=b)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    # Time kernel.
    start.record()
    for _ in range(TIMED_ITERS):
        _ = w8a8_obf16_bias_weight_asym(x, w, b, sx, sw, su, zp)
    end.record()
    torch.cuda.synchronize()
    kernel_ms = start.elapsed_time(end) / TIMED_ITERS

    # Time torch.linear bf16.
    start.record()
    for _ in range(TIMED_ITERS):
        _ = F.linear(x_bf16, w_bf16, bias=b)
    end.record()
    torch.cuda.synchronize()
    torch_ms = start.elapsed_time(end) / TIMED_ITERS

    tflops = 2.0 * M * N * K / 1e9
    print()
    print(f"Wall-clock (M={M}, N={N}, K={K}, {TIMED_ITERS} iters avg):")
    print(f"  w8a8_obf16 kernel : {kernel_ms:7.3f} ms  ({tflops / kernel_ms:6.1f} TFLOPS effective)")
    print(f"  torch bf16 linear : {torch_ms:7.3f} ms  ({tflops / torch_ms:6.1f} TFLOPS effective)")
    print(f"  speedup ratio     : {torch_ms / kernel_ms:5.2f}x  (kernel vs torch)")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required")

    print("Phase 25 observational bench: W8A8 bf16-output kernel")
    print()
    numerical_check()
    wall_clock_compare()


if __name__ == "__main__":
    main()
