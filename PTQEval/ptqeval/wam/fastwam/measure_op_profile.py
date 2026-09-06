"""Measure a fastwam inference's op-type kernel breakdown AND FLOPs/call.

The committed, reproducible source for the two calc_cross_ckpt inputs that the
per-task eval does NOT produce:
  - `op_profile.json`     -> op_breakdown_measured.png
      {"op_per_call_ms": {"linear","attention","memcpy","other"}, "_meta": {...}}
  - `measured_flops.json` -> roofline.png
      {"flops_per_call_tf": <float>, "_meta": {...}}

Why standalone (not folded into the eval, unlike lingbot_va's --profile_ops):
  FastWAM's op-breakdown and FLOPs are **shape-invariant** — they depend only on
  the fixed inference shapes (S_video=120, S_action=32, 30 layers, 10 action
  steps), not on the observation/prompt values, which are all that differ across
  tasks. A single synthetic-but-correct-shape run is therefore representative of
  every task, and avoids running torch.profiler / FlopCounterMode inside the
  14 h eval worker (host-RAM pressure alongside the resident T5). Run this once
  per variant; the JSONs feed calc_cross_ckpt for all tasks.

Method:
  - op times: torch.profiler over N infer_action calls; classify each CUDA
    kernel by name into linear/attention/memcpy/other using **self** CUDA time
    (parent/child scopes not double-counted). 'other' kernels are printed for
    audit. Caveat: attention attribution relies on the fused SDPA kernel name;
    if SDPA falls back to a math backend (bmm+softmax), bmm would land in
    'linear' and only softmax in 'attention' — inspect the 'other'/linear dump
    if attention looks too small.
  - FLOPs: FlopCounterMode over one infer_action (dtype-invariant, so the value
    is valid for bf16 and any quantized variant).
  Observation only; no assert.

CLI:
    python -m ptqeval.wam.fastwam.measure_op_profile \\
        --ckpt models/robotwin2.0-fastwam/robotwin_uncond_3cam_384.pt \\
        --variant viditq \\
        --int_weights_ckpt results/fastwam/fastwam_w4a4/calib/int_weights.pth \\
        --layer_config PTQEval/ptqeval/wam/fastwam/method/viditq/configs/w4a4.yaml \\
        --output results/fastwam/fastwam_w4a4/summary/op_profile.json \\
        --flops_output results/fastwam/fastwam_w4a4/measured_flops.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils.flop_counter import FlopCounterMode

import ptqeval.wam.fastwam  # noqa: F401
from ptqeval.wam.fastwam.method.viditq.loader import load_quant_model
from ptqeval.wam.fastwam.method.viditq.ptq import load_fastwam_model

# Substring rules over lowercased kernel names. Order: attention first
# (fused SDPA / softmax), then linear (GEMM + qwan quant kernels), memcpy,
# else other. Mirrors ptqeval.wam.lingbot_va.server classification, extended
# with the fastwam W4A4 kernel names (w4a4/atom/pack_atom/act_quant_*group).
_ATTENTION = ("fmha", "flash", "scaled_dot", "efficient_attention", "mem_eff",
              "attention", "softmax")
_LINEAR = ("gemm", "cublas", "cutlass", "w8a8", "w4a8", "w4a4", "atom",
           "qlinear", "addmm", "act_quant", "pack_atom", "ampere", "gemv",
           "wgrad", "sgemm", "hgemm", "igemm", "_mm_", "matmul", "nn_")
_MEMCPY = ("memcpy",)


def classify(name: str) -> str:
    n = name.lower()
    if any(t in n for t in _ATTENTION):
        return "attention"
    if any(t in n for t in _MEMCPY):
        return "memcpy"
    if any(t in n for t in _LINEAR):
        return "linear"
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--variant", default="viditq", choices=["", "bf16", "viditq"])
    ap.add_argument("--int_weights_ckpt", default=None)
    ap.add_argument("--layer_config", default=None)
    ap.add_argument("--output", required=True,
                    help="op_profile.json output path.")
    ap.add_argument("--flops_output", default=None,
                    help="measured_flops.json output path (FlopCounterMode). "
                         "Omit to skip the FLOP pass.")
    ap.add_argument("--n_warmup", type=int, default=2)
    ap.add_argument("--n_measure", type=int, default=3)
    ap.add_argument("--text_len", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    dev = torch.device(args.device)
    # Skip the T5 text encoder (11 GB host+GPU) and feed a synthetic context of
    # the right shape instead: the profiled path is the video/action expert
    # kernels, not text encoding, and torch.profiler already accumulates events
    # in host RAM — dropping T5 keeps the run inside memory. umt5-xxl encoder
    # dim = 4096 (projected to hidden by text_embedding inside pre_dit).
    model = load_fastwam_model(args.ckpt, device=dev, dtype=torch.bfloat16)

    image = torch.rand(1, 3, 384, 320, device=dev, dtype=model.torch_dtype) * 2 - 1
    proprio = torch.zeros(1, model.proprio_dim, device=dev, dtype=model.torch_dtype)
    context = torch.randn(1, args.text_len, 4096, device=dev, dtype=model.torch_dtype)
    context_mask = torch.ones(1, args.text_len, device=dev, dtype=torch.bool)

    def one_call():
        with torch.no_grad():
            model.infer_action(
                prompt=None, context=context, context_mask=context_mask,
                input_image=image, action_horizon=32, proprio=proprio,
                num_inference_steps=10, seed=0, rand_device="cpu")

    # --- FLOP pass FIRST, on the FP (bf16) model, BEFORE any quant swap. ---
    # Count FLOPs of ONLY the two expert regions that the roofline uses as its
    # latency denominator (transformer stage = prefill_video_cache x1;
    # action_head stage = forward_action_with_video_cache x10). Counting the
    # whole infer_action would also include VAE encode + pre/post_dit, inflating
    # FLOPs relative to the transformer+action_head time. Measure on the FP model
    # because FlopCounterMode is dispatcher-level and cannot see the custom
    # qwan_extension int GEMM; FLOPs are quantization-invariant (same shapes).
    flops_per_call_tf = None
    if args.flops_output:
        _mot = model.mot
        _orig_prefill = _mot.prefill_video_cache
        _orig_action = _mot.forward_action_with_video_cache
        _flop_tf = {"v": 0.0}

        def _prefill_fc(*a, **k):
            fc = FlopCounterMode(display=False)
            with fc:
                out = _orig_prefill(*a, **k)
            _flop_tf["v"] += fc.get_total_flops() / 1e12
            return out

        def _action_fc(*a, **k):
            fc = FlopCounterMode(display=False)
            with fc:
                out = _orig_action(*a, **k)
            _flop_tf["v"] += fc.get_total_flops() / 1e12
            return out

        _mot.prefill_video_cache = _prefill_fc
        _mot.forward_action_with_video_cache = _action_fc
        one_call()
        _mot.prefill_video_cache = _orig_prefill
        _mot.forward_action_with_video_cache = _orig_action
        flops_per_call_tf = _flop_tf["v"]
        torch.cuda.synchronize(dev)

    # Now swap to the quantized kernels for the op-time profiler pass.
    if args.variant == "viditq":
        if not args.int_weights_ckpt:
            ap.error("--variant viditq requires --int_weights_ckpt")
        load_quant_model(model, args.int_weights_ckpt,
                         layer_config=args.layer_config, device=dev)

    for _ in range(args.n_warmup):
        one_call()
    torch.cuda.synchronize(dev)

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(args.n_measure):
            one_call()
        torch.cuda.synchronize(dev)

    bucket_us: dict[str, float] = defaultdict(float)
    other_kernels: list[tuple[float, str]] = []
    for evt in prof.key_averages():
        self_cuda_us = float(getattr(evt, "self_device_time_total",
                                     getattr(evt, "self_cuda_time_total", 0.0)))
        if self_cuda_us <= 0:
            continue
        cat = classify(evt.key)
        bucket_us[cat] += self_cuda_us
        if cat == "other":
            other_kernels.append((self_cuda_us, evt.key))

    n = args.n_measure
    op_per_call_ms = {k: bucket_us.get(k, 0.0) / 1000.0 / n
                      for k in ("linear", "attention", "memcpy", "other")}
    total = sum(op_per_call_ms.values())

    print("\n===== op breakdown (self CUDA time, per replan) =====")
    for k in ("linear", "attention", "memcpy", "other"):
        v = op_per_call_ms[k]
        print(f"  {k:<10} {v:8.2f} ms  ({v / total * 100 if total else 0:5.1f}%)")
    print(f"  {'TOTAL':<10} {total:8.2f} ms")
    print("\ntop 'other' kernels (self CUDA us total over "
          f"{n} calls):")
    for us, name in sorted(other_kernels, reverse=True)[:12]:
        print(f"  {us / 1000:8.2f} ms  {name[:88]}")

    out = {
        "op_per_call_ms": op_per_call_ms,
        "_meta": {
            "variant": args.variant or "bf16",
            "n_measure": n, "n_warmup": args.n_warmup,
            "int_weights_ckpt": args.int_weights_ckpt,
            "note": "self CUDA time per infer_action (1 video prefill + 10 action steps)",
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.output}")

    if args.flops_output and flops_per_call_tf is not None:
        flops_out = {
            "flops_per_call_tf": flops_per_call_tf,
            "_meta": {
                "source": "FlopCounterMode over one infer_action (1 video prefill "
                          "+ 10 action steps), measured on the FP model BEFORE "
                          "quant swap (custom int GEMM is invisible to the "
                          "dispatcher). Quantization-invariant -> valid for all variants.",
                "measured_on": "fp_model",
                "seq": {"S_video": 120, "S_action": 32, "num_inference_steps": 10},
            },
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.flops_output)), exist_ok=True)
        with open(args.flops_output, "w") as f:
            json.dump(flops_out, f, indent=2)
        print(f"wrote {args.flops_output}  (flops_per_call_tf={flops_per_call_tf:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
