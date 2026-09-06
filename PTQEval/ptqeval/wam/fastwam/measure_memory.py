"""Measure the FastWAM inference VRAM breakdown into weight + KV + activation
segments, for calc_cross_ckpt's memory_breakdown.png.

Emits a measure_kv_cache-schema JSON (`{"samples": {...}, "meta": {...}}`) that
`ptqeval.eval.calc_cross_ckpt --measured_kv_cache tag=<path>` consumes:

    samples = {
      transformer_weight_mb,    # mot (video+action experts), quantized footprint
      vae_weight_mb,            # VAE
      text_encoder_weight_mb,   # T5 (resident throughout FastWAM inference)
      kv_cache_mb,              # video KV cache from one prefill
      activation_peak_mb,       # transient activation = peak - static weights - kv
    }

Segments are measured by summing each component's param+buffer bytes (FastWAM
instantiates the whole model at once, so per-module byte sums are cleaner than
lingbot_va's incremental-load deltas). For the quantized variant, mot's byte sum
reflects the real int_weight/scale/zp footprint.

Why this matters: FastWAM keeps **T5 resident** for the whole inference (it does
NOT transient-swap it like LingBot-VA Phase 41), so the single ~20 GB peak is
T5-dominated and hides the weight-quantization saving. The breakdown separates
the quant-shrinkable transformer weight from the quant-invariant T5/VAE/KV floor.
Shape-invariant across tasks -> one standalone run per variant.

CLI:
    python -m ptqeval.wam.fastwam.measure_memory \\
        --ckpt models/robotwin2.0-fastwam/robotwin_uncond_3cam_384.pt \\
        --variant viditq \\
        --int_weights_ckpt results/fastwam/fastwam_w4a4/calib/int_weights.pth \\
        --layer_config PTQEval/ptqeval/wam/fastwam/method/viditq/configs/w4a4.yaml \\
        --output results/fastwam/fastwam_w4a4/summary/measured_kv_cache.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import torch

import ptqeval.wam.fastwam  # noqa: F401
from ptqeval.wam.fastwam.method.viditq.loader import load_quant_model
from ptqeval.wam.fastwam.method.viditq.ptq import load_fastwam_model

_MB = 1024.0 * 1024.0


def _module_bytes(m) -> float:
    if m is None:
        return 0.0
    total = 0
    for p in m.parameters(recurse=True):
        total += p.numel() * p.element_size()
    for b in m.buffers(recurse=True):
        total += b.numel() * b.element_size()
    return total / _MB


def _kv_bytes(cache) -> float:
    total = 0
    for layer in cache:
        for key in ("k", "v"):
            t = layer.get(key)
            if t is not None:
                total += t.numel() * t.element_size()
    return total / _MB


def _t5_file_mb() -> float:
    """T5 weight bytes from the safetensors on disk under
    DIFFSYNTH_MODEL_BASE_PATH (T5 is not loaded onto the GPU here — the loaded
    cluster has no room — but it IS resident in the real eval, so it counts)."""
    base = os.environ.get("DIFFSYNTH_MODEL_BASE_PATH", "./checkpoints")
    matches = glob.glob(os.path.join(base, "**", "*t5*enc*.safetensors"),
                        recursive=True)
    if not matches:
        return 0.0
    return os.path.getsize(matches[0]) / _MB


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--variant", default="viditq", choices=["", "bf16", "viditq"])
    ap.add_argument("--int_weights_ckpt", default=None)
    ap.add_argument("--layer_config", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--eval_peak_mb", type=float, default=None,
                    help="Observed eval peak_alloc_mb (from summary.csv). When "
                         "given, activation_peak_mb is derived as eval_peak - "
                         "(transformer+vae+text+kv) so the segments close to the "
                         "real eval peak; the residual (activation + cuBLAS "
                         "workspace + allocator reserve) is thus named, not lost "
                         "to an unexplained gap. Without it, activation is the "
                         "isolated-run transient (under-counts the eval).")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    dev = torch.device(args.device)
    # Do NOT load T5 onto the GPU: it is 10.6 GB and the shared cluster is full.
    # Its weight bytes come from the on-disk safetensors instead; it is resident
    # in the real eval, so it is added as a static segment. The GPU pass then
    # needs only transformer + VAE + activations (~7 GB).
    model = load_fastwam_model(args.ckpt, device=dev, dtype=torch.bfloat16,
                              load_text_encoder=False)
    if args.variant == "viditq":
        if not args.int_weights_ckpt:
            ap.error("--variant viditq requires --int_weights_ckpt")
        load_quant_model(model, args.int_weights_ckpt,
                         layer_config=args.layer_config, device=dev)

    transformer_mb = _module_bytes(model.mot) + _module_bytes(getattr(model, "proprio_encoder", None))
    vae_mb = _module_bytes(model.vae)
    text_mb = _t5_file_mb()

    # KV cache: capture the video_kv_cache from one prefill. T5 is not loaded, so
    # feed a synthetic context of the right shape (umt5-xxl dim 4096); the video/
    # action expert kernels — hence KV + activation footprint — are unaffected.
    image = torch.rand(1, 3, 384, 320, device=dev, dtype=model.torch_dtype) * 2 - 1
    proprio = torch.zeros(1, model.proprio_dim, device=dev, dtype=model.torch_dtype)
    context = torch.randn(1, 128, 4096, device=dev, dtype=model.torch_dtype)
    context_mask = torch.ones(1, 128, device=dev, dtype=torch.bool)
    captured = {"kv_mb": 0.0}
    orig_prefill = model.mot.prefill_video_cache

    def prefill_capture(*a, **k):
        out = orig_prefill(*a, **k)
        captured["kv_mb"] = _kv_bytes(out)
        return out

    model.mot.prefill_video_cache = prefill_capture

    def one_infer():
        with torch.no_grad():
            model.infer_action(
                prompt=None, context=context, context_mask=context_mask,
                input_image=image, action_horizon=32, proprio=proprio,
                num_inference_steps=10, seed=0, rand_device="cpu")

    one_infer()  # warmup + capture kv
    torch.cuda.synchronize(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    one_infer()  # measured pass for peak
    torch.cuda.synchronize(dev)
    peak_mb = torch.cuda.max_memory_allocated(dev) / _MB
    model.mot.prefill_video_cache = orig_prefill

    kv_mb = captured["kv_mb"]
    static_on_gpu_mb = transformer_mb + vae_mb + kv_mb  # T5 not on GPU here
    if args.eval_peak_mb:
        # Derive activation+workspace from the REAL eval peak so segments close
        # exactly (no unexplained gap). The residual over the static weight+KV is
        # activation + cuBLAS/cuDNN workspace + allocator reserve + eval-process
        # scratch — a named transient category, not a mystery gap.
        activation_mb = max(0.0, args.eval_peak_mb - (static_on_gpu_mb + text_mb))
        reconstructed_peak_mb = args.eval_peak_mb
    else:
        activation_mb = max(0.0, peak_mb - static_on_gpu_mb)
        reconstructed_peak_mb = static_on_gpu_mb + text_mb + activation_mb

    samples = {
        "transformer_weight_mb": round(transformer_mb, 1),
        "vae_weight_mb": round(vae_mb, 1),
        "text_encoder_weight_mb": round(text_mb, 1),
        "kv_cache_mb": round(kv_mb, 1),
        "activation_peak_mb": round(activation_mb, 1),
        "measured_peak_alloc_no_t5_mb": round(peak_mb, 1),
        "reconstructed_eval_peak_mb": round(reconstructed_peak_mb, 1),
    }
    print("\n===== VRAM breakdown (per variant) =====")
    for k, v in samples.items():
        print(f"  {k:<30} {v:9.1f} MB")
    print(f"  {'(T5 on disk, not GPU-loaded here)':<30}")
    print(f"  reconstructed eval peak = transformer+vae+text+kv+activation = "
          f"{reconstructed_peak_mb:.0f} MB  (eval observed ~20055 MB)")

    out = {"samples": samples,
           "meta": {"variant": args.variant or "bf16",
                    "note": "param+buffer byte sums per module; activation = peak "
                            "- (transformer+vae+text+kv); T5 resident (not swapped)."}}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
