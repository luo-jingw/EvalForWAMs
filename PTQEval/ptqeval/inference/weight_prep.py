# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Offline w4a4 weight generation for deployment (M1).

Thin wrapper over method.viditq.ptq. w4a4 is weight-quantization +
QuaRoT, data-free (the w4a4 layer config sets smooth_quant=false and
declares no calib_data_path; activations are quantized per-token at
runtime), so this needs only the FP transformer -- NO calibration data.

Run on a resource-rich machine, ship the resulting int_weights.pth (an
overlay on the FP transformer, not standalone) to the device.

    from ptqeval.inference.weight_prep import build_int_weights
    build_int_weights("models/lingbot-va-posttrain-robotwin",
                      "int_weights_w4a4.pth")

or CLI:

    python -m ptqeval.inference.weight_prep \\
        --model_path models/lingbot-va-posttrain-robotwin \\
        --output int_weights_w4a4.pth
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

# Default layer config: the packaged w4a4 spec (only w4a4 is supported for
# deployment). Resolved relative to this file so the deploy package is
# self-contained regardless of cwd.
_DEFAULT_W4A4 = os.path.join(os.path.dirname(__file__), "configs", "w4a4.yaml")


def build_int_weights(
    model_path: str,
    output: str,
    layer_config: str = _DEFAULT_W4A4,
    device: str = "cuda:0",
    load_dtype: str = "bf16",
) -> str:
    """FP transformer -> int_weights.pth via ptq.py (subprocess, same as
    derive_calib_ptq drives it). `model_path` is the model root
    (containing transformer/); the FP transformer dir is model_path/
    transformer. Returns the output path. Raises on ptq failure."""
    transformer_dir = os.path.join(model_path, "transformer")
    if not os.path.isdir(transformer_dir):
        raise FileNotFoundError(f"no transformer/ under {model_path}")
    if not os.path.isfile(layer_config):
        raise FileNotFoundError(f"layer_config not found: {layer_config}")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    cmd = [
        sys.executable, "-m", "ptqeval.wam.lingbot_va.method.viditq.ptq",
        "--model_path", transformer_dir,
        "--layer_config", layer_config,
        "--output", output,
        "--device", device,
        "--load_dtype", load_dtype,
    ]
    print("[weight_prep] " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    size_mb = os.path.getsize(output) / (1024.0 * 1024.0)
    print(f"[weight_prep] wrote {output} ({size_mb:.1f} MB)")
    return output


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_path", required=True,
                    help="FP model root (contains transformer/).")
    ap.add_argument("--output", required=True,
                    help="Output int_weights.pth path.")
    ap.add_argument("--layer_config", default=_DEFAULT_W4A4,
                    help="w4a4 layer config yaml (default: packaged w4a4).")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--load_dtype", default="bf16",
                    choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()
    build_int_weights(args.model_path, args.output, args.layer_config,
                      args.device, args.load_dtype)
    return 0


if __name__ == "__main__":
    sys.exit(main())
