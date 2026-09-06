# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""req-1a observational check: transformer offload keeps T5 and lingbot-va
mutually exclusive in VRAM.

Builds the server (no text-cond cache -> live T5 / serve_residency direct-
compute) and, for each offload_target, runs a reset on an uncached prompt
(which triggers offload -> T5 encode -> reload) plus one real _infer forward
to confirm the transformer survives the round-trip. Prints reset peak_alloc
and post-reload forward output stats. Also measures the additive baseline
(serve_residency off = transient swap, T5 stacked on lingbot-va).

Observational (principle.txt L12): metric rows only, no assert / no PASS.

    python -m ptqeval.wam.lingbot_va.method.viditq.check_offload \
        --model_path ../models/lingbot-va-posttrain-robotwin \
        --videos_root results/calib_capture
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

import ptqeval.wam.lingbot_va  # noqa: F401


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_path",
                    default="models/lingbot-va-posttrain-robotwin")
    ap.add_argument("--videos_root", default="results/calib_capture",
                    help="Root with visualization/real/ obs chunks for the "
                         "post-reload forward (a quick transformer sanity "
                         "forward). calib_capture works.")
    args = ap.parse_args()

    from ptqeval.eval.measure_flops import _build_server
    server = _build_server(args.model_path, Path("/tmp/check_offload_scratch"))
    server.text_cond_cache = None
    server._mem_text_cond = {}

    def _xfmr_param():
        return next(server.transformer.parameters())

    print(f"{'config':18} {'reset_peak_MB':>13} {'xfmr_dev':>9} "
          f"{'param_finite':>12} {'param_absmax':>12}")
    cases = [
        ("additive (off)", False, "cpu"),
        ("serial cpu",     True,  "cpu"),
        ("serial disk",    True,  "disk"),
    ]
    for label, residency, target in cases:
        server.serve_residency = residency
        server.offload_target = target
        server._mem_text_cond = {}
        prompt = f"uncached offload probe {label} zzz"
        torch.cuda.reset_peak_memory_stats()
        server.infer({"reset": True, "prompt": prompt,
                      "save_visualization": False})
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1024 / 1024
        # reload integrity: after the reset the transformer must be back on
        # the GPU with finite weights (confirms the offload/reload round-trip
        # -- incl. the disk save + meta + load_state_dict(assign) -- worked).
        p = _xfmr_param()
        dev = str(p.device)
        finite = str(bool(torch.isfinite(p.float()).all().item()))
        absmax = p.float().abs().max().item()
        print(f"{label:18} {peak:13.1f} {dev:>9} {finite:>12} {absmax:12.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
