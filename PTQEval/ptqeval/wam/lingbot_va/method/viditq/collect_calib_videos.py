# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Calibration step 1: collect bf16 inference videos in RoboTwin.

Thin wrapper around ptqeval.eval.run_eval (pool mode, bf16, no quant
variant). The eval client saves per-episode obs_data_*.pt /
actions_*.pt / latents_*.pt under <save_root>/visualization/...; those
.pt files are the raw calibration corpus consumed by derive_calib_ptq.py
to drive the FP transformer offline (no RoboTwin sim required at
PTQ-time).

Why a separate script: keeps run_eval's surface focused on eval, makes
the "collect videos -> derive calib + PTQ" two-step flow obvious in the
codebase, and lets calibration be re-run independently of any eval.

Default: 50 task * 5 ep (CALIB_TASKS_ALL), demo_randomized config,
output under results/calib_capture/. Re-runs overwrite the same path
per the no-_v2 / no-backward-compat convention (ProjectDescription.txt
section "关键约定" item 7).
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task_list_name", default="CALIB_TASKS_ALL",
                   help="Attribute in ptqeval.wam.<wam>.tasks to iterate.")
    p.add_argument("--test_num", type=int, default=5,
                   help="Episodes per task.")
    p.add_argument("--task_config", default="demo_randomized",
                   help="RoboTwin task_config (demo_clean / demo_randomized).")
    p.add_argument("--save_root", type=Path,
                   default=Path("/home/arash/EvalForWAMs/results/calib_capture"),
                   help="Eval output root: visualization + perf + logs.")
    p.add_argument("--wam_name", default="lingbot_va")
    p.add_argument("--min_free_mb", type=int, default=32000)
    args = p.parse_args()

    cmd = [
        sys.executable, "-m", "ptqeval.eval.run_eval",
        "--mode", "pool",
        "--test_num", str(args.test_num),
        "--task_list_name", args.task_list_name,
        "--task_config", args.task_config,
        "--save_root", str(args.save_root),
        "--wam_name", args.wam_name,
        "--min_free_mb", str(args.min_free_mb),
    ]
    print("[collect_calib_videos] " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
