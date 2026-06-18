# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Calibration step 1: collect bf16 inference videos in RoboTwin.

Peer to ptqeval.eval.run_eval. Uses the same _pool_runner primitives
(start_server / run_client_blocking / GPU pool worker loop) but with a
calibration-focused CLI surface (no --variant, no smoke/single modes,
no --mode at all -- always pool). The eval_client saves per-episode
obs_data_*.pt / actions_*.pt / latents_*.pt under
<save_root>/visualization/...; those .pt files are the raw calibration
corpus consumed by derive_calib_ptq.py to drive the FP transformer
offline.

Peer-script discipline: this script and run_eval.py do NOT call each
other. Both import the shared orchestration primitives from
ptqeval.eval._pool_runner. The calibration pipeline (collect_calib_videos
+ derive_calib_ptq + ptq.py) is fully independent of the eval pipeline.

Default: 50 task * 5 ep (CALIB_TASKS_ALL), demo_randomized config,
output under results/calib_capture/. Re-runs overwrite the same path
per the no-_v2 / no-backward-compat convention (ProjectDescription.txt
section "关键约定" item 7).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ptqeval.eval._pool_runner import (
    Config,
    cleanup_all_sessions,
    install_signal_handlers,
    run_pool,
)


def parse_args() -> Config:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- task scope ---
    p.add_argument("--task_list_name", default="CALIB_TASKS_ALL",
                   help="Attribute in ptqeval.wam.<wam>.tasks to iterate. "
                        "Default CALIB_TASKS_ALL (50 RoboTwin tasks).")
    p.add_argument("--test_num", type=int, default=5,
                   help="Episodes per task. Default 5.")
    p.add_argument("--task_config", default="demo_randomized",
                   help="RoboTwin task_config (demo_clean / demo_randomized). "
                        "Default demo_randomized so the calib corpus matches "
                        "the eval distribution.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rerun_all", action="store_true",
                   help="Include tasks regardless of prior res.json under save_root.")

    # --- output ---
    p.add_argument("--save_root", type=Path,
                   default=Path("results/calib_capture"),
                   help="Calib corpus root: visualization + perf + logs.")
    p.add_argument("--perf_log_dir", type=Path, default=None,
                   help="Per-call perf JSONL dir. Default: <save_root>/perf.")

    # --- pool / GPU detection ---
    p.add_argument("--min_free_mb", type=int, default=32000)
    p.add_argument("--gpu_wait_timeout", type=int, default=0,
                   help="0 = wait forever.")
    p.add_argument("--gpu_poll_interval", type=int, default=30)

    # --- WAM + RoboTwin paths ---
    p.add_argument("--wam_name", default="lingbot_va")
    p.add_argument("--wam_model_path", type=Path,
                   default=Path("models/"
                                "lingbot-va-posttrain-robotwin"))
    p.add_argument("--robotwin_root", type=Path,
                   default=Path("RoboTwin"))

    # --- conda envs ---
    p.add_argument("--server_env", default="lingbot-jw")
    p.add_argument("--client_env", default="RoboTwin-jw")

    args = p.parse_args()

    # Resolve to absolute paths immediately. eval_client.py does
    # os.chdir(ROBOTWIN_ROOT) on import, so any relative --save_root passed
    # to the client subprocess would land under RoboTwin/ instead of the
    # repo root (silently splitting res.json + visualization across the
    # wrong tree from server-side perf JSONL written by the server cwd).
    # Same fix as run_eval.py:139 (eval workflow audit 2026-06-17);
    # collect_calib_videos.py was missed in that pass.
    save_root = args.save_root.resolve()
    perf_log_dir = (args.perf_log_dir if args.perf_log_dir else save_root / "perf").resolve()
    save_root.mkdir(parents=True, exist_ok=True)
    perf_log_dir.mkdir(parents=True, exist_ok=True)

    return Config(
        mode="pool",                # fixed: only pool makes sense for calib
        task_name=None,
        test_num=args.test_num,
        gpu_id=0,                   # unused in pool mode
        seed=args.seed,
        min_free_mb=args.min_free_mb,
        gpu_wait_timeout=args.gpu_wait_timeout,
        gpu_poll_interval=args.gpu_poll_interval,
        rerun_all=args.rerun_all,
        wam_name=args.wam_name,
        wam_model_path=str(args.wam_model_path.resolve()),
        robotwin_root=str(args.robotwin_root.resolve()),
        variant="",                 # bf16: variant intentionally empty
        variant_args="",
        task_list_name=args.task_list_name,
        task_config=args.task_config,
        server_env=args.server_env,
        client_env=args.client_env,
        save_root=save_root,
        perf_log_dir=perf_log_dir,
    )


def main() -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    cfg = parse_args()
    install_signal_handlers()
    try:
        run_pool(cfg)
    finally:
        cleanup_all_sessions()
    return 0


if __name__ == "__main__":
    sys.exit(main())
