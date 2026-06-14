# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""WAM RoboTwin eval orchestrator (thin CLI).

Modes:
  smoke  -- 1 GPU, 1 task, test_num=1; sanity check.
  single -- 1 GPU, sequential over task list (or one task via --task_name).
  pool   -- N GPUs, worker queue; one server per task per GPU.

All real work lives in ptqeval.eval._pool_runner (shared with peer
calibration scripts -- see collect_calib_videos.py / derive_calib_ptq.py
under ptqeval/wam/lingbot_va/method/viditq/).

Required CLI args: --mode, --save_root. Defaults exist for everything
else; see --help.

Examples:
  python -m ptqeval.eval.run_eval \\
      --mode smoke \\
      --save_root /home/arash/EvalForWAMs/results/smoke_bf16

  python -m ptqeval.eval.run_eval \\
      --mode pool --test_num 25 \\
      --save_root /home/arash/EvalForWAMs/results/bf16

  python -m ptqeval.eval.run_eval \\
      --mode pool \\
      --variant viditq \\
      --variant_args .../runtime_args_w8a8_viditq.yaml \\
      --save_root /home/arash/EvalForWAMs/results/viditq_w8a8_viditq
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
    run_single,
    run_smoke,
)


def parse_args() -> Config:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )

    # --- mode + per-task ---
    p.add_argument("--mode", choices=["smoke", "single", "pool"], required=True)
    p.add_argument("--task_name",
                   help="smoke / single only. Default adjust_bottle for smoke.")
    p.add_argument("--test_num", type=int,
                   help="Episodes per task. Default: 1 for smoke, 25 for single/pool.")
    p.add_argument("--gpu_id", type=int, default=0,
                   help="Used by smoke / single.")
    p.add_argument("--seed", type=int, default=0)

    # --- pool / GPU detection ---
    p.add_argument("--min_free_mb", type=int, default=32000,
                   help="GPU is usable when free memory >= this (MB).")
    p.add_argument("--gpu_wait_timeout", type=int, default=0,
                   help="Seconds to wait for GPU memory before failing; 0 = forever.")
    p.add_argument("--gpu_poll_interval", type=int, default=30)
    p.add_argument("--rerun_all", action="store_true",
                   help="pool: include all tasks regardless of prior res.json.")

    # --- WAM + RoboTwin paths ---
    p.add_argument("--wam_name", default="lingbot_va",
                   help="Picks ptqeval.wam.<wam_name>.*.")
    p.add_argument("--wam_model_path", type=Path,
                   default=Path("/home/arash/EvalForWAMs/models/"
                                "lingbot-va-posttrain-robotwin"),
                   help="FP checkpoint dir passed to server.py --model_path.")
    p.add_argument("--robotwin_root", type=Path,
                   default=Path("/home/arash/EvalForWAMs/RoboTwin"),
                   help="RoboTwin simulator root.")

    # --- output ---
    p.add_argument("--save_root", type=Path, required=True,
                   help="Eval output root (visualization + logs + summary).")
    p.add_argument("--perf_log_dir", type=Path, default=None,
                   help="Per-call perf JSONL dir. Default: <save_root>/perf.")

    # --- variant ---
    p.add_argument("--variant", default="",
                   help="Quant method name; resolves to "
                        "ptqeval.wam.<wam>.method.<variant>.loader. "
                        "Empty -> bf16 baseline.")
    p.add_argument("--variant_args", type=Path, default=None,
                   help="Runtime args yaml (layer_config + int_weights_ckpt).")

    # --- task list ---
    p.add_argument("--task_list_name", default="SELECTED_15_TASKS",
                   help="Attribute in ptqeval.wam.<wam>.tasks to iterate.")
    p.add_argument("--task_config", default="demo_clean",
                   help="RoboTwin task_config yaml stem (demo_clean / "
                        "demo_randomized). Forwarded to eval_client. "
                        "demo_randomized adds background / light / table-height "
                        "randomization -> harder, eval-realistic.")

    # --- conda envs ---
    p.add_argument("--server_env", default="lingbot-jw")
    p.add_argument("--client_env", default="RoboTwin-jw")

    # --- optional op-level profiling ---
    p.add_argument("--profile_ops", action="store_true",
                   help="Forward --profile_ops to each server: wrap the "
                        "first --profile_n_calls infer() calls in "
                        "torch.profiler and dump op_profile.json (kernel "
                        "ms classified into linear / attention / other). "
                        "Off by default; profiler ~5-10x slowdown so "
                        "only enable with small --test_num.")
    p.add_argument("--profile_n_calls", type=int, default=5,
                   help="Number of post-warmup infer() calls to profile "
                        "per server when --profile_ops is on.")

    args = p.parse_args()

    save_root = args.save_root
    perf_log_dir = args.perf_log_dir if args.perf_log_dir else save_root / "perf"
    save_root.mkdir(parents=True, exist_ok=True)
    perf_log_dir.mkdir(parents=True, exist_ok=True)

    return Config(
        mode=args.mode,
        task_name=args.task_name,
        test_num=args.test_num,
        gpu_id=args.gpu_id,
        seed=args.seed,
        min_free_mb=args.min_free_mb,
        gpu_wait_timeout=args.gpu_wait_timeout,
        gpu_poll_interval=args.gpu_poll_interval,
        rerun_all=args.rerun_all,
        wam_name=args.wam_name,
        wam_model_path=str(args.wam_model_path),
        robotwin_root=str(args.robotwin_root),
        variant=args.variant,
        variant_args=str(args.variant_args) if args.variant_args else "",
        task_list_name=args.task_list_name,
        task_config=args.task_config,
        server_env=args.server_env,
        client_env=args.client_env,
        save_root=save_root,
        perf_log_dir=perf_log_dir,
        profile_ops=args.profile_ops,
        profile_n_calls=args.profile_n_calls,
    )


def main() -> int:
    # Line-buffer stdout/stderr so progress prints appear in the redirected
    # nohup log file in real time. Default Python buffering is block-mode
    # (4KB) when stdout is a file, which hides progress for orchestrator
    # runs that print a few lines per task transition.
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    cfg = parse_args()
    install_signal_handlers()
    try:
        if cfg.mode == "smoke":
            run_smoke(cfg)
        elif cfg.mode == "single":
            run_single(cfg)
        elif cfg.mode == "pool":
            run_pool(cfg)
    finally:
        cleanup_all_sessions()
    return 0


if __name__ == "__main__":
    sys.exit(main())
