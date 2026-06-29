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
      --save_root results/smoke_bf16

  python -m ptqeval.eval.run_eval \\
      --mode pool \\
      --save_root results/bf16
  # Defaults: task_list_name=ALL_TASKS (50 tasks), test_num=100 per
  # task = 5000 episodes/variant; matches the Phase 42 5-way cross_summary
  # convention. For the legacy 15-task x 25-ep scope add
  # --task_list_name SELECTED_15_TASKS --test_num 25.

  python -m ptqeval.eval.run_eval \\
      --mode pool \\
      --variant viditq \\
      --variant_args .../runtime_args_w8a8.yaml \\
      --save_root results/viditq_w8a8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ptqeval.eval._pool_runner import (
    Config,
    _set_cleanup_context,
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
                   help="Episodes per task. Default: 1 for smoke, 100 for "
                        "single/pool (promoted from 25 on 2026-06-26 to "
                        "match the paper-style full-coverage eval — see "
                        "_pool_runner.run_pool / run_single).")
    p.add_argument("--gpu_id", type=int, default=0,
                   help="Used by smoke / single.")
    p.add_argument("--seed", type=int, default=0)

    # --- pool / GPU detection ---
    p.add_argument("--min_free_mb", type=int, default=33000,
                   help="GPU is usable when free memory >= this (MB). "
                        "Default 33 GB matches bf16 init_peak (~12 GB) + "
                        "KV cache (~13 GB) + activations (~7 GB) headroom. "
                        "Used to need ~40 GB while the text encoder was "
                        "resident; the v3 swap pattern frees those 11 GB "
                        "so 33 GB is a comfortable floor.")
    p.add_argument("--gpu_wait_timeout", type=int, default=0,
                   help="Seconds to wait for GPU memory before failing; 0 = forever.")
    p.add_argument("--gpu_poll_interval", type=int, default=30)
    p.add_argument("--rerun_all", action="store_true",
                   help="pool: include all tasks regardless of prior res.json.")
    p.add_argument("--gpus", type=str, default="",
                   help="pool: comma-separated GPU ids to consider "
                        "(e.g. '0,2,5'). Default = scan all 8. "
                        "Still subject to --min_free_mb filter.")
    p.add_argument("--max_gpus", type=int, default=None,
                   help="pool: cap to at most N GPUs (taken from the "
                        "front of the usable-list after --gpus + "
                        "--min_free_mb filtering). Default = no cap.")

    # --- WAM + RoboTwin paths ---
    p.add_argument("--wam_name", default="lingbot_va",
                   help="Picks ptqeval.wam.<wam_name>.*.")
    p.add_argument("--wam_model_path", type=Path,
                   default=Path("models/"
                                "lingbot-va-posttrain-robotwin"),
                   help="FP checkpoint dir passed to server.py --model_path.")
    p.add_argument("--robotwin_root", type=Path,
                   default=Path("RoboTwin"),
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
    p.add_argument("--task_list_name", default="ALL_TASKS",
                   help="Attribute in ptqeval.wam.<wam>.tasks to iterate. "
                        "Default ALL_TASKS (50 RoboTwin tasks) was "
                        "promoted from SELECTED_15_TASKS on 2026-06-26 to "
                        "match the paper-style full-coverage eval used by "
                        "the 5-way cross_summary (Phase 42). Pass "
                        "--task_list_name SELECTED_15_TASKS for the older "
                        "15-task subset (Phase 39 / 40 historical scope).")
    p.add_argument("--task_config", default="demo_clean",
                   help="RoboTwin task_config yaml stem (demo_clean / "
                        "demo_randomized). Forwarded to eval_client. "
                        "demo_randomized adds background / light / table-height "
                        "randomization -> harder, eval-realistic.")

    # --- conda envs ---
    p.add_argument("--server_env", default="lingbot-jw")
    p.add_argument("--client_env", default="RoboTwin-jw")

    # --- op-level profiling ---
    # ON BY DEFAULT (since the workflow audit on 2026-06-17 showed that the
    # earlier _speed sidecar pattern was a workaround, not the intended
    # path). The profiler only wraps the first --profile_n_calls (default 5)
    # post-warmup infer() calls; the remaining N x 25 ep x ~10 calls/ep run
    # un-instrumented, so SR data is unaffected. One eval run now produces
    # both the SR table AND op_breakdown.png without a separate test_num=5
    # follow-up. Pass --no_profile_ops to disable explicitly.
    p.add_argument("--profile_ops", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Wrap the first --profile_n_calls post-warmup infer "
                        "calls in torch.profiler and dump op_profile.json "
                        "(kernel ms classified into linear / attention / "
                        "memcpy / other). Default ON; remaining calls run "
                        "un-instrumented so SR is unaffected. Disable with "
                        "--no-profile_ops.")
    p.add_argument("--profile_n_calls", type=int, default=5,
                   help="Number of post-warmup infer() calls to profile "
                        "per server when --profile_ops is on.")
    p.add_argument("--save_visualization", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Save per-episode obs_data/latents/actions .pt under "
                        "<save_root>/visualization/. Default OFF (raw camera "
                        "frames ~20 MB/ep, not needed for SR/latency/memory). "
                        "Pass --save_visualization to opt in for inspection; "
                        "calibration (collect_calib_videos) always saves.")

    args = p.parse_args()

    # Resolve to absolute paths immediately. eval_client.py does
    # os.chdir(ROBOTWIN_ROOT) on import, so any relative --save_root passed
    # to the client subprocess would land under RoboTwin/ instead of the
    # repo root (silently splitting res.json + visualization across the
    # wrong tree from server-side perf JSONL written by the server cwd).
    # Same logic for perf_log_dir / wam_model_path / variant_args / robotwin_root.
    save_root = args.save_root.resolve()
    perf_log_dir = (args.perf_log_dir if args.perf_log_dir else save_root / "perf").resolve()
    save_root.mkdir(parents=True, exist_ok=True)
    perf_log_dir.mkdir(parents=True, exist_ok=True)

    gpu_ids: list[int] | None = None
    if args.gpus:
        gpu_ids = [int(g.strip()) for g in args.gpus.split(",") if g.strip()]
        if not gpu_ids:
            raise ValueError(f"--gpus parsed to empty list from '{args.gpus}'")

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
        wam_model_path=str(args.wam_model_path.resolve()),
        robotwin_root=str(args.robotwin_root.resolve()),
        variant=args.variant,
        variant_args=str(args.variant_args.resolve()) if args.variant_args else "",
        task_list_name=args.task_list_name,
        task_config=args.task_config,
        server_env=args.server_env,
        client_env=args.client_env,
        save_root=save_root,
        perf_log_dir=perf_log_dir,
        profile_ops=args.profile_ops,
        profile_n_calls=args.profile_n_calls,
        save_visualization=args.save_visualization,
        gpu_ids=gpu_ids,
        max_gpus=args.max_gpus,
    )


def _ensure_kv_cache_measurement(cfg: Config) -> None:
    """One-shot baseline GPU memory measurement (~30 sec) before the
    pool spawns server processes. Writes <save_root>/measured_kv_cache.json
    consumed by calc_cross_ckpt's memory_breakdown chart (replaces the
    earlier hardcoded reverse-fit constants for KV / VAE / xfmr_bf16
    with real torch.cuda.memory_allocated() deltas).

    Skipped when the JSON already exists (idempotent resume) or when the
    user passed --skip_kv_measurement. Errors are logged but do not
    abort the eval (chart falls back to theoretical defaults).

    Runs in a subprocess to isolate the measurement's CUDA context from
    the pool servers; otherwise the lingering allocator state would
    leak ~25 GB into GPU 0 before any worker starts.
    """
    import subprocess
    target = cfg.save_root / "measured_kv_cache.json"
    if target.exists():
        print(f"[run_eval] measured_kv_cache.json already at {target}, skipping")
        return
    if getattr(cfg, "skip_kv_measurement", False):
        print(f"[run_eval] skipping KV cache measurement (--skip_kv_measurement)")
        return
    print(f"[run_eval] running KV cache measurement -> {target} (~30 sec)")
    cmd = [
        sys.executable, "-m", "ptqeval.eval.measure_kv_cache",
        "--model_path", cfg.wam_model_path,
        "--output", str(target),
        "--device", "cuda:0",
    ]
    # When this eval run quantizes (--variant given), measure the
    # quantized model footprint instead of bf16 baseline. Captures real
    # per-variant transformer weight + activation peak.
    if cfg.variant:
        cmd += ["--variant", cfg.variant]
        if cfg.variant_args:
            cmd += ["--variant_args", cfg.variant_args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[run_eval] WARNING: KV cache measurement failed "
              f"(continuing; chart will fall back to theoretical defaults):\n"
              f"  stderr tail: {result.stderr[-400:]}")
    else:
        print(f"[run_eval] measurement OK; wrote {target}")


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
    _set_cleanup_context(cfg)
    install_signal_handlers()
    try:
        # Phase: one-shot baseline KV cache + weight measurement before
        # pool/single/smoke launch a server. Idempotent (skips if JSON
        # already at save_root). Smoke skips by default (too short for
        # the cost to matter; smoke output isn't consumed by cross_ckpt).
        if cfg.mode in ("single", "pool"):
            _ensure_kv_cache_measurement(cfg)

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
