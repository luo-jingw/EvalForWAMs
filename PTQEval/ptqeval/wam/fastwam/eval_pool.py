"""Multi-GPU task pool over the single-task fastwam eval runner.

One worker thread per GPU pulls tasks off a shared queue and runs
`python -m ptqeval.wam.fastwam.eval_runner` as a subprocess (peer scripts are
invoked, never imported). Each task therefore gets a fresh process: RoboTwin's
eval_policy.py loads the model per invocation, so no state leaks across tasks.

A failing task is logged and the pool continues; the exit code is non-zero if
any task failed.

Per-task stdout goes to <save_root>/logs/<task>.log. SR (res.json) and perf
JSONL land under <save_root> via the runner, ready for
`python -m ptqeval.eval.aggregator`.

CLI:
    python -m ptqeval.wam.fastwam.eval_pool \\
        --gpus 0,1,2,3 --task_list ALL_TASKS --eval_num_episodes 100 \\
        --ckpt models/robotwin2.0-fastwam/robotwin_uncond_3cam_384.pt \\
        --dataset_stats models/robotwin2.0-fastwam/robotwin_uncond_3cam_384_dataset_stats.json \\
        --variant viditq \\
        --int_weights_ckpt results/fastwam/fastwam_w4a4/calib/int_weights.pth \\
        --layer_config PTQEval/ptqeval/wam/fastwam/method/viditq/configs/w4a4.yaml \\
        --save_root results/fastwam/fastwam_w4a4
"""
from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import time
from typing import Optional

from ptqeval.wam.fastwam.tasks import ALL_TASKS, SELECTED_15_TASKS, SMOKE_5_TASKS

_TASK_LISTS: dict[str, list[str]] = {
    "ALL_TASKS": ALL_TASKS,
    "SELECTED_15_TASKS": SELECTED_15_TASKS,
    "SMOKE_5_TASKS": SMOKE_5_TASKS,
}

_print_lock = threading.Lock()


def _log(msg: str) -> None:
    with _print_lock:
        print(f"[eval_pool {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _run_one(task: str, gpu: int, args: argparse.Namespace, log_dir: str) -> int:
    cmd = [
        sys.executable, "-u", "-m", "ptqeval.wam.fastwam.eval_runner",
        "--task", task,
        "--gpu", str(gpu),
        "--ckpt", args.ckpt,
        "--dataset_stats", args.dataset_stats,
        "--robotwin_root", args.robotwin_root,
        "--task_config", args.task_config,
        "--instruction_type", args.instruction_type,
        "--eval_num_episodes", str(args.eval_num_episodes),
        "--seed", str(args.seed),
        "--save_root", args.save_root,
        "--variant", args.variant,
    ]
    if args.int_weights_ckpt:
        cmd += ["--int_weights_ckpt", args.int_weights_ckpt]
    if args.layer_config:
        cmd += ["--layer_config", args.layer_config]
    if args.calib_out:
        cmd += ["--calib_out", args.calib_out, "--profile_ops", "false"]

    log_path = os.path.join(log_dir, f"{task}.log")
    with open(log_path, "w") as fp:
        proc = subprocess.run(cmd, stdout=fp, stderr=subprocess.STDOUT)
    return proc.returncode


def _worker(gpu: int, q: "queue.Queue[str]", args: argparse.Namespace,
            log_dir: str, results: dict[str, int]) -> None:
    while True:
        try:
            task = q.get_nowait()
        except queue.Empty:
            return
        _log(f"gpu{gpu} start {task}")
        t0 = time.time()
        rc = _run_one(task, gpu, args, log_dir)
        results[task] = rc
        status = "ok" if rc == 0 else f"FAILED rc={rc}"
        _log(f"gpu{gpu} done  {task} ({status}, {time.time() - t0:.0f}s), "
             f"{q.qsize()} left")
        q.task_done()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", required=True, help="Comma-separated GPU ids, e.g. 0,1,2,3")
    ap.add_argument("--task_list", default="ALL_TASKS", choices=sorted(_TASK_LISTS))
    ap.add_argument("--tasks", default=None,
                    help="Comma-separated task names; overrides --task_list.")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset_stats", required=True)
    ap.add_argument("--robotwin_root", default="FastWAM/third_party/RoboTwin")
    ap.add_argument("--task_config", default="demo_randomized")
    ap.add_argument("--instruction_type", default="unseen")
    ap.add_argument("--eval_num_episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_root", required=True)
    ap.add_argument("--variant", default="", choices=["", "bf16", "viditq"])
    ap.add_argument("--int_weights_ckpt", default=None)
    ap.add_argument("--layer_config", default=None)
    ap.add_argument("--calib_out", default=None,
                    help="On-policy SmoothQuant calib: run bf16 + absmax hooks, "
                         "all tasks merge per-channel absmax into this one path "
                         "(via _CalibState flock). Use with --variant bf16 "
                         "--eval_num_episodes 1.")
    args = ap.parse_args()

    if args.calib_out and args.variant == "viditq":
        ap.error("--calib_out needs the FP model; use --variant bf16 (or '').")

    gpus = [int(g) for g in args.gpus.split(",") if g.strip() != ""]
    tasks = ([t.strip() for t in args.tasks.split(",") if t.strip()]
             if args.tasks else list(_TASK_LISTS[args.task_list]))
    if not gpus:
        ap.error("--gpus resolved to an empty list")
    if not tasks:
        ap.error("task list resolved to empty")

    log_dir = os.path.join(args.save_root, "logs")
    os.makedirs(log_dir, exist_ok=True)

    q: "queue.Queue[str]" = queue.Queue()
    for t in tasks:
        q.put(t)
    results: dict[str, int] = {}

    _log(f"variant={args.variant or 'bf16'} tasks={len(tasks)} gpus={gpus} "
         f"episodes/task={args.eval_num_episodes} config={args.task_config} "
         f"instruction={args.instruction_type}")
    _log(f"save_root={args.save_root} logs={log_dir}")

    t_start = time.time()
    threads = [threading.Thread(target=_worker, args=(g, q, args, log_dir, results),
                                daemon=False, name=f"gpu{g}") for g in gpus]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    failed = {t: rc for t, rc in results.items() if rc != 0}
    _log(f"pool done in {time.time() - t_start:.0f}s; "
         f"{len(results) - len(failed)}/{len(tasks)} ok, {len(failed)} failed")
    if failed:
        _log(f"failed tasks: {sorted(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
