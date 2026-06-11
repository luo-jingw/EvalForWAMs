# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""WAM RoboTwin eval orchestrator. Replaces run_eval.sh.

Modes:
  smoke  -- 1 GPU, 1 task, test_num=1; sanity check.
  single -- 1 GPU, sequential over task list (or one task via --task_name).
  pool   -- N GPUs, worker queue; one server per task per GPU.

Process lifecycle fix vs run_eval.sh:
  Each server is launched in a NEW POSIX session (subprocess.Popen with
  start_new_session=True). When the task finishes, os.killpg(pid, SIGTERM)
  takes down the whole launcher + torch.distributed.run + worker tree.
  The bash version only killed the launcher PID, leaking the
  torch.distributed.run grandchild on GPU; the leaked process kept the
  model resident in VRAM, blocking the next pool worker from acquiring
  the GPU until OOM-style failure cascaded.

Env vars (same names as the bash version):
  WAM_NAME, WAM_MODEL_PATH, ROBOTWIN_ROOT, SAVE_ROOT, PERF_LOG_DIR,
  VARIANT, VARIANT_ARGS, CALIBRATE_OUT, TASK_LIST_NAME,
  SERVER_ENV, CLIENT_ENV, MIN_FREE_MB, GPU_WAIT_TIMEOUT, GPU_POLL_INTERVAL.

Invocation:
  python -m ptqeval.eval.run_eval --mode pool --test_num 5
  python -m ptqeval.eval.run_eval --mode smoke
  python -m ptqeval.eval.run_eval --mode single --task_name adjust_bottle
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    mode: str
    task_name: Optional[str]
    test_num: Optional[int]
    gpu_id: int
    seed: int
    min_free_mb: int
    gpu_wait_timeout: int
    gpu_poll_interval: int
    rerun_all: bool
    wam_name: str
    wam_model_path: str
    robotwin_root: str
    variant: str
    variant_args: str
    calibrate_out: str
    task_list_name: str
    server_env: str
    client_env: str
    save_root: Path
    perf_log_dir: Path


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return int(v)


def parse_args() -> Config:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__.split("\n\n", 1)[0],
        epilog="Env vars: WAM_NAME, WAM_MODEL_PATH, ROBOTWIN_ROOT, SAVE_ROOT, "
               "PERF_LOG_DIR, VARIANT, VARIANT_ARGS, CALIBRATE_OUT, "
               "TASK_LIST_NAME (default SELECTED_15_TASKS), SERVER_ENV, "
               "CLIENT_ENV, MIN_FREE_MB, GPU_WAIT_TIMEOUT, GPU_POLL_INTERVAL.",
    )
    p.add_argument("--mode", choices=["smoke", "single", "pool"], required=True)
    p.add_argument("--task_name", help="smoke / single. Default adjust_bottle for smoke.")
    p.add_argument("--test_num", type=int, help="Episodes per task. Smoke=1, single/pool=25.")
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min_free_mb", type=int, default=_env_int("MIN_FREE_MB", 40000))
    p.add_argument("--gpu_wait_timeout", type=int, default=_env_int("GPU_WAIT_TIMEOUT", 0),
                   help="0 = wait forever.")
    p.add_argument("--gpu_poll_interval", type=int, default=_env_int("GPU_POLL_INTERVAL", 30))
    p.add_argument("--rerun_all", action="store_true",
                   help="pool: include all tasks regardless of prior res.json.")
    args = p.parse_args()

    wam_name = os.environ.get("WAM_NAME", "lingbot_va")
    variant = os.environ.get("VARIANT", "")
    variant_args = os.environ.get("VARIANT_ARGS", "")

    save_root_env = os.environ.get("SAVE_ROOT")
    if save_root_env:
        save_root = Path(save_root_env)
    elif variant and variant_args:
        tag = Path(variant_args).stem
        save_root = Path(f"/home/arash/EvalForWAMs/results/{variant}_{tag}")
    elif variant:
        save_root = Path(f"/home/arash/EvalForWAMs/results/{variant}")
    else:
        save_root = Path("/home/arash/EvalForWAMs/results/bf16")

    perf_log_dir_env = os.environ.get("PERF_LOG_DIR")
    perf_log_dir = Path(perf_log_dir_env) if perf_log_dir_env else save_root / "perf"
    save_root.mkdir(parents=True, exist_ok=True)
    perf_log_dir.mkdir(parents=True, exist_ok=True)

    return Config(
        mode=args.mode, task_name=args.task_name, test_num=args.test_num,
        gpu_id=args.gpu_id, seed=args.seed, min_free_mb=args.min_free_mb,
        gpu_wait_timeout=args.gpu_wait_timeout,
        gpu_poll_interval=args.gpu_poll_interval, rerun_all=args.rerun_all,
        wam_name=wam_name,
        wam_model_path=os.environ.get(
            "WAM_MODEL_PATH",
            "/home/arash/EvalForWAMs/models/lingbot-va-posttrain-robotwin"),
        robotwin_root=os.environ.get(
            "ROBOTWIN_ROOT", "/home/arash/EvalForWAMs/RoboTwin"),
        variant=variant, variant_args=variant_args,
        calibrate_out=os.environ.get("CALIBRATE_OUT", ""),
        task_list_name=os.environ.get("TASK_LIST_NAME", "SELECTED_15_TASKS"),
        server_env=os.environ.get("SERVER_ENV", "lingbot-jw"),
        client_env=os.environ.get("CLIENT_ENV", "RoboTwin-jw"),
        save_root=save_root, perf_log_dir=perf_log_dir,
    )


# ---------------------------------------------------------------------------
# Task list
# ---------------------------------------------------------------------------

def load_tasks(cfg: Config) -> list[str]:
    mod = importlib.import_module(f"ptqeval.wam.{cfg.wam_name}.tasks")
    return list(getattr(mod, cfg.task_list_name))


# ---------------------------------------------------------------------------
# GPU monitoring
# ---------------------------------------------------------------------------

def gpu_free_mb(gpu: int) -> int:
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
         "-i", str(gpu)],
        capture_output=True, text=True, check=False)
    try:
        return int(r.stdout.strip())
    except (ValueError, AttributeError):
        return -1


def wait_for_gpu(cfg: Config, gpu: int) -> bool:
    elapsed = 0
    while True:
        mb = gpu_free_mb(gpu)
        if mb < 0:
            print(f"[wait_for_gpu] non-numeric from nvidia-smi on GPU {gpu}", file=sys.stderr)
            return False
        if mb >= cfg.min_free_mb:
            print(f"[wait_for_gpu] GPU {gpu} has {mb} MB free (>= {cfg.min_free_mb} MB). Proceeding.")
            return True
        print(f"[wait_for_gpu] GPU {gpu} has {mb} MB free, need {cfg.min_free_mb}. "
              f"Sleeping {cfg.gpu_poll_interval}s (elapsed {elapsed}s).")
        time.sleep(cfg.gpu_poll_interval)
        elapsed += cfg.gpu_poll_interval
        if cfg.gpu_wait_timeout > 0 and elapsed >= cfg.gpu_wait_timeout:
            print(f"[wait_for_gpu] Timeout after {elapsed}s on GPU {gpu}", file=sys.stderr)
            return False


def wait_for_port(port: int, timeout: float = 600.0) -> bool:
    elapsed = 0.0
    while elapsed < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2.0):
                return True
        except OSError:
            time.sleep(2)
            elapsed += 2
    print(f"[wait_for_port] timeout on port {port}", file=sys.stderr)
    return False


# ---------------------------------------------------------------------------
# Conda + subprocess
# ---------------------------------------------------------------------------

_CONDA_BASE: Optional[str] = None


def conda_base() -> str:
    global _CONDA_BASE
    if _CONDA_BASE is None:
        r = subprocess.run(["conda", "info", "--base"],
                            capture_output=True, text=True, check=True)
        _CONDA_BASE = r.stdout.strip()
    return _CONDA_BASE


def launch_in_session(bash_cmd: str, log_path: Path) -> subprocess.Popen:
    """Spawn `bash -c bash_cmd` in a new POSIX session. The returned PID is
    the new session/group leader; os.killpg on it cleans up the whole
    launcher + torch.distributed.run + worker tree."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logf = open(log_path, "w")
    return subprocess.Popen(
        ["bash", "-c", bash_cmd],
        stdout=logf, stderr=subprocess.STDOUT,
        env=os.environ.copy(),
        start_new_session=True, close_fds=True,
    )


def kill_session(pid: int, timeout: float = 5.0) -> None:
    """SIGTERM the process group; escalate to SIGKILL after timeout."""
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    start = time.time()
    while time.time() - start < timeout:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.2)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def start_server(cfg: Config, gpu: int, port: int, master_port: int,
                  task_name: str, server_log: Path) -> subprocess.Popen:
    extra: list[str] = []
    if cfg.variant:
        extra.append(f"--variant {cfg.variant}")
        if cfg.variant_args:
            extra.append(f"--variant_args {cfg.variant_args}")
    if cfg.calibrate_out:
        extra.append(f"--calibrate_out {cfg.calibrate_out}")
    extra_cli = " ".join(extra)

    cmd = (
        f"source {conda_base()}/etc/profile.d/conda.sh\n"
        f"conda activate {cfg.server_env}\n"
        f"exec env CUDA_VISIBLE_DEVICES={gpu} "
        f"         PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        f"  python -m torch.distributed.run"
        f"    --nproc_per_node 1"
        f"    --master_port {master_port}"
        f"    --module ptqeval.wam.{cfg.wam_name}.server"
        f"    --config-name robotwin"
        f"    --port {port}"
        f"    --save_root {cfg.save_root}/visualization"
        f"    --perf_log_dir {cfg.perf_log_dir}"
        f"    --perf_task_name {task_name}"
        f"    --model_path {cfg.wam_model_path}"
        f"    {extra_cli}\n"
    )
    return launch_in_session(cmd, server_log)


def run_client_blocking(cfg: Config, gpu: int, task_name: str, port: int,
                         test_num: int, client_log: Path) -> int:
    cmd = (
        f"source {conda_base()}/etc/profile.d/conda.sh\n"
        f"conda activate {cfg.client_env}\n"
        f"export LD_LIBRARY_PATH=/usr/lib64:/usr/lib:${{LD_LIBRARY_PATH:-}}\n"
        f"export ROBOTWIN_ROOT={cfg.robotwin_root}\n"
        f"exec env CUDA_VISIBLE_DEVICES={gpu}"
        f"         PYTHONWARNINGS=ignore::UserWarning"
        f"         XLA_PYTHON_CLIENT_MEM_FRACTION=0.9"
        f"  python -m ptqeval.wam.{cfg.wam_name}.eval_client"
        f"    --config {cfg.robotwin_root}/policy/ACT/deploy_policy.yml"
        f"    --overrides"
        f"    --task_name {task_name}"
        f"    --task_config demo_clean"
        f"    --train_config_name 0"
        f"    --model_name 0"
        f"    --ckpt_setting 0"
        f"    --seed {cfg.seed}"
        f"    --policy_name ACT"
        f"    --save_root {cfg.save_root}"
        f"    --video_guidance_scale 5"
        f"    --action_guidance_scale 1"
        f"    --test_num {test_num}"
        f"    --port {port}\n"
    )
    proc = launch_in_session(cmd, client_log)
    with _SESSIONS_LOCK:
        _SESSIONS.add(proc.pid)
    try:
        proc.wait()
    finally:
        with _SESSIONS_LOCK:
            _SESSIONS.discard(proc.pid)
    return proc.returncode


# ---------------------------------------------------------------------------
# Session bookkeeping for cleanup-on-signal
# ---------------------------------------------------------------------------

_SESSIONS: set[int] = set()
_SESSIONS_LOCK = threading.Lock()


def cleanup_all_sessions() -> None:
    with _SESSIONS_LOCK:
        pids = list(_SESSIONS)
        _SESSIONS.clear()
    for pid in pids:
        kill_session(pid)


def install_signal_handlers() -> None:
    def handler(sig, _frame):
        print(f"\n[run_eval] received signal {sig}; cleaning up sessions...",
              file=sys.stderr)
        cleanup_all_sessions()
        sys.exit(128 + sig)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, handler)


# ---------------------------------------------------------------------------
# Per-task and pool execution
# ---------------------------------------------------------------------------

def run_one_task(cfg: Config, task: str, gpu: int, port: int, master_port: int,
                  test_num: int, tag: str) -> None:
    log_dir = cfg.save_root / "logs" / tag
    log_dir.mkdir(parents=True, exist_ok=True)
    server_log = log_dir / f"server_{task}.log"
    client_log = log_dir / f"client_{task}.log"

    print(f"[run_one_task] task={task} gpu={gpu} port={port} test_num={test_num}")
    if not wait_for_gpu(cfg, gpu):
        return
    sp = start_server(cfg, gpu, port, master_port, task, server_log)
    with _SESSIONS_LOCK:
        _SESSIONS.add(sp.pid)
    try:
        if not wait_for_port(port):
            return
        run_client_blocking(cfg, gpu, task, port, test_num, client_log)
    finally:
        kill_session(sp.pid)
        with _SESSIONS_LOCK:
            _SESSIONS.discard(sp.pid)


def task_needs_run(cfg: Config, task: str, target: int) -> bool:
    f = cfg.save_root / "stseed-10000" / "metrics" / task / "res.json"
    if not f.exists():
        return True
    try:
        with open(f) as fp:
            d = json.load(fp)
        return int(d.get("total_num", 0)) < target
    except Exception:
        return True


def run_smoke(cfg: Config) -> None:
    task = cfg.task_name or "adjust_bottle"
    test_num = cfg.test_num if cfg.test_num is not None else 1
    run_one_task(cfg, task, cfg.gpu_id, 29056, 29061, test_num, "smoke")


def run_single(cfg: Config) -> None:
    test_num = cfg.test_num if cfg.test_num is not None else 25
    if cfg.task_name:
        run_one_task(cfg, cfg.task_name, cfg.gpu_id, 29056, 29061,
                      test_num, "single")
        return
    for task in load_tasks(cfg):
        run_one_task(cfg, task, cfg.gpu_id, 29056, 29061, test_num, "single")


def run_pool(cfg: Config) -> None:
    test_num = cfg.test_num if cfg.test_num is not None else 25
    all_tasks = load_tasks(cfg)
    pending = [t for t in all_tasks
                if cfg.rerun_all or task_needs_run(cfg, t, test_num)]
    if not pending:
        print(f"[pool] all tasks already have >= {test_num} episodes. Nothing to do.")
        return

    print(f"[pool] queue ({len(pending)} tasks):")
    for t in pending:
        print(f"  - {t}")

    usable: list[int] = []
    for g in range(8):
        mb = gpu_free_mb(g)
        if mb >= cfg.min_free_mb:
            usable.append(g)
        else:
            print(f"[pool] skipping GPU {g}: free={mb}MB < {cfg.min_free_mb}MB")
    if not usable:
        print(f"[pool] no GPU has >= {cfg.min_free_mb} MB free.", file=sys.stderr)
        sys.exit(1)
    n_workers = min(len(usable), len(pending))
    print(f"[pool] using GPUs: {usable[:n_workers]}")

    q: queue.Queue[str] = queue.Queue()
    for t in pending:
        q.put(t)

    log_dir = cfg.save_root / "logs" / "pool"
    log_dir.mkdir(parents=True, exist_ok=True)

    def worker(gpu: int) -> None:
        port = 29556 + gpu
        master_port = 29661 + gpu
        while True:
            try:
                task = q.get(block=False)
            except queue.Empty:
                print(f"[pool worker gpu={gpu}] queue empty, exit")
                return
            server_log = log_dir / f"server_{gpu}_{task}.log"
            client_log = log_dir / f"client_{gpu}_{task}.log"
            print(f"[pool worker gpu={gpu}] task={task}")
            if not wait_for_gpu(cfg, gpu):
                continue
            sp = start_server(cfg, gpu, port, master_port, task, server_log)
            with _SESSIONS_LOCK:
                _SESSIONS.add(sp.pid)
            try:
                if not wait_for_port(port):
                    continue
                run_client_blocking(cfg, gpu, task, port, test_num, client_log)
            finally:
                kill_session(sp.pid)
                with _SESSIONS_LOCK:
                    _SESSIONS.discard(sp.pid)

    threads = [threading.Thread(target=worker, args=(g,), daemon=True)
                for g in usable[:n_workers]]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main() -> int:
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
    print(f"[run_eval] mode={cfg.mode} done. logs under {cfg.save_root}/logs/, "
          f"perf under {cfg.perf_log_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
