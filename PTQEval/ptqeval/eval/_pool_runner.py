# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Shared orchestration primitives for ptqeval evaluation scripts.

This module is the private home of the server-launch / client-blocking /
GPU-pool / session-cleanup logic used by the eval-tier CLIs:

    ptqeval.eval.run_eval                       (full eval)
    ptqeval.wam.lingbot_va.method.viditq.collect_calib_videos
                                                (calib video collection)

Each consumer script owns its own argparse surface and constructs a
Config dataclass; the run_{smoke,single,pool} dispatchers and the
underlying primitives are imported from here. The consumer scripts do
NOT call each other -- they are peers that share these helpers.

Process lifecycle: each server is launched in a fresh POSIX session
(subprocess.Popen with start_new_session=True). kill_session(pid) walks
/proc to take down the launcher + torch.distributed.run + worker tree
(torch.distributed.run setsids its worker into a separate group, so the
top-level os.killpg is insufficient -- the /proc walk catches that).
"""
from __future__ import annotations

import ctypes
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
# Orphan-prevention: PR_SET_PDEATHSIG (Linux)
# ---------------------------------------------------------------------------
# Tell the Linux kernel to send SIGTERM to a child process automatically
# when its parent dies. This works even when the parent is SIGKILLed (no
# Python signal handler runs), unlike `kill_session()` which depends on
# Python-level cleanup to walk /proc and tear down descendants. PRCTL 1
# is PR_SET_PDEATHSIG. Used as preexec_fn in launch_in_session below.
#
# Before this hook: orchestrator killed -> children reparent to init ->
# orphan servers/clients eat GPU memory until manually killed. This was
# the 2026-06-21 incident that prompted this fix.
#
# After this hook: kernel guarantees children get SIGTERM the moment the
# orchestrator exits (signal-killed, OOM, panic, anything). Python-level
# kill_session() still runs on graceful exit as defense in depth.

_PR_SET_PDEATHSIG = 1


def _set_pdeathsig() -> None:
    """preexec_fn for subprocess.Popen. Runs in the child between fork
    and exec, registering kernel-level orphan prevention. No-op on
    non-Linux (libc.so.6 missing) since other platforms aren't supported
    by our eval pipeline anyway."""
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
    except OSError:
        # Non-Linux fallback: silently skip. Defensive only; we never
        # ship to a non-Linux platform.
        pass


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
    task_list_name: str
    task_config: str
    server_env: str
    client_env: str
    save_root: Path
    perf_log_dir: Path
    profile_ops: bool = False
    profile_n_calls: int = 5
    # Save per-episode obs_data/latents/actions .pt under
    # <save_root>/visualization/. Default False for eval (the dumps are
    # raw camera frames, ~20 MB/ep, not needed for SR/latency/memory).
    # collect_calib_videos sets True (the .pt files ARE the calib corpus).
    save_visualization: bool = False
    # Optional pool GPU selection (default = scan all 8 + filter by free
    # memory). When gpu_ids is set, only those GPUs are considered (still
    # subject to min_free_mb filter). When max_gpus is set, after the
    # usable set is determined, only the first N are used. Both may be
    # combined.
    gpu_ids: Optional[list[int]] = None
    max_gpus: Optional[int] = None


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


def wait_for_port(port: int, timeout: float = 600.0,
                   proc: Optional[subprocess.Popen] = None) -> bool:
    """Poll the port; bail out early if the spawned proc has died (saves
    waiting the full timeout when server crashed on import / model load /
    OOM)."""
    elapsed = 0.0
    while elapsed < timeout:
        if proc is not None and proc.poll() is not None:
            print(f"[wait_for_port] server proc {proc.pid} died early "
                  f"(rc={proc.returncode}) before port {port} came up",
                  file=sys.stderr)
            return False
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
    """Spawn `bash -c bash_cmd` in a new POSIX session, with
    PR_SET_PDEATHSIG=SIGTERM so the kernel auto-kills this child the
    moment we (the orchestrator) die for any reason -- including
    SIGKILL where no Python finalizer would run.

    The returned PID is the new session/group leader; kill_session()
    can still walk /proc to clean up the launcher + torch.distributed.
    run + worker tree on the graceful-exit path.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logf = open(log_path, "w")
    return subprocess.Popen(
        ["bash", "-c", bash_cmd],
        stdout=logf, stderr=subprocess.STDOUT,
        env=os.environ.copy(),
        start_new_session=True, close_fds=True,
        preexec_fn=_set_pdeathsig,
    )


def _find_descendants(root_pid: int) -> list[int]:
    """Walk /proc to find every descendant of root_pid via PPID chain.

    Needed because torch.distributed.run creates its worker process in a
    fresh POSIX session (setsid), so os.killpg(launcher_pid) does NOT
    reach the worker. The descendant walk catches that case.
    """
    parent_map: dict[int, list[int]] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as f:
                stat = f.read()
            # comm field is in parens and may contain spaces;
            # PPID is the 4th field after the closing paren.
            comm_close = stat.rindex(")")
            ppid = int(stat[comm_close + 2:].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        parent_map.setdefault(ppid, []).append(int(entry))

    out: list[int] = []
    stack = [root_pid]
    seen: set[int] = set()
    while stack:
        cur = stack.pop()
        for child in parent_map.get(cur, []):
            if child in seen:
                continue
            seen.add(child)
            out.append(child)
            stack.append(child)
    return out


def kill_session(pid: int, timeout: float = 5.0) -> None:
    """SIGTERM the process and all its descendants; SIGKILL survivors
    after `timeout`. Descendants found via /proc walking so we catch
    torch.distributed.run workers that setsid into their own group."""
    # Snapshot the tree BEFORE killing (otherwise reaping the parent
    # would orphan grandchildren and we'd lose their PIDs).
    targets = [pid] + _find_descendants(pid)

    for p in targets:
        try:
            os.kill(p, signal.SIGTERM)
        except ProcessLookupError:
            pass

    deadline = time.time() + timeout
    while time.time() < deadline:
        alive: list[int] = []
        for p in targets:
            try:
                os.kill(p, 0)
                alive.append(p)
            except ProcessLookupError:
                pass
        if not alive:
            return
        time.sleep(0.2)

    for p in targets:
        try:
            os.kill(p, signal.SIGKILL)
        except ProcessLookupError:
            pass


# ---------------------------------------------------------------------------
# Server / client launchers
# ---------------------------------------------------------------------------

def start_server(cfg: Config, gpu: int, port: int, master_port: int,
                  task_name: str, server_log: Path) -> subprocess.Popen:
    extra: list[str] = []
    if cfg.variant:
        extra.append(f"--variant {cfg.variant}")
        if cfg.variant_args:
            extra.append(f"--variant_args {cfg.variant_args}")
    if cfg.profile_ops:
        extra.append("--profile_ops")
        extra.append(f"--profile_n_calls {cfg.profile_n_calls}")
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
        # SAPIEN_USE_VULKAN_DEVICE_ID pins sapien's vulkan renderer to the
        # same physical GPU as torch. Without this, sapien defaults to
        # vulkan device 0 for every worker -- 8 concurrent pool workers
        # then all hit GPU 0's vulkan surface, slowing each sapien step
        # ~17x and triggering reset-stall + ConnectionClosedError. Sapien
        # does NOT honor CUDA_VISIBLE_DEVICES for its vulkan path. The
        # diagnostic script (scripts/diagnose_l40s_eval.sh) confirmed
        # single-worker runs work end-to-end on L40s.
        f"exec env CUDA_VISIBLE_DEVICES={gpu}"
        f"         SAPIEN_USE_VULKAN_DEVICE_ID={gpu}"
        f"         PYTHONWARNINGS=ignore::UserWarning"
        f"         XLA_PYTHON_CLIENT_MEM_FRACTION=0.9"
        f"  python -m ptqeval.wam.{cfg.wam_name}.eval_client"
        f"    --config {cfg.robotwin_root}/policy/ACT/deploy_policy.yml"
        f"    --overrides"
        f"    --task_name {task_name}"
        f"    --task_config {cfg.task_config}"
        f"    --train_config_name 0"
        f"    --model_name 0"
        f"    --ckpt_setting 0"
        f"    --seed {cfg.seed}"
        f"    --policy_name ACT"
        f"    --save_root {cfg.save_root}"
        f"    --video_guidance_scale 5"
        f"    --action_guidance_scale 1"
        f"    --test_num {test_num}"
        f"    --save_visualization {cfg.save_visualization}"
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
_CFG_FOR_CLEANUP: Optional["Config"] = None
_SIGNAL_COUNT = 0


def _set_cleanup_context(cfg: "Config") -> None:
    """Stash the active Config so orphan scanners can match `save_root`
    in /proc/*/cmdline as a last-resort filter."""
    global _CFG_FOR_CLEANUP
    _CFG_FOR_CLEANUP = cfg


def _scan_orphans_by_save_root(save_root: Optional[str]) -> list[int]:
    """Find any lingering lingbot_va.server PIDs whose cmdline mentions
    `save_root`. Catches race-condition orphans where a server was
    spawned but never added to _SESSIONS (e.g. signal arrived during
    Popen handoff)."""
    if not save_root:
        return []
    needle = str(save_root)
    out: list[int] = []
    self_pid = os.getpid()
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid == self_pid:
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\x00", b" ").decode("utf-8", "replace")
        except (OSError, ValueError):
            continue
        if "ptqeval.wam" in cmd and "server" in cmd and needle in cmd:
            out.append(pid)
    return out


def cleanup_all_sessions() -> None:
    """Idempotent: kill every tracked session, then do a /proc rescan
    by save_root to catch race-condition orphans (server spawned but
    not yet added to _SESSIONS when signal fired)."""
    with _SESSIONS_LOCK:
        pids = list(_SESSIONS)
        _SESSIONS.clear()
    for pid in pids:
        try:
            kill_session(pid)
        except Exception as e:
            print(f"[_pool_runner] kill_session({pid}) failed: {e}",
                  file=sys.stderr)
    # Last-resort sweep -- finds anything we lost track of, including
    # children spawned milliseconds before SIGINT arrived.
    save_root = getattr(_CFG_FOR_CLEANUP, "save_root", None)
    if save_root is not None:
        leftover = _scan_orphans_by_save_root(save_root)
        if leftover:
            print(f"[_pool_runner] orphan sweep found {len(leftover)} extra "
                  f"server pid(s) matching {save_root}; killing.",
                  file=sys.stderr)
            for pid in leftover:
                try:
                    kill_session(pid, timeout=2.0)
                except Exception:
                    pass


def install_signal_handlers() -> None:
    """Register SIGINT/SIGTERM/SIGHUP handlers + atexit fallback.

    On 1st signal: graceful cleanup_all_sessions then sys.exit.
    On 2nd signal: skip wait, SIGKILL every tracked descendant then
                   os._exit (covers double-Ctrl+C impatient user).
    SIGHUP is included because closing the terminal sends SIGHUP, not
    SIGINT, and we still need orphans cleaned up in that case.
    atexit fires on normal SystemExit so it catches SIGINT/SIGHUP-driven
    exits even if the signal handler itself raced; harmless duplicate
    when handler already ran (cleanup is idempotent)."""
    import atexit

    def handler(sig, _frame):
        global _SIGNAL_COUNT
        _SIGNAL_COUNT += 1
        if _SIGNAL_COUNT == 1:
            print(f"\n[_pool_runner] received signal {sig}; cleaning up "
                  f"sessions (press again to force SIGKILL)...",
                  file=sys.stderr)
            cleanup_all_sessions()
            sys.exit(128 + sig)
        else:
            print(f"\n[_pool_runner] signal {sig} again; SIGKILL all "
                  f"tracked descendants and abort.", file=sys.stderr)
            # Snapshot and SIGKILL without waiting.
            with _SESSIONS_LOCK:
                pids = list(_SESSIONS)
            for pid in pids:
                for child in [pid] + _find_descendants(pid):
                    try:
                        os.kill(child, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            os._exit(128 + sig)

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, handler)
    atexit.register(cleanup_all_sessions)


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
    # Default test_num bumped 25 -> 100 on 2026-06-26 alongside the
    # task_list_name default flip (SELECTED_15_TASKS -> ALL_TASKS)
    # so the production scope matches paper-style full-coverage eval
    # (50 tasks x 100 ep = 5000 episodes/variant); see run_eval.py
    # --test_num help text for rationale.
    test_num = cfg.test_num if cfg.test_num is not None else 100
    if cfg.task_name:
        run_one_task(cfg, cfg.task_name, cfg.gpu_id, 29056, 29061,
                      test_num, "single")
        return
    for task in load_tasks(cfg):
        run_one_task(cfg, task, cfg.gpu_id, 29056, 29061, test_num, "single")


def run_pool(cfg: Config) -> None:
    # Default test_num: see run_single comment above (25 -> 100 on 2026-06-26).
    test_num = cfg.test_num if cfg.test_num is not None else 100
    all_tasks = load_tasks(cfg)
    pending = [t for t in all_tasks
                if cfg.rerun_all or task_needs_run(cfg, t, test_num)]
    if not pending:
        print(f"[pool] all tasks already have >= {test_num} episodes. Nothing to do.")
        return

    print(f"[pool] queue ({len(pending)} tasks):")
    for t in pending:
        print(f"  - {t}")

    # GPU selection: scan candidates, filter by free memory, optionally
    # cap by max_gpus. Candidates come from cfg.gpu_ids if given,
    # otherwise the full range(8).
    candidates = cfg.gpu_ids if cfg.gpu_ids else list(range(8))
    if cfg.gpu_ids:
        print(f"[pool] candidate GPUs (from --gpus): {candidates}")
    usable: list[int] = []
    for g in candidates:
        mb = gpu_free_mb(g)
        if mb >= cfg.min_free_mb:
            usable.append(g)
        else:
            print(f"[pool] skipping GPU {g}: free={mb}MB < {cfg.min_free_mb}MB")
    if not usable:
        print(f"[pool] no GPU in {candidates} has >= {cfg.min_free_mb} MB free.",
              file=sys.stderr)
        sys.exit(1)
    if cfg.max_gpus is not None and cfg.max_gpus > 0:
        if len(usable) > cfg.max_gpus:
            print(f"[pool] usable={usable}, capping to first {cfg.max_gpus} "
                  f"per --max_gpus")
            usable = usable[:cfg.max_gpus]
    n_workers = min(len(usable), len(pending))
    print(f"[pool] using GPUs: {usable[:n_workers]}")

    q: queue.Queue[str] = queue.Queue()
    for t in pending:
        q.put(t)

    log_dir = cfg.save_root / "logs" / "pool"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Track per-worker progress so the "queue empty" message at end is
    # not mistaken for a mid-run abort.
    worker_done_counts: dict[int, int] = {g: 0 for g in usable[:n_workers]}
    worker_failed_counts: dict[int, int] = {g: 0 for g in usable[:n_workers]}

    def worker(gpu: int) -> None:
        port = 29556 + gpu
        master_port = 29661 + gpu
        while True:
            try:
                task = q.get(block=False)
            except queue.Empty:
                print(f"[pool worker gpu={gpu}] queue drained for this worker "
                      f"(done={worker_done_counts[gpu]}, "
                      f"failed={worker_failed_counts[gpu]}); "
                      f"exiting. Other workers may still be running.")
                return
            server_log = log_dir / f"server_{gpu}_{task}.log"
            client_log = log_dir / f"client_{gpu}_{task}.log"
            print(f"[pool worker gpu={gpu}] task={task} starting")
            if not wait_for_gpu(cfg, gpu):
                print(f"[pool worker gpu={gpu}] task={task} SKIPPED: GPU wait failed",
                      file=sys.stderr)
                worker_failed_counts[gpu] += 1
                continue
            sp = start_server(cfg, gpu, port, master_port, task, server_log)
            with _SESSIONS_LOCK:
                _SESSIONS.add(sp.pid)
            try:
                if not wait_for_port(port, proc=sp):
                    print(f"[pool worker gpu={gpu}] task={task} SKIPPED: "
                          f"server failed to come up (see {server_log})",
                          file=sys.stderr)
                    worker_failed_counts[gpu] += 1
                    continue
                rc = run_client_blocking(cfg, gpu, task, port, test_num, client_log)
                if rc != 0:
                    # Previously this branch silently incremented done -- a
                    # crashed client (e.g. websockets ConnectionClosedError
                    # because server hung in reset and got SIGTERMed) looked
                    # identical to a successful run in the orchestrator log.
                    # Now: report the actual rc so the user sees real failure
                    # counts instead of a misleading "done".
                    print(f"[pool worker gpu={gpu}] task={task} FAILED "
                          f"(client rc={rc}; see {client_log} and "
                          f"{server_log.parent / f'server_{gpu}_{task}.log'})",
                          file=sys.stderr)
                    worker_failed_counts[gpu] += 1
                else:
                    worker_done_counts[gpu] += 1
                    print(f"[pool worker gpu={gpu}] task={task} done "
                          f"(worker total done={worker_done_counts[gpu]})")
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

    total_done = sum(worker_done_counts.values())
    total_failed = sum(worker_failed_counts.values())
    print(f"[pool] all workers exited. "
          f"completed {total_done} task(s), failed {total_failed} task(s) "
          f"out of {len(pending)} queued.")
    if total_failed:
        print(f"[pool] per-worker breakdown: done={worker_done_counts}, "
              f"failed={worker_failed_counts}")
