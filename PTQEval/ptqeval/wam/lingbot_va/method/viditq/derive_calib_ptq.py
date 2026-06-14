# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Calibration step 2: replay collected obs through bf16 FP transformer
with hooks, then run PTQ to produce int_weights.pth.

Reads the per-episode visualization directories saved by
collect_calib_videos.py (obs_data_*.pt files under <videos_root>/
visualization/real/<prompt>_<timestamp>/), filters by task list, and
drives replay through one or more bf16 in-process VA_Server workers
pinned to dedicated GPUs. Per-worker forward_pre_hooks aggregate
per-channel input absmax into a single calib_data.pth shared across
workers via fcntl.flock + merge-on-write (get_calib_data._CalibState).

GPU mode (--gpus):
  auto                Claim every currently-free GPU at start. Run a
                      background monitor that ALSO claims any GPU that
                      stays free for --gpu_stable_secs (default 300 =
                      5 min) -- so the calib opportunistically grows
                      its worker pool when other users free up GPUs.
  <list>              Comma-separated explicit GPU ids, e.g. "0,1,2".
                      No dynamic claim; pool stays at exactly this size.

Each worker is its own subprocess invocation of this script with
--_worker_mode + a per-process env that pins CUDA_VISIBLE_DEVICES and
unique torch.distributed MASTER_PORT. Workers pull episodes from a
file-backed queue (one episode path per line, popped under fcntl.flock).
This keeps torch out of the orchestrator process and lets the
orchestrator stay on a tiny stdlib-only footprint.

No RoboTwin simulator involvement at this stage -- pure FP transformer
forward driven by previously-captured observations.

After workers drain the queue, the orchestrator runs ptq.py as a
subprocess with the supplied --layer_config so PTQ produces the
int_weights.pth for any variant (W8A8 dynamic / smooth / quarot /
viditq / viditq-static). Layer configs read calib_data path from the
yaml itself; this script writes calib_data.pth at the path declared
by the config (or --calib_out override).

Task subset selection:
  --task_list <name1,name2,...>   comma-separated short names (default:
                                  SELECTED_15_TASKS contents).
  --all                           use every episode under videos_root,
                                  ignoring --task_list. Default OFF.
"""
from __future__ import annotations

import argparse
import fcntl
import logging
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from ptqeval.wam.lingbot_va.tasks import SELECTED_15_TASKS


logger = logging.getLogger("derive_calib_ptq")


# ---------------------------------------------------------------------------
# Episode dir enumeration + task filtering (no torch needed)
# ---------------------------------------------------------------------------

# visualization/real/<prompt>_<YYYYMMDD_HHMMSS>/  -- trailing timestamp is
# 1 underscore + 8 date digits + 1 underscore + 6 time digits = 16 chars.
_TIMESTAMP_SUFFIX_LEN = len("_YYYYMMDD_HHMMSS")  # 16


def _strip_timestamp(dir_name: str) -> str:
    return dir_name[:-_TIMESTAMP_SUFFIX_LEN] if len(dir_name) > _TIMESTAMP_SUFFIX_LEN else dir_name


def _build_prompt_to_task(save_root: Path) -> dict[str, str]:
    """Walk stseed-*/visualization/<task>/<...>.mp4 to learn prompt->task
    mapping. mp4 filename format produced by eval_client.py:
        <test_num_idx>_<prompt_with_spaces_as_underscores>_<succ>.mp4
    Underscoring is reversible because no RoboTwin task prompt in the
    selected_15/calib_all set contains a literal underscore.
    """
    pat = re.compile(r"^(\d+)_(.+)_(True|False)$")
    mapping: dict[str, str] = {}
    for mp4 in save_root.glob("stseed-*/visualization/*/*.mp4"):
        task = mp4.parent.name
        m = pat.match(mp4.stem)
        if not m:
            continue
        prompt = m.group(2).replace("_", " ")
        mapping[prompt] = task
    return mapping


def _filter_episodes(videos_root: Path, task_list: Optional[list[str]],
                     use_all: bool, per_task_ep: int = 0) -> list[Path]:
    vis_root = videos_root / "visualization" / "real"
    if not vis_root.exists():
        raise FileNotFoundError(
            f"No visualization/real/ under {videos_root}. Run "
            f"collect_calib_videos.py first."
        )
    ep_dirs = sorted(p for p in vis_root.iterdir() if p.is_dir())
    if use_all:
        if per_task_ep > 0:
            logger.warning("--per_task_ep ignored because --all is set")
        return ep_dirs

    assert task_list, "task_list required when --all is not set"
    prompt_to_task = _build_prompt_to_task(videos_root)
    by_task: dict[str, list[Path]] = {}
    skipped_task = 0
    skipped_unknown = 0
    for ep_dir in ep_dirs:
        prompt = _strip_timestamp(ep_dir.name)
        task = prompt_to_task.get(prompt)
        if task is None:
            skipped_unknown += 1
            continue
        if task not in task_list:
            skipped_task += 1
            continue
        by_task.setdefault(task, []).append(ep_dir)
    # Sorted iteration already gives chronological order per task; keep
    # the first per_task_ep when set.
    selected: list[Path] = []
    capped = 0
    for task, eps in by_task.items():
        keep = eps[:per_task_ep] if per_task_ep > 0 else eps
        capped += len(eps) - len(keep)
        selected.extend(keep)
    logger.info(
        f"episode filter: {len(selected)} kept, {skipped_task} other-task, "
        f"{skipped_unknown} unmapped (no matching mp4 -> prompt mapping)"
        + (f", {capped} dropped by --per_task_ep={per_task_ep}"
           if per_task_ep > 0 else "")
    )
    return selected


# ---------------------------------------------------------------------------
# GPU probe (orchestrator side; mirrors _pool_runner.gpu_free_mb but stays
# self-contained so the orchestrator has no cross-script imports beyond
# tasks.py)
# ---------------------------------------------------------------------------

def _gpu_free_mb(gpu: int) -> int:
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
         "-i", str(gpu)],
        capture_output=True, text=True, check=False)
    try:
        return int(r.stdout.strip())
    except (ValueError, AttributeError):
        return -1


def _detect_n_gpus() -> int:
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False)
    return len([line for line in r.stdout.splitlines() if line.strip()])


# ---------------------------------------------------------------------------
# File-backed episode queue (stdlib + fcntl, so workers can be separate
# Python processes without an mp.Manager)
# ---------------------------------------------------------------------------

def _queue_init(queue_path: Path, ep_dirs: list[Path]) -> None:
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    with open(queue_path, "w") as f:
        for ep in ep_dirs:
            f.write(str(ep) + "\n")


def _queue_pop(queue_path: Path) -> Optional[str]:
    """Atomic pop of first line from queue. Returns None on empty."""
    with open(queue_path, "r+") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        lines = f.read().splitlines()
        if not lines:
            return None
        first = lines[0]
        f.seek(0)
        f.truncate()
        if len(lines) > 1:
            f.write("\n".join(lines[1:]) + "\n")
    return first


def _queue_remaining(queue_path: Path) -> int:
    """Approximate remaining-work count (line count). Cheap, no lock."""
    try:
        with open(queue_path) as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


# ---------------------------------------------------------------------------
# Worker subprocess: in-process VA_Server replay loop
# ---------------------------------------------------------------------------

def _build_server(model_path: str, save_root: Path):
    """Construct a bf16 VA_Server in-process. Side-effect saves
    (visualization, perf logs) redirected to save_root so they don't
    stomp the calib video corpus.

    Imports deferred so this only runs in worker subprocess (after
    CUDA_VISIBLE_DEVICES + RANK/LOCAL_RANK/WORLD_SIZE/MASTER_PORT are
    set in env by the orchestrator's Popen call)."""
    import ptqeval.wam.lingbot_va as _lingbot_va_pkg  # noqa: F401
    _wan_va_dir = os.path.join(_lingbot_va_pkg.LINGBOT_VA_PATH, "wan_va")
    if _wan_va_dir not in sys.path:
        sys.path.insert(0, _wan_va_dir)
    from distributed.util import init_distributed
    from ptqeval.wam.lingbot_va.server import VA_Server
    from configs import VA_CONFIGS

    init_distributed(world_size=1, local_rank=0, rank=0)

    cfg = VA_CONFIGS["robotwin"]
    cfg.save_root = str(save_root)
    cfg.perf_log_dir = None
    cfg.perf_task_name = None
    cfg.wan22_pretrained_model_name_or_path = model_path
    cfg.rank = 0
    cfg.local_rank = 0
    cfg.world_size = 1
    save_root.mkdir(parents=True, exist_ok=True)
    return VA_Server(cfg)


_CHUNK_RE = re.compile(r"obs_data_(\d+)\.pt$")


def _replay_episode(server, ep_dir: Path) -> int:
    """Drive server.infer over every saved obs chunk in ep_dir using
    reset + 1-frame _infer per chunk. See module docstring for rationale.
    Returns number of _infer calls fired."""
    import torch  # deferred; only worker imports torch
    chunks = sorted(
        (int(_CHUNK_RE.search(p.name).group(1)), p)
        for p in ep_dir.glob("obs_data_*.pt")
    )
    if not chunks:
        return 0

    first = torch.load(chunks[0][1], weights_only=False, map_location="cpu")
    prompt = first[0]["task"]

    n_calls = 0
    for _, chunk_path in chunks:
        obs_list = torch.load(chunk_path, weights_only=False, map_location="cpu")
        server.infer({"reset": True, "prompt": prompt, "save_visualization": False})
        server.infer({
            "obs": [obs_list[0]],
            "prompt": prompt,
            "save_visualization": False,
        })
        n_calls += 1
    return n_calls


def _worker_main(args) -> int:
    """Subprocess entry point. CUDA_VISIBLE_DEVICES + RANK/LOCAL_RANK/
    WORLD_SIZE/MASTER_PORT come pre-set in env (orchestrator does that
    via Popen(env=...)). Loop on queue_pop until empty, then exit
    naturally so install_calib_hooks's atexit dump fires."""
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [worker pid={os.getpid()}] %(levelname)s %(message)s",
    )
    wlog = logging.getLogger(f"derive_worker_pid{os.getpid()}")

    from ptqeval.wam.lingbot_va.method.viditq.get_calib_data import install_calib_hooks

    wlog.info(f"building bf16 server from {args.model_path} "
              f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")
    server = _build_server(args.model_path, args.replay_save_root)

    wlog.info("installing calib hooks on 180 target Linears")
    state = install_calib_hooks(server.transformer, str(args.calib_out))

    n_done = 0
    while True:
        ep_path = _queue_pop(args._worker_queue_path)
        if ep_path is None:
            wlog.info(f"queue drained; exiting after {n_done} episodes")
            break
        try:
            _replay_episode(server, Path(ep_path))
        except Exception as e:
            wlog.exception(f"replay failed for {ep_path}: {e}")
        n_done += 1
        if n_done % 5 == 0:
            wlog.info(f"replayed {n_done} episodes")

    state.dump()
    wlog.info(f"final dump done (replayed {n_done} episodes)")
    return 0


# ---------------------------------------------------------------------------
# Orchestrator: multi-GPU spawn + dynamic claim monitor
# ---------------------------------------------------------------------------

def _spawn_worker_subprocess(gpu_id: int, queue_path: Path, calib_out: Path,
                              model_path: str, replay_save_root: Path,
                              skip_ptq_extra_args: dict) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["RANK"] = "0"
    env["LOCAL_RANK"] = "0"
    env["WORLD_SIZE"] = "1"
    env["MASTER_ADDR"] = "127.0.0.1"
    # Unique port per worker so concurrent distributed inits do not clash.
    env["MASTER_PORT"] = str(29577 + gpu_id)
    worker_save_root = replay_save_root / f"gpu{gpu_id}"
    cmd = [
        sys.executable, "-m", "ptqeval.wam.lingbot_va.method.viditq.derive_calib_ptq",
        "--_worker_mode",
        "--_worker_queue_path", str(queue_path),
        "--model_path", model_path,
        "--calib_out", str(calib_out),
        "--replay_save_root", str(worker_save_root),
        # Dummy values for orchestrator-only args (worker ignores these
        # but argparse still needs them defined; main()'s argparse
        # accepts them in both modes).
        "--videos_root", str(Path("/tmp")),
        "--skip_ptq",
    ]
    log_path = worker_save_root / "worker.log"
    worker_save_root.mkdir(parents=True, exist_ok=True)
    logf = open(log_path, "w")
    return subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT,
                              start_new_session=True, close_fds=True)


def _orchestrate(ep_dirs: list[Path], gpus: object, calib_out: Path,
                  model_path: str, replay_save_root: Path,
                  min_free_mb: int, stable_secs: float) -> int:
    """Spawn workers (subprocesses), monitor for newly-free GPUs in
    'auto' mode, wait for queue to drain. Returns 0 on normal exit."""
    queue_path = replay_save_root / "ep_queue.txt"
    _queue_init(queue_path, ep_dirs)
    logger.info(f"queue initialized with {len(ep_dirs)} episodes at {queue_path}")

    workers: dict[int, subprocess.Popen] = {}
    workers_lock = threading.Lock()
    stop_event = threading.Event()
    n_total_spawned = 0

    def spawn(gpu_id: int) -> None:
        nonlocal n_total_spawned
        with workers_lock:
            if gpu_id in workers and workers[gpu_id].poll() is None:
                return
            p = _spawn_worker_subprocess(
                gpu_id, queue_path, calib_out, model_path, replay_save_root, {})
            workers[gpu_id] = p
            n_total_spawned += 1
            logger.info(
                f"spawned worker #{n_total_spawned} on GPU {gpu_id} "
                f"(pid={p.pid}, log={replay_save_root}/gpu{gpu_id}/worker.log)"
            )

    # Initial pool: claim free GPUs at T=0 (no wait).
    if gpus == "auto":
        n_gpus = _detect_n_gpus() or 8
        for g in range(n_gpus):
            if _gpu_free_mb(g) >= min_free_mb:
                spawn(g)
    else:
        for g in gpus:
            spawn(g)

    if not workers:
        if gpus == "auto":
            logger.warning(
                f"no GPU with >= {min_free_mb} MB free at start; "
                f"monitor will claim as GPUs become available."
            )
        else:
            logger.error(f"no usable GPU in --gpus={gpus}")
            return 1

    # Dynamic claim monitor (auto only): poll every 30s; track first-free
    # time per non-claimed GPU; claim after stable_secs of continuous
    # free-ness. Stops when queue is drained.
    monitor_thread = None
    if gpus == "auto":
        def monitor():
            first_free_at: dict[int, float] = {}
            n_gpus = _detect_n_gpus() or 8
            while not stop_event.is_set():
                # Stop spawning if queue empty (no more work).
                if _queue_remaining(queue_path) == 0:
                    stop_event.wait(30)
                    continue
                for g in range(n_gpus):
                    with workers_lock:
                        active = (g in workers and workers[g].poll() is None)
                    if active:
                        first_free_at.pop(g, None)
                        continue
                    mb = _gpu_free_mb(g)
                    if mb >= min_free_mb:
                        first_free_at.setdefault(g, time.time())
                        if time.time() - first_free_at[g] >= stable_secs:
                            logger.info(
                                f"monitor: GPU {g} stable free for "
                                f"{stable_secs:.0f}s, claiming"
                            )
                            spawn(g)
                            first_free_at.pop(g, None)
                    else:
                        first_free_at.pop(g, None)
                stop_event.wait(30)

        monitor_thread = threading.Thread(target=monitor,
                                            name="gpu-claim-monitor",
                                            daemon=True)
        monitor_thread.start()

    # Wait for all live workers; spawn-set may grow during the wait
    # (monitor can add). Loop until all workers exited.
    try:
        while True:
            with workers_lock:
                alive = [p for p in workers.values() if p.poll() is None]
            if not alive:
                # If queue not drained yet but no workers alive AND
                # not in auto mode, surface the partial work loss.
                if _queue_remaining(queue_path) > 0 and gpus != "auto":
                    logger.warning(
                        f"all manual-mode workers exited with "
                        f"{_queue_remaining(queue_path)} episodes left in queue"
                    )
                break
            time.sleep(15)
    finally:
        stop_event.set()
        if monitor_thread:
            monitor_thread.join(timeout=5)
        with workers_lock:
            for gpu_id, p in workers.items():
                if p.poll() is None:
                    logger.warning(f"terminating lingering worker gpu={gpu_id} pid={p.pid}")
                    try:
                        os.killpg(os.getpgid(p.pid), 15)  # SIGTERM whole session
                    except ProcessLookupError:
                        pass
                    p.wait(timeout=10)

    # Don't unlink queue_path -- keeping it makes a partial-run recoverable
    # by re-invoking with the same calib_out (queue_pop would just see
    # whatever wasn't consumed). For the all-drained happy path it's an
    # empty file that takes 0 bytes.
    logger.info(f"orchestrator: workers complete, queue remaining={_queue_remaining(queue_path)}")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--videos_root", type=Path,
        default=Path("/home/arash/EvalForWAMs/results/calib_capture"),
        help="Root saved by collect_calib_videos.py "
             "(contains visualization/real/<prompt>_<ts>/ and "
             "stseed-*/visualization/<task>/...).",
    )
    p.add_argument(
        "--task_list", default=",".join(SELECTED_15_TASKS),
        help="Comma-separated task short names. Default: SELECTED_15_TASKS.",
    )
    p.add_argument(
        "--all", action="store_true",
        help="Use all episodes under --videos_root regardless of task. "
             "Default OFF (filter by --task_list).",
    )
    p.add_argument(
        "--per_task_ep", type=int, default=0,
        help="If >0, keep only the first N episodes per task (after "
             "task_list filter) so calib mass per task is uniform. "
             "Default 0 = use every matching episode (legacy behavior).",
    )
    p.add_argument(
        "--calib_out", type=Path,
        default=Path("/home/arash/EvalForWAMs/results/calib_data/calib_data.pth"),
        help="Per-channel absmax dump path. Overwrites in place.",
    )
    p.add_argument(
        "--layer_config", type=Path,
        help="ptq.py --layer_config yaml. Required unless --skip_ptq.",
    )
    p.add_argument(
        "--int_weights_out", type=Path,
        help="ptq.py --output int_weights.pth. Required unless --skip_ptq.",
    )
    p.add_argument(
        "--skip_ptq", action="store_true",
        help="Stop after dumping calib_data.pth (skip ptq subprocess).",
    )
    p.add_argument(
        "--model_path",
        default="/home/arash/EvalForWAMs/models/lingbot-va-posttrain-robotwin",
        help="FP bf16 transformer dir.",
    )
    p.add_argument(
        "--replay_save_root", type=Path,
        default=Path("/tmp/derive_calib_replay"),
        help="Throwaway dir for per-worker visualization saves + the "
             "shared ep_queue.txt file.",
    )

    # Multi-GPU configuration.
    p.add_argument(
        "--gpus", default="auto",
        help='"auto" (default): claim every currently-free GPU at start, '
             'then dynamically claim any GPU that stays free for '
             '--gpu_stable_secs. Or "<gpu_id_csv>", e.g. "0,1,2".',
    )
    p.add_argument(
        "--gpu_stable_secs", type=int, default=300,
        help="In --gpus=auto, how long a non-claimed GPU must stay free "
             "before the monitor claims it. Default 300 (5 min).",
    )
    p.add_argument(
        "--min_free_mb", type=int, default=32000,
        help="Threshold for considering a GPU 'free'.",
    )

    # Worker-only flags (hidden; the orchestrator passes these when
    # spawning a worker subprocess).
    p.add_argument("--_worker_mode", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--_worker_queue_path", type=Path, default=None,
                   help=argparse.SUPPRESS)

    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # ----- Worker mode (subprocess entry) -----
    if args._worker_mode:
        if args._worker_queue_path is None:
            p.error("--_worker_mode requires --_worker_queue_path")
        return _worker_main(args)

    # ----- Orchestrator mode -----
    if not args.skip_ptq and (args.layer_config is None or args.int_weights_out is None):
        p.error("--layer_config and --int_weights_out required unless --skip_ptq")

    task_list = None
    if not args.all:
        task_list = [t.strip() for t in args.task_list.split(",") if t.strip()]
        logger.info(f"task subset: {task_list}")
    else:
        logger.info("task subset: ALL (--all)")

    ep_dirs = _filter_episodes(args.videos_root, task_list, args.all,
                                per_task_ep=args.per_task_ep)
    if not ep_dirs:
        logger.error("no episode dirs matched; nothing to do.")
        return 1

    args.calib_out.parent.mkdir(parents=True, exist_ok=True)
    if args.calib_out.exists():
        logger.info(f"removing prior calib at {args.calib_out}")
        args.calib_out.unlink()

    # Parse --gpus
    raw = args.gpus.strip().lower()
    if raw == "auto":
        gpus_arg: object = "auto"
        logger.info(
            f"GPU mode: auto (initial=free-at-start, "
            f"dynamic-claim after {args.gpu_stable_secs}s stable, "
            f"min_free={args.min_free_mb} MB)"
        )
    else:
        gpus_arg = [int(g) for g in raw.split(",") if g.strip()]
        logger.info(f"GPU mode: manual list {gpus_arg}")

    rc = _orchestrate(
        ep_dirs, gpus_arg, args.calib_out, args.model_path,
        args.replay_save_root, args.min_free_mb, args.gpu_stable_secs,
    )
    if rc != 0:
        return rc

    logger.info(f"calib_data.pth ready at {args.calib_out}")

    if args.skip_ptq:
        return 0

    cmd = [
        sys.executable, "-m", "ptqeval.wam.lingbot_va.method.viditq.ptq",
        "--layer_config", str(args.layer_config),
        "--output", str(args.int_weights_out),
    ]
    logger.info("running ptq: " + " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
