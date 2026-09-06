"""Shared SmoothQuant calibration corpus generator (multi-WAM, step W2a).

Generate ONE model-neutral 50-episode RoboTwin expert-demo corpus (raw obs)
shared across all four WAMs {lingbot_va, FastWAM, Motus, ImageWAM}. This is a
thin orchestrator over RoboTwin/collect_data.sh: it iterates the 50 ALL_TASKS,
collecting one expert demo per task (task_config `calib_shared` -> episode_num:1),
pooled across GPUs.

Neutral by construction: RoboTwin's scripted expert (motion planning) drives the
simulator, NOT any WAM, so the corpus does not favour any model. The resulting
raw-obs hdf5 (head/wrist rgb + joint_action + endpose + qpos + instruction) is
consumed by each WAM's calib-derive step, which converts raw obs -> that WAM's
policy obs and replays through its FP transformer to collect its own per-channel
activation absmax (plan_multiwam.txt §1 SHARED CALIB ARCHITECTURE).

Output per task: <robotwin>/data/<task>/<task_config>/data/episode0.hdf5
Resumable: a task whose episode0.hdf5 already exists is skipped (unless --rerun_all).

CLI:
    python -m ptqeval.eval.gen_calib_episodes --gpus 4,5,6,7
    python -m ptqeval.eval.gen_calib_episodes --gpus auto --task_list_name SMOKE_5
"""
from __future__ import annotations

import argparse
import importlib
import logging
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

logger = logging.getLogger("ptqeval.eval.gen_calib_episodes")


def _repo_root() -> Path:
    # .../PTQEval/ptqeval/eval/gen_calib_episodes.py -> repo root is 4 parents up.
    return Path(__file__).resolve().parents[3]


def _default_robotwin_dir() -> Path:
    root = _repo_root()
    for cand in (root / "RoboTwin", root / "FastWAM" / "third_party" / "RoboTwin"):
        if (cand / "collect_data.sh").is_file():
            return cand
    raise SystemExit(f"no RoboTwin/collect_data.sh found under {root}")


def _resolve_tasks(module_name: str, list_name: str) -> list[str]:
    mod = importlib.import_module(module_name)
    tasks = getattr(mod, list_name)
    return list(tasks)


def _episode0_path(robotwin: Path, task: str, task_config: str) -> Path:
    return robotwin / "data" / task / task_config / "data" / "episode0.hdf5"


def _free_gpus() -> list[int]:
    """GPUs with < 1024 MiB used (best-effort via nvidia-smi)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"],
            text=True, timeout=30,
        )
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"--gpus auto needs nvidia-smi: {e}")
    free = []
    for line in out.strip().splitlines():
        idx, used = (x.strip() for x in line.split(","))
        if int(used) < 1024:
            free.append(int(idx))
    if not free:
        raise SystemExit("--gpus auto found no free GPU (all >1024 MiB used)")
    return free


def _collect_one(robotwin: Path, task: str, task_config: str, gpu: int,
                 timeout: int) -> tuple[str, bool, str]:
    """Run collect_data.sh <task> <config> <gpu> in the RoboTwin dir."""
    cmd = ["bash", "collect_data.sh", task, task_config, str(gpu)]
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, cwd=str(robotwin), timeout=timeout,
            capture_output=True, text=True,
        )
    except subprocess.TimeoutExpired:
        return task, False, f"timeout after {timeout}s"
    dt = time.time() - t0
    ok = _episode0_path(robotwin, task, task_config).is_file()
    if not ok:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        return task, False, f"rc={proc.returncode} in {dt:.0f}s; tail={tail}"
    return task, True, f"ok in {dt:.0f}s"


def _worker(gpu: int, q: "queue.Queue[str]", robotwin: Path, task_config: str,
            timeout: int, results: list, lock: threading.Lock) -> None:
    while True:
        try:
            task = q.get_nowait()
        except queue.Empty:
            return
        logger.info(f"[gpu{gpu}] collecting {task} ...")
        name, ok, msg = _collect_one(robotwin, task, task_config, gpu, timeout)
        with lock:
            results.append((name, ok, msg))
        logger.info(f"[gpu{gpu}] {name}: {'OK' if ok else 'FAIL'} ({msg})")
        q.task_done()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpus", default="0",
                   help='Comma list "4,5,6,7" or "auto" (claim currently-free GPUs).')
    p.add_argument("--tasks_module", default="ptqeval.wam.lingbot_va.tasks",
                   help="Module exposing the task list attribute.")
    p.add_argument("--task_list_name", default="ALL_TASKS",
                   help="Attribute in tasks_module (ALL_TASKS = 50 RoboTwin tasks).")
    p.add_argument("--task_config", default="calib_shared",
                   help="RoboTwin task_config yml name (episode_num:1 for calib).")
    p.add_argument("--robotwin_dir", default=None,
                   help="RoboTwin root (default: repo ./RoboTwin).")
    p.add_argument("--per_task_timeout", type=int, default=1800,
                   help="Seconds before abandoning a task's expert collection.")
    p.add_argument("--rerun_all", action="store_true",
                   help="Recollect even tasks whose episode0.hdf5 already exists.")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    robotwin = Path(args.robotwin_dir).resolve() if args.robotwin_dir \
        else _default_robotwin_dir()
    if not (robotwin / "task_config" / f"{args.task_config}.yml").is_file():
        raise SystemExit(
            f"missing {robotwin}/task_config/{args.task_config}.yml")

    gpus = _free_gpus() if args.gpus == "auto" \
        else [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    tasks = _resolve_tasks(args.tasks_module, args.task_list_name)

    pending = tasks if args.rerun_all else [
        t for t in tasks
        if not _episode0_path(robotwin, t, args.task_config).is_file()]
    done_already = len(tasks) - len(pending)
    logger.info(f"RoboTwin={robotwin} config={args.task_config} gpus={gpus}")
    logger.info(f"{len(tasks)} tasks, {done_already} already collected, "
                f"{len(pending)} to collect")
    if not pending:
        logger.info("nothing to do.")
        return 0

    q: "queue.Queue[str]" = queue.Queue()
    for t in pending:
        q.put(t)
    results: list = []
    lock = threading.Lock()
    threads = [
        threading.Thread(target=_worker,
                         args=(g, q, robotwin, args.task_config,
                               args.per_task_timeout, results, lock),
                         daemon=True)
        for g in gpus
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    ok = [n for n, s, _ in results if s]
    fail = [(n, m) for n, s, m in results if not s]
    logger.info(f"DONE: {len(ok)} collected, {len(fail)} failed "
                f"(+{done_already} pre-existing)")
    for n, m in fail:
        logger.warning(f"  FAILED {n}: {m}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
