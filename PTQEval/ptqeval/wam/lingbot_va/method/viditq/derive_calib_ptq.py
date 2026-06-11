# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Calibration step 2: replay collected obs through bf16 FP transformer
with hooks, then run PTQ to produce int_weights.pth.

Reads the per-episode visualization directories saved by
collect_calib_videos.py (obs_data_*.pt files under <videos_root>/
visualization/real/<prompt>_<timestamp>/), filters by task list, spins
up a single in-process VA_Server (bf16), installs forward_pre_hooks on
the 180 target Linears, and drives `server.infer(...)` over the saved
obs chunks. Hooks aggregate per-channel input absmax into calib_data.pth
via the same get_calib_data._CalibState used by the live-eval path
(merge-on-write + fcntl.flock for crash safety).

No RoboTwin simulator involvement at this stage -- pure FP transformer
forward driven by previously-captured observations.

After hooks dump, invokes ptq.py with the supplied --layer_config so
PTQ produces the int_weights.pth for any variant (W8A8 dynamic /
smooth / quarot / viditq / viditq-static). Layer configs read calib_data
path from the yaml itself; this script writes the calib_data.pth at the
path declared by the config (or --calib_out override).

Task subset selection:
  --task_list <name1,name2,...>   comma-separated short names (default:
                                  SELECTED_15_TASKS contents).
  --all                           use every episode under videos_root,
                                  ignoring --task_list. Default OFF.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Force single-GPU non-distributed BEFORE importing torch.distributed.
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29577")

import torch

# Triggers ptqeval.wam.lingbot_va package init (adds lingbot-va to sys.path).
import ptqeval.wam.lingbot_va  # noqa: F401
from ptqeval.wam.lingbot_va.tasks import SELECTED_15_TASKS
from ptqeval.wam.lingbot_va.method.viditq.get_calib_data import install_calib_hooks


logger = logging.getLogger("derive_calib_ptq")


# ---------------------------------------------------------------------------
# Episode dir enumeration + task filtering
# ---------------------------------------------------------------------------

# visualization/real/<prompt>_<YYYYMMDD_HHMMSS>/  -- trailing timestamp is
# 1 underscore + 8 date digits + 1 underscore + 6 time digits = 16 chars.
_TIMESTAMP_SUFFIX_LEN = len("_YYYYMMDD_HHMMSS")  # 16


def _strip_timestamp(dir_name: str) -> str:
    return dir_name[:-_TIMESTAMP_SUFFIX_LEN] if len(dir_name) > _TIMESTAMP_SUFFIX_LEN else dir_name


def _build_prompt_to_task(save_root: Path) -> dict[str, str]:
    """Walk stseed-*/visualization/<task>/<...>.mp4 to learn prompt->task
    mapping. The mp4 filename format produced by eval_client.py:
        <test_num_idx>_<prompt_with_spaces_as_underscores>_<succ>.mp4
    Underscoring is reversible because no RoboTwin task prompt in the
    selected_15/calib_all set contains a literal underscore (verified
    by inspection of obs_data 'task' fields).
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
                     use_all: bool) -> list[Path]:
    vis_root = videos_root / "visualization" / "real"
    if not vis_root.exists():
        raise FileNotFoundError(
            f"No visualization/real/ under {videos_root}. Run "
            f"collect_calib_videos.py first."
        )
    ep_dirs = sorted(p for p in vis_root.iterdir() if p.is_dir())
    if use_all:
        return ep_dirs

    assert task_list, "task_list required when --all is not set"
    prompt_to_task = _build_prompt_to_task(videos_root)
    selected = []
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
        selected.append(ep_dir)
    logger.info(
        f"episode filter: {len(selected)} kept, {skipped_task} other-task, "
        f"{skipped_unknown} unmapped (no matching mp4 -> prompt mapping)."
    )
    return selected


# ---------------------------------------------------------------------------
# Server in-process construction
# ---------------------------------------------------------------------------

def _build_server(model_path: str, save_root: Path) -> "VA_Server":
    """Construct a bf16 VA_Server in-process with side-effect saves
    redirected to save_root (so visualization writes during replay do
    not stomp the calib video corpus)."""
    # Imports deferred so distributed env vars (set above) are picked up.
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


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

_CHUNK_RE = re.compile(r"obs_data_(\d+)\.pt$")


def _replay_episode(server: "VA_Server", ep_dir: Path) -> int:
    """Drive server.infer over every chunk in ep_dir. Returns number of
    chunks driven (== number of times hooks fired per Linear)."""
    chunks = sorted(
        (int(_CHUNK_RE.search(p.name).group(1)), p)
        for p in ep_dir.glob("obs_data_*.pt")
    )
    if not chunks:
        return 0

    first = torch.load(chunks[0][1], weights_only=False, map_location="cpu")
    prompt = first[0]["task"]

    server.infer({"reset": True, "prompt": prompt, "save_visualization": False})

    # Force fresh init_latent computation each chunk by resetting
    # frame_st_id (server._infer treats frame_st_id == 0 as the first
    # chunk and re-encodes obs via VAE).
    for _, chunk_path in chunks:
        obs_list = torch.load(chunk_path, weights_only=False, map_location="cpu")
        server.frame_st_id = 0
        server.infer({
            "obs": obs_list,
            "prompt": prompt,
            "save_visualization": False,
        })
    return len(chunks)


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
        help="FP bf16 transformer dir (transformer subdir inside).",
    )
    p.add_argument(
        "--replay_save_root", type=Path,
        default=Path("/tmp/derive_calib_replay"),
        help="Throwaway dir for server's per-call visualization saves "
             "during replay (kept off the calib video corpus).",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if not args.skip_ptq and (args.layer_config is None or args.int_weights_out is None):
        p.error("--layer_config and --int_weights_out required unless --skip_ptq")

    task_list = None
    if not args.all:
        task_list = [t.strip() for t in args.task_list.split(",") if t.strip()]
        logger.info(f"task subset: {task_list}")
    else:
        logger.info("task subset: ALL (--all)")

    ep_dirs = _filter_episodes(args.videos_root, task_list, args.all)
    if not ep_dirs:
        logger.error("no episode dirs matched; nothing to do.")
        return 1

    args.calib_out.parent.mkdir(parents=True, exist_ok=True)
    # Wipe any prior calib so replay produces a fresh aggregation.
    if args.calib_out.exists():
        logger.info(f"removing prior calib at {args.calib_out}")
        args.calib_out.unlink()

    logger.info(f"building bf16 server in-process from {args.model_path}")
    server = _build_server(args.model_path, args.replay_save_root)
    logger.info("installing calib hooks on 180 target Linears")
    state = install_calib_hooks(server.transformer, str(args.calib_out))

    total_chunks = 0
    for i, ep_dir in enumerate(ep_dirs):
        n = _replay_episode(server, ep_dir)
        total_chunks += n
        if (i + 1) % 10 == 0 or (i + 1) == len(ep_dirs):
            logger.info(
                f"replayed {i+1}/{len(ep_dirs)} episodes "
                f"({total_chunks} chunks total)"
            )

    state.dump()
    logger.info(f"final calib dumped to {args.calib_out}")

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
