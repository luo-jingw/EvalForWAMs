"""In-process FastWAM RoboTwin eval runner (server-less), variant-aware.

Forked from FastWAM's experiments/robotwin/eval_robotwin_single.py. Differences:
  - symlinks OUR RoboTwin policy package (ptqeval/wam/fastwam/robotwin_policy)
    into <robotwin_root>/policy/fastwam_viditq, so RoboTwin's eval_policy.py
    loads ptqeval.wam.fastwam.policy (variant dispatch + PerfProbe).
  - passes variant / int_weights_ckpt / layer_config / perf_log_path through
    the override plumbing so a quantized variant can be evaluated.
  - plain argparse (no hydra); one task, one GPU. Multi-GPU pooling wraps this.

Runs RoboTwin's script/eval_policy.py as a subprocess with cwd=robotwin_root,
exactly like the FastWAM runner, so the episode loop + SR accounting are the
upstream harness (unchanged).

Example (bf16):
  python -m ptqeval.wam.fastwam.eval_runner \
      --ckpt models/robotwin2.0-fastwam/robotwin_uncond_3cam_384.pt \
      --dataset_stats models/robotwin2.0-fastwam/robotwin_uncond_3cam_384_dataset_stats.json \
      --robotwin_root FastWAM/third_party/RoboTwin \
      --task click_alarmclock --eval_num_episodes 5 --gpu 4 \
      --save_root results/fastwam/fastwam_bf16

Example (viditq w4a4):
  python -m ptqeval.wam.fastwam.eval_runner ... \
      --variant viditq \
      --int_weights_ckpt results/fastwam/fastwam_w4a4/calib/int_weights.pth \
      --layer_config PTQEval/ptqeval/wam/fastwam/method/viditq/configs/w4a4.yaml \
      --save_root results/fastwam/fastwam_w4a4
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import ptqeval.wam.fastwam  # noqa: F401  (puts FastWAM paths on sys.path)
from ptqeval.wam.fastwam import FASTWAM_ROOT

POLICY_NAME = "fastwam_viditq"
_POLICY_SRC = Path(__file__).resolve().parent / "robotwin_policy"


def _abspath(p: str) -> str:
    return str(Path(p).expanduser().resolve())


def _ensure_policy_symlink(robotwin_root: Path) -> None:
    policy_root = robotwin_root / "policy"
    if not policy_root.is_dir():
        raise FileNotFoundError(f"RoboTwin policy dir not found: {policy_root}")
    target = policy_root / POLICY_NAME
    src = _POLICY_SRC.resolve()
    if target.is_symlink():
        if target.resolve() != src:
            raise RuntimeError(f"policy symlink conflict: {target} -> {target.resolve()}")
        return
    if target.exists():
        raise RuntimeError(f"{target} exists and is not a symlink; handle manually.")
    target.symlink_to(src, target_is_directory=True)


def _override(ovr: list[str], key: str, value) -> None:
    if value is None:
        value = "None"
    elif isinstance(value, bool):
        value = "True" if value else "False"
    else:
        value = repr(str(value)) if not isinstance(value, (int, float)) else str(value)
    ovr.extend([f"--{key}", value])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset_stats", required=True)
    ap.add_argument("--robotwin_root", default="FastWAM/third_party/RoboTwin")
    ap.add_argument("--task", required=True)
    ap.add_argument("--task_config", default="demo_randomized")
    ap.add_argument("--instruction_type", default="unseen")
    ap.add_argument("--eval_num_episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--save_root", required=True,
                    help="Results root; perf JSONL goes to <save_root>/perf/<task>.jsonl.")
    ap.add_argument("--sim_task", default="robotwin_uncond_3cam_384_1e-4")
    ap.add_argument("--replan_steps", type=int, default=24)
    ap.add_argument("--num_inference_steps", type=int, default=10)
    # variant dispatch
    ap.add_argument("--variant", default="", choices=["", "bf16", "viditq"])
    ap.add_argument("--int_weights_ckpt", default=None)
    ap.add_argument("--layer_config", default=None)
    ap.add_argument("--profile_ops", default="true")
    ap.add_argument("--calib_out", default=None,
                    help="On-policy SmoothQuant calibration: FP model + absmax "
                         "hooks, merge per-channel absmax into this path. "
                         "Requires variant bf16; run with --eval_num_episodes 1.")
    args = ap.parse_args()

    if args.variant == "viditq" and not args.int_weights_ckpt:
        ap.error("--variant viditq requires --int_weights_ckpt")
    if args.calib_out and args.variant == "viditq":
        ap.error("--calib_out needs the FP model; use --variant bf16 (or '').")

    robotwin_root = Path(_abspath(args.robotwin_root))
    if not robotwin_root.exists():
        raise FileNotFoundError(f"RoboTwin root not found: {robotwin_root}")
    _ensure_policy_symlink(robotwin_root)

    save_root = _abspath(args.save_root)
    perf_dir = os.path.join(save_root, "perf")
    os.makedirs(perf_dir, exist_ok=True)
    # ptqeval.eval.aggregator keys perf logs by filename via the pattern
    # "<task>_rank<N>_<YYYYMMDD>_<HHMMSS>.jsonl"; conform so the shared
    # aggregator picks them up unchanged. rank = gpu index (one worker/gpu).
    perf_log_path = os.path.join(
        perf_dir, f"{args.task}_rank{args.gpu}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    )
    eval_output_dir = os.path.join(save_root, "robotwin_eval", args.task)
    os.makedirs(eval_output_dir, exist_ok=True)

    sim_cfg_path = os.path.join(FASTWAM_ROOT, "configs", "sim_robotwin.yaml")

    ovr: list[str] = []
    _override(ovr, "task_name", args.task)
    _override(ovr, "task_config", args.task_config)
    _override(ovr, "ckpt_setting", _abspath(args.ckpt))
    _override(ovr, "seed", args.seed)
    _override(ovr, "policy_name", POLICY_NAME)
    _override(ovr, "instruction_type", args.instruction_type)
    _override(ovr, "eval_num_episodes", args.eval_num_episodes)
    _override(ovr, "sim_cfg_path", sim_cfg_path)
    _override(ovr, "sim_task", args.sim_task)
    _override(ovr, "eval_output_dir", eval_output_dir)
    _override(ovr, "mixed_precision", "bf16")
    _override(ovr, "device", "cuda")
    _override(ovr, "dataset_stats_path", _abspath(args.dataset_stats))
    _override(ovr, "replan_steps", args.replan_steps)
    _override(ovr, "num_inference_steps", args.num_inference_steps)
    _override(ovr, "rand_device", "cpu")
    _override(ovr, "tiled", False)
    _override(ovr, "timing_enabled", False)
    _override(ovr, "skip_get_obs_within_replan", True)
    # variant dispatch (abspath: eval_policy.py runs with cwd=robotwin_root)
    _override(ovr, "variant", args.variant)
    _override(ovr, "int_weights_ckpt",
              _abspath(args.int_weights_ckpt) if args.int_weights_ckpt else None)
    _override(ovr, "layer_config",
              _abspath(args.layer_config) if args.layer_config else None)
    _override(ovr, "profile_ops", args.profile_ops)
    _override(ovr, "perf_log_path", perf_log_path)
    _override(ovr, "calib_out",
              _abspath(args.calib_out) if args.calib_out else None)

    cmd = [
        sys.executable, "-u", "script/eval_policy.py",
        "--config", f"policy/{POLICY_NAME}/deploy_policy.yml",
        "--overrides", *ovr,
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["SAPIEN_USE_VULKAN_DEVICE_ID"] = str(args.gpu)
    env["PYTHONUNBUFFERED"] = "1"

    print(f"[eval_runner] task={args.task} variant={args.variant or 'bf16'} "
          f"gpu={args.gpu} perf={perf_log_path}", flush=True)
    # Tee subprocess stdout to console while capturing the final SR line, which
    # RoboTwin's eval_policy.py prints as "Success rate: <suc>/<total>".
    proc = subprocess.Popen(
        cmd, cwd=str(robotwin_root), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    # RoboTwin prints the SR line with ANSI color codes around the numbers
    # ("Success rate: \x1b[96m2/2\x1b[0m ..."); strip them before matching.
    ansi_re = re.compile(r"\x1b\[[0-9;]*m")
    sr_re = re.compile(r"Success rate:\s*(\d+)\s*/\s*(\d+)")
    last_suc: int | None = None
    last_total: int | None = None
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        m = sr_re.search(ansi_re.sub("", line))
        if m:
            last_suc, last_total = int(m.group(1)), int(m.group(2))
    rc = proc.wait()

    # Emit res.json in the ptqeval.eval.aggregator contract so the shared
    # aggregator (unchanged) can read fastwam SR alongside its perf JSONL.
    if last_suc is not None and last_total:
        _write_res_json(save_root, args.seed, args.task, last_suc, last_total)
    else:
        print(f"[eval_runner] WARNING: no 'Success rate' line captured for "
              f"{args.task}; res.json not written.", flush=True)
    return rc


def _write_res_json(save_root: str, seed: int, task: str, suc: int, total: int) -> None:
    out_dir = os.path.join(save_root, f"stseed-{seed}", "metrics", task)
    os.makedirs(out_dir, exist_ok=True)
    res = {
        "task_name": task,
        "succ_num": suc,
        "total_num": total,
        "succ_rate": (suc / total) if total else 0.0,
    }
    with open(os.path.join(out_dir, "res.json"), "w") as fp:
        json.dump(res, fp, indent=2)
    print(f"[eval_runner] wrote res.json: {task} SR={suc}/{total}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
