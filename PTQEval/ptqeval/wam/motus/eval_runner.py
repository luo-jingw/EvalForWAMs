"""In-process Motus RoboTwin eval runner (server-less, variant-aware).

Motus uses the standard external RoboTwin (no vendored sim of its own): the
policy package is symlinked into <robotwin>/policy/<name> and RoboTwin's
script/eval_policy.py imports it and drives get_model/eval/reset_model. Motus's
get_model reads its own inference/robotwin/Motus/utils/robotwin.yml for the model
config, so the overrides are FLAT scalars (like fastwam, unlike imagewam's
hydra model_overrides): the ONLY extra keys vs fastwam are --wan_path / --vlm_path
(required by Motus get_model) and the ptqeval ViDiT-Q variant keys.

Adds: the robotwin_policy symlink, variant/int_weights/layer_config/calib_out/
perf keys, and SR-parse -> res.json in the ptqeval.eval.aggregator contract.

CLI (per task; eval_pool fans over GPUs):
  python -m ptqeval.wam.motus.eval_runner --task click_alarmclock --gpu 0 \
    --ckpt models/motus_ckpt/Motus_robotwin2 \
    --wan_path models/motus_ckpt/Wan2.2-TI2V-5B \
    --vlm_path models/motus_ckpt/Qwen3-VL-2B-Instruct \
    --task_config demo_clean --eval_num_episodes 100 \
    --variant viditq --int_weights_ckpt .../int_weights_clean.pth \
    --layer_config PTQEval/ptqeval/wam/motus/method/viditq/configs/w4a4_smooth.yaml \
    --save_root results/motus/motus_w4a4_smooth_clean
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
from typing import Any

import ptqeval.wam.motus  # noqa: F401  (puts Motus paths on sys.path)

POLICY_NAME = "motus_viditq"
_POLICY_SRC = Path(__file__).resolve().parent / "robotwin_policy"


def _abspath(p: str) -> str:
    return str(Path(p).expanduser().resolve())


def _default_robotwin_root() -> Path:
    root = Path(__file__).resolve().parents[4]  # repo root
    for cand in (root / "FastWAM" / "third_party" / "RoboTwin", root / "RoboTwin"):
        if (cand / "script" / "eval_policy.py").is_file():
            return cand
    raise SystemExit(f"no RoboTwin with script/eval_policy.py found under {root}")


def _override(ovr: list[str], key: str, value: Any) -> None:
    if value is None:
        value = "None"
    elif isinstance(value, bool):
        value = "True" if value else "False"
    else:
        value = str(value)
    ovr.extend([f"--{key}", value])


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


def _write_res_json(save_root: str, seed: int, task: str, suc: int, total: int) -> None:
    out_dir = os.path.join(save_root, f"stseed-{seed}", "metrics", task)
    os.makedirs(out_dir, exist_ok=True)
    res = {"task_name": task, "succ_num": suc, "total_num": total,
           "succ_rate": (suc / total) if total else 0.0}
    with open(os.path.join(out_dir, "res.json"), "w") as fp:
        json.dump(res, fp, indent=2)
    print(f"[eval_runner] wrote res.json: {task} SR={suc}/{total}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="RoboTwin task_name")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--ckpt", required=True, help="Motus release ckpt dir (DeepSpeed)")
    ap.add_argument("--wan_path", required=True, help="WAN dir (config + Wan2.2_VAE.pth)")
    ap.add_argument("--vlm_path", required=True, help="Qwen3-VL dir")
    ap.add_argument("--robotwin_root", default=None,
                    help="default: FastWAM/third_party/RoboTwin (has script/eval_policy.py)")
    ap.add_argument("--task_config", default="demo_randomized")
    ap.add_argument("--instruction_type", default="unseen")
    ap.add_argument("--eval_num_episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_root", required=True)
    # variant dispatch
    ap.add_argument("--variant", default="", choices=["", "bf16", "viditq"])
    ap.add_argument("--int_weights_ckpt", default=None)
    ap.add_argument("--layer_config", default=None)
    ap.add_argument("--profile_ops", default="true")
    ap.add_argument("--calib_out", default=None)
    args = ap.parse_args()

    if args.variant == "viditq" and not args.int_weights_ckpt:
        ap.error("--variant viditq requires --int_weights_ckpt")
    if args.calib_out and args.variant == "viditq":
        ap.error("--calib_out needs the FP model; use --variant bf16 (or '').")

    robotwin_root = Path(_abspath(args.robotwin_root)) if args.robotwin_root \
        else _default_robotwin_root()
    if not robotwin_root.exists():
        raise FileNotFoundError(f"RoboTwin root not found: {robotwin_root}")
    _ensure_policy_symlink(robotwin_root)

    save_root = _abspath(args.save_root)
    perf_dir = os.path.join(save_root, "perf")
    os.makedirs(perf_dir, exist_ok=True)
    perf_log_path = os.path.join(
        perf_dir, f"{args.task}_rank{args.gpu}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    eval_output_dir = os.path.join(save_root, "robotwin_eval", args.task)
    os.makedirs(eval_output_dir, exist_ok=True)

    ovr: list[str] = []
    _override(ovr, "task_name", args.task)
    _override(ovr, "task_config", args.task_config)
    _override(ovr, "ckpt_setting", _abspath(args.ckpt))
    _override(ovr, "seed", args.seed)
    _override(ovr, "policy_name", POLICY_NAME)
    _override(ovr, "instruction_type", args.instruction_type)
    _override(ovr, "eval_num_episodes", args.eval_num_episodes)
    _override(ovr, "eval_output_dir", eval_output_dir)
    # Motus get_model requires these two extra backbone paths:
    _override(ovr, "wan_path", _abspath(args.wan_path))
    _override(ovr, "vlm_path", _abspath(args.vlm_path))
    # ptqeval variant dispatch (read by ptqeval.wam.motus.policy.get_model)
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
    proc = subprocess.Popen(
        cmd, cwd=str(robotwin_root), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
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

    if last_suc is not None and last_total:
        _write_res_json(save_root, args.seed, args.task, last_suc, last_total)
    else:
        print(f"[eval_runner] WARNING: no 'Success rate' line captured for "
              f"{args.task}; res.json not written.", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
