"""In-process ImageWAM RoboTwin eval runner (server-less, variant-aware).

Forks ImageWAM's experiments/robotwin/eval_robotwin_single.py (hydra-compose the
flux2 model config -> model_overrides, then subprocess RoboTwin's
script/eval_policy.py) and adds the ptqeval bits: the robotwin_policy symlink,
the ViDiT-Q variant / int_weights / layer_config / calib_out / perf keys, and
SR-parse -> res.json in the ptqeval.eval.aggregator contract.

Unlike fastwam's runner (flat scalar overrides), ImageWAM's get_model rebuilds
the model from `model_overrides` = the resolved cfg.model dict; we compose
sim_robotwin.yaml with `task=<flux2 task>` and fill the model's flux2/AE/src
paths, then forward the whole cfg.model (ImageWAM's get_model filters it to the
create fn's kwargs). Value serialization mirrors ImageWAM's repr()-based
_format_override_value so RoboTwin's eval_policy.py literal_evals them back.

CLI (per task; eval_pool fans this over GPUs):
  python -m ptqeval.wam.imagewam.eval_runner \
    --task click_alarmclock --gpu 0 \
    --ckpt .../imagewam_release/robotwin/flux2_klein_4b/model.pt \
    --flux2_model_path .../FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors \
    --ae_model_path    .../FLUX.2-dev/ae.safetensors \
    --dataset_stats    .../flux2_klein_4b/dataset_stats.json \
    --task_config demo_clean --eval_num_episodes 100 \
    --variant viditq --int_weights_ckpt .../int_weights_clean.pth \
    --layer_config PTQEval/ptqeval/wam/imagewam/method/viditq/configs/w4a4_smooth.yaml \
    --save_root results/imagewam/imagewam_w4a4_smooth_clean
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

import ptqeval.wam.imagewam  # noqa: F401  (puts ImageWAM paths on sys.path)
from ptqeval.wam.imagewam import IMAGEWAM_FLUX2_SRC, IMAGEWAM_ROOT

POLICY_NAME = "imagewam_viditq"
_POLICY_SRC = Path(__file__).resolve().parent / "robotwin_policy"


def _abspath(p: str) -> str:
    return str(Path(p).expanduser().resolve())


def _format_override_value(value: Any) -> str:
    # Mirrors ImageWAM eval_robotwin_single._format_override_value.
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (dict, list, tuple)):
        return repr(value)
    return repr(str(value))


def _append_override(ovr: list[str], key: str, value: Any, *, skip_none: bool = True) -> None:
    if skip_none and value is None:
        return
    ovr.extend([f"--{key}", _format_override_value(value)])


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


def _compose_model_overrides(hydra_task: str, flux2_model_path: str,
                             ae_model_path: str, flux2_src_path: str | None) -> dict:
    """Compose sim_robotwin.yaml with the flux2 task + filled paths; return the
    resolved cfg.model dict (ImageWAM get_model filters it to the create kwargs)."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    if flux2_src_path is None:
        flux2_src_path = IMAGEWAM_FLUX2_SRC
    configs_root = os.path.join(IMAGEWAM_ROOT, "configs")
    overrides = [
        f"task={hydra_task}",
        f"model.flux2_model_path={flux2_model_path}",
        f"model.ae_model_path={ae_model_path}",
        f"model.flux2_src_path={flux2_src_path}",
    ]
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=configs_root):
        cfg = compose(config_name="sim_robotwin.yaml", overrides=overrides)
    return OmegaConf.to_container(cfg.model, resolve=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="RoboTwin task_name")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--ckpt", required=True, help="ImageWAM released model.pt (ckpt_setting)")
    ap.add_argument("--flux2_model_path", required=True)
    ap.add_argument("--ae_model_path", required=True)
    ap.add_argument("--flux2_src_path", default=None)
    ap.add_argument("--hydra_task", default="robotwin_flux2_klein_4b_base_imagewam")
    ap.add_argument("--dataset_stats", required=True)
    ap.add_argument("--robotwin_root", default=None,
                    help="default: ImageWAM/third_party/RoboTwin")
    ap.add_argument("--task_config", default="demo_randomized")
    ap.add_argument("--instruction_type", default="unseen")
    ap.add_argument("--eval_num_episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_root", required=True)
    ap.add_argument("--replan_steps", type=int, default=8)
    ap.add_argument("--num_inference_steps", type=int, default=10)
    ap.add_argument("--robotwin_camera_layout", default="compact_288x256")
    # variant dispatch
    ap.add_argument("--variant", default="", choices=["", "bf16", "viditq"])
    ap.add_argument("--int_weights_ckpt", default=None)
    ap.add_argument("--layer_config", default=None)
    ap.add_argument("--profile_ops", default="true")
    ap.add_argument("--calib_out", default=None,
                    help="On-policy SmoothQuant calib (FP model + absmax hooks); "
                         "requires variant bf16, run with --eval_num_episodes 1.")
    args = ap.parse_args()

    if args.variant == "viditq" and not args.int_weights_ckpt:
        ap.error("--variant viditq requires --int_weights_ckpt")
    if args.calib_out and args.variant == "viditq":
        ap.error("--calib_out needs the FP model; use --variant bf16 (or '').")

    robotwin_root = Path(_abspath(args.robotwin_root)) if args.robotwin_root \
        else Path(IMAGEWAM_ROOT) / "third_party" / "RoboTwin"
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

    sim_cfg_path = os.path.join(IMAGEWAM_ROOT, "configs", "sim_robotwin.yaml")
    model_overrides = _compose_model_overrides(
        args.hydra_task, _abspath(args.flux2_model_path), _abspath(args.ae_model_path),
        _abspath(args.flux2_src_path) if args.flux2_src_path else None)

    ovr: list[str] = []
    _append_override(ovr, "task_name", args.task)
    _append_override(ovr, "task_config", args.task_config)
    _append_override(ovr, "ckpt_setting", _abspath(args.ckpt))
    _append_override(ovr, "seed", args.seed)
    _append_override(ovr, "policy_name", POLICY_NAME)
    _append_override(ovr, "instruction_type", args.instruction_type)
    _append_override(ovr, "eval_num_episodes", args.eval_num_episodes)
    _append_override(ovr, "sim_cfg_path", sim_cfg_path)
    _append_override(ovr, "sim_task", args.hydra_task)
    _append_override(ovr, "eval_output_dir", eval_output_dir)
    _append_override(ovr, "mixed_precision", "bf16")
    _append_override(ovr, "device", "cuda")
    _append_override(ovr, "dataset_stats_path", _abspath(args.dataset_stats))
    _append_override(ovr, "replan_steps", args.replan_steps)
    _append_override(ovr, "num_inference_steps", args.num_inference_steps)
    _append_override(ovr, "rand_device", "cpu")
    _append_override(ovr, "tiled", False)
    _append_override(ovr, "timing_enabled", False)
    _append_override(ovr, "robotwin_camera_layout", args.robotwin_camera_layout)
    _append_override(ovr, "skip_get_obs_within_replan", True)
    _append_override(ovr, "model_overrides", model_overrides)
    # ptqeval variant dispatch (read by ptqeval.wam.imagewam.policy.get_model)
    _append_override(ovr, "variant", args.variant)
    _append_override(ovr, "int_weights_ckpt",
                     _abspath(args.int_weights_ckpt) if args.int_weights_ckpt else None)
    _append_override(ovr, "layer_config",
                     _abspath(args.layer_config) if args.layer_config else None)
    _append_override(ovr, "profile_ops", args.profile_ops)
    _append_override(ovr, "perf_log_path", perf_log_path)
    _append_override(ovr, "calib_out",
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


def _write_res_json(save_root: str, seed: int, task: str, suc: int, total: int) -> None:
    out_dir = os.path.join(save_root, f"stseed-{seed}", "metrics", task)
    os.makedirs(out_dir, exist_ok=True)
    res = {"task_name": task, "succ_num": suc, "total_num": total,
           "succ_rate": (suc / total) if total else 0.0}
    with open(os.path.join(out_dir, "res.json"), "w") as fp:
        json.dump(res, fp, indent=2)
    print(f"[eval_runner] wrote res.json: {task} SR={suc}/{total}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
