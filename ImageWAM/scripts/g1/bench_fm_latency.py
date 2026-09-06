#!/usr/bin/env python3
"""Directly measure the flow-matching (denoise) latency per method via the backbone's
CUDA-event profiler (profile_gpu_event_timing=True), instead of deriving it from a
step-count slope.

Per call it reads output["timing"]:
    fm_loop   = segments["action_denoise_loop_s"]   # the whole FM sampling loop
    fm_predict= action_predict_total_s              # pure denoise-network forwards (sum over steps)
    nfe       = num_inference_steps                 # actual forward count

Runs baseline 10/3/2, ProbeFlow, DASH on the same consecutive real frames (DASH keeps
one drift session), and reports mean/std/min/p50/p90/max of the FM latency per method.
Drives model.infer_action directly (reusing the engine's preprocessing) so numbers match
the production path. One GPU, self-terminating.

    ssh nnmc61
    S=/shared/user75/workspace/jingwu/ImagewamFT
    cd $S/ImageWAM
    D=runs/g1_stack_cubes_flux2_klein_4b/2026-07-23_02-40-24
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTHONPATH=$S/ImageWAM/src \
      $S/ImageWAM/.venv/bin/python scripts/g1/bench_fm_latency.py \
        --ckpt $D/checkpoints/weights/step_009850.pt --dataset-stats $D/dataset_stats.json \
        --prompt-embeds ../checkpoints/imagewam_g1/g1_stack_cubes_prompt_embeds.npz \
        --task-config g1_stack_cubes_flux2_klein_4b \
        --dataset-root /shared/user64/workspace/yuhao/pi/data/stack-v3 \
        --prompt "Stack the blocks by color: put the red block in the center, then stack the blue block on the red block, then stack the yellow block on the blue block." \
        --flux2-model-path $S/checkpoints/flux2/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors \
        --ae-model-path $S/checkpoints/flux2/FLUX.2-dev/ae.safetensors \
        --flux2-src-path $S/ImageWAM/third_party/flux2 \
        --episode 1 --num-calls 40 --warmup 5 --stride 8
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from serve_imagewam_g1 import RuntimeConfig, build_engine  # noqa: E402
from inference_methods import (  # noqa: E402
    BaselineParams, ProbeFlowParams, DashParams,
    build_baseline_kwargs, build_probeflow_kwargs, FLUX2_REF_ENCODE_TIME_VALUE,
)
from bench_methods import build_obs_sequence, load_task, resolve_prompt  # noqa: E402


def stats(xs: list[float]) -> dict:
    a = np.asarray(xs, dtype=np.float64)
    return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "min": float(a.min()), "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90)), "max": float(a.max())}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--dataset-stats", type=Path, required=True)
    p.add_argument("--prompt-embeds", type=Path, required=True)
    p.add_argument("--task-config", default="g1_stack_cubes_flux2_klein_4b")
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--episode", type=int, default=1)
    p.add_argument("--num-calls", type=int, default=40)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--prompt", default=None)
    p.add_argument("--flux2-model-path", type=Path, default=os.environ.get("FLUX2_MODEL_PATH"))
    p.add_argument("--ae-model-path", type=Path, default=os.environ.get("FLUX2_AE_MODEL_PATH"))
    p.add_argument("--flux2-src-path", type=Path, default=os.environ.get("FLUX2_SRC"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--camera-layout", default="compact_288x256")
    args = p.parse_args()

    engine = build_engine(
        ckpt=args.ckpt, dataset_stats=args.dataset_stats, prompt_embeds=args.prompt_embeds,
        task_config=args.task_config, flux2_model_path=args.flux2_model_path,
        ae_model_path=args.ae_model_path, flux2_src_path=args.flux2_src_path,
        device=args.device, camera_layout=args.camera_layout, host="127.0.0.1", port=0,
        runtime=RuntimeConfig(baseline=BaselineParams(num_inference_steps=10, seed=0)),
    )
    model = engine._model
    PROF = {"profile_gpu_event_timing": True}

    total = args.warmup + args.num_calls
    frame_idxs = [i * args.stride for i in range(total)]
    candidate = args.prompt or load_task(args.dataset_root, args.episode)
    prompt = resolve_prompt(candidate, engine.prompts)
    if prompt is None:
        raise SystemExit(f"prompt not in bank: {candidate!r}")
    obs_seq = build_obs_sequence(args.dataset_root, args.episode, frame_idxs, prompt)
    print(f"dataset {args.dataset_root}  ep {args.episode}  frames: warmup {args.warmup} + timed {args.num_calls}\n")

    baseline_kw = {
        "10step": build_baseline_kwargs(BaselineParams(num_inference_steps=10, seed=0)),
        "3step": build_baseline_kwargs(BaselineParams(num_inference_steps=3, seed=0)),
        "2step": build_baseline_kwargs(BaselineParams(num_inference_steps=2, seed=0)),
        "probeflow": build_probeflow_kwargs(ProbeFlowParams(seed=0)),
    }
    dash_params = DashParams(seed=0)
    session = engine.new_dash_session()

    acc: dict[str, dict[str, list]] = {
        k: {"loop": [], "predict": [], "nfe": []}
        for k in ("10step", "3step", "2step", "probeflow", "dash")
    }

    _dumped = {"done": False}

    def read_timing(pred, label="") -> tuple[float, float, int]:
        t = pred.get("timing")
        if t is None:
            if not _dumped["done"]:
                _dumped["done"] = True
                print(f"[debug] {label}: no 'timing'; pred keys = {sorted(pred.keys())}")
            return float("nan"), float("nan"), 0
        seg = t.get("segments", {})
        loop = float(seg.get("action_denoise_loop_s", float("nan"))) * 1e3
        predict = float(t.get("action_predict_total_s", float("nan"))) * 1e3
        nfe = int(t.get("num_inference_steps", 0))
        if not _dumped["done"]:
            _dumped["done"] = True
            print(f"[debug] {label}: timing segments = {sorted(seg.keys())}")
        return loop, predict, nfe

    with torch.no_grad():
        for i, obs in enumerate(obs_seq):
            image = engine._images(obs["image"])
            proprio = engine._state.normalize_state(np.asarray(obs["state"], dtype=np.float32).reshape(-1))
            context, cmask = engine._prompts.get(prompt)
            base = dict(prompt=None, input_image=image, action_horizon=engine._runtime.action_horizon,
                        proprio=proprio, context=context, context_mask=cmask)

            rec = {}
            for label in ("10step", "3step", "2step", "probeflow"):
                pred = model.infer_action(**base, **baseline_kw[label], **PROF)
                rec[label] = read_timing(pred, label)

            ctrl = session.controller(dash_params, engine._scheduler_shift)
            ref_tokens, ref_ids = model._encode_flux2_image_tokens(image, time_value=FLUX2_REF_ENCODE_TIME_VALUE)
            plan = ctrl.plan_replan(ref_tokens, ref_ids, replan_index=ctrl._replan_count)
            dash_base = {"num_inference_steps": int(dash_params.num_inference_steps),
                         "sigma_shift": dash_params.sigma_shift, "seed": dash_params.seed}
            pred = model.infer_action(**base, **dash_base, **plan.infer_kwargs, **PROF)
            ctrl.commit(ref_tokens)
            rec["dash"] = read_timing(pred, "dash")

            if i >= args.warmup:
                for label, (loop, predict, nfe) in rec.items():
                    acc[label]["loop"].append(loop)
                    acc[label]["predict"].append(predict)
                    acc[label]["nfe"].append(nfe)

    print("=" * 96)
    print(f"Flow-matching latency (CUDA-event measured, ms) over {args.num_calls} frames")
    print(f"{'method':10s} {'nfe':>5s} | {'FM loop: mean':>13s} {'std':>6s} {'min':>7s} {'p50':>7s} "
          f"{'p90':>7s} {'max':>7s} | {'predict_only':>12s}")
    print("-" * 96)
    out = {}
    for label in ("10step", "3step", "2step", "probeflow", "dash"):
        s = stats(acc[label]["loop"])
        sp = stats(acc[label]["predict"])
        nfe = float(np.mean(acc[label]["nfe"]))
        out[label] = {"nfe_mean": nfe, "fm_loop_ms": s, "fm_predict_ms": sp}
        print(f"{label:10s} {nfe:5.2f} | {s['mean']:13.1f} {s['std']:6.1f} {s['min']:7.1f} {s['p50']:7.1f} "
              f"{s['p90']:7.1f} {s['max']:7.1f} | {sp['mean']:12.1f}")
    print("=" * 96)
    print("FM loop = segments[action_denoise_loop_s] (denoise sampling loop: network forwards +\n"
          "scheduler steps + speculative verify). predict_only = action_predict_total_s (pure network\n"
          "forwards). nfe = actual forward passes.")
    print("\nJSON " + json.dumps(out))
    print("\nFM_OK")


if __name__ == "__main__":
    main()
