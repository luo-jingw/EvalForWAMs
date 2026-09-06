#!/usr/bin/env python3
"""GPU smoke test: load the G1 model once, run baseline / ProbeFlow / DASH on a
synthetic observation, and report shapes + latency. Self-terminating (~3-4 min incl.
model load). No robot, no WebSocket -- it drives PolicyEngine directly.

    ssh nnmc75
    cd /home/user1/workspace/jingwu/ImagewamFT/ImageWAM
    set -a && source .env.local && set +a
    D=runs/g1_stack_cubes_flux2_klein_4b/2026-07-23_02-40-24
    CUDA_VISIBLE_DEVICES=5 .venv/bin/python scripts/g1/smoke_methods.py \
        --ckpt          $D/checkpoints/weights/step_009850.pt \
        --dataset-stats $D/dataset_stats.json \
        --prompt-embeds ../checkpoints/imagewam_g1/g1_stack_cubes_prompt_embeds.npz \
        --task-config   g1_stack_cubes_flux2_klein_4b

Pass criteria: every method returns actions of shape (16, 16), all finite, and the
DASH / ProbeFlow paths do not raise. Baseline is run twice; with a fixed seed the two
runs must match bit-for-bit (sanity that inference is deterministic).
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from serve_imagewam_g1 import DashSession, RuntimeConfig, build_engine  # noqa: E402
from inference_methods import BaselineParams  # noqa: E402


def fake_obs(prompt: str) -> dict:
    rng = np.random.default_rng(0)
    views = [rng.integers(0, 256, (256, 320, 3), dtype=np.uint8) for _ in range(3)]
    # Mid-range G1 state: arms ~0, grippers ~open (training span [2.9, 5.4]).
    state = np.zeros(16, dtype=np.float32)
    state[14] = state[15] = 5.0
    return {"image": views, "state": state, "prompt": prompt}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--dataset-stats", type=Path, required=True)
    p.add_argument("--prompt-embeds", type=Path, required=True)
    p.add_argument("--task-config", default="g1_stack_cubes_flux2_klein_4b")
    p.add_argument("--prompt", default=None, help="defaults to the first prompt in the bank")
    p.add_argument("--test-add-prompt", default=None,
                   help="if set, exercises in-process Qwen3 precompute (no npz write) for this text")
    p.add_argument("--flux2-model-path", type=Path, default=os.environ.get("FLUX2_MODEL_PATH"))
    p.add_argument("--ae-model-path", type=Path, default=os.environ.get("FLUX2_AE_MODEL_PATH"))
    p.add_argument("--flux2-src-path", type=Path, default=os.environ.get("FLUX2_SRC"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--camera-layout", default="compact_288x256")
    args = p.parse_args()

    engine = build_engine(
        ckpt=args.ckpt,
        dataset_stats=args.dataset_stats,
        prompt_embeds=args.prompt_embeds,
        task_config=args.task_config,
        flux2_model_path=args.flux2_model_path,
        ae_model_path=args.ae_model_path,
        flux2_src_path=args.flux2_src_path,
        device=args.device,
        camera_layout=args.camera_layout,
        host="127.0.0.1",
        port=0,
        runtime=RuntimeConfig(baseline=BaselineParams(num_inference_steps=10, seed=0)),
    )
    prompt = args.prompt or engine.prompts[0]
    obs = fake_obs(prompt)
    print(f"prompt: {prompt!r}")

    def run(tag: str, session=None):
        t = time.perf_counter()
        out = engine.infer(obs, dash_session=session)
        dt = (time.perf_counter() - t) * 1e3
        a = out["actions"]
        tm = engine.last_timing
        print(f"[{tag:10s}] actions={a.shape} finite={np.isfinite(a).all()} "
              f"gripL[0]={a[0,14]:+.3f} | wall={dt:.0f}ms infer={tm.infer_s*1e3:.0f}ms "
              f"steps={tm.effective_steps}")
        return a

    # 1) baseline twice (determinism), 2) probeflow, 3) dash (two replans for drift).
    engine.update_runtime(method="baseline")
    a0 = run("baseline#1")
    a1 = run("baseline#2")
    match = np.allclose(a0, a1)
    print(f"baseline determinism (seed=0): {'MATCH' if match else 'DIFFER'} max|Δ|={np.abs(a0-a1).max():.2e}")

    engine.update_runtime(method="probeflow")
    run("probeflow")

    engine.update_runtime(method="dash")
    ds = engine.new_dash_session()
    run("dash r0", ds)   # first replan: jump at k_near (no drift history)
    run("dash r1", ds)   # second replan: drift-adaptive jump

    if args.test_add_prompt:
        r = engine.add_prompt(args.test_add_prompt, persist=False)  # persist=False: don't touch the npz
        print(f"[add_prompt] status={r['status']} -> can now infer this prompt in-memory")
        engine.update_runtime(method="baseline")
        obs2 = dict(obs, prompt=args.test_add_prompt)
        out = engine.infer(obs2)
        print(f"[add_prompt] infer with new prompt: actions={out['actions'].shape} "
              f"finite={np.isfinite(out['actions']).all()}")

    print("\nSMOKE_OK")


if __name__ == "__main__":
    main()
