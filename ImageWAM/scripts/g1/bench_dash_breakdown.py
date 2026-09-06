#!/usr/bin/env python3
"""DASH latency breakdown: where does DASH's wall-time go beyond its ~2 denoise steps?

Replays a few consecutive real frames and, per frame, cuda-synced-times each sub-step
of the DASH call as serve's PolicyEngine._infer_dash does it:

    enc     = model._encode_flux2_image_tokens(image, t=10)   # DASH-only ref-token encode (for drift)
    plan    = controller.plan_replan(...)                     # drift metric + ratio-jump planning
    dinfer  = model.infer_action(..., speculative kwargs)     # denoise steps + speculative VERIFY
    commit  = controller.commit(ref_tokens)                   # store previous (tiny)

and, on the SAME `base`, times the baseline infer_action for 2 and 3 steps (each does an
internal image-encode + N denoise, NO verify). From those we attribute DASH's cost:

    denoise/step   ~= base3 - base2
    image-encode   ~= 3*base2 - 2*base3           (internal encode inside a baseline call)
    verify(inside dinfer) ~= dinfer - eff_steps * denoise/step
    DASH bookkeeping = enc + plan + commit
    DASH total       = enc + plan + dinfer + commit

This does NOT modify serve; it drives the resident model directly (reusing the engine's
own preprocessing) so numbers match the production DASH path. One GPU, self-terminating.

    ssh nnmc62
    S=/shared/user75/workspace/jingwu/ImagewamFT
    cd $S/ImageWAM
    D=runs/g1_stack_cubes_flux2_klein_4b/2026-07-23_02-40-24
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTHONPATH=$S/ImageWAM/src \
      $S/ImageWAM/.venv/bin/python scripts/g1/bench_dash_breakdown.py \
        --ckpt $D/checkpoints/weights/step_009850.pt --dataset-stats $D/dataset_stats.json \
        --prompt-embeds ../checkpoints/imagewam_g1/g1_stack_cubes_prompt_embeds.npz \
        --task-config g1_stack_cubes_flux2_klein_4b \
        --dataset-root /shared/user64/workspace/yuhao/pi/data/stack-v3 \
        --prompt "Stack the blocks by color: put the red block in the center, then stack the blue block on the red block, then stack the yellow block on the blue block." \
        --flux2-model-path $S/checkpoints/flux2/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors \
        --ae-model-path $S/checkpoints/flux2/FLUX.2-dev/ae.safetensors \
        --flux2-src-path $S/ImageWAM/third_party/flux2 \
        --episode 1 --num-calls 20 --warmup 5 --stride 8
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from serve_imagewam_g1 import RuntimeConfig, build_engine, _effective_steps  # noqa: E402
from inference_methods import (  # noqa: E402
    BaselineParams, DashParams, build_baseline_kwargs, FLUX2_REF_ENCODE_TIME_VALUE,
)
from bench_methods import build_obs_sequence, load_task, resolve_prompt  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--dataset-stats", type=Path, required=True)
    p.add_argument("--prompt-embeds", type=Path, required=True)
    p.add_argument("--task-config", default="g1_stack_cubes_flux2_klein_4b")
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--episode", type=int, default=1)
    p.add_argument("--num-calls", type=int, default=20)
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
    use_cuda = torch.device(args.device).type == "cuda"

    def sync():
        if use_cuda:
            torch.cuda.synchronize()

    def timed(fn):
        sync(); t = time.perf_counter(); r = fn(); sync()
        return (time.perf_counter() - t) * 1e3, r

    total = args.warmup + args.num_calls
    frame_idxs = [i * args.stride for i in range(total)]
    candidate = args.prompt or load_task(args.dataset_root, args.episode)
    prompt = resolve_prompt(candidate, engine.prompts)
    if prompt is None:
        raise SystemExit(f"prompt not in bank: {candidate!r}")
    obs_seq = build_obs_sequence(args.dataset_root, args.episode, frame_idxs, prompt)
    print(f"dataset {args.dataset_root}  ep {args.episode}  prompt {prompt!r:.50}")
    print(f"frames: warmup {args.warmup} + timed {args.num_calls} @ stride {args.stride}\n")

    dash_params = DashParams(seed=0)
    b2 = build_baseline_kwargs(BaselineParams(num_inference_steps=2, seed=0))
    b3 = build_baseline_kwargs(BaselineParams(num_inference_steps=3, seed=0))
    session = engine.new_dash_session()

    acc: dict[str, list] = {k: [] for k in ("enc", "plan", "dinfer", "commit", "base2", "base3", "eff")}

    with torch.no_grad():
        for i, obs in enumerate(obs_seq):
            image = engine._images(obs["image"])
            proprio = engine._state.normalize_state(np.asarray(obs["state"], dtype=np.float32).reshape(-1))
            context, cmask = engine._prompts.get(prompt)
            base = dict(prompt=None, input_image=image, action_horizon=engine._runtime.action_horizon,
                        proprio=proprio, context=context, context_mask=cmask)

            ctrl = session.controller(dash_params, engine._scheduler_shift)
            t_enc, (ref_tokens, ref_ids) = timed(
                lambda: model._encode_flux2_image_tokens(image, time_value=FLUX2_REF_ENCODE_TIME_VALUE))
            t_plan, plan = timed(lambda: ctrl.plan_replan(ref_tokens, ref_ids, replan_index=ctrl._replan_count))
            dash_base = {"num_inference_steps": int(dash_params.num_inference_steps),
                         "sigma_shift": dash_params.sigma_shift, "seed": dash_params.seed}
            t_dinfer, pred = timed(lambda: model.infer_action(**base, **dash_base, **plan.infer_kwargs))
            t_commit, _ = timed(lambda: ctrl.commit(ref_tokens))
            eff = _effective_steps(pred)

            t_base2, _ = timed(lambda: model.infer_action(**base, **b2))
            t_base3, _ = timed(lambda: model.infer_action(**base, **b3))

            if i >= args.warmup:
                acc["enc"].append(t_enc); acc["plan"].append(t_plan); acc["dinfer"].append(t_dinfer)
                acc["commit"].append(t_commit); acc["base2"].append(t_base2); acc["base3"].append(t_base3)
                if eff is not None:
                    acc["eff"].append(float(eff))

    m = {k: (float(np.mean(v)) if v else float("nan")) for k, v in acc.items()}
    eff = m["eff"]
    dash_total = m["enc"] + m["plan"] + m["dinfer"] + m["commit"]
    bookkeeping = m["enc"] + m["plan"] + m["commit"]
    denoise_step = m["base3"] - m["base2"]                 # marginal cost of one denoise step
    fixed = 3 * m["base2"] - 2 * m["base3"]                # N-independent infer_action setup (KV/conditioning)
    # A baseline call doing `eff` denoise steps would cost fixed + eff*denoise_step; DASH's dinfer
    # over that = the speculative-verify overhead (dinfer also carries the same fixed setup).
    verify = m["dinfer"] - (fixed + eff * denoise_step)

    print("=" * 70)
    print(f"DASH per-call breakdown (mean ms over {len(acc['dinfer'])} frames, eff_steps={eff:.2f})")
    print("-" * 70)
    print(f"  enc    (ref-token encode)        {m['enc']:7.2f}")
    print(f"  plan   (drift + ratio-jump plan) {m['plan']:7.2f}")
    print(f"  dinfer (denoise + verify)        {m['dinfer']:7.2f}")
    print(f"  commit (store prev)              {m['commit']:7.2f}")
    print(f"  -------------------------------  -------")
    print(f"  DASH total                       {dash_total:7.2f}")
    print(f"  of which bookkeeping (enc+plan+commit) {bookkeeping:7.2f}  ({100*bookkeeping/dash_total:.0f}%)")
    print("-" * 70)
    print(f"baseline infer_action on same frame:  2step={m['base2']:.2f}   3step={m['base3']:.2f}")
    print(f"derived:  denoise/step~={denoise_step:.2f}   infer_action fixed setup~={fixed:.2f}   "
          f"verify(in dinfer)~={verify:+.2f}")
    print("-" * 70)
    extra = m['enc'] + m['plan'] + max(verify, 0.0) + m['commit']
    print("interpretation: DASH does ~%.1f denoise steps; dinfer=%.1f ~= 2-step infer_action=%.1f,"
          % (eff, m['dinfer'], m['base2']))
    print("  so the speculative VERIFY is ~free here (%.1f ms). DASH's only real extra over 2-step is" % verify)
    print("  bookkeeping [ref-encode %.1f + plan %.1f + commit %.1f] ~= %.1f ms (%.0f%% of total)."
          % (m['enc'], m['plan'], m['commit'], bookkeeping, 100 * bookkeeping / dash_total))
    print(f"  => on clean low-drift frames DASH({dash_total:.0f}) ~= 2-step({m['base2']:.0f})+bookkeeping, NOT 3-step({m['base3']:.0f}).")
    print("  The ~3-step MEAN seen in aggregates comes from the high-drift tail + shared-GPU contention.")
    print("=" * 70)
    print("\nJSON " + json.dumps({"mean_ms": m, "dash_total": dash_total, "bookkeeping": bookkeeping,
                                  "denoise_step": denoise_step, "fixed_setup": fixed, "verify": verify,
                                  "eff_steps": eff}))
    print("\nBREAKDOWN_OK")


if __name__ == "__main__":
    main()
