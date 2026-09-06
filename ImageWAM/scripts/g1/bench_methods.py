#!/usr/bin/env python3
"""Open-loop latency + effective-step benchmark over REAL dataset frames.

Loads the G1 model once (one resident copy of the weights) and replays consecutive
real frames from one or more LeRobot episodes through PolicyEngine.infer, once per
method:

    baseline 10-step / 3-step / 2-step  (num_inference_steps = 10 / 3 / 2)
    ProbeFlow                           (training-free adaptive FM solver)
    DASH                                (drift-adaptive speculative ratio-jump)

Every method sees the SAME frames in the SAME order (fair comparison). Latency is
measured with an explicit torch.cuda.synchronize() on both sides of each call --
serve's own InferTiming.infer_s is NOT cuda-synced and under-reports. For ProbeFlow
and DASH we also report the effective FM-step count (mean / variance) -- the backend
`parallel_calls` (DASH) / `forward_calls` (ProbeFlow), i.e. how many denoise-network
forward passes each replan actually spent. Baseline steps are fixed (= the config),
so they are not aggregated.

Open-loop = we feed the dataset's OWN observations at each timestep; we do NOT roll
the predicted actions back into the sim. DASH drift is meaningful because the frames
are consecutive; a fresh DashSession per episode accumulates drift as it would online.

Multi-episode: pass --num-episodes N (+ --seed) to randomly sample N episodes, or
--episodes 3,7,12 for an explicit set. Latency and effective-step stats are aggregated
across ALL timed calls of ALL episodes into one final table.

No robot, no WebSocket. Self-terminating. One GPU.

    ssh nnmc62   # or nnmc75
    S=/shared/user75/workspace/jingwu/ImagewamFT     # on nnmc75 use the local /home path
    cd $S/ImageWAM
    D=runs/g1_stack_cubes_flux2_klein_4b/2026-07-23_02-40-24
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$S/ImageWAM/src $S/ImageWAM/.venv/bin/python \
      scripts/g1/bench_methods.py \
        --ckpt          $D/checkpoints/weights/step_009850.pt \
        --dataset-stats $D/dataset_stats.json \
        --prompt-embeds ../checkpoints/imagewam_g1/g1_stack_cubes_prompt_embeds.npz \
        --task-config   g1_stack_cubes_flux2_klein_4b \
        --dataset-root  ../dataset/stack_the_cubes \
        --flux2-model-path $S/checkpoints/flux2/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors \
        --ae-model-path    $S/checkpoints/flux2/FLUX.2-dev/ae.safetensors \
        --flux2-src-path   $S/ImageWAM/third_party/flux2 \
        --num-episodes 10 --seed 42 --num-calls 60 --warmup 5 --stride 8
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import av
import numpy as np
import torch

from serve_imagewam_g1 import RuntimeConfig, build_engine  # noqa: E402
from inference_methods import BaselineParams  # noqa: E402

# LeRobot camera video keys, ordered to match ImagePreprocessor's expected views:
#   v0 = head/top (cam_left_high), v1 = left wrist, v2 = right wrist.
CAM_KEYS = (
    "observation.images.cam_left_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
DEFAULT_PROMPT = (
    "A video recorded from a robot's point of view executing the following instruction: {task}"
)

# (label, nominal_steps, update_runtime kwargs). Effective steps for probeflow/dash
# come from the backend at run time; baseline steps are fixed at nominal.
METHOD_PLAN = (
    ("10step", 10, dict(method="baseline", baseline={"num_inference_steps": 10, "seed": 0})),
    ("3step", 3, dict(method="baseline", baseline={"num_inference_steps": 3, "seed": 0})),
    ("2step", 2, dict(method="baseline", baseline={"num_inference_steps": 2, "seed": 0})),
    # Same seed as baseline so MSE-vs-10step reflects only the solver, not the init noise.
    ("probeflow", 10, dict(method="probeflow", probeflow={"seed": 0})),
    ("dash", 10, dict(method="dash", dash={"seed": 0})),
)
REF_LABEL = "10step"  # reference method for the vs-reference deviation MSE


def list_episodes(root: Path) -> dict[int, int]:
    """episode_index -> length, from meta/episodes.jsonl."""
    lengths: dict[int, int] = {}
    with open(root / "meta" / "episodes.jsonl") as f:
        for line in f:
            obj = json.loads(line)
            lengths[int(obj["episode_index"])] = int(obj["length"])
    return lengths


def load_states(root: Path, episode: int) -> np.ndarray:
    """[T, 16] float32 raw joint state from the episode parquet."""
    import pandas as pd  # local import: only needed once per episode
    pq = root / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"
    df = pd.read_parquet(pq, columns=["observation.state"])
    return np.stack(df["observation.state"].to_numpy()).astype(np.float32)


def load_actions(root: Path, episode: int) -> np.ndarray:
    """[T, 16] float32 ground-truth action from the episode parquet."""
    import pandas as pd
    pq = root / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"
    df = pd.read_parquet(pq, columns=["action"])
    return np.stack(df["action"].to_numpy()).astype(np.float32)


def load_task(root: Path, episode: int) -> str:
    """The bare task string for this episode (as stored in meta/episodes.jsonl)."""
    with open(root / "meta" / "episodes.jsonl") as f:
        for line in f:
            obj = json.loads(line)
            if int(obj["episode_index"]) == episode:
                return obj["tasks"][0]
    raise KeyError(f"episode {episode} not found in {root}/meta/episodes.jsonl")


def resolve_prompt(candidate: str, bank: tuple[str, ...]) -> str | None:
    """Match `candidate` against the PromptBank keys, trying the bare task and the
    DEFAULT_PROMPT-wrapped form (the export/serve convention varies per npz). Returns
    the matching bank key, or None if neither form is present."""
    for form in (candidate, DEFAULT_PROMPT.format(task=candidate)):
        if form in bank:
            return form
    return None


def decode_frames(video_path: Path, want: list[int]) -> dict[int, np.ndarray]:
    """Decode the requested frame indices as RGB uint8 HxWx3 (sequential PyAV walk)."""
    need = set(want)
    out: dict[int, np.ndarray] = {}
    container = av.open(str(video_path))
    try:
        for i, frame in enumerate(container.decode(video=0)):
            if i in need:
                out[i] = frame.to_ndarray(format="rgb24")
                if len(out) == len(need):
                    break
    finally:
        container.close()
    missing = need - set(out)
    if missing:
        raise RuntimeError(f"{video_path.name}: frames {sorted(missing)} past EOF")
    return out


def build_obs_sequence(root: Path, episode: int, frame_idxs: list[int], prompt: str) -> list[dict]:
    states = load_states(root, episode)
    if frame_idxs[-1] >= len(states):
        raise ValueError(f"frame {frame_idxs[-1]} >= episode length {len(states)}")
    views_per_cam = [
        decode_frames(root / "videos" / "chunk-000" / cam / f"episode_{episode:06d}.mp4", frame_idxs)
        for cam in CAM_KEYS
    ]
    obs_seq = []
    for f in frame_idxs:
        obs_seq.append({
            "image": [views_per_cam[c][f] for c in range(len(CAM_KEYS))],
            "state": states[f].reshape(-1),
            "prompt": prompt,
        })
    return obs_seq


def summarize(latencies_ms: list[float], steps: list[float | None]) -> dict:
    lat = np.asarray(latencies_ms, dtype=np.float64)
    out = {
        "n": int(lat.size),
        "lat_mean": float(lat.mean()),
        "lat_std": float(lat.std(ddof=1)) if lat.size > 1 else 0.0,
        "lat_p50": float(np.percentile(lat, 50)),
        "lat_p90": float(np.percentile(lat, 90)),
        "lat_min": float(lat.min()),
        "lat_max": float(lat.max()),
    }
    eff = np.asarray([s for s in steps if s is not None], dtype=np.float64)
    if eff.size:
        out["step_mean"] = float(eff.mean())
        out["step_var"] = float(eff.var(ddof=1)) if eff.size > 1 else 0.0
        out["step_min"] = float(eff.min())
        out["step_max"] = float(eff.max())
    return out


def choose_episodes(args, lengths: dict[int, int]) -> list[int]:
    if args.episodes:
        eps = [int(x) for x in args.episodes.split(",")]
        missing = [e for e in eps if e not in lengths]
        if missing:
            raise SystemExit(f"episodes not in dataset: {missing}")
        return eps
    if args.num_episodes and args.num_episodes > 1:
        pool = sorted(lengths)
        rng = random.Random(args.seed)
        k = min(args.num_episodes, len(pool))
        return sorted(rng.sample(pool, k))
    return [args.episode]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--dataset-stats", type=Path, required=True)
    p.add_argument("--prompt-embeds", type=Path, required=True)
    p.add_argument("--task-config", default="g1_stack_cubes_flux2_klein_4b")
    p.add_argument("--dataset-root", type=Path, required=True,
                   help="LeRobot dataset root, e.g. ../dataset/stack_the_cubes")
    p.add_argument("--episode", type=int, default=0, help="single-episode mode (if no --num-episodes/--episodes)")
    p.add_argument("--num-episodes", type=int, default=1, help="randomly sample this many episodes")
    p.add_argument("--episodes", default=None, help="explicit comma list of episode indices (overrides --num-episodes)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for random episode sampling")
    p.add_argument("--num-calls", type=int, default=60, help="timed calls per method per episode")
    p.add_argument("--warmup", type=int, default=5, help="discarded calls before timing (per episode per method)")
    p.add_argument("--stride", type=int, default=8,
                   help="frames between consecutive infer calls (mimics exec-horizon replan cadence)")
    p.add_argument("--prompt", default=None, help="override the task string (single-episode use)")
    p.add_argument("--methods", default=None,
                   help="comma list subset of: 10step,3step,2step,probeflow,dash")
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
    use_cuda = torch.device(args.device).type == "cuda"

    def timed(obs: dict, session) -> tuple[float, float | None, np.ndarray]:
        if use_cuda:
            torch.cuda.synchronize()
        t = time.perf_counter()
        out = engine.infer(obs, dash_session=session)
        if use_cuda:
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t) * 1e3
        return dt, engine.last_timing.effective_steps, np.asarray(out["actions"], dtype=np.float64)

    lengths = list_episodes(args.dataset_root)
    episodes = choose_episodes(args, lengths)
    selected = set(args.methods.split(",")) if args.methods else None
    plan = [m for m in METHOD_PLAN if selected is None or m[0] in selected]
    nominal = {label: nom for label, nom, _ in plan}

    print(f"dataset      : {args.dataset_root}")
    print(f"episodes     : {episodes}  (seed {args.seed})")
    print(f"per episode  : warmup {args.warmup} + timed {args.num_calls} @ stride {args.stride}")
    print(f"methods      : {[m[0] for m in plan]}\n")

    # Global per-method accumulators across all episodes.
    agg: dict[str, dict[str, list]] = {
        label: {"lat": [], "steps": [], "mse_gt": [], "mse_ref": []} for label, _, _ in plan
    }
    ref_in_plan = any(lb == REF_LABEL for lb, _, _ in plan)
    used: list[dict] = []
    dash_calls: list[tuple[float, float | None, int | None]] = []  # (lat_ms, eff_steps, branch_count)

    for ep in episodes:
        length = lengths[ep]
        usable = length // args.stride
        if usable <= args.warmup + 1:
            print(f"[ep {ep:3d}] len {length} too short for stride {args.stride}; skipped")
            continue
        n_timed = min(args.num_calls, usable - args.warmup)
        frame_idxs = [i * args.stride for i in range(args.warmup + n_timed)]
        timed_frames = frame_idxs[args.warmup:]

        candidate = args.prompt or load_task(args.dataset_root, ep)
        prompt = resolve_prompt(candidate, engine.prompts)
        if prompt is None:
            print(f"[ep {ep:3d}] task not in prompt bank; skipped ({candidate!r:.60})")
            continue

        obs_seq = build_obs_sequence(args.dataset_root, ep, frame_idxs, prompt)
        gt_actions = load_actions(args.dataset_root, ep)   # [T,16] ground-truth
        per_ep = {"episode": ep, "length": length, "n_timed": n_timed}
        preds: dict[str, list[np.ndarray]] = {}
        line = [f"[ep {ep:3d}] len {length:4d} timed {n_timed:3d}:"]
        for label, _, changes in plan:
            engine.update_runtime(**changes)
            session = engine.new_dash_session() if changes["method"] == "dash" else None
            for obs in obs_seq[:args.warmup]:           # warm caches / fill drift, discard
                engine.infer(obs, dash_session=session)
            lat_ms, steps, plist = [], [], []
            for obs in obs_seq[args.warmup:]:
                dt, eff, act = timed(obs, session)
                lat_ms.append(dt)
                steps.append(eff)
                plist.append(act)                       # [H,16] predicted chunk
                if changes["method"] == "dash":         # per-call diagnostic for tail attribution
                    d = engine.last_timing.dash or {}
                    dash_calls.append((dt, eff, d.get("branch_count")))
            preds[label] = plist
            agg[label]["lat"].extend(lat_ms)
            agg[label]["steps"].extend(steps)
            lat_mean = float(np.mean(lat_ms))
            eff_vals = [s for s in steps if s is not None]
            per_ep[label] = {"lat_mean": lat_mean,
                             "step_mean": (float(np.mean(eff_vals)) if eff_vals else None)}
            tag = f" {label}={lat_mean:6.1f}ms"
            if eff_vals:
                tag += f"({np.mean(eff_vals):.2f}st)"
            line.append(tag)
        print(" ".join(line))

        # ---- per-episode MSE: vs ground-truth action, and vs the reference method ----
        ref_plist = preds.get(REF_LABEL)
        mse_line = [f"           mse_gt:"]
        ref_line = [f"           vs{REF_LABEL}:"]
        for label, _, _ in plan:
            gt_ep, ref_ep = [], []
            for i, n in enumerate(timed_frames):
                pred = preds[label][i]                  # [H,16]
                h = min(pred.shape[0], len(gt_actions) - n)
                if h >= 1:
                    m = float(np.mean((pred[:h] - gt_actions[n:n + h]) ** 2))
                    gt_ep.append(m)
                    agg[label]["mse_gt"].append(m)
                if ref_plist is not None and label != REF_LABEL:
                    d = float(np.mean((pred - ref_plist[i]) ** 2))
                    ref_ep.append(d)
                    agg[label]["mse_ref"].append(d)
            per_ep[label]["mse_gt"] = float(np.mean(gt_ep)) if gt_ep else None
            per_ep[label]["mse_ref"] = float(np.mean(ref_ep)) if ref_ep else None
            mse_line.append(f" {label}={per_ep[label]['mse_gt']:.4f}")
            if label != REF_LABEL and per_ep[label]["mse_ref"] is not None:
                ref_line.append(f" {label}={per_ep[label]['mse_ref']:.5f}")
        print(" ".join(mse_line))
        if ref_in_plan and len(ref_line) > 1:
            print(" ".join(ref_line))
        used.append(per_ep)

    if not used:
        raise SystemExit("no episodes were benchmarked (all skipped)")

    # ---- aggregate table across all episodes --------------------------------
    results = {label: summarize(agg[label]["lat"], agg[label]["steps"]) for label, _, _ in plan}
    for label, _, _ in plan:
        g, r = agg[label]["mse_gt"], agg[label]["mse_ref"]
        results[label]["mse_gt_mean"] = float(np.mean(g)) if g else None
        results[label]["mse_ref_mean"] = float(np.mean(r)) if r else None
    total_calls = results[plan[0][0]]["n"]
    print("\n" + "=" * 100)
    print(f"AGGREGATE over {len(used)} episodes, {total_calls} timed calls/method "
          f"(dataset {args.dataset_root.name})")
    hdr = f"{'method':10s} {'nom':>4s} {'n':>5s} {'lat_mean':>9s} {'p50':>8s} {'p90':>8s} " \
          f"{'eff_mean':>9s} {'mse_gt':>9s} {'mse_vs'+REF_LABEL[:5]:>9s}"
    print(hdr)
    print("-" * 100)
    for label, _, _ in plan:
        s = results[label]
        em = f"{s['step_mean']:9.2f}" if "step_mean" in s else f"{'-':>9s}"
        mg = f"{s['mse_gt_mean']:9.4f}" if s.get("mse_gt_mean") is not None else f"{'-':>9s}"
        mr = f"{s['mse_ref_mean']:9.5f}" if s.get("mse_ref_mean") is not None else f"{'-':>9s}"
        print(f"{label:10s} {nominal[label]:>4d} {s['n']:>5d} {s['lat_mean']:9.1f} "
              f"{s['lat_p50']:8.1f} {s['lat_p90']:8.1f} {em} {mg} {mr}")
    print("=" * 100)
    print("lat_* in ms (cuda-synced). eff_mean = effective FM steps (DASH parallel_calls /\n"
          f"ProbeFlow forward_calls). mse_gt = MSE of predicted action chunk vs dataset "
          f"ground-truth\n(raw radians, all 16 dims x horizon). mse_vs{REF_LABEL} = MSE of "
          f"the prediction vs {REF_LABEL}'s\nprediction at the same frame+seed (isolates the "
          "accelerator's deviation).")

    # ---- DASH tail attribution: bucket per-call latency by effective steps + branch_count ----
    dash_buckets = None
    if dash_calls:
        def _bucket(key_fn):
            out = {}
            for lat, eff, bc in dash_calls:
                k = key_fn(eff, bc)
                out.setdefault(k, []).append(lat)
            return {k: {"n": len(v), "mean": float(np.mean(v)), "p50": float(np.percentile(v, 50)),
                        "p90": float(np.percentile(v, 90)), "max": float(np.max(v)),
                        "share": len(v) / len(dash_calls)} for k, v in sorted(out.items(), key=lambda kv: (kv[0] is None, kv[0]))}
        by_eff = _bucket(lambda eff, bc: (None if eff is None else int(round(eff))))
        by_branch = _bucket(lambda eff, bc: bc)
        dash_buckets = {"by_eff_steps": by_eff, "by_branch_count": by_branch}
        print("\n" + "-" * 100)
        print(f"DASH tail attribution ({len(dash_calls)} calls): per-call latency bucketed")
        print(f"{'eff_steps':>10s} {'n':>5s} {'share':>6s} {'lat_mean':>9s} {'p50':>8s} {'p90':>8s} {'max':>8s}")
        for k, s in by_eff.items():
            print(f"{str(k):>10s} {s['n']:>5d} {s['share']*100:5.0f}% {s['mean']:9.1f} "
                  f"{s['p50']:8.1f} {s['p90']:8.1f} {s['max']:8.1f}")
        print(f"{'branch_ct':>10s} {'n':>5s} {'share':>6s} {'lat_mean':>9s} {'p50':>8s} {'p90':>8s} {'max':>8s}")
        for k, s in by_branch.items():
            print(f"{str(k):>10s} {s['n']:>5d} {s['share']*100:5.0f}% {s['mean']:9.1f} "
                  f"{s['p50']:8.1f} {s['p90']:8.1f} {s['max']:8.1f}")
        print("-" * 100)
        print("If the tail concentrates in one eff/branch bucket -> intrinsic (bigger speculative batch +\n"
              "resume steps). If every bucket has a fat p90/max -> GPU contention, not DASH's own cost.")

    payload = {"dataset": str(args.dataset_root), "episodes": episodes, "seed": args.seed,
               "stride": args.stride, "warmup": args.warmup, "num_calls": args.num_calls,
               "nominal_steps": nominal, "per_episode": used, "aggregate": results,
               "dash_buckets": dash_buckets}
    print("\nJSON " + json.dumps(payload))
    print("\nBENCH_OK")


if __name__ == "__main__":
    main()
