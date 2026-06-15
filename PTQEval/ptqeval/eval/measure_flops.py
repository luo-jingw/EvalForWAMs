# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Measure true FLOPs per LingBot-VA inference call using
torch.utils.flop_counter.FlopCounterMode (PyTorch dispatcher-level
counter; doesn't need profiler, ~zero overhead beyond model load).

Replaces the architecture estimate in `_ARCH` for roofline +
op_breakdown. bf16 path only -- quantization doesn't change algorithm
FLOPs (only the dtype the matmul runs in), so a single bf16 measurement
applies to all variants.

Usage (one shot, ~3 min):
    python -m ptqeval.eval.measure_flops \\
        --videos_root /home/arash/EvalForWAMs/results/bf16 \\
        --task adjust_bottle --n_calls 3 \\
        --output /home/arash/EvalForWAMs/results/measured_flops.json

Output JSON schema:
    {
      "_meta": {"source": "torch.utils.flop_counter.FlopCounterMode",
                "n_calls": N, "task": ..., "episode": ...,
                "model_path": ...},
      "flops_per_call_tf": <float>,          # total mean
      "flops_per_call_tf_per_call": [...],   # per-call sample
      "by_op": {"aten::mm": ..., ...}        # top ops by FLOPs
    }
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional


logger = logging.getLogger("measure_flops")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")


# Mirrors derive_calib_ptq._build_server / profile_ops._build_server but
# bf16-only since FLOPs don't depend on weight dtype.
def _build_server(model_path: str, save_root: Path):
    # init_distributed reads MASTER_ADDR/PORT from env; set them up-front
    # since we are running directly (not via torch.distributed.run launcher).
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29680")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

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


def _pick_episode(videos_root: Path, task_substr: Optional[str]) -> Path:
    vis_root = videos_root / "visualization" / "real"
    if not vis_root.exists():
        raise FileNotFoundError(f"no visualization/real/ under {videos_root}")
    eps = sorted(p for p in vis_root.iterdir() if p.is_dir())
    if task_substr:
        eps = [p for p in eps if task_substr.lower() in p.name.lower()]
        if not eps:
            raise FileNotFoundError(
                f"no episode under {vis_root} matches --task {task_substr!r}")
    return eps[0]


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--videos_root", type=Path, required=True,
                   help="Root with visualization/real/<prompt>/ obs chunks; "
                        "usually results/bf16/.")
    p.add_argument("--task", default=None,
                   help="Episode prompt substring filter (e.g. 'adjust_bottle').")
    p.add_argument("--n_calls", type=int, default=3,
                   help="Number of post-warmup infer() calls to measure.")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--model_path",
                   default="/home/arash/EvalForWAMs/models/lingbot-va-posttrain-robotwin")
    p.add_argument("--save_root", type=Path,
                   default=Path("/tmp/measure_flops_scratch"))
    p.add_argument("--output", required=True,
                   help="Output JSON path.")
    args = p.parse_args()

    import torch
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError as e:
        logger.error(f"FlopCounterMode requires torch>=2.1: {e}")
        return 1

    server = _build_server(args.model_path, Path(args.save_root))
    ep_dir = _pick_episode(Path(args.videos_root), args.task)
    logger.info(f"using episode {ep_dir.name}")
    chunks = sorted(
        (int(_CHUNK_RE.search(p.name).group(1)), p)
        for p in ep_dir.glob("obs_data_*.pt")
    )
    if not chunks:
        raise RuntimeError(f"no obs_data_*.pt under {ep_dir}")
    obs_list = torch.load(chunks[0][1], weights_only=False, map_location="cpu")
    prompt = obs_list[0]["task"]
    obs_dict = {"obs": [obs_list[0]], "prompt": prompt,
                "save_visualization": False}

    # Each measurement is independent: reset + 1-frame _infer (matches
    # derive_calib_ptq._replay_episode pattern). Repeating infer() with
    # the same single-frame obs without resetting between calls causes
    # the streaming VAE to mismatch its conv3d kernel against the
    # 1-frame input on subsequent calls.
    def _reset_and_warm():
        server.infer({"reset": True, "prompt": prompt,
                      "save_visualization": False})

    _reset_and_warm()
    for _ in range(args.warmup):
        server.infer(obs_dict)
        _reset_and_warm()
    torch.cuda.synchronize()
    logger.info(f"warmup done; counting FLOPs over {args.n_calls} calls")

    per_call_flops: list[int] = []
    by_op_total: dict = {}
    for i in range(args.n_calls):
        with FlopCounterMode(display=False) as fcm:
            server.infer(obs_dict)
        torch.cuda.synchronize()
        _reset_and_warm()  # restore clean state for next sample
        total = int(fcm.get_total_flops())
        per_call_flops.append(total)
        # by-op aggregation (FlopCounterMode tracks per-aten-op).
        for mod_or_op, ops_dict in fcm.flop_counts.items():
            for op, flops in ops_dict.items():
                key = str(op)
                by_op_total[key] = by_op_total.get(key, 0) + int(flops)
        logger.info(f"call {i + 1}/{args.n_calls}: {total / 1e12:.2f} TF")

    mean_tf = sum(per_call_flops) / len(per_call_flops) / 1e12
    by_op_per_call = {k: v / args.n_calls / 1e12 for k, v in by_op_total.items()}
    by_op_sorted = dict(sorted(by_op_per_call.items(),
                               key=lambda kv: -kv[1]))

    payload = {
        "_meta": {
            "source": "torch.utils.flop_counter.FlopCounterMode",
            "torch_version": torch.__version__,
            "n_calls": args.n_calls,
            "warmup": args.warmup,
            "task": args.task,
            "episode": ep_dir.name,
            "model_path": args.model_path,
            "note": ("FLOPs counted at PyTorch dispatcher level (aten ops). "
                     "Custom CUDA kernels outside aten dispatch are not "
                     "counted; bf16 path uses cuBLAS aten::mm/addmm/bmm "
                     "which ARE counted, so bf16 baseline is accurate. "
                     "Use this to override the architecture-estimated "
                     "flops_per_call in calc_cross_ckpt's _ARCH."),
        },
        "flops_per_call_tf": mean_tf,
        "flops_per_call_tf_per_call": [f / 1e12 for f in per_call_flops],
        "by_op_tf_per_call": by_op_sorted,
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"wrote {args.output}: {mean_tf:.2f} TF/call mean "
                f"({len(by_op_sorted)} distinct ops)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
