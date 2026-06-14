# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Op-level kernel profiler for LingBot-VA inference.

Drives one in-process VA_Server through N inference calls under
torch.profiler, then aggregates per-kernel CUDA time into three
categories (linear / attention / other) and writes a JSON consumable
by `ptqeval.eval.calc_cross_ckpt --op_profile`.

Lightweight by design: profiles a single episode's first chunk replayed
N times (default 5), so total wall clock is ~3-5 min per variant.
Profiler overhead is real (~5-10x slowdown) so don't infer wall-clock
ratios from this; only use it for op-type SHARE within a variant.

Usage (one invocation per variant, then merge):
    python -m ptqeval.eval.profile_ops \\
        --tag bf16 \\
        --videos_root /home/arash/EvalForWAMs/results/bf16 \\
        --output /tmp/op_profile_bf16.json

    python -m ptqeval.eval.profile_ops \\
        --tag viditq_w8a8_dynamic --variant viditq \\
        --variant_args .../runtime_args_w8a8_viditq.yaml \\
        --videos_root /home/arash/EvalForWAMs/results/bf16 \\
        --output /tmp/op_profile_dynamic.json

    # Merge:
    python -c "
    import json
    merged = {'_meta': {'unit': 'ms', 'source': 'torch.profiler'}}
    for p in ['/tmp/op_profile_bf16.json',
              '/tmp/op_profile_dynamic.json',
              '/tmp/op_profile_static.json']:
        merged.update({k: v for k, v in json.load(open(p)).items()
                       if not k.startswith('_')})
    json.dump(merged, open('/tmp/op_profile.json', 'w'), indent=2)
    "

    # Then re-render cross_summary with the measured chart:
    python -m ptqeval.eval.calc_cross_ckpt \\
        --variant bf16=... --variant viditq_w8a8_dynamic=... --variant viditq_w8a8_static=... \\
        --op_profile /tmp/op_profile.json --out_dir results/cross_summary
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


logger = logging.getLogger("profile_ops")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")


# Substring rules for classifying CUDA kernel names. Order matters --
# first match wins. Tuned for our stack (cuBLAS bf16 + our W8A8 GEMM
# + PyTorch SDPA + ViDiT-Q kernels). Inspect prof.key_averages() output
# and extend rules if anything large falls into "other".
_LINEAR_TOKENS = (
    "gemm",          # cuBLAS gemm + cuBLASLt + cutlass + our w8a8 gemm
    "cublas",        # any cuBLAS dispatch
    "cutlass",
    "w8a8",          # our quantized GEMM kernels
    "qlinear",
    "linear",        # PyTorch nn.Linear lowered name (rare)
    "addmm", "mm_kernel", "mm_out",
    "act_quant",     # activation-quant pre-GEMM (counts as linear path)
)
_ATTENTION_TOKENS = (
    "sdpa",
    "scaled_dot_product",
    "flash",         # flash-attn variants
    "fmha",          # fused multi-head attention
    "mha",
    "attention_kernel",
    "softmax_backward",  # only relevant if we ever profile train
    "softmax",       # most softmax kernels in our stack are attention softmax
)


def _classify(kernel_name: str) -> str:
    name = kernel_name.lower()
    for tok in _LINEAR_TOKENS:
        if tok in name:
            return "linear"
    for tok in _ATTENTION_TOKENS:
        if tok in name:
            return "attention"
    return "other"


# ---------------------------------------------------------------------------
# Server construction (mirrors derive_calib_ptq._build_server but accepts
# --variant + --variant_args so we can profile bf16 or any quant path).
# ---------------------------------------------------------------------------

def _build_server(model_path: str, save_root: Path, variant: Optional[str],
                  variant_args_path: Optional[str]):
    """Build an in-process VA_Server (bf16 or any --variant)."""
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
    cfg.perf_log_dir = None     # disable PerfProbe so it doesn't intermix
    cfg.perf_task_name = None
    cfg.wan22_pretrained_model_name_or_path = model_path
    cfg.rank = 0
    cfg.local_rank = 0
    cfg.world_size = 1
    if variant:
        cfg.variant = variant
    if variant_args_path:
        cfg.variant_args = variant_args_path
    save_root.mkdir(parents=True, exist_ok=True)
    return VA_Server(cfg)


_CHUNK_RE = re.compile(r"obs_data_(\d+)\.pt$")


def _pick_first_episode(videos_root: Path, task_substr: Optional[str]) -> Path:
    """Return the first episode directory under videos_root whose
    prompt path string contains task_substr (or any episode when None)."""
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


# ---------------------------------------------------------------------------
# Profiling loop
# ---------------------------------------------------------------------------

def profile_variant(args) -> dict:
    import torch
    from torch.profiler import profile, ProfilerActivity, record_function

    server = _build_server(args.model_path, Path(args.save_root),
                           args.variant, args.variant_args)

    ep_dir = _pick_first_episode(Path(args.videos_root), args.task)
    logger.info(f"profiling episode {ep_dir.name}")
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

    # Warmup (no profiler) -- kicks off compile / autotune.
    server.infer({"reset": True, "prompt": prompt, "save_visualization": False})
    for _ in range(args.warmup):
        server.infer(obs_dict)
    torch.cuda.synchronize()

    logger.info(f"profiling {args.n_calls} infer() calls")
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
                 record_shapes=False) as prof:
        for _ in range(args.n_calls):
            server.infer(obs_dict)
    torch.cuda.synchronize()

    # Aggregate per kernel name across all events.
    by_cat_us: dict[str, float] = {"linear": 0.0, "attention": 0.0,
                                    "other": 0.0}
    # Use key_averages() to collapse repeated launches of the same kernel.
    # Each KernelStat entry exposes self_cuda_time_total (CPU events have
    # cuda_time_total == 0 so they fall into "other" with ~0 weight).
    rows = prof.key_averages()
    per_kernel_dump = []
    for row in rows:
        name = row.key
        cuda_us = float(row.self_cuda_time_total)
        if cuda_us <= 0:
            continue
        cat = _classify(name)
        by_cat_us[cat] += cuda_us
        per_kernel_dump.append({"name": name, "cat": cat,
                                "self_cuda_us": cuda_us,
                                "count": int(row.count)})

    # Convert microseconds total -> ms per inference call.
    by_cat_ms_per_call = {
        cat: us / 1000.0 / args.n_calls
        for cat, us in by_cat_us.items()
    }
    total_ms = sum(by_cat_ms_per_call.values())
    logger.info(f"per-call: linear={by_cat_ms_per_call['linear']:.1f} ms, "
                f"attention={by_cat_ms_per_call['attention']:.1f} ms, "
                f"other={by_cat_ms_per_call['other']:.1f} ms, "
                f"total={total_ms:.1f} ms (profiler-instrumented, "
                f"not wall-clock comparable)")

    return {
        "_meta": {
            "unit": "ms",
            "source": "torch.profiler",
            "variant": args.variant or "bf16",
            "variant_args": args.variant_args,
            "episode": ep_dir.name,
            "n_calls": args.n_calls,
            "warmup": args.warmup,
            "note": ("Profiler overhead inflates total ms; use op-share "
                     "(linear vs attention) not absolute speed."),
        },
        args.tag: {
            "linear": by_cat_ms_per_call["linear"],
            "attention": by_cat_ms_per_call["attention"],
            "other": by_cat_ms_per_call["other"],
        },
        "_per_kernel": sorted(per_kernel_dump,
                              key=lambda r: -r["self_cuda_us"])[:50],
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--tag", required=True,
                   help="Variant tag used as the top-level JSON key; "
                        "must match the --variant tag passed to "
                        "calc_cross_ckpt later (e.g. bf16, "
                        "viditq_w8a8_dynamic, viditq_w8a8_static).")
    p.add_argument("--variant", default=None,
                   help="VA_Server --variant (e.g. 'viditq'); omit "
                        "for bf16 baseline.")
    p.add_argument("--variant_args", default=None,
                   help="Path to runtime_args YAML for the quant "
                        "variant; omit for bf16.")
    p.add_argument("--videos_root", type=Path, required=True,
                   help="Root containing visualization/real/<prompt>/ "
                        "obs_data chunks; usually results/bf16/.")
    p.add_argument("--task", default=None,
                   help="Substring filter for the episode prompt path "
                        "(e.g. 'adjust_bottle'); default picks any.")
    p.add_argument("--n_calls", type=int, default=5,
                   help="Number of infer() calls under the profiler.")
    p.add_argument("--warmup", type=int, default=2,
                   help="Warmup calls before profiling (no instrument).")
    p.add_argument("--model_path",
                   default="/home/arash/EvalForWAMs/models/lingbot-va-posttrain-robotwin",
                   help="bf16 backbone path.")
    p.add_argument("--save_root", type=Path,
                   default=Path("/tmp/profile_ops_scratch"),
                   help="Scratch directory for server output (deleted "
                        "fine to leave).")
    p.add_argument("--output", required=True,
                   help="Output JSON path.")
    args = p.parse_args()

    result = profile_variant(args)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
