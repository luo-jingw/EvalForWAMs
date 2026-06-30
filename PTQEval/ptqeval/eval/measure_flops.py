# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Measure true FLOPs and DRAM bytes per LingBot-VA inference call.

FLOPs: torch.utils.flop_counter.FlopCounterMode (PyTorch dispatcher-
level counter; includes aten::addmm/bmm/conv AND SDPA via the
registered aten._scaled_dot_product_*_attention formulas, which apply
the standard 4*B*H*S_q*S_kv*D MatMul-only count).

Bytes: per-call activation traffic estimated via two hooks installed
on the model after build:
  - WanAttention.attn_op wrapper records (B, S_q, S_kv, H, D) per call
    -> KV-cache + attention I/O bytes
  - WanTransformerBlock.forward wrapper records (B, S, d) per call
    -> block-input-read + block-output-write activation bytes
Weight bytes are computed separately from N_PARAMS * dtype_bytes in
the downstream roofline (per variant: bf16 vs w8a8).

The bf16 path is used for measurement; the FLOPs and the KV /
activation byte counts apply unchanged to W8A8 variants since
quantization touches only weights + linear activations (which are
re-quantized per layer; KV cache and inter-block hidden_states stay
bf16 per ViDiT-Q convention).

Standard convention (matches FlashAttention paper, Megatron, PaLM,
PyTorch FlopCounterMode, MLPerf LLM inference roofline):
  - 1 MAC = 2 FLOPs
  - Attention: 4*B*H*S_q*S_kv*D per call (no causal halving since model
    uses causal=False default)
  - Softmax / scale / mask FLOPs excluded (~2% vs matmul)
  - No L2-cache reuse discount on bytes (conservative upper bound)

Rollout mode (Phase 39c): multi-chunk replay with KV cache
accumulating across chunks (matches production inference loop, NOT
the 1-frame+reset calibration pattern). Reset is called once at the
start, then warmup chunks are consumed sequentially to fill the KV
cache to attn_window/2 capacity. Measurement chunks (n_calls) are
each wrapped in a separate FlopCounterMode, so per-call FLOP / byte
counts reflect steady-state cache usage. This gives realistic
attention proportion (which 1-frame+reset under-estimated since
KV cache stayed empty).

Usage (one shot, ~5-10 min including 36-chunk warmup):
    python -m ptqeval.eval.measure_flops \\
        --videos_root results/calib_capture \\
        --warmup 36 --n_calls 3 \\
        --output results/measured_flops.json
  (Phase 45.9-A removed the variant obs_data raw frames; calib_capture
   keeps its VAE-primed obs chunks, the same format, so it is now the
   obs source for measure_flops.)

Output JSON schema:
    {
      "_meta": {...},
      "flops_per_call_tf": <float>,                   # total mean (FlopCounterMode 'Global')
      "flops_per_call_tf_per_call": [...],            # per-call sample
      "by_op_tf_per_call": {"aten::addmm": ..., ...}, # Global-only, sums to flops_per_call_tf
      "bytes_per_call_b": {                           # measured via hooks (bf16 path)
        "kv_attention_bytes_b": <int>,                # K/V load + O write + cache write
        "block_activation_bytes_b": <int>,            # per-block input R + output W
        "n_attn_calls_per_infer": <int>,
        "n_block_calls_per_infer": <int>
      }
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


def _install_byte_hooks(transformer):
    """Install in-place hooks on every WanAttention.attn_op and every
    WanTransformerBlock.forward to record per-call shapes.

    Returns (attn_records, block_records, teardown_fn) where:
      - attn_records: list of dicts {B, S_q, S_kv, H, D, dtype_bytes,
        is_self_attn} appended on every attn_op invocation
      - block_records: list of dicts {B, S, d, dtype_bytes} appended on
        every block forward
      - teardown_fn: restores original methods (idempotent)

    Class matching is by __class__.__name__ so we don't depend on the
    wan_va import path; both hooks are guarded against double-install
    via a sentinel attribute.
    """
    attn_records: list[dict] = []
    block_records: list[dict] = []
    teardown: list = []

    # Match by class-name suffix so FSDP-wrapped subclasses
    # ('FSDPWanAttention', 'FSDPWanTransformerBlock') also get hooked.
    for _, m in transformer.named_modules():
        cls = m.__class__.__name__
        if cls.endswith("WanAttention") and not getattr(
                m, "_byte_hook_installed", False):
            orig_attn_op = m.attn_op
            # WanAttention.attn_caches is None for cross-attention (which
            # has cross_attention_dim_head set in __init__); self-attn
            # has a dict. Capture at install time since this can't change.
            is_self_attn = m.attn_caches is not None

            def _make_wrap(orig, is_self):
                def _wrapped(q, k, v, *args, **kwargs):
                    # flash_attn_func and custom_sdpa both receive
                    # tensors in [B, S, H, D] convention (see model.py).
                    attn_records.append({
                        "B": q.shape[0], "S_q": q.shape[1],
                        "S_kv": k.shape[1], "H": q.shape[2],
                        "D": q.shape[3],
                        "dtype_bytes": q.element_size(),
                        "is_self_attn": is_self,
                    })
                    return orig(q, k, v, *args, **kwargs)
                return _wrapped

            m.attn_op = _make_wrap(orig_attn_op, is_self_attn)
            m._byte_hook_installed = True
            teardown.append(lambda mm=m, op=orig_attn_op: (
                setattr(mm, "attn_op", op),
                delattr(mm, "_byte_hook_installed"),
            ))

        elif cls.endswith("WanTransformerBlock") and not getattr(
                m, "_byte_hook_installed", False):
            orig_forward = m.forward

            def _make_wrap(orig):
                def _wrapped(hidden_states, *args, **kwargs):
                    block_records.append({
                        "B": hidden_states.shape[0],
                        "S": hidden_states.shape[1],
                        "d": hidden_states.shape[-1],
                        "dtype_bytes": hidden_states.element_size(),
                    })
                    return orig(hidden_states, *args, **kwargs)
                return _wrapped

            m.forward = _make_wrap(orig_forward)
            m._byte_hook_installed = True
            teardown.append(lambda mm=m, fn=orig_forward: (
                setattr(mm, "forward", fn),
                delattr(mm, "_byte_hook_installed"),
            ))

    def _teardown():
        for fn in teardown:
            fn()
    return attn_records, block_records, _teardown


def _attn_bytes_per_call(attn_records: list[dict]) -> int:
    """KV-cache + attention I/O bytes for one infer() call.

    For each attention invocation we count (matches FlashAttention paper
    Appendix B DRAM cost + standard MLPerf KV-cache traffic model):
      - K load:      B * H * S_kv * D * dt
      - V load:      B * H * S_kv * D * dt
      - O write:     B * H * S_q  * D * dt
      - cache write: B * H * S_q  * D * dt * 2  (only for self-attn,
                     which appends new K and V to its cache pool)
    Q is assumed to stay in SRAM (FlashAttention tiles Q in registers);
    this matches standard roofline accounting for fused attention.
    """
    total = 0
    for r in attn_records:
        dt = r["dtype_bytes"]
        k_load = r["B"] * r["H"] * r["S_kv"] * r["D"] * dt
        v_load = k_load
        o_write = r["B"] * r["H"] * r["S_q"] * r["D"] * dt
        bytes_ = k_load + v_load + o_write
        if r["is_self_attn"]:
            bytes_ += 2 * r["B"] * r["H"] * r["S_q"] * r["D"] * dt
        total += bytes_
    return total


def _block_bytes_per_call(block_records: list[dict]) -> int:
    """Inter-block activation traffic for one infer() call.

    For each WanTransformerBlock forward we count one read of its input
    hidden_states + one write of its output (= same shape). Intra-block
    activations (attn1->norm2, ffn intermediates) are assumed to stay in
    L2 within the block -- standard simplification; intra-block reuse
    is the reason we don't sum all aten op tensor I/O.
    """
    total = 0
    for r in block_records:
        total += 2 * r["B"] * r["S"] * r["d"] * r["dtype_bytes"]
    return total


def _pick_episode(videos_root: Path, task_substr: Optional[str]) -> Path:
    """Pick the episode with the most obs_data chunks (subject to --task
    filter), since multi-chunk rollout measurement needs enough chunks
    to fill the KV cache warmup window (~36+) and still leave several
    for measurement."""
    vis_root = videos_root / "visualization" / "real"
    if not vis_root.exists():
        raise FileNotFoundError(f"no visualization/real/ under {videos_root}")
    eps = [p for p in vis_root.iterdir() if p.is_dir()]
    if task_substr:
        eps = [p for p in eps if task_substr.lower() in p.name.lower()]
        if not eps:
            raise FileNotFoundError(
                f"no episode under {vis_root} matches --task {task_substr!r}")
    eps_with_counts = [
        (sum(1 for _ in p.glob("obs_data_*.pt")), p) for p in eps
    ]
    eps_with_counts.sort(key=lambda x: (-x[0], x[1].name))
    return eps_with_counts[0][1]


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--videos_root", type=Path, required=True,
                   help="Root with visualization/real/<prompt>/ obs chunks; "
                        "results/calib_capture/ (variant obs removed in "
                        "Phase 45.9-A).")
    p.add_argument("--task", default=None,
                   help="Episode prompt substring filter (e.g. 'adjust_bottle').")
    p.add_argument("--n_calls", type=int, default=3,
                   help="Number of post-warmup chunks to wrap in "
                        "FlopCounterMode (each = one server.infer call).")
    p.add_argument("--warmup", type=int, default=36,
                   help="Number of warmup chunks consumed sequentially "
                        "WITHOUT reset before measurement starts. Default "
                        "36 = attn_window/2 for robotwin (attn_window=72, "
                        "frame_chunk_size=2), enough to fill the KV cache "
                        "to steady-state capacity.")
    p.add_argument("--model_path",
                   default="models/lingbot-va-posttrain-robotwin")
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
    attn_records, block_records, _teardown_hooks = _install_byte_hooks(
        server.transformer)
    logger.info("installed byte hooks on WanAttention.attn_op and "
                "WanTransformerBlock.forward")
    ep_dir = _pick_episode(Path(args.videos_root), args.task)
    logger.info(f"using episode {ep_dir.name}")
    chunks = sorted(
        (int(_CHUNK_RE.search(p.name).group(1)), p)
        for p in ep_dir.glob("obs_data_*.pt")
    )
    if not chunks:
        raise RuntimeError(f"no obs_data_*.pt under {ep_dir}")
    # Episode chunk count is informational only -- we drive the model
    # via frame_st_id auto-increment using the same init_obs (see
    # below); chunks file count == max frame_st_id divided by
    # frame_chunk_size, which bounds how many warmup+measure calls the
    # episode actually supports before frame_st_id exceeds the recorded
    # rollout length. Use the longest-episode picked above.
    first = torch.load(chunks[0][1], weights_only=False, map_location="cpu")
    prompt = first[0]["task"]
    frame_chunk_size = server.job_config.frame_chunk_size
    # Mirror production generate() at server.py:900 exactly:
    # reset once, then call server._infer DIRECTLY (bypassing the
    # infer() wrapper) with an explicit frame_st_id = chunk_id *
    # frame_chunk_size. The wrapper does NOT auto-increment
    # self.frame_st_id (only _compute_kv_cache does), so calling
    # infer() repeatedly would always pass frame_st_id=0 and re-run
    # the VAE encoder, which crashes on chunk 2+ because the encoder's
    # streaming feat_cache only holds 1 frame from chunk 0 -- not
    # enough for the time_conv kernel size 3.
    init_obs = {"obs": [first[0]], "prompt": prompt,
                "save_visualization": False}
    n_total = len(chunks)
    logger.info(
        f"episode has {n_total} chunks; chunk_size={frame_chunk_size}; "
        f"using same init_obs + explicit frame_st_id across all calls "
        f"(production generate() flow). warmup={args.warmup} chunks "
        f"to fill KV cache, then measuring next {args.n_calls} chunks.")

    server.infer({"reset": True, "prompt": prompt,
                  "save_visualization": False})
    # _reset clears self.frame_st_id; production uses chunk_id * size
    # explicitly so we don't depend on the server attribute.
    for chunk_id in range(args.warmup):
        server._infer(init_obs, frame_st_id=chunk_id * frame_chunk_size)
    torch.cuda.synchronize()
    logger.info(f"warmup done; measuring next {args.n_calls} chunks")

    per_call_flops: list[int] = []
    per_call_kv_bytes: list[int] = []
    per_call_block_bytes: list[int] = []
    per_call_n_attn: list[int] = []
    per_call_n_block: list[int] = []
    by_op_total: dict = {}
    for i in range(args.n_calls):
        chunk_id = args.warmup + i
        # Hooks accumulate across the model's forward; clear before each
        # measurement so per-call counts are independent.
        attn_records.clear()
        block_records.clear()
        with FlopCounterMode(display=False) as fcm:
            server._infer(init_obs, frame_st_id=chunk_id * frame_chunk_size)
        torch.cuda.synchronize()
        total = int(fcm.get_total_flops())
        per_call_flops.append(total)
        kv_b = _attn_bytes_per_call(attn_records)
        blk_b = _block_bytes_per_call(block_records)
        per_call_kv_bytes.append(kv_b)
        per_call_block_bytes.append(blk_b)
        per_call_n_attn.append(len(attn_records))
        per_call_n_block.append(len(block_records))
        # by-op aggregation: use 'Global' bucket only. fcm.flop_counts
        # increments once per (module-ancestor, op) pair, so iterating
        # all keys would double-count by hierarchy depth; the 'Global'
        # bucket sums to fcm.get_total_flops() exactly.
        for op, flops in fcm.flop_counts.get("Global", {}).items():
            key = str(op)
            by_op_total[key] = by_op_total.get(key, 0) + int(flops)
        logger.info(
            f"call {i + 1}/{args.n_calls}: {total / 1e12:.2f} TF, "
            f"KV/attn-IO {kv_b / 1e9:.2f} GB, "
            f"block-act {blk_b / 1e9:.2f} GB "
            f"({len(attn_records)} attn calls, "
            f"{len(block_records)} block calls)"
        )

    mean_tf = sum(per_call_flops) / len(per_call_flops) / 1e12
    by_op_per_call = {k: v / args.n_calls / 1e12 for k, v in by_op_total.items()}
    by_op_sorted = dict(sorted(by_op_per_call.items(),
                               key=lambda kv: -kv[1]))
    mean_kv_b = sum(per_call_kv_bytes) / len(per_call_kv_bytes)
    mean_blk_b = sum(per_call_block_bytes) / len(per_call_block_bytes)
    mean_n_attn = sum(per_call_n_attn) / len(per_call_n_attn)
    mean_n_block = sum(per_call_n_block) / len(per_call_n_block)

    payload = {
        "_meta": {
            "source": "torch.utils.flop_counter.FlopCounterMode "
                      "+ WanAttention/WanTransformerBlock byte hooks",
            "torch_version": torch.__version__,
            "n_calls": args.n_calls,
            "warmup_chunks": args.warmup,
            "rollout_mode": "multi_chunk_no_reset_between",
            "task": args.task,
            "episode": ep_dir.name,
            "model_path": args.model_path,
            "flops_note": (
                "FLOPs counted at PyTorch dispatcher level (aten ops). "
                "Includes aten._scaled_dot_product_*_attention via the "
                "registered 4*B*H*S_q*S_kv*D formula. Custom CUDA "
                "kernels outside aten dispatch (e.g. W8A8 GEMM) are not "
                "counted, but FLOPs are dtype-invariant so the bf16 "
                "measurement applies to all variants."),
            "bytes_note": (
                "kv_attention_bytes_b counts K+V loads, output writes, "
                "and self-attention cache writes (Q stays in SRAM). "
                "block_activation_bytes_b counts per-block input read + "
                "output write only; intra-block activations assumed to "
                "stay in L2 (standard simplification, matches MLPerf / "
                "vLLM roofline). No L2 reuse discount applied -- "
                "conservative upper bound on bytes. Activations stay "
                "bf16 in W8A8 path (per ViDiT-Q), so these byte counts "
                "are dtype-invariant; weight bytes are computed "
                "per-variant downstream."),
        },
        "flops_per_call_tf": mean_tf,
        "flops_per_call_tf_per_call": [f / 1e12 for f in per_call_flops],
        "by_op_tf_per_call": by_op_sorted,
        "bytes_per_call_b": {
            "kv_attention_bytes_b": int(mean_kv_b),
            "block_activation_bytes_b": int(mean_blk_b),
            "n_attn_calls_per_infer": float(mean_n_attn),
            "n_block_calls_per_infer": float(mean_n_block),
        },
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(
        f"wrote {args.output}: {mean_tf:.2f} TF/call, "
        f"KV/attn {mean_kv_b / 1e9:.2f} GB/call, "
        f"block-act {mean_blk_b / 1e9:.2f} GB/call "
        f"({len(by_op_sorted)} distinct ops, "
        f"{mean_n_attn:.0f} attn calls, {mean_n_block:.0f} block calls)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
