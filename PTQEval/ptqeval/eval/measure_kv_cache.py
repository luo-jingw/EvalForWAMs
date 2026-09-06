# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Measure GPU memory occupied by KV cache + transformer weights + (optional)
VAE + (optional) text encoder, by sampling torch.cuda.memory_allocated() at
checkpoints during model construction.

Strict spatial decomposition without residual / reverse-fit:

    1. baseline                  ~ 0 GB             (after empty_cache)
    2. after transformer load    -> xfmr_weight_GB
    3. after vae load            -> + vae_GB
    4. after init_kv_cache       -> + kv_cache_GB   (cache_name='pos')
    5. after first _compute_kv_cache + _infer       -> + transient_GB
       (activation peak observed during one full _infer call)

Output: a JSON with absolute bytes per stage + per-segment deltas. Consumed
by calc_cross_ckpt for the memory_breakdown chart (replaces the previous
reverse-fit `_KV_CACHE_GB = 8.92` hardcoded constant).

Run on a single free GPU (~30 sec total, peak ~15 GB during alloc):

    python -m ptqeval.eval.measure_kv_cache \\
        --model_path models/lingbot-va-posttrain-robotwin \\
        --output results/measured_kv_cache.json

Note: KV cache is BF16 across all quantization variants (paper W8A8/W4A8
do not touch KV); a single bf16 measurement applies to bf16/W8A8/W4A8/
mixed variants identically. Re-run only if attn_window / frame_chunk_size
/ obs_cam_keys change.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from pathlib import Path

import torch

# Triggers ptqeval.wam.lingbot_va package init, which puts lingbot-va/ on sys.path.
import ptqeval.wam.lingbot_va  # noqa: F401


logger = logging.getLogger("ptqeval.eval.measure_kv_cache")


def _alloc_mb() -> float:
    """Current resident allocation (NOT peak). Use for delta math."""
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024 / 1024


def _peak_mb() -> float:
    """Peak since last reset_peak_memory_stats."""
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024 / 1024


def measure(model_path: str, device: torch.device,
            dtype: torch.dtype,
            variant: str = "",
            variant_args_path: str = "") -> dict:
    from wan_va.modules.utils import load_transformer
    from diffusers import AutoencoderKLWan

    samples: dict[str, float] = {}

    # ---- 1. baseline (empty CUDA) ----
    torch.cuda.empty_cache()
    gc.collect()
    samples["baseline_mb"] = _alloc_mb()
    logger.info(f"baseline alloc: {samples['baseline_mb']:.1f} MB")

    # ---- 2. transformer weight load ----
    # If variant given, load quantized model via the variant's loader so
    # the measurement reflects the actual deployed weight size (W8/W4/
    # mixed instead of bf16 baseline). Otherwise load fp baseline.
    if variant:
        import importlib
        from omegaconf import OmegaConf
        loader_mod = importlib.import_module(
            f"ptqeval.wam.lingbot_va.method.{variant}.loader"
        )
        variant_args = {}
        if variant_args_path:
            variant_args = OmegaConf.to_container(
                OmegaConf.load(variant_args_path), resolve=True
            )
        transformer = loader_mod.load_quant_model(
            wan_model_path=os.path.join(model_path, "transformer"),
            variant_args=variant_args,
            device=device,
            dtype=dtype,
        )
    else:
        transformer = load_transformer(
            os.path.join(model_path, "transformer"),
            torch_dtype=dtype,
            torch_device=device,
            attn_mode="torch",
        )
    transformer.eval()
    samples["after_transformer_mb"] = _alloc_mb()
    samples["transformer_weight_mb"] = (
        samples["after_transformer_mb"] - samples["baseline_mb"]
    )
    logger.info(
        f"transformer weight ({variant or 'bf16'}): "
        f"{samples['transformer_weight_mb']:.1f} MB "
        f"(now {samples['after_transformer_mb']:.1f} MB total)"
    )

    # ---- 3. vae weight load ----
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(model_path, "vae"),
        torch_dtype=dtype,
    ).to(device).eval()
    samples["after_vae_mb"] = _alloc_mb()
    samples["vae_weight_mb"] = (
        samples["after_vae_mb"] - samples["after_transformer_mb"]
    )
    logger.info(
        f"vae weight: {samples['vae_weight_mb']:.1f} MB "
        f"(now {samples['after_vae_mb']:.1f} MB total)"
    )

    # ---- 3b. text encoder weight (own segment) ----
    # Measured as a load delta then freed. Eval offloads it to CPU at
    # steady state, but the transient swap-to-GPU during text encoding
    # makes it a real peak contributor; the memory_breakdown shows it as
    # its own segment. Freed before KV so the KV delta stays clean.
    from wan_va.modules.utils import load_text_encoder
    text_encoder = load_text_encoder(
        os.path.join(model_path, "text_encoder"),
        torch_dtype=dtype,
        torch_device=device,
    )
    samples["after_text_encoder_mb"] = _alloc_mb()
    samples["text_encoder_weight_mb"] = (
        samples["after_text_encoder_mb"] - samples["after_vae_mb"]
    )
    logger.info(
        f"text encoder weight: {samples['text_encoder_weight_mb']:.1f} MB "
        f"(freed before KV measurement)"
    )
    del text_encoder
    gc.collect()
    torch.cuda.empty_cache()

    # ---- 4. init_kv_cache ----
    # Mirror server.py's create_empty_cache call with the same RoboTwin
    # config values that determine total_tolen.
    from wan_va.configs.va_robotwin_cfg import va_robotwin_cfg as cfg

    attn_window = cfg.attn_window
    frame_chunk_size = cfg.frame_chunk_size
    n_cams = len(cfg.obs_cam_keys)
    patch_size = cfg.patch_size
    height = cfg.height
    width = cfg.width
    action_per_frame = cfg.action_per_frame
    use_cfg = (cfg.guidance_scale > 1) or (cfg.action_guidance_scale > 1)
    batch_size = 2 if use_cfg else 1

    latent_height = height // 16
    latent_width = (width // 16) * n_cams
    latent_token_per_chunk = (
        frame_chunk_size * latent_height * latent_width
    ) // (patch_size[0] * patch_size[1] * patch_size[2])
    action_token_per_chunk = frame_chunk_size * action_per_frame

    cache_name = "pos"
    transformer.create_empty_cache(
        cache_name,
        attn_window,
        latent_token_per_chunk,
        action_token_per_chunk,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
    )
    samples["after_kv_cache_mb"] = _alloc_mb()
    samples["kv_cache_mb"] = (
        samples["after_kv_cache_mb"] - samples["after_vae_mb"]
    )
    logger.info(
        f"kv_cache: {samples['kv_cache_mb']:.1f} MB "
        f"(now {samples['after_kv_cache_mb']:.1f} MB total)"
    )

    # ---- 5. activation peak ----
    # Not directly measured here (requires rotary_emb / temb / encoder
    # shapes matching the production _infer call, fragile). Instead the
    # chart in calc_cross_ckpt derives activation+scratch = measured_
    # peak_alloc_mb (from eval summary.csv, real production peak) - sum
    # of (text + xfmr + KV + VAE) measured here. All subtrahends are now
    # real torch.cuda.memory_allocated() deltas, so the derived
    # activation IS a measurement (just expressed as a difference), not
    # a reverse-fit.
    samples["activation_peak_mb"] = None

    samples_meta = {
        "attn_window": attn_window,
        "frame_chunk_size": frame_chunk_size,
        "n_cams": n_cams,
        "patch_size": list(patch_size),
        "height": height,
        "width": width,
        "action_per_frame": action_per_frame,
        "use_cfg": use_cfg,
        "batch_size": batch_size,
        "latent_token_per_chunk": latent_token_per_chunk,
        "action_token_per_chunk": action_token_per_chunk,
        "total_tolen": (attn_window // 2) * latent_token_per_chunk
                       + (attn_window // 2) * action_token_per_chunk,
        "num_layers": len(transformer.blocks),
        "num_heads": transformer.num_attention_heads,
        "head_dim": transformer.attention_head_dim,
        "dtype": str(dtype),
        "kv_dtype_bytes": 2,
        "variant": variant or "bf16",
    }

    # Sanity check: derived theoretical vs measured.
    theoretical_kv_mb = (
        samples_meta["num_layers"] * 2
        * batch_size
        * samples_meta["total_tolen"]
        * samples_meta["num_heads"]
        * samples_meta["head_dim"]
        * samples_meta["kv_dtype_bytes"]
    ) / 1024 / 1024
    samples_meta["theoretical_kv_mb"] = theoretical_kv_mb
    logger.info(
        f"theoretical KV: {theoretical_kv_mb:.1f} MB "
        f"(measured: {samples['kv_cache_mb']:.1f} MB; "
        f"delta {samples['kv_cache_mb'] - theoretical_kv_mb:+.1f} MB)"
    )

    return {"samples": samples, "meta": samples_meta}


def measure_activation(model_path: str, videos_root: str,
                       task: str | None, warmup: int, n_calls: int,
                       variant: str = "", variant_args_path: str = "") -> dict:
    """Measure the activation+scratch transient of a REAL forward by
    reusing measure_flops' server harness (build server -> warmup the KV
    cache -> measured calls). Resets the peak counter right before the
    measured calls and reads max_memory_allocated relative to the
    resident-before allocation. This is a direct measurement of the
    forward transient in one process (reset_peak -> forward -> read
    peak), the same allocation-delta method used for the weight/KV
    segments -- NOT a chart-side cross-source subtraction.

    Returns: forward_resident_mb, forward_peak_mb, activation_peak_mb
    (= forward_peak - forward_resident). Needs obs chunks under
    videos_root/visualization/real/ (use results/calib_capture/; variant
    obs removed in Phase 45.9-A)."""
    from pathlib import Path
    from ptqeval.eval.measure_flops import (
        _build_server, _pick_episode, _CHUNK_RE)

    server = _build_server(model_path,
                           Path("/tmp/measure_kv_cache_activation_scratch"))
    ep_dir = _pick_episode(Path(videos_root), task)
    chunks = sorted(
        (int(_CHUNK_RE.search(p.name).group(1)), p)
        for p in ep_dir.glob("obs_data_*.pt"))
    if not chunks:
        raise RuntimeError(f"no obs_data_*.pt under {ep_dir}")
    first = torch.load(chunks[0][1], weights_only=False, map_location="cpu")
    prompt = first[0]["task"]
    frame_chunk_size = server.job_config.frame_chunk_size
    init_obs = {"obs": [first[0]], "prompt": prompt,
                "save_visualization": False}
    logger.info(f"activation: episode {ep_dir.name}, warmup {warmup} chunks")
    server.infer({"reset": True, "prompt": prompt,
                  "save_visualization": False})
    for chunk_id in range(warmup):
        server._infer(init_obs, frame_st_id=chunk_id * frame_chunk_size)
    torch.cuda.synchronize()
    resident_before = _alloc_mb()
    torch.cuda.reset_peak_memory_stats()
    for j in range(n_calls):
        server._infer(init_obs,
                      frame_st_id=(warmup + j) * frame_chunk_size)
    peak = _peak_mb()
    logger.info(
        f"activation: resident_before={resident_before:.1f} MB, "
        f"forward_peak={peak:.1f} MB, "
        f"activation={peak - resident_before:.1f} MB")
    return {
        "forward_resident_mb": resident_before,
        "forward_peak_mb": peak,
        "activation_peak_mb": peak - resident_before,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=str,
                        default="models/lingbot-va-posttrain-robotwin",
                        help="HF model root (with transformer/ and vae/ subdirs).")
    parser.add_argument("--output", type=str, default="results/measured_kv_cache.json")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bf16",
                        choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--variant", default="",
                        help="Quant variant (resolves "
                             "ptqeval.wam.lingbot_va.method.<variant>.loader). "
                             "Empty -> bf16 baseline.")
    parser.add_argument("--variant_args", type=str, default="",
                        help="Variant runtime_args yaml (layer_config + "
                             "int_weights_ckpt paths).")
    parser.add_argument("--activation_videos_root", type=str, default="",
                        help="When set, ALSO measure the forward "
                             "activation+scratch transient via a real "
                             "server._infer (reuses measure_flops harness). "
                             "Points at a root with visualization/real/ obs "
                             "chunks (results/calib_capture; variant obs "
                             "removed Phase 45.9-A). Empty -> "
                             "activation_peak_mb stays None.")
    parser.add_argument("--activation_task", default=None,
                        help="Episode prompt substring filter for activation.")
    parser.add_argument("--activation_warmup", type=int, default=36,
                        help="Warmup chunks to fill KV before measuring "
                             "activation (default 36 = attn_window/2).")
    parser.add_argument("--activation_n_calls", type=int, default=3,
                        help="Measured _infer calls for activation peak.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    dtype = {"bf16": torch.bfloat16,
             "fp16": torch.float16,
             "fp32": torch.float32}[args.dtype]
    device = torch.device(args.device)

    result = measure(args.model_path, device, dtype,
                     variant=args.variant,
                     variant_args_path=args.variant_args)

    if args.activation_videos_root:
        gc.collect()
        torch.cuda.empty_cache()
        act = measure_activation(
            args.model_path, args.activation_videos_root,
            args.activation_task, args.activation_warmup,
            args.activation_n_calls,
            variant=args.variant, variant_args_path=args.variant_args)
        result["samples"].update(act)
        # Observational sum-check (principle.txt L12): do the measured
        # resident segments + measured activation reconstruct the
        # measured forward peak? Reported, never used to derive a value.
        s = result["samples"]
        seg_sum = (s.get("transformer_weight_mb", 0.0)
                   + s.get("vae_weight_mb", 0.0)
                   + s.get("kv_cache_mb", 0.0)
                   + s.get("activation_peak_mb", 0.0))
        s["segsum_vs_forwardpeak_mb"] = seg_sum - s.get("forward_peak_mb", 0.0)
        logger.info(
            f"sum-check: transformer+vae+kv+activation = {seg_sum:.1f} MB "
            f"vs forward_peak {s.get('forward_peak_mb', 0.0):.1f} MB "
            f"(gap {s['segsum_vs_forwardpeak_mb']:+.1f} MB)")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
