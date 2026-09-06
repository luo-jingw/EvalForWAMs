"""PTQ runner for Motus (3-expert MoT): FP model -> int weights.

Reuses the WAM-agnostic quantization core from the lingbot_va ViDiT-Q method
(compute_int_state_dict / _flatten_to_state_dict / _parse_bit_alloc_key). The
only Motus-specific piece is loading the FP model (load_motus_model: MotusConfig
+ released checkpoint, no hydra) and the layer_config regex targeting the WAN/
Motus block Linears.

Quantized targets (see configs/w4a4.yaml): per expert per block —
  video : video_model.wan_model.blocks.N.self_attn.{q,k,v,o} + ffn.{0,2}
  action: action_expert.blocks.N.wan_action_o + ffn.{0,2}
  und   : und_expert.blocks.N.wan_und_o + ffn.{0,2}
blocks.0 (all experts) + all non-block Linears stay FP. action/und Q/K/V are
fused nn.Parameter (wan_*_qkv) -> not nn.Linear -> unquantizable.

The blocks invoke these Linears as plain submodule calls, so the runtime loader
swaps them in place. Output: flat torch state_dict; per quantized layer the
lingbot_va ptq buffer layout (int_weight / scale_weight / bias +
zp_weight|szeros_weight + optional quarot_sign / act_channel_div). Keys are the
model's named_modules names, e.g.
  video_model.wan_model.blocks.1.self_attn.q.int_weight

CLI:
    python -m ptqeval.wam.motus.method.viditq.ptq \\
        --ckpt      <motus release ckpt dir> \\
        --wan_path  <WAN dir: config.json + Wan2.2_VAE.pth> \\
        --vlm_path  <Qwen3-VL checkpoint> \\
        --layer_config PTQEval/ptqeval/wam/motus/method/viditq/configs/w4a4_smooth.yaml \\
        --calib_data_path results/motus/motus_w4a4_smooth/calib/calib_data_clean.pth \\
        --output          results/motus/motus_w4a4_smooth/calib/int_weights_clean.pth
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import OmegaConf

# Puts FastWAM/{,/src,/experiments/robotwin} on sys.path.
import ptqeval.wam.motus  # noqa: F401

# WAM-agnostic quant core reused verbatim from the lingbot_va method. After
# the plan.txt T8 fix, importing this module no longer pulls in wan_va.
from ptqeval.wam.lingbot_va.method.viditq.ptq import (  # noqa: E402
    _flatten_to_state_dict,
    _parse_bit_alloc_key,
    compute_int_state_dict,
)

logger = logging.getLogger("ptqeval.wam.motus.method.viditq.ptq")


def load_motus_model(
    ckpt_path: str,
    device: torch.device,
    dtype: torch.dtype,
    wan_path: str,
    vlm_path: str,
    config_path: Optional[str] = None,
) -> nn.Module:
    """Instantiate the Motus FP model and load released weights.

    Mirrors MotusPolicy._load_model / _create_model_config (no hydra): read
    the deploy robotwin.yml (common + model), build a MotusConfig with
    load_pretrained_backbones=False (only WAN + Qwen3-VL config/skeleton, no
    backbone weights; VAE always built), then Motus(config) + load_checkpoint.

    Needs wan_path (WAN config.json + Wan2.2_VAE.pth) and vlm_path (Qwen3-VL) to
    even instantiate; the released ckpt (DeepSpeed dir with mp_rank_00_model_
    states.pt / a dict with 'module') carries the trained weights.

    NOTE: verified against MotusPolicy._create_model_config; to be smoke-checked
    once the Motus env + checkpoints are available.
    """
    import yaml

    from models.motus import Motus, MotusConfig  # Motus root on sys.path
    from ptqeval.wam.motus import MOTUS_POLICY_DIR

    if config_path is None:
        config_path = os.path.join(MOTUS_POLICY_DIR, "utils", "robotwin.yml")
    with open(config_path, "r", encoding="utf-8") as f:
        cd = yaml.safe_load(f)
    common = cd["common"]
    model_cfg = cd["model"]
    vae_path = os.path.join(wan_path, "Wan2.2_VAE.pth")
    hidden_size = model_cfg["action_expert"]["hidden_size"]
    ffn_mult = model_cfg["action_expert"]["ffn_dim_multiplier"]

    config = MotusConfig(
        wan_checkpoint_path=wan_path, vae_path=vae_path, wan_config_path=wan_path,
        video_precision="bfloat16", vlm_checkpoint_path=vlm_path,
        und_expert_hidden_size=512, und_expert_ffn_dim_multiplier=4,
        und_expert_norm_eps=1e-5, und_layers_to_extract=None,
        vlm_adapter_input_dim=2048, vlm_adapter_projector_type="mlp3x_silu",
        num_layers=30, action_state_dim=common["state_dim"],
        action_dim=common["action_dim"], action_expert_dim=hidden_size,
        action_expert_ffn_dim_multiplier=ffn_mult, action_expert_norm_eps=1e-6,
        global_downsample_rate=common["global_downsample_rate"],
        video_action_freq_ratio=common["video_action_freq_ratio"],
        num_video_frames=common["num_video_frames"], video_loss_weight=1.0,
        action_loss_weight=1.0, batch_size=1, video_height=common["video_height"],
        video_width=common["video_width"], load_pretrained_backbones=False,
        training_mode="finetune",
    )
    model = Motus(config).to(device)
    model.load_checkpoint(ckpt_path, strict=False)
    model = model.eval().requires_grad_(False)
    return model


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute INT weights + per-channel scales for the FastWAM "
        "MoT model (video + action experts). Output: torch state_dict at --output."
    )
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Released Motus checkpoint (DeepSpeed dir / dict with 'module').")
    parser.add_argument("--wan_path", type=str, required=True,
                        help="WAN dir (config.json + Wan2.2_VAE.pth).")
    parser.add_argument("--vlm_path", type=str, required=True,
                        help="Qwen3-VL checkpoint/dir (vlm_checkpoint_path).")
    parser.add_argument("--config", type=str, default=None,
                        help="robotwin.yml (default: Motus inference/robotwin/Motus/utils/robotwin.yml).")
    parser.add_argument("--layer_config", type=str, required=True,
                        help="YAML with remain_fp_regex + weight/act bits + preprocessing.")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for the flat state_dict (.pth).")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--load_dtype", type=str, default="bf16",
                        choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--calib_data_path", type=str, default=None,
                        help="Override the config's calib_data_path (e.g. per-run "
                             "clean vs randomized calibration). SmoothQuant only.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    cfg = OmegaConf.load(args.layer_config)
    remain_fp_regex = str(cfg.remain_fp_regex)

    # Same schema as lingbot_va ptq main(): mixed-precision configs declare
    # weight_bits_default + bit_alloc; homogeneous configs declare weight_bits.
    bit_alloc_raw = cfg.get("bit_alloc", None)
    bit_alloc: Optional[dict] = None
    if bit_alloc_raw:
        bit_alloc = {
            _parse_bit_alloc_key(k): list(v)
            for k, v in OmegaConf.to_container(bit_alloc_raw).items()
        }
        weight_bits = int(cfg.get("weight_bits_default", cfg.get("weight_bits", 8)))
    else:
        weight_bits = int(cfg.weight_bits)
    act_bits = int(cfg.get("act_bits_default", cfg.get("act_bits", 8)))
    if weight_bits not in (4, 8):
        raise ValueError(f"weight_bits (default) must be 8 or 4, got {weight_bits}")
    if act_bits not in (4, 8):
        raise ValueError(f"act_bits (default) must be 8 or 4, got {act_bits}")
    if bit_alloc:
        for (w, a_opt) in bit_alloc:
            if w not in (4, 8):
                raise ValueError(f"bit_alloc weight_bits must be 4 or 8, got {w}")
            if a_opt is not None and a_opt not in (4, 8):
                raise ValueError(f"bit_alloc act_bits must be 4/8/omitted, got {a_opt}")

    # Phase 24d/28 support asymmetric weight only; W4A4 tier is per-group
    # symmetric inside _quantize_one. Same guard as lingbot_va ptq.
    weight_sym = bool(cfg.get("weight_sym", True))
    has_w4a4_tier = bit_alloc and any(
        (w == 4 and a_opt == 4) for (w, a_opt) in bit_alloc.keys()
    )
    if weight_sym and not has_w4a4_tier:
        raise ValueError(
            f"layer_config {args.layer_config} has weight_sym=true but no W4A4 "
            f"tier; set weight_sym: false for W8A8 / W4A8 configs."
        )

    quarot_enabled = bool(cfg.get("quarot", False))
    quarot_seed_base = int(cfg.get("quarot_seed_base", 0))
    quarot_layer_regex = str(cfg.get("quarot_layer_regex", ".*"))
    smooth_quant_enabled = bool(cfg.get("smooth_quant", False))
    smooth_alpha = float(cfg.get("smooth_alpha", 0.99))
    calib_data_path = args.calib_data_path or cfg.get("calib_data_path", None)
    calib_data = None
    if smooth_quant_enabled:
        if not calib_data_path:
            raise ValueError(
                f"layer_config {args.layer_config} enables smooth_quant but "
                f"declares no calib_data_path; run FastWAM calibration first."
            )
        calib_data = torch.load(str(calib_data_path), weights_only=True)
        if not isinstance(calib_data, dict):
            raise ValueError(
                f"calib_data_path {calib_data_path} did not load a dict."
            )

    load_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                  "fp32": torch.float32}[args.load_dtype]
    device = torch.device(args.device)
    logger.info(f"loading FP Motus model from {args.ckpt} (dtype={load_dtype}, device={device})")
    model = load_motus_model(
        args.ckpt, device=device, dtype=load_dtype,
        wan_path=args.wan_path, vlm_path=args.vlm_path, config_path=args.config,
    )

    n_linear_total = sum(1 for _, m in model.named_modules() if isinstance(m, nn.Linear))
    logger.info(
        f"weight_bits={weight_bits} act_bits={act_bits} bit_alloc={bit_alloc} "
        f"remain_fp_regex={remain_fp_regex!r}"
    )
    logger.info(
        f"smooth_quant={smooth_quant_enabled} alpha={smooth_alpha} "
        f"quarot={quarot_enabled} seed_base={quarot_seed_base} "
        f"layer_regex={quarot_layer_regex!r}; total nn.Linear={n_linear_total}"
    )

    entries = compute_int_state_dict(
        model,
        remain_fp_regex,
        weight_bits,
        act_bits=act_bits,
        quarot_enabled=quarot_enabled,
        quarot_seed_base=quarot_seed_base,
        quarot_layer_regex=quarot_layer_regex,
        smooth_quant_enabled=smooth_quant_enabled,
        smooth_alpha=smooth_alpha,
        calib_data=calib_data,
        bit_alloc=bit_alloc,
    )
    n_quant = len(entries)
    logger.info(f"quantized {n_quant} layers; kept {n_linear_total - n_quant} as FP")

    # Observation: per-expert target-Linear counts (no assert).
    per_expert: dict[str, int] = {}
    for name in entries:
        exp = ("video" if "video_model" in name
               else "action" if "action_expert" in name
               else "und" if "und_expert" in name else "other")
        per_expert[exp] = per_expert.get(exp, 0) + 1
    logger.info(f"quantized target counts by expert: {per_expert}")

    sd = _flatten_to_state_dict(entries)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(sd, args.output)
    size_mb = os.path.getsize(args.output) / (1024.0 * 1024.0)
    logger.info(f"wrote {args.output} ({size_mb:.1f} MB; {len(sd)} tensor keys)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
