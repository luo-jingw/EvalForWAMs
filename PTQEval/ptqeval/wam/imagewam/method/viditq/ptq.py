"""PTQ runner for ImageWAM (FLUX.2 Klein-4B): FP MoT model -> int weights.

Reuses the WAM-agnostic quantization core from the lingbot_va ViDiT-Q method
(compute_int_state_dict / _flatten_to_state_dict / _parse_bit_alloc_key). The
only ImageWAM-specific piece is loading the FP model (load_imagewam_model:
hydra-composed flux2 sim config + released checkpoint) and the layer_config
regex targeting the FLUX.2 DiT block Linears.

Quantized targets (see configs/w4a4.yaml): per expert per block —
  video  double_blocks.N: {img,txt}_attn.{qkv,proj} + {img,txt}_mlp.{0,2}
  video  single_blocks.N: linear1 / linear2
  action double_blocks.N: img_attn.{qkv,proj} + img_mlp.{0,2}
  action single_blocks.N: linear1 / linear2
double_blocks.0 (both experts) and all non-block Linears (modulation, img_in/
txt_in/time_in, final_layer, AE, Qwen3 text encoder) stay FP.

The FLUX.2 DiT invokes these Linears as plain submodule calls, so the runtime
loader swaps them in place (no block subclass). Output: flat torch state_dict;
per quantized layer the lingbot_va ptq buffer layout (int_weight / scale_weight
/ bias + zp_weight|szeros_weight + optional quarot_sign / act_channel_div). Keys
are the model's named_modules names, e.g.
  video_expert.transformer.double_blocks.1.img_attn.qkv.int_weight

CLI:
    python -m ptqeval.wam.imagewam.method.viditq.ptq \\
        --ckpt        .../imagewam_release/robotwin/flux2_klein_4b/model.pt \\
        --flux2_model_path .../FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors \\
        --ae_model_path    .../FLUX.2-dev/ae.safetensors \\
        --layer_config PTQEval/ptqeval/wam/imagewam/method/viditq/configs/w4a4_smooth.yaml \\
        --calib_data_path results/imagewam/imagewam_w4a4_smooth/calib/calib_data_clean.pth \\
        --output          results/imagewam/imagewam_w4a4_smooth/calib/int_weights_clean.pth
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
import ptqeval.wam.imagewam  # noqa: F401

# WAM-agnostic quant core reused verbatim from the lingbot_va method. After
# the plan.txt T8 fix, importing this module no longer pulls in wan_va.
from ptqeval.wam.lingbot_va.method.viditq.ptq import (  # noqa: E402
    _flatten_to_state_dict,
    _parse_bit_alloc_key,
    compute_int_state_dict,
)

logger = logging.getLogger("ptqeval.wam.imagewam.method.viditq.ptq")


def load_imagewam_model(
    ckpt_path: str,
    device: torch.device,
    dtype: torch.dtype,
    flux2_model_path: str,
    ae_model_path: str,
    flux2_src_path: Optional[str] = None,
    task: str = "robotwin_flux2_klein_4b_base_imagewam",
    load_text_encoder: bool = False,
) -> nn.Module:
    """Instantiate the FLUX.2 ImageWAM FP model and load released weights.

    Composes ImageWAM/configs/sim_robotwin.yaml with `task=<flux2 task>` (whose
    defaults select the imagewam_flux2_klein_4b_base model), filling the model
    config's three `???`/external paths: the FLUX.2 backbone .safetensors
    (flux2_model_path), the FLUX.2 autoencoder .safetensors (ae_model_path), and
    the vendored FLUX.2 source tree (flux2_src_path -> IMAGEWAM_FLUX2_SRC).

    `load_text_encoder=False` (default) for PTQ: only the DiT transformer weights
    are quantized/walked. The released checkpoint is a dict with a "mot" key
    (LoRA-merge/remap handled by ImageWAM.load_checkpoint).

    NOTE: exact hydra composition (unresolved `???` in the flux2 task's data
    group, extra model `???` keys) is to be verified once the ImageWAM env is up.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate

    from ptqeval.wam.imagewam import IMAGEWAM_ROOT, IMAGEWAM_FLUX2_SRC

    if flux2_src_path is None:
        flux2_src_path = IMAGEWAM_FLUX2_SRC
    configs_root = os.path.join(IMAGEWAM_ROOT, "configs")
    overrides = [
        f"task={task}",
        f"model.flux2_model_path={flux2_model_path}",
        f"model.ae_model_path={ae_model_path}",
        f"model.flux2_src_path={flux2_src_path}",
        f"model.load_text_encoder={bool(load_text_encoder)}",
    ]
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=configs_root):
        cfg = compose(config_name="sim_robotwin.yaml", overrides=overrides)

    import inspect
    from hydra.utils import get_method

    model_dict = OmegaConf.to_container(cfg.model, resolve=True)
    # sim_robotwin.yaml's inline `model:` block carries WAN22-only keys
    # (e.g. skip_dit_load_from_pretrain) that merge in when the flux2 task
    # overrides /model but which create_imagewam_flux2_klein rejects. Filter to
    # the target's accepted kwargs (mirrors ImageWAM get_model's allowed-keys).
    _sig = set(inspect.signature(get_method(model_dict["_target_"])).parameters)
    model_dict = {k: v for k, v in model_dict.items() if k == "_target_" or k in _sig}
    model_cfg = OmegaConf.create(model_dict)
    model = instantiate(model_cfg, model_dtype=dtype, device=str(device))
    model.load_checkpoint(ckpt_path)
    model = model.to(device).eval().requires_grad_(False)
    return model


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute INT weights + per-channel scales for the FastWAM "
        "MoT model (video + action experts). Output: torch state_dict at --output."
    )
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Released ImageWAM checkpoint (model.pt) with the 'mot' key.")
    parser.add_argument("--flux2_model_path", type=str, required=True,
                        help="FLUX.2 backbone .safetensors (…/FLUX.2-klein-base-4B/"
                             "flux-2-klein-base-4b.safetensors).")
    parser.add_argument("--ae_model_path", type=str, required=True,
                        help="FLUX.2 autoencoder .safetensors (…/FLUX.2-dev/ae.safetensors).")
    parser.add_argument("--flux2_src_path", type=str, default=None,
                        help="FLUX.2 source tree (default: ImageWAM/third_party/flux2).")
    parser.add_argument("--task", type=str,
                        default="robotwin_flux2_klein_4b_base_imagewam",
                        help="Hydra task config selecting the flux2 model.")
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
    logger.info(f"loading FP FastWAM model from {args.ckpt} (dtype={load_dtype}, device={device})")
    model = load_imagewam_model(
        args.ckpt, device=device, dtype=load_dtype,
        flux2_model_path=args.flux2_model_path,
        ae_model_path=args.ae_model_path,
        flux2_src_path=args.flux2_src_path,
        task=args.task,
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
        exp = ("video" if "video_expert" in name
               else "action" if "action_expert" in name else "other")
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
