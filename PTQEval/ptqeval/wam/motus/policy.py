"""Motus RoboTwin policy with ViDiT-Q variant dispatch.

Thin wrapper over Motus's deploy_policy (inference/robotwin/Motus/deploy_policy.py:
get_model -> MotusPolicy, eval, reset_model):
  - get_model builds the bf16 MotusPolicy via Motus's own get_model, then, for
    variant=="viditq", swaps the target Linears in place on policy.model (the
    Motus nn.Module) via load_quant_model. Only the video expert's self_attn +
    all FFN + action/und output proj are quantized (action/und Q/K/V are fused
    nn.Parameter, not Linear).
  - calib_out (on-policy SmoothQuant): install per-input-channel absmax hooks on
    policy.model (FP / variant bf16); pooled per-task processes merge into the
    shared calib_out via _CalibState's flock merge-on-write.
  - profile_ops installs a MINIMAL PerfProbe (one-time init peak only; Motus runs
    a single fused inference_step via model.get_action() with no fastwam-style
    prefill/action-loop split — detailed stages are a refinement).

eval / reset_model delegate to Motus's own deploy_policy (which drive
model.set_instruction/update_obs/get_action + TASK_ENV.take_action).

Injected usr_args keys (via deploy_policy.yml or eval override):
    wan_path / vlm_path: required by Motus get_model (WAN + Qwen3-VL paths)
    variant / int_weights_ckpt / layer_config / calib_out / profile_ops /
    perf_log_path: ptqeval ViDiT-Q dispatch.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import torch

# Puts Motus/{,/bak,/inference/robotwin/Motus} on sys.path.
import ptqeval.wam.motus  # noqa: F401
from deploy_policy import (  # noqa: E402  (Motus inference/robotwin/Motus/deploy_policy.py)
    eval,  # noqa: F401  (re-exported for the RoboTwin policy contract)
    get_model as _motus_get_model,
    reset_model,  # noqa: F401
)

from ptqeval.eval.perf_probe import PerfProbe, StageRecord  # noqa: E402
from ptqeval.wam.motus.method.viditq.loader import load_quant_model  # noqa: E402

_MB = 1024.0 * 1024.0
logger = logging.getLogger("ptqeval.wam.motus.policy")


def _truthy(v: Any, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def _is_none_like(v: Any) -> bool:
    return v is None or (isinstance(v, str) and v.strip().lower() in {"", "none", "null"})


def get_model(usr_args: Dict[str, Any]):
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger("ptqeval.wam.motus").setLevel(logging.INFO)
    policy = _motus_get_model(usr_args)

    variant = str(usr_args.get("variant", "") or "").strip().lower()
    if variant == "viditq":
        int_weights = usr_args.get("int_weights_ckpt")
        if _is_none_like(int_weights):
            raise ValueError("variant=viditq requires `int_weights_ckpt`.")
        layer_config = usr_args.get("layer_config")
        load_quant_model(
            policy.model,
            str(int_weights),
            layer_config=None if _is_none_like(layer_config) else str(layer_config),
            device=policy.model.device if hasattr(policy.model, "device")
            else next(policy.model.parameters()).device,
        )
    elif variant not in ("", "bf16"):
        raise ValueError(f"unknown variant {variant!r} (expected '' | 'bf16' | 'viditq')")

    calib_out = usr_args.get("calib_out")
    if not _is_none_like(calib_out):
        if variant == "viditq":
            raise ValueError("calib_out requires the FP model; do not set variant=viditq.")
        from ptqeval.wam.motus.method.viditq.get_calib_data import install_calib_hooks
        install_calib_hooks(policy.model, str(calib_out))
        logger.info(f"calib mode: hooks on full Motus model -> {calib_out} (PerfProbe skipped)")
        return policy

    if _truthy(usr_args.get("profile_ops", True), default=True):
        perf_log_path = usr_args.get("perf_log_path")
        if not _is_none_like(perf_log_path):
            dev = (policy.model.device.index if getattr(policy.model, "device", None)
                   is not None and policy.model.device.index is not None else 0)
            probe = PerfProbe(str(perf_log_path),
                              task_name=str(usr_args.get("task_name", "task")), device=dev)
            probe.begin_call()
            probe.append_stage(StageRecord(
                stage="init", elapsed_ms=0.0,
                peak_alloc_mb=torch.cuda.max_memory_allocated(dev) / _MB,
                peak_reserved_mb=torch.cuda.max_memory_reserved(dev) / _MB))
            probe.end_call()

    return policy
