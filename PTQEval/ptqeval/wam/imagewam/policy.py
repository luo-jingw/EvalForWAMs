"""ImageWAM (FLUX.2 Klein-4B) RoboTwin policy with ViDiT-Q variant dispatch.

Thin wrapper over ImageWAM's WorldActionRobotWinPolicy (imagewam_policy/
deploy_policy.py):
  - get_model builds the bf16 policy via ImageWAM's own get_model, then, for
    variant=="viditq", swaps the FLUX.2 DiT target Linears in place via
    load_quant_model (no ImageWAM edit).
  - calib_out (on-policy SmoothQuant calib): install per-input-channel absmax
    hooks on the FP model (variant bf16); each pooled per-task process merges
    into the shared calib_out via _CalibState's flock merge-on-write.
  - profile_ops installs a MINIMAL PerfProbe: the one-time init peak + a single
    per-call "action_head" timing around policy._infer_action_chunk. (The
    detailed per-stage breakdown — video prefill vs action loop, mapped onto
    ImageWAM.infer_action's internals — is a refinement; ImageWAM's method names
    differ from fastwam's mot.prefill_video_cache / forward_action_with_video_cache.)

RoboTwin policy contract (unchanged): get_model / eval / reset_model.

Injected usr_args keys (via deploy_policy.yml or eval override):
    variant:            "" -> bf16 ; "viditq" -> quantized
    int_weights_ckpt:   path to the PTQ int_weights .pth (required for viditq)
    layer_config:       optional yaml path (logging only)
    calib_out:          if set -> FP model + absmax hooks (requires variant bf16)
    profile_ops:        bool, default True -> install minimal PerfProbe
    perf_log_path:      JSONL output path for PerfProbe (required if profiling)
"""
from __future__ import annotations

import logging
import time
import types
from typing import Any, Dict, Optional

import torch

# Puts ImageWAM/{,/src,/third_party/flux2/src,/experiments/robotwin} on sys.path.
import ptqeval.wam.imagewam  # noqa: F401
from imagewam_policy.deploy_policy import (  # noqa: E402
    WorldActionRobotWinPolicy,
    encode_obs,
    get_model as _imagewam_get_model,
)

from ptqeval.eval.perf_probe import PerfProbe, StageRecord  # noqa: E402

_MB = 1024.0 * 1024.0

logger = logging.getLogger("ptqeval.wam.imagewam.policy")


def _truthy(v: Any, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def _is_none_like(v: Any) -> bool:
    return v is None or (isinstance(v, str) and v.strip().lower() in {"", "none", "null"})


def _install_probe(policy: WorldActionRobotWinPolicy, probe: PerfProbe) -> None:
    """Minimal timing: wrap _infer_action_chunk as one 'action_head' stage per
    call. Detailed video/action sub-stages are a refinement (see module docstring)."""
    orig_infer = policy._infer_action_chunk

    def infer_wrapped(self, *args, **kwargs):
        probe.begin_call()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        out = orig_infer(*args, **kwargs)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0
        dev = self.model.device.index if self.model.device.index is not None else 0
        probe.append_stage(StageRecord(
            stage="action_head", elapsed_ms=dt,
            peak_alloc_mb=torch.cuda.max_memory_allocated(dev) / _MB,
            peak_reserved_mb=torch.cuda.max_memory_reserved(dev) / _MB))
        probe.end_call()
        return out

    policy._infer_action_chunk = types.MethodType(infer_wrapped, policy)


def get_model(usr_args: Dict[str, Any]) -> WorldActionRobotWinPolicy:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger("ptqeval.wam.imagewam").setLevel(logging.INFO)
    policy = _imagewam_get_model(usr_args)

    variant = str(usr_args.get("variant", "") or "").strip().lower()
    if variant == "viditq":
        int_weights = usr_args.get("int_weights_ckpt")
        if _is_none_like(int_weights):
            raise ValueError("variant=viditq requires `int_weights_ckpt`.")
        layer_config = usr_args.get("layer_config")
        # Lazy import: qwan_extension (the W4A4 CUDA kernel) is only needed to
        # quantize. Importing it at module top would force the bf16 path to load
        # the kernel .so as well, which fails when the shared build is compiled
        # for a different torch ABI. Mirrors the install_calib_hooks lazy import.
        from ptqeval.wam.imagewam.method.viditq.loader import load_quant_model
        load_quant_model(
            policy.model,
            str(int_weights),
            layer_config=None if _is_none_like(layer_config) else str(layer_config),
            device=policy.model.device,
        )
    elif variant not in ("", "bf16"):
        raise ValueError(f"unknown variant {variant!r} (expected '' | 'bf16' | 'viditq')")

    # Calibration mode (on-policy SmoothQuant): install absmax hooks on the whole
    # ImageWAM model (same object ptq walks -> matching names). FP model only.
    calib_out = usr_args.get("calib_out")
    if not _is_none_like(calib_out):
        if variant == "viditq":
            raise ValueError("calib_out requires the FP model; do not set variant=viditq.")
        from ptqeval.wam.imagewam.method.viditq.get_calib_data import install_calib_hooks
        install_calib_hooks(policy.model, str(calib_out))
        logger.info(f"calib mode: hooks on full ImageWAM model -> {calib_out} (PerfProbe skipped)")
        return policy

    if _truthy(usr_args.get("profile_ops", True), default=True):
        perf_log_path = usr_args.get("perf_log_path")
        if _is_none_like(perf_log_path):
            raise ValueError("profile_ops is on but `perf_log_path` is not set.")
        device_index = policy.model.device.index if policy.model.device.index is not None else 0
        task_name = str(usr_args.get("task_name", "task"))
        probe = PerfProbe(str(perf_log_path), task_name=task_name, device=device_index)
        probe.begin_call()
        probe.append_stage(StageRecord(
            stage="init", elapsed_ms=0.0,
            peak_alloc_mb=torch.cuda.max_memory_allocated(device_index) / _MB,
            peak_reserved_mb=torch.cuda.max_memory_reserved(device_index) / _MB))
        probe.end_call()
        _install_probe(policy, probe)

    return policy


def eval(TASK_ENV, model, observation: Optional[Dict[str, Any]]):  # noqa: A001
    model.step(TASK_ENV, encode_obs(observation))


def reset_model(model):
    model.reset()
