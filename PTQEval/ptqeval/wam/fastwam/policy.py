"""FastWAM RoboTwin policy with ViDiT-Q variant dispatch + PerfProbe.

Thin wrapper over FastWAM's WorldActionRobotWinPolicy (deploy_policy.py):
  - get_model builds the bf16 policy via FastWAM's own get_model, then, for
    variant=="viditq", swaps the two experts' target Linears in place via
    load_quant_model (no FastWAM edit, no block subclass).
  - installs a PerfProbe that times the inference stages by monkeypatching
    the model's sub-methods. Stage names reuse ptqeval.eval.aggregator's
    _KNOWN_STAGES so the aggregator needs no change:
        init          <- one-time model-load peak (emitted once at get_model)
        text_encoder  <- model.encode_prompt
        vae_encode    <- model._encode_input_image_latents_tensor
        transformer   <- mot.prefill_video_cache        (video expert, 1x)
        action_head   <- sum of mot.forward_action_with_video_cache (action, 10x)

NOTE on `profile_ops`: here it toggles the **stage-level** PerfProbe (wall time
+ peak memory per stage), NOT the op-level (linear/attention/memcpy/other)
classification that the shared run_eval / lingbot_va server attach to the same
flag name. FastWAM's op-level breakdown + FLOPs are produced separately and
reproducibly by `ptqeval.wam.fastwam.measure_op_profile` (they are shape-
invariant across tasks, so a single standalone run is representative — see that
module's docstring). Do not expect `<task>_op_profile.json` from this eval path.

RoboTwin policy contract (unchanged): get_model / eval / reset_model.

Injected usr_args keys (via deploy_policy.yml or eval override):
    variant:            "" -> bf16 ; "viditq" -> quantized
    int_weights_ckpt:   path to the PTQ int_weights .pth (required for viditq)
    layer_config:       optional yaml path (logging only)
    profile_ops:        bool, default True -> install stage-level PerfProbe
    perf_log_path:      JSONL output path for PerfProbe (required if profiling)
"""
from __future__ import annotations

import logging
import types
from typing import Any, Dict, Optional

import torch

# Puts FastWAM/{,/src,/experiments/robotwin} on sys.path.
import ptqeval.wam.fastwam  # noqa: F401
from fastwam_policy.deploy_policy import (  # noqa: E402
    WorldActionRobotWinPolicy,
    encode_obs,
    get_model as _fastwam_get_model,
)

from ptqeval.eval.perf_probe import PerfProbe, StageRecord  # noqa: E402
from ptqeval.wam.fastwam.method.viditq.loader import load_quant_model  # noqa: E402

_MB = 1024.0 * 1024.0


def _truthy(v: Any, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def _is_none_like(v: Any) -> bool:
    return v is None or (isinstance(v, str) and v.strip().lower() in {"", "none", "null"})


def _install_probe(policy: WorldActionRobotWinPolicy, probe: PerfProbe) -> None:
    """Monkeypatch model sub-methods to emit per-stage PerfProbe records, and
    wrap the policy's per-replan inference with begin_call / end_call.
    """
    model = policy.model
    mot = model.mot
    dev = probe.device

    # --- single-shot stages: wrap the method in a probe.stage context ---
    def _wrap_stage(obj: Any, attr: str, stage_name: str) -> None:
        orig = getattr(obj, attr)

        def wrapped(*a, **k):
            with probe.stage(stage_name):
                return orig(*a, **k)

        setattr(obj, attr, wrapped)

    _wrap_stage(model, "encode_prompt", "text_encoder")
    _wrap_stage(model, "_encode_input_image_latents_tensor", "vae_encode")
    _wrap_stage(mot, "prefill_video_cache", "transformer")

    # --- action loop: accumulate the 10 forward_action calls into ONE record ---
    orig_fa = mot.forward_action_with_video_cache
    acc = {"ms": 0.0, "first": True}

    def fa_wrapped(*a, **k):
        if acc["first"]:
            torch.cuda.synchronize(dev)
            torch.cuda.reset_peak_memory_stats(dev)
            acc["first"] = False
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        out = orig_fa(*a, **k)
        e1.record()
        torch.cuda.synchronize(dev)
        acc["ms"] += e0.elapsed_time(e1)
        return out

    mot.forward_action_with_video_cache = fa_wrapped

    # --- bracket each replan inference with begin_call / end_call ---
    orig_infer = policy._infer_action_chunk

    def infer_wrapped(self, observation, instruction):
        probe.begin_call()
        acc["ms"] = 0.0
        acc["first"] = True
        try:
            return orig_infer(observation, instruction)
        finally:
            probe.append_stage(StageRecord(
                stage="action_head",
                elapsed_ms=acc["ms"],
                peak_alloc_mb=torch.cuda.max_memory_allocated(dev) / _MB,
                peak_reserved_mb=torch.cuda.max_memory_reserved(dev) / _MB,
            ))
            probe.end_call()

    policy._infer_action_chunk = types.MethodType(infer_wrapped, policy)


def get_model(usr_args: Dict[str, Any]) -> WorldActionRobotWinPolicy:
    # eval_policy.py / FastWAM configure logging before this runs; surface the
    # loader's swap histogram (correctness-relevant) by forcing INFO on both
    # the root and our package logger regardless of the pre-existing config.
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger("ptqeval.wam.fastwam").setLevel(logging.INFO)
    policy = _fastwam_get_model(usr_args)

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
            device=policy.model.device,
        )
    elif variant not in ("", "bf16"):
        raise ValueError(f"unknown variant {variant!r} (expected '' | 'bf16' | 'viditq')")

    # Calibration mode (on-policy SmoothQuant calib): install per-input-channel
    # absmax hooks on the FULL model (policy.model) — the same object ptq passes
    # to compute_int_state_dict — so the hook keys are exactly the names ptq looks
    # up (e.g. `action_expert.blocks.0.ffn.0`, no `mot.` prefix). Must run on the
    # FP (bf16) model; PerfProbe is skipped. Under eval_pool each per-task process
    # merges into the shared calib_out via _CalibState's flock merge-on-write.
    calib_out = usr_args.get("calib_out")
    if not _is_none_like(calib_out):
        if variant == "viditq":
            raise ValueError("calib_out requires the FP model; do not set variant=viditq.")
        from ptqeval.wam.fastwam.method.viditq.get_calib_data import install_calib_hooks
        install_calib_hooks(policy.model, str(calib_out))
        logging.getLogger("ptqeval.wam.fastwam.policy").info(
            f"calib mode: hooks on full FastWAM model -> {calib_out} (PerfProbe skipped)")
        return policy

    if _truthy(usr_args.get("profile_ops", True), default=True):
        perf_log_path = usr_args.get("perf_log_path")
        if _is_none_like(perf_log_path):
            raise ValueError("profile_ops is on but `perf_log_path` is not set.")
        device_index = policy.model.device.index if policy.model.device.index is not None else 0
        task_name = str(usr_args.get("task_name", "task"))
        probe = PerfProbe(str(perf_log_path), task_name=task_name, device=device_index)
        # init stage: emit the one-time model-load peak (weights + T5 + VAE
        # transiently resident). aggregator reads init_peak_mb from this record;
        # it is excluded from diffusion_peak (44g) so it does not pollute the
        # working-set memory metric.
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
