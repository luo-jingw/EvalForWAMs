# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Deployment config for the lingbot-va inference package (M2).

Single source of truth for the runtime parameters a robot deployment
needs. The heavy model-side defaults (schedulers, patch sizes, action
norm stats, ...) are inherited from wan_va's va_robotwin_cfg
(VA_CONFIGS['robotwin']); this dataclass only carries the deploy-facing
subset and overrides them in to_job_config().

No torch / wan_va import at module load, so `from
ptqeval.inference.config import InferenceConfig` stays light; the heavy
imports happen inside to_job_config() (which builds the VA_Server config).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Any, Optional


@dataclass
class InferenceConfig:
    """Robot-facing runtime parameters. Paths are resolved as given (no
    implicit cwd magic); pass absolute paths on the device.

    The STRUCTURAL params that define the robot/model -- env_type,
    attn_window, frame_chunk_size, the action-channel mapping, norm_stat,
    obs_cam_keys, action_per_frame -- are NOT hardcoded here; they are
    inherited from base_config (a VA_CONFIGS key). Use 'g1_server' for the
    Unitree G1 (env_type=none, attn_window=30, joint action mapping) and
    'robotwin' for the RoboTwin layout (attn_window=72, EEF mapping).
    Getting base_config wrong silently corrupts actions, so it is explicit.

    The Optional fields below default to None = inherit base_config's
    value; set one only to override a sampling knob at deploy time."""
    model_path: str                         # assembled model dir (transformer/ vae/ text_encoder/ tokenizer/)
    base_config: str = "g1_server"          # VA_CONFIGS key -> structural params (robot/model layout)
    int_weights_ckpt: str = ""              # weight_prep output; "" -> bf16 baseline
    layer_config: str = ""                  # quant layer config (w4a4); "" -> bf16
    variant: str = "viditq"                 # method.<variant>.loader namespace
    text_cond_cache: str = ""               # precompute_text output; "" -> needs T5
    serve_residency: bool = False           # serial text/diffusion residency (T5 vs xfmr mutually exclusive)
    offload_target: str = "cpu"             # serve_residency transformer offload: cpu|disk
    offload_dir: str = ""
    dtype: str = "bf16"                     # bf16|fp16|fp32
    device: str = "cuda:0"
    save_root: str = "/tmp/va_infer"
    # Optional overrides (None -> inherit base_config). attn_window is a
    # structural param -- leave None so the robot's value wins (g1=30).
    attn_window: Optional[int] = None
    env_type: Optional[str] = None
    enable_offload: Optional[bool] = None   # None -> inherit (g1_server: True)
    guidance_scale: Optional[float] = None
    action_guidance_scale: Optional[float] = None
    num_inference_steps: Optional[int] = None
    action_num_inference_steps: Optional[int] = None

    @classmethod
    def from_yaml(cls, path: str) -> "InferenceConfig":
        """Load from a flat yaml (keys = dataclass field names). Unknown
        keys raise; missing required (model_path) raises."""
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown keys in {path}: {sorted(unknown)}")
        return cls(**raw)

    def _local_rank(self) -> int:
        return int(self.device.split(":")[-1]) if ":" in self.device else 0

    def _write_variant_args(self) -> str:
        """VA_Server reads variant_args as a yaml PATH (OmegaConf.load).
        Materialise one from int_weights_ckpt + layer_config so the deploy
        config stays flat. Written under save_root."""
        import yaml
        os.makedirs(self.save_root, exist_ok=True)
        path = os.path.join(self.save_root, "_variant_args.yaml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                {"layer_config": self.layer_config,
                 "int_weights_ckpt": self.int_weights_ckpt}, f)
        return path

    def to_job_config(self) -> Any:
        """Build the EasyDict VA_Server consumes: a deep copy of the chosen
        base_config with the deploy overrides applied. Structural params
        (env_type, action mapping, norm_stat, cams, ...) are inherited from
        base_config; only explicitly-set (non-None) fields are overridden.
        Deep copy so we never mutate the shared VA_CONFIGS singleton."""
        import copy
        import torch
        from ptqeval.inference._env import ensure_wan_va_on_path
        ensure_wan_va_on_path()
        from configs import VA_CONFIGS

        if self.base_config not in VA_CONFIGS:
            raise ValueError(
                f"base_config '{self.base_config}' not in VA_CONFIGS "
                f"({sorted(VA_CONFIGS)})")
        cfg = copy.deepcopy(VA_CONFIGS[self.base_config])
        # Always-applied deploy overrides.
        cfg.wan22_pretrained_model_name_or_path = self.model_path
        cfg.param_dtype = {"bf16": torch.bfloat16,
                           "fp16": torch.float16,
                           "fp32": torch.float32}[self.dtype]
        cfg.save_root = self.save_root
        # Optional overrides: None -> keep base_config's (structural) value.
        if self.attn_window is not None:
            cfg.attn_window = self.attn_window
        if self.env_type is not None:
            cfg.env_type = self.env_type
        if self.enable_offload is not None:
            cfg.enable_offload = self.enable_offload
        if self.guidance_scale is not None:
            cfg.guidance_scale = self.guidance_scale
        if self.action_guidance_scale is not None:
            cfg.action_guidance_scale = self.action_guidance_scale
        if self.num_inference_steps is not None:
            cfg.num_inference_steps = self.num_inference_steps
        if self.action_num_inference_steps is not None:
            cfg.action_num_inference_steps = self.action_num_inference_steps
        # Measurement / eval-only knobs stay off in deployment.
        cfg.perf_log_dir = None
        cfg.perf_task_name = "deploy"
        cfg.profile_ops = False
        cfg.profile_n_calls = 0
        # Text-cond residency.
        cfg.text_cond_cache = self.text_cond_cache or None
        cfg.serve_residency = self.serve_residency
        cfg.offload_target = self.offload_target
        cfg.offload_dir = self.offload_dir
        # Quant variant (overlay on the FP transformer). "" -> bf16 baseline.
        if self.int_weights_ckpt:
            cfg.variant = self.variant
            cfg.variant_args = self._write_variant_args()
        else:
            cfg.variant = None
            cfg.variant_args = None
        # Single-GPU deployment.
        cfg.rank = 0
        cfg.local_rank = self._local_rank()
        cfg.world_size = 1
        return cfg
