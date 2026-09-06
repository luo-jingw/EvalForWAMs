# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Inference engine for robot deployment (M3).

Thin wrapper over server.VA_Server: it (1) brings up a single-process
torch.distributed group (VA_Server's build path calls collectives / sets
the device via init_distributed), (2) builds the VA_Server from an
InferenceConfig, (3) exposes reset / infer / compute_kv_cache. All forward
logic (diffusion, KV cache, VAE encode, text-cond injection) is reused
from VA_Server unchanged.

The infer(request) entry mirrors VA_Server.infer's dict protocol so a
future server layer can forward request dicts verbatim (server interface
is TBD -- see infer_plan.txt Phase 6).
"""
from __future__ import annotations

import os
from typing import Any, Optional

import ptqeval.wam.lingbot_va  # noqa: F401  (package init -> wan_va on sys.path)
from ptqeval.inference.config import InferenceConfig


def _ensure_single_process_dist(local_rank: int) -> None:
    """Bring up a 1-process nccl group if none exists. VA_Server's build
    path (and wan_va's distributed util) assume an initialized group +
    a device set via init_distributed. Idempotent: a no-op when a group
    is already up (e.g. launched under torchrun)."""
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", str(local_rank))
    os.environ.setdefault("WORLD_SIZE", "1")
    from ptqeval.inference._env import ensure_wan_va_on_path
    ensure_wan_va_on_path()
    from distributed.util import init_distributed
    init_distributed(world_size=1, local_rank=local_rank, rank=0)


class InferenceEngine:
    """Owns the model + KV cache + current prompt embeds. One per device."""

    def __init__(self, config: InferenceConfig) -> None:
        from ptqeval.wam.lingbot_va.server import VA_Server
        self.config = config
        job_config = config.to_job_config()
        _ensure_single_process_dist(job_config.local_rank)
        self.server = VA_Server(job_config)

    def infer(self, request: dict) -> dict:
        """Forward a request dict to VA_Server.infer. Branches on keys:
        {'reset': True, 'prompt': str} -> reset;
        {'compute_kv_cache': True, 'obs': ...} -> advance KV cache;
        {'obs': ..., 'prompt': ...} -> one action chunk (returns
        {'action': ndarray})."""
        return self.server.infer(request)

    def reset(self, prompt: Optional[str] = None) -> None:
        """Convenience: reset the server for a new episode/prompt (injects
        the cached text embeds, allocates the KV buffer)."""
        self.server.infer({"reset": True, "prompt": prompt})

    def compute_kv_cache(self, obs: dict) -> None:
        """Convenience: advance the sliding-window KV cache with keyframe
        obs (passthrough of the eval-time compute_kv_cache call)."""
        req = dict(obs)
        req["compute_kv_cache"] = True
        self.server.infer(req)
