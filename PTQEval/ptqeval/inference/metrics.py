# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Optional lightweight observation wrapper for the inference engine (M4).

Deployment cares about at most two numbers: overall peak VRAM and
per-infer wall time. This wraps an InferenceEngine and records them.
Observational only (principle.txt L12): it returns measured values, makes
no pass/fail judgement and no threshold check.
"""
from __future__ import annotations

import time
from typing import Optional

from ptqeval.inference.engine import InferenceEngine


class MeasuredEngine:
    """Transparent wrapper: same infer/reset/compute_kv_cache surface as
    InferenceEngine, plus peak-VRAM and last-infer-time getters."""

    def __init__(self, engine: InferenceEngine) -> None:
        self.engine = engine
        self._device = engine.config.device
        self._last_infer_ms: float = float("nan")
        self.reset_peak()

    def _torch(self):
        import torch
        return torch

    def reset_peak(self) -> None:
        torch = self._torch()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self._device)

    def peak_alloc_mb(self) -> float:
        """Overall peak allocated VRAM (MB) since the last reset_peak."""
        torch = self._torch()
        if not torch.cuda.is_available():
            return float("nan")
        return torch.cuda.max_memory_allocated(self._device) / (1024.0 * 1024.0)

    def last_infer_ms(self) -> float:
        """Wall time (ms) of the last infer() one-chunk call."""
        return self._last_infer_ms

    def infer(self, request: dict) -> dict:
        torch = self._torch()
        is_chunk = not (request.get("reset") or request.get("compute_kv_cache"))
        if is_chunk and torch.cuda.is_available():
            torch.cuda.synchronize(self._device)
        t0 = time.perf_counter()
        out = self.engine.infer(request)
        if is_chunk:
            if torch.cuda.is_available():
                torch.cuda.synchronize(self._device)
            self._last_infer_ms = (time.perf_counter() - t0) * 1000.0
        return out

    def reset(self, prompt: Optional[str] = None) -> None:
        self.engine.reset(prompt)

    def compute_kv_cache(self, obs: dict) -> None:
        self.engine.compute_kv_cache(obs)
