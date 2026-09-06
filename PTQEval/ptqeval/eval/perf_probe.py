# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Per-stage CUDA latency and peak memory probe for LingBot-VA inference.

One PerfProbe instance owns one JSONL log file. The probe wraps inference
stages via context manager. CUDA Events measure GPU time. Peak memory is
read from torch.cuda.max_memory_allocated and max_memory_reserved.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

import torch


_BYTES_PER_MB: float = 1024.0 * 1024.0


@dataclass
class StageRecord:
    stage: str
    elapsed_ms: float
    peak_alloc_mb: float
    peak_reserved_mb: float
    # KV cache occupancy snapshot at stage exit (queried via
    # PerfProbe.kv_introspect; None if probe was not given an introspect
    # callable). filled / total are slot counts inside the K/V mask
    # tensor for the first transformer block (representative — all
    # blocks share the same allocation pattern). Used by aggregator +
    # cross_ckpt to render per-task KV cache occupancy chart.
    kv_filled_slots: Optional[int] = None
    kv_total_slots: Optional[int] = None


@dataclass
class CallRecord:
    call_id: int
    task_name: str
    timestamp: str
    total_ms: float
    stages: list[StageRecord] = field(default_factory=list)


class StageContext:
    """Context manager returned by PerfProbe.stage(). Times one stage."""

    def __init__(self, probe: "PerfProbe", name: str) -> None:
        self._probe: PerfProbe = probe
        self._name: str = name
        self._start_event: Optional[torch.cuda.Event] = None
        self._end_event: Optional[torch.cuda.Event] = None

    def __enter__(self) -> "StageContext":
        torch.cuda.synchronize(self._probe.device)
        torch.cuda.reset_peak_memory_stats(self._probe.device)
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._end_event = torch.cuda.Event(enable_timing=True)
        self._start_event.record()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        assert self._start_event is not None
        assert self._end_event is not None
        self._end_event.record()
        torch.cuda.synchronize(self._probe.device)
        elapsed_ms: float = self._start_event.elapsed_time(self._end_event)
        peak_alloc_mb: float = (
            torch.cuda.max_memory_allocated(self._probe.device) / _BYTES_PER_MB
        )
        peak_reserved_mb: float = (
            torch.cuda.max_memory_reserved(self._probe.device) / _BYTES_PER_MB
        )
        # Optional KV cache fill snapshot. Callable returns (filled, total)
        # slot counts; None if probe was not given an introspect callable.
        kv_filled: Optional[int] = None
        kv_total: Optional[int] = None
        if self._probe.kv_introspect is not None:
            try:
                kv_filled, kv_total = self._probe.kv_introspect()
            except Exception:  # noqa: BLE001
                # Don't let introspect bugs break the eval timing path.
                kv_filled, kv_total = None, None
        record = StageRecord(
            stage=self._name,
            elapsed_ms=elapsed_ms,
            peak_alloc_mb=peak_alloc_mb,
            peak_reserved_mb=peak_reserved_mb,
            kv_filled_slots=kv_filled,
            kv_total_slots=kv_total,
        )
        self._probe.append_stage(record)


class PerfProbe:
    """JSONL-backed per-call perf probe. One probe per server process."""

    def __init__(self, log_path: str, task_name: str, device: int) -> None:
        self.log_path: str = log_path
        self.task_name: str = task_name
        self.device: int = device
        self._call_id: int = 0
        self._call_start_ms: Optional[float] = None
        self._stages: list[StageRecord] = []
        # KV cache introspect callable: () -> (filled_slots, total_slots).
        # Set by server.py after the model is loaded. None disables
        # kv_filled_slots / kv_total_slots fields on StageRecord.
        self.kv_introspect: Optional[Callable[[], tuple[int, int]]] = None
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._fp = open(log_path, "a", buffering=1)

    def begin_call(self) -> None:
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self._stages = []
        self._call_start_ms = time.perf_counter() * 1000.0

    def stage(self, name: str) -> StageContext:
        return StageContext(self, name)

    def append_stage(self, record: StageRecord) -> None:
        self._stages.append(record)

    def end_call(self) -> CallRecord:
        assert self._call_start_ms is not None, "begin_call must precede end_call"
        total_ms: float = time.perf_counter() * 1000.0 - self._call_start_ms
        record = CallRecord(
            call_id=self._call_id,
            task_name=self.task_name,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            total_ms=total_ms,
            stages=list(self._stages),
        )
        self._fp.write(json.dumps(asdict(record)) + "\n")
        self._call_id += 1
        self._call_start_ms = None
        self._stages = []
        return record

    def close(self) -> None:
        if not self._fp.closed:
            self._fp.close()
