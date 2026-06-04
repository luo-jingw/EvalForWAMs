# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Aggregator: merge perf JSONL records with RoboTwin SR results.

Inputs
- perf_log_dir : directory with one JSONL per server process.
  Each line is a CallRecord dict written by PerfProbe.
- save_root    : RoboTwin client output root. SR lives in
  <save_root>/stseed-<seed>/metrics/<task_name>/res.json.

Outputs
- <out_dir>/summary.csv
- <out_dir>/summary.json

Schema (one row per task)
  task_name, num_episodes, success_rate,
  init_peak_mb, mean_total_ms, p50_total_ms, p95_total_ms,
  mean_transformer_ms, mean_action_head_ms, peak_alloc_mb, peak_reserved_mb
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import statistics
from dataclasses import asdict, dataclass, field
from typing import Optional


# Filename convention written by PerfProbe:
#   <task_name>_rank<N>_<YYYYMMDD_HHMMSS>.jsonl
_PERF_FILENAME_RE = re.compile(r"^(.+)_rank\d+_\d{8}_\d{6}\.jsonl$")


def _task_from_perf_filename(fname: str) -> Optional[str]:
    m = _PERF_FILENAME_RE.match(fname)
    return m.group(1) if m else None


_KNOWN_STAGES: tuple[str, ...] = (
    "init",
    "text_encoder",
    "vae_encode",
    "transformer",
    "action_head",
)


@dataclass
class TaskSR:
    task_name: str
    succ_num: float
    total_num: float
    succ_rate: float


@dataclass
class TaskPerf:
    task_name: str
    num_calls: int
    init_peak_mb: Optional[float]
    mean_total_ms: Optional[float]
    p50_total_ms: Optional[float]
    p95_total_ms: Optional[float]
    per_stage_mean_ms: dict[str, float] = field(default_factory=dict)
    per_stage_peak_alloc_mb: dict[str, float] = field(default_factory=dict)
    per_stage_peak_reserved_mb: dict[str, float] = field(default_factory=dict)
    overall_peak_alloc_mb: Optional[float] = None
    overall_peak_reserved_mb: Optional[float] = None


@dataclass
class SummaryRow:
    task_name: str
    num_episodes: float
    success_rate: float
    init_peak_mb: Optional[float]
    mean_total_ms: Optional[float]
    p50_total_ms: Optional[float]
    p95_total_ms: Optional[float]
    mean_transformer_ms: Optional[float]
    mean_action_head_ms: Optional[float]
    peak_alloc_mb: Optional[float]
    peak_reserved_mb: Optional[float]


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    k = (len(sorted_v) - 1) * pct
    f_floor = int(k)
    f_ceil = min(f_floor + 1, len(sorted_v) - 1)
    return sorted_v[f_floor] + (sorted_v[f_ceil] - sorted_v[f_floor]) * (k - f_floor)


def load_perf_logs(perf_log_dir: str) -> dict[str, list[dict]]:
    """Per task, keep only the latest jsonl by mtime. Older jsonl files from
    interrupted or OOM-aborted prior runs are ignored (preserved on disk)."""
    by_task: dict[str, tuple[float, str]] = {}
    skipped: list[str] = []
    for path in glob.glob(os.path.join(perf_log_dir, "*.jsonl")):
        task = _task_from_perf_filename(os.path.basename(path))
        if task is None:
            continue
        mtime = os.path.getmtime(path)
        cur = by_task.get(task)
        if cur is None or mtime > cur[0]:
            if cur is not None:
                skipped.append(cur[1])
            by_task[task] = (mtime, path)
        else:
            skipped.append(path)
    for stale in sorted(skipped):
        print(f"[load_perf_logs] skipping stale jsonl: {stale}")

    grouped: dict[str, list[dict]] = {}
    for _, path in sorted(by_task.values()):
        with open(path, "r") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                raw_name: str = record.get("task_name", "unknown")
                task = _normalize_task_name(raw_name)
                grouped.setdefault(task, []).append(record)
    return grouped


def _normalize_task_name(raw: str) -> str:
    """Strip per-call prefixes added by the server (reset:, infer:, kv:)."""
    for prefix in ("reset:", "infer:", "kv:"):
        if raw.startswith(prefix):
            return raw[len(prefix):]
    if raw == "_init":
        return "_init"
    return raw


def compute_task_perf(task_name: str, records: list[dict]) -> TaskPerf:
    init_peak_mb: Optional[float] = None
    total_ms_list: list[float] = []
    stage_ms: dict[str, list[float]] = {s: [] for s in _KNOWN_STAGES}
    stage_alloc: dict[str, list[float]] = {s: [] for s in _KNOWN_STAGES}
    stage_reserved: dict[str, list[float]] = {s: [] for s in _KNOWN_STAGES}

    for rec in records:
        for stage_rec in rec.get("stages", []):
            stage = stage_rec.get("stage")
            if stage not in stage_ms:
                continue
            stage_ms[stage].append(float(stage_rec.get("elapsed_ms", 0.0)))
            stage_alloc[stage].append(float(stage_rec.get("peak_alloc_mb", 0.0)))
            stage_reserved[stage].append(float(stage_rec.get("peak_reserved_mb", 0.0)))
            if stage == "init" and init_peak_mb is None:
                init_peak_mb = float(stage_rec.get("peak_alloc_mb", 0.0))
        if any(s.get("stage") != "init" for s in rec.get("stages", [])):
            total_ms_list.append(float(rec.get("total_ms", 0.0)))

    per_stage_mean = {
        s: statistics.fmean(v) for s, v in stage_ms.items() if v
    }
    per_stage_alloc = {
        s: max(v) for s, v in stage_alloc.items() if v
    }
    per_stage_reserved = {
        s: max(v) for s, v in stage_reserved.items() if v
    }
    overall_peak_alloc = max(per_stage_alloc.values()) if per_stage_alloc else None
    overall_peak_reserved = max(per_stage_reserved.values()) if per_stage_reserved else None

    return TaskPerf(
        task_name=task_name,
        num_calls=len(total_ms_list),
        init_peak_mb=init_peak_mb,
        mean_total_ms=statistics.fmean(total_ms_list) if total_ms_list else None,
        p50_total_ms=_percentile(total_ms_list, 0.50) if total_ms_list else None,
        p95_total_ms=_percentile(total_ms_list, 0.95) if total_ms_list else None,
        per_stage_mean_ms=per_stage_mean,
        per_stage_peak_alloc_mb=per_stage_alloc,
        per_stage_peak_reserved_mb=per_stage_reserved,
        overall_peak_alloc_mb=overall_peak_alloc,
        overall_peak_reserved_mb=overall_peak_reserved,
    )


def load_robotwin_sr(save_root: str) -> dict[str, TaskSR]:
    """Walks save_root for stseed-*/metrics/<task>/res.json files."""
    results: dict[str, TaskSR] = {}
    pattern = os.path.join(save_root, "stseed-*", "metrics", "*", "res.json")
    for path in glob.glob(pattern):
        task_name = os.path.basename(os.path.dirname(path))
        with open(path, "r") as fp:
            data = json.load(fp)
        existing = results.get(task_name)
        if existing is None or float(data.get("total_num", 0)) > existing.total_num:
            results[task_name] = TaskSR(
                task_name=task_name,
                succ_num=float(data.get("succ_num", 0.0)),
                total_num=float(data.get("total_num", 0.0)),
                succ_rate=float(data.get("succ_rate", 0.0)),
            )
    return results


def build_summary(perf: dict[str, TaskPerf], sr: dict[str, TaskSR]) -> list[SummaryRow]:
    rows: list[SummaryRow] = []
    task_names = sorted(set(perf.keys()) | set(sr.keys()))
    for name in task_names:
        if name == "_init":
            continue
        p = perf.get(name)
        s = sr.get(name)
        rows.append(SummaryRow(
            task_name=name,
            num_episodes=s.total_num if s else 0.0,
            success_rate=s.succ_rate if s else float("nan"),
            init_peak_mb=p.init_peak_mb if p else None,
            mean_total_ms=p.mean_total_ms if p else None,
            p50_total_ms=p.p50_total_ms if p else None,
            p95_total_ms=p.p95_total_ms if p else None,
            mean_transformer_ms=p.per_stage_mean_ms.get("transformer") if p else None,
            mean_action_head_ms=p.per_stage_mean_ms.get("action_head") if p else None,
            peak_alloc_mb=p.overall_peak_alloc_mb if p else None,
            peak_reserved_mb=p.overall_peak_reserved_mb if p else None,
        ))
    return rows


def write_csv(rows: list[SummaryRow], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow([
            "task_name", "num_episodes", "success_rate",
            "init_peak_mb", "mean_total_ms", "p50_total_ms", "p95_total_ms",
            "mean_transformer_ms", "mean_action_head_ms",
            "peak_alloc_mb", "peak_reserved_mb",
        ])
        for r in rows:
            writer.writerow([
                r.task_name, r.num_episodes, r.success_rate,
                r.init_peak_mb, r.mean_total_ms, r.p50_total_ms, r.p95_total_ms,
                r.mean_transformer_ms, r.mean_action_head_ms,
                r.peak_alloc_mb, r.peak_reserved_mb,
            ])


def write_json(rows: list[SummaryRow], perf: dict[str, TaskPerf], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "summary": [asdict(r) for r in rows],
        "per_stage_breakdown": {name: asdict(p) for name, p in perf.items()},
    }
    with open(path, "w") as fp:
        json.dump(payload, fp, indent=2)


def main(save_root: str, perf_log_dir: str, out_dir: str) -> None:
    perf_records = load_perf_logs(perf_log_dir)
    perf_stats: dict[str, TaskPerf] = {
        name: compute_task_perf(name, recs) for name, recs in perf_records.items()
    }
    sr_stats = load_robotwin_sr(save_root)
    rows = build_summary(perf_stats, sr_stats)
    csv_path = os.path.join(out_dir, "summary.csv")
    json_path = os.path.join(out_dir, "summary.json")
    write_csv(rows, csv_path)
    write_json(rows, perf_stats, json_path)
    print(f"wrote {csv_path} ({len(rows)} rows)")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_root", required=True)
    parser.add_argument("--perf_log_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    main(args.save_root, args.perf_log_dir, args.out_dir)
