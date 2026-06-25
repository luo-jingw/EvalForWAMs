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
    # KV cache occupancy across all stage records in this task's run.
    # `kv_total_slots` is fixed (= attn_window-derived container size);
    # `mean_kv_filled_slots` / `max_kv_filled_slots` show how the actual
    # sliding-window fill varies during real eval.
    mean_kv_filled_slots: Optional[float] = None
    max_kv_filled_slots: Optional[int] = None
    kv_total_slots: Optional[int] = None


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
    mean_kv_filled_slots: Optional[float] = None
    max_kv_filled_slots: Optional[int] = None
    kv_total_slots: Optional[int] = None


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
    kv_filled_list: list[int] = []
    kv_total: Optional[int] = None

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
            # KV occupancy: ignore init stage (cache not yet allocated)
            # and reset stage (mask cleared, would skew average down).
            if stage in ("transformer", "action_head", "vae_encode"):
                kv_filled = stage_rec.get("kv_filled_slots")
                kv_total_rec = stage_rec.get("kv_total_slots")
                if kv_filled is not None and kv_total_rec is not None:
                    kv_filled_list.append(int(kv_filled))
                    if kv_total is None:
                        kv_total = int(kv_total_rec)
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
        mean_kv_filled_slots=(statistics.fmean(kv_filled_list)
                              if kv_filled_list else None),
        max_kv_filled_slots=(max(kv_filled_list)
                             if kv_filled_list else None),
        kv_total_slots=kv_total,
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
            mean_kv_filled_slots=p.mean_kv_filled_slots if p else None,
            max_kv_filled_slots=p.max_kv_filled_slots if p else None,
            kv_total_slots=p.kv_total_slots if p else None,
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
            "mean_kv_filled_slots", "max_kv_filled_slots", "kv_total_slots",
        ])
        for r in rows:
            writer.writerow([
                r.task_name, r.num_episodes, r.success_rate,
                r.init_peak_mb, r.mean_total_ms, r.p50_total_ms, r.p95_total_ms,
                r.mean_transformer_ms, r.mean_action_head_ms,
                r.peak_alloc_mb, r.peak_reserved_mb,
                r.mean_kv_filled_slots, r.max_kv_filled_slots, r.kv_total_slots,
            ])


def write_json(rows: list[SummaryRow], perf: dict[str, TaskPerf], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "summary": [asdict(r) for r in rows],
        "per_stage_breakdown": {name: asdict(p) for name, p in perf.items()},
    }
    with open(path, "w") as fp:
        json.dump(payload, fp, indent=2)


_OTHER_SUBCAT_PATTERNS = (
    # (label, substring-match list against kernel name; first match wins)
    ("norm",       ("layer_norm", "rms_norm", "native_batch_norm",
                    "native_layer_norm")),
    ("softmax",    ("softmax_warp_forward", "_softmax", "Softmax")),
    ("reduction",  ("reduce_kernel", "ReduceOp", "sum_kernel",
                    "mean_kernel", "max_kernel", "min_kernel")),
    ("index_gather", ("index_elementwise_kernel", "IndexKernel",
                     "gather", "scatter", "index_select")),
    ("copy",       ("direct_copy_kernel_cuda", "_copy_kernel",
                    "CopyKernel", "fill_kernel")),
    ("elementwise", ("elementwise_kernel", "gpu_kernel_impl",
                     "binary_internal", "unary_internal")),
)

# One-line description per sub-cat for figure legends / report blurbs.
# Embedded in the merged op_profile.json so downstream chart renderers
# can show it without hardcoding the list. `beyond_top50` is the
# residual bucket: per-task profilers cap at the top-50 most-expensive
# kernels by self_cuda_time, so any kernel below rank 50 contributes
# to `op_per_call_ms.other` but cannot be sub-attributed.
_OTHER_SUBCAT_DOC = {
    "elementwise":  "pointwise math (add / mul / sub / silu / gelu / cast)",
    "copy":         "direct_copy_kernel_cuda (residual saves, cache writes)",
    "index_gather": "indexing / gather / scatter / index_select",
    "norm":         "layer_norm / rms_norm forward",
    "softmax":      "attention softmax kernels",
    "reduction":    "sum / mean / max reduction kernels",
    "other_misc":   "matched 'other' but no sub-cat pattern hit",
    "beyond_top50": ("kernels ranked > 50 per task (each <1 ms but many; "
                     "names truncated at profile time, only the total ms "
                     "is recoverable)"),
}


def _classify_other_subcat(name: str) -> str:
    """Sub-classify a kernel previously bucketed as 'other' by matching
    its mangled CUDA name against a list of substring patterns. Returns
    'other_misc' if no pattern matches (preserves the catch-all)."""
    for label, pats in _OTHER_SUBCAT_PATTERNS:
        for p in pats:
            if p in name:
                return label
        # Fast path: exact-prefix match via the label itself (covers
        # short kernel names without templates).
    return "other_misc"


def merge_op_profiles(perf_log_dir: str, out_dir: str) -> None:
    """Merge per-task <task>_op_profile.json files (written by server
    when --profile_ops is on) into a single <out_dir>/op_profile.json
    containing the mean op-share across tasks. Also emits
    `_other_subcats` decomposing the 'other' bucket by kernel-name
    pattern when per-task files carry `_per_kernel_top50`; older
    per-task files without that field still aggregate the 4-class
    `op_per_call_ms` unchanged. No-op if no per-task file exists."""
    import glob
    files = sorted(glob.glob(os.path.join(perf_log_dir, "*_op_profile.json")))
    if not files:
        return
    per_call_ms = {"linear": 0.0, "attention": 0.0,
                    "memcpy": 0.0, "other": 0.0}
    # Sub-decomposition of 'other' by kernel-name pattern. Same per-call
    # ms semantics: sum across tasks, divide by n at the end.
    other_subcats: dict[str, float] = {}
    n = 0
    sources = []
    for f in files:
        try:
            with open(f) as fh:
                p = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        slot = p.get("op_per_call_ms")
        if not slot:
            continue
        # Accept legacy 3-key profiles (linear/attention/other only)
        # by treating their `other` as `other` and leaving memcpy at 0.
        for k in per_call_ms:
            per_call_ms[k] += float(slot.get(k, 0.0))
        # Per-task n_calls divisor used in server.py:
        #     per_call_ms[cat] = sum(self_cuda_us[cat]) / 1000 / n_calls
        # so `_per_kernel_top50` entries are *summed cuda_us across all
        # profiled calls*. Divide by n_calls to get per-call us.
        n_calls = max(1, int(p.get("_meta", {}).get("n_calls", 1)))
        for kr in p.get("_per_kernel_top50", []):
            if kr.get("cat") != "other":
                continue
            sub = _classify_other_subcat(kr.get("name", ""))
            per_call_us = float(kr.get("self_cuda_us", 0.0)) / n_calls
            other_subcats[sub] = other_subcats.get(sub, 0.0) + per_call_us / 1000.0
        n += 1
        sources.append(os.path.basename(f))
    if n == 0:
        return
    per_call_ms = {k: v / n for k, v in per_call_ms.items()}
    other_subcats = {k: v / n for k, v in other_subcats.items()}
    out_path = os.path.join(out_dir, "op_profile.json")
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "_meta": {
            "unit": "ms",
            "source": "torch.profiler (per-task averaged)",
            "n_tasks": n,
            "task_files": sources,
        },
        "op_per_call_ms": per_call_ms,
    }
    # Only emit the sub-decomposition when at least one per-task file
    # actually carried _per_kernel_top50 (otherwise the dict is empty
    # and the consumer would print confusing zero rows). Verified
    # _other_subcats covers only the 'other' bucket — sum should
    # approximately equal per_call_ms['other'] modulo the top-50 cap
    # per task (kernels outside top50 stay in `other` headline but
    # don't get sub-attributed; this gap is the implicit
    # 'beyond_top50' sub-bucket which we expose explicitly).
    if other_subcats:
        attributed = sum(other_subcats.values())
        gap = max(0.0, per_call_ms["other"] - attributed)
        if gap > 0.01:
            other_subcats["beyond_top50"] = gap
        payload["_other_subcats"] = dict(sorted(other_subcats.items(),
                                                 key=lambda kv: -kv[1]))
        # Only ship docs for sub-cats actually present (keeps the JSON
        # tight; downstream renderer can show them as legend tooltips
        # or figure footer).
        payload["_other_subcats_doc"] = {
            k: _OTHER_SUBCAT_DOC[k]
            for k in other_subcats if k in _OTHER_SUBCAT_DOC
        }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {out_path} (averaged across {n} tasks)")


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
    # Merge optional op_profile artifacts if any server ran with
    # --profile_ops; harmless when absent.
    merge_op_profiles(perf_log_dir, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_root", required=True)
    parser.add_argument("--perf_log_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    main(args.save_root, args.perf_log_dir, args.out_dir)
