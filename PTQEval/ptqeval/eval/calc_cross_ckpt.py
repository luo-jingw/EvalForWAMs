# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Cross-checkpoint aggregator: compare per-task perf + SR across variants.

Joins multiple per-variant summary.csv (each produced by
ptqeval.eval.aggregator) on task_name. First --variant is treated as the
baseline; subsequent variants are compared against it.

Outputs to --out_dir:
    cross_summary.csv    flat table with per-variant columns + delta columns
    cross_summary.json   structured payload (variants meta + per-task rows)
    report.md            English markdown report (headline, per-task, aggregates)

Usage:
    python -m ptqeval.eval.calc_cross_ckpt \\
        --variant bf16=results/bf16/summary/summary.csv \\
        --variant viditq_w8a8=results/viditq_runtime_args_w8a8/summary/summary.csv \\
        --out_dir results/cross_summary
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from dataclasses import asdict, dataclass


# Float-valued fields in summary.csv (everything except task_name).
_FLOAT_FIELDS = (
    "num_episodes",
    "success_rate",
    "init_peak_mb",
    "mean_total_ms",
    "p50_total_ms",
    "p95_total_ms",
    "mean_transformer_ms",
    "mean_action_head_ms",
    "peak_alloc_mb",
    "peak_reserved_mb",
)


@dataclass
class SummaryRow:
    task_name: str
    num_episodes: float
    success_rate: float
    init_peak_mb: float
    mean_total_ms: float
    p50_total_ms: float
    p95_total_ms: float
    mean_transformer_ms: float
    mean_action_head_ms: float
    peak_alloc_mb: float
    peak_reserved_mb: float


def load_summary(csv_path: str) -> dict[str, SummaryRow]:
    rows: dict[str, SummaryRow] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            data = {"task_name": raw["task_name"]}
            for k in _FLOAT_FIELDS:
                data[k] = float(raw[k]) if raw.get(k) not in (None, "") else float("nan")
            rows[raw["task_name"]] = SummaryRow(**data)
    return rows


def _inspect_quant_scope(int_weights_pth_path: str | None) -> dict | None:
    """Read an int_weights.pth produced by `ptq.py` and classify the
    quantized Linears into runtime-active (attn1 self-attn + ffn) vs
    cross-attn FP (attn2, present in state_dict but not loaded by the
    wrapper) vs other. Returns per-category Linear count + param count
    (summed over int_weight.numel() to match the original FP weight
    parameter count, unaffected by int8/int4 storage)."""
    if not int_weights_pth_path or not os.path.exists(int_weights_pth_path):
        return None
    try:
        import torch
    except ImportError:
        return None
    sd = torch.load(int_weights_pth_path, map_location="cpu", weights_only=True)
    out = {
        "ckpt_path": int_weights_pth_path,
        "active": {"n_linears": 0, "params": 0},      # attn1 + ffn, loaded by wrapper
        "cross_attn_fp": {"n_linears": 0, "params": 0},  # attn2, runtime FP
        "other_quantized": {"n_linears": 0, "params": 0},
    }
    for k, v in sd.items():
        if not k.endswith(".int_weight"):
            continue
        name = k[:-len(".int_weight")]
        n = int(v.numel())
        # For W4A8 the stored int_weight has C_in halved (nibble packing).
        # If we ever cross-summary mix W4 and W8, multiply by 2 here when
        # we detect packing. Phase 27 (this report) is W8A8 only.
        if ".attn2." in name:
            out["cross_attn_fp"]["n_linears"] += 1
            out["cross_attn_fp"]["params"] += n
        elif ".attn1." in name or ".ffn." in name:
            out["active"]["n_linears"] += 1
            out["active"]["params"] += n
        else:
            out["other_quantized"]["n_linears"] += 1
            out["other_quantized"]["params"] += n
    return out


def _load_step_limits(path: str | None) -> dict[str, int]:
    """Parse RoboTwin's task_config/_eval_step_limit.yml (simple
    `task_name: int` per line). Returns an empty dict if the path is
    missing or unreadable; downstream code then omits the step column."""
    if not path or not os.path.exists(path):
        return {}
    limits: dict[str, int] = {}
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            k, v = line.split(":", 1)
            try:
                limits[k.strip()] = int(v.strip())
            except ValueError:
                continue
    return limits


def _parse_variant_arg(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"--variant must be TAG=PATH, got {spec!r}")
    tag, path = spec.split("=", 1)
    if not tag or not path:
        raise ValueError(f"--variant must be TAG=PATH (non-empty), got {spec!r}")
    return tag, path


def _ratio(num: float, denom: float) -> float:
    return num / denom if denom != 0 else float("nan")


# --------------------------------------------------------------------------
# CSV emission
# --------------------------------------------------------------------------

def write_cross_csv(
    out_path: str,
    tags: list[str],
    baseline_tag: str,
    tasks: list[str],
    summaries: dict[str, dict[str, SummaryRow]],
) -> None:
    metric_fields = (
        "success_rate",
        "mean_total_ms",
        "mean_transformer_ms",
        "peak_alloc_mb",
    )
    header = ["task_name", "num_episodes"]
    for t in tags:
        for m in metric_fields:
            header.append(f"{m}__{t}")
    for t in tags:
        if t == baseline_tag:
            continue
        header.append(f"success_rate_delta__{t}_vs_{baseline_tag}")
        header.append(f"mean_total_ms_ratio__{t}_vs_{baseline_tag}")
        header.append(f"mean_transformer_ms_ratio__{t}_vs_{baseline_tag}")
        header.append(f"peak_alloc_mb_savings__{baseline_tag}_minus_{t}")

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for task in tasks:
            base_row = summaries[baseline_tag][task]
            line: list = [task, int(base_row.num_episodes)]
            for t in tags:
                r = summaries[t][task]
                line.append(f"{r.success_rate:.4f}")
                line.append(f"{r.mean_total_ms:.2f}")
                line.append(f"{r.mean_transformer_ms:.2f}")
                line.append(f"{r.peak_alloc_mb:.2f}")
            for t in tags:
                if t == baseline_tag:
                    continue
                r = summaries[t][task]
                line.append(f"{r.success_rate - base_row.success_rate:+.4f}")
                line.append(f"{_ratio(r.mean_total_ms, base_row.mean_total_ms):.4f}")
                line.append(f"{_ratio(r.mean_transformer_ms, base_row.mean_transformer_ms):.4f}")
                line.append(f"{base_row.peak_alloc_mb - r.peak_alloc_mb:+.2f}")
            w.writerow(line)


# --------------------------------------------------------------------------
# JSON emission
# --------------------------------------------------------------------------

def write_cross_json(
    out_path: str,
    variant_paths: list[tuple[str, str]],
    baseline_tag: str,
    tasks: list[str],
    summaries: dict[str, dict[str, SummaryRow]],
    aggregates: dict[str, dict[str, float]],
    step_limits: dict[str, int],
) -> None:
    payload = {
        "baseline": baseline_tag,
        "variants": [{"tag": t, "summary_csv": p} for t, p in variant_paths],
        "aggregates": aggregates,
        "per_task": [],
    }
    for task in tasks:
        row = {"task_name": task,
               "num_episodes": int(summaries[baseline_tag][task].num_episodes)}
        if task in step_limits:
            row["step_limit"] = step_limits[task]
        for tag in [t for (t, _) in variant_paths]:
            row[tag] = {
                k: getattr(summaries[tag][task], k)
                for k in _FLOAT_FIELDS
            }
        payload["per_task"].append(row)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


# --------------------------------------------------------------------------
# Aggregates
# --------------------------------------------------------------------------

def compute_aggregates(
    tags: list[str],
    tasks: list[str],
    summaries: dict[str, dict[str, SummaryRow]],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for t in tags:
        srs    = [summaries[t][task].success_rate       for task in tasks]
        totals = [summaries[t][task].mean_total_ms      for task in tasks]
        trans  = [summaries[t][task].mean_transformer_ms for task in tasks]
        peaks  = [summaries[t][task].peak_alloc_mb      for task in tasks]
        inits  = [summaries[t][task].init_peak_mb       for task in tasks]
        out[t] = {
            "mean_success_rate":         statistics.mean(srs),
            "median_success_rate":       statistics.median(srs),
            "min_success_rate":          min(srs),
            "max_success_rate":          max(srs),
            "mean_total_ms":             statistics.mean(totals),
            "median_total_ms":           statistics.median(totals),
            "mean_transformer_ms":       statistics.mean(trans),
            "mean_peak_alloc_mb":        statistics.mean(peaks),
            "mean_init_peak_mb":         statistics.mean(inits),
        }
    return out


# --------------------------------------------------------------------------
# Markdown report
# --------------------------------------------------------------------------

def _fmt_pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _fmt_sr_delta(d: float) -> str:
    if d > 0:
        return f"+{d * 100:.1f}pp"
    if d < 0:
        return f"{d * 100:.1f}pp"
    return "+0.0pp"


def _fmt_ms(v: float) -> str:
    return f"{v:.0f} ms"


def _fmt_mb(v: float) -> str:
    return f"{v:.0f} MB"


def _fmt_gb(v: float) -> str:
    return f"{v / 1024:.2f} GB"


def _make_plots(
    out_dir: str,
    variant_paths: list[tuple[str, str]],
    baseline_tag: str,
    tasks: list[str],
    summaries: dict[str, dict[str, SummaryRow]],
    step_limits: dict[str, int],
    op_data: dict[str, dict[str, float]] | None = None,
    measured_flops_tf: float | None = None,
    measured_kv_bytes: float | None = None,
    measured_act_bytes: float | None = None,
) -> list[tuple[str, str]]:
    """Renders charts under <out_dir>/plots/:
      1. sr_by_task.png           per-task SR per variant
      2. total_ms_speedup.png     latency + speedup curve per task
      3. speedup_by_task.png      speedup ratio per task (omitted when single variant)
      4. latency_distribution.png mean / p50 / p95 per task
      5. roofline.png             achieved (AI, throughput) vs A6000 ceilings
                                  (requires --measured_flops; skipped otherwise)
      6. memory_breakdown.png     VRAM peak = model weights + transient
    Main also writes op_breakdown_measured.png when --op_profile entries
    are supplied. Returns [(title, report-relative path)] pairs for
    embedding in report.md. Empty list if matplotlib is missing."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("warning: matplotlib not available; skipping plots.")
        return []

    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    tags = [t for t, _ in variant_paths]
    other_tags = [t for t in tags if t != baseline_tag]

    sorted_tasks = (
        sorted(tasks, key=lambda t: (step_limits.get(t, 10**9), t))
        if step_limits else list(tasks)
    )
    n = len(sorted_tasks)
    x = list(range(n))
    x_labels = [
        (f"{t}\n({step_limits[t]})" if t in step_limits else t)
        for t in sorted_tasks
    ]
    fig_w = max(10.0, n * 0.85)

    charts: list[tuple[str, str]] = []

    # ------ Chart 1: SR per task per variant ------
    fig, ax = plt.subplots(figsize=(fig_w, 5.4))
    bw = 0.8 / max(1, len(tags))
    for i, tag in enumerate(tags):
        srs = [summaries[tag][t].success_rate * 100.0 for t in sorted_tasks]
        offset = (i - (len(tags) - 1) / 2.0) * bw
        ax.bar([xi + offset for xi in x], srs, width=bw, label=tag)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Per-task success rate (sorted by step limit)", pad=24)
    # Legend above plot area as a single row so it doesn't occlude bars
    # in any corner (lower-left collided with adjust_bottle / beat_block
    # at 0-30%; lower-right would collide with put_bottles_dustbin).
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005),
              ncol=len(tags), frameon=False, fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p1 = os.path.join(plots_dir, "sr_by_task.png")
    fig.savefig(p1, dpi=120)
    plt.close(fig)
    charts.append(("Per-task success rate (sorted by step limit)",
                   "plots/sr_by_task.png"))

    # ------ Chart 2: total_ms bars (+ speedup line when N>=2) ------
    # In single-variant mode, the speedup subplot has nothing to draw and
    # would render as an empty pane; collapse to a single-axis figure
    # showing only the per-task latency bars.
    if other_tags:
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(fig_w, 7.5),
            gridspec_kw={"height_ratios": [2, 1]}, sharex=True)
    else:
        fig, ax_top = plt.subplots(figsize=(fig_w, 5.0))
        ax_bot = None
    for i, tag in enumerate(tags):
        ms = [summaries[tag][t].mean_total_ms for t in sorted_tasks]
        offset = (i - (len(tags) - 1) / 2.0) * bw
        ax_top.bar([xi + offset for xi in x], ms, width=bw, label=tag)
    ax_top.set_ylabel("Mean total ms / call")
    if other_tags:
        ax_top.set_title("Latency and speedup vs task (sorted by step limit)")
    else:
        ax_top.set_title("Per-task mean latency (sorted by step limit)")
        ax_top.set_xticks(x)
        ax_top.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax_top.legend(loc="upper right")
    ax_top.grid(True, axis="y", alpha=0.3)

    base_ms = [summaries[baseline_tag][t].mean_total_ms for t in sorted_tasks]
    if ax_bot is not None:
        for tag in other_tags:
            var_ms = [summaries[tag][t].mean_total_ms for t in sorted_tasks]
            speedup = [
                (b / v if v > 0 else float("nan")) for b, v in zip(base_ms, var_ms)
            ]
            ax_bot.plot(x, speedup, marker="o", linewidth=1.5,
                        label=f"{tag} vs {baseline_tag}")
        ax_bot.axhline(1.0, color="gray", linestyle="--", linewidth=0.8,
                       alpha=0.7, label="parity (1.0x)")
        ax_bot.set_xticks(x)
        ax_bot.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
        ax_bot.set_ylabel("Speedup ratio")
        ax_bot.legend(loc="upper right")
        ax_bot.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p2 = os.path.join(plots_dir, "total_ms_speedup.png")
    fig.savefig(p2, dpi=120)
    plt.close(fig)
    charts.append(
        ("Mean total ms per call + speedup ratio" if other_tags
         else "Per-task mean latency",
         "plots/total_ms_speedup.png"))

    # ------ Chart 3: Speedup per task (histogram); skip in single-variant mode ------
    if other_tags:
        fig, ax = plt.subplots(figsize=(fig_w, 5.0))
        bw3 = 0.8 / max(1, len(other_tags))
        for i, tag in enumerate(other_tags):
            var_ms = [summaries[tag][t].mean_total_ms for t in sorted_tasks]
            speedup = [
                (b / v if v > 0 else 0.0) for b, v in zip(base_ms, var_ms)
            ]
            offset = (i - (len(other_tags) - 1) / 2.0) * bw3
            ax.bar([xi + offset for xi in x], speedup, width=bw3,
                   label=f"{tag} vs {baseline_tag}")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7,
                   label="parity (1.0x)")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Speedup ratio (baseline / variant)")
        ax.set_title("Speedup ratio per task (tasks sorted by step limit)")
        ax.legend(loc="upper right")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        p3 = os.path.join(plots_dir, "speedup_by_task.png")
        fig.savefig(p3, dpi=120)
        plt.close(fig)
        charts.append(("Speedup ratio per task (sorted by step limit)",
                       "plots/speedup_by_task.png"))

    # ------ Chart 4: Latency distribution (mean / p50 / p95) ------
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))
    bw6 = 0.8 / max(1, len(tags))
    for i, tag in enumerate(tags):
        means = [summaries[tag][t].mean_total_ms for t in sorted_tasks]
        p50s  = [summaries[tag][t].p50_total_ms  for t in sorted_tasks]
        p95s  = [summaries[tag][t].p95_total_ms  for t in sorted_tasks]
        offset = (i - (len(tags) - 1) / 2.0) * bw6
        positions = [xi + offset for xi in x]
        # Vertical line from p50 to p95 (whisker).
        for pos, p50, p95 in zip(positions, p50s, p95s):
            ax.plot([pos, pos], [p50, p95], color=f"C{i}",
                    linewidth=2.5, alpha=0.85, solid_capstyle="butt")
            # Cap at p95.
            ax.plot([pos - bw6 * 0.25, pos + bw6 * 0.25], [p95, p95],
                    color=f"C{i}", linewidth=1.5)
        # Bullets for p50 (filled) and mean (open).
        ax.scatter(positions, p50s, marker="_", s=100, color=f"C{i}",
                   linewidth=2.0,
                   label=(f"{tag} p50→p95" if i == 0 else None))
        ax.scatter(positions, means, marker="o", s=45, color=f"C{i}",
                   edgecolors="white", linewidth=0.8,
                   label=f"{tag} mean", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Total ms / call (log scale)")
    ax.set_yscale("log")
    ax.set_title(
        "Latency distribution per task "
        "(circle = mean, short dash = p50, vertical bar reaches p95)"
    )
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3, which="both")
    fig.tight_layout()
    p_lat = os.path.join(plots_dir, "latency_distribution.png")
    fig.savefig(p_lat, dpi=120)
    plt.close(fig)
    charts.append(("Latency distribution (mean / p50 / p95)",
                   "plots/latency_distribution.png"))

    # ------ Chart 5: Roofline (achieved throughput vs arithmetic intensity) ------
    # Hardware: RTX A6000 (sm_86). Two compute ceilings (bf16 tensor core,
    # int8 tensor core) and one memory-bandwidth ceiling. Workload
    # parameters estimated for LingBot-VA per-call:
    #   - 5.09B params (ProjectDescription) -> weight memory dominates
    #   - per-call forward count and token count are aggregate proxies;
    #     we keep absolute FLOPs CONSTANT across variants and only vary
    #     the bytes-per-weight (bf16=2, w8a8=1) for arithmetic intensity.
    #   - achieved throughput = FLOPs / mean_(transformer + action_head)_ms
    #     since those are the stages the quant touches.
    # Variants land at different (AI, throughput) points; the diagonal
    # bandwidth line separates memory-bound (below diagonal) from
    # compute-bound (right of knee).
    # A6000 spec (NVIDIA datasheet, dense / no structured sparsity):
    #   FP32 CUDA cores       38.7 TFLOPS  <- NOT used here, common confusion
    #   BF16/FP16 Tensor Core 154.8 TFLOPS (4x FP32 since TC fuse mma)
    #   INT8 Tensor Core      309.7 TOPS   (2x BF16 since 8b vs 16b mma)
    #   GDDR6 bandwidth       768 GB/s
    # Sparse peaks would be 2x of these but our path is dense.
    PEAK_BF16_TFLOPS = 154.8    # A6000 BF16 Tensor Core dense peak
    PEAK_INT8_TOPS   = 309.7    # A6000 INT8 Tensor Core dense peak (= 2x BF16 TC)
    PEAK_BW_GBS      = 768.0    # A6000 GDDR6 bandwidth
    N_PARAMS         = 5.09e9   # LingBot-VA WAN transformer
    # FLOPs per call from FlopCounterMode dispatcher-level measurement
    # (ptqeval.eval.measure_flops). Required: roofline depends on real
    # FLOPs/call, not a hand-tuned architecture estimate. Pass
    # --measured_flops <path> to enable.
    if not (measured_flops_tf and measured_flops_tf > 0):
        print("warning: roofline chart needs --measured_flops <path> "
              "(JSON from ptqeval.eval.measure_flops); skipping.",
              file=sys.stderr)
        return charts
    flops_per_call = float(measured_flops_tf) * 1e12

    bytes_per_param = {"bf16": 2.0, "fp16": 2.0}
    # All variants other than bf16 carry W8A8 weights (int8 = 1 byte/param).
    def _weight_bytes(tag: str) -> float:
        return bytes_per_param.get(tag.lower(), 1.0)

    # Effective device-to-device bandwidth used to convert profile-measured
    # memcpy ms back into bytes. A6000 spec is 768 GB/s; the practical
    # DtoD copy bandwidth (cudaMemcpyAsync within the same device) tops
    # out around 600-700 GB/s due to allocator + scheduler overhead.
    _DTOD_BW_GBS = 600.0

    def _extra_bytes_from_profile(tag: str) -> float:
        """Bytes/call beyond raw weight load, estimated from profile
        memcpy ms. Returns 0 when op_data missing."""
        if op_data is None:
            return 0.0
        entry = op_data.get(tag)
        if not entry:
            return 0.0
        memcpy_ms = float(entry.get("memcpy", 0.0))
        return memcpy_ms * 1e-3 * _DTOD_BW_GBS * 1e9

    # KV-cache + attention I/O + inter-block activation R/W from the
    # measure_flops hook. Dtype-invariant across variants because
    # ViDiT-Q keeps the attention path + inter-block hidden_states at
    # bf16. None when measure_flops JSON predates the hook.
    measured_extra_b = 0.0
    if measured_kv_bytes is not None:
        measured_extra_b += float(measured_kv_bytes)
    if measured_act_bytes is not None:
        measured_extra_b += float(measured_act_bytes)

    # VLA-paper Fig 2 idiom (Williams roofline): one circle per
    # variant, ceilings as solid piecewise lines clipped to where each
    # bound is actually active, regions tinted with light shading,
    # numeric label sits inline next to the marker without a box,
    # caption goes under the figure. Taller figsize so the title at
    # the top and the caption at the bottom both fit without squeezing
    # the plot area.
    fig, ax = plt.subplots(figsize=(9.0, 6.8))
    ai_min, ai_max = 1.0, 1.0e5
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(ai_min, ai_max)
    y_top = max(PEAK_INT8_TOPS, PEAK_BF16_TFLOPS) * 1.8
    y_bottom = 0.5
    ax.set_ylim(y_bottom, y_top)

    bw_tbps = PEAK_BW_GBS / 1000.0
    ai_knee_bf16 = PEAK_BF16_TFLOPS / bw_tbps   # ~50 FLOPs/byte
    ai_knee_int8 = PEAK_INT8_TOPS / bw_tbps     # ~200 FLOPs/byte

    # Region shading: memory-bound (left of bf16 knee, under the BW
    # line) gets a cool tint, compute-bound (right of bf16 knee, under
    # the compute ceilings) gets a warm tint. Both transparent so
    # gridlines and markers stay readable.
    ax.axvspan(ai_min, ai_knee_bf16, color="#cfe2ff", alpha=0.30, zorder=0)
    ax.axvspan(ai_knee_bf16, ai_max, color="#fde8d0", alpha=0.30, zorder=0)

    # Solid-line roofline, three segments clipped to where each is
    # active so the diagonal only sits BELOW the horizontals and the
    # horizontals only sit TO THE RIGHT of the diagonal.
    # Diagonal (memory-BW roof) runs from ai_min up to the last knee it
    # actually caps -- the int8 knee -- and stops there.
    ai_diag = [ai_min, ai_knee_int8]
    bw_diag = [a * bw_tbps for a in ai_diag]
    ax.plot(ai_diag, bw_diag, linestyle="-", color="#444", lw=2.0,
            label=f"mem BW ({PEAK_BW_GBS:.0f} GB/s)", zorder=3)
    # bf16 ceiling: horizontal from its knee to plot right edge.
    ax.plot([ai_knee_bf16, ai_max], [PEAK_BF16_TFLOPS, PEAK_BF16_TFLOPS],
            linestyle="-", color="#1f77b4", lw=2.0,
            label=f"bf16 TC peak ({PEAK_BF16_TFLOPS:.1f} TFLOPS)",
            zorder=3)
    # int8 ceiling: horizontal from its knee to plot right edge.
    ax.plot([ai_knee_int8, ai_max], [PEAK_INT8_TOPS, PEAK_INT8_TOPS],
            linestyle="-", color="#d62728", lw=2.0,
            label=f"int8 TC peak ({PEAK_INT8_TOPS:.1f} TOPS)",
            zorder=3)

    # Region labels in lower band so they don't collide with ceilings
    # or marker labels along the top.
    ax.text(ai_knee_bf16 / 6.0, y_bottom * 4.0, "memory\nbound",
            color="#666", fontsize=10, fontstyle="italic",
            ha="center", va="center")
    ax.text(ai_knee_bf16 * 60.0, y_bottom * 4.0, "compute\nbound",
            color="#666", fontsize=10, fontstyle="italic",
            ha="center", va="center")

    # Per-variant markers: all circles, distinguished by color alone.
    # Saturated edge ring + semi-transparent fill so overlapping
    # markers at near-identical (AI, throughput) (which happens when
    # multiple int8 variants share weight bytes => identical AI) still
    # read distinctly through the layered alpha.
    marker_palette = ["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928"]
    # (dx, dy, ha, va) per index -- stagger label slots so coincident
    # markers do not stack their numeric labels on the same pixels.
    label_offsets = [
        (-12, -10, "right", "top"),       # idx 0: left-down
        (12,   12, "left",  "bottom"),    # idx 1: right-up
        (12,  -10, "left",  "top"),       # idx 2: right-down
        (-12,  12, "right", "bottom"),    # idx 3: left-up
        (0,    18, "center", "bottom"),   # idx 4: above
        (0,   -22, "center", "top"),      # idx 5: below
    ]
    from matplotlib.colors import to_rgba
    for idx, tag in enumerate(tags):
        wbytes = _weight_bytes(tag)
        weight_bytes = N_PARAMS * wbytes
        extra_bytes = _extra_bytes_from_profile(tag) + measured_extra_b
        ai = flops_per_call / (weight_bytes + extra_bytes)
        t_ms = statistics.mean(
            summaries[tag][t].mean_transformer_ms + summaries[tag][t].mean_action_head_ms
            for t in sorted_tasks)
        if t_ms <= 0:
            continue
        tflops = (flops_per_call / 1e12) / (t_ms / 1000.0)
        if wbytes >= 2.0:
            peak = PEAK_BF16_TFLOPS
        else:
            peak = PEAK_INT8_TOPS
        pct_peak = tflops / peak * 100.0 if peak > 0 else 0.0
        edge_rgba = to_rgba(marker_palette[idx % len(marker_palette)], 1.0)
        face_rgba = to_rgba(marker_palette[idx % len(marker_palette)], 0.35)
        ax.scatter([ai], [tflops], s=180, marker="o",
                   facecolors=[face_rgba], edgecolors=[edge_rgba],
                   linewidths=2.2, zorder=5, label=tag)
        # Inline annotation is only the numeric % peak; tag identity is
        # already carried by marker color in the legend. Text color
        # matches the marker edge so each label visually attaches to
        # its own dot.
        dx, dy, ha, va = label_offsets[idx % len(label_offsets)]
        ax.annotate(f"{pct_peak:.1f}% peak",
                    xy=(ai, tflops),
                    xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va=va,
                    fontsize=9.5, color=edge_rgba, fontweight="bold",
                    zorder=6)

    ax.set_xlabel("Arithmetic intensity (FLOPs / byte)", fontsize=10)
    ax.set_ylabel("Throughput (TFLOPs/s)", fontsize=10)
    ax.set_title(
        "Roofline placement of LingBot-VA per-variant inference (RTX A6000)",
        fontsize=11, pad=8)
    # Legend in upper-left like the VLA paper, lists ceilings then
    # markers; compact.
    ax.legend(loc="upper left", fontsize=8.0, framealpha=0.92, ncol=1)
    ax.grid(True, which="both", alpha=0.20, zorder=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    # Caption below the figure (VLA paper convention). Use fig.text so
    # it shows in the standalone PNG even when the report.md embed
    # strips off ax titles.
    fig.subplots_adjust(top=0.92, bottom=0.18)
    # Build AI-denominator description that matches what was actually
    # added per-variant above (weight + optional KV/activation +
    # optional memcpy). Keeps caption honest about scope.
    ai_parts = ["weight bytes"]
    if measured_kv_bytes is not None or measured_act_bytes is not None:
        kv_gb = (float(measured_kv_bytes or 0)) / 1e9
        act_gb = (float(measured_act_bytes or 0)) / 1e9
        ai_parts.append(
            f"hook-measured KV/attn I/O ({kv_gb:.2f} GB) + "
            f"inter-block activation ({act_gb:.2f} GB)")
    if op_data is not None:
        ai_parts.append(f"profile memcpy ({_DTOD_BW_GBS:.0f} GB/s effective)")
    if measured_kv_bytes is None and measured_act_bytes is None and \
            op_data is None:
        ai_parts.append("KV/activation NOT counted -- AI over-estimated")
    ai_caption = " + ".join(ai_parts)
    fig.text(
        0.5, 0.02,
        f"Each dot is one variant: arithmetic intensity = FLOPs / "
        f"({ai_caption}); throughput = FLOPs / measured "
        f"transformer+action_head time. FLOPs measured by "
        f"FlopCounterMode ({flops_per_call/1e12:.1f} TF/call). Label "
        f"shows the achieved fraction of the relevant tensor-core peak "
        f"(bf16 peak for bf16-weight variants, int8 peak for w8a8 variants).",
        ha="center", va="bottom", fontsize=8.5, color="#333",
        wrap=True)
    p_rl = os.path.join(plots_dir, "roofline.png")
    fig.savefig(p_rl, dpi=130)
    plt.close(fig)
    charts.append(("Roofline (achieved throughput vs arithmetic intensity, A6000)",
                   "plots/roofline.png"))

    # ------ Chart 6: Memory breakdown (spatial, analytical) ------
    # Decompose the observed VRAM peak into the four physical
    # components that actually occupy it (per PI question: "how is the
    # breakdown of the 32 GB peak VRAM"). Quantities are computed from
    # model architecture + KV-cache config, then validated against the
    # measured peak; the gap goes to a residual "activations + scratch"
    # segment so the bar lengths match summary.csv exactly.
    #
    #   Text encoder (UMT5-XXL)    : 11.0 GB const  (not quantized in ViDiT-Q)
    #   Transformer weights        : per-variant (bf16 ~10 GB, W8A8 ~5.5 GB)
    #   KV cache                   : 8.92 GB const  (attn_window=72,
    #                                                frame_chunk_size=2,
    #                                                latent 32x40, action=16,
    #                                                30 layers, 24 heads,
    #                                                head_dim=128, bf16)
    #   VAE (AutoencoderKLWan)     : 2.7 GB const
    #   Activations / scratch / frag: measured_peak - sum(above)
    _TEXT_ENCODER_GB = 11.0     # disk: text_encoder/ ~ 11 GB; fp16/bf16 weights, loaded on GPU
    _VAE_GB          = 2.7      # disk: vae/ ~ 2.7 GB
    _KV_CACHE_GB     = 8.92     # 30 layers * 2 (K+V) * B=1 * 24192 tokens * 24 H * 128 D * 2 bytes
    _N_PARAMS_XFMR   = N_PARAMS  # 5.09e9 (defined in roofline block above)
    # Fraction of transformer params that get quantized (everything
    # inside the 30 WanTransformerBlocks; embedders / condition_embedder
    # / proj_out / scale_shift_table stay FP per remain_fp_regex).
    _XFMR_QUANT_FRAC = 0.95
    _QUANT_META_GB = 0.03  # per-channel scales + smooth + Hadamard sign

    def _xfmr_weight_gb(tag: str) -> float:
        # bf16: all params at 2 bytes/param
        # W8A8: quant_frac * 1 byte + (1-quant_frac) * 2 bytes + metadata
        if tag.lower() in ("bf16", "fp16"):
            return _N_PARAMS_XFMR * 2 / 1e9
        return (_N_PARAMS_XFMR * _XFMR_QUANT_FRAC * 1 / 1e9
                + _N_PARAMS_XFMR * (1 - _XFMR_QUANT_FRAC) * 2 / 1e9
                + _QUANT_META_GB)

    measured_peak_gb = [
        statistics.mean(summaries[tag][t].peak_alloc_mb for t in sorted_tasks) / 1024.0
        for tag in tags
    ]

    # 5 segments per bar in fixed left-to-right order. Last is the
    # residual so the visualization always matches the measured total.
    seg_names = ["Text encoder (UMT5)", "Transformer weights",
                 "KV cache (self-attn)", "VAE", "Activations + scratch"]
    seg_colors = ["#6b4596", "#1f78b4", "#ff7f00", "#33a02c", "#999999"]
    seg_data = []  # list per variant: [text, xfmr, kv, vae, residual]  in GB
    for idx, tag in enumerate(tags):
        text = _TEXT_ENCODER_GB
        xfmr = _xfmr_weight_gb(tag)
        kv   = _KV_CACHE_GB
        vae  = _VAE_GB
        residual = max(0.0, measured_peak_gb[idx] - (text + xfmr + kv + vae))
        seg_data.append([text, xfmr, kv, vae, residual])

    fig, ax = plt.subplots(figsize=(13.0, max(4.0, 1.0 * len(tags) + 2.2)))
    y_pos = list(range(len(tags)))
    bar_total_gb = [sum(row) for row in seg_data]
    bar_max_gb = max(bar_total_gb)
    for s_idx, (s_name, s_color) in enumerate(zip(seg_names, seg_colors)):
        widths = [row[s_idx] for row in seg_data]
        lefts  = [sum(row[:s_idx]) for row in seg_data]
        ax.barh(y_pos, widths, left=lefts, color=s_color,
                label=s_name, edgecolor="white",
                linewidth=1.0, height=0.78)
        # Inline label only when segment is big enough to fit text legibly.
        for i, w in enumerate(widths):
            if w >= 0.045 * bar_max_gb:
                tot = bar_total_gb[i]
                pct = w / tot * 100 if tot > 0 else 0
                center = lefts[i] + w / 2
                txt_color = "white" if s_idx in (0, 1, 3) else "#1a1a1a"
                ax.text(center, i,
                        f"{s_name.split(' (')[0]}\n{w:.1f} GB\n({pct:.0f}%)",
                        ha="center", va="center", fontsize=8.5,
                        color=txt_color, fontweight="bold")
    # Total at bar right edge.
    for i, tot in enumerate(bar_total_gb):
        ax.text(tot + 0.012 * bar_max_gb, i, f"{tot:.1f} GB total",
                va="center", fontsize=10, color="#333", fontweight="bold")
    # Bf16 -> W8A8 savings arrow (skip if only one variant).
    if len(tags) >= 2:
        bf16_idx = next((i for i, t in enumerate(tags) if t.lower() in ("bf16", "fp16")), 0)
        for j, tag in enumerate(tags):
            if j == bf16_idx:
                continue
            saving = bar_total_gb[bf16_idx] - bar_total_gb[j]
            if saving > 0.1:
                ax.annotate(
                    f"-{saving:.1f} GB ({saving / bar_total_gb[bf16_idx] * 100:.1f}%)",
                    xy=(bar_total_gb[j], j),
                    xytext=(bar_total_gb[bf16_idx] + 0.18 * bar_max_gb, j - 0.30),
                    arrowprops=dict(arrowstyle="->", color="#b22222", lw=1.6,
                                    connectionstyle="arc3,rad=-0.25"),
                    fontsize=9.5, color="#b22222", fontweight="bold",
                    ha="center", va="center")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(tags, fontsize=11, fontweight="bold")
    ax.set_ylim(-0.6, len(tags) - 0.4)
    ax.invert_yaxis()
    ax.set_xlabel("VRAM peak alloc (GB)", fontsize=10)
    ax.set_title("Per-variant VRAM peak breakdown "
                 "(spatial decomposition of the 32 GB)",
                 fontsize=12, pad=8)
    ax.legend(loc="lower right", framealpha=0.92, fontsize=8.5, ncol=2)
    ax.set_xlim(0, bar_max_gb * 1.32)
    ax.grid(True, axis="x", alpha=0.20)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.subplots_adjust(left=0.18, bottom=0.18)
    floor_gb = _TEXT_ENCODER_GB + _KV_CACHE_GB + _VAE_GB
    xfmr_cap_gb = _N_PARAMS_XFMR * 2 / 1e9 - _xfmr_weight_gb('viditq_w8a8')
    fig.text(
        0.5, 0.04,
        f"Text encoder, VAE, KV cache: unquantized.  "
        f"KV cache analytical: attn_window=72, 30 layers, 24 heads × 128 d, "
        f"24192 tokens/layer, bf16.  "
        f"Transformer: N_PARAMS={_N_PARAMS_XFMR/1e9:.2f}B, W8A8 quantizes "
        f"~{_XFMR_QUANT_FRAC*100:.0f}% (remain_fp_regex).  "
        f"Activations+scratch = residual.",
        ha="center", va="bottom", fontsize=8.0, color="#444", wrap=True)
    fig.text(
        0.5, 0.008,
        f"Floor (text + KV + VAE) = {floor_gb:.1f} GB  •  "
        f"W8A8-on-transformer cap = -{xfmr_cap_gb:.1f} GB  •  "
        f"next levers: KV int8 quant / text-encoder CPU offload.",
        ha="center", va="bottom", fontsize=8.5, color="#b22222",
        fontstyle="italic")
    p_mem = os.path.join(plots_dir, "memory_breakdown.png")
    fig.savefig(p_mem, dpi=130)
    plt.close(fig)
    charts.append(("VRAM peak spatial breakdown per variant "
                   "(text encoder / transformer / KV / VAE / scratch)",
                   "plots/memory_breakdown.png"))

    return charts


# --------------------------------------------------------------------------
# Op-type measured kernel time chart
# --------------------------------------------------------------------------


def _load_op_profile_for_tag(path: str) -> dict[str, float] | None:
    """Read a single aggregator op_profile.json:
        {"_meta": {...}, "op_per_call_ms": {"linear": .., "attention": .., "other": ..}}
    Returns the op_per_call_ms dict or None when unreadable / malformed."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    slot = payload.get("op_per_call_ms")
    if not isinstance(slot, dict):
        return None
    # Legacy 3-key profiles (no memcpy) fall back to 0 for memcpy.
    return {k: float(slot.get(k, 0.0))
            for k in ("linear", "attention", "memcpy", "other")}


def _collect_op_profile(specs: list[str]) -> dict[str, dict[str, float]] | None:
    """Parse multiple `--op_profile TAG=PATH` entries into a per-tag dict.
    Returns None if specs is empty or every path is unreadable."""
    if not specs:
        return None
    out: dict[str, dict[str, float]] = {}
    for s in specs:
        if "=" not in s:
            print(f"warning: --op_profile {s!r} must be TAG=PATH; skipped")
            continue
        tag, path = s.split("=", 1)
        data = _load_op_profile_for_tag(path)
        if data is None:
            print(f"warning: --op_profile {tag}={path!r} not readable; skipped")
            continue
        out[tag] = data
    return out or None


def _render_op_breakdown(
    plots_dir: str,
    tags: list[str],
    baseline_tag: str,
    FancyArrowPatch,
    data: dict[str, dict[str, float]],
    unit_label: str,
    unit_fmt: str,
    xlabel: str,
    title: str,
    caption: str,
    out_name: str,
) -> None:
    """Stacked horizontal bar of Linear vs Attention per variant, with
    DeltaQuant Fig 3 red speedup arrows. Data is supplied by the caller
    so the same renderer serves both the estimated-FLOPs and the
    measured-time variants."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    op_keys = ["linear", "attention", "memcpy", "other"]
    op_labels = {"linear": "Linear", "attention": "Attention",
                 "memcpy": "Memcpy (KV cache)", "other": "Other"}
    op_colors = ["#a14b6b", "#5db1a8", "#7e8fbf", "#f1d49a"]
    seg = {k: [] for k in op_keys}
    totals: list[float] = []
    for tag in tags:
        entry = data.get(tag, {k: 0.0 for k in op_keys})
        for k in op_keys:
            seg[k].append(float(entry.get(k, 0.0)))
        totals.append(sum(seg[k][-1] for k in op_keys))

    fig, ax = plt.subplots(figsize=(13.0, max(4.0, 1.0 * len(tags) + 1.8)))
    y_pos = list(range(len(tags)))
    left = [0.0] * len(tags)
    for key, color in zip(op_keys, op_colors):
        vals = seg[key]
        ax.barh(y_pos, vals, left=left, color=color,
                label=op_labels[key], edgecolor="white",
                linewidth=1.0, height=0.85)
        for i, (v, l) in enumerate(zip(vals, left)):
            if v <= 0:
                continue
            share = v / totals[i] * 100.0 if totals[i] > 0 else 0.0
            label = f"{op_labels[key]}: {unit_fmt.format(v)} {unit_label}\n({share:.1f}%)"
            # Show inner label only when segment is wide enough; small
            # segments rely on legend + the percentage in the next
            # widest segment's neighborhood.
            if v >= 0.12 * max(totals):
                ax.text(l + v / 2, i, label, ha="center", va="center",
                        fontsize=9.5, color="#1a1a1a", fontweight="bold")
        left = [l + v for l, v in zip(left, vals)]

    bar_max = max(left) if left else 1.0
    # Total label at bar end.
    for i, (tot, bar_end) in enumerate(zip(totals, left)):
        is_baseline = (tags[i] == baseline_tag)
        pad = 0.010 * bar_max if is_baseline else 0.040 * bar_max
        ax.text(bar_end + pad, i, f"{unit_fmt.format(tot)} {unit_label}",
                va="center", fontsize=10, color="#333", fontweight="bold")

    # Reference dotted vertical line + curved red speedup arrows.
    base_idx = tags.index(baseline_tag) if baseline_tag in tags else 0
    base_end = left[base_idx]
    ax.plot([base_end, base_end],
            [base_idx, len(tags) - 1 + 0.4],
            linestyle=(0, (2, 3)), color="#222", lw=1.0, alpha=0.55,
            zorder=4)
    arrow_color = "#c0392b"
    arrow_lw = 1.8
    arrow_anchor_pad = 0.018 * bar_max
    src_y_pull = 0.30
    for k in range(1, len(tags)):
        src_x = left[k - 1] + arrow_anchor_pad
        dst_x = left[k] + arrow_anchor_pad
        src_y = (k - 1) + src_y_pull
        dst_y = k - src_y_pull
        arrow = FancyArrowPatch(
            (src_x, src_y), (dst_x, dst_y),
            arrowstyle="->,head_width=7,head_length=9",
            color=arrow_color, lw=arrow_lw,
            connectionstyle="arc3,rad=-0.32",
            zorder=5,
        )
        ax.add_patch(arrow)
        sp = left[k - 1] / left[k] if left[k] > 0 else float("nan")
        mid_x = (src_x + dst_x) / 2.0
        mid_y = (src_y + dst_y) / 2.0
        ax.annotate(f"{sp:.2f}x",
                    xy=(mid_x, mid_y),
                    xytext=(8, 0), textcoords="offset points",
                    color=arrow_color, fontsize=14, fontweight="bold",
                    ha="left", va="center", zorder=6)

    if len(tags) >= 3:
        last = len(tags) - 1
        src_x = left[base_idx] + arrow_anchor_pad
        dst_x = left[last] + arrow_anchor_pad
        src_y = base_idx + src_y_pull
        dst_y = last - src_y_pull
        arrow = FancyArrowPatch(
            (src_x, src_y), (dst_x, dst_y),
            arrowstyle="->,head_width=8,head_length=10",
            color=arrow_color, lw=arrow_lw,
            connectionstyle="arc3,rad=-0.55",
            zorder=5,
        )
        ax.add_patch(arrow)
        cum_sp = left[base_idx] / left[last] if left[last] > 0 else float("nan")
        mid_x = (src_x + dst_x) / 2.0
        mid_y = (src_y + dst_y) / 2.0
        ax.annotate(f"{cum_sp:.2f}x",
                    xy=(mid_x, mid_y),
                    xytext=(8, 0), textcoords="offset points",
                    color=arrow_color, fontsize=14, fontweight="bold",
                    ha="left", va="center", zorder=6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(tags, fontsize=11, fontweight="bold")
    ax.set_ylim(-0.6, len(tags) - 0.4)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(title, fontsize=12, pad=8)
    ax.legend(loc="upper right", framealpha=0.92, fontsize=10)
    right_pad = 1.20 if len(tags) >= 3 else 1.13
    ax.set_xlim(0, bar_max * right_pad)
    ax.grid(True, axis="x", alpha=0.20)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    # Generous left margin so long variant tags (e.g.
    # `viditq_w8a8_dynamic`) are not clipped at the y-tick gutter.
    fig.subplots_adjust(left=0.15, bottom=0.20)
    fig.text(0.5, 0.02, caption,
             ha="center", va="bottom", fontsize=8.0, color="#444", wrap=True)
    p = os.path.join(plots_dir, out_name)
    fig.savefig(p, dpi=130)
    plt.close(fig)


def write_report_md(
    out_path: str,
    variant_paths: list[tuple[str, str]],
    baseline_tag: str,
    tasks: list[str],
    summaries: dict[str, dict[str, SummaryRow]],
    aggregates: dict[str, dict[str, float]],
    step_limits: dict[str, int],
    charts: list[tuple[str, str]] | None = None,
    quant_scope: dict | None = None,
) -> None:
    tags = [t for (t, _) in variant_paths]
    other_tags = [t for t in tags if t != baseline_tag]

    lines: list[str] = []
    a = lines.append

    # ----------------- header -----------------
    a("# LingBot-VA on RoboTwin: Cross-Checkpoint Evaluation Report")
    a("")
    a("Generated by `ptqeval.eval.calc_cross_ckpt`. Joins per-variant "
      "`summary.csv` outputs of `ptqeval.eval.aggregator` on `task_name`.")
    a("")
    a("## Variants")
    a("")
    a("| Tag | Role | Summary CSV |")
    a("| --- | --- | --- |")
    for tag, path in variant_paths:
        role = "baseline" if tag == baseline_tag else f"compared vs {baseline_tag}"
        a(f"| `{tag}` | {role} | `{path}` |")
    a("")
    a(f"- **Task set**: {len(tasks)} tasks shared across all variants")
    a(f"- **Episodes per task**: "
      f"{int(summaries[baseline_tag][tasks[0]].num_episodes)} "
      f"(uniform across variants if both ran the same task config)")
    a("")

    # ----------------- quantization scope -----------------
    a("## Quantization Scope")
    a("")
    a("LingBot-VA's `WanTransformer3DModel` has **30 `WanTransformerBlock`** "
      "instances plus 14 outer Linears. Per block there are 10 `nn.Linear` "
      "modules. The W8A8 wrapper "
      "(`PTQEval/.../method/viditq/block.py::QuantWanTransformerBlockWithCudaKernel`) "
      "swaps a subset:")
    a("")
    a("| Group | Linears | Per-block count | Runtime status |")
    a("| --- | --- | --- | --- |")
    a("| Self-attention (`attn1`) | `to_q`, `to_k`, `to_v`, `to_out[0]` | 4 | **quantized W8A8** |")
    a("| Feed-forward (`ffn`) | `net[0].proj` (up), `net[2]` (down) | 2 | **quantized W8A8** |")
    a("| Cross-attention (`attn2`) | `to_q`, `to_k`, `to_v`, `to_out[0]` | 4 | FP (inherited from ViDiT-Q image-DiT convention; K=text_seq_len small, per-token scale unstable) |")
    a("")
    a("Outer Linears (`patch_embedding_mlp`, `action_embedder`, `proj_out`, "
      "`action_proj_out`, `condition_embedder`, `condition_embedder_action`, "
      "`time_proj`, plus the `scale_shift_table` parameter) are excluded "
      "by `remain_fp_regex` in `configs/w8a8.yaml`. The PTQ run (`ptq.py`) "
      "quantizes everything outside that regex (300 Linears), but the "
      "wrapper only swaps the 6 per-block Linears above (180 total), so "
      "the remaining 120 cross-attn `attn2` Linears appear in "
      "`int_weights.pth` yet stay FP at runtime.")
    a("")
    a("### By Linear count")
    a("")
    a("| Category | Count | Share of all 314 Linears |")
    a("| --- | ---: | ---: |")
    a("| Quantized at runtime (`attn1` + `ffn`, 6 per block × 30) | 180 | 57.3% |")
    a("| Present in `int_weights.pth` but FP at runtime (`attn2`, 4 per block × 30) | 120 | 38.2% |")
    a("| Outer FP (`remain_fp_regex`) | 14 | 4.5% |")
    a("| **Total** | **314** | 100.0% |")
    a("")
    if quant_scope is not None:
        a("### By parameter count")
        a("")
        a(f"Counted from `{quant_scope['ckpt_path']}` "
          f"(summed `int_weight.numel()` per category; matches the original "
          f"FP weight parameter count, dtype-independent).")
        a("")
        active_p   = quant_scope["active"]["params"]
        ca_p       = quant_scope["cross_attn_fp"]["params"]
        other_q_p  = quant_scope["other_quantized"]["params"]
        # Total transformer Linear params we can directly measure
        # (sum over int_weights.pth, i.e. the 300 quantized layers
        # including both active and cross_attn_fp).
        body_p = active_p + ca_p + other_q_p
        total_model_params = 5_090_000_000  # 5.09B per ProjectDescription
        a("| Category | Linears | Params | % of summed transformer Linears | % of full 5.09B model |")
        a("| --- | ---: | ---: | ---: | ---: |")
        a(f"| Quantized at runtime (`attn1` + `ffn`) | "
          f"{quant_scope['active']['n_linears']} | "
          f"{active_p / 1e9:.3f} B | "
          f"{active_p / body_p * 100:.1f}% | "
          f"{active_p / total_model_params * 100:.1f}% |")
        a(f"| Cross-attn FP (`attn2`, PTQ-quantized but runtime FP) | "
          f"{quant_scope['cross_attn_fp']['n_linears']} | "
          f"{ca_p / 1e9:.3f} B | "
          f"{ca_p / body_p * 100:.1f}% | "
          f"{ca_p / total_model_params * 100:.1f}% |")
        if quant_scope["other_quantized"]["n_linears"] > 0:
            a(f"| Other quantized (PTQ-only, not loaded) | "
              f"{quant_scope['other_quantized']['n_linears']} | "
              f"{other_q_p / 1e9:.3f} B | "
              f"{other_q_p / body_p * 100:.1f}% | "
              f"{other_q_p / total_model_params * 100:.1f}% |")
        outer_p_est = total_model_params - body_p
        a(f"| Outer + non-Linear (norms, scale_shift_table, etc., FP, estimated as residual) | 14 + non-Linear | "
          f"{outer_p_est / 1e9:.3f} B | n/a | "
          f"{outer_p_est / total_model_params * 100:.1f}% |")
        a(f"| **Total** | **314 Linears + non-Linear** | **{total_model_params / 1e9:.2f} B** | — | **100.0%** |")
        a("")
        a(f"**Effective quantization coverage** = "
          f"{active_p / total_model_params * 100:.1f}% of total model parameters "
          f"(runtime-active W8A8 path).")
        a("")
    else:
        a("### By parameter count")
        a("")
        a("_Provide `--int_weights_ckpt <path>` (e.g. "
          "`results/viditq_w8a8_kernel/calib/int_weights.pth`) to populate "
          "this section with exact per-category parameter counts._")
        a("")

    # ----------------- headline -----------------
    a("## Headline")
    a("")
    headline_fields = [
        ("mean_success_rate",   "Mean success rate",  _fmt_pct),
        ("mean_total_ms",       "Mean total ms/call", _fmt_ms),
        ("mean_transformer_ms", "Mean transformer ms/call", _fmt_ms),
        ("mean_peak_alloc_mb",  "Mean peak alloc",    _fmt_mb),
        ("mean_init_peak_mb",   "Init peak alloc",    _fmt_mb),
    ]
    head_header = ["Metric"] + [f"`{t}`" for t in tags]
    for t in other_tags:
        head_header.append(f"Δ (`{t}` vs `{baseline_tag}`)")
    a("| " + " | ".join(head_header) + " |")
    a("| " + " | ".join(["---"] * len(head_header)) + " |")
    for field, label, fmt in headline_fields:
        row = [label] + [fmt(aggregates[t][field]) for t in tags]
        base_v = aggregates[baseline_tag][field]
        for t in other_tags:
            v = aggregates[t][field]
            if field == "mean_success_rate":
                row.append(_fmt_sr_delta(v - base_v))
            elif "ms" in field:
                row.append(f"{_ratio(v, base_v):.2f}x")
            elif "_mb" in field:
                row.append(f"{base_v - v:+.0f} MB ({_ratio(base_v - v, base_v) * 100:+.1f}%)")
            else:
                row.append(f"{v - base_v:+.4f}")
        a("| " + " | ".join(row) + " |")
    a("")

    # Embed overall-view charts here so they sit right after the
    # headline numbers they visualize.
    chart_map = {rel: title for (title, rel) in (charts or [])}
    overall_charts = [
        "plots/op_breakdown_measured.png",  # present when --op_profile
        "plots/memory_breakdown.png",
        "plots/roofline.png",               # present when --measured_flops
    ]
    for rel in overall_charts:
        if rel in chart_map:
            a(f"![{chart_map[rel]}]({rel})")
            a("")

    # ----------------- per-task -----------------
    a("## Per-Task Comparison")
    a("")
    if step_limits:
        a(f"`Steps` is the RoboTwin per-episode step limit "
          f"(`task_config/_eval_step_limit.yml`); a proxy for task "
          f"horizon length. Tasks are listed in ascending step order.")
        a("")
    has_steps = bool(step_limits)
    pt_header = ["Task"]
    if has_steps:
        pt_header.append("Steps")
    pt_header.append("N")
    pt_header += [f"SR `{t}`" for t in tags]
    for t in other_tags:
        pt_header.append(f"ΔSR vs `{baseline_tag}`")
    for t in tags:
        pt_header.append(f"total_ms `{t}`")
    for t in other_tags:
        pt_header.append(f"speedup vs `{baseline_tag}`")
    for t in tags:
        pt_header.append(f"peak_MB `{t}`")
    for t in other_tags:
        pt_header.append(f"peak save vs `{baseline_tag}`")
    a("| " + " | ".join(pt_header) + " |")
    a("| " + " | ".join(["---"] * len(pt_header)) + " |")
    # Sort by step limit ascending if available, else lexicographic.
    if has_steps:
        sorted_tasks = sorted(tasks, key=lambda t: (step_limits.get(t, 10**9), t))
    else:
        sorted_tasks = tasks
    for task in sorted_tasks:
        base_row = summaries[baseline_tag][task]
        cells: list[str] = [task]
        if has_steps:
            cells.append(str(step_limits.get(task, "?")))
        cells.append(str(int(base_row.num_episodes)))
        for t in tags:
            cells.append(_fmt_pct(summaries[t][task].success_rate))
        for t in other_tags:
            cells.append(_fmt_sr_delta(
                summaries[t][task].success_rate - base_row.success_rate))
        for t in tags:
            cells.append(_fmt_ms(summaries[t][task].mean_total_ms))
        for t in other_tags:
            cells.append(
                f"{_ratio(base_row.mean_total_ms, summaries[t][task].mean_total_ms):.2f}x")
        for t in tags:
            cells.append(_fmt_mb(summaries[t][task].peak_alloc_mb))
        for t in other_tags:
            cells.append(
                f"{base_row.peak_alloc_mb - summaries[t][task].peak_alloc_mb:+.0f} MB")
        a("| " + " | ".join(cells) + " |")
    a("")

    # Per-task charts embedded after the per-task comparison table.
    per_task_charts = [
        "plots/sr_by_task.png",
        "plots/total_ms_speedup.png",
        "plots/speedup_by_task.png",
        "plots/latency_distribution.png",
    ]
    for rel in per_task_charts:
        if rel in chart_map:
            a(f"![{chart_map[rel]}]({rel})")
            a("")

    # ----------------- notable deltas -----------------
    notable_threshold = 0.10  # |ΔSR| > 10 pp counts as notable
    a(f"## Notable SR Deltas (|ΔSR| > {int(notable_threshold * 100)} pp)")
    a("")
    any_notable = False
    for t in other_tags:
        wins: list[tuple[str, float]] = []
        losses: list[tuple[str, float]] = []
        for task in tasks:
            delta = summaries[t][task].success_rate - summaries[baseline_tag][task].success_rate
            if delta > notable_threshold:
                wins.append((task, delta))
            elif delta < -notable_threshold:
                losses.append((task, delta))
        if wins or losses:
            any_notable = True
            a(f"### `{t}` vs `{baseline_tag}`")
            a("")
            if wins:
                a(f"**Wins ({len(wins)})**: variant exceeds baseline by >10 pp")
                a("")
                a("| Task | baseline SR | variant SR | ΔSR |")
                a("| --- | --- | --- | --- |")
                for task, delta in sorted(wins, key=lambda x: -x[1]):
                    a(f"| {task} | "
                      f"{_fmt_pct(summaries[baseline_tag][task].success_rate)} | "
                      f"{_fmt_pct(summaries[t][task].success_rate)} | "
                      f"{_fmt_sr_delta(delta)} |")
                a("")
            if losses:
                a(f"**Losses ({len(losses)})**: variant falls short of baseline by >10 pp")
                a("")
                a("| Task | baseline SR | variant SR | ΔSR |")
                a("| --- | --- | --- | --- |")
                for task, delta in sorted(losses, key=lambda x: x[1]):
                    a(f"| {task} | "
                      f"{_fmt_pct(summaries[baseline_tag][task].success_rate)} | "
                      f"{_fmt_pct(summaries[t][task].success_rate)} | "
                      f"{_fmt_sr_delta(delta)} |")
                a("")
    if not any_notable:
        a("_No |ΔSR| exceeds the 10 pp notable threshold._")
        a("")

    # ----------------- aggregates -----------------
    a("## Detailed Aggregates")
    a("")
    agg_header = ["Aggregate"] + [f"`{t}`" for t in tags]
    a("| " + " | ".join(agg_header) + " |")
    a("| " + " | ".join(["---"] * len(agg_header)) + " |")
    agg_rows = [
        ("mean_success_rate",   "Mean SR",          _fmt_pct),
        ("median_success_rate", "Median SR",        _fmt_pct),
        ("min_success_rate",    "Min SR",           _fmt_pct),
        ("max_success_rate",    "Max SR",           _fmt_pct),
        ("mean_total_ms",       "Mean total ms",    _fmt_ms),
        ("median_total_ms",     "Median total ms",  _fmt_ms),
        ("mean_transformer_ms", "Mean trans ms",    _fmt_ms),
        ("mean_peak_alloc_mb",  "Mean peak alloc",  _fmt_mb),
        ("mean_init_peak_mb",   "Init peak alloc",  _fmt_mb),
    ]
    for field, label, fmt in agg_rows:
        a("| " + " | ".join([label] + [fmt(aggregates[t][field]) for t in tags]) + " |")
    a("")

    # ----------------- footer -----------------
    a("## Method Note")
    a("")
    a(f"- `total_ms` is the per-inference-call wall clock recorded by "
      f"`PerfProbe` in `ptqeval.eval.perf_probe`, averaged over all "
      f"calls within a task.")
    a(f"- `success_rate` is reported by the RoboTwin client to "
      f"`<save_root>/stseed-*/metrics/<task>/res.json`.")
    a(f"- `peak_alloc_mb` is the maximum allocated CUDA memory across "
      f"the run, reset between calls.")
    a(f"- SR deltas in percentage points (pp); ms / MB deltas as ratios "
      f"or absolute savings.")
    a("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description="Cross-checkpoint aggregation: produce a joined "
                    "summary + English markdown report from multiple "
                    "per-variant summary.csv inputs.")
    p.add_argument("--variant", action="append", required=True,
                   metavar="TAG=CSV_PATH",
                   help="Variant tag and path to its summary.csv. First "
                        "--variant is treated as the baseline. Repeat "
                        "for each variant.")
    p.add_argument("--out_dir", required=True,
                   help="Output directory. Writes cross_summary.csv, "
                        "cross_summary.json, report.md.")
    p.add_argument("--step_limit_yaml",
                   default="/home/arash/EvalForWAMs/RoboTwin/task_config/_eval_step_limit.yml",
                   help="Path to RoboTwin's _eval_step_limit.yml. When "
                        "readable, the per-task table gains a `Steps` "
                        "column and rows are sorted by step limit. Set "
                        "to '' to disable.")
    p.add_argument("--no_plots", action="store_true",
                   help="Skip the matplotlib charts (SR per task, "
                        "latency+speedup, speedup per task). Set if "
                        "matplotlib is unavailable or unwanted.")
    p.add_argument("--int_weights_ckpt",
                   default="/home/arash/EvalForWAMs/results/viditq_w8a8_kernel/calib/int_weights.pth",
                   help="Path to a PTQ-produced int_weights.pth. When "
                        "readable, the report's Quantization Scope "
                        "section gains per-category parameter counts "
                        "(active vs cross_attn_fp vs other). Set to "
                        "'' to skip.")
    p.add_argument("--op_profile", action="append", default=[],
                   help="Per-variant profiler JSON: <TAG>=<path>. Path "
                        "points at a JSON produced by aggregator's "
                        "merge_op_profiles (i.e. <summary>/op_profile.json) "
                        "containing {op_per_call_ms: {linear, attention, "
                        "other}}. Repeat the flag once per variant. When "
                        "any are supplied, emits "
                        "plots/op_breakdown_measured.png next to the "
                        "architecture-estimated version.")
    p.add_argument("--measured_flops", default="",
                   help="Path to measure_flops JSON {flops_per_call_tf}. "
                        "When supplied, roofline uses this measured "
                        "value instead of the architecture-estimated "
                        "linear+attention sum -- gives true dispatcher-"
                        "level FLOPs/call.")
    args = p.parse_args()

    variant_specs = [_parse_variant_arg(v) for v in args.variant]
    if len(variant_specs) < 1:
        print("error: need at least 1 --variant entry", file=sys.stderr)
        return 2
    baseline_tag = variant_specs[0][0]
    # Single-variant mode: all comparison-only charts and report
    # sections degrade to "report only" rather than disappearing the
    # variant entirely. Useful for previewing a new run before the
    # paired baseline finishes.
    if len(variant_specs) == 1:
        print(f"single-variant mode: only {baseline_tag} rendered; "
              f"comparison-only charts/columns are omitted.")

    summaries: dict[str, dict[str, SummaryRow]] = {}
    for tag, path in variant_specs:
        if not os.path.exists(path):
            print(f"error: summary csv not found: {path}", file=sys.stderr)
            return 2
        summaries[tag] = load_summary(path)
        print(f"loaded {tag}: {len(summaries[tag])} tasks from {path}")

    common = set.intersection(*(set(s.keys()) for s in summaries.values()))
    if not common:
        print("error: no task_name overlap across variants", file=sys.stderr)
        return 2
    missing_report = []
    for tag, _ in variant_specs:
        only_here = set(summaries[tag].keys()) - common
        if only_here:
            missing_report.append(f"  {tag}: {len(only_here)} tasks not in intersection ({sorted(only_here)[:3]}...)")
    if missing_report:
        print("warning: some tasks dropped because not present in all variants:")
        for m in missing_report:
            print(m)

    tasks = sorted(common)
    tags = [t for t, _ in variant_specs]
    aggregates = compute_aggregates(tags, tasks, summaries)
    step_limits = _load_step_limits(args.step_limit_yaml)
    if step_limits:
        missing = [t for t in tasks if t not in step_limits]
        if missing:
            print(f"warning: step_limit missing for {len(missing)} task(s); "
                  f"will display '?' in the table: {missing[:5]}")
    elif args.step_limit_yaml:
        print(f"warning: --step_limit_yaml {args.step_limit_yaml!r} not "
              f"readable; per-task table will omit the Steps column.")

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path  = os.path.join(args.out_dir, "cross_summary.csv")
    json_path = os.path.join(args.out_dir, "cross_summary.json")
    md_path   = os.path.join(args.out_dir, "report.md")

    # Load op_profile data up-front so roofline can use measured
    # memcpy bytes for AI estimation (not just weight-bytes-only).
    op_data = _collect_op_profile(args.op_profile)

    # Optional measured FLOPs/call AND measured bytes/call from
    # ptqeval.eval.measure_flops. The bytes block is dtype-invariant
    # (KV cache + inter-block activations stay bf16 even in W8A8) and
    # is added to the roofline AI denominator on top of per-variant
    # weight bytes.
    measured_flops_tf: float | None = None
    measured_kv_bytes: float | None = None
    measured_act_bytes: float | None = None
    if args.measured_flops and os.path.exists(args.measured_flops):
        try:
            with open(args.measured_flops) as f:
                mf = json.load(f)
            measured_flops_tf = float(mf.get("flops_per_call_tf", 0)) or None
            bpc = mf.get("bytes_per_call_b") or {}
            measured_kv_bytes = float(bpc.get("kv_attention_bytes_b", 0)) or None
            measured_act_bytes = float(bpc.get("block_activation_bytes_b", 0)) or None
            if measured_flops_tf:
                msg = (f"loaded measured_flops: {measured_flops_tf:.2f} "
                       f"TF/call from {args.measured_flops}")
                if measured_kv_bytes or measured_act_bytes:
                    msg += (f"; KV {measured_kv_bytes / 1e9:.2f} GB + "
                            f"act {measured_act_bytes / 1e9:.2f} GB / call")
                print(msg)
        except (OSError, json.JSONDecodeError, ValueError) as e:
            print(f"warning: --measured_flops unreadable: {e}")

    charts: list[tuple[str, str]] = []
    if not args.no_plots:
        charts = _make_plots(args.out_dir, variant_specs, baseline_tag,
                             tasks, summaries, step_limits,
                             op_data=op_data,
                             measured_flops_tf=measured_flops_tf,
                             measured_kv_bytes=measured_kv_bytes,
                             measured_act_bytes=measured_act_bytes)

        # Overlay a profiler-measured op_breakdown chart alongside the
        # architecture-estimated default when op_profile is available.
        if op_data is not None:
            from matplotlib.patches import FancyArrowPatch
            plots_dir = os.path.join(args.out_dir, "plots")
            missing = [t for t in tags if t not in op_data]
            if missing:
                print(f"warning: --op_profile missing entries for "
                      f"{missing}; measured op_breakdown will show 0 for "
                      f"those variants.")
            srcs = ", ".join(sorted(op_data.keys()))
            _render_op_breakdown(
                plots_dir, tags, baseline_tag, FancyArrowPatch,
                data=op_data,
                unit_label="ms", unit_fmt="{:.0f}",
                xlabel="Mean op-type kernel time per inference call (ms)",
                title=("Per-variant op-type measured kernel time "
                       "(Linear vs Attention)"),
                caption=(
                    "Memcpy ↓24×: W8A8 kernel fuses bias-add in-register; "
                    "bf16 cuBLAS addmm stages bias via separate Memcpy DtoD "
                    "(~37 us × 324K launches/task).\n"
                    "Other = LayerNorm/RMSNorm + GELU + RoPE complex mul + "
                    "residual/AdaLN elementwise + KV-cache slot scatter "
                    "+ dtype cast (bf16↔fp32 for norms) + W8A8-only: "
                    "act_quant kernel (per-token int8 scale, ~+350 ms vs bf16)."
                ),
                out_name="op_breakdown_measured.png",
            )
            charts.append(("Op-type measured kernel time per variant "
                           "(Linear vs Attention)",
                           "plots/op_breakdown_measured.png"))
            print(f"loaded op_profile for {len(op_data)} variants; "
                  f"rendered op_breakdown_measured.png")
        elif args.op_profile:
            print(f"warning: --op_profile {args.op_profile!r} produced "
                  f"no usable data; only architecture-estimated "
                  f"op_breakdown will appear.")

    quant_scope = _inspect_quant_scope(args.int_weights_ckpt)
    if quant_scope is None and args.int_weights_ckpt:
        print(f"warning: --int_weights_ckpt {args.int_weights_ckpt!r} not "
              f"readable (or torch unavailable); Quantization Scope param "
              f"table will be omitted.")

    write_cross_csv(csv_path, tags, baseline_tag, tasks, summaries)
    write_cross_json(json_path, variant_specs, baseline_tag, tasks, summaries, aggregates, step_limits)
    write_report_md(md_path, variant_specs, baseline_tag, tasks, summaries,
                    aggregates, step_limits, charts, quant_scope)

    print(f"wrote {csv_path}  ({len(tasks)} task rows x "
          f"{len(tags)} variants)")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    if charts:
        print(f"wrote {len(charts)} plot(s) under {os.path.join(args.out_dir, 'plots')}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
