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
) -> list[tuple[str, str]]:
    """Renders 3 charts (SR per task, latency + speedup curve, speedup per
    task) under <out_dir>/plots/. Returns [(title, report-relative path)]
    pairs for embedding in report.md. Empty list if matplotlib is missing
    (graceful fallback)."""
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
    fig, ax = plt.subplots(figsize=(fig_w, 5.0))
    bw = 0.8 / max(1, len(tags))
    for i, tag in enumerate(tags):
        srs = [summaries[tag][t].success_rate * 100.0 for t in sorted_tasks]
        offset = (i - (len(tags) - 1) / 2.0) * bw
        ax.bar([xi + offset for xi in x], srs, width=bw, label=tag)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Per-task success rate (tasks sorted by step limit)")
    ax.legend(loc="lower left")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p1 = os.path.join(plots_dir, "sr_by_task.png")
    fig.savefig(p1, dpi=120)
    plt.close(fig)
    charts.append(("Per-task success rate (sorted by step limit)",
                   "plots/sr_by_task.png"))

    # ------ Chart 2: total_ms bars + speedup line ------
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(fig_w, 7.5),
        gridspec_kw={"height_ratios": [2, 1]}, sharex=True)
    for i, tag in enumerate(tags):
        ms = [summaries[tag][t].mean_total_ms for t in sorted_tasks]
        offset = (i - (len(tags) - 1) / 2.0) * bw
        ax_top.bar([xi + offset for xi in x], ms, width=bw, label=tag)
    ax_top.set_ylabel("Mean total ms / call")
    ax_top.set_title("Latency and speedup vs task (sorted by step limit)")
    ax_top.legend(loc="upper right")
    ax_top.grid(True, axis="y", alpha=0.3)

    base_ms = [summaries[baseline_tag][t].mean_total_ms for t in sorted_tasks]
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
    charts.append(("Mean total ms per call + speedup ratio",
                   "plots/total_ms_speedup.png"))

    # ------ Chart 3: Speedup per task (histogram) ------
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

    # ------ Chart 4: ΔSR vs step_limit scatter ------
    # Only meaningful if we have step limits to anchor the x-axis.
    if step_limits:
        fig, ax = plt.subplots(figsize=(max(8.0, n * 0.55), 5.0))
        for tag in other_tags:
            xs, ys, names = [], [], []
            for t in sorted_tasks:
                if t not in step_limits:
                    continue
                xs.append(step_limits[t])
                ys.append((summaries[tag][t].success_rate
                           - summaries[baseline_tag][t].success_rate) * 100.0)
                names.append(t)
            ax.scatter(xs, ys, s=70, alpha=0.85,
                       label=f"{tag} vs {baseline_tag}")
            # Trend line (linear fit if >= 3 points, else just connect).
            if len(xs) >= 3:
                paired = sorted(zip(xs, ys))
                xs_s, ys_s = zip(*paired)
                # Plain linear regression via closed form (no numpy dep).
                n_pts = len(xs_s)
                mean_x = sum(xs_s) / n_pts
                mean_y = sum(ys_s) / n_pts
                num = sum((xs_s[i] - mean_x) * (ys_s[i] - mean_y) for i in range(n_pts))
                den = sum((xs_s[i] - mean_x) ** 2 for i in range(n_pts))
                if den != 0:
                    slope = num / den
                    intercept = mean_y - slope * mean_x
                    xline = [min(xs_s), max(xs_s)]
                    yline = [slope * v + intercept for v in xline]
                    ax.plot(xline, yline, linestyle="--", linewidth=1.2,
                            alpha=0.7,
                            label=f"{tag} trend (slope {slope:+.3f} pp/step)")
            for xi, yi, ni in zip(xs, ys, names):
                ax.annotate(ni, (xi, yi), xytext=(4, 3),
                            textcoords="offset points", fontsize=7, alpha=0.75)
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_xlabel("Step limit (task horizon proxy)")
        ax.set_ylabel("ΔSR (variant - baseline), pp")
        ax.set_title("SR delta vs task horizon (does quant error compound with horizon?)")
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p4 = os.path.join(plots_dir, "sr_delta_vs_steps.png")
        fig.savefig(p4, dpi=120)
        plt.close(fig)
        charts.append(("ΔSR vs step limit (horizon-error correlation)",
                       "plots/sr_delta_vs_steps.png"))

    # ------ Chart 5: Stage breakdown stacked bars ------
    # Per task, one stacked bar per variant: transformer / action_head /
    # other (= total - transformer - action_head; captures vae_encode +
    # FSDP/comm/launch overhead).
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))
    bw5 = 0.8 / max(1, len(tags))
    stage_colors = {
        "transformer": "#1f77b4",
        "action_head": "#ff7f0e",
        "other":       "#7f7f7f",
    }
    for i, tag in enumerate(tags):
        trans = [summaries[tag][t].mean_transformer_ms for t in sorted_tasks]
        head  = [summaries[tag][t].mean_action_head_ms for t in sorted_tasks]
        total = [summaries[tag][t].mean_total_ms       for t in sorted_tasks]
        other = [max(0.0, total[j] - trans[j] - head[j]) for j in range(n)]
        offset = (i - (len(tags) - 1) / 2.0) * bw5
        positions = [xi + offset for xi in x]
        ax.bar(positions, trans, width=bw5,
               color=stage_colors["transformer"],
               label="transformer" if i == 0 else None)
        ax.bar(positions, head, width=bw5,
               bottom=trans,
               color=stage_colors["action_head"],
               label="action_head" if i == 0 else None)
        ax.bar(positions, other, width=bw5,
               bottom=[trans[j] + head[j] for j in range(n)],
               color=stage_colors["other"],
               label="other (vae_encode + FSDP / comm)" if i == 0 else None)
        # Variant label above each bar group's first cluster.
        for j, pos in enumerate(positions):
            if j == 0:
                ax.text(pos, total[j] + 50, tag, ha="center",
                        fontsize=6, alpha=0.7, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean ms / call (stacked stages)")
    ax.set_title(
        "Per-call latency breakdown by stage "
        f"(per task: {' + '.join(tags)} side-by-side, tasks sorted by step)"
    )
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p5 = os.path.join(plots_dir, "stage_breakdown.png")
    fig.savefig(p5, dpi=120)
    plt.close(fig)
    charts.append(("Per-call latency stage breakdown",
                   "plots/stage_breakdown.png"))

    # ------ Chart 6: Latency distribution (mean / p50 / p95) ------
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
    ax.set_ylabel("Total ms / call")
    ax.set_title(
        "Latency distribution per task "
        "(circle = mean, short dash = p50, vertical bar reaches p95)"
    )
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p6 = os.path.join(plots_dir, "latency_distribution.png")
    fig.savefig(p6, dpi=120)
    plt.close(fig)
    charts.append(("Latency distribution (mean / p50 / p95)",
                   "plots/latency_distribution.png"))

    # ------ Chart 7: Memory init vs runtime delta ------
    # Per variant: one stacked bar with init_peak (load-time, weights) +
    # runtime_delta (= peak_alloc - init_peak, activations / kernel
    # scratch / FSDP intermediates). Aggregated across tasks (mean) since
    # values barely vary within a variant.
    fig, ax = plt.subplots(figsize=(max(5.0, len(tags) * 1.6), 5.0))
    init_means = []
    runtime_deltas = []
    for tag in tags:
        inits = [summaries[tag][t].init_peak_mb for t in sorted_tasks]
        peaks = [summaries[tag][t].peak_alloc_mb for t in sorted_tasks]
        init_means.append(statistics.mean(inits))
        runtime_deltas.append(statistics.mean(peaks) - statistics.mean(inits))
    xv = list(range(len(tags)))
    ax.bar(xv, init_means, width=0.6,
           color="#4c72b0", label="init peak (FP weights + framework)")
    ax.bar(xv, runtime_deltas, width=0.6, bottom=init_means,
           color="#dd8452",
           label="runtime delta (activations / FSDP / kernel scratch)")
    for xi, init, delta in zip(xv, init_means, runtime_deltas):
        total = init + delta
        ax.text(xi, total + 200, f"total {total / 1024:.2f} GB",
                ha="center", fontsize=9)
        ax.text(xi, init / 2, f"{init / 1024:.2f} GB",
                ha="center", va="center", fontsize=9, color="white")
        ax.text(xi, init + delta / 2, f"{delta / 1024:.2f} GB",
                ha="center", va="center", fontsize=9, color="white")
    ax.set_xticks(xv)
    ax.set_xticklabels(tags)
    ax.set_ylabel("Peak allocated memory (MB), averaged across tasks")
    ax.set_title("Memory breakdown: init vs runtime delta")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p7 = os.path.join(plots_dir, "memory_init_vs_runtime.png")
    fig.savefig(p7, dpi=120)
    plt.close(fig)
    charts.append(("Memory breakdown (init vs runtime delta)",
                   "plots/memory_init_vs_runtime.png"))

    # ------ Chart 8: Pareto trade-off summary ------
    # Each variant a single bubble: x = mean speedup vs baseline,
    # y = mean SR (%), size = mean peak alloc savings vs baseline (GB).
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    base_mean_total = statistics.mean(
        summaries[baseline_tag][t].mean_total_ms for t in sorted_tasks)
    base_mean_peak = statistics.mean(
        summaries[baseline_tag][t].peak_alloc_mb for t in sorted_tasks)
    for tag in tags:
        v_mean_total = statistics.mean(
            summaries[tag][t].mean_total_ms for t in sorted_tasks)
        v_mean_peak = statistics.mean(
            summaries[tag][t].peak_alloc_mb for t in sorted_tasks)
        v_mean_sr = statistics.mean(
            summaries[tag][t].success_rate for t in sorted_tasks) * 100.0
        speedup = base_mean_total / v_mean_total if v_mean_total > 0 else 1.0
        savings_gb = (base_mean_peak - v_mean_peak) / 1024.0
        # Size scales with abs savings, with a floor so baseline (0 savings)
        # is still visible.
        size = 200 + abs(savings_gb) * 600
        marker = "o" if tag != baseline_tag else "s"
        ax.scatter([speedup], [v_mean_sr], s=size, alpha=0.65,
                   marker=marker, edgecolors="black", linewidth=1.0)
        annot = (f"{tag}\n"
                 f"speed {speedup:.2f}x, "
                 f"SR {v_mean_sr:.1f}%, "
                 f"peak save {savings_gb:+.2f} GB")
        ax.annotate(annot, (speedup, v_mean_sr),
                    xytext=(12, 10), textcoords="offset points",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec="gray", alpha=0.85))
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel(f"Mean speedup ratio vs `{baseline_tag}` (1.0 = parity)")
    ax.set_ylabel("Mean success rate (%)")
    ax.set_title(
        "Pareto summary: speed vs SR (bubble size = peak alloc savings)\n"
        "Top-right is best (higher SR, higher speedup)"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p8 = os.path.join(plots_dir, "pareto_tradeoff.png")
    fig.savefig(p8, dpi=120)
    plt.close(fig)
    charts.append(("Pareto trade-off (speedup vs SR, sized by peak alloc savings)",
                   "plots/pareto_tradeoff.png"))

    return charts


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

    # ----------------- visualizations (files only, not embedded) -----------------
    if charts:
        a("## Visualizations")
        a("")
        a("Saved under `plots/` next to this report (not embedded inline):")
        a("")
        for title, rel in charts:
            a(f"- `{rel}` — {title}")
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
    args = p.parse_args()

    variant_specs = [_parse_variant_arg(v) for v in args.variant]
    if len(variant_specs) < 2:
        print("error: need at least 2 --variant entries "
              "(1 baseline + 1 comparison)", file=sys.stderr)
        return 2
    baseline_tag = variant_specs[0][0]

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

    charts: list[tuple[str, str]] = []
    if not args.no_plots:
        charts = _make_plots(args.out_dir, variant_specs, baseline_tag,
                             tasks, summaries, step_limits)

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
