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
    """Renders 6 charts under <out_dir>/plots/:
      1. sr_by_task.png           per-task SR per variant
      2. total_ms_speedup.png     latency + speedup curve per task
      3. speedup_by_task.png      speedup ratio per task
      4. latency_distribution.png mean / p50 / p95 per task
      5. compute_breakdown.png    horizontal stacked-bar with speedup arrows
                                  (paper-figure style, image.png-inspired)
      6. roofline.png             achieved (AI, throughput) vs A6000 ceilings
    Returns [(title, report-relative path)] pairs for embedding in
    report.md. Empty list if matplotlib is missing (graceful fallback)."""
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

    # ------ Chart 5: Compute breakdown per variant (DeltaQuant Fig 3 style) ------
    # Horizontal stacked bars, one row per variant, stack order:
    # transformer | action_head | other (= total - transformer - action_head;
    # absorbs text_encoder + vae_encode + comm + dispatch + PerfProbe overlap
    # residual). Adjacent variants get a curved red speedup arrow above the
    # bar end (paper-figure style); a single cumulative arrow on the far right
    # carries baseline -> last-variant ratio when N >= 3.
    from matplotlib.patches import FancyArrowPatch  # local import; matplotlib already loaded

    # Segment keys map to PerfProbe stage fields:
    #   transformer  -> "Video"  (Algorithm 1 line 4: video stream 3
    #                   Euler-step partial denoising loop, runs through
    #                   the d_v=3072 video-stream transformer blocks)
    #   action_head  -> "Action" (Algorithm 1 line 7: action stream 10
    #                   Euler-step full denoising loop conditioned on
    #                   the predicted video chunk, runs through the
    #                   d_a=768 action-stream blocks with joint cross-
    #                   modal attention into the video space)
    #   other        -> "Other" (text_encoder + vae_encode + dispatch +
    #                   PerfProbe stage-overlap residual)
    # Source fields stay as transformer/action_head for backward
    # compatibility; only the visible label is renamed.
    breakdown_keys = ["transformer", "action_head", "other"]
    breakdown_labels = {"transformer": "Video", "action_head": "Action",
                        "other": "Other"}
    breakdown_colors = ["#a14b6b", "#5db1a8", "#f1d49a"]
    seg_data: dict[str, list[float]] = {k: [] for k in breakdown_keys}
    totals: list[float] = []
    for tag in tags:
        trans = statistics.mean(
            summaries[tag][t].mean_transformer_ms for t in sorted_tasks)
        head  = statistics.mean(
            summaries[tag][t].mean_action_head_ms for t in sorted_tasks)
        total = statistics.mean(
            summaries[tag][t].mean_total_ms for t in sorted_tasks)
        other = max(0.0, total - trans - head)
        seg_data["transformer"].append(trans)
        seg_data["action_head"].append(head)
        seg_data["other"].append(other)
        totals.append(total)

    # Compact row spacing to mirror DeltaQuant Fig 3 (bars nearly
    # touching). Bar height 0.85 with ylim padding 0.15 keeps a thin
    # visual seam between rows; explicit ylim because matplotlib default
    # around invert_yaxis can squash bars into a thin band.
    fig, ax = plt.subplots(figsize=(13.0, max(4.0, 1.0 * len(tags) + 1.8)))
    y_pos = list(range(len(tags)))
    left = [0.0] * len(tags)
    for key, color in zip(breakdown_keys, breakdown_colors):
        vals = seg_data[key]
        ax.barh(y_pos, vals, left=left, color=color,
                label=breakdown_labels[key],
                edgecolor="white", linewidth=1.0, height=0.85)
        for i, (v, l) in enumerate(zip(vals, left)):
            if v <= 0:
                continue
            share = v / totals[i] * 100.0 if totals[i] > 0 else 0.0
            label = f"{breakdown_labels[key]}: {v:.0f} ms\n({share:.1f}%)"
            if v >= 0.08 * max(totals):
                ax.text(l + v / 2, i, label, ha="center", va="center",
                        fontsize=9.5, color="#1a1a1a", fontweight="bold")
        left = [l + v for l, v in zip(left, vals)]

    # Total ms label at actual bar end (sum of segments). Padding is
    # tight on the baseline bar (no arrow head ever lands there) and
    # generous on every other bar (otherwise the red arrow head whose
    # dst sits at `bar_end + arrow_anchor_pad` overstrikes the digits).
    # ms padding must exceed the arrow_anchor_pad used a few lines
    # below; pick 0.04 so the label sits clearly right of the arrow tip.
    bar_max = max(left) if left else 1.0
    for i, (tot, bar_end) in enumerate(zip(totals, left)):
        is_baseline = (tags[i] == baseline_tag)
        pad = 0.010 * bar_max if is_baseline else 0.040 * bar_max
        ax.text(bar_end + pad, i,
                f"{tot:.0f} ms",
                va="center", fontsize=10, color="#333", fontweight="bold")

    # Reference dotted vertical line from the baseline bar end down to
    # the last bar -- mirrors the black dashed reference in DeltaQuant
    # Fig 3 that visually anchors the speedup arrows. Drawn before the
    # arrows so arrows sit on top.
    base_idx = tags.index(baseline_tag) if baseline_tag in tags else 0
    base_end = left[base_idx]
    ax.plot([base_end, base_end],
            [base_idx, len(tags) - 1 + 0.4],
            linestyle=(0, (2, 3)), color="#222", lw=1.0, alpha=0.55,
            zorder=4)

    # Curved red speedup arrows (DeltaQuant Fig 3 idiom). Adjacent
    # arrows are anchored at the actual bar ends -- so 1.38x emerges
    # from just past the bf16 bar tip (`4657 ms` corner) and curls into
    # the dynamic bar tip below it -- rather than pushed into a
    # separate right-side lane. Cumulative arrow (N>=3) keeps its own
    # right-side column so it does not pile on top of the adjacent
    # arcs and their labels.
    arrow_color = "#c0392b"
    arrow_lw = 1.8
    arrow_anchor_pad = 0.018 * bar_max   # tiny gap past bar end
    # src/dst y are pulled toward the gap between rows (instead of row
    # center) so the arc starts at the bar's lower-right corner and
    # ends at the next bar's upper-right corner -- "from the bottom-
    # left of `4657 ms`". This shortens the arc and keeps the head
    # well clear of the next bar's interior ms label.
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
        # Label sits at the arc midpoint (DeltaQuant Fig 3 idiom) so
        # the digit rides on the curve rather than off to its side.
        # Small upward offset in display points keeps the digits from
        # overlapping the red arc itself.
        mid_x = (src_x + dst_x) / 2.0
        mid_y = (src_y + dst_y) / 2.0
        ax.annotate(f"{sp:.2f}x",
                    xy=(mid_x, mid_y),
                    xytext=(8, 0), textcoords="offset points",
                    color=arrow_color, fontsize=14, fontweight="bold",
                    ha="left", va="center", zorder=6)

    if len(tags) >= 3:
        last = len(tags) - 1
        # Cumulative arrow shares endpoints with the adjacent chain:
        # src matches the 1.38x source (baseline bar end, lower-right
        # corner of `bf16`), dst matches the 0.97x dst (last bar end,
        # upper-right corner of `viditq_static`). A larger curvature
        # pushes the arc further out so it skirts the inner adjacent
        # arcs and labels instead of overlapping them.
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
        # Label rides the cumulative arc at its midpoint, mirroring
        # the adjacent labels' "on-the-curve" placement so the three
        # speedup digits sit in a consistent visual slot.
        mid_x = (src_x + dst_x) / 2.0
        mid_y = (src_y + dst_y) / 2.0
        ax.annotate(f"{cum_sp:.2f}x",
                    xy=(mid_x, mid_y),
                    xytext=(8, 0), textcoords="offset points",
                    color=arrow_color, fontsize=14, fontweight="bold",
                    ha="left", va="center", zorder=6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(tags, fontsize=11, fontweight="bold")
    # Tight half-row padding so bars are nearly touching (DeltaQuant
    # Fig 3 idiom). Invert so baseline reads on top.
    ax.set_ylim(-0.6, len(tags) - 0.4)
    ax.invert_yaxis()
    ax.set_xlabel(
        f"Mean per-call latency (ms), averaged across {len(sorted_tasks)} tasks",
        fontsize=10)
    ax.set_title("Per-variant compute breakdown", fontsize=12, pad=8)
    # Legend at upper right corner avoids landing on speedup-arrow
    # lanes which all sit in the lower-right quadrant of the plot area.
    ax.legend(loc="upper right", framealpha=0.92, fontsize=10)
    # Right padding sized to fit the adjacent-arc labels (~0.05 past
    # bar_max) and, when >=3 variants, the wider cumulative arc that
    # bulges further right (~0.13 past bar_max with rad=-0.55) plus
    # its `X.XXx` label.
    right_pad = 1.20 if len(tags) >= 3 else 1.13
    ax.set_xlim(0, bar_max * right_pad)
    ax.grid(True, axis="x", alpha=0.20)
    # Hide top/right spines for a cleaner paper look.
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    p_bd = os.path.join(plots_dir, "compute_breakdown.png")
    fig.savefig(p_bd, dpi=130)
    plt.close(fig)
    charts.append(("Compute breakdown per variant "
                   "(transformer / action_head / other; curved speedup arrows)",
                   "plots/compute_breakdown.png"))

    # ------ Chart 6: Roofline (achieved throughput vs arithmetic intensity) ------
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
    # Pull raw FLOPs (linear + attention, no bf16-equivalent scaling)
    # from the same architecture model used by op_breakdown so the
    # two charts agree on absolute work. _estimate_op_pf("bf16") path
    # skips the /2 w8a8 scaling and returns honest BF16 FLOPs in TF.
    _arch_pf = _estimate_op_pf("bf16")
    flops_per_call = (_arch_pf["linear"] + _arch_pf["attention"]) * 1e12

    bytes_per_param = {"bf16": 2.0, "fp16": 2.0}
    # All variants other than bf16 carry W8A8 weights (int8 = 1 byte/param).
    def _weight_bytes(tag: str) -> float:
        return bytes_per_param.get(tag.lower(), 1.0)

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
        ai = flops_per_call / (N_PARAMS * wbytes)
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
    fig.text(
        0.5, 0.02,
        "Each dot is one variant: arithmetic intensity from FLOPs / "
        "weight bytes; throughput from FLOPs / measured "
        "transformer+action_head time. Label shows the achieved "
        "fraction of the relevant tensor-core peak (bf16 peak for "
        "bf16-weight variants, int8 peak for w8a8 variants).",
        ha="center", va="bottom", fontsize=8.5, color="#333",
        wrap=True)
    p_rl = os.path.join(plots_dir, "roofline.png")
    fig.savefig(p_rl, dpi=130)
    plt.close(fig)
    charts.append(("Roofline (achieved throughput vs arithmetic intensity, A6000)",
                   "plots/roofline.png"))

    # ------ Chart 7: Op-type breakdown (Linear vs Attention) ------
    # Two data sources, selected per call site:
    #   1. Measured: profiler JSON keyed by tag -> {linear, attention,
    #      other} kernel ms (preferred; ground truth).
    #   2. Estimated: architecture FLOPs derived from _ARCH constants,
    #      reported in BF16-equivalent TFLOPs (fallback when no
    #      profiler data exists; DeltaQuant Fig 3 idiom).
    # _make_plots stays data-agnostic: it always emits the estimated
    # version. main() additionally renders the measured chart when
    # --op_profile points at a readable JSON.
    _render_op_breakdown(
        plots_dir, tags, baseline_tag, FancyArrowPatch,
        data={tag: _estimate_op_pf(tag) for tag in tags},
        unit_label="TF", unit_fmt="{:.1f}",
        xlabel="BF16-equivalent FLOPs per inference call (TFLOPs)",
        title="Per-variant op-type FLOPs breakdown (Linear vs Attention)",
        caption=(
            "FLOPs estimated from LingBot-VA architecture (d_v=3072 + "
            "d_a=768 MoT, 30 layers, K=4 chunk, video=3 Euler*2 CFG, "
            "action=10 Euler). Linear includes self-attn QKV+O, FFN, "
            "cross-attn, MoT projection; Attention is QK^T + softmax-V. "
            "w8a8 Linear divided by 2 (A6000 INT8 TC peak 309.7 TOPS = "
            "2x BF16 TC peak 154.8 TFLOPS). Op-level kernel time not "
            "measured -- pass --op_profile <path> to use real profiler data."
        ),
        out_name="op_breakdown.png",
    )
    charts.append(("Op-type FLOPs breakdown per variant "
                   "(Linear vs Attention, BF16-equivalent; estimated)",
                   "plots/op_breakdown.png"))

    # ------ Chart 8: Memory breakdown (model weights + transient) ------
    # Two-segment stacked horizontal bar per variant decomposing the
    # observed VRAM peak into model weights (init_peak_mb, never freed)
    # and the additional activations/KV cache/scratch buffer (peak -
    # init). Uses summary.csv fields only -- no extra instrumentation.
    seg_weights = [
        statistics.mean(summaries[tag][t].init_peak_mb for t in sorted_tasks)
        for tag in tags
    ]
    seg_transient = [
        max(0.0, statistics.mean(summaries[tag][t].peak_alloc_mb for t in sorted_tasks)
            - statistics.mean(summaries[tag][t].init_peak_mb for t in sorted_tasks))
        for tag in tags
    ]
    fig, ax = plt.subplots(figsize=(13.0, max(4.0, 1.0 * len(tags) + 1.8)))
    y_pos = list(range(len(tags)))
    mem_colors = ("#4a6fa5", "#e09f3e")  # weights (cool) + transient (warm)
    mem_labels = ("Model weights (init peak, persistent)",
                  "Activations + KV cache + scratch (transient peak)")
    bar_max_mem = max(w + t for w, t in zip(seg_weights, seg_transient))
    # Segment 1: weights.
    ax.barh(y_pos, seg_weights, color=mem_colors[0],
            label=mem_labels[0], edgecolor="white",
            linewidth=1.0, height=0.85)
    for i, w in enumerate(seg_weights):
        if w >= 0.10 * bar_max_mem:
            ax.text(w / 2, i,
                    f"weights\n{w/1024:.1f} GB\n({w / (w + seg_transient[i]) * 100:.1f}%)",
                    ha="center", va="center", fontsize=9.5,
                    color="white", fontweight="bold")
    # Segment 2: transient.
    ax.barh(y_pos, seg_transient, left=seg_weights, color=mem_colors[1],
            label=mem_labels[1], edgecolor="white",
            linewidth=1.0, height=0.85)
    for i, (t, w) in enumerate(zip(seg_transient, seg_weights)):
        if t >= 0.05 * bar_max_mem:
            tot = w + t
            ax.text(w + t / 2, i,
                    f"transient\n{t/1024:.1f} GB\n({t / tot * 100:.1f}%)",
                    ha="center", va="center", fontsize=9.5,
                    color="#1a1a1a", fontweight="bold")
    # Total label at bar end.
    for i, (w, t) in enumerate(zip(seg_weights, seg_transient)):
        tot = w + t
        ax.text(tot + 0.012 * bar_max_mem, i, f"{tot/1024:.1f} GB total",
                va="center", fontsize=10, color="#333", fontweight="bold")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tags, fontsize=11, fontweight="bold")
    ax.set_ylim(-0.6, len(tags) - 0.4)
    ax.invert_yaxis()
    ax.set_xlabel("VRAM peak alloc (MB)", fontsize=10)
    ax.set_title("Per-variant VRAM peak breakdown "
                 "(model weights vs runtime transient)",
                 fontsize=12, pad=8)
    ax.legend(loc="lower right", framealpha=0.92, fontsize=9.5)
    ax.set_xlim(0, bar_max_mem * 1.18)
    ax.grid(True, axis="x", alpha=0.20)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.subplots_adjust(left=0.15, bottom=0.18)
    fig.text(0.5, 0.02,
             "Weights = init_peak_mb (PerfProbe samples allocator right "
             "after model load). Transient = (max stage peak_alloc) - "
             "(init peak), aggregating activations + KV cache + scratch + "
             "fragmentation across the 5 stages (init / text_encoder / "
             "vae_encode / transformer / action_head); on this workload "
             "the transient peak falls in vae_encode.",
             ha="center", va="bottom", fontsize=8.0, color="#444", wrap=True)
    p_mem = os.path.join(plots_dir, "memory_breakdown.png")
    fig.savefig(p_mem, dpi=130)
    plt.close(fig)
    charts.append(("VRAM peak breakdown per variant "
                   "(model weights vs transient activations + KV)",
                   "plots/memory_breakdown.png"))

    return charts


# --------------------------------------------------------------------------
# Op-type FLOPs estimation + chart
# --------------------------------------------------------------------------

# LingBot-VA architecture constants (paper §4.2 + ptq.py inspection).
# Encapsulated so a future profiler-based replacement can swap them out
# without disturbing the rest of the chart pipeline.
_ARCH = dict(
    d_v=3072,             # video stream hidden dim (Wan2.2-5B backbone)
    d_a=768,              # action stream hidden dim
    n_layers=30,
    ffn_inner_v=14336,    # video ffn inner dim
    ffn_inner_a=3584,     # action ffn inner dim (4x smaller)
    n_tokens_per_frame=192,  # spatial tokens per frame after VAE+patchify
    chunk_K=4,            # video frames per chunk
    tau=4,                # action tokens per video frame
    video_steps=3,        # Euler steps to s=0.5 (Algorithm 1 line 4)
    action_steps=10,      # Euler steps to s=1.0 (Algorithm 1 line 7)
    video_cfg_forwards=2, # CFG=5.0 -> cond+uncond batched as 2 forwards
    action_cfg_forwards=1,  # CFG=1.0 -> no CFG batch
    text_seq_len=256,     # T5 text encoder typical output length
)


def _estimate_op_pf(tag: str) -> dict:
    """Return {linear, attention, other} in PF (BF16-equivalent FLOPs)
    per inference call. Architecture-based; assumes the workload is
    one chunk of K video frames + tau*K action tokens.

    BF16-equivalent: w8a8 Linear FLOPs are divided by 4 since INT8 TC
    runs 4x faster per FLOP than BF16 TC on A6000 (154.8 TOPS vs 38.7
    TFLOPS). Attention stays in BF16 SDPA across all variants so its
    PF is variant-independent."""
    a = _ARCH
    # Linear FLOPs per token per layer (4 self-attn + 2 ffn + 4 cross-attn
    # to text encoder, matching ViDiT-Q remain_fp_regex scope).
    lin_per_tok_v = (8 * a["d_v"]**2                  # self-attn (q/k/v/out)
                     + 4 * a["d_v"] * a["ffn_inner_v"]  # ffn up + down
                     + 8 * a["d_v"]**2)               # cross-attn attn2
    lin_per_tok_a = (8 * a["d_a"]**2
                     + 4 * a["d_a"] * a["ffn_inner_a"]
                     + 8 * a["d_a"]**2)
    # Video stream per chunk: K frames * N tokens/frame video tokens,
    # all 30 layers, video_steps * cfg_forwards forward passes.
    n_video_tok = a["chunk_K"] * a["n_tokens_per_frame"]
    n_video_fwd = a["video_steps"] * a["video_cfg_forwards"]
    video_linear_flops = n_video_fwd * a["n_layers"] * n_video_tok * lin_per_tok_v
    # Action stream per chunk: tau*K action tokens, all 30 layers,
    # action_steps * cfg_forwards forward passes.
    n_action_tok = a["tau"] * a["chunk_K"]
    n_action_fwd = a["action_steps"] * a["action_cfg_forwards"]
    action_linear_flops = n_action_fwd * a["n_layers"] * n_action_tok * lin_per_tok_a
    # Cross-modal projection in MoT (action <-> video dim each layer).
    n_cross_proj = 2 * a["d_v"] * a["d_a"]  # 2 Linears (in + out)
    cross_proj_flops = n_action_fwd * a["n_layers"] * n_action_tok * 2 * n_cross_proj
    linear_flops = video_linear_flops + action_linear_flops + cross_proj_flops

    # Attention FLOPs (self-attn within chunk; joint attn for action).
    # Self-attn: 4 * S^2 * d  (Q.K^T + softmax-weighted V, both 2*S*S*d).
    video_self_attn = n_video_fwd * a["n_layers"] * 4 * n_video_tok**2 * a["d_v"]
    # Action joint attn uses video-dim space; combined seq len = video + action.
    joint_seq = n_video_tok + n_action_tok
    action_joint_attn = n_action_fwd * a["n_layers"] * 4 * joint_seq**2 * a["d_v"]
    # Cross-attn to text encoder (video + action queries vs text K/V).
    video_text_attn = n_video_fwd * a["n_layers"] * 4 * n_video_tok * a["text_seq_len"] * a["d_v"]
    action_text_attn = n_action_fwd * a["n_layers"] * 4 * n_action_tok * a["text_seq_len"] * a["d_a"]
    attn_flops = video_self_attn + action_joint_attn + video_text_attn + action_text_attn

    # BF16-equivalent: a w8a8 Linear with N raw FLOPs takes N / INT8_TC
    # seconds; the same N FLOPs would take N / BF16_TC seconds on the
    # bf16 path. The "bf16-equivalent" view scales so they share the
    # bf16 time axis: bf16_eq_FLOPs = N * (BF16_TC / INT8_TC) = N / 2.
    # Attention stays bf16 either way (no int8 attention kernel).
    is_w8a8 = (_weight_bytes_lookup(tag) < 2.0)
    linear_bf16eq = linear_flops / 2.0 if is_w8a8 else linear_flops

    TF = 1e12
    return {
        "linear": linear_bf16eq / TF,
        "attention": attn_flops / TF,
        "memcpy": 0.0,    # not knowable from architecture; profiler-only
        "other": 0.0,
    }


def _weight_bytes_lookup(tag: str) -> float:
    """Mirror of _weight_bytes inner-fn from roofline; bf16/fp16 -> 2.0,
    everything else (w8a8 / int8 quant) -> 1.0."""
    return 2.0 if tag.lower() in ("bf16", "fp16") else 1.0


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
        "plots/compute_breakdown.png",
        "plots/op_breakdown.png",
        "plots/op_breakdown_measured.png",  # only present when --op_profile
        "plots/memory_breakdown.png",
        "plots/roofline.png",
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

    charts: list[tuple[str, str]] = []
    if not args.no_plots:
        charts = _make_plots(args.out_dir, variant_specs, baseline_tag,
                             tasks, summaries, step_limits)

        # Optionally overlay a profiler-measured op_breakdown chart
        # alongside the architecture-estimated default. Each
        # --op_profile entry is TAG=PATH; PATH points at an
        # aggregator-produced op_profile.json.
        op_data = _collect_op_profile(args.op_profile)
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
                    f"Measured kernel time per inference call, aggregated "
                    f"by torch.profiler over the first 5 post-warmup "
                    f"infer() calls of each task and averaged across tasks "
                    f"(sources: {srcs}). Linear = cuBLAS/cuBLASLt + W8A8 "
                    f"GEMM kernels; Attention = SDPA / fused mha + softmax "
                    f"kernels; Other = elementwise + layernorm + launch. "
                    f"Profiler overhead inflates absolute ms -- compare "
                    f"op-share within a variant, not absolute speed."
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
