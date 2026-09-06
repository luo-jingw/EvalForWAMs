#!/usr/bin/env python3
"""Build a LeRobot frame-filter JSON that excludes outlier episodes of the G1 dataset.

Two explicit exclusion rules:
  1. task-level   episodes whose task is listed in DROP_TASKS.
  2. zscore-level episodes with any |z| > Z_THRESHOLD, evaluated on the remainder.

The z-score statistics reproduce BaseLerobotDataset.get_dataset_stats, which reads
every episode regardless of this filter. The statistics are therefore computed over
all episodes, while the filter only removes episodes from sampling.

Output is consumed by `data.train.nonidle_filter_path`. An episode mapped to an empty
range list contributes zero frames; see lerobot_dataset.py::_load_nonidle_filter.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq
from numpy.typing import NDArray

ACTION_HORIZON: int = 16
Z_THRESHOLD: float = 5.0
DROP_TASKS: frozenset[str] = frozenset({"pick_red_bottle"})

ACTION_COLUMN: str = "action"
STD_REGULARIZER: float = 1e-8

DropReason = Literal["task", "zscore"]


@dataclass(frozen=True)
class EpisodeRecord:
    episode_index: int
    task: str
    action: NDArray[np.float64]


@dataclass(frozen=True)
class ActionStats:
    mean: NDArray[np.float64]
    std: NDArray[np.float64]


@dataclass(frozen=True)
class DroppedEpisode:
    episode_index: int
    task: str
    num_frames: int
    reason: DropReason
    max_abs_z: float
    argmax_dim: int
    outlier_frame_ratio: float


@dataclass(frozen=True)
class FrameFilterReport:
    output_path: Path
    total_episodes: int
    dropped: tuple[DroppedEpisode, ...]
    kept_frames: int
    total_frames: int
    kept_abs_z_max: float


def _episode_task_map(dataset_root: Path) -> dict[int, str]:
    tasks: dict[int, str] = {}
    with (dataset_root / "meta" / "episodes.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            tasks[int(record["episode_index"])] = str(record["tasks"][0])
    return tasks


def load_episode_records(dataset_root: Path) -> tuple[EpisodeRecord, ...]:
    """Read the action column of every episode parquet, in episode-index order."""
    task_by_index = _episode_task_map(dataset_root)
    records: list[EpisodeRecord] = []
    for episode_index in sorted(task_by_index):
        path = (
            dataset_root
            / "data"
            / f"chunk-{episode_index // 1000:03d}"
            / f"episode_{episode_index:06d}.parquet"
        )
        column = pq.read_table(path)[ACTION_COLUMN].to_numpy(zero_copy_only=False)
        action = np.stack(column).astype(np.float64)
        records.append(
            EpisodeRecord(
                episode_index=episode_index,
                task=task_by_index[episode_index],
                action=action,
            )
        )
    return tuple(records)


def sliding_window_with_replication(action: NDArray[np.float64], horizon: int) -> NDArray[np.float64]:
    """Mirror base_lerobot_dataset.sliding_window_with_replication: [N, D] -> [N, horizon, D]."""
    num_frames = action.shape[0]
    offsets = np.arange(num_frames)[:, None] + np.arange(horizon)[None, :]
    return action[np.clip(offsets, 0, num_frames - 1)]


def compute_action_stats(records: Sequence[EpisodeRecord], horizon: int) -> ActionStats:
    """Reproduce the z-score branch of BaseLerobotDataset.get_dataset_stats.

    Per episode the sliding-window tensor [N, horizon, D] is reduced to a per-step
    mean and variance. The pooled statistics are then averaged over the episode axis
    and the horizon axis. `ddof=1` matches torch.Tensor.var, which is unbiased.
    """
    means: list[NDArray[np.float64]] = []
    variances: list[NDArray[np.float64]] = []
    for record in records:
        window = sliding_window_with_replication(record.action, horizon)
        means.append(window.mean(axis=0))
        variances.append(window.var(axis=0, ddof=1))
    stacked_means = np.stack(means)
    stacked_variances = np.stack(variances)
    global_mean = stacked_means.mean(axis=(0, 1))
    global_std = np.sqrt(
        (stacked_variances + (stacked_means - global_mean) ** 2).mean(axis=(0, 1))
    )
    return ActionStats(mean=global_mean, std=global_std)


def _abs_z(action: NDArray[np.float64], stats: ActionStats) -> NDArray[np.float64]:
    return np.abs((action - stats.mean) / (stats.std + STD_REGULARIZER))


def find_dropped_episodes(
    records: Sequence[EpisodeRecord],
    stats: ActionStats,
    drop_tasks: frozenset[str],
    threshold: float,
) -> tuple[DroppedEpisode, ...]:
    """Apply the task rule first, then the z-score rule on the remaining episodes."""
    dropped: list[DroppedEpisode] = []
    for record in records:
        abs_z = _abs_z(record.action, stats)
        max_abs_z = float(abs_z.max())
        argmax_dim = int(abs_z.max(axis=0).argmax())
        outlier_frame_ratio = float((abs_z.max(axis=1) > threshold).mean())
        if record.task in drop_tasks:
            reason: DropReason = "task"
        elif max_abs_z > threshold:
            reason = "zscore"
        else:
            continue
        dropped.append(
            DroppedEpisode(
                episode_index=record.episode_index,
                task=record.task,
                num_frames=int(record.action.shape[0]),
                reason=reason,
                max_abs_z=max_abs_z,
                argmax_dim=argmax_dim,
                outlier_frame_ratio=outlier_frame_ratio,
            )
        )
    return tuple(dropped)


def build_frame_filter(
    dataset_root: Path,
    output_path: Path,
    drop_tasks: frozenset[str],
    threshold: float,
    horizon: int,
) -> tuple[FrameFilterReport, ActionStats]:
    records = load_episode_records(dataset_root)
    stats = compute_action_stats(records, horizon)
    dropped = find_dropped_episodes(records, stats, drop_tasks, threshold)
    dropped_indices = {item.episode_index for item in dropped}

    kept = [record for record in records if record.episode_index not in dropped_indices]
    kept_abs_z_max = max(float(_abs_z(record.action, stats).max()) for record in kept)

    payload = {
        "format": "imagewam_nonidle_ranges_v1",
        "dataset_dir": str(dataset_root),
        "rules": {
            "drop_tasks": sorted(drop_tasks),
            "z_threshold": threshold,
            "action_horizon": horizon,
        },
        "summary": {
            "total_episodes": len(records),
            "dropped_episodes": len(dropped),
            "total_frames": sum(int(record.action.shape[0]) for record in records),
            "kept_frames": sum(int(record.action.shape[0]) for record in kept),
            "kept_abs_z_max": kept_abs_z_max,
        },
        "episodes": {str(index): [] for index in sorted(dropped_indices)},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    report = FrameFilterReport(
        output_path=output_path,
        total_episodes=len(records),
        dropped=dropped,
        kept_frames=int(payload["summary"]["kept_frames"]),
        total_frames=int(payload["summary"]["total_frames"]),
        kept_abs_z_max=kept_abs_z_max,
    )
    return report, stats


def _print_report(report: FrameFilterReport, stats: ActionStats, threshold: float) -> None:
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    print("action global_mean", stats.mean)
    print("action global_std ", stats.std)

    by_reason: Mapping[DropReason, list[DroppedEpisode]] = {
        "task": [item for item in report.dropped if item.reason == "task"],
        "zscore": [item for item in report.dropped if item.reason == "zscore"],
    }
    task_dropped = by_reason["task"]
    print(f"\nreason=task   dropped {len(task_dropped)} episodes")
    task_names = sorted({item.task for item in task_dropped})
    for name in task_names:
        count = sum(1 for item in task_dropped if item.task == name)
        frames = sum(item.num_frames for item in task_dropped if item.task == name)
        print(f"  {name:24s} episodes={count:3d} frames={frames}")

    print(f"\nreason=zscore dropped {len(by_reason['zscore'])} episodes")
    for item in sorted(by_reason["zscore"], key=lambda entry: entry.max_abs_z, reverse=True):
        print(
            f"  ep{item.episode_index:03d} {item.task:24s} frames={item.num_frames:5d} "
            f"max_abs_z={item.max_abs_z:6.2f} dim={item.argmax_dim:2d} "
            f"outlier_frame_ratio={item.outlier_frame_ratio * 100:5.1f}%"
        )

    kept_ratio = report.kept_frames / max(report.total_frames, 1)
    print(
        f"\nkept_episodes={report.total_episodes - len(report.dropped)}/{report.total_episodes} "
        f"kept_frames={report.kept_frames}/{report.total_frames} ({kept_ratio * 100:.2f}%)"
    )
    print(f"kept_abs_z_max={report.kept_abs_z_max:.2f} (clamp boundary {threshold})")
    print(f"wrote {report.output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--z-threshold", type=float, default=Z_THRESHOLD)
    parser.add_argument("--action-horizon", type=int, default=ACTION_HORIZON)
    args = parser.parse_args()

    report, stats = build_frame_filter(
        dataset_root=args.dataset_root,
        output_path=args.output,
        drop_tasks=DROP_TASKS,
        threshold=args.z_threshold,
        horizon=args.action_horizon,
    )
    _print_report(report, stats, args.z_threshold)


if __name__ == "__main__":
    main()
