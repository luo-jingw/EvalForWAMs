#!/usr/bin/env python3
"""Rewrite the G1 dataset task names into natural-language instructions.

meta/tasks.jsonl ships snake_case identifiers, which reach the Qwen3 text encoder
verbatim through RobotVideoDataset. The pretrained LIBERO/RoboTwin checkpoints were
trained on natural-language instructions, so the identifiers are rewritten in place.

Only the `task` field is rewritten. `task_index` is preserved so the parquet
`task_index` column stays valid. Tasks excluded by the frame filter keep their rows.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

G1_TASK_TEXT: Mapping[str, str] = {
    "open_lid_add_potato": "open the lid and put the potato into the pot",
    "pick_red_bottle": "pick up the red bottle",
    "pick_and_move_bottle": "pick up the bottle and move it to the target position",
    "put_carrot_n_cup": "put the carrot into the cup",
    "put_cup_n_broccoli": "put the cup and the broccoli on the plate",
}


@dataclass(frozen=True)
class TaskRewrite:
    task_index: int
    before: str
    after: str


@dataclass(frozen=True)
class TaskRewriteReport:
    tasks_path: Path
    rewrites: tuple[TaskRewrite, ...]


def rewrite_tasks_jsonl(dataset_root: Path, mapping: Mapping[str, str]) -> TaskRewriteReport:
    """Overwrite meta/tasks.jsonl with natural-language task strings."""
    tasks_path = dataset_root / "meta" / "tasks.jsonl"
    records: list[dict[str, object]] = []
    with tasks_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    rewrites: list[TaskRewrite] = []
    for record in records:
        before = str(record["task"])
        after = mapping.get(before, before)
        record["task"] = after
        rewrites.append(
            TaskRewrite(task_index=int(record["task_index"]), before=before, after=after)
        )

    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    tasks_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return TaskRewriteReport(tasks_path=tasks_path, rewrites=tuple(rewrites))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    args = parser.parse_args()

    report = rewrite_tasks_jsonl(args.dataset_root, G1_TASK_TEXT)
    for rewrite in report.rewrites:
        changed = "changed" if rewrite.before != rewrite.after else "unchanged"
        print(f"task_index={rewrite.task_index} {changed}")
        print(f"  before: {rewrite.before}")
        print(f"  after : {rewrite.after}")
    print(f"wrote {report.tasks_path}")


if __name__ == "__main__":
    main()
