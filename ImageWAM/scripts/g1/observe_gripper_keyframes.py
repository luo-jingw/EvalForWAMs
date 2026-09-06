#!/usr/bin/env python3
"""Observe the gripper-transition loss weight the training pipeline emits.

Mirrors openpi's scripts/probe_agibot_batch.py: build the training dataset for a
task config, draw samples, and print shapes/stats of action_loss_weight. Pure
observation, no assertions. If the config has no gripper_keyframe_weight, the field
is absent and that is reported.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from imagewam.utils import misc
from imagewam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()


def observe(task: str, num_samples: int, overrides: Sequence[str], seed: int, work_dir: Path) -> None:
    misc.register_work_dir(work_dir)
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=[f"task={task}", *overrides])

    dataset = instantiate(cfg.data.train)
    print(f"len(dataset) = {len(dataset)}")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(0, len(dataset), (num_samples,), generator=generator).tolist()

    flagged = 0
    total = 0
    for index in indices:
        sample = dataset[index]
        weight = sample.get("action_loss_weight")
        if weight is None:
            print(f"idx={index}: action_loss_weight absent (weighting disabled)")
            continue
        weight = np.asarray(weight)
        key = int((weight > weight.min()).sum())
        flagged += key
        total += weight.size
        print(
            f"idx={index}: shape={tuple(weight.shape)} "
            f"min={weight.min():.1f} max={weight.max():.1f} "
            f"key_frames={key}/{weight.size} "
            f"weight={np.array2string(weight, precision=1, max_line_width=200)}"
        )

    if total:
        print(
            f"\nflagged frames across {num_samples} samples: "
            f"{flagged}/{total} = {flagged / total:.1%}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="g1_stack_cubes_flux2_klein_4b")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--work-dir", type=Path, default=REPO_ROOT / "runs" / "observe_keyframes")
    parser.add_argument("overrides", nargs="*", default=[])
    args = parser.parse_args()
    observe(args.task, args.num_samples, args.overrides, args.seed, args.work_dir)


if __name__ == "__main__":
    main()
