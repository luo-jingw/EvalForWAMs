#!/usr/bin/env python3
"""Print observations for samples drawn from the G1 training dataset.

Observation only: every value is reported as measured. No thresholds, no pass/fail.

Instantiates data.train from the task config, so it exercises the same path the
trainer uses: LeRobot loading, the outlier frame filter, video decoding, camera
composition, normalization and the action/state merger.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from imagewam.utils import misc
from imagewam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()


@dataclass(frozen=True)
class SampleObservation:
    index: int
    elapsed_s: float
    video_shape: tuple[int, ...]
    video_min: float
    video_max: float
    action_shape: tuple[int, ...]
    action_absmax: float
    proprio_shape: tuple[int, ...]
    proprio_absmax: float
    action_pad_ratio: float
    image_pad_ratio: float
    action_dim_is_pad_sum: int
    instruction: str


def _load_config(task: str, overrides: Sequence[str]) -> DictConfig:
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base="1.3"):
        return compose(config_name="train", overrides=[f"task={task}", *overrides])


def _observe_sample(dataset: Any, index: int) -> SampleObservation:
    start = time.perf_counter()
    sample = dataset[index]
    elapsed = time.perf_counter() - start
    video: torch.Tensor = sample["video"]
    action: torch.Tensor = sample["action"]
    proprio: torch.Tensor = sample["proprio"]
    action_dim_is_pad = sample.get("action_dim_is_pad")
    return SampleObservation(
        index=index,
        elapsed_s=elapsed,
        video_shape=tuple(video.shape),
        video_min=float(video.min()),
        video_max=float(video.max()),
        action_shape=tuple(action.shape),
        action_absmax=float(action.abs().max()),
        proprio_shape=tuple(proprio.shape),
        proprio_absmax=float(proprio.abs().max()),
        action_pad_ratio=float(torch.as_tensor(sample["action_is_pad"]).float().mean()),
        image_pad_ratio=float(torch.as_tensor(sample["image_is_pad"]).float().mean()),
        action_dim_is_pad_sum=int(torch.as_tensor(action_dim_is_pad).sum())
        if action_dim_is_pad is not None
        else -1,
        instruction=str(sample["instruction"]),
    )


def observe(
    task: str,
    num_samples: int,
    overrides: Sequence[str],
    seed: int,
    work_dir: Path,
) -> None:
    # RobotVideoDataset writes the recomputed norm stats into the registered work dir.
    misc.register_work_dir(work_dir)
    cfg = _load_config(task, overrides)
    print("data.train.dataset_dirs      ", OmegaConf.to_container(cfg.data.train.dataset_dirs))
    print("data.train.nonidle_filter_path", cfg.data.train.nonidle_filter_path)
    print("data.train.pretrained_norm_stats", cfg.data.train.pretrained_norm_stats)

    dataset = instantiate(cfg.data.train)
    print(f"len(dataset) = {len(dataset)}")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(0, len(dataset), (num_samples,), generator=generator).tolist()

    observations: list[SampleObservation] = []
    for index in indices:
        observation = _observe_sample(dataset, index)
        observations.append(observation)
        print(f"\nidx={observation.index} elapsed={observation.elapsed_s:.3f}s")
        print(
            f"  video   {observation.video_shape} "
            f"min={observation.video_min:.3f} max={observation.video_max:.3f}"
        )
        print(f"  action  {observation.action_shape} absmax={observation.action_absmax:.3f}")
        print(f"  proprio {observation.proprio_shape} absmax={observation.proprio_absmax:.3f}")
        print(
            f"  action_is_pad={observation.action_pad_ratio:.3f} "
            f"image_is_pad={observation.image_pad_ratio:.3f} "
            f"action_dim_is_pad_sum={observation.action_dim_is_pad_sum}"
        )
        print(f"  instruction: {observation.instruction}")

    print("\n--- aggregate over sampled items ---")
    print(f"  action absmax  max={max(o.action_absmax for o in observations):.3f}")
    print(f"  proprio absmax max={max(o.proprio_absmax for o in observations):.3f}")
    print(f"  elapsed_s      mean={sum(o.elapsed_s for o in observations) / len(observations):.3f}")
    print(f"  distinct instructions: {len({o.instruction for o in observations})}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="g1_flux2_klein_4b_base_imagewam")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--work-dir", type=Path, default=REPO_ROOT / "runs" / "observe_g1_sample")
    parser.add_argument("overrides", nargs="*", default=[])
    args = parser.parse_args()

    observe(
        task=args.task,
        num_samples=args.num_samples,
        overrides=args.overrides,
        seed=args.seed,
        work_dir=args.work_dir,
    )


if __name__ == "__main__":
    main()
