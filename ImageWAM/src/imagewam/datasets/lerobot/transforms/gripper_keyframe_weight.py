"""Per-timestep loss multiplier that upweights gripper open/close transitions.

Grasp and release are brief, near-binary events that MSE averaging over the action
horizon drowns out; the frames around each transition are upweighted so the model
commits to them. Mirrors the intent of openpi's examples/agibot_g2/frame_weights.py,
adapted for G1: the G1 gripper is a continuous position channel (not a binary suction
command), so the signal is binarized by a threshold before edge detection, and both
directions are treated as key events (close = open->closed, release = closed->open).

The weight is derived from the raw action chunk [T, D] and returned as [T]. It is a
final multiplier, not a flag: base frames get `base_weight`, frames within the window
of any transition get `base_weight * key_weight`. When two grippers transition in
overlapping windows the larger weight wins (max, not product).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class GripperKeyframeWeight:
    """Build the per-timestep loss multiplier for one action chunk.

    Args:
        gripper_dims: indices of the gripper channels in the raw action layout.
        thresholds: binarization threshold per gripper dim, same length as gripper_dims.
            A value above the threshold is "open" (1), at or below is "closed" (0).
        pre: frames before a transition to include in the upweight window.
        post: frames after a transition to include.
        key_weight: multiplier applied inside a transition window.
        base_weight: multiplier applied everywhere else.
    """

    gripper_dims: Sequence[int]
    thresholds: Sequence[float]
    pre: int = 4
    post: int = 4
    key_weight: float = 3.0
    base_weight: float = 1.0

    def __post_init__(self) -> None:
        if len(self.gripper_dims) != len(self.thresholds):
            raise ValueError(
                f"gripper_dims and thresholds must match, "
                f"got {len(self.gripper_dims)} and {len(self.thresholds)}"
            )
        if self.pre < 0 or self.post < 0:
            raise ValueError(f"pre/post must be non-negative, got {self.pre}/{self.post}")

    def __call__(self, action: Tensor) -> Tensor:
        """action [T, D] raw -> weight [T] float32."""
        if action.ndim != 2:
            raise ValueError(f"Expected action [T, D], got {tuple(action.shape)}")
        num_frames = action.shape[0]
        weight = torch.full((num_frames,), float(self.base_weight), dtype=torch.float32)
        if num_frames < 2:
            return weight

        key = torch.zeros(num_frames, dtype=torch.bool)
        for dim, threshold in zip(self.gripper_dims, self.thresholds):
            binary = action[:, dim] > float(threshold)
            # A transition sits between frame t-1 and t; flag frame t as its anchor.
            changed = binary[1:] != binary[:-1]
            for anchor in torch.nonzero(changed, as_tuple=False).flatten().tolist():
                onset = anchor + 1
                lo = max(0, onset - self.pre)
                hi = min(num_frames, onset + self.post + 1)
                key[lo:hi] = True

        weight[key] = float(self.base_weight) * float(self.key_weight)
        return weight
