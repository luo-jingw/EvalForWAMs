#!/usr/bin/env python3
"""Expand the ImageWAM RoboTwin checkpoint from 14 action dimensions to G1's 16.

ImageWAM.load_checkpoint loads `mot` with strict=False and `proprio_encoder` with
strict=True. Neither tolerates a shape mismatch, so the three 14-wide tensors are
rewritten before training.

RoboTwin's 14 dims are [left arm 6, left gripper, right arm 6, right gripper];
G1's 16 dims are [left arm 7, right arm 7, left gripper, right gripper]. Slices are
therefore remapped by G1_TO_SRC_DIM rather than prefix-copied. The two wrist-yaw
dims have no RoboTwin counterpart and are zero-initialised.

Every other tensor is passed through untouched, including the shared storage of the
video expert, so the output file stays the same size as the input.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor

SRC_ACTION_DIM: int = 14
DST_ACTION_DIM: int = 16

# index i holds the RoboTwin dim feeding G1 dim i, or None when there is no counterpart.
G1_TO_SRC_DIM: tuple[int | None, ...] = (
    0, 1, 2, 3, 4, 5, None,      # G1 left arm 7  <- RoboTwin left arm 6, left wrist yaw unmapped
    7, 8, 9, 10, 11, 12, None,   # G1 right arm 7 <- RoboTwin right arm 6, right wrist yaw unmapped
    6,                           # G1 left gripper  <- RoboTwin dim 6
    13,                          # G1 right gripper <- RoboTwin dim 13
)

# (payload section, tensor key inside that section, axis carrying the action dim)
EXPAND_SPEC: tuple[tuple[str, str, int], ...] = (
    ("mot", "mixtures.action.action_encoder.weight", 1),
    ("mot", "mixtures.action.head.linear.weight", 0),
    ("proprio_encoder", "weight", 1),
)


@dataclass(frozen=True)
class TensorExpansion:
    section: str
    key: str
    axis: int
    src_shape: tuple[int, ...]
    dst_shape: tuple[int, ...]
    zero_init_indices: tuple[int, ...]
    src_absmax: float
    dst_absmax: float


@dataclass(frozen=True)
class ExpandReport:
    src_path: Path
    dst_path: Path
    step: int
    expansions: tuple[TensorExpansion, ...]


def expand_along_axis(
    tensor: Tensor,
    axis: int,
    index_map: Sequence[int | None],
    dst_dim: int,
) -> Tensor:
    """Build a tensor whose `axis` has length `dst_dim`, filled per `index_map`."""
    if tensor.shape[axis] != SRC_ACTION_DIM:
        raise ValueError(
            f"Expected axis {axis} of size {SRC_ACTION_DIM}, got {tuple(tensor.shape)}"
        )
    if len(index_map) != dst_dim:
        raise ValueError(f"index_map length {len(index_map)} does not match dst_dim {dst_dim}")

    dst_shape = list(tensor.shape)
    dst_shape[axis] = dst_dim
    out = torch.zeros(dst_shape, dtype=tensor.dtype)
    for dst_index, src_index in enumerate(index_map):
        if src_index is None:
            continue
        out.select(axis, dst_index).copy_(tensor.select(axis, src_index))
    return out


def expand_checkpoint(src_path: Path, dst_path: Path) -> ExpandReport:
    payload: dict[str, Any] = torch.load(src_path, map_location="cpu", weights_only=False)
    zero_init_indices = tuple(
        index for index, source in enumerate(G1_TO_SRC_DIM) if source is None
    )

    expansions: list[TensorExpansion] = []
    for section, key, axis in EXPAND_SPEC:
        source_tensor = payload[section][key]
        expanded = expand_along_axis(source_tensor, axis, G1_TO_SRC_DIM, DST_ACTION_DIM)
        payload[section][key] = expanded
        expansions.append(
            TensorExpansion(
                section=section,
                key=key,
                axis=axis,
                src_shape=tuple(source_tensor.shape),
                dst_shape=tuple(expanded.shape),
                zero_init_indices=zero_init_indices,
                src_absmax=float(source_tensor.abs().max()),
                dst_absmax=float(expanded.abs().max()),
            )
        )

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, dst_path)
    return ExpandReport(
        src_path=src_path,
        dst_path=dst_path,
        step=int(payload["step"]),
        expansions=tuple(expansions),
    )


def _print_report(report: ExpandReport) -> None:
    print(f"src  {report.src_path}")
    print(f"dst  {report.dst_path}")
    print(f"step {report.step}")
    print(f"dim map {G1_TO_SRC_DIM}")
    for expansion in report.expansions:
        print(f"\n{expansion.section}/{expansion.key}")
        print(f"  axis={expansion.axis} {expansion.src_shape} -> {expansion.dst_shape}")
        print(f"  zero_init_indices={expansion.zero_init_indices}")
        print(f"  absmax {expansion.src_absmax:.6f} -> {expansion.dst_absmax:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src_path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = expand_checkpoint(args.src_path, args.output)
    _print_report(report)


if __name__ == "__main__":
    main()
