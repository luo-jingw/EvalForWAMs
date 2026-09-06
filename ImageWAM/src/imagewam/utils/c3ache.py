"""Cross-chunk residual cache (C3ache) for flow-matching action inference.

Training-free acceleration from "C3ache: Accelerating World Action Models with
Cross Inference Chunk Cache".  At a fixed denoising step ``k``, cache the MoT
stack residual ``R = h_L - h_0`` on refresh replans and reuse it on subsequent
replans while the observation embedding ``h_0`` is recomputed from the current
chunk context.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class C3acheController:
    """Stateful cross-replan residual cache for one eval episode."""

    num_steps: int
    refresh_interval: int = 4
    cache_max_step: int = 6
    replan_index: int = 0
    _residual_cache: dict[int, torch.Tensor] = field(default_factory=dict, init=False, repr=False)
    _refresh_replan: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        steps = int(self.num_steps)
        if steps <= 0:
            raise ValueError(f"`num_steps` must be positive, got {steps}.")
        self.num_steps = steps
        self.refresh_interval = int(self.refresh_interval)
        self.cache_max_step = int(self.cache_max_step)
        if self.cache_max_step < 0 or self.cache_max_step >= steps:
            raise ValueError(
                f"`cache_max_step` must be in [0, num_steps), got {self.cache_max_step} "
                f"for num_steps={steps}."
            )

    def reset_episode(self) -> None:
        self.replan_index = 0
        self._residual_cache.clear()
        self._refresh_replan = False

    def begin_replan(self) -> None:
        if self.refresh_interval == 0:
            self._refresh_replan = self.replan_index == 0
        else:
            self._refresh_replan = self.replan_index % self.refresh_interval == 0

    def end_replan(self) -> None:
        self.replan_index += 1

    def should_use_cache(self, step_index: int) -> bool:
        step = int(step_index)
        if step > self.cache_max_step:
            return False
        if self._refresh_replan:
            return False
        return step in self._residual_cache

    def should_store_residual(self, step_index: int) -> bool:
        return int(step_index) <= self.cache_max_step and self._refresh_replan

    def get_cached_residual(self, step_index: int) -> torch.Tensor:
        return self._residual_cache[int(step_index)]

    def store_residual(self, step_index: int, residual: torch.Tensor) -> None:
        step = int(step_index)
        if step > self.cache_max_step:
            return
        self._residual_cache[step] = residual.detach().clone()
