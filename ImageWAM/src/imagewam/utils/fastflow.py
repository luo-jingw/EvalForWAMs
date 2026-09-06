"""Bandit controller and finite-difference helpers for FastFlow inference.

This implements the training-free method from "FastFlow: Accelerating the
Generative Flow Matching Models with Bandit Inference" (ICLR 2026).  The
controller intentionally stores only Python scalars so that model tensors are
never retained between generations.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


def default_fastflow_arms(num_steps: int) -> tuple[int, ...]:
    """Return the paper's arm set, filtered for the requested horizon.

    The paper uses ``[0, 1, 2, 3]`` for 10-step inference and
    ``[0, 2, 4, 6]`` for 25/50-step inference.  An arm is the number of model
    evaluations skipped before the next exact velocity evaluation.
    """

    steps = int(num_steps)
    if steps < 3:
        raise ValueError(f"FastFlow requires at least 3 inference steps, got {steps}.")
    candidates = (0, 1, 2, 3) if steps <= 10 else (0, 2, 4, 6)
    return tuple(arm for arm in candidates if arm <= steps - 2)


def normalize_fastflow_arms(arms: Sequence[int] | None, *, num_steps: int) -> tuple[int, ...]:
    """Validate a custom arm set or resolve the paper default."""

    if arms is None:
        return default_fastflow_arms(num_steps)
    normalized = tuple(sorted({int(arm) for arm in arms}))
    if not normalized:
        raise ValueError("`fastflow_arms` must not be empty.")
    if normalized[0] < 0:
        raise ValueError(f"`fastflow_arms` must be non-negative, got {normalized}.")
    if 0 not in normalized:
        raise ValueError(f"`fastflow_arms` must include arm 0, got {normalized}.")
    if normalized[-1] > int(num_steps) - 2:
        raise ValueError(
            "A FastFlow arm cannot skip past the final exact velocity index: "
            f"max arm={normalized[-1]}, num_steps={num_steps}."
        )
    return normalized


def finite_difference_velocity(
    velocity_current: torch.Tensor,
    velocity_previous: torch.Tensor,
    *,
    time_current: float,
    time_previous: float,
    time_target: float,
) -> torch.Tensor:
    """Extrapolate velocity to ``time_target`` using paper Eq. (4).

    Times may be increasing or decreasing and need not be uniformly spaced,
    which is required for ImageWAM's shifted Wan schedule.
    """

    denominator = float(time_current) - float(time_previous)
    if abs(denominator) <= 1e-12:
        raise ValueError(
            "FastFlow finite differences require distinct exact-evaluation times, "
            f"got current={time_current} previous={time_previous}."
        )
    scale = (float(time_target) - float(time_current)) / denominator
    return velocity_current + (velocity_current - velocity_previous) * scale


def velocity_mse(approximation: torch.Tensor, exact: torch.Tensor) -> float:
    """Compute the scalar velocity discrepancy used by the bandit reward."""

    approx_float = approximation.detach().to(device="cpu", dtype=torch.float32)
    exact_float = exact.detach().to(device="cpu", dtype=torch.float32)
    return float(torch.mean((approx_float - exact_float) ** 2).item())


class FastFlowBanditController:
    """Per-timestep UCB bandits used by FastFlow.

    A dense first generation initializes every feasible arm from exact
    velocities, matching the paper's first-prompt initialization while avoiding
    extra model evaluations.  Later generations update an arm with the exact
    velocity observed after its skip.
    """

    def __init__(
        self,
        *,
        num_steps: int,
        arms: Sequence[int] | None = None,
        mu: float | None = None,
        gamma: float = 2.0,
    ) -> None:
        self.num_steps = int(num_steps)
        self.arms = normalize_fastflow_arms(arms, num_steps=self.num_steps)
        if mu is not None and float(mu) < 0.0:
            raise ValueError(f"`fastflow_mu` must be non-negative, got {mu}.")
        if float(gamma) < 0.0:
            raise ValueError(f"`fastflow_gamma` must be non-negative, got {gamma}.")
        self.configured_mu = None if mu is None else float(mu)
        self.gamma = float(gamma)
        self.resolved_mu: float | None = None
        self.initialized = False
        self._counts: dict[int, dict[int, int]] = {}
        self._means: dict[int, dict[int, float]] = {}

    def feasible_arms(self, step_index: int) -> tuple[int, ...]:
        """Return arms whose next exact evaluation remains on the model grid."""

        step = int(step_index)
        if not (1 <= step < self.num_steps - 1):
            return ()
        max_skip = self.num_steps - step - 2
        return tuple(arm for arm in self.arms if arm <= max_skip)

    def _ensure_step(self, step_index: int) -> None:
        step = int(step_index)
        if step not in self._counts:
            feasible = self.feasible_arms(step)
            self._counts[step] = {arm: 0 for arm in feasible}
            self._means[step] = {arm: 0.0 for arm in feasible}

    def _reward(self, arm: int, loss: float) -> float:
        if self.resolved_mu is None:
            raise RuntimeError("FastFlow controller must be initialized before computing rewards.")
        return self.resolved_mu * int(arm) - float(loss)

    def initialize_from_dense(
        self,
        *,
        times: Sequence[float],
        velocities: Sequence[torch.Tensor],
    ) -> dict[str, float | int]:
        """Warm-start all feasible arms from one full model trajectory."""

        if self.initialized:
            raise RuntimeError("FastFlow controller is already initialized.")
        if len(times) != self.num_steps or len(velocities) != self.num_steps:
            raise ValueError(
                "Dense FastFlow initialization must contain one time and velocity per inference step, "
                f"got times={len(times)} velocities={len(velocities)} num_steps={self.num_steps}."
            )

        losses: dict[tuple[int, int], float] = {}
        for step in range(1, self.num_steps - 1):
            self._ensure_step(step)
            for arm in self.feasible_arms(step):
                next_exact = step + arm + 1
                approximation = finite_difference_velocity(
                    velocities[step],
                    velocities[step - 1],
                    time_current=float(times[step]),
                    time_previous=float(times[step - 1]),
                    time_target=float(times[next_exact]),
                )
                losses[(step, arm)] = velocity_mse(approximation, velocities[next_exact])

        max_loss = max(losses.values(), default=0.0)
        # Section 4 normalizes mu from the first full generation.  A tiny floor
        # preserves the paper's positive efficiency incentive on exactly linear
        # synthetic trajectories.
        self.resolved_mu = (
            self.configured_mu
            if self.configured_mu is not None
            else max(max_loss / float(self.num_steps), 1e-12)
        )
        for (step, arm), loss in losses.items():
            self._counts[step][arm] = 1
            self._means[step][arm] = self._reward(arm, loss)
        self.initialized = True
        return {
            "max_warmup_mse": float(max_loss),
            "resolved_mu": float(self.resolved_mu),
            "initialized_arms": int(len(losses)),
        }

    def select(self, step_index: int) -> tuple[int, float]:
        """Select a skip length with the paper's UCB objective."""

        if not self.initialized:
            raise RuntimeError("FastFlow controller must be initialized before arm selection.")
        step = int(step_index)
        self._ensure_step(step)
        feasible = self.feasible_arms(step)
        if not feasible:
            raise ValueError(f"No FastFlow arm is feasible at step {step}.")

        invocations = sum(self._counts[step][arm] for arm in feasible)
        log_n = math.log(max(invocations, 1))

        def score(arm: int) -> float:
            count = self._counts[step][arm]
            if count == 0:
                return math.inf
            return self._means[step][arm] + self.gamma * math.sqrt(log_n / count)

        # Prefer the larger skip on an exact tie: it has the same learned
        # fidelity estimate and better efficiency.
        selected = max(feasible, key=lambda arm: (score(arm), arm))
        return int(selected), float(score(selected))

    def update(self, *, step_index: int, arm: int, loss: float) -> float:
        """Update one timestep bandit and return the observed reward."""

        if not self.initialized:
            raise RuntimeError("FastFlow controller must be initialized before updates.")
        step = int(step_index)
        chosen = int(arm)
        self._ensure_step(step)
        if chosen not in self._counts[step]:
            raise ValueError(
                f"FastFlow arm {chosen} is not feasible at step {step}; "
                f"expected one of {tuple(self._counts[step])}."
            )
        reward = self._reward(chosen, float(loss))
        old_count = self._counts[step][chosen]
        new_count = old_count + 1
        old_mean = self._means[step][chosen]
        self._counts[step][chosen] = new_count
        self._means[step][chosen] = old_mean + (reward - old_mean) / new_count
        return float(reward)

    def state_dict(self) -> dict[str, object]:
        """Return JSON-friendly controller diagnostics."""

        return {
            "num_steps": self.num_steps,
            "arms": list(self.arms),
            "configured_mu": self.configured_mu,
            "resolved_mu": self.resolved_mu,
            "gamma": self.gamma,
            "initialized": self.initialized,
            "counts": {step: dict(values) for step, values in self._counts.items()},
            "means": {step: dict(values) for step, values in self._means.items()},
        }

