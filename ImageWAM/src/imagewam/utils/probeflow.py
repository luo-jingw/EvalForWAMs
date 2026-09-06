"""Training-free ProbeFlow adaptive solver for flow-matching action inference."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def velocity_cosine_similarity(v_start: torch.Tensor, v_probe: torch.Tensor) -> float:
    """Return cosine similarity between two vector-field predictions."""
    start_flat = v_start.detach().to(device="cpu", dtype=torch.float32).reshape(1, -1)
    probe_flat = v_probe.detach().to(device="cpu", dtype=torch.float32).reshape(1, -1)
    return float(F.cosine_similarity(start_flat, probe_flat, dim=1, eps=1e-8).item())


def schedule_step_count(
    similarity: float,
    *,
    epsilon: float,
    n_min: int,
    n_max: int,
    delta_n: int,
) -> int:
    """Map ProbeFlow linearity score ``S`` to an integration step count ``N``."""
    if not (0.0 < float(epsilon) <= 1.0):
        raise ValueError(f"`epsilon` must be in (0, 1], got {epsilon}")
    if int(n_min) <= 0 or int(n_max) <= 0 or int(n_min) > int(n_max):
        raise ValueError(f"Expected 0 < n_min <= n_max, got n_min={n_min}, n_max={n_max}")
    if int(delta_n) <= 0:
        raise ValueError(f"`delta_n` must be positive, got {delta_n}")

    similarity_clamped = min(1.0, max(-1.0, float(similarity)))
    extra = int(delta_n) * int((1.0 - similarity_clamped) // float(epsilon))
    return int(min(int(n_max), max(int(n_min), int(n_min) + extra)))


def probe_sigma_targets(
    *,
    sigma_start: float,
    dt_probe: float,
) -> tuple[float, float, float]:
    """Return ``(sigma_probe, delta_probe, sigma_remain)`` for ProbeFlow."""
    if not (0.0 < float(dt_probe) < 1.0):
        raise ValueError(f"`dt_probe` must be in (0, 1), got {dt_probe}")
    sigma_start_f = float(sigma_start)
    sigma_probe = sigma_start_f * (1.0 - float(dt_probe))
    delta_probe = sigma_probe - sigma_start_f
    sigma_remain = 0.0 - sigma_probe
    return float(sigma_probe), float(delta_probe), float(sigma_remain)
