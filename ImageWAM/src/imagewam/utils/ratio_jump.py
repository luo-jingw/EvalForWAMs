"""Training-free helpers for speculative, drift-adaptive ratio-jump inference."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def theoretical_ratio_jump_r(
    step_k: int,
    num_inference_steps: int,
    shift: float = 5.0,
) -> float:
    """Return the scalar flow-progress ratio from snapshot ``x_1`` to ``x_k``."""
    if num_inference_steps <= 0:
        raise ValueError(f"`num_inference_steps` must be positive, got {num_inference_steps}")
    if not (1 <= int(step_k) <= int(num_inference_steps)):
        raise ValueError(
            f"`step_k` must be in [1, num_inference_steps={num_inference_steps}], got {step_k}"
        )
    if float(shift) <= 0:
        raise ValueError(f"`shift` must be positive, got {shift}")

    u_steps = torch.linspace(1.0, 0.0, int(num_inference_steps) + 1, dtype=torch.float64)
    sigma_steps = float(shift) * u_steps / (1.0 + (float(shift) - 1.0) * u_steps)
    sigma_1 = float(sigma_steps[1].item())
    sigma_k = float(sigma_steps[int(step_k)].item())
    denominator = 1.0 - sigma_1
    if abs(denominator) < 1e-12:
        raise ValueError(
            f"Degenerate ratio: 1 - sigma_1={denominator} "
            f"(shift={shift}, num_inference_steps={num_inference_steps})"
        )
    return float((1.0 - sigma_k) / denominator)


def drift_to_jump_step(
    drift: float,
    *,
    k_near: int,
    k_far: int,
    drift_low: float,
    drift_high: float,
) -> int:
    """Map low visual drift to a farther jump and high drift to a nearer jump."""
    if k_near > k_far:
        raise ValueError(f"`k_near` must be <= `k_far`, got {k_near} > {k_far}")
    if not (0.0 <= float(drift_low) < float(drift_high) <= 1.0):
        raise ValueError(
            "Expected 0 <= drift_low < drift_high <= 1, "
            f"got drift_low={drift_low}, drift_high={drift_high}"
        )
    if float(drift) <= float(drift_low):
        return int(k_far)
    if float(drift) >= float(drift_high):
        return int(k_near)
    alpha = (float(drift) - float(drift_low)) / (float(drift_high) - float(drift_low))
    return int(round(float(k_far) - float(k_far - k_near) * alpha))


def latent_drift_metrics(
    current: torch.Tensor,
    previous: torch.Tensor,
    *,
    l2_scale: float = 0.08,
) -> tuple[float, float, float]:
    """Return ``(drift, cosine, relative-norm-delta)`` for two visual latents."""
    if tuple(current.shape) != tuple(previous.shape):
        raise ValueError(
            f"Visual latent shapes must match, got {tuple(current.shape)} and {tuple(previous.shape)}"
        )
    current_flat = current.detach().to(device="cpu", dtype=torch.float32).reshape(1, -1)
    previous_flat = previous.detach().to(device="cpu", dtype=torch.float32).reshape(1, -1)
    cosine = float(F.cosine_similarity(current_flat, previous_flat, dim=1, eps=1e-8).item())
    current_norm = float(torch.linalg.vector_norm(current_flat).item())
    previous_norm = float(torch.linalg.vector_norm(previous_flat).item())
    relative_norm_delta = abs(current_norm - previous_norm) / max(previous_norm, 1e-8)
    l2_ratio = min(max(relative_norm_delta / max(float(l2_scale), 1e-8), 0.0), 1.0)
    stability = max(0.0, cosine) * (1.0 - l2_ratio)
    drift = min(1.0, max(0.0, 1.0 - stability))
    return float(drift), float(cosine), float(relative_norm_delta)


def derive_inference_seed(base_seed: int, *, episode_index: int, replan_index: int) -> int:
    """Derive a stable, distinct diffusion seed for each episode and replan."""
    return int(base_seed) + int(episode_index) * 1_000_003 + int(replan_index) * 10_007
