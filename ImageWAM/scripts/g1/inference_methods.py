"""Robot-framework-agnostic reproduction of DASH's inference-method configuration.

This module extracts, from ``DASH-main/experiments/robotwin/imagewam_policy/
deploy_policy.py`` (the ``WorldActionRobotWinPolicy`` class + ``_infer_action_chunk``),
exactly the pieces a WebSocket policy server needs to drive the three ImageWAM
inference methods **without** RoboTwin / hydra / omegaconf:

    "baseline"   -- plain N-step flow-matching action solve.
    "probeflow"  -- training-free adaptive FM solver (stateless kwargs).
    "dash"       -- drift-adaptive speculative ratio-jump (stateful, per-episode).

Only stdlib + torch are used. ``theoretical_ratio_jump_r`` / ``drift_to_jump_step``
/ ``latent_drift_metrics`` are vendored inline (byte-for-byte equivalent to
``imagewam.utils.ratio_jump``) so this file has zero imagewam/RoboTwin import.

NOTE (backbone accelerators intentionally NOT exposed here):
    The ImageWAM backbone (``imagewam.models.backbones.imagewam.infer_action`` /
    ``infer_action_flux2``) also supports ``fastflow_enabled`` and ``c3ache_enabled``.
    Per the deploy contract only one accelerator may be enabled at a time, and this
    server surface deliberately exposes only baseline / probeflow / dash.


================================================================================
THE CRITICAL PART -- how DASH obtains ``current_ref_tokens`` (call SEQUENCE)
================================================================================

DASH does **NOT** read the reference tokens out of ``infer_action``'s return dict.
The return dict of ``model.infer_action(...)`` contains only:
    {"action": <cpu float32 [T,D]>,            # always
     "ratio_jump_speculative": {...},          # only when speculative ran
     "timing": {...},                          # only when profiling
     "probeflow"/"fastflow"/"c3ache": {...}}   # only for those modes
-- there is no ref-token key.

Instead, deploy_policy **pre-encodes the current image separately**, BEFORE the
infer call, and feeds the tokens back into the very same call as
``precomputed_flux2_ref_tokens`` / ``precomputed_flux2_ref_img_ids``.

Per-replan sequence for DASH (deploy_policy._infer_action_chunk):

  1. Build the FLUX.2 image tensor from the CURRENT observation.

  2. Pre-encode the visual latent (external encode, no_grad)::

         # deploy_policy.py lines 830-833
         current_ref_tokens, current_ref_img_ids = self.model._encode_flux2_image_tokens(
             image_tensor,
             time_value=10.0,
         )

  3. Choose the jump step ``ratio_jump_step_k`` (deploy_policy.py lines 857-881)::

         ratio_jump_step_k = int(self._spec_ratio_jump_k_near)        # <-- default
         if (self.spec_ratio_jump_fixed_k is None
                 and self._previous_flux2_ref_tokens is not None):
             (ratio_jump_drift, ratio_jump_cosine,
              ratio_jump_relative_norm_delta) = latent_drift_metrics(
                 current_ref_tokens,
                 self._previous_flux2_ref_tokens,           # <-- PREVIOUS replan
                 l2_scale=_SPEC_RATIO_JUMP_DRIFT_L2_SCALE,  # 0.08
             )
             ratio_jump_step_k = drift_to_jump_step(
                 ratio_jump_drift,
                 k_near=self._spec_ratio_jump_k_near,
                 k_far=self._spec_ratio_jump_k_far,
                 drift_low=self.spec_ratio_jump_drift_low,
                 drift_high=self.spec_ratio_jump_drift_high,
             )

  4. Assemble the infer kwargs, passing the freshly-encoded tokens back in
     (deploy_policy.py lines 907-934)::

         infer_kwargs.update({
             "precomputed_flux2_ref_tokens": current_ref_tokens,
             "precomputed_flux2_ref_img_ids": current_ref_img_ids,
             "ratio_jump_multiplier": torch.tensor(R_by_k[step_k], dtype=torch.float32),
             "ratio_jump_to_step": ratio_jump_step_k,
             "ratio_jump_speculative": self.spec_ratio_jump_speculative,
             # + candidate multipliers + verify_* when speculative
         })

  5. Call ``pred = self.model.infer_action(**infer_kwargs)`` (line 973).

  6. AFTER the call, store the CURRENT tokens as previous (line 992)::

         self._previous_flux2_ref_tokens = current_ref_tokens.detach().clone()

DRIFT LAG SEMANTICS (important, and stated explicitly per the brief):
    Drift is measured between the CURRENT replan's freshly-encoded ref tokens and
    the PREVIOUS replan's ref tokens, and the resulting jump step is applied to
    the CURRENT replan's infer call. So the *ref tokens driving the jump are the
    current ones* (no lag on the tokens themselves) -- but the drift SIGNAL is
    "how much did the scene change since the last replan", i.e. it is computed
    against last replan and consumed this replan. There is therefore a one-replan
    lag in the drift *signal*: replan i's jump reflects the change from replan
    i-1 -> i. The current tokens are used both for the metric and for this call.

FIRST REPLAN BEHAVIOR -- READ CAREFULLY (mismatch vs. the task brief):
    On the first replan of an episode ``_previous_flux2_ref_tokens is None``, so
    step 3 leaves ``ratio_jump_step_k == k_near`` (default 7). deploy_policy
    therefore STILL EMITS A SPECULATIVE JUMP at k_near on the first replan -- it
    does NOT run plain baseline. The task brief describes "fall back to plain
    baseline (no jump)" on the first replan; that is NOT what the authoritative
    source does. This module defaults to the FAITHFUL source behavior
    (``baseline_on_first_replan=False`` -> jump at k_near). Set
    ``baseline_on_first_replan=True`` if you instead want the brief's described
    baseline-on-first-replan behavior.


================================================================================
KWARG-NAME MAPPING  (DASH doc name  ->  model.infer_action kwarg)
================================================================================
    num_inference_steps                -> num_inference_steps
    spec_ratio_jump_k_near             -> (picks ratio_jump_to_step; also lower
                                           bound of the R lookup table)
    spec_ratio_jump_k_far              -> (upper bound for the chosen step; must be
                                           N-1 so speculative candidates are complete)
    spec_ratio_jump_drift_low/high     -> drift_to_jump_step thresholds (NOT a
                                           model kwarg)
    (chosen step k)                    -> ratio_jump_to_step
    (theoretical R at k)               -> ratio_jump_multiplier   (0-dim f32 tensor)
    (R for k..N-1)                     -> ratio_jump_candidate_multipliers (f32 tensor)
    spec_ratio_jump_speculative        -> ratio_jump_speculative
    spec_ratio_jump_verify_rel_only    -> ratio_jump_verify_rel_only
    spec_ratio_jump_verify_rel_l2_max  -> ratio_jump_verify_rel_l2_max
    spec_ratio_jump_verify_cos_min     -> ratio_jump_verify_cos_min
    spec_ratio_jump_verify_cos_only    -> ratio_jump_verify_cos_only
    current_ref_tokens                 -> precomputed_flux2_ref_tokens
    current_ref_img_ids                -> precomputed_flux2_ref_img_ids

    (``model.infer_action`` internally forwards precomputed_flux2_ref_tokens ->
     infer_action_flux2's precomputed_ref_tokens; the server always calls the
     public ``infer_action``, so use the ``*_flux2_*`` names shown above.)
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

# --------------------------------------------------------------------------- #
# Defaults lifted verbatim from deploy_policy.py module constants.
# --------------------------------------------------------------------------- #
_SPEC_RATIO_JUMP_K_NEAR = 7
_SPEC_RATIO_JUMP_DRIFT_LOW = 0.25
_SPEC_RATIO_JUMP_DRIFT_HIGH = 0.45
_SPEC_RATIO_JUMP_DRIFT_L2_SCALE = 0.08  # l2_scale fed to latent_drift_metrics
_SPEC_RATIO_JUMP_VERIFY_COS_MIN = 0.998
_SPEC_RATIO_JUMP_VERIFY_REL_L2_MAX = 0.02  # backbone default; DASH doc overrides -> 0.03

_PROBEFLOW_DT_PROBE = 0.5
_PROBEFLOW_EPSILON = 0.008
_PROBEFLOW_N_MIN = 2
_PROBEFLOW_DELTA_N = 2

# The image time-value deploy_policy uses when externally encoding ref tokens.
FLUX2_REF_ENCODE_TIME_VALUE = 10.0


# --------------------------------------------------------------------------- #
# (1) Method identifier
# --------------------------------------------------------------------------- #
class InferenceMethod(str, enum.Enum):
    """The three inference methods this server surface exposes.

    ``fastflow`` and ``c3ache`` also exist in the ImageWAM backbone but are
    intentionally NOT exposed here (see module docstring).
    """

    BASELINE = "baseline"
    PROBEFLOW = "probeflow"
    DASH = "dash"


# Convenience: the accepted string literals.
METHODS = (InferenceMethod.BASELINE.value, InferenceMethod.PROBEFLOW.value, InferenceMethod.DASH.value)


def coerce_method(value: Any) -> InferenceMethod:
    """Coerce a str/enum into :class:`InferenceMethod` (raises on unknown)."""
    if isinstance(value, InferenceMethod):
        return value
    key = str(value).strip().lower()
    for method in InferenceMethod:
        if method.value == key:
            return method
    raise ValueError(f"Unknown inference method {value!r}; expected one of {METHODS}.")


# --------------------------------------------------------------------------- #
# (2) Frozen parameter dataclasses (DASH doc REAL_ROBOT_DEPLOY.md 3.3/3.4)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BaselineParams:
    """Plain N-step solve. Stateless."""

    num_inference_steps: int = 10
    sigma_shift: Optional[float] = None
    seed: Optional[int] = None


@dataclass(frozen=True)
class ProbeFlowParams:
    """ProbeFlow adaptive FM solver. Stateless (values from PROBEFLOW_EXTRA)."""

    num_inference_steps: int = 10
    probeflow_dt_probe: float = _PROBEFLOW_DT_PROBE      # 0.5
    probeflow_epsilon: float = _PROBEFLOW_EPSILON        # 0.008
    probeflow_n_min: int = _PROBEFLOW_N_MIN              # 2
    probeflow_n_max: Optional[int] = 10                  # = N; None -> backbone uses N
    probeflow_delta_n: int = _PROBEFLOW_DELTA_N          # 2
    sigma_shift: Optional[float] = None
    seed: Optional[int] = None


@dataclass(frozen=True)
class DashParams:
    """DASH drift-adaptive speculative ratio-jump (values from DASH_EXTRA).

    Field names keep the ``spec_ratio_jump_*`` doc names; the mapping onto the
    backbone ``infer_action`` kwargs is applied by :class:`DashDriftController`
    (see module docstring "KWARG-NAME MAPPING").
    """

    num_inference_steps: int = 10
    spec_ratio_jump_k_near: int = 7
    spec_ratio_jump_k_far: int = 9  # = N - 1
    spec_ratio_jump_drift_low: float = _SPEC_RATIO_JUMP_DRIFT_LOW    # 0.25
    spec_ratio_jump_drift_high: float = _SPEC_RATIO_JUMP_DRIFT_HIGH  # 0.45
    spec_ratio_jump_speculative: bool = True
    spec_ratio_jump_verify_rel_only: bool = True
    spec_ratio_jump_verify_rel_l2_max: float = 0.03  # DASH doc override (backbone default 0.02)
    # Carried for faithful completeness (deploy defaults; unused when rel_only=True):
    spec_ratio_jump_verify_cos_min: float = _SPEC_RATIO_JUMP_VERIFY_COS_MIN  # 0.998
    spec_ratio_jump_verify_cos_only: bool = False
    spec_ratio_jump_fixed_k: Optional[int] = None  # pin the jump step (disables drift adaptation)
    drift_l2_scale: float = _SPEC_RATIO_JUMP_DRIFT_L2_SCALE  # 0.08
    sigma_shift: Optional[float] = None
    seed: Optional[int] = None


# --------------------------------------------------------------------------- #
# Vendored ratio-jump helpers (equivalent to imagewam.utils.ratio_jump).
# --------------------------------------------------------------------------- #
def theoretical_ratio_jump_r(step_k: int, num_inference_steps: int, shift: float = 5.0) -> float:
    """Scalar flow-progress ratio from snapshot ``x_1`` to ``x_k`` (see ratio_jump.py)."""
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
    """Low visual drift -> farther jump (k_far); high drift -> nearer jump (k_near)."""
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
    cosine = float(
        torch.nn.functional.cosine_similarity(current_flat, previous_flat, dim=1, eps=1e-8).item()
    )
    current_norm = float(torch.linalg.vector_norm(current_flat).item())
    previous_norm = float(torch.linalg.vector_norm(previous_flat).item())
    relative_norm_delta = abs(current_norm - previous_norm) / max(previous_norm, 1e-8)
    l2_ratio = min(max(relative_norm_delta / max(float(l2_scale), 1e-8), 0.0), 1.0)
    stability = max(0.0, cosine) * (1.0 - l2_ratio)
    drift = min(1.0, max(0.0, 1.0 - stability))
    return float(drift), float(cosine), float(relative_norm_delta)


# --------------------------------------------------------------------------- #
# (5) Stateless helpers for baseline / probeflow
# --------------------------------------------------------------------------- #
def build_baseline_kwargs(params: BaselineParams) -> Dict[str, Any]:
    """Return plain ``model.infer_action`` kwargs for the baseline method.

    These are the accelerator-free knobs; all fastflow/probeflow/c3ache/ratio-jump
    kwargs are left at their backbone defaults (off).
    """
    return {
        "num_inference_steps": int(params.num_inference_steps),
        "sigma_shift": params.sigma_shift,
        "seed": params.seed,
    }


def build_probeflow_kwargs(params: ProbeFlowParams) -> Dict[str, Any]:
    """Return ``model.infer_action`` kwargs for ProbeFlow (deploy lines 936-944)."""
    return {
        "num_inference_steps": int(params.num_inference_steps),
        "sigma_shift": params.sigma_shift,
        "seed": params.seed,
        "probeflow_enabled": True,
        "probeflow_dt_probe": float(params.probeflow_dt_probe),
        "probeflow_epsilon": float(params.probeflow_epsilon),
        "probeflow_n_min": int(params.probeflow_n_min),
        "probeflow_n_max": None if params.probeflow_n_max is None else int(params.probeflow_n_max),
        "probeflow_delta_n": int(params.probeflow_delta_n),
    }


# --------------------------------------------------------------------------- #
# (3) Stateful DASH controller
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DashReplanPlan:
    """Result of one DASH replan planning step.

    ``infer_kwargs`` holds the ratio-jump kwargs (plus the precomputed ref
    tokens) to splice into ``model.infer_action(**base_kwargs, **infer_kwargs)``.
    The remaining fields are for logging/telemetry only.
    """

    infer_kwargs: Dict[str, Any]
    step_k: Optional[int]           # chosen jump step; None when running baseline-on-first
    is_jump: bool                   # True if a ratio jump is emitted this replan
    drift: Optional[float]          # None on the first replan (no previous)
    cosine: Optional[float]
    relative_norm_delta: Optional[float]


class DashDriftController:
    """Per-episode, stateful driver for DASH's drift-adaptive ratio jump.

    Faithfully mirrors ``deploy_policy._infer_action_chunk``'s DASH branch. The
    controller is model-agnostic: the caller owns the model, encodes the ref
    tokens (``model._encode_flux2_image_tokens(image, time_value=10.0)``), and
    performs ``model.infer_action(...)``. The controller only:
      * precomputes the theoretical R lookup table,
      * turns (current ref tokens, previous ref tokens) into a jump step,
      * builds the ratio_jump_* + precomputed_flux2_ref_* kwargs,
      * tracks ``_previous_flux2_ref_tokens`` across replans.

    Typical per-replan usage on the server::

        ctrl = DashDriftController(DashParams(), scheduler_shift=model.infer_action_scheduler.shift)
        ctrl.reset()                                      # at each episode boundary
        ...
        tokens, img_ids = model._encode_flux2_image_tokens(image, time_value=10.0)
        plan = ctrl.plan_replan(tokens, img_ids, replan_index=i)
        pred = model.infer_action(prompt=..., input_image=image, action_horizon=H,
                                  proprio=proprio, num_inference_steps=params.num_inference_steps,
                                  sigma_shift=..., seed=..., **plan.infer_kwargs)
        ctrl.commit(tokens)                               # AFTER the infer call
    """

    def __init__(
        self,
        params: DashParams,
        *,
        scheduler_shift: float,
        baseline_on_first_replan: bool = False,
    ) -> None:
        """Precompute the R table and validate ranges (mirrors deploy __init__).

        Args:
            params: DASH parameters.
            scheduler_shift: ``model.infer_action_scheduler.shift``. This is the
                resolved shift used to compute theoretical ratios. deploy uses
                ``params.sigma_shift if not None else scheduler.shift``; pass the
                already-resolved value, or leave ``params.sigma_shift`` None and
                pass the scheduler shift here.
            baseline_on_first_replan: If True, the first replan of each episode
                emits NO jump (plain baseline). If False (default, faithful to
                deploy_policy), the first replan jumps at ``k_near``.
        """
        self.params = params
        self.baseline_on_first_replan = bool(baseline_on_first_replan)

        n = int(params.num_inference_steps)
        if n < 3:
            raise ValueError(f"DASH requires at least 3 inference steps, got {n}.")

        # Resolve shift exactly as deploy_policy does (sigma_shift override wins).
        self._shift = float(params.sigma_shift) if params.sigma_shift is not None else float(scheduler_shift)
        if self._shift <= 0:
            raise ValueError(f"Resolved shift must be positive, got {self._shift}.")

        self._k_near = int(params.spec_ratio_jump_k_near)
        self._k_far = int(params.spec_ratio_jump_k_far)
        if not (1 <= self._k_near <= self._k_far <= n - 1):
            raise ValueError(
                "Expected 1 <= spec_ratio_jump_k_near <= spec_ratio_jump_k_far <= N-1, "
                f"got k_near={self._k_near}, k_far={self._k_far}, N={n}."
            )
        if params.spec_ratio_jump_fixed_k is not None and not (2 <= params.spec_ratio_jump_fixed_k < n):
            raise ValueError(
                "`spec_ratio_jump_fixed_k` must be in [2, num_inference_steps) when set, "
                f"got {params.spec_ratio_jump_fixed_k}."
            )
        if not (0.0 <= params.spec_ratio_jump_drift_low < params.spec_ratio_jump_drift_high <= 1.0):
            raise ValueError(
                "Expected 0 <= drift_low < drift_high <= 1, "
                f"got low={params.spec_ratio_jump_drift_low}, high={params.spec_ratio_jump_drift_high}."
            )

        # Precompute theoretical R for every step the chosen jump / speculative
        # candidates can reference: k_near .. N-1. deploy uses range(k_min, k_far+1);
        # with the default k_far == N-1 these coincide, and covering up to N-1 keeps
        # the speculative candidate list (range(step_k, N)) complete for any step_k.
        k_min = self._k_near if params.spec_ratio_jump_fixed_k is None else min(
            self._k_near, int(params.spec_ratio_jump_fixed_k)
        )
        self._r_by_k: Dict[int, float] = {
            step_k: theoretical_ratio_jump_r(step_k, n, self._shift)
            for step_k in range(k_min, n)  # k_min .. N-1 inclusive
        }

        # Per-episode state.
        self._previous_flux2_ref_tokens: Optional[torch.Tensor] = None
        self._replan_count: int = 0

    # -- state management --------------------------------------------------- #
    def reset(self) -> None:
        """Clear per-episode state. Call at each robot-episode boundary."""
        self._previous_flux2_ref_tokens = None
        self._replan_count = 0

    def commit(self, current_ref_tokens: torch.Tensor) -> None:
        """Store the CURRENT replan's ref tokens as previous.

        Call AFTER ``model.infer_action`` returns, mirroring deploy_policy line
        992: ``self._previous_flux2_ref_tokens = current_ref_tokens.detach().clone()``.
        """
        self._previous_flux2_ref_tokens = current_ref_tokens.detach().clone()
        self._replan_count += 1

    @property
    def has_previous(self) -> bool:
        return self._previous_flux2_ref_tokens is not None

    @property
    def r_by_k(self) -> Dict[int, float]:
        """Read-only view of the precomputed theoretical R lookup table."""
        return dict(self._r_by_k)

    # -- core planning ------------------------------------------------------ #
    def plan_replan(
        self,
        current_ref_tokens: torch.Tensor,
        current_ref_img_ids: torch.Tensor,
        replan_index: int,
    ) -> DashReplanPlan:
        """Build this replan's ratio-jump kwargs from the current ref tokens.

        Mirrors deploy_policy lines 854-934. ``current_ref_tokens`` /
        ``current_ref_img_ids`` come from
        ``model._encode_flux2_image_tokens(image, time_value=10.0)``.

        Returns a :class:`DashReplanPlan`; splice ``plan.infer_kwargs`` into the
        ``model.infer_action(**base, **plan.infer_kwargs)`` call. Does NOT mutate
        previous-token state -- call :meth:`commit` after the infer call.
        """
        params = self.params
        n = int(params.num_inference_steps)

        # Step choice (deploy lines 857-881).
        if params.spec_ratio_jump_fixed_k is not None:
            step_k = int(params.spec_ratio_jump_fixed_k)
        else:
            step_k = int(self._k_near)

        drift: Optional[float] = None
        cosine: Optional[float] = None
        rel_norm_delta: Optional[float] = None

        if params.spec_ratio_jump_fixed_k is None and self._previous_flux2_ref_tokens is not None:
            drift, cosine, rel_norm_delta = latent_drift_metrics(
                current_ref_tokens,
                self._previous_flux2_ref_tokens,
                l2_scale=params.drift_l2_scale,
            )
            step_k = drift_to_jump_step(
                drift,
                k_near=self._k_near,
                k_far=self._k_far,
                drift_low=params.spec_ratio_jump_drift_low,
                drift_high=params.spec_ratio_jump_drift_high,
            )

        # First-replan baseline fallback (opt-in; NOT deploy's default -- see docstring).
        if self.baseline_on_first_replan and self._previous_flux2_ref_tokens is None:
            infer_kwargs = {
                # Reuse the freshly-encoded tokens (avoids a second internal encode)
                # without requesting any ratio jump.
                "precomputed_flux2_ref_tokens": current_ref_tokens,
                "precomputed_flux2_ref_img_ids": current_ref_img_ids,
            }
            return DashReplanPlan(
                infer_kwargs=infer_kwargs,
                step_k=None,
                is_jump=False,
                drift=None,
                cosine=None,
                relative_norm_delta=None,
            )

        # Assemble ratio-jump kwargs (deploy lines 907-934).
        infer_kwargs: Dict[str, Any] = {
            "precomputed_flux2_ref_tokens": current_ref_tokens,
            "precomputed_flux2_ref_img_ids": current_ref_img_ids,
            "ratio_jump_multiplier": torch.tensor(self._r_by_k[step_k], dtype=torch.float32),
            "ratio_jump_to_step": int(step_k),
            "ratio_jump_speculative": bool(params.spec_ratio_jump_speculative),
        }
        if params.spec_ratio_jump_speculative:
            infer_kwargs.update(
                {
                    "ratio_jump_candidate_multipliers": torch.tensor(
                        [self._r_by_k[k] for k in range(int(step_k), n)],
                        dtype=torch.float32,
                    ),
                    "ratio_jump_verify_cos_min": float(params.spec_ratio_jump_verify_cos_min),
                    "ratio_jump_verify_rel_l2_max": float(params.spec_ratio_jump_verify_rel_l2_max),
                    "ratio_jump_verify_cos_only": bool(params.spec_ratio_jump_verify_cos_only),
                    "ratio_jump_verify_rel_only": bool(params.spec_ratio_jump_verify_rel_only),
                }
            )

        return DashReplanPlan(
            infer_kwargs=infer_kwargs,
            step_k=int(step_k),
            is_jump=True,
            drift=drift,
            cosine=cosine,
            relative_norm_delta=rel_norm_delta,
        )


__all__ = [
    "InferenceMethod",
    "METHODS",
    "coerce_method",
    "BaselineParams",
    "ProbeFlowParams",
    "DashParams",
    "DashReplanPlan",
    "DashDriftController",
    "build_baseline_kwargs",
    "build_probeflow_kwargs",
    "theoretical_ratio_jump_r",
    "drift_to_jump_step",
    "latent_drift_metrics",
    "FLUX2_REF_ENCODE_TIME_VALUE",
]
