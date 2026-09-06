#!/usr/bin/env python3
"""WebSocket policy server for the Unitree G1 ImageWAM checkpoint.

Speaks the msgpack/WebSocket protocol of g1-client's PolicyClient, so
`python fastwam/main.py --server-host <this host> --server-port 8000` drives it
unchanged.

Wire contract (mirrors g1-client/fastwam/fastwam_policy.py):
    recv {"image": [head, left_wrist, right_wrist] RGB uint8 HWC,
          "state": float32 (16,) raw joint radians,
          "prompt": str}
    send {"actions": float64 [H, 16] raw joint radians}
    recv {"reset": True, "prompt": str} -> send {"ok": True}
A metadata dict is sent once on connect.

Text is never encoded here. The prompt embeddings come from the npz written by
scripts/g1/export_g1_prompt_embeds.py, and ImageWAM.infer_action takes them via
`context`/`context_mask`. This keeps `load_text_encoder=false` as in training and
saves the ~8 GB Qwen3 resident cost.

This module exposes a *hot-swappable* `PolicyEngine`: the active checkpoint and the
inference method/config (baseline / ProbeFlow / DASH, num_inference_steps, horizon,
per-method params) can be changed at runtime under a lock, without restarting the
process. The standalone `main()` still serves a single fixed baseline config for
back-compat; the Gradio control panel (scripts/g1/gradio_control_panel.py) drives
the same `PolicyEngine` live.

torch.compile is never used here -- inference must stay eager (per project rule).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
import torchvision.transforms.functional as transforms_F
import websockets.sync.server
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from imagewam.datasets.lerobot.processors.imagewam_processor import ImageWAMProcessor
from imagewam.datasets.lerobot.robot_video_dataset import RobotVideoDataset
from imagewam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from imagewam.utils.config_resolvers import register_default_resolvers

# Method configs + the stateful DASH drift controller live in their own module so
# the wire/server code stays independent of the acceleration math.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from inference_methods import (  # noqa: E402
    FLUX2_REF_ENCODE_TIME_VALUE,
    METHODS,
    BaselineParams,
    DashDriftController,
    DashParams,
    ProbeFlowParams,
    build_baseline_kwargs,
    build_probeflow_kwargs,
)

log = logging.getLogger("serve_imagewam_g1")

ACTION_DIM: int = 16
NUM_CAMERAS: int = 3


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BootConfig:
    """Immutable, set once at process start -- what needs a full restart to change."""

    task_config: str
    host: str
    port: int
    device: str
    robotwin_camera_layout: str
    flux2_model_path: Path
    ae_model_path: Path
    flux2_src_path: Path


@dataclass
class RuntimeConfig:
    """Mutable inference configuration, swappable live by the control panel."""

    method: str = "baseline"  # one of inference_methods.METHODS
    # If set, the server uses this prompt for every inference and ignores the prompt
    # the robot client sends. None -> use the client's per-request prompt.
    prompt_override: Optional[str] = None
    action_horizon: int = 16
    # Advisory: how many steps of each chunk the *client* should execute before
    # replanning. Our WebSocket server always returns a full `action_horizon`
    # chunk; execution cadence is a client decision (fastwam --exec-steps). We
    # publish this in metadata so the client can honor it, but the server does
    # not truncate the chunk.
    exec_horizon: int = 16
    baseline: BaselineParams = field(default_factory=BaselineParams)
    probeflow: ProbeFlowParams = field(default_factory=ProbeFlowParams)
    dash: DashParams = field(default_factory=DashParams)

    def summary(self) -> dict[str, Any]:
        params = {"baseline": self.baseline, "probeflow": self.probeflow, "dash": self.dash}[self.method]
        return {
            "method": self.method,
            "prompt_override": self.prompt_override,
            "action_horizon": self.action_horizon,
            "exec_horizon": self.exec_horizon,
            "params": vars(params).copy(),
        }


@dataclass
class ActiveProfile:
    """Which weights + stats + prompt bank are currently loaded."""

    name: str
    ckpt_path: Path
    dataset_stats_path: Path
    prompt_embeds_path: Path
    task_config: str


@dataclass(frozen=True)
class InferTiming:
    preprocess_s: float
    infer_s: float
    postprocess_s: float
    effective_steps: Optional[float] = None  # ProbeFlow/DASH may spend < num_inference_steps
    dash: Optional[dict[str, Any]] = None     # DASH ratio-jump telemetry for this replan


# --------------------------------------------------------------------------- #
# Stateless helpers (unchanged from the original single-config server)         #
# --------------------------------------------------------------------------- #
# All G1 prompt-embeds npz were exported at this context length; new prompts must
# match it (not Qwen3's 512 default) to stay compatible with the trained model.
DEFAULT_CONTEXT_LEN: int = 128


class PromptBank:
    """Task string -> precomputed Qwen3 context tensors, resident on the model device.

    Loads from an npz when present; otherwise starts empty (for bootstrapping a new
    task's bank from the panel via `add()` + persist).
    """

    def __init__(self, path: Optional[Path], device: torch.device, dtype: torch.dtype,
                 context_len: int = DEFAULT_CONTEXT_LEN) -> None:
        self._context: dict[str, torch.Tensor] = {}
        self._context_mask: dict[str, torch.Tensor] = {}
        self.tasks: tuple[str, ...] = ()
        if path is not None and Path(path).exists():
            payload = np.load(path, allow_pickle=True)
            tasks = [str(t) for t in payload["tasks"]]
            hidden = torch.from_numpy(payload["hidden"]).to(device=device, dtype=dtype)
            mask = torch.from_numpy(payload["mask"]).to(device=device, dtype=torch.bool)
            for index, task in enumerate(tasks):
                self._context[task] = hidden[index].unsqueeze(0)
                self._context_mask[task] = mask[index].unsqueeze(0)
            self.context_len = int(payload["context_len"])
            self.tasks = tuple(tasks)
        else:
            self.context_len = int(context_len)

    def get(self, task: str) -> tuple[torch.Tensor, torch.Tensor]:
        if task not in self._context:
            raise KeyError(
                f"No precomputed embedding for prompt {task!r}. "
                f"Known prompts: {list(self.tasks)}. "
                "Re-run scripts/g1/export_g1_prompt_embeds.py with --task."
            )
        return self._context[task], self._context_mask[task]

    def __contains__(self, task: str) -> bool:
        return task in self._context

    def add(self, task: str, hidden: torch.Tensor, mask: torch.Tensor) -> None:
        """Add one prompt's embedding. hidden [1,L,D] / [L,D], mask [1,L] / [L]."""
        if hidden.ndim == 2:
            hidden = hidden.unsqueeze(0)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if hidden.shape[1] != self.context_len or mask.shape[1] != self.context_len:
            raise ValueError(
                f"prompt length {hidden.shape[1]}/{mask.shape[1]} != bank context_len {self.context_len}"
            )
        self._context[task] = hidden
        self._context_mask[task] = mask
        if task not in self.tasks:
            self.tasks = (*self.tasks, task)

    def export_arrays(self) -> tuple[list[str], NDArray[np.float32], NDArray[np.bool_], int]:
        """Stack the whole bank into npz-ready arrays (tasks, hidden[N,L,D], mask[N,L])."""
        tasks = list(self.tasks)
        hidden = torch.cat([self._context[t] for t in tasks], dim=0).to("cpu", torch.float32).numpy()
        mask = torch.cat([self._context_mask[t] for t in tasks], dim=0).to("cpu", torch.bool).numpy()
        return tasks, hidden, mask, self.context_len


class ImagePreprocessor:
    """Three RGB views -> one robotwin-composed conditioning frame on the GPU.

    Mirrors RobotVideoDataset: scale to [0,1] first, resize with antialias, compose
    (head on top, wrists side by side underneath), then map to [-1,1]. Resizing runs
    on the device in float, so there is one host-to-device copy per view and no PIL
    round trip.
    """

    def __init__(self, layout: str, device: torch.device, dtype: torch.dtype) -> None:
        # Reuse the dataset's layout table so serving cannot drift from training.
        top, left, right = RobotVideoDataset._robotwin_camera_sizes(layout)
        self._sizes: tuple[list[int], list[int], list[int]] = (top, left, right)
        self._device = device
        self._dtype = dtype

    def __call__(self, views: Sequence[NDArray[np.uint8]]) -> torch.Tensor:
        if len(views) != NUM_CAMERAS:
            raise ValueError(f"Expected {NUM_CAMERAS} views, got {len(views)}")
        resized: list[torch.Tensor] = []
        for view, size in zip(views, self._sizes):
            array = np.asarray(view)
            if array.ndim != 3 or array.shape[2] != 3:
                raise ValueError(f"Expected RGB HWC uint8, got shape {array.shape}")
            # msgpack yields a read-only view; copy so torch.from_numpy is safe.
            tensor = torch.from_numpy(np.array(array, dtype=np.uint8)).to(
                device=self._device, non_blocking=True
            )
            tensor = tensor.permute(2, 0, 1).to(dtype=torch.float32).div_(255.0)
            resized.append(
                transforms_F.resize(
                    tensor,
                    size=size,
                    interpolation=transforms_F.InterpolationMode.BILINEAR,
                    antialias=True,
                )
            )
        bottom = torch.cat([resized[1], resized[2]], dim=-1)
        image = torch.cat([resized[0], bottom], dim=-2)
        image = image.mul_(2.0).sub_(1.0)
        return image.unsqueeze(0).to(dtype=self._dtype)


class StateCodec:
    """Normalize the raw 16-dim state and un-normalize predicted action chunks."""

    def __init__(self, processor: ImageWAMProcessor) -> None:
        state_meta = processor.shape_meta["state"]
        action_meta = processor.shape_meta["action"]
        if len(state_meta) != 1 or len(action_meta) != 1:
            raise ValueError("Expected exactly one merged state key and one action key.")
        self._processor = processor
        self._state_key = state_meta[0]["key"]
        self._action_key = action_meta[0]["key"]

    def normalize_state(self, state: NDArray[np.float32]) -> torch.Tensor:
        if state.shape != (ACTION_DIM,):
            raise ValueError(f"Expected state shape ({ACTION_DIM},), got {state.shape}")
        batch = {
            "state": {
                self._state_key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            }
        }
        batch = self._processor.action_state_transform(batch)
        batch = self._processor.normalizer.forward(batch)
        return batch["state"][self._state_key]

    def denormalize_action(self, action: torch.Tensor) -> NDArray[np.float64]:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"Expected action [B,T,D], got {tuple(action.shape)}")
        normalizer = self._processor.normalizer.normalizers["action"][self._action_key]
        denorm = normalizer.backward(action.to(dtype=torch.float32, device="cpu"))
        return denorm.numpy()[0].astype(np.float64)


def _effective_steps(pred: Any) -> Optional[float]:
    """Effective number of FM denoise evaluations this replan actually spent.

    For DASH speculative ratio-jump the backbone returns ``ratio_jump_speculative``
    with ``parallel_calls`` = the network forward passes for that replan (the DASH
    speedup shows up as this being well below the nominal ``num_inference_steps``).
    ProbeFlow/others attach their own info dict; probe plausible keys, else give up.
    """
    if not isinstance(pred, dict):
        return None
    spec = pred.get("ratio_jump_speculative")
    if isinstance(spec, dict) and isinstance(spec.get("parallel_calls"), (int, float)):
        return float(spec["parallel_calls"])
    for key in ("probeflow", "fastflow", "c3ache"):
        info = pred.get(key)
        if isinstance(info, dict):
            # ProbeFlow reports `forward_calls` = denoise-network evaluations this
            # replan spent (= 2 when it takes the sparse 2-step path, else n+1),
            # directly comparable to DASH's parallel_calls.
            for field_name in ("forward_calls", "effective_steps", "num_steps",
                               "steps", "n_eval", "nfe"):
                value = info.get(field_name)
                if isinstance(value, (int, float)):
                    return float(value)
    return None


def _dash_info(pred: Any) -> Optional[dict[str, Any]]:
    """Merge DASH per-replan telemetry: planned jump step + realized speculative result.

    ``planned_step_k`` / ``drift`` come from the controller (server side); ``base_step``,
    ``chosen_step``, ``accepted_links``, ``parallel_calls`` come from the backbone's
    speculative verification. ``parallel_calls`` is the effective FM-step count.
    """
    if not isinstance(pred, dict):
        return None
    info: dict[str, Any] = {}
    planned = pred.get("_dash")
    if isinstance(planned, dict):
        info["planned_step_k"] = planned.get("step_k")
        info["drift"] = None if planned.get("drift") is None else round(float(planned["drift"]), 3)
    spec = pred.get("ratio_jump_speculative")
    if isinstance(spec, dict):
        for key in ("base_step", "chosen_step", "accepted_links", "parallel_calls", "branch_count"):
            if key in spec:
                info[key] = spec[key]
    return info or None


def build_processor(cfg: DictConfig, dataset_stats_path: Path) -> ImageWAMProcessor:
    processor = instantiate(cfg.data.train.processor)
    processor.eval()
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(dataset_stats_path)))
    return processor


def load_config(
    task_config: str,
    flux2_model_path: Path,
    ae_model_path: Path,
    flux2_src_path: Path,
    extra_overrides: Sequence[str],
) -> DictConfig:
    """Compose the training task config, filling the paths it marks mandatory.

    configs/model/imagewam_flux2_klein_4b_base.yaml leaves flux2_model_path and
    ae_model_path as `???`, and ships a placeholder flux2_src_path; the training
    entrypoints supply all three from the environment. Serving must do the same.
    """
    register_default_resolvers()
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base="1.3"):
        return compose(
            config_name="train",
            overrides=[
                f"task={task_config}",
                f"model.flux2_model_path={flux2_model_path}",
                f"model.ae_model_path={ae_model_path}",
                f"model.flux2_src_path={flux2_src_path}",
                # The checkpoint is saved with checkpoint_format="lora_merged": the
                # LoRA deltas are already folded into the plain FLUX.2 weights. Keeping
                # LoRA modules here would leave freshly built fp32 lora_A/lora_B on a
                # bf16 backbone (load_state_dict preserves the destination dtype), which
                # fails the first matmul. Serving loads the merged weights directly.
                "model.flux2_lora_config.enabled=false",
                *extra_overrides,
            ],
        )


# --------------------------------------------------------------------------- #
# Per-connection DASH state                                                    #
# --------------------------------------------------------------------------- #
class DashSession:
    """Per-connection holder for the stateful DASH drift controller.

    The controller precomputes its ratio-jump table from `DashParams` +
    scheduler shift at construction, so it must be rebuilt when the panel changes
    DASH params live. A rebuild resets the per-episode drift history (the next
    replan is treated as the first) -- fine, since benchmarks don't retune mid
    episode. `reset()` clears drift history at each robot-episode boundary.
    """

    def __init__(self) -> None:
        self._ctrl: Optional[DashDriftController] = None
        self._sig: Any = None

    def controller(self, params: DashParams, scheduler_shift: float) -> DashDriftController:
        sig = (params, scheduler_shift)
        if self._ctrl is None or self._sig != sig:
            self._ctrl = DashDriftController(params, scheduler_shift=scheduler_shift)
            self._sig = sig
        return self._ctrl

    def reset(self) -> None:
        if self._ctrl is not None:
            self._ctrl.reset()


class RunningStats:
    """Online mean/variance (Welford) over effective FM steps, plus n / min / max.

    Accumulates across replans; reset when the regime changes (config apply /
    checkpoint swap) so the stats describe the currently-selected method.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._min: Optional[float] = None
        self._max: Optional[float] = None

    def add(self, value: float) -> None:
        x = float(value)
        self._n += 1
        delta = x - self._mean
        self._mean += delta / self._n
        self._m2 += delta * (x - self._mean)
        self._min = x if self._min is None else min(self._min, x)
        self._max = x if self._max is None else max(self._max, x)

    def summary(self) -> dict[str, Any]:
        var = self._m2 / (self._n - 1) if self._n > 1 else 0.0  # sample variance (ddof=1)
        return {
            "n": self._n,
            "mean": round(self._mean, 3) if self._n else None,
            "var": round(var, 4) if self._n else None,
            "std": round(var ** 0.5, 3) if self._n else None,
            "min": self._min,
            "max": self._max,
        }


# --------------------------------------------------------------------------- #
# Hot-swappable engine                                                         #
# --------------------------------------------------------------------------- #
class PolicyEngine:
    """Owns the single resident model; serves inference and swaps checkpoint/config live.

    Thread-safety: `_lock` serializes every model touch. The WebSocket handler thread
    calls `infer()`; the Gradio thread calls `swap_checkpoint()` / `update_runtime()`.
    Only one runs at a time, so a checkpoint reload briefly stalls inference (the robot
    client simply waits for its next chunk) rather than racing half-loaded weights.
    """

    def __init__(self, boot: BootConfig, cfg: DictConfig, profile: ActiveProfile,
                 runtime: RuntimeConfig) -> None:
        self._boot = boot
        self._cfg = cfg
        self._device = torch.device(boot.device)
        self._lock = threading.RLock()
        self._runtime = runtime
        self._profile = profile

        self._model = instantiate(cfg.model, model_dtype=torch.bfloat16, device=boot.device)
        self._model.load_checkpoint(str(profile.ckpt_path), optimizer=None)
        self._model.eval()
        self._model.requires_grad_(False)
        if self._model.stack != "flux2":
            raise ValueError(f"Serving path supports the flux2 stack only, got {self._model.stack!r}")
        # Resolved flow-matching shift; DASH's theoretical ratio table needs it.
        self._scheduler_shift = float(self._model.infer_action_scheduler.shift)

        self._images = ImagePreprocessor(boot.robotwin_camera_layout, self._device, self._model.torch_dtype)
        self._prompts = PromptBank(profile.prompt_embeds_path, self._device, self._model.torch_dtype)
        self._state = StateCodec(build_processor(cfg, profile.dataset_stats_path))
        self.last_timing: Optional[InferTiming] = None
        # When True, an unknown prompt from the robot is precomputed on the fly
        # (loads Qwen3 + stalls that one chunk) instead of erroring. Default off.
        self._auto_precompute = False
        # Running mean/variance of effective FM steps, reset on config/checkpoint change.
        self._fm_stats = RunningStats()

    # -- introspection ----------------------------------------------------- #
    @property
    def prompts(self) -> tuple[str, ...]:
        return self._prompts.tasks

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "profile": vars(self._profile).copy(),
                "runtime": self._runtime.summary(),
                "fm_step_stats": self._fm_stats.summary(),
                "prompts": list(self._prompts.tasks),
                "context_len": self._prompts.context_len,
            }

    def metadata(self) -> dict[str, Any]:
        with self._lock:
            return {
                "server": "serve_imagewam_g1",
                "action_dim": ACTION_DIM,
                "action_horizon": self._runtime.action_horizon,
                "exec_horizon": self._runtime.exec_horizon,
                "method": self._runtime.method,
                "num_inference_steps": self._runtime.baseline.num_inference_steps
                if self._runtime.method == "baseline"
                else getattr({"probeflow": self._runtime.probeflow, "dash": self._runtime.dash}[self._runtime.method],
                             "num_inference_steps", 10),
                "camera_layout": self._boot.robotwin_camera_layout,
                "context_len": self._prompts.context_len,
                "prompts": list(self._prompts.tasks),
                "checkpoint": str(self._profile.ckpt_path),
                "profile": self._profile.name,
            }

    def new_dash_session(self) -> DashSession:
        """One session per client connection; holds cross-replan drift state."""
        return DashSession()

    # -- live reconfiguration ---------------------------------------------- #
    def update_runtime(self, **changes: Any) -> dict[str, Any]:
        """Mutate method / horizon / per-method params. Returns the new summary."""
        with self._lock:
            if "method" in changes:
                method = str(changes.pop("method"))
                if method not in METHODS:
                    raise ValueError(f"Unknown method {method!r}; expected one of {METHODS}")
                self._runtime.method = method
            for key in ("action_horizon", "exec_horizon"):
                if key in changes:
                    setattr(self._runtime, key, int(changes.pop(key)))
            for group in ("baseline", "probeflow", "dash"):
                if group in changes and changes[group] is not None:
                    current = getattr(self._runtime, group)
                    setattr(self._runtime, group, replace(current, **changes.pop(group)))
            if changes:
                raise ValueError(f"Unrecognized runtime keys: {sorted(changes)}")
            self._fm_stats.reset()  # new regime -> fresh FM-step stats
            return self._runtime.summary()

    def swap_checkpoint(self, profile: ActiveProfile) -> dict[str, Any]:
        """Load a different run's weights + stats + prompt bank into the resident model.

        All G1 klein-4b profiles share the model architecture, so only the weights,
        normalization stats and prompt bank change -- the model is not rebuilt.
        """
        with self._lock:
            t0 = time.perf_counter()
            self._model.load_checkpoint(str(profile.ckpt_path), optimizer=None)
            self._model.eval()
            self._model.requires_grad_(False)
            self._prompts = PromptBank(profile.prompt_embeds_path, self._device, self._model.torch_dtype)
            self._state = StateCodec(build_processor(self._cfg, profile.dataset_stats_path))
            self._profile = profile
            self._runtime.prompt_override = None  # new bank may not contain the old override
            self._fm_stats.reset()  # new weights -> fresh FM-step stats
            torch.cuda.synchronize() if self._device.type == "cuda" else None
            log.info("swapped to profile %s (%.1fs)", profile.name, time.perf_counter() - t0)
            return self.snapshot()

    # -- prompt precompute ------------------------------------------------- #
    def set_auto_precompute(self, enabled: bool) -> None:
        with self._lock:
            self._auto_precompute = bool(enabled)

    def reset_fm_stats(self) -> dict[str, Any]:
        with self._lock:
            self._fm_stats.reset()
            return self._fm_stats.summary()

    def set_prompt_override(self, prompt: Optional[str]) -> dict[str, Any]:
        """Force every inference to use `prompt` (must be in the bank); None -> client's."""
        with self._lock:
            value = str(prompt).strip() if prompt else None
            if value is not None and value not in self._prompts:
                raise ValueError(f"prompt not in bank: {value!r}")
            self._runtime.prompt_override = value
            return self._runtime.summary()

    def add_prompt(self, text: str, persist: bool = True) -> dict[str, Any]:
        """Encode a new prompt in-process (lazy-loads Qwen3) and add it to the bank.

        Uses the model's own FLUX.2 text path (`_encode_flux2_prompts`), so the
        embedding is identical to the training/inference text encoder -- no format
        drift vs. the offline `export_g1_prompt_embeds.py` npz. Encodes at the bank's
        `context_len` (128 here, not Qwen3's 512 default) so it stacks into the npz.

        First call loads Qwen3 (~8 GB, ~15-30 s) and stalls inference for that once;
        it then stays resident. If `persist`, the active profile's npz is rewritten
        with the new prompt appended.
        """
        text = str(text).strip()
        with self._lock:
            if not text:
                return {"status": "empty"}
            if text in self._prompts:
                return {"status": "exists", "prompt": text, "prompts": list(self._prompts.tasks)}
            # Lazy-load the text encoder, then pin its length to the bank's context_len.
            if getattr(self._model, "text_encoder", None) is None:
                self._model._load_flux2_text_encoder_for_inference()
            self._model.text_encoder.max_length = int(self._prompts.context_len)
            with torch.no_grad():
                hidden, mask = self._model._encode_flux2_prompts([text])
            self._prompts.add(text, hidden, mask)
            result = {"status": "added", "prompt": text, "prompts": list(self._prompts.tasks)}
            if persist:
                result["npz"] = self._persist_prompts()
            return result

    def _persist_prompts(self) -> str:
        """Rewrite the active profile's npz with the full current bank (atomic replace)."""
        path = Path(self._profile.prompt_embeds_path)
        tasks, hidden, mask, context_len = self._prompts.export_arrays()
        # Tmp name must already end in .npz, else np.savez appends another .npz.
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez(
            tmp,
            tasks=np.array(tasks, dtype=object),
            hidden=hidden,
            mask=mask,
            context_len=np.int64(context_len),
        )
        os.replace(tmp, path)
        return str(path)

    # -- inference --------------------------------------------------------- #
    def infer(self, obs: Mapping[str, Any], dash_session: Optional[DashSession] = None) -> dict[str, Any]:
        with self._lock:
            runtime = replace(self._runtime)  # snapshot under lock; model call stays locked
            t0 = time.perf_counter()
            prompt = runtime.prompt_override or str(obs["prompt"])
            if self._auto_precompute and prompt not in self._prompts:
                log.info("auto-precomputing unknown prompt: %r", prompt)
                self.add_prompt(prompt, persist=True)
            context, context_mask = self._prompts.get(prompt)
            image = self._images(obs["image"])
            proprio = self._state.normalize_state(
                np.asarray(obs["state"], dtype=np.float32).reshape(-1)
            )
            t1 = time.perf_counter()

            base = dict(prompt=None, input_image=image, action_horizon=runtime.action_horizon,
                        proprio=proprio, context=context, context_mask=context_mask)

            with torch.no_grad():
                if runtime.method == "dash":
                    pred = self._infer_dash(base, image, runtime.dash, dash_session)
                elif runtime.method == "probeflow":
                    pred = self._model.infer_action(**base, **build_probeflow_kwargs(runtime.probeflow))
                else:
                    pred = self._model.infer_action(**base, **build_baseline_kwargs(runtime.baseline))
            t2 = time.perf_counter()

            actions = self._state.denormalize_action(pred["action"])
            if actions.shape[1] != ACTION_DIM:
                raise ValueError(f"Expected action dim {ACTION_DIM}, got {actions.shape}")
            t3 = time.perf_counter()

            effective = _effective_steps(pred)
            if effective is not None:
                self._fm_stats.add(effective)
            self.last_timing = InferTiming(
                preprocess_s=t1 - t0,
                infer_s=t2 - t1,
                postprocess_s=t3 - t2,
                effective_steps=effective,
                dash=_dash_info(pred) if runtime.method == "dash" else None,
            )
            return {"actions": actions}

    def _infer_dash(self, base: dict[str, Any], image: torch.Tensor, params: DashParams,
                    session: Optional[DashSession]) -> dict[str, Any]:
        """DASH: externally encode ref tokens, plan the ratio jump, run, then commit.

        Mirrors deploy_policy: pre-encode the current visual latent, derive the jump
        step from drift vs. the previous replan, pass the tokens back into the same
        infer call (so the backbone does not re-encode), and store them as previous.
        """
        session = session if session is not None else DashSession()
        ctrl = session.controller(params, self._scheduler_shift)
        ref_tokens, ref_img_ids = self._model._encode_flux2_image_tokens(
            image, time_value=FLUX2_REF_ENCODE_TIME_VALUE
        )
        plan = ctrl.plan_replan(ref_tokens, ref_img_ids, replan_index=ctrl._replan_count)
        dash_base = {
            "num_inference_steps": int(params.num_inference_steps),
            "sigma_shift": params.sigma_shift,
            "seed": params.seed,
        }
        pred = self._model.infer_action(**base, **dash_base, **plan.infer_kwargs)
        ctrl.commit(ref_tokens)
        if isinstance(pred, dict) and plan.step_k is not None:
            pred.setdefault("_dash", {"step_k": plan.step_k, "drift": plan.drift, "is_jump": plan.is_jump})
        return pred


# --------------------------------------------------------------------------- #
# WebSocket wire loop                                                          #
# --------------------------------------------------------------------------- #
def load_wire_codec(g1_client_path: Path) -> tuple[Any, Any]:
    """Import the client's own msgpack codec; both sides must use the same module."""
    if str(g1_client_path) not in sys.path:
        sys.path.insert(0, str(g1_client_path))
    from g1_client.msgpack_numpy import Packer, unpackb

    return Packer, unpackb


def serve(engine: PolicyEngine, host: str, port: int, codec: tuple[Any, Any],
          ready: Optional[threading.Event] = None) -> None:
    Packer, unpackb = codec
    packer = Packer()

    def handler(connection) -> None:
        peer = connection.remote_address
        log.info("client connected: %s", peer)
        # Per-connection DASH state: one robot stream == one episode sequence. The
        # session is inert unless the active method is "dash".
        dash_session = engine.new_dash_session()
        connection.send(packer.pack(engine.metadata()))
        try:
            while True:
                obs = unpackb(connection.recv())
                if obs.get("reset"):
                    dash_session.reset()
                    log.info("reset: prompt=%r", obs.get("prompt"))
                    connection.send(packer.pack({"ok": True}))
                    continue
                try:
                    result = engine.infer(obs, dash_session=dash_session)
                except Exception:
                    # PolicyClient raises on a str payload; use it as the error channel
                    # instead of dropping the connection with the cause hidden.
                    detail = traceback.format_exc()
                    log.error("inference failed:\n%s", detail)
                    connection.send(detail)
                    continue
                connection.send(packer.pack(result))
                timing = engine.last_timing
                if timing.dash is not None:
                    d = timing.dash
                    extra = (
                        f" | DASH FMsteps={d.get('parallel_calls','?')}"
                        f" jump={d.get('base_step','?')}->{d.get('chosen_step','?')}"
                        f" drift={d.get('drift','?')}"
                    )
                elif timing.effective_steps is not None:
                    extra = f" | steps={timing.effective_steps:.1f}"
                else:
                    extra = ""
                log.info(
                    "chunk H=%d | pre=%.0fms infer=%.0fms post=%.0fms%s",
                    result["actions"].shape[0],
                    timing.preprocess_s * 1e3,
                    timing.infer_s * 1e3,
                    timing.postprocess_s * 1e3,
                    extra,
                )
        except Exception as error:  # noqa: BLE001 - report to the client, keep serving
            log.warning("client %s disconnected: %s", peer, error)

    log.info("listening on ws://%s:%d", host, port)
    with websockets.sync.server.serve(
        handler, host, port, compression=None, max_size=None, ping_interval=None
    ) as server:
        if ready is not None:
            ready.set()
        server.serve_forever()


# --------------------------------------------------------------------------- #
# Standalone entrypoint (single fixed baseline config)                        #
# --------------------------------------------------------------------------- #
def build_engine(
    ckpt: Path,
    dataset_stats: Path,
    prompt_embeds: Path,
    task_config: str,
    flux2_model_path: Path,
    ae_model_path: Path,
    flux2_src_path: Path,
    device: str,
    camera_layout: str,
    host: str,
    port: int,
    runtime: RuntimeConfig,
    overrides: Sequence[str] = (),
    profile_name: str = "cli",
) -> PolicyEngine:
    boot = BootConfig(
        task_config=task_config,
        host=host,
        port=port,
        device=device,
        robotwin_camera_layout=camera_layout,
        flux2_model_path=flux2_model_path,
        ae_model_path=ae_model_path,
        flux2_src_path=flux2_src_path,
    )
    cfg = load_config(task_config, flux2_model_path, ae_model_path, flux2_src_path, overrides)
    log.info("config: %s", OmegaConf.to_container(cfg.model, resolve=True)["_target_"])
    profile = ActiveProfile(
        name=profile_name,
        ckpt_path=ckpt,
        dataset_stats_path=dataset_stats,
        prompt_embeds_path=prompt_embeds,
        task_config=task_config,
    )
    return PolicyEngine(boot, cfg, profile, runtime)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--dataset-stats", type=Path, required=True)
    parser.add_argument("--prompt-embeds", type=Path, required=True)
    parser.add_argument("--g1-client-path", type=Path, required=True)
    # Defaults come from .env.local, the same source the training entrypoints read.
    parser.add_argument("--flux2-model-path", type=Path, default=os.environ.get("FLUX2_MODEL_PATH"))
    parser.add_argument("--ae-model-path", type=Path, default=os.environ.get("FLUX2_AE_MODEL_PATH"))
    parser.add_argument("--flux2-src-path", type=Path, default=os.environ.get("FLUX2_SRC"))
    parser.add_argument("--task-config", default="g1_flux2_klein_4b_base_imagewam")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--action-horizon", type=int, default=16, help="steps per chunk")
    parser.add_argument("--exec-horizon", type=int, default=16, help="advisory client exec steps")
    parser.add_argument("--num-inference-steps", type=int, default=10, help="flow-matching steps")
    parser.add_argument("--method", default="baseline", choices=list(METHODS))
    parser.add_argument("--sigma-shift", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--robotwin-camera-layout", default="compact_288x256")
    parser.add_argument("overrides", nargs="*", default=[], help="extra hydra overrides")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    for name, value in (
        ("--flux2-model-path/FLUX2_MODEL_PATH", args.flux2_model_path),
        ("--ae-model-path/FLUX2_AE_MODEL_PATH", args.ae_model_path),
        ("--flux2-src-path/FLUX2_SRC", args.flux2_src_path),
    ):
        if value is None:
            raise ValueError(f"{name} is required (set it or source .env.local)")

    runtime = RuntimeConfig(
        method=args.method,
        action_horizon=args.action_horizon,
        exec_horizon=args.exec_horizon,
        baseline=BaselineParams(
            num_inference_steps=args.num_inference_steps,
            sigma_shift=args.sigma_shift,
            seed=args.seed,
        ),
    )
    codec = load_wire_codec(args.g1_client_path)
    engine = build_engine(
        ckpt=args.ckpt,
        dataset_stats=args.dataset_stats,
        prompt_embeds=args.prompt_embeds,
        task_config=args.task_config,
        flux2_model_path=args.flux2_model_path,
        ae_model_path=args.ae_model_path,
        flux2_src_path=args.flux2_src_path,
        device=args.device,
        camera_layout=args.robotwin_camera_layout,
        host=args.host,
        port=args.port,
        runtime=runtime,
        overrides=args.overrides,
    )
    log.info("metadata: %s", engine.metadata())
    serve(engine, args.host, args.port, codec)


if __name__ == "__main__":
    main()
