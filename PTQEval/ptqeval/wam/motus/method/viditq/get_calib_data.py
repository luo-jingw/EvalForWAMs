"""Calibration data collection for Motus SmoothQuant (w4a4_smooth / any
smooth_quant config). Only needed when a config sets smooth_quant: true; the
QuaRoT-only w4a4.yaml is data-free and does not use this.

install_calib_hooks(model, out_path) registers a forward_pre_hook on the Motus
target Linears (per expert per block), each accumulating a per-input-channel
running absmax keyed by full module name.

Install on the WHOLE Motus model (policy.model), the same object motus ptq walks,
so the hook keys are exactly the names ptq looks up. named_modules reports the
first-registered top-level names, e.g.
  "video_model.wan_model.blocks.5.self_attn.q" / "action_expert.blocks.0.ffn.0".

Target Linears (all in .blocks.):
  video : video_model.wan_model.blocks.N.self_attn.{q,k,v,o} + ffn.{0,2}
  action: action_expert.blocks.N.wan_action_o + ffn.{0,2}
  und   : und_expert.blocks.N.wan_und_o + ffn.{0,2}
(action/und Q/K/V are fused nn.Parameter -> not Linear -> not hooked/quantized.)

The accumulator / pool-safe dump / atexit+SIGTERM machinery is reused verbatim
from the lingbot_va collector (_CalibState / _make_hook are WAM-agnostic).
"""
from __future__ import annotations

import atexit
import logging
import signal

import torch.nn as nn

import ptqeval.wam.motus  # noqa: F401
from ptqeval.wam.lingbot_va.method.viditq.get_calib_data import (
    _CalibState,
    _make_hook,
)

logger = logging.getLogger("ptqeval.wam.motus.method.viditq.get_calib_data")

_TARGET_SUFFIXES: tuple[str, ...] = (
    "self_attn.q",
    "self_attn.k",
    "self_attn.v",
    "self_attn.o",
    "wan_action_o",
    "wan_und_o",
    "ffn.0",
    "ffn.2",
)


def _is_target(name: str) -> bool:
    # Must be inside an expert transformer block (video_model.wan_model.blocks.N /
    # action_expert.blocks.N / und_expert.blocks.N).
    if ".blocks." not in name:
        return False
    return any(name.endswith("." + s) for s in _TARGET_SUFFIXES)


def install_calib_hooks(model: nn.Module, out_path: str) -> _CalibState:
    """Install per-layer input-absmax hooks on the Motus target Linears.

    Returns the _CalibState; call state.dump() (also on atexit / SIGTERM) to
    persist dict[full_module_name -> bf16 per-channel absmax] to out_path.
    """
    state = _CalibState(out_path)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and _is_target(name):
            state.handles.append(
                module.register_forward_pre_hook(_make_hook(state, name))
            )
    logger.info(
        f"install_calib_hooks: registered {len(state.handles)} hooks on the "
        f"full Motus model (video 6/block + action 3/block + und 3/block "
        f"x 30 blocks) -> {out_path}"
    )
    atexit.register(state.dump)

    def _on_sigterm(signum, frame):
        logger.info("SIGTERM received; dumping calib data and exiting.")
        state.dump()
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        import os
        os.kill(os.getpid(), signal.SIGTERM)

    try:
        signal.signal(signal.SIGTERM, _on_sigterm)
    except ValueError:
        logger.warning("install_calib_hooks: not main thread; SIGTERM dump disabled.")
    return state


# On-policy calibration flow (mirrors fastwam/imagewam): the RoboTwin eval is run
# at variant=bf16 with a `calib_out` arg; policy.get_model installs these hooks on
# policy.model, and each pooled per-task process merges its per-channel absmax into
# the shared calib_out via _CalibState's flock merge-on-write. Then run ptq with
# w4a4_smooth.yaml (--calib_data_path <calib_out>).
