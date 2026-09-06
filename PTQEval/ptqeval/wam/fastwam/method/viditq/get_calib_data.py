"""Calibration data collection for FastWAM SmoothQuant (w4a4_smooth / any
smooth_quant config). Only needed when a config sets smooth_quant: true; the
QuaRoT-only w4a4.yaml is data-free and does not use this.

install_calib_hooks(model, out_path) registers a forward_pre_hook on the FastWAM
target Linears (per expert per block: self_attn.{q,k,v,o} + ffn.{0,2}), each
accumulating a per-input-channel running absmax keyed by full module name.

Install on the WHOLE FastWAM model (policy.model), the same object fastwam ptq
passes to compute_int_state_dict, so the hook keys are exactly the names ptq
looks up. The experts are double-registered and named_modules uses the
first-registered top-level name WITHOUT a `mot.` prefix, e.g.
"video_expert.blocks.5.self_attn.q" / "action_expert.blocks.0.ffn.0" (verified in
the w4a4_smooth smoke: all 348 quantized layers received act_channel_div, i.e.
zero missing_calib). Output = dict[str, bf16[C_in]]. (blocks.0 is also hooked but
unused downstream — W4A4 keeps blocks.0 FP.)

The accumulator / pool-safe dump / atexit+SIGTERM machinery is reused verbatim
from the lingbot_va collector (_CalibState / _make_hook are WAM-agnostic and
pull in no wan_va); only the target-identification differs (FastWAM's
`{video,action}_expert.blocks.N.<suffix>` naming vs lingbot_va's
`blocks.N.attn1.*`).

NOT run yet: producing calib_data.pth requires a bf16 forward pass over
representative episodes with these hooks installed. See the module footer for
the intended invocation.
"""
from __future__ import annotations

import atexit
import logging
import signal

import torch.nn as nn

import ptqeval.wam.fastwam  # noqa: F401
# Reuse the WAM-agnostic accumulator + hook + pool-safe dump (no wan_va pulled).
from ptqeval.wam.lingbot_va.method.viditq.get_calib_data import (
    _CalibState,
    _make_hook,
)

logger = logging.getLogger("ptqeval.wam.fastwam.method.viditq.get_calib_data")

# Target Linear suffixes, per expert per block (same set ptq.py quantizes and
# loader.py swaps). Both experts, all blocks except blocks.0 (kept FP), are
# hooked; ptq only consumes calib entries for layers it actually quantizes.
_TARGET_SUFFIXES: tuple[str, ...] = (
    "self_attn.q",
    "self_attn.k",
    "self_attn.v",
    "self_attn.o",
    "ffn.0",
    "ffn.2",
)


def _is_target(name: str) -> bool:
    # Must be inside an expert block, e.g. "video_expert.blocks.5.self_attn.q".
    if "_expert.blocks." not in name:
        return False
    return any(name.endswith("." + s) for s in _TARGET_SUFFIXES)


def install_calib_hooks(model: nn.Module, out_path: str) -> _CalibState:
    """Install per-layer input-absmax hooks on the FastWAM target Linears.

    Returns the _CalibState; call state.dump() (also done on atexit / SIGTERM)
    to persist dict[full_module_name -> bf16 per-channel absmax] to out_path.
    """
    state = _CalibState(out_path)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and _is_target(name):
            state.handles.append(
                module.register_forward_pre_hook(_make_hook(state, name))
            )
    logger.info(
        f"install_calib_hooks: registered {len(state.handles)} hooks on the "
        f"full model (6 suffixes x 30 blocks x 2 experts = 360; blocks.0 is "
        f"collected but unused since W4A4 keeps it FP) -> {out_path}"
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


# Intended invocation (NOT run yet — no calibration performed):
#   Collect activation stats with a bf16 forward pass over representative
#   episodes, then run ptq with w4a4_smooth.yaml:
#     1. build the bf16 FastWAM model (ptq.load_fastwam_model, load_text_encoder
#        as needed) and `install_calib_hooks(model, calib_out)`;
#     2. run infer_action over N calib episodes (real obs+prompt) so the hooks
#        see representative activations; state.dump() writes calib_data.pth;
#     3. python -m ptqeval.wam.fastwam.method.viditq.ptq
#            --ckpt ... --layer_config .../configs/w4a4_smooth.yaml
#            --output results/fastwam/fastwam_w4a4_smooth/calib/int_weights.pth
#        (ptq reads calib_data_path from the config and applies SmoothQuant).
