"""Calibration data collection for ImageWAM SmoothQuant (w4a4_smooth / any
smooth_quant config). Only needed when a config sets smooth_quant: true; the
QuaRoT-only w4a4.yaml is data-free and does not use this.

install_calib_hooks(model, out_path) registers a forward_pre_hook on the ImageWAM
FLUX.2 DiT target Linears (per expert per block), each accumulating a
per-input-channel running absmax keyed by full module name.

Install on the WHOLE ImageWAM model (policy.model), the same object imagewam ptq
passes to compute_int_state_dict, so the hook keys are exactly the names ptq
looks up. The experts are registered before self.mot, so named_modules reports
the top-level names WITHOUT a `mot.` prefix, e.g.
"video_expert.transformer.double_blocks.0.img_attn.qkv" /
"action_expert.single_blocks.5.linear1". Output = dict[str, bf16[C_in]].

Target Linears (all bias=False), gated to the DiT blocks (double_blocks./
single_blocks.):
  video (MMDiT): double_blocks.N.{img,txt}_attn.{qkv,proj} + {img,txt}_mlp.{0,2}
                 single_blocks.N.{linear1,linear2}
  action (SlimFlux2): double_blocks.N.img_attn.{qkv,proj} + img_mlp.{0,2}
                      single_blocks.N.{linear1,linear2}

The accumulator / pool-safe dump / atexit+SIGTERM machinery is reused verbatim
from the lingbot_va collector (_CalibState / _make_hook are WAM-agnostic and
pull in no wan_va); only the target-identification (suffix set + block gate)
differs.
"""
from __future__ import annotations

import atexit
import logging
import signal

import torch.nn as nn

import ptqeval.wam.imagewam  # noqa: F401
# Reuse the WAM-agnostic accumulator + hook + pool-safe dump (no wan_va pulled).
from ptqeval.wam.lingbot_va.method.viditq.get_calib_data import (
    _CalibState,
    _make_hook,
)

logger = logging.getLogger("ptqeval.wam.imagewam.method.viditq.get_calib_data")

# Target Linear suffixes (same set ptq.py quantizes and loader.py swaps). Gated
# to double_blocks./single_blocks. so only the DiT block Linears are hooked
# (modulation `.lin`, img_in/txt_in/time_in, final_layer, AE, etc. are excluded);
# ptq only consumes calib entries for the layers it actually quantizes.
_TARGET_SUFFIXES: tuple[str, ...] = (
    "img_attn.qkv",
    "img_attn.proj",
    "img_mlp.0",
    "img_mlp.2",
    "txt_attn.qkv",
    "txt_attn.proj",
    "txt_mlp.0",
    "txt_mlp.2",
    "linear1",
    "linear2",
)


def _is_target(name: str) -> bool:
    # Must be inside a FLUX.2 DiT block, e.g.
    # "video_expert.transformer.double_blocks.0.img_attn.qkv".
    if ".double_blocks." not in name and ".single_blocks." not in name:
        return False
    return any(name.endswith("." + s) for s in _TARGET_SUFFIXES)


def install_calib_hooks(model: nn.Module, out_path: str) -> _CalibState:
    """Install per-layer input-absmax hooks on the ImageWAM target Linears.

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
        f"full ImageWAM model (video 8/double x5 + 2/single x20; action "
        f"4/double x5 + 2/single x20 = up to 140) -> {out_path}"
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


# On-policy calibration flow (mirrors fastwam): the RoboTwin eval is run at
# variant=bf16 with a `calib_out` arg; policy.get_model then installs these hooks
# on policy.model, and each pooled per-task process merges its per-channel absmax
# into the shared calib_out via _CalibState's flock merge-on-write. Then:
#   python -m ptqeval.wam.imagewam.method.viditq.ptq
#       --ckpt ... --layer_config .../configs/w4a4_smooth.yaml
#       --calib_data_path <calib_out> --output .../int_weights.pth
#   (ptq reads calib_data and applies SmoothQuant to the layers it quantizes.)
