# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Phase 31: Calibration data collection.

install_calib_hooks(model, out_path) walks model.blocks (the 30
WanTransformerBlocks) and installs a forward_pre_hook on the 180
Linears that block.py swaps to W8A8 (attn1.to_q/k/v/out[0] +
ffn.net[0].proj + ffn.net[2]). Each hook updates a per-layer running
per-channel input absmax tensor on CPU bf16. A persistent state is
dumped to out_path:
  - on Python exit (atexit)
  - on SIGTERM (signal handler; run_eval.sh cleanup sends this)
  - every PERIODIC_DUMP_INTERVAL hook invocations as a crash-safety net.

Output schema (matches Part V Phase 32 / Part VI Phase 36 consumers):
  dict[str, torch.Tensor]
    key   = full module name (e.g. "blocks.0.attn1.to_q")
    value = bf16 [in_features], per-input-channel max(|x|) across all
            forward calls and all token positions seen so far.

Single-tensor-per-layer (NOT per-call) keeps storage trivial (~1 MB
total) while still being sufficient for SmoothQuant's channel_mask
formula and for the simple max-based static activation scale used in
the Part V plan.
"""
from __future__ import annotations

import atexit
import logging
import os
import signal
from typing import Optional

import torch
import torch.nn as nn


logger = logging.getLogger("ptqeval.wam.lingbot_va.method.viditq.get_calib_data")


# Names AFTER block.py swap (which is also the form ptq.py uses for
# remain_fp_regex filter). Listed relative to a WanTransformerBlock.
_TARGET_SUFFIXES: tuple[str, ...] = (
    "attn1.to_q",
    "attn1.to_k",
    "attn1.to_v",
    "attn1.to_out.0",
    "ffn.net.0.proj",
    "ffn.net.2",
)

PERIODIC_DUMP_INTERVAL: int = 100   # hook invocations between safety dumps


class _CalibState:
    def __init__(self, out_path: str) -> None:
        self.out_path = out_path
        # Load existing dump if present so sequential calib runs across
        # multiple tasks accumulate via running max instead of overwriting.
        # Safe with single-server sequential invocation (e.g. a bash loop
        # over SELECTED_15_TASKS); NOT safe with pool-mode parallel runs
        # (multiple servers writing same path = race condition).
        if os.path.exists(out_path):
            try:
                self.absmax: dict[str, torch.Tensor] = torch.load(
                    out_path, weights_only=True, map_location="cpu"
                )
                logger.info(
                    f"calib resume: loaded existing dump ({len(self.absmax)} layers) "
                    f"from {out_path}; new stats will be max-merged in place."
                )
            except Exception as e:
                logger.warning(
                    f"calib resume: failed to load existing dump from {out_path} "
                    f"({e}); starting fresh."
                )
                self.absmax = {}
        else:
            self.absmax = {}
        self.call_count = 0
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

    def update(self, name: str, x: torch.Tensor) -> None:
        # x: [..., C_in], bf16. Reduce over leading dims, keep per-channel.
        flat = x.detach().reshape(-1, x.shape[-1])
        cur = flat.abs().amax(dim=0).to(torch.bfloat16).cpu()
        prev = self.absmax.get(name)
        if prev is None:
            self.absmax[name] = cur
        else:
            # element-wise running max in bf16; cast through fp32 to avoid
            # bf16 max() ambiguity on equal values.
            self.absmax[name] = torch.maximum(prev.float(), cur.float()).to(torch.bfloat16)
        self.call_count += 1
        if self.call_count % PERIODIC_DUMP_INTERVAL == 0:
            self.dump()

    def dump(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.out_path)), exist_ok=True)
        # Atomic-ish: write to .tmp then rename. Avoids torch.load on a
        # half-written file if SIGKILL hits mid-dump.
        tmp = self.out_path + ".tmp"
        torch.save(self.absmax, tmp)
        os.replace(tmp, self.out_path)
        logger.info(
            f"calib dump (call_count={self.call_count}, layers={len(self.absmax)}) "
            f"-> {self.out_path}"
        )


def _make_hook(state: _CalibState, name: str):
    def hook(module: nn.Module, inputs: tuple) -> None:
        x = inputs[0]
        if not isinstance(x, torch.Tensor):
            return
        state.update(name, x)
    return hook


def install_calib_hooks(model: nn.Module, out_path: str) -> _CalibState:
    """Install per-layer absmax hooks on the 180 target Linears in model.

    Identifies targets by name suffix (attn1.to_{q,k,v,out.0} / ffn.net.0.proj
    / ffn.net.2) anywhere under a parent module named with "blocks.N"
    pattern (model.blocks[i]).
    """
    state = _CalibState(out_path)

    def is_target(name: str) -> bool:
        # Must live inside blocks.<idx> (excludes any other matching name,
        # e.g. action_head if it ever existed at top level).
        if not name.startswith("blocks."):
            return False
        # The path between "blocks.N." and the suffix must not introduce
        # extra dots (avoids matching attn1.to_q.something.subname).
        for suffix in _TARGET_SUFFIXES:
            if name.endswith("." + suffix):
                return True
        return False

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not is_target(name):
            continue
        handle = module.register_forward_pre_hook(_make_hook(state, name))
        state.handles.append(handle)

    logger.info(
        f"install_calib_hooks: registered {len(state.handles)} hooks "
        f"(expected 180 for stock WAN 30-block model) -> {out_path}"
    )

    # Dump on normal Python exit.
    atexit.register(state.dump)

    # Dump on SIGTERM (run_eval.sh cleanup sends this when the client is
    # done). Signal handlers must be installed in the main thread, which
    # is the case here because server init runs on the main thread.
    def _on_sigterm(signum, frame):
        logger.info("SIGTERM received; dumping calib data and exiting.")
        state.dump()
        # Re-raise default behavior so the rest of the server shutdown runs.
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        os.kill(os.getpid(), signal.SIGTERM)
    try:
        signal.signal(signal.SIGTERM, _on_sigterm)
    except ValueError:
        # Not the main thread; fall back to atexit-only behavior.
        logger.warning(
            "install_calib_hooks: not in main thread; SIGTERM dump disabled, "
            "atexit + periodic dump remain active."
        )

    return state
