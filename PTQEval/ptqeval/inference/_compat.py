# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""torch compat shims for old Jetson torch builds (one job: dtype symbols).

JetPack 5.1.1 ships torch 2.1.0a0 (nv23.06) — an alpha built from a June-2023
snapshot whose Python API predates the FP8 dtype symbols (torch.float8_e4m3fn
etc., which landed in the 2.1 release). diffusers / transformers reference
those symbols at import time, so importing them raises `AttributeError:
module 'torch' has no attribute 'float8_e4m3fn'`.

The w4a4 / bf16 deploy path never uses FP8 (int4 weights + bf16 activations;
Orin/Ampere has no FP8 hardware anyway), so we register harmless placeholders
(a real torch.dtype, uint8) for any missing FP8 symbol. This only makes the
symbol EXIST so the library import succeeds; no FP8 tensor is ever created on
the deploy path. On a torch build that already has them, this is a no-op.

Import this module BEFORE importing diffusers/transformers (the inference
package __init__ does so first).
"""
import sys
import types

import torch

# ---- FP8 dtype symbols (added in torch 2.1 final; absent in the nv23.06
# alpha). Placeholder = a real torch.dtype (uint8); never used on the
# int4/bf16 deploy path. ----
_FP8_NAMES = (
    "float8_e4m3fn",
    "float8_e5m2",
    "float8_e4m3fnuz",
    "float8_e5m2fnuz",
)

for _name in _FP8_NAMES:
    if not hasattr(torch, _name):
        setattr(torch, _name, torch.uint8)


# ---- torch.compiler namespace (added in torch 2.1 final; absent in the
# nv23.06 alpha). diffusers/transformers reference torch.compiler.disable /
# is_compiling at import time. The deploy path never torch.compile()s
# (attn_mode='torch'), so provide a no-op namespace: is_compiling -> False,
# disable -> identity decorator, any other attr -> pass-through callable. ----
def _install_fake_compiler() -> None:
    mod = types.ModuleType("torch.compiler")

    def disable(fn=None, *args, **kwargs):
        # supports @disable, @disable(...), and disable(fn)
        if callable(fn):
            return fn
        def _wrap(f):
            return f
        return _wrap

    def _false(*args, **kwargs) -> bool:
        return False

    def compile(model=None, *args, **kwargs):
        return model if model is not None else (lambda f: f)

    mod.disable = disable
    mod.is_compiling = _false
    mod.is_dynamo_compiling = _false
    mod.compile = compile

    def _fallback(name):
        # unknown attr -> callable that is a no-op, or identity if used as a
        # decorator (@torch.compiler.something).
        def _f(*args, **kwargs):
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]
            return None
        return _f

    mod.__getattr__ = _fallback  # PEP 562 module-level __getattr__
    torch.compiler = mod
    sys.modules["torch.compiler"] = mod


if not hasattr(torch, "compiler"):
    _install_fake_compiler()
