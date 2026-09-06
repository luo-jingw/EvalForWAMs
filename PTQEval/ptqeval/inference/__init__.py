# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Robot-deployment inference package for lingbot-va.

Two offline weight-prep helpers + one runtime engine (+ optional
metrics). See infer_plan.txt for the module map. Export declarations only;
implementations live in the per-module files (one file, one job).
"""
# FP8 dtype shim for old Jetson torch builds — MUST run before any
# diffusers/transformers import (see _compat.py).
import ptqeval.inference._compat  # noqa: F401

from ptqeval.inference.config import InferenceConfig
from ptqeval.inference.engine import InferenceEngine
from ptqeval.inference.metrics import MeasuredEngine
from ptqeval.inference.weight_prep import build_int_weights
from ptqeval.inference.precompute_text import build_text_cond_cache

__all__ = [
    "InferenceConfig",
    "InferenceEngine",
    "MeasuredEngine",
    "build_int_weights",
    "build_text_cond_cache",
]
