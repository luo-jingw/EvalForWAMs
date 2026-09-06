# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""sys.path setup for wan_va's bare imports (one job: path).

wan_va uses bare top-level imports (`from configs import VA_CONFIGS`,
`from distributed.util import ...`, `from modules.utils import ...`) that
resolve relative to lingbot-va/wan_va/. The ptqeval.wam.lingbot_va package
init puts lingbot-va/ on sys.path (so `from wan_va.* import` works) but not
wan_va/ itself; server.py adds it inline. This helper does the same so the
inference package can call the bare-import modules directly.
"""
from __future__ import annotations

import os
import sys

import ptqeval.wam.lingbot_va as _lingbot_va_pkg


def ensure_wan_va_on_path() -> None:
    """Idempotently add lingbot-va/wan_va/ to sys.path."""
    wan_va_dir = os.path.join(_lingbot_va_pkg.LINGBOT_VA_PATH, "wan_va")
    if wan_va_dir not in sys.path:
        sys.path.insert(0, wan_va_dir)
