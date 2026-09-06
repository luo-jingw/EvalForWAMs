# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Deployment websocket server (Phase 6).

Serves an InferenceEngine over the same websocket protocol as
lingbot-va/wan_va/wan_va_server.py (the upstream reference): it builds the
model, then hands it to wan_va's run_async_server_mode, which wraps it in
a WebsocketPolicyServer that calls model.infer(request) per message.

The wire protocol is the VA_Server.infer dict protocol -- identical to
what the g1-client (g1_client/lingbot_va) already sends:
    {'reset': True, 'prompt': str}          -> reset
    {'obs': ..., 'prompt': ...}             -> {'action': [16, F, S]}
    {'obs': ..., 'compute_kv_cache': True}  -> advance KV cache
So an existing g1-client connects unchanged; only --server-host/--server-
port (and the task prompt) differ.

Unlike wan_va_server.py this path loads the QUANTIZED transformer + the
precomputed text-cond cache (via InferenceConfig / VA_Server), so the
device runs w4a4 and T5-free.

    python -m ptqeval.inference.server_ws \\
        --config ptqeval/inference/configs/deploy.yaml \\
        --host 0.0.0.0 --port 29056
"""
from __future__ import annotations

import argparse
import sys

from ptqeval.inference.config import InferenceConfig
from ptqeval.inference.engine import InferenceEngine


def serve(config: InferenceConfig, host: str = "0.0.0.0",
          port: int = 29056) -> None:
    """Build the engine and serve forever. Blocks. The engine's __init__
    brings up the single-process group + VA_Server; run_async_server_mode
    (rank 0) then runs the WebsocketPolicyServer loop."""
    engine = InferenceEngine(config)
    # wan_va bare import (engine.__init__ already put wan_va on sys.path).
    from utils import run_async_server_mode
    run_async_server_mode(engine, local_rank=config._local_rank(),
                          host=host, port=port)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True,
                    help="InferenceConfig yaml (see configs/deploy.yaml).")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=29056)
    args = ap.parse_args()
    serve(InferenceConfig.from_yaml(args.config), args.host, args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
