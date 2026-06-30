# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Phase 44 observational check — text-condition cache (44a/44b).

Two modes, both observational (principle.txt L12: emit metrics only, no
assert / no PASS judgement):

  (default) cache inspection — no model needed, fast:
    per-entry {key_consistent, seq_len, dim, embed_l2}, plus whether the
    empty-negative key is present.

  --reencode (needs GPU + model): builds the server, re-encodes a few
    cached prompts live, prints max_abs( cached - live ) so the caller can
    see how deterministic the T5 path is (cache validity).

    python -m ptqeval.wam.lingbot_va.method.viditq.check_text_cond_cache \\
        --cache results/text_cond_cache.pt
"""
from __future__ import annotations

import argparse
import sys

import torch

import ptqeval.wam.lingbot_va  # noqa: F401
from ptqeval.wam.lingbot_va.text_cond_cache import (
    cache_key, load_cache)

_MAX_SEQ = 512


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, help="text_cond_cache .pt path.")
    ap.add_argument("--reencode", type=int, default=0,
                    help="Re-encode the first N prompts live and print "
                         "max_abs vs cached (needs GPU + model). 0 = skip.")
    ap.add_argument("--model_path",
                    default="models/lingbot-va-posttrain-robotwin")
    args = ap.parse_args()

    cache = load_cache(args.cache)
    empty_key = cache_key("", _MAX_SEQ)
    print(f"entries={len(cache)}  empty_negative_present={empty_key in cache}")
    print(f"{'key_consistent':>14} {'seq':>4} {'dim':>5} {'embed_l2':>10}  prompt")
    for k, e in cache.items():
        consistent = (k == cache_key(e.prompt, e.max_sequence_length))
        l2 = e.prompt_embeds.float().norm().item()
        print(f"{str(consistent):>14} {e.seq_len:>4} {e.dim:>5} {l2:>10.2f}  "
              f"'{e.prompt[:44]}'")

    if args.reencode > 0:
        from pathlib import Path
        from ptqeval.eval.measure_flops import _build_server
        server = _build_server(args.model_path,
                               Path("/tmp/check_text_cond_scratch"))
        server.text_encoder.to(server.device)
        items = list(cache.values())[:args.reencode]
        print(f"\n--reencode {len(items)} prompts (max_abs cached vs live):")
        for e in items:
            live, _ = server.encode_prompt(
                prompt=e.prompt, negative_prompt=None,
                do_classifier_free_guidance=False, num_videos_per_prompt=1,
                prompt_embeds=None, negative_prompt_embeds=None,
                max_sequence_length=e.max_sequence_length,
                device=server.device, dtype=server.dtype)
            live = live.detach().to("cpu").float()
            cached = e.prompt_embeds.float()
            max_abs = (live - cached).abs().max().item()
            print(f"  max_abs={max_abs:.3e}  '{e.prompt[:40]}'")
        server.text_encoder.to("cpu")
    return 0


if __name__ == "__main__":
    sys.exit(main())
