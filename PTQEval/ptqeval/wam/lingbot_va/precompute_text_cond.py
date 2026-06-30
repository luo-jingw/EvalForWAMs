# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Phase 44b: precompute the text-condition cache (offline).

Builds the VA server (T5 on GPU), encodes every unique RoboTwin prompt
plus the empty negative prompt, and stores the T5 outputs via
text_cond_cache so eval can inject them (server _reset) and skip the
text encoder entirely -- so T5 never occupies VRAM during eval.

Prompts are read from the `task` field of obs_data_*.pt under
<videos_root>/visualization/real/ (the exact strings the model saw), so
a cached key matches the eval-time prompt verbatim.

Observational (principle.txt L12): prints per-prompt seq_len + embed L2;
no assert / no PASS judgement.

    python -m ptqeval.wam.lingbot_va.precompute_text_cond \\
        --videos_root results/bf16 \\
        --output results/text_cond_cache.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

import ptqeval.wam.lingbot_va  # noqa: F401  (package init: sys.path for wan_va)
from ptqeval.wam.lingbot_va.text_cond_cache import (
    TextCondEntry, cache_key, store_cache)

# Matches server._reset's encode_prompt(max_sequence_length=512).
_MAX_SEQ = 512


def collect_prompts(videos_root: Path) -> list[str]:
    """Unique prompt strings from obs_data_*.pt 'task' fields under
    <videos_root>/visualization/real/, in first-seen order."""
    vis = videos_root / "visualization" / "real"
    if not vis.exists():
        raise FileNotFoundError(f"no visualization/real/ under {videos_root}")
    seen: dict[str, None] = {}
    for ep in sorted(vis.iterdir()):
        if not ep.is_dir():
            continue
        chunks = sorted(ep.glob("obs_data_*.pt"))
        if not chunks:
            continue
        first = torch.load(chunks[0], weights_only=False, map_location="cpu")
        seen[first[0]["task"]] = None
    return list(seen)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos_root", type=Path, required=True,
                    help="Root with visualization/real/<prompt>/ obs chunks "
                         "(e.g. results/bf16).")
    ap.add_argument("--model_path",
                    default="models/lingbot-va-posttrain-robotwin")
    ap.add_argument("--output", required=True,
                    help="Cache output path. A DIRECTORY path (no .pt "
                         "extension) writes the LazyCache format (index + "
                         "per-prompt embeds) so the eval server loads only "
                         "the prompts it needs -- use this at full scope "
                         "(thousands of prompts, tens of GB). A .pt path "
                         "writes a single eager file (small caches / smoke).")
    ap.add_argument("--limit", type=int, default=0,
                    help="Encode only the first N prompts (0 = all). Smoke.")
    args = ap.parse_args()

    from ptqeval.eval.measure_flops import _build_server
    server = _build_server(args.model_path,
                           Path("/tmp/precompute_text_cond_scratch"))
    server.text_encoder.to(server.device)

    prompts = collect_prompts(args.videos_root)
    if args.limit > 0:
        prompts = prompts[:args.limit]
    # Empty negative prompt is shared across all tasks; encode it too.
    todo = [""] + prompts
    print(f"encoding {len(todo)} prompts (incl. empty negative) "
          f"from {args.videos_root}")

    entries: dict[str, TextCondEntry] = {}
    for prompt in todo:
        embeds, _ = server.encode_prompt(
            prompt=prompt,
            negative_prompt=None,
            do_classifier_free_guidance=False,   # one embed per prompt; neg = ""
            num_videos_per_prompt=1,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            max_sequence_length=_MAX_SEQ,
            device=server.device,
            dtype=server.dtype,
        )
        emb = embeds.detach().to("cpu")
        k = cache_key(prompt, _MAX_SEQ)
        entries[k] = TextCondEntry(
            prompt=prompt, max_sequence_length=_MAX_SEQ,
            prompt_embeds=emb, seq_len=int(emb.shape[-2]), dim=int(emb.shape[-1]))
        print(f"  seq={emb.shape[-2]} dim={emb.shape[-1]} "
              f"l2={emb.float().norm().item():.1f}  '{prompt[:48]}'")

    server.text_encoder.to("cpu")
    store_cache(args.output, entries)
    print(f"wrote {len(entries)} entries to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
