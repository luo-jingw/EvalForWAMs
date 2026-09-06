# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Offline text-condition precompute for deployment (M5).

The device runs T5-free: the UMT5-XXL text encoder (~11 GB) never stays
resident. For a fixed deployment prompt set, encode the prompts here
(loading T5 once) and ship the small unpadded cache; the engine injects
the cached embeds at reset (server cache-hit path). If a prompt is not in
the cache the engine falls back to serial residency (config
serve_residency) so T5 and the transformer are never co-resident.

Reuses the eval-side encode harness (measure_flops._build_server) and the
unpadded cache schema (text_cond_cache.store_cache).

    from ptqeval.inference.precompute_text import build_text_cond_cache
    build_text_cond_cache(["put the bottle in the bin"],
                          "models/lingbot-va-posttrain-robotwin",
                          "text_cond_cache")

or CLI:

    python -m ptqeval.inference.precompute_text \\
        --model_path models/lingbot-va-posttrain-robotwin \\
        --prompt "put the bottle in the bin" \\
        --output text_cond_cache
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import ptqeval.wam.lingbot_va  # noqa: F401  (package init -> wan_va on sys.path)
from ptqeval.wam.lingbot_va.text_cond_cache import (
    TextCondEntry, cache_key, store_cache)

# Matches server._reset's encode_prompt(max_sequence_length=512).
_MAX_SEQ = 512


def build_text_cond_cache(
    prompts: list[str],
    model_path: str,
    output: str,
    device: str = "cuda:0",
) -> str:
    """Encode `prompts` (+ the shared empty negative prompt) with T5 and
    store an unpadded text-cond cache at `output`. A directory `output`
    (no .pt suffix) writes the LazyCache format. Returns the output path.
    Observational: prints per-prompt seq_len + embed L2, no assertion."""
    from ptqeval.eval.measure_flops import _build_server
    server = _build_server(model_path, Path("/tmp/precompute_text_scratch"))
    server.text_encoder.to(server.device)

    # Dedup, preserve first-seen order; empty negative shared across tasks.
    seen: dict[str, None] = {}
    for p in [""] + list(prompts):
        seen.setdefault(p, None)
    todo = list(seen)
    print(f"[precompute_text] encoding {len(todo)} prompts "
          f"(incl. empty negative)")

    entries: dict[str, TextCondEntry] = {}
    for prompt in todo:
        embeds, _ = server.encode_prompt(
            prompt=prompt,
            negative_prompt=None,
            do_classifier_free_guidance=False,
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
            prompt_embeds=emb, seq_len=int(emb.shape[-2]),
            dim=int(emb.shape[-1]))
        print(f"  seq={emb.shape[-2]} dim={emb.shape[-1]} "
              f"l2={emb.float().norm().item():.1f}  '{prompt[:48]}'")

    server.text_encoder.to("cpu")
    store_cache(output, entries)
    print(f"[precompute_text] wrote {len(entries)} entries to {output}")
    return output


def _read_prompts(args: argparse.Namespace) -> list[str]:
    prompts: list[str] = list(args.prompt or [])
    if args.prompts_file:
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            prompts += [ln.strip() for ln in f if ln.strip()]
    if not prompts:
        raise ValueError("provide at least one --prompt or --prompts_file")
    return prompts


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_path", required=True,
                    help="FP model root (contains text_encoder/).")
    ap.add_argument("--output", required=True,
                    help="Cache output. Directory path -> LazyCache format; "
                         ".pt path -> single eager file.")
    ap.add_argument("--prompt", action="append",
                    help="A deployment prompt (repeatable).")
    ap.add_argument("--prompts_file",
                    help="File with one prompt per line.")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    build_text_cond_cache(_read_prompts(args), args.model_path,
                          args.output, args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
