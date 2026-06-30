# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Text-condition cache for the lingbot-va WAM (Phase 44).

RoboTwin per-task prompts are fixed, so the UMT5-XXL text encoder output
is deterministic and therefore cacheable. precompute_text_cond.py fills
the cache offline; the server injects the cached prompt_embeds into
encode_prompt() to skip the T5 encoder entirely, so the text encoder
never occupies VRAM during eval.

One file, one job: the on-disk cache schema + key + load/store. No model
code here (the encoder lives in the server / precompute script).
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import torch


@dataclass
class TextCondEntry:
    """One cached T5 encoding. prompt_embeds is the _get_t5_prompt_embeds
    output for `prompt` (CPU tensor; the server moves it to GPU on use)."""
    prompt: str
    max_sequence_length: int
    prompt_embeds: torch.Tensor   # [seq_len, dim]
    seq_len: int
    dim: int


def cache_key(prompt: str, max_sequence_length: int) -> str:
    """Stable key for (prompt, max_sequence_length): sha1 of the raw
    prompt + the sequence length. Keying on the raw prompt (not a
    cleaned form) keeps the cache module free of any tokenizer/diffusers
    dependency; the precompute script and the server both pass the same
    raw prompt string."""
    h = hashlib.sha1()
    h.update(prompt.encode("utf-8"))
    h.update(f"|{max_sequence_length}".encode("utf-8"))
    return h.hexdigest()


class LazyCache:
    """Directory-backed cache that loads embeds on demand. Only the small
    index (key -> prompt/seq_len/dim) is read at construction; each
    prompt's [seq_len, dim] embed (~4 MB padded) is torch.load-ed on first
    access and memoised. A per-task eval server (start_server is launched
    per task) thus holds only its task's prompts (~1 GB), not the whole
    cache (tens of GB at full scope). Implements the read API the server
    and check scripts use: `key in cache`, `cache[key]`, len/keys/values/
    items."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._index: dict[str, dict] = torch.load(
            os.path.join(path, "index.pt"), weights_only=False)
        self._mem: dict[str, TextCondEntry] = {}

    def __contains__(self, key: str) -> bool:
        return key in self._index

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, key: str) -> TextCondEntry:
        if key not in self._mem:
            meta = self._index[key]
            emb = torch.load(os.path.join(self._path, "embeds", f"{key}.pt"),
                             map_location="cpu", weights_only=True)
            self._mem[key] = TextCondEntry(
                prompt=meta["prompt"],
                max_sequence_length=meta["max_sequence_length"],
                prompt_embeds=emb, seq_len=meta["seq_len"], dim=meta["dim"])
        return self._mem[key]

    def keys(self):
        return self._index.keys()

    def values(self):
        return (self[k] for k in self._index)

    def items(self):
        return ((k, self[k]) for k in self._index)


def load_cache(path: str):
    """Load a text-cond cache. A directory path -> LazyCache (loads the
    index now, embeds on demand; scales to tens of thousands of prompts
    without per-worker OOM). A single-file path (legacy / small caches) ->
    the eager {key: TextCondEntry} dict (weights_only=False: the payload
    holds TextCondEntry instances)."""
    if os.path.isdir(path):
        return LazyCache(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def store_cache(path: str, entries: dict[str, TextCondEntry]) -> None:
    """Persist the cache. A path ending in .pt -> single eager file (small
    caches / smoke). Otherwise -> a LazyCache directory: index.pt (metadata
    only) + embeds/<key>.pt per prompt, so the eval server can load just
    the prompts it needs."""
    if path.endswith(".pt"):
        torch.save(entries, path)
        return
    os.makedirs(os.path.join(path, "embeds"), exist_ok=True)
    index: dict[str, dict] = {}
    for k, e in entries.items():
        torch.save(e.prompt_embeds, os.path.join(path, "embeds", f"{k}.pt"))
        index[k] = dict(prompt=e.prompt,
                        max_sequence_length=e.max_sequence_length,
                        seq_len=e.seq_len, dim=e.dim)
    torch.save(index, os.path.join(path, "index.pt"))
