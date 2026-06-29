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


def load_cache(path: str) -> dict[str, TextCondEntry]:
    """Load {cache_key: TextCondEntry} from a torch-saved cache file.
    weights_only=False because the payload contains TextCondEntry
    dataclass instances, not bare tensors."""
    return torch.load(path, map_location="cpu", weights_only=False)


def store_cache(path: str, entries: dict[str, TextCondEntry]) -> None:
    """Persist {cache_key: TextCondEntry} via torch.save."""
    torch.save(entries, path)
