#!/usr/bin/env python3
"""Export precomputed Qwen3 text embeddings into a single npz for the G1 policy server.

The server must not load Qwen3: the text encoder costs ~8 GB of VRAM and adds
startup latency, while the set of task prompts is fixed and known offline.
ImageWAM.infer_action accepts `context`/`context_mask` in place of `prompt`, so
serving only needs the embeddings.

Input is the cache written by scripts/flux2/precompute_flux2_qwen3_embeds.py.
That script owns the encoding; this one only reshapes its output. Run it first.

Cache entries are keyed by sha256 of the *wrapped* prompt, i.e.
DEFAULT_PROMPT.format(task=<task>), which is what RobotVideoDataset feeds the
encoder during training. The same wrapping is applied here so serving matches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from numpy.typing import NDArray

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from imagewam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT

CACHE_SUFFIX: str = "qwen3_flux2"


@dataclass(frozen=True)
class PromptEmbedding:
    task: str
    prompt: str
    hidden: NDArray[np.float32]  # [L, D]
    mask: NDArray[np.bool_]  # [L]


@dataclass(frozen=True)
class ExportReport:
    output_path: Path
    entries: tuple[PromptEmbedding, ...]
    context_len: int
    hidden_dim: int


def wrap_prompt(task: str) -> str:
    """Apply the same prompt template RobotVideoDataset uses during training."""
    return DEFAULT_PROMPT.format(task=task)


def cache_path_for(cache_dir: Path, prompt: str, context_len: int) -> Path:
    hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return cache_dir / f"{hashed}.{CACHE_SUFFIX}_len{context_len}.pt"


def load_prompt_embedding(cache_dir: Path, task: str, context_len: int) -> PromptEmbedding:
    prompt = wrap_prompt(task)
    path = cache_path_for(cache_dir, prompt, context_len)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Qwen3 cache for task {task!r} at {path}. "
            "Run scripts/flux2/precompute_flux2_qwen3_embeds.py for this prompt first."
        )
    payload = torch.load(path, map_location="cpu")
    hidden = payload["text_hidden_states"]
    mask = payload["text_attention_mask"]
    if hidden.ndim != 2 or mask.ndim != 1:
        raise ValueError(
            f"Expected hidden [L,D] and mask [L], got {tuple(hidden.shape)} and {tuple(mask.shape)}"
        )
    if hidden.shape[0] != context_len or mask.shape[0] != context_len:
        raise ValueError(
            f"context_len mismatch for {task!r}: expected {context_len}, "
            f"got hidden={hidden.shape[0]} mask={mask.shape[0]}"
        )
    # npz has no bfloat16; the server casts back to the model dtype on load.
    return PromptEmbedding(
        task=task,
        prompt=prompt,
        hidden=hidden.to(dtype=torch.float32).numpy(),
        mask=mask.to(dtype=torch.bool).numpy(),
    )


def read_tasks_jsonl(path: Path) -> tuple[str, ...]:
    tasks: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                tasks.append(str(json.loads(line)["task"]))
    return tuple(tasks)


def export_prompt_embeds(
    cache_dir: Path,
    tasks: Sequence[str],
    output_path: Path,
    context_len: int,
) -> ExportReport:
    entries = tuple(load_prompt_embedding(cache_dir, task, context_len) for task in tasks)
    hidden = np.stack([entry.hidden for entry in entries])
    mask = np.stack([entry.mask for entry in entries])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        tasks=np.array([entry.task for entry in entries], dtype=object),
        prompts=np.array([entry.prompt for entry in entries], dtype=object),
        hidden=hidden,
        mask=mask,
        context_len=np.int64(context_len),
    )
    return ExportReport(
        output_path=output_path,
        entries=entries,
        context_len=context_len,
        hidden_dim=int(hidden.shape[-1]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache_dir", type=Path, help="flux2_qwen3_cache_4b directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks-file", type=Path, default=None, help="LeRobot meta/tasks.jsonl")
    parser.add_argument("--task", action="append", default=[], help="extra task string, repeatable")
    parser.add_argument("--context-len", type=int, default=128)
    args = parser.parse_args()

    tasks: list[str] = []
    if args.tasks_file is not None:
        tasks.extend(read_tasks_jsonl(args.tasks_file))
    tasks.extend(args.task)
    if not tasks:
        raise ValueError("No tasks given. Pass --tasks-file and/or --task.")

    report = export_prompt_embeds(
        cache_dir=args.cache_dir,
        tasks=tuple(dict.fromkeys(tasks)),
        output_path=args.output,
        context_len=args.context_len,
    )
    print(f"context_len={report.context_len} hidden_dim={report.hidden_dim}")
    for entry in report.entries:
        print(f"  [{entry.mask.sum():3d}/{report.context_len} tokens] {entry.task}")
    print(f"wrote {report.output_path} ({report.output_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
