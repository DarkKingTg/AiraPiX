from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset, IterableDataset

from .tokenizer import TokenizerWrapper


def iter_jsonl(path: str | Path) -> Iterable[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def render_example(row: dict) -> str:
    category = row.get("category", "text")
    input_text = row.get("input", "")
    output_text = row.get("output", "")
    if category in {"chat", "reasoning", "tool_use"} and input_text:
        return f"User:\n{input_text}\n\nAssistant:\n{output_text}"
    if category == "code" and input_text:
        return f"Prompt:\n{input_text}\n\nCode:\n{output_text}"
    return "\n\n".join(part for part in [input_text, output_text] if part)


def encode_for_lm(
    tokenizer: TokenizerWrapper,
    text: str,
    context_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    ids = tokenizer.encode(text)
    ids = ids[:context_len]
    labels = list(ids)
    if len(ids) < context_len:
        pad_count = context_len - len(ids)
        ids.extend([tokenizer.pad_token_id] * pad_count)
        labels.extend([-100] * pad_count)
    return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class JsonlLMDataset(Dataset):
    def __init__(self, path: str | Path, tokenizer: TokenizerWrapper, context_len: int, max_rows: int | None = None) -> None:
        self.rows = []
        for idx, row in enumerate(iter_jsonl(path)):
            if max_rows is not None and idx >= max_rows:
                break
            self.rows.append(render_example(row))
        self.tokenizer = tokenizer
        self.context_len = context_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        input_ids, labels = encode_for_lm(self.tokenizer, self.rows[idx], self.context_len)
        return {"input_ids": input_ids, "labels": labels}


class MixtureJsonlDataset(IterableDataset):
    def __init__(
        self,
        category_paths: dict[str, str | Path],
        weights: dict[str, float],
        tokenizer: TokenizerWrapper,
        context_len: int,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.category_paths = {k: Path(v) for k, v in category_paths.items() if Path(v).exists()}
        if not self.category_paths:
            expected = ", ".join(str(v) for v in category_paths.values())
            raise FileNotFoundError(
                "No category shard files were found. Run the Phase 1 dataset builder first. "
                f"Expected one or more of: {expected}"
            )
        self.weights = {k: float(weights.get(k, 0.0)) for k in self.category_paths}
        total = sum(self.weights.values())
        if total <= 0:
            self.weights = {k: 1.0 / len(self.category_paths) for k in self.category_paths}
        else:
            self.weights = {k: v / total for k, v in self.weights.items()}
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.seed = seed

    def _cycle(self, path: Path):
        while True:
            yielded = False
            for row in iter_jsonl(path):
                yielded = True
                yield row
            if not yielded:
                raise RuntimeError(f"No rows found in {path}")

    def __iter__(self):
        rng = random.Random(self.seed)
        categories = list(self.category_paths.keys())
        weights = [self.weights[c] for c in categories]
        streams = {cat: self._cycle(path) for cat, path in self.category_paths.items()}
        while True:
            category = rng.choices(categories, weights=weights, k=1)[0]
            row = next(streams[category])
            input_ids, labels = encode_for_lm(self.tokenizer, render_example(row), self.context_len)
            yield {"input_ids": input_ids, "labels": labels}
