from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, IterableDataset

from .dataset import encode_for_lm, iter_jsonl, render_example
from .tokenizer import TokenizerWrapper


class ContinualReplayMixture(IterableDataset):
    """
    Continual Learning 80/20 Experience Replay Mixture (Strategy 7).
    To prevent catastrophic forgetting during active/dynamic training:
    - 80% of tokens are sampled from the foundational pretraining corpus
      (core grammar, reasoning, programming, mathematical definitions).
    - 20% of tokens are sampled from fresh dynamic experiences
      (verified dream consolidation, recent chat sessions, vault notes).
    """

    def __init__(
        self,
        base_pretrain_paths: List[str | Path],
        dynamic_experience_paths: List[str | Path],
        tokenizer: TokenizerWrapper,
        context_len: int = 1024,
        replay_ratio: float = 0.80,  # 80% base, 20% dynamic
        seed: int = 42,
        mask_prompt: bool = True,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.replay_ratio = replay_ratio
        self.seed = seed
        self.mask_prompt = mask_prompt

        # Validate base paths
        self.base_paths = [Path(p) for p in base_pretrain_paths if Path(p).exists() and Path(p).stat().st_size > 0]
        if not self.base_paths:
            raise FileNotFoundError(f"No valid base pretrain shards found: {base_pretrain_paths}")

        # Validate dynamic paths (with fallback to base if none exist yet)
        self.dynamic_paths = [Path(p) for p in dynamic_experience_paths if Path(p).exists() and Path(p).stat().st_size > 0]
        if not self.dynamic_paths:
            # Fallback: if no dynamic experiences exist yet, sample from base
            self.dynamic_paths = self.base_paths

    def _cycle(self, paths: List[Path]) -> Iterator[Dict[str, Any]]:
        rng = random.Random(self.seed)
        while True:
            shuffled_paths = list(paths)
            rng.shuffle(shuffled_paths)
            for path in shuffled_paths:
                for row in iter_jsonl(path):
                    yield row

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        rng = random.Random(self.seed)
        base_stream = self._cycle(self.base_paths)
        dynamic_stream = self._cycle(self.dynamic_paths)

        while True:
            # 80/20 probabilistic choice
            if rng.random() < self.replay_ratio:
                row = next(base_stream)
                source_type = "base_pretrain"
            else:
                row = next(dynamic_stream)
                source_type = "dynamic_experience"

            example = render_example(row)
            input_ids, labels = encode_for_lm(
                self.tokenizer,
                example,
                self.context_len,
                mask_prompt=self.mask_prompt
            )

            yield {
                "input_ids": input_ids,
                "labels": labels,
            }


class EWCConsolidator:
    """
    Elastic Weight Consolidation (EWC) Manager (Strategy 7).
    Protects foundational pretraining weights from catastrophic forgetting
    by penalizing movement along directions of high Fisher Information.

    Loss_total = Loss_experience + (lambda / 2) * sum( F_i * (theta_i - theta_star_i)^2 )
    """

    def __init__(self, model: nn.Module, lambda_reg: float = 200.0):
        self.model = model
        self.lambda_reg = lambda_reg
        self.reference_params: Dict[str, torch.Tensor] = {}
        self.fisher_diagonal: Dict[str, torch.Tensor] = {}
        self.is_initialized = False

    def register_reference_weights(self) -> None:
        """Stores the current optimal baseline weights theta_star."""
        self.reference_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.reference_params[name] = param.detach().clone()

    def compute_fisher_information(
        self,
        dataloader: Iterable[Dict[str, torch.Tensor]],
        num_batches: int = 10,
        device: Optional[torch.device] = None,
    ) -> Dict[str, float]:
        """
        Computes the empirical Fisher Information Matrix diagonal:
        F_i = (1 / N) * sum( (grad_theta_i)^2 )
        over a representative sample of baseline pretraining data.
        """
        device = device or next(self.model.parameters()).device
        self.register_reference_weights()
        self.fisher_diagonal = {
            name: torch.zeros_like(param, device=device)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        self.model.eval()
        batches_processed = 0

        for batch in dataloader:
            if batches_processed >= num_batches:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            self.model.zero_grad()
            outputs = self.model(input_ids)
            logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

            # Shift tokens for causal loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            loss.backward()

            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.fisher_diagonal[name] += param.grad.data.pow(2)

            batches_processed += 1

        # Normalize across batches
        if batches_processed > 0:
            for name in self.fisher_diagonal:
                self.fisher_diagonal[name] /= float(batches_processed)

        self.is_initialized = True

        # Compute summary metrics
        total_importance = sum(f.sum().item() for f in self.fisher_diagonal.values())
        return {
            "batches_processed": batches_processed,
            "total_fisher_mass": round(total_importance, 4),
            "num_parameters_protected": len(self.fisher_diagonal)
        }

    def compute_ewc_loss(self) -> torch.Tensor:
        """
        Calculates the EWC quadratic penalty:
        (lambda / 2) * sum( F_i * (theta_i - theta_star_i)^2 )
        """
        if not self.is_initialized or not self.fisher_diagonal:
            device = next(self.model.parameters()).device
            return torch.tensor(0.0, device=device)

        ewc_loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        for name, param in self.model.named_parameters():
            if name in self.fisher_diagonal and param.requires_grad:
                fisher = self.fisher_diagonal[name]
                ref_param = self.reference_params[name]
                diff = param - ref_param
                ewc_loss += (fisher * diff.pow(2)).sum()

        return (self.lambda_reg / 2.0) * ewc_loss

    def save_checkpoint(self, path: str | Path) -> None:
        """Persists the Fisher matrix and reference parameters to disk."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "fisher_diagonal": {k: v.cpu() for k, v in self.fisher_diagonal.items()},
                "reference_params": {k: v.cpu() for k, v in self.reference_params.items()},
                "lambda_reg": self.lambda_reg,
            },
            save_path
        )

    def load_checkpoint(self, path: str | Path, device: Optional[torch.device] = None) -> None:
        """Loads a precomputed Fisher matrix from disk."""
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"EWC checkpoint not found: {load_path}")
        data = torch.load(load_path, map_location=device or "cpu")
        self.fisher_diagonal = data["fisher_diagonal"]
        self.reference_params = data["reference_params"]
        self.lambda_reg = data.get("lambda_reg", self.lambda_reg)
        self.is_initialized = True
