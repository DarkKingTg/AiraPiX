from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from airapix.model.config import config_from_preset
from airapix.model.model import AiraForCausalLM, count_parameters
from airapix.training.optimizer import build_optimizer


def main() -> None:
    cfg = config_from_preset(
        "tiny",
        vocab_size=128,
        context_len=32,
        dropout=0.0,
        gradient_checkpointing=False,
    )
    model = AiraForCausalLM(cfg)
    optimizer = build_optimizer(
        model,
        {
            "peak_lr": 1e-3,
            "weight_decay": 0.01,
            "adamw_betas": [0.9, 0.95],
            "adamw_eps": 1e-8,
            "muon_momentum": 0.95,
            "muon_ns_steps": 2,
        },
    )
    input_ids = torch.randint(0, cfg.vocab_size, (2, cfg.context_len))
    labels = input_ids.clone()
    out = model(input_ids, labels=labels)
    loss = out["loss"]
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    generated = model.generate(input_ids[:1, :4], max_new_tokens=4)
    print(
        json.dumps(
            {
                "ok": True,
                "params": count_parameters(model),
                "loss": float(loss.detach()),
                "generated_shape": list(generated.shape),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
