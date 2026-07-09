from __future__ import annotations

import math


def cosine_with_warmup(
    step: int,
    max_steps: int,
    warmup_steps: int,
    peak_lr: float,
    min_lr_ratio: float,
) -> float:
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    min_lr = peak_lr * min_lr_ratio
    return min_lr + (peak_lr - min_lr) * cosine
