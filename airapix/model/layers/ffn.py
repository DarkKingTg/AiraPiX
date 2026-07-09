from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def round_up_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_mult: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = round_up_to_multiple(int(d_model * hidden_mult * 2 / 3), 64)
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))
