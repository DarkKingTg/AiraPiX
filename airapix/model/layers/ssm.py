from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class DiagonalSSMMixer(nn.Module):
    """Pure-PyTorch diagonal SSM mixer.

    This is the dependency-light fallback path for Phase 2. It borrows the
    gated/convolutional shape of Mamba-like blocks, but uses a simple diagonal
    recurrent scan so it runs anywhere PyTorch runs.
    """

    def __init__(
        self,
        d_model: int,
        inner_mult: float = 2.0,
        conv_kernel: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.inner_dim = int(d_model * inner_mult)
        self.in_proj = nn.Linear(d_model, 2 * self.inner_dim, bias=False)
        self.conv = nn.Conv1d(
            self.inner_dim,
            self.inner_dim,
            kernel_size=conv_kernel,
            groups=self.inner_dim,
            padding=conv_kernel - 1,
            bias=True,
        )
        self.dt_proj = nn.Linear(self.inner_dim, self.inner_dim, bias=True)
        self.b_proj = nn.Linear(self.inner_dim, self.inner_dim, bias=False)
        self.c_proj = nn.Linear(self.inner_dim, self.inner_dim, bias=False)
        self.a_log = nn.Parameter(torch.zeros(self.inner_dim))
        self.d_skip = nn.Parameter(torch.ones(self.inner_dim))
        self.out_proj = nn.Linear(self.inner_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        u, gate = self.in_proj(x).chunk(2, dim=-1)
        u = self.conv(u.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        u = F.silu(u)

        dt = F.softplus(self.dt_proj(u)).clamp(max=10.0)
        b_term = torch.tanh(self.b_proj(u))
        c_term = self.c_proj(u)

        state = torch.zeros(bsz, self.inner_dim, device=x.device, dtype=torch.float32)
        a = -torch.exp(self.a_log.float()).view(1, self.inner_dim)
        d_skip = self.d_skip.float().view(1, self.inner_dim)
        outputs = []

        u_f = u.float()
        dt_f = dt.float()
        b_f = b_term.float()
        c_f = c_term.float()
        for idx in range(seq_len):
            decay = torch.exp(dt_f[:, idx, :] * a)
            state = decay * state + (1.0 - decay) * b_f[:, idx, :]
            y = c_f[:, idx, :] * state + d_skip * u_f[:, idx, :]
            outputs.append(y)

        y = torch.stack(outputs, dim=1).to(dtype=x.dtype)
        y = y * F.silu(gate)
        return self.out_proj(self.dropout(y))
