from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


# Fast PyTorch scan for SSM recurrence
def _ssm_recurrent_scan(
    dt_f: torch.Tensor,
    a: torch.Tensor,
    b_f: torch.Tensor,
    c_f: torch.Tensor,
    d_skip: torch.Tensor,
    u_f: torch.Tensor,
) -> torch.Tensor:
    bsz = u_f.size(0)
    seq_len = u_f.size(1)
    dim = u_f.size(2)
    decay_all = torch.exp(dt_f * a)
    e_all = (1.0 - decay_all) * b_f
    state = torch.zeros((bsz, dim), device=u_f.device, dtype=u_f.dtype)
    outputs = torch.empty((bsz, seq_len, dim), device=u_f.device, dtype=u_f.dtype)

    for i in range(seq_len):
        state = decay_all[:, i, :] * state + e_all[:, i, :]
        outputs[:, i, :] = c_f[:, i, :] * state + d_skip * u_f[:, i, :]

    return outputs


# Optional compilation only if requested explicitly via environment variable
import os
import sys

_compiled_ssm_scan = None
if os.environ.get("USE_TORCH_COMPILE", "0") == "1" and sys.platform != "win32":
    try:
        _compiled_ssm_scan = torch.compile(_ssm_recurrent_scan, mode="reduce-overhead")
    except Exception:
        _compiled_ssm_scan = None


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

        a = -torch.exp(self.a_log.float()).view(1, self.inner_dim)
        d_skip = self.d_skip.float().view(1, self.inner_dim)

        u_f = u.float()
        dt_f = dt.float()
        b_f = b_term.float()
        c_f = c_term.float()

        if _compiled_ssm_scan is not None and self.training and x.is_cuda:
            try:
                y = _compiled_ssm_scan(dt_f, a, b_f, c_f, d_skip, u_f).to(dtype=x.dtype)
            except Exception:
                y = _ssm_recurrent_scan(dt_f, a, b_f, c_f, d_skip, u_f).to(dtype=x.dtype)
        else:
            y = _ssm_recurrent_scan(dt_f, a, b_f, c_f, d_skip, u_f).to(dtype=x.dtype)

        y = y * F.silu(gate)
        return self.out_proj(self.dropout(y))


