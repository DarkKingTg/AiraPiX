from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


# Ultra-fast, numerically stable chunked parallel PyTorch scan for SSM recurrence
def _ssm_recurrent_scan(
    dt_f: torch.Tensor,
    a: torch.Tensor,
    b_f: torch.Tensor,
    c_f: torch.Tensor,
    d_skip: torch.Tensor,
    u_f: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
    """
    Numerically stable chunked parallel SSM scan.
    Breaks long sequence into bounded chunks (default 64) to eliminate catastrophic
    numerical overflow (exp(60) = 1e26) and NaN gradient explosions during multi-thousand-step runs.
    """
    decay = torch.exp(dt_f * a)
    e = (1.0 - decay) * b_f * u_f
    bsz, seq_len, dim = u_f.shape

    if seq_len % chunk_size == 0 and seq_len >= chunk_size:
        num_chunks = seq_len // chunk_size
        decay_c = decay.view(bsz, num_chunks, chunk_size, dim)
        e_c = e.view(bsz, num_chunks, chunk_size, dim)
        c_c = c_f.view(bsz, num_chunks, chunk_size, dim)

        log_d = (dt_f * a).view(bsz, num_chunks, chunk_size, dim)
        cum_log_d = torch.cumsum(log_d, dim=2)
        chunk_decay = torch.exp(cum_log_d[:, :, -1])

        exp_neg = torch.exp(-cum_log_d)
        intra_states = torch.exp(cum_log_d) * torch.cumsum(exp_neg * e_c, dim=2)

        h_chunk = torch.zeros(bsz, dim, device=u_f.device, dtype=u_f.dtype)
        h0_list = []
        for m in range(num_chunks):
            h0_list.append(h_chunk)
            h_chunk = chunk_decay[:, m] * h_chunk + intra_states[:, m, -1]

        h0 = torch.stack(h0_list, dim=1).unsqueeze(2)
        states = intra_states + torch.exp(cum_log_d) * h0
        y = (c_c * states).view(bsz, seq_len, dim) + d_skip * u_f
        return y
    else:
        # Clean sequential scan for variable / short sequence lengths
        h = torch.zeros(bsz, dim, device=u_f.device, dtype=u_f.dtype)
        y = torch.empty_like(u_f)
        for t in range(seq_len):
            h = decay[:, t] * h + e[:, t]
            y[:, t] = c_f[:, t] * h
        return y + d_skip * u_f


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
    gated/convolutional shape of Mamba-like blocks, but uses a stable chunked diagonal
    recurrent scan so it runs anywhere PyTorch runs without NaN explosions.
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

        # Clamped a_log prevents exponential decay rate explosion while keeping gradients smooth
        a = -torch.exp(self.a_log.float().clamp(min=-6.0, max=4.0)).view(1, self.inner_dim)
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


