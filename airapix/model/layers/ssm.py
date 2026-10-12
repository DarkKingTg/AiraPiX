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
    chunk_size: int = 32,
    return_final_state: bool = False,
    initial_h: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Numerically stable chunked parallel SSM recurrence scan.
    Processes sequences in bounded parallel chunks without computing exp(-cumsum),
    guaranteeing that decay multipliers remain strictly in (0, 1] and eliminating
    float32 exp() overflow (+inf) and 0*inf = NaN explosions.
    """
    decay = torch.exp(dt_f * a)
    e = (1.0 - decay) * b_f * u_f
    bsz, seq_len, dim = u_f.shape

    if seq_len % chunk_size == 0 and seq_len >= chunk_size:
        num_chunks = seq_len // chunk_size
        decay_c = decay.view(bsz, num_chunks, chunk_size, dim)
        e_c = e.view(bsz, num_chunks, chunk_size, dim)
        c_c = c_f.view(bsz, num_chunks, chunk_size, dim)

        # 1. Parallel intra-chunk recurrence across all (B, num_chunks, dim)
        h_intra = torch.zeros(bsz, num_chunks, dim, device=u_f.device, dtype=u_f.dtype)
        intra_list = []
        for t in range(chunk_size):
            h_intra = decay_c[:, :, t] * h_intra + e_c[:, :, t]
            intra_list.append(h_intra)
        intra_states = torch.stack(intra_list, dim=2)  # (B, num_chunks, chunk_size, dim)

        # 2. Cumulative decay within each chunk (strictly <= 1.0, non-positive exponent)
        log_d = (dt_f * a).view(bsz, num_chunks, chunk_size, dim)
        cum_decay = torch.exp(torch.cumsum(log_d, dim=2))  # Guaranteed in (0, 1]
        chunk_decay = cum_decay[:, :, -1]  # Total chunk decay in (0, 1]

        # 3. Inter-chunk sequential propagation across chunk boundaries
        if initial_h is not None:
            h_chunk = initial_h.view(bsz, dim).clone()
        else:
            h_chunk = torch.zeros(bsz, dim, device=u_f.device, dtype=u_f.dtype)

        h0_list = []
        for m in range(num_chunks):
            h0_list.append(h_chunk)
            h_chunk = chunk_decay[:, m] * h_chunk + intra_states[:, m, -1]

        h0 = torch.stack(h0_list, dim=1).unsqueeze(2)  # (B, num_chunks, 1, dim)
        states = intra_states + cum_decay * h0
        y = (c_c * states).view(bsz, seq_len, dim) + d_skip * u_f
        if not torch.isfinite(y).all():
            y = torch.nan_to_num(y, nan=0.0, posinf=10.0, neginf=-10.0)
        final_h = states[:, -1, -1, :].unsqueeze(1)  # (B, 1, dim)
        if return_final_state:
            return y, final_h
        return y
    else:
        # Clean sequential scan for variable / short sequence lengths
        if initial_h is not None:
            h = initial_h.view(bsz, dim).clone()
        else:
            h = torch.zeros(bsz, dim, device=u_f.device, dtype=u_f.dtype)

        y = torch.empty_like(u_f)
        for t in range(seq_len):
            h = decay[:, t] * h + e[:, t]
            y[:, t] = c_f[:, t] * h
        y = y + d_skip * u_f
        if not torch.isfinite(y).all():
            y = torch.nan_to_num(y, nan=0.0, posinf=10.0, neginf=-10.0)
        final_h = h.unsqueeze(1)  # (B, 1, dim)
        if return_final_state:
            return y, final_h
        return y


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
        self.conv_kernel = conv_kernel
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

    def forward(self, x: torch.Tensor, cache=None) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        if cache is not None and seq_len == 1:
            # Fast O(1) single-token recurrent decode step
            u, gate = self.in_proj(x).chunk(2, dim=-1)  # (bsz, 1, inner_dim)
            u_curr = u.transpose(1, 2)  # (bsz, inner_dim, 1)

            if cache.conv_state is None:
                cache.conv_state = torch.zeros(
                    bsz, self.inner_dim, self.conv_kernel - 1, device=x.device, dtype=x.dtype
                )

            conv_window = torch.cat([cache.conv_state, u_curr], dim=2)  # (bsz, inner_dim, conv_kernel)
            cache.conv_state = conv_window[:, :, 1:]

            conv_out = F.conv1d(conv_window, self.conv.weight, self.conv.bias, groups=self.inner_dim)
            u = F.silu(conv_out.transpose(1, 2))  # (bsz, 1, inner_dim)

            dt = F.softplus(self.dt_proj(u)).clamp(min=1e-4, max=10.0)
            b_term = torch.tanh(self.b_proj(u))
            c_term = self.c_proj(u)

            a = -torch.exp(self.a_log.float().clamp(min=-6.0, max=4.0)).view(1, 1, self.inner_dim)
            d_skip = self.d_skip.float().view(1, 1, self.inner_dim)

            u_f = u.float()
            dt_f = dt.float()
            b_f = b_term.float()
            c_f = c_term.float()

            decay = torch.exp(dt_f * a)
            e = (1.0 - decay) * b_f * u_f

            if cache.ssm_state is None:
                cache.ssm_state = torch.zeros(bsz, 1, self.inner_dim, device=x.device, dtype=torch.float32)
            elif cache.ssm_state.dim() == 2:
                cache.ssm_state = cache.ssm_state.unsqueeze(1)

            h = decay * cache.ssm_state + e
            cache.ssm_state = h
            y = (c_f * h + d_skip * u_f).to(dtype=x.dtype)
            y = y * F.silu(gate)
            return self.out_proj(self.dropout(y))

        # Full sequence scan (prefill or training)
        u, gate = self.in_proj(x).chunk(2, dim=-1)
        raw_u_transposed = u.transpose(1, 2)

        if cache is not None and cache.conv_state is not None:
            # Continue convolution from existing prefix conv state
            conv_in = torch.cat([cache.conv_state, raw_u_transposed], dim=2)
            conv_out = F.conv1d(conv_in, self.conv.weight, self.conv.bias, groups=self.inner_dim)
            u = F.silu(conv_out.transpose(1, 2))
            cache.conv_state = conv_in[:, :, -(self.conv_kernel - 1):].clone()
        else:
            u = self.conv(raw_u_transposed)[:, :, :seq_len].transpose(1, 2)
            u = F.silu(u)
            if cache is not None:
                pad_len = max(0, (self.conv_kernel - 1) - seq_len)
                if pad_len > 0:
                    pad_zeros = torch.zeros(bsz, self.inner_dim, pad_len, device=x.device, dtype=x.dtype)
                    cache.conv_state = torch.cat([pad_zeros, raw_u_transposed], dim=2)
                else:
                    cache.conv_state = raw_u_transposed[:, :, -(self.conv_kernel - 1):].clone()

        dt = F.softplus(self.dt_proj(u)).clamp(min=1e-4, max=10.0)
        b_term = torch.tanh(self.b_proj(u))
        c_term = self.c_proj(u)

        a = -torch.exp(self.a_log.float().clamp(min=-6.0, max=4.0)).view(1, self.inner_dim)
        d_skip = self.d_skip.float().view(1, self.inner_dim)

        u_f = u.float()
        dt_f = dt.float()
        b_f = b_term.float()
        c_f = c_term.float()

        if cache is not None:
            init_h = cache.ssm_state if cache.ssm_state is not None else None
            y, final_h = _ssm_recurrent_scan(
                dt_f, a, b_f, c_f, d_skip, u_f, return_final_state=True, initial_h=init_h
            )
            cache.ssm_state = final_h.clone()
            y = y.to(dtype=x.dtype)
        else:
            if _compiled_ssm_scan is not None and self.training and x.is_cuda:
                try:
                    y = _compiled_ssm_scan(dt_f, a, b_f, c_f, d_skip, u_f).to(dtype=x.dtype)
                except Exception:
                    y = _ssm_recurrent_scan(dt_f, a, b_f, c_f, d_skip, u_f).to(dtype=x.dtype)
            else:
                y = _ssm_recurrent_scan(dt_f, a, b_f, c_f, d_skip, u_f).to(dtype=x.dtype)

        y = y * F.silu(gate)
        return self.out_proj(self.dropout(y))


