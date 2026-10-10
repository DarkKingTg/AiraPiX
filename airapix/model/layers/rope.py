from __future__ import annotations

import torch


_ROPE_CACHE: dict[tuple[int, int, float, str, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}


def build_rope_cache(
    seq_len: int,
    dim: int,
    base: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if dim % 2 != 0:
        raise ValueError("RoPE dimension must be even.")
    key = (seq_len, dim, float(base), str(device), dtype)
    if key in _ROPE_CACHE:
        return _ROPE_CACHE[key]

    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype=dtype)[None, None, :, :]
    sin = emb.sin().to(dtype=dtype)[None, None, :, :]
    _ROPE_CACHE[key] = (cos, sin)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)
