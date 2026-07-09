from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .rope import apply_rope, build_rope_cache


class MLAAttention(nn.Module):
    """A small MLA-style attention module.

    Keys and values are reconstructed from a compressed latent vector. A small
    separate RoPE key path preserves positional information without storing full
    per-head K/V tensors.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        latent_dim: int,
        qk_nope_dim: int,
        qk_rope_dim: int,
        v_head_dim: int,
        rope_base: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if qk_rope_dim % 2 != 0:
            raise ValueError("qk_rope_dim must be even.")
        self.n_heads = n_heads
        self.qk_nope_dim = qk_nope_dim
        self.qk_rope_dim = qk_rope_dim
        self.v_head_dim = v_head_dim
        self.rope_base = rope_base
        self.dropout = dropout

        q_dim = n_heads * (qk_nope_dim + qk_rope_dim)
        kv_dim = n_heads * (qk_nope_dim + v_head_dim)
        self.q_proj = nn.Linear(d_model, q_dim, bias=False)
        self.kv_down = nn.Linear(d_model, latent_dim, bias=False)
        self.kv_up = nn.Linear(latent_dim, kv_dim, bias=False)
        self.k_rope_proj = nn.Linear(d_model, qk_rope_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * v_head_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(
            bsz,
            seq_len,
            self.n_heads,
            self.qk_nope_dim + self.qk_rope_dim,
        )
        q_nope, q_rope = torch.split(q, [self.qk_nope_dim, self.qk_rope_dim], dim=-1)

        latent = self.kv_down(x)
        kv = self.kv_up(latent).view(
            bsz,
            seq_len,
            self.n_heads,
            self.qk_nope_dim + self.v_head_dim,
        )
        k_nope, value = torch.split(kv, [self.qk_nope_dim, self.v_head_dim], dim=-1)
        k_rope = self.k_rope_proj(x).view(bsz, seq_len, 1, self.qk_rope_dim)
        k_rope = k_rope.expand(bsz, seq_len, self.n_heads, self.qk_rope_dim)

        q_nope = q_nope.transpose(1, 2)
        q_rope = q_rope.transpose(1, 2)
        k_nope = k_nope.transpose(1, 2)
        k_rope = k_rope.transpose(1, 2)
        value = value.transpose(1, 2)

        cos, sin = build_rope_cache(
            seq_len,
            self.qk_rope_dim,
            self.rope_base,
            device=x.device,
            dtype=x.dtype,
        )
        q_rope = apply_rope(q_rope, cos, sin)
        k_rope = apply_rope(k_rope, cos, sin)

        query = torch.cat([q_nope, q_rope], dim=-1)
        key = torch.cat([k_nope, k_rope], dim=-1)

        attn = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attn = attn.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_heads * self.v_head_dim)
        return self.out_proj(attn)
