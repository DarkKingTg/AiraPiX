from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .rope import apply_rope, build_rope_cache


class MLAAttention(nn.Module):
    """Multi-Head Latent Attention with QK normalization.

    Keys and values are reconstructed from a compressed latent vector. A small
    separate RoPE key path preserves positional information without storing full
    per-head K/V tensors.

    QK normalization stabilizes attention logits, especially with long context,
    mixed precision, and low-rank latent projections (DeepSeek-V2 style).
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
        use_qk_norm: bool = True,
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
        self.use_qk_norm = use_qk_norm

        q_dim = n_heads * (qk_nope_dim + qk_rope_dim)
        kv_dim = n_heads * (qk_nope_dim + v_head_dim)
        self.q_proj = nn.Linear(d_model, q_dim, bias=False)
        self.kv_down = nn.Linear(d_model, latent_dim, bias=False)
        self.kv_up = nn.Linear(latent_dim, kv_dim, bias=False)
        self.k_rope_proj = nn.Linear(d_model, qk_rope_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * v_head_dim, d_model, bias=False)

        # QK normalization: per-head RMSNorm on Q and K before attention scores
        if use_qk_norm:
            head_dim = qk_nope_dim + qk_rope_dim
            self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
            self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)

    def forward(self, x: torch.Tensor, cache=None, start_pos: int = 0) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        if cache is not None and seq_len == 1:
            # Fast O(1) single-token decode step
            q = self.q_proj(x).view(
                bsz,
                1,
                self.n_heads,
                self.qk_nope_dim + self.qk_rope_dim,
            )
            q_nope, q_rope = torch.split(q, [self.qk_nope_dim, self.qk_rope_dim], dim=-1)

            latent = self.kv_down(x)
            kv = self.kv_up(latent).view(
                bsz,
                1,
                self.n_heads,
                self.qk_nope_dim + self.v_head_dim,
            )
            k_nope, value = torch.split(kv, [self.qk_nope_dim, self.v_head_dim], dim=-1)
            k_rope = self.k_rope_proj(x).view(bsz, 1, 1, self.qk_rope_dim)
            k_rope = k_rope.expand(bsz, 1, self.n_heads, self.qk_rope_dim)

            q_nope = q_nope.transpose(1, 2)  # (bsz, n_heads, 1, qk_nope_dim)
            q_rope = q_rope.transpose(1, 2)  # (bsz, n_heads, 1, qk_rope_dim)
            k_nope = k_nope.transpose(1, 2)  # (bsz, n_heads, 1, qk_nope_dim)
            k_rope = k_rope.transpose(1, 2)  # (bsz, n_heads, 1, qk_rope_dim)
            value = value.transpose(1, 2)    # (bsz, n_heads, 1, v_head_dim)

            pos = start_pos
            cos, sin = build_rope_cache(
                pos + 1,
                self.qk_rope_dim,
                self.rope_base,
                device=x.device,
                dtype=x.dtype,
            )
            cos_t = cos[:, :, pos : pos + 1, :]
            sin_t = sin[:, :, pos : pos + 1, :]
            q_rope = apply_rope(q_rope, cos_t, sin_t)
            k_rope = apply_rope(k_rope, cos_t, sin_t)

            query = torch.cat([q_nope, q_rope], dim=-1)
            key = torch.cat([k_nope, k_rope], dim=-1)

            if self.use_qk_norm:
                query = self.q_norm(query)
                key = self.k_norm(key)

            if cache.key_cache is not None:
                full_key = torch.cat([cache.key_cache, key], dim=2)
                full_value = torch.cat([cache.value_cache, value], dim=2)
            else:
                full_key = key
                full_value = value

            cache.key_cache = full_key
            cache.value_cache = full_value
            if cache.latent_cache is not None:
                cache.latent_cache = torch.cat([cache.latent_cache, latent], dim=1)
            else:
                cache.latent_cache = latent

            attn = F.scaled_dot_product_attention(
                query,
                full_key,
                full_value,
                dropout_p=0.0,
                is_causal=False,
            )
            attn = attn.transpose(1, 2).contiguous().view(bsz, 1, self.n_heads * self.v_head_dim)
            return self.out_proj(attn)

        # Full sequence forward (prefill or training)
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

        # Transpose to (bsz, n_heads, seq_len, head_dim)
        q_nope = q_nope.transpose(1, 2)
        q_rope = q_rope.transpose(1, 2)
        k_nope = k_nope.transpose(1, 2)
        k_rope = k_rope.transpose(1, 2)
        value = value.transpose(1, 2)

        cos, sin = build_rope_cache(
            start_pos + seq_len,
            self.qk_rope_dim,
            self.rope_base,
            device=x.device,
            dtype=x.dtype,
        )
        cos_slice = cos[:, :, start_pos : start_pos + seq_len, :]
        sin_slice = sin[:, :, start_pos : start_pos + seq_len, :]
        q_rope = apply_rope(q_rope, cos_slice, sin_slice)
        k_rope = apply_rope(k_rope, cos_slice, sin_slice)

        query = torch.cat([q_nope, q_rope], dim=-1)
        key = torch.cat([k_nope, k_rope], dim=-1)

        # QK normalization: stabilize logits before scaled_dot_product_attention
        if self.use_qk_norm:
            query = self.q_norm(query)
            key = self.k_norm(key)

        if cache is not None:
            if start_pos > 0 and cache.key_cache is not None:
                full_key = torch.cat([cache.key_cache, key], dim=2)
                full_value = torch.cat([cache.value_cache, value], dim=2)
            else:
                full_key = key
                full_value = value
            cache.key_cache = full_key
            cache.value_cache = full_value
            if cache.latent_cache is not None and start_pos > 0:
                cache.latent_cache = torch.cat([cache.latent_cache, latent], dim=1)
            else:
                cache.latent_cache = latent.clone()
        else:
            full_key = key
            full_value = value

        if start_pos > 0:
            k_len = full_key.size(2)
            prefix_len = k_len - seq_len
            mask = torch.ones(seq_len, k_len, dtype=torch.bool, device=x.device)
            mask[:, prefix_len:] = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
            attn = F.scaled_dot_product_attention(
                query,
                full_key,
                full_value,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            attn = F.scaled_dot_product_attention(
                query,
                full_key,
                full_value,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        attn = attn.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_heads * self.v_head_dim)
        return self.out_proj(attn)
