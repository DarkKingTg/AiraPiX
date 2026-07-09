from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import AiraConfig
from .layers import DiagonalSSMMixer, MLAAttention, RMSNorm, SwiGLU


class AiraBlock(nn.Module):
    def __init__(self, config: AiraConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.kind = config.layer_type(layer_idx)
        self.norm1 = RMSNorm(config.d_model)
        if self.kind == "mla":
            self.mixer = MLAAttention(
                d_model=config.d_model,
                n_heads=config.n_heads,
                latent_dim=config.mla_latent_dim,
                qk_nope_dim=config.qk_nope_dim,
                qk_rope_dim=config.qk_rope_dim,
                v_head_dim=config.v_head_dim,
                rope_base=config.rope_base,
                dropout=config.dropout,
            )
        else:
            self.mixer = DiagonalSSMMixer(
                d_model=config.d_model,
                inner_mult=config.ssm_inner_mult,
                conv_kernel=config.ssm_conv_kernel,
                dropout=config.dropout,
            )
        self.norm2 = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config.d_model, hidden_mult=config.ffn_hidden_mult, dropout=config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class AiraForCausalLM(nn.Module):
    def __init__(self, config: AiraConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.blocks = nn.ModuleList([AiraBlock(config, i) for i in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        if input_ids.size(1) > self.config.context_len:
            raise ValueError(f"Sequence length {input_ids.size(1)} exceeds context_len={self.config.context_len}")
        x = self.embed_tokens(input_ids)
        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int | None = 50,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        eos_token_id = self.config.eos_token_id if eos_token_id is None else eos_token_id
        for _ in range(max_new_tokens):
            context = input_ids[:, -self.config.context_len :]
            logits = self(context)["logits"][:, -1, :]
            logits = logits / max(temperature, 1e-6)
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = torch.where(logits < values[:, [-1]], torch.full_like(logits, -float("inf")), logits)
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if bool((next_token == eos_token_id).all()):
                break
        return input_ids


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
