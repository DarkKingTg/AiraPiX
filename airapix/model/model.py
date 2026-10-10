from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import AiraConfig
from .layers import DeepSeekMoE, DiagonalSSMMixer, MLAAttention, RMSNorm, SwiGLU


class AiraBlock(nn.Module):
    """Single transformer block — either Mamba-2/SSD or MLA, with dense SwiGLU or MoE FFN.

    The 3:1 interleaved schedule is handled by AiraConfig.layer_type():
      Layers 0,1,2 → SSM;  Layer 3 → MLA;  Layers 4,5,6 → SSM;  Layer 7 → MLA; ...

    MoE replaces dense SwiGLU only in upper layers (controlled by config.is_moe_layer).
    """

    def __init__(self, config: AiraConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.kind = config.layer_type(layer_idx)
        self.use_moe = config.is_moe_layer(layer_idx)

        # Pre-norm 1 → Mixer
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
                use_qk_norm=config.use_qk_norm,
            )
        else:
            self.mixer = DiagonalSSMMixer(
                d_model=config.d_model,
                inner_mult=config.ssm_inner_mult,
                conv_kernel=config.ssm_conv_kernel,
                dropout=config.dropout,
            )

        # Pre-norm 2 → FFN (dense SwiGLU or MoE)
        self.norm2 = RMSNorm(config.d_model)
        if self.use_moe:
            self.ffn = DeepSeekMoE(
                d_model=config.d_model,
                hidden_mult=config.ffn_hidden_mult,
                num_experts=config.moe_num_experts,
                top_k=config.moe_top_k,
                num_shared_experts=config.num_shared_experts,
                dropout=config.dropout,
            )
        else:
            self.ffn = SwiGLU(config.d_model, hidden_mult=config.ffn_hidden_mult, dropout=config.dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.mixer(self.norm1(x))
        if self.use_moe:
            ffn_out, aux_loss = self.ffn(self.norm2(x))
            x = x + ffn_out
        else:
            x = x + self.ffn(self.norm2(x))
            aux_loss = torch.tensor(0.0, device=x.device)
        return x, aux_loss


class MTPHead(nn.Module):
    """Multi-Token Prediction auxiliary head.

    Each MTP head predicts the (k+1)-th future token using a lightweight
    shared-embedding projection. During training these provide denser
    learning signals; at inference the heads serve as draft predictors
    for speculative decoding.
    """

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(hidden))


class AiraForCausalLM(nn.Module):
    """AiraPix v2 language model.

    Architecture: 3:1 interleaved Mamba-2/SSD + MLA blocks, with
    staged MoE in upper layers and multi-token prediction heads.
    """

    def __init__(self, config: AiraConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.blocks = nn.ModuleList([AiraBlock(config, i) for i in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Multi-Token Prediction heads
        self.mtp_heads = nn.ModuleList([
            MTPHead(config.d_model, config.vocab_size)
            for _ in range(config.num_mtp_heads)
        ]) if config.num_mtp_heads > 0 else nn.ModuleList()

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

        total_aux_loss = torch.tensor(0.0, device=x.device)
        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training:
                x, aux = checkpoint(block, x, use_reentrant=False)
            else:
                x, aux = block(x)
            total_aux_loss = total_aux_loss + aux

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # Primary next-token loss (computed in float32 for FP16 numerical stability)
            shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1)).float()
            shift_labels = labels[:, 1:].reshape(-1)
            valid_mask = (shift_labels != -100)
            if valid_mask.sum() > 0:
                loss = F.cross_entropy(
                    shift_logits,
                    shift_labels,
                    ignore_index=-100,
                )
            else:
                loss = torch.tensor(0.0, device=x.device, requires_grad=True)

            # MTP auxiliary losses: predict token at position +2, +3, ...
            for k, mtp_head in enumerate(self.mtp_heads, start=2):
                if labels.size(1) > k:
                    mtp_logits = mtp_head(x[:, :-k, :]).reshape(-1, logits.size(-1)).float()
                    mtp_labels = labels[:, k:].reshape(-1)
                    mtp_mask = (mtp_labels != -100)
                    if mtp_mask.sum() > 0:
                        mtp_loss = F.cross_entropy(
                            mtp_logits,
                            mtp_labels,
                            ignore_index=-100,
                        )
                        if torch.isfinite(mtp_loss):
                            loss = loss + 0.1 * mtp_loss
                    del mtp_logits, mtp_labels

            # Add MoE auxiliary load-balancing loss
            num_moe_layers = sum(1 for b in self.blocks if b.use_moe)
            if num_moe_layers > 0 and torch.isfinite(total_aux_loss):
                loss = loss + 0.01 * (total_aux_loss / num_moe_layers)

            if not torch.isfinite(loss):
                loss = torch.tensor(0.0, device=x.device, requires_grad=True)

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
