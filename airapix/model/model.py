from __future__ import annotations

import time
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import AiraConfig
from .layers import DeepSeekMoE, DiagonalSSMMixer, MLAAttention, RMSNorm, SwiGLU


@dataclass
class LayerCache:
    layer_type: str
    # SSM state
    conv_state: torch.Tensor | None = None
    ssm_state: torch.Tensor | None = None
    # MLA cache
    key_cache: torch.Tensor | None = None
    value_cache: torch.Tensor | None = None
    latent_cache: torch.Tensor | None = None

    def clone(self) -> "LayerCache":
        return LayerCache(
            layer_type=self.layer_type,
            conv_state=self.conv_state.clone() if self.conv_state is not None else None,
            ssm_state=self.ssm_state.clone() if self.ssm_state is not None else None,
            key_cache=self.key_cache.clone() if self.key_cache is not None else None,
            value_cache=self.value_cache.clone() if self.value_cache is not None else None,
            latent_cache=self.latent_cache.clone() if self.latent_cache is not None else None,
        )

    def truncate(self, length: int) -> None:
        if self.key_cache is not None and self.key_cache.size(2) > length:
            self.key_cache = self.key_cache[:, :, :length, :].clone()
        if self.value_cache is not None and self.value_cache.size(2) > length:
            self.value_cache = self.value_cache[:, :, :length, :].clone()
        if self.latent_cache is not None and self.latent_cache.size(1) > length:
            self.latent_cache = self.latent_cache[:, :length, :].clone()


class AiraInferenceCache:
    """Stateful inference cache for AiraPix hybrid models.
    Tracks O(1) recurrent states for SSM layers and compact latent/key/value states for MLA layers.
    """

    def __init__(self, config: AiraConfig, batch_size: int = 1) -> None:
        self.config = config
        self.batch_size = batch_size
        self.seq_len = 0
        self.layers: list[LayerCache] = [
            LayerCache(layer_type=config.layer_type(i))
            for i in range(config.n_layers)
        ]

    def reset(self) -> None:
        self.seq_len = 0
        for layer in self.layers:
            layer.conv_state = None
            layer.ssm_state = None
            layer.key_cache = None
            layer.value_cache = None
            layer.latent_cache = None

    def clone(self) -> "AiraInferenceCache":
        new_cache = AiraInferenceCache(self.config, batch_size=self.batch_size)
        new_cache.seq_len = self.seq_len
        new_cache.layers = [layer.clone() for layer in self.layers]
        return new_cache

    def truncate(self, length: int) -> None:
        self.seq_len = min(self.seq_len, length)
        for layer in self.layers:
            layer.truncate(length)


class PrefixCacheEntry:
    def __init__(self, token_ids: tuple[int, ...], cache: AiraInferenceCache) -> None:
        self.token_ids = token_ids
        self.cache = cache
        self.last_accessed = time.time()
        self.hit_count = 0


class AiraPrefixCacheManager:
    """Radix-style Prefix Cache Manager for AiraPix models.
    Caches historical token prefixes and their corresponding AiraInferenceCache states.
    Allows multi-turn chat dialogues and repeated system prompts to skip prefill compute.
    """

    def __init__(self, max_entries: int = 32) -> None:
        self.max_entries = max_entries
        self.entries: dict[tuple[int, ...], PrefixCacheEntry] = {}

    def match_prefix(self, token_ids: list[int] | tuple[int, ...]) -> tuple[int, AiraInferenceCache | None]:
        """Finds the longest cached prefix matching the start of token_ids.
        Returns (matched_length, cloned_cache) or (0, None).
        """
        token_tuple = tuple(token_ids)
        best_match_len = 0
        best_entry: PrefixCacheEntry | None = None

        for prefix, entry in self.entries.items():
            prefix_len = len(prefix)
            if prefix_len > best_match_len and prefix_len <= len(token_tuple):
                if token_tuple[:prefix_len] == prefix:
                    best_match_len = prefix_len
                    best_entry = entry

        if best_entry is not None and best_match_len > 0:
            best_entry.last_accessed = time.time()
            best_entry.hit_count += 1
            cloned = best_entry.cache.clone()
            if cloned.seq_len > best_match_len:
                cloned.truncate(best_match_len)
            return best_match_len, cloned

        return 0, None

    def store_prefix(self, token_ids: list[int] | tuple[int, ...], cache: AiraInferenceCache) -> None:
        """Stores a token prefix and a clone of its inference cache."""
        token_tuple = tuple(token_ids)
        if len(token_tuple) == 0:
            return

        # Evict LRU entry if at capacity
        if len(self.entries) >= self.max_entries and token_tuple not in self.entries:
            oldest_key = min(self.entries.keys(), key=lambda k: self.entries[k].last_accessed)
            del self.entries[oldest_key]

        self.entries[token_tuple] = PrefixCacheEntry(
            token_ids=token_tuple,
            cache=cache.clone(),
        )

    def clear(self) -> None:
        self.entries.clear()


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

    def forward(
        self,
        x: torch.Tensor,
        cache: LayerCache | None = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.kind == "mla":
            mixer_out = self.mixer(self.norm1(x), cache=cache, start_pos=start_pos)
        else:
            mixer_out = self.mixer(self.norm1(x), cache=cache)
        x = x + mixer_out
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
        cache: AiraInferenceCache | None = None,
        start_pos: int | None = None,
    ) -> dict[str, torch.Tensor | None]:
        if cache is None and input_ids.size(1) > self.config.context_len:
            raise ValueError(f"Sequence length {input_ids.size(1)} exceeds context_len={self.config.context_len}")
        x = self.embed_tokens(input_ids)

        curr_start_pos = cache.seq_len if (cache is not None and start_pos is None) else (start_pos or 0)
        total_aux_loss = torch.tensor(0.0, device=x.device)
        for i, block in enumerate(self.blocks):
            layer_cache = cache.layers[i] if cache is not None else None
            if self.config.gradient_checkpointing and self.training:
                x, aux = checkpoint(block, x, use_reentrant=False)
            else:
                x, aux = block(x, cache=layer_cache, start_pos=curr_start_pos)
            total_aux_loss = total_aux_loss + aux

        if cache is not None:
            cache.seq_len = curr_start_pos + input_ids.size(1)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # Primary next-token loss (calculated in float32 for numerical stability)
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
                loss = F.cross_entropy(shift_logits, labels[:, :-1].reshape(-1))

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

        return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int | None = 50,
        eos_token_id: int | None = None,
        use_cache: bool = True,
        prefix_cache: AiraPrefixCacheManager | None = None,
    ) -> torch.Tensor:
        self.eval()
        eos_token_id = self.config.eos_token_id if eos_token_id is None else eos_token_id

        if not use_cache:
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

        # Fast O(1) Cache-Accelerated Generation
        bsz = input_ids.size(0)
        prompt_len = input_ids.size(1)
        if prompt_len > self.config.context_len:
            input_ids = input_ids[:, -self.config.context_len :]
            prompt_len = input_ids.size(1)

        cache = None
        matched_len = 0

        # Radix Prefix Cache match
        if prefix_cache is not None and bsz == 1:
            prompt_tokens = input_ids[0].tolist()
            matched_len, matched_cache = prefix_cache.match_prefix(prompt_tokens)
            if matched_len > 0 and matched_cache is not None:
                cache = matched_cache

        if cache is None:
            cache = AiraInferenceCache(self.config, batch_size=bsz)

        # 1. Prefill pass (evaluates only newly arriving tokens past the matched prefix)
        if matched_len < prompt_len:
            unprocessed_tokens = input_ids[:, matched_len:]
            out = self(unprocessed_tokens, cache=cache, start_pos=matched_len)
            logits = out["logits"][:, -1, :]
        else:
            last_token = input_ids[:, -1:]
            out = self(last_token, cache=cache, start_pos=prompt_len - 1)
            logits = out["logits"][:, -1, :]

        # Register prompt state into Prefix Cache for subsequent conversation turns
        if prefix_cache is not None and bsz == 1:
            prefix_cache.store_prefix(input_ids[0, :cache.seq_len].tolist(), cache)

        logits = logits / max(temperature, 1e-6)
        if top_k is not None:
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = torch.where(logits < values[:, [-1]], torch.full_like(logits, -float("inf")), logits)
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = [next_token]

        if bool((next_token == eos_token_id).all()):
            return torch.cat([input_ids, next_token], dim=1)

        # 2. Decode pass: single token at a time in O(1) time
        curr_pos = prompt_len
        curr_token = next_token

        for _ in range(max_new_tokens - 1):
            if curr_pos >= self.config.context_len:
                break
            out = self(curr_token, cache=cache, start_pos=curr_pos)
            logits = out["logits"][:, -1, :]
            logits = logits / max(temperature, 1e-6)
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = torch.where(logits < values[:, [-1]], torch.full_like(logits, -float("inf")), logits)
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated.append(next_token)
            curr_pos += 1
            curr_token = next_token
            if bool((next_token == eos_token_id).all()):
                break

        full_output = torch.cat([input_ids] + generated, dim=1)

        # Register verified conversation trajectory into prefix cache
        if prefix_cache is not None and bsz == 1:
            prefix_cache.store_prefix(full_output[0, :cache.seq_len].tolist(), cache)

        return full_output


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
