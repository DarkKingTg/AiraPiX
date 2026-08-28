from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass
class AiraConfig:
    vocab_size: int = 12000
    context_len: int = 1024
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    attention_every: int = 4
    mla_latent_dim: int = 256
    qk_nope_dim: int = 48
    qk_rope_dim: int = 16
    v_head_dim: int = 64
    rope_base: float = 10000.0
    ssm_inner_mult: float = 2.0
    ssm_conv_kernel: int = 4
    ffn_hidden_mult: float = 4.0
    dropout: float = 0.0
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    tie_embeddings: bool = True
    gradient_checkpointing: bool = True

    def layer_type(self, layer_idx: int) -> str:
        return "mla" if layer_idx % self.attention_every == 0 else "ssm"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AiraConfig":
        allowed = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in data.items() if k in allowed})

    @classmethod
    def from_json(cls, path: str | Path) -> "AiraConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


MODEL_SIZE_PRESETS = {
    "tiny": dict(d_model=128, n_layers=4, n_heads=4, mla_latent_dim=64, qk_nope_dim=24, qk_rope_dim=8, v_head_dim=32),
    "60m": dict(d_model=384, n_layers=12, n_heads=6, mla_latent_dim=192, qk_nope_dim=48, qk_rope_dim=16, v_head_dim=64),
    "90m": dict(d_model=448, n_layers=14, n_heads=7, mla_latent_dim=224, qk_nope_dim=48, qk_rope_dim=16, v_head_dim=64),
    "125m": dict(d_model=512, n_layers=16, n_heads=8, mla_latent_dim=256, qk_nope_dim=48, qk_rope_dim=16, v_head_dim=64),
    "1.5b": dict(d_model=1536, n_layers=24, n_heads=16, mla_latent_dim=512, qk_nope_dim=64, qk_rope_dim=32, v_head_dim=64),
    "3b": dict(d_model=2560, n_layers=28, n_heads=20, mla_latent_dim=640, qk_nope_dim=96, qk_rope_dim=32, v_head_dim=96),
    "7b": dict(d_model=3584, n_layers=30, n_heads=28, mla_latent_dim=896, qk_nope_dim=128, qk_rope_dim=64, v_head_dim=128),
    "8b": dict(d_model=4096, n_layers=32, n_heads=32, mla_latent_dim=1024, qk_nope_dim=128, qk_rope_dim=64, v_head_dim=128),
}


def config_from_preset(name: str, **overrides) -> AiraConfig:
    if name not in MODEL_SIZE_PRESETS:
        raise KeyError(f"Unknown model preset {name!r}. Choose one of {sorted(MODEL_SIZE_PRESETS)}")
    data = dict(MODEL_SIZE_PRESETS[name])
    data.update(overrides)
    return AiraConfig(**data)
