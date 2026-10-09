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
    attention_every: int = 4  # 3:1 schedule: 3 SSM/SSD layers followed by 1 MLA layer (Layers 3, 7, 11...)
    mla_latent_dim: int = 256
    qk_nope_dim: int = 48
    qk_rope_dim: int = 16
    v_head_dim: int = 64
    rope_base: float = 10000.0
    use_qk_norm: bool = True
    use_yarn: bool = True
    yarn_scale: float = 1.0

    # State Space Model (Mamba-2 SSD) Settings
    ssm_inner_mult: float = 2.0
    ssm_conv_kernel: int = 4
    use_ssd_chunking: bool = True
    chunk_size: int = 64

    # Feed-Forward Network & Staged Single-GPU MoE Settings
    ffn_hidden_mult: float = 4.0
    use_moe: bool = True
    moe_num_experts: int = 4        # Staged single-GPU default: 4 routed experts
    moe_top_k: int = 1             # Top-1 routing for low routing overhead
    num_shared_experts: int = 1    # 1 always-active shared expert
    moe_start_layer: int = 8       # Upper 1/3 of layers (Layer 8 for 12-layer model)
    moe_stride: int = 2            # MoE in every 2nd upper block

    # Multi-Token Prediction (MTP) / Speculative Draft Head Settings
    num_mtp_heads: int = 2

    # SigLIP 2 Vision ("Pix") Settings
    use_vision: bool = False
    vision_model_name: str = "siglip2-b"
    vision_image_size: int = 224
    vision_patch_size: int = 14
    vision_hidden_dim: int = 768
    num_visual_tokens: int = 64     # Resampled visual latent tokens per image

    dropout: float = 0.0
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    tie_embeddings: bool = True
    gradient_checkpointing: bool = True

    def layer_type(self, layer_idx: int) -> str:
        # 3:1 schedule: Layers (idx + 1) % 4 == 0 are global MLA attention
        return "mla" if (layer_idx + 1) % self.attention_every == 0 else "ssm"

    def is_moe_layer(self, layer_idx: int) -> bool:
        if not self.use_moe:
            return False
        if layer_idx < self.moe_start_layer:
            return False
        return (layer_idx - self.moe_start_layer) % self.moe_stride == 0

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
    "tiny": dict(d_model=128, n_layers=4, n_heads=4, mla_latent_dim=64, qk_nope_dim=24, qk_rope_dim=8, v_head_dim=32, moe_start_layer=2, moe_stride=2),
    "60m": dict(d_model=384, n_layers=12, n_heads=6, mla_latent_dim=192, qk_nope_dim=48, qk_rope_dim=16, v_head_dim=64, moe_start_layer=8, moe_stride=2),
    "90m": dict(d_model=448, n_layers=14, n_heads=7, mla_latent_dim=224, qk_nope_dim=48, qk_rope_dim=16, v_head_dim=64, moe_start_layer=9, moe_stride=2),
    "125m": dict(d_model=512, n_layers=16, n_heads=8, mla_latent_dim=256, qk_nope_dim=48, qk_rope_dim=16, v_head_dim=64, moe_start_layer=10, moe_stride=2),
    "400m": dict(d_model=896, n_layers=20, n_heads=14, mla_latent_dim=384, qk_nope_dim=64, qk_rope_dim=16, v_head_dim=64, moe_start_layer=12, moe_stride=2),
    "1.5b": dict(d_model=1536, n_layers=24, n_heads=16, mla_latent_dim=512, qk_nope_dim=64, qk_rope_dim=32, v_head_dim=64, moe_start_layer=16, moe_stride=2),
    "3b": dict(d_model=2560, n_layers=28, n_heads=20, mla_latent_dim=640, qk_nope_dim=96, qk_rope_dim=32, v_head_dim=96, moe_start_layer=18, moe_stride=2),
    "7b": dict(d_model=3584, n_layers=30, n_heads=28, mla_latent_dim=896, qk_nope_dim=128, qk_rope_dim=64, v_head_dim=128, moe_start_layer=20, moe_stride=2),
    "8b": dict(d_model=4096, n_layers=32, n_heads=32, mla_latent_dim=1024, qk_nope_dim=128, qk_rope_dim=64, v_head_dim=128, moe_start_layer=21, moe_stride=2),
}


def config_from_preset(name: str, **overrides) -> AiraConfig:
    if name not in MODEL_SIZE_PRESETS:
        raise KeyError(f"Unknown model preset {name!r}. Choose one of {sorted(MODEL_SIZE_PRESETS)}")
    data = dict(MODEL_SIZE_PRESETS[name])
    data.update(overrides)
    return AiraConfig(**data)
