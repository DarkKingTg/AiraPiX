from __future__ import annotations

import math
from typing import Sequence
import torch
from torch import nn


class LoRALinear(nn.Module):
    """Wraps a linear module with low-rank adapter (LoRA) weights."""

    def __init__(
        self,
        base_layer: nn.Module,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0

        in_features = getattr(base_layer, "in_features", None)
        out_features = getattr(base_layer, "out_features", None)
        if in_features is None or out_features is None:
            # Fallback for wrapped bitsandbytes or non-standard layers
            in_features = getattr(base_layer, "in_proj", None) or base_layer.weight.shape[1]
            out_features = getattr(base_layer, "out_proj", None) or base_layer.weight.shape[0]

        self.in_features = in_features
        self.out_features = out_features

        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
            self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
            self._reset_parameters()
        else:
            self.lora_A = None
            self.lora_B = None
            self.dropout = nn.Identity()

    def _reset_parameters(self) -> None:
        if self.lora_A is not None:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        if self.lora_B is not None:
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(x)
        if self.r > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
            result = result + lora_out * self.scaling
        return result


def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    """Freeze all base model parameters, leaving only LoRA parameters trainable."""
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
        else:
            p.requires_grad = True

    if bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoRALinear) and hasattr(m.base_layer, "bias") and m.base_layer.bias is not None:
                m.base_layer.bias.requires_grad = True


def apply_lora_to_model(
    model: nn.Module,
    r: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.05,
    target_modules: Sequence[str] = (
        "q_proj",
        "kv_down",
        "kv_up",
        "k_rope_proj",
        "out_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "in_proj",
    ),
) -> nn.Module:
    """Recursively replace specified Linear modules with LoRALinear modules."""
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear) or ("Linear" in child.__class__.__name__):
            if any(target in name for target in target_modules):
                setattr(model, name, LoRALinear(child, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout))
        else:
            apply_lora_to_model(child, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, target_modules=target_modules)

    mark_only_lora_as_trainable(model)
    return model


def get_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Returns a state_dict containing only trainable LoRA parameters."""
    return {k: v for k, v in model.state_dict().items() if "lora_" in k}
