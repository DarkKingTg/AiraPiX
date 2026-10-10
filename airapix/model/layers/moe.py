from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .ffn import SwiGLU


class DeepSeekMoE(nn.Module):
    """DeepSeek-style Mixture of Experts (1 Shared Expert + N Routed Experts).

    Configured for single-GPU workstation execution (1 shared + 4 routed, Top-1).
    """

    def __init__(
        self,
        d_model: int,
        hidden_mult: float = 4.0,
        num_experts: int = 4,
        top_k: int = 1,
        num_shared_experts: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.num_shared_experts = num_shared_experts

        if num_shared_experts > 0:
            shared_mult = hidden_mult * num_shared_experts / (num_shared_experts + self.top_k)
            self.shared_expert = SwiGLU(d_model, hidden_mult=shared_mult, dropout=dropout)
        else:
            self.shared_expert = None

        routed_mult = hidden_mult / (num_shared_experts + self.top_k) if num_shared_experts > 0 else hidden_mult / self.top_k
        self.experts = nn.ModuleList([
            SwiGLU(d_model, hidden_mult=routed_mult, dropout=dropout)
            for _ in range(num_experts)
        ])
        self.router = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)

        shared_out = self.shared_expert(x) if self.shared_expert is not None else torch.zeros_like(x)

        router_logits = self.router(x_flat)
        routing_weights = F.softmax(router_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        topk_weights = (topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-6)).to(dtype=x_flat.dtype)

        me = routing_weights.mean(dim=0)
        ce = (F.one_hot(topk_indices[:, 0], num_classes=self.num_experts).to(dtype=me.dtype)).mean(dim=0)
        aux_loss = (me * ce).sum() * self.num_experts

        routed_out = torch.zeros_like(x_flat)
        if self.top_k == 1:
            expert_indices = topk_indices.squeeze(-1)
            for i in range(self.num_experts):
                idx = (expert_indices == i).nonzero(as_tuple=True)[0]
                if idx.numel() > 0:
                    expert_tokens = x_flat[idx]
                    expert_output = self.experts[i](expert_tokens.unsqueeze(1)).squeeze(1)
                    res = (expert_output * topk_weights[idx, 0:1]).to(dtype=routed_out.dtype)
                    routed_out[idx] = res
        else:
            for k in range(self.top_k):
                expert_indices = topk_indices[:, k]
                weight_k = topk_weights[:, k : k + 1]
                for i in range(self.num_experts):
                    idx = (expert_indices == i).nonzero(as_tuple=True)[0]
                    if idx.numel() > 0:
                        expert_tokens = x_flat[idx]
                        expert_output = self.experts[i](expert_tokens.unsqueeze(1)).squeeze(1)
                        res = (expert_output * weight_k[idx]).to(dtype=routed_out.dtype)
                        routed_out[idx] += res

        final_out = shared_out + routed_out.view(bsz, seq_len, dim)
        return final_out, aux_loss
