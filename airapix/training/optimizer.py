from __future__ import annotations

import math
from typing import Iterable

import torch


def zeropower_via_newtonschulz5(update: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    if update.ndim != 2:
        return update
    original_dtype = update.dtype
    x = update.float()
    if x.size(0) > x.size(1):
        x = x.T
        transposed = True
    else:
        transposed = False
    x = x / (x.norm() + eps)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        xx_t = x @ x.T
        x = a * x + (b * xx_t + c * (xx_t @ xx_t)) @ x
    if transposed:
        x = x.T
    return x.to(dtype=original_dtype)


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
        nesterov: bool = True,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            nesterov=nesterov,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            ns_steps = group["ns_steps"]
            nesterov = group["nesterov"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients.")
                if weight_decay:
                    p.mul_(1.0 - lr * weight_decay)
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)
                update = grad.add(buf, alpha=momentum) if nesterov else buf
                if update.ndim == 2:
                    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
                    scale = math.sqrt(max(1.0, update.size(0) / max(1, update.size(1))))
                    update = update * scale
                p.add_(update, alpha=-lr)
        return loss


class HybridMuonAdamW:
    def __init__(self, muon: Muon | None, adamw: torch.optim.AdamW | None) -> None:
        self.muon = muon
        self.adamw = adamw
        self.param_groups = []
        if self.muon is not None:
            self.param_groups.extend(self.muon.param_groups)
        if self.adamw is not None:
            self.param_groups.extend(self.adamw.param_groups)

    def step(self) -> None:
        if self.muon is not None:
            self.muon.step()
        if self.adamw is not None:
            self.adamw.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        if self.muon is not None:
            self.muon.zero_grad(set_to_none=set_to_none)
        if self.adamw is not None:
            self.adamw.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        return {
            "muon": self.muon.state_dict() if self.muon is not None else None,
            "adamw": self.adamw.state_dict() if self.adamw is not None else None,
        }

    def load_state_dict(self, state: dict) -> None:
        if self.muon is not None and state.get("muon") is not None:
            self.muon.load_state_dict(state["muon"])
        if self.adamw is not None and state.get("adamw") is not None:
            self.adamw.load_state_dict(state["adamw"])


def split_muon_adamw_params(model: torch.nn.Module):
    muon_params = []
    adamw_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        use_muon = (
            param.ndim == 2
            and "embed_tokens" not in name
            and "lm_head" not in name
            and "norm" not in name
        )
        if use_muon:
            muon_params.append(param)
        else:
            adamw_params.append(param)
    return muon_params, adamw_params


def build_optimizer(model: torch.nn.Module, cfg: dict) -> HybridMuonAdamW:
    peak_lr = float(cfg["peak_lr"])
    weight_decay = float(cfg.get("weight_decay", 0.1))
    muon_params, adamw_params = split_muon_adamw_params(model)

    muon = Muon(
        muon_params,
        lr=peak_lr * float(cfg.get("muon_lr_multiplier", 1.0)),
        momentum=float(cfg.get("muon_momentum", 0.95)),
        weight_decay=weight_decay,
        ns_steps=int(cfg.get("muon_ns_steps", 5)),
    ) if muon_params else None

    betas = tuple(float(x) for x in cfg.get("adamw_betas", [0.9, 0.95]))
    adamw = torch.optim.AdamW(
        adamw_params,
        lr=peak_lr * float(cfg.get("adamw_lr_multiplier", 1.0)),
        betas=betas,
        eps=float(cfg.get("adamw_eps", 1e-8)),
        weight_decay=weight_decay,
    ) if adamw_params else None
    return HybridMuonAdamW(muon, adamw)


def set_optimizer_lr(optimizer: HybridMuonAdamW, base_lr: float, cfg: dict) -> None:
    muon_mult = float(cfg.get("muon_lr_multiplier", 1.0))
    adamw_mult = float(cfg.get("adamw_lr_multiplier", 1.0))
    if optimizer.muon is not None:
        for group in optimizer.muon.param_groups:
            group["lr"] = base_lr * muon_mult
    if optimizer.adamw is not None:
        for group in optimizer.adamw.param_groups:
            group["lr"] = base_lr * adamw_mult
