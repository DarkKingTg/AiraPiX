from __future__ import annotations

import math
from typing import Iterable

import torch


def zeropower_via_newtonschulz5(update: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    if update.ndim != 2:
        return update
    original_dtype = update.dtype
    try:
        x = update.float()
        if not torch.isfinite(x).all():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        norm_val = x.norm()
        if norm_val < eps:
            return update
        if x.size(0) > x.size(1):
            x = x.T
            transposed = True
        else:
            transposed = False
        x = x / (norm_val + eps)
        a, b, c = 3.4445, -4.7750, 2.0315
        for _ in range(steps):
            xx_t = x @ x.T
            if not torch.isfinite(xx_t).all():
                return update / (norm_val + eps)
            xx_t = xx_t.clamp(min=-50.0, max=50.0)
            x = a * x + (b * xx_t + c * (xx_t @ xx_t)) @ x
            if not torch.isfinite(x).all():
                return update / (norm_val + eps)
        if transposed:
            x = x.T
        res = x.to(dtype=original_dtype)
        if not torch.isfinite(res).all():
            return update / (norm_val + eps)
        return res
    except Exception:
        return update


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
                    state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)
                buf = state["momentum_buffer"]
                grad_f32 = grad.float()
                if not torch.isfinite(grad_f32).all():
                    grad_f32 = torch.nan_to_num(grad_f32, nan=0.0, posinf=1.0, neginf=-1.0)
                buf.mul_(momentum).add_(grad_f32)
                update = grad_f32.add(buf, alpha=momentum) if nesterov else buf
                if update.ndim == 2:
                    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
                    scale = math.sqrt(max(1.0, update.size(0) / max(1, update.size(1))))
                    update = (update * scale).clamp(min=-5.0, max=5.0)
                p.add_(update.to(dtype=p.dtype), alpha=-lr)
                if not torch.isfinite(p).all():
                    p.nan_to_num_(nan=0.0)
                p.clamp_(min=-50.0, max=50.0)
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
            and "router" not in name
            and "mtp_heads" not in name
            and "norm" not in name
            and "dt_proj" not in name
            and "b_proj" not in name
            and "c_proj" not in name
            and "a_log" not in name
            and "d_skip" not in name
        )
        if use_muon:
            muon_params.append(param)
        else:
            adamw_params.append(param)
    return muon_params, adamw_params


def build_optimizer(model: torch.nn.Module, cfg: dict) -> HybridMuonAdamW:
    peak_lr = float(cfg.get("peak_lr", 3.5e-4))
    muon_lr = float(cfg.get("muon_lr", 0.02))
    weight_decay = float(cfg.get("weight_decay", 0.01))
    muon_params, adamw_params = split_muon_adamw_params(model)

    muon = Muon(
        muon_params,
        lr=muon_lr,
        momentum=float(cfg.get("muon_momentum", 0.95)),
        weight_decay=weight_decay,
        ns_steps=int(cfg.get("muon_ns_steps", 2)),
    ) if muon_params else None

    betas = tuple(float(x) for x in cfg.get("adamw_betas", [0.9, 0.95]))
    adamw_lr = float(cfg.get("adamw_lr", peak_lr))
    # Exempt 1D parameters and sensitive SSM recurrent projections (dt_proj, b_proj, c_proj) from weight decay
    ssm_recurrent_names = {"dt_proj", "b_proj", "c_proj"}
    ssm_recurrent_params = {
        param for name, param in model.named_parameters()
        if any(target in name for target in ssm_recurrent_names)
    }

    decay_params = [p for p in adamw_params if p.ndim >= 2 and p not in ssm_recurrent_params]
    nodecay_params = [p for p in adamw_params if p.ndim < 2 or p in ssm_recurrent_params]
    adamw_groups = []
    if decay_params:
        adamw_groups.append({"params": decay_params, "weight_decay": weight_decay, "lr": adamw_lr})
    if nodecay_params:
        adamw_groups.append({"params": nodecay_params, "weight_decay": 0.0, "lr": adamw_lr})

    adamw = torch.optim.AdamW(
        adamw_groups,
        betas=betas,
        eps=float(cfg.get("adamw_eps", 1e-8)),
    ) if adamw_groups else None
    return HybridMuonAdamW(muon, adamw)


def set_optimizer_lr(
    optimizer: HybridMuonAdamW,
    decay_scale: float,
    peak_muon_lr: float = 0.02,
    peak_adamw_lr: float = 3.5e-4,
) -> tuple[float, float]:
    """Dynamically scales Muon and AdamW learning rates using the scheduler decay ratio."""
    current_muon_lr = peak_muon_lr * decay_scale
    current_adamw_lr = peak_adamw_lr * decay_scale

    if optimizer.muon is not None:
        for group in optimizer.muon.param_groups:
            group["lr"] = current_muon_lr
    if optimizer.adamw is not None:
        for group in optimizer.adamw.param_groups:
            group["lr"] = current_adamw_lr

    return current_muon_lr, current_adamw_lr
