"""Parameter groups, optimizers, and SAM."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import Optimizer as OptName, TrainConfig
from .models import get_classifier


def build_param_groups(model: nn.Module, cfg: TrainConfig) -> list[dict]:
    """Weight-decay exclusion, layer-wise LR decay, and the head LR multiplier.

    Uses timm's layer-decay helper when LLRD is on, since it knows each architecture's
    depth structure via `group_matcher`; hand-rolling that mapping is where LLRD
    implementations usually go wrong.
    """
    O = cfg.optimization
    skip = set()
    if O.no_weight_decay_on_norm_bias:
        # Architectures declare their own no-decay params (pos_embed, cls_token, ...).
        for attr in ("no_weight_decay",):
            fn = getattr(model, attr, None)
            if callable(fn):
                try:
                    skip |= set(fn())
                except Exception:
                    pass

    if O.layer_lr_decay < 1.0:
        from timm.optim import param_groups_layer_decay

        groups = param_groups_layer_decay(
            model,
            weight_decay=O.weight_decay,
            no_weight_decay_list=skip,
            layer_decay=O.layer_lr_decay,
        )
        # timm emits relative `lr_scale`; materialise it against the base LR.
        for g in groups:
            g["lr"] = O.lr * g.get("lr_scale", 1.0)
        return _apply_head_mult(groups, model, O.head_lr_mult, O.lr)

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        excluded = (
            O.no_weight_decay_on_norm_bias
            and (p.ndim <= 1 or name.endswith(".bias") or name in skip)
        )
        (no_decay if excluded else decay).append((name, p))

    groups = [
        {"params": [p for _, p in decay], "weight_decay": O.weight_decay, "lr": O.lr,
         "param_names": [n for n, _ in decay]},
        {"params": [p for _, p in no_decay], "weight_decay": 0.0, "lr": O.lr,
         "param_names": [n for n, _ in no_decay]},
    ]
    return _apply_head_mult([g for g in groups if g["params"]], model, O.head_lr_mult, O.lr)


def _apply_head_mult(groups: list[dict], model: nn.Module, mult: float, base_lr: float) -> list[dict]:
    """Split the classifier head into its own group at `mult` x the base LR."""
    if mult == 1.0:
        return groups
    head = get_classifier(model)
    if head is None:
        return groups
    head_ids = {id(p) for p in head.parameters()}

    out: list[dict] = []
    for g in groups:
        keep = [p for p in g["params"] if id(p) not in head_ids]
        moved = [p for p in g["params"] if id(p) in head_ids]
        if keep:
            out.append({**g, "params": keep})
        if moved:
            out.append({**{k: v for k, v in g.items() if k != "param_names"},
                        "params": moved, "lr": base_lr * mult, "is_head": True})
    return out


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    O = cfg.optimization
    groups = build_param_groups(model, cfg)
    for g in groups:
        g.pop("param_names", None)

    fused_ok = torch.cuda.is_available()

    if O.optimizer == OptName.ADAMW:
        kw = dict(lr=O.lr, betas=tuple(O.betas), eps=O.eps, weight_decay=O.weight_decay)
        if fused_ok:
            kw["fused"] = True
        return torch.optim.AdamW(groups, **kw)

    if O.optimizer == OptName.SGD:
        kw = dict(lr=O.lr, momentum=O.momentum, nesterov=O.nesterov, weight_decay=O.weight_decay)
        if fused_ok:
            kw["fused"] = True
        return torch.optim.SGD(groups, **kw)

    if O.optimizer == OptName.LION:
        from timm.optim import Lion

        return Lion(groups, lr=O.lr, betas=tuple(O.betas), weight_decay=O.weight_decay)

    return torch.optim.RMSprop(groups, lr=O.lr, momentum=O.momentum,
                               eps=O.eps, weight_decay=O.weight_decay)


def suggest_lr(base_lr: float, base_batch: int, batch: int, rule: str) -> float:
    """LR suggestion when batch size changes. Advisory only — never auto-applied."""
    if rule == "linear":
        return base_lr * batch / base_batch
    if rule == "sqrt":
        return base_lr * math.sqrt(batch / base_batch)
    return base_lr


# --------------------------------------------------------------------------------------
# SAM
# --------------------------------------------------------------------------------------

class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization (Foret et al., ICLR 2021).

    Two forward-backward passes per step, so it doubles training time. Large win on
    ViTs (+5 points on ViT-B/16), marginal on well-regularized CNNs (+0.8).
    `adaptive=True` gives ASAM, whose rho is scale-invariant and easier to tune.
    """

    def __init__(self, base_optimizer: torch.optim.Optimizer, rho: float = 0.05,
                 adaptive: bool = False):
        if rho < 0:
            raise ValueError(f"rho must be non-negative, got {rho}")
        self.base_optimizer = base_optimizer
        self.rho = rho
        self.adaptive = adaptive
        super().__init__(base_optimizer.param_groups, dict(rho=rho, adaptive=adaptive))
        self.defaults.update(base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self) -> None:
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = self.rho / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if self.adaptive else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

    @torch.no_grad()
    def restore(self) -> None:
        """Undo the ascent step, leaving the descent step to the caller.

        Split out from `second_step` so the caller can drive the actual update through a
        GradScaler when AMP is active; `second_step` remains for direct use.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if "old_p" not in self.state[p]:
                    continue
                p.data = self.state[p]["old_p"]  # back to the original point
                del self.state[p]["old_p"]       # don't pin a weight-sized copy per param

    @torch.no_grad()
    def second_step(self) -> None:
        self.restore()
        self.base_optimizer.step()

    def _grad_norm(self) -> torch.Tensor:
        shared = self.param_groups[0]["params"][0].device
        return torch.norm(torch.stack([
            ((torch.abs(p) if self.adaptive else 1.0) * p.grad).norm(p=2).to(shared)
            for group in self.param_groups for p in group["params"] if p.grad is not None
        ]), p=2)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, sd):
        self.base_optimizer.load_state_dict(sd)
        self.param_groups = self.base_optimizer.param_groups
