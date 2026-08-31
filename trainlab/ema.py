"""Exponential moving average of weights, and SWA.

The horizon trap this guards against: `ema_decay=0.9998` averages over 1/(1-d) = 5,000
steps. On a small dataset that is longer than the entire run, so the EMA weights never
leave initialisation and the EMA metric is meaningless. `resolve_decay` sizes the horizon
against the actual run length when decay is 'auto'.
"""

from __future__ import annotations

import copy
from typing import Iterable

import torch
import torch.nn as nn

#: Fraction of total training steps the EMA horizon should span under 'auto'.
AUTO_HORIZON_FRACTION = 0.1


def resolve_decay(decay: float | str, total_steps: int) -> tuple[float, str | None]:
    """Returns (decay, note). `note` is set when the value was changed or is suspect."""
    if decay == "auto":
        horizon = max(10.0, AUTO_HORIZON_FRACTION * total_steps)
        d = 1.0 - 1.0 / horizon
        return d, f"ema_decay auto -> {d:.6f} (horizon {horizon:,.0f} of {total_steps:,} steps)"

    d = float(decay)
    horizon = 1.0 / max(1e-12, 1.0 - d)
    if horizon > 0.5 * total_steps:
        return d, (f"ema_decay {d} has a horizon of {horizon:,.0f} steps but the run is only "
                   f"{total_steps:,} steps — the EMA metric will be unreliable. "
                   f"Consider ema_decay='auto'.")
    return d, None


class ModelEMA:
    """Keeps a shadow copy of the weights updated as ema = d*ema + (1-d)*live.

    Buffers (BatchNorm running stats) are copied rather than averaged, matching timm.
    """

    def __init__(self, model: nn.Module, decay: float, device=None, warmup_steps: int = 0):
        self.module = copy.deepcopy(model).eval()
        self.module.requires_grad_(False)
        if device is not None:
            self.module.to(device)
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.updates = 0

    def _current_decay(self) -> float:
        if self.warmup_steps <= 0:
            return self.decay
        # Ramp the decay in, so early steps are not dominated by the initialisation.
        # The ramp constant is `warmup_steps`, not a hardcoded 10 — the parameter was
        # accepted, stored and then ignored, so passing 100 behaved exactly like 1.
        return min(self.decay, (1 + self.updates) / (self.warmup_steps + self.updates))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = self._current_decay()
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if not v.dtype.is_floating_point:
                v.copy_(msd[k])
                continue
            v.mul_(d).add_(msd[k].detach().to(v.device), alpha=1.0 - d)

    def state_dict(self):
        return self.module.state_dict()


class SWA:
    """Equal-weight average of checkpoints collected during the SWA phase."""

    def __init__(self, model: nn.Module, device=None):
        self.module = copy.deepcopy(model).eval()
        self.module.requires_grad_(False)
        if device is not None:
            self.module.to(device)
        self.n = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        self.n += 1
        for k, v in self.module.state_dict().items():
            src = msd[k].detach().to(v.device)
            if not v.dtype.is_floating_point:
                v.copy_(src)
            else:
                v.add_((src - v) / self.n)

    @torch.no_grad()
    def update_bn(self, loader, device, max_batches: int = 100) -> None:
        """Recompute BatchNorm statistics — averaged weights invalidate the old ones."""
        bns = [m for m in self.module.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
        if not bns:
            return
        for m in bns:
            m.reset_running_stats()
            m.momentum = None
        self.module.train()
        for i, batch in enumerate(loader):    # datasets yield (image, label, index)
            if i >= max_batches:
                break
            if batch is None:                 # a batch whose images were all unreadable
                continue
            self.module(batch[0].to(device))
        self.module.eval()

    def state_dict(self):
        return self.module.state_dict()
