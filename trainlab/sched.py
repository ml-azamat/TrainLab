"""Learning-rate schedules.

Stepped per-optimizer-step (not per-epoch) so warmup and cosine are smooth regardless
of dataset size. Each param group keeps its own base LR, so layer-wise LR decay and the
head multiplier survive scheduling.
"""

from __future__ import annotations

import math

from .config import Scheduler, TrainConfig


class LRSchedule:
    """Computes a multiplicative factor on each group's base LR."""

    def __init__(self, cfg: TrainConfig, steps_per_epoch: int):
        s = cfg.schedule
        self.kind = s.scheduler
        self.base_lr = cfg.optimization.lr
        self.total_steps = max(1, steps_per_epoch * s.epochs)
        self.warmup_steps = int(steps_per_epoch * s.warmup_epochs)
        self.warmup_start = s.warmup_start_lr
        self.min_lr = s.min_lr
        self.steps_per_epoch = steps_per_epoch
        self.step_size = max(1, s.epochs // 3) * steps_per_epoch
        self.step_gamma = 0.1
        self._plateau_factor = 1.0

    #: One-cycle shape constants, matching torch.optim.lr_scheduler.OneCycleLR defaults.
    ONE_CYCLE_PCT_START = 0.3
    ONE_CYCLE_DIV = 25.0          # initial lr = base / 25
    ONE_CYCLE_FINAL_DIV = 1e4     # final lr = base / 1e4

    def factor(self, step: int) -> float:
        """Multiplier on the base LR at global step `step`."""
        if self.kind == Scheduler.ONE_CYCLE:
            # A real one-cycle (Smith 2018): ramp up to the base LR over the first
            # 30% of training, then anneal down well below the starting point. The
            # cycle IS the warmup, so the generic warmup branch below does not apply
            # — the previous implementation skipped the up-phase entirely and was
            # just cosine with a deeper floor wearing the one_cycle name.
            up = max(1, int(self.total_steps * self.ONE_CYCLE_PCT_START))
            lo0 = 1.0 / self.ONE_CYCLE_DIV
            if step < up:
                t = step / up
                return lo0 + (1 - lo0) * 0.5 * (1 - math.cos(math.pi * t))
            lo1 = 1.0 / self.ONE_CYCLE_FINAL_DIV
            t = min(1.0, (step - up) / max(1, self.total_steps - up))
            return lo1 + (1 - lo1) * 0.5 * (1 + math.cos(math.pi * t))

        if self.warmup_steps > 0 and step < self.warmup_steps:
            w = step / max(1, self.warmup_steps)
            lr = self.warmup_start + (self.base_lr - self.warmup_start) * w
            return lr / self.base_lr

        if self.kind == Scheduler.CONSTANT:
            return 1.0
        if self.kind == Scheduler.PLATEAU:
            return self._plateau_factor
        if self.kind == Scheduler.STEP:
            return self.step_gamma ** ((step - self.warmup_steps) // self.step_size)

        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, progress))

        floor = self.min_lr / self.base_lr
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    # The only mutable state: everything else is derived from the global step, which is
    # what makes a resumed schedule pick up exactly where it stopped once that step is
    # restored. The plateau factor is the exception — it is a product of *decisions*
    # (halvings), not of time, so it has to travel in the checkpoint.
    def state_dict(self) -> dict:
        return {"plateau_factor": self._plateau_factor}

    def load_state_dict(self, state: dict) -> None:
        self._plateau_factor = float(state.get("plateau_factor", 1.0))

    def on_plateau(self, improved: bool) -> None:
        """Only meaningful for scheduler='plateau'."""
        if self.kind == Scheduler.PLATEAU and not improved:
            self._plateau_factor = max(self.min_lr / self.base_lr, self._plateau_factor * 0.5)

    def apply(self, optimizer, step: int) -> float:
        f = self.factor(step)
        lr_seen = 0.0
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
            group["lr"] = group["initial_lr"] * f
            lr_seen = max(lr_seen, group["lr"])
        return lr_seen
