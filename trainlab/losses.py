"""Loss functions and mixup/cutmix wiring.

The critical invariant: label smoothing is applied in exactly ONE place. When mixup is
active timm's Mixup bakes smoothing into the soft targets it emits, so the loss must not
smooth again — otherwise the effective epsilon is silently wrong.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ClassWeights, LossName, TrainConfig


class SoftTargetCrossEntropy(nn.Module):
    """Cross-entropy against a soft target distribution (as produced by mixup/cutmix).

    Also accepts integer class indices, which it one-hots (with `smoothing`, since the
    mixup collator that normally bakes smoothing in is not producing these targets).
    Integer targets are not an edge case: `mixup_off_epoch` disables mixing for the
    final epochs while this criterion stays installed for the whole run — without the
    fallback that switch crashed the run mid-training (or, when batch_size happened to
    equal num_classes, broadcast silently into a garbage loss).
    """

    def __init__(self, smoothing: float = 0.0):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == 1:
            n = x.shape[-1]
            off = self.smoothing / n
            target = F.one_hot(target, n).float() * (1.0 - self.smoothing) + off
        return torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1).mean()


class BCEWithSoftTargets(nn.Module):
    """Multi-label BCE, per ResNet-Strikes-Back.

    A mixed image genuinely contains two concepts, so forcing the targets to sum to 1 is
    the wrong model of the data. Each present concept gets target 1 (or 1-eps).
    """

    def __init__(self, smoothing: float = 0.0, target_threshold: float | None = None):
        super().__init__()
        self.smoothing = smoothing
        self.target_threshold = target_threshold

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == 1:
            target = F.one_hot(target, x.shape[-1]).float()
        if self.target_threshold is not None:
            target = target.gt(self.target_threshold).float()
        if self.smoothing > 0:
            off = self.smoothing / x.shape[-1]
            target = target.clamp(min=off, max=1.0 - self.smoothing + off)
        return F.binary_cross_entropy_with_logits(x, target)


class FocalLoss(nn.Module):
    """Down-weights easy examples. Only appropriate under severe imbalance."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else torch.empty(0))
        self.label_smoothing = label_smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        w = self.weight if self.weight.numel() else None
        ce = F.cross_entropy(x, target, weight=w, reduction="none",
                             label_smoothing=self.label_smoothing)
        # pt must be the model's probability for the true class. Deriving it as
        # exp(-ce) is only valid for a plain unweighted, unsmoothed cross-entropy: with
        # class weights or smoothing folded in, exp(-ce) is off by the weight factor and
        # the focal term (1-pt)^gamma down-weights the wrong examples — silently, since
        # the loss still looks like a plausible number. Read it from the logits instead.
        with torch.no_grad():
            logp = F.log_softmax(x, dim=-1)
            pt = logp.gather(1, target.unsqueeze(1)).squeeze(1).exp()
        return ((1 - pt) ** self.gamma * ce).mean()


class LogitAdjustedLoss(nn.Module):
    """Adds tau * log(prior) to the logits during training (Menon et al., ICLR 2021).

    A Bayes-consistent correction for label-distribution shift: it fixes the decision
    rule rather than reweighting the data or the loss, which is why it composes better
    than resampling.
    """

    def __init__(self, priors: torch.Tensor, tau: float = 1.0, label_smoothing: float = 0.0):
        super().__init__()
        self.register_buffer("adjust", tau * torch.log(priors.clamp_min(1e-12)))
        self.label_smoothing = label_smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(x + self.adjust, target, label_smoothing=self.label_smoothing)


def class_weights_from_counts(counts: np.ndarray, mode: ClassWeights,
                              beta: float = 0.9999,
                              manual: list[float] | None = None) -> torch.Tensor | None:
    if mode == ClassWeights.NONE:
        return None
    if mode == ClassWeights.MANUAL:
        if not manual:
            raise ValueError("class_weights='manual' but manual_class_weights is empty.")
        w = np.asarray(manual, dtype=np.float64)
    elif mode == ClassWeights.INVERSE_FREQUENCY:
        w = 1.0 / np.maximum(counts, 1)
    else:  # effective-number: (1 - beta^n) / (1 - beta)
        eff = (1.0 - np.power(beta, counts)) / (1.0 - beta)
        w = 1.0 / np.maximum(eff, 1e-12)
    w = w / w.sum() * len(w)  # normalise so the average weight is 1
    return torch.tensor(w, dtype=torch.float32)


def build_loss(cfg: TrainConfig, counts: np.ndarray, device) -> nn.Module:
    L = cfg.loss
    mixup_on = cfg.mixup_active
    kind = cfg.effective_loss
    # When mixup is active it has already applied smoothing to the targets.
    smoothing = 0.0 if mixup_on else L.label_smoothing
    weights = class_weights_from_counts(counts, L.class_weights, L.cb_beta, L.manual_class_weights)
    if weights is not None:
        weights = weights.to(device)

    if kind == LossName.SOFT_TARGET_CE:
        # The smoothing argument only applies to INTEGER targets (epochs after
        # mixup_off_epoch, or soft_target_ce chosen without mixup). Soft targets from
        # the mixup collator arrive pre-smoothed and pass through untouched, so the
        # one-place-only smoothing invariant holds in both phases.
        return SoftTargetCrossEntropy(smoothing=L.label_smoothing)
    if kind == LossName.BCE:
        return BCEWithSoftTargets(smoothing=L.label_smoothing if not mixup_on else 0.0)
    if kind == LossName.FOCAL:
        return FocalLoss(L.focal_gamma, weights, smoothing).to(device)
    if kind == LossName.LOGIT_ADJUSTED:
        priors = torch.tensor(counts / counts.sum(), dtype=torch.float32, device=device)
        return LogitAdjustedLoss(priors, L.logit_adjust_tau, smoothing).to(device)
    # cross_entropy and class_balanced_ce differ only in whether weights are supplied.
    return nn.CrossEntropyLoss(weight=weights, label_smoothing=smoothing)


def build_mixup(cfg: TrainConfig, num_classes: int):
    """timm Mixup collator, or None when mixing is disabled."""
    if not cfg.mixup_active:
        return None
    from timm.data import Mixup

    a = cfg.augmentation
    return Mixup(
        mixup_alpha=a.mixup_alpha,
        cutmix_alpha=a.cutmix_alpha,
        prob=a.mixup_prob,
        switch_prob=a.mixup_switch_prob,
        mode="batch",
        label_smoothing=cfg.loss.label_smoothing,
        num_classes=num_classes,
    )


def mixup_enabled_this_epoch(cfg: TrainConfig, epoch: int) -> bool:
    off = cfg.augmentation.mixup_off_epoch
    return cfg.mixup_active and (off == 0 or epoch < off)
