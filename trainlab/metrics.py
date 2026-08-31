"""Metric computation (torchmetrics) and test-time augmentation."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import torchmetrics as tm

from .config import TTA, Metric, TrainConfig, fpr_at_fnr_key


def fpr_at_fnr(scores: torch.Tensor, is_positive: torch.Tensor, target_fnr: float) -> float:
    """False-positive rate at the operating point that misses at most `target_fnr` positives.

    The pair of numbers a detection system is actually deployed on: fix the miss rate you
    can tolerate, and report how many false accepts that costs. Threshold-free metrics
    (accuracy, AUROC) hide exactly this trade-off.

    The threshold is the `k`-th lowest positive score, where `k = floor(target * n_pos)` —
    so the realised FNR is at most the target rather than nearest to it, and the reported
    FPR is never optimistic. Scores exactly at the threshold are accepted, which is the
    conservative reading when many samples tie.

    Returns NaN when either class is absent, since neither rate is defined then.
    """
    pos = scores[is_positive]
    neg = scores[~is_positive]
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    k = min(int(math.floor(target_fnr * pos.numel())), pos.numel() - 1)
    threshold = torch.sort(pos).values[k]
    return (neg >= threshold).float().mean().item()


class FprAtFnr(tm.Metric):
    """`fpr_at_fnr` as an accumulating metric, one instance per FNR target.

    Scores are kept rather than binned: the targets that matter here go down to 1e-3 and
    below, where a fixed threshold grid quantises the answer into uselessness.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self, target_fnr: float, pos_index: int) -> None:
        super().__init__()
        self.target_fnr = target_fnr
        self.pos_index = pos_index
        self.add_state("scores", default=[], dist_reduce_fx="cat")
        self.add_state("labels", default=[], dist_reduce_fx="cat")

    def update(self, probs: torch.Tensor, target: torch.Tensor) -> None:
        self.scores.append(probs[:, self.pos_index].detach().float())
        self.labels.append((target == self.pos_index).detach())

    def compute(self) -> torch.Tensor:
        scores = torch.cat(self.scores)
        labels = torch.cat(self.labels)
        return torch.tensor(fpr_at_fnr(scores, labels, self.target_fnr), device=scores.device)


def unavailable_metrics(cfg: TrainConfig, num_classes: int) -> dict[str, str]:
    """Requested metrics that cannot be computed here, mapped to the reason.

    Split out from `build_metrics` so the caller can refuse to start a run whose PRIMARY
    metric is one of them. Silently dropping it left `row.get(primary)` returning -inf
    every epoch: nothing ever "improved", no `best.ckpt` was written, and the run still
    exited 0 reporting `best acc@5 = -inf`.
    """
    out: dict[str, str] = {}
    want = set(cfg.validation.metrics) | {cfg.validation.primary_metric}
    if Metric.ACC5 in want and num_classes < 5:
        out[Metric.ACC5.value] = (
            f"top-5 accuracy is undefined with only {num_classes} classes"
        )
    if Metric.FPR_AT_FNR in want and num_classes != 2:
        out[Metric.FPR_AT_FNR.value] = (
            f"fpr@fnr is a binary-detection metric and this dataset has {num_classes} "
            f"classes; there is no single positive class to miss"
        )
    return out


def _positive_index(cfg: TrainConfig, class_names: list[str] | None) -> int:
    """Index of the configured positive class in the dataset's class order.

    Class indices come from sorting the class names, so the index cannot be assumed —
    `live`/`spoof` puts `live` at 0, `real`/`fake` puts `fake` at 0. The name is what the
    user chose and the order is what the data says; this is where the two meet. Validation
    has already refused a name that is not in the list, so the fallback only covers a
    caller that passed no names at all.
    """
    name = cfg.validation.positive_class
    if class_names and name in class_names:
        return class_names.index(name)
    return 1


def build_metrics(cfg: TrainConfig, num_classes: int, device,
                  class_names: list[str] | None = None) -> tm.MetricCollection:
    want = set(cfg.validation.metrics) | {cfg.validation.primary_metric}
    task = "multiclass"
    m: dict[str, tm.Metric] = {}

    if Metric.ACC1 in want:
        m["acc@1"] = tm.Accuracy(task=task, num_classes=num_classes, top_k=1)
    if Metric.ACC5 in want and num_classes >= 5:
        m["acc@5"] = tm.Accuracy(task=task, num_classes=num_classes, top_k=5)
    if Metric.MACRO_F1 in want:
        m["macro-F1"] = tm.F1Score(task=task, num_classes=num_classes, average="macro")
    if Metric.BALANCED_ACC in want:
        m["balanced-accuracy"] = tm.Recall(task=task, num_classes=num_classes, average="macro")
    if Metric.PER_CLASS_RECALL in want:
        m["per-class-recall"] = tm.Recall(task=task, num_classes=num_classes, average=None)
    if Metric.AUROC in want:
        m["auroc"] = tm.AUROC(task=task, num_classes=num_classes, average="macro")
    if Metric.MCC in want:
        m["mcc"] = tm.MatthewsCorrCoef(task=task, num_classes=num_classes)
    if Metric.ECE in want:
        m["ece"] = tm.CalibrationError(task=task, num_classes=num_classes, n_bins=15)
    if Metric.FPR_AT_FNR in want and num_classes == 2:
        # One metric per target rather than one metric returning several numbers, so each
        # lands in the collection under its own key and can be selected as primary,
        # plotted and compared like any other.
        pos = _positive_index(cfg, class_names)
        for target in cfg.validation.fpr_at_fnr_targets:
            m[fpr_at_fnr_key(target)] = FprAtFnr(target_fnr=target, pos_index=pos)

    # compute_groups=False is load-bearing, not a performance toggle.
    #
    # MetricCollection's default groups metrics whose internal state looks identical after
    # the first update and then updates only one member per group. With {acc@1, acc@5,
    # macro-F1} on real data this merges F1Score into the top-5 Accuracy group, and
    # macro-F1 silently reports the acc@5 value: measured 0.9954 vs a true 0.9287 on
    # Imagenette. The merge is data-dependent — in the same run the EMA collection stayed
    # correct while the raw one did not — so it fails silently and intermittently.
    return tm.MetricCollection(m, compute_groups=False).to(device)


@torch.no_grad()
def forward_tta(model, x: torch.Tensor, mode: TTA, *, test_size: int | None = None) -> torch.Tensor:
    """Averaged probabilities under test-time augmentation.

    Averaging is done in probability space, not logit space — logit averaging is
    dominated by whichever view happens to be most confident.
    """
    if mode == TTA.NONE:
        return F.softmax(model(x), dim=-1)

    views = [x]
    if mode == TTA.HFLIP:
        views.append(torch.flip(x, dims=[3]))
    elif mode == TTA.MULTI_CROP:
        views.append(torch.flip(x, dims=[3]))
        s = x.shape[-1]
        c = int(s * 0.875)
        crop = x[:, :, (s - c) // 2:(s - c) // 2 + c, (s - c) // 2:(s - c) // 2 + c]
        up = F.interpolate(crop, size=s, mode="bicubic", align_corners=False)
        views += [up, torch.flip(up, dims=[3])]
    elif mode == TTA.MULTI_SCALE:
        base = test_size or x.shape[-1]
        for scale in (0.9, 1.1):
            sz = max(32, int(round(base * scale / 32) * 32))
            views.append(F.interpolate(x, size=sz, mode="bicubic", align_corners=False))
        views.append(torch.flip(x, dims=[3]))

    probs = torch.stack([F.softmax(model(v), dim=-1) for v in views])
    return probs.mean(0)
