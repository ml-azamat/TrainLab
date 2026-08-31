"""Automatic run names and tags derived from the config.

You should never have to name a run by hand — the name is a lossy but readable summary,
and the tags are what you actually filter the leaderboard by.
"""

from __future__ import annotations

from .config import TTA, AutoAugment, FreezePolicy, LossName, TrainConfig


def _aug_token(cfg: TrainConfig) -> str | None:
    a = cfg.augmentation.auto_augment
    return {
        AutoAugment.RANDAUGMENT: f"randaug{cfg.augmentation.randaugment_m}",
        AutoAugment.TRIVIALAUGMENT_WIDE: "trivialaug",
        AutoAugment.AUTOAUGMENT_IMAGENET: "autoaug",
        AutoAugment.THREE_AUGMENT: "3aug",
        AutoAugment.NONE: None,
    }[a]


def run_name(cfg: TrainConfig) -> str:
    """e.g. 'convnext_tiny · 224 · randaug9 · mixup · llrd0.75 · 30ep'"""
    parts: list[str] = [cfg.model.backbone, str(cfg.input.input_size)]

    if cfg.effective_test_input_size != cfg.input.input_size:
        parts[-1] = f"{cfg.input.input_size}→{cfg.effective_test_input_size}"

    if cfg.ssl_active:
        # SSL is one of the largest differences between two runs, so it belongs in the
        # name — otherwise pretrained and SSL-pretrained runs are indistinguishable in
        # the comparison table.
        E = cfg.experimental
        tok = f"{E.ssl_method.value}{E.ssl_epochs}"
        if E.ssl_extra_data_dir:
            tok += "+extra"
        parts.append(tok)

    if (t := _aug_token(cfg)):
        parts.append(t)
    if cfg.mixup_active:
        parts.append("mixup")
    if cfg.optimization.layer_lr_decay < 1.0:
        parts.append(f"llrd{cfg.optimization.layer_lr_decay:g}")
    if cfg.optimization.sam:
        parts.append("asam" if cfg.optimization.sam_adaptive else "sam")
    if cfg.effective_loss != LossName.CROSS_ENTROPY:
        parts.append(cfg.effective_loss.value)
    if cfg.model.ema:
        parts.append("ema")
    if cfg.validation.tta != TTA.NONE:
        parts.append(f"tta-{cfg.validation.tta.value}")
    if not cfg.model.pretrained:
        parts.append("scratch")
    if cfg.model.freeze_policy != FreezePolicy.NONE:
        parts.append(f"freeze{cfg.model.freeze_epochs}")
    parts.append(f"{cfg.schedule.epochs}ep")
    return " · ".join(parts)


def run_tags(cfg: TrainConfig, extra: dict | None = None) -> dict[str, str]:
    """Structured, filterable tags. Keep values short — they render in table chips."""
    tags = {
        "preset": cfg.tracking.preset.value,
        "backbone": cfg.model.backbone,
        "input_size": str(cfg.input.input_size),
        "optimizer": cfg.optimization.optimizer.value,
        "scheduler": cfg.schedule.scheduler.value,
        "loss": cfg.effective_loss.value,
        "epochs": str(cfg.schedule.epochs),
        "batch_size": str(cfg.schedule.batch_size),
        "pretrained": str(cfg.model.pretrained).lower(),
        "auto_augment": cfg.augmentation.auto_augment.value,
        "mixup": str(cfg.mixup_active).lower(),
        "ema": str(cfg.model.ema).lower(),
        "sam": str(cfg.optimization.sam).lower(),
        "llrd": f"{cfg.optimization.layer_lr_decay:g}",
        "tta": cfg.validation.tta.value,
        "primary_metric": cfg.primary_metric_key,
        # Experimental stages, so the leaderboard can be filtered by them.
        "ssl_method": cfg.experimental.ssl_method.value,
        "crt": str(cfg.experimental.classifier_retrain_epochs > 0).lower(),
        "pseudo_label": str(bool(cfg.experimental.pseudo_label_dir)).lower(),
        "curriculum": str(cfg.experimental.curriculum_by_loss).lower(),
        "noise_filter": str(cfg.experimental.label_noise_filter).lower(),
        "ensemble": str(len(cfg.experimental.ensemble_run_ids)),
    }
    tags.update(cfg.tracking.tags)
    tags.update(extra or {})
    return {k: str(v) for k, v in tags.items()}


def flatten(cfg_dict: dict, prefix: str = "") -> dict[str, str]:
    """Flatten nested config to dotted keys for the tracker's params table."""
    out: dict[str, str] = {}
    for k, v in cfg_dict.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, f"{key}."))
        elif isinstance(v, (list, tuple)):
            out[key] = ",".join(str(x) for x in v) if v else ""
        else:
            out[key] = "" if v is None else str(v)
    return out
