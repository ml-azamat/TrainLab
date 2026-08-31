"""Presets.

Note on mixup: no preset enables it, and neither does any rung of the augmentation
strength ladder (`AUG_PRESETS` in config.py). The evidence for mixup/cutmix comes from
300-600 epoch from-scratch ImageNet training; when fine-tuning a pretrained backbone on a
short schedule it is measurably negative (RESEARCH.md section 0). It stays one slider away
and fully supported -- opt-in rather than on by default.

Note on the `augmentation` overlays below: they name a rung of the ladder instead of
listing values, and `AUG_PRESETS` expands it. Anything listed *alongside* the rung is a
deliberate deviation from it, which is why `small-dataset` ends up reporting
`augmentation.preset == custom`: it wants heavy strength with a tuning-free policy.

Note on `tracking.preset`: a preset names a whole config, so "is this still that preset?"
is answered by comparing configs -- see `describes` at the bottom of this module.
`TrainConfig` calls it to demote a label that has stopped being true.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from typing import Any

from .config import Preset, TrainConfig

PRESETS: dict[Preset, dict[str, Any]] = {
    Preset.FAST_BASELINE: {
        "_description": "Sanity check. Small model, 3 epochs, no augmentation. "
                        "Confirms the pipeline works before you spend real time.",
        "input": {"input_size": 160, "rrc_scale": (0.65, 1.0)},
        "augmentation": {"preset": "light"},
        "model": {"backbone": "resnet18", "drop_path_rate": 0.0, "ema": False},
        "loss": {"label_smoothing": 0.0},
        "optimization": {"lr": 1e-3, "weight_decay": 0.01, "layer_lr_decay": 1.0},
        "schedule": {"epochs": 3, "batch_size": 64, "warmup_epochs": 0.5},
        "validation": {"metrics": ["acc@1"], "tta": "none"},
        "checkpoint": {"early_stopping": False, "save_top_k": 1},
    },
    Preset.BALANCED: {
        "_description": "The recommended starting point for fine-tuning a pretrained "
                        "backbone on a custom dataset.",
        "input": {"input_size": 224, "rrc_scale": (0.65, 1.0)},
        "augmentation": {"preset": "medium"},
        "model": {"backbone": "convnext_tiny", "drop_path_rate": 0.1,
                  "ema": True, "ema_decay": "auto"},
        "loss": {"loss": "cross_entropy", "label_smoothing": 0.1},
        "optimization": {"lr": 3e-4, "weight_decay": 0.05, "layer_lr_decay": 0.75},
        "schedule": {"epochs": 30, "batch_size": 64, "warmup_epochs": 3},
        "validation": {"metrics": ["acc@1", "acc@5", "macro-F1"], "tta": "none"},
        "checkpoint": {"early_stopping": True, "es_patience": 10},
    },
    Preset.MAX_ACCURACY: {
        "_description": "Long schedule, heavy augmentation, EMA and TTA. Expect ~4x the "
                        "runtime of Balanced for a low-single-digit accuracy gain.",
        "input": {"input_size": 224, "test_input_size": 256, "rrc_scale": (0.4, 1.0),
                  "val_crop_pct": 0.95},
        "augmentation": {"preset": "heavy"},
        "model": {"backbone": "convnext_base", "drop_path_rate": 0.3,
                  "ema": True, "ema_decay": "auto"},
        "loss": {"loss": "cross_entropy", "label_smoothing": 0.1},
        "optimization": {"lr": 2e-4, "weight_decay": 0.05, "layer_lr_decay": 0.8},
        "schedule": {"epochs": 120, "batch_size": 32, "warmup_epochs": 5},
        "validation": {"metrics": ["acc@1", "acc@5", "macro-F1", "ece"], "tta": "hflip"},
        "checkpoint": {"early_stopping": False, "save_top_k": 3},
    },
    Preset.SMALL_DATASET: {
        "_description": "Heavy regularization for a few hundred images per class: "
                        "tuning-free augmentation, aggressive LLRD, tiny head init.",
        "input": {"input_size": 224, "rrc_scale": (0.7, 1.0)},
        # Heavy strength, but with the policy that needs no tuning -- see the module note.
        "augmentation": {"preset": "heavy", "auto_augment": "trivialaugment-wide"},
        "model": {"backbone": "convnext_tiny", "drop_path_rate": 0.2,
                  "head_init_scale": 0.001, "ema": True, "ema_decay": "auto"},
        "loss": {"loss": "cross_entropy", "label_smoothing": 0.1},
        "optimization": {"lr": 1e-4, "weight_decay": 0.1, "layer_lr_decay": 0.65},
        "schedule": {"epochs": 40, "batch_size": 32, "warmup_epochs": 3},
        "validation": {"metrics": ["acc@1", "macro-F1"], "tta": "hflip"},
        "checkpoint": {"early_stopping": True, "es_patience": 15},
        "data": {"val_split": 0.2},
    },
    Preset.IMBALANCED: {
        "_description": "Logit adjustment plus macro-F1 as the objective. Deliberately "
                        "uses plain random sampling -- decoupled-training work found "
                        "instance-balanced sampling learns the best representations.",
        "input": {"input_size": 224, "rrc_scale": (0.65, 1.0)},
        "augmentation": {"preset": "medium"},
        "model": {"backbone": "convnext_tiny", "drop_path_rate": 0.1,
                  "ema": True, "ema_decay": "auto"},
        "loss": {"loss": "logit_adjusted", "logit_adjust_tau": 1.0, "label_smoothing": 0.1,
                 "class_weights": "none"},
        "optimization": {"lr": 3e-4, "weight_decay": 0.05, "layer_lr_decay": 0.75},
        "schedule": {"epochs": 40, "batch_size": 64, "warmup_epochs": 3},
        "validation": {"metrics": ["acc@1", "macro-F1", "balanced-accuracy",
                                   "per-class-recall"],
                       "primary_metric": "macro-F1", "tta": "hflip"},
        "checkpoint": {"early_stopping": True, "es_patience": 15},
        "data": {"sampler": "random"},
    },
}


def describe(p: Preset) -> str:
    return PRESETS.get(p, {}).get("_description", "")


def _overlay(d: dict[str, Any], preset: Preset) -> dict[str, Any]:
    """Write `preset`'s groups into a config dump, in place."""
    overlay = deepcopy(PRESETS.get(preset, {}))
    overlay.pop("_description", None)

    for group, fields in overlay.items():
        # The augmentation group is replaced, not merged: it is described by a rung of the
        # strength ladder, and merging would leave the base config's knobs sitting inside
        # a group that now claims to be `light`. AugmentationConfig fills the rest in.
        if group == "augmentation" and "preset" in fields:
            d[group] = dict(fields)
        else:
            d.setdefault(group, {}).update(fields)
    return d


def apply_preset(preset: Preset, base: TrainConfig | None = None) -> TrainConfig:
    """Overlay a preset onto `base` (or schema defaults), preserving data paths.

    The result is tagged with the preset -- but only if it still *is* the preset. Overlaid
    onto a `base` carrying settings the preset does not name, `TrainConfig` demotes the
    tag to `custom`, because the config that comes out is no longer the one the preset
    describes. Paths and tracker settings are exempt (`PRESET_IDENTITY_FIELDS`).
    """
    d = _overlay((base or TrainConfig()).model_dump(mode="json"), preset)
    d.setdefault("tracking", {})["preset"] = preset.value
    return TrainConfig.model_validate(d)


# --------------------------------------------------------------------------------------
# Keeping `tracking.preset` honest
# --------------------------------------------------------------------------------------
#
# The tag is what `naming.run_tags` puts on the run and what the comparison view filters
# and groups by, so it has to describe the config that ran. A preset names a complete
# config, so "does this label still hold?" is a comparison against that config -- with
# two kinds of field left out.

#: Fields a preset does not describe, and which therefore must never demote its label:
#:
#:   * where things live -- dataset directories, output directory. Two people running
#:     `balanced` on their own data are both running `balanced`.
#:   * what the app fills in for you -- `num_classes`/`class_names` are auto-detected
#:     from the dataset and written back into the config by the form.
#:   * the whole `tracking` group -- it says where a run is recorded, not how it trains.
#:     That includes the label itself, which is the thing being derived.
#:
#: Mirrored client-side in frontend/src/lib/preset.ts, which uses the same list to decide
#: what a preset click preserves; keep the two in step.
PRESET_IDENTITY_FIELDS = (
    "schema_version",
    "data.train_dir", "data.val_dir", "data.num_classes", "data.class_names",
    "checkpoint.output_dir", "checkpoint.resume_from",
    "tracking",
)


def recipe(config: dict[str, Any]) -> dict[str, Any]:
    """A config dump reduced to the part a preset actually describes."""
    out = deepcopy(config)
    for path in PRESET_IDENTITY_FIELDS:
        *parents, leaf = path.split(".")
        node: Any = out
        for part in parents:
            node = node.get(part) if isinstance(node, dict) else None
        if isinstance(node, dict):
            node.pop(leaf, None)
    return out


@lru_cache(maxsize=None)
def _preset_recipe(preset: Preset) -> dict[str, Any]:
    """The comparable config `preset` stands for.

    Built with the label neutralised to `custom` on purpose: expanding a preset has to
    validate a `TrainConfig`, and a labelled one would re-enter the very check this table
    exists to answer. Cached because it is recomputed on every validation otherwise;
    treat the result as read-only.
    """
    d = _overlay(TrainConfig().model_dump(mode="json"), preset)
    d.setdefault("tracking", {})["preset"] = Preset.CUSTOM.value
    return recipe(TrainConfig.model_validate(d).model_dump(mode="json"))


def describes(preset: Preset, config: dict[str, Any]) -> bool:
    """True when `config` is still the config `preset` names."""
    if preset not in PRESETS:
        return False        # `custom` names no config, so it describes nothing
    return recipe(config) == _preset_recipe(preset)
