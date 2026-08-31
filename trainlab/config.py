"""
Single source of truth for the training configuration.

Everything downstream is derived from this file:

    TrainConfig  ──┬──> JSON Schema + UI metadata  ──> React form (auto-generated)
                   ├──> YAML round-trip            ──> `python train.py --config run.yaml`
                   ├──> flattened dict             ──> MLflow params
                   └──> validation warnings        ──> UI warning panel + CLI stderr

Adding a hyperparameter is a ONE-LINE change here. The form, the YAML, the tracker
params and the sweep search space all pick it up automatically.

Design notes
------------
* Every field carries `ui(...)` metadata: which accordion group it belongs to,
  whether it is "advanced", its tooltip, and when it should be shown/disabled.
* `Field(default=...)` values are the *recommended* values (see RESEARCH.md), never
  neutral zeros. A user who changes nothing and clicks Train gets a solid result.
* Validation is split in two:
    - `@model_validator` raises on configs that CANNOT run (hard errors).
    - `warnings()` returns non-fatal advice (soft warnings). The UI mirrors this
      logic client-side for instant feedback; this is the authoritative copy.
* Nothing here imports torch. The schema must stay importable in a bare venv so the
  API can serve the form without a training stack present.
"""

from __future__ import annotations

import copy
import math
import os
import re
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1.0"

#: Where the tracker is, when nothing else says. `make api` sets TRAINLAB_TRACKING_URI
#: from the same MLFLOW_HOST/MLFLOW_PORT that start the server, so moving the tracker does
#: not mean retyping its address in the form every session. The field still wins per run
#: (the form, a YAML file, `train.py --tracking-uri`); this only decides where it starts.
DEFAULT_TRACKING_URI = "http://127.0.0.1:5050"


# --------------------------------------------------------------------------------------
# UI metadata helper
# --------------------------------------------------------------------------------------

class Group(str, Enum):
    """Accordion groups, in display order."""
    DATA = "data"
    INPUT = "input"
    AUGMENTATION = "augmentation"
    MODEL = "model"
    LOSS = "loss"
    OPTIMIZATION = "optimization"
    SCHEDULE = "schedule"
    VALIDATION = "validation"
    CHECKPOINT = "checkpoint"
    TRACKING = "tracking"
    EXPERIMENTAL = "experimental"


#: Groups expanded on first load. Everything else starts collapsed.
DEFAULT_EXPANDED = [Group.DATA, Group.MODEL, Group.OPTIMIZATION]


def ui(
    *,
    label: str,
    tooltip: str,
    advanced: bool = False,
    widget: str | None = None,
    step: float | None = None,
    unit: str | None = None,
    show_if: str | None = None,
    disable_if: str | None = None,
    summary_priority: int = 0,
) -> dict[str, Any]:
    """Attach UI metadata to a field.

    Args:
        tooltip: MUST say both what it does *and when to change it*.
        advanced: hidden behind the group's "Advanced" sub-toggle.
        show_if / disable_if: JS-evaluable expression over the flat config, e.g.
            "optimizer == 'sgd'". Evaluated by a tiny safe interpreter, not eval().
        summary_priority: higher values appear first in the collapsed-header summary
            of non-default values. 0 means "only show if nothing better is modified".
    """
    return {
        "json_schema_extra": {
            "x-ui": {
                "label": label,
                "tooltip": tooltip,
                "advanced": advanced,
                "widget": widget,
                "step": step,
                "unit": unit,
                "showIf": show_if,
                "disableIf": disable_if,
                "summaryPriority": summary_priority,
            }
        }
    }


class Severity(str, Enum):
    ERROR = "error"      # config cannot run
    WARNING = "warning"  # config runs but is probably wrong
    INFO = "info"        # FYI / auto-applied transformation


class Warning_(BaseModel):
    """A single validation finding surfaced in the UI warning panel."""
    severity: Severity
    field: str | None = None
    message: str
    fix: str | None = None


# --------------------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------------------

class DatasetFormat(str, Enum):
    IMAGEFOLDER = "imagefolder"
    CSV = "csv"
    HUGGINGFACE = "huggingface"


class Sampler(str, Enum):
    RANDOM = "random"
    WEIGHTED = "weighted"
    BALANCED_BATCH = "balanced-batch"


class CacheMode(str, Enum):
    NONE = "none"
    RAM = "ram"
    DISK_DECODED = "disk-decoded"


class TrainResizePolicy(str, Enum):
    RRC = "random-resized-crop"
    SHORTEST_THEN_CROP = "resize-shortest-then-random-crop"
    SQUASH = "resize-squash"
    PAD_SQUARE = "resize-pad-to-square"


class Interpolation(str, Enum):
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    LANCZOS = "lanczos"


class Normalization(str, Enum):
    IMAGENET = "imagenet"
    DATASET = "dataset-computed"
    CUSTOM = "custom"


class AugPreset(str, Enum):
    NONE = "none"
    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"
    CUSTOM = "custom"


class AutoAugment(str, Enum):
    NONE = "none"
    RANDAUGMENT = "randaugment"
    TRIVIALAUGMENT_WIDE = "trivialaugment-wide"   # torchvision v2 (timm has no impl)
    AUTOAUGMENT_IMAGENET = "autoaugment-imagenet"
    THREE_AUGMENT = "3a"                          # DeiT III — timm 'auto_augment=3a'


class FreezePolicy(str, Enum):
    NONE = "none"
    BACKBONE_FIRST_N = "backbone-only-first-N-epochs"
    FIRST_K_STAGES = "freeze-first-K-stages"


class LossName(str, Enum):
    CROSS_ENTROPY = "cross_entropy"
    SOFT_TARGET_CE = "soft_target_ce"
    BCE = "bce"
    FOCAL = "focal"
    CLASS_BALANCED_CE = "class_balanced_ce"
    LOGIT_ADJUSTED = "logit_adjusted"


#: Losses that can consume the soft targets produced by mixup/cutmix.
SOFT_TARGET_CAPABLE = {LossName.SOFT_TARGET_CE, LossName.BCE}


class ClassWeights(str, Enum):
    NONE = "none"
    INVERSE_FREQUENCY = "inverse-frequency"
    EFFECTIVE_NUMBER = "effective-number"
    MANUAL = "manual"


class Optimizer(str, Enum):
    ADAMW = "adamw"
    SGD = "sgd"
    LION = "lion"
    RMSPROP = "rmsprop"


class Scheduler(str, Enum):
    COSINE = "cosine"
    ONE_CYCLE = "one_cycle"
    STEP = "step"
    PLATEAU = "plateau"
    CONSTANT = "constant"


class Amp(str, Enum):
    OFF = "off"
    FP16 = "fp16"
    BF16 = "bf16"


class Device(str, Enum):
    AUTO = "auto"
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


class Metric(str, Enum):
    ACC1 = "acc@1"
    ACC5 = "acc@5"
    MACRO_F1 = "macro-F1"
    BALANCED_ACC = "balanced-accuracy"
    PER_CLASS_RECALL = "per-class-recall"
    AUROC = "auroc"
    MCC = "mcc"
    ECE = "ece"
    #: A family rather than a single number: one FPR per FNR target in
    #: `ValidationConfig.fpr_at_fnr_targets`, keyed by `fpr_at_fnr_key`.
    FPR_AT_FNR = "fpr@fnr"

    @property
    def higher_is_better(self) -> bool:
        """Optimisation direction.

        Load-bearing, not cosmetic: checkpoint selection, early stopping, top-K
        retention, the leaderboard star and the sweep objective all compare scores,
        and every one of them was previously hardcoded to maximise. Calibration error
        and `fpr@fnr` are the metrics in this enum where that is backwards — selecting
        `ece` as the primary metric used to save the WORST-calibrated epoch as
        `best.ckpt`.
        """
        return self not in (Metric.ECE, Metric.FPR_AT_FNR)


#: Direction for metric keys that never appear in the `Metric` enum but are surfaced
#: for comparison and sweeps (losses, timings). Anything unlisted is assumed
#: higher-is-better, which matches every metric this app computes.
LOWER_IS_BETTER_KEYS = frozenset({
    "val_loss", "train_loss", "ece", "epoch_time_s", "peak_vram_gb",
    # Loop timings. `imgs_per_s` and `compute_share` are deliberately absent: more
    # throughput and a busier GPU are better, so the default direction already fits.
    "data_wait_s", "compute_s", "val_time_s", "ms_per_step",
})


#: Prefix of every key in the `fpr@fnr` family, in both spellings the app uses: raw, and
#: sanitised by the tracker (`@` -> `_at_`). Membership cannot be a set lookup like the
#: keys above, because the FNR target is part of the name.
_FPR_AT_FNR_PREFIXES = ("fpr@fnr", "fpr_at_fnr")


def fpr_at_fnr_key(target_fnr: float) -> str:
    """Metric key for the FPR measured at a given FNR target.

    `0.01` becomes `fpr@fnr0_01`. The decimal point is an underscore because these keys
    become `MetricCollection` entries, which torch registers as submodule names, and a
    submodule name may not contain a dot. `:g` keeps the rest short (no trailing zeros,
    and 1e-4 stays 0_0001), and the `@` follows the convention `acc@1` already sets — the
    tracker rewrites it to `_at_` the same way.
    """
    return f"fpr@fnr{target_fnr:g}".replace(".", "_")


def metric_higher_is_better(key: str) -> bool:
    """Direction for an arbitrary logged metric key.

    Accepts raw keys (`acc@1`), tracker-sanitised keys (`acc_at_1`) and stage-prefixed
    keys (`ema/val_loss`, `crt/acc@1`), so a single rule serves the engine, the sweep
    driver and the comparison view instead of three divergent ones.
    """
    k = key.rsplit("/", 1)[-1]
    if k.startswith(_FPR_AT_FNR_PREFIXES):
        return False           # a false-positive rate: smaller is better at every target
    return k.replace("_at_", "@") not in LOWER_IS_BETTER_KEYS


class TTA(str, Enum):
    NONE = "none"
    HFLIP = "hflip"
    MULTI_CROP = "multi-crop"
    MULTI_SCALE = "multi-scale"


class SSLMethod(str, Enum):
    NONE = "none"
    SIMSIAM = "simsiam"   # negative-free siamese; any backbone
    MAE = "mae"           # asymmetric masked autoencoder; plain ViT only
    SIMMIM = "simmim"     # full-token masked modelling; ViT and Swin


#: Backbone prefixes that build a plain timm `VisionTransformer` — what MAE requires,
#: because it drops masked tokens and therefore needs a flat token sequence.
#:
#: This list must agree with the `isinstance` checks in `trainlab/ssl.py`, which are the
#: real gate. It previously advertised `beit_`, `beitv2_`, `eva_`, `eva02_` and
#: `samvit_`, none of which subclass `VisionTransformer` in timm — so the form showed a
#: clean validation panel and the run then died at model construction. `vit3d_` is not a
#: timm prefix at all. Anything added here needs a matching isinstance branch in ssl.py;
#: `test_ssl_prefix_lists_match_runtime_support` fails the build if the two drift again.
VIT_LIKE_PREFIXES = ("vit_", "deit_", "deit3_", "flexivit_")

#: SimMIM keeps every token, so it also works on hierarchical Swin encoders. `swinv2_`
#: builds `SwinTransformerV2`/`SwinTransformerV2Cr`, which do not share
#: `SwinTransformer`'s patch_embed→layers→norm NHWC structure and are not supported.
SIMMIM_LIKE_PREFIXES = VIT_LIKE_PREFIXES + ("swin_",)

#: Old field names -> new, for controls that were RENAMED and kept their meaning.
#:
#: Every rename in `experimental` must be listed here. With `extra="forbid"`, a rename
#: with no entry makes every run recorded before it permanently un-clonable.
_LEGACY_SSL_FIELDS = {
    "mim_mask_ratio": "mae_mask_ratio",
    "mim_decoder_dim": "mae_decoder_dim",
    "mim_decoder_depth": "mae_decoder_depth",
    "mim_decoder_heads": "mae_decoder_heads",
    "mim_norm_pix_loss": "mae_norm_pix_loss",
}

#: Controls that were REMOVED because the technique they described was redesigned, not
#: renamed. Their values have no equivalent in the current schema, so they are dropped
#: on load rather than aliased onto a field that means something else.
#:
#: `ssl_warmup_epochs` conflated "warm up the SSL learning rate" with "pretrain on a
#: large corpus" (see RESEARCH.md §12); it was replaced by `ssl_method` plus an explicit
#: budget. Aliasing it onto `ssl_lr_warmup_epochs` would silently fabricate a schedule
#: the original run never had.
#: `distributed`/`num_gpus` were declared controls with NO runtime behind them — no DDP
#: wrapper, no process spawn, nothing. Worse than dead: `effective_batch_size`
#: multiplied by `num_gpus`, so enabling them silently distorted the EMA-horizon
#: warning, the LR suggestion and the displayed effective batch while training ran on a
#: single device. Dropped rather than aliased; there is nothing to alias them to.
_LEGACY_DROPPED_FIELDS = {
    "experimental": ("ssl_warmup_epochs",),
    "schedule": ("distributed", "num_gpus"),
}


#: Names the form injects into the expression scope that are not config fields.
UI_SCOPE_EXTRAS = frozenset({"resolved_device"})

_EXPR_STRING = re.compile(r"'[^']*'|\"[^\"]*\"")
_EXPR_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")
_EXPR_KEYWORDS = frozenset({"in", "and", "or", "not", "true", "false", "null", "None"})


def ui_expression_identifiers(expr: str) -> set[str]:
    """Field names an ``x-ui`` expression reads (string literals excluded)."""
    return {
        tok for tok in _EXPR_IDENT.findall(_EXPR_STRING.sub(" ", expr))
        if tok not in _EXPR_KEYWORDS
    }


def validate_ui_expressions() -> list[str]:
    """Check every showIf/disableIf against the fields that actually exist.

    These expressions are hand-written strings resolved against a flat namespace at
    render time. A field renamed without updating them leaves the dependent control
    permanently hidden in the form, with nothing anywhere to indicate why — the one
    place where "the schema generates everything" stops being true. Returns a list of
    problems; empty means consistent.
    """
    leaves: dict[str, list[str]] = {}
    for section, field in TrainConfig.model_fields.items():
        ann = field.annotation
        sub = getattr(ann, "model_fields", None)
        if not sub:
            continue
        for name in sub:
            leaves.setdefault(name, []).append(section)

    problems: list[str] = []
    for section, field in TrainConfig.model_fields.items():
        sub = getattr(field.annotation, "model_fields", None)
        if not sub:
            continue
        for name, f in sub.items():
            meta = ((f.json_schema_extra or {}).get("x-ui") or {})
            for kind in ("showIf", "disableIf"):
                expr = meta.get(kind)
                if not expr:
                    continue
                for ident in sorted(ui_expression_identifiers(expr)):
                    if ident in UI_SCOPE_EXTRAS:
                        continue
                    if ident not in leaves:
                        problems.append(
                            f"{section}.{name}: {kind}={expr!r} references unknown field "
                            f"'{ident}'"
                        )
                    elif len(leaves[ident]) > 1:
                        problems.append(
                            f"{section}.{name}: {kind}={expr!r} references '{ident}', which "
                            f"is ambiguous across {leaves[ident]}"
                        )
    return problems


def load_config_leniently(raw: Any) -> tuple["TrainConfig", list[str]]:
    """Validate a stored config, dropping keys the current schema no longer knows.

    Returns (config, dropped_field_paths). Use this for replaying recorded history
    (cloning a tracked run, opening an old YAML); use plain `model_validate` for input a
    user just typed, where an unknown key is a typo worth reporting.

    The strict path is still what decides validity — this only removes `extra_forbidden`
    keys, then revalidates. Any other error propagates unchanged.
    """
    from pydantic import ValidationError

    dropped: list[str] = []
    data = raw
    for _ in range(20):                      # bounded: each pass removes ≥1 key
        try:
            return TrainConfig.model_validate(data), dropped
        except ValidationError as e:
            extras = [err["loc"] for err in e.errors()
                      if err["type"] == "extra_forbidden"]
            if not extras:
                raise
            data = copy.deepcopy(data)
            for loc in extras:
                node = data
                for part in loc[:-1]:
                    node = node[part]
                node.pop(loc[-1], None)
                dropped.append(".".join(str(p) for p in loc))
    return TrainConfig.model_validate(data), dropped


class Preset(str, Enum):
    FAST_BASELINE = "fast-baseline"
    BALANCED = "balanced"
    MAX_ACCURACY = "max-accuracy"
    SMALL_DATASET = "small-dataset"
    IMBALANCED = "imbalanced"
    CUSTOM = "custom"


# --------------------------------------------------------------------------------------
# Groups
# --------------------------------------------------------------------------------------

class _G(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, use_enum_values=False)


class DataConfig(_G):
    dataset_format: DatasetFormat = Field(
        DatasetFormat.IMAGEFOLDER,
        **ui(label="Dataset format", summary_priority=3,
             tooltip="ImageFolder: one subdirectory per class. CSV: a manifest with "
                     "path,label columns — use it when images live in a flat directory "
                     "or a file belongs to several splits."))
    train_dir: str = Field(
        "", **ui(label="Train directory", widget="path", summary_priority=5,
                 tooltip="Root of the training set. For ImageFolder, its immediate "
                         "subdirectories become the class list."))
    val_dir: str | None = Field(
        None, **ui(label="Val directory", widget="path",
                   tooltip="Leave empty to carve a validation split out of the training "
                           "set using val_split. Always prefer a real held-out directory "
                           "if you have one — a random split leaks near-duplicates."))
    val_split: float = Field(
        0.1, ge=0.0, lt=1.0,
        **ui(label="Val split", step=0.05,
             tooltip="Fraction held out for validation when no val directory is given. "
                     "Raise to 0.2 for datasets under ~2k images so the metric is less noisy."))
    split_strategy: Literal["stratified", "random", "grouped"] = Field(
        "stratified",
        **ui(label="Split strategy", advanced=True,
             tooltip="Stratified keeps class proportions identical across splits — "
                     "always what you want unless you have groups (e.g. multiple crops "
                     "of the same subject) that must not straddle the split."))
    group_column: str | None = Field(
        None, **ui(label="Group column", advanced=True, show_if="split_strategy == 'grouped'",
                   tooltip="CSV column identifying the group (patient, session, source "
                           "image). Rows sharing a value never land in different splits."))
    num_classes: int | None = Field(
        None, **ui(label="Num classes", widget="readonly", summary_priority=1,
                   tooltip="Auto-detected from the dataset. Read-only."))
    class_names: list[str] = Field(
        default_factory=list,
        **ui(label="Class names", widget="readonly",
             tooltip="Auto-detected from the dataset and sorted alphabetically. Read-only. "
                     "This order defines the class indices, so manual class weights and "
                     "per-class metrics follow it — check it if those look mislabelled."))

    sampler: Sampler = Field(
        Sampler.RANDOM,
        **ui(label="Sampler", summary_priority=4,
             tooltip="random: standard. weighted: oversample rare classes. balanced-batch: "
                     "every batch has equal class counts. Note that decoupled-training work "
                     "(Kang 2020) found plain random sampling learns the BEST representations "
                     "— fix imbalance in the loss or logits instead. See RESEARCH.md §7."))
    oversample_factor: float = Field(
        1.0, ge=0.0, le=10.0,
        **ui(label="Oversample factor", advanced=True, step=0.1,
             show_if="sampler != 'random'",
             tooltip="Interpolates between no rebalancing (0) and full inverse-frequency "
                     "rebalancing (1). Values above 1 over-correct toward rare classes."))

    num_workers: int = Field(
        8, ge=0, le=64,
        **ui(label="Num workers", advanced=True,
             tooltip="DataLoader processes. Rule of thumb: number of physical cores. If GPU "
                     "utilization is spiky, the dataloader — not the model — is your bottleneck."))
    pin_memory: bool = Field(
        True, **ui(label="Pin memory", advanced=True,
                   tooltip="Faster host→device copies on CUDA. No effect on MPS/CPU."))
    persistent_workers: bool = Field(
        True, **ui(label="Persistent workers", advanced=True,
                   tooltip="Keeps workers alive between epochs. Meaningful win for short "
                           "epochs, where worker respawn can dominate."))
    prefetch_factor: int = Field(
        2, ge=1, le=16,
        **ui(label="Prefetch factor", advanced=True,
             tooltip="Batches preloaded per worker. Raise if input is I/O-bound; costs RAM."))
    broken_image_tolerance: float = Field(
        0.02, ge=0.0, le=1.0,
        **ui(label="Broken image tolerance", advanced=True, step=0.01,
             tooltip="Unreadable images (truncated files, a path that no longer resolves) "
                     "are skipped and listed in broken_images.tsv rather than ending the "
                     "run. Above this fraction of the dataset the run stops instead: past "
                     "a point it is not a few bad files, it is the wrong directory or an "
                     "unmounted share, and training on what is left is not what you asked "
                     "for. 0 disables the check."))
    cache_mode: CacheMode = Field(
        CacheMode.NONE,
        **ui(label="Cache mode", advanced=True,
             tooltip="RAM caches decoded tensors — a large win when the dataset fits in "
                     "memory, since JPEG decode is pure waste after epoch 1. "
                     "disk-decoded writes a decoded cache for datasets too big for RAM."))


class InputConfig(_G):
    input_size: int = Field(
        224, ge=32, le=1024,
        **ui(label="Input size", unit="px", summary_priority=5,
             tooltip="Train-time square resolution. Bigger is usually more accurate and "
                     "quadratically slower. Match the backbone's pretrained size unless "
                     "you have a reason not to."))
    train_resize_policy: TrainResizePolicy = Field(
        TrainResizePolicy.RRC,
        **ui(label="Train resize policy", summary_priority=2,
             tooltip="RandomResizedCrop is the standard and doubles as strong augmentation. "
                     "Use resize-pad-to-square when aspect ratio carries information "
                     "(documents, X-rays); use squash only if objects are always centered "
                     "and framing is consistent."))
    rrc_scale: tuple[float, float] = Field(
        (0.08, 1.0),
        **ui(label="RRC scale", widget="range2", step=0.01, summary_priority=3,
             show_if="train_resize_policy == 'random-resized-crop'",
             tooltip="Area fraction range for the random crop. 0.08 is the ImageNet "
                     "from-scratch default and is FAR too aggressive for small or "
                     "fine-grained datasets — the crop often contains no evidence of the "
                     "class at all. Use 0.65–1.0 there, and check the augmentation preview."))
    rrc_ratio: tuple[float, float] = Field(
        (3 / 4, 4 / 3),
        **ui(label="RRC aspect ratio", widget="range2", advanced=True, step=0.01,
             show_if="train_resize_policy == 'random-resized-crop'",
             tooltip="Aspect ratio range of the crop. Narrow it toward (1.0, 1.0) when "
                     "distortion destroys the signal."))
    val_resize_policy: Literal["resize-shortest-center-crop", "resize-squash", "resize-pad-to-square"] = Field(
        "resize-shortest-center-crop",
        **ui(label="Val resize policy", advanced=True,
             tooltip="Should mirror how the model will see data in production. Mismatch "
                     "between this and the train policy is a common silent accuracy leak."))
    val_crop_pct: float = Field(
        0.875, gt=0.0, le=1.0,
        **ui(label="Val crop %", advanced=True, step=0.005,
             tooltip="Resize to input_size/crop_pct then center-crop. 0.875 is the torchvision "
                     "convention; ResNet-Strikes-Back found 0.95 better, and modern "
                     "ConvNeXt/ViT weights often prefer 0.95–1.0. Cheap thing to try."))
    test_input_size: int | None = Field(
        None, ge=32, le=1024,
        **ui(label="Test input size", unit="px", summary_priority=4,
             tooltip="Defaults to input_size. RandomResizedCrop makes objects look LARGER "
                     "in training than at test time; testing ~1.15x higher than you trained "
                     "usually recovers ~0.5–1 point for zero training cost (FixRes, 2019). "
                     "One of the best free wins available."))
    interpolation: Interpolation = Field(
        Interpolation.BICUBIC,
        **ui(label="Interpolation", advanced=True,
             tooltip="Match the backbone's pretraining. timm weights are overwhelmingly "
                     "bicubic; using bilinear costs a small but measurable amount."))
    antialias: bool = Field(
        True, **ui(label="Antialias", advanced=True,
                   tooltip="Antialiased downsampling. Leave on — off reproduces a "
                           "well-known torchvision-vs-PIL train/test skew."))
    normalization: Normalization = Field(
        Normalization.IMAGENET,
        **ui(label="Normalization", advanced=True,
             tooltip="Use ImageNet stats when using pretrained weights, even if your domain "
                     "differs — the weights expect them. dataset-computed only makes sense "
                     "when training from scratch."))
    custom_mean: tuple[float, float, float] | None = Field(
        None, **ui(label="Custom mean", advanced=True, show_if="normalization == 'custom'",
                   tooltip="Per-channel RGB mean in [0,1]. Only set this when you are "
                           "reproducing another codebase's exact preprocessing — with "
                           "pretrained weights the ImageNet statistics are what the "
                           "backbone expects, even for unusual domains."))
    custom_std: tuple[float, float, float] | None = Field(
        None, **ui(label="Custom std", advanced=True, show_if="normalization == 'custom'",
                   tooltip="Per-channel RGB standard deviation in [0,1]. Must come from the "
                           "same source as the custom mean — mixing the two is a silent "
                           "accuracy leak that the augmentation preview will reveal."))
    channels: Literal["RGB", "L"] = Field(
        "RGB", **ui(label="Channels", advanced=True,
                    tooltip="Grayscale ('L') is replicated to 3 channels for pretrained "
                            "backbones — it saves I/O, not compute."))
    channels_last: bool = Field(
        True, **ui(label="channels_last", advanced=True,
                   tooltip="NHWC memory format. Real throughput win for convnets on CUDA "
                           "tensor cores; little effect for ViTs, and no meaningful effect "
                           "on MPS (auto-disabled there)."))


# --------------------------------------------------------------------------------------
# Augmentation strength ladder
# --------------------------------------------------------------------------------------
#
# `augmentation.preset` is not a label, it IS the group: naming a rung sets every control
# the group contains. Each table below is therefore complete rather than a delta — a
# partial one would leave whatever the previous rung set behind, and "medium" would mean
# something different depending on what you clicked before it.
#
# Two deliberate omissions from the ladder:
#   * vflip / rotation / grayscale / blur stay at 0 on every rung. Each encodes a claim
#     about the domain (there is no canonical "up"; orientation is not part of the label),
#     not a strength dial, so a strength preset must not switch them on behind the user's
#     back. The `heavy` rung turns the *tuned* knobs up instead.
#   * No rung enables mixup/cutmix, for the reason in `presets.py`: the evidence is from
#     300-600 epoch from-scratch training and it is measurably negative on a short
#     pretrained fine-tune. Mixing stays opt-in at every strength.
#
# MEDIUM must stay equal to this group's schema defaults, because the default `preset` is
# MEDIUM and an untouched config must not contradict its own label.
# `test_aug_preset_medium_matches_schema_defaults` pins that.

_AUG_SHARED: dict[str, Any] = {
    "vflip": 0.0,
    "rotation_deg": 0.0,
    "grayscale_p": 0.0,
    "blur_p": 0.0,
    "random_erasing_mode": "pixel",
    "mixup_alpha": 0.0,
    "cutmix_alpha": 0.0,
    "mixup_prob": 1.0,
    "mixup_switch_prob": 0.5,
    "mixup_off_epoch": 0,
    # Inert unless auto_augment == 'randaugment', and identical across rungs: switching a
    # policy back on should not reveal a magnitude some earlier rung quietly rewrote.
    "randaugment_n": 2,
    "randaugment_mstd": 0.5,
}

#: Concrete values behind each named rung of `AugmentationConfig.preset`.
AUG_PRESETS: dict[AugPreset, dict[str, Any]] = {
    # Resize/crop only — the crop itself lives in the Input group. The honest baseline for
    # measuring what augmentation is worth on your data.
    AugPreset.NONE: {
        **_AUG_SHARED, "hflip": 0.0, "color_jitter": 0.0,
        "auto_augment": AutoAugment.NONE.value, "randaugment_m": 7,
        "random_erasing_p": 0.0,
    },
    # Horizontal flip only: free, near-universal, and enough for a 3-epoch sanity run.
    AugPreset.LIGHT: {
        **_AUG_SHARED, "hflip": 0.5, "color_jitter": 0.0,
        "auto_augment": AutoAugment.NONE.value, "randaugment_m": 7,
        "random_erasing_p": 0.0,
    },
    # The fine-tuning default: RandAugment at the magnitude that suits a short schedule
    # (5-7, per the randaugment_m tooltip) plus light erasing. `color_jitter` is inert
    # while an auto-augment policy is active — it is carried so that turning the policy
    # off leaves a sensible jitter rather than nothing.
    AugPreset.MEDIUM: {
        **_AUG_SHARED, "hflip": 0.5, "color_jitter": 0.4,
        "auto_augment": AutoAugment.RANDAUGMENT.value, "randaugment_m": 7,
        "random_erasing_p": 0.1,
    },
    # For 100+ epoch schedules, where regularization has time to pay off: the ImageNet
    # standard magnitude and the DeiT/ConvNeXt erasing probability.
    AugPreset.HEAVY: {
        **_AUG_SHARED, "hflip": 0.5, "color_jitter": 0.4,
        "auto_augment": AutoAugment.RANDAUGMENT.value, "randaugment_m": 9,
        "random_erasing_p": 0.25,
    },
}


def _aug_value_eq(a: Any, b: Any) -> bool:
    """Compare a supplied augmentation value against a preset's.

    Tolerant on purpose: the same value arrives as an enum member from Python, as a
    string from YAML and as an int from JSON, and all three must count as "unchanged"
    or every config would report itself as `custom`.
    """
    a = a.value if isinstance(a, Enum) else a
    b = b.value if isinstance(b, Enum) else b
    if a is None or b is None or isinstance(a, bool) or isinstance(b, bool):
        return a == b
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    except (TypeError, ValueError):
        return a == b


def with_aug_preset(aug: dict[str, Any], rung: Any) -> dict[str, Any]:
    """An augmentation group with `rung`'s values written into it.

    For callers that hold an already-complete group and are acting on "the user picked a
    rung": `--set augmentation.preset=heavy`, a sweep over `augmentation.preset`, the
    form's own select. `AugmentationConfig` deliberately lets explicit values win over the
    rung — that is what keeps `preset: heavy` plus `vflip: 0.5` in a YAML file — and in a
    complete group *every* value is explicit, so validation alone would relabel and change
    nothing. Overrides applied after this call still win, so `--set` stays last-wins.

    An unknown rung is passed through untouched for field validation to report.
    """
    named = rung.value if isinstance(rung, AugPreset) else rung
    try:
        table = AUG_PRESETS.get(AugPreset(named))
    except ValueError:
        table = None
    return {**aug, **(table or {}), "preset": rung}


class AugmentationConfig(_G):
    preset: AugPreset = Field(
        AugPreset.MEDIUM,
        **ui(label="Augmentation preset", summary_priority=6,
             tooltip="Sets the whole group at once: none (resize/crop only), light "
                     "(flip only), medium (RandAugment 7 + light erasing), heavy "
                     "(RandAugment 9 + more erasing). Switches to 'custom' as soon as "
                     "you touch any individual control below. No preset enables "
                     "mixup/cutmix — those stay opt-in."))
    hflip: float = Field(
        0.5, ge=0.0, le=1.0,
        **ui(label="Horizontal flip p", step=0.05, summary_priority=1,
             tooltip="Free and near-universal. Set to 0 only when left/right is semantic "
                     "(text, road signs, chirality)."))
    vflip: float = Field(
        0.0, ge=0.0, le=1.0,
        **ui(label="Vertical flip p", step=0.05, summary_priority=2,
             tooltip="Off by default — wrong for natural photos. Turn on for overhead, "
                     "microscopy, or satellite imagery where there is no canonical up."))
    rotation_deg: float = Field(
        0.0, ge=0.0, le=180.0,
        **ui(label="Rotation", unit="deg", step=5, summary_priority=2,
             tooltip="Random rotation range. Useful for rotation-invariant domains; "
                     "harmful where orientation distinguishes classes (6 vs 9)."))
    color_jitter: float = Field(
        0.4, ge=0.0, le=1.0,
        **ui(label="Color jitter", step=0.05,
             tooltip="Brightness/contrast/saturation strength. Ignored when an "
                     "auto-augment policy is active, which already includes color ops. "
                     "Reduce when color IS the class signal."))
    auto_augment: AutoAugment = Field(
        AutoAugment.RANDAUGMENT,
        **ui(label="Auto-augment policy", summary_priority=8,
             tooltip="randaugment: the tuned workhorse. trivialaugment-wide: matches or "
                     "beats it with ZERO hyperparameters — the best choice if you don't "
                     "want to tune. 3a: DeiT III's 3-Augment, best for ViTs and small data."))
    randaugment_n: int = Field(
        2, ge=1, le=5,
        **ui(label="RandAugment N", show_if="auto_augment == 'randaugment'",
             tooltip="Ops applied per image. 2 is standard; rarely worth changing."))
    # 7, not the literature's 9: the default schedule is a 30-epoch pretrained fine-tune,
    # which is exactly the case the tooltip says to drop to 5-7 for. 9 is one rung up
    # (`heavy`), where the schedule length that justifies it also lives.
    randaugment_m: int = Field(
        7, ge=0, le=30,
        **ui(label="RandAugment M", show_if="auto_augment == 'randaugment'", summary_priority=7,
             tooltip="Magnitude. 9 suits long from-scratch schedules; drop to 5–7 for short "
                     "fine-tunes. THE augmentation knob worth sweeping."))
    randaugment_mstd: float = Field(
        0.5, ge=0.0, le=2.0,
        **ui(label="RandAugment M std", advanced=True, step=0.1,
             show_if="auto_augment == 'randaugment'",
             tooltip="Jitters magnitude per-sample. 0.5 is timm's default and beats a "
                     "fixed magnitude."))
    # 0.1 for the same reason as randaugment_m: 0.25 is the long-schedule value, so it
    # belongs to `heavy` rather than to the default 30-epoch fine-tune.
    random_erasing_p: float = Field(
        0.1, ge=0.0, le=1.0,
        **ui(label="Random erasing p", step=0.05, summary_priority=3,
             tooltip="Occludes a random rectangle. 0.25 comes from the DeiT/ConvNeXt "
                     "lineage; ResNet-Strikes-Back used none and torchvision used 0.1. "
                     "Use 0–0.1 on short schedules."))
    random_erasing_mode: Literal["pixel", "const", "rand"] = Field(
        "pixel", **ui(label="Random erasing fill", advanced=True,
                      show_if="random_erasing_p > 0",
                      tooltip="'pixel' fills with per-pixel noise and beats a constant fill."))
    # Default OFF, matching every preset.
    #
    # These were 0.2/1.0 on the reasoning that they are the literature values. But the
    # UI boots into the `balanced` preset, so the raw defaults were only ever reached by
    # the headless path — the one entry point with no preset button and no warning panel.
    # `python train.py --train-dir X` therefore produced a config the tool immediately
    # warned was "measurably NEGATIVE", and silently swapped the loss function to match.
    # Literature values belong in the tooltip; the default should be the one that is
    # right for this app's default schedule. `max-accuracy` is where they turn on.
    mixup_alpha: float = Field(
        0.0, ge=0.0, le=2.0,
        **ui(label="Mixup alpha", step=0.05, summary_priority=9,
             tooltip="Beta(a,a) blend of image AND label pairs. Strongly positive on 300+ "
                     "epoch from-scratch runs (RSB uses 0.2); measurably NEGATIVE when "
                     "fine-tuning a pretrained backbone for <50 epochs (RESEARCH.md §0), "
                     "which is why it is off by default. Forces a soft-target-capable "
                     "loss. Try 0.2 with 100+ epochs."))
    cutmix_alpha: float = Field(
        0.0, ge=0.0, le=2.0,
        **ui(label="CutMix alpha", step=0.05, summary_priority=9,
             tooltip="Pastes a rectangular patch from another image; label weight is the "
                     "area ratio. Same schedule-length caveat as mixup, and off for the "
                     "same reason. RSB uses 1.0 on a 300–600 epoch schedule."))
    mixup_prob: float = Field(
        1.0, ge=0.0, le=1.0,
        **ui(label="Mixup/CutMix p", step=0.05,
             tooltip="Probability that a batch gets mixed at all. The cheapest way to soften "
                     "mixing without turning it off — try 0.5 on medium schedules."))
    mixup_switch_prob: float = Field(
        0.5, ge=0.0, le=1.0,
        **ui(label="Switch prob", advanced=True, step=0.05,
             tooltip="Given a batch is mixed, chance of CutMix over Mixup."))
    mixup_off_epoch: int = Field(
        0, ge=0,
        **ui(label="Mixup off epoch", advanced=True, summary_priority=4,
             tooltip="Disable mixing after this epoch (0 = never). Regularize early, then "
                     "let the model fit clean labels at the end. Genuinely useful and "
                     "widely underused."))
    grayscale_p: float = Field(
        0.0, ge=0.0, le=1.0,
        **ui(label="Grayscale p", advanced=True, step=0.05,
             tooltip="A 3-Augment component. Forces shape over color cues — helps when "
                     "color is a spurious correlate of the label."))
    blur_p: float = Field(
        0.0, ge=0.0, le=1.0,
        **ui(label="Gaussian blur p", advanced=True, step=0.05,
             tooltip="A 3-Augment component. Useful when deployment images are lower "
                     "quality than your training set."))

    @model_validator(mode="before")
    @classmethod
    def _expand_preset(cls, data: Any) -> Any:
        """Expand `preset` into the values it names, and keep the label honest.

        Every construction path goes through here — the form's POST, a hand-written YAML,
        `presets.apply_preset` — so `preset: heavy` means heavy augmentation everywhere
        instead of only in whichever caller remembered to expand it.

        Explicitly supplied values always win over the rung, and any value that
        contradicts it demotes the label to `custom`: a preset that no longer describes
        the group is worse than no preset at all, because it is what the run gets tagged
        and compared by.
        """
        if not isinstance(data, dict):
            return data

        raw = data.get("preset", AugPreset.MEDIUM)
        try:
            named = AugPreset(raw.value if isinstance(raw, AugPreset) else raw)
        except ValueError:
            return data          # unknown value: let field validation report it verbatim
        table = AUG_PRESETS.get(named)
        if table is None:
            return data          # 'custom' names no values, so there is nothing to expand

        contradicted = any(not _aug_value_eq(v, table[k])
                           for k, v in data.items() if k in table)
        return {**table, **data,
                "preset": AugPreset.CUSTOM if contradicted else named}


class ModelConfig(_G):
    backbone: str = Field(
        "convnext_tiny",
        **ui(label="Backbone", widget="timm-search", summary_priority=10,
             tooltip="Searchable over the full timm catalogue, with pretrained availability, "
                     "parameter count and native input size. The single biggest lever you "
                     "have — try a larger model before tuning anything else."))
    pretrained: bool = Field(
        True, **ui(label="Pretrained", summary_priority=6,
                   tooltip="Almost always keep this on. Worth +5 to +30 points on small "
                           "datasets. Turn off only to measure the value of pretraining."))
    pretrained_source: str = Field(
        "default", **ui(label="Pretrained tag", advanced=True, widget="timm-pretrained-tag",
                        show_if="pretrained == true",
                        tooltip="timm often ships several weight sets per architecture "
                                "(in1k, in22k-ft-in1k, augreg, ...). in22k-pretrained tags "
                                "are usually stronger for transfer."))
    pretrained_checkpoint: str | None = Field(
        None,
        **ui(label="Pretrained checkpoint", widget="path", advanced=True,
             tooltip="Initialise the backbone from a checkpoint instead of the pretrained "
                     "tag: a TrainLab .ckpt (or a previous run's directory — its best.ckpt "
                     "is used), or a raw state dict. Only the weights are taken; optimizer, "
                     "schedule and epoch state in a full checkpoint are ignored, so a new "
                     "schedule starts clean — use checkpoint.resume_from to continue a run "
                     "instead. The classifier head transfers only when the class list "
                     "matches; otherwise it stays freshly initialised."))
    drop_rate: float = Field(
        0.0, ge=0.0, le=0.9,
        **ui(label="Dropout", step=0.05,
             tooltip="Classifier-head dropout. Mostly redundant once drop-path is on; "
                     "raise only if the head specifically overfits."))
    drop_path_rate: float = Field(
        0.1, ge=0.0, le=0.9,
        **ui(label="Drop path (stochastic depth)", step=0.05, summary_priority=7,
             tooltip="THE primary regularizer for deep ConvNeXt/ViT/Swin models — far more "
                     "effective than dropout. Scale with model size and schedule length: "
                     "~0.05 tiny/short, 0.2 for ConvNeXt-T fine-tuning, up to 0.5 huge/long."))
    global_pool: Literal["avg", "max", "avgmax", "token", ""] = Field(
        "avg", **ui(label="Global pool", advanced=True,
                    tooltip="Pooling before the classifier. 'token' uses the ViT CLS token. "
                            "Leave at the architecture's default unless experimenting."))
    head_init_scale: float = Field(
        1.0, gt=0.0, le=10.0,
        **ui(label="Head init scale", advanced=True, step=0.001,
             tooltip="Scales the freshly-initialized classifier weights. ConvNeXt's own "
                     "fine-tuning recipe uses 0.001 — a tiny head produces small early "
                     "gradients and stops a random head from wrecking good pretrained "
                     "features. Cheaper than a freeze schedule and often as effective."))
    freeze_policy: FreezePolicy = Field(
        FreezePolicy.NONE,
        **ui(label="Freeze policy", summary_priority=5,
             tooltip="Head-only warmup: freeze the backbone for the first few epochs while "
                     "the random head stabilizes. Useful on very small datasets; "
                     "head_init_scale often achieves the same thing for free."))
    freeze_epochs: int = Field(
        0, ge=0, **ui(label="Freeze epochs", show_if="freeze_policy != 'none'",
                      tooltip="Epochs to keep frozen before unfreezing. 1–3 is plenty."))
    freeze_stages: int = Field(
        0, ge=0, le=8, **ui(label="Freeze first K stages", advanced=True,
                            show_if="freeze_policy == 'freeze-first-K-stages'",
                            tooltip="Permanently freeze the first K backbone stages. Saves "
                                    "memory and time; costs accuracy when your domain is "
                                    "far from ImageNet (low-level features must adapt)."))
    freeze_bn: bool = Field(
        False, **ui(label="Freeze BatchNorm", advanced=True,
                    tooltip="Freeze BN running stats and affine params. Helps at batch "
                            "sizes under ~16 where BN statistics get noisy. Irrelevant "
                            "for ConvNeXt/ViT, which use LayerNorm."))
    torch_compile: bool = Field(
        False, **ui(label="torch.compile", advanced=True,
                    tooltip="~40% faster training on CUDA, but costs 30–120s of compile time "
                            "on the first step — bad for short runs, good for long ones. "
                            "Recompiles when input shape changes, so it fights progressive "
                            "resizing. Partial support on MPS."))
    ema: bool = Field(
        True, **ui(label="Weight EMA", summary_priority=4,
                   tooltip="Exponential moving average of weights, evaluated alongside the "
                           "raw model. Worth ~+0.25 points for free, plus better calibration "
                           "and robustness to label noise. Keep on."))
    ema_decay: float | Literal["auto"] = Field(
        0.9998,
        **ui(label="EMA decay", advanced=True, show_if="ema == true",
             tooltip="Averaging horizon is 1/(1-decay) steps. 0.9998 means 5,000 steps — "
                     "LONGER THAN AN ENTIRE RUN on a small dataset, which makes the EMA "
                     "metric meaningless. Use 'auto' to size it at ~10% of total steps."))
    gradient_checkpointing: bool = Field(
        False, **ui(label="Gradient checkpointing", advanced=True,
                    tooltip="Recompute activations in the backward pass: ~40% less VRAM for "
                            "~30% more time. The right answer when you want a bigger model "
                            "or batch than fits."))


class LossConfig(_G):
    loss: LossName = Field(
        LossName.CROSS_ENTROPY,
        **ui(label="Loss", summary_priority=8,
             tooltip="Auto-switched to soft_target_ce when mixup/cutmix is active, because "
                     "standard cross-entropy cannot consume soft targets. bce is the "
                     "ResNet-Strikes-Back choice and slightly beats CE on long, heavily "
                     "augmented schedules."))
    label_smoothing: float = Field(
        0.1, ge=0.0, lt=1.0,
        **ui(label="Label smoothing", step=0.01, summary_priority=5,
             tooltip="Near-universal small win for accuracy and calibration. Set to 0 if "
                     "this run will be a distillation TEACHER — smoothing collapses the "
                     "inter-class structure that distillation depends on."))
    focal_gamma: float = Field(
        2.0, ge=0.0, le=10.0,
        **ui(label="Focal gamma", step=0.5, show_if="loss == 'focal'",
             tooltip="Down-weights easy examples. 2.0 is standard. Only use focal loss under "
                     "SEVERE imbalance — on balanced data it measurably hurts."))
    class_weights: ClassWeights = Field(
        ClassWeights.NONE,
        **ui(label="Class weights", summary_priority=6,
             tooltip="effective-number ((1-b^n)/(1-b)) is better than raw inverse-frequency, "
                     "which over-corrects on very rare classes. Do not stack this with a "
                     "weighted sampler — you will double-correct."))
    manual_class_weights: list[float] | None = Field(
        None, **ui(label="Manual weights", advanced=True, show_if="class_weights == 'manual'",
                   tooltip="One weight per class, in sorted class-name order."))
    cb_beta: float = Field(
        0.9999, ge=0.0, lt=1.0,
        **ui(label="Class-balanced beta", advanced=True,
             show_if="class_weights == 'effective-number'",
             tooltip="Controls the effective-number reweighting. 0.9999 is the paper "
                     "default; lower values reweight less aggressively."))
    logit_adjust_tau: float = Field(
        1.0, ge=0.0, le=2.0,
        **ui(label="Logit adjustment tau", step=0.1, show_if="loss == 'logit_adjusted'",
             tooltip="Adds tau*log(class_prior) to the logits. The best-supported single fix "
                     "for long-tailed data — principled, cheap, and beats loss reweighting. "
                     "1.0 is the theoretically-motivated value."))


class OptimizationConfig(_G):
    optimizer: Optimizer = Field(
        Optimizer.ADAMW,
        **ui(label="Optimizer", summary_priority=9,
             tooltip="AdamW is right for ViT/ConvNeXt and fine for CNNs. SGD+Nesterov is "
                     "still competitive for pure convnets on long schedules but needs an LR "
                     "~1000x larger. Lion needs LR 3-10x SMALLER and weight decay 3-10x "
                     "larger than AdamW."))
    lr: float = Field(
        3e-4, gt=0.0, le=10.0,
        **ui(label="Learning rate", widget="log", summary_priority=10,
             tooltip="With weight decay, the hyperparameter most worth tuning per-dataset. "
                     "3e-4 suits AdamW fine-tuning. The UI suggests a rescaled value when "
                     "you change optimizer or batch size, and shows which rule it used."))
    lr_scaling_rule: Literal["none", "linear", "sqrt"] = Field(
        "sqrt",
        **ui(label="LR scaling rule", advanced=True,
             tooltip="How to rescale LR when batch size changes. Linear (Goyal 2017) was "
                     "derived for SGD; sqrt is the better empirical fit for adaptive "
                     "optimizers. This only drives the SUGGESTION — it never silently "
                     "overwrites a value you typed."))
    weight_decay: float = Field(
        0.05, ge=0.0, le=1.0,
        **ui(label="Weight decay", widget="log", summary_priority=8,
             tooltip="Spread across published recipes is enormous: 0.05 (ConvNeXt pretrain), "
                     "2e-5 (torchvision SGD), 1e-8 (ConvNeXt fine-tune). Not transferable "
                     "across optimizers or regimes — sweep it alongside LR."))
    no_weight_decay_on_norm_bias: bool = Field(
        True, **ui(label="No WD on norm/bias", advanced=True,
                   tooltip="Excludes LayerNorm/BatchNorm gains, biases and position embeddings "
                           "from decay. Every modern reference implementation does this; "
                           "leaving it off is a small, well-known bug. Keep on."))
    betas: tuple[float, float] = Field(
        (0.9, 0.999),
        **ui(label="Betas", advanced=True, show_if="optimizer in ('adamw', 'lion')",
             tooltip="AdamW momentum coefficients. Lion wants (0.9, 0.99). Lower beta2 "
                     "(0.95) can stabilize large-batch transformer training."))
    eps: float = Field(
        1e-8, gt=0.0,
        **ui(label="Epsilon", advanced=True, widget="log",
             show_if="optimizer in ('adamw', 'rmsprop')",
             tooltip="Numerical floor. Raise to 1e-6 if you see NaNs under fp16."))
    momentum: float = Field(
        0.9, ge=0.0, lt=1.0,
        **ui(label="Momentum", advanced=True, show_if="optimizer in ('sgd', 'rmsprop')",
             tooltip="SGD momentum. 0.9 is essentially always right."))
    nesterov: bool = Field(
        True, **ui(label="Nesterov", advanced=True, show_if="optimizer == 'sgd'",
                   tooltip="Nesterov momentum. Free and marginally better than plain momentum."))
    layer_lr_decay: float = Field(
        1.0, gt=0.0, le=1.0,
        **ui(label="Layer-wise LR decay", step=0.05, summary_priority=9,
             tooltip="1.0 = off. Gives layer L a LR of lr * decay^(depth-L), so early "
                     "generic layers move slowly and late task-specific layers move fast. "
                     "One of the highest-value and least-known knobs for fine-tuning: try "
                     "0.75. Published values: 0.65 (ViT-B), 0.8 (ViT-L, ConvNeXt)."))
    head_lr_mult: float = Field(
        1.0, gt=0.0, le=100.0,
        **ui(label="Head LR multiplier", advanced=True, step=0.5,
             tooltip="Extra LR multiplier on the classifier head. A coarse alternative to "
                     "layer_lr_decay; 10x is the classic discriminative-LR setting."))
    sam: bool = Field(
        False, **ui(label="SAM", summary_priority=7,
                    tooltip="Sharpness-Aware Minimization: seeks flat minima using TWO "
                            "forward-backward passes per step, so it DOUBLES training time. "
                            "Gain is architecture-dependent — +5.3 points on ViT-B/16, only "
                            "+0.8 on ResNet-152. Worth it for ViTs without heavy augmentation."))
    sam_rho: float = Field(
        0.05, gt=0.0, le=1.0,
        **ui(label="SAM rho", advanced=True, step=0.01, show_if="sam == true",
             tooltip="Neighborhood radius. 0.05 for SAM. Larger values (0.5-2.0) for the "
                     "adaptive variant, which is easier to tune."))
    sam_adaptive: bool = Field(
        False, **ui(label="Adaptive SAM (ASAM)", advanced=True, show_if="sam == true",
                    tooltip="Scale-invariant rho — much less sensitive to tuning, and better "
                            "suited to a pretrain-then-finetune pipeline."))
    grad_clip_norm: float = Field(
        1.0, ge=0.0,
        **ui(label="Gradient clip norm", advanced=True, step=0.1,
             tooltip="0 disables. Cheap insurance against loss spikes; matters most for "
                     "transformers and under fp16."))
    grad_accum_steps: int = Field(
        1, ge=1, le=64,
        **ui(label="Gradient accumulation", advanced=True, summary_priority=3,
             tooltip="Multiplies effective batch size without more VRAM, at proportionally "
                     "more wall-clock. Caveat: BatchNorm still only sees the micro-batch, so "
                     "this is NOT fully equivalent to a real large batch for convnets."))


class ScheduleConfig(_G):
    epochs: int = Field(
        30, ge=1, le=10_000,
        **ui(label="Epochs", summary_priority=10,
             tooltip="Regularization strength should scale WITH this. Heavy augmentation on "
                     "a short schedule is just underfitting."))
    batch_size: int = Field(
        64, ge=1, le=4096,
        **ui(label="Batch size", summary_priority=9,
             tooltip="Largest that fits, generally. Changing it should change the LR — the "
                     "UI will suggest the rescaled value. Under ~16, consider freeze_bn "
                     "for BatchNorm backbones."))
    scheduler: Scheduler = Field(
        Scheduler.COSINE,
        **ui(label="Scheduler", summary_priority=6,
             tooltip="Cosine+warmup is what essentially every modern recipe uses. one_cycle "
                     "is a good short-schedule alternative. plateau when you genuinely don't "
                     "know the right epoch count. step is legacy, for reproducing old baselines."))
    warmup_epochs: float = Field(
        3.0, ge=0.0,
        **ui(label="Warmup epochs", step=0.5, summary_priority=4,
             tooltip="Linear ramp from warmup_start_lr. Matters most at large batch, high LR "
                     "and for transformers, where it prevents an early collapse. ConvNeXt "
                     "fine-tuning uses 0."))
    warmup_start_lr: float = Field(
        1e-6, ge=0.0, **ui(label="Warmup start LR", advanced=True, widget="log",
                           tooltip="LR at step 0 of the warmup ramp, before it climbs "
                                   "linearly to the base LR. Effectively always fine at "
                                   "1e-6; raise it only if warmup is eating a meaningful "
                                   "fraction of a very short schedule."))
    min_lr: float = Field(
        1e-6, ge=0.0, **ui(label="Min LR", advanced=True, widget="log",
                           tooltip="Floor the cosine decays to. Non-zero avoids a completely "
                                   "dead final epoch."))
    amp: Amp = Field(
        Amp.BF16,
        **ui(label="Mixed precision", summary_priority=5,
             tooltip="1.5-2x throughput. bf16 needs Ampere+ and needs no loss scaler; fp16 "
                     "works on older GPUs. On Apple MPS bf16 is auto-downgraded to fp16 — "
                     "MPS autocast disables itself for bf16 and lacks optimized bf16 kernels."))
    seed: int = Field(
        42, ge=0, **ui(label="Seed", tooltip="Seeds torch, numpy and Python RNGs plus the "
                                             "dataloader workers. Vary it to measure run-to-run "
                                             "variance — do that BEFORE trusting a 0.2-point "
                                             "difference between two configs."))
    deterministic: bool = Field(
        False, **ui(label="Deterministic", advanced=True,
                    tooltip="Forces deterministic kernels. Costs 10-30% throughput and some "
                            "ops have no deterministic implementation. Use to debug, not to train."))
    device: Device = Field(
        Device.AUTO, **ui(label="Device", advanced=True,
                          tooltip="auto picks cuda > mps > cpu. The resolved value is logged "
                                  "to the tracker, along with any device-driven downgrades."))
    progressive_resizing: bool = Field(
        False, **ui(label="Progressive resizing", advanced=True, summary_priority=2,
                    tooltip="Start at a low resolution and ramp to input_size. Early epochs "
                            "run ~2x faster and it acts as a curriculum. Note it conflicts "
                            "with torch.compile, which recompiles on every shape change."))
    progressive_start_size: int = Field(
        160, ge=32, le=1024, **ui(label="Start size", advanced=True, unit="px",
                                  show_if="progressive_resizing == true",
                                  tooltip="Resolution at epoch 0, ramped up to input_size. "
                                          "About 0.7x the target is a good starting point; "
                                          "going much lower makes the early epochs cheap but "
                                          "teaches features at a scale the model never sees "
                                          "again."))
    progressive_end_epoch: float = Field(
        0.75, gt=0.0, le=1.0, **ui(label="Ramp ends at", advanced=True, step=0.05,
                                   show_if="progressive_resizing == true",
                                   tooltip="Fraction of training by which full resolution is "
                                           "reached. Always finish at full res so the final "
                                           "weights match the test resolution."))


class ValidationConfig(_G):
    eval_every_n_epochs: int = Field(
        1, ge=1, **ui(label="Eval every N epochs",
                      tooltip="Raise for very long runs where validation is expensive. "
                              "Interacts with early-stopping patience, which counts evals."))
    metrics: list[Metric] = Field(
        default_factory=lambda: [Metric.ACC1, Metric.ACC5, Metric.MACRO_F1],
        **ui(label="Metrics", widget="multiselect", summary_priority=3,
             tooltip="All computed via torchmetrics. Add ECE if you care about calibrated "
                     "probabilities; add per-class recall to spot a collapsed rare class."))
    primary_metric: Metric = Field(
        Metric.ACC1,
        **ui(label="Primary metric", summary_priority=8,
             tooltip="Drives checkpointing, early stopping and the leaderboard sort. Switch "
                     "to macro-F1 or balanced-accuracy on imbalanced data — accuracy is a "
                     "misleading objective there. Choosing fpr@fnr follows the FIRST FNR "
                     "target below, since that metric is a list."))
    positive_class: str | None = Field(
        None,
        **ui(label="Positive class", widget="class-select", summary_priority=2,
             show_if="'fpr@fnr' in metrics",
             tooltip="Which class counts as a positive detection — the one whose misses are "
                     "the FNR and whose false accepts are the FPR. There is no sane default: "
                     "on a live/spoof problem, picking the wrong one silently swaps the two "
                     "error rates and every number reads backwards."))
    fpr_at_fnr_targets: list[Annotated[float, Field(gt=0.0, lt=1.0)]] = Field(
        default_factory=lambda: [0.01, 0.001],
        **ui(label="FNR targets", widget="float-list", show_if="'fpr@fnr' in metrics",
             tooltip="One FPR is reported per target, e.g. 0.01 gives fpr@fnr0_01 — the "
                     "false-positive rate at the threshold that misses at most 1% of "
                     "positives. Enter fractions, not percents. As the primary metric, the "
                     "FIRST entry is the one that decides the best checkpoint."))
    eval_ema_weights: bool = Field(
        True, **ui(label="Evaluate EMA weights", show_if="ema == true",
                   tooltip="Reports both raw and EMA metrics. Watch the EMA-horizon warning: "
                           "on a short run an over-long horizon makes this number meaningless."))
    tta: TTA = Field(
        TTA.NONE,
        **ui(label="Test-time augmentation", summary_priority=4,
             tooltip="hflip costs 2x inference for ~+0.7 points — the best accuracy-per-compute "
                     "on offer. multi-crop/multi-scale add much less for much more. Averaging "
                     "can HURT where the augmentation destroys the label, so always compare "
                     "against the no-TTA number."))
    log_confusion_matrix: bool = Field(
        True, **ui(label="Log confusion matrix",
                   tooltip="Logged as an artifact each eval. The fastest way to see which "
                           "two classes the model actually conflates."))
    log_worst_predictions: bool = Field(
        True, **ui(label="Log worst predictions",
                   tooltip="Top-32 highest-loss validation images. Usually the single highest-ROI "
                           "thing in the app: most of them are MISLABELLED rather than hard, "
                           "and fixing labels beats any hyperparameter."))


class CheckpointConfig(_G):
    save_top_k: int = Field(
        1, ge=0, le=20, **ui(label="Save top K",
                             tooltip="Best-K checkpoints by primary metric. 0 saves none — "
                                     "useful for cheap sweeps where you only want the metric."))
    save_last: bool = Field(
        True, **ui(label="Save last",
                   tooltip="Keeps last.ckpt (weights + optimizer + epoch) so an interrupted "
                           "run can be resumed. Turn off only when disk is tight and you "
                           "are running a large sweep, where the best checkpoint is all "
                           "you actually keep."))
    early_stopping: bool = Field(
        True, **ui(label="Early stopping", summary_priority=3,
                   tooltip="Saves time, not accuracy. Be careful with a noisy small val set — "
                           "raise patience or it will stop on noise."))
    es_patience: int = Field(
        10, ge=1, **ui(label="ES patience", show_if="early_stopping == true",
                       tooltip="Evaluations (not epochs) without improvement before stopping."))
    es_min_delta: float = Field(
        0.001, ge=0.0, **ui(label="ES min delta", advanced=True, show_if="early_stopping == true",
                            tooltip="Improvement below this doesn't count. Set near your "
                                    "run-to-run seed variance."))
    resume_from: str | None = Field(
        None, **ui(label="Resume from", widget="path", advanced=True,
                   tooltip="A .ckpt file, or a previous run's directory (its last.ckpt is "
                           "used). Restores weights, EMA, optimizer, scaler, schedule and "
                           "best-so-far tracking, then continues at the next epoch under "
                           "the CURRENT config — so to train longer, raise epochs and "
                           "resume. Classes and architecture must match the original run."))
    output_dir: str = Field(
        "./runs", **ui(label="Output dir", widget="path",
                       tooltip="Root for checkpoints and artifacts. Each run gets a subdirectory."))


class TrackingConfig(_G):
    enabled: bool = Field(
        True, **ui(label="Track this run",
                   tooltip="Turn off only for throwaway debugging. Untracked runs are "
                           "invisible to the comparison view."))
    backend: Literal["mlflow", "none"] = Field(
        "mlflow", **ui(label="Backend", tooltip="Behind an adapter interface — see "
                                                "tracking/base.py to add another."))
    tracking_uri: str = Field(
        default_factory=lambda: os.environ.get("TRAINLAB_TRACKING_URI",
                                               DEFAULT_TRACKING_URI),
        **ui(label="Tracking URI",
             tooltip="MLflow server; docker-compose and `make up-local` both bind here. "
                     "Prefer 127.0.0.1 over localhost — some proxy configurations "
                     "intercept the hostname but not the literal address."))
    experiment_name: str = Field(
        "default", **ui(label="Experiment", tooltip="Groups related runs. One per dataset "
                                                     "is a good convention."))
    run_name: str | None = Field(
        None, **ui(label="Run name",
                   tooltip="Leave empty — auto-generated from the config, e.g. "
                           "'convnext_tiny · 224 · randaug · mixup · 30ep'. You should never "
                           "have to name a run by hand."))
    tags: dict[str, str] = Field(
        default_factory=dict, **ui(label="Extra tags", advanced=True,
                                   tooltip="Free-form key/value tags merged with the "
                                           "auto-generated ones."))
    preset: Preset = Field(
        Preset.CUSTOM, **ui(label="Derived from preset", widget="readonly",
                            tooltip="Auto-tagged so you can filter the leaderboard by "
                                    "preset. Falls back to 'custom' as soon as anything "
                                    "the preset describes is changed — paths, the output "
                                    "directory and these tracking settings excepted."))
    log_augmentation_preview: bool = Field(
        True, **ui(label="Log augmentation preview", advanced=True,
                   tooltip="Uploads the preview grid as a run artifact, so you can see what "
                           "the model actually saw months later."))


class ExperimentalConfig(_G):
    """Higher-variance techniques. Expect to spend runs before these pay off."""

    kd_enabled: bool = Field(
        False, **ui(label="Knowledge distillation", summary_priority=5,
                    tooltip="Needs a LONG schedule and the SAME aggressive augmentation for "
                            "teacher and student ('function matching', Beyer 2022). Short-schedule "
                            "KD mostly disappoints. Note a label-smoothed teacher is a worse teacher."))
    kd_teacher_ckpt: str | None = Field(
        None, **ui(label="Teacher checkpoint", widget="path", show_if="kd_enabled == true",
                   tooltip="A previous run's best checkpoint, or a timm model name."))
    kd_temperature: float = Field(
        4.0, gt=0.0, **ui(label="KD temperature", show_if="kd_enabled == true",
                          tooltip="Softens teacher logits. Higher shares more inter-class "
                                  "structure but over-smooths past ~10."))
    kd_alpha: float = Field(
        0.5, ge=0.0, le=1.0, **ui(label="KD alpha", show_if="kd_enabled == true",
                                  tooltip="Weight on the distillation term vs the hard-label "
                                          "term. 1.0 ignores ground-truth labels entirely."))
    swa: bool = Field(
        False, **ui(label="SWA", summary_priority=4,
                    tooltip="Equal-weight averaging of late checkpoints under a high constant "
                            "LR. Worth +0.6-0.9 on ImageNet ResNets, but requires changing the "
                            "schedule — unlike EMA, which is free. Try EMA first."))
    swa_start_epoch: float = Field(
        0.75, gt=0.0, le=1.0, **ui(label="SWA start", step=0.05, show_if="swa == true",
                                   tooltip="Fraction of training after which averaging begins."))
    swa_lr: float = Field(
        1e-4, gt=0.0, **ui(label="SWA LR", widget="log", show_if="swa == true",
                           tooltip="Constant LR held during the averaging phase. SWA needs a "
                                   "HIGH constant LR to keep exploring the basin it averages "
                                   "over — roughly the value your cosine schedule reaches "
                                   "around the SWA start epoch. Too low and it just averages "
                                   "near-identical checkpoints and gains nothing."))
    pseudo_label_dir: str | None = Field(
        None, **ui(label="Pseudo-label directory", widget="path",
                   tooltip="Unlabeled images. Trains, predicts, keeps confident predictions "
                           "and retrains. High variance — it amplifies whatever bias the "
                           "model already has."))
    pseudo_label_threshold: float = Field(
        0.95, gt=0.0, lt=1.0, **ui(label="Confidence threshold",
                                   show_if="pseudo_label_dir != null",
                                   tooltip="Minimum softmax confidence to accept a pseudo-label."))
    curriculum_by_loss: bool = Field(
        False, **ui(label="Curriculum by loss",
                    tooltip="Train on the easiest samples first and grow the pool as training "
                            "proceeds, ranking by per-sample loss. Mixed evidence in the "
                            "literature and easy to make worse — it costs an extra scoring "
                            "pass over the training set each epoch."))
    curriculum_start_frac: float = Field(
        0.5, gt=0.0, le=1.0,
        **ui(label="Start with easiest", step=0.05, show_if="curriculum_by_loss == true",
             tooltip="Fraction of the training set available at the first curriculum epoch. "
                     "0.5 starts on the easier half. Too low and early epochs see a badly "
                     "unrepresentative slice of your classes."))
    curriculum_epochs: int = Field(
        10, ge=1, **ui(label="Ramp to full over", show_if="curriculum_by_loss == true",
                       tooltip="Epochs by which the whole training set is in play. After this "
                               "the run is ordinary training."))
    label_noise_filter: bool = Field(
        False, **ui(label="Label-noise filtering",
                    tooltip="Drops persistently high-loss training samples after a warmup "
                            "phase. Powerful on scraped data; dangerous when high loss means "
                            "'hard but correct' rather than 'mislabelled'."))
    label_noise_percentile: float = Field(
        0.02, ge=0.0, le=0.5, **ui(label="Drop top-% loss", show_if="label_noise_filter == true",
                                   tooltip="Fraction of the training set to discard."))
    label_noise_start_epoch: int = Field(
        5, ge=0, **ui(label="Start filtering at epoch", show_if="label_noise_filter == true",
                      tooltip="Epochs of normal training before any sample is dropped. Filtering "
                              "from epoch 0 discards whatever the model happens to find hard "
                              "while it is still random, which is not the same as mislabelled."))

    # ---- in-domain self-supervised pretraining -------------------------------------
    ssl_method: SSLMethod = Field(
        SSLMethod.NONE,
        **ui(label="In-domain SSL pretraining", summary_priority=6,
             tooltip="Label-free pretraining on your OWN images before supervised training — "
                     "it reuses the train split and ignores the labels, so no extra data is "
                     "required. Worth it when your domain is far from ImageNet (medical, "
                     "satellite, industrial) AND you can afford hundreds of epochs; a handful "
                     "of epochs does nothing. SimSiam: any backbone. MAE: plain ViT only (it drops masked tokens). SimMIM: ViT or Swin (it keeps every token)."))
    ssl_epochs: int = Field(
        30, ge=1, **ui(label="SSL epochs", show_if="ssl_method != 'none'", summary_priority=2,
                       tooltip="Default 30 suits the default pretrained=true, where SSL is "
                               "domain adaptation and 20-50 epochs is the useful range. From "
                               "scratch (pretrained=false) SSL needs a LONG schedule — "
                               "SimSiam's CIFAR-10 recipe is 800 epochs, its ImageNet "
                               "baseline 100 — so raise this well above 100 there."))
    ssl_extra_data_dir: str | None = Field(
        None, **ui(label="Extra unlabeled data", widget="path", show_if="ssl_method != 'none'",
                   tooltip="Optional directory of unlabeled images, searched recursively and "
                           "with no class structure required. When set, SSL pretrains on the "
                           "train split PLUS these images (all labels ignored). When empty, "
                           "SSL runs on the train split alone."))
    ssl_input_size: int | None = Field(
        None, ge=32, le=1024,
        **ui(label="SSL input size", unit="px", advanced=True, show_if="ssl_method != 'none'",
             tooltip="Defaults to the supervised input_size. Pretraining at a lower resolution "
                     "(e.g. 128) is a cheap way to afford far more SSL epochs, and the "
                     "representation still transfers to the higher supervised resolution."))
    ssl_batch_size: int = Field(
        32, ge=1, le=4096,
        **ui(label="SSL batch size", show_if="ssl_method != 'none'",
             tooltip="SimSiam encodes TWO views per step, so its activation memory is roughly "
                     "double a supervised step at the same batch size: measured 10.4 GB at "
                     "batch 64 / 224px / convnext_tiny, which is why the default is 32 "
                     "(~5 GB). Both methods work fine at small batches — neither needs the "
                     "large-batch negatives SimCLR does — so raise this for speed, not quality."))
    ssl_num_workers: int | None = Field(
        None, ge=0, le=64,
        **ui(label="SSL dataloader workers", advanced=True, show_if="ssl_method != 'none'",
             tooltip="Defaults to the Data group's num_workers. SSL augmentation is heavier "
                     "than supervised augmentation, so this stage is more often input-bound."))
    ssl_amp: Amp = Field(
        Amp.BF16, **ui(label="SSL mixed precision", advanced=True, show_if="ssl_method != 'none'",
                       tooltip="Resolved per device exactly like the supervised setting: bf16 "
                               "becomes fp16 on MPS, and AMP is disabled on CPU."))
    ssl_lr: float | Literal["auto"] = Field(
        "auto", **ui(label="SSL learning rate", widget="log", advanced=True,
                     show_if="ssl_method != 'none'",
                     tooltip="'auto' applies each method's published rule scaled to your batch "
                             "size: SimSiam uses SGD at 0.05*bs/256, the masked methods use AdamW at "
                             "1.5e-4*bs/256. Both differ enough from supervised LRs that "
                             "carrying one over is a mistake."))
    ssl_weight_decay: float | Literal["auto"] = Field(
        "auto", **ui(label="SSL weight decay", widget="log", advanced=True,
                     show_if="ssl_method != 'none'",
                     tooltip="'auto' uses 1e-4 for SimSiam (SGD) and 0.05 for MAE/SimMIM (AdamW), "
                             "per the respective papers."))
    ssl_lr_warmup_epochs: int = Field(
        10, ge=0, **ui(label="SSL LR warmup", advanced=True, show_if="ssl_method != 'none'",
                       tooltip="Linear LR ramp inside the SSL stage before its cosine decay. "
                               "MAE uses 40 warmup epochs out of 800; scale it to your budget."))
    ssl_save_encoder: bool = Field(
        True, **ui(label="Save SSL encoder", advanced=True, show_if="ssl_method != 'none'",
                   tooltip="Logs the pretrained encoder as a run artifact so a later run can "
                           "reuse it via Model → pretrained tag instead of paying for SSL again. "
                           "This is the real way to use SSL: pretrain once, fine-tune many times."))

    simsiam_proj_hidden_dim: int = Field(
        2048, ge=64, **ui(label="SimSiam projector width", advanced=True,
                          show_if="ssl_method == 'simsiam'",
                          tooltip="Hidden width of the 3-layer projection MLP. 2048 is the paper "
                                  "value; shrink it only if the projector dominates memory on a "
                                  "small backbone."))
    simsiam_out_dim: int = Field(
        2048, ge=64, **ui(label="SimSiam output dim", advanced=True,
                          show_if="ssl_method == 'simsiam'",
                          tooltip="Embedding dimension the loss is computed in."))
    simsiam_pred_hidden_dim: int = Field(
        512, ge=16, **ui(label="SimSiam predictor bottleneck",
                         show_if="ssl_method == 'simsiam'",
                         tooltip="The predictor's bottleneck (512 against a 2048 output) is what "
                                 "stops SimSiam collapsing to a constant, together with the "
                                 "stop-gradient. Widening it to match the output dim defeats the "
                                 "mechanism — this is SimSiam's one load-bearing hyperparameter."))
    simsiam_fix_pred_lr: bool = Field(
        True, **ui(label="Constant predictor LR", advanced=True,
                   show_if="ssl_method == 'simsiam'",
                   tooltip="Holds the predictor's LR constant while the rest of the network "
                           "decays. The SimSiam paper found this consistently helps and never "
                           "hurts. Leave on."))

    mae_mask_ratio: float = Field(
        0.75, gt=0.0, lt=1.0,
        **ui(label="MAE mask ratio", step=0.05, show_if="ssl_method == 'mae'",
             tooltip="Fraction of patches hidden from the encoder. MAE's central finding is that "
                     "a HIGH ratio (0.75) is what makes the task non-trivial — images are "
                     "spatially redundant, so masking 15% the way language models do is too easy."))
    mae_decoder_dim: int = Field(
        512, ge=32, **ui(label="MAE decoder width", advanced=True, show_if="ssl_method == 'mae'",
                         tooltip="The decoder is discarded after pretraining, so it can be far "
                                 "narrower than the encoder. 512 is MAE's value; 256 is a "
                                 "reasonable economy."))
    mae_decoder_depth: int = Field(
        8, ge=1, le=24, **ui(label="MAE decoder depth", advanced=True, show_if="ssl_method == 'mae'",
                             tooltip="MAE uses 8 blocks. Fewer (1-2) trains much faster and still "
                                     "gives usable features — the lightly reference example uses 1."))
    mae_decoder_heads: int = Field(
        16, ge=1, le=32, **ui(label="MAE decoder heads", advanced=True,
                              show_if="ssl_method == 'mae'",
                              tooltip="Attention heads in the decoder. Must divide the decoder width."))
    mae_norm_pix_loss: bool = Field(
        True, **ui(label="Normalized pixel targets", advanced=True, show_if="ssl_method == 'mae'",
                   tooltip="Regress per-patch normalized pixels rather than raw ones. MAE reports "
                           "this improves representation quality; turn it off only to inspect "
                           "reconstructions, which look wrong under normalization."))

    simmim_mask_ratio: float = Field(
        0.6, gt=0.0, lt=1.0,
        **ui(label="SimMIM mask ratio", step=0.05, show_if="ssl_method == 'simmim'",
             tooltip="Fraction of mask units hidden. SimMIM is far less sensitive to this than "
                     "MAE — the paper reports a wide 0.1-0.7 plateau — because the encoder still "
                     "sees every position, so the task stays well-posed at lower ratios."))
    simmim_mask_patch_size: int = Field(
        32, ge=4, le=128,
        **ui(label="SimMIM mask unit", unit="px", show_if="ssl_method == 'simmim'",
             tooltip="Size of each masked square in PIXELS, independent of the model's patch "
                     "size. SimMIM's finding is that the masked unit must be large (32) relative "
                     "to the patch size: with a small unit the model can inpaint from immediate "
                     "neighbours and learns nothing. Must divide the input size and be a multiple "
                     "of the encoder's patch stride."))

    # ---- decoupled classifier retraining --------------------------------------------
    classifier_retrain_epochs: int = Field(
        0, ge=0, **ui(label="Classifier retraining (cRT)", summary_priority=3,
                      tooltip="Two-stage decoupled training for long-tailed data: after normal "
                              "training, re-train ONLY the classifier with class-balanced "
                              "sampling. Kang (2020) found instance-balanced sampling learns "
                              "the best representations and only the classifier needs rebalancing."))
    classifier_retrain_lr: float | Literal["auto"] = Field(
        "auto",
        **ui(label="cRT learning rate", widget="log", advanced=True,
             show_if="classifier_retrain_epochs > 0",
             tooltip="'auto' uses 10x the base LR, since only a freshly reinitialised linear "
                     "head is training and the backbone is frozen."))
    classifier_retrain_reinit: bool = Field(
        True, **ui(label="Reinitialise the head", advanced=True,
                   show_if="classifier_retrain_epochs > 0",
                   tooltip="cRT re-initialises the classifier before rebalancing; LWS instead "
                           "keeps it and only rescales its weight norms. Reinitialising is the "
                           "stronger of the two in the original paper."))

    # ---- pseudo-labeling --------------------------------------------------------------
    pseudo_label_epochs: int = Field(
        10, ge=1, **ui(label="Pseudo-label epochs", show_if="pseudo_label_dir != null",
                       tooltip="Extra epochs trained on the labelled set plus the accepted "
                               "pseudo-labelled images, per round."))
    pseudo_label_rounds: int = Field(
        1, ge=1, le=5, **ui(label="Pseudo-label rounds", advanced=True,
                            show_if="pseudo_label_dir != null",
                            tooltip="Each round re-predicts the unlabeled pool with the improved "
                                    "model and re-selects. More rounds compound both the gains "
                                    "and the confirmation bias."))
    pseudo_label_max_ratio: float = Field(
        1.0, gt=0.0, le=10.0,
        **ui(label="Max pseudo:real ratio", advanced=True, show_if="pseudo_label_dir != null",
             tooltip="Caps accepted pseudo-labels at this multiple of the labelled set size, "
                     "keeping the highest-confidence ones. Stops a large unlabeled pool from "
                     "drowning out your ground truth."))

    # ---- ensembling ---------------------------------------------------------------------
    ensemble_run_ids: list[str] = Field(
        default_factory=list,
        **ui(label="Ensemble with runs", widget="run-picker",
             tooltip="MLflow run IDs, or local checkpoint paths. Their probabilities are averaged "
                     "with this run's at final evaluation and reported as separate ensemble/* "
                     "metrics, so you can see what the ensemble bought you."))


# --------------------------------------------------------------------------------------
# Root config
# --------------------------------------------------------------------------------------

class TrainConfig(BaseModel):
    """The complete run configuration. Serializes to/from YAML losslessly."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    schema_version: str = Field(SCHEMA_VERSION, **ui(label="Schema version", widget="hidden",
                                                     tooltip="For migrating old YAML files."))

    data: DataConfig = Field(default_factory=DataConfig)
    input: InputConfig = Field(default_factory=InputConfig)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    experimental: ExperimentalConfig = Field(default_factory=ExperimentalConfig)

    # ---------------------------------------------------------------- migrations

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy(cls, data: Any) -> Any:
        """Bring configs written against older schema versions up to date.

        Runs stored in the tracker keep their original YAML, and `extra="forbid"` turns
        any since-renamed field into a hard validation error — so without this the Clone
        button fails permanently on every run recorded before a rename.
        """
        if not isinstance(data, dict):
            return data

        out = dict(data)
        exp = out.get("experimental")
        if isinstance(exp, dict):
            exp = dict(exp)
            if exp.get("ssl_method") == "mim":
                # The old `mim` was an asymmetric masked autoencoder, i.e. MAE.
                exp["ssl_method"] = "mae"
            for old, new in _LEGACY_SSL_FIELDS.items():
                if old in exp:
                    exp.setdefault(new, exp.pop(old))
                exp.pop(old, None)
            out["experimental"] = exp

        for section, dropped in _LEGACY_DROPPED_FIELDS.items():
            node = out.get(section)
            if isinstance(node, dict) and any(k in node for k in dropped):
                node = dict(node)
                for k in dropped:
                    node.pop(k, None)
                out[section] = node
        return out

    # ---------------------------------------------------------------- derived properties

    @property
    def mixup_active(self) -> bool:
        a = self.augmentation
        return a.mixup_prob > 0 and (a.mixup_alpha > 0 or a.cutmix_alpha > 0)

    @property
    def effective_loss(self) -> LossName:
        """The loss that will actually be constructed.

        Derived rather than written back into `loss.loss`, so the stored config always
        reflects what the user chose and the switch reverses cleanly when mixup is turned
        off again. Writing it back made the auto-switch sticky across preset changes.
        """
        if self.mixup_active and self.loss.loss == LossName.CROSS_ENTROPY:
            return LossName.SOFT_TARGET_CE
        return self.loss.loss

    @property
    def loss_was_autoswitched(self) -> bool:
        return self.effective_loss != self.loss.loss

    @property
    def effective_test_input_size(self) -> int:
        """Test resolution, defaulting to the train resolution. Derived for the same
        reason as `effective_loss`: so it tracks `input_size` instead of freezing."""
        return self.input.test_input_size or self.input.input_size

    @property
    def primary_metric_key(self) -> str:
        """The logged metric key that decides which checkpoint is best.

        Every metric is its own key except `fpr@fnr`, which names a family — one FPR per
        FNR target. The first target is the one that decides, so that selecting the metric
        as primary is a complete answer rather than an ambiguous one; the rest are still
        computed and logged alongside.
        """
        V = self.validation
        if V.primary_metric is Metric.FPR_AT_FNR and V.fpr_at_fnr_targets:
            return fpr_at_fnr_key(V.fpr_at_fnr_targets[0])
        return V.primary_metric.value

    @property
    def fpr_at_fnr_active(self) -> bool:
        return Metric.FPR_AT_FNR in self.validation.metrics \
            or self.validation.primary_metric is Metric.FPR_AT_FNR

    @property
    def effective_batch_size(self) -> int:
        return self.schedule.batch_size * self.optimization.grad_accum_steps

    def steps_per_epoch(self, n_train: int) -> int:
        return max(1, math.ceil(n_train / self.effective_batch_size))

    def total_steps(self, n_train: int) -> int:
        return self.steps_per_epoch(n_train) * self.schedule.epochs

    # ---------------------------------------------------------------- hard validation

    @model_validator(mode="after")
    def _hard_errors(self) -> TrainConfig:
        # A loss taking integer targets cannot consume mixup's soft targets. Plain
        # cross_entropy is silently upgraded via `effective_loss`; anything else is a
        # deliberate choice we cannot safely reinterpret, so it's a hard error.
        if (self.mixup_active
                and self.loss.loss not in SOFT_TARGET_CAPABLE
                and self.loss.loss != LossName.CROSS_ENTROPY):
            raise ValueError(
                f"loss={self.loss.loss.value} cannot consume the soft targets produced by "
                f"mixup/cutmix. Use soft_target_ce or bce, or disable mixing."
            )

        if self.schedule.warmup_epochs >= self.schedule.epochs:
            raise ValueError(
                f"warmup_epochs ({self.schedule.warmup_epochs}) must be less than "
                f"epochs ({self.schedule.epochs})."
            )

        if self.model.freeze_policy != FreezePolicy.NONE and self.model.freeze_epochs >= self.schedule.epochs:
            raise ValueError("freeze_epochs must be less than epochs, or the backbone never trains.")

        if self.model.pretrained_checkpoint and self.checkpoint.resume_from:
            raise ValueError(
                "model.pretrained_checkpoint and checkpoint.resume_from are both set. "
                "They answer different questions — 'start a NEW run from these weights' "
                "versus 'CONTINUE that run, optimizer and schedule included' — and "
                "resume_from already restores the weights. Choose one."
            )
        if self.model.pretrained_checkpoint and self.ssl_active:
            raise ValueError(
                "model.pretrained_checkpoint and an SSL method are both set. The SSL "
                "stage builds and pretrains its own encoder and then overwrites the "
                "supervised model's weights, so the checkpoint would be silently "
                "discarded. Choose one initialisation."
            )

        if self.fpr_at_fnr_active:
            # A metric family with no members produces nothing, and as the primary metric
            # it would leave every epoch scoring -inf with no best checkpoint written.
            if not self.validation.fpr_at_fnr_targets:
                raise ValueError(
                    "fpr@fnr is selected but no FNR targets are set. Add at least one, "
                    "e.g. 0.01 for the FPR at a 1% miss rate."
                )
            # Which class is positive decides which of the two error rates is which, and
            # nothing in the data says. Guessing would report a plausible number for the
            # wrong question, so it is required rather than defaulted.
            if not self.validation.positive_class:
                raise ValueError(
                    "fpr@fnr needs validation.positive_class: name the class whose misses "
                    "count as false negatives. Point the form at a dataset first if the "
                    "class list is still empty."
                )
            known = self.data.class_names
            if known and self.validation.positive_class not in known:
                raise ValueError(
                    f"validation.positive_class='{self.validation.positive_class}' is not "
                    f"one of the detected classes {known}."
                )

        if self.experimental.kd_enabled and not self.experimental.kd_teacher_ckpt:
            raise ValueError("Knowledge distillation is enabled but no teacher checkpoint is set.")

        E = self.experimental
        if E.ssl_method != SSLMethod.NONE and E.ssl_lr_warmup_epochs >= E.ssl_epochs:
            raise ValueError(
                f"ssl_lr_warmup_epochs ({E.ssl_lr_warmup_epochs}) must be less than "
                f"ssl_epochs ({E.ssl_epochs})."
            )
        if E.ssl_method == SSLMethod.MAE and E.mae_decoder_dim % E.mae_decoder_heads != 0:
            raise ValueError(
                f"mae_decoder_dim ({E.mae_decoder_dim}) must be divisible by "
                f"mae_decoder_heads ({E.mae_decoder_heads})."
            )
        if E.ssl_method == SSLMethod.SIMMIM:
            size = self.effective_ssl_input_size
            if size % E.simmim_mask_patch_size != 0:
                raise ValueError(
                    f"simmim_mask_patch_size ({E.simmim_mask_patch_size}) must divide the SSL "
                    f"input size ({size}); the mask grid has to tile the image exactly."
                )
        if E.label_noise_filter and E.label_noise_start_epoch >= self.schedule.epochs:
            raise ValueError(
                "label_noise_start_epoch must be less than epochs, or filtering never runs."
            )

        return self

    # ---------------------------------------------------------------- preset label

    @model_validator(mode="after")
    def _demote_stale_preset(self) -> TrainConfig:
        """Give up `tracking.preset` once the config stops matching it.

        The same rule as `AugmentationConfig._expand_preset`, one level up. This label is
        what `naming.run_tags` tags the run with and what the comparison view filters and
        groups by, so one that no longer describes the config is worse than none at all:
        `--preset balanced --epochs 100` and the form's "click Balanced, then change
        something" both used to launch runs tagged `preset=balanced`.

        Demotion only, and the form works the same way: a config never acquires a label it
        was not given. Applying a preset is what grants one — anything else can only take
        it away — so a hand-written config is not retitled on load for matching by chance.
        """
        if self.tracking.preset != Preset.CUSTOM:
            from . import presets      # imports this module — resolved at call time
            if not presets.describes(self.tracking.preset, self.model_dump(mode="json")):
                self.tracking.preset = Preset.CUSTOM
        return self

    # ---------------------------------------------------------------- experimental helpers

    @property
    def ssl_active(self) -> bool:
        return self.experimental.ssl_method != SSLMethod.NONE

    @property
    def effective_ssl_input_size(self) -> int:
        return self.experimental.ssl_input_size or self.input.input_size

    def effective_ssl_num_workers(self, ) -> int:
        w = self.experimental.ssl_num_workers
        return self.data.num_workers if w is None else w

    def resolved_ssl_lr(self) -> float:
        """Published per-method rules, linearly scaled to batch size."""
        E = self.experimental
        if E.ssl_lr != "auto":
            return float(E.ssl_lr)
        bs = E.ssl_batch_size
        if E.ssl_method == SSLMethod.SIMSIAM:
            return 0.05 * bs / 256      # SimSiam: SGD, lr = 0.05 x bs/256
        if E.ssl_method == SSLMethod.SIMMIM:
            return 1e-4 * bs / 256      # SimMIM: AdamW, base 8e-4 at batch 2048
        return 1.5e-4 * bs / 256        # MAE: AdamW, base 1.5e-4 x bs/256

    def resolved_ssl_weight_decay(self) -> float:
        E = self.experimental
        if E.ssl_weight_decay != "auto":
            return float(E.ssl_weight_decay)
        return 1e-4 if E.ssl_method == SSLMethod.SIMSIAM else 0.05

    def resolved_crt_lr(self) -> float:
        lr = self.experimental.classifier_retrain_lr
        return self.optimization.lr * 10 if lr == "auto" else float(lr)

    # ---------------------------------------------------------------- soft warnings

    def warnings(self, *, n_train: int | None = None, imbalance_ratio: float | None = None,
                 resolved_device: str | None = None) -> list[Warning_]:
        """Non-fatal advice. Mirrored client-side for instant feedback in the form.

        Args:
            n_train: training set size, once known — enables step-count-dependent checks.
            imbalance_ratio: max_class_count / min_class_count, once known.
            resolved_device: 'cuda' | 'mps' | 'cpu', once resolved.
        """
        w: list[Warning_] = []
        A, M, L, O, S, V = (self.augmentation, self.model, self.loss,
                            self.optimization, self.schedule, self.validation)

        # --- auto-switched loss --------------------------------------------------------
        if self.loss_was_autoswitched:
            w.append(Warning_(
                severity=Severity.INFO, field="loss.loss",
                message=f"Mixup/CutMix produces soft targets, which standard cross-entropy "
                        f"cannot consume. Using {self.effective_loss.value} for this run.",
                fix="Disable mixup/cutmix to go back to plain cross_entropy."))

        # --- optimizer / LR sanity ---------------------------------------------------
        if O.optimizer == Optimizer.ADAMW and O.lr > 1e-2:
            w.append(Warning_(severity=Severity.WARNING, field="optimization.lr",
                              message=f"lr={O.lr:g} is very high for AdamW; typical range is 1e-5 to 3e-3.",
                              fix="Did you mean to use SGD? SGD needs ~1000x the LR of AdamW."))
        if O.optimizer == Optimizer.SGD and O.lr < 1e-3:
            w.append(Warning_(severity=Severity.WARNING, field="optimization.lr",
                              message=f"lr={O.lr:g} is very low for SGD; typical range is 0.01 to 0.5.",
                              fix="Switch to AdamW, or raise LR by ~1000x."))
        if O.optimizer == Optimizer.LION and O.lr > 3e-4:
            w.append(Warning_(severity=Severity.WARNING, field="optimization.lr",
                              message="Lion needs an LR 3-10x SMALLER than AdamW (and weight decay 3-10x larger).",
                              fix=f"Try lr={O.lr / 5:.1e} and weight_decay={O.weight_decay * 5:g}."))

        # --- mixup on a short fine-tune (RESEARCH.md §0) ------------------------------
        if self.mixup_active and M.pretrained and S.epochs < 50:
            w.append(Warning_(
                severity=Severity.WARNING, field="augmentation.mixup_alpha",
                message=f"Mixup/CutMix with only {S.epochs} epochs of fine-tuning is measurably "
                        f"NEGATIVE in published results — the pretrained features only need weak "
                        f"augmentation to transfer.",
                fix="Set mixup_alpha and cutmix_alpha to 0, or raise epochs above ~100."))

        # --- EMA horizon (RESEARCH.md §5.6) ------------------------------------------
        if M.ema and M.ema_decay != "auto" and n_train:
            horizon = 1.0 / (1.0 - float(M.ema_decay))
            total = self.total_steps(n_train)
            if horizon > 0.5 * total:
                w.append(Warning_(
                    severity=Severity.WARNING, field="model.ema_decay",
                    message=f"EMA horizon is {horizon:,.0f} steps but the whole run is only "
                            f"{total:,} steps. The EMA weights will never converge away from "
                            f"initialization, and the EMA metric will be meaningless.",
                    fix=f"Use ema_decay='auto', or set it to {1 - 1 / (0.1 * total):.5f}."))

        # --- imbalance handling stacked (RESEARCH.md §7) ------------------------------
        rebalancers = [
            self.data.sampler != Sampler.RANDOM,
            L.class_weights != ClassWeights.NONE,
            L.loss in (LossName.LOGIT_ADJUSTED, LossName.CLASS_BALANCED_CE, LossName.FOCAL),
        ]
        if sum(rebalancers) > 1:
            w.append(Warning_(
                severity=Severity.WARNING, field="loss.class_weights",
                message="More than one imbalance correction is active (sampler, class weights, "
                        "and/or a rebalancing loss). These are substitutes, not complements — "
                        "stacking them over-corrects toward rare classes.",
                fix="Pick one. Logit adjustment is the best-supported single choice."))

        if L.loss == LossName.CLASS_BALANCED_CE and L.class_weights == ClassWeights.NONE:
            w.append(Warning_(
                severity=Severity.WARNING, field="loss.class_weights",
                message="loss='class_balanced_ce' with class_weights='none' is exactly "
                        "plain cross-entropy — the loss only differs by the weights it is "
                        "given, so this run is not doing any class balancing.",
                fix="Set class_weights='effective-number' (the Cui et al. formulation "
                    "this loss is named for), or switch the loss back to cross_entropy."))

        if L.loss == LossName.FOCAL and imbalance_ratio is not None and imbalance_ratio < 3:
            w.append(Warning_(
                severity=Severity.WARNING, field="loss.loss",
                message=f"Focal loss on near-balanced data (ratio {imbalance_ratio:.1f}:1) is "
                        f"documented to HURT — it down-weights easy examples that still matter.",
                fix="Use cross_entropy with label smoothing."))

        if self.fpr_at_fnr_active:
            n = self.data.num_classes
            if n is not None and n != 2:
                w.append(Warning_(
                    severity=Severity.ERROR, field="validation.metrics",
                    message=f"fpr@fnr is a binary-detection metric, but the dataset has "
                            f"{n} classes. It will not be computed.",
                    fix="Drop fpr@fnr, or point at a two-class dataset."))
            if V.primary_metric is Metric.FPR_AT_FNR and len(V.fpr_at_fnr_targets) > 1:
                w.append(Warning_(
                    severity=Severity.INFO, field="validation.primary_metric",
                    message=f"Checkpointing and early stopping follow "
                            f"{fpr_at_fnr_key(V.fpr_at_fnr_targets[0])}, the first target; "
                            f"the other {len(V.fpr_at_fnr_targets) - 1} are computed and "
                            f"logged but do not decide the best epoch.",
                    fix="Reorder the targets to change which one decides."))
            # Below roughly one positive per 1/target, the metric is quantised to the point
            # of being noise: with 200 positives, "FNR <= 0.1%" is the same operating point
            # as "FNR <= 0.5%", and both move by a whole sample between epochs.
            if n_train is not None and V.fpr_at_fnr_targets:
                tightest = min(V.fpr_at_fnr_targets)
                if n_train * (1 - self.data.val_split) < 1 / tightest:
                    w.append(Warning_(
                        severity=Severity.WARNING, field="validation.fpr_at_fnr_targets",
                        message=f"An FNR target of {tightest:g} needs at least ~{1 / tightest:,.0f} "
                                f"positive validation samples to be measurable at all; "
                                f"below that the metric jumps a whole sample at a time.",
                        fix="Use a looser target, or validate on more data."))

        if M.pretrained_checkpoint and M.pretrained:
            w.append(Warning_(
                severity=Severity.INFO, field="model.pretrained_checkpoint",
                message="A pretrained checkpoint is set, so the timm weights (and the "
                        "pretrained tag) are skipped — the checkpoint is the "
                        "initialisation.",
                fix=None))

        if imbalance_ratio is not None and imbalance_ratio > 10 and V.primary_metric == Metric.ACC1:
            w.append(Warning_(
                severity=Severity.WARNING, field="validation.primary_metric",
                message=f"Class imbalance is {imbalance_ratio:.0f}:1 but the primary metric is "
                        f"top-1 accuracy, which a majority-class-only model can score well on.",
                fix="Use macro-F1 or balanced-accuracy."))

        # --- augmentation sanity ------------------------------------------------------
        if self.input.rrc_scale[0] < 0.3 and n_train is not None and n_train < 20_000:
            w.append(Warning_(
                severity=Severity.WARNING, field="input.rrc_scale",
                message=f"rrc_scale lower bound of {self.input.rrc_scale[0]} on a "
                        f"{n_train:,}-image dataset will often crop away all evidence of the class.",
                fix="Try 0.65-1.0, and check the augmentation preview."))
        if A.vflip > 0 or A.rotation_deg > 45:
            w.append(Warning_(
                severity=Severity.INFO, field="augmentation.vflip",
                message="Vertical flip / large rotation assumes there is no canonical 'up'. "
                        "Correct for satellite or microscopy, wrong for natural photos."))

        # --- schedule / regularization balance ---------------------------------------
        if S.epochs > 200 and A.auto_augment == AutoAugment.NONE and not self.mixup_active:
            w.append(Warning_(
                severity=Severity.INFO, field="schedule.epochs",
                message="Long schedule with no strong augmentation will likely overfit.",
                fix="Enable an auto-augment policy, or shorten the schedule."))
        if S.epochs < 20 and A.randaugment_m >= 9 and A.auto_augment == AutoAugment.RANDAUGMENT:
            w.append(Warning_(
                severity=Severity.INFO, field="augmentation.randaugment_m",
                message="RandAugment magnitude 9 on a short schedule is likely to underfit.",
                fix="Drop to 5-7, or use trivialaugment-wide."))

        # --- distillation teacher quality --------------------------------------------
        if L.label_smoothing > 0 and self.experimental.kd_enabled:
            w.append(Warning_(
                severity=Severity.INFO, field="loss.label_smoothing",
                message="Label smoothing collapses the inter-class logit structure that "
                        "distillation relies on. It degrades a model as a TEACHER (this run "
                        "is a student, so this only matters if you distil from it later)."))

        # --- batch-norm at small batch -------------------------------------------------
        if S.batch_size < 16 and not M.freeze_bn:
            w.append(Warning_(
                severity=Severity.INFO, field="schedule.batch_size",
                message=f"Batch size {S.batch_size} makes BatchNorm statistics noisy.",
                fix="Enable freeze_bn, raise batch size, or use a LayerNorm backbone "
                    "(ConvNeXt/ViT), which is unaffected."))

        # --- device-specific ------------------------------------------------------------
        if resolved_device == "mps":
            if S.amp == Amp.BF16:
                w.append(Warning_(
                    severity=Severity.INFO, field="schedule.amp",
                    message="MPS autocast disables itself for bf16 and lacks optimized bf16 "
                            "kernels. Downgraded to fp16 for this run.",
                    fix="Set amp=fp16 explicitly to silence this."))
            if self.input.channels_last:
                w.append(Warning_(severity=Severity.INFO, field="input.channels_last",
                                  message="channels_last has no meaningful effect on MPS. Ignored."))
            if M.torch_compile:
                w.append(Warning_(severity=Severity.WARNING, field="model.torch_compile",
                                  message="torch.compile support on MPS is partial; expect graph "
                                          "breaks and possibly no speedup."))
        if resolved_device == "cpu" and S.amp != Amp.OFF:
            w.append(Warning_(severity=Severity.INFO, field="schedule.amp",
                              message="AMP on CPU is rarely a win. Disabled for this run."))

        # --- throughput ------------------------------------------------------------------
        if M.torch_compile and S.progressive_resizing:
            w.append(Warning_(
                severity=Severity.WARNING, field="model.torch_compile",
                message="torch.compile recompiles on every input-shape change, which "
                        "progressive resizing triggers repeatedly.",
                fix="Disable one of them."))
        if O.sam:
            w.append(Warning_(severity=Severity.INFO, field="optimization.sam",
                              message="SAM doubles training time (two forward-backward passes "
                                      "per step). Large win on ViTs (+5 pts), marginal on "
                                      "well-regularized CNNs (+0.8)."))

        # --- self-supervised pretraining -------------------------------------------------
        E = self.experimental
        if E.ssl_method == SSLMethod.MAE and not M.backbone.startswith(VIT_LIKE_PREFIXES):
            w.append(Warning_(
                severity=Severity.ERROR, field="experimental.ssl_method",
                message=f"MAE drops masked tokens before the encoder, so it needs a plain ViT "
                        f"token sequence, but the backbone is '{M.backbone}'.",
                fix="Use a vit_*/deit_*/deit3_*/flexivit_* backbone (beit_*, eva_* and "
                    "samvit_* are ViT-like in name only and build different timm classes); "
                    "or SimMIM, which keeps every token and also runs on Swin; or SimSiam, "
                    "which works with any architecture."))

        if E.ssl_method == SSLMethod.SIMMIM and not M.backbone.startswith(SIMMIM_LIKE_PREFIXES):
            w.append(Warning_(
                severity=Severity.ERROR, field="experimental.ssl_method",
                message=f"SimMIM masks patch embeddings, so it needs a patch-based encoder "
                        f"(timm VisionTransformer or SwinTransformer), but the backbone is "
                        f"'{M.backbone}'.",
                fix="Use a vit_*/deit_*/swin_* backbone, or SimSiam for a convnet. Note "
                    "swinv2_* builds SwinTransformerV2, which has a different internal "
                    "structure and is not supported. Masked modelling on plain convnets "
                    "needs sparse convolutions (SparK / ConvNeXt-V2 FCMAE), which this app "
                    "does not implement."))

        if E.ssl_method != SSLMethod.NONE:
            # The epoch budget means different things depending on where SSL starts, so
            # the advice does too. Warning about SimSiam's 800-epoch from-scratch CIFAR
            # recipe while the run initialises from ImageNet weights compared the config
            # against an experiment it is not running — and contradicted the adjacent
            # note that a long SSL stage is what pulls those weights apart.
            if M.pretrained:
                if E.ssl_epochs < 10:
                    w.append(Warning_(
                        severity=Severity.WARNING, field="experimental.ssl_epochs",
                        message=f"{E.ssl_epochs} SSL epochs is too few to move the pretrained "
                                f"encoder meaningfully, but still costs {E.ssl_epochs} full "
                                f"passes over the data before supervised training starts.",
                        fix="Use 20-50 epochs for domain adaptation, or turn SSL off and "
                            "rely on the pretrained weights."))
                w.append(Warning_(
                    severity=Severity.INFO, field="experimental.ssl_method",
                    message=f"SSL starts from the ImageNet-pretrained weights, so this is "
                            f"domain adaptation rather than pretraining from scratch: "
                            f"20-50 epochs is the useful range, and {E.ssl_epochs} is "
                            f"{'in it' if 20 <= E.ssl_epochs <= 50 else 'outside it'}. A LONG "
                            f"stage can pull the encoder away from a good initialisation "
                            f"faster than the pretext task adds anything back.",
                    fix="Compare against the same config with ssl_method='none' before "
                        "trusting the result."))
            elif E.ssl_epochs < 100:
                w.append(Warning_(
                    severity=Severity.WARNING, field="experimental.ssl_epochs",
                    message=f"Training from scratch (pretrained=false), {E.ssl_epochs} SSL "
                            f"epochs is very likely to be wasted time. Self-supervised "
                            f"objectives need hundreds of epochs to build useful features "
                            f"from random init — SimSiam's CIFAR-10 recipe uses 800.",
                    fix="Raise ssl_epochs above ~100, or enable pretrained weights and "
                        "treat this as domain adaptation instead."))
            if E.ssl_save_encoder:
                w.append(Warning_(
                    severity=Severity.INFO, field="experimental.ssl_save_encoder",
                    message="The SSL encoder is logged as an artifact. Pretrain once, then point "
                            "later runs at that checkpoint instead of repeating this stage."))

        # --- pseudo-labeling ---------------------------------------------------------------
        if E.pseudo_label_dir:
            w.append(Warning_(
                severity=Severity.INFO, field="experimental.pseudo_label_dir",
                message=f"Pseudo-labeling adds {E.pseudo_label_rounds} x "
                        f"{E.pseudo_label_epochs} epochs after the main run, and amplifies "
                        f"whatever bias the model already has.",
                fix="Compare against the same config with pseudo-labeling off before trusting it."))
            if E.pseudo_label_threshold < 0.9:
                w.append(Warning_(
                    severity=Severity.WARNING, field="experimental.pseudo_label_threshold",
                    message=f"A threshold of {E.pseudo_label_threshold} accepts many wrong "
                            f"labels; confirmation bias compounds over rounds.",
                    fix="0.95 or higher is the usual operating point."))

        # --- cRT ---------------------------------------------------------------------------
        if E.classifier_retrain_epochs > 0 and imbalance_ratio is not None and imbalance_ratio < 3:
            w.append(Warning_(
                severity=Severity.INFO, field="experimental.classifier_retrain_epochs",
                message=f"cRT targets long-tailed data; this dataset is near-balanced "
                        f"({imbalance_ratio:.1f}:1), so expect little from it."))

        # --- curriculum / noise filtering ---------------------------------------------------
        if E.curriculum_by_loss and E.label_noise_filter:
            w.append(Warning_(
                severity=Severity.WARNING, field="experimental.curriculum_by_loss",
                message="Curriculum and noise filtering both rank samples by loss and both "
                        "withhold the hardest ones. Together they can starve the model of every "
                        "difficult example.",
                fix="Enable one at a time so you can attribute the effect."))

        return w
