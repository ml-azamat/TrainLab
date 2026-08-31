"""Transform pipelines.

Routing decision worth knowing about: RandAugment / AutoAugment / 3-Augment come from
`timm` (which supports magnitude-std jitter and the DeiT III `3a` policy), while
TrivialAugmentWide comes from `torchvision.transforms.v2` because timm has no
implementation of it. Everything else is torchvision v2.
"""

from __future__ import annotations

import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import InterpolationMode

from .config import (
    AutoAugment,
    Interpolation,
    Normalization,
    TrainConfig,
    TrainResizePolicy,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_INTERP = {
    Interpolation.NEAREST: InterpolationMode.NEAREST,
    Interpolation.BILINEAR: InterpolationMode.BILINEAR,
    Interpolation.BICUBIC: InterpolationMode.BICUBIC,
    Interpolation.LANCZOS: InterpolationMode.LANCZOS,
}


def resolve_norm(cfg: TrainConfig, dataset_stats: tuple | None = None):
    n = cfg.input.normalization
    if n == Normalization.IMAGENET:
        return IMAGENET_MEAN, IMAGENET_STD
    if n == Normalization.CUSTOM:
        return (tuple(cfg.input.custom_mean or IMAGENET_MEAN),
                tuple(cfg.input.custom_std or IMAGENET_STD))
    if dataset_stats is not None:
        return dataset_stats
    return IMAGENET_MEAN, IMAGENET_STD


def denormalize(x: torch.Tensor, mean, std) -> torch.Tensor:
    m = torch.tensor(mean, device=x.device).view(-1, 1, 1)
    s = torch.tensor(std, device=x.device).view(-1, 1, 1)
    return (x * s + m).clamp(0, 1)


def _timm_auto_augment(cfg: TrainConfig, mean, interp: Interpolation):
    """Build the timm PIL-stage auto-augment op, or None."""
    from timm.data.auto_augment import auto_augment_transform, rand_augment_transform
    from timm.data.transforms import str_to_pil_interp

    a = cfg.augmentation
    hparams = {
        "translate_const": int(cfg.input.input_size * 0.45),
        # timm passes these straight through to PIL as `fillcolor` and `resample`, so
        # img_mean must be 0-255 ints and interpolation must be a PIL constant, not a name.
        "img_mean": tuple(round(m * 255) for m in mean),
        "interpolation": str_to_pil_interp(interp.value),
    }

    if a.auto_augment == AutoAugment.RANDAUGMENT:
        cfg_str = f"rand-m{a.randaugment_m}-n{a.randaugment_n}-mstd{a.randaugment_mstd}-inc1"
        return rand_augment_transform(cfg_str, hparams)
    if a.auto_augment == AutoAugment.AUTOAUGMENT_IMAGENET:
        return auto_augment_transform("original-mstd0.5", hparams)
    if a.auto_augment == AutoAugment.THREE_AUGMENT:
        # DeiT III 3-Augment: one of {grayscale, solarize, blur}, equal probability.
        return auto_augment_transform("3a", hparams)
    return None


def _geometric(cfg: TrainConfig, size: int, interp: InterpolationMode) -> list:
    p = cfg.input.train_resize_policy
    aa = {"interpolation": interp, "antialias": cfg.input.antialias}

    if p == TrainResizePolicy.RRC:
        return [v2.RandomResizedCrop(size, scale=tuple(cfg.input.rrc_scale),
                                     ratio=tuple(cfg.input.rrc_ratio), **aa)]
    if p == TrainResizePolicy.SHORTEST_THEN_CROP:
        return [v2.Resize(size, **aa), v2.RandomCrop(size, pad_if_needed=True)]
    if p == TrainResizePolicy.SQUASH:
        return [v2.Resize((size, size), **aa)]
    if p == TrainResizePolicy.PAD_SQUARE:
        # Preserve aspect ratio; pad to square. Correct when framing carries information.
        return [_PadToSquare(size, interp, cfg.input.antialias)]
    raise ValueError(p)


class _PadToSquare(v2.Transform):
    """Resize longest side to `size`, then zero-pad the short side to square."""

    def __init__(self, size: int, interp: InterpolationMode, antialias: bool):
        super().__init__()
        self.size, self.interp, self.antialias = size, interp, antialias

    def forward(self, img):
        from torchvision.transforms.v2 import functional as F
        w, h = F.get_size(img)[::-1]
        scale = self.size / max(w, h)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        img = F.resize(img, [nh, nw], interpolation=self.interp, antialias=self.antialias)
        pad_w, pad_h = self.size - nw, self.size - nh
        return F.pad(img, [pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2])


def build_train_transform(cfg: TrainConfig, *, size: int | None = None,
                          dataset_stats=None, normalize: bool = True) -> v2.Compose:
    a, i = cfg.augmentation, cfg.input
    size = size or i.input_size
    interp = _INTERP[i.interpolation]
    mean, std = resolve_norm(cfg, dataset_stats)

    ops: list = list(_geometric(cfg, size, interp))

    if a.hflip > 0:
        ops.append(v2.RandomHorizontalFlip(a.hflip))
    if a.vflip > 0:
        ops.append(v2.RandomVerticalFlip(a.vflip))
    if a.rotation_deg > 0:
        ops.append(v2.RandomRotation(a.rotation_deg, interpolation=interp))

    if a.auto_augment == AutoAugment.TRIVIALAUGMENT_WIDE:
        ops.append(v2.TrivialAugmentWide(interpolation=interp))
    elif a.auto_augment != AutoAugment.NONE:
        op = _timm_auto_augment(cfg, mean, i.interpolation)
        if op is not None:
            ops.append(op)

    # Auto-augment policies already contain colour operations, so applying jitter on top
    # double-counts. 3-Augment is the exception: colour jitter is part of that recipe.
    if a.color_jitter > 0 and a.auto_augment in (AutoAugment.NONE, AutoAugment.THREE_AUGMENT):
        cj = a.color_jitter
        ops.append(v2.ColorJitter(brightness=cj, contrast=cj, saturation=cj))

    if a.grayscale_p > 0:
        ops.append(v2.RandomGrayscale(a.grayscale_p))
    if a.blur_p > 0:
        ops.append(v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=a.blur_p))

    ops += [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
    if normalize:
        ops.append(v2.Normalize(mean=list(mean), std=list(std)))

    if a.random_erasing_p > 0:
        ops.append(v2.RandomErasing(
            p=a.random_erasing_p,
            value="random" if a.random_erasing_mode in ("pixel", "rand") else 0,
        ))
    return v2.Compose(ops)


def build_eval_transform(cfg: TrainConfig, *, size: int | None = None,
                         dataset_stats=None, normalize: bool = True) -> v2.Compose:
    i = cfg.input
    size = size or i.input_size
    interp = _INTERP[i.interpolation]
    mean, std = resolve_norm(cfg, dataset_stats)
    aa = {"interpolation": interp, "antialias": i.antialias}

    if i.val_resize_policy == "resize-squash":
        ops: list = [v2.Resize((size, size), **aa)]
    elif i.val_resize_policy == "resize-pad-to-square":
        ops = [_PadToSquare(size, interp, i.antialias)]
    else:
        resize_to = int(round(size / i.val_crop_pct))
        ops = [v2.Resize(resize_to, **aa), v2.CenterCrop(size)]

    ops += [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
    if normalize:
        ops.append(v2.Normalize(mean=list(mean), std=list(std)))
    return v2.Compose(ops)


def progressive_size(cfg: TrainConfig, epoch: int) -> int:
    """Resolution for a given epoch under progressive resizing."""
    s = cfg.schedule
    if not s.progressive_resizing:
        return cfg.input.input_size
    end = max(1, s.progressive_end_epoch * s.epochs)
    frac = min(1.0, epoch / end)
    lo, hi = s.progressive_start_size, cfg.input.input_size
    # Round to a multiple of 32 so convnet strides stay clean.
    return int(round((lo + (hi - lo) * frac) / 32) * 32)
