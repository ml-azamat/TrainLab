"""Live augmentation preview and dataset introspection.

The preview is the app's most valuable debugging surface, so it runs on the real
pipeline built from the real config — never an approximation. Same images each time
(seeded) so that changing one knob shows the effect of that knob, not of resampling.
"""

from __future__ import annotations

import base64
import io
import random
from functools import lru_cache

import numpy as np
from PIL import Image

from trainlab import data as data_mod
from trainlab import transforms as tf_mod
from trainlab.config import TrainConfig


@lru_cache(maxsize=8)
def _cached_manifest(train_dir: str, fmt: str, group_col: str | None):
    cfg = TrainConfig.model_validate({
        "data": {"train_dir": train_dir, "dataset_format": fmt,
                 "group_column": group_col},
    })
    return data_mod.load_manifest(cfg, "train")


def inspect_dataset(cfg: TrainConfig) -> dict:
    """Class list, counts and imbalance — drives num_classes and several warnings."""
    try:
        m = _cached_manifest(cfg.data.train_dir, cfg.data.dataset_format.value,
                             cfg.data.group_column)
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    counts = {m.class_names[k]: v for k, v in sorted(m.counts.items())}
    return {
        "ok": True,
        "num_classes": len(m.class_names),
        "class_names": m.class_names,
        "num_images": len(m),
        "class_counts": counts,
        "imbalance_ratio": round(m.imbalance_ratio, 2),
        "fingerprint": m.fingerprint(),
        "smallest_class": min(counts, key=counts.get) if counts else None,
        "largest_class": max(counts, key=counts.get) if counts else None,
    }


def _encode(img: Image.Image, size: int = 224) -> str:
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=82)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()


def augmentation_preview(cfg: TrainConfig, n: int = 8, seed: int = 0,
                         include_original: bool = True) -> dict:
    """Render `n` sampled training images through the current train pipeline."""
    try:
        m = _cached_manifest(cfg.data.train_dir, cfg.data.dataset_format.value,
                             cfg.data.group_column)
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(m), size=min(n, len(m)), replace=False)

    mean, std = tf_mod.resolve_norm(cfg)
    transform = tf_mod.build_train_transform(cfg)

    # Seed torch/random so a re-render with an unchanged config is stable, and a changed
    # config shows the change rather than a different random draw.
    import torch
    torch.manual_seed(seed)
    random.seed(seed)

    items = []
    for i in idx:
        i = int(i)
        try:
            src = Image.open(m.paths[i]).convert("RGB")
            x = transform(src)
            arr = tf_mod.denormalize(x, mean, std).permute(1, 2, 0).numpy()
            aug = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
            item = {
                "label": m.class_names[m.labels[i]],
                "augmented": _encode(aug),
                "path": m.paths[i],
            }
            if include_original:
                thumb = src.copy()
                thumb.thumbnail((256, 256))
                item["original"] = _encode(thumb)
            items.append(item)
        except Exception as e:  # one bad file must not break the whole grid
            items.append({"label": "?", "error": f"{type(e).__name__}: {e}",
                          "path": m.paths[i]})

    return {
        "ok": True,
        "items": items,
        "pipeline": [type(t).__name__ for t in transform.transforms],
        "input_size": cfg.input.input_size,
        "test_input_size": cfg.effective_test_input_size,
        "normalization": f"mean={tuple(round(v, 3) for v in mean)} "
                         f"std={tuple(round(v, 3) for v in std)}",
    }


def eval_preview(cfg: TrainConfig, n: int = 4, seed: int = 0) -> dict:
    """The validation pipeline, so train/eval mismatch is visible side by side."""
    try:
        m = _cached_manifest(cfg.data.train_dir, cfg.data.dataset_format.value,
                             cfg.data.group_column)
    except Exception as e:
        return {"ok": False, "error": str(e)}

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(m), size=min(n, len(m)), replace=False)
    mean, std = tf_mod.resolve_norm(cfg)
    transform = tf_mod.build_eval_transform(cfg, size=cfg.effective_test_input_size)

    items = []
    for i in idx:
        i = int(i)
        # Guarded like the train-time grid above: a truncated file shows as one bad tile
        # rather than a failed request with no preview at all.
        try:
            x = transform(Image.open(m.paths[i]).convert("RGB"))
        except Exception as e:
            items.append({"label": "?", "error": f"{type(e).__name__}: {e}",
                          "path": m.paths[i]})
            continue
        arr = tf_mod.denormalize(x, mean, std).permute(1, 2, 0).numpy()
        items.append({
            "label": m.class_names[m.labels[i]],
            "augmented": _encode(Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))),
        })
    return {"ok": True, "items": items,
            "pipeline": [type(t).__name__ for t in transform.transforms]}


def clear_cache() -> None:
    _cached_manifest.cache_clear()
