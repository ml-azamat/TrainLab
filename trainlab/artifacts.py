"""Visual artifacts: augmentation preview, confusion matrix, worst predictions.

All rendered to PNG on disk and handed to the tracker, so a run stays interpretable
months later without re-running anything.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # noqa: E402  headless: this runs inside a training subprocess
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from .transforms import denormalize  # noqa: E402


def _to_numpy_img(t: torch.Tensor, mean, std) -> np.ndarray:
    return denormalize(t.detach().float().cpu(), mean, std).permute(1, 2, 0).numpy()


def augmentation_preview(dataset, transform, mean, std, out_path: Path,
                         n: int = 16, seed: int = 0, titles: list[str] | None = None) -> Path:
    """Grid of training images pushed through the *current* pipeline.

    The highest-value debugging surface in the app: RRC cropping the subject out,
    double normalisation, channel swaps and aspect squash are all visible at a glance.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset), size=min(n, len(dataset)), replace=False)
    cols = min(4, len(idx))
    rows = math.ceil(len(idx) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.4))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")

    from PIL import Image
    for ax, i in zip(axes, idx):
        # One unreadable file used to cost the whole grid: the preview is the app's best
        # debugging surface, and losing it for a run because a random draw happened to
        # include a truncated JPEG is the worst possible trade. The tile says so instead.
        try:
            img = Image.open(dataset.m.paths[int(i)]).convert("RGB")
            x = transform(img)
        except Exception as e:
            ax.text(0.5, 0.5, f"unreadable\n{type(e).__name__}", ha="center", va="center",
                    fontsize=6, color="crimson", transform=ax.transAxes)
            ax.set_title(Path(dataset.m.paths[int(i)]).name[:18], fontsize=6)
            ax.axis("off")
            continue
        ax.imshow(np.clip(_to_numpy_img(x, mean, std), 0, 1))
        label = dataset.m.class_names[dataset.m.labels[int(i)]]
        ax.set_title(label[:18], fontsize=7)
        ax.axis("off")

    fig.suptitle(titles[0] if titles else "Augmentation preview", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


def confusion_matrix_png(cm: np.ndarray, class_names: list[str], out_path: Path,
                         normalize: bool = True) -> Path:
    if normalize:
        with np.errstate(invalid="ignore", divide="ignore"):
            cm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    n = len(class_names)
    size = max(4.0, min(16.0, n * 0.45))
    fig, ax = plt.subplots(figsize=(size, size * 0.9))
    im = ax.imshow(cm, cmap="magma", vmin=0, vmax=1 if normalize else None)
    fig.colorbar(im, ax=ax, fraction=0.046)

    show_labels = n <= 40
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    if show_labels:
        ax.set_xticklabels([c[:14] for c in class_names], rotation=90, fontsize=6)
        ax.set_yticklabels([c[:14] for c in class_names], fontsize=6)
    else:
        ax.set_xticklabels([]), ax.set_yticklabels([])
    if n <= 15:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{cm[i, j]:.2f}" if normalize else str(int(cm[i, j])),
                        ha="center", va="center", fontsize=6,
                        color="white" if cm[i, j] < 0.6 else "black")

    ax.set_xlabel("predicted"), ax.set_ylabel("true")
    ax.set_title("Confusion matrix" + (" (row-normalised)" if normalize else ""))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def worst_predictions_png(records: list[dict], out_path: Path, mean, std, k: int = 32) -> Path:
    """Highest-loss validation images.

    In practice most of these are MISLABELLED rather than hard, which makes this the
    cheapest route to a real accuracy gain on a custom dataset.
    """
    records = sorted(records, key=lambda r: -r["loss"])[:k]
    if not records:
        return out_path

    cols = 8
    rows = math.ceil(len(records) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.9, rows * 2.25))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")

    from PIL import Image
    for ax, r in zip(axes, records):
        ax.imshow(Image.open(r["path"]).convert("RGB"))
        ax.set_title(f"T:{r['true'][:11]}\nP:{r['pred'][:11]} ({r['conf']:.2f})",
                     fontsize=6, color="crimson")
        ax.axis("off")

    fig.suptitle(f"Top-{len(records)} highest-loss validation images "
                 f"— check these for label noise", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path
