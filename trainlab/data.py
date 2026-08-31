"""Dataset construction, splitting, sampling and caching."""

from __future__ import annotations

import csv
import hashlib
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import (
    DataLoader, Dataset, Sampler, SubsetRandomSampler, WeightedRandomSampler,
)
from torch.utils.data._utils.collate import default_collate

from .config import CacheMode, DatasetFormat, Sampler as SamplerKind, TrainConfig

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".ppm", ".gif"}


# --------------------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------------------

@dataclass
class Manifest:
    """A flat list of (path, label_index) plus the class vocabulary."""
    paths: list[str]
    labels: list[int]
    class_names: list[str]
    groups: list[str] | None = None

    def __len__(self) -> int:
        return len(self.paths)

    @property
    def counts(self) -> Counter:
        return Counter(self.labels)

    @property
    def imbalance_ratio(self) -> float:
        c = self.counts
        return (max(c.values()) / max(1, min(c.values()))) if c else 1.0

    def fingerprint(self) -> dict:
        """Stable identity of the data this run saw.

        Hashes the sorted (path-relative-to-the-dataset-root, label) pairs, so
        re-ordering the filesystem or moving the dataset does not change the
        fingerprint, but adding, removing or relabelling a file does. Hashing only the
        basename — as this did — collides across classes whenever two class directories
        contain a file of the same name, which is the norm for exported datasets
        (`0001.jpg` in every class).
        """
        h = hashlib.sha256()
        root = os.path.commonpath([os.path.dirname(p) for p in self.paths]) if self.paths else ""
        for p, y in sorted(zip(self.paths, self.labels)):
            h.update(os.path.relpath(p, root).encode() if root else p.encode())
            h.update(b"\x00")
            h.update(str(y).encode())
        return {
            "num_files": len(self.paths),
            "num_classes": len(self.class_names),
            "sha256": h.hexdigest()[:16],
            "class_counts": {self.class_names[k]: v for k, v in sorted(self.counts.items())},
        }


def _scan_imagefolder(root: str | Path) -> Manifest:
    root = Path(root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Train directory does not exist: {root}")
    classes = sorted(d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))
    if not classes:
        raise ValueError(
            f"No class subdirectories found in {root}. ImageFolder expects one directory "
            f"per class. Use dataset_format='csv' for a flat directory."
        )
    paths, labels = [], []
    for idx, cname in enumerate(classes):
        for p in sorted((root / cname).rglob("*")):
            if p.suffix.lower() in IMG_EXTS:
                paths.append(str(p))
                labels.append(idx)
    if not paths:
        raise ValueError(f"No images found under {root} (looked for {sorted(IMG_EXTS)}).")
    if len(classes) < 2:
        raise ValueError(
            f"Found only one class ('{classes[0]}') under {root}. Classification needs at "
            f"least two. If your class directories are nested one level deeper, point "
            f"train_dir at that level instead."
        )
    return Manifest(paths, labels, classes)


def _scan_csv(path: str | Path, group_column: str | None = None) -> Manifest:
    path = Path(path).expanduser()
    base = path.parent
    rows = list(csv.DictReader(path.open()))
    if not rows:
        raise ValueError(f"CSV manifest {path} is empty.")
    cols = {c.lower(): c for c in rows[0]}
    pcol = cols.get("path") or cols.get("filename") or cols.get("image")
    lcol = cols.get("label") or cols.get("class") or cols.get("target")
    if not pcol or not lcol:
        raise ValueError(f"CSV must contain path/label columns; found {list(rows[0])}.")

    classes = sorted({r[lcol] for r in rows})
    lut = {c: i for i, c in enumerate(classes)}
    paths = [str(base / r[pcol]) if not os.path.isabs(r[pcol]) else r[pcol] for r in rows]
    labels = [lut[r[lcol]] for r in rows]
    groups = [r[group_column] for r in rows] if group_column and group_column in rows[0] else None
    return Manifest(paths, labels, classes, groups)


def scan_unlabeled(root: str | Path) -> list[str]:
    """Every image under `root`, recursively, with no class structure required.

    Used by SSL pretraining and pseudo-labeling, where the directory layout carries no
    label information and must not be interpreted as one.
    """
    root = Path(root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Unlabeled directory does not exist: {root}")
    paths = sorted(str(p) for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not paths:
        raise ValueError(f"No images found under {root} (looked for {sorted(IMG_EXTS)}).")
    return paths


def _scan_hf(spec: str) -> Manifest:
    """`spec` is 'dataset_name[:split]', materialised to a local cache directory."""
    from datasets import load_dataset

    name, _, split = spec.partition(":")
    ds = load_dataset(name, split=split or "train")
    cache = Path.home() / ".cache" / "trainlab" / name.replace("/", "__")
    cache.mkdir(parents=True, exist_ok=True)
    label_feat = ds.features["label"]
    names = getattr(label_feat, "names", None)
    if names:
        # ClassLabel feature: raw labels are already indices into `names`.
        classes = list(names)
        to_index = None
    else:
        # Plain values: build the vocabulary from the values themselves and map
        # through it. Indexing raw values directly while the class list came from
        # `sorted(str(...))` pointed labels at the wrong classes whenever the raw
        # values were not already 0..N-1 in sorted-string order.
        classes = sorted({str(x) for x in ds["label"]})
        to_index = {c: i for i, c in enumerate(classes)}
    paths, labels = [], []
    for i, ex in enumerate(ds):
        p = cache / f"{i:07d}.jpg"
        if not p.exists():
            ex["image"].convert("RGB").save(p, quality=95)
        paths.append(str(p))
        labels.append(int(ex["label"]) if to_index is None else to_index[str(ex["label"])])
    return Manifest(paths, labels, classes)


def load_manifest(cfg: TrainConfig, which: str = "train") -> Manifest | None:
    src = cfg.data.train_dir if which == "train" else cfg.data.val_dir
    if not src:
        return None
    fmt = cfg.data.dataset_format
    if fmt == DatasetFormat.IMAGEFOLDER:
        return _scan_imagefolder(src)
    if fmt == DatasetFormat.CSV:
        return _scan_csv(src, cfg.data.group_column)
    return _scan_hf(src)


# --------------------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------------------

def split_manifest(m: Manifest, cfg: TrainConfig) -> tuple[Manifest, Manifest]:
    """Carve a validation split out of `m` when no val directory was given."""
    rng = np.random.default_rng(cfg.schedule.seed)
    frac, strategy = cfg.data.val_split, cfg.data.split_strategy
    n = len(m)
    idx = np.arange(n)

    if strategy == "grouped" and m.groups:
        uniq = sorted(set(m.groups))
        rng.shuffle(uniq)
        n_val_groups = max(1, round(len(uniq) * frac))
        val_groups = set(uniq[:n_val_groups])
        val_mask = np.array([g in val_groups for g in m.groups])
    elif strategy == "stratified":
        val_mask = np.zeros(n, dtype=bool)
        labels = np.array(m.labels)
        for c in np.unique(labels):
            ci = idx[labels == c]
            rng.shuffle(ci)
            k = max(1, round(len(ci) * frac)) if len(ci) > 1 else 0
            val_mask[ci[:k]] = True
    else:
        rng.shuffle(idx)
        val_mask = np.zeros(n, dtype=bool)
        val_mask[idx[: round(n * frac)]] = True

    def take(mask):
        return Manifest(
            [m.paths[i] for i in range(n) if mask[i]],
            [m.labels[i] for i in range(n) if mask[i]],
            m.class_names,
            [m.groups[i] for i in range(n) if mask[i]] if m.groups else None,
        )

    return take(~val_mask), take(val_mask)


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------

#: Root of the decoded-pixels disk cache (cache_mode='disk-decoded').
_DECODED_CACHE_ROOT = Path.home() / ".cache" / "trainlab" / "decoded"


def _load_decoded(path: str) -> Image.Image:
    """Load via a disk cache of decoded RGB pixels.

    JPEG decode is paid once; every later epoch (and worker) reads raw pixels. This
    backs cache_mode='disk-decoded', which used to be a declared enum value with no
    implementation — it silently behaved exactly like 'none'.
    """
    h = hashlib.sha1(path.encode()).hexdigest()
    cp = _DECODED_CACHE_ROOT / h[:2] / f"{h}.npy"
    if cp.exists():
        try:
            return Image.fromarray(np.load(cp, allow_pickle=False))
        except Exception:
            pass  # corrupt or partially-written entry: fall through and rewrite it
    img = Image.open(path).convert("RGB")
    cp.parent.mkdir(parents=True, exist_ok=True)
    tmp = cp.parent / f"{cp.name}.{os.getpid()}.tmp"
    try:
        with open(tmp, "wb") as f:
            np.save(f, np.asarray(img), allow_pickle=False)
        os.replace(tmp, cp)   # atomic: concurrent workers race to identical content
    except OSError:
        tmp.unlink(missing_ok=True)   # cache is best-effort; the image is already decoded
    return img


def load_image(path: str, cache_mode: CacheMode) -> Image.Image:
    if cache_mode == CacheMode.DISK_DECODED:
        return _load_decoded(path)
    return Image.open(path).convert("RGB")


# --------------------------------------------------------------------------------------
# Unreadable images
# --------------------------------------------------------------------------------------
#
# A truncated JPEG, a file that vanished from a network mount, a path with a typo: any of
# them raises inside a DataLoader worker, and the exception is re-raised in the main
# process at the `for batch in loader` line — hours into a run, naming one file out of
# millions, with everything since the last checkpoint lost.
#
# So the read is guarded, the sample is dropped from its batch, and the path is appended
# to a file the run reports on. Dropping rather than substituting is deliberate: a black
# image trains the model on a lie, and a duplicated neighbour quietly reweights the
# dataset. A batch is simply a few samples shorter, and the count is on the record.
#
# The log is a file rather than an attribute because workers are separate processes: an
# in-memory list would be filled in a fork and thrown away with it. Lines are short and
# opened O_APPEND, which POSIX keeps atomic across the workers writing concurrently.


def record_broken_image(log_path, image_path: str, err: BaseException) -> None:
    """Append one unreadable image to the run's log. Never raises."""
    if log_path is None:
        return
    try:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"{image_path}\t{type(err).__name__}: {err}\n")
    except OSError:
        pass          # the log is a diagnostic; losing a line must not end the run


def read_broken_images(log_path) -> list[tuple[str, str]]:
    """(path, error) for every image that failed to load, de-duplicated, in order.

    Workers re-read the same file every epoch, so the raw log repeats itself; what the
    user needs is the set of files to go and fix.
    """
    if log_path is None or not Path(log_path).exists():
        return []
    seen: dict[str, str] = {}
    for line in Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines():
        path, _, err = line.partition("\t")
        if path and path not in seen:
            seen[path] = err
    return list(seen.items())


def collate_skip_broken(batch):
    """Drop the samples that could not be read; `None` when the whole batch was.

    Callers must treat `None` as "no batch here" rather than as data — a batch of zero
    images through a model is either a crash or, worse, a silent NaN.
    """
    kept = [b for b in batch if b is not None]
    if not kept:
        return None
    return default_collate(kept)


class ImageListDataset(Dataset):
    """Reads from a Manifest. `transform` is mutable so progressive resizing can swap it.

    Yields `(image, label, index)`. The index is what lets per-sample loss tracking,
    curriculum ordering, label-noise filtering and worst-prediction reporting refer back
    to a specific file — previously the eval path reconstructed identity from batch order,
    which silently assumed the loader never shuffles.
    """

    def __init__(self, manifest: Manifest, transform=None, cache_mode: CacheMode = CacheMode.NONE,
                 broken_log=None):
        self.m = manifest
        self.transform = transform
        self.cache_mode = cache_mode
        self.broken_log = broken_log
        self._cache: dict[int, Image.Image] = {}

    def __len__(self) -> int:
        return len(self.m)

    def _load(self, i: int) -> Image.Image:
        if self.cache_mode == CacheMode.RAM and i in self._cache:
            return self._cache[i]
        img = load_image(self.m.paths[i], self.cache_mode)
        if self.cache_mode == CacheMode.RAM:
            self._cache[i] = img
        return img

    def __getitem__(self, i: int):
        try:
            img = self._load(i)
            if self.transform is not None:
                img = self.transform(img)
        except Exception as e:
            # The transform is inside the guard on purpose: a file can decode far enough
            # to produce an image object and then fail on the truncated tail when the
            # pixels are actually read.
            record_broken_image(self.broken_log, self.m.paths[i], e)
            return None
        return img, self.m.labels[i], i


class UnlabeledDataset(Dataset):
    """Flat list of image paths with no labels.

    `transform` may return a list of views (SSL transforms do), which is passed through
    untouched so the training loop can decide how to consume them.
    """

    def __init__(self, paths: list[str], transform=None, cache_mode: CacheMode = CacheMode.NONE,
                 broken_log=None):
        self.paths = paths
        self.transform = transform
        self.broken_log = broken_log
        self.cache_mode = cache_mode
        self._cache: dict[int, Image.Image] = {}

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        try:
            if self.cache_mode == CacheMode.RAM and i in self._cache:
                img = self._cache[i]
            else:
                img = load_image(self.paths[i], self.cache_mode)
                if self.cache_mode == CacheMode.RAM:
                    self._cache[i] = img
            return (self.transform(img) if self.transform is not None else img), i
        except Exception as e:
            record_broken_image(self.broken_log, self.paths[i], e)
            return None


class BalancedBatchSampler(Sampler[list[int]]):
    """Every batch contains an (approximately) equal number of samples per class."""

    def __init__(self, labels: list[int], batch_size: int, seed: int = 0):
        self.by_class: dict[int, list[int]] = {}
        for i, y in enumerate(labels):
            self.by_class.setdefault(y, []).append(i)
        self.n_classes = len(self.by_class)
        self.batch_size = batch_size
        self.per_class = max(1, batch_size // self.n_classes)
        self.n_batches = max(1, len(labels) // batch_size)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self):
        pools = {c: list(self.rng.permutation(v)) for c, v in self.by_class.items()}
        classes = list(self.by_class)
        for _ in range(self.n_batches):
            batch: list[int] = []

            def draw(c: int) -> int:
                if not pools[c]:
                    pools[c] = list(self.rng.permutation(self.by_class[c]))
                return int(pools[c].pop())

            for c in classes:
                batch += [draw(c) for _ in range(self.per_class)]
            # `batch_size // n_classes` leaves a remainder whenever the classes don't
            # divide the batch evenly (64 over 10 classes gave batches of 60). Top up
            # from classes chosen at random so the batch is the size that was asked for.
            while len(batch) < self.batch_size:
                batch.append(draw(int(self.rng.choice(classes))))
            self.rng.shuffle(batch)
            yield batch[: self.batch_size]


def _make_sampler(m: Manifest, cfg: TrainConfig, active: list[int] | None = None,
                  force_balanced: bool = False):
    """Returns (sampler, batch_sampler, shuffle).

    `active` restricts training to a subset of indices (curriculum / noise filtering).
    `force_balanced` overrides the configured sampler with class-balanced sampling, which
    is what the cRT stage needs regardless of what the main run used.
    """
    kind = SamplerKind.WEIGHTED if force_balanced else cfg.data.sampler

    if active is not None:
        counts = Counter(m.labels[i] for i in active)
        if kind == SamplerKind.RANDOM:
            return SubsetRandomSampler(active), None, False
        if kind == SamplerKind.BALANCED_BATCH:
            sub = Manifest([m.paths[i] for i in active], [m.labels[i] for i in active],
                           m.class_names)
            return None, _IndexRemapBatchSampler(
                BalancedBatchSampler(sub.labels, cfg.schedule.batch_size, cfg.schedule.seed),
                active), False
        f = 1.0 if force_balanced else cfg.data.oversample_factor
        weights = torch.zeros(len(m), dtype=torch.double)
        for i in active:
            weights[i] = (1.0 / counts[m.labels[i]]) ** f
        return WeightedRandomSampler(weights, num_samples=len(active), replacement=True), None, False

    if kind == SamplerKind.RANDOM:
        return None, None, True
    if kind == SamplerKind.BALANCED_BATCH:
        return None, BalancedBatchSampler(m.labels, cfg.schedule.batch_size, cfg.schedule.seed), False

    # Weighted: interpolate between uniform and full inverse-frequency by oversample_factor.
    counts = m.counts
    f = 1.0 if force_balanced else cfg.data.oversample_factor
    w = torch.tensor([(1.0 / counts[y]) ** f for y in m.labels], dtype=torch.double)
    return WeightedRandomSampler(w, num_samples=len(m), replacement=True), None, False


class _IndexRemapBatchSampler(Sampler[list[int]]):
    """Maps a batch sampler defined over a subset back onto dataset-level indices."""

    def __init__(self, inner: Sampler[list[int]], active: list[int]):
        self.inner, self.active = inner, active

    def __len__(self) -> int:
        return len(self.inner)   # type: ignore[arg-type]

    def __iter__(self):
        for batch in self.inner:
            yield [self.active[i] for i in batch]


def loader_kwargs(cfg: TrainConfig, num_workers: int | None = None) -> dict:
    d = cfg.data
    nw = d.num_workers if num_workers is None else num_workers
    if d.cache_mode == CacheMode.RAM and nw > 0:
        # Worker processes get a forked copy of the dataset, so each fills its own cache:
        # memory is multiplied by num_workers and the parent's cache stays empty. Decoding
        # in-process is the only way the cache is actually a cache.
        nw = 0
    kw = dict(
        num_workers=nw,
        pin_memory=d.pin_memory and torch.cuda.is_available(),
        persistent_workers=d.persistent_workers and nw > 0,
    )
    if nw > 0:
        kw["prefetch_factor"] = d.prefetch_factor
    return kw


def make_train_loader(cfg: TrainConfig, dataset: "ImageListDataset", manifest: Manifest,
                      active: list[int] | None = None,
                      force_balanced: bool = False) -> DataLoader:
    """Build (or rebuild) the training loader, optionally over a subset of indices."""
    sampler, batch_sampler, shuffle = _make_sampler(manifest, cfg, active, force_balanced)
    common = loader_kwargs(cfg)
    if batch_sampler is not None:
        return DataLoader(dataset, batch_sampler=batch_sampler,
                          collate_fn=collate_skip_broken, **common)

    # drop_last keeps batch statistics (and BatchNorm) stable, but when the dataset is
    # smaller than one batch it drops EVERY batch: the loop body never runs, the model
    # never trains, and the run still reports `loss 0.0000` and a plausible accuracy from
    # the untouched backbone. Below one full batch, keep the short batch instead.
    n_available = len(active) if active is not None else len(manifest)
    drop_last = n_available >= cfg.schedule.batch_size
    return DataLoader(dataset, batch_size=cfg.schedule.batch_size, shuffle=shuffle,
                      sampler=sampler, drop_last=drop_last,
                      collate_fn=collate_skip_broken, **common)


def check_class_vocabularies(train_m: Manifest, val_m: Manifest) -> None:
    """Refuse to run when train and val disagree about what the class indices mean.

    A separate val directory is scanned independently, so its class list comes from its
    own subdirectories. If the two listings differ at all — a missing class, an extra
    one, a stray directory — `sorted()` assigns different indices to the same class name
    and every validation label silently points at the wrong class. The run does not
    fail: it reports a plausible accuracy computed against scrambled targets.
    """
    if list(train_m.class_names) == list(val_m.class_names):
        return

    train_set, val_set = set(train_m.class_names), set(val_m.class_names)
    lines = []
    if missing := sorted(train_set - val_set):
        lines.append(f"    in train but not val: {missing}")
    if extra := sorted(val_set - train_set):
        lines.append(f"    in val but not train: {extra}")
    if not lines:  # same names, different order — still a different index mapping
        lines.append(f"    same classes, different order:\n"
                     f"      train: {list(train_m.class_names)}\n"
                     f"      val:   {list(val_m.class_names)}")
    shifted = sorted(
        c for c in train_set & val_set
        if train_m.class_names.index(c) != val_m.class_names.index(c)
    )
    if shifted:
        lines.append(f"    classes whose label index differs between the two: {shifted}")

    raise ValueError(
        "Train and validation directories describe different class vocabularies "
        f"({len(train_m.class_names)} vs {len(val_m.class_names)} classes).\n"
        + "\n".join(lines)
        + "\n  Class indices come from sorting each directory's subdirectories, so a "
          "mismatch means validation labels refer to different classes than the model "
          "was trained on, and every reported metric would be wrong. Make both "
          "directories contain exactly the same class subdirectories (an empty "
          "directory is fine for a class with no val images)."
    )


def build_loaders(cfg: TrainConfig, train_tf, val_tf, broken_log=None
                  ) -> tuple[DataLoader, DataLoader, Manifest, Manifest]:
    train_m = load_manifest(cfg, "train")
    if train_m is None:
        raise ValueError("data.train_dir is required.")
    val_m = load_manifest(cfg, "val")
    if val_m is None:
        train_m, val_m = split_manifest(train_m, cfg)
    else:
        check_class_vocabularies(train_m, val_m)
    if len(val_m) == 0:
        raise ValueError("Validation split is empty. Increase val_split or provide val_dir.")

    train_ds = ImageListDataset(train_m, train_tf, cfg.data.cache_mode, broken_log=broken_log)
    val_ds = ImageListDataset(val_m, val_tf, cfg.data.cache_mode, broken_log=broken_log)

    train_loader = make_train_loader(cfg, train_ds, train_m)
    val_loader = DataLoader(val_ds, batch_size=cfg.schedule.batch_size, shuffle=False,
                            collate_fn=collate_skip_broken, **loader_kwargs(cfg))
    return train_loader, val_loader, train_m, val_m


def extend_manifest(base: Manifest, paths: list[str], labels: list[int]) -> Manifest:
    """Append pseudo-labelled images to a manifest, preserving the class vocabulary."""
    return Manifest(
        paths=list(base.paths) + list(paths),
        labels=list(base.labels) + list(labels),
        class_names=base.class_names,
        groups=None,
    )
