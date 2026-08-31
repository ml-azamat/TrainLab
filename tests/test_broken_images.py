"""Unreadable images: skipped, listed by path, and refused past a tolerance."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trainlab.config import CacheMode, TrainConfig  # noqa: E402
from trainlab.data import (  # noqa: E402
    ImageListDataset, Manifest, UnlabeledDataset, collate_skip_broken,
    read_broken_images, record_broken_image,
)


def _make(tmp_path: Path, good: int = 3, truncated: int = 0, missing: int = 0) -> Manifest:
    """A manifest mixing readable files with the two ways a read actually fails."""
    paths, labels = [], []
    for i in range(good):
        p = tmp_path / f"good{i}.jpg"
        Image.new("RGB", (32, 32), (i * 40, 90, 120)).save(p)
        paths.append(str(p)); labels.append(i % 2)
    for i in range(truncated):
        src = tmp_path / f"src{i}.jpg"
        Image.new("RGB", (64, 64), (200, 30, 30)).save(src)
        blob = src.read_bytes()
        p = tmp_path / f"cut{i}.jpg"
        p.write_bytes(blob[: len(blob) // 2])       # a real half-written JPEG
        src.unlink()
        paths.append(str(p)); labels.append(0)
    for i in range(missing):
        paths.append(str(tmp_path / f"gone{i}.jpg")); labels.append(1)
    return Manifest(paths, labels, ["a", "b"])


def _tensor(img):
    import numpy as np
    return torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float()


def test_a_truncated_file_is_dropped_instead_of_raising(tmp_path):
    """The whole point: the run does not die on a file nobody can name in advance."""
    log = tmp_path / "broken.tsv"
    ds = ImageListDataset(_make(tmp_path, good=2, truncated=1), _tensor,
                          CacheMode.NONE, broken_log=log)
    items = [ds[i] for i in range(len(ds))]
    assert items[2] is None
    assert [i is None for i in items] == [False, False, True]


def test_a_missing_file_is_dropped_the_same_way(tmp_path):
    log = tmp_path / "broken.tsv"
    ds = ImageListDataset(_make(tmp_path, good=1, missing=1), _tensor,
                          CacheMode.NONE, broken_log=log)
    assert ds[0] is not None and ds[1] is None
    assert "FileNotFoundError" in read_broken_images(log)[0][1]


def test_the_failing_paths_are_recorded_by_name(tmp_path):
    """'I don't know which ones' is the actual problem; the log is the answer to it."""
    log = tmp_path / "broken.tsv"
    m = _make(tmp_path, good=1, truncated=2)
    ds = ImageListDataset(m, _tensor, CacheMode.NONE, broken_log=log)
    for i in range(len(ds)):
        ds[i]
    recorded = dict(read_broken_images(log))
    assert set(recorded) == {m.paths[1], m.paths[2]}
    assert all(err for err in recorded.values())


def test_the_log_is_deduplicated_across_epochs(tmp_path):
    """Workers re-read the same file every epoch; the user needs the set, not the tally."""
    log = tmp_path / "broken.tsv"
    ds = ImageListDataset(_make(tmp_path, good=0, truncated=1), _tensor,
                          CacheMode.NONE, broken_log=log)
    for _ in range(4):
        ds[0]
    assert len(read_broken_images(log)) == 1


def test_recording_never_raises_even_when_the_log_cannot_be_written(tmp_path):
    """A diagnostic that can end the run is worse than no diagnostic."""
    record_broken_image(tmp_path / "no" / "such" / "dir" / "x.tsv", "/a.jpg", OSError("x"))
    record_broken_image(None, "/a.jpg", OSError("x"))


def test_collate_drops_the_broken_and_keeps_the_batch(tmp_path):
    batch = [(torch.zeros(3, 4, 4), 0, 0), None, (torch.ones(3, 4, 4), 1, 2)]
    x, y, idx = collate_skip_broken(batch)
    assert x.shape[0] == 2 and y.tolist() == [0, 1] and idx.tolist() == [0, 2]


def test_a_wholly_unreadable_batch_becomes_none(tmp_path):
    """`None` rather than an empty tensor: zero images through a model is a crash or a
    silent NaN, and every loop in the engine checks for it."""
    assert collate_skip_broken([None, None]) is None


def test_a_dataset_of_readable_images_is_unaffected(tmp_path):
    log = tmp_path / "broken.tsv"
    ds = ImageListDataset(_make(tmp_path, good=4), _tensor, CacheMode.NONE, broken_log=log)
    batch = collate_skip_broken([ds[i] for i in range(4)])
    assert batch is not None and batch[0].shape[0] == 4
    assert read_broken_images(log) == []


def test_the_unlabeled_dataset_is_guarded_too(tmp_path):
    """SSL pretraining reads the same files through a different dataset."""
    log = tmp_path / "broken.tsv"
    m = _make(tmp_path, good=1, truncated=1)
    ds = UnlabeledDataset(m.paths, _tensor, CacheMode.NONE, broken_log=log)
    assert ds[0] is not None and ds[1] is None
    assert len(read_broken_images(log)) == 1


def test_a_loader_over_broken_data_still_yields_usable_batches(tmp_path):
    """End to end through a real DataLoader with workers, which is where the exception
    used to surface — in the main process, hours in, naming one file."""
    from torch.utils.data import DataLoader

    log = tmp_path / "broken.tsv"
    ds = ImageListDataset(_make(tmp_path, good=6, truncated=3), _tensor,
                          CacheMode.NONE, broken_log=log)
    seen = 0
    for batch in DataLoader(ds, batch_size=3, num_workers=2,
                            collate_fn=collate_skip_broken):
        if batch is not None:
            seen += batch[0].shape[0]
    assert seen == 6
    assert len(read_broken_images(log)) == 3       # written from the worker processes


def test_the_tolerance_is_configurable_and_defaults_low():
    assert TrainConfig().data.broken_image_tolerance == pytest.approx(0.02)
    assert TrainConfig.model_validate(
        {"data": {"broken_image_tolerance": 0.5}}).data.broken_image_tolerance == 0.5
    with pytest.raises(ValueError):
        TrainConfig.model_validate({"data": {"broken_image_tolerance": 1.5}})
