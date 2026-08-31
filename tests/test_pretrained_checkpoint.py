"""`model.pretrained_checkpoint`: a new run that starts from trained weights.

The counterpart to resume: nothing but the weights may carry over. The optimizer, the
schedule, the epoch counter and best-so-far tracking all start clean, which is what lets
a backbone trained under one config move into another — new LR, new schedule, new head.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from trainlab.config import TrainConfig
from trainlab.device import resolve_runtime
from trainlab.engine import Trainer, _load_checkpoint


class _NullTracker:
    def start_run(self, **kw): return "test"
    def log_params(self, *a, **k): pass
    def log_metrics(self, *a, **k): pass
    def log_text(self, *a, **k): pass
    def log_artifact(self, *a, **k): pass
    def set_tags(self, *a, **k): pass
    def end_run(self, *a, **k): pass


def _dataset(root, classes):
    rng = np.random.default_rng(0)
    for split, n in (("train", 16), ("val", 8)):
        for ci, cname in enumerate(classes):
            d = root / split / cname
            d.mkdir(parents=True)
            for i in range(n):
                base = 40 + ci * 160
                arr = np.clip(base + rng.normal(0, 12, (32, 32, 3)), 0, 255).astype(np.uint8)
                Image.fromarray(arr).save(d / f"{i:02d}.png")
    return root


def _cfg(data_dir, **over) -> TrainConfig:
    cfg = TrainConfig()
    cfg.data.train_dir = str(data_dir / "train")
    cfg.data.val_dir = str(data_dir / "val")
    cfg.data.num_workers = 0
    cfg.input.input_size = 32
    cfg.model.backbone = "resnet18"
    cfg.model.pretrained = False
    cfg.model.ema = False
    cfg.schedule.device = "cpu"
    cfg.schedule.epochs = 2
    cfg.schedule.warmup_epochs = 0
    cfg.schedule.batch_size = 8
    cfg.checkpoint.early_stopping = False
    cfg.validation.log_confusion_matrix = False
    cfg.validation.log_worst_predictions = False
    cfg.tracking.enabled = False
    for path, val in over.items():
        section, _, name = path.partition(".")
        setattr(getattr(cfg, section), name, val)
    return cfg


@pytest.fixture
def donor(tmp_path):
    """A completed run whose checkpoints every test warm-starts from."""
    data = _dataset(tmp_path / "data", ["dark", "light"])
    cfg = _cfg(data)
    t = Trainer(cfg, resolve_runtime(cfg), _NullTracker(), tmp_path / "run1")
    t.setup()
    t.fit()
    return data, tmp_path / "run1"


def _setup(data, out, **over) -> Trainer:
    cfg = _cfg(data, **over)
    t = Trainer(cfg, resolve_runtime(cfg), _NullTracker(), out)
    t.setup()
    return t


# ------------------------------------------------------------------- weights only

def test_weights_transfer_but_nothing_else_does(donor, tmp_path):
    data, run1 = donor
    t = _setup(data, tmp_path / "warm",
               **{"model.pretrained_checkpoint": str(run1 / "last.ckpt")})
    ck = _load_checkpoint(run1 / "last.ckpt")
    for k, v in t.model.state_dict().items():
        assert torch.equal(v, ck["model"][k]), f"{k} was not transferred"
    # The run itself is new: no optimizer trajectory, no epoch offset, no inherited best.
    assert t.start_epoch == 0
    assert t.state.global_step == 0
    assert not any(t.optimizer.state.values())
    assert t.state.best_metric in (-float("inf"), float("inf"))


def test_a_run_directory_means_best_ckpt(donor, tmp_path):
    """Resume takes last.ckpt from a directory; a warm start takes best.ckpt — a new
    run wants the best weights the old one produced, not wherever it stopped."""
    data, run1 = donor
    t = _setup(data, tmp_path / "warm", **{"model.pretrained_checkpoint": str(run1)})
    best = _load_checkpoint(run1 / "best.ckpt")
    which = "ema" if best.get("best_weights") == "ema" and best.get("ema") else "model"
    for k, v in t.model.state_dict().items():
        assert torch.equal(v, best[which][k])


def test_the_head_moves_only_with_its_classes(donor, tmp_path):
    """Same class COUNT, different names: shapes match, so name+shape filtering alone
    would load a head that answers for the wrong classes."""
    data, run1 = donor
    other = _dataset(tmp_path / "other", ["cats", "dogs"])
    t = _setup(other, tmp_path / "warm",
               **{"model.pretrained_checkpoint": str(run1 / "last.ckpt")})
    ck = _load_checkpoint(run1 / "last.ckpt")
    assert not torch.equal(t.model.fc.weight, ck["model"]["fc.weight"]), \
        "a head trained on ['dark','light'] was loaded for ['cats','dogs']"
    assert torch.equal(t.model.conv1.weight, ck["model"]["conv1.weight"]), \
        "the backbone should still transfer"


def test_raw_and_wrapped_state_dicts_are_accepted(donor, tmp_path):
    data, run1 = donor
    sd = _load_checkpoint(run1 / "last.ckpt")["model"]
    torch.save(sd, tmp_path / "raw.pt")
    torch.save({"state_dict": sd}, tmp_path / "wrapped.pt")
    torch.save({f"module.{k}": v for k, v in sd.items()}, tmp_path / "ddp.pt")
    # torch.compile's wrapper — the prefix a real compiled run's checkpoint carries.
    # 782 of 782 tensors failed to match on the first real tf_efficientnetv2_s file
    # because only DDP's prefix was stripped.
    torch.save({f"_orig_mod.{k}": v for k, v in sd.items()}, tmp_path / "compiled.pt")
    for name in ("raw.pt", "wrapped.pt", "ddp.pt", "compiled.pt"):
        t = _setup(data, tmp_path / f"warm-{name}",
                   **{"model.pretrained_checkpoint": str(tmp_path / name)})
        assert torch.equal(t.model.conv1.weight, sd["conv1.weight"]), name


def test_a_different_architecture_is_refused_with_both_names(donor, tmp_path):
    data, run1 = donor
    with pytest.raises(RuntimeError, match="resnet34.*resnet18|resnet18.*resnet34"):
        _setup(data, tmp_path / "warm",
               **{"model.backbone": "resnet34",
                  "model.pretrained_checkpoint": str(run1 / "last.ckpt")})


def test_an_unreadable_payload_is_a_clear_error(donor, tmp_path):
    data, _ = donor
    torch.save({"optimizer_only": {"lr": 0.1}}, tmp_path / "junk.ckpt")
    with pytest.raises(RuntimeError, match="not a checkpoint this app can read"):
        _setup(data, tmp_path / "warm",
               **{"model.pretrained_checkpoint": str(tmp_path / "junk.ckpt")})


def test_training_proceeds_under_the_new_config(donor, tmp_path):
    """The stated use case end to end: new schedule, warm weights, fresh best."""
    data, run1 = donor
    cfg = _cfg(data, **{"model.pretrained_checkpoint": str(run1 / "last.ckpt"),
                        "schedule.epochs": 1, "optimization.lr": 1e-4})
    t = Trainer(cfg, resolve_runtime(cfg), _NullTracker(), tmp_path / "warm")
    t.setup()
    result = t.fit()
    assert [h["epoch"] for h in result["history"]] == [0]
    assert result["best_metric"] == result["history"][0][cfg.primary_metric_key]


# ------------------------------------------------------------------- config guards

def test_conflicts_are_config_errors(tmp_path):
    with pytest.raises(ValueError, match="resume_from"):
        TrainConfig.model_validate({
            "model": {"pretrained_checkpoint": "/a.ckpt"},
            "checkpoint": {"resume_from": "/b.ckpt"},
        })
    with pytest.raises(ValueError, match="SSL"):
        TrainConfig.model_validate({
            "model": {"pretrained_checkpoint": "/a.ckpt"},
            "experimental": {"ssl_method": "simsiam"},
        })
