"""`checkpoint.resume_from`: real save → real resume on the tiny synthetic dataset.

The field existed, the form rendered it and the tooltip promised restoration, while the
engine never read it — a run configured to resume silently trained from scratch. These
tests pin the whole contract: what continues (weights, EMA, optimizer, schedule position,
best-so-far tracking), where the loop restarts, and what is refused loudly.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from trainlab.config import TrainConfig
from trainlab.device import resolve_runtime
from trainlab.engine import Trainer, _load_checkpoint
from trainlab.sched import LRSchedule


class _NullTracker:
    def start_run(self, **kw): return "test"
    def log_params(self, *a, **k): pass
    def log_metrics(self, *a, **k): pass
    def log_text(self, *a, **k): pass
    def log_artifact(self, *a, **k): pass
    def set_tags(self, *a, **k): pass
    def end_run(self, *a, **k): pass


@pytest.fixture
def tiny_data(tmp_path):
    rng = np.random.default_rng(0)
    for split, n in (("train", 16), ("val", 8)):
        for ci, cname in enumerate(["dark", "light"]):
            d = tmp_path / "data" / split / cname
            d.mkdir(parents=True)
            for i in range(n):
                base = 40 if ci == 0 else 200
                arr = np.clip(base + rng.normal(0, 12, (32, 32, 3)), 0, 255).astype(np.uint8)
                Image.fromarray(arr).save(d / f"{i:02d}.png")
    return tmp_path / "data"


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


def _train(cfg, out_dir) -> tuple[Trainer, dict]:
    t = Trainer(cfg, resolve_runtime(cfg), _NullTracker(), out_dir)
    t.setup()
    return t, t.fit()


@pytest.fixture
def first_run(tiny_data, tmp_path):
    """Two completed epochs whose last.ckpt every test resumes from."""
    cfg = _cfg(tiny_data)
    t, result = _train(cfg, tmp_path / "run1")
    return tiny_data, tmp_path / "run1", t, result


# ------------------------------------------------------------------- the happy path

def test_resume_continues_instead_of_restarting(first_run, tmp_path):
    tiny_data, run1, t1, r1 = first_run
    cfg = _cfg(tiny_data, **{"schedule.epochs": 4,
                             "checkpoint.resume_from": str(run1 / "last.ckpt")})
    t2, r2 = _train(cfg, tmp_path / "run2")

    assert t2.start_epoch == 2
    # Only epochs 2 and 3 were trained here; 0 and 1 travelled in via the history.
    assert [h["epoch"] for h in r2["history"]] == [0, 1, 2, 3]
    # The step counter kept counting: 4 steps/epoch (16 img/class / batch 8) × 4 epochs.
    assert t2.state.global_step == t1.state.global_step + 2 * t1.steps_per_epoch


def test_resumed_weights_are_the_checkpoints_weights(first_run, tmp_path):
    """The original bug shape: configured to resume, trained from scratch."""
    tiny_data, run1, _, _ = first_run
    ck = _load_checkpoint(run1 / "last.ckpt")
    cfg = _cfg(tiny_data, **{"schedule.epochs": 3,
                             "checkpoint.resume_from": str(run1 / "last.ckpt")})
    t = Trainer(cfg, resolve_runtime(cfg), _NullTracker(), tmp_path / "run2")
    t.setup()          # restore happens here; fit() would change the weights again
    for k, v in t.model.state_dict().items():
        assert torch.equal(v, ck["model"][k]), f"{k} was not restored"


def test_optimizer_momentum_survives_the_resume(first_run, tmp_path):
    tiny_data, run1, _, _ = first_run
    cfg = _cfg(tiny_data, **{"schedule.epochs": 3,
                             "checkpoint.resume_from": str(run1 / "last.ckpt")})
    t = Trainer(cfg, resolve_runtime(cfg), _NullTracker(), tmp_path / "run2")
    t.setup()
    states = [s for g in t.optimizer.state.values() for s in [g] if s]
    assert states, "optimizer state is empty — the trajectory restarted from zero"


def test_a_run_directory_is_accepted_and_means_last_ckpt(first_run, tmp_path):
    tiny_data, run1, _, _ = first_run
    cfg = _cfg(tiny_data, **{"schedule.epochs": 3,
                             "checkpoint.resume_from": str(run1)})
    t = Trainer(cfg, resolve_runtime(cfg), _NullTracker(), tmp_path / "run2")
    t.setup()
    assert t.start_epoch == 2


def test_best_so_far_tracking_continues(first_run, tmp_path):
    """A resumed run that never improves must still report the ORIGINAL best, not
    crown its own first epoch by comparing against ±inf."""
    tiny_data, run1, t1, r1 = first_run
    cfg = _cfg(tiny_data, **{"schedule.epochs": 3, "optimization.lr": 1e-12,
                             "checkpoint.resume_from": str(run1 / "last.ckpt")})
    t2, r2 = _train(cfg, tmp_path / "run2")
    assert r2["best_metric"] == pytest.approx(r1["best_metric"], abs=0.15)
    assert t2.state.best_metric != -float("inf")


def test_ema_resumes_from_the_checkpoint_not_from_init(first_run, tmp_path):
    tiny_data, run1, _, _ = first_run
    # The first run had no EMA; resuming with EMA on must seed it from the RESUMED
    # weights rather than the fresh initialisation the module was deep-copied from.
    cfg = _cfg(tiny_data, **{"schedule.epochs": 3, "model.ema": True,
                             "checkpoint.resume_from": str(run1 / "last.ckpt")})
    t = Trainer(cfg, resolve_runtime(cfg), _NullTracker(), tmp_path / "run2")
    t.setup()
    ck = _load_checkpoint(run1 / "last.ckpt")
    for k, v in t.ema.module.state_dict().items():
        assert torch.equal(v.cpu(), ck["model"][k]), f"EMA {k} still holds init weights"


def test_schedule_position_is_restored(first_run, tmp_path):
    """The LR must continue down the cosine, not restart at the top of warmup."""
    tiny_data, run1, t1, _ = first_run
    cfg = _cfg(tiny_data, **{"schedule.epochs": 4,
                             "checkpoint.resume_from": str(run1 / "last.ckpt")})
    t = Trainer(cfg, resolve_runtime(cfg), _NullTracker(), tmp_path / "run2")
    t.setup()
    resumed_factor = t.schedule.factor(t.state.global_step)
    fresh_factor = t.schedule.factor(0)
    assert resumed_factor < fresh_factor, "schedule restarted from step 0"


# ------------------------------------------------------------------- refusals

def test_an_exhausted_schedule_is_refused(first_run, tmp_path):
    """Resuming a finished run with unchanged epochs used to be the silent no-train
    case; now it says what to change."""
    tiny_data, run1, _, _ = first_run
    cfg = _cfg(tiny_data, **{"schedule.epochs": 2,
                             "checkpoint.resume_from": str(run1 / "last.ckpt")})
    t = Trainer(cfg, resolve_runtime(cfg), _NullTracker(), tmp_path / "run2")
    with pytest.raises(ValueError, match="Raise schedule.epochs"):
        t.setup()


def test_a_class_mismatch_is_refused(first_run, tmp_path):
    tiny_data, run1, _, _ = first_run
    other = tiny_data.parent / "other"
    for split in ("train", "val"):
        for cname in ("cats", "dogs"):          # different names, same count
            src = tiny_data / split / ("dark" if cname == "cats" else "light")
            dst = other / split / cname
            dst.mkdir(parents=True)
            for f in src.iterdir():
                dst.joinpath(f.name).write_bytes(f.read_bytes())
    cfg = _cfg(other, **{"schedule.epochs": 3,
                         "checkpoint.resume_from": str(run1 / "last.ckpt")})
    t = Trainer(cfg, resolve_runtime(cfg), _NullTracker(), tmp_path / "run2")
    with pytest.raises(ValueError, match="classes"):
        t.setup()


def test_a_different_backbone_is_refused_by_name(first_run, tmp_path):
    tiny_data, run1, _, _ = first_run
    cfg = _cfg(tiny_data, **{"schedule.epochs": 3, "model.backbone": "resnet34",
                             "checkpoint.resume_from": str(run1 / "last.ckpt")})
    t = Trainer(cfg, resolve_runtime(cfg), _NullTracker(), tmp_path / "run2")
    with pytest.raises(RuntimeError, match="resnet18.*resnet34|resnet34.*resnet18"):
        t.setup()


def test_a_missing_path_is_a_clear_error(tiny_data, tmp_path):
    cfg = _cfg(tiny_data, **{"checkpoint.resume_from": str(tmp_path / "nope.ckpt")})
    t = Trainer(cfg, resolve_runtime(cfg), _NullTracker(), tmp_path / "run")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        t.setup()


# ------------------------------------------------------------------- compatibility

def test_a_checkpoint_from_before_resume_existed_still_resumes(first_run, tmp_path):
    """Older payloads carry no run_state/scaler/sched keys; weights, optimizer and the
    epoch are still enough, with the step derived from the epoch."""
    tiny_data, run1, t1, _ = first_run
    ck = _load_checkpoint(run1 / "last.ckpt")
    for key in ("run_state", "scaler", "sched"):
        ck.pop(key, None)
    legacy = tmp_path / "legacy.ckpt"
    torch.save(ck, legacy)

    cfg = _cfg(tiny_data, **{"schedule.epochs": 3,
                             "checkpoint.resume_from": str(legacy)})
    t = Trainer(cfg, resolve_runtime(cfg), _NullTracker(), tmp_path / "run2")
    t.setup()
    assert t.start_epoch == 2
    assert t.state.global_step == 2 * t1.steps_per_epoch      # derived, not zero


def test_the_payload_stays_safe_to_load(first_run):
    """Everything resume adds must survive torch's safe unpickler — one numpy scalar in
    the history and every resume needs TRAINLAB_ALLOW_UNSAFE_CHECKPOINTS=1."""
    _, run1, _, _ = first_run
    ck = torch.load(run1 / "last.ckpt", map_location="cpu", weights_only=True)
    assert "run_state" in ck and "sched" in ck
    for h in ck["run_state"]["history"]:
        assert all(isinstance(v, (int, float, str)) for v in h.values())


def test_plateau_factor_round_trips():
    cfg = TrainConfig.model_validate({"schedule": {"scheduler": "plateau"}})
    s1 = LRSchedule(cfg, steps_per_epoch=10)
    s1.on_plateau(improved=False)
    s1.on_plateau(improved=False)
    s2 = LRSchedule(cfg, steps_per_epoch=10)
    s2.load_state_dict(s1.state_dict())
    assert s2.factor(1000) == s1.factor(1000) != 1.0
