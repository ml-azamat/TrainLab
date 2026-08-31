"""End-to-end engine tests on a tiny synthetic dataset.

Deliberately real training runs rather than mocks: the defects these cover were about
how the loop's pieces interact (metric direction driving checkpoint selection, an epoch
that takes no optimizer step, checkpoint payloads matching the score they report), and a
mocked loop would define exactly those interactions away.

Kept fast with a 32px, 2-class, 16-image dataset on CPU — a few seconds per run.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from trainlab.config import Device, Metric, TrainConfig
from trainlab.device import resolve_runtime
from trainlab.engine import Trainer, _load_checkpoint


class _NullTracker:
    """Tracker interface with no I/O — these tests are about the loop, not MLflow."""
    def start_run(self, **kw): return "test"
    def log_params(self, *a, **k): pass
    def log_metrics(self, *a, **k): pass
    def log_text(self, *a, **k): pass
    def log_artifact(self, *a, **k): pass
    def set_tags(self, *a, **k): pass
    def end_run(self, *a, **k): pass


@pytest.fixture
def tiny_data(tmp_path):
    """Two visually separable classes, so training actually moves the metric."""
    rng = np.random.default_rng(0)
    for split, n in (("train", 16), ("val", 8)):
        for ci, cname in enumerate(["dark", "light"]):
            d = tmp_path / split / cname
            d.mkdir(parents=True)
            for i in range(n):
                base = 40 if ci == 0 else 200
                arr = np.clip(base + rng.normal(0, 12, (32, 32, 3)), 0, 255).astype(np.uint8)
                Image.fromarray(arr).save(d / f"{i:02d}.png")
    return tmp_path


def _cfg(tiny_data, **over) -> TrainConfig:
    cfg = TrainConfig()
    cfg.data.train_dir = str(tiny_data / "train")
    cfg.data.val_dir = str(tiny_data / "val")
    cfg.data.num_workers = 0
    cfg.input.input_size = 32
    cfg.model.backbone = "resnet18"
    cfg.model.pretrained = False
    cfg.model.ema = False
    cfg.schedule.device = Device.CPU
    cfg.schedule.epochs = 3
    cfg.schedule.warmup_epochs = 0
    cfg.schedule.batch_size = 8
    cfg.checkpoint.early_stopping = False
    cfg.validation.log_confusion_matrix = False
    cfg.validation.log_worst_predictions = False
    cfg.tracking.enabled = False
    for path, val in over.items():
        section, _, field = path.partition(".")
        setattr(getattr(cfg, section), field, val)
    return cfg


def _train(cfg, out_dir) -> tuple[Trainer, dict]:
    rt = resolve_runtime(cfg)
    t = Trainer(cfg, rt, _NullTracker(), out_dir)
    t.setup()
    return t, t.fit()


# ------------------------------------------------------------------ metric direction

def test_lower_is_better_metric_selects_the_lowest_epoch(tiny_data, tmp_path):
    """With ECE as the primary metric, `improved = score > best` saved the WORST-
    calibrated epoch as best.ckpt and reported it as the run's result."""
    # es_min_delta=0: the loop only crowns an epoch that improves by MORE than the
    # delta, while this test asserts exact-min selection — with the 0.001 default, an
    # epoch that wins by less sits inside the dead zone and the assertion flakes on
    # numeric noise. Zeroing it makes the test state the rule it actually verifies:
    # direction, not dead-zone width.
    cfg = _cfg(tiny_data, **{"validation.primary_metric": Metric.ECE,
                             "validation.metrics": [Metric.ACC1, Metric.ECE],
                             "checkpoint.es_min_delta": 0.0})
    out = tmp_path / "run"
    trainer, result = _train(cfg, out)

    per_epoch = [h["ece"] for h in result["history"]]
    assert result["best_metric"] == pytest.approx(min(per_epoch)), (
        f"reported best {result['best_metric']} is not the lowest of {per_epoch}")
    assert result["best_epoch"] == int(np.argmin(per_epoch))
    assert (out / "best.ckpt").exists()

    ck = _load_checkpoint(out / "best.ckpt")
    assert ck["score"] == pytest.approx(min(per_epoch))
    assert ck["primary_metric"] == "ece"


def test_higher_is_better_metric_still_selects_the_highest(tiny_data, tmp_path):
    cfg = _cfg(tiny_data, **{"validation.primary_metric": Metric.ACC1,
                             "validation.metrics": [Metric.ACC1]})
    out = tmp_path / "run"
    _, result = _train(cfg, out)
    per_epoch = [h["acc@1"] for h in result["history"]]
    assert result["best_metric"] == pytest.approx(max(per_epoch))


def test_top_k_retention_keeps_the_best_end_for_ece(tiny_data, tmp_path):
    cfg = _cfg(tiny_data, **{"validation.primary_metric": Metric.ECE,
                             "validation.metrics": [Metric.ACC1, Metric.ECE],
                             "checkpoint.save_top_k": 2, "schedule.epochs": 4})
    out = tmp_path / "run"
    _, result = _train(cfg, out)

    # Filenames carry the score to 4 dp, so compare at that precision.
    kept = sorted(float(p.stem.split("-", 1)[1]) for p in out.glob("epoch*.ckpt"))
    all_scores = sorted(round(h["ece"], 4) for h in result["history"])
    assert kept == all_scores[:2], (
        f"kept {kept}, but the two best (lowest) ECEs were {all_scores[:2]}")
    assert max(kept) < all_scores[-1], "a worse checkpoint was retained over a better one"


# ------------------------------------------------------------------ refusal to start

def test_unavailable_primary_metric_is_refused(tiny_data, tmp_path):
    """acc@5 on 2 classes used to score -inf every epoch: nothing ever improved, no
    best.ckpt was written, and the run exited 0 reporting `best acc@5 = -inf`."""
    cfg = _cfg(tiny_data, **{"validation.primary_metric": Metric.ACC5})
    rt = resolve_runtime(cfg)
    t = Trainer(cfg, rt, _NullTracker(), tmp_path / "run")
    with pytest.raises(ValueError, match="cannot be computed"):
        t.setup()


def test_epoch_with_no_optimizer_step_fails_loudly(tiny_data, tmp_path):
    """grad_accum larger than the number of batches meant `(i+1) % accum == 0` never
    fired: the model never trained, and the epoch still reported a finite loss."""
    cfg = _cfg(tiny_data, **{"schedule.batch_size": 8, "optimization.grad_accum_steps": 50})
    rt = resolve_runtime(cfg)
    t = Trainer(cfg, rt, _NullTracker(), tmp_path / "run")
    t.setup()
    with pytest.raises(RuntimeError, match="no optimizer step"):
        t.train_epoch(0)


# --------------------------------------------------------------------- checkpoints

def test_checkpoint_records_which_weights_the_score_belongs_to(tiny_data, tmp_path):
    """Consumers preferred `ema` whenever it existed, even when the reported score was
    measured on the raw weights."""
    cfg = _cfg(tiny_data, **{"model.ema": True, "validation.eval_ema_weights": True})
    out = tmp_path / "run"
    _train(cfg, out)
    ck = _load_checkpoint(out / "best.ckpt")
    assert ck["best_weights"] in ("model", "ema")
    assert ck[ck["best_weights"]] is not None


def test_safe_checkpoint_loading_is_the_default(tmp_path, monkeypatch):
    """`weights_only=False` executes arbitrary pickle opcodes from whoever wrote the
    file. Our own checkpoints must load without it."""
    p = tmp_path / "c.ckpt"
    torch.save({"model": {"w": torch.zeros(2)}, "config": {"a": 1},
                "class_names": ["x", "y"]}, p)
    ck = _load_checkpoint(p)
    assert ck["class_names"] == ["x", "y"]


def test_unsafe_checkpoint_needs_an_explicit_opt_in(tmp_path, monkeypatch):
    class Evil:
        def __reduce__(self):
            return (print, ("pwned",))

    p = tmp_path / "evil.ckpt"
    torch.save({"model": Evil()}, p)

    monkeypatch.delenv("TRAINLAB_ALLOW_UNSAFE_CHECKPOINTS", raising=False)
    with pytest.raises(RuntimeError, match="safe unpickler"):
        _load_checkpoint(p)

    monkeypatch.setenv("TRAINLAB_ALLOW_UNSAFE_CHECKPOINTS", "1")
    _load_checkpoint(p)          # allowed, loudly


def test_kd_teacher_with_different_classes_is_refused(tiny_data, tmp_path):
    """Distilling across mismatched class vocabularies aligns the wrong logits."""
    cfg = _cfg(tiny_data)
    teacher = tmp_path / "teacher.ckpt"
    torch.save({"model": {}, "config": cfg.model_dump(mode="json"),
                "class_names": ["totally", "different", "classes"]}, teacher)

    cfg.experimental.kd_enabled = True
    cfg.experimental.kd_teacher_ckpt = str(teacher)
    rt = resolve_runtime(cfg)
    t = Trainer(cfg, rt, _NullTracker(), tmp_path / "run")
    with pytest.raises(ValueError, match="different\n?\\s*classes|different classes"):
        t.setup()


# --------------------------------------------------------------------------- amp/sam

def test_sam_runs_with_grad_clipping(tiny_data, tmp_path):
    """The SAM branch bypassed the GradScaler and called `unscale_` in a way that
    conflicted with the descent step. It must complete a full epoch."""
    cfg = _cfg(tiny_data, **{"optimization.sam": True,
                             "optimization.grad_clip_norm": 1.0,
                             "schedule.epochs": 1})
    _, result = _train(cfg, tmp_path / "run")
    assert result["history"], "SAM run produced no evaluated epoch"
    assert np.isfinite(result["best_metric"])


def test_grad_accumulation_produces_the_expected_step_count(tiny_data, tmp_path):
    cfg = _cfg(tiny_data, **{"schedule.batch_size": 4, "optimization.grad_accum_steps": 2,
                             "schedule.epochs": 1})
    rt = resolve_runtime(cfg)
    t = Trainer(cfg, rt, _NullTracker(), tmp_path / "run")
    t.setup()
    n_batches = len(t.train_loader)
    t.train_epoch(0)
    assert t.state.global_step == n_batches // 2


# ------------------------------------------------------------------ epoch boundaries

def test_mixup_off_epoch_switch_completes_the_run(tiny_data, tmp_path):
    """`mixup_off_epoch` disables mixing for the final epochs while the auto-switched
    SoftTargetCrossEntropy stays installed. The integer targets those epochs produce
    used to crash the run at the switch (or, when batch_size happened to equal
    num_classes, broadcast silently into a garbage loss)."""
    cfg = _cfg(tiny_data, **{"schedule.epochs": 3})
    cfg.augmentation.mixup_alpha = 0.2
    cfg.augmentation.cutmix_alpha = 0.0
    cfg.augmentation.mixup_off_epoch = 2       # epochs 2+ train on integer targets
    _, result = _train(cfg, tmp_path / "run")
    assert len(result["history"]) == 3
    assert all(np.isfinite(h["train_loss"]) for h in result["history"])


def test_soft_target_ce_int_targets_match_smoothed_ce():
    """The integer-target fallback must apply the label smoothing the mixup collator
    would otherwise have baked in — same value as F.cross_entropy(label_smoothing=)."""
    import torch.nn.functional as F

    from trainlab.losses import SoftTargetCrossEntropy

    torch.manual_seed(0)
    x = torch.randn(8, 5)
    y = torch.randint(0, 5, (8,))
    for eps in (0.0, 0.1):
        ours = SoftTargetCrossEntropy(smoothing=eps)(x, y)
        ref = F.cross_entropy(x, y, label_smoothing=eps)
        assert torch.allclose(ours, ref, atol=1e-6), f"eps={eps}: {ours} vs {ref}"


def test_progressive_resizing_actually_changes_batch_resolution(tiny_data, tmp_path):
    """With persistent workers, mutating dataset.transform never reached the worker
    processes: the log said 'progressive resize -> 64px' while every batch stayed at
    the start size for the whole run. The loader must be rebuilt on a size change, and
    this test asserts on the tensors the model actually receives, with real workers."""
    import types

    cfg = _cfg(tiny_data, **{"schedule.epochs": 2, "input.input_size": 64})
    cfg.schedule.progressive_resizing = True
    cfg.schedule.progressive_start_size = 32
    cfg.schedule.progressive_end_epoch = 0.75   # epoch 1 of 2 reaches full size
    cfg.data.num_workers = 2
    cfg.data.persistent_workers = True

    rt = resolve_runtime(cfg)
    t = Trainer(cfg, rt, _NullTracker(), tmp_path / "run")
    t.setup()

    seen: list[set[int]] = [set(), set()]
    epoch_box = {"i": 0}
    orig = t._forward_loss

    def spy(self, x, y):
        seen[epoch_box["i"]].add(x.shape[-1])
        return orig(x, y)

    t._forward_loss = types.MethodType(spy, t)
    for epoch in range(2):
        epoch_box["i"] = epoch
        t.train_epoch(epoch)

    assert seen[0] == {32}, f"epoch 0 should train at the start size, saw {seen[0]}"
    assert seen[1] == {64}, f"epoch 1 should train at full size, saw {seen[1]}"


def test_progressive_resizing_downgrades_for_patch_models(tiny_data, tmp_path):
    """ViT/Swin bake the input resolution into patch_embed at construction; feeding
    them the ramp is a mid-run shape error. The engine must disable the ramp loudly
    instead of crashing at epoch 1."""
    cfg = _cfg(tiny_data, **{"input.input_size": 32, "schedule.epochs": 1})
    cfg.model.backbone = "vit_tiny_patch16_224"
    cfg.input.input_size = 224
    cfg.schedule.progressive_resizing = True
    cfg.schedule.progressive_start_size = 160
    rt = resolve_runtime(cfg)
    t = Trainer(cfg, rt, _NullTracker(), tmp_path / "run")
    t.setup()
    assert t.progressive_enabled is False


# ----------------------------------------------------------------------------- SWA

def test_swa_logs_nothing_when_training_ends_before_the_phase(tiny_data, tmp_path):
    """Early stopping before swa_start_epoch left the SWA module holding the
    pre-training deepcopy from setup(); evaluating it logged plausible-looking swa/*
    metrics that described the initialization."""
    logged: list[str] = []

    class SpyTracker(_NullTracker):
        def log_metrics(self, metrics, step=None):
            logged.extend(metrics)

    cfg = _cfg(tiny_data, **{"schedule.epochs": 2})
    cfg.experimental.swa = True
    cfg.experimental.swa_start_epoch = 0.95    # phase would begin at epoch 1.9: never
    rt = resolve_runtime(cfg)
    t = Trainer(cfg, rt, SpyTracker(), tmp_path / "run")
    t.setup()
    t.fit()
    assert t.swa.n == 0
    assert not [k for k in logged if k.startswith("swa/")], (
        "swa/* metrics were logged although no checkpoint was ever averaged")


# ---------------------------------------------------------------------- SSL end-to-end

def test_ssl_run_end_to_end_produces_a_json_safe_result(tiny_data, tmp_path):
    """The full SSL -> supervised pipeline with DEFAULT settings. `result` crosses a
    JSON boundary (the finished event and result.json); SSLResult.encoder_path is a
    Path, and left raw it crashed `emit("finished")` AFTER all training work was done —
    every SSL run with the default ssl_save_encoder=true ended status FAILED."""
    import json

    cfg = _cfg(tiny_data, **{"schedule.epochs": 1})
    cfg.experimental.ssl_method = "simsiam"
    cfg.experimental.ssl_epochs = 1
    cfg.experimental.ssl_lr_warmup_epochs = 0
    cfg.experimental.ssl_batch_size = 8
    cfg.experimental.ssl_amp = "off"
    assert cfg.experimental.ssl_save_encoder is True   # the default that used to crash

    out = tmp_path / "run"
    _, result = _train(cfg, out)

    json.dumps({k: v for k, v in result.items() if k != "history"})   # must not raise
    assert result["ssl"]["method"] == "simsiam"
    assert isinstance(result["ssl"]["encoder_path"], str)
    assert (out / "ssl_simsiam_encoder.pt").exists()
