"""Regression tests for the silent-failure bugs.

Every test here corresponds to a defect that produced a plausible-looking but wrong
number, or hid a real failure behind a successful-looking run. They are grouped by the
module that owns the invariant, and each names the symptom it prevents so a future
change that reintroduces it fails with an explanation rather than a bare assert.

Nothing here needs a dataset or a GPU: image fixtures are generated in a tmp_path.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from trainlab import data as data_mod
from trainlab import losses, metrics as metrics_mod
from trainlab.config import (
    LOWER_IS_BETTER_KEYS, Metric, TrainConfig, load_config_leniently,
    metric_higher_is_better, ui_expression_identifiers, validate_ui_expressions,
)


# --------------------------------------------------------------------------------- utils

def _make_dataset(root, classes: dict[str, int], size: int = 32) -> None:
    """Create an ImageFolder tree: {class_name: n_images}."""
    rng = np.random.default_rng(0)
    for cname, n in classes.items():
        d = root / cname
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            arr = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
            Image.fromarray(arr).save(d / f"{i:03d}.png")


def _to_tensor(img: Image.Image) -> torch.Tensor:
    """Minimal transform so batches collate; the pixels are irrelevant here."""
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)


def _cfg(**over) -> TrainConfig:
    cfg = TrainConfig()
    for path, val in over.items():
        section, _, field = path.partition(".")
        setattr(getattr(cfg, section), field, val)
    return cfg


# ------------------------------------------------------------------- metric direction

def test_ece_is_lower_is_better():
    """Selecting ECE as the primary metric used to save the WORST-calibrated epoch."""
    assert Metric.ECE.higher_is_better is False
    assert Metric.ACC1.higher_is_better is True
    assert Metric.MACRO_F1.higher_is_better is True


@pytest.mark.parametrize("key,expected", [
    ("acc@1", True), ("acc_at_1", True), ("macro-F1", True), ("auroc", True),
    ("val_loss", False), ("train_loss", False), ("ece", False),
    ("ema/val_loss", False), ("ema/acc_at_1", True), ("crt/ece", False),
])
def test_metric_direction_handles_prefixes_and_sanitised_keys(key, expected):
    """The tracker sanitises `@`->`_at_` and prefixes stage metrics; one rule must
    cover the engine, the sweep driver and the comparison view."""
    assert metric_higher_is_better(key) is expected


def test_every_lower_is_better_key_is_reachable_from_the_enum_or_logged():
    """Guards against the frontend's mirror of this set drifting out of step."""
    assert "ece" in LOWER_IS_BETTER_KEYS
    assert Metric.ECE.value in LOWER_IS_BETTER_KEYS


def test_checkpoint_retention_keeps_the_best_end_for_a_lower_is_better_metric():
    """`sort(key=-score)` retained exactly the K WORST checkpoints for ECE."""
    scores = [0.30, 0.05, 0.12, 0.40]
    for maximize, expected in [(True, [0.40, 0.30]), (False, [0.05, 0.12])]:
        ranked = sorted(scores, reverse=maximize)[:2]
        assert ranked == expected


# ----------------------------------------------------------------------- data loading

def test_train_val_class_mismatch_is_refused(tmp_path):
    """The headline silent bug: same class COUNT, different classes.

    Label indices come from sorting each directory independently, so validation targets
    silently referred to different classes and the run reported a believable accuracy.
    """
    _make_dataset(tmp_path / "train", {"a": 2, "b": 2, "c": 2})
    _make_dataset(tmp_path / "val", {"a": 2, "b": 2, "d": 2})
    train_m = data_mod._scan_imagefolder(tmp_path / "train")
    val_m = data_mod._scan_imagefolder(tmp_path / "val")

    with pytest.raises(ValueError, match="different class vocabularies"):
        data_mod.check_class_vocabularies(train_m, val_m)


def test_train_val_class_mismatch_names_the_offending_classes(tmp_path):
    _make_dataset(tmp_path / "train", {"a": 2, "b": 2, "c": 2})
    _make_dataset(tmp_path / "val", {"a": 2, "b": 2})
    train_m = data_mod._scan_imagefolder(tmp_path / "train")
    val_m = data_mod._scan_imagefolder(tmp_path / "val")
    with pytest.raises(ValueError) as e:
        data_mod.check_class_vocabularies(train_m, val_m)
    assert "'c'" in str(e.value) or "['c']" in str(e.value)


def test_identical_vocabularies_pass(tmp_path):
    _make_dataset(tmp_path / "train", {"a": 2, "b": 2})
    _make_dataset(tmp_path / "val", {"a": 1, "b": 1})
    data_mod.check_class_vocabularies(
        data_mod._scan_imagefolder(tmp_path / "train"),
        data_mod._scan_imagefolder(tmp_path / "val"),
    )


def test_dataset_smaller_than_one_batch_still_yields_batches(tmp_path):
    """`drop_last=True` dropped every batch, so the model never trained and the run
    still reported `loss 0.0000` and a plausible accuracy from the untouched backbone."""
    _make_dataset(tmp_path / "train", {"a": 10, "b": 10})
    m = data_mod._scan_imagefolder(tmp_path / "train")
    cfg = _cfg(**{"schedule.batch_size": 64, "data.num_workers": 0})
    ds = data_mod.ImageListDataset(m, transform=_to_tensor)
    loader = data_mod.make_train_loader(cfg, ds, m)
    assert len(list(loader)) > 0, "a dataset smaller than one batch must still train"


def test_full_batches_still_drop_the_ragged_tail(tmp_path):
    """The small-dataset fix must not disable drop_last where it earns its keep."""
    _make_dataset(tmp_path / "train", {"a": 50, "b": 50})
    m = data_mod._scan_imagefolder(tmp_path / "train")
    cfg = _cfg(**{"schedule.batch_size": 32, "data.num_workers": 0})
    ds = data_mod.ImageListDataset(m, transform=_to_tensor)
    loader = data_mod.make_train_loader(cfg, ds, m)
    assert len(loader) == 100 // 32
    assert all(len(b[1]) == 32 for b in loader)


def test_single_class_directory_is_rejected(tmp_path):
    """Used to surface as a raw torchmetrics error after the model had been built."""
    _make_dataset(tmp_path / "train", {"only": 4})
    with pytest.raises(ValueError, match="only one class"):
        data_mod._scan_imagefolder(tmp_path / "train")


def test_balanced_batch_sampler_fills_the_requested_batch_size(tmp_path):
    """batch_size // n_classes left a remainder: 64 over 10 classes gave batches of 60."""
    labels = [i % 10 for i in range(1000)]
    s = data_mod.BalancedBatchSampler(labels, batch_size=64, seed=0)
    batches = list(s)
    assert batches, "sampler produced no batches"
    assert all(len(b) == 64 for b in batches), {len(b) for b in batches}


def test_balanced_batch_sampler_stays_balanced():
    labels = [0] * 900 + [1] * 100
    s = data_mod.BalancedBatchSampler(labels, batch_size=64, seed=0)
    first = next(iter(s))
    counts = np.bincount([labels[i] for i in first], minlength=2)
    # Perfect balance is impossible at 64/2 with a top-up, but it must be far closer
    # than the 9:1 the underlying distribution would give.
    assert min(counts) >= 24, counts


def test_ram_cache_forces_single_process_loading():
    """Each worker forks its own cache: memory x num_workers, parent cache never fills."""
    from trainlab.config import CacheMode

    cfg = _cfg(**{"data.cache_mode": CacheMode.RAM, "data.num_workers": 8})
    assert data_mod.loader_kwargs(cfg)["num_workers"] == 0
    cfg2 = _cfg(**{"data.cache_mode": CacheMode.NONE, "data.num_workers": 8})
    assert data_mod.loader_kwargs(cfg2)["num_workers"] == 8


def test_fingerprint_distinguishes_same_named_files_in_different_classes(tmp_path):
    """Hashing only the basename collided across classes — `0001.jpg` is in every one."""
    _make_dataset(tmp_path / "a", {"cls1": 2, "cls2": 2})
    m = data_mod._scan_imagefolder(tmp_path / "a")
    swapped = data_mod.Manifest(list(m.paths), list(reversed(m.labels)), m.class_names)
    assert m.fingerprint()["sha256"] != swapped.fingerprint()["sha256"]


# --------------------------------------------------------------------------- metrics

def test_macro_f1_is_not_merged_into_acc5():
    """torchmetrics merges compute groups whose state matches after the first update.

    With {acc@1, acc@5, macro-F1} on a working model, macro-F1 and balanced-accuracy
    silently returned the acc@5 value. `compute_groups=False` is the fix.
    """
    import torchmetrics as tm

    nc = 10
    torch.manual_seed(0)
    probs, targets = [], []
    for _ in range(8):
        y = torch.randint(0, nc, (64,))
        logits = torch.randn(64, nc) * 0.3
        logits[torch.arange(64), y] += 5.0
        flip = torch.rand(64) < 0.15
        logits[flip] = torch.randn(int(flip.sum()), nc) * 0.3
        logits[flip, y[flip]] += 1.0
        probs.append(logits.softmax(-1))
        targets.append(y)

    cfg = _cfg(**{"validation.metrics": [Metric.ACC1, Metric.ACC5, Metric.MACRO_F1,
                                         Metric.BALANCED_ACC]})
    mc = metrics_mod.build_metrics(cfg, nc, torch.device("cpu"))
    for p, y in zip(probs, targets):
        mc.update(p, y)
    got = {k: float(v) for k, v in mc.compute().items()}

    ref = tm.MetricCollection({
        "macro-F1": tm.F1Score(task="multiclass", num_classes=nc, average="macro"),
    }, compute_groups=False)
    for p, y in zip(probs, targets):
        ref.update(p, y)
    expected_f1 = float(ref.compute()["macro-F1"])

    assert got["macro-F1"] == pytest.approx(expected_f1, abs=1e-6)
    assert got["macro-F1"] != pytest.approx(got["acc@5"], abs=1e-9), (
        "macro-F1 collapsed onto acc@5 — compute_groups must stay disabled")
    assert got["balanced-accuracy"] != pytest.approx(got["acc@5"], abs=1e-9)


def test_acc5_is_reported_unavailable_below_five_classes():
    """It was dropped silently; if it was the primary metric every epoch scored -inf,
    no best checkpoint was written, and the run still exited 0."""
    cfg = _cfg(**{"validation.primary_metric": Metric.ACC5,
                  "validation.metrics": [Metric.ACC1, Metric.ACC5]})
    blocked = metrics_mod.unavailable_metrics(cfg, num_classes=3)
    assert "acc@5" in blocked
    assert metrics_mod.unavailable_metrics(cfg, num_classes=10) == {}


# ---------------------------------------------------------------------------- losses

def test_focal_modulation_uses_the_true_class_probability():
    """`pt = exp(-ce)` is only valid for unweighted, unsmoothed CE. With class weights
    it is off by the weight factor, so the focal term down-weights the wrong examples."""
    torch.manual_seed(0)
    x = torch.randn(16, 4)
    y = torch.randint(0, 4, (16,))
    w = torch.tensor([4.0, 1.0, 1.0, 1.0])

    weighted = losses.FocalLoss(gamma=2.0, weight=w)(x, y)

    pt = torch.softmax(x, -1).gather(1, y[:, None]).squeeze(1)
    ce = torch.nn.functional.cross_entropy(x, y, weight=w, reduction="none")
    expected = ((1 - pt) ** 2 * ce).mean()
    assert torch.allclose(weighted, expected, atol=1e-6)

    # The old formulation and the correct one must genuinely differ, or this proves
    # nothing about the bug it guards.
    old = ((1 - torch.exp(-ce)) ** 2 * ce).mean()
    assert not torch.allclose(old, expected, atol=1e-4)


def test_class_balanced_ce_without_weights_is_flagged():
    """It is byte-identical to plain cross-entropy, which the UI gave no hint of."""
    from trainlab.config import ClassWeights, LossName

    cfg = _cfg(**{"loss.loss": LossName.CLASS_BALANCED_CE,
                  "loss.class_weights": ClassWeights.NONE})
    fields = {w.field for w in cfg.warnings()}
    assert "loss.class_weights" in fields


def test_mixup_is_off_in_the_bare_defaults():
    """The UI boots into `balanced`, so the raw defaults were reached only by the CLI —
    which then warned that its own default config was harmful."""
    cfg = TrainConfig()
    assert cfg.mixup_active is False
    assert cfg.effective_loss.value == "cross_entropy"
    assert cfg.loss_was_autoswitched is False


# ------------------------------------------------------------------------- migration

def test_removed_ssl_warmup_epochs_still_loads():
    """3 of 5 runs in the author's own tracker were permanently un-clonable, because
    `extra="forbid"` turns any un-migrated rename into a hard failure."""
    cfg = TrainConfig.model_validate({
        "experimental": {"ssl_warmup_epochs": 0, "ssl_method": "none"},
    })
    assert cfg.experimental.ssl_method.value == "none"
    assert not hasattr(cfg.experimental, "ssl_warmup_epochs")


def test_legacy_mim_still_maps_to_mae():
    cfg = TrainConfig.model_validate({
        "experimental": {"ssl_method": "mim", "mim_mask_ratio": 0.6,
                         "ssl_lr_warmup_epochs": 0, "ssl_epochs": 10},
    })
    assert cfg.experimental.ssl_method.value == "mae"
    assert cfg.experimental.mae_mask_ratio == 0.6


def test_lenient_load_drops_unknown_keys_and_reports_them():
    """Replaying recorded history must survive future renames; the strict path is for
    input a user just typed."""
    cfg, dropped = load_config_leniently({
        "optimization": {"lr": 0.001, "some_removed_knob": 7},
        "model": {"backbone": "resnet18", "another_gone": "x"},
    })
    assert cfg.optimization.lr == 0.001
    assert cfg.model.backbone == "resnet18"
    assert set(dropped) == {"optimization.some_removed_knob", "model.another_gone"}


def test_lenient_load_still_rejects_genuinely_invalid_values():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        load_config_leniently({"optimization": {"lr": "not-a-number"}})


# --------------------------------------------------------- schema/runtime consistency

def test_every_ui_expression_references_a_real_unambiguous_field():
    """showIf/disableIf are hand-written strings resolved against a flat namespace at
    render time. A renamed field leaves the dependent control permanently hidden with
    nothing to indicate why — the one place the schema does not generate everything."""
    problems = validate_ui_expressions()
    assert problems == [], "\n".join(problems)


def test_ui_expression_identifier_extraction_ignores_string_literals():
    assert ui_expression_identifiers("optimizer == 'adamw'") == {"optimizer"}
    assert ui_expression_identifiers("ssl_method in ('mae', 'simmim')") == {"ssl_method"}
    assert ui_expression_identifiers("a == 1 and b != null") == {"a", "b"}


@pytest.mark.parametrize("backbone,method,supported", [
    ("vit_tiny_patch16_224", "mae", True),
    ("deit_small_patch16_224", "mae", True),
    ("beit_base_patch16_224", "mae", False),
    ("eva02_tiny_patch14_224", "mae", False),
    ("samvit_base_patch16", "mae", False),
    ("convnext_tiny", "mae", False),
    ("swin_tiny_patch4_window7_224", "simmim", True),
    ("swinv2_tiny_window8_256", "simmim", False),
    ("vit_tiny_patch16_224", "simmim", True),
    ("convnext_tiny", "simsiam", True),
])
def test_ssl_prefix_lists_match_runtime_support(backbone, method, supported):
    """`config.py`'s prefix tuples exist so the schema can warn without importing timm.
    When they disagreed with `ssl.py`'s isinstance checks the form showed a clean
    validation panel and the run then died at model construction, minutes in.
    """
    pytest.importorskip("timm")
    from trainlab import ssl as ssl_mod
    from trainlab.config import SSLMethod

    cfg = _cfg(**{"model.backbone": backbone,
                  "experimental.ssl_method": SSLMethod(method)})
    runtime_ok, _ = ssl_mod.supported_backbone(cfg)
    schema_blocks = any(w.severity.value == "error" for w in cfg.warnings())

    assert runtime_ok is supported
    assert runtime_ok is not schema_blocks, (
        f"schema pre-flight and runtime disagree for {backbone}/{method}")


# ------------------------------------------------------------------------- device

def test_unavailable_device_falls_back_instead_of_crashing():
    """`_pick_device` returned the request verbatim, so the downgrade branch below it
    was unreachable and `device=cuda` on a Mac raised a bare AssertionError."""
    from trainlab.config import Device
    from trainlab.device import resolve_runtime

    cfg = _cfg(**{"schedule.device": Device.CUDA})
    rt = resolve_runtime(cfg)
    if not torch.cuda.is_available():
        assert rt.device_str != "cuda"
        assert any("cuda unavailable" in d for d in rt.downgrades)


def test_amp_downgrades_are_device_appropriate():
    from trainlab.config import Amp
    from trainlab.device import resolve_amp_for

    dtype, scaler, note = resolve_amp_for(Amp.BF16, "cpu")
    assert dtype is None and scaler is False

    dtype, scaler, note = resolve_amp_for(Amp.BF16, "mps")
    assert dtype is torch.float16 and "bf16 -> fp16" in note
    # fp16 without loss scaling flushes small gradients to zero.
    assert scaler is True
