"""FPR at a target FNR: the operating-point metric, its wiring and its guard rails."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trainlab.config import (  # noqa: E402
    Metric, Severity, TrainConfig, fpr_at_fnr_key, metric_higher_is_better,
)
from trainlab.metrics import (  # noqa: E402
    FprAtFnr, _positive_index, build_metrics, fpr_at_fnr, unavailable_metrics,
)


def _cfg(**validation) -> TrainConfig:
    base = {"metrics": ["acc@1", "fpr@fnr"], "positive_class": "live"}
    return TrainConfig.model_validate({
        "data": {"num_classes": 2, "class_names": ["live", "spoof"]},
        "validation": {**base, **validation},
    })


# --------------------------------------------------------------------------- the metric

def test_perfectly_separated_scores_cost_nothing():
    scores = torch.tensor([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    pos = torch.tensor([True, True, True, False, False, False])
    assert fpr_at_fnr(scores, pos, 0.01) == 0.0


def test_the_threshold_misses_at_most_the_target_fraction():
    """`floor(target * n_pos)` positives may fall below the threshold — never more, so a
    reported FPR is never the one belonging to a looser operating point."""
    pos_scores = torch.arange(100, dtype=torch.float) / 100      # 0.00 .. 0.99
    neg_scores = torch.full((100,), 0.5)
    scores = torch.cat([pos_scores, neg_scores])
    labels = torch.cat([torch.ones(100), torch.zeros(100)]).bool()

    # Allowing 10 misses puts the threshold at the 10th-lowest positive (0.10), which is
    # below every negative, so every negative is still accepted.
    assert fpr_at_fnr(scores, labels, 0.10) == pytest.approx(1.0)
    # Allowing 60 puts it at 0.60, above the negatives, so none are.
    assert fpr_at_fnr(scores, labels, 0.60) == pytest.approx(0.0)


def test_a_looser_target_can_only_help():
    """Monotonicity: tolerate more misses and the false accepts cannot go up."""
    g = torch.Generator().manual_seed(0)
    scores = torch.cat([torch.rand(500, generator=g) * 0.6 + 0.4,
                        torch.rand(500, generator=g) * 0.6])
    labels = torch.cat([torch.ones(500), torch.zeros(500)]).bool()
    values = [fpr_at_fnr(scores, labels, t) for t in (0.001, 0.01, 0.05, 0.2)]
    assert values == sorted(values, reverse=True)


def test_ties_at_the_threshold_are_accepted():
    """The conservative reading: a tied negative counts as a false accept rather than
    being quietly dropped to flatter the number."""
    scores = torch.tensor([0.5, 0.5, 0.5, 0.5])
    labels = torch.tensor([True, True, False, False])
    assert fpr_at_fnr(scores, labels, 0.0) == pytest.approx(1.0)


def test_one_sided_data_is_undefined_rather_than_zero():
    scores = torch.tensor([0.9, 0.8])
    assert fpr_at_fnr(scores, torch.tensor([True, True]), 0.01) != \
        fpr_at_fnr(scores, torch.tensor([True, True]), 0.01)      # NaN
    assert fpr_at_fnr(scores, torch.tensor([False, False]), 0.01) != \
        fpr_at_fnr(scores, torch.tensor([False, False]), 0.01)


def test_a_target_of_one_does_not_run_off_the_end():
    scores = torch.tensor([0.9, 0.1])
    labels = torch.tensor([True, False])
    assert 0.0 <= fpr_at_fnr(scores, labels, 0.999) <= 1.0


def test_the_accumulating_metric_matches_the_function():
    """Batched updates must give the same answer as one shot over the whole set."""
    g = torch.Generator().manual_seed(1)
    probs = torch.rand(400, 2, generator=g)
    target = (torch.rand(400, generator=g) > 0.5).long()
    metric = FprAtFnr(target_fnr=0.05, pos_index=1)
    for i in range(0, 400, 64):
        metric.update(probs[i:i + 64], target[i:i + 64])
    assert metric.compute().item() == pytest.approx(
        fpr_at_fnr(probs[:, 1], target == 1, 0.05))


# --------------------------------------------------------------------------- wiring

def test_one_metric_is_built_per_target():
    keys = build_metrics(_cfg(fpr_at_fnr_targets=[0.01, 0.001]), num_classes=2,
                         device="cpu", class_names=["live", "spoof"]).keys()
    assert "fpr@fnr0_01" in keys and "fpr@fnr0_001" in keys


def test_the_positive_class_is_resolved_by_name_not_by_position():
    """Class indices come from sorting the names, so 'live' is 0 and 'spoof' is 1 —
    assuming index 1 would measure the opposite error rate and look plausible."""
    assert _positive_index(_cfg(), ["live", "spoof"]) == 0
    assert _positive_index(_cfg(positive_class="spoof"), ["live", "spoof"]) == 1


def test_the_first_target_decides_the_best_checkpoint():
    cfg = _cfg(primary_metric="fpr@fnr", fpr_at_fnr_targets=[0.02, 0.001])
    assert cfg.primary_metric_key == "fpr@fnr0_02"
    assert cfg.validation.primary_metric.higher_is_better is False


def test_reordering_the_targets_changes_which_one_decides():
    assert _cfg(primary_metric="fpr@fnr",
                fpr_at_fnr_targets=[0.001, 0.02]).primary_metric_key == "fpr@fnr0_001"


def test_other_primary_metrics_are_unaffected():
    assert _cfg(primary_metric="acc@1").primary_metric_key == "acc@1"


def test_lower_is_better_at_every_target_and_stage():
    for key in ("fpr@fnr0_01", "fpr@fnr0_001", "ema/fpr@fnr0_01", "fpr_at_fnr0_001"):
        assert metric_higher_is_better(key) is False, key
    assert metric_higher_is_better("acc@1") is True


def test_key_formatting_stays_short():
    assert fpr_at_fnr_key(0.01) == "fpr@fnr0_01"
    assert fpr_at_fnr_key(0.001) == "fpr@fnr0_001"
    assert fpr_at_fnr_key(1e-4) == "fpr@fnr0_0001"


# --------------------------------------------------------------------------- guard rails

def test_the_positive_class_must_be_chosen():
    with pytest.raises(ValueError, match="positive_class"):
        TrainConfig.model_validate({"validation": {"metrics": ["fpr@fnr"]}})


def test_the_positive_class_must_be_one_of_the_detected_classes():
    with pytest.raises(ValueError, match="not one of the detected"):
        TrainConfig.model_validate({
            "data": {"class_names": ["live", "spoof"]},
            "validation": {"metrics": ["fpr@fnr"], "positive_class": "genuine"},
        })


def test_an_empty_target_list_is_refused():
    with pytest.raises(ValueError, match="no FNR targets"):
        _cfg(fpr_at_fnr_targets=[])


def test_targets_must_be_fractions():
    for bad in ([1.0], [0.0], [-0.1], [5]):
        with pytest.raises(ValueError):
            _cfg(fpr_at_fnr_targets=bad)


def test_targets_are_floats_even_when_typed_as_something_else():
    """The form posts whatever the text box produced; the schema is what makes it a
    number, and everything downstream formats the key from it."""
    cfg = _cfg(fpr_at_fnr_targets=["0.01", 0.5])
    assert cfg.validation.fpr_at_fnr_targets == [0.01, 0.5]
    assert all(isinstance(t, float) for t in cfg.validation.fpr_at_fnr_targets)


def test_a_non_binary_dataset_blocks_the_metric():
    cfg = TrainConfig.model_validate({
        "data": {"num_classes": 3, "class_names": ["a", "b", "c"]},
        "validation": {"metrics": ["fpr@fnr"], "positive_class": "a"},
    })
    assert Metric.FPR_AT_FNR.value in unavailable_metrics(cfg, num_classes=3)
    assert unavailable_metrics(cfg, num_classes=2) == {}
    assert any(w.severity is Severity.ERROR and "binary" in w.message
               for w in cfg.warnings())


def test_a_val_set_too_small_for_the_target_warns():
    cfg = _cfg(fpr_at_fnr_targets=[0.001])
    assert any("measurable" in w.message for w in cfg.warnings(n_train=200))
    assert not any("measurable" in w.message for w in cfg.warnings(n_train=100_000))


def test_which_target_decides_is_stated_when_there_are_several():
    cfg = _cfg(primary_metric="fpr@fnr", fpr_at_fnr_targets=[0.02, 0.001])
    assert any("fpr@fnr0_02" in w.message and w.severity is Severity.INFO
               for w in cfg.warnings())
