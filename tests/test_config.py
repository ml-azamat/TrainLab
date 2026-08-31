"""Schema, preset and validation tests. No dataset or GPU required."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trainlab import naming, presets  # noqa: E402
from trainlab.config import (  # noqa: E402
    AUG_PRESETS, DEFAULT_EXPANDED, DEFAULT_TRACKING_URI, AugmentationConfig, AugPreset,
    AutoAugment, Group, LossName, Preset, Severity, TrainConfig, with_aug_preset,
)


# --------------------------------------------------------------------------- schema

def test_defaults_validate():
    cfg = TrainConfig()
    assert cfg.model.backbone == "convnext_tiny"
    assert cfg.schedule.epochs == 30
    assert cfg.optimization.lr == 3e-4


def test_yaml_roundtrip_is_lossless():
    cfg = TrainConfig()
    text = yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False)
    assert TrainConfig.model_validate(yaml.safe_load(text)).model_dump(mode="json") \
        == cfg.model_dump(mode="json")


def test_every_field_carries_ui_metadata():
    """The form is generated from this metadata, so a field without it is invisible."""
    schema = TrainConfig.model_json_schema()
    missing = []
    for name, defn in schema.get("$defs", {}).items():
        if not name.endswith("Config"):
            continue
        for field, prop in defn.get("properties", {}).items():
            if "x-ui" not in prop:
                missing.append(f"{name}.{field}")
    assert not missing, f"fields missing x-ui metadata: {missing}"


def test_tooltips_explain_when_to_change():
    """A tooltip that only restates the label is not worth showing."""
    schema = TrainConfig.model_json_schema()
    weak = []
    for defn in schema.get("$defs", {}).values():
        for field, prop in defn.get("properties", {}).items():
            ui = prop.get("x-ui")
            if ui and len(ui["tooltip"]) < 40:
                weak.append(field)
    assert not weak, f"tooltips too short to be useful: {weak}"


def test_three_groups_expanded_by_default():
    assert set(DEFAULT_EXPANDED) == {Group.DATA, Group.MODEL, Group.OPTIMIZATION}


# --------------------------------------------------------------------------- derived

def test_mixup_upgrades_cross_entropy_without_mutating_config():
    cfg = TrainConfig.model_validate({"augmentation": {"mixup_alpha": 0.2}})
    assert cfg.mixup_active
    assert cfg.loss.loss == LossName.CROSS_ENTROPY      # user's choice preserved
    assert cfg.effective_loss == LossName.SOFT_TARGET_CE
    assert cfg.loss_was_autoswitched


def test_loss_autoswitch_reverses_when_mixup_disabled():
    """Regression: writing the switch back into the config made it sticky."""
    on = TrainConfig.model_validate({"augmentation": {"mixup_alpha": 0.2}})
    off = TrainConfig.model_validate({
        **on.model_dump(mode="json"),
        "augmentation": {**on.augmentation.model_dump(mode="json"),
                         "mixup_alpha": 0.0, "cutmix_alpha": 0.0},
    })
    assert off.effective_loss == LossName.CROSS_ENTROPY


def test_incompatible_loss_with_mixup_is_a_hard_error():
    with pytest.raises(ValueError, match="soft targets"):
        TrainConfig.model_validate({
            "augmentation": {"mixup_alpha": 0.2},
            "loss": {"loss": "focal"},
        })


def test_test_input_size_tracks_input_size():
    """Regression: freezing this at validation time broke later input_size edits."""
    cfg = TrainConfig.model_validate({"input": {"input_size": 320}})
    assert cfg.effective_test_input_size == 320
    cfg2 = TrainConfig.model_validate({"input": {"input_size": 224, "test_input_size": 256}})
    assert cfg2.effective_test_input_size == 256


def test_warmup_must_be_shorter_than_training():
    with pytest.raises(ValueError, match="warmup_epochs"):
        TrainConfig.model_validate({"schedule": {"epochs": 2, "warmup_epochs": 5}})


def test_effective_batch_size_accounts_for_accumulation():
    cfg = TrainConfig.model_validate({
        "schedule": {"batch_size": 32}, "optimization": {"grad_accum_steps": 4},
    })
    assert cfg.effective_batch_size == 128


# --------------------------------------------------------------------------- warnings

def _fields(cfg, **kw):
    return {w.field for w in cfg.warnings(**kw)}


def test_ema_horizon_warning_on_short_run():
    cfg = TrainConfig.model_validate({"model": {"ema": True, "ema_decay": 0.9998}})
    assert "model.ema_decay" in _fields(cfg, n_train=2000)


def test_no_ema_warning_when_auto():
    cfg = TrainConfig.model_validate({"model": {"ema": True, "ema_decay": "auto"}})
    assert "model.ema_decay" not in _fields(cfg, n_train=2000)


def test_mixup_on_short_finetune_warns():
    cfg = TrainConfig.model_validate({
        "augmentation": {"mixup_alpha": 0.2},
        "schedule": {"epochs": 30}, "model": {"pretrained": True},
    })
    assert "augmentation.mixup_alpha" in _fields(cfg)


def test_adamw_with_sgd_scale_lr_warns():
    cfg = TrainConfig.model_validate({"optimization": {"optimizer": "adamw", "lr": 0.05}})
    assert "optimization.lr" in _fields(cfg)


def test_stacked_imbalance_corrections_warn():
    cfg = TrainConfig.model_validate({
        "data": {"sampler": "weighted"},
        "loss": {"class_weights": "effective-number"},
    })
    assert "loss.class_weights" in _fields(cfg)


def test_focal_on_balanced_data_warns():
    # Focal cannot consume mixup's soft targets, so mixing must be off for this config
    # to be constructible at all.
    cfg = TrainConfig.model_validate({
        "loss": {"loss": "focal"},
        "augmentation": {"mixup_alpha": 0.0, "cutmix_alpha": 0.0},
    })
    assert "loss.loss" in _fields(cfg, imbalance_ratio=1.1)


def test_accuracy_as_objective_under_imbalance_warns():
    cfg = TrainConfig()
    assert "validation.primary_metric" in _fields(cfg, imbalance_ratio=50)


def test_mps_downgrades_bf16():
    cfg = TrainConfig()
    msgs = [w.message for w in cfg.warnings(resolved_device="mps")]
    assert any("bf16" in m for m in msgs)


def test_clean_config_has_no_warnings():
    cfg = presets.apply_preset(Preset.BALANCED)
    warns = [w for w in cfg.warnings(n_train=50_000, imbalance_ratio=1.0,
                                     resolved_device="cuda")
             if w.severity != Severity.INFO]
    assert warns == [], [w.message for w in warns]


# --------------------------------------------------------------------------- presets

@pytest.mark.parametrize("preset", [p for p in Preset if p != Preset.CUSTOM])
def test_preset_produces_valid_config(preset):
    cfg = presets.apply_preset(preset)
    assert cfg.tracking.preset == preset
    assert presets.describe(preset)


@pytest.mark.parametrize("preset", [p for p in Preset if p != Preset.CUSTOM])
def test_no_preset_enables_mixup(preset):
    """Mixup is opt-in: the evidence for it is specific to long from-scratch schedules."""
    cfg = presets.apply_preset(preset)
    assert not cfg.mixup_active, f"{preset.value} enables mixup"


def test_imbalanced_preset_optimises_macro_f1():
    cfg = presets.apply_preset(Preset.IMBALANCED)
    assert cfg.validation.primary_metric.value == "macro-F1"
    assert cfg.loss.loss == LossName.LOGIT_ADJUSTED
    # Decoupled-training result: instance-balanced sampling learns better representations.
    assert cfg.data.sampler.value == "random"


def test_preset_preserves_data_paths():
    base = TrainConfig.model_validate({"data": {"train_dir": "/my/data"}})
    out = presets.apply_preset(Preset.MAX_ACCURACY, base)
    assert out.data.train_dir == "/my/data"


def test_top_level_presets_expand_their_augmentation_rung():
    """Regression: presets named a rung ('heavy') that nothing ever expanded, so the
    augmentation they advertised was whatever the schema defaults happened to be."""
    light = presets.apply_preset(Preset.FAST_BASELINE).augmentation
    assert (light.preset, light.auto_augment.value, light.random_erasing_p) == \
        (AugPreset.LIGHT, "none", 0.0)

    heavy = presets.apply_preset(Preset.MAX_ACCURACY).augmentation
    assert (heavy.preset, heavy.randaugment_m, heavy.random_erasing_p) == \
        (AugPreset.HEAVY, 9, 0.25)

    medium = presets.apply_preset(Preset.BALANCED).augmentation
    assert (medium.preset, medium.randaugment_m, medium.random_erasing_p) == \
        (AugPreset.MEDIUM, 7, 0.1)


def test_preset_deviating_from_its_rung_reports_custom():
    """small-dataset wants heavy strength with a tuning-free policy. It gets heavy's
    values, but the label has to admit the group is no longer that rung."""
    aug = presets.apply_preset(Preset.SMALL_DATASET).augmentation
    assert aug.auto_augment == AutoAugment.TRIVIALAUGMENT_WIDE
    assert aug.random_erasing_p == AUG_PRESETS[AugPreset.HEAVY]["random_erasing_p"]
    assert aug.preset == AugPreset.CUSTOM


# ------------------------------------------------------------------- preset label (tag)
#
# `tracking.preset` is what naming.run_tags tags the run with and what the comparison view
# filters and groups by, so it has to describe the config that ran.


def _edited(preset: Preset, **groups) -> TrainConfig:
    """`preset` applied, then edited the way the form or `--set` edits a config."""
    d = presets.apply_preset(preset).model_dump(mode="json")
    for group, fields in groups.items():
        d[group].update(fields)
    return TrainConfig.model_validate(d)


@pytest.mark.parametrize("preset", [p for p in Preset if p != Preset.CUSTOM])
def test_a_preset_still_describes_itself(preset):
    assert presets.describes(preset, presets.apply_preset(preset).model_dump(mode="json"))


def test_editing_a_preset_gives_up_its_tag():
    """Regression: clicking Balanced and then changing anything still launched the run
    tagged `preset=balanced`, so the leaderboard grouped it with untouched Balanced runs."""
    assert _edited(Preset.BALANCED, schedule={"epochs": 100}).tracking.preset == Preset.CUSTOM
    assert _edited(Preset.MAX_ACCURACY,
                   optimization={"lr": 1e-5}).tracking.preset == Preset.CUSTOM
    assert _edited(Preset.IMBALANCED,
                   model={"backbone": "resnet50"}).tracking.preset == Preset.CUSTOM


def test_editing_an_augmentation_control_gives_up_both_labels():
    """The group's own rung and the run's tag describe different scopes, but an edit
    falsifies both at once."""
    cfg = _edited(Preset.BALANCED, augmentation={"hflip": 0.0})
    assert cfg.augmentation.preset == AugPreset.CUSTOM
    assert cfg.tracking.preset == Preset.CUSTOM


def test_the_tag_survives_everything_a_preset_does_not_describe():
    """Where the data lives, where output goes, what the dataset scan detected and how the
    run is tracked are not the recipe — two people running Balanced on their own data are
    both running Balanced."""
    cfg = _edited(
        Preset.BALANCED,
        data={"train_dir": "/data/train", "val_dir": "/data/val",
              "num_classes": 10, "class_names": [f"c{i}" for i in range(10)]},
        checkpoint={"output_dir": "/scratch/runs", "resume_from": "/scratch/last.ckpt"},
        tracking={"experiment_name": "birds", "tracking_uri": "http://10.0.0.2:5050",
                  "run_name": "second try", "tags": {"owner": "me"}, "enabled": False},
    )
    assert cfg.tracking.preset == Preset.BALANCED


def test_a_config_that_merely_matches_a_preset_is_not_retitled():
    """Demotion only. Promotion is the form's job, where the highlighted button is live
    feedback; here it would silently retitle hand-written YAML on load."""
    d = presets.apply_preset(Preset.BALANCED).model_dump(mode="json")
    d["tracking"]["preset"] = Preset.CUSTOM.value
    assert TrainConfig.model_validate(d).tracking.preset == Preset.CUSTOM


def test_schema_defaults_are_not_tagged_with_a_preset():
    assert TrainConfig().tracking.preset == Preset.CUSTOM


def test_the_tag_survives_a_yaml_roundtrip():
    """The launched config is stored and replayed; a tag that only survived the POST would
    make every cloned run report `custom`."""
    cfg = presets.apply_preset(Preset.SMALL_DATASET)
    text = yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False)
    back = TrainConfig.model_validate(yaml.safe_load(text))
    assert back.tracking.preset == Preset.SMALL_DATASET
    assert naming.run_tags(back)["preset"] == "small-dataset"


def test_overlaying_a_preset_onto_a_customised_config_reports_custom():
    """`apply_preset` keeps whatever the base set outside the preset's own groups, so the
    result is only that preset when the base was not carrying anything else."""
    base = TrainConfig.model_validate({"experimental": {"swa": True}})
    assert presets.apply_preset(Preset.BALANCED, base).tracking.preset == Preset.CUSTOM


def test_cli_preset_plus_an_override_reports_custom():
    """`--preset balanced --set schedule.epochs=100` is the headless spelling of the
    same bug: `--train-dir`, `--output-dir` and `--experiment` must not cost the tag."""
    import train                    # local: the CLI pulls in torch, this module must not

    d = presets.apply_preset(Preset.BALANCED).model_dump(mode="json")
    d["data"]["train_dir"] = "/data/train"
    d["checkpoint"]["output_dir"] = "/scratch"
    d["tracking"]["experiment_name"] = "birds"
    assert TrainConfig.model_validate(d).tracking.preset == Preset.BALANCED

    edited = train._apply_overrides(d, ["schedule.epochs=100"])
    assert TrainConfig.model_validate(edited).tracking.preset == Preset.CUSTOM


def test_identity_fields_all_exist_in_the_schema():
    """A rename that left a dead entry here would quietly start demoting every config
    that touches the renamed field."""
    dump = TrainConfig().model_dump(mode="json")
    for path in presets.PRESET_IDENTITY_FIELDS:
        node = dump
        for part in path.split("."):
            assert isinstance(node, dict) and part in node, path
            node = node[part]


def test_identity_fields_are_the_only_ones_that_keep_a_tag():
    """Every other field is part of the recipe: changing it must cost the label. Catches a
    new group or field being added to the schema and quietly escaping the comparison."""
    for group, fields in TrainConfig.model_fields.items():
        sub = getattr(fields.annotation, "model_fields", None)
        if not sub or group == "tracking":
            continue
        for name in sub:
            if f"{group}.{name}" in presets.PRESET_IDENTITY_FIELDS:
                continue
            base = presets.apply_preset(Preset.BALANCED).model_dump(mode="json")
            base[group][name] = _mutate(base[group][name])
            assert not presets.describes(Preset.BALANCED, base), f"{group}.{name}"


def _mutate(value):
    """Some value the schema would accept that is not the one it already holds."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return 0 if value else 1
    if isinstance(value, list):
        return [] if value else ["x"]
    return None if value is not None else "x"


# ----------------------------------------------------------------- augmentation ladder

_AUG_CONTROLS = set(AugmentationConfig.model_fields) - {"preset"}


def test_every_selectable_rung_has_a_table():
    """`custom` is the only value of the enum that names no values. Any other rung the
    form offers must expand, or picking it would be the no-op this ladder replaced."""
    assert set(AUG_PRESETS) == {p for p in AugPreset if p != AugPreset.CUSTOM}


@pytest.mark.parametrize("rung", list(AUG_PRESETS))
def test_aug_preset_covers_every_control(rung):
    """A rung sets the whole group, so a control missing from the table would silently
    keep whatever the previous rung left there — which is how 'medium' stops meaning
    medium. Adding an augmentation field must mean adding it to all four tables."""
    assert set(AUG_PRESETS[rung]) == _AUG_CONTROLS


def test_aug_preset_medium_matches_schema_defaults():
    """The default preset is `medium`, so an untouched config must not contradict it."""
    defaults = AugmentationConfig().model_dump(mode="json")
    assert AUG_PRESETS[AugPreset.MEDIUM] == {k: v for k, v in defaults.items()
                                             if k != "preset"}


@pytest.mark.parametrize("rung", list(AUG_PRESETS))
def test_aug_preset_expands_into_real_values(rung):
    """Regression: selecting a rung set the enum and nothing else, so the preview, the
    warnings and the launched run all stayed on the previous values."""
    aug = TrainConfig.model_validate({"augmentation": {"preset": rung.value}}).augmentation
    assert aug.preset == rung
    for field, expected in AUG_PRESETS[rung].items():
        actual = getattr(aug, field)
        actual = actual.value if hasattr(actual, "value") else actual
        assert actual == expected, field


def test_aug_preset_none_disables_augmentation():
    aug = TrainConfig.model_validate({"augmentation": {"preset": "none"}}).augmentation
    assert (aug.hflip, aug.color_jitter, aug.random_erasing_p) == (0.0, 0.0, 0.0)
    assert aug.auto_augment == AutoAugment.NONE


def test_aug_ladder_is_monotone_in_strength():
    order = [AugPreset.NONE, AugPreset.LIGHT, AugPreset.MEDIUM, AugPreset.HEAVY]
    for weaker, stronger in zip(order, order[1:]):
        a, b = AUG_PRESETS[weaker], AUG_PRESETS[stronger]
        assert a["random_erasing_p"] <= b["random_erasing_p"]
        assert a["hflip"] <= b["hflip"]


def test_no_aug_rung_enables_mixup():
    for rung, values in AUG_PRESETS.items():
        assert values["mixup_alpha"] == 0.0 and values["cutmix_alpha"] == 0.0, rung


def test_value_contradicting_the_rung_demotes_it_to_custom():
    """A label that no longer describes the group is worse than no label: runs are
    filtered and compared by it."""
    aug = TrainConfig.model_validate({
        "augmentation": {"preset": "heavy", "vflip": 0.5},
    }).augmentation
    assert aug.preset == AugPreset.CUSTOM
    assert aug.vflip == 0.5
    assert aug.randaugment_m == 9        # the rest of the rung still applied


def test_restating_a_rungs_own_value_is_not_a_contradiction():
    """The form POSTs the whole group every time, so equal-but-respelled values (ints for
    floats, strings for enums) must not demote every config to custom."""
    aug = TrainConfig.model_validate({
        "augmentation": {"preset": "heavy", "hflip": 0.5, "randaugment_m": 9,
                         "auto_augment": "randaugment", "mixup_off_epoch": 0},
    }).augmentation
    assert aug.preset == AugPreset.HEAVY


def test_custom_preset_leaves_values_untouched():
    aug = TrainConfig.model_validate({
        "augmentation": {"preset": "custom", "randaugment_m": 3},
    }).augmentation
    assert aug.preset == AugPreset.CUSTOM
    assert aug.randaugment_m == 3
    assert aug.hflip == 0.5              # unnamed fields still fall back to the schema


def test_aug_preset_survives_a_yaml_roundtrip():
    cfg = TrainConfig.model_validate({"augmentation": {"preset": "heavy"}})
    text = yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False)
    back = TrainConfig.model_validate(yaml.safe_load(text))
    assert back.augmentation.preset == AugPreset.HEAVY
    assert back.model_dump(mode="json") == cfg.model_dump(mode="json")


def test_unknown_aug_preset_is_still_a_validation_error():
    with pytest.raises(ValueError, match="preset"):
        TrainConfig.model_validate({"augmentation": {"preset": "extreme"}})


def test_with_aug_preset_applies_a_rung_to_a_complete_group():
    """`--set augmentation.preset=heavy` and a sweep over the rung both set it on a group
    where every value is already explicit, which validation alone would only relabel."""
    complete = TrainConfig().augmentation.model_dump(mode="json")
    out = TrainConfig.model_validate({
        "augmentation": with_aug_preset(complete, "heavy"),
    }).augmentation
    assert out.preset == AugPreset.HEAVY
    assert (out.randaugment_m, out.random_erasing_p) == (9, 0.25)


def test_with_aug_preset_leaves_later_overrides_the_last_word():
    complete = TrainConfig().augmentation.model_dump(mode="json")
    raw = with_aug_preset(complete, "heavy")
    raw["randaugment_m"] = 3                     # a --set applied after the rung
    out = TrainConfig.model_validate({"augmentation": raw}).augmentation
    assert (out.randaugment_m, out.random_erasing_p) == (3, 0.25)
    assert out.preset == AugPreset.CUSTOM        # no longer the rung it names


def test_cli_set_expands_an_augmentation_rung():
    import train                    # local: the CLI pulls in torch, this module must not

    d = TrainConfig().model_dump(mode="json")
    out = TrainConfig.model_validate(
        train._apply_overrides(d, ["augmentation.preset=heavy"])).augmentation
    assert (out.preset, out.randaugment_m, out.random_erasing_p) == \
        (AugPreset.HEAVY, 9, 0.25)


def test_cli_set_after_a_rung_still_wins():
    import train

    d = TrainConfig().model_dump(mode="json")
    out = TrainConfig.model_validate(train._apply_overrides(
        d, ["augmentation.preset=none", "augmentation.hflip=0.5"])).augmentation
    assert (out.hflip, out.auto_augment) == (0.5, AutoAugment.NONE)
    assert out.preset == AugPreset.CUSTOM


# --------------------------------------------------------------------------- naming

def test_run_name_describes_the_config():
    cfg = presets.apply_preset(Preset.BALANCED)
    name = naming.run_name(cfg)
    assert "convnext_tiny" in name and "30ep" in name and "llrd0.75" in name


def test_flatten_produces_dotted_scalar_params():
    flat = naming.flatten(TrainConfig().model_dump(mode="json"))
    assert flat["optimization.lr"] == "0.0003"
    assert all(isinstance(v, str) for v in flat.values())


def test_tags_are_all_strings():
    tags = naming.run_tags(TrainConfig())
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in tags.items())


# --------------------------------------------------------------------------- metrics

def test_metric_collection_disables_compute_groups():
    """Regression: torchmetrics' compute-group merging silently aliased macro-F1 to acc@5.

    On Imagenette this reported macro-F1 = 0.9954 (the acc@5 value) instead of 0.9287.
    The merge is data-dependent, so this asserts the setting rather than the symptom.
    """
    pytest.importorskip("torch")
    from trainlab.metrics import build_metrics

    cfg = TrainConfig.model_validate({
        "validation": {"metrics": ["acc@1", "acc@5", "macro-F1"]},
    })
    mc = build_metrics(cfg, num_classes=10, device="cpu")
    assert mc._enable_compute_groups is False
    assert set(mc.keys()) == {"acc@1", "acc@5", "macro-F1"}


def test_acc5_dropped_when_fewer_than_five_classes():
    pytest.importorskip("torch")
    from trainlab.metrics import build_metrics

    cfg = TrainConfig.model_validate({"validation": {"metrics": ["acc@1", "acc@5"]}})
    assert "acc@5" not in build_metrics(cfg, num_classes=3, device="cpu").keys()


# --------------------------------------------------------------------------- SSL

def test_ssl_off_by_default():
    assert TrainConfig().ssl_active is False


@pytest.mark.parametrize("method", ["simsiam", "mae"])
def test_ssl_method_resolves_published_hyperparameters(method):
    """'auto' must apply each method's own rule, not carry over the supervised LR."""
    backbone = "vit_small_patch16_224" if method == "mae" else "convnext_tiny"
    cfg = TrainConfig.model_validate({
        "model": {"backbone": backbone},
        "experimental": {"ssl_method": method, "ssl_batch_size": 256},
    })
    assert cfg.ssl_active
    if method == "simsiam":
        assert cfg.resolved_ssl_lr() == pytest.approx(0.05)    # SGD, 0.05*bs/256
        assert cfg.resolved_ssl_weight_decay() == pytest.approx(1e-4)
    else:
        assert cfg.resolved_ssl_lr() == pytest.approx(1.5e-4)  # AdamW, 1.5e-4*bs/256
        assert cfg.resolved_ssl_weight_decay() == pytest.approx(0.05)


def test_ssl_lr_scales_with_batch_size():
    def lr(bs):
        return TrainConfig.model_validate({
            "experimental": {"ssl_method": "simsiam", "ssl_batch_size": bs},
        }).resolved_ssl_lr()
    assert lr(512) == pytest.approx(2 * lr(256))


def test_ssl_explicit_values_override_auto():
    cfg = TrainConfig.model_validate({
        "experimental": {"ssl_method": "simsiam", "ssl_lr": 0.123, "ssl_weight_decay": 0.456},
    })
    assert cfg.resolved_ssl_lr() == 0.123
    assert cfg.resolved_ssl_weight_decay() == 0.456


def test_ssl_input_size_falls_back_to_supervised():
    cfg = TrainConfig.model_validate({
        "input": {"input_size": 192}, "experimental": {"ssl_method": "simsiam"},
    })
    assert cfg.effective_ssl_input_size == 192
    cfg2 = TrainConfig.model_validate({
        "input": {"input_size": 192},
        "experimental": {"ssl_method": "simsiam", "ssl_input_size": 128},
    })
    assert cfg2.effective_ssl_input_size == 128


def test_ssl_workers_fall_back_to_data_group():
    cfg = TrainConfig.model_validate({
        "data": {"num_workers": 12}, "experimental": {"ssl_method": "simsiam"},
    })
    assert cfg.effective_ssl_num_workers() == 12
    cfg2 = TrainConfig.model_validate({
        "data": {"num_workers": 12},
        "experimental": {"ssl_method": "simsiam", "ssl_num_workers": 0},
    })
    assert cfg2.effective_ssl_num_workers() == 0


def test_mae_on_a_convnet_is_an_error_severity_warning():
    """MAE is token-based; a convnet cannot do it. Surfaced before the run starts."""
    cfg = TrainConfig.model_validate({
        "model": {"backbone": "convnext_tiny"},
        "experimental": {"ssl_method": "mae"},
    })
    errs = [w for w in cfg.warnings()
            if w.severity == Severity.ERROR and w.field == "experimental.ssl_method"]
    assert errs, "expected an error-severity warning for MIM on a convnet"


def test_mae_on_a_vit_is_accepted():
    cfg = TrainConfig.model_validate({
        "model": {"backbone": "vit_base_patch16_224"},
        "experimental": {"ssl_method": "mae", "ssl_epochs": 200},
    })
    assert not [w for w in cfg.warnings() if w.severity == Severity.ERROR]


def test_short_ssl_schedule_warns():
    cfg = TrainConfig.model_validate({
        "experimental": {"ssl_method": "simsiam", "ssl_epochs": 5, "ssl_lr_warmup_epochs": 0},
    })
    assert "experimental.ssl_epochs" in {w.field for w in cfg.warnings()}


def test_ssl_warmup_must_be_shorter_than_ssl_schedule():
    with pytest.raises(ValueError, match="ssl_lr_warmup_epochs"):
        TrainConfig.model_validate({
            "experimental": {"ssl_method": "simsiam", "ssl_epochs": 10,
                             "ssl_lr_warmup_epochs": 10},
        })


def test_mae_decoder_heads_must_divide_width():
    with pytest.raises(ValueError, match="divisible"):
        TrainConfig.model_validate({
            "model": {"backbone": "vit_small_patch16_224"},
            "experimental": {"ssl_method": "mae", "mae_decoder_dim": 100,
                             "mae_decoder_heads": 16},
        })


# --------------------------------------------------------------------------- other stages

def test_crt_lr_defaults_to_ten_times_base():
    cfg = TrainConfig.model_validate({
        "optimization": {"lr": 1e-4},
        "experimental": {"classifier_retrain_epochs": 10},
    })
    assert cfg.resolved_crt_lr() == pytest.approx(1e-3)


def test_noise_filter_start_must_be_inside_the_schedule():
    with pytest.raises(ValueError, match="label_noise_start_epoch"):
        TrainConfig.model_validate({
            "schedule": {"epochs": 5},
            "experimental": {"label_noise_filter": True, "label_noise_start_epoch": 5},
        })


def test_curriculum_and_noise_filter_together_warn():
    cfg = TrainConfig.model_validate({
        "schedule": {"epochs": 30},
        "experimental": {"curriculum_by_loss": True, "label_noise_filter": True},
    })
    assert "experimental.curriculum_by_loss" in {w.field for w in cfg.warnings()}


def test_low_pseudo_label_threshold_warns():
    cfg = TrainConfig.model_validate({
        "experimental": {"pseudo_label_dir": "/tmp/unlabeled",
                         "pseudo_label_threshold": 0.6},
    })
    assert "experimental.pseudo_label_threshold" in {w.field for w in cfg.warnings()}


def test_every_experimental_and_schedule_field_is_read_by_the_engine():
    """Regression: controls used to exist in the schema and do nothing.

    Six experimental fields once rendered, logged and did nothing; later,
    `schedule.distributed`/`schedule.num_gpus` were declared with no DDP behind them
    while still distorting `effective_batch_size` — dead controls hid exactly where
    this guard wasn't looking, so it now covers `schedule` too.

    Scans the runtime modules plus the bodies of TrainConfig's `resolved_*`/`effective_*`
    helpers — a field read inside one of those is legitimately consumed. Field
    *declarations* are excluded, so simply existing in the schema does not satisfy this.
    """
    import inspect
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sources = "\n".join(
        p.read_text() for p in [
            root / "trainlab" / "engine.py", root / "trainlab" / "ssl.py",
            root / "trainlab" / "data.py", root / "trainlab" / "sched.py",
            root / "trainlab" / "device.py", root / "trainlab" / "transforms.py",
            root / "trainlab" / "optim.py", root / "trainlab" / "losses.py",
            root / "trainlab" / "metrics.py", root / "train.py",
        ]
    )
    for name in dir(TrainConfig):
        if name.startswith(("resolved_", "effective_")):
            attr = inspect.getattr_static(TrainConfig, name)
            fn = attr.fget if isinstance(attr, property) else attr
            if callable(fn):
                sources += "\n" + inspect.getsource(fn)

    cfg = TrainConfig()
    for group in ("experimental", "schedule"):
        fields = set(getattr(cfg, group).model_dump().keys())
        unread = sorted(f for f in fields if f not in sources)
        assert not unread, f"{group} controls not referenced by any runtime code: {unread}"


# --------------------------------------------------------------------------- SimMIM

def test_simmim_runs_on_vit_and_swin_but_not_convnets():
    from trainlab.config import SSLMethod

    for backbone in ("vit_small_patch16_224", "swin_tiny_patch4_window7_224"):
        cfg = TrainConfig.model_validate({
            "model": {"backbone": backbone},
            "experimental": {"ssl_method": "simmim", "ssl_epochs": 200},
        })
        assert cfg.experimental.ssl_method == SSLMethod.SIMMIM
        assert not [w for w in cfg.warnings() if w.severity == Severity.ERROR], backbone

    cfg = TrainConfig.model_validate({
        "model": {"backbone": "convnext_tiny"},
        "experimental": {"ssl_method": "simmim", "ssl_epochs": 200},
    })
    errs = [w for w in cfg.warnings() if w.severity == Severity.ERROR]
    assert errs and "SparK" in errs[0].fix


def test_mae_refuses_swin_but_simmim_accepts_it():
    """The architectural split is the whole reason both methods exist."""
    def errors(method):
        cfg = TrainConfig.model_validate({
            "model": {"backbone": "swin_tiny_patch4_window7_224"},
            "experimental": {"ssl_method": method, "ssl_epochs": 200},
        })
        return [w for w in cfg.warnings() if w.severity == Severity.ERROR]

    assert errors("mae"), "MAE drops tokens; it cannot use a hierarchical Swin encoder"
    assert not errors("simmim")


def test_simmim_mask_unit_must_tile_the_image():
    with pytest.raises(ValueError, match="simmim_mask_patch_size"):
        TrainConfig.model_validate({
            "model": {"backbone": "vit_small_patch16_224"},
            "input": {"input_size": 224},
            "experimental": {"ssl_method": "simmim", "simmim_mask_patch_size": 30},
        })


def test_simmim_uses_its_own_auto_lr():
    """SimMIM's published base LR differs from MAE's, so 'auto' must not share one rule."""
    def lr(method):
        return TrainConfig.model_validate({
            "model": {"backbone": "vit_small_patch16_224"},
            "experimental": {"ssl_method": method, "ssl_batch_size": 256},
        }).resolved_ssl_lr()

    assert lr("simmim") == pytest.approx(1e-4)
    assert lr("mae") == pytest.approx(1.5e-4)
    assert lr("simmim") != lr("mae")


# --------------------------------------------------------------------------- migration

def test_legacy_mim_configs_still_load():
    """Runs recorded before the rename live in the tracker; Clone must keep working."""
    from trainlab.config import SSLMethod

    cfg = TrainConfig.model_validate({
        "model": {"backbone": "vit_small_patch16_224"},
        "experimental": {
            "ssl_method": "mim", "ssl_epochs": 400,
            "mim_mask_ratio": 0.8, "mim_decoder_dim": 256,
            "mim_decoder_depth": 2, "mim_decoder_heads": 8,
            "mim_norm_pix_loss": False,
        },
    })
    assert cfg.experimental.ssl_method == SSLMethod.MAE   # the old 'mim' WAS MAE
    assert cfg.experimental.mae_mask_ratio == 0.8
    assert cfg.experimental.mae_decoder_depth == 2
    assert cfg.experimental.mae_norm_pix_loss is False


def test_migration_leaves_current_configs_untouched():
    cfg = TrainConfig.model_validate({
        "model": {"backbone": "vit_small_patch16_224"},
        "experimental": {"ssl_method": "mae", "mae_mask_ratio": 0.5},
    })
    assert cfg.experimental.mae_mask_ratio == 0.5


def test_no_legacy_mim_field_names_survive_in_the_schema():
    schema = TrainConfig.model_json_schema()
    props = schema["$defs"]["ExperimentalConfig"]["properties"]
    assert not [k for k in props if k.startswith("mim_")]
    assert {"mae_mask_ratio", "simmim_mask_ratio", "simmim_mask_patch_size"} <= set(props)


# --------------------------------------------------------------------------- tracker URI

def test_tracking_uri_defaults_to_the_schema_constant(monkeypatch):
    monkeypatch.delenv("TRAINLAB_TRACKING_URI", raising=False)
    assert TrainConfig().tracking.tracking_uri == DEFAULT_TRACKING_URI


def test_tracking_uri_default_follows_the_environment(monkeypatch):
    """`make api` passes the tracker's address in, so moving it does not mean retyping
    the URL in the form every session."""
    monkeypatch.setenv("TRAINLAB_TRACKING_URI", "http://127.0.0.1:5555")
    assert TrainConfig().tracking.tracking_uri == "http://127.0.0.1:5555"
    # Presets carry it too — none of them overlays the tracking group.
    assert presets.apply_preset(Preset.BALANCED).tracking.tracking_uri == \
        "http://127.0.0.1:5555"


def test_an_explicit_tracking_uri_still_wins(monkeypatch):
    monkeypatch.setenv("TRAINLAB_TRACKING_URI", "http://127.0.0.1:5555")
    cfg = TrainConfig.model_validate({"tracking": {"tracking_uri": "http://host:9999"}})
    assert cfg.tracking.tracking_uri == "http://host:9999"


def test_the_tracker_address_does_not_affect_the_preset_tag(monkeypatch):
    """Two people running Balanced against their own trackers are both running Balanced."""
    monkeypatch.setenv("TRAINLAB_TRACKING_URI", "http://127.0.0.1:5555")
    assert presets.apply_preset(Preset.BALANCED).tracking.preset == Preset.BALANCED
