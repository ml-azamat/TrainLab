#!/usr/bin/env python
"""Headless training entry point.

    python train.py --config runs/my_run.yaml
    python train.py --preset balanced --train-dir ./data/train --epochs 5
    python train.py --config base.yaml --set optimization.lr=1e-4 schedule.epochs=50

The UI is a config generator; this script is the only training entry point and never
imports anything from the web app.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from trainlab import env, naming, presets  # noqa: E402
from trainlab.config import Preset, Severity, TrainConfig, with_aug_preset  # noqa: E402
from trainlab.engine import Trainer, emit, log  # noqa: E402


def _coerce(text: str):
    """Parse a --set value into a Python scalar.

    Deliberately not plain `yaml.safe_load`: YAML 1.1 resolves `off`/`on`/`yes`/`no` to
    booleans, which silently breaks enum values like `--set schedule.amp=off`. Strings
    are left alone and Pydantic does the final coercion, so genuine bool fields still
    accept `off` while enums keep their literal value.

    `none` is likewise left as a literal string: it is a legitimate enum value on many
    fields (tta, auto_augment, ssl_method, freeze_policy, class_weights, cache_mode,
    sampler), and mapping it to Python None made every one of them unsettable from the
    CLI — the same bug class as `off`. Use `null` or `~` to set a nullable field.
    """
    t = text.strip()
    low = t.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "~"):
        return None
    for cast in (int, float):
        try:
            return cast(t)
        except ValueError:
            pass
    if t.startswith(("[", "{")):
        try:
            return yaml.safe_load(t)
        except yaml.YAMLError:
            return t
    return t


def _apply_overrides(d: dict, overrides: list[str]) -> dict:
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got '{item}'")
        key, _, val = item.partition("=")
        # Naming an augmentation rung sets the whole group, exactly as clicking it in the
        # form does. Applied in place so a later --set of an individual knob still wins.
        if key.strip() == "augmentation.preset":
            d["augmentation"] = with_aug_preset(d.get("augmentation") or {},
                                                _coerce(val.strip()))
            continue
        node = d
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce(val.strip())
    return d


def build_config(args) -> TrainConfig:
    if args.config:
        raw = yaml.safe_load(Path(args.config).read_text())
        # An empty file yields None; coercing that to {} silently trained a full set
        # of defaults from what was almost certainly a truncated or misnamed file.
        # (The API refuses the same input — the CLI should not be more permissive.)
        if raw is None:
            raise SystemExit(f"config file '{args.config}' is empty")
        if not isinstance(raw, dict):
            raise SystemExit(
                f"config file '{args.config}' must be a YAML mapping of config "
                f"sections, got {type(raw).__name__}")
        cfg = TrainConfig.model_validate(raw)
    elif args.preset:
        cfg = presets.apply_preset(Preset(args.preset))
    else:
        cfg = TrainConfig()

    d = cfg.model_dump(mode="json")
    if args.train_dir:
        d["data"]["train_dir"] = args.train_dir
    if args.val_dir:
        d["data"]["val_dir"] = args.val_dir
    if args.epochs:
        d["schedule"]["epochs"] = args.epochs
    if args.output_dir:
        d["checkpoint"]["output_dir"] = args.output_dir
    if args.experiment:
        d["tracking"]["experiment_name"] = args.experiment
    if args.no_tracking:
        d["tracking"]["enabled"] = False
    if args.tracking_uri:
        d["tracking"]["tracking_uri"] = args.tracking_uri
    if args.set:
        d = _apply_overrides(d, args.set)
    return TrainConfig.model_validate(d)


def main() -> int:
    ap = argparse.ArgumentParser(description="Train an image classifier from a YAML config.")
    ap.add_argument("--config", type=str, help="Path to a YAML config.")
    ap.add_argument("--preset", choices=[p.value for p in Preset if p != Preset.CUSTOM])
    ap.add_argument("--train-dir"), ap.add_argument("--val-dir")
    ap.add_argument("--epochs", type=int), ap.add_argument("--output-dir")
    ap.add_argument("--experiment"), ap.add_argument("--tracking-uri")
    ap.add_argument("--no-tracking", action="store_true")
    ap.add_argument("--set", nargs="*", action="extend", default=[], metavar="KEY=VALUE",
                    help="Dotted-path overrides, e.g. optimization.lr=1e-4. "
                         "Repeatable; later values win. Use `null` to clear a nullable "
                         "field ('none' is a literal enum value on several fields).")
    ap.add_argument("--print-config", action="store_true",
                    help="Resolve the config, print it as YAML, and exit.")
    ap.add_argument("--force", action="store_true",
                    help="Start even when validation reports error-severity findings.")
    args = ap.parse_args()

    cfg = build_config(args)
    if args.print_config:
        print(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))
        return 0

    # The CLI is the one entry point with no warning panel, so give the validation
    # gate the same information the UI has: dataset size and imbalance unlock the
    # checks that matter most headlessly (rrc_scale on a small dataset, EMA horizon,
    # focal on balanced data). Scanning the manifest needs PIL, not torch, so
    # --print-config above stays instant. Advisory only — a broken data dir surfaces
    # with its own clearer error at setup.
    n_train = imbalance_ratio = None
    try:
        from trainlab import data as data_mod

        m = data_mod.load_manifest(cfg, "train")
        if m is not None:
            n_train = len(m)
            imbalance_ratio = m.imbalance_ratio
    except Exception:
        pass

    # Import torch only after config resolution, so --print-config stays instant.
    from trainlab.device import resolve_runtime
    from trainlab.tracking.mlflow_adapter import build_tracker

    rt = resolve_runtime(cfg)
    name = cfg.tracking.run_name or naming.run_name(cfg)

    log(f"\n▶ {name}")
    log(f"  device: {rt.device_name} ({rt.device_str})"
        + (f", amp {str(rt.amp_dtype).replace('torch.', '')}" if rt.amp_enabled else ", amp off"))
    for d in rt.downgrades:
        log(f"  ! {d}")

    # WARNING/INFO findings are advisory and never block. ERROR findings describe a
    # config that cannot produce a valid result, and used to be printed and then ignored
    # — so a run the UI would refuse to launch proceeded happily from the CLI and failed
    # later, or worse, didn't.
    findings = cfg.warnings(n_train=n_train, imbalance_ratio=imbalance_ratio,
                            resolved_device=rt.device_str)
    for w in findings:
        log(f"  [{w.severity.value}] {w.message}" + (f"  -> {w.fix}" if w.fix else ""))
    fatal = [w for w in findings if w.severity == Severity.ERROR]
    if fatal and not args.force:
        log("\n✗ refusing to start: " + "; ".join(w.message.split(".")[0] for w in fatal))
        log("  Fix the above, or pass --force to run anyway.")
        return 2

    # The API cancels runs with SIGTERM to the process group. Python's default SIGTERM
    # disposition kills the process without unwinding, so the `finally` below never ran:
    # the MLflow run stayed RUNNING forever, no "cancelled" event was emitted, and the
    # Compare tab showed cancelled runs as live indefinitely. Route it through the same
    # KeyboardInterrupt path Ctrl-C uses so cancellation is recorded like any other end.
    #
    # The flag matters as much as the raise: the group signal also kills the dataloader
    # workers, and their death can surface in the main process as a RuntimeError before
    # our KeyboardInterrupt lands. Any exception that follows a SIGTERM is part of the
    # cancellation, not a training failure.
    got_sigterm = {"flag": False}

    def _sigterm(_signum, _frame):
        got_sigterm["flag"] = True
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _sigterm)

    out_dir = Path(cfg.checkpoint.output_dir) / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_yaml = yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False)
    (out_dir / "config.yaml").write_text(cfg_yaml)

    tracker = build_tracker(cfg)
    trainer = Trainer(cfg, rt, tracker, out_dir)
    status, result = "FINISHED", {}
    try:
        # Start the run BEFORE setup: setup can emit notes (e.g. the EMA-horizon warning)
        # via tracker.set_tags, and those must land on this run rather than implicitly
        # opening an anonymous one.
        run_id = tracker.start_run(
            run_name=name,
            tags=naming.run_tags(cfg, {"downgrades": "; ".join(rt.downgrades) or "none"}),
            experiment=cfg.tracking.experiment_name,
        )
        emit("started", run_id=run_id, run_name=name, output_dir=str(out_dir),
             device=rt.device_str, downgrades=rt.downgrades)

        params = naming.flatten(cfg.model_dump(mode="json"))
        params.update(env.full_context(rt))
        tracker.log_params(params)
        tracker.log_text(cfg_yaml, "config.yaml")

        trainer.setup()

        # The dataset fingerprint is only knowable once the manifest is scanned.
        tracker.log_params({
            f"data.fingerprint.{k}": json.dumps(v) if isinstance(v, dict) else str(v)
            for k, v in trainer.train_m.fingerprint().items()
        })

        result = trainer.fit()
        log(f"\n✓ best {result['primary_metric']} = {result['best_metric']:.4f} "
            f"at epoch {result['best_epoch'] + 1}")
        emit("finished", **{k: v for k, v in result.items() if k != "history"},
             run_id=run_id, output_dir=str(out_dir))

    except KeyboardInterrupt:
        status = "KILLED"
        log("\n✗ interrupted")
        emit("cancelled")
        return 130
    except Exception as e:
        if got_sigterm["flag"]:
            # Collateral of the group SIGTERM (dead workers, broken pipes), not a
            # training failure — record it as the cancellation it is.
            status = "KILLED"
            log("\n✗ interrupted")
            emit("cancelled")
            return 130
        status = "FAILED"
        log(f"\n✗ {type(e).__name__}: {e}")
        emit("failed", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        tracker.set_tags({"status": status})
        tracker.end_run(status)

    (out_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
