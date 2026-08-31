#!/usr/bin/env python
"""End-to-end smoke test: config -> training -> tracker -> comparison view.

Runs two short, seeded Imagenette runs that differ in exactly ONE parameter
(`layer_lr_decay`), so afterwards the Compare tab has something meaningful to show:
a two-run diff with a single differing parameter and a real metric delta.

    make smoke          # downloads Imagenette if needed, then runs this
    python scripts/smoke_test.py --epochs 2

Exits non-zero if any stage fails.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trainlab import presets  # noqa: E402
from trainlab.config import Preset, TrainConfig  # noqa: E402

DATA = ROOT / "data" / "imagenette2-160"
TRACKING_URI = "http://127.0.0.1:5050"
EXPERIMENT = "smoke-test"


def ok(msg: str) -> None:
    print(f"  \033[32m✓\033[0m {msg}")


def fail(msg: str) -> None:
    print(f"  \033[31m✗\033[0m {msg}")


def step(n: int, msg: str) -> None:
    print(f"\n\033[1m[{n}]\033[0m {msg}")


def build_config(epochs: int, llrd: float, tracking: bool) -> TrainConfig:
    """A deliberately small, fast configuration — this tests plumbing, not accuracy."""
    cfg = presets.apply_preset(Preset.FAST_BASELINE)
    d = cfg.model_dump(mode="json")
    d["data"].update(train_dir=str(DATA / "train"), val_dir=str(DATA / "val"),
                     num_workers=4, cache_mode="none")
    d["input"].update(input_size=128, rrc_scale=[0.65, 1.0])
    d["model"].update(backbone="resnet18", pretrained=True, ema=True, ema_decay="auto")
    d["optimization"].update(lr=1e-3, layer_lr_decay=llrd)
    d["schedule"].update(epochs=epochs, batch_size=64, warmup_epochs=0.5, seed=42)
    d["validation"].update(metrics=["acc@1", "acc@5", "macro-F1"], primary_metric="acc@1")
    d["checkpoint"].update(output_dir="./runs/smoke", save_top_k=1, early_stopping=False)
    d["tracking"].update(enabled=tracking, tracking_uri=TRACKING_URI,
                         experiment_name=EXPERIMENT)
    return TrainConfig.model_validate(d)


def tracker_reachable() -> bool:
    import urllib.error
    import urllib.request
    try:
        urllib.request.urlopen(f"{TRACKING_URI}/health", timeout=3)
        return True
    except (urllib.error.URLError, OSError):
        return False


def run_training(cfg: TrainConfig, tag: str) -> tuple[bool, float | None]:
    path = ROOT / "runs" / f"smoke-{tag}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))

    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "-u", str(ROOT / "train.py"), "--config", str(path)],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    elapsed = time.time() - t0
    tail = (proc.stderr or "").strip().splitlines()

    for line in tail:
        if line.startswith(("  epoch", "  train", "  !", "✓", "✗")) or "[warning]" in line:
            print(f"      {line.strip()}")

    if proc.returncode != 0:
        fail(f"training failed (exit {proc.returncode})")
        print("\n".join(f"      {l}" for l in tail[-15:]))
        return False, None

    best = None
    for line in tail:
        if line.startswith("✓ best"):
            try:
                best = float(line.split("=")[1].split("at")[0].strip())
            except (IndexError, ValueError):
                pass
    ok(f"{tag}: finished in {elapsed:.0f}s, best acc@1 = {best}")
    return True, best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--skip-second", action="store_true",
                    help="Run only one training job (faster, no diff to compare)")
    args = ap.parse_args()

    print("\n\033[1mTrainLab smoke test\033[0m")
    print("Exercises: config schema -> YAML -> training -> tracker -> comparison view")

    # -------------------------------------------------------------- 1. dataset
    step(1, "Dataset")
    if not DATA.exists():
        fail(f"{DATA} not found. Run `make smoke` (it downloads Imagenette) "
             f"or fetch it manually.")
        return 1
    n_train = len(list((DATA / "train").rglob("*.JPEG")))
    n_val = len(list((DATA / "val").rglob("*.JPEG")))
    classes = sorted(p.name for p in (DATA / "train").iterdir() if p.is_dir())
    ok(f"Imagenette: {n_train:,} train / {n_val:,} val across {len(classes)} classes")

    # -------------------------------------------------------------- 2. schema
    step(2, "Config schema")
    cfg = build_config(args.epochs, 1.0, tracking=False)
    text = yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False)
    reloaded = TrainConfig.model_validate(yaml.safe_load(text))
    assert reloaded.model_dump(mode="json") == cfg.model_dump(mode="json")
    ok(f"YAML round-trip lossless ({len(text.splitlines())} lines)")

    warns = cfg.warnings(n_train=n_train)
    ok(f"validation produced {len(warns)} advisory message(s)")
    for w in warns:
        print(f"      [{w.severity.value}] {w.message[:88]}")

    # -------------------------------------------------------------- 3. tracker
    step(3, "Tracking server")
    tracking = tracker_reachable()
    if tracking:
        ok(f"MLflow reachable at {TRACKING_URI}")
    else:
        fail(f"MLflow not reachable at {TRACKING_URI} — continuing untracked.")
        print("      Start it with `make up-local` (no Docker) or `make up` (compose).")

    # -------------------------------------------------------------- 4. training
    step(4, f"Training run A — {args.epochs} epochs, layer_lr_decay=1.0")
    good, best_a = run_training(build_config(args.epochs, 1.0, tracking), "a-llrd1.0")
    if not good:
        return 1

    best_b = None
    if not args.skip_second:
        step(5, f"Training run B — identical except layer_lr_decay=0.75")
        good, best_b = run_training(build_config(args.epochs, 0.75, tracking), "b-llrd0.75")
        if not good:
            return 1

    # -------------------------------------------------------------- 5. compare
    if tracking:
        step(6, "Comparison view")
        sys.path.insert(0, str(ROOT / "backend"))
        from app import registry

        runs = registry.list_runs(TRACKING_URI, EXPERIMENT)
        ok(f"{len(runs)} run(s) recorded in experiment '{EXPERIMENT}'")

        varying = registry.varying_params(runs)
        ok(f"varying parameters detected: {varying or '(none)'}")

        if len(runs) >= 2:
            d = registry.diff(TRACKING_URI, runs[0]["run_id"], runs[1]["run_id"])
            ok(f"diff: {len(d['params'])} differing, {d['identical_params']} identical hidden")
            for p in d["params"]:
                print(f"      {p['key']}: {p['a']} -> {p['b']}")
            acc = next((m for m in d["metrics"] if m["key"] == "acc_at_1"), None)
            if acc and acc["delta"] is not None:
                print(f"      Δ acc@1 = {acc['delta']:+.4f}")

            pc = registry.parallel_coordinates(runs, "acc_at_1")
            ok(f"parallel-coordinates: {len(pc['axes'])} axes, {len(pc['lines'])} lines")

            cloned = registry.config_from_run(TRACKING_URI, runs[0]["run_id"])
            TrainConfig.model_validate(cloned)
            ok("clone: config recovered from tracker and re-validated")

    # -------------------------------------------------------------- done
    print("\n\033[1;32mSmoke test passed.\033[0m")
    print(f"  run A best acc@1: {best_a}")
    if best_b is not None:
        print(f"  run B best acc@1: {best_b}")
    if tracking:
        print(f"\n  MLflow UI:  {TRACKING_URI}")
        print(f"  TrainLab:   http://127.0.0.1:8000  → Compare tab → experiment "
              f"'{EXPERIMENT}'")
    print("\n  These are 2-epoch plumbing checks, not accuracy results — Imagenette "
          "reaches ~99% with the Balanced preset.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
