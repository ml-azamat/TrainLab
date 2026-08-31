"""Tests for the API layer: run supervision, SSE fan-out, sweeps and the diff.

These used the real subprocess machinery rather than mocking it, because the defects
they cover were all about timing between the process, the pipe-reader threads and the
consumer — which a mock would define away.
"""

from __future__ import annotations

import asyncio
import sys
import textwrap
import time

import pytest

from backend.app import registry
from backend.app.runner import RunManager
from backend.app.sweeps import SearchSpace, SweepConfig, _SweepStopped, SweepManager


# --------------------------------------------------------------------------- helpers

@pytest.fixture
def manager(tmp_path):
    return RunManager(workdir=tmp_path / "cfgs")


def _script(manager: RunManager, body: str, monkeypatch) -> None:
    """Point the manager at a stub 'trainer' instead of train.py."""
    path = manager.workdir / "fake_train.py"
    path.write_text(textwrap.dedent(body))

    real_popen_cmd = []

    import subprocess

    orig = subprocess.Popen

    def patched(cmd, *a, **kw):
        # cmd is [python, -u, train.py, --config, x.yaml]; swap the script.
        new = list(cmd)
        new[2] = str(path)
        real_popen_cmd.append(new)
        return orig(new, *a, **kw)

    monkeypatch.setattr(subprocess, "Popen", patched)


EMIT = 'print("@@TRAINLAB@@ " + __import__("json").dumps(ev), flush=True)'


# ------------------------------------------------------------------------ run manager

def test_final_epoch_is_visible_once_done_is_set(manager, monkeypatch):
    """`proc.poll()` returning None-no-longer raced the pipe readers: a consumer could
    see the process exit before the last epoch event had been parsed, and conclude the
    trial produced no metric."""
    _script(manager, f'''
        import json, sys
        for i in range(3):
            ev = {{"event": "epoch", "epoch": i, "total": 3,
                   "metrics": {{"acc@1": 0.5 + i / 10}}, "best": 0.5 + i / 10,
                   "primary_metric": "acc@1", "higher_is_better": True}}
            {EMIT}
        sys.exit(0)
    ''', monkeypatch)

    run = manager.start({"schedule": {"epochs": 3}})
    assert manager.wait(run, timeout=60), "run did not finish"

    # done implies drained: the last epoch must already be in latest.
    assert run.latest["epoch"] == 2
    assert run.latest["metrics"]["acc@1"] == pytest.approx(0.7)
    assert run.status == "finished"
    assert len(run.events) == 3


def test_failed_run_reports_nonzero_exit(manager, monkeypatch):
    _script(manager, '''
        import sys
        print("boom", file=sys.stderr, flush=True)
        sys.exit(3)
    ''', monkeypatch)
    run = manager.start({})
    assert manager.wait(run, timeout=60)
    assert run.status == "failed"
    assert "3" in (run.error or "")


def test_cancel_marks_the_run_cancelled(manager, monkeypatch):
    _script(manager, '''
        import time
        while True:
            time.sleep(0.2)
    ''', monkeypatch)
    run = manager.start({})
    time.sleep(1.5)
    assert manager.cancel(run.id) is True
    assert manager.wait(run, timeout=60)
    assert run.status == "cancelled"


def test_slow_subscriber_never_loses_the_terminal_status(manager, monkeypatch):
    """A full queue silently swallowed whatever arrived next — including the status
    frame, which left the UI showing a finished run as still running."""
    _script(manager, f'''
        import sys
        for i in range(4000):
            print("line %d" % i, file=sys.stderr, flush=True)
        ev = {{"event": "epoch", "epoch": 0, "total": 1, "metrics": {{"acc@1": 1.0}},
               "best": 1.0, "primary_metric": "acc@1", "higher_is_better": True}}
        {EMIT}
        sys.exit(0)
    ''', monkeypatch)

    async def scenario():
        loop = asyncio.get_running_loop()
        manager.bind_loop(loop)
        run = manager.start({})
        q = await manager.subscribe(run.id)
        await loop.run_in_executor(None, manager.wait, run, 120)
        await asyncio.sleep(0.3)          # let queued call_soon_threadsafe callbacks run

        drained = []
        while not q.empty():
            drained.append(q.get_nowait())
        return run, drained

    run, drained = asyncio.run(scenario())
    kinds = [m.get("type") for m in drained]
    assert "status" in kinds, "terminal status frame was dropped under back-pressure"
    statuses = [m for m in drained if m.get("type") == "status"]
    assert statuses[-1]["status"] == "finished"
    # Log loss is acceptable and must be accounted for, not silent.
    assert run.dropped_events > 0
    assert any(m.get("type") == "event" for m in drained), "epoch event was dropped"


def test_finished_runs_are_evicted_once_the_list_grows(manager, monkeypatch):
    from backend.app import runner as runner_mod

    monkeypatch.setattr(runner_mod, "MAX_RETAINED_RUNS", 5)
    _script(manager, "import sys; sys.exit(0)", monkeypatch)
    ids = []
    for _ in range(9):
        r = manager.start({})
        manager.wait(r, timeout=60)
        ids.append(r.id)
    assert len(manager.runs) <= 6
    assert ids[-1] in manager.runs, "the newest run must never be evicted"


# --------------------------------------------------------------------------- presets

def test_presets_endpoint_serves_the_augmentation_ladder():
    """The form applies these client-side so the sliders move when you pick a rung. If the
    payload stops carrying them the select silently goes back to being an inert label."""
    from trainlab.config import AUG_PRESETS, AugmentationConfig, AugPreset

    from backend.app.main import get_presets

    served = {r["key"]: r["values"] for r in get_presets()["aug_presets"]}
    assert set(served) == {p.value for p in AugPreset if p != AugPreset.CUSTOM}
    controls = set(AugmentationConfig.model_fields) - {"preset"}
    for key, values in served.items():
        assert set(values) == controls, key
        assert values == AUG_PRESETS[AugPreset(key)]


# ---------------------------------------------------------------------------- sweeps

def test_sweep_config_derives_direction_from_the_metric():
    """Defaulting to 'maximize' meant a val_loss sweep maximised val_loss."""
    assert SweepConfig(base_config={}, metric="acc_at_1").resolved_direction == "maximize"
    assert SweepConfig(base_config={}, metric="macro-F1").resolved_direction == "maximize"
    assert SweepConfig(base_config={}, metric="val_loss").resolved_direction == "minimize"
    assert SweepConfig(base_config={}, metric="ece").resolved_direction == "minimize"
    # An explicit choice still wins.
    assert SweepConfig(base_config={}, metric="val_loss",
                       direction="maximize").resolved_direction == "maximize"


def test_sweep_metric_key_matches_what_the_engine_emits():
    assert SweepConfig(base_config={}, metric="acc_at_1").metric_key == "acc@1"
    assert SweepConfig(base_config={}, metric="macro-F1").metric_key == "macro-F1"


class _StubTrial:
    """Picks the first candidate of every space, so the assembly is deterministic."""
    number = 0

    def suggest_categorical(self, name, values):
        return values[0]

    def suggest_int(self, name, low, high, step=1):
        return low

    def suggest_float(self, name, low, high, step=None, log=False):
        return low


def test_sweeping_the_augmentation_rung_changes_the_pixels_not_just_the_label():
    """`augmentation.preset` names a whole group. Set like any other scalar it would only
    relabel the group to 'custom', and every trial would train identically."""
    from trainlab.config import AUG_PRESETS, AugPreset, TrainConfig

    from backend.app.sweeps import _trial_config

    base = TrainConfig().model_dump(mode="json")
    config, chosen = _trial_config(
        base, {"augmentation.preset": SearchSpace(kind="list", values=["heavy", "light"])},
        _StubTrial())

    assert chosen["augmentation.preset"] == "heavy"
    aug = TrainConfig.model_validate(config).augmentation
    assert aug.preset == AugPreset.HEAVY
    assert aug.randaugment_m == AUG_PRESETS[AugPreset.HEAVY]["randaugment_m"]


def test_a_swept_knob_survives_a_swept_rung():
    """Sweeping the rung and one of its own knobs together: the knob is the more specific
    instruction, so it must not be overwritten by whatever the rung says."""
    from trainlab.config import AugPreset, TrainConfig

    from backend.app.sweeps import _trial_config

    base = TrainConfig().model_dump(mode="json")
    config, _ = _trial_config(base, {
        "augmentation.randaugment_m": SearchSpace(kind="int_range", low=3, high=12, step=1),
        "augmentation.preset": SearchSpace(kind="list", values=["heavy"]),
    }, _StubTrial())

    aug = TrainConfig.model_validate(config).augmentation
    assert aug.randaugment_m == 3            # the swept knob, not heavy's 9
    assert aug.random_erasing_p == 0.25      # the rest of the rung still applied
    assert aug.preset == AugPreset.CUSTOM


def test_objective_reads_the_declared_metric_not_the_runs_primary():
    """The headline sweep bug: the objective was `run.latest['best']`, the best-so-far of
    the run's OWN primary metric, so a sweep on macro-F1 silently optimised acc@1."""
    class FakeRun:
        latest = {"primary_metric": "acc@1", "best": 0.99, "higher_is_better": True}
        events = [
            {"event": "epoch", "metrics": {"acc@1": 0.90, "macro-F1": 0.70}},
            {"event": "epoch", "metrics": {"acc@1": 0.99, "macro-F1": 0.80}},
        ]

    cfg = SweepConfig(base_config={}, metric="macro-F1")
    assert SweepManager._observed(FakeRun(), cfg) == pytest.approx(0.80)

    cfg_acc = SweepConfig(base_config={}, metric="acc_at_1")
    assert SweepManager._observed(FakeRun(), cfg_acc) == pytest.approx(0.99)


def test_objective_takes_the_minimum_for_a_lower_is_better_metric():
    class FakeRun:
        latest = {}
        events = [
            {"event": "epoch", "metrics": {"val_loss": 0.5}},
            {"event": "epoch", "metrics": {"val_loss": 0.2}},
            {"event": "epoch", "metrics": {"val_loss": 0.4}},
        ]

    cfg = SweepConfig(base_config={}, metric="val_loss")
    assert SweepManager._observed(FakeRun(), cfg) == pytest.approx(0.2)


def test_objective_returns_none_when_the_metric_was_never_logged():
    class FakeRun:
        latest = {"primary_metric": "acc@1", "best": 0.9}
        events = [{"event": "epoch", "metrics": {"acc@1": 0.9}}]

    assert SweepManager._observed(FakeRun(), SweepConfig(base_config={}, metric="auroc")) is None


def test_pruning_reports_raw_values_so_optuna_direction_applies():
    """Intermediate values used to be NEGATED for minimize objectives ("Optuna's
    pruners assume larger-is-better"). That premise is false — pruners read
    study.direction — so the negation inverted every pruning decision: a val_loss
    sweep with the default median pruner killed its best trials and kept the worst.
    The objective must hand optuna the raw metric value."""
    import threading

    class FlipEvent:
        """wait() times out once (so the poll body runs and reports), then is done."""
        def __init__(self):
            self.calls = 0
        def wait(self, timeout=None):
            self.calls += 1
            return self.calls > 1

    class FakeRun:
        id = "r1"
        status = "finished"
        error = None
        mlflow_run_id = "m1"
        def __init__(self):
            self.done = FlipEvent()
            self.latest = {"epoch": 0, "metrics": {"val_loss": 0.4}}
            self.events = [{"event": "epoch", "metrics": {"val_loss": 0.4}}]

    run = FakeRun()

    class FakeRM:
        def start(self, config, sweep_id=None):
            return run
        def cancel(self, run_id):
            pass

    reported: list[float] = []

    class FakeTrial:
        number = 0
        def report(self, value, step):
            reported.append(value)
        def should_prune(self):
            return False

    from backend.app.sweeps import Sweep

    cfg = SweepConfig(base_config={}, metric="val_loss", pruning="median")
    assert cfg.resolved_direction == "minimize"
    sm = SweepManager(FakeRM())
    sweep = Sweep(id="s", config=cfg)
    value = sm._objective(sweep, FakeTrial())

    assert value == pytest.approx(0.4)
    assert reported == [pytest.approx(0.4)], (
        f"intermediate values must be raw (study.direction handles the ordering); "
        f"got {reported}")


def test_stop_signal_is_not_swallowed_by_optuna_catch():
    """`KeyboardInterrupt` is a BaseException that `except Exception` missed, so a
    stopped sweep stayed in 'stopping' forever. The replacement must also not be caught
    by `study.optimize(catch=(Exception,))`."""
    assert issubclass(_SweepStopped, BaseException)
    assert not issubclass(_SweepStopped, Exception)


def test_search_space_validation_rejects_impossible_ranges():
    from backend.app.sweeps import _suggest

    class T:
        def suggest_float(self, *a, **k): return 0.0
        def suggest_int(self, *a, **k): return 0
        def suggest_categorical(self, name, values): return values[0]

    with pytest.raises(ValueError, match="no values"):
        _suggest(T(), "x", SearchSpace(kind="list", values=[]))
    with pytest.raises(ValueError, match="low and high"):
        _suggest(T(), "x", SearchSpace(kind="range", low=None, high=1))
    with pytest.raises(ValueError, match="low .* > high"):
        _suggest(T(), "x", SearchSpace(kind="range", low=5, high=1))
    with pytest.raises(ValueError, match="strictly positive"):
        _suggest(T(), "x", SearchSpace(kind="log_range", low=0, high=1))
    # A fractional int step used to truncate to 0 and make suggest_int raise.
    _suggest(T(), "x", SearchSpace(kind="int_range", low=1, high=9, step=0.5))


# ----------------------------------------------------------------------- cancellation

def test_sigterm_is_recorded_as_cancellation(tmp_path):
    """The API cancels with SIGTERM to the process group. Python's default SIGTERM
    disposition kills the process without unwinding, so train.py's cleanup never ran:
    no 'cancelled' event, exit code showed a raw signal death, and the tracker run
    stayed RUNNING forever. train.py must convert SIGTERM into the same recorded
    cancellation Ctrl-C produces (exit 130 + a cancelled event)."""
    import json
    import os
    import signal
    import subprocess
    from pathlib import Path

    import numpy as np
    import yaml
    from PIL import Image

    root = Path(__file__).resolve().parents[1]
    rng = np.random.default_rng(0)
    for cname in ("a", "b"):
        d = tmp_path / "data" / cname
        d.mkdir(parents=True)
        for i in range(8):
            arr = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
            Image.fromarray(arr).save(d / f"{i}.png")

    cfg = {
        "data": {"train_dir": str(tmp_path / "data"), "num_workers": 0, "val_split": 0.25},
        "input": {"input_size": 32},
        "model": {"backbone": "resnet18", "pretrained": False, "ema": False},
        "schedule": {"epochs": 500, "batch_size": 4, "warmup_epochs": 0,
                     "device": "cpu", "amp": "off"},
        "checkpoint": {"output_dir": str(tmp_path / "runs")},
        "tracking": {"enabled": False},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    proc = subprocess.Popen(
        [sys.executable, "-u", str(root / "train.py"), "--config", str(cfg_path)],
        cwd=str(root), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, bufsize=1, start_new_session=True,
    )
    try:
        events = []
        # Wait for training to actually be under way before signalling.
        deadline = time.time() + 120
        for line in proc.stdout:
            if line.startswith("@@TRAINLAB@@"):
                events.append(json.loads(line.split(" ", 1)[1]))
                if events[-1]["event"] in ("progress", "epoch"):
                    break
            if time.time() > deadline:
                pytest.fail("run never started")

        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        for line in proc.stdout:
            if line.startswith("@@TRAINLAB@@"):
                events.append(json.loads(line.split(" ", 1)[1]))
        code = proc.wait(timeout=60)
    finally:
        if proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)

    assert code == 130, f"expected the KeyboardInterrupt exit code, got {code}"
    assert any(e["event"] == "cancelled" for e in events), (
        f"no cancelled event was emitted; saw {[e['event'] for e in events]}")


# ---------------------------------------------------------------------------- diff

def _run(params, metrics=None):
    return {"run_id": "r", "name": "n", "status": "FINISHED", "start_time": 0,
            "end_time": 1, "duration_s": 1, "params": params,
            "metrics": metrics or {}, "tags": {}}


def test_diff_counts_identical_and_noise_separately(monkeypatch):
    """Equal noise params inflated `identical`, and noise params that DIFFERED were
    dropped from both counts, so the reported total matched neither."""
    a = _run({"optimization.lr": "0.1", "model.backbone": "resnet18",
              "env.git_commit": "aaa", "checkpoint.output_dir": "/x"})
    b = _run({"optimization.lr": "0.2", "model.backbone": "resnet18",
              "env.git_commit": "bbb", "checkpoint.output_dir": "/y"})
    monkeypatch.setattr(registry, "get_run", lambda uri, rid: a if rid == "a" else b)

    d = registry.diff("uri", "a", "b")
    assert [p["key"] for p in d["params"]] == ["optimization.lr"]
    assert d["identical_params"] == 1          # model.backbone only
    assert d["hidden_noise_params"] == 2       # git_commit + output_dir
    assert d["confounded"] is False


def test_diff_flags_multiple_differing_params_as_confounded(monkeypatch):
    a = _run({"optimization.lr": "0.1", "schedule.epochs": "10"})
    b = _run({"optimization.lr": "0.2", "schedule.epochs": "20"})
    monkeypatch.setattr(registry, "get_run", lambda uri, rid: a if rid == "a" else b)
    assert registry.diff("uri", "a", "b")["confounded"] is True


def test_parallel_coordinates_marks_log_axes_only_for_positive_ranges():
    runs = [_run({"optimization.lr": "1e-5"}, {"acc_at_1": 0.5}),
            _run({"optimization.lr": "1e-2"}, {"acc_at_1": 0.9})]
    out = registry.parallel_coordinates(runs, "acc_at_1")
    lr_axis = next(a for a in out["axes"] if a["key"] == "optimization.lr")
    assert lr_axis["log"] is True

    runs2 = [_run({"optimization.weight_decay": "0.0"}, {"acc_at_1": 0.5}),
             _run({"optimization.weight_decay": "0.1"}, {"acc_at_1": 0.9})]
    wd_axis = next(a for a in registry.parallel_coordinates(runs2, "acc_at_1")["axes"]
                   if a["key"] == "optimization.weight_decay")
    assert wd_axis["log"] is False, "a range including 0 cannot be plotted logarithmically"


# --------------------------------------------------------------------------- host guard

def test_allowed_hosts_keeps_the_defaults_when_nothing_is_named():
    from backend.app.main import allowed_hosts

    for empty in (None, "", "  ", ",,"):
        assert allowed_hosts(empty) == ["127.0.0.1", "localhost", "testserver"]


def test_allowed_hosts_adds_named_hosts_without_dropping_the_check():
    """Serving on another interface has to name the host you browse it by, or every
    request is a 400 — but the guard against DNS rebinding must survive that."""
    from backend.app.main import allowed_hosts

    out = allowed_hosts(" 192.168.1.5 , trainlab.lan ")
    assert out == ["127.0.0.1", "localhost", "testserver", "192.168.1.5", "trainlab.lan"]


def test_allowed_hosts_does_not_repeat_a_default():
    """`make api` passes $(HOST) through, which is 127.0.0.1 unless it was overridden."""
    from backend.app.main import allowed_hosts

    assert allowed_hosts("127.0.0.1,") == ["127.0.0.1", "localhost", "testserver"]
