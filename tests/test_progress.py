"""Loop instrumentation: the wait/compute split and what it concludes from it."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trainlab.config import TrainConfig  # noqa: E402
from trainlab.progress import (  # noqa: E402
    INPUT_BOUND_SHARE, StepMeter, fmt_duration, input_bound_advice,
)


class FakeClock:
    """Hand-cranked time, so the timings under test are exact rather than approximate."""

    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def run_epoch(steps: int, wait: float, compute: float, *, batch: int = 32,
              total: int | None = None) -> tuple[StepMeter, FakeClock]:
    clock = FakeClock()
    m = StepMeter(total if total is not None else steps, clock=clock)
    m.start()
    for _ in range(steps):
        clock.advance(wait)
        m.batch_ready()
        clock.advance(compute)
        m.step_done(batch)
    return m, clock


def test_wait_and_compute_are_accounted_separately():
    m, _ = run_epoch(10, wait=0.02, compute=0.08)
    assert m.steps == 10
    assert m.wait_s == pytest.approx(0.2)
    assert m.compute_s == pytest.approx(0.8)
    assert m.elapsed_s == pytest.approx(1.0)


def test_a_gpu_bound_run_reports_a_high_compute_share():
    """Workers keeping up shows as a wait near zero — the healthy case."""
    m, _ = run_epoch(50, wait=0.001, compute=0.1)
    assert m.compute_share > 0.98
    assert input_bound_advice(m.epoch_metrics()["compute_share"], TrainConfig()) is None


def test_an_input_bound_run_is_named_as_one():
    """The case the instrumentation exists for: fast model, slow loader."""
    m, _ = run_epoch(50, wait=0.2, compute=0.05)
    assert m.compute_share == pytest.approx(0.2, abs=0.01)
    advice = input_bound_advice(m.epoch_metrics()["compute_share"], TrainConfig())
    assert advice is not None and "input-bound" in advice


def test_throughput_and_step_time_use_the_whole_step():
    """img/s has to price the waiting too, or it describes a run nobody is having."""
    m, _ = run_epoch(20, wait=0.05, compute=0.05, batch=64)
    assert m.step_s == pytest.approx(0.1)
    assert m.imgs_per_s == pytest.approx(640)
    assert m.epoch_metrics()["ms_per_step"] == pytest.approx(100, abs=1)


def test_eta_counts_the_steps_that_are_left():
    m, _ = run_epoch(10, wait=0.02, compute=0.08, total=100)
    assert m.eta_epoch_s == pytest.approx(90 * 0.1, rel=0.05)


def test_smoothing_follows_a_change_in_pace():
    """The first steps of an epoch are not representative — autotuning, warmup, compile —
    so the live figures have to move off them rather than average them in forever."""
    clock = FakeClock()
    m = StepMeter(100, clock=clock)
    m.start()
    for _ in range(3):                       # slow start
        clock.advance(0.5); m.batch_ready()
        clock.advance(0.5); m.step_done(32)
    slow = m.step_s
    for _ in range(60):                      # settled pace
        clock.advance(0.01); m.batch_ready()
        clock.advance(0.04); m.step_done(32)
    assert slow > 0.9
    assert m.step_s == pytest.approx(0.05, abs=0.01)


def test_reporting_is_throttled_by_time_not_by_step_count():
    clock = FakeClock()
    m = StepMeter(1000, report_every_s=10.0, clock=clock)
    m.start()
    clock.advance(0.05); m.batch_ready()
    clock.advance(0.05); m.step_done(32)
    assert m.due()              # the first step always reports
    clock.advance(0.1); m.batch_ready()
    clock.advance(0.1); m.step_done(32)
    assert not m.due()
    clock.advance(9.0)
    assert not m.due()
    clock.advance(2.0)
    assert m.due()
    assert not m.due()          # the window restarts on report


def test_an_epoch_shorter_than_the_report_interval_still_says_something():
    """A 5-second epoch used to go by in silence, then print only its summary."""
    clock = FakeClock()
    m = StepMeter(50, report_every_s=10.0, clock=clock)
    m.start()
    reported = 0
    for _ in range(50):
        clock.advance(0.01); m.batch_ready()
        clock.advance(0.04); m.step_done(32)
        reported += m.due()
    assert reported == 1


def test_an_epoch_with_no_steps_does_not_divide_by_zero():
    clock = FakeClock()
    m = StepMeter(0, clock=clock)
    m.start()
    assert m.imgs_per_s == 0.0 and m.compute_share == 0.0
    assert m.epoch_metrics()["ms_per_step"] >= 0
    assert m.line(loss=0.0, lr=1e-4)


def test_epoch_metrics_are_exact_sums_not_smoothed():
    """These land in the tracker and get compared across runs, where a smoothed value
    would depend on where in the epoch the averaging happened to be."""
    m, _ = run_epoch(10, wait=0.02, compute=0.08)
    em = m.epoch_metrics()
    assert em["data_wait_s"] == pytest.approx(0.2)
    assert em["compute_s"] == pytest.approx(0.8)
    assert em["compute_share"] == pytest.approx(0.8, abs=0.01)


def test_advice_does_not_suggest_what_is_already_set():
    """Telling a run with 32 workers to raise num_workers wastes the one line that had
    the reader's attention."""
    cfg = TrainConfig.model_validate({
        "data": {"num_workers": 32, "prefetch_factor": 8, "cache_mode": "ram",
                 "persistent_workers": True},
    })
    advice = input_bound_advice(0.2, cfg)
    assert advice is not None
    assert "num_workers" not in advice and "prefetch_factor" not in advice
    assert "cache_mode" not in advice and "persistent_workers" not in advice
    assert "input_size" in advice           # the fallback lever always applies


def test_input_bound_threshold_is_the_documented_one():
    cfg = TrainConfig()
    assert input_bound_advice(INPUT_BOUND_SHARE, cfg) is None
    assert input_bound_advice(INPUT_BOUND_SHARE - 0.01, cfg) is not None


@pytest.mark.parametrize("seconds,expected", [
    (0, "0s"), (45, "45s"), (90, "1m 30s"), (3600, "1h 00m"), (5430, "1h 30m"),
    (None, "?"), (float("nan"), "?"),
])
def test_duration_formatting(seconds, expected):
    assert fmt_duration(seconds) == expected
