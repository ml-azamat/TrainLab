"""Per-step timing for the training loop.

Splits the wall clock into the two things that can be the bottleneck — waiting for the
next batch, and running the model on it — so a run that is starving the GPU says so
instead of merely being slow.

Two properties of the loop make the split mean what it says:

* **Dataloader workers prefetch.** `wait` is therefore NOT how long loading a batch takes;
  it is how much of that loading the main process could not hide behind compute. Workers
  keeping up shows as a wait near zero, which is the healthy case — the reading is "the
  GPU was idle this long waiting for input", not "loading is free".
* **CUDA is asynchronous.** The `compute` window is only real because the loop
  synchronises once per iteration, at the `loss.item()` in `train_epoch`. Without a sync
  inside the window it would measure how long it takes to *enqueue* kernels, and every
  run would look input-bound with a suspiciously fast compute time. If that `.item()`
  ever moves, this measurement moves with it.

Live figures are exponentially smoothed, because the first steps of an epoch are not
representative (cudnn autotuning, allocator warmup, `torch.compile`) and a raw
instantaneous rate is unreadable. Epoch totals are exact sums, and are what gets logged
to the tracker.
"""

from __future__ import annotations

import time
from typing import Callable

#: Seconds between human-readable progress lines. Time-based rather than every-N-steps so
#: the cadence does not depend on how fast a step happens to be.
REPORT_EVERY_S = 10.0

#: Below this share of wall clock spent computing, the input pipeline is the bottleneck
#: and the run is worth reconfiguring rather than waiting out.
INPUT_BOUND_SHARE = 0.7


def fmt_duration(seconds: float | None) -> str:
    """Compact duration: `45s`, `12m 30s`, `1h 04m`."""
    if seconds is None or seconds != seconds or seconds < 0:      # None/NaN/negative
        return "?"
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m"


class StepMeter:
    """Times one epoch of the training loop, one batch at a time.

    Usage mirrors the shape of the loop::

        meter = StepMeter(len(loader))
        meter.start()
        for batch in loader:
            meter.batch_ready()      # closes the wait window
            ...                      # forward, backward, step, and the .item() sync
            meter.step_done(n)       # closes the compute window, reopens the wait one
    """

    def __init__(self, total_steps: int, *, ema_decay: float = 0.9,
                 report_every_s: float = REPORT_EVERY_S,
                 clock: Callable[[], float] = time.perf_counter) -> None:
        self.total_steps = max(1, int(total_steps))
        self.ema_decay = ema_decay
        self.report_every_s = report_every_s
        self._clock = clock

        self.steps = 0
        self.images = 0
        self.wait_s = 0.0
        self.compute_s = 0.0

        self._wait_ema: float | None = None
        self._compute_ema: float | None = None
        self._t_start = 0.0
        self._t_mark = 0.0
        self._t_report = 0.0

    # ---------------------------------------------------------------- recording

    def start(self) -> None:
        self._t_start = self._t_mark = self._t_report = self._clock()

    def batch_ready(self) -> None:
        """The loader handed over a batch: everything since the last step was waiting."""
        now = self._clock()
        waited = now - self._t_mark
        self.wait_s += waited
        self._wait_ema = self._blend(self._wait_ema, waited)
        self._t_mark = now

    def step_done(self, n_images: int) -> None:
        """The batch is through the model and the optimizer has stepped."""
        now = self._clock()
        spent = now - self._t_mark
        self.compute_s += spent
        self._compute_ema = self._blend(self._compute_ema, spent)
        self._t_mark = now
        self.steps += 1
        self.images += int(n_images)

    def _blend(self, prev: float | None, value: float) -> float:
        return value if prev is None else self.ema_decay * prev + (1 - self.ema_decay) * value

    def due(self) -> bool:
        """True on the first step, then at most every `report_every_s`.

        The first step is always worth printing: an epoch shorter than the reporting
        interval would otherwise go by in silence, and the pace it establishes is what
        tells you whether to keep waiting or kill the run.
        """
        now = self._clock()
        if self.steps <= 1 or now - self._t_report >= self.report_every_s:
            self._t_report = now
            return True
        return False

    # ---------------------------------------------------------------- live figures

    @property
    def elapsed_s(self) -> float:
        return max(0.0, self._t_mark - self._t_start)

    @property
    def step_s(self) -> float:
        """Smoothed seconds per batch, wait included — what ETA is built from."""
        return (self._wait_ema or 0.0) + (self._compute_ema or 0.0)

    @property
    def imgs_per_s(self) -> float:
        """Throughput at the current smoothed pace, using the average batch size."""
        if not self.steps or self.step_s <= 0:
            return 0.0
        return (self.images / self.steps) / self.step_s

    @property
    def compute_share(self) -> float:
        """Fraction of the smoothed step spent computing rather than waiting for data.

        1.0 means the input pipeline never made the model wait. Low means the GPU is
        idling on the loader — see `INPUT_BOUND_SHARE`.
        """
        total = self.step_s
        return (self._compute_ema or 0.0) / total if total > 0 else 0.0

    @property
    def eta_epoch_s(self) -> float:
        return max(0, self.total_steps - self.steps) * self.step_s

    # ---------------------------------------------------------------- output

    def snapshot(self) -> dict:
        """Live figures for the progress event the UI consumes."""
        return {
            "imgs_per_s": round(self.imgs_per_s, 1),
            "ms_per_step": round(self.step_s * 1000, 1),
            "ms_data_wait": round((self._wait_ema or 0.0) * 1000, 1),
            "ms_compute": round((self._compute_ema or 0.0) * 1000, 1),
            "compute_share": round(self.compute_share, 3),
            "eta_epoch_s": round(self.eta_epoch_s, 1),
            "elapsed_s": round(self.elapsed_s, 1),
        }

    def epoch_metrics(self) -> dict:
        """Exact epoch totals, for the tracker. Sums, not smoothed values."""
        elapsed = max(1e-9, self.elapsed_s)
        return {
            "data_wait_s": round(self.wait_s, 2),
            "compute_s": round(self.compute_s, 2),
            "imgs_per_s": round(self.images / elapsed, 1),
            "compute_share": round(self.compute_s / elapsed, 3),
            "ms_per_step": round(elapsed / max(1, self.steps) * 1000, 1),
        }

    def line(self, *, loss: float, lr: float, eta_run_s: float | None = None) -> str:
        """One human-readable progress line."""
        pct = 100 * self.steps / self.total_steps
        eta = f"eta {fmt_duration(self.eta_epoch_s)}"
        if eta_run_s is not None:
            eta += f" (run {fmt_duration(eta_run_s)})"
        return (
            f"    {self.steps}/{self.total_steps} ({pct:.0f}%)  "
            f"loss {loss:.4f}  lr {lr:.2e}  "
            f"{self.imgs_per_s:,.0f} img/s  {self.step_s * 1000:.0f} ms/step "
            f"[data {(self._wait_ema or 0) * 1000:.0f} + compute {(self._compute_ema or 0) * 1000:.0f}]  "
            f"compute {self.compute_share * 100:.0f}%  {eta}"
        )


def input_bound_advice(share: float, cfg) -> str | None:
    """A line explaining what to change when the loader is the bottleneck, or None.

    Deliberately names the fields as the form spells them, and reads the ones already set
    so it does not suggest what is in place: 'raise num_workers' is useless advice to a
    run that has 32 of them and is bound by disk instead.
    """
    if share >= INPUT_BOUND_SHARE:
        return None
    d = cfg.data
    fixes: list[str] = []
    if d.num_workers < 8:
        fixes.append(f"raise data.num_workers (currently {d.num_workers})")
    if d.prefetch_factor < 4:
        fixes.append(f"raise data.prefetch_factor (currently {d.prefetch_factor})")
    if d.cache_mode.value == "none":
        fixes.append("set data.cache_mode='ram' if the dataset fits, or 'disk-decoded'")
    if not d.persistent_workers:
        fixes.append("enable data.persistent_workers")
    fixes.append(f"or lower input.input_size (currently {cfg.input.input_size}px), "
                 f"which cuts decode and resize cost")
    return (f"input-bound: only {share * 100:.0f}% of the step is compute, the rest is the "
            f"GPU waiting for data. Try: " + "; ".join(fixes))
