"""Hyperparameter sweeps.

The search space is stored SEPARATELY from TrainConfig (keyed by dotted path), so a
plain config stays a plain YAML file and train.py never needs to know sweeps exist.
Each trial is an ordinary tracked run tagged with `sweep_id`, which is what makes sweep
results show up in the same table, parallel-coordinates and diff views as hand-launched
runs.
"""

from __future__ import annotations

import copy
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from trainlab.config import metric_higher_is_better, with_aug_preset

#: The one swept path that names a whole group rather than a single value.
AUG_PRESET_PATH = "augmentation.preset"

from .runner import Run, RunManager


class SearchSpace(BaseModel):
    """One searchable parameter."""
    kind: Literal["list", "range", "log_range", "int_range"]
    values: list[Any] | None = None          # kind='list'
    low: float | None = None
    high: float | None = None
    step: float | None = None


class SweepConfig(BaseModel):
    base_config: dict
    parameters: dict[str, SearchSpace] = Field(default_factory=dict)
    algorithm: Literal["grid", "random", "tpe"] = "tpe"
    budget: int = 20
    max_concurrent: int = 1
    metric: str = "acc_at_1"
    #: None means "derive from the metric", which is almost always what you want:
    #: an explicit default of "maximize" silently minimised nothing and happily
    #: maximised val_loss.
    direction: Literal["maximize", "minimize"] | None = None
    pruning: Literal["none", "median", "hyperband"] = "median"
    experiment_name: str = "default"

    @property
    def resolved_direction(self) -> str:
        if self.direction is not None:
            return self.direction
        return "maximize" if metric_higher_is_better(self.metric) else "minimize"

    @property
    def metric_key(self) -> str:
        """The metric name as the engine emits it (`acc@1`, not the tracker's `acc_at_1`)."""
        return self.metric.replace("_at_", "@")


@dataclass
class Sweep:
    id: str
    config: SweepConfig
    status: str = "running"
    trials: list[dict] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    best: dict | None = None
    error: str | None = None
    _stop: threading.Event = field(default_factory=threading.Event)

    def summary(self) -> dict:
        return {
            "id": self.id, "status": self.status,
            "algorithm": self.config.algorithm, "budget": self.config.budget,
            "metric": self.config.metric, "direction": self.config.resolved_direction,
            "completed": len([t for t in self.trials if t["status"] == "finished"]),
            "total": len(self.trials), "best": self.best,
            "started_at": self.started_at,
            "parameters": list(self.config.parameters),
        }


class _SweepStopped(BaseException):
    """Raised inside a trial when the user stops the sweep.

    A BaseException on purpose, so `study.optimize(..., catch=(Exception,))` does not
    swallow it and quietly carry on to the next trial. `_drive` catches it explicitly —
    which is what the old bare `KeyboardInterrupt` failed to do, leaving a stopped sweep
    wedged in "stopping" forever because `except Exception` never matched it.
    """


def _set_path(d: dict, path: str, value: Any) -> None:
    node = d
    parts = path.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def _trial_config(base_config: dict, parameters: dict[str, "SearchSpace"],
                  trial) -> tuple[dict, dict[str, Any]]:
    """One trial's config, plus the values chosen for it.

    Split out of `_objective` so the assembly can be tested without a subprocess.
    """
    config = copy.deepcopy(base_config)
    chosen: dict[str, Any] = {}
    params = dict(parameters)

    # An augmentation rung names a whole group, so it is expanded before the rest: a sweep
    # over the rung *and* one of its knobs must leave the knob at the swept value, not at
    # whatever the rung says. Setting it through `_set_path` like any other scalar would
    # relabel the group and change no pixels — see `with_aug_preset`.
    rung = params.pop(AUG_PRESET_PATH, None)
    if rung is not None:
        v = _suggest(trial, AUG_PRESET_PATH, rung)
        config["augmentation"] = with_aug_preset(config.get("augmentation") or {}, v)
        chosen[AUG_PRESET_PATH] = v

    for path, space in params.items():
        v = _suggest(trial, path, space)
        _set_path(config, path, v)
        chosen[path] = v
    return config, chosen


def _suggest(trial, path: str, space: SearchSpace) -> Any:
    name = path
    if space.kind == "list":
        if not space.values:
            raise ValueError(f"search space '{path}' is kind='list' but has no values")
        return trial.suggest_categorical(name, space.values)
    if space.low is None or space.high is None:
        raise ValueError(f"search space '{path}' (kind={space.kind}) needs low and high")
    if space.low > space.high:
        raise ValueError(
            f"search space '{path}' has low ({space.low}) > high ({space.high})")
    if space.kind == "int_range":
        # A fractional step truncates to 0 and makes suggest_int raise; round up so a
        # step of 0.5 behaves like the 1 the user must have meant.
        return trial.suggest_int(name, int(space.low), int(space.high),
                                 step=max(1, int(space.step or 1)))
    if space.kind == "log_range":
        if space.low <= 0:
            raise ValueError(
                f"search space '{path}' is kind='log_range' but low is {space.low}; "
                f"a log scale needs a strictly positive lower bound")
        return trial.suggest_float(name, space.low, space.high, log=True)
    return trial.suggest_float(name, space.low, space.high, step=space.step)


class SweepManager:
    def __init__(self, run_manager: RunManager):
        self.rm = run_manager
        self.sweeps: dict[str, Sweep] = {}

    def start(self, cfg: SweepConfig) -> Sweep:
        sweep = Sweep(id=uuid.uuid4().hex[:10], config=cfg)
        self.sweeps[sweep.id] = sweep
        threading.Thread(target=self._drive, args=(sweep,), daemon=True).start()
        return sweep

    def _drive(self, sweep: Sweep) -> None:
        try:
            import optuna

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            cfg = sweep.config

            sampler = {
                "grid": optuna.samplers.GridSampler(self._grid(cfg)),
                "random": optuna.samplers.RandomSampler(seed=42),
                "tpe": optuna.samplers.TPESampler(seed=42),
            }[cfg.algorithm]
            pruner = {
                "none": optuna.pruners.NopPruner(),
                "median": optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2),
                "hyperband": optuna.pruners.HyperbandPruner(),
            }[cfg.pruning]

            study = optuna.create_study(direction=cfg.resolved_direction, sampler=sampler,
                                        pruner=pruner, study_name=f"sweep-{sweep.id}")
            # Each trial thread only supervises a subprocess, so concurrency here costs
            # threads rather than GPU memory — but the trials themselves share one
            # accelerator, so this stays opt-in and defaults to sequential.
            n_jobs = max(1, min(cfg.max_concurrent, cfg.budget))
            study.optimize(lambda t: self._objective(sweep, t), n_trials=cfg.budget,
                           n_jobs=n_jobs, catch=(Exception,))

            sweep.status = "stopped" if sweep._stop.is_set() else "finished"
        except _SweepStopped:
            # Raised by stop(); not an error, and it must not leave the sweep wedged in
            # "stopping" the way an escaping BaseException did.
            sweep.status = "stopped"
        except BaseException as e:
            sweep.status = "failed"
            sweep.error = f"{type(e).__name__}: {e}"
        finally:
            if sweep.status in ("running", "stopping"):
                sweep.status = "stopped" if sweep._stop.is_set() else "finished"

    def _grid(self, cfg: SweepConfig) -> dict:
        grid: dict[str, list] = {}
        for path, sp in cfg.parameters.items():
            if sp.kind == "list":
                grid[path] = list(sp.values or [])
            elif sp.kind == "int_range":
                step = int(sp.step or 1)
                grid[path] = list(range(int(sp.low), int(sp.high) + 1, step))
            else:
                n = 5
                lo, hi = sp.low, sp.high
                if sp.kind == "log_range":
                    import math
                    grid[path] = [10 ** (math.log10(lo) + i * (math.log10(hi) - math.log10(lo)) / (n - 1))
                                  for i in range(n)]
                else:
                    grid[path] = [lo + i * (hi - lo) / (n - 1) for i in range(n)]
        return grid

    @staticmethod
    def _observed(run: Run, cfg: SweepConfig) -> float | None:
        """Best value of the DECLARED sweep metric seen so far in this run.

        Previously the objective was `run.latest['best']`, i.e. the best-so-far of the
        run's own `validation.primary_metric` — so a sweep configured to optimise
        macro-F1 silently optimised whatever the base config happened to be checkpointing
        on. The declared metric is now read from the per-epoch metrics directly, and
        `best` is used only when the run's primary metric IS the sweep metric.
        """
        key = cfg.metric_key
        maximize = cfg.resolved_direction == "maximize"
        values = [
            ev["metrics"][key]
            for ev in run.events
            if isinstance(ev.get("metrics"), dict) and key in ev["metrics"]
            and isinstance(ev["metrics"][key], (int, float))
        ]
        if values:
            return max(values) if maximize else min(values)

        latest = run.latest or {}
        if latest.get("primary_metric") == key and latest.get("best") is not None:
            return float(latest["best"])
        return None

    def _objective(self, sweep: Sweep, trial) -> float:
        if sweep._stop.is_set():
            raise _SweepStopped

        cfg = sweep.config
        config, chosen = _trial_config(cfg.base_config, cfg.parameters, trial)

        config.setdefault("tracking", {})["experiment_name"] = cfg.experiment_name
        config["tracking"].setdefault("tags", {}).update(
            {"sweep_id": sweep.id, "sweep_trial": str(trial.number)})

        run = self.rm.start(config, sweep_id=sweep.id)
        record = {"trial": trial.number, "run_id": run.id, "params": chosen,
                  "status": "running", "value": None}
        sweep.trials.append(record)

        # Poll the subprocess, reporting intermediate values so pruning can act. The
        # intermediate value must be the same quantity as the final objective, or pruning
        # decides on one metric while the study optimises another.
        maximize = cfg.resolved_direction == "maximize"
        last_epoch = -1
        while not run.done.wait(1.5):
            if sweep._stop.is_set():
                self.rm.cancel(run.id)
                raise _SweepStopped
            latest = run.latest or {}
            ep = latest.get("epoch")
            if ep is not None and ep > last_epoch:
                last_epoch = ep
                val = (latest.get("metrics") or {}).get(cfg.metric_key)
                if val is not None:
                    # Report the RAW value. Optuna's pruners read the study's own
                    # direction (see optuna/pruners/_percentile.py), so the study
                    # created with `direction=cfg.resolved_direction` already prunes
                    # the correct tail. This used to negate for minimize objectives
                    # on the belief that pruners assume larger-is-better — which
                    # inverted the comparison and pruned the BEST val_loss trials
                    # while keeping the worst.
                    trial.report(float(val), ep)
                    if trial.should_prune():
                        self.rm.cancel(run.id)
                        record["status"] = "pruned"
                        import optuna
                        raise optuna.TrialPruned()

        # `run.done` is set only after the output pipes have drained, so the final
        # epoch's metrics are guaranteed to be visible here. Polling `proc.poll()`
        # raced the reader threads and could miss the last epoch entirely.
        record["status"] = run.status
        record["mlflow_run_id"] = run.mlflow_run_id
        value = self._observed(run, cfg)
        if value is None:
            record["status"] = "failed" if run.status != "cancelled" else "cancelled"
            raise RuntimeError(
                run.error
                or f"trial produced no value for metric '{cfg.metric}'. Check that it is "
                   f"listed in validation.metrics of the base config."
            )

        record["value"] = float(value)
        record["metric"] = cfg.metric
        if sweep.best is None or (
            (maximize and record["value"] > sweep.best["value"])
            or (not maximize and record["value"] < sweep.best["value"])
        ):
            sweep.best = {"value": record["value"], "params": chosen,
                          "run_id": run.id, "trial": trial.number, "metric": cfg.metric}
        # Optuna maximises when direction='maximize'; the study is created with the
        # resolved direction, so hand back the raw value.
        return record["value"]

    def stop(self, sweep_id: str) -> bool:
        sweep = self.sweeps.get(sweep_id)
        if not sweep:
            return False
        sweep._stop.set()
        for t in sweep.trials:
            if t["status"] == "running":
                self.rm.cancel(t["run_id"])
        sweep.status = "stopping"
        return True

    def list(self) -> list[dict]:
        return sorted((s.summary() for s in self.sweeps.values()),
                      key=lambda s: -s["started_at"])

    def get(self, sweep_id: str) -> Sweep | None:
        return self.sweeps.get(sweep_id)
