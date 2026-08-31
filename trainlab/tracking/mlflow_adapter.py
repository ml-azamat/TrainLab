"""MLflow implementation of the Tracker protocol."""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

#: MLflow rejects params over 6000 chars and batches of more than 100 at a time.
_MAX_PARAM_LEN = 5900
_PARAM_BATCH = 100


class MLflowTracker:
    def __init__(self, tracking_uri: str):
        import mlflow

        self.mlflow = mlflow
        mlflow.set_tracking_uri(tracking_uri)
        self.run_id: str | None = None
        self._active = False

    def start_run(self, *, run_name: str, tags: dict[str, str], experiment: str) -> str:
        # MLflow's fluent set_tags/log_metrics implicitly start a run when none is active.
        # If anything did that before us, adopt-and-close it rather than colliding.
        if self.mlflow.active_run() is not None:
            self.mlflow.end_run()
        self.mlflow.set_experiment(experiment)
        run = self.mlflow.start_run(run_name=run_name, tags=tags)
        self.run_id = run.info.run_id
        self._active = True
        return self.run_id

    def log_params(self, params: dict[str, str]) -> None:
        clean = {k: (v[:_MAX_PARAM_LEN] if isinstance(v, str) else v) for k, v in params.items()}
        items = list(clean.items())
        for i in range(0, len(items), _PARAM_BATCH):
            try:
                self.mlflow.log_params(dict(items[i:i + _PARAM_BATCH]))
            except Exception as e:  # a bad param must never kill a training run
                log.warning("mlflow.log_params failed: %s", e)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        clean = {
            k.replace("@", "_at_").replace(" ", "_"): float(v)
            for k, v in metrics.items()
            if v is not None and not (isinstance(v, float) and v != v)  # drop NaN
        }
        if not clean:
            return
        try:
            self.mlflow.log_metrics(clean, step=step)
        except Exception as e:
            log.warning("mlflow.log_metrics failed: %s", e)

    def log_artifact(self, path: Path, artifact_dir: str | None = None) -> None:
        try:
            self.mlflow.log_artifact(str(path), artifact_path=artifact_dir)
        except Exception as e:
            log.warning("mlflow.log_artifact(%s) failed: %s", path, e)

    def log_text(self, text: str, filename: str) -> None:
        try:
            self.mlflow.log_text(text, filename)
        except Exception as e:
            log.warning("mlflow.log_text(%s) failed: %s", filename, e)

    def set_tags(self, tags: dict[str, str]) -> None:
        try:
            self.mlflow.set_tags({k: str(v) for k, v in tags.items()})
        except Exception as e:
            log.warning("mlflow.set_tags failed: %s", e)

    def end_run(self, status: str = "FINISHED") -> None:
        if self._active:
            try:
                self.mlflow.end_run(status=status)
            except Exception as e:
                # This was the one unguarded call in the adapter, and it sits in
                # train.py's `finally`: a tracker that died mid-run turned a
                # SUCCESSFULLY finished training run into exit code 1, status
                # FAILED, and no result.json. The run record staying RUNNING on the
                # dead server is the tracker's problem, not the training run's.
                log.warning("mlflow.end_run failed: %s", e)
            finally:
                self._active = False


def build_tracker(cfg) -> object:
    """Factory: returns a Tracker for the configured backend."""
    from .base import NoopTracker

    if not cfg.tracking.enabled or cfg.tracking.backend == "none":
        return NoopTracker()
    try:
        return MLflowTracker(cfg.tracking.tracking_uri)
    except Exception as e:
        log.warning("Falling back to no-op tracker: could not reach MLflow (%s)", e)
        return NoopTracker()
