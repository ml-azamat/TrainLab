"""Tracker adapter interface.

Training code depends only on this Protocol. Swapping MLflow for W&B / ClearML / Aim
means writing one new adapter — no changes anywhere in trainlab/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class Tracker(Protocol):
    run_id: str | None

    def start_run(self, *, run_name: str, tags: dict[str, str],
                  experiment: str) -> str: ...

    def log_params(self, params: dict[str, str]) -> None: ...

    def log_metrics(self, metrics: dict[str, float], step: int) -> None: ...

    def log_artifact(self, path: Path, artifact_dir: str | None = None) -> None: ...

    def log_text(self, text: str, filename: str) -> None: ...

    def set_tags(self, tags: dict[str, str]) -> None: ...

    def end_run(self, status: str = "FINISHED") -> None: ...


class NoopTracker:
    """Used when tracking is disabled. Keeps training code branch-free."""

    run_id: str | None = None

    def start_run(self, *, run_name, tags, experiment) -> str:
        self.run_id = "untracked"
        return self.run_id

    def log_params(self, params): ...
    def log_metrics(self, metrics, step): ...
    def log_artifact(self, path, artifact_dir=None): ...
    def log_text(self, text, filename): ...
    def set_tags(self, tags): ...
    def end_run(self, status="FINISHED"): ...
