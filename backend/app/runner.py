"""Training-run supervision.

Runs are subprocesses (`python train.py --config <yaml>`), never in-process:
  * the API never blocks on a training step,
  * cancel is a real SIGTERM (then SIGKILL) on a process group,
  * a segfault in CUDA/MPS kernels cannot take the web server down with it.

stdout carries machine-readable `@@TRAINLAB@@ {json}` events; stderr is human log text.
Both are fanned out to any number of SSE subscribers.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from trainlab.engine import EVENT_PREFIX

REPO_ROOT = Path(__file__).resolve().parents[2]
MAX_BUFFERED_LINES = 4000
#: Per-epoch events are retained for the run detail view; a 1000-epoch sweep trial
#: would otherwise keep every one of them alive for the lifetime of the server.
MAX_EPOCH_EVENTS = 2000
#: Finished runs kept in memory. The API process is long-lived and a sweep can create
#: hundreds; the tracker is the durable record, this dict is only a live view.
MAX_RETAINED_RUNS = 500


@dataclass
class Run:
    id: str
    config: dict
    status: str = "starting"       # starting|running|finished|failed|cancelled
    proc: subprocess.Popen | None = None
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    lines: deque = field(default_factory=lambda: deque(maxlen=MAX_BUFFERED_LINES))
    events: list[dict] = field(default_factory=list)
    latest: dict = field(default_factory=dict)
    mlflow_run_id: str | None = None
    run_name: str = ""
    output_dir: str | None = None
    error: str | None = None
    sweep_id: str | None = None
    config_path: str | None = None
    #: Set only after the process has exited AND both pipe readers have drained, so a
    #: waiter that sees it is guaranteed to observe the final epoch's metrics.
    done: threading.Event = field(default_factory=threading.Event)
    dropped_events: int = 0
    _subscribers: list[asyncio.Queue] = field(default_factory=list)
    _pumps: list[threading.Thread] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "run_name": self.run_name,
            "mlflow_run_id": self.mlflow_run_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_s": (self.ended_at or time.time()) - self.started_at,
            "latest": self.latest,
            "error": self.error,
            "sweep_id": self.sweep_id,
            "output_dir": self.output_dir,
            "backbone": self.config.get("model", {}).get("backbone"),
            "epochs": self.config.get("schedule", {}).get("epochs"),
        }


class RunManager:
    """Owns every live training subprocess."""

    def __init__(self, workdir: Path | None = None):
        self.runs: dict[str, Run] = {}
        self.workdir = workdir or (REPO_ROOT / "runs" / "_configs")
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """SSE fan-out happens from reader threads, so we need the API's loop."""
        self._loop = loop

    # ------------------------------------------------------------------ launch

    def start(self, config: dict, *, sweep_id: str | None = None,
              extra_args: list[str] | None = None) -> Run:
        run_id = uuid.uuid4().hex[:12]
        cfg_path = self.workdir / f"{run_id}.yaml"
        cfg_path.write_text(yaml.safe_dump(config, sort_keys=False))

        run = Run(id=run_id, config=config, sweep_id=sweep_id, config_path=str(cfg_path))
        self.runs[run_id] = run
        self._evict_old_runs()

        cmd = [sys.executable, "-u", str(REPO_ROOT / "train.py"), "--config", str(cfg_path)]
        cmd += extra_args or []

        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        # Reduce thread thrash when several runs share one machine.
        env.setdefault("OMP_NUM_THREADS", "1")

        run.proc = subprocess.Popen(
            cmd, cwd=str(REPO_ROOT), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
            start_new_session=True,  # own process group, so cancel kills dataloader workers too
        )
        run.status = "running"

        run._pumps = [
            threading.Thread(target=self._pump, args=(run, run.proc.stdout, "stdout"),
                             daemon=True),
            threading.Thread(target=self._pump, args=(run, run.proc.stderr, "stderr"),
                             daemon=True),
        ]
        for t in run._pumps:
            t.start()
        threading.Thread(target=self._reap, args=(run,), daemon=True).start()
        return run

    # ------------------------------------------------------------------ streaming

    def _pump(self, run: Run, stream, which: str) -> None:
        for raw in iter(stream.readline, ""):
            line = raw.rstrip("\n")
            if not line:
                continue
            if line.startswith(EVENT_PREFIX):
                try:
                    ev = json.loads(line[len(EVENT_PREFIX):].strip())
                except json.JSONDecodeError:
                    continue
                self._handle_event(run, ev)
                self._broadcast(run, {"type": "event", **ev})
            else:
                run.lines.append(line)
                self._broadcast(run, {"type": "log", "stream": which, "line": line})
        with contextlib.suppress(Exception):
            stream.close()

    def _handle_event(self, run: Run, ev: dict) -> None:
        kind = ev.get("event")
        if kind == "started":
            run.mlflow_run_id = ev.get("run_id")
            run.run_name = ev.get("run_name", "")
            run.output_dir = ev.get("output_dir")
        elif kind == "epoch":
            run.latest = {
                "epoch": ev.get("epoch"), "total": ev.get("total"),
                "metrics": ev.get("metrics", {}), "best": ev.get("best"),
                "primary_metric": ev.get("primary_metric"),
                "higher_is_better": ev.get("higher_is_better", True),
            }
            run.events.append(ev)
            del run.events[:-MAX_EPOCH_EVENTS]
        elif kind == "progress":
            run.latest.update({"batch": ev.get("batch"),
                               "total_batches": ev.get("total_batches"),
                               "running_loss": ev.get("loss"), "lr": ev.get("lr")})
        elif kind == "failed":
            run.error = ev.get("error")

    def _broadcast(self, run: Run, payload: dict) -> None:
        if self._loop is None:
            return
        for q in list(run._subscribers):
            with contextlib.suppress(Exception):
                self._loop.call_soon_threadsafe(self._offer, run, q, payload)

    @staticmethod
    def _offer(run: Run, q: asyncio.Queue, payload: dict) -> None:
        """Enqueue for one subscriber, dropping the oldest log line under back-pressure.

        A slow SSE client used to lose whatever happened to arrive while its queue was
        full — including the terminal status message, which left the UI showing a run as
        "running" forever. Status and event frames now displace buffered log lines
        instead, and dropped lines are counted rather than vanishing silently.
        """
        try:
            q.put_nowait(payload)
            return
        except asyncio.QueueFull:
            pass

        if payload.get("type") == "log":
            run.dropped_events += 1
            return
        # Make room for a frame that carries state, not just text.
        for _ in range(min(64, q.qsize())):
            try:
                if q.get_nowait().get("type") == "log":
                    run.dropped_events += 1
                    break
            except asyncio.QueueEmpty:
                break
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            run.dropped_events += 1

    def _reap(self, run: Run) -> None:
        code = run.proc.wait()
        # The process exiting does not mean its output has been consumed: the last epoch
        # event can still be sitting in the pipe. Draining before publishing the terminal
        # status is what lets a waiter trust `run.latest` and stops SSE clients closing
        # the stream on "finished" before the final metrics arrive.
        for t in run._pumps:
            t.join(timeout=30)

        run.ended_at = time.time()
        if run.status == "cancelling":
            run.status = "cancelled"
        elif code == 0:
            run.status = "finished"
        else:
            run.status = "failed"
            run.error = run.error or f"exited with code {code}"
        self._broadcast(run, {"type": "status", "status": run.status,
                              "error": run.error, "exit_code": code})
        run.done.set()

    def wait(self, run: Run, timeout: float | None = None) -> bool:
        """Block until the run is finished AND its output fully consumed."""
        return run.done.wait(timeout)

    async def subscribe(self, run_id: str) -> asyncio.Queue:
        run = self.runs[run_id]
        q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        # Replay recent history so a late subscriber sees context, not a blank pane.
        for line in list(run.lines)[-200:]:
            q.put_nowait({"type": "log", "stream": "stderr", "line": line})
        for ev in run.events[-50:]:
            q.put_nowait({"type": "event", **ev})
        q.put_nowait({"type": "status", "status": run.status})
        run._subscribers.append(q)
        return q

    def unsubscribe(self, run_id: str, q: asyncio.Queue) -> None:
        run = self.runs.get(run_id)
        if run and q in run._subscribers:
            run._subscribers.remove(q)

    # ------------------------------------------------------------------ control

    def cancel(self, run_id: str) -> bool:
        run = self.runs.get(run_id)
        if not run or not run.proc or run.proc.poll() is not None:
            return False
        run.status = "cancelling"
        try:
            # Signal the whole group: SIGTERM alone leaves dataloader workers orphaned.
            os.killpg(os.getpgid(run.proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            return False

        def _force() -> None:
            time.sleep(10)
            if run.proc.poll() is None:
                with contextlib.suppress(Exception):
                    os.killpg(os.getpgid(run.proc.pid), signal.SIGKILL)

        threading.Thread(target=_force, daemon=True).start()
        return True

    def _evict_old_runs(self) -> None:
        """Forget the oldest finished runs once the in-memory list gets long."""
        if len(self.runs) <= MAX_RETAINED_RUNS:
            return
        done = sorted(
            (r for r in self.runs.values()
             if r.status in ("finished", "failed", "cancelled") and not r._subscribers),
            key=lambda r: r.ended_at or r.started_at,
        )
        for r in done[: len(self.runs) - MAX_RETAINED_RUNS]:
            self.runs.pop(r.id, None)

    def list(self) -> list[dict]:
        return sorted((r.summary() for r in self.runs.values()),
                      key=lambda r: -r["started_at"])

    def get(self, run_id: str) -> Run | None:
        return self.runs.get(run_id)

    def active_count(self) -> int:
        return sum(1 for r in self.runs.values() if r.status in ("starting", "running"))

    def shutdown(self) -> None:
        for run_id, run in self.runs.items():
            if run.proc and run.proc.poll() is None:
                self.cancel(run_id)


manager = RunManager()
