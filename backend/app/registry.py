"""Read side of the tracker: run listing, comparison, diffing, cloning.

Everything the Compare tab needs. Deliberately talks to MLflow through its client rather
than through the training-side adapter, because the read model (many runs, sparse
columns) has nothing in common with the write model (one run, append-only).
"""

from __future__ import annotations

import json
import os
from typing import Any

# Interactive reads must fail fast. MLflow's client defaults to several retries with
# exponential backoff, so a stopped tracker turned every Compare-tab request into a
# multi-minute stall instead of a prompt "tracking server unreachable".
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")

#: Config params that always differ between runs and would otherwise dominate the diff.
#: Dataset *identity* is not lost by hiding the path spellings: `data.class_names`,
#: `data.num_classes` and the fingerprint travel with every run, and the Compare tab is
#: scoped to one experiment ("one experiment per dataset" is the documented convention).
NOISE_PARAMS = {
    "env.git_commit", "env.git_dirty", "env.git_branch",
    "tracking.run_name", "tracking.tags", "checkpoint.output_dir",
    "data.class_names", "data.num_classes", "schema_version",
    "hw.python_executable", "checkpoint.resume_from",
    # Path/URI spellings: `/abs/path` vs `data/...` vs a moved dataset root produced
    # phantom "varying" axes in the parallel-coordinates plot and diff clutter.
    "data.train_dir", "data.val_dir", "tracking.tracking_uri",
}
NOISE_PREFIXES = ("data.fingerprint.", "lib.", "hw.")


def _client(uri: str):
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(uri)
    return MlflowClient(tracking_uri=uri)


def _to_row(r) -> dict:
    info, d = r.info, r.data
    return {
        "run_id": info.run_id,
        "name": d.tags.get("mlflow.runName", info.run_id[:8]),
        "status": d.tags.get("status", info.status),
        "start_time": info.start_time,
        "end_time": info.end_time,
        "duration_s": ((info.end_time or info.start_time) - info.start_time) / 1000,
        "params": dict(d.params),
        "metrics": dict(d.metrics),
        "tags": {k: v for k, v in d.tags.items() if not k.startswith("mlflow.")},
    }


def list_experiments(uri: str) -> list[dict]:
    c = _client(uri)
    return [{"id": e.experiment_id, "name": e.name} for e in c.search_experiments()]


def list_runs(uri: str, experiment: str = "default", limit: int = 500,
              filter_string: str = "") -> list[dict]:
    c = _client(uri)
    exp = c.get_experiment_by_name(experiment)
    if exp is None:
        return []
    runs = c.search_runs([exp.experiment_id], filter_string=filter_string,
                         max_results=limit, order_by=["attributes.start_time DESC"])
    return [_to_row(r) for r in runs]


class TrackerUnreachable(RuntimeError):
    """The tracking server could not be contacted (as opposed to: no such run)."""


def _check_reachable(uri: str) -> None:
    """Raise TrackerUnreachable if the server is down.

    Callers that swallow per-run errors need this to tell "that run does not exist" from
    "nothing is listening", which otherwise both surfaced as a bland "run not found".
    """
    try:
        _client(uri).search_experiments(max_results=1)
    except Exception as e:
        raise TrackerUnreachable(str(e)) from e


def get_run(uri: str, run_id: str) -> dict | None:
    c = _client(uri)
    try:
        return _to_row(c.get_run(run_id))
    except Exception:
        _check_reachable(uri)     # re-raises if the server itself is gone
        return None


def metric_history(uri: str, run_id: str, key: str) -> list[dict]:
    c = _client(uri)
    safe = key.replace("@", "_at_").replace(" ", "_")
    try:
        return [{"step": m.step, "value": m.value, "timestamp": m.timestamp}
                for m in c.get_metric_history(run_id, safe)]
    except Exception:
        _check_reachable(uri)     # an empty chart and a dead tracker are different things
        return []


def _is_noise(key: str) -> bool:
    return key in NOISE_PARAMS or key.startswith(NOISE_PREFIXES)


def varying_params(runs: list[dict], *, include_noise: bool = False) -> list[str]:
    """Params that are not identical across every run.

    This is what makes a 100-run table readable: constant columns carry no information
    for comparison, so they are hidden by default.
    """
    seen: dict[str, set[str]] = {}
    for r in runs:
        for k, v in r["params"].items():
            if not include_noise and _is_noise(k):
                continue
            seen.setdefault(k, set()).add(v)
    return sorted(k for k, vals in seen.items() if len(vals) > 1)


def common_metrics(runs: list[dict]) -> list[str]:
    keys: dict[str, int] = {}
    for r in runs:
        for k in r["metrics"]:
            keys[k] = keys.get(k, 0) + 1
    # Headline metrics first, then everything else alphabetically.
    priority = ["acc_at_1", "acc_at_5", "macro-F1", "balanced-accuracy", "val_loss",
                "train_loss", "auroc", "mcc", "ece"]
    ordered = [k for k in priority if k in keys]
    ordered += sorted(k for k in keys if k not in ordered)
    return ordered


def _num(v: str) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parallel_coordinates(runs: list[dict], metric: str,
                         params: list[str] | None = None) -> dict:
    """Axis definitions plus one line per run, ready to plot.

    Numeric axes carry their range; categorical axes carry an ordinal encoding so the
    frontend can draw them without re-deriving the mapping.
    """
    params = params or varying_params(runs)
    axes: list[dict] = []

    for p in params:
        raw = [r["params"].get(p) for r in runs]
        nums = [_num(v) for v in raw]
        if all(n is not None for n in nums) and len(set(nums)) > 1:
            lo, hi = min(nums), max(nums)
            log = lo > 0 and hi / max(lo, 1e-12) >= 100
            axes.append({"key": p, "type": "number", "min": lo, "max": hi, "log": log})
        else:
            cats = sorted({str(v) for v in raw if v is not None})
            axes.append({"key": p, "type": "category", "categories": cats})

    mvals = [r["metrics"].get(metric) for r in runs]
    valid = [m for m in mvals if m is not None]
    axes.append({"key": metric, "type": "metric",
                 "min": min(valid) if valid else 0, "max": max(valid) if valid else 1})

    lines = []
    for r in runs:
        if r["metrics"].get(metric) is None:
            continue
        lines.append({
            "run_id": r["run_id"], "name": r["name"],
            "metric": r["metrics"][metric],
            "values": {p: r["params"].get(p) for p in params},
        })
    return {"axes": axes, "lines": lines, "metric": metric}


def diff(uri: str, run_a: str, run_b: str, *, include_noise: bool = False) -> dict:
    """Only the parameters that differ, plus the metric deltas.

    Reports the number of differing parameters explicitly: with more than one, the
    comparison is confounded and the UI says so rather than implying causality.
    """
    a, b = get_run(uri, run_a), get_run(uri, run_b)
    if a is None or b is None:
        return {"error": "run not found"}

    keys = set(a["params"]) | set(b["params"])
    param_diff = []
    identical = 0
    hidden_noise = 0
    for k in sorted(keys):
        va, vb = a["params"].get(k), b["params"].get(k)
        noisy = not include_noise and _is_noise(k)
        if noisy:
            # Counted separately: folding equal noise params into `identical` inflated
            # the "N identical parameters hidden" line, and noise params that DIFFERED
            # were dropped from both counts, so the UI reported a total that matched
            # neither what was shown nor what existed.
            hidden_noise += 1
            continue
        if va == vb:
            identical += 1
            continue
        param_diff.append({"key": k, "a": va, "b": vb})

    metric_diff = []
    for k in sorted(set(a["metrics"]) | set(b["metrics"])):
        ma, mb = a["metrics"].get(k), b["metrics"].get(k)
        metric_diff.append({
            "key": k, "a": ma, "b": mb,
            "delta": (mb - ma) if (ma is not None and mb is not None) else None,
        })

    return {
        "a": {"run_id": a["run_id"], "name": a["name"], "tags": a["tags"]},
        "b": {"run_id": b["run_id"], "name": b["name"], "tags": b["tags"]},
        "params": param_diff,
        "metrics": metric_diff,
        "identical_params": identical,
        "hidden_noise_params": hidden_noise,
        "confounded": len(param_diff) > 1,
    }


def config_from_run(uri: str, run_id: str) -> dict | None:
    """Rebuild a TrainConfig dict from a logged run, for the Clone button.

    Prefers the exact `config.yaml` artifact; falls back to un-flattening the params
    table when the artifact store is unreachable.
    """
    import yaml
    from mlflow.artifacts import download_artifacts

    try:
        path = download_artifacts(run_id=run_id, artifact_path="config.yaml",
                                  tracking_uri=uri)
        return yaml.safe_load(open(path))
    except Exception:
        pass

    r = get_run(uri, run_id)
    if r is None:
        return None
    nested: dict[str, Any] = {}
    for k, v in r["params"].items():
        if _is_noise(k) or k.startswith(("env.", "lib.", "hw.")):
            continue
        node = nested
        parts = k.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce_param(v)
    return nested or None


def _coerce_param(v: str):
    if v in ("True", "true"):
        return True
    if v in ("False", "false"):
        return False
    if v in ("None", "", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    if "," in v:
        return [_coerce_param(x) for x in v.split(",")]
    return v
