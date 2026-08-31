"""FastAPI app.

The form is generated from `GET /api/schema`, so the Pydantic model in
`trainlab/config.py` is the only place a hyperparameter is ever declared.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
from pathlib import Path, PurePosixPath

import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from trainlab import naming, presets as presets_mod  # noqa: E402
from trainlab.config import (  # noqa: E402
    AUG_PRESETS, DEFAULT_EXPANDED, Group, Preset, TrainConfig, fpr_at_fnr_key,
    load_config_leniently, validate_ui_expressions,
)
from trainlab.optim import suggest_lr  # noqa: E402

from . import catalog, preview, registry  # noqa: E402
from .runner import manager  # noqa: E402
from .sweeps import SweepConfig, SweepManager  # noqa: E402

@contextlib.asynccontextmanager
async def lifespan(_: FastAPI):
    # Reader threads fan out to SSE subscribers, so they need the serving loop.
    manager.bind_loop(asyncio.get_running_loop())
    # showIf/disableIf are hand-written strings resolved against field names at render
    # time; a stale one silently removes a control from the form. Surface it at startup
    # instead of leaving it to be noticed by whoever misses the missing widget.
    for problem in validate_ui_expressions():
        print(f"[trainlab] schema expression problem: {problem}", file=sys.stderr)
    yield
    manager.shutdown()


app = FastAPI(title="TrainLab API", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"], allow_headers=["*"],
)
# The server binds loopback, but loopback alone does not stop DNS rebinding: a hostile
# page can point its own hostname at 127.0.0.1 and reach an API that spawns
# subprocesses and reads caller-supplied filesystem paths. CORS doesn't help there —
# the rebound request is same-origin from the browser's point of view. Rejecting
# unexpected Host headers closes it. ("testserver" is Starlette's TestClient host.)
from fastapi.middleware.trustedhost import TrustedHostMiddleware  # noqa: E402

#: Hosts the API answers to regardless of where it is bound.
DEFAULT_ALLOWED_HOSTS = ("127.0.0.1", "localhost", "testserver")


def allowed_hosts(extra: str | None) -> list[str]:
    """The Host headers to accept, given `TRAINLAB_ALLOWED_HOSTS`.

    Serving on another interface (`make api HOST=192.168.1.5`) means browsers send that
    name in the Host header, and the guard above would reject every request with a 400
    that says nothing about why. Naming the extra hosts keeps the check on instead of
    turning it off — each one is a deliberate answer to "who may reach this API".
    """
    named = [h.strip() for h in (extra or "").split(",") if h.strip()]
    return list(DEFAULT_ALLOWED_HOSTS) + [h for h in named
                                          if h not in DEFAULT_ALLOWED_HOSTS]


app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=allowed_hosts(os.environ.get("TRAINLAB_ALLOWED_HOSTS")),
)
sweep_manager = SweepManager(manager)


# --------------------------------------------------------------------------------------
# Schema, defaults, presets
# --------------------------------------------------------------------------------------

GROUP_META = {
    Group.DATA: ("Data", "Where the images are and how they're loaded."),
    Group.INPUT: ("Input & preprocessing", "Resolution, cropping, normalization."),
    Group.AUGMENTATION: ("Augmentation", "What the model sees during training."),
    Group.MODEL: ("Model / backbone", "Architecture, pretrained weights, regularization."),
    Group.LOSS: ("Loss", "Objective and class weighting."),
    Group.OPTIMIZATION: ("Optimization", "Optimizer, learning rate, weight decay."),
    Group.SCHEDULE: ("Schedule & runtime", "Epochs, batch size, precision, device."),
    Group.VALIDATION: ("Validation & metrics", "What gets measured and how."),
    Group.CHECKPOINT: ("Checkpointing & early stopping", "What gets saved and when to stop."),
    Group.TRACKING: ("Experiment tracking", "Where runs are recorded."),
    Group.EXPERIMENTAL: ("Advanced / experimental", "Higher-variance techniques."),
}


@app.get("/api/schema")
def get_schema() -> dict:
    """JSON Schema plus the x-ui metadata that drives the form layout."""
    return {
        "schema": TrainConfig.model_json_schema(),
        "defaults": TrainConfig().model_dump(mode="json"),
        "groups": [
            {"key": g.value, "title": GROUP_META[g][0], "description": GROUP_META[g][1],
             "expandedByDefault": g in DEFAULT_EXPANDED}
            for g in Group
        ],
    }


@app.get("/api/presets")
def get_presets() -> dict:
    out = []
    for p in Preset:
        if p == Preset.CUSTOM:
            continue
        out.append({
            "key": p.value,
            "label": p.value.replace("-", " ").title(),
            "description": presets_mod.describe(p),
            "config": presets_mod.apply_preset(p).model_dump(mode="json"),
        })
    # The augmentation group's own strength ladder. The form applies these client-side so
    # the sliders move the moment you pick a rung; `AugmentationConfig` applies the same
    # table server-side, so the two cannot drift.
    aug = [{"key": p.value, "values": values} for p, values in AUG_PRESETS.items()]
    return {"presets": out, "aug_presets": aug}


class ConfigBody(BaseModel):
    config: dict


class ValidateBody(ConfigBody):
    n_train: int | None = None
    imbalance_ratio: float | None = None


@app.post("/api/validate")
def validate(body: ValidateBody) -> dict:
    """Authoritative validation. The client mirrors this for instant feedback."""
    try:
        cfg = TrainConfig.model_validate(body.config)
    except ValidationError as e:
        return {
            "valid": False,
            "errors": [{"field": ".".join(str(x) for x in err["loc"]),
                        "message": err["msg"]} for err in e.errors()],
            "warnings": [],
        }

    device = _resolved_device()
    warns = cfg.warnings(n_train=body.n_train, imbalance_ratio=body.imbalance_ratio,
                         resolved_device=device)
    return {
        "valid": True,
        "errors": [],
        "warnings": [w.model_dump(mode="json") for w in warns],
        "resolved": {
            "loss": cfg.effective_loss.value,
            "loss_autoswitched": cfg.loss_was_autoswitched,
            "test_input_size": cfg.effective_test_input_size,
            "effective_batch_size": cfg.effective_batch_size,
            "device": device,
        },
        "run_name": naming.run_name(cfg),
        "tags": naming.run_tags(cfg),
    }


def _resolved_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


@app.post("/api/config/yaml", response_class=PlainTextResponse)
def to_yaml(body: ConfigBody) -> str:
    cfg = TrainConfig.model_validate(body.config)
    return yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False)


class YamlBody(BaseModel):
    yaml: str


@app.post("/api/config/parse")
def from_yaml(body: YamlBody) -> dict:
    try:
        raw = yaml.safe_load(body.yaml)
    except yaml.YAMLError as e:
        raise HTTPException(400, f"{type(e).__name__}: {e}")
    # `yaml.safe_load` returns None for an empty document or a bare `null`. Coercing
    # that to `{}` silently handed back a full set of defaults, so importing a truncated
    # or empty file looked like it had loaded successfully.
    if raw is None:
        raise HTTPException(400, "the YAML document is empty")
    if not isinstance(raw, dict):
        raise HTTPException(
            400, f"expected a YAML mapping of config sections, got {type(raw).__name__}")
    try:
        cfg, dropped = load_config_leniently(raw)
    except ValidationError as e:
        raise HTTPException(400, f"{type(e).__name__}: {e}")
    return {"config": cfg.model_dump(mode="json"), "dropped_fields": dropped}


@app.get("/api/lr-suggestion")
def lr_suggestion(optimizer: str, batch_size: int, rule: str = "sqrt",
                  current_lr: float | None = None) -> dict:
    """Advisory only. Never auto-applied to a value the user typed."""
    base = {"adamw": 3e-4, "sgd": 0.05, "lion": 3e-5, "rmsprop": 1e-4}.get(optimizer, 3e-4)
    suggested = suggest_lr(base, 64, batch_size, rule)
    return {
        "suggested_lr": suggested,
        "rule": rule,
        "explanation": (
            f"{optimizer} base LR {base:g} at batch 64, scaled by "
            f"{'batch/64' if rule == 'linear' else 'sqrt(batch/64)'} = {suggested:.2e}. "
            f"Linear scaling was derived for SGD; sqrt fits adaptive optimizers better."
        ),
        "current_lr": current_lr,
    }


# --------------------------------------------------------------------------------------
# Backbones
# --------------------------------------------------------------------------------------

@app.get("/api/backbones")
def backbones(q: str = "", pretrained_only: bool = False,
              family: str | None = None, limit: int = Query(60, le=300)) -> dict:
    return catalog.search(q, only_pretrained=pretrained_only, family=family, limit=limit)


@app.get("/api/backbones/families")
def backbone_families() -> dict:
    return {"families": catalog.families()}


@app.get("/api/backbones/{name}")
def backbone_detail(name: str) -> dict:
    return catalog.detail(name)


@app.post("/api/estimate")
def estimate(body: ConfigBody) -> dict:
    cfg = TrainConfig.model_validate(body.config)
    from trainlab.device import resolve_runtime

    rt = resolve_runtime(cfg)
    mem = catalog.estimate_memory_gb(
        cfg.model.backbone, cfg.schedule.batch_size, cfg.input.input_size,
        amp=rt.amp_enabled, grad_checkpointing=cfg.model.gradient_checkpointing)
    return {
        "memory": mem,
        "device_memory_gb": rt.total_memory_gb,
        "device": rt.device_str,
        "device_name": rt.device_name,
        "downgrades": rt.downgrades,
        "risk": ("high" if rt.total_memory_gb and mem["total_gb"] > 0.9 * rt.total_memory_gb
                 else "medium" if rt.total_memory_gb and mem["total_gb"] > 0.7 * rt.total_memory_gb
                 else "low"),
    }


# --------------------------------------------------------------------------------------
# Dataset + preview
# --------------------------------------------------------------------------------------

@app.post("/api/dataset/inspect")
def dataset_inspect(body: ConfigBody) -> dict:
    return preview.inspect_dataset(TrainConfig.model_validate(body.config))


class PreviewBody(ConfigBody):
    n: int = 8
    seed: int = 0
    include_original: bool = True


@app.post("/api/preview/augmentation")
def preview_augmentation(body: PreviewBody) -> dict:
    cfg = TrainConfig.model_validate(body.config)
    return preview.augmentation_preview(cfg, n=body.n, seed=body.seed,
                                        include_original=body.include_original)


@app.post("/api/preview/eval")
def preview_eval(body: PreviewBody) -> dict:
    return preview.eval_preview(TrainConfig.model_validate(body.config),
                                n=body.n, seed=body.seed)


# --------------------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------------------

@app.post("/api/runs")
def start_run(body: ConfigBody) -> dict:
    cfg = TrainConfig.model_validate(body.config)   # fail fast on a bad config
    run = manager.start(cfg.model_dump(mode="json"))
    return run.summary()


@app.get("/api/runs")
def list_runs() -> dict:
    return {"runs": manager.list(), "active": manager.active_count()}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    run = manager.get(run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    return {**run.summary(), "log_tail": list(run.lines)[-400:], "events": run.events}


@app.post("/api/runs/{run_id}/cancel")
def cancel_run(run_id: str) -> dict:
    return {"cancelled": manager.cancel(run_id)}


@app.get("/api/runs/{run_id}/stream")
async def stream_run(run_id: str):
    if manager.get(run_id) is None:
        raise HTTPException(404, "run not found")

    async def gen():
        q = await manager.subscribe(run_id)
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=20)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"       # keep proxies from closing the stream
                    continue
                yield f"data: {json.dumps(msg)}\n\n"
                if msg.get("type") == "status" and msg["status"] in (
                        "finished", "failed", "cancelled"):
                    break
        finally:
            manager.unsubscribe(run_id, q)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# --------------------------------------------------------------------------------------
# Tracker (comparison view)
# --------------------------------------------------------------------------------------

#: Taken from the schema rather than restated, so `TRAINLAB_TRACKING_URI` moves the whole
#: app at once. Requests carry the configured URI (the Compare tab sends the value in the
#: form), and this is what answers when one does not.
DEFAULT_TRACKING_URI = TrainConfig().tracking.tracking_uri


def _uri(tracking_uri: str | None) -> str:
    return tracking_uri or DEFAULT_TRACKING_URI


@app.get("/api/tracker/experiments")
def experiments(tracking_uri: str | None = None) -> dict:
    try:
        return {"experiments": registry.list_experiments(_uri(tracking_uri))}
    except Exception as e:
        raise HTTPException(503, f"tracking server unreachable: {e}")


@app.get("/api/tracker/runs")
def tracker_runs(experiment: str = "default", limit: int = 500,
                 tracking_uri: str | None = None, filter: str = "") -> dict:
    try:
        runs = registry.list_runs(_uri(tracking_uri), experiment, limit, filter)
    except Exception as e:
        raise HTTPException(503, f"tracking server unreachable: {e}")
    return {
        "runs": runs,
        "varying_params": registry.varying_params(runs),
        "metrics": registry.common_metrics(runs),
    }


@app.get("/api/tracker/parallel")
def tracker_parallel(experiment: str = "default", metric: str = "acc_at_1",
                     tracking_uri: str | None = None,
                     params: str | None = None) -> dict:
    # Same 503 treatment as the other tracker reads. Without it, a dead tracker left
    # MLflow's client retrying inside the request and the endpoint never returned —
    # the Compare tab span forever with no error to show.
    try:
        runs = registry.list_runs(_uri(tracking_uri), experiment)
    except Exception as e:
        raise HTTPException(503, f"tracking server unreachable: {e}")
    selected = [p for p in (params or "").split(",") if p] or None
    return registry.parallel_coordinates(runs, metric, selected)


@app.get("/api/tracker/diff")
def tracker_diff(a: str, b: str, tracking_uri: str | None = None,
                 include_noise: bool = False) -> dict:
    try:
        return registry.diff(_uri(tracking_uri), a, b, include_noise=include_noise)
    except Exception as e:
        raise HTTPException(503, f"tracking server unreachable: {e}")


@app.get("/api/tracker/history")
def tracker_history(run_id: str, key: str, tracking_uri: str | None = None) -> dict:
    try:
        return {"history": registry.metric_history(_uri(tracking_uri), run_id, key)}
    except Exception as e:
        raise HTTPException(503, f"tracking server unreachable: {e}")


@app.get("/api/tracker/clone")
def tracker_clone(run_id: str, tracking_uri: str | None = None) -> dict:
    """Load a past run's config back into the form.

    Strictness is right for a config a user just typed, but wrong for replaying recorded
    history: `extra="forbid"` made every run predating any rename permanently
    un-clonable. Unknown keys are dropped here and reported, so the rest of the config
    still loads and the user can see exactly what was discarded.
    """
    raw = registry.config_from_run(_uri(tracking_uri), run_id)
    if raw is None:
        raise HTTPException(404, "could not recover a config for that run")
    try:
        cfg, dropped = load_config_leniently(raw)
    except ValidationError as e:
        raise HTTPException(422, f"stored config no longer matches the schema: {e}")
    return {"config": cfg.model_dump(mode="json"), "cloned_from": run_id,
            "dropped_fields": dropped}


@app.get("/api/tracker/artifact")
def tracker_artifact(run_id: str, path: str, tracking_uri: str | None = None):
    from mlflow.artifacts import download_artifacts

    # `path` comes straight off the query string. Without this, `..` segments walk out
    # of the run's artifact directory and FileResponse happily serves whatever they
    # reach. Artifact paths are always relative and never traverse upward.
    clean = PurePosixPath(path)
    if clean.is_absolute() or ".." in clean.parts:
        raise HTTPException(400, "artifact path must be relative and must not contain '..'")

    try:
        local = download_artifacts(run_id=run_id, artifact_path=str(clean),
                                   tracking_uri=_uri(tracking_uri))
    except Exception as e:
        raise HTTPException(404, f"artifact not found: {e}")
    return FileResponse(local)


# --------------------------------------------------------------------------------------
# Sweeps
# --------------------------------------------------------------------------------------

@app.post("/api/sweeps")
def start_sweep(cfg: SweepConfig) -> dict:
    base = TrainConfig.model_validate(cfg.base_config)
    if not cfg.parameters:
        raise HTTPException(400, "a sweep needs at least one search space")

    # Fail here rather than after the first trial has burned a full training run: the
    # objective can only read a metric the run actually computes.
    computed = {m.value for m in base.validation.metrics} | {
        base.validation.primary_metric.value, base.primary_metric_key,
        "val_loss", "train_loss"}
    # `fpr@fnr` is a family: the run logs one key per target, and none of them is the
    # bare enum value a sweep would otherwise be told to optimise.
    if base.fpr_at_fnr_active:
        computed |= {fpr_at_fnr_key(t) for t in base.validation.fpr_at_fnr_targets}
    if cfg.metric_key not in computed:
        raise HTTPException(
            400,
            f"sweep metric '{cfg.metric}' is not computed by the base config. "
            f"Add it to validation.metrics (currently: {sorted(computed)}).",
        )
    return sweep_manager.start(cfg).summary()


@app.get("/api/sweeps")
def list_sweeps() -> dict:
    return {"sweeps": sweep_manager.list()}


@app.get("/api/sweeps/{sweep_id}")
def get_sweep(sweep_id: str) -> dict:
    s = sweep_manager.get(sweep_id)
    if s is None:
        raise HTTPException(404, "sweep not found")
    return {**s.summary(), "trials": s.trials}


@app.post("/api/sweeps/{sweep_id}/stop")
def stop_sweep(sweep_id: str) -> dict:
    return {"stopped": sweep_manager.stop(sweep_id)}


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "device": _resolved_device(), "active_runs": manager.active_count()}


# --------------------------------------------------------------------------------------
# Static frontend (served in production; Vite dev server is used during development)
# --------------------------------------------------------------------------------------

_DIST = REPO_ROOT / "frontend" / "dist"
if _DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="ui")
