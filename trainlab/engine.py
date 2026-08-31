"""The training loop."""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import artifacts, data, ema as ema_mod, losses, metrics as metrics_mod
from . import models, naming, optim as optim_mod, progress, sched, transforms
from .config import FreezePolicy, TTA, TrainConfig
from .device import ResolvedRuntime, peak_memory_gb, reset_peak_memory

#: Prefix for machine-readable progress lines the API parses off stdout.
EVENT_PREFIX = "@@TRAINLAB@@"


def emit(kind: str, **payload) -> None:
    """Structured progress event. Human-readable text goes to stderr instead."""
    print(f"{EVENT_PREFIX} {json.dumps({'event': kind, **payload})}", flush=True)


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _load_checkpoint(path, map_location="cpu") -> dict:
    """Load a checkpoint, preferring torch's safe unpickler.

    `weights_only=False` hands arbitrary pickle opcodes to the interpreter, i.e. loading
    someone else's checkpoint runs their code. Everything this app writes (tensors, a
    config dict, a list of class names) deserialises fine under `weights_only=True`, so
    that is the default path. The permissive fallback stays available for checkpoints
    written by other tools, but it is opt-in per process and says so loudly, rather than
    being the silent default it was.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as e:
        if os.environ.get("TRAINLAB_ALLOW_UNSAFE_CHECKPOINTS") != "1":
            raise RuntimeError(
                f"Could not load '{path}' with PyTorch's safe unpickler ({type(e).__name__}: "
                f"{e}). This usually means the checkpoint was written by another tool and "
                f"contains pickled Python objects. Loading it executes arbitrary code from "
                f"whoever produced the file. If you trust it, re-run with "
                f"TRAINLAB_ALLOW_UNSAFE_CHECKPOINTS=1."
            ) from e
        log(f"  ! loading {path} with weights_only=False — this executes code embedded "
            f"in the checkpoint (TRAINLAB_ALLOW_UNSAFE_CHECKPOINTS=1 is set)")
        return torch.load(path, map_location=map_location, weights_only=False)


def _unwrap_state(state: dict) -> dict:
    """Strip the wrapper prefixes training harnesses add to state-dict keys.

    `torch.compile` wraps the model in an OptimizedModule whose state dict prefixes
    every key with `_orig_mod.`; DDP does the same with `module.`. The tensors are the
    plain model's either way, but by name they match nothing — a compiled run's
    checkpoint loaded into an uncompiled model matches 0 of 782 tensors while naming the
    same backbone. Prefixes are stripped repeatedly, so a DDP-of-compiled dict unwraps
    too.
    """
    changed = True
    while changed and state:
        changed = False
        for prefix in ("_orig_mod.", "module."):
            if all(k.startswith(prefix) for k in state):
                state = {k[len(prefix):]: v for k, v in state.items()}
                changed = True
    return state


def _unwrap_model(model):
    """The plain module inside a torch.compile wrapper, or the model itself."""
    return getattr(model, "_orig_mod", model)


@dataclass
class RunState:
    #: Overwritten in Trainer.__init__ with +inf when the primary metric is
    #: lower-is-better, so the first evaluation always counts as an improvement.
    best_metric: float = -float("inf")
    best_epoch: int = -1
    epochs_without_improvement: int = 0
    global_step: int = 0
    history: list[dict] = field(default_factory=list)
    checkpoints: list[tuple[float, Path]] = field(default_factory=list)


class Trainer:
    def __init__(self, cfg: TrainConfig, rt: ResolvedRuntime, tracker, out_dir: Path):
        self.cfg, self.rt, self.tracker, self.out_dir = cfg, rt, tracker, out_dir
        self.state = RunState()
        # Direction of the primary metric decides checkpoint selection, early stopping
        # and top-K retention. Seeding `best_metric` at the losing end of the scale is
        # what makes the first evaluation count as an improvement either way.
        self.maximize = cfg.validation.primary_metric.higher_is_better
        self.state.best_metric = -float("inf") if self.maximize else float("inf")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # Populated only when the corresponding technique is enabled.
        self.ssl_result = None
        self._sample_scores: np.ndarray | None = None
        self._dropped_indices: set[int] = set()
        self._active_subset: list[int] | None = None
        self._ensemble: list[torch.nn.Module] = []
        self._best_is_ema = False
        self._input_bound_warned = False
        #: First epoch `fit()` runs. Moved past 0 by `checkpoint.resume_from`.
        self.start_epoch = 0
        self.progressive_enabled = cfg.schedule.progressive_resizing

    # ---------------------------------------------------------------- setup

    def setup(self) -> None:
        cfg, rt = self.cfg, self.rt

        # Cheap architectural preconditions first. An unsupported SSL backbone used to
        # surface only at model construction, after the dataset had been scanned and the
        # tracker run opened — minutes into a run the schema had called valid.
        if cfg.ssl_active:
            from . import ssl as ssl_mod

            ssl_mod.check_backbone(cfg)

        torch.manual_seed(cfg.schedule.seed)
        np.random.seed(cfg.schedule.seed)
        if cfg.schedule.deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)

        self.mean, self.std = transforms.resolve_norm(cfg)
        train_tf = transforms.build_train_transform(cfg)
        val_tf = transforms.build_eval_transform(cfg, size=cfg.effective_test_input_size)
        # Unreadable images are recorded here by every worker; `_report_broken_images`
        # turns the file into a summary, a run artifact and, past the tolerance, a stop.
        self.broken_log = self.out_dir / "broken_images.tsv"
        self.broken_log.unlink(missing_ok=True)
        self._broken_seen = 0
        self.train_loader, self.val_loader, self.train_m, self.val_m = data.build_loaders(
            cfg, train_tf, val_tf, broken_log=self.broken_log)

        self.num_classes = len(self.train_m.class_names)
        self.class_names = self.train_m.class_names
        counts = np.array([self.train_m.counts.get(i, 0) for i in range(self.num_classes)])

        # Refuse to optimise a metric that will never be computed, rather than scoring
        # every epoch as -inf and reporting a successful run with no best checkpoint.
        blocked = metrics_mod.unavailable_metrics(cfg, self.num_classes)
        primary = cfg.validation.primary_metric.value
        if primary in blocked:
            raise ValueError(
                f"primary_metric='{primary}' cannot be computed: {blocked[primary]}. "
                f"Nothing would ever register as an improvement, so no best checkpoint "
                f"would be saved. Choose a different validation.primary_metric."
            )
        for name, why in blocked.items():
            log(f"  ! metric '{name}' not computed: {why}")

        log(f"  train {len(self.train_m):,} · val {len(self.val_m):,} · "
            f"{self.num_classes} classes · imbalance {self.train_m.imbalance_ratio:.1f}:1")

        self.model = models.create_model(
            cfg, self.num_classes,
            pretrained_override=False if cfg.model.pretrained_checkpoint else None)

        # Patch/window models bake the input resolution into patch_embed/pos_embed at
        # construction, so feeding them the ramped sizes progressive resizing produces
        # is a mid-run shape error. Downgrade loudly rather than crash at epoch 1.
        if self.progressive_enabled and hasattr(self.model, "patch_embed"):
            self.progressive_enabled = False
            note = (f"progressive resizing disabled: {cfg.model.backbone} bakes its "
                    f"input resolution into patch_embed at construction")
            log(f"  ! {note}")
            self.tracker.set_tags({"progressive_resizing_note": note[:250]})

        if cfg.model.pretrained_checkpoint:
            self._init_from_checkpoint(cfg.model.pretrained_checkpoint)

        # In-domain SSL runs BEFORE the optimizer exists, so the supervised stage starts
        # from the pretrained encoder rather than re-initialising over it.
        if cfg.ssl_active:
            from . import ssl as ssl_mod

            state, self.ssl_result = ssl_mod.pretrain(
                cfg, rt, self.train_m, self.out_dir, tracker=self.tracker, log=log, emit=emit)
            ssl_mod.load_encoder_into(self.model, state, log=log)
            self.tracker.set_tags({
                "ssl_method": self.ssl_result.method,
                "ssl_epochs": str(self.ssl_result.epochs),
                "ssl_images": str(self.ssl_result.num_images),
            })

        self.model = self.model.to(rt.device)
        if rt.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        if cfg.model.freeze_bn:
            models.freeze_bn(self.model)
        if rt.torch_compile:
            log("  compiling model (first step will be slow)...")
            self.model = torch.compile(self.model)

        total, trainable = models.count_params(self.model)
        log(f"  {cfg.model.backbone}: {total/1e6:.1f}M params ({trainable/1e6:.1f}M trainable)")

        self.criterion = losses.build_loss(cfg, counts, rt.device)
        self.mixup_fn = losses.build_mixup(cfg, self.num_classes)
        self.optimizer = optim_mod.build_optimizer(self.model, cfg)

        self.steps_per_epoch = max(1, len(self.train_loader) // cfg.optimization.grad_accum_steps)
        self.total_steps = self.steps_per_epoch * cfg.schedule.epochs
        self.schedule = sched.LRSchedule(cfg, self.steps_per_epoch)

        self.sam = None
        if cfg.optimization.sam:
            if cfg.optimization.grad_accum_steps > 1:
                log("  ! SAM disabled: not supported with gradient accumulation")
            else:
                self.sam = optim_mod.SAM(self.optimizer, cfg.optimization.sam_rho,
                                         cfg.optimization.sam_adaptive)

        self.ema = None
        if cfg.model.ema:
            decay, note = ema_mod.resolve_decay(cfg.model.ema_decay, self.total_steps)
            if note:
                log(f"  ! {note}")
                self.tracker.set_tags({"ema_note": note[:250]})
            self.ema = ema_mod.ModelEMA(self.model, decay, rt.device, warmup_steps=100)
            self.resolved_ema_decay = decay

        self.swa = None
        if cfg.experimental.swa:
            self.swa = ema_mod.SWA(self.model, rt.device)

        # The device argument is load-bearing: `GradScaler(enabled=True)` defaults to
        # CUDA and silently disables itself (scale stays 1.0) on any other backend, so
        # fp16 would run unscaled while appearing to be protected.
        self.scaler = torch.amp.GradScaler(device=rt.device_str, enabled=rt.use_grad_scaler)
        self.teacher = self._build_teacher()

        # Last, so that everything it writes into (model, EMA, optimizer, scaler,
        # schedule) already exists in its final shape.
        if cfg.checkpoint.resume_from:
            self._restore(cfg.checkpoint.resume_from)

    def _build_teacher(self):
        e = self.cfg.experimental
        if not e.kd_enabled or not e.kd_teacher_ckpt:
            return None
        ck = _load_checkpoint(e.kd_teacher_ckpt, map_location="cpu")

        # Build the teacher from the architecture it was TRAINED with, not the student's.
        # Using the student's config silently restricted distillation to same-backbone
        # pairs, which is the one case where distillation is least interesting.
        teacher_cfg = self.cfg
        if isinstance(ck, dict) and ck.get("config"):
            try:
                teacher_cfg = TrainConfig.model_validate(ck["config"])
            except Exception as exc:
                log(f"  ! teacher config unreadable ({type(exc).__name__}); "
                    f"assuming the same architecture as the student")

        names = ck.get("class_names") if isinstance(ck, dict) else None
        if names and list(names) != list(self.class_names):
            raise ValueError(
                f"KD teacher '{e.kd_teacher_ckpt}' was trained on {len(names)} different "
                f"classes ({list(names)[:4]}...). Distilling across mismatched class "
                f"vocabularies aligns the wrong logits and silently teaches nonsense."
            )

        t = models.create_model(teacher_cfg, self.num_classes).to(self.rt.device)
        which = ck.get("best_weights") if isinstance(ck, dict) else None
        sd = (ck.get(which) if which and isinstance(ck, dict) else None)
        if sd is None:
            sd = ck.get("model", ck) if isinstance(ck, dict) else ck
        t.load_state_dict(_unwrap_state(sd))
        t.eval().requires_grad_(False)
        log(f"  distilling from {teacher_cfg.model.backbone} @ {e.kd_teacher_ckpt}")
        return t

    def _improves(self, score: float, reference: float) -> bool:
        """Is `score` better than `reference` by at least the early-stopping delta?"""
        delta = self.cfg.checkpoint.es_min_delta
        return score > reference + delta if self.maximize else score < reference - delta

    def _worst_score(self) -> float:
        return -float("inf") if self.maximize else float("inf")

    def in_swa_phase(self, epoch: int) -> bool:
        E = self.cfg.experimental
        return E.swa and epoch >= E.swa_start_epoch * self.cfg.schedule.epochs

    def _autocast(self):
        if not self.rt.amp_enabled:
            return nullcontext()
        return torch.autocast(device_type=self.rt.device_str, dtype=self.rt.amp_dtype)

    # ---------------------------------------------------------------- train

    def train_epoch(self, epoch: int) -> dict:
        cfg, rt = self.cfg, self.rt
        self.model.train()
        if cfg.model.freeze_bn:
            models.freeze_bn(self.model)

        if models.apply_freeze(self.model, cfg, epoch):
            frozen = cfg.model.freeze_policy == FreezePolicy.BACKBONE_FIRST_N and epoch < cfg.model.freeze_epochs
            log(f"  backbone {'frozen' if frozen else 'unfrozen'} at epoch {epoch}")

        if self.progressive_enabled:
            size = transforms.progressive_size(cfg, epoch)
            if size != getattr(self, "_cur_size", None):
                self._cur_size = size
                self.train_loader.dataset.transform = transforms.build_train_transform(cfg, size=size)
                # Mutating the transform is not enough: with persistent workers the
                # dataset was pickled into the worker processes once, and they keep
                # serving the OLD size while the log claims otherwise. Rebuilding the
                # loader respawns the workers with the new transform.
                self.train_loader = data.make_train_loader(
                    cfg, self.train_loader.dataset, self.train_m, active=self._active_subset)
                log(f"  progressive resize -> {size}px")

        use_mixup = self.mixup_fn is not None and losses.mixup_enabled_this_epoch(cfg, epoch)
        accum = cfg.optimization.grad_accum_steps
        running_loss, n_seen, lr_now = 0.0, 0, cfg.optimization.lr
        t0 = time.time()

        self.optimizer.zero_grad(set_to_none=True)
        batches_seen, steps_taken = 0, 0
        meter = progress.StepMeter(len(self.train_loader))
        meter.start()
        for i, batch in enumerate(self.train_loader):
            # Everything since the previous step ended was spent blocked here, waiting for
            # workers that prefetch in parallel — so this is idle GPU time, not load time.
            meter.batch_ready()
            if batch is None:
                # Every image in this batch was unreadable. Reported at the end of the
                # epoch by path; here there is simply nothing to train on.
                continue
            x, y, _idx = batch
            batches_seen += 1
            x = x.to(rt.device, non_blocking=True)
            y = y.to(rt.device, non_blocking=True)
            if rt.channels_last:
                x = x.to(memory_format=torch.channels_last)
            if use_mixup:
                x, y = self.mixup_fn(x, y)

            loss = self._forward_loss(x, y) / accum

            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % accum == 0:
                lr_now = self.schedule.apply(self.optimizer, self.state.global_step)
                if self.in_swa_phase(epoch):
                    # SWA averages under a HIGH CONSTANT LR — that is what keeps the
                    # iterates exploring the basin being averaged over. Letting the cosine
                    # schedule continue to decay here collapses SWA into "average a few
                    # nearly-identical checkpoints", which gains nothing.
                    for g in self.optimizer.param_groups:
                        g["lr"] = cfg.experimental.swa_lr
                    lr_now = cfg.experimental.swa_lr
                self._optimizer_step(x, y)
                self.state.global_step += 1
                steps_taken += 1
                if self.ema is not None:
                    self.ema.update(self.model)

            # `.item()` is the only synchronisation point in the iteration, which is what
            # makes the timing below a measurement of compute rather than of kernel
            # launches. Moving it moves the meaning of `compute_s` with it.
            running_loss += loss.item() * accum * x.size(0)
            n_seen += x.size(0)
            meter.step_done(x.size(0))

            if i % 20 == 0:
                emit("progress", epoch=epoch, batch=i, total_batches=len(self.train_loader),
                     loss=running_loss / max(1, n_seen), lr=lr_now,
                     eta_run_s=self._eta_run_s(epoch, meter), **meter.snapshot())
            if meter.due():
                log(meter.line(loss=running_loss / max(1, n_seen), lr=lr_now,
                               eta_run_s=self._eta_run_s(epoch, meter)))

        # An epoch that took no optimizer step trained nothing, but still produces a
        # finite loss and a plausible validation score. Fail loudly instead.
        if steps_taken == 0:
            raise RuntimeError(
                f"Epoch {epoch} completed {batches_seen} batches but took no optimizer "
                f"step, so the model did not train. "
                + (f"grad_accum_steps={accum} is larger than the {batches_seen} batches "
                   f"in an epoch — lower it, or lower batch_size."
                   if batches_seen else
                   f"The training loader produced no batches at all "
                   f"({len(self.train_m)} images, batch_size={cfg.schedule.batch_size}).")
            )

        timing = meter.epoch_metrics()
        # Said once per run, not once per epoch: the fix is a config change, and repeating
        # it every epoch would bury the metrics it is meant to draw attention to.
        if not self._input_bound_warned:
            advice = progress.input_bound_advice(timing["compute_share"], cfg)
            if advice:
                self._input_bound_warned = True
                log(f"  ! {advice}")
                self.tracker.set_tags({"input_bound": "true"})

        return {"train_loss": running_loss / max(1, n_seen), "lr": lr_now,
                "epoch_time_s": time.time() - t0, **timing}

    def _report_broken_images(self) -> None:
        """Say which images could not be read, and stop if too many could not.

        Called once per epoch because the list only grows: a mount that goes away mid-run
        should be reported when it happens, not in the summary of a run that spent hours
        training on a fraction of its data.
        """
        broken = data.read_broken_images(self.broken_log)
        if len(broken) == self._broken_seen:
            return
        self._broken_seen = len(broken)

        total = max(1, len(self.train_m) + len(self.val_m))
        frac = len(broken) / total
        kinds: dict[str, int] = {}
        for _, err in broken:
            kinds[err.split(":", 1)[0]] = kinds.get(err.split(":", 1)[0], 0) + 1
        summary = ", ".join(f"{n} {k}" for k, n in sorted(kinds.items(), key=lambda kv: -kv[1]))

        log(f"  ! {len(broken):,} unreadable image(s) skipped ({frac:.2%} of the dataset): "
            f"{summary}. Full list: {self.broken_log}")
        for path, err in broken[:5]:
            log(f"      {path}  [{err}]")
        if len(broken) > 5:
            log(f"      … and {len(broken) - 5:,} more")
        self.tracker.set_tags({"broken_images": str(len(broken))})
        try:
            self.tracker.log_artifact(self.broken_log, "diagnostics")
        except Exception:
            pass          # the local file is the source of truth; upload is a convenience

        tol = self.cfg.data.broken_image_tolerance
        if tol > 0 and frac > tol:
            raise RuntimeError(
                f"{len(broken):,} of {total:,} images ({frac:.1%}) could not be read, over "
                f"the {tol:.1%} tolerance. That is usually a directory that moved or a "
                f"share that is not mounted rather than a few bad files — the list is in "
                f"{self.broken_log}. Raise data.broken_image_tolerance to train anyway."
            )

    def _eta_run_s(self, epoch: int, meter: "progress.StepMeter | None" = None) -> float:
        """Seconds left in the whole run: the rest of this epoch, plus the ones after it.

        Later epochs are priced at what a full epoch has actually cost so far (training
        AND validation), because eval is not free and an estimate built from training
        alone under-reports the wait by exactly the part the user is sitting through.
        Before any epoch has finished there is no such measurement, so the current pace
        stands in. Called with no meter (at the end of an epoch) the current epoch is
        already done and contributes nothing.
        """
        eta = meter.eta_epoch_s if meter is not None else 0.0
        remaining = self.cfg.schedule.epochs - epoch - 1
        if remaining <= 0:
            return eta
        done = [h["epoch_time_s"] + h.get("val_time_s", 0.0) for h in self.state.history
                if "epoch_time_s" in h]
        if done:
            per_epoch = sum(done) / len(done)
        elif meter is not None:
            per_epoch = meter.step_s * meter.total_steps
        else:
            per_epoch = 0.0
        return eta + remaining * per_epoch

    def _forward_loss(self, x, y) -> torch.Tensor:
        with self._autocast():
            out = self.model(x)
            loss = self.criterion(out, y)
            if self.teacher is not None:
                e = self.cfg.experimental
                with torch.no_grad():
                    t_out = self.teacher(x)
                kd = F.kl_div(
                    F.log_softmax(out / e.kd_temperature, dim=-1),
                    F.log_softmax(t_out / e.kd_temperature, dim=-1),
                    reduction="batchmean", log_target=True,
                ) * (e.kd_temperature ** 2)
                loss = (1 - e.kd_alpha) * loss + e.kd_alpha * kd
        return loss

    def _optimizer_step(self, x, y) -> None:
        clip = self.cfg.optimization.grad_clip_norm

        if self.sam is not None:
            # SAM needs two backward passes, but a GradScaler permits only one
            # `unscale_` per `update()`. So the ascent pass is handled without touching
            # the scaler's bookkeeping, and the descent pass — the one that actually
            # updates the weights — goes through it normally.
            #
            # This branch previously bypassed the scaler entirely: the second backward
            # ran unscaled under fp16 autocast, so its gradients underflowed to zero and
            # `scaler.update()` never ran. Training looked healthy and learned from noise.
            scaled = self.scaler.is_enabled()
            scale = self.scaler.get_scale() if scaled else 1.0

            # The ascent direction is `rho * g/||g||`, so a constant factor on `g`
            # cancels and the perturbation is correct on scaled gradients as-is. Only
            # clipping cares about true magnitudes, and scaling the threshold by the
            # same factor is equivalent to unscaling the gradients.
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip * scale)
            self.sam.first_step()                 # climb to w + e(w)

            self.optimizer.zero_grad(set_to_none=True)
            loss = self._forward_loss(x, y)       # second pass at the perturbed weights
            if scaled:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
            else:
                loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)

            # Back to the original point before stepping. Going through the scaler keeps
            # its inf/NaN detection authoritative: an overflowing step is skipped, and
            # the weights have already been restored, so nothing is left perturbed.
            self.sam.restore()
            if scaled:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            return

        if self.scaler.is_enabled():
            if clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    # ---------------------------------------------------------------- eval

    @torch.no_grad()
    def evaluate(self, model, epoch: int, *, collect_details: bool = False,
                 ensemble_with: list[torch.nn.Module] | None = None) -> dict:
        cfg, rt = self.cfg, self.rt
        model.eval()
        for extra in ensemble_with or []:
            extra.eval()
        mc = metrics_mod.build_metrics(cfg, self.num_classes, rt.device,
                                       class_names=self.class_names)
        cm = torch.zeros(self.num_classes, self.num_classes, dtype=torch.long, device=rt.device)
        total_loss, n = 0.0, 0
        worst: list[dict] = []

        for batch in self.val_loader:
            if batch is None:
                continue
            x, y, sample_idx = batch
            x = x.to(rt.device, non_blocking=True)
            y = y.to(rt.device, non_blocking=True)
            if rt.channels_last:
                x = x.to(memory_format=torch.channels_last)

            with self._autocast():
                probs = metrics_mod.forward_tta(model, x, cfg.validation.tta,
                                                test_size=cfg.effective_test_input_size)
                if ensemble_with:
                    # Average in probability space: logit averaging is dominated by
                    # whichever member happens to be most confident.
                    stack = [probs.float()]
                    for extra in ensemble_with:
                        stack.append(metrics_mod.forward_tta(
                            extra, x, cfg.validation.tta,
                            test_size=cfg.effective_test_input_size).float())
                    probs = torch.stack(stack).mean(0)
            probs = probs.float()
            per_sample = F.nll_loss(torch.log(probs.clamp_min(1e-12)), y, reduction="none")
            total_loss += per_sample.sum().item()
            n += y.numel()

            mc.update(probs, y)
            pred = probs.argmax(-1)
            idx = y * self.num_classes + pred
            cm.view(-1).scatter_add_(0, idx, torch.ones_like(idx))

            if collect_details and cfg.validation.log_worst_predictions:
                conf = probs.max(-1).values
                for j in range(y.numel()):
                    worst.append({
                        # Real dataset index, so this stays correct under any loader order.
                        "path": self.val_m.paths[int(sample_idx[j])],
                        "loss": per_sample[j].item(),
                        "true": self.class_names[y[j].item()],
                        "pred": self.class_names[pred[j].item()],
                        "conf": conf[j].item(),
                    })

        out: dict = {"val_loss": total_loss / max(1, n)}
        for k, v in mc.compute().items():
            if v.ndim == 0:
                out[k] = v.item()
            else:
                for ci, val in enumerate(v.tolist()):
                    out[f"{k}/{self.class_names[ci]}"] = val
        out["_cm"] = cm.cpu().numpy()
        out["_worst"] = worst
        return out

    # ---------------------------------------------------------------- checkpoints

    # ---------------------------------------------------------------- warm start

    def _init_from_checkpoint(self, ref: str) -> None:
        """Initialise the model's weights from a checkpoint — and nothing else.

        The counterpart to `_restore`: that continues a run (optimizer, schedule, epoch
        and best tracking included), this begins a NEW one that merely starts from
        trained weights. A full TrainLab payload is accepted and everything beyond the
        weights is deliberately ignored, which is what lets a backbone trained under one
        config carry into a different one — new schedule, new LR, new head.
        """
        cfg = self.cfg
        path = self._resolve_resume_path(ref, prefer=("best.ckpt", "last.ckpt"),
                                         field="pretrained_checkpoint")
        raw = _load_checkpoint(path, map_location="cpu")

        ck_classes: list[str] | None = None
        if isinstance(raw, dict) and isinstance(raw.get("model"), dict):
            # A full TrainLab checkpoint. `best_weights` names the weights its score was
            # measured on — for best.ckpt that may be the EMA copy, which is the copy a
            # warm start wants.
            # The payload records what built it; a different backbone is refused by
            # name here, before tensor matching gets a chance to half-succeed.
            was_bb = ((raw.get("config") or {}).get("model") or {}).get("backbone")
            if was_bb and was_bb != cfg.model.backbone:
                raise RuntimeError(
                    f"pretrained_checkpoint: {path} was trained with backbone "
                    f"'{was_bb}', but this config builds '{cfg.model.backbone}'. "
                    f"Same-family architectures share tensor names, so a partial load "
                    f"would quietly mix trained and random weights."
                )
            which = raw.get("best_weights", "model")
            state = raw["ema"] if (which == "ema" and raw.get("ema")) else raw["model"]
            ck_classes = raw.get("class_names")
            log(f"  warm start: {path} (epoch {raw.get('epoch', '?')}, "
                f"{raw.get('primary_metric', 'score')} {raw.get('score', float('nan')):.4f}, "
                f"{which} weights)")
        elif isinstance(raw, dict) and isinstance(raw.get("state_dict"), dict):
            state = raw["state_dict"]          # the other common wrapper convention
        elif isinstance(raw, dict) and raw and all(
                isinstance(v, torch.Tensor) for v in raw.values()):
            state = raw                        # a bare state dict
        else:
            raise RuntimeError(
                f"pretrained_checkpoint: {path} is not a checkpoint this app can read — "
                f"expected a TrainLab .ckpt, a {{'state_dict': ...}} wrapper, or a raw "
                f"state dict of tensors."
            )

        state = _unwrap_state(state)

        # The head transfers only when it provably means the same classes. Name+shape
        # matching alone would load a head trained on a same-sized but DIFFERENT class
        # list, and every prediction would be plausibly, silently wrong.
        if ck_classes is not None and list(ck_classes) != list(self.class_names):
            head_keys = models.classifier_param_names(self.model)
            state = {k: v for k, v in state.items() if k not in head_keys}
            log(f"  warm start: classifier left at fresh init — checkpoint classes "
                f"{list(ck_classes)} != dataset classes {list(self.class_names)}")

        try:
            models.load_state_into(self.model, state, source=str(path), log=log)
        except RuntimeError as e:
            was = ""
            if isinstance(raw, dict):
                was_bb = ((raw.get("config") or {}).get("model") or {}).get("backbone")
                was = f" It was written by backbone '{was_bb}'." if was_bb else ""
            raise RuntimeError(
                f"pretrained_checkpoint does not fit model.backbone="
                f"'{cfg.model.backbone}'.{was} ({e})"
            ) from e
        self.tracker.set_tags({"pretrained_checkpoint": str(path)[:250]})

    # ---------------------------------------------------------------- resume

    @staticmethod
    def _resolve_resume_path(ref: str, prefer: tuple[str, ...] = ("last.ckpt", "best.ckpt"),
                             field: str = "resume_from") -> Path:
        """A checkpoint file, or a run directory — resolved by `prefer` order.

        Resume prefers `last.ckpt`: it continues the optimizer trajectory, and the last
        state is the one the trajectory actually passed through most recently. A warm
        start (`pretrained_checkpoint`) prefers `best.ckpt`: a new run wants the best
        weights the old one produced, not wherever it happened to stop.
        """
        p = Path(ref).expanduser()
        if p.is_dir():
            for name in prefer:
                if (p / name).exists():
                    return p / name
            raise FileNotFoundError(
                f"{field}: no {' or '.join(prefer)} in {p}. Point at the run "
                f"directory that contains them, or at a .ckpt file directly."
            )
        if not p.exists():
            raise FileNotFoundError(f"{field}: {p} does not exist.")
        return p

    def _restore(self, ref: str) -> None:
        """Continue training from a checkpoint written by `_save_checkpoint`.

        The CURRENT config governs the resumed run — that is what makes "raise
        schedule.epochs and resume" the way to train longer — so anything that must
        match the original run (classes, architecture, optimizer shape) is checked
        rather than assumed, and anything that is honest to reset (top-K retention,
        the broken-image log, RNG state and therefore shuffle order) starts fresh.
        """
        cfg = self.cfg
        path = self._resolve_resume_path(ref)
        ck = _load_checkpoint(path, map_location="cpu")

        was = ck.get("class_names")
        if was and list(was) != list(self.class_names):
            raise ValueError(
                f"resume_from: {path} was trained on classes {list(was)}, but the "
                f"current dataset has {list(self.class_names)}. Label indices would "
                f"silently refer to different classes."
            )

        try:
            _unwrap_model(self.model).load_state_dict(_unwrap_state(ck["model"]))
        except RuntimeError as e:
            was_bb = ((ck.get("config") or {}).get("model") or {}).get("backbone", "unknown")
            raise RuntimeError(
                f"resume_from: {path} does not fit this model — it was written by "
                f"backbone '{was_bb}', the current config builds "
                f"'{cfg.model.backbone}'. ({e})"
            ) from e

        if self.ema is not None:
            # The EMA module was deep-copied from the *initial* weights before this
            # restore ran; left alone it would average the resumed model against an
            # initialisation it never trained from.
            _unwrap_model(self.ema.module).load_state_dict(
                _unwrap_state(ck["ema"] if ck.get("ema") else ck["model"]))

        try:
            self.optimizer.load_state_dict(ck["optimizer"])
        except (ValueError, KeyError) as e:
            raise RuntimeError(
                f"resume_from: the optimizer state in {path} does not match the current "
                f"optimization settings (optimizer, layer_lr_decay, head_lr_mult and "
                f"no_weight_decay_on_norm_bias all shape the parameter groups). Resume "
                f"with the settings the run was started with. ({e})"
            ) from e

        if ck.get("scaler") and self.scaler.is_enabled():
            self.scaler.load_state_dict(ck["scaler"])
        if ck.get("sched"):
            self.schedule.load_state_dict(ck["sched"])

        rs = ck.get("run_state") or {}
        st = self.state
        st.global_step = int(rs.get("global_step", (ck["epoch"] + 1) * self.steps_per_epoch))
        st.best_metric = float(rs["best_metric"]) if "best_metric" in rs else st.best_metric
        st.best_epoch = int(rs.get("best_epoch", ck["epoch"]))
        st.epochs_without_improvement = int(rs.get("epochs_without_improvement", 0))
        st.history = list(rs.get("history", []))
        self._best_is_ema = ck.get("best_weights") == "ema"
        if self.ema is not None:
            # Without the counter the decay ramp restarts and the first resumed steps
            # drag the (good) EMA hard toward the live weights.
            self.ema.updates = int(rs.get("ema_updates", st.global_step))

        self.start_epoch = int(ck["epoch"]) + 1
        if self.start_epoch >= cfg.schedule.epochs:
            raise ValueError(
                f"resume_from: {path} already completed epoch {ck['epoch'] + 1} of "
                f"{cfg.schedule.epochs} — there is nothing left to train. Raise "
                f"schedule.epochs above {self.start_epoch} to continue."
            )

        log(f"  resumed from {path} — continuing at epoch {self.start_epoch + 1}/"
            f"{cfg.schedule.epochs} (step {st.global_step:,}, "
            f"best {cfg.primary_metric_key} {st.best_metric:.4f} @ epoch {st.best_epoch + 1})")
        self.tracker.set_tags({"resumed_from": str(path)[:250],
                               "resumed_at_epoch": str(self.start_epoch)})

    def _save_checkpoint(self, epoch: int, score: float, is_best: bool) -> None:
        cfg = self.cfg
        payload = {
            # Unwrapped names: a compiled run's checkpoint must load into an uncompiled
            # model (warm start, KD teacher, ensembling, a resume with compile off).
            "model": _unwrap_model(self.model).state_dict(),
            "ema": _unwrap_model(self.ema.module).state_dict() if self.ema else None,
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "score": score,
            "primary_metric": cfg.primary_metric_key,
            # Which of the two weight sets `score` was actually measured on. Consumers
            # (ensembling, KD) must load that one; blindly preferring `ema` loads weights
            # the reported score never described.
            "best_weights": "ema" if (is_best and self._best_is_ema) else "model",
            "config": cfg.model_dump(mode="json"),
            "class_names": self.class_names,
            # Everything `resume_from` needs beyond the weights. Values are coerced to
            # plain Python so the payload stays loadable under torch's safe unpickler —
            # a numpy scalar in the history would force weights_only=False on every
            # resume. `state.checkpoints` is deliberately absent: top-K retention manages
            # files, and a resumed run must not reach into another run's directory to
            # delete things.
            "scaler": self.scaler.state_dict() if self.scaler.is_enabled() else None,
            "sched": self.schedule.state_dict(),
            "run_state": {
                "global_step": int(self.state.global_step),
                "best_metric": float(self.state.best_metric),
                "best_epoch": int(self.state.best_epoch),
                "epochs_without_improvement": int(self.state.epochs_without_improvement),
                "ema_updates": int(self.ema.updates) if self.ema else 0,
                "history": [
                    {k: (float(v) if isinstance(v, (int, float)) else v)
                     for k, v in h.items() if isinstance(v, (int, float, str))}
                    for h in self.state.history
                ],
            },
        }
        if cfg.checkpoint.save_last:
            torch.save(payload, self.out_dir / "last.ckpt")

        if cfg.checkpoint.save_top_k > 0:
            p = self.out_dir / f"epoch{epoch:03d}-{score:.4f}.ckpt"
            torch.save(payload, p)
            self.state.checkpoints.append((score, p))
            # Best-first, honouring the metric's direction: sorting descending for a
            # lower-is-better metric retains exactly the K worst checkpoints.
            self.state.checkpoints.sort(key=lambda t: t[0], reverse=self.maximize)
            for _, old in self.state.checkpoints[cfg.checkpoint.save_top_k:]:
                old.unlink(missing_ok=True)
            del self.state.checkpoints[cfg.checkpoint.save_top_k:]
            if is_best:
                shutil.copy2(p, self.out_dir / "best.ckpt")

    # ---------------------------------------------------------------- sample selection

    @torch.no_grad()
    def _score_train_samples(self) -> np.ndarray:
        """Per-sample loss over the training set, scored with EVAL transforms.

        Deliberately a separate clean pass rather than reusing the training loss:
        under mixup the training loss belongs to a blended pair and carries no per-sample
        meaning, and augmentation noise would dominate the ranking. Costs roughly a third
        of an epoch, and only runs when curriculum or noise filtering is active.
        """
        cfg, rt = self.cfg, self.rt
        was_training = self.model.training
        self.model.eval()

        # Scored at the TRAIN resolution: the ranking should reflect what the model
        # finds hard at the resolution it is being trained at, not the (possibly
        # larger) FixRes test resolution.
        scorer = data.ImageListDataset(
            self.train_m,
            transforms.build_eval_transform(cfg, size=cfg.input.input_size),
            cfg.data.cache_mode,
            broken_log=self.broken_log,
        )
        loader = DataLoader(scorer, batch_size=cfg.schedule.batch_size, shuffle=False,
                            collate_fn=data.collate_skip_broken, **data.loader_kwargs(cfg))

        scores = np.zeros(len(self.train_m), dtype=np.float64)
        for batch in loader:
            if batch is None:
                continue
            x, y, idx = batch
            x = x.to(rt.device, non_blocking=True)
            y = y.to(rt.device, non_blocking=True)
            if rt.channels_last:
                x = x.to(memory_format=torch.channels_last)
            with self._autocast():
                logits = self.model(x)
            per = F.cross_entropy(logits.float(), y, reduction="none")
            scores[idx.numpy()] = per.detach().cpu().numpy()

        if was_training:
            self.model.train()
        return scores

    def _active_indices(self, epoch: int) -> tuple[list[int] | None, dict]:
        """Which training samples this epoch may use, plus stats for logging."""
        cfg = self.cfg
        E = cfg.experimental
        if not (E.curriculum_by_loss or E.label_noise_filter):
            return None, {}

        filtering = E.label_noise_filter and epoch >= E.label_noise_start_epoch
        if not filtering and not E.curriculum_by_loss:
            return None, {}

        if self._sample_scores is None:
            return None, {}

        n = len(self._sample_scores)
        order = np.argsort(self._sample_scores)          # easiest first
        keep = order

        stats: dict = {}
        if filtering:
            # Drop the persistently hardest tail — on scraped data these are mostly
            # mislabelled rather than genuinely difficult.
            n_drop = int(round(n * E.label_noise_percentile))
            if n_drop > 0:
                dropped = order[-n_drop:]
                keep = order[:-n_drop]
                self._dropped_indices = set(int(i) for i in dropped)
                stats["noise_dropped"] = n_drop

        if E.curriculum_by_loss:
            # Grow the pool from the easiest `start_frac` to everything, by the epoch the
            # curriculum ends.
            end = max(1, E.curriculum_epochs)
            frac = E.curriculum_start_frac + (1.0 - E.curriculum_start_frac) * min(1.0, epoch / end)
            n_keep = max(cfg.schedule.batch_size, int(round(len(keep) * frac)))
            keep = keep[:n_keep]
            stats["curriculum_frac"] = round(frac, 3)

        stats["active_samples"] = len(keep)
        return [int(i) for i in keep], stats

    def _refresh_sample_selection(self, epoch: int) -> dict:
        """Rescore and rebuild the training loader if selection is active."""
        E = self.cfg.experimental
        if not (E.curriculum_by_loss or E.label_noise_filter):
            return {}

        # Scoring needs a model that has learned something; before that, use everything.
        min_epoch = 1 if E.curriculum_by_loss else E.label_noise_start_epoch
        if epoch < min_epoch:
            return {}

        self._sample_scores = self._score_train_samples()
        active, stats = self._active_indices(epoch)
        # Remembered so the progressive-resize loader rebuild keeps the subset.
        self._active_subset = active
        if active is None:
            return stats

        self.train_loader = data.make_train_loader(
            self.cfg, self.train_loader.dataset, self.train_m, active=active)
        log(f"  sample selection: {stats.get('active_samples', '?')}/{len(self.train_m)} active"
            + (f", {stats['noise_dropped']} dropped as noisy" if "noise_dropped" in stats else "")
            + (f", curriculum {stats['curriculum_frac']:.2f}" if "curriculum_frac" in stats else ""))
        return stats

    def _log_dropped_samples(self) -> None:
        """Record which files noise filtering discarded — the point is to inspect them."""
        if not self._dropped_indices:
            return
        rows = sorted(
            ((float(self._sample_scores[i]), self.train_m.paths[i],
              self.class_names[self.train_m.labels[i]]) for i in self._dropped_indices),
            reverse=True,
        )
        path = self.out_dir / "dropped_as_noisy.csv"
        with path.open("w") as f:
            f.write("loss,label,path\n")
            for loss, p, label in rows:
                f.write(f"{loss:.6f},{label},{p}\n")
        self.tracker.log_artifact(path, "diagnostics")
        log(f"  wrote {len(rows)} dropped samples to {path.name} — check these for label noise")

    # ---------------------------------------------------------------- ensembling

    def _load_ensemble_models(self) -> list[torch.nn.Module]:
        """Load previous runs' checkpoints for probability averaging at final eval.

        Accepts local checkpoint paths or MLflow run IDs. Members whose class list or
        head shape disagrees with this run are refused rather than silently misaligned.
        """
        ids = self.cfg.experimental.ensemble_run_ids
        if not ids:
            return []

        out: list[torch.nn.Module] = []
        for ref in ids:
            try:
                ckpt_path = self._resolve_checkpoint(ref)
                if ckpt_path is None:
                    log(f"  ! ensemble: could not resolve '{ref}'")
                    continue
                ck = _load_checkpoint(ckpt_path, map_location="cpu")
                names = ck.get("class_names")
                if names and list(names) != list(self.class_names):
                    log(f"  ! ensemble: '{ref}' was trained on different classes — skipped")
                    continue
                member_cfg = TrainConfig.model_validate(ck["config"])
                member = models.create_model(member_cfg, self.num_classes)
                # Load the weights the checkpoint's score was measured on. Older
                # checkpoints carry no marker, so fall back to the previous behaviour.
                which = ck.get("best_weights") or ("ema" if ck.get("ema") else "model")
                member.load_state_dict(
                    _unwrap_state(ck[which] if ck.get(which) else ck["model"]))
                member.to(self.rt.device).eval().requires_grad_(False)
                out.append(member)
                log(f"  ensemble member: {member_cfg.model.backbone} from {ref[:12]}")
            except Exception as e:
                log(f"  ! ensemble: '{ref}' failed to load ({type(e).__name__}: {e})")
        return out

    def _resolve_checkpoint(self, ref: str) -> str | None:
        p = Path(ref)
        if p.exists():
            return str(p / "best.ckpt") if p.is_dir() else str(p)
        try:
            from mlflow.artifacts import download_artifacts

            return download_artifacts(
                run_id=ref, artifact_path="checkpoints/best.ckpt",
                tracking_uri=self.cfg.tracking.tracking_uri)
        except Exception:
            return None

    # ---------------------------------------------------------------- cRT

    def _classifier_retrain(self, epoch_offset: int) -> dict:
        """Stage 2 of decoupled training: rebalance ONLY the classifier.

        Kang et al. (2020) found instance-balanced (plain random) sampling learns the best
        representations, and that only the classifier needs rebalancing — so the backbone
        is frozen here and the head is retrained with class-balanced sampling.
        """
        cfg = self.cfg
        n_epochs = cfg.experimental.classifier_retrain_epochs
        lr = cfg.resolved_crt_lr()
        log(f"\n  cRT: retraining the classifier for {n_epochs} epochs at lr {lr:.2e} "
            f"with class-balanced sampling (backbone frozen)")

        head = models.get_classifier(self.model)
        if head is None:
            log("  ! cRT skipped: could not locate a classifier head")
            return {}

        # cRT freezes the whole backbone. Remember what was trainable so the model is
        # handed back to later stages in the state they expect rather than fully frozen.
        was_trainable = {id(p): p.requires_grad for p in self.model.parameters()}
        for p in self.model.parameters():
            p.requires_grad_(False)
        if cfg.experimental.classifier_retrain_reinit:
            for m in ([head] if isinstance(head, torch.nn.Module) else []):
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.trunc_normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
        for p in head.parameters():
            p.requires_grad_(True)

        opt = torch.optim.SGD([p for p in head.parameters()], lr=lr, momentum=0.9)
        loader = data.make_train_loader(cfg, self.train_loader.dataset, self.train_m,
                                        force_balanced=True)
        # Mixup produces soft targets the plain CE below cannot consume, and cRT is about
        # the decision boundary rather than augmentation — so it trains on clean labels.
        crit = torch.nn.CrossEntropyLoss(label_smoothing=cfg.loss.label_smoothing)

        # cRT has its own optimizer, so it needs its own scaler; sharing the main one
        # would interleave two different step schedules through the same scale state.
        scaler = torch.amp.GradScaler(device=self.rt.device_str,
                                      enabled=self.rt.use_grad_scaler)

        best: dict = {}
        primary = cfg.primary_metric_key
        try:
            for e in range(n_epochs):
                # The entire frozen backbone stays in eval mode: BN running stats must
                # not drift, and drop-path/dropout noise on frozen features only makes
                # the head's targets stochastic for no benefit. Gradients still flow to
                # the head — eval mode changes module behaviour, not autograd. (This
                # used to call model.train() and only switch BN modules back, leaving
                # stochastic depth active during head retraining.)
                self.model.eval()
                total, seen = 0.0, 0
                for batch in loader:
                    if batch is None:
                        continue
                    x, y, _ = batch
                    x = x.to(self.rt.device, non_blocking=True)
                    y = y.to(self.rt.device, non_blocking=True)
                    if self.rt.channels_last:
                        x = x.to(memory_format=torch.channels_last)
                    with self._autocast():
                        loss = crit(self.model(x), y)
                    opt.zero_grad(set_to_none=True)
                    # Same fp16 underflow hazard as the main loop: an unscaled backward
                    # under autocast silently zeroes small gradients.
                    if scaler.is_enabled():
                        scaler.scale(loss).backward()
                        scaler.step(opt)
                        scaler.update()
                    else:
                        loss.backward()
                        opt.step()
                    total += loss.item() * x.size(0)
                    seen += x.size(0)

                ev = self.evaluate(self.model, epoch_offset + e)
                row = {f"crt/{k}": v for k, v in ev.items() if not k.startswith("_")}
                row["crt/train_loss"] = total / max(1, seen)
                self.tracker.log_metrics(row, step=epoch_offset + e)
                log(f"  cRT epoch {e+1}/{n_epochs}  loss {row['crt/train_loss']:.4f}  "
                    f"{primary} {ev.get(primary, float('nan')):.4f}")
                best = row

                # cRT is a real candidate for the deployed model, so let it compete for
                # `best.ckpt`. Previously its metrics were logged under `crt/*` and then
                # ignored: a cRT stage that improved the primary metric left the reported
                # best, and the saved checkpoint, at the pre-cRT values.
                score = ev.get(primary)
                if score is not None and self._improves(score, self.state.best_metric):
                    self.state.best_metric = score
                    self.state.best_epoch = epoch_offset + e
                    self._best_is_ema = False
                    self._save_checkpoint(epoch_offset + e, score, is_best=True)
                    log(f"    cRT improved {primary} to {score:.4f} — new best checkpoint")
        finally:
            for p in self.model.parameters():
                p.requires_grad_(was_trainable.get(id(p), True))
        return best

    # ---------------------------------------------------------------- pseudo-labeling

    @torch.no_grad()
    def _predict_unlabeled(self, paths: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Confidences and predicted labels for an unlabeled pool."""
        cfg, rt = self.cfg, self.rt
        self.model.eval()
        ds = data.UnlabeledDataset(
            paths, transforms.build_eval_transform(cfg, size=cfg.effective_test_input_size),
            cfg.data.cache_mode, broken_log=self.broken_log)
        loader = DataLoader(ds, batch_size=cfg.schedule.batch_size, shuffle=False,
                            collate_fn=data.collate_skip_broken, **data.loader_kwargs(cfg))
        conf = np.zeros(len(paths), dtype=np.float32)
        pred = np.zeros(len(paths), dtype=np.int64)
        for batch in loader:
            if batch is None:
                continue
            x, idx = batch
            x = x.to(rt.device, non_blocking=True)
            if rt.channels_last:
                x = x.to(memory_format=torch.channels_last)
            with self._autocast():
                probs = metrics_mod.forward_tta(self.model, x, cfg.validation.tta,
                                                test_size=cfg.effective_test_input_size)
            probs = probs.float()
            c, p = probs.max(-1)
            conf[idx.numpy()] = c.cpu().numpy()
            pred[idx.numpy()] = p.cpu().numpy()
        return conf, pred

    def _pseudo_label_stage(self, epoch_offset: int) -> dict:
        """Self-training rounds on an unlabeled directory.

        Deviates from Noisy Student in one respect worth knowing: it continues training the
        SAME model rather than training a fresh (larger) student from scratch each round,
        which is far cheaper and fits an interactive tool, but gives up some of the
        published gain and is more prone to confirmation bias.
        """
        cfg = self.cfg
        E = cfg.experimental
        pool = data.scan_unlabeled(E.pseudo_label_dir)
        log(f"\n  pseudo-labeling: {len(pool):,} unlabeled images, "
            f"{E.pseudo_label_rounds} round(s) x {E.pseudo_label_epochs} epochs, "
            f"threshold {E.pseudo_label_threshold}")

        base_m = self.train_m
        row: dict = {}
        step = epoch_offset

        for rnd in range(E.pseudo_label_rounds):
            conf, pred = self._predict_unlabeled(pool)
            accepted = np.where(conf >= E.pseudo_label_threshold)[0]
            # Cap the pseudo set so a large pool cannot drown out the ground truth.
            cap = int(len(base_m) * E.pseudo_label_max_ratio)
            if len(accepted) > cap:
                accepted = accepted[np.argsort(-conf[accepted])[:cap]]

            if len(accepted) == 0:
                log(f"  round {rnd+1}: no predictions above threshold — stopping")
                break

            per_class = np.bincount(pred[accepted], minlength=self.num_classes)
            log(f"  round {rnd+1}: accepted {len(accepted):,}/{len(pool):,} "
                f"({100*len(accepted)/len(pool):.1f}%), mean conf "
                f"{conf[accepted].mean():.3f}, per-class {per_class.tolist()}")
            self.tracker.log_metrics({
                "pseudo/accepted": float(len(accepted)),
                "pseudo/accept_rate": float(len(accepted) / len(pool)),
                "pseudo/mean_confidence": float(conf[accepted].mean()),
            }, step=step)

            extended = data.extend_manifest(
                base_m, [pool[i] for i in accepted], [int(pred[i]) for i in accepted])
            ds = data.ImageListDataset(
                extended, transforms.build_train_transform(cfg), cfg.data.cache_mode,
                broken_log=self.broken_log)
            loader = data.make_train_loader(cfg, ds, extended)

            saved_loader, saved_m = self.train_loader, self.train_m
            self.train_loader, self.train_m = loader, extended
            try:
                for e in range(E.pseudo_label_epochs):
                    tr = self.train_epoch(step)
                    ev = self.evaluate(self.model, step)
                    row = {f"pseudo/{k}": v for k, v in {**tr, **ev}.items()
                           if not k.startswith("_")}
                    self.tracker.log_metrics(row, step=step)
                    primary = cfg.primary_metric_key
                    log(f"  pseudo round {rnd+1} epoch {e+1}/{E.pseudo_label_epochs}  "
                        f"{primary} {ev.get(primary, float('nan')):.4f}")
                    step += 1
            finally:
                self.train_loader, self.train_m = saved_loader, saved_m
        return row

    # ---------------------------------------------------------------- main loop

    def fit(self) -> dict:
        cfg = self.cfg
        primary = cfg.primary_metric_key
        reset_peak_memory(self.rt.device_str)

        if cfg.tracking.log_augmentation_preview:
            try:
                p = artifacts.augmentation_preview(
                    self.train_loader.dataset, self.train_loader.dataset.transform,
                    self.mean, self.std, self.out_dir / "augmentation_preview.png",
                    seed=cfg.schedule.seed)
                self.tracker.log_artifact(p, "previews")
            except Exception as e:
                log(f"  ! augmentation preview failed: {e}")

        for epoch in range(self.start_epoch, cfg.schedule.epochs):
            sel_stats = self._refresh_sample_selection(epoch)
            tr = self.train_epoch(epoch)
            tr.update({f"selection/{k}": v for k, v in sel_stats.items()})

            if self.in_swa_phase(epoch):
                self.swa.update(self.model)

            do_eval = ((epoch + 1) % cfg.validation.eval_every_n_epochs == 0
                       or epoch == cfg.schedule.epochs - 1)
            if not do_eval:
                self.tracker.log_metrics(tr, step=epoch)
                continue

            is_last = epoch == cfg.schedule.epochs - 1
            t_val = time.time()
            ev = self.evaluate(self.model, epoch, collect_details=is_last)
            row = {**tr, **{k: v for k, v in ev.items() if not k.startswith("_")}}

            if self.ema is not None and cfg.validation.eval_ema_weights:
                ev_ema = self.evaluate(self.ema.module, epoch, collect_details=False)
                for k, v in ev_ema.items():
                    if not k.startswith("_"):
                        row[f"ema/{k}"] = v
            # Evaluating twice (raw + EMA) is a real share of the epoch, and it is not in
            # `epoch_time_s`. Without it, "why is my epoch 3x the training time?" has no
            # answer in the numbers.
            row["val_time_s"] = round(time.time() - t_val, 2)

            raw_score = row.get(primary, self._worst_score())
            ema_score = row.get(f"ema/{primary}", self._worst_score())
            use_ema = (
                self.ema is not None and cfg.validation.eval_ema_weights
                and f"ema/{primary}" in row
                and (ema_score > raw_score if self.maximize else ema_score < raw_score)
            )
            score = ema_score if use_ema else raw_score
            # Which weights the score belongs to decides which weights `best.ckpt` must
            # contain. Recording it stops the EMA copy being scored and the raw copy saved.
            self._best_is_ema = use_ema

            self._report_broken_images()
            row["broken_images"] = self._broken_seen
            row["peak_vram_gb"] = peak_memory_gb(self.rt.device_str)
            self.state.history.append({"epoch": epoch, **row})
            self.tracker.log_metrics(row, step=epoch)

            improved = self._improves(score, self.state.best_metric)
            self.schedule.on_plateau(improved)
            if improved:
                self.state.best_metric, self.state.best_epoch = score, epoch
                self.state.epochs_without_improvement = 0
            else:
                self.state.epochs_without_improvement += 1

            self._save_checkpoint(epoch, score, improved)

            log(f"  epoch {epoch+1}/{cfg.schedule.epochs}  "
                f"loss {row['train_loss']:.4f}  val_loss {row['val_loss']:.4f}  "
                f"{primary} {row.get(primary, float('nan')):.4f}"
                f"{'  *best*' if improved else ''}  "
                f"({row['epoch_time_s']:.1f}s train [data {row.get('data_wait_s', 0):.1f}s "
                f"+ compute {row.get('compute_s', 0):.1f}s, {row.get('imgs_per_s', 0):,.0f} img/s] "
                f"+ {row['val_time_s']:.1f}s val, "
                f"eta {progress.fmt_duration(self._eta_run_s(epoch))})")
            emit("epoch", epoch=epoch, total=cfg.schedule.epochs, metrics=
                 {k: v for k, v in row.items() if isinstance(v, (int, float))},
                 best=self.state.best_metric, primary_metric=primary,
                 # Consumers (the sweep driver, the UI) must not assume "bigger is better".
                 higher_is_better=self.maximize)

            if is_last or (cfg.checkpoint.early_stopping
                           and self.state.epochs_without_improvement >= cfg.checkpoint.es_patience):
                self._final_artifacts(ev, epoch)
                if not is_last:
                    log(f"  early stopping at epoch {epoch+1} "
                        f"(no improvement for {cfg.checkpoint.es_patience} evals)")
                break

        if self.swa is not None:
            if self.swa.n == 0:
                # Early stopping fired before swa_start_epoch: the SWA module still
                # holds the pre-training deepcopy from setup(). Evaluating it would
                # log plausible-looking swa/* metrics that describe the initialization.
                log("  ! SWA skipped: training ended before the SWA phase began "
                    "(no checkpoints were averaged)")
            else:
                self.swa.update_bn(self.train_loader, self.rt.device)
                ev = self.evaluate(self.swa.module, cfg.schedule.epochs)
                self.tracker.log_metrics({f"swa/{k}": v for k, v in ev.items()
                                          if not k.startswith("_")}, step=cfg.schedule.epochs)

        # ---- post-training stages, in dependency order --------------------------------
        extra: dict = {}
        step = cfg.schedule.epochs + 1

        self._log_dropped_samples()

        if cfg.experimental.pseudo_label_dir:
            extra["pseudo"] = self._pseudo_label_stage(step)
            step += cfg.experimental.pseudo_label_rounds * cfg.experimental.pseudo_label_epochs

        if cfg.experimental.classifier_retrain_epochs > 0:
            extra["crt"] = self._classifier_retrain(step)
            step += cfg.experimental.classifier_retrain_epochs

        # Ensembling is last: it evaluates whatever the preceding stages produced.
        if cfg.experimental.ensemble_run_ids:
            self._ensemble = self._load_ensemble_models()
            if self._ensemble:
                ev = self.evaluate(self.model, step, ensemble_with=self._ensemble)
                ens = {f"ensemble/{k}": v for k, v in ev.items() if not k.startswith("_")}
                self.tracker.log_metrics(ens, step=step)
                solo = self.state.history[-1].get(primary) if self.state.history else None
                got = ev.get(primary)
                log(f"\n  ensemble of {len(self._ensemble) + 1} models: {primary} {got:.4f}"
                    + (f"  (single model {solo:.4f}, delta {got - solo:+.4f})"
                       if solo is not None and got is not None else ""))
                extra["ensemble"] = ens

        # Everything in this dict crosses a JSON boundary (the @@TRAINLAB@@ "finished"
        # event and result.json), so it must contain only JSON-serialisable values.
        # `SSLResult.encoder_path` is a Path — left raw, it made every SSL run with
        # the default ssl_save_encoder=true crash in `emit("finished")` AFTER all the
        # training work was done, and the run was recorded as FAILED.
        ssl_info = None
        if self.ssl_result:
            ssl_info = {k: (str(v) if isinstance(v, Path) else v)
                        for k, v in vars(self.ssl_result).items()}

        return {
            "best_metric": self.state.best_metric,
            "best_epoch": self.state.best_epoch,
            "primary_metric": primary,
            "history": self.state.history,
            "stages": extra,
            "ssl": ssl_info,
        }

    def _final_artifacts(self, ev: dict, epoch: int) -> None:
        cfg = self.cfg
        try:
            if cfg.validation.log_confusion_matrix:
                p = artifacts.confusion_matrix_png(
                    ev["_cm"], self.class_names, self.out_dir / "confusion_matrix.png")
                self.tracker.log_artifact(p, "diagnostics")
            if cfg.validation.log_worst_predictions and ev["_worst"]:
                p = artifacts.worst_predictions_png(
                    ev["_worst"], self.out_dir / "worst_predictions.png", self.mean, self.std)
                self.tracker.log_artifact(p, "diagnostics")
        except Exception as e:
            log(f"  ! artifact generation failed: {e}")

        best = self.out_dir / "best.ckpt"
        if best.exists() and cfg.checkpoint.save_top_k > 0:
            self.tracker.log_artifact(best, "checkpoints")
