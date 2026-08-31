"""Device resolution and device-aware config downgrades.

The config records what the user *asked for*; this module decides what will actually
run, and returns the list of downgrades so they can be surfaced in the UI and logged
to the tracker. A run record should always reflect what executed, not what was requested.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field

import torch

from .config import Amp, Device, TrainConfig


@dataclass
class ResolvedRuntime:
    device: torch.device
    device_str: str          # 'cuda' | 'mps' | 'cpu'
    device_name: str         # human-readable accelerator name
    amp_dtype: torch.dtype | None
    amp_enabled: bool
    use_grad_scaler: bool
    channels_last: bool
    torch_compile: bool
    total_memory_gb: float | None
    downgrades: list[str] = field(default_factory=list)


def _available(kind: str) -> bool:
    if kind == "cuda":
        return torch.cuda.is_available()
    if kind == "mps":
        return torch.backends.mps.is_available()
    return True


def _pick_device(requested: Device) -> str:
    """Resolve the requested device against what this machine actually has.

    An explicit request used to be returned verbatim, which made the "device unavailable"
    downgrade below unreachable (`kind` was always equal to the request) and turned
    `device=cuda` on a Mac into a bare `AssertionError: Torch not compiled with CUDA
    enabled` from somewhere deep in the first tensor move.
    """
    if requested != Device.AUTO:
        return requested.value if _available(requested.value) else _fallback()
    return _fallback()


def _fallback() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _accelerator_name(kind: str) -> str:
    if kind == "cuda":
        return torch.cuda.get_device_name(0)
    if kind == "mps":
        return f"Apple {platform.machine()} (MPS)"
    return platform.processor() or platform.machine() or "CPU"


def _total_memory_gb(kind: str) -> float | None:
    if kind == "cuda":
        return torch.cuda.get_device_properties(0).total_memory / 1024**3
    if kind == "mps":
        try:
            return torch.mps.recommended_max_memory() / 1024**3
        except Exception:
            return None
    return None


def resolve_amp_for(amp: Amp, kind: str) -> tuple[torch.dtype | None, bool, str | None]:
    """Resolve an AMP request against a device.

    Shared by the supervised loop and the SSL stage so both downgrade identically.
    Returns (dtype, needs_grad_scaler, note) where note describes any downgrade.
    """
    if amp == Amp.OFF:
        return None, False, None

    if kind == "cpu":
        return None, False, "AMP disabled: rarely a win on CPU"

    if kind == "mps":
        # MPS autocast disables itself for bf16, and where bf16 does run the backend
        # lacks optimized Metal kernels for it. fp16 is the correct choice here.
        note = "amp bf16 -> fp16: MPS autocast does not support bf16" if amp == Amp.BF16 else None
        # fp16 has ~6e-8 minimum subnormal, so small gradients flush to zero without loss
        # scaling. torch.amp.GradScaler supports MPS, and running fp16 unscaled was a
        # silent accuracy risk on the one device this app is most used on.
        return torch.float16, True, note

    # cuda
    if amp == Amp.BF16:
        if not torch.cuda.is_bf16_supported():
            return torch.float16, True, "amp bf16 -> fp16: GPU lacks bf16 support (needs Ampere+)"
        return torch.bfloat16, False, None
    return torch.float16, True, None


def resolve_runtime(cfg: TrainConfig) -> ResolvedRuntime:
    """Decide what actually runs, and record every downgrade applied."""
    kind = _pick_device(cfg.schedule.device)
    downgrades: list[str] = []

    if cfg.schedule.device != Device.AUTO and cfg.schedule.device.value != kind:
        downgrades.append(f"device {cfg.schedule.device.value} unavailable; using {kind}")

    amp_dtype, use_scaler, amp_note = resolve_amp_for(cfg.schedule.amp, kind)
    if amp_note:
        downgrades.append(amp_note)

    # ---- channels_last ----------------------------------------------------------------
    channels_last = cfg.input.channels_last
    if channels_last and kind != "cuda":
        channels_last = False
        if cfg.input.channels_last:
            downgrades.append(f"channels_last ignored: no meaningful effect on {kind}")

    # ---- torch.compile ----------------------------------------------------------------
    compile_ = cfg.model.torch_compile
    if compile_ and cfg.schedule.progressive_resizing:
        compile_ = False
        downgrades.append("torch.compile disabled: recompiles on every progressive-resize step")

    return ResolvedRuntime(
        device=torch.device(kind),
        device_str=kind,
        device_name=_accelerator_name(kind),
        amp_dtype=amp_dtype,
        amp_enabled=amp_dtype is not None,
        use_grad_scaler=use_scaler,
        channels_last=channels_last,
        torch_compile=compile_,
        total_memory_gb=_total_memory_gb(kind),
        downgrades=downgrades,
    )


def peak_memory_gb(kind: str) -> float:
    """Peak device memory in GB.

    CUDA has a true resettable high-water mark. MPS does not expose one, so this reports
    current driver allocation instead — an approximation that is only meaningful directly
    after the stage being measured, and never below what earlier stages left cached.
    `reset_peak_memory` empties the MPS cache to keep successive readings comparable.
    """
    if kind == "cuda":
        return torch.cuda.max_memory_allocated() / 1024**3
    if kind == "mps":
        try:
            return torch.mps.driver_allocated_memory() / 1024**3
        except Exception:
            return 0.0
    return 0.0


def reset_peak_memory(kind: str) -> None:
    if kind == "cuda":
        torch.cuda.reset_peak_memory_stats()
    elif kind == "mps":
        # No resettable counter on MPS; releasing cached blocks is the closest equivalent,
        # so a later stage's reading is not dominated by an earlier one's cache.
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
