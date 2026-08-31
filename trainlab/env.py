"""Reproducibility context logged with every run."""

from __future__ import annotations

import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

_TRACKED = ["torch", "torchvision", "timm", "torchmetrics", "numpy", "pillow",
            "mlflow", "optuna", "pydantic"]


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=Path(__file__).resolve().parent.parent,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def git_info() -> dict[str, str]:
    commit = _git("rev-parse", "HEAD")
    if not commit:
        return {"git_commit": "unknown", "git_dirty": "unknown"}
    status = _git("status", "--porcelain")
    return {
        "git_commit": commit[:12],
        "git_dirty": str(bool(status)).lower(),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
    }


def library_versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for pkg in _TRACKED:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            continue
    return out


def hardware_info(runtime=None) -> dict[str, str]:
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_executable": sys.executable,
    }
    if runtime is not None:
        info.update({
            "device": runtime.device_str,
            "accelerator": runtime.device_name,
            "amp_dtype": str(runtime.amp_dtype).replace("torch.", "") if runtime.amp_dtype else "off",
            "channels_last": str(runtime.channels_last).lower(),
            "torch_compile": str(runtime.torch_compile).lower(),
        })
        if runtime.total_memory_gb:
            info["device_memory_gb"] = f"{runtime.total_memory_gb:.1f}"
    return info


def full_context(runtime=None) -> dict[str, str]:
    ctx: dict[str, str] = {}
    ctx.update({f"env.{k}": v for k, v in git_info().items()})
    ctx.update({f"lib.{k}": v for k, v in library_versions().items()})
    ctx.update({f"hw.{k}": v for k, v in hardware_info(runtime).items()})
    return ctx
