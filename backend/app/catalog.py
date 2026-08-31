"""timm/torchvision backbone catalogue.

Listing is cheap (name + pretrained tags + native input size come from timm's config
registry). Parameter count requires instantiating the architecture, so it is computed
lazily per model and cached.
"""

from __future__ import annotations

import functools
import re


@functools.lru_cache(maxsize=1)
def _catalog() -> list[dict]:
    import timm

    pretrained = set()
    for tag in timm.list_pretrained():
        pretrained.add(tag.split(".")[0])

    out: list[dict] = []
    for name in timm.list_models():
        cfg = {}
        try:
            pc = timm.get_pretrained_cfg(name, allow_unregistered=True)
            if pc is not None:
                cfg = {
                    "input_size": (pc.input_size or (3, 224, 224))[-1],
                    "crop_pct": pc.crop_pct,
                    "interpolation": pc.interpolation,
                }
        except Exception:
            pass
        out.append({
            "name": name,
            "family": re.split(r"[_\d]", name, maxsplit=1)[0],
            "pretrained": name in pretrained,
            "input_size": cfg.get("input_size", 224),
            "crop_pct": cfg.get("crop_pct"),
            "interpolation": cfg.get("interpolation"),
            "source": "timm",
        })
    out.sort(key=lambda m: (not m["pretrained"], m["name"]))
    return out


#: name -> parameter count, filled in on demand by `param_count`.
_PARAM_CACHE: dict[str, int | None] = {}


def param_count(name: str) -> int | None:
    """Instantiate once (no weights) to count parameters. Cached; ~0.1-2s per model."""
    if name in _PARAM_CACHE:
        return _PARAM_CACHE[name]
    try:
        import timm

        m = timm.create_model(name, pretrained=False, num_classes=1)
        n = sum(p.numel() for p in m.parameters())
    except Exception:
        n = None
    _PARAM_CACHE[name] = n
    return n


@functools.lru_cache(maxsize=256)
def pretrained_tags(name: str) -> list[str]:
    try:
        import timm

        tags = [t.split(".", 1)[1] for t in timm.list_pretrained(name + ".*") if "." in t]
        return ["default"] + sorted(set(tags))
    except Exception:
        return ["default"]


def search(query: str = "", *, only_pretrained: bool = False,
           family: str | None = None, limit: int = 60) -> dict:
    q = query.strip().lower()
    items = _catalog()
    if only_pretrained:
        items = [m for m in items if m["pretrained"]]
    if family:
        items = [m for m in items if m["family"] == family]
    if q:
        exact = [m for m in items if m["name"] == q]
        prefix = [m for m in items if m["name"].startswith(q) and m["name"] != q]
        sub = [m for m in items if q in m["name"] and not m["name"].startswith(q)]
        items = exact + prefix + sub
    total = len(items)
    # Parameter count needs a real instantiation, which is far too slow to do for a whole
    # result page. Serve whatever is already memoised and let the client pull
    # /backbones/{name} for the focused row.
    page = [{**m, "params": _PARAM_CACHE.get(m["name"])} for m in items[:limit]]
    return {"total": total, "items": page}


def families(limit: int = 60) -> list[str]:
    seen: dict[str, int] = {}
    for m in _catalog():
        seen[m["family"]] = seen.get(m["family"], 0) + 1
    return [f for f, _ in sorted(seen.items(), key=lambda kv: -kv[1])[:limit]]


def detail(name: str) -> dict:
    for m in _catalog():
        if m["name"] == name:
            return {**m, "params": param_count(name), "tags": pretrained_tags(name)}
    return {"name": name, "found": False}


#: Rough per-backbone activation-memory coefficients (GB per sample at 224px, fp32),
#: measured once on reference hardware. Used only for the OOM heuristic in the UI, which
#: is labelled as an estimate rather than a promise.
_MEM_COEFF = {
    "resnet18": 0.014, "resnet50": 0.043, "convnext_tiny": 0.052, "convnext_small": 0.078,
    "convnext_base": 0.110, "vit_small_patch16_224": 0.038, "vit_base_patch16_224": 0.082,
    "swin_tiny_patch4_window7_224": 0.060, "efficientnet_b0": 0.030,
}
_DEFAULT_COEFF = 0.06


def estimate_memory_gb(backbone: str, batch_size: int, input_size: int, *,
                       amp: bool = True, grad_checkpointing: bool = False,
                       params: int | None = None) -> dict:
    coeff = _MEM_COEFF.get(backbone, _DEFAULT_COEFF)
    scale = (input_size / 224) ** 2
    activations = coeff * batch_size * scale
    if amp:
        activations *= 0.55
    if grad_checkpointing:
        activations *= 0.6

    p = params if params is not None else (param_count(backbone) or 25_000_000)
    # weights + grads + 2 Adam moments, in fp32
    weights = p * 4 * 4 / 1024**3
    return {
        "activations_gb": round(activations, 2),
        "weights_gb": round(weights, 2),
        "total_gb": round(activations + weights, 2),
        "heuristic": True,
    }
