"""Backbone construction via timm, with a torchvision fallback."""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import FreezePolicy, TrainConfig


def create_model(cfg: TrainConfig, num_classes: int,
                 pretrained_override: bool | None = None) -> nn.Module:
    m = cfg.model
    # `pretrained_override=False` skips the timm/HF weight download when the caller is
    # about to overwrite every tensor anyway (model.pretrained_checkpoint) — the config's
    # own `pretrained` flag is untouched, so tags and the recorded run stay honest.
    pretrained = m.pretrained if pretrained_override is None else pretrained_override
    try:
        import timm
    except ImportError:
        timm = None

    if timm is not None and m.backbone in timm.list_models():
        kwargs = dict(
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=m.drop_rate,
        )
        # Not every architecture supports these; probe rather than maintain a whitelist.
        for key, val in (("drop_path_rate", m.drop_path_rate), ("global_pool", m.global_pool)):
            if val not in (None, ""):
                kwargs[key] = val
        if pretrained and m.pretrained_source not in ("", "default"):
            kwargs["pretrained_cfg_overlay"] = dict(tag=m.pretrained_source)

        # Patch-based models bake the input resolution into patch_embed and pos_embed, so
        # a ViT at any resolution other than its pretrained one must be told the size (the
        # position embeddings get interpolated). Convnets ignore img_size, hence the probe.
        size = cfg.input.input_size
        try:
            model = timm.create_model(m.backbone, img_size=size, **kwargs)
        except TypeError:
            # This architecture takes no img_size (convnets); fall back cleanly.
            try:
                model = timm.create_model(m.backbone, **kwargs)
            except TypeError:
                kwargs.pop("drop_path_rate", None)
                kwargs.pop("global_pool", None)
                model = timm.create_model(m.backbone, **kwargs)
        except ValueError as e:
            # The architecture DOES accept img_size and rejected this value — e.g. a
            # resolution that is not a multiple of the patch size or window size.
            # Silently retrying without it built the model at its native resolution
            # while the loaders kept feeding `input_size`, which is either a late shape
            # error or a quietly mis-scaled position embedding.
            raise ValueError(
                f"backbone '{m.backbone}' cannot be built at input_size={size}: {e}. "
                f"Patch- and window-based models only accept resolutions compatible "
                f"with their patch/window geometry — try the architecture's native size."
            ) from e
    else:
        model = _torchvision_model(m.backbone, pretrained, num_classes)

    if m.head_init_scale != 1.0:
        _scale_head(model, m.head_init_scale)
    if m.gradient_checkpointing and hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(True)
    return model


def _torchvision_model(name: str, pretrained: bool, num_classes: int) -> nn.Module:
    from torchvision import models as tvm

    if not hasattr(tvm, name):
        raise ValueError(
            f"Backbone '{name}' not found in timm or torchvision. "
            f"Use the backbone search in the UI, or `timm.list_models(pretrained=True)`."
        )
    model = getattr(tvm, name)(weights="DEFAULT" if pretrained else None)
    # Retarget whichever head convention this architecture uses.
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, "classifier"):
        c = model.classifier
        if isinstance(c, nn.Linear):
            model.classifier = nn.Linear(c.in_features, num_classes)
        else:
            for i in range(len(c) - 1, -1, -1):
                if isinstance(c[i], nn.Linear):
                    c[i] = nn.Linear(c[i].in_features, num_classes)
                    break
    elif hasattr(model, "head") and isinstance(model.head, nn.Linear):
        model.head = nn.Linear(model.head.in_features, num_classes)
    return model


def get_classifier(model: nn.Module) -> nn.Module | None:
    if hasattr(model, "get_classifier"):
        try:
            return model.get_classifier()
        except Exception:
            pass
    for attr in ("fc", "head", "classifier"):
        mod = getattr(model, attr, None)
        if isinstance(mod, nn.Linear):
            return mod
    return None


def classifier_param_names(model: nn.Module) -> set[str]:
    """Fully-qualified names of the classifier head's params and buffers.

    Needed to keep a head OUT of a weight transfer: name+shape matching alone would
    happily load a head trained on a same-sized but different class list, and every
    prediction would be plausibly, silently wrong.
    """
    head = get_classifier(model)
    if head is None:
        return set()
    for mod_name, mod in model.named_modules():
        if mod is head and mod_name:
            prefix = mod_name + "."
            return {n for n, _ in model.named_parameters() if n.startswith(prefix)} | \
                   {n for n, _ in model.named_buffers() if n.startswith(prefix)}
    return set()


def load_state_into(model: nn.Module, state: dict, *, source: str, log=print,
                    min_loaded_fraction: float = 0.5,
                    min_covered_fraction: float = 0.85) -> tuple[int, int]:
    """Copy `state` into `model` wherever name and shape agree; report the rest.

    The generic twin of `ssl.load_encoder_into`, for weights arriving from a checkpoint
    rather than an SSL stage. Tolerant on purpose — a changed head or an extra buffer
    must not block a backbone transfer — but not silently so, and in BOTH directions:

    * `min_loaded_fraction` of the incoming tensors must land, or the file describes a
      different architecture;
    * `min_covered_fraction` of the model's tensors must be filled, or the model is a
      different (larger) architecture than the file. This direction is the one a
      same-family upgrade slips through — every resnet18 tensor matches resnet34 by name
      and shape, and without the coverage check the "warm-started" model was 44% random
      init.
    """
    tgt = model.state_dict()
    loadable = {k: v for k, v in state.items()
                if k in tgt and tgt[k].shape == v.shape}
    skipped = len(state) - len(loadable)
    missing = [k for k in tgt if k not in loadable]
    fraction = len(loadable) / max(1, len(state))
    covered = len(loadable) / max(1, len(tgt))
    if fraction < min_loaded_fraction:
        raise RuntimeError(
            f"{source} holds {len(state)} tensors but only {len(loadable)} "
            f"({fraction:.0%}) match this model by name and shape — it looks like a "
            f"different architecture. Nothing was loaded."
        )
    if covered < min_covered_fraction:
        raise RuntimeError(
            f"{source} fills only {len(loadable)} of this model's {len(tgt)} tensors "
            f"({covered:.0%}) — the model is larger than what the checkpoint describes, "
            f"and the other {len(missing)} tensors would train from random init while "
            f"the run records itself as warm-started. Nothing was loaded."
        )
    model.load_state_dict(loadable, strict=False)
    log(f"  loaded {len(loadable)} tensors from {source} "
        f"({skipped} name/shape mismatches, {len(missing)} left at init)")
    return len(loadable), skipped


@torch.no_grad()
def _scale_head(model: nn.Module, scale: float) -> None:
    """Shrink the freshly-initialised head.

    A random head produces large early gradients that can damage good pretrained
    features; ConvNeXt's fine-tuning recipe uses scale=0.001 for exactly this reason.
    """
    head = get_classifier(model)
    if isinstance(head, nn.Linear):
        head.weight.mul_(scale)
        if head.bias is not None:
            head.bias.mul_(scale)


def apply_freeze(model: nn.Module, cfg: TrainConfig, epoch: int) -> bool:
    """Apply the freeze schedule for `epoch`. Returns True if anything changed."""
    m = cfg.model
    if m.freeze_policy == FreezePolicy.NONE:
        return False

    head = get_classifier(model)
    head_params = set(id(p) for p in head.parameters()) if head is not None else set()

    if m.freeze_policy == FreezePolicy.BACKBONE_FIRST_N:
        frozen = epoch < m.freeze_epochs
        changed = False
        for p in model.parameters():
            want = True if id(p) in head_params else not frozen
            if p.requires_grad != want:
                p.requires_grad_(want)
                changed = True
        return changed

    if m.freeze_policy == FreezePolicy.FIRST_K_STAGES and epoch == 0:
        stages = _stage_modules(model)
        for stage in stages[: m.freeze_stages]:
            for p in stage.parameters():
                p.requires_grad_(False)
        return True
    return False


def _stage_modules(model: nn.Module) -> list[nn.Module]:
    """Best-effort ordered list of backbone stages across timm architectures."""
    for attr in ("stages", "layers", "blocks"):
        seq = getattr(model, attr, None)
        if isinstance(seq, (nn.Sequential, nn.ModuleList)):
            return list(seq)
    return [m for _, m in model.named_children()]


def freeze_bn(model: nn.Module) -> None:
    """Freeze BatchNorm running stats and affine params (helps at batch < ~16)."""
    for mod in model.modules():
        if isinstance(mod, nn.modules.batchnorm._BatchNorm):
            mod.eval()
            for p in mod.parameters():
                p.requires_grad_(False)


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
