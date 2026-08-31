"""In-domain self-supervised pretraining: SimSiam, MAE and SimMIM.

SimSiam and MAE are built from `lightly` (heads, losses, masking utilities, view
transforms). SimMIM is not in lightly, so its masking, single-linear head and L1 objective
are implemented here — but still on top of lightly's masked-ViT wrapper for the ViT path,
so only the parts the library does not cover are hand-written.

What this stage does
--------------------
Pretrains the *backbone* on images with no labels, then hands its weights to the
supervised trainer. The pretraining set is:

    train split  +  ssl_extra_data_dir (if set)      -- labels ignored throughout

so no extra data is required: in-domain SSL reuses the images you already labelled and
discards the labels, extracting more signal per image than a cross-entropy objective can
(log2(num_classes) bits per image).

Method choice
-------------
* **SimSiam** (Chen & He, CVPR 2021) works with *any* backbone. Two augmented views, a
  projector, a bottlenecked predictor, and a stop-gradient — no negatives, no memory bank,
  no momentum encoder, so it trains fine at small batch sizes.
* **MAE** (He et al., CVPR 2022) is an *asymmetric* masked autoencoder: masked tokens are
  DROPPED, the encoder runs on the visible ~25%, and a Transformer decoder reconstructs the
  rest. That token-dropping is where its speed comes from and why it needs a plain ViT.
* **SimMIM** (Xie et al., CVPR 2022) keeps every token, replacing masked ones with a
  learnable mask token at the encoder input, and predicts pixels through a single linear
  layer with an L1 loss. Because nothing is dropped it also runs on hierarchical **Swin**
  encoders, which MAE cannot.

Backbone support is decided by isinstance checks, not name heuristics.

Everything here is device-agnostic: the caller passes a resolved runtime and this module
never names a device.
"""

from __future__ import annotations

import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import data as data_mod
from .config import SSLMethod, TrainConfig
from .device import ResolvedRuntime, peak_memory_gb, resolve_amp_for
from .transforms import resolve_norm


@dataclass
class SSLResult:
    method: str
    epochs: int
    final_loss: float
    num_images: int
    wall_time_s: float
    peak_mem_gb: float
    encoder_path: Path | None


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------

def build_ssl_paths(cfg: TrainConfig, train_manifest) -> tuple[list[str], int, int]:
    """train-split images + optional unlabeled directory. Returns (paths, n_train, n_extra)."""
    paths = list(train_manifest.paths)
    n_train = len(paths)
    n_extra = 0
    extra_dir = cfg.experimental.ssl_extra_data_dir
    if extra_dir:
        extra = data_mod.scan_unlabeled(extra_dir)
        # De-duplicate: pointing the extra directory at the train root is an easy mistake
        # and would silently double-count those images.
        seen = set(paths)
        extra = [p for p in extra if p not in seen]
        paths += extra
        n_extra = len(extra)
    return paths, n_train, n_extra


def build_ssl_transform(cfg: TrainConfig):
    """lightly's per-method view transform, using the run's normalization statistics."""
    from lightly.transforms import MAETransform, SimSiamTransform

    size = cfg.effective_ssl_input_size
    mean, std = resolve_norm(cfg)
    normalize = {"mean": list(mean), "std": list(std)}

    if cfg.experimental.ssl_method == SSLMethod.SIMSIAM:
        # SimSiam's view pipeline is the SimCLR one: RRC + flip + colour jitter +
        # grayscale + blur, applied independently to produce two correlated views.
        return SimSiamTransform(input_size=size, normalize=normalize)
    # Both masked methods deliberately use only RRC + flip: the masking IS the pretext
    # task, so heavy photometric augmentation is unnecessary and slightly harmful. MAE and
    # SimMIM specify near-identical pipelines, so one transform serves both.
    return MAETransform(input_size=size, normalize=normalize)


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------

class SimSiamNet(nn.Module):
    """Backbone -> projector -> predictor, with stop-gradient on the target branch."""

    def __init__(self, backbone: nn.Module, feat_dim: int, cfg: TrainConfig):
        super().__init__()
        from lightly.models.modules import SimSiamPredictionHead, SimSiamProjectionHead

        E = cfg.experimental
        self.backbone = backbone
        self.projection_head = SimSiamProjectionHead(
            feat_dim, E.simsiam_proj_hidden_dim, E.simsiam_out_dim)
        self.prediction_head = SimSiamPredictionHead(
            E.simsiam_out_dim, E.simsiam_pred_hidden_dim, E.simsiam_out_dim)

    def forward(self, x: torch.Tensor):
        f = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(f)
        p = self.prediction_head(z)
        return z.detach(), p     # stop-gradient on z is the anti-collapse mechanism


class MAENet(nn.Module):
    """MAE: mask most patches, encode the visible ones, reconstruct the rest."""

    def __init__(self, vit, cfg: TrainConfig):
        super().__init__()
        from lightly.models.modules import MAEDecoderTIMM, MaskedVisionTransformerTIMM

        E = cfg.experimental
        self.vit = vit                       # kept so we can export plain timm weights
        self.mask_ratio = E.mae_mask_ratio
        self.norm_pix_loss = E.mae_norm_pix_loss
        self.patch_size = vit.patch_embed.patch_size[0]
        # The `idx_mask - num_prefix` target indexing below and lightly's wrapper both
        # assume exactly one prefix token; `supported_backbone` gates on this, and the
        # assert makes the assumption loud if construction is reached another way.
        self.num_prefix = getattr(vit, "num_prefix_tokens", 1)
        assert self.num_prefix == 1, (
            f"MAE requires a plain ViT with a single class token; "
            f"got num_prefix_tokens={self.num_prefix}"
        )
        self.backbone = MaskedVisionTransformerTIMM(vit=vit)
        self.sequence_length = self.backbone.sequence_length
        self.decoder = MAEDecoderTIMM(
            num_patches=vit.patch_embed.num_patches,
            patch_size=self.patch_size,
            embed_dim=vit.embed_dim,
            decoder_embed_dim=E.mae_decoder_dim,
            decoder_depth=E.mae_decoder_depth,
            decoder_num_heads=E.mae_decoder_heads,
        )

    def forward(self, images: torch.Tensor):
        from lightly.models import utils as lu

        b = images.shape[0]
        idx_keep, idx_mask = lu.random_token_mask(
            size=(b, self.sequence_length), mask_ratio=self.mask_ratio, device=images.device)

        x_enc = self.backbone.encode(images=images, idx_keep=idx_keep)
        x_dec = self.decoder.embed(x_enc)
        x_masked = lu.repeat_token(self.decoder.mask_token, (b, self.sequence_length))
        x_masked = lu.set_at_index(x_masked, idx_keep, x_dec.type_as(x_masked))
        x_pred = self.decoder.predict(lu.get_at_index(self.decoder.decode(x_masked), idx_mask))

        # Offset by the prefix-token count (asserted == 1 in __init__) to map sequence
        # indices back to image patches — the class token has no corresponding patch.
        patches = lu.patchify(images, self.patch_size)
        target = lu.get_at_index(patches, idx_mask - self.num_prefix)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6) ** 0.5
        return x_pred, target


def _simmim_mask(batch: int, coarse_grid: int, ratio: float, device) -> torch.Tensor:
    """SimMIM's mask generator: uniform random over COARSE units, not model patches.

    Masking whole 32x32 squares rather than individual patches is the paper's key
    detail — with a small masked unit the model inpaints from immediate neighbours and
    learns nothing useful.
    """
    n = coarse_grid * coarse_grid
    k = max(1, int(round(n * ratio)))
    idx = torch.rand(batch, n, device=device).argsort(dim=1)[:, :k]
    mask = torch.zeros(batch, n, device=device)
    mask.scatter_(1, idx, 1.0)
    return mask.view(batch, coarse_grid, coarse_grid), k


def _unpatchify(pred: torch.Tensor, grid: int, patch: int) -> torch.Tensor:
    """(B, grid*grid, patch*patch*3) -> (B, 3, grid*patch, grid*patch)."""
    b = pred.shape[0]
    x = pred.reshape(b, grid, grid, patch, patch, 3)
    x = x.permute(0, 5, 1, 3, 2, 4)                       # b, 3, gh, ph, gw, pw
    return x.reshape(b, 3, grid * patch, grid * patch)


class SimMIMNet(nn.Module):
    """SimMIM: keep every token, replace masked ones with a learnable token, predict pixels.

    Differs from MAE in all four of its defining choices:
      * the encoder sees the FULL token set (masked positions carry a mask token) rather
        than only the visible subset,
      * the decoder is a SINGLE linear layer, not a Transformer stack,
      * the loss is L1 on raw pixels, not MSE on normalised ones,
      * masking happens in large pixel-space units rather than per patch.

    Keeping every token is also what lets it run on hierarchical Swin encoders, which
    MAE's token-dropping design cannot support.
    """

    def __init__(self, backbone: nn.Module, cfg: TrainConfig):
        super().__init__()
        from timm.models.swin_transformer import SwinTransformer
        from timm.models.vision_transformer import VisionTransformer

        E = cfg.experimental
        self.img_size = cfg.effective_ssl_input_size
        self.mask_ratio = E.simmim_mask_ratio
        self.mask_unit = E.simmim_mask_patch_size
        self.coarse_grid = self.img_size // self.mask_unit
        self.encoder_model = backbone            # plain timm model, for weight export

        if isinstance(backbone, VisionTransformer):
            from lightly.models.modules import MaskedVisionTransformerTIMM

            self.kind = "vit"
            self.embed_stride = backbone.patch_embed.patch_size[0]
            self.out_stride = self.embed_stride          # ViT keeps resolution throughout
            self.num_prefix = backbone.num_prefix_tokens
            # lightly's wrapper assumes a single class token; register-token/GAP ViT
            # variants crash inside its pos-embed initialisation. Gate here as well as
            # in supported_backbone so no path reaches the confusing lightly error.
            if self.num_prefix != 1:
                raise ValueError(
                    f"ssl_method='simmim' on a ViT requires a plain ViT with a single "
                    f"class token, but '{cfg.model.backbone}' has "
                    f"{self.num_prefix} prefix tokens (register tokens / GAP). Use a "
                    f"plain vit_*/deit_* backbone, or ssl_method='simsiam'."
                )
            self.encoder = MaskedVisionTransformerTIMM(vit=backbone)
            out_dim = backbone.embed_dim
        elif isinstance(backbone, SwinTransformer):
            self.kind = "swin"
            self.embed_stride = backbone.patch_embed.patch_size[0]   # 4
            self.encoder = backbone
            with torch.no_grad():
                probe = backbone.patch_embed(torch.zeros(1, 3, self.img_size, self.img_size))
                feat = backbone.norm(backbone.layers(probe))
            # Swin downsamples, so predictions are made at the FINAL stride (32), which is
            # what the SimMIM paper does for hierarchical encoders.
            self.out_stride = self.img_size // feat.shape[1]
            out_dim = feat.shape[-1]
            self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, backbone.embed_dim))
            nn.init.trunc_normal_(self.mask_token, std=0.02)
        else:
            raise ValueError(
                f"ssl_method='simmim' needs a patch-based encoder (ViT or Swin), but "
                f"backbone '{cfg.model.backbone}' builds a {type(backbone).__name__}. "
                f"Masked modelling on plain convnets requires sparse convolutions "
                f"(SparK / ConvNeXt-V2 FCMAE), which this app does not implement. "
                f"Use ssl_method='simsiam' for a convnet."
            )

        if self.mask_unit % self.embed_stride != 0:
            raise ValueError(
                f"simmim_mask_patch_size ({self.mask_unit}) must be a multiple of the "
                f"encoder's patch stride ({self.embed_stride})."
            )
        self.embed_grid = self.img_size // self.embed_stride
        self.out_grid = self.img_size // self.out_stride

        # SimMIM's "decoder" is one linear layer — the paper's point is that a heavier
        # head lets the decoder do the work the encoder was supposed to learn.
        self.head = nn.Linear(out_dim, self.out_stride ** 2 * 3)

    def forward(self, images: torch.Tensor):
        b = images.shape[0]
        coarse, k = _simmim_mask(b, self.coarse_grid, self.mask_ratio, images.device)

        scale = self.mask_unit // self.embed_stride
        enc_mask = coarse.repeat_interleave(scale, 1).repeat_interleave(scale, 2)

        if self.kind == "vit":
            n_masked = k * scale * scale
            idx_mask = (enc_mask.flatten(1).argsort(dim=1, descending=True)[:, :n_masked]
                        + self.num_prefix)
            tokens = self.encoder.encode(images=images, idx_mask=idx_mask)
            tokens = tokens[:, self.num_prefix:]                  # drop class token
            pred = self.head(tokens)
        else:
            t = self.encoder.patch_embed(images)                  # (B, H', W', C)
            m = enc_mask.unsqueeze(-1).type_as(t)
            t = t * (1 - m) + self.mask_token.type_as(t) * m
            feat = self.encoder.norm(self.encoder.layers(t))      # (B, h, w, C)
            pred = self.head(feat).flatten(1, 2)

        recon = _unpatchify(pred, self.out_grid, self.out_stride)
        pixel_mask = (coarse.repeat_interleave(self.mask_unit, 1)
                            .repeat_interleave(self.mask_unit, 2).unsqueeze(1))
        return recon, images, pixel_mask


class SimMIMLoss(nn.Module):
    """L1 on masked pixels only, normalised by the number of masked values."""

    def forward(self, recon: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        diff = (recon - target).abs() * mask
        return diff.sum() / (mask.sum() * target.shape[1] + 1e-5)


def _build_backbone(cfg: TrainConfig):
    """Feature-extractor form of the configured backbone (no classifier)."""
    import timm

    m = cfg.model
    kwargs = dict(pretrained=m.pretrained, num_classes=0)
    if m.pretrained and m.pretrained_source not in ("", "default"):
        kwargs["pretrained_cfg_overlay"] = dict(tag=m.pretrained_source)
    if cfg.experimental.ssl_method in (SSLMethod.MAE, SSLMethod.SIMMIM):
        # Masked methods drive patch_embed/pos_embed at a fixed grid, so img_size must
        # match the SSL resolution rather than the architecture's pretrained default.
        kwargs["img_size"] = cfg.effective_ssl_input_size
    try:
        return timm.create_model(m.backbone, **kwargs)
    except TypeError:
        kwargs.pop("img_size", None)
        return timm.create_model(m.backbone, **kwargs)


def supported_backbone(cfg: TrainConfig) -> tuple[bool, str]:
    """Can the configured SSL method actually run on the configured backbone?

    The authority on this question — `config.py`'s name-prefix tuples only exist so the
    schema can warn without importing timm, and are asserted against this function by the
    test suite. Constructing the architecture without pretrained weights is cheap and is
    the only way to know which timm class a name resolves to.
    """
    import timm
    from timm.models.swin_transformer import SwinTransformer
    from timm.models.vision_transformer import VisionTransformer

    method = cfg.experimental.ssl_method
    if method == SSLMethod.NONE or method == SSLMethod.SIMSIAM:
        return True, ""   # SimSiam is architecture-agnostic

    try:
        probe = timm.create_model(cfg.model.backbone, pretrained=False, num_classes=0)
    except Exception as e:
        return False, f"backbone '{cfg.model.backbone}' could not be built ({e})"

    cls = type(probe).__name__

    def _plain_vit(p) -> tuple[bool, str]:
        # isinstance is necessary but NOT sufficient: register-token and GAP ViT
        # variants (vit_*_reg4_*, vit_*_gap_*) subclass VisionTransformer yet crash
        # inside lightly's masked wrapper (sine-cosine pos-embed init assumes exactly
        # one class token), and the pixel-target indexing assumes one prefix token.
        # Passing them through here produced a clean validation panel and a crash at
        # model construction — the exact failure this gate exists to prevent.
        n = getattr(p, "num_prefix_tokens", 1)
        if n != 1:
            return False, (
                f"'{cfg.model.backbone}' is a ViT variant with {n} prefix tokens "
                f"(register tokens / GAP); the masked-ViT wrapper only supports plain "
                f"ViTs with a single class token. Use a plain vit_*/deit_* backbone, "
                f"or ssl_method='simsiam', which works with any architecture."
            )
        return True, ""

    if method == SSLMethod.MAE:
        if isinstance(probe, VisionTransformer):
            return _plain_vit(probe)
        return False, (
            f"ssl_method='mae' drops masked tokens before the encoder, so it needs a plain "
            f"ViT token sequence, but '{cfg.model.backbone}' builds a {cls}. Use a "
            f"vit_*/deit_*/flexivit_* backbone; or ssl_method='simmim', which keeps every "
            f"token and also supports Swin; or ssl_method='simsiam', which works with any "
            f"architecture."
        )
    if isinstance(probe, VisionTransformer):
        return _plain_vit(probe)
    if isinstance(probe, SwinTransformer):
        return True, ""
    return False, (
        f"ssl_method='simmim' needs a patch-based encoder (timm VisionTransformer or "
        f"SwinTransformer), but '{cfg.model.backbone}' builds a {cls}. Note that "
        f"swinv2_* builds SwinTransformerV2, which does not share SwinTransformer's "
        f"internal structure and is not supported. Use ssl_method='simsiam' instead."
    )


def check_backbone(cfg: TrainConfig) -> None:
    """Raise before any expensive setup if the SSL method cannot run here."""
    ok, why = supported_backbone(cfg)
    if not ok:
        raise ValueError(why)


def build_ssl_model(cfg: TrainConfig) -> nn.Module:
    backbone = _build_backbone(cfg)
    method = cfg.experimental.ssl_method

    if method == SSLMethod.MAE:
        from timm.models.vision_transformer import VisionTransformer

        if not isinstance(backbone, VisionTransformer):
            raise ValueError(
                f"ssl_method='mae' drops masked tokens before the encoder, so it needs a "
                f"plain ViT token sequence, but backbone '{cfg.model.backbone}' builds a "
                f"{type(backbone).__name__}. Use a plain vit_*/deit_* backbone (eva_* "
                f"and beit_* build different timm classes); or ssl_method='simmim', "
                f"which keeps every token and also supports Swin; or "
                f"ssl_method='simsiam', which works with any architecture."
            )
        return MAENet(backbone, cfg)

    if method == SSLMethod.SIMMIM:
        return SimMIMNet(backbone, cfg)   # raises with guidance on unsupported backbones

    feat_dim = getattr(backbone, "num_features", None)
    if not feat_dim:
        with torch.no_grad():
            size = cfg.effective_ssl_input_size
            feat_dim = backbone(torch.zeros(1, 3, size, size)).flatten(1).shape[1]
    return SimSiamNet(backbone, feat_dim, cfg)


def encoder_state_dict(model: nn.Module) -> dict:
    """Plain timm-compatible weights, with SSL-only heads/decoders dropped."""
    if isinstance(model, MAENet):
        inner = model.vit
    elif isinstance(model, SimMIMNet):
        inner = model.encoder_model
    else:
        inner = model.backbone
    return {k: v.detach().cpu().clone() for k, v in inner.state_dict().items()}


# --------------------------------------------------------------------------------------
# Optimizer / schedule
# --------------------------------------------------------------------------------------

def _build_optimizer(model: nn.Module, cfg: TrainConfig):
    lr, wd = cfg.resolved_ssl_lr(), cfg.resolved_ssl_weight_decay()

    if cfg.experimental.ssl_method == SSLMethod.SIMSIAM:
        params = list(model.parameters())
        if cfg.experimental.simsiam_fix_pred_lr:
            # The predictor is held at a constant LR while everything else decays —
            # the SimSiam paper reports this helps consistently and never hurts.
            pred_ids = {id(p) for p in model.prediction_head.parameters()}
            rest = [p for p in params if id(p) not in pred_ids]
            groups = [
                {"params": rest, "fix_lr": False},
                {"params": list(model.prediction_head.parameters()), "fix_lr": True},
            ]
        else:
            groups = [{"params": params, "fix_lr": False}]
        return torch.optim.SGD(groups, lr=lr, momentum=0.9, weight_decay=wd)

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or name.endswith(".bias") else decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": wd, "fix_lr": False},
         {"params": no_decay, "weight_decay": 0.0, "fix_lr": False}],
        lr=lr, betas=(0.9, 0.95))


def _lr_at(cfg: TrainConfig, epoch_frac: float) -> float:
    """Linear warmup then cosine decay, in epoch units."""
    E = cfg.experimental
    base = cfg.resolved_ssl_lr()
    if E.ssl_lr_warmup_epochs > 0 and epoch_frac < E.ssl_lr_warmup_epochs:
        return base * (epoch_frac + 1e-8) / E.ssl_lr_warmup_epochs
    span = max(1e-8, E.ssl_epochs - E.ssl_lr_warmup_epochs)
    progress = min(1.0, (epoch_frac - E.ssl_lr_warmup_epochs) / span)
    return base * 0.5 * (1 + math.cos(math.pi * progress))


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------

def pretrain(cfg: TrainConfig, rt: ResolvedRuntime, train_manifest, out_dir: Path,
             tracker=None, log=print, emit=None) -> tuple[dict, SSLResult]:
    """Run the SSL stage. Returns (encoder_state_dict, result)."""
    from lightly.loss import NegativeCosineSimilarity

    E = cfg.experimental
    method = E.ssl_method.value

    paths, n_train, n_extra = build_ssl_paths(cfg, train_manifest)
    dataset = data_mod.UnlabeledDataset(paths, build_ssl_transform(cfg), cfg.data.cache_mode,
                                        broken_log=out_dir / "broken_images.tsv")
    # Same rule as the supervised loader: drop_last keeps batch statistics stable,
    # but on a dataset smaller than one batch it drops EVERY batch — the loop body
    # never runs, "final loss 0.0000" is reported, and an untrained encoder is saved
    # as if pretrained. Below one full batch, keep the short batch instead.
    loader = DataLoader(
        dataset, batch_size=E.ssl_batch_size, shuffle=True,
        drop_last=len(dataset) >= E.ssl_batch_size,
        collate_fn=data_mod.collate_skip_broken,
        **data_mod.loader_kwargs(cfg, cfg.effective_ssl_num_workers()),
    )

    model = build_ssl_model(cfg).to(rt.device)
    if rt.channels_last and isinstance(model, SimSiamNet):
        model = model.to(memory_format=torch.channels_last)

    optimizer = _build_optimizer(model, cfg)
    # The SSL stage has its own AMP setting, resolved against this device exactly like
    # the supervised one.
    amp_dtype, use_scaler, amp_note = resolve_amp_for(E.ssl_amp, rt.device_str)
    # `device=` matters: without it GradScaler assumes CUDA and quietly disables itself.
    scaler = torch.amp.GradScaler(device=rt.device_str, enabled=use_scaler)
    criterion = {
        SSLMethod.SIMSIAM: NegativeCosineSimilarity,
        SSLMethod.MAE: nn.MSELoss,          # MAE regresses (optionally normalised) pixels
        SSLMethod.SIMMIM: SimMIMLoss,       # SimMIM uses L1 on masked pixels
    }[E.ssl_method]()

    n_params = sum(p.numel() for p in model.parameters())
    log(f"  SSL [{method}]: {len(paths):,} images "
        f"({n_train:,} from train split + {n_extra:,} unlabeled) "
        f"@ {cfg.effective_ssl_input_size}px, batch {E.ssl_batch_size}")
    log(f"  SSL model: {n_params/1e6:.1f}M params, lr {cfg.resolved_ssl_lr():.2e}, "
        f"wd {cfg.resolved_ssl_weight_decay():g}"
        + (f", amp {str(amp_dtype).replace('torch.', '')}" if amp_dtype else ", amp off"))
    if amp_note:
        log(f"  ! {amp_note}")

    steps_per_epoch = max(1, len(loader))
    t0 = time.time()
    final_loss = float("nan")

    for epoch in range(E.ssl_epochs):
        model.train()
        running, seen, steps_taken = 0.0, 0, 0
        for step, batch in enumerate(loader):
            if batch is None:      # every image in this batch was unreadable
                continue
            views, _ = batch
            steps_taken += 1
            lr = _lr_at(cfg, epoch + step / steps_per_epoch)
            for g in optimizer.param_groups:
                # `fix_lr` groups (SimSiam's predictor) hold the base LR.
                g["lr"] = cfg.resolved_ssl_lr() if g.get("fix_lr") else lr

            ctx = (torch.autocast(device_type=rt.device_str, dtype=amp_dtype)
                   if amp_dtype else nullcontext())

            if E.ssl_method == SSLMethod.SIMSIAM:
                x0, x1 = views[0].to(rt.device, non_blocking=True), views[1].to(rt.device, non_blocking=True)
                if rt.channels_last:
                    x0, x1 = x0.to(memory_format=torch.channels_last), x1.to(memory_format=torch.channels_last)
                with ctx:
                    z0, p0 = model(x0)
                    z1, p1 = model(x1)
                    loss = 0.5 * (criterion(z0, p1) + criterion(z1, p0))
                bs = x0.size(0)
            else:
                x = (views[0] if isinstance(views, (list, tuple)) else views).to(
                    rt.device, non_blocking=True)
                with ctx:
                    out = model(x)
                    # MAE returns (prediction, target); SimMIM adds the pixel mask, since
                    # its loss is restricted to the masked region.
                    loss = criterion(*out)
                bs = x.size(0)

            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running += loss.item() * bs
            seen += bs
            if step % 50 == 0 and emit:
                emit("ssl_progress", epoch=epoch, step=step, total_steps=steps_per_epoch,
                     loss=running / max(1, seen), lr=lr, method=method)

        # Mirror of the supervised zero-step guard: an SSL epoch that took no
        # optimizer step "pretrained" nothing, yet would report a plausible-looking
        # final loss of 0.0 and hand a random encoder to the supervised stage.
        if steps_taken == 0:
            raise RuntimeError(
                f"SSL epoch {epoch} produced no batches "
                f"({len(dataset)} images, ssl_batch_size={E.ssl_batch_size}) — "
                f"the encoder would be saved untrained. Lower ssl_batch_size or add "
                f"images (ssl_extra_data_dir)."
            )

        final_loss = running / max(1, seen)
        if tracker is not None:
            tracker.log_metrics({"ssl/loss": final_loss, "ssl/lr": lr}, step=epoch)
        if epoch % max(1, E.ssl_epochs // 20) == 0 or epoch == E.ssl_epochs - 1:
            log(f"  ssl epoch {epoch+1}/{E.ssl_epochs}  loss {final_loss:.4f}  lr {lr:.2e}")
        if emit:
            emit("ssl_epoch", epoch=epoch, total=E.ssl_epochs, loss=final_loss, method=method)

    state = encoder_state_dict(model)

    encoder_path = None
    if E.ssl_save_encoder:
        encoder_path = out_dir / f"ssl_{method}_encoder.pt"
        torch.save({
            "backbone": cfg.model.backbone,
            "ssl_method": method,
            "ssl_epochs": E.ssl_epochs,
            "input_size": cfg.effective_ssl_input_size,
            "num_images": len(paths),
            "state_dict": state,
        }, encoder_path)
        if tracker is not None:
            tracker.log_artifact(encoder_path, "ssl")

    result = SSLResult(
        method=method, epochs=E.ssl_epochs, final_loss=final_loss, num_images=len(paths),
        wall_time_s=time.time() - t0, peak_mem_gb=peak_memory_gb(rt.device_str),
        encoder_path=encoder_path,
    )
    log(f"  SSL done in {result.wall_time_s/60:.1f} min, final loss {final_loss:.4f}"
        + (f", peak {result.peak_mem_gb:.1f} GB" if result.peak_mem_gb else ""))
    return state, result


def load_encoder_into(model: nn.Module, state: dict, log=print,
                      min_loaded_fraction: float = 0.5) -> tuple[int, int]:
    """Copy SSL-pretrained weights into the supervised model, keeping its classifier.

    Raises if almost nothing transferred. `strict=False` means a wholesale key mismatch
    loads zero tensors and returns quietly, so hours of pretraining could be discarded
    with only a log line to show for it — and the supervised run would proceed from the
    ImageNet initialisation while the tracker recorded it as an SSL run.
    """
    tgt = model.state_dict()
    loadable = {k: v for k, v in state.items()
                if k in tgt and tgt[k].shape == v.shape}
    skipped = len(state) - len(loadable)
    missing = [k for k in tgt if k not in loadable]
    fraction = len(loadable) / max(1, len(state))
    if fraction < min_loaded_fraction:
        raise RuntimeError(
            f"SSL pretraining produced {len(state)} tensors but only {len(loadable)} "
            f"({fraction:.0%}) matched the supervised model by name and shape. The "
            f"pretrained encoder would be almost entirely discarded. This usually means "
            f"the SSL stage and the supervised stage built different architectures — "
            f"check that model.backbone and input sizes agree across both."
        )
    model.load_state_dict(loadable, strict=False)
    log(f"  loaded {len(loadable)} SSL tensors into the supervised model "
        f"({skipped} shape/name mismatches, {len(missing)} left at init — "
        f"the classifier head is expected here)")
    return len(loadable), skipped
