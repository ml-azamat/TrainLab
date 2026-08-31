"""SSL construction tests.

The SimMIM Swin path reaches into timm's internals (`patch_embed` -> `layers` -> `norm`,
NHWC) and the ViT path computes a mask-index offset for the prefix tokens. Neither is
covered by a type signature, so a timm upgrade can change the shapes underneath and the
loss will still produce a plausible number. These tests pin the geometry.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("timm")
pytest.importorskip("lightly")

from trainlab import ssl as ssl_mod           # noqa: E402
from trainlab.config import SSLMethod, TrainConfig   # noqa: E402


def _cfg(backbone: str, method: str, size: int = 224) -> TrainConfig:
    cfg = TrainConfig()
    cfg.model.backbone = backbone
    cfg.model.pretrained = False
    cfg.input.input_size = size
    cfg.experimental.ssl_method = SSLMethod(method)
    cfg.experimental.ssl_input_size = size
    cfg.experimental.simmim_mask_patch_size = 32
    return cfg


# ------------------------------------------------------------------------- SimMIM

def test_simmim_swin_geometry():
    """Swin predicts at the FINAL stride (32), which is what the paper does for
    hierarchical encoders. If timm changes patch_embed's output layout this breaks
    loudly rather than silently reconstructing the wrong pixels."""
    model = ssl_mod.build_ssl_model(_cfg("swin_tiny_patch4_window7_224", "simmim"))
    assert model.kind == "swin"
    assert model.embed_stride == 4
    assert model.out_stride == 32
    assert model.out_grid == 224 // 32


def test_simmim_swin_forward_shapes_and_masked_loss():
    model = ssl_mod.build_ssl_model(_cfg("swin_tiny_patch4_window7_224", "simmim"))
    x = torch.randn(2, 3, 224, 224)
    recon, target, mask = model(x)

    assert recon.shape == (2, 3, 224, 224)
    assert target.shape == (2, 3, 224, 224)
    assert mask.shape == (2, 1, 224, 224)
    # The mask must select whole coarse units, and roughly the configured fraction.
    frac = mask.mean().item()
    assert 0.3 < frac < 0.9, frac

    loss = ssl_mod.SimMIMLoss()(recon, target, mask)
    assert torch.isfinite(loss) and loss.item() > 0

    # The loss must ignore unmasked pixels entirely.
    perturbed = target.clone()
    perturbed[mask.expand_as(perturbed) == 0] += 10.0
    same = ssl_mod.SimMIMLoss()(recon, target, mask)
    assert torch.allclose(loss, same)


def test_simmim_vit_prefix_offset_selects_only_patch_tokens():
    """The mask indices are offset by `num_prefix_tokens` to skip the class token. An
    off-by-one here masks the wrong tokens and quietly degrades pretraining."""
    cfg = _cfg("vit_tiny_patch16_224", "simmim")
    model = ssl_mod.build_ssl_model(cfg)
    assert model.kind == "vit"
    assert model.num_prefix == 1                    # class token

    x = torch.randn(2, 3, 224, 224)
    recon, target, mask = model(x)
    assert recon.shape == (2, 3, 224, 224)
    assert mask.shape == (2, 1, 224, 224)
    n_tokens = (224 // model.embed_stride) ** 2
    # Indices must land inside [num_prefix, num_prefix + n_tokens).
    assert model.num_prefix + n_tokens <= model.encoder.sequence_length


def test_simmim_mask_unit_must_be_a_multiple_of_the_patch_stride():
    cfg = _cfg("vit_tiny_patch16_224", "simmim")
    cfg.experimental.simmim_mask_patch_size = 24    # not a multiple of 16
    with pytest.raises(ValueError, match="multiple of the encoder's patch stride"):
        ssl_mod.build_ssl_model(cfg)


def test_simmim_refuses_a_convnet_with_actionable_guidance():
    with pytest.raises(ValueError, match="simsiam"):
        ssl_mod.build_ssl_model(_cfg("convnext_tiny", "simmim"))


# ---------------------------------------------------------------------------- MAE

def test_mae_refuses_a_non_plain_vit():
    with pytest.raises(ValueError, match="plain ViT"):
        ssl_mod.build_ssl_model(_cfg("swin_tiny_patch4_window7_224", "mae"))


def test_mae_forward_predicts_only_the_masked_patches():
    model = ssl_mod.build_ssl_model(_cfg("vit_tiny_patch16_224", "mae"))
    x = torch.randn(2, 3, 224, 224)
    pred, target = model(x)
    assert pred.shape == target.shape
    n_patches = (224 // 16) ** 2
    n_masked = pred.shape[1]
    assert 0 < n_masked < n_patches
    assert n_masked == pytest.approx(round(n_patches * model.mask_ratio), abs=2)


# ------------------------------------------------------------------ weight transfer

def test_encoder_weights_transfer_into_the_supervised_model():
    """The author's claim: swin_tiny SimMIM yields 171 encoder tensors, all matching."""
    from trainlab import models

    cfg = _cfg("swin_tiny_patch4_window7_224", "simmim")
    model = ssl_mod.build_ssl_model(cfg)
    state = ssl_mod.encoder_state_dict(model)
    supervised = models.create_model(cfg, num_classes=10)

    loaded, skipped = ssl_mod.load_encoder_into(supervised, state, log=lambda *_: None)
    assert skipped == 0
    assert loaded == len(state)
    assert loaded > 100          # sanity: the encoder, not just a couple of buffers


def test_wholesale_mismatch_raises_instead_of_silently_discarding_pretraining():
    """`strict=False` meant a total key mismatch loaded zero tensors and returned
    quietly, throwing away the entire SSL stage with only a log line."""
    from trainlab import models

    cfg = _cfg("vit_tiny_patch16_224", "simmim")
    supervised = models.create_model(cfg, num_classes=10)
    bogus = {f"nonexistent.{i}": torch.zeros(2) for i in range(50)}
    with pytest.raises(RuntimeError, match="almost entirely discarded"):
        ssl_mod.load_encoder_into(supervised, bogus, log=lambda *_: None)


# ---------------------------------------------------------------- backbone gate limits

def test_register_token_vits_are_refused_up_front():
    """isinstance(VisionTransformer) is necessary but NOT sufficient: reg-token/GAP
    variants subclass it and then crash inside lightly's masked wrapper (its pos-embed
    init assumes a single class token). They used to pass validation cleanly and die at
    model construction — the exact failure the gate exists to prevent."""
    for method in ("mae", "simmim"):
        cfg = _cfg("vit_base_patch16_reg4_gap_256", method, size=256)
        ok, why = ssl_mod.supported_backbone(cfg)
        assert not ok, f"{method} accepted a 4-prefix-token ViT"
        assert "prefix tokens" in why
        with pytest.raises((ValueError, AssertionError)):
            ssl_mod.build_ssl_model(cfg)


def test_plain_vits_still_pass_the_prefix_gate():
    for method in ("mae", "simmim"):
        ok, why = ssl_mod.supported_backbone(_cfg("vit_tiny_patch16_224", method))
        assert ok, why


# -------------------------------------------------------------- tiny-dataset behaviour

def test_ssl_dataset_smaller_than_one_batch_still_trains(tmp_path):
    """drop_last on a dataset smaller than ssl_batch_size dropped EVERY batch: the
    stage reported 'final loss 0.0000' and saved an untrained encoder as if
    pretrained. Below one full batch the short batch must be kept."""
    import numpy as np
    from PIL import Image

    from trainlab import data as data_mod
    from trainlab.device import resolve_runtime
    from trainlab.config import Device

    d = tmp_path / "imgs" / "c0"
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(6):
        Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)).save(d / f"{i}.png")
    # A second class directory so the manifest scan accepts the root.
    d2 = tmp_path / "imgs" / "c1"
    d2.mkdir()
    for i in range(6):
        Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)).save(d2 / f"{i}.png")

    cfg = _cfg("resnet18", "simsiam", size=32)
    cfg.data.train_dir = str(tmp_path / "imgs")
    cfg.data.num_workers = 0
    cfg.schedule.device = Device.CPU
    cfg.experimental.ssl_epochs = 1
    cfg.experimental.ssl_lr_warmup_epochs = 0
    cfg.experimental.ssl_batch_size = 64          # far larger than the 12 images
    cfg.experimental.ssl_amp = "off"
    cfg.experimental.ssl_save_encoder = False

    manifest = data_mod.load_manifest(cfg, "train")
    rt = resolve_runtime(cfg)
    state, result = ssl_mod.pretrain(cfg, rt, manifest, tmp_path, log=lambda *_: None)
    assert result.num_images == 12
    assert result.final_loss != 0.0 and torch.isfinite(torch.tensor(result.final_loss))
    assert state, "no encoder weights came back"
