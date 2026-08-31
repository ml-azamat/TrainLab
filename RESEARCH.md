# RESEARCH.md — What actually improves image-classification accuracy

**Compiled:** 2026-07-27
**Purpose:** justify every control exposed in the training UI. Each technique below either maps to a control in the app (see the [mapping table](#mapping-table)) or is explicitly listed as [deliberately excluded](#deliberately-excluded).

**Verified library versions at time of writing:** `torch` 2.13.0 · `torchvision` 0.28.0 · `timm` 1.0.28 · `torchmetrics` 1.9.0 · `mlflow` 3.14.0 · `optuna` 4.9.0.

---

## 0. The single most important framing

Almost every famous recipe in this document — ResNet Strikes Back, DeiT III, the torchvision SOTA recipe, the ConvNeXt paper — is a **from-scratch ImageNet-1k run: 300–600 epochs, batch size 1024–4096, 8+ GPUs.**

That is not the workload this tool is for. The realistic workload is **fine-tuning a pretrained backbone for ~10–50 epochs on a custom dataset of 10³–10⁵ images.** Several knobs *invert* between those two regimes:

| Knob | From-scratch ImageNet (600 ep) | Fine-tuning custom data (30 ep) |
|---|---|---|
| Mixup / CutMix | Strongly positive | **Often negative** — see §3.4 |
| RandAugment magnitude | 7–9 | 5–7, or TrivialAugment |
| `rrc_scale` lower bound | 0.08 | **0.5–0.9** (0.08 destroys fine-grained classes) |
| Learning rate | High (1e-3 … 8e-3) | Low (1e-4 … 5e-4) + LLRD |
| Warmup | 5–50 epochs | 1–3 epochs |
| Epochs | 300–600 | 10–50 |
| EMA decay | 0.9998 (≈5 000-step horizon) | **must be rescaled** — see §5.5 |
| Label smoothing | 0.1 | 0.1 (transfers fine) |
| Stochastic depth | 0.05–0.5 | 0.05–0.2 |

The evidence for the mixup inversion is direct: Dong et al. (2022), fine-tuning CLIP ViT-B, report that **under a 50-epoch fine-tuning budget, removing Mixup/CutMix gave the best result**, because the pretrained features are already good and only need weak augmentation to transfer; strong mixing is far from the pretraining distribution and forces the representation to move more than it should.[^clipft]

**Consequence for this app's defaults:** the "Balanced (recommended)" preset ships with mixup/cutmix **off** and `rrc_scale=(0.65, 1.0)`. The user-specified values (`mixup_alpha=0.2`, `cutmix_alpha=1.0`, `rrc_scale=0.08`) remain the field defaults and the "Max accuracy" preset, where the long schedule justifies them. This is flagged in §14.

---

## 1. Reference recipes that moved the needle

### 1.1 ResNet Strikes Back (RSB), Wightman/Touvron/Jégou, 2021 — [arXiv:2110.00476](https://arxiv.org/abs/2110.00476)

Took a **vanilla, unmodified ResNet-50** from its 2015 accuracy of 76.1% to **80.4% top-1** purely by changing the training procedure. The most persuasive demonstration in the literature that *recipe > architecture* within a fixed compute budget.

The training-procedure table (Table 2 in the arXiv version), reproduced verbatim for the ingredients relevant to this app:

| Ingredient | A1 | A2 | A3 |
|---|---|---|---|
| Epochs | 600 | 300 | 100 |
| Batch size | 2048 | 2048 | 2048 |
| Optimizer | LAMB | LAMB | LAMB |
| LR | 5e-3 | 5e-3 | 8e-3 |
| LR schedule | cosine | cosine | cosine |
| Weight decay | 0.01 | 0.02 | 0.02 |
| Warmup epochs | 5 | 5 | 5 |
| Label smoothing ε | 0.1 | ✗ | ✗ |
| Dropout | ✗ | ✗ | ✗ |
| Stochastic depth | 0.05 | 0.05 | ✗ |
| Repeated augmentation | ✓ | ✓ | ✗ |
| Gradient clipping | ✗ | ✗ | ✗ |
| H. flip / RRC | ✓ | ✓ | ✓ |
| RandAugment (m/mstd) | 7/0.5 | 7/0.5 | 6/0.5 |
| Mixup α | 0.2 | 0.1 | 0.1 |
| CutMix α | 1.0 | 1.0 | 1.0 |
| Random erasing | ✗ | ✗ | ✗ |
| ColorJitter | ✗ | ✗ | ✗ |
| EMA | ✗ | ✗ | ✗ |
| CE loss | ✗ | ✗ | ✗ |
| **BCE loss** | ✓ | ✓ | ✓ |
| Mixed precision | ✓ | ✓ | ✓ |
| Train res | 224 | 224 | 160 |
| Test res | 224 | 224 | 224 |
| Test crop ratio | 0.95 | 0.95 | 0.95 |
| **Top-1** | **80.4** | **79.8** | **78.1** |

Three findings worth carrying forward:

1. **BCE-with-soft-targets slightly beats cross-entropy** at each method's own best configuration. The reasoning: with Mixup/CutMix, an image genuinely contains two concepts, so forcing the targets to sum to 1 is wrong. BCE treats it as multi-label and sets *each* mixed target to 1 (or 1−ε).
2. **LAMB** was chosen because SGD+BCE was hard to converge at batch 2048 with repeated augmentation. This is a large-batch artifact — **it does not imply LAMB is right at batch 64.**
3. **A3 trains at 160px and tests at 224px** — a deliberate FixRes exploitation (§6.5), and it's *cheaper* as well as more accurate.

⚠️ **Superseded/contextual:** the LAMB choice and batch 2048 are consequences of an 8-GPU budget. At the batch sizes this app targets (32–256), **AdamW is the better default** and LAMB is not exposed.

### 1.2 DeiT III: Revenge of the ViT, Touvron et al., ECCV 2022 — [arXiv:2204.07118](https://arxiv.org/abs/2204.07118)

Notable for moving in the *opposite* direction on augmentation: it replaces the heavy RandAugment stack with **3-Augment** — pick exactly one of {grayscale, solarization, Gaussian blur} with equal probability, then apply color jitter and horizontal flip.[^deit3] Simpler, closer to self-supervised practice, and better for ViTs. Also uses LayerScale (init 1e-4) and simple random cropping instead of RRC for ImageNet-21k pretraining.

**Relevance:** `timm` implements this as the `3a` auto-augment policy, so it costs nothing to expose it as an `auto_augment` option — and it is a genuinely good choice for ViT backbones and for smaller datasets where RandAugment is too destructive.

### 1.3 torchvision SOTA recipe, 2021 — [pytorch.org/blog](https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/)

ResNet-50 → **80.858% top-1**. 8 GPUs × batch 128, 600 epochs, SGD momentum 0.9, **LR 0.5**, cosine, 5 warmup epochs, **weight decay 2e-5**, label smoothing 0.1, Mixup α=0.2, CutMix α=1.0, **TrivialAugmentWide**, random erasing p=0.1, and EMA (`model_ema_steps=32`, `decay=0.99998`). The blog quantifies EMA's contribution at **+0.254 points** — a real but modest gain, obtained for free at inference time.

Note this recipe reaches roughly the same place as RSB A1 via SGD+CE+TrivialAugment instead of LAMB+BCE+RandAugment. **There is no single right recipe** — which is precisely the argument for a tool that makes the comparison cheap.

### 1.4 ConvNeXt fine-tuning schedule, 2022 — [arXiv:2201.03545](https://arxiv.org/abs/2201.03545)

Pretraining: AdamW, base LR 4e-3, wd 0.05, batch 4096, 300 epochs, cosine with 20–50 warmup epochs, RandAugment(9, 0.5), mixup 0.8, cutmix 1.0, random erasing 0.25, label smoothing 0.1, stochastic depth.

**Fine-tuning** (the regime that matters here) is dramatically gentler: LR **5e-5**, weight decay **1e-8**, batch 32, **30 epochs**, **0 warmup epochs**, `layer_decay` **0.8**, `drop_path` 0.2, and **`head_init_scale=0.001`**.[^convnext] The tiny head-init scale and near-zero weight decay are both deliberate: they keep the pretrained trunk from being disturbed early in training.

### 1.5 What the recipes share

Strip away the disagreements and five things are common to every modern recipe:

1. **AdamW-family optimizer with decoupled weight decay**, norm/bias params excluded from decay.
2. **Cosine schedule with linear warmup.**
3. **Label smoothing ≈ 0.1** (except where BCE+mixup subsumes it).
4. **Some automated augmentation policy** — the specific one matters much less than having one.
5. **Long schedules**, and regularization strength scaled *with* schedule length. Heavy augmentation on a short schedule is just underfitting.

---

## 2. Augmentation

### 2.1 The automated policies

| Method | Year | Search cost | Hyperparams | Notes |
|---|---|---|---|---|
| AutoAugment | 2019 ([arXiv:1805.09501](https://arxiv.org/abs/1805.09501)) | ~15 000 GPU-hours (RL search) | policy id | **Largely superseded.** The learned ImageNet policy transfers, but there's no reason to prefer it over the alternatives. |
| RandAugment | 2020 ([arXiv:1909.13719](https://arxiv.org/abs/1909.13719)) | None | N (ops), M (magnitude) | The workhorse. `timm` adds `mstd` (magnitude jitter), which is what `rand-m9-mstd0.5-inc1` means. |
| TrivialAugment(Wide) | ICCV 2021 ([arXiv:2103.10158](https://arxiv.org/abs/2103.10158)) | None | **none** | Sample one op uniformly, sample its strength uniformly. Beat AutoAugment (82.9) and RandAugment (83.3) at **84.33%** on CIFAR-100/WRN-28-10, and beat both on ImageNet/ResNet-50 at 78.07% top-1. |
| 3-Augment | ECCV 2022 | None | none | DeiT III. One of {grayscale, solarize, blur} + jitter + flip. Best for ViTs. |

**Recommendation:** `randaugment` remains the default because it is the most-tuned option and `randaugment_m` is a genuinely useful sweep axis. But **TrivialAugmentWide is the better zero-thought choice** and is the first thing to try when you don't want to tune — its whole selling point is having no knobs.

⚙️ **Implementation note:** `timm` supports `rand-*`, `augmix-*`, `original`, `v0`, `3a` — but **not** TrivialAugment. TrivialAugmentWide comes from `torchvision.transforms.v2.TrivialAugmentWide`. The app therefore routes RandAugment/AutoAugment/3-Augment through `timm` and TrivialAugmentWide through torchvision.

### 2.2 Random Erasing — [arXiv:1708.04896](https://arxiv.org/abs/1708.04896) (AAAI 2020)

Occludes a random rectangle. Cheap, reliably mildly positive on long schedules. `mode=pixel` (random noise fill) is `timm`'s default and beats constant fill. Note RSB used **none** and torchvision used only **0.1** — the commonly-copied 0.25 comes from the DeiT/ConvNeXt lineage. On short fine-tunes, 0.0–0.1 is the safer range.

### 2.3 Mixup and CutMix

- **Mixup** ([arXiv:1710.09412](https://arxiv.org/abs/1710.09412), ICLR 2018): convex combination of both images *and* labels, λ ~ Beta(α, α).
- **CutMix** ([arXiv:1905.04899](https://arxiv.org/abs/1905.04899), ICCV 2019): paste a rectangular patch from image B into image A; label weight = area ratio.

`timm`'s `Mixup` class handles `mixup_prob` (apply at all), `switch_prob` (mixup vs cutmix when both are on), and `mixup_off_epoch` (disable for the final epochs — a genuinely useful trick: regularize early, clean up late).

### 2.4 ⚠️ Mixup ↔ loss ↔ label-smoothing interaction (the part people get wrong)

Three separate constraints, all enforced by the app:

1. **Mixup produces soft targets.** A standard `nn.CrossEntropyLoss` taking integer class indices *cannot consume them*. The loss must be `soft_target_ce` (KL against a soft distribution) or `bce`. This is why the app auto-switches `loss` when mixup/cutmix is enabled, with a visible note.
2. **Label smoothing must not be double-applied.** `timm`'s `Mixup` bakes `label_smoothing` into the soft targets it emits. If you *also* wrap in a smoothing loss, you smooth twice and the effective ε is wrong. The app applies smoothing in exactly one place, chosen by whether mixup is active.
3. **BCE-with-soft-targets is the RSB choice** and slightly beats soft-target CE — but only in a long-schedule, high-augmentation regime. Don't assume it transfers to a 20-epoch fine-tune.

Separately: label smoothing improves calibration and accuracy but **degrades the model as a distillation teacher** (Müller et al. 2019, [arXiv:1906.02629](https://arxiv.org/abs/1906.02629)) by collapsing the inter-class similarity structure in the logits. If you plan to distill from a run, train the teacher with ε=0.

### 2.5 Live augmentation preview

Not a technique from the literature, but empirically the highest-value debugging surface in a tool like this. Roughly half of "my model won't train" cases are visible in one glance at the augmented batch: RRC scale cropping the subject out entirely, normalization applied twice, channels swapped, aspect-ratio squash destroying a fine-grained cue, or a rotation that makes a "6"/"9" class ambiguous. This is why it's a first-class feature and not an afterthought.

---

## 3. Regularization

| Technique | Default | Evidence & when to change |
|---|---|---|
| **Label smoothing** | 0.1 | Near-universal. Improves accuracy and calibration. Set to 0 for a distillation teacher (§2.4). |
| **Stochastic depth (drop-path)** | 0.1 | [arXiv:1603.09382](https://arxiv.org/abs/1603.09382). The **primary** regularizer for deep ConvNeXt/ViT/Swin models — far more effective than dropout for them. Scale with model size and schedule length: ~0.05 (tiny/short) → 0.5 (huge/long). ConvNeXt-T fine-tune uses 0.2. |
| **Dropout** | 0.0 | Classifier-head only in modern backbones. Mostly redundant once drop-path is on. Keep at 0 unless the head is overfitting specifically. |
| **Weight decay** | 0.05 (AdamW) | Note the enormous spread: 0.05 (ConvNeXt pretrain) vs 2e-5 (torchvision SGD) vs 1e-8 (ConvNeXt fine-tune). WD is **not** transferable across optimizers or regimes — it is one of the highest-value sweep axes. |
| **No WD on norm/bias** | true | Applying decay to BatchNorm/LayerNorm γ,β and to biases is a well-known small bug that costs a fraction of a point. Every modern reference implementation excludes them. Leave on. |
| **EMA of weights** | true | See §5.5. |

---

## 4. Optimization

### 4.1 Optimizer choice

| Optimizer | When |
|---|---|
| **AdamW** ([arXiv:1711.05101](https://arxiv.org/abs/1711.05101)) | **The default.** Required for ViT/ConvNeXt-class models, fine for CNNs. Decoupled weight decay is the whole point — don't use plain Adam. |
| **SGD + Nesterov** | Still competitive for pure CNNs (ResNet/EfficientNet) on long schedules, and torchvision's 80.858% recipe uses it. Needs ~1000× larger LR than AdamW (0.5 vs 5e-4) — a classic footgun the app guards against. |
| **Lion** ([arXiv:2302.06675](https://arxiv.org/abs/2302.06675)) | Memory-light (no second moment), sign-based updates. **LR must be 3–10× smaller than AdamW and weight decay 3–10× larger** to keep effective strength comparable. β=(0.9, 0.99). Worth an experiment; not a safe default. |
| **LAMB / LARS** | Large-batch (≥1024) only. **Not exposed** — this app targets single-GPU batch sizes where it has no advantage. |

### 4.2 LR ↔ batch-size scaling

Two competing rules, both implemented as suggestions in the UI:

- **Linear scaling** (Goyal et al. 2017, [arXiv:1706.02677](https://arxiv.org/abs/1706.02677)): `lr = base_lr × batch/base_batch`. Derived for SGD; the paper pairs it with gradual warmup, which is why warmup and large batch are always mentioned together.
- **Square-root scaling**: `lr = base_lr × sqrt(batch/base_batch)`. Empirically better for adaptive optimizers (Adam/AdamW), since the update is already normalized by gradient magnitude.

The app shows the suggested LR and the rule it used, and lets you accept or ignore it. It never silently overwrites a value you typed.

### 4.3 Schedules

**Cosine + linear warmup** is the default and is what essentially every recipe in §1 uses. Warmup matters most at large batch, high LR, and for transformers (it prevents an early attention-collapse failure mode). One-cycle (Smith, [arXiv:1708.07120](https://arxiv.org/abs/1708.07120)) is a good short-schedule alternative. `plateau` is the choice when you truly don't know the right epoch count. `step` is legacy — exposed for reproducing old baselines only.

### 4.4 SAM / ASAM

SAM ([arXiv:2010.01412](https://arxiv.org/abs/2010.01412), ICLR 2021) minimizes loss in a neighborhood rather than at a point, seeking flat minima. It requires **two forward-backward passes per step ≈ 2× the wall-clock cost.**

The gain is extremely architecture-dependent. Chen et al. 2021 ([arXiv:2106.01548](https://arxiv.org/abs/2106.01548)): SAM lifts **ViT-B/16 from 74.6% → 79.9%** and Mixer-B/16 from 66.4% → 77.4%, but a comparable ResNet-152 gains only **0.8%**.

**Reading:** SAM is a large win for attention/MLP models trained without strong augmentation or pretraining, and a marginal one for well-regularized CNNs. Given it doubles cost, `sam=false` is correct as a default — but it belongs in the "Max accuracy" search space for ViT backbones. ASAM (ICML 2021) makes ρ scale-invariant and hence easier to tune.

### 4.5 Layer-wise LR decay (LLRD)

Layer ℓ gets `lr × decay^(depth − ℓ)`, so early layers (generic features) move slowly and late layers (task-specific) move fast. Originated in ULMFiT/ELECTRA, now standard for vision fine-tuning.

Published values: **BEiT-v2 uses 0.65 for ViT-B and 0.8 for ViT-L**;[^beit2] ConvNeXt fine-tuning uses 0.8; other work reports 0.55 for ViT-S/B and 0.7–0.85 for segmentation heads. **Larger models want a value closer to 1.0** (they have more layers, so the compounding is more aggressive).

**This is one of the highest-value knobs in the whole app for the fine-tuning regime, and it is almost always left at its "off" value by people who don't know about it.** 0.75 is a good first thing to try.

### 4.6 Gradient clipping and accumulation

`grad_clip_norm=1.0` is cheap insurance, essentially free, and matters for transformers and for AMP. RSB used none; DeiT-family recipes use it. Keep it on.

`grad_accum_steps` trades wall-clock for effective batch size — the correct answer to OOM when you need a large effective batch. Note it changes BatchNorm statistics (BN still sees only the micro-batch), which is a real and frequently-missed caveat for CNNs.

---

## 5. Transfer learning strategy

### 5.1 Freeze / unfreeze

Full fine-tuning beats linear probing whenever you have more than a few hundred images per class. **Head-only warmup** (freeze the backbone for 1–3 epochs while the randomly-initialized head stabilizes, then unfreeze) is a cheap way to avoid the large early gradients from a random head corrupting good pretrained features. `head_init_scale` (ConvNeXt uses **0.001**) attacks the same problem more directly and more cheaply.

### 5.2 BatchNorm freezing

For **small batch sizes (<16)**, BN running statistics get noisy and freezing them (or switching to GroupNorm) helps. Irrelevant for ConvNeXt/ViT, which use LayerNorm — hence `freeze_bn` is an advanced control, not a headline one.

### 5.3 Discriminative LRs

A coarse special case of LLRD (§4.5): separate LR for backbone vs head, typically 10× higher for the head. Subsumed by `layer_lr_decay` + `head_lr_mult`.

### 5.4 Progressive resizing

Train early epochs at low resolution, ramp up. Popularized by fast.ai and used implicitly by RSB-A3 (160→224). Real throughput win — early epochs are ~2× faster at 160px than 224px — and it doubles as a curriculum. Interacts with FixRes (§5.5) in a useful direction.

### 5.5 FixRes — train/test resolution discrepancy — [arXiv:1906.06423](https://arxiv.org/abs/1906.06423) (NeurIPS 2019)

The most under-used free win in this list.

`RandomResizedCrop` at training time zooms in on a random sub-region, so **objects appear larger during training than at test time**, where a plain center crop is used. The scale statistics don't match. The fix: **test at a higher resolution than you trained at**, optionally fine-tuning just the classifier and the final norm layer to recalibrate the activation statistics.

Reported effect for a ResNet-50 trained at 224 (paper Table 5): **77.1%** at test 224, **78.6%** at test 288 (+1.5) and **79.0%** at test 384 (+1.9) with a brief classifier/BN fine-tune at the target resolution; at test 448 it *declines* to 78.4%, so the win peaks around 1.5–1.7× the train resolution rather than growing indefinitely. Without any fine-tuning the optimum sits around test 288 (78.4%). For a fixed target test resolution, a *lower* train resolution gives better test accuracy — RSB-A3 (train 160 / test 224) is exactly this.

**This is why `test_input_size` is a separate control from `input_size`.** Setting it ~1.15× `input_size` is often a free fraction of a point at zero training cost.

### 5.6 EMA horizon — ⚠️ a default that is actively wrong for small datasets

`ema_decay = 0.9998` has an effective averaging horizon of `1/(1−0.9998) = 5 000 optimizer steps`.

A 5 000-image dataset at batch 64 is 78 steps/epoch; a 30-epoch run is **2 340 total steps**. The EMA horizon is **longer than the entire run**, so the EMA weights never converge away from initialization and `eval_ema_weights=true` reports garbage — a metric that looks like a broken model while the raw weights are fine.

**Mitigation in the app:** `ema_decay` gets an `auto` mode that solves for a horizon of ~10% of total steps, and a hard validation warning fires when `1/(1−decay) > 0.5 × total_steps`.

**SWA** ([arXiv:1803.05407](https://arxiv.org/abs/1803.05407), UAI 2018) is the alternative: equal-weight averaging of checkpoints under a high constant LR, worth ~+0.6–0.9 points on ImageNet ResNets. EMA is cheaper and more flexible (it also improves robustness to label noise, prediction consistency, and calibration); SWA needs a schedule change. EMA is the default; SWA lives in the experimental group.

---

## 6. Loss functions

| Loss | Use it when |
|---|---|
| **Cross-entropy (+LS)** | Default. Correct answer for balanced or mildly imbalanced data. |
| **Soft-target CE** | **Required** when mixup/cutmix is on. Auto-selected. |
| **BCE with soft targets** | RSB's choice; slightly beats CE on long, heavily-augmented schedules. Treats mixed images as genuinely multi-label. |
| **Focal** ([arXiv:1708.02002](https://arxiv.org/abs/1708.02002)) | **Severe** imbalance, or when minority-class recall dominates. ⚠️ On balanced data focal loss **hurts** — it down-weights easy examples that are still important. Its other documented use is calibration: Mukhoti et al. 2020 ([arXiv:2002.09437](https://arxiv.org/abs/2002.09437)) show focal-trained models are better calibrated than CE-trained ones, since focal implicitly adds a max-entropy regularizer. |
| **Class-balanced CE** ([arXiv:1901.05555](https://arxiv.org/abs/1901.05555)) | Reweights by *effective number* of samples `(1−β^n)/(1−β)` rather than raw inverse frequency, which over-corrects. |
| **Logit-adjusted** ([arXiv:2007.07314](https://arxiv.org/abs/2007.07314), ICLR 2021) | **The best-supported single fix for long-tailed data.** Add `τ·log(class_prior)` to the logits. Principled (it's a Bayes-consistent correction), has a post-hoc variant needing no retraining, and reported to beat both loss-reweighting and post-hoc alternatives. |

---

## 7. Imbalanced and long-tailed data

Four families, and **they are substitutes, not complements** — stacking them double-corrects and overshoots the minority classes:

1. **Weighted / balanced sampling** — resample the input distribution. Simple, but oversampling the tail risks overfitting the few real minority images.
2. **Weighted loss** — reweight the gradient. Inverse-frequency over-corrects; effective-number (§6) is the better formulation.
3. **Logit adjustment** — correct the decision rule instead of the data or the loss. Cheapest and best-supported.
4. **Two-stage decoupled training** (Kang et al., ICLR 2020, [arXiv:1910.09217](https://arxiv.org/abs/1910.09217)) — the key finding is that **instance-balanced (i.e. plain random) sampling learns the best *representations***; only the *classifier* needs rebalancing. So: train normally, then re-train just the classifier with class-balanced sampling (cRT) or rescale its weight norms (LWS). Consistently strong, and it explains why naive resampling underperforms.

**App behavior:** the `Imbalanced dataset` preset selects logit adjustment + macro-F1 as the primary metric (accuracy is a misleading objective under imbalance), and a validation warning fires if you combine a weighted sampler *and* a weighted loss.

---

## 8. Inference-time gains

| Technique | Cost | Typical gain |
|---|---|---|
| **hflip TTA** | 2× inference | **+0.7%** on a ResNet-18 ImageNet baseline — the best accuracy-per-compute point on this list. |
| **FiveCrop TTA** | 5× | +0.79% beyond hflip |
| **TenCrop TTA** | ~9× | only +0.42% beyond FiveCrop — clearly past the knee |
| **Multi-scale TTA** | k× | ~+0.6 points (mIoU, segmentation) |
| **EMA / SWA** | **free** | +0.25% (torchvision, EMA) / +0.6–0.9% (SWA, ImageNet) |
| **Ensembling** | N× | Largest single gain available, always. |
| **Temperature scaling** | free | Fixes calibration without touching accuracy (Guo et al. 2017, [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)). Measure with ECE. |

⚠️ Naive TTA averaging can *hurt* on classes where the augmentation is label-destroying (Shanmugam et al., [arXiv:2011.11156](https://arxiv.org/abs/2011.11156)) — vertical flip on digits being the obvious case. Default TTA to `none`, apply at eval only, and always compare against the no-TTA number.

**Threshold tuning** matters whenever the deployed decision isn't argmax — per-class thresholds tuned on validation can move macro-F1 substantially under imbalance, at zero training cost.

---

## 9. Throughput — the levers that let you run more experiments

This section is about experiments-per-hour, which for an experimentation tool is as important as accuracy-per-experiment.

| Lever | Effect | Caveats |
|---|---|---|
| **AMP (bf16/fp16)** | ~1.5–2× | bf16 needs Ampere+ (SM80). No loss scaler needed for bf16; fp16 needs one. |
| **`channels_last`** | Meaningful for convnets | NHWC matches what cuDNN/tensor cores want. Little or no effect for ViTs. |
| **`torch.compile`** | +43% training on average across 163 models on A100; +51% at AMP precision, +21% at fp32 | First-call compile cost of 30–120 s — **bad for 2-minute runs, good for 2-hour runs**. Recompiles on input-shape changes, so it fights progressive resizing. |
| **cuDNN benchmark** | Small | Only when input shapes are static. |
| **Fused optimizers** (`fused=True`) | ~5–10% | AdamW/SGD on CUDA. Free. |
| **DataLoader tuning** | Often the actual bottleneck | `num_workers`, `persistent_workers=True` (avoids per-epoch worker respawn), `pin_memory=True`, `prefetch_factor`. If GPU utilization is spiky, this — not the model — is your problem. |
| **Decode-cache / RAM cache** | Large for small datasets | If the dataset fits in RAM, JPEG decode is pure waste after epoch 1. |

### 9.1 ⚠️ Apple Silicon (this machine is an M3 Pro)

The defaults above assume CUDA. On MPS, three of them are wrong:

- **bf16 is a trap.** MPS autocast currently disables itself with a warning when the target dtype is bf16 ([pytorch#139386](https://github.com/pytorch/pytorch/issues/139386)), and where bf16 does run, the backend lacks optimized Metal kernels for it in convolutions and some linear projections — reports of a ~10× throughput collapse on consumer M-series chips. **fp16 is the correct autocast dtype on MPS.**
- **`channels_last`** is not meaningfully accelerated on MPS.
- **`torch.compile`** support on MPS is partial; treat it as opt-in and expect graph breaks.

**App behavior:** `amp` stays `bf16` as the schema default (correct on CUDA), but the runtime **resolves device-aware** — on MPS it downgrades bf16→fp16 and logs a visible warning to both the UI and the tracker, so the run record reflects what actually executed rather than what was requested. Same for `channels_last` and `torch_compile`.

---

## 10. Optional / higher-variance techniques

- **Knowledge distillation** (Hinton et al. 2015; Beyer et al. 2022 "Patient and consistent", [arXiv:2106.05237](https://arxiv.org/abs/2106.05237)). The 2022 paper's finding is the actionable one: KD works as **function matching** and needs *aggressive augmentation shared between teacher and student* plus a *very* long schedule. Short-schedule KD mostly disappoints. Also: a label-smoothed teacher is a worse teacher (§2.4).
- **Self-supervised / in-domain pretraining.** The framing that matters: in-domain SSL **reuses your own labelled training images and discards the labels** — no extra data is required. That is worth doing because cross-entropy over N classes carries at most log₂(N) bits of supervision per image (≈3.3 bits for 10 classes), while an SSL objective extracts far more signal from the same pixels. Extra unlabeled data is *optional* and simply enlarges the pretraining set (`ssl_extra_data_dir`).

  Two methods are exposed, and the choice is constrained by architecture:

  - **SimSiam** ([arXiv:2011.10566](https://arxiv.org/abs/2011.10566), CVPR 2021) — two augmented views, projector, bottlenecked predictor, **stop-gradient**. No negatives, no memory bank, no momentum encoder, so it trains at small batch sizes where SimCLR (which needs 512–4096 for in-batch negatives) does not. Works with **any** backbone.
  - **MAE** ([arXiv:2111.06377](https://arxiv.org/abs/2111.06377), CVPR 2022) — mask 75% of patches, **drop** them, encode only the visible ~25%, and reconstruct the rest with a Transformer decoder that is discarded afterwards. The high mask ratio is the central finding: images are spatially redundant, so the 15% masking used in language models is too easy. Token-dropping is where its speed comes from, and why it structurally requires a **plain ViT**.
  - **SimMIM** ([arXiv:2111.09886](https://arxiv.org/abs/2111.09886), CVPR 2022) — the deliberately simple counterpart. The encoder sees **every** token, with masked ones replaced by a learnable mask token at the input; prediction is a **single linear layer**; the loss is **L1** on raw pixels. Because nothing is dropped it also runs on hierarchical **Swin** encoders. Its own finding is that the masked *unit* must be large (32px) relative to the patch size — with a small unit the model just inpaints from immediate neighbours.

  | | MAE | SimMIM |
  |---|---|---|
  | Encoder input | visible patches only | all patches (mask token at masked positions) |
  | Decoder | Transformer, 8 blocks / 512-d | single linear layer |
  | Loss | MSE, optionally on normalised pixels | L1 on raw pixels |
  | Mask ratio | 0.75 | ~0.6, wide plateau |
  | Backbones | plain ViT | ViT **and Swin** |
  | Auto LR (AdamW) | 1.5e-4 x bs/256 | 1e-4 x bs/256 |

  Neither works on a plain convnet: convolutions leak information across masked regions, so convnet MIM needs sparse convolutions ([SparK](https://arxiv.org/abs/2301.03580), or the FCMAE in [ConvNeXt V2](https://arxiv.org/abs/2301.00808)). Not implemented here — use SimSiam for convnets. Backbone support is decided by an isinstance check, not a name heuristic.

  ⚠️ **The honest caveat: "warmup" is the wrong frame.** SSL needs hundreds of epochs — SimSiam's CIFAR-10 recipe is 800 at batch 512, its ImageNet baseline 100; MAE runs 800–1600. A handful of epochs does approximately nothing, and starting from `pretrained=True` a short SSL phase can pull good ImageNet weights away from a strong initialisation faster than it adds anything back. The app's warnings are conditioned on the starting point: from pretrained weights this is domain adaptation (useful range 20–50 epochs, the default is 30, and fewer than 10 warns); from scratch anything under ~100 warns. The realistic workflow is **pretrain once, fine-tune many times**: `ssl_save_encoder` logs the encoder as an artifact for exactly that.

  Best return when the domain gap from ImageNet is large. Supporting evidence is mostly medical/scientific: [98.84% from 4,000 retinal images](https://arxiv.org/html/2404.10166v1) (2024), [Azizi et al. on medical classification](https://arxiv.org/abs/2101.05224) (2021), [SEM particle segmentation](https://www.nature.com/articles/s41524-025-01802-3) (2025). Counterpoint worth knowing: [Chen et al. 2022](https://arxiv.org/pdf/2205.14443) find MIM beats contrastive on data-*sufficient* downstream tasks and loses on data-*insufficient* ones — neither method wins everywhere.
- **Semi-supervised pseudo-labeling** (Noisy Student [arXiv:1911.04252](https://arxiv.org/abs/1911.04252), FixMatch [arXiv:2001.07685](https://arxiv.org/abs/2001.07685)). Requires unlabeled data and confidence thresholding. High variance — it amplifies your model's existing biases.
- **Label-noise filtering.** The `log_worst_predictions` feature is the cheap manual version and frequently the highest-ROI hour you can spend on a real dataset: the top-32 highest-loss validation images are, in practice, *mostly mislabeled* rather than hard.
- **Curriculum / sample reweighting.** Mixed evidence; easy to make things worse.

---

## 11. Recommended experiment order

Ordered by expected accuracy gain per unit of your time — this is also the "recommended experiment order" section of the README.

1. **Sanity check.** `Fast baseline` preset. Confirm the loss decreases and the augmentation preview looks like your data. Look at the worst-predictions grid — fix label noise *before* tuning anything.
2. **Backbone and resolution.** The single biggest lever. `convnext_tiny` @224 → try `convnext_small`, `vit_base_patch16_224`, or a higher `input_size`. Bigger/higher usually wins if you can afford it.
3. **LR × weight decay.** A 3×3 random search here beats everything else in this list. These are the two hyperparameters that actually need tuning per-dataset.
4. **`layer_lr_decay = 0.75`.** One run. Frequently a free win when fine-tuning a large backbone, and widely unknown.
5. **Augmentation strength**, matched to schedule length. Short run → lighter aug. Try TrivialAugmentWide as the no-tuning option.
6. **`test_input_size` ≈ 1.15 × `input_size`** (FixRes). Costs zero training time.
7. **EMA** (check the horizon warning) and **hflip TTA**. Both nearly free.
8. **Longer schedule + heavier augmentation together.** Only now — they only pay off jointly.
9. **SAM**, if you're on a ViT and can afford 2× time.
10. **Ensemble** your best 3–5 runs. Always the biggest remaining gain, always the most expensive.

---

## 12. ⚠️ Defaults I'm flagging as questionable

Per your instruction to say so rather than silently include:

| Your default | Concern | What the app does |
|---|---|---|
| `mixup_alpha=0.2`, `cutmix_alpha=1.0`, `mixup_prob=1.0` **on**, with `epochs=30` and `pretrained=true` | Direct evidence this **hurts** in short fine-tuning (§0, §3.4). It also silently forces the loss away from `cross_entropy`. | **Now off by default** (`mixup_alpha=0.0`, `cutmix_alpha=0.0`). Keeping the literature values as defaults only affected the headless path — the UI boots into `Balanced`, so `python train.py` was the sole consumer of the raw defaults, and it had neither the preset buttons nor the warning panel. It therefore produced a config the tool immediately warned was harmful. `Max accuracy` turns mixing on with `epochs=120`; the literature values live in the tooltips. Warning still shown when mixup is on with `epochs < 50` and `pretrained=true`. |
| `ema_decay=0.9998` | Horizon (5 000 steps) exceeds the entire run for any dataset under ~10 k images (§5.6). Produces a silently-broken EMA metric. | `auto` mode added; hard warning when horizon > 50% of total steps. |
| `amp=bf16` | Broken/slow on MPS, which is this machine (§9.1). | Device-aware resolution to fp16 + visible warning; the tracker logs the resolved value. |
| `rrc_scale=(0.08, 1.0)` | Correct for ImageNet-scale from-scratch; **destroys fine-grained and small datasets** — the crop often contains no evidence of the class. | Default kept, but presets set 0.65; the augmentation preview makes the damage immediately visible, which is the real fix. |
| `val_crop_pct=0.875` | RSB found **0.95** better, and modern ConvNeXt/ViT weights often prefer 0.95–1.0. | Kept at 0.875 (it matches torchvision convention) with a tooltip pointing at 0.95. |
| `head_init_scale=1.0` | ConvNeXt's own fine-tuning recipe uses **0.001** (§1.4, §5.1). | Kept at 1.0; tooltip and the `Small dataset` preset use 0.001. |
| `class_weights` + `loss=class_balanced_ce` + `logit_adjusted` + `sampler=weighted` | **Four overlapping ways to fix the same problem** (§7). Stacking them over-corrects. | All kept — they're legitimately different techniques worth comparing — but a validation warning fires when more than one is active. |
| `distributed`, `num_gpus` | Declared controls with **no DDP runtime behind them** — and `effective_batch_size` multiplied by `num_gpus`, so enabling them distorted the EMA-horizon warning and LR suggestion while training ran on one device. | **Removed.** Old configs still load (the fields are dropped and reported). The reads-guard test now covers the `schedule` group so dead controls can't hide there again. |
| Six schema-only controls | SSL warmup, pseudo-labeling, cRT, curriculum, noise filtering and ensembling were **declared in the schema but read by no runtime code** — they rendered, logged to the tracker, and did nothing. | All six are now implemented. A test (`test_every_experimental_field_is_read_by_the_engine`) fails the build if a control is ever added without runtime code reading it. |
| `ssl_warmup_epochs` as a bare integer | The original control implied SSL was a cheap warmup, and its tooltip described large-corpus pretraining (needs a data path) while its shape implied in-domain pretraining (needs only an epoch count) — two different techniques merged into one field. | Replaced by `ssl_method` (simsiam/mae/simmim) plus an explicit budget, optional `ssl_extra_data_dir`, and per-method parameters. The field is listed in `_LEGACY_DROPPED_FIELDS` so runs recorded before the change still clone (it is dropped rather than aliased, because the replacement means something different). The epoch-count warning is now conditioned on `pretrained`: from a pretrained encoder this is domain adaptation and 20–50 epochs is the useful range, so warning against SimSiam's 800-epoch from-scratch CIFAR recipe was comparing the config to an experiment it was not running. |
| `swa_lr` | Declared and tooltipped, but **never applied** — SWA averaged under the decaying cosine LR, which is the one thing that makes SWA pointless. | The LR is now pinned to `swa_lr` for every step of the averaging phase. |
| `focal` on balanced data | Documented to *hurt* (§6). | Kept; warning when `focal` is selected and the computed class-imbalance ratio is < 3:1. |

---

<a name="mapping-table"></a>
## 13. Mapping table — technique → control → default → effect → source

| # | Technique | UI group | Control | Default | Expected effect | Source (year) |
|---|---|---|---|---|---|---|
| 1 | Backbone choice | Model | `backbone` | `convnext_tiny` | Largest single lever | ConvNeXt (2022) |
| 2 | Transfer learning | Model | `pretrained` | `true` | +5–30 pts on small data | ubiquitous |
| 3 | Stochastic depth | Model | `drop_path_rate` | 0.1 | +0.2–0.5; primary regularizer for deep nets | [1603.09382](https://arxiv.org/abs/1603.09382) (2016) |
| 4 | Dropout | Model | `drop_rate` | 0.0 | ~0; redundant with drop-path | Srivastava (2014) |
| 5 | Head-only warmup | Model | `freeze_policy`, `freeze_epochs` | none, 0 | Protects features from random-head gradients | ULMFiT (2018) |
| 6 | Head init scaling | Model | `head_init_scale` | 1.0 (try 0.001) | Same goal as #5, cheaper | ConvNeXt (2022) |
| 7 | BN freezing | Model | `freeze_bn` | false | Helps at batch < 16 | FixRes (2019) |
| 8 | Weight EMA | Model | `ema`, `ema_decay` | true, 0.9998/auto | **+0.25** free | torchvision (2021) |
| 9 | Gradient checkpointing | Model | `gradient_checkpointing` | false | ~30% slower, ~40% less VRAM | — |
| 10 | RandomResizedCrop | Input | `train_resize_policy`, `rrc_scale` | RRC, 0.08–1.0 | Core aug; **use 0.65 for fine-grained** | AlexNet lineage |
| 11 | Interpolation | Input | `interpolation` | bicubic | Match pretraining; small but real | timm |
| 12 | FixRes | Input | `test_input_size` | = `input_size` | **~+1 pt for free** at 1.15× | [1906.06423](https://arxiv.org/abs/1906.06423) (2019) |
| 13 | Test crop ratio | Input | `val_crop_pct` | 0.875 (0.95 better) | +0.1–0.3 | RSB (2021) |
| 14 | channels_last | Input | `channels_last` | true | Throughput (CUDA convnets) | PyTorch |
| 15 | Horizontal flip | Aug | `hflip` | 0.5 | Free, universal | — |
| 16 | RandAugment | Aug | `auto_augment`, `randaugment_n/m/mstd` | randaugment, 2/9/0.5 | +1–2 on long schedules | [1909.13719](https://arxiv.org/abs/1909.13719) (2020) |
| 17 | TrivialAugmentWide | Aug | `auto_augment=trivialaugment-wide` | — | Matches/beats RandAug, **zero tuning** | [2103.10158](https://arxiv.org/abs/2103.10158) (2021) |
| 18 | 3-Augment | Aug | `auto_augment=3a` | — | Best for ViTs / small data | DeiT III (2022) |
| 19 | Random erasing | Aug | `random_erasing_p/mode` | 0.25, pixel | +0.1–0.3 long schedules; use 0–0.1 short | [1708.04896](https://arxiv.org/abs/1708.04896) (2020) |
| 20 | Mixup | Aug | `mixup_alpha`, `mixup_prob` | **0.0** (off; lit. value 0.2 in tooltip/Max-accuracy), 1.0 | +1–2 long; **negative short** | [1710.09412](https://arxiv.org/abs/1710.09412) (2018) |
| 21 | CutMix | Aug | `cutmix_alpha`, `mixup_switch_prob` | **0.0** (off; lit. value 1.0 in tooltip/Max-accuracy), 0.5 | as #20 | [1905.04899](https://arxiv.org/abs/1905.04899) (2019) |
| 22 | Mixup annealing | Aug | `mixup_off_epoch` | 0 (never) | Regularize early, clean up late | timm |
| 23 | Color jitter | Aug | `color_jitter` | 0.4 | Small; part of 3-Augment | — |
| 24 | Grayscale/blur | Aug | `grayscale_p`, `blur_p` | 0.0 | 3-Augment components | DeiT III (2022) |
| 25 | Label smoothing | Loss | `label_smoothing` | 0.1 | +0.2–0.5, better calibration | [1906.02629](https://arxiv.org/abs/1906.02629) (2019) |
| 26 | Soft-target CE | Loss | `loss` (auto) | auto w/ mixup | **Correctness requirement** | — |
| 27 | BCE soft targets | Loss | `loss=bce` | — | Slightly > CE on RSB-style runs | RSB (2021) |
| 28 | Focal loss | Loss | `loss=focal`, `focal_gamma` | 2.0 | Severe imbalance only; **hurts balanced** | [1708.02002](https://arxiv.org/abs/1708.02002) (2017) |
| 29 | Class-balanced loss | Loss | `class_weights=effective-number` | none | Better than inverse-freq | [1901.05555](https://arxiv.org/abs/1901.05555) (2019) |
| 30 | Logit adjustment | Loss | `loss=logit_adjusted`, `logit_adjust_tau` | 1.0 | **Best single long-tail fix** | [2007.07314](https://arxiv.org/abs/2007.07314) (2021) |
| 31 | Balanced sampling | Data | `sampler`, `oversample_factor` | random, 1.0 | Substitute for #29/#30, not additive | Decoupling (2020) |
| 32 | Two-stage decoupling (cRT) | Advanced | `classifier_retrain_epochs`, `classifier_retrain_lr`, `classifier_retrain_reinit` | 0, auto (10x), true | Strong long-tail; freezes the backbone and rebalances only the head | [1910.09217](https://arxiv.org/abs/1910.09217) (2020) |
| 33 | AdamW | Opt | `optimizer`, `lr`, `weight_decay` | adamw, 3e-4, 0.05 | Default | [1711.05101](https://arxiv.org/abs/1711.05101) (2019) |
| 34 | SGD+Nesterov | Opt | `optimizer=sgd`, `momentum`, `nesterov` | 0.9, true | Competitive for CNNs; needs ~1000× LR | torchvision (2021) |
| 35 | Lion | Opt | `optimizer=lion` | — | LR ÷3–10, WD ×3–10 | [2302.06675](https://arxiv.org/abs/2302.06675) (2023) |
| 36 | No-WD on norm/bias | Opt | `no_weight_decay_on_norm_bias` | true | Small, free, always on | ubiquitous |
| 37 | LLRD | Opt | `layer_lr_decay` | 1.0 (**try 0.75**) | **+0.5–2 when fine-tuning** | BEiT-v2 (2022), ConvNeXt (2022) |
| 38 | SAM | Opt | `sam`, `sam_rho` | false, 0.05 | **+5 ViT** / +0.8 CNN, at **2× cost** | [2010.01412](https://arxiv.org/abs/2010.01412) (2021) |
| 39 | Gradient clipping | Opt | `grad_clip_norm` | 1.0 | Stability, ~free | — |
| 40 | Gradient accumulation | Opt | `grad_accum_steps` | 1 | Big effective batch under VRAM limits | — |
| 41 | LR↔batch scaling | Opt | suggestion UI | sqrt for AdamW | Prevents a classic misconfiguration | [1706.02677](https://arxiv.org/abs/1706.02677) (2017) |
| 42 | Warmup + cosine | Sched | `scheduler`, `warmup_epochs`, `min_lr` | cosine, 3, 1e-6 | Universal | [1608.03983](https://arxiv.org/abs/1608.03983) (2017) |
| 43 | One-cycle | Sched | `scheduler=one_cycle` | — | Good short-schedule alternative | [1708.07120](https://arxiv.org/abs/1708.07120) (2018) |
| 44 | AMP | Sched | `amp` | bf16 (→fp16 on MPS) | **1.5–2× throughput** | [1710.03740](https://arxiv.org/abs/1710.03740) (2018) |
| 45 | Progressive resizing | Sched | `progressive_resizing` | off | Throughput + curriculum | fast.ai; RSB-A3 |
| 46 | torch.compile | Model | `torch_compile` | false | **+43% train**; 30–120 s warmup | PyTorch 2.x |
| 47 | DataLoader tuning | Data | `num_workers`, `persistent_workers`, `prefetch_factor`, `pin_memory` | 8, true, 2, true | Often *the* bottleneck | PyTorch |
| 48 | Dataset caching | Data | `cache_mode` | none | Large win when data fits in RAM | — |
| 49 | TTA | Val | `tta` | none | **+0.7 hflip**; diminishing after | [2011.11156](https://arxiv.org/abs/2011.11156) (2021) |
| 50 | SWA | Advanced | `swa`, `swa_start_epoch`, `swa_lr` | off, 0.75, 1e-4 | +0.6–0.9; `swa_lr` is held CONSTANT during the averaging phase | [1803.05407](https://arxiv.org/abs/1803.05407) (2018) |
| 51 | Calibration (ECE) | Val | `metrics` incl. ECE | on | Diagnostic; enables temp. scaling | [1706.04599](https://arxiv.org/abs/1706.04599) (2017) |
| 52 | Worst-prediction review | Val | `log_worst_predictions` | true | **Finds label noise** — highest real-world ROI | — |
| 53 | Confusion matrix | Val | `log_confusion_matrix` | true | Diagnostic | — |
| 54 | Knowledge distillation | Advanced | `kd_teacher`, `kd_temperature`, `kd_alpha` | off | Needs long schedule + shared aug | [2106.05237](https://arxiv.org/abs/2106.05237) (2022) |
| 55 | Pseudo-labeling | Advanced | `pseudo_label_dir`, `_threshold`, `_epochs`, `_rounds`, `_max_ratio` | off, 0.95, 10, 1, 1.0 | High variance; continues the same model rather than retraining a fresh student | [1911.04252](https://arxiv.org/abs/1911.04252) (2020) |
| 56 | Ensembling | Advanced | `ensemble_run_ids` (run IDs or ckpt paths) | [] | Largest remaining gain; logged as separate `ensemble/*` metrics | ubiquitous |
| 57 | Early stopping | Ckpt | `early_stopping`, `es_patience` | true, 10 | Saves time, not accuracy | — |
| 58 | Sweeps (TPE) | Sweeps | search-space mode | — | Best accuracy per GPU-hour overall | [1907.10902](https://arxiv.org/abs/1907.10902) (2019) |
| 59 | In-domain SSL: SimSiam | Advanced | `ssl_method=simsiam`, `ssl_epochs`, `simsiam_*` | none, 30 (domain-adaptation range; raise ≫100 from scratch) | Any backbone; negative-free, small-batch friendly | [2011.10566](https://arxiv.org/abs/2011.10566) (2021) |
| 60 | In-domain SSL: MAE | Advanced | `ssl_method=mae`, `mae_mask_ratio`, `mae_decoder_*` | none, 0.75 | **Plain ViT only** (drops masked tokens); 75% masking is the key hyperparameter | [2111.06377](https://arxiv.org/abs/2111.06377) (2022) |
| 60b | In-domain SSL: SimMIM | Advanced | `ssl_method=simmim`, `simmim_mask_ratio`, `simmim_mask_patch_size` | none, 0.6, 32px | **ViT and Swin**; single linear head, L1 on masked pixels | [2111.09886](https://arxiv.org/abs/2111.09886) (2022) |
| 61 | SSL data / budget | Advanced | `ssl_extra_data_dir`, `ssl_batch_size`, `ssl_num_workers`, `ssl_amp`, `ssl_input_size` | none, 32, auto, bf16, = input_size | Train split + optional unlabeled dir; all device-resolved. Batch 32 because SimSiam encodes two views: measured 10.4 GB at batch 64/224px/convnext_tiny, ~5.3 GB at 32 | — |
| 62 | SSL encoder reuse | Advanced | `ssl_save_encoder` | true | Pretrain once, fine-tune many times | — |
| 63 | Curriculum by loss | Advanced | `curriculum_by_loss`, `curriculum_start_frac`, `curriculum_epochs` | off, 0.5, 10 | Mixed evidence; costs a scoring pass per epoch | Bengio (2009) |
| 64 | Label-noise filtering | Advanced | `label_noise_filter`, `label_noise_percentile`, `label_noise_start_epoch` | off, 0.02, 5 | Drops the persistent high-loss tail; writes `dropped_as_noisy.csv` | Han (2018) |

---

<a name="deliberately-excluded"></a>
## 14. Deliberately excluded

| Technique | Why not |
|---|---|
| **LAMB / LARS optimizers** | Only useful at batch ≥1024. This tool targets single-GPU batch 32–256, where they are strictly worse than AdamW. RSB used LAMB because of an 8-GPU budget, not because it's better. |
| **Repeated augmentation** | Used by RSB and DeiT. Only pays off with very large batches and very long schedules; adds sampler complexity and *reduces* effective dataset diversity per epoch. Wrong trade at this scale. |
| **AugMix** | Designed for corruption robustness, not clean accuracy. Different objective from this tool's. |
| **AutoAugment policy search** | The *search* costs ~15 000 GPU-hours. The pre-searched ImageNet policy is exposed; running the search is out of scope. |
| **CutOut** | Strictly subsumed by Random Erasing. |
| **LayerScale / architecture edits** | Belongs to model definitions, not training config. Already baked into the relevant `timm` models. |
| **Batch-size finder / LR range test** | Genuinely useful, but it's a *separate tool* — a one-off probe, not a hyperparameter. Candidate for a later "Estimate" button rather than a config field. |
| **Quantization / pruning / export** | Deployment concerns, orthogonal to training accuracy. |
| **Multi-node / DDP orchestration** | Not exposed at all. A flag existed briefly with no launcher behind it, which was worse than absence (see §12); a control this app cannot honour does not belong in the schema. Out of scope for a local single-user tool. |
| **`nesterov` for AdamW** (NAdamW) | Marginal and confusing next to the SGD `nesterov` flag; the field is shown only when `optimizer=sgd`. |

---

## Footnotes

[^clipft]: Dong et al., "CLIP Itself is a Strong Fine-tuner", 2022 — [arXiv:2212.06138](https://arxiv.org/abs/2212.06138). Also recommends shortened epochs, LLRD, and EMA to slow the drift of pretrained low-level features.
[^deit3]: Touvron et al., "DeiT III: Revenge of the ViT", ECCV 2022 — [ECVA PDF](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840509.pdf).
[^convnext]: Fine-tuning configuration per the ConvNeXt reference training configs.
[^beit2]: Peng et al., "BEiT v2", 2022 — [arXiv:2208.06366](https://arxiv.org/abs/2208.06366).
