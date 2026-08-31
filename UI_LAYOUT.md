# UI layout spec — for approval before build

The form is **generated from `backend/app/schema.py`**, not hand-written. The backend serves
`GET /api/schema` (JSON Schema + the `x-ui` block on every field) and the React form renders
groups, widgets, tooltips, `showIf`/`disableIf` and the advanced split from that. Adding a
hyperparameter is a one-line change in Python; the form, YAML, tracker params and sweep
search space all pick it up.

---

## 1. Screen layout

```
┌───────────────────────────────────────────────────────────────────────────────────────┐
│  TrainLab        [Fast baseline] [Balanced ✓] [Max accuracy] [Small data] [Imbalanced]│
│                                    ⟳ Reset all   ↓ Export YAML   ↑ Import   ▶ Train   │
├─────────────────────────────────────────────┬─────────────────────────────────────────┤
│  CONFIG  (scrollable, ~62%)                 │  RIGHT RAIL (sticky, ~38%)              │
│                                             │                                         │
│  ▼ Data                          ⟳ 2 mod.   │  ┌── Augmentation preview ───────────┐  │
│      [ ] Advanced                           │  │  [img][img][img][img]             │  │
│      Train dir      ▸ /data/train        ⓘ  │  │  [img][img][img][img]   ⟳ resample│  │
│      Val dir        ▸ (auto 10% split)   ⓘ  │  │  live, debounced 300 ms           │  │
│      Classes        37 (read-only)          │  └───────────────────────────────────┘  │
│      Sampler        [random        ▾]  ●    │                                         │
│                                             │  ┌── Validation ─────────────────────┐  │
│  ▶ Input & preprocessing    224px · bicubic │  │ ⚠ Mixup with 30 ep of fine-tuning │  │
│                                             │  │   is measurably negative.  [Fix]  │  │
│  ▶ Augmentation    RandAugment(2,9), Mixup  │  │ ⚠ EMA horizon 5,000 > run 2,370   │  │
│                    0.2, RErase 0.25, +2 ▲   │  │   steps.                   [Fix]  │  │
│                                             │  │ ⚠ rrc_scale 0.08 on 5k images     │  │
│  ▼ Model / backbone           convnext_tiny │  │ ℹ bf16 → fp16 on MPS              │  │
│      [ ] Advanced                           │  └───────────────────────────────────┘  │
│      Backbone     [🔍 convnext_tiny      ]  │                                         │
│                    28.6M · 224px · in1k ✓   │  ┌── Estimates ──────────────────────┐  │
│      Pretrained   [✓]                       │  │ VRAM   ~4.1 GB / 18 GB      ▓▓░░░ │  │
│      Drop path    [0.1        ] ────●────   │  │ Epoch  ~48 s   Total ~24 min      │  │
│      EMA          [✓]  decay 0.9998   ⚠     │  │ Steps  2,370                      │  │
│                                             │  └───────────────────────────────────┘  │
│  ▼ Optimization        adamw · 3e-4 · wd.05 │                                         │
│  ▶ Loss             soft_target_ce (auto) ⓘ │  ┌── Recent runs ────────────────────┐  │
│  ▶ Schedule & runtime      30ep · bs64 · bf16│ │ ● convnext_t·224·mixup   82.4  ⧉  │  │
│  ▶ Validation & metrics          acc@1 ▸ 3  │  │ ● convnext_t·224·noaug   81.1  ⧉  │  │
│  ▶ Checkpointing & early stopping           │  │ ○ vit_b16·224·randaug  running... │  │
│  ▶ Experiment tracking             mlflow   │  └───────────────────────────────────┘  │
│  ▶ Advanced / experimental  ⚗ higher variance│                                        │
└─────────────────────────────────────────────┴─────────────────────────────────────────┘
```

Tabs across the top of the whole app: **Configure** · **Runs** · **Compare** · **Sweeps**.

---

## 2. Group behaviour

| Group | On load | Basic fields (always visible) | Behind "Advanced" |
|---|---|---|---|
| 1. Data | **expanded** | format, train_dir, val_dir, val_split, num_classes, sampler | split_strategy, group_column, oversample_factor, num_workers, pin_memory, persistent_workers, prefetch_factor, cache_mode |
| 2. Input & preprocessing | collapsed | input_size, train_resize_policy, rrc_scale, test_input_size | rrc_ratio, val_resize_policy, val_crop_pct, interpolation, antialias, normalization, custom mean/std, channels, channels_last |
| 3. Augmentation | collapsed | preset, hflip, vflip, rotation, color_jitter, auto_augment, randaug N/M, erasing_p, mixup_alpha, cutmix_alpha, mixup_prob | randaugment_mstd, erasing_mode, switch_prob, mixup_off_epoch, grayscale_p, blur_p |
| 4. Model / backbone | **expanded** | backbone, pretrained, drop_rate, drop_path_rate, freeze_policy, freeze_epochs, ema | pretrained_source, pretrained_checkpoint, global_pool, head_init_scale, freeze_stages, freeze_bn, torch_compile, ema_decay, gradient_checkpointing |
| 5. Loss | collapsed | loss, label_smoothing, focal_gamma, class_weights, logit_adjust_tau | manual_class_weights, cb_beta |
| 6. Optimization | **expanded** | optimizer, lr, weight_decay, layer_lr_decay, sam | lr_scaling_rule, no_wd_on_norm_bias, betas, eps, momentum, nesterov, head_lr_mult, sam_rho, sam_adaptive, grad_clip_norm, grad_accum_steps |
| 7. Schedule & runtime | collapsed | epochs, batch_size, scheduler, warmup_epochs, amp, seed | warmup_start_lr, min_lr, deterministic, device, progressive_* |
| 8. Validation & metrics | collapsed | metrics, primary_metric, tta, eval_ema_weights, log_confusion_matrix, log_worst_predictions | eval_every_n_epochs |
| 9. Checkpointing | collapsed | save_top_k, save_last, early_stopping, es_patience, output_dir | es_min_delta, resume_from |
| 10. Experiment tracking | collapsed | enabled, experiment_name, run_name | backend, tracking_uri, tags, log_augmentation_preview |
| 11. Advanced / experimental | collapsed, ⚗ badge | ssl_method + its revealed sub-controls, kd_enabled, swa, pseudo_label_dir, classifier_retrain_epochs, curriculum_by_loss, label_noise_filter, ensemble_run_ids | tuning sub-parameters of each (ssl_lr/wd/workers/amp, simsiam_proj/out/fix_pred_lr, mae_decoder_*, pseudo_label_rounds, crt lr/reinit, …) — 19 of the group's 44 controls |

**Progressive disclosure inside the group.** The experimental group is the one place where
a single choice unlocks a whole sub-form, so it leans hardest on `showIf`. Picking an SSL
method reveals the shared budget controls (`ssl_epochs`, `ssl_extra_data_dir`,
`ssl_batch_size`, …) plus *only* that method's parameters — `simsiam_pred_hidden_dim` for
SimSiam, `mae_mask_ratio` + decoder shape for MAE, `simmim_mask_ratio` +
`simmim_mask_patch_size` for SimMIM. With `ssl_method=none` all of them stay hidden. Same pattern for `classifier_retrain_epochs > 0`, `pseudo_label_dir`,
`curriculum_by_loss` and `label_noise_filter`.

**Rule that keeps it non-intimidating:** with the three default-expanded groups showing only
basic fields, the form is **14 controls on first paint**. Everything else is one click away.

### Collapsed-header summaries
Each collapsed header renders its non-default values ordered by `summaryPriority`, truncated
to fit one line with a `+N more` affordance:

```
▶ Augmentation      RandAugment(2,9), Mixup 0.2, RErase 0.25, +2 more ▲
▶ Optimization      adamw · lr 3e-4 · wd 0.05 · LLRD 0.75
▶ Schedule & runtime  30ep · bs 64 · cosine · bf16
▶ Loss              soft_target_ce (auto) · ls 0.1
```
When a group is fully default the header shows a muted `defaults` instead.

### Per-field affordances
- **Modified marker** — a small amber dot `●` left of the label plus a left border on the
  input. Click it to revert that one field.
- **Tooltip** — `ⓘ` on hover/focus, sourced from the `tooltip=` in the schema. Every one says
  what it does *and when to change it*.
- **Reset group** — `⟳ N modified` in the group header; disabled at 0.
- **Auto-set fields** (`loss` under mixup, `test_input_size`) render read-only with an inline
  `(auto)` chip and a one-line explanation, overridable by clicking the chip.

---

## 3. Presets

Applying a preset repopulates the whole form and tags the run. Deltas from schema defaults:

| | Fast baseline | Balanced *(default)* | Max accuracy | Small data | Imbalanced |
|---|---|---|---|---|---|
| backbone | `resnet18` | `convnext_tiny` | `convnext_base` | `convnext_tiny` | `convnext_tiny` |
| input_size | 160 | 224 | 224 (test 256) | 224 | 224 |
| epochs | 3 | 30 | 120 | 40 | 40 |
| batch_size | 64 | 64 | 32 | 32 | 64 |
| lr / wd | 1e-3 / 0.01 | 3e-4 / 0.05 | 2e-4 / 0.05 | 1e-4 / 0.1 | 3e-4 / 0.05 |
| layer_lr_decay | 1.0 | **0.75** | 0.8 | **0.65** | 0.75 |
| auto_augment | none | randaugment(2,7) | randaugment(2,9) | trivialaugment-wide | randaugment(2,7) |
| mixup / cutmix | 0 / 0 | **0 / 0** ⚠ | **0 / 0** ⚠ | 0 / 0 | 0 / 0 |
| random_erasing_p | 0 | 0.1 | 0.25 | 0.25 | 0.1 |
| rrc_scale | 0.65–1.0 | 0.65–1.0 | 0.4–1.0 | 0.7–1.0 | 0.65–1.0 |
| drop_path | 0.0 | 0.1 | 0.3 | 0.2 | 0.1 |
| label_smoothing | 0.0 | 0.1 | 0.1 | 0.1 | 0.1 |
| loss | cross_entropy | cross_entropy | soft_target_ce | cross_entropy | **logit_adjusted** |
| head_init_scale | 1.0 | 1.0 | 1.0 | **0.001** | 1.0 |
| ema | off | on (auto decay) | on (auto decay) | on (auto decay) | on (auto decay) |
| tta | none | none | **hflip** | hflip | hflip |
| primary_metric | acc@1 | acc@1 | acc@1 | acc@1 | **macro-F1** |
| metrics | acc@1 | +acc@5, macro-F1 | +ECE | +macro-F1 | +per-class recall, balanced acc |
| early_stopping | off | on (p=10) | off | on (p=15) | on (p=15) |

⚠ **No preset enables mixup/cutmix.** Per RESEARCH.md §0/§12, the evidence for mixup is
specific to 300–600 epoch from-scratch training, and it is measurably negative when
fine-tuning a pretrained backbone on a short schedule. Mixup is therefore **opt-in only**:
every preset zeroes it, and so do the schema defaults, because the headless
`python train.py --train-dir X` path has no preset button and no warning panel to catch it —
the reasoning is spelled out at the field in `config.py`. Turning either one on auto-switches
the loss to `soft_target_ce` and raises a warning if `epochs < 50` and `pretrained` is set.

### The tag is derived, not remembered

`tracking.preset` says which preset a run came from, and the Compare tab filters and groups
by it — so it describes the config rather than the last button pressed. Change anything a
preset names and both the tag and the bar's highlight fall back to `custom`; click the preset
again and it comes back. The server applies the same rule to whatever it is handed
(`TrainConfig._demote_stale_preset`), so `--preset balanced --epochs 100` is tagged `custom`
too, and no hand-written YAML can claim a preset it has stopped matching.

Exempt from that comparison, listed in `PRESET_IDENTITY_FIELDS`: the dataset directories, the
output directory, the auto-detected `num_classes`/`class_names`, and the whole tracking group.
Those say where a run lives rather than how it trains — two people running Balanced on their
own data are both running Balanced — so applying a preset preserves them, and editing one
never costs the tag.

### Augmentation preset — the strength ladder

The Augmentation group carries its own preset, independent of the five above: it sets that
one group and nothing else. Selecting a rung **writes its values into the controls** — the
form applies the table served by `GET /api/presets`, and `AugmentationConfig` applies the
same table server-side, so a hand-written `preset: heavy` in YAML expands identically.
Touching any individual control demotes the label to `custom`, in the form and on the server
alike, because the label is what runs get tagged and compared by.

| | none | light | medium *(default)* | heavy |
|---|---|---|---|---|
| hflip | 0 | 0.5 | 0.5 | 0.5 |
| color_jitter | 0 | 0 | 0.4 † | 0.4 † |
| auto_augment | none | none | randaugment(2,7) | randaugment(2,9) |
| random_erasing_p | 0 | 0 | 0.1 | 0.25 |
| vflip / rotation / grayscale / blur | 0 | 0 | 0 | 0 |
| mixup / cutmix | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

† Inert while an auto-augment policy is active — the policy already contains colour ops.

`none` is resize/crop only (the crop itself lives in the Input group), which makes it the
honest baseline for measuring what augmentation is worth on your data. The ladder moves only
the *tuned* knobs: `vflip`, `rotation`, `grayscale` and `blur` each assert something about
the domain rather than a strength, so no rung switches them on behind your back. `medium`
must equal the group's schema defaults — the default label is `medium`, and an untouched
config may not contradict its own label.

The five presets above name a rung instead of listing augmentation values. `small-dataset` is
the one that reports `custom`: it deliberately deviates, wanting heavy strength with the
tuning-free policy.

---

## 4. Validation panel

Three severities, rendered in the sticky right rail, each with a one-click **[Fix]** applying
the suggested value. Client-side mirror of `TrainConfig.warnings()` so it updates as you type;
the server copy is authoritative at launch. Currently implemented checks:

*Errors (block Train):* mixup with a non-soft-target loss that can't auto-switch · warmup ≥ epochs ·
freeze_epochs ≥ epochs · KD enabled without a teacher.

*Warnings:* AdamW with lr > 1e-2 · SGD with lr < 1e-3 · Lion with an AdamW-sized lr · mixup on a
< 50-epoch pretrained fine-tune · EMA horizon > 50% of total steps · more than one imbalance
correction stacked · focal loss on near-balanced data · acc@1 as primary metric at > 10:1
imbalance · rrc_scale < 0.3 on a < 20k-image dataset · torch.compile + progressive resizing ·
torch.compile on MPS.

*Info:* bf16 → fp16 downgrade on MPS · channels_last ignored on MPS · AMP off on CPU · vflip /
large rotation assumptions · long schedule with no augmentation · high RandAugment on a short
schedule · label smoothing degrading this run as a future teacher · noisy BN at batch < 16 ·
SAM doubling runtime.

**OOM estimate:** `activation ≈ batch × (input_size/224)² × per-backbone coefficient`, calibrated
from a one-time measured table of timm backbones, then adjusted for AMP (×0.55), gradient
checkpointing (×0.6) and grad-accum. Shown as a bar against detected device memory. It is a
*heuristic*, labelled as such — not a promise.

---

## 5. Compare tab — the reason the app exists

Three linked panels over the same filtered run set:

1. **Run table** — sortable/filterable; columns are primary metric, all secondary metrics, and
   *only those hyperparameters that vary across the visible set* (constant columns auto-hide,
   which is what makes the table readable at 100+ runs). Row actions: `⧉ Clone`, `⊕ Compare`,
   `⚑ Tag`, `↓ YAML`.
2. **Parallel coordinates** — one axis per varying hyperparameter, final axis = primary metric,
   lines colored by metric. Brush any axis to filter; the table and diff follow the brush.
   Categorical axes get stable ordinal encodings, log axes for lr/wd.
3. **Run diff** — pick two runs, get *only the differing parameters* side by side with the
   metric delta on top:
   ```
   Δ acc@1   +1.34   (82.41 → 83.75)
   ─────────────────────────────────────────────────
   layer_lr_decay          1.0    →   0.75      ← likely cause
   optimization.lr        3e-4    →   2e-4
   ─────────────────────────────────────────────────
   47 identical parameters hidden                [show]
   ```
   With >1 differing parameter it says *"2 parameters differ — this comparison is confounded"*
   rather than implying causality.

**Clone this run** loads a past config back into the form, marks it `cloned-from: <run_id>`,
and highlights fields differing from the current preset.

---

## 6. Sweeps

Any numeric or enum field gets a `⋯` menu → **Make search space**, turning the input into a
`list` / `range(min,max,step)` / `log-range(min,max)` / `choice[...]` editor. The sweep overlay
is stored separately from `TrainConfig` (as `SweepConfig.parameters: dict[dotted_path, SearchSpace]`),
so a plain config stays a plain YAML file and `train.py` never needs to know sweeps exist.

Launch bar: **grid | random | Optuna-TPE**, run budget, max concurrency, optional
median/Hyperband pruning. Each trial is a normal tracked run tagged `sweep_id`, so sweep
results appear in the same table, parallel-coordinates and diff views as hand-launched runs.

---

## 7. Stack

- **Backend** FastAPI + Pydantic v2. Training is a **subprocess** (`python train.py --config <yaml>`)
  supervised by the API, streaming stdout over SSE — so the UI never blocks, runs are cancellable
  with a real SIGTERM, and a crashed trainer cannot take the server down.
- **Frontend** React + TypeScript + Vite + Tailwind + shadcn/ui. Chosen over Gradio/Streamlit
  specifically for this brief: your collapsible-group/summary-header/modified-marker UX is
  fighting Streamlit's rerun-everything model, and the parallel-coordinates + brushing view
  is not something Gradio does well. The parallel-coordinates plot is hand-rolled SVG
  rather than a charting library, because brushing has to drive selection in both the run
  table and the diff view and owning the scales keeps that coupling simple — it also keeps
  the bundle at ~120 kB gzipped.
- **Model zoo** `timm` (≈1.0.28, ~1400 pretrained backbones), torchvision as fallback.
  TrivialAugmentWide comes from `torchvision.transforms.v2` since timm has no implementation.
- **Metrics** `torchmetrics`. **Sweeps** `optuna`. **Transforms** torchvision v2 + timm.
- **Tracking** MLflow 3.x self-hosted via docker-compose (Postgres backend store + MinIO
  artifact store), behind a `TrackerAdapter` protocol so W&B/ClearML/Aim can be swapped in
  without touching training code. Agreeing with your default: local, open, no account gate,
  and params+metrics+artifacts+registry in one server.
- **Headless contract** `train.py --config run.yaml` is the only training entry point. The UI
  writes YAML and shells out; it is a config generator, not a dependency.
