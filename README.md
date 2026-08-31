# TrainLab

A local web UI for configuring, launching and **comparing** PyTorch image-classification
experiments. Built for the case where you run hundreds of experiments against one dataset
and need to see which knobs actually helped.

- **The config schema is the single source of truth.** One Pydantic model
  ([`trainlab/config.py`](trainlab/config.py)) generates the form, the YAML, the tracker
  params and the sweep search space. Adding a hyperparameter is a one-line change.
- **The UI is a config generator, not a dependency.** `python train.py --config run.yaml`
  is the only training entry point and never imports the web app.
- **Runs are subprocesses.** The UI never blocks, Cancel is a real signal to a process
  group, and a crash in a CUDA/MPS kernel can't take the server down.

The technique research behind every default is in [`RESEARCH.md`](RESEARCH.md); the form
layout spec is in [`UI_LAYOUT.md`](UI_LAYOUT.md).

---

## Setup

Requires Python ≥3.10 and Node ≥18.

```bash
make setup
```

Then, in two terminals:

```bash
make up-local
```

```bash
make api
```

- **TrainLab** → http://127.0.0.1:8000
- **MLflow** → http://127.0.0.1:5050

`make up-local` runs MLflow in-process on SQLite with local artifacts — no Docker needed.
If you have Docker, `make up` brings up the full stack instead (Postgres metadata store +
MinIO artifact store, via [`docker-compose.yml`](docker-compose.yml)). Both bind to
`127.0.0.1:5050`, so nothing else changes. (5050 rather than the MLflow-conventional
5000 because macOS AirPlay Receiver listens on 5000 and 5001.)

For frontend work, `make ui` runs the Vite dev server on :5173 with hot reload, proxying
`/api` to :8000.

### Moving the ports

Every process takes its address from a Make variable, all defaulting to loopback:

| | variable | default |
|---|---|---|
| API + built UI (`make api`) | `HOST`, `PORT` | `127.0.0.1:8000` |
| MLflow (`make up`, `make up-local`) | `MLFLOW_HOST`, `MLFLOW_PORT` | `127.0.0.1:5050` |
| The tracker the app talks to (`make api`) | `TRACKING_URI` | `http://$(MLFLOW_HOST):$(MLFLOW_PORT)` |
| Vite dev server (`make ui`) | `UI_HOST`, `UI_PORT`, `API_URL` | `127.0.0.1:5173`, proxying `http://$(HOST):$(PORT)` |

```bash
make api PORT=9000
```

Moving MLflow moves the app with it — start both with the same variable and nothing else
needs saying:

```bash
make up-local MLFLOW_PORT=5555
```

```bash
make api MLFLOW_PORT=5555
```

`make api` passes `TRACKING_URI` in as `TRAINLAB_TRACKING_URI`, which becomes the schema
default for `tracking.tracking_uri` — so the form opens on the right tracker, launched runs
record there, and the Compare tab reads from there. The field still wins per run (edit it in
the form, or pass `--tracking-uri` to `train.py`), and changing it in the form redirects the
Compare tab too, not just the next run.

Serving the API off loopback needs one more thing. It rejects unexpected `Host` headers —
loopback alone does not stop DNS rebinding, and this API spawns subprocesses and reads
caller-supplied paths — so the name you browse it by has to be named:

```bash
make api HOST=0.0.0.0 TRAINLAB_ALLOWED_HOSTS=trainlab.lan
```

`make api HOST=…` passes that host through automatically, so `make api HOST=192.168.1.5`
needs nothing extra; the variable is for the case where the name in the URL differs from
the interface you bound.

---

## First run

```bash
make smoke
```

Downloads Imagenette (~95 MB) and runs two seeded ~1-minute jobs that differ in exactly
one parameter, then verifies the comparison path end to end. It exercises: config schema →
YAML round-trip → training → MLflow → varying-parameter detection → run diff → parallel
coordinates → clone.

Verified output on an Apple M3 Pro (MPS), 2 epochs at 128px with `resnet18`:

```
[4] Training run A — 2 epochs, layer_lr_decay=1.0
      ! amp bf16 -> fp16: MPS autocast does not support bf16
      ! ema_decay auto -> 0.965986 (horizon 29 of 294 steps)
      epoch 1/2  loss 0.8422  val_loss 0.2876  acc@1 0.9108  *best*  (35.2s)
      epoch 2/2  loss 0.1619  val_loss 0.1665  acc@1 0.9470  *best*  (21.2s)
[6] Comparison view
  ✓ varying parameters detected: ['optimization.layer_lr_decay']
  ✓ diff: 1 differing, 161 identical hidden
```

Once it passes, point **Data → Train directory** at your own dataset. Everything else has
a working default.

---

## Headless use

The UI writes YAML; `train.py` consumes it. You never need the UI to train.

```bash
python train.py --config runs/_configs/abc123.yaml
```

```bash
python train.py --preset balanced --train-dir ./data/train --epochs 40
```

```bash
python train.py --config base.yaml --set optimization.lr=1e-4 optimization.layer_lr_decay=0.75
```

`--print-config` resolves everything and prints the YAML without importing torch, which
makes it fast enough for scripting. `--set` takes dotted paths and is repeatable.

---

## How it fits together

```
trainlab/config.py ──► JSON Schema + x-ui ──► React form (generated, not hand-written)
        │
        ├──► YAML ──► train.py ──► trainlab/engine.py ──► TrackerAdapter ──► MLflow
        │                                                        │
        └──► warnings() ─────────────────────────────────────────┘
             (mirrored client-side for instant feedback)
```

| Path | What lives there |
|---|---|
| [`trainlab/config.py`](trainlab/config.py) | The schema. Every hyperparameter, its default, tooltip, group, and the validation warnings. |
| [`trainlab/engine.py`](trainlab/engine.py) | Training loop: mixup, AMP, EMA, SAM, grad accumulation, checkpointing, early stopping. |
| [`trainlab/transforms.py`](trainlab/transforms.py) | Augmentation pipelines (timm + torchvision v2). |
| [`trainlab/presets.py`](trainlab/presets.py) | The five presets. (The augmentation group's own none/light/medium/heavy ladder lives with the schema, in `config.py`.) |
| [`trainlab/tracking/`](trainlab/tracking/) | `Tracker` protocol + MLflow adapter. Swap backends here. |
| [`backend/app/`](backend/app/) | FastAPI: schema, preview, run supervision (SSE), registry, sweeps. |
| [`frontend/src/`](frontend/src/) | React + TS + Tailwind + shadcn-style primitives. Four tabs. |

---

## Recommended experiment order

Ordered by expected accuracy gain per unit of *your* time. Full reasoning and citations in
[`RESEARCH.md`](RESEARCH.md#11-recommended-experiment-order).

1. **Sanity check** — `Fast baseline` preset. Confirm loss decreases. Then look hard at the
   **augmentation preview** and the **worst-predictions grid**. Fix label noise before
   tuning anything; it is routinely worth more than every hyperparameter combined.
2. **Backbone and resolution** — the single biggest lever. `convnext_tiny` → `convnext_small`,
   `vit_base_patch16_224`, or a higher `input_size`.
3. **LR × weight decay** — a 3×3 random search here beats everything below it. These are the
   two parameters that genuinely need tuning per dataset.
4. **`layer_lr_decay = 0.75`** — one run. Frequently a free win when fine-tuning a large
   backbone, and widely unknown.
5. **Augmentation strength**, matched to schedule length. Short run → lighter augmentation.
   The Augmentation group's `preset` moves the whole group between none / light / medium /
   heavy in one click; `trivialaugment-wide` is the good no-tuning policy.
6. **`test_input_size` ≈ 1.15 × `input_size`** (FixRes) — costs zero training time.
7. **EMA** (watch the horizon warning) and **hflip TTA** — both nearly free.
8. **Longer schedule + heavier augmentation together** — they only pay off jointly.
9. **SAM**, if you're on a ViT and can afford 2× the time.
10. **Ensemble** your best 3–5 runs.

Everything in the Advanced group sits *after* this list, not inside it. Reach for it only
once steps 1–8 have stopped paying: cRT if your data is long-tailed, pseudo-labeling if you
have an unlabeled pool, SSL if your domain is far from ImageNet **and** you can afford
hundreds of epochs. Each costs materially more than anything above and is higher variance.

**Note on mixup/cutmix:** neither the five presets nor any rung of the augmentation ladder
enables them. The evidence for mixup is specific to 300–600 epoch from-scratch ImageNet
training; when fine-tuning a pretrained backbone for <50 epochs it is
[measurably negative](https://arxiv.org/abs/2212.06138). Both default to 0 and are one
slider away — opt-in rather than on by default, and turning either on auto-switches the loss
to `soft_target_ce` and raises a warning if your schedule is short.

---

## Comparing runs

The **Compare** tab is the reason the app exists.

- **Run table** — sortable, and it shows *only the parameters that vary across the visible
  runs*. Constant columns are hidden, which is what keeps the table readable at 100+ runs.
- **Parallel coordinates** — one axis per varying hyperparameter, final axis is the metric,
  lines coloured by metric. Drag on an axis to brush-filter; double-click to clear; click a
  line to select it.
- **Run diff** — select two runs to get *only* the differing parameters with the metric
  delta on top. When more than one parameter differs it says the comparison is
  **confounded** rather than implying causality.
- **Clone** — loads any past run's config back into the form.

Sweeps (grid / random / Optuna-TPE, with median or Hyperband pruning) tag every trial with
`sweep_id`, so sweep results land in the same three views as hand-launched runs.

---

## Advanced / experimental techniques

Everything in the Advanced group is wired into the training loop — a test
(`test_every_experimental_field_is_read_by_the_engine`) fails the build if a control is
ever added to the schema without runtime code reading it.

Stages run in this order after the main loop: **noise-filter report → pseudo-labeling →
cRT → ensembling**. Each logs under its own metric prefix (`pseudo/*`, `crt/*`,
`ensemble/*`) so you can see what each stage bought you rather than one blended number.

### In-domain self-supervised pretraining

Label-free pretraining on your **own** images before supervised training. It reuses the
train split and discards the labels, so **no extra data is required** — the point is that
cross-entropy over N classes carries at most log₂(N) bits per image (≈3.3 for 10 classes)
while an SSL objective extracts far more from the same pixels.

| | SimSiam | MAE | SimMIM |
|---|---|---|---|
| Backbone | **any** | **plain ViT** | **ViT + Swin** |
| Mechanism | 2 views → projector → bottlenecked predictor → stop-gradient | drop 75% of tokens, reconstruct with a Transformer decoder | keep all tokens (mask token at masked positions), predict via one linear layer |
| Loss | negative cosine | MSE, optionally normalised pixels | L1 on masked pixels |
| Key knob | `simsiam_pred_hidden_dim` (the bottleneck prevents collapse) | `mae_mask_ratio` (0.75) | `simmim_mask_patch_size` (32px unit) |
| Auto LR | SGD, `0.05 × bs/256` | AdamW, `1.5e-4 × bs/256` | AdamW, `1e-4 × bs/256` |

All three are fine at small batch sizes — none needs SimCLR's large-batch negatives.
Backbone support is an isinstance check, not a name heuristic. Neither masked method works
on a plain convnet: convolutions leak information across masked regions, so that needs
sparse convolutions (SparK / ConvNeXt-V2 FCMAE), which this app does not implement. Use
SimSiam there.

Set `ssl_extra_data_dir` to fold an unlabeled directory into pretraining — it is scanned
recursively, needs no class structure, and is de-duplicated against the train split:

```bash
python train.py --preset balanced --train-dir ./data/train --set experimental.ssl_method=simsiam experimental.ssl_epochs=400 experimental.ssl_extra_data_dir=./data/unlabeled
```

SimSiam and MAE are built on [`lightly`](https://github.com/lightly-ai/lightly) (heads,
losses, masking utilities, view transforms). SimMIM is not in lightly, so its masking,
linear head and L1 objective are implemented here — on top of lightly's masked-ViT wrapper
for the ViT path, so only the uncovered parts are hand-written.

⚠️ **SSL is not a "warmup".** From scratch it needs hundreds of epochs — SimSiam's
CIFAR-10 recipe is 800, MAE runs 800–1600 — and the app warns under ~100 there. From
the default `pretrained=true` it is domain adaptation instead: 20–50 epochs is the
useful range (the default is 30), and a LONG stage can pull good ImageNet weights away
faster than it adds anything back. The realistic workflow is **pretrain once, fine-tune
many times**: `ssl_save_encoder` (on by default) logs the encoder as a run artifact for
reuse.

### The other four

- **Classifier retraining (cRT)** — freezes the backbone, reinitialises the head, retrains
  it with class-balanced sampling at 10× the base LR. The backbone stays in eval mode so
  BatchNorm statistics don't drift while only the head is meant to be learning.
- **Pseudo-labeling** — predicts on an unlabeled directory, keeps predictions above
  `pseudo_label_threshold`, caps them at `pseudo_label_max_ratio ×` the labelled set, and
  continues training. *Deviates from Noisy Student*: it continues the same model rather
  than training a fresh larger student each round — much cheaper, but gives up some of the
  published gain and is more prone to confirmation bias.
- **Curriculum by loss** — trains on the easiest `curriculum_start_frac` first, growing to
  the full set by `curriculum_epochs`.
- **Label-noise filtering** — drops the persistently highest-loss `label_noise_percentile`
  after `label_noise_start_epoch`, and writes **`dropped_as_noisy.csv`** as an artifact.
  Read that file — the point is to inspect what it threw away.

Curriculum and noise filtering both rank by per-sample loss, computed in a **separate
clean scoring pass** with eval transforms (no augmentation, no mixup) each epoch. That
costs roughly a third of an epoch but makes the ranking meaningful — under mixup the
training loss belongs to a blended pair and carries no per-sample signal. Enabling both at
once triggers a warning: together they can starve the model of every hard example.

- **Ensembling** — `ensemble_run_ids` accepts MLflow run IDs *or* local checkpoint paths.
  Probabilities are averaged (not logits), and members trained on a different class list
  are refused rather than silently misaligned.

## Extending it

### Add a hyperparameter

One line in the relevant group in [`trainlab/config.py`](trainlab/config.py):

```python
my_knob: float = Field(
    0.5, ge=0.0, le=1.0,
    **ui(label="My knob", advanced=True, step=0.05,
         tooltip="What it does, and when you'd want to change it."))
```

The form control, YAML field, tracker param and sweep search space all appear
automatically. Read it in the engine as `cfg.<group>.my_knob`. Two tests enforce that every
field carries UI metadata and that tooltips are substantive enough to be worth showing.

**If you rename or remove a field**, two things do *not* update themselves:

1. **`showIf` / `disableIf` strings** reference fields by bare name through a flat
   namespace, resolved at render time. `validate_ui_expressions()` checks them — it runs
   in `test_every_ui_expression_references_a_real_unambiguous_field` and again at API
   startup, which prints any problem to stderr.
2. **Stored configs.** `extra="forbid"` means every run recorded before the rename fails
   to load. Add an entry to `_LEGACY_SSL_FIELDS` (if the field was renamed and kept its
   meaning) or `_LEGACY_DROPPED_FIELDS` (if the control was replaced by a different
   design, so aliasing it would fabricate a value). Cloning uses
   `load_config_leniently`, which drops still-unknown keys and reports them, so history
   stays loadable even when a migration entry is missed.

### Add a backbone

Nothing to do — the picker searches the entire `timm` catalogue (~1400 architectures) and
`create_model` falls back to torchvision for anything timm doesn't have. To register a
custom architecture, use timm's `@register_model` decorator and import it in
[`trainlab/models.py`](trainlab/models.py).

### Add an augmentation

Add the enum value to `AutoAugment` (or a new field) in `config.py`, then handle it in
`build_train_transform` in [`trainlab/transforms.py`](trainlab/transforms.py). RandAugment,
AutoAugment and 3-Augment route through `timm`; TrivialAugmentWide comes from
`torchvision.transforms.v2` because timm has no implementation of it. The live preview
picks up new ops with no extra work.

### Swap the experiment tracker

Implement the `Tracker` protocol in [`trainlab/tracking/base.py`](trainlab/tracking/base.py)
(eight methods) and return it from `build_tracker`. Training code depends only on the
protocol. The Compare tab reads through [`backend/app/registry.py`](backend/app/registry.py),
which would need a matching implementation.

---

## Notes on this machine (Apple Silicon)

The defaults assume CUDA; the runtime resolves them per device and logs every downgrade to
both the UI and the tracker, so a run record reflects what actually executed:

- **`amp: bf16` → `fp16` on MPS.** MPS autocast [disables itself for bf16](https://github.com/pytorch/pytorch/issues/139386)
  and lacks optimized bf16 Metal kernels.
- **`channels_last` is ignored on MPS** — no meaningful effect there.
- **`torch.compile`** support on MPS is partial; treat it as opt-in.
- **Multi-GPU/DDP is not implemented.** The former `distributed`/`num_gpus` controls
  were declared with no runtime behind them (while still distorting the effective
  batch size the config layer reasoned about) and have been removed; old configs
  that carry them still load, with the fields dropped and reported.

---

## Testing

```bash
make test
```

205 tests — 153 Python, 52 frontend — none of which need a dataset or a GPU. Under a
minute total.

| File | Covers |
|---|---|
| `tests/test_config.py` | Schema, YAML round-trip, derived values, every warning, all five presets |
| `tests/test_correctness.py` | Metric direction, train/val vocabulary agreement, loader edge cases, focal weighting, legacy migration, schema↔runtime consistency |
| `tests/test_engine.py` | Real training runs on a 32px synthetic set: checkpoint selection, top-K retention, zero-step detection, SAM, checkpoint safety, progressive resizing with real workers, the mixup-off epoch boundary, SWA phase guard, SSL end-to-end with default settings |
| `tests/test_backend.py` | Run supervision against real subprocesses, SSE back-pressure, sweep objective, pruning reports raw values, SIGTERM recorded as cancellation, diff accounting |
| `tests/test_ssl.py` | SimMIM/MAE tensor geometry and weight transfer (pins timm's internal structure), register-token ViT refusal, smaller-than-one-batch pretraining |
| `frontend/src/lib/expr.test.ts` | The `showIf`/`disableIf` grammar, including the forms that must fail loudly |
| `frontend/src/components/ParallelCoordinates.test.ts` | Plot degenerate cases and metric-direction colouring |

Run one side only with `make test-py` or `make test-ui`.

**What the tests are for.** Most of them pin a specific way the app used to produce a
plausible-looking but wrong number — a metric optimised in the wrong direction, a
validation set whose labels meant something different from the training set's, an epoch
that took no optimizer step. Each test names that symptom in its docstring, so a
regression reports what broke rather than which assert failed.

Two of them are consistency checks rather than behaviour tests, and are the ones most
worth keeping green:

* `test_every_ui_expression_references_a_real_unambiguous_field` — the `showIf`/`disableIf`
  strings are the one part of the schema that is not type-checked. A renamed field leaves
  the dependent control permanently hidden with nothing to indicate why.
* `test_ssl_prefix_lists_match_runtime_support` — `config.py` gates SSL on backbone name
  prefixes (so the schema can warn without importing timm) while `ssl.py` gates on
  `isinstance`. When those disagree the form says the config is fine and the run dies at
  model construction.
