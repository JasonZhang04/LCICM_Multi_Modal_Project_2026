# Multi-Modal Aortic-Dilation Project — System Overview, Results & Analysis

> **Purpose of this document.** A single, careful walk-through of *how the current
> system actually works* — the data it uses, the preprocessing, how data flows through
> files and models, what the latest results are, and what they mean — followed by an
> honest list of **bugs / concerns** and a prioritized set of **concrete ideas to improve
> the results**. Written for someone who wants to fully re-understand the pipeline before
> changing it.
>
> Author: Claude (code read-through), 2026-07-16. Based on the state of `master` at commit
> `39b8c75` ("Add v3 multimodal pipeline and M0-M3 rigor/evaluation foundation").
>
> Abbreviations are expanded on first use; there is also a **glossary at the end**.

---

> **UPDATE 2026-07-16:** the top improvement ideas below have now been tested — see
> [notes/experiments_2026-07-16.md](experiments_2026-07-16.md). Headline: PCA-reducing the frozen
> embeddings is a large, robust win, and a multi-instance CXR learner + EHR ensemble
> **significantly beats even the full-cohort EHR floor for the ascending aorta** (the positive
> result the project was missing). New repo scripts: `scripts/train_reduced_fusion.py`,
> `scripts/extract_cxr_anatomy_rois.py`, `scripts/extract_raddino_anatomy_roi_embeddings.py`.

## 0. TL;DR (read this first)

**The task.** Predict how dilated a patient's aorta is — at two sites, the **aortic root
(sinus of Valsalva)** and the **ascending aorta** — from three "cheap" modalities the
patient already has: a **12-lead ECG**, a **chest X-ray (CXR)**, and **basic EHR /
demographics** (age, sex, height, weight, BMI, BSA, blood pressure). The ground-truth
label is the aortic diameter (cm) measured on a real **echocardiogram (echo)**. Echo is
used **only as the label source**, never as a model input.

**The honest headline.** The models *work* (the pipeline runs, produces calibrated
predictions, and beats chance), but **the multimodal signal is weak** and the current
best model **does not convincingly beat a plain age/sex/body-size baseline** for the
primary endpoint.

**The nuance that matters (and is currently hidden by the reporting).** When you compare
fairly — every model trained on the *same* 522 patients — imaging **does** add value for
the **ascending aorta** (CXR-alone AUROC 0.69 vs EHR-alone 0.56), which is physiologically
sensible (the ascending aorta forms part of the visible mediastinal silhouette on a CXR).
The reason the headline says "fusion loses to EHR" is that the EHR "floor" is trained on
**~4,525 patients** while the fusion models are trained on only **522** — an 8× data
advantage, not a modeling advantage. See [§8](#8-results-what-the-numbers-actually-say).

**The single most actionable finding.** Reducing the 1,088-dim frozen embeddings with PCA
before the tree model jumped AUROC from **0.63 → 0.70 (root)** and **0.68 → 0.71 (asc)**.
The embeddings are drowning the tiny sample. **Dimensionality is the bottleneck, not the
modality.** See [§10](#10-opportunities--innovations-prioritized).

---

## 1. The clinical problem & how it is framed as ML

**Aortic dilation (AD)** = the aorta is wider than normal; at the extreme it becomes an
**aneurysm** that can dissect or rupture. It is usually caught by echo or CT, which are
expensive/limited. The scientific bet of this project: *can we screen for AD opportunistically
from data a patient already has (ECG + CXR + EHR)?*

Two anatomical **sites** are predicted independently, each with its own heads:

| Site | Column in labels | Clinical name |
|---|---|---|
| `root` | `sinus_diam_cm` | Aortic root / sinus of Valsalva |
| `asc`  | `ascending_diam_cm` | Ascending aorta |

Each site's continuous diameter (cm) is turned into **four framings of the same number**:

- **Regression** — predict the diameter in cm directly.
- **Binary ≥ 4.0 cm** — "any AD". **This is the PRIMARY endpoint.**
- **Binary ≥ 4.5 cm** — "moderate+", the clinically actionable 3-class boundary.
- **Ordinal grade {0,1,2,3}** — cuts at 4.0 / 4.5 / 5.0 cm (normal / mild / moderate / severe),
  modeled ordinally (see CORAL, [§6.3](#63-deep-fusion-v3fusionmodel)).

Defined in `src/multimodal_aorta/data/targets.py`.

### The brutal reality of the label distribution

The whole project is shaped by this table (from the plan + confirmed in `event_counts.csv`):

| Endpoint | root positives / 522 | asc positives / 513 |
|---|---|---|
| ≥ 4.0 cm (any AD) | **32** (6.1%) | **48** (9.4%) |
| ≥ 4.5 cm (moderate+) | **8** | **10** |
| ≥ 5.0 cm (severe) | **1** | **2** |

With **1–2 severe cases**, "severe" is statistically un-learnable and un-evaluable. This is
why: (a) the ordinal/CORAL framing is used (it shares statistical strength across cutpoints),
(b) the primary endpoint is the ≥4.0 binary, and (c) evaluation uses **5-fold cross-validation
+ bootstrap confidence intervals** instead of a single test split. This is a *correct*
response to the small-n reality.

---

## 2. Data sources

Large datasets are **not** in the repo; they live on `/scratch4/rsteven1/` and are referenced
by absolute path in `src/multimodal_aorta/configs/default_config.py`.

| Modality | Source dataset | What is used |
|---|---|---|
| **Label** (echo) | MIMIC-IV-Echo structured measurements | `aortic_labels.csv` — one row/patient: `sinus_diam_cm`, `ascending_diam_cm`, `any_dilation`. 4,579 patients. |
| **ECG** | MIMIC-IV-ECG (matched subset, ~800k records, 161k patients) | 12-lead, 10 s @ 500 Hz waveforms (`.hea`/`.dat`, read via `wfdb`). |
| **CXR** | MIMIC-CXR-JPG (+ DICOM fallback) | Frontal chest X-rays; best view PA > AP chosen. |
| **EHR** | MIMIC-IV v3.1 `hosp` module | `patients.csv.gz` (age/sex) + `omr.csv.gz` (height/weight/BMI/BP). |
| **Echo imaging** (Track 2, parallel) | `jooyoung/mimic_4_echo` DICOM | PanEcho embeddings — see [§11](#11-track-2-echo-only-model-parallel-not-the-main-line). |

**Cohort sizes** (single most important structural fact):

- **4,579** patients have an echo aortic label.
- **~2,874** of those also have an ECG in window.
- **~522** have **all of** ECG + CXR + label → the **"triple cohort"**, which is what every
  v3 multimodal model trains and is evaluated on.
- The CXR is the binding constraint: only ~550 of the labelled patients have a chest X-ray.

The triple cohort (`pretrained_checkpoints/cohort_triple.csv`, 522 rows) is the beating heart
of the project. **Its small size is the dominant difficulty in everything that follows.**

---

## 3. How data flows through the system (the map)

The v3 pipeline is a **two-phase, offline-embedding design**. Heavy backbones are run
**once, offline**, cached as vectors, and every experiment then trains a *tiny* model on
those frozen vectors. This is deliberate — with only 522 patients you cannot fine-tune a
44M-parameter vision transformer without catastrophic overfitting.

```
                          ┌─────────────────────── PHASE 1: build once (offline) ───────────────────────┐

  aortic_labels.csv ─┐
  MIMIC-IV-ECG ──────┤   build_cohort()          ┌─ extract_pclr_embeddings.py ──►  pclr_embeddings.pt      {sid: (320,)}
  MIMIC-CXR ─────────┼──► (dataset.py)  ──────────┼─ extract_raddino_embeddings.py ► raddino_embeddings.pt   {sid: (768,)}
  MIMIC-IV hosp ─────┘   temporal + view match    ├─ extract_raddino_roi_embeddings ► raddino_roi_embeddings.pt
                          │                        └─ extract_ehr_features.py ──────►  ehr_features.csv        (12 features)
                          │
                          ├─ build_triple_cohort.py  ──►  cohort_triple.csv   (522 rows + grade columns)
                          └─ build_fold_assignments.py ─► fold_assignments.csv (immutable 5-fold split)

                          └────────────────────────────────────────────────────────────────────────────┘

                          ┌─────────────────────── PHASE 2: train tiny models (fast) ───────────────────┐
   frozen vectors  ──────►  train_gbdt_fusion.py       (tree, early fusion)
   + cohort        ──────►  train_deep_fusion.py        (MLP + CORAL heads)
   + folds         ──────►  train_late_fusion_v2.py     (stacking / meta-learner)   } each writes:
                   ──────►  train_ehr_baseline.py       (the "floor")                 outputs/<model>/
                   ──────►  train_residual_fusion.py    (imaging on top of clinical)  ├─ results.json
                   ──────►  train_cxr_roi_compare.py    (whole-image vs ROI)          └─ oof_predictions.csv  ◄─┐
                                                                                                                │
   make_eval_report.py  ◄──────────────── reads every outputs/*/oof_predictions.csv ◄─────────────────────────┘
        └──►  outputs/eval_report/{metrics_summary.csv, deltas_vs_ehr.csv, calibration_metrics.csv, plots/}
                          └────────────────────────────────────────────────────────────────────────────┘
```

The key data contract that ties it together is the **standardized per-patient OOF file**
(`oof_predictions.csv`, schema in `training/oof_io.py`): every model, regardless of type,
emits one prediction per patient tagged with its immutable fold id. `make_eval_report.py`
then aligns all models **patient-for-patient** and does paired comparisons. This is a
genuinely good piece of engineering — it makes the "does model A beat model B on the *same*
patients" question answerable rigorously.

**"OOF" = out-of-fold.** Each patient's prediction always comes from a model that never saw
that patient in training. It is how you get an honest, leakage-free prediction for *every*
patient using cross-validation, instead of throwing 80% of the data away in a single test split.

---

## 4. Cohort construction (`data/dataset.py::build_cohort`)

This is the most intricate data code. Steps:

1. **Load & QC labels** (`_load_echo_labels`): read `aortic_labels.csv`; auto-detect mm-vs-cm
   (if median > 20, divide by 10); drop rows where both sites are NaN; **clip implausible
   values to NaN** (root outside 1.5–6.0 cm, asc outside 1.5–7.0 cm).
2. **Attach an echo date**: joined from `structured-measurement.csv` by taking the
   **minimum `measurement_datetime` per patient**. ⚠️ *This is a soft spot — see
   [§9, Bug #2](#9-bugs-concerns--things-that-are-quietly-wrong).*
3. **Index ECG & CXR** by patient. The CXR index is cached in `cxr_metadata_cache.csv` and
   pulls `ViewPosition` + `StudyDate` from the official MIMIC-CXR metadata CSV.
4. **Temporal + view matching** (per patient):
   - **ECG**: pick the ECG **closest in time** to the echo date, within `max_days_offset`
     (default **±180 days**). (`_select_best_ecg`)
   - **CXR**: pick the best **view** (PA=0 > AP=1 > other=2), breaking ties by closeness
     in time, within ±180 days. (`_select_best_cxr`)
5. **`cohort_mode`**: `"triple"` keeps only patients with **both** ECG and CXR → the 522.
   (`"any"` is the older, deprecated v2 behavior.)

The ±180-day window is a real modeling assumption: it treats an ECG/CXR taken up to ~6 months
from the echo as representing the same aortic state. For a slowly-changing structural feature
this is reasonable, but it *does* inject label noise.

---

## 5. Preprocessing, per modality

### 5.1 ECG
Two different preprocessing paths exist, which is a subtle inconsistency:

- **Supervised path** (`data/preprocessing.py::load_ecg`): resample to 500 Hz → center-crop/pad
  to **5000 samples (10 s)** → robust per-lead z-score (clip ±4σ, then standardize). Output
  `(12, 5000)`.
- **PCLR embedding path** (`scripts/extract_pclr_embeddings.py::preprocess_for_pclr`): resample
  to 500 Hz → **`np.interp` the whole recording to 4096 samples** → **no amplitude
  normalization**. Output `(4096, 12)`, fed to the PCLR Keras model.

Because the v3 pipeline uses the **frozen PCLR embeddings**, the second path is the one that
matters, and its lack of normalization + full-length interpolation is worth auditing
([§9, Bug #4](#9-bugs-concerns--things-that-are-quietly-wrong)).

**PCLR** = "Patient Contrastive Learning of Representations" — a published self-supervised ECG
encoder that maps a 12-lead ECG to a 320-dim vector. Here it is a *frozen feature extractor*:
run once, never trained.

### 5.2 CXR
`data/preprocessing.py::load_cxr`: load PNG or DICOM → grayscale → replicate to 3 channels →
(optionally crop to a fractional **ROI**) → resize to 224×224 → ImageNet normalize. Training
augmentations (h-flip, ±10° rotate, brightness/contrast jitter) exist but are **only used in
the deprecated end-to-end path**; the frozen-embedding path uses `is_train=False` (no aug).

**RAD-DINO** = a ViT-B/16 (vision transformer) pretrained by Microsoft on 882k chest X-rays
via DINOv2 self-supervision. Here it too is a frozen extractor producing a **768-dim CLS
token** per image. The **CLS token is a single whole-image summary** — this is a known
weakness for a *localized* finding like a widened aortic knob ([§10](#10-opportunities--innovations-prioritized)).

### 5.3 EHR (`data/ehr.py`)
Pulls a small, clinically-motivated tabular set and matches each value to the echo date by
**nearest chartdate** (body size changes slowly). Features (12):

`age, sex, height_cm, weight_kg, bmi, bsa, sbp, dbp` + 4 `*_missing` indicator flags.

Nice touches: unit conversions with plausibility clipping, a **height fallback from
BMI+weight** (`height = √(weight/BMI)`) to reduce missingness, and **BSA via the Mosteller
formula** (`√(height·weight/3600)`). Raw units with NaN preserved — imputation/scaling is
deliberately deferred to train-time (per fold) to avoid leakage.

**Why EHR is not "just another modality":** aortic diameter is clinically *indexed* to body
size (BSA), age, and sex. These are genuine **confounders** — a big person has a bigger aorta.
A model must account for them, and it turns out they carry most of the predictable signal.

---

## 6. The models

There are effectively **two generations** in the repo.

### 6.0 Generation 1 (v2, end-to-end) — mostly deprecated
`models/full_model.py` (`AortaModel`) wires a trainable **ResNet1D-34 ECG encoder**
(`ecg_encoder.py`) + **end-to-end RAD-DINO** (`cxr_encoder.py`) + a **3-token fusion
transformer** (`fusion.py`, tokens = `[CLS, ECG, CXR]` with learned mask tokens for missing
modalities) + a regression head. It has clever pieces (per-sample **modality dropout**,
per-component learning rates, freeze scheduling), **but the notes record it collapsing to the
population mean** (val MAE ≈ population std, R²≈0.07; fusion not fusing). The v3 redesign
froze everything and moved to precomputed embeddings. **`AortaModel`/`FusionTransformer` are
no longer the active path** — treat them as legacy. `train.py`, `evaluate.py`, `run_training.py`
belong to this generation.

### 6.1 Generation 2 (v3, frozen-feature fusion) — the active line
Everything trains on the three cached vectors `[PCLR(320) | RAD-DINO(768) | EHR(12)]`. Four
fusion strategies, plus baselines:

### 6.2 GBDT early fusion (`scripts/train_gbdt_fusion.py`) — the fastest real baseline
**Early / feature fusion**: literally concatenate the three vectors → 1,100-dim feature matrix
→ `HistGradientBoostingClassifier/Regressor` (a LightGBM-style gradient-boosted tree; **NaN-native**,
so EHR missingness needs no imputation) under 5-fold CV with OOF predictions. Reports per-site
binary/ordinal/regression, a **per-modality ablation** (ECG / CXR / EHR / ECG+CXR / all), and a
**PCA-reduced variant** (PCLR→32, RAD-DINO→64). This is the most *diagnostic* script — the
ablation and PCA numbers are the most informative results in the whole project.

### 6.3 Deep fusion (`scripts/train_deep_fusion.py` + `models/fusion_v3.py`)
`V3FusionModel`: project ECG→128, CXR→128, EHR→64 via small MLPs, concatenate, pass through a
LayerNorm→Dropout→Linear→GELU **trunk (128-wide)**, then two heads **per site**:

- **CORAL ordinal head** → 3 cumulative logits `P(y≥4.0), P(y≥4.5), P(y≥5.0)`.
- **Linear regression head** → z-scored diameter.

**CORAL explained simply:** for an ordered 4-grade label you train 3 *shared-slope* binary
classifiers ("is it ≥4.0?", "≥4.5?", "≥5.0?") that are forced to stay rank-consistent (a bias
per threshold, descending). It borrows statistical strength across cutpoints so the rare upper
grades aren't learned from scratch. Loss = masked BCE over the cumulative targets + `λ=0.3` ×
masked MSE on the diameter. Trained with AdamW, early-stopping on **mean validation AUROC**.

### 6.4 Late fusion / stacking (`train_late_fusion.py`, `train_late_fusion_v2.py`) — the best line
**Stacking explained simply:** train a separate model on *each* modality alone ("base
learners"), take their predictions as a handful of new features, and train a small "meta-learner"
(logistic regression / ridge) on top. The win: **each base learner can use its full
per-modality cohort** — the ECG model trains on all ~2,874 ECG patients, the EHR model on all
~4,525 EHR patients — not just the 522. Only the tiny meta-learner is limited to the 522.

`late_fusion_v2` is the current strongest fusion model. It improves on v1 by:
1. injecting **EHR's continuous diameter prediction** as a meta-feature (not just the binary prob),
2. adding the **aorta-ROI CXR** base learner alongside whole-image,
3. **nested inner-CV** to pick the meta-learner regularization (so the blend can't overfit and
   discard EHR),
4. reporting the **paired delta vs the EHR floor** — the clean "does the blend finally beat the
   floor?" test.

### 6.5 Residual fusion (`train_residual_fusion.py`) & CXR-ROI compare (`train_cxr_roi_compare.py`)
Two focused probes:
- **Residual fusion**: fit clinical (EHR) → diameter first, then ask whether imaging can predict
  the **residual** (the part EHR *couldn't* explain). This is the sharpest test of "does imaging
  add *size-independent* signal?" **Answer: essentially no** (residual R² ≈ 0, deltas straddle 0).
- **CXR-ROI compare**: whole-image vs a cropped **upper-mediastinum ROI** `(0.25, 0.08, 0.75, 0.60)`.
  The ROI helps a *little* but not significantly.

### 6.6 EHR baseline (`train_ehr_baseline.py`) — "the floor"
HistGradientBoosting / LogReg / Ridge on the 12 EHR features, **trained on the full ~4,525-patient
cohort** and reported OOF on the 522. Explicitly *not* meant to be the headline model — it exists
to answer "does ECG/CXR add anything beyond age/sex/body-size?"

---

## 7. Evaluation methodology (this part is done well)

- **5-fold stratified CV** on the binary any-AD label, from an **immutable
  `fold_assignments.csv`** so every model uses the identical split (`build_fold_assignments.py`).
- **OOF predictions** for every patient, standardized schema (`oof_io.py`).
- **Bootstrap 95% CIs** over patients for every metric (`training/bootstrap.py`): resample
  patients 2,000× → percentile CI. Essential at n=522 with ~32 positives where a point AUROC is
  very noisy.
- **Paired bootstrap deltas vs the EHR floor** (`make_eval_report.py` → `deltas_vs_ehr.csv`):
  both models scored on the *same* resampled patients each iteration; a CI excluding 0 means
  "distinguishable". This is the correct, rigorous way to ask "did imaging add value?"
- **Calibration** (Brier, calibration slope/intercept), **clinical utility** (sensitivity / PPV /
  number-needed-to-echo at top-5/10/20% risk).
- Metrics: AUROC + **AUPRC** (AUPRC matters under 6% prevalence), MAE / R² for regression,
  quadratic-weighted kappa for the ordinal grade.

Two honestly-disclosed limitations in the code comments: (a) the late-fusion/EHR base learners'
internal CV is **not nested inside the outer folds** ("second-order optimism"), and (b) the
EHR floor is trained on far more data than the fusion models (see below).

---

## 8. Results: what the numbers actually say

All numbers are OOF over the 522 triple cohort, from `outputs/eval_report/metrics_summary.csv`
and the per-model `results.json`. **Primary endpoint = binary any-AD (≥4.0 cm), AUROC.**

### 8.1 Headline table (as currently reported)

| Model | root ≥4.0 AUROC | asc ≥4.0 AUROC | root diam R² | asc diam R² | Trained on |
|---|---|---|---|---|---|
| **EHR baseline (floor)** | **0.776** [.69,.86] | 0.668 [.59,.75] | 0.309 | 0.143 | **~4,525** |
| late_fusion_v2 | 0.769 [.68,.85] | **0.695** [.62,.76] | 0.326 | 0.187 | 522 meta / full bases |
| late_fusion (v1) | 0.728 | 0.678 | 0.329 | 0.184 | 522 meta / full bases |
| deep_fusion | 0.651 | 0.669 | 0.242 | 0.082 | 522 |
| gbdt_concat | 0.630 | 0.683 | 0.307 | 0.092 | 522 |
| cxr_roi_only | 0.562 | 0.642 | — | — | 522 |

**Reading it straight:** on the **root**, nothing beats EHR-alone. On the **ascending aorta**,
`late_fusion_v2` (0.695) edges the EHR floor (0.668) but the paired-delta CI still includes 0
(`deltas_vs_ehr.csv`: +0.027 [−0.027, +0.083]) — i.e. **not statistically distinguishable**.
Every "excludes_0 = True" delta vs EHR in the report is **negative** (fusion *worse*, mostly the
≥4.5 endpoint where positives are tiny). So as reported: **imaging does not beat the clinical floor.**

### 8.2 The reframe: at equal sample size, imaging *does* help the ascending aorta

The GBDT ablation trains **every arm on the same 522 patients and same folds**, so it is the
*fair* comparison. From `gbdt_fusion/results.json`:

| Modality (all on 522) | root ≥4.0 AUROC | asc ≥4.0 AUROC |
|---|---|---|
| EHR only | 0.614 | 0.555 |
| ECG only | 0.582 | 0.535 |
| **CXR only** | 0.579 | **0.686** |
| ECG + CXR | 0.564 | **0.694** |
| all (ECG+CXR+EHR) | 0.630 | 0.683 |
| **all + PCA (ECG→32, CXR→64)** | **0.699** | **0.709** |

Two things jump out:

1. **For the ascending aorta, CXR alone (0.686) crushes EHR alone (0.555)** and ECG+CXR (0.694)
   is the best non-PCA arm. This is physiologically expected — the ascending aorta is part of the
   right upper mediastinal border on a frontal CXR. **The imaging modalities carry real
   ascending-aorta signal; the headline hides it because the reported EHR floor gets 8× more
   training data.**
2. **PCA reduction is the biggest single lever** (root 0.63→0.70, asc 0.68→0.71). 1,088 embedding
   dims on 522 patients is a massive overfitting risk; compressing them helps a lot.

### 8.3 Regression is where all models are "least bad"
Diameter R² ≈ 0.31 (root, EHR) means **age/sex/body-size explain ~31% of aortic-root diameter
variance** — and imaging adds ~nothing to that (`residual_fusion`: residual R² ≈ 0). If you
instead predict the **BSA-indexed** diameter (`residual_fusion.py` `indexed_per_bsa`), R² jumps to
**~0.38** — but that is partly mechanical (dividing by BSA re-introduces a body-size term the model
can exploit), so interpret with care.

### 8.4 Bottom line
- **Root AD** is largely a **body-size / age / sex** phenomenon in this cohort; ECG and CXR add
  little beyond that.
- **Ascending AD** has **genuine imaging signal (CXR)**, currently masked by an unfair
  data-advantaged floor and by embedding over-dimensionality.
- Everything is **severely sample-limited** (522 patients, 32–48 positives). This — not the model
  architecture — is the dominant constraint.

---

## 9. Bugs, concerns & things that are quietly wrong

Ordered roughly by how much they could be distorting the results. I've tried to be honest about
severity and to separate "real bug" from "design smell."

**#1 — The headline comparison is confounded by unequal training data (HIGH severity, but it's a
methodology issue, not a crash).** The EHR floor trains on ~4,525 patients; `gbdt_concat` and
`deep_fusion` on 522. "Fusion < EHR" therefore conflates *modeling* with *sample size*. The fair
floor is the **EHR-only GBDT ablation on 522** (root 0.614, asc 0.555), against which fusion looks
much better. *Fix:* always report the equal-n EHR arm next to fusion, and make `late_fusion_v2`
(which gives every modality its full cohort) the standard comparator.

**#2 — Echo date = `min(measurement_datetime)` per patient (MEDIUM).** In
`dataset.py::_load_echo_labels`, if a patient has several structured-measurement rows, the
**earliest** timestamp is used as "the echo date." If the aortic measurement in `aortic_labels.csv`
came from a *different/later* study, the ECG/CXR temporal matching anchors to the wrong date and
silently picks a less-relevant ECG/CXR. Worth verifying that `aortic_labels.csv` is truly one
study per patient and that the min-date is that study's date.

**#3 — Model selection on ~3–6 positive validation cases (MEDIUM).** `train_deep_fusion.py`
early-stops on **mean validation AUROC**, where the inner val split contains only a *handful* of
positives (15% of ~420, of which ~6% are positive → ~4 positives). AUROC on 4 positives is nearly
noise, so the "best epoch" is close to random. This plausibly explains why deep_fusion
underperforms the GBDT and late-fusion arms. *Fix:* select on validation **loss** or regression
**MAE** (dense signal), or on a pooled multi-site AUROC, and/or use more CV-robust selection.

**#4 — PCLR preprocessing is unnormalized and full-length-interpolated (MEDIUM, needs
verification).** `preprocess_for_pclr` does `np.interp` of the entire recording to 4,096 samples
with **no amplitude normalization**. If PCLR was trained on a fixed window (e.g. a fixed number of
seconds) and/or on normalized input, this is a train/inference mismatch that would degrade every
ECG embedding — consistent with ECG-only AUROC being ~0.53–0.58 (barely above chance). *Action:*
confirm the exact PCLR input contract (sample count, sampling rate, expected units/normalization)
against its reference implementation, and re-extract if it differs.

**#5 — Second-order leakage in stacking / floor (LOW–MEDIUM, disclosed).** The base learners'
internal CV is not nested inside the outer folds, so meta-features for outer-train patients were
produced by models that saw outer-test patients. The code comments acknowledge this. Effect is
usually small but it mildly optimistically biases late-fusion and the EHR floor. *Fix:* fully
nested CV (the plan's "M4").

**#6 — `stratify_bins`/`make_splits` is dead-ish for v3 (LOW).** `DataConfig.stratify_bins=3` and
`make_splits` (single train/val/test) belong to the legacy path; v3 uses `make_cv_folds`. Not a
bug, but a source of confusion when reading config. Similarly `huber_delta`, the whole
`ModelConfig` end-to-end section, and `run_training.py`/`train.py`/`evaluate.py` are legacy.

**#7 — Modality dropout loops per-sample in Python (LOW, legacy path only).**
`full_model.py::_apply_modality_dropout` uses a Python `for i in range(B)` with `random.random()`;
fine for correctness, slow, and only in the deprecated end-to-end model.

**#8 — `target_asc` cohort count differs (513 vs 522) (LOW, expected).** Some patients have a root
measurement but not an ascending one (or it was QC-clipped), so the ascending endpoint has n=513.
This is handled correctly via NaN-masking; just be aware the two sites have slightly different n.

No memory-corruption / crashing bugs were found — the pipeline is clean and runs. The issues above
are about *statistical validity and signal preservation*, which is exactly what matters here.

---

## 10. Opportunities & innovations (prioritized)

Grouped by expected payoff. The theme: **the bottleneck is sample size and embedding
dimensionality, not fancier fusion.** Spend effort there first.

### Tier 1 — highest expected payoff, low effort

1. **Compress the embeddings before fusion (do this first).** The PCA variant already showed
   root 0.63→0.70, asc 0.68→0.71. Make dimensionality reduction a first-class, *fold-safe*
   step (fit PCA on train fold only) across **all** fusion models, and sweep target dims. Also try
   **supervised** reductions (PLS, or a linear probe) and simple **L2/feature-selection** on the
   1,088 dims. This is the cheapest win available.

2. **Report the equal-n EHR floor as the primary comparator.** Add "EHR-only on 522" beside every
   fusion result. This will change the story from "imaging fails" to "imaging adds ascending-aorta
   signal." It's a reporting change, not a modeling change, but it's the difference between a null
   result and a positive one.

3. **Fix deep-fusion model selection** (Bug #3): select on val loss / MAE, not AUROC-on-4-positives.
   Likely lifts deep_fusion into contention with GBDT/late-fusion for free.

4. **Verify & fix PCLR preprocessing** (Bug #4). If the input contract is wrong, *every* ECG result
   so far understates ECG. Re-extract and re-run — cheap and potentially unlocks the ECG modality.

### Tier 2 — architectural, medium effort, genuinely novel for *this* task

5. **Use CXR patch tokens + attention, not the CLS token.** Aortic dilation is *localized* (the
   aortic knob / mediastinal width). The 768-dim CLS token averages the whole image away. Expose
   RAD-DINO's ~197 patch tokens and pool them with **attention-MIL** or let a query attend over
   them (the v2 plan's "cross-attention fusion"). This is the most promising imaging upgrade and
   directly targets where the ascending-aorta signal lives. (Re-extract patch-token embeddings once,
   offline, to keep the frozen-feature discipline.)

6. **Predict the BSA-indexed aortic size as the target** (aortic size index, ASI = diameter/BSA).
   This removes the trivial body-size confound and makes "does imaging add *size-independent*
   signal?" the actual learning objective — a cleaner and more publishable question. Pair with a
   proper size-independent evaluation (don't let BSA leak back in mechanically as in the current
   `indexed_per_bsa`).

7. **Give the ECG modality a real chance with a stronger backbone.** PCLR is a generic contrastive
   embedding. Options, in increasing effort: (a) a **supervised ECG model fine-tuned end-to-end on
   the full ~2,874 ECG-patient cohort** for aortic diameter (not a frozen 320-dim vector + tiny
   head); (b) **ECG-FM** (`wanglab/ecg-fm`, wav2vec 2.0, pretrained on ~1.5M ECGs — the plan's Step
   6); (c) SimCLR self-pretrain the existing ResNet1D (a checkpoint `ecg_pretrain.pt` already exists
   in `pretrained_checkpoints/`). Do this only after confirming PCLR preprocessing is correct, so
   you know whether ECG is genuinely weak or just mis-fed.

8. **Cross-modal contrastive pre-alignment (CLIP-style) on the 522 pairs** before the tiny head, or
   auxiliary self-supervision, to make the fusion trunk learn from more than 32 positives.

### Tier 3 — data & scope (the real ceiling)

9. **The dominant constraint is 522 patients / 32–48 positives.** No architecture fixes that. Levers:
   - **Download more CXRs.** CXR is the binding modality (only ~550 labelled patients have one).
     Every additional CXR-bearing patient enlarges the triple cohort. `download_cohort_cxr.py`
     exists for this.
   - **Lean into the larger cohorts** where a modality *does* have data: an **ECG(2,874)+EHR** model
     and an **EHR(4,525)** model are the honest large-n baselines; late fusion already exploits this
     and should be the headline.
   - Consider **regression as the primary endpoint** (dense signal, R²≈0.31 already) and *derive*
     the binary from it, rather than training on 32 positives directly.

10. **Track 2 (echo-only) as the "upper bound."** PanEcho embeddings on the MIMIC echo clips (already
    extracted, `panecho_embeddings.pt`) → attention-MIL → same heads. This answers "how well can the
    *imaging that the label came from* recover the grade?" — a ceiling to contextualize the ECG/CXR
    screening numbers. See [§11](#11-track-2-echo-only-model-parallel-not-the-main-line).

### Tier 4 — evaluation/rigor polish

11. **Nested CV** for stacking + floor (Bug #5 / plan "M4"); **heteroscedastic regression head**
    (predict μ and σ, Gaussian NLL) for calibrated uncertainty; **test-time augmentation** (average
    over ECG crops × CXR augmentations); **decision-curve analysis** for the screening use-case.

### A note on model choice / "look online"
For the imaging side specifically, the most relevant recent direction is **chest-X-ray foundation
models with spatial outputs** (patch tokens) and **CXR-based cardiovascular risk models** — the
signal for great-vessel size is spatial, so any approach that preserves location beats a CLS token.
For ECG, **ECG-FM / ECG foundation models** are the current best-in-class over PCLR. Neither changes
the fundamental sample-size ceiling, but both are reasonable upgrades once the Tier-1 fixes are in.

---

## 11. Track 2: echo-only model (parallel, not the main line)

A separate, MIMIC-native branch (`extract_panecho_embeddings.py`, `outputs/echo_panecho`, the
`echo_infer_AD` scripts). Idea: run **PanEcho** (an echo foundation model, already integrated) on
the cohort's echo clips → frozen embeddings → attention-MIL over a study's clips → same
ordinal+regression heads. It is framed as an **automated-measurement upper bound** (it reads the
aorta from the same echo the label came from), *not* a like-for-like screening comparison with
ECG/CXR/EHR. View selection (keeping PLAX clips) is the hard, uncertain part — the plan proposes
using PanEcho's aortic-root prediction itself as a rough PLAX selector. Coverage of the 522
`subject_id`s in the echo DICOM download is unverified.

---

## 12. Suggested next steps (concrete, in order)

1. **Reporting fix**: add the equal-n EHR floor to `make_eval_report.py`; re-tell the result. *(hours)*
2. **PCA / dim-reduction as a fold-safe step** across all fusion models; sweep dims. *(≈1 day)*
3. **Fix deep-fusion early-stopping** to val loss/MAE. *(hours)*
4. **Audit PCLR preprocessing** against the reference; re-extract if wrong. *(≈1 day)*
5. **CXR patch-token + attention-MIL** embeddings and fusion. *(≈2–3 days)*
6. **BSA-indexed target** as the clean "does imaging add size-independent signal" experiment. *(≈1 day)*
7. Only then consider **ECG-FM** and **more CXR downloads / bigger cohort**. *(week+)*

Steps 1–4 are cheap and directly address the reasons the current results look worse than the data
actually supports. Steps 5–6 are where a genuine, publishable positive finding is most likely to come
from.

---

## Glossary

| Term | Meaning |
|---|---|
| **AD** | Aortic dilation — the aorta is wider than normal. |
| **Root / sinus of Valsalva** | The base of the aorta just above the heart valve; label column `sinus_diam_cm`. |
| **Ascending aorta** | The section rising from the root; visible on a frontal CXR; `ascending_diam_cm`. |
| **Echo (echocardiogram)** | Ultrasound of the heart. Source of the ground-truth aortic diameter. **Label only, not an input.** |
| **CXR** | Chest X-ray. |
| **ECG / EKG** | Electrocardiogram; here 12-lead, 10 s @ 500 Hz. |
| **EHR** | Electronic Health Record; here just age/sex/body-size/BP. |
| **BSA** | Body Surface Area (m²); aortic size is clinically indexed to it. Mosteller: √(height·weight/3600). |
| **ASI** | Aortic Size Index = diameter / BSA. |
| **PCLR** | Patient Contrastive Learning of Representations — frozen self-supervised ECG encoder → 320-dim vector. |
| **RAD-DINO** | Microsoft ViT-B/16 CXR encoder (DINOv2 self-supervised on 882k CXRs) → 768-dim CLS token. |
| **PanEcho** | Echo foundation model (Track 2). |
| **ViT** | Vision Transformer. **CLS token** = its single whole-image summary vector. **Patch tokens** = per-region vectors. |
| **Frozen embedding** | Run a big pretrained model once, offline; cache its output vector; never train it. |
| **Early / feature fusion** | Concatenate modality vectors, then one model on top (GBDT, deep_fusion). |
| **Late fusion / stacking** | One model per modality (base learners) → meta-learner on their predictions. |
| **CORAL** | Ordinal-regression method: rank-consistent cumulative binary classifiers sharing a slope. |
| **GBDT / HistGradientBoosting** | Gradient-Boosted Decision Trees (LightGBM-style); NaN-native. |
| **CV** | Cross-Validation. **5-fold** = split into 5 parts, each is the test set once. |
| **OOF** | Out-Of-Fold — a patient's prediction always comes from a model that didn't train on that patient. |
| **AUROC** | Area Under ROC Curve; 0.5 = chance, 1.0 = perfect ranking. |
| **AUPRC** | Area Under Precision-Recall Curve; more informative than AUROC under low prevalence. |
| **QWK** | Quadratic-Weighted Kappa — agreement metric for ordinal grades. |
| **Bootstrap CI** | Resample patients with replacement many times → percentile confidence interval. |
| **Paired delta** | Difference between two models scored on the *same* resampled patients; CI excluding 0 = distinguishable. |
| **Prevalence** | Fraction of positives. Here 6–9% for ≥4.0 cm. |
| **The "floor"** | The EHR-only baseline every model is compared against. |
| **Modality dropout** | Randomly hide a modality during training so the model tolerates missing inputs. |
| **ROI** | Region Of Interest — here a cropped upper-mediastinum box on the CXR. |
| **Triple cohort** | The ~522 patients with ECG **and** CXR **and** an echo label — what v3 trains on. |
