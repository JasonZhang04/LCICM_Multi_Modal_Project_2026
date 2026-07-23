# Multimodal Aorta — v3 Redesign Plan

Status: **DRAFT for review** (no code written yet). Author: Claude + Chenjia, 2026-06.

This plan supersedes the v2 setup (`outputs/multimodal_pclr_v2`). It restricts the
multimodal model to the triple-modality cohort, adds an EHR modality, shifts the
primary objective from regression to (ordinal) classification, keeps all pretrained
backbones frozen, and reorganizes evaluation around the small-n reality.

---

## 0. Decisions locked (from discussion)


| #                   | Decision                                                                                                                                                                                                        |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Cohort              | Restrict multimodal model to the **~522 patients with ECG + CXR + echo-label** in-window. Echo is the **label source only**, not a fusion input (avoids circularity; MIMIC echo imaging not linked for fusion). |
| Label               | One source = echo aortic diameter (cm) → **two targets**: continuous diameter (regression) + ordinal grade (classification).                                                                                    |
| Grades              | Cuts at **4.0 / 4.5 / 5.0 cm**. Headlines: **binary ≥4.0 (any AD)** primary; **3-class {normal / mild / moderate+severe}** via ≥4.5 (clinically actionable). 4-grade reported as ordinal output only.           |
| Sites               | Keep **per-site heads: root (sinus) + ascending**, each with its own ordinal + regression head.                                                                                                                 |
| Backbones           | All pretrained encoders **frozen** (PCLR + RAD-DINO already precomputed; EHR has no backbone).                                                                                                                  |
| Fusion order        | **Early/feature fusion first** (concat → GBDT and concat → small MLP), single-stage on 522. **Late fusion / stacking** is the documented next step, not in scope for v3.0.                                      |
| EHR treatment       | Tabular features → small MLP trained from scratch (deep model) and raw columns (GBDT). Not frozen (nothing to freeze).                                                                                          |
| Track 2 (echo-only) | Separate, MIMIC-native, parallel. View selection = run a view model on **our** clips to make pseudo-labels (existing teammate CSV is NOT usable). Flagged as the hard/uncertain part.                           |


---

## 1. Why the task/cohort change (evidence)

Patient-level AD class counts (more-dilated of root/asc), full label set n=4579, and
estimated counts after restricting to 522:


| Class              | Full (n=4579) | Est. in 522 |
| ------------------ | ------------- | ----------- |
| no AD (<4.0)       | 3956 (86%)    | ~451        |
| mild (4.0–4.4)     | 437 (9.5%)    | ~50         |
| moderate (4.5–4.9) | 119 (2.6%)    | ~14         |
| severe (≥5.0)      | 20 (0.4%)     | **~2**      |


Implications baked into the plan:

- A flat 4-way softmax is **not** trainable/evaluable at n=522 (severe ≈ 2). Use **ordinal**
modeling so all cutpoints share statistical strength; report binary + 3-class as the real endpoints.
- A single 70/15/15 split gives test n≈78 → noisy (we already saw this with the v2 dual subset
n=89). **Switch the evaluation protocol to stratified 5-fold CV** (+ bootstrap CIs).
- **Where we will need more patients (flag):** the moderate/severe grades and any honest 4-class
metric. Mitigations later = late-fusion/stacking on the larger per-modality cohorts (ECG≈2874,
EHR≈thousands) or more echo. Out of scope for v3.0.

---

## 2. Data layer

### 2.1 Cohort restriction → triple-modality (522)

- `data/dataset.py::build_cohort`: add `cohort_mode` param (`"any"` | `"triple"`). For `"triple"`,
keep rows with `has_ecg & has_cxr` (all rows already have an echo label). Log the resulting
count + per-grade class counts for root and ascending.
- `configs/default_config.py`: `DataConfig.cohort_mode = "triple"`.
- Expected: ~522 patients → with 5-fold CV, ~417 train / ~105 held-out per fold.

### 2.2 EHR extraction (NEW)

New module `data/ehr.py` + one-time script `scripts/extract_ehr_features.py` →
cache `pretrained_checkpoints/ehr_features.csv` (one row per subject_id), mirroring the
embedding-cache pattern.

Sources (MIMIC-IV v3.1, on cluster, no download):

- `…/mimiciv/3.1/hosp/patients.csv.gz` → `gender`, `anchor_age`, `anchor_year`.
- `…/mimiciv/3.1/hosp/omr.csv.gz` → `result_name` ∈ {`Height (Inches)`, `Weight (Lbs)`,
`BMI (kg/m2)`, `Blood Pressure` "SBP/DBP"}, with `chartdate`.

Features (per subject, matched to echo date):


| Feature      | Derivation                                                                                                            |
| ------------ | --------------------------------------------------------------------------------------------------------------------- |
| `age`        | `anchor_age + (echo_year − anchor_year)`                                                                              |
| `sex`        | `gender` → M=1 / F=0                                                                                                  |
| `height_cm`  | `Height (Inches)` × 2.54, nearest omr record to echo_date; **fallback** from BMI+weight: `sqrt(weight_kg / BMI)` ×100 |
| `weight_kg`  | `Weight (Lbs)` × 0.4536, nearest to echo_date                                                                         |
| `bmi`        | `BMI (kg/m2)` direct, else weight/height²                                                                             |
| `bsa`        | Mosteller: `sqrt(height_cm × weight_kg / 3600)`                                                                       |
| `sbp`, `dbp` | parse `Blood Pressure` "S/D" (optional, lower priority)                                                               |
| `*_missing`  | binary indicators for height/weight/bsa/bp                                                                            |


Notes:

- Height is sparser than weight/BMI → the BMI+weight fallback materially reduces missingness.
- Numeric features z-scored on the **train fold only** (store stats per fold).
- Clinical rationale: aortic size is normally indexed to BSA/age/sex, so EHR are confounders the
model genuinely needs — not just "another modality."

### 2.3 Label / target redesign

New `data/targets.py` (or extend `dataset.py`):

- `diam_cm` per site (existing).
- `grade` per site ∈ {0,1,2,3} by cuts {4.0,4.5,5.0}.
- **Ordinal encoding (CORN/CORAL):** for K=4, three cumulative binary targets per site:
`y≥4.0`, `y≥4.5`, `y≥5.0`. NaN site → all targets for that site masked.
- Dataset `__getitem_`_ returns: `ecg`(320), `cxr`(768), `ehr`(F), `diam`(2), `grade_cum`(2×3),
masks.

### 2.4 Splits → stratified 5-fold CV

- `data/splits.py`: add `make_cv_folds(cohort, n_splits=5, stratify_on="any_AD", seed)` using
`StratifiedKFold` on the binary ≥4.0 label (guarantees positives in every fold).
- Keep a fixed held-out **test** option too, but CV is the default reporting protocol at this n.

---

## 3. Model layer (deep fusion path)

All inputs are frozen precomputed features; only the pieces below are trained (tiny).

### 3.1 EHR encoder (NEW) — `models/ehr_encoder.py`

- Small MLP: `Linear(F→64) → GELU → Dropout → Linear(64→d_ehr)`. ~few k params.

### 3.2 Fusion (early/feature) — adapt `models/full_model.py`

- **v3.0 default = concatenation MLP** (simplest, fewest params for n≈420 train):
`concat[proj_ecg ⊕ proj_cxr ⊕ ehr_emb] → MLP(→256→128) → shared trunk`.
- Keep the existing `FusionTransformer` as an optional `fusion_type="transformer"` (now a
3-token ECG/CXR/EHR variant), but it is **not** the v3.0 baseline.
- Projections: `Linear(320→d)`, `Linear(768→d)`, EHR MLP→d (or just concat raw dims for the MLP variant).

### 3.3 Heads (NEW) — `models/heads.py`

- Per site (root, asc): an **ordinal head** `Linear(trunk→3)` (logits for the 3 cumulative cuts) +
a **regression head** `Linear(trunk→1)` (z-scored diameter).
- 4 heads total (2 sites × {ordinal, reg}).

### 3.4 Loss — `training/losses.py`

- Ordinal: CORN/CORAL conditional BCE over the 3 cumulative targets per site (masked for NaN sites).
- Regression: existing masked MSE on z-scored diameter (auxiliary, weight λ_reg≈0.3).
- Total = Σ_site (ordinal + λ_reg·reg). Class imbalance handled via ordinal framing + optional
`pos_weight` on the cumulative BCE.

---

## 4. GBDT baseline (NEW, first deliverable) — `scripts/train_gbdt_fusion.py`

The strongest, fastest baseline at n≈522.

- Feature matrix = concat raw `[PCLR(320) ⊕ RAD-DINO(768) ⊕ EHR(~10)]` per subject.
- Targets: per-site binary (≥4.0, ≥4.5) and ordinal (via independent or `OrdinalClassifier`
wrappers); diameter via `XGBRegressor`.
- Protocol: stratified 5-fold CV; report mean ± CI.
- **Dimensionality note:** 1088 embedding dims vs ~420 train rows → rely on XGBoost regularization;
also try PCA (PCLR→32, RAD-DINO→64) and compare. Report both.
- Free wins: per-modality ablation (drop each block), `feature_importances`_, SHAP optional.

---

## 5. Evaluation overhaul — `training/evaluate.py`

- Metrics per site:
  - Binary ≥4.0 and ≥4.5: **AUROC + AUPRC** (AUPRC matters under imbalance).
  - Ordinal 4-grade: **quadratic-weighted kappa** + confusion matrix.
  - Diameter: MAE, RMSE, R², Pearson.
- Baselines reported alongside: **mean/prevalence predictor** + **per-modality ablations**
(ECG-only, CXR-only, EHR-only, all) — now meaningful since all 522 have ECG+CXR.
- **CV aggregation + bootstrap CIs** (replace single-split point estimates).
- Keep v2's clean reporting style; drop the dual-modality-subset machinery (no longer needed —
the cohort is fully ECG+CXR by construction; EHR missingness handled via indicators).

---

## 6. Build order (milestones)

1. **EHR extraction** — `data/ehr.py` + `scripts/extract_ehr_features.py` → cache CSV. Verify
  omr parsing + coverage of the 522.
2. **Cohort + targets + CV splits** — triple-mode cohort, ordinal/regression targets, stratified 5-fold.
3. **GBDT fusion baseline** — `scripts/train_gbdt_fusion.py` (fastest path to a real number + ablation).
4. **Deep fusion** — EHR encoder + concat-MLP trunk + ordinal/reg heads + losses.
5. **Eval overhaul** — AUROC/AUPRC/QWK/MAE, CV + bootstrap, per-modality ablations.
6. *(next phase)* **Late fusion / stacking** — unimodal models on full per-modality cohorts →
  tiny combiner on 522.
7. *(parallel)* **Track 2 echo-only** (Section 7).

Each milestone is independently runnable/reviewable. Milestones 1–3 deliver a usable result.

---

## 7. Track 2 — Echo-only model (MIMIC-native, parallel)

Goal: can echo *imaging* recover the aortic grade (an automated-measurement upper bound),
fully on MIMIC, sharing the same labels.

Reality check on view selection (per Chenjia):

- The existing `MIMIC_ECHO_View_Classifications.csv` is a **teammate's EchoVIEW model output** on a
**tiny subset with little overlap** with our cohort → **not ground truth, not usable coverage.**
- Proper approach = obtain a view-classification model and **run inference on OUR cohort's clips**
to produce *machine pseudo view labels*, then keep PLAX. This is non-trivial.

Two viable paths:

- **(Recommended, pragmatic) PanEcho AORoot proxy on our clips.** PanEcho is already integrated and
predicts aortic-root cm (only measurable from PLAX). Run it on our MIMIC clips → use AORoot in a
plausible range as a PLAX selector. No external view model needed; runs on the right data. Approximate.
- **(Higher fidelity, more work) Dedicated view classifier.** Obtain a public 23-view model
(taxonomy matching the EchoVIEW columns) or the teammate's weights; run on our clips for pseudo-labels.

Then: frozen **PanEcho/EchoJEPA** embeddings on PLAX clips → attention-MIL over a study's clips →
same ordinal + regression heads. Backbones frozen (consistent with Decision row 5).

Open verification before starting: confirm `jooyoung/mimic_4_echo/files` DICOM coverage of the 522
`subject_id`s (download may be partial).

---

## 8. Risks / open items

- **Rare classes (moderate/severe) untrainable at n=522** → report but don't over-trust; needs
bigger cohorts (late fusion) or more echo. Primary endpoint stays binary ≥4.0.
- **Small test size** → mitigated by 5-fold CV + bootstrap CIs.
- **EHR missingness** (esp. height) → BMI+weight fallback + missingness indicators.
- **Echo DICOM coverage** of the 522 unverified (Track 2).
- **Echo-only is an "automated-measurement" task**, not screening (it reads the aorta from the same
echo the label came from) — useful as an upper bound, not a like-for-like comparison with the
ECG/CXR/EHR screening model.

---

## 9. New/changed files (summary)

New:

- `data/ehr.py`, `scripts/extract_ehr_features.py`
- `data/targets.py` (ordinal/grade logic) — or fold into `dataset.py`
- `models/ehr_encoder.py`, `models/heads.py`
- `scripts/train_gbdt_fusion.py`

Changed:

- `data/dataset.py` (cohort_mode, EHR + ordinal targets in `__getitem_`_)
- `data/splits.py` (stratified k-fold CV)
- `models/full_model.py` (EHR token/concat, ordinal+reg heads, concat-MLP trunk)
- `training/losses.py` (ordinal CORN/CORAL + masked reg)
- `training/evaluate.py` (AUROC/AUPRC/QWK/MAE, CV, bootstrap, ablations)
- `configs/default_config.py` (cohort_mode, ehr paths/dims, fusion_type, loss weights, CV)
- `run_training.py` (wire EHR cache + CV loop)

