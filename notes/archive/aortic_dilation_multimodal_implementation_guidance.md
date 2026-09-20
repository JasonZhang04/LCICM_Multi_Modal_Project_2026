# Implementation Guidance: Multimodal Aortic Dilation Prediction

**Project:** Detecting aortic dilation using ECG, CXR, and EHR instead of echo as the only screening signal  
**Primary targets:** Aortic root and ascending aorta diameter/dilation  
**Primary modalities:** 12-lead ECG, chest X-ray, structured EHR  
**Current evaluation setting:** MIMIC-derived echo labels, ECG/CXR temporally aligned to echo, triple-modality cohort around 522 patients  
**Purpose of this document:** Provide implementation-focused guidance for coding agents. This file emphasizes model, architecture, preprocessing, target-design, leakage-control, and evaluation changes needed to strengthen the paper.

---

## 0. Executive Implementation Summary

The current system already contains useful components: frozen PCLR ECG embeddings, frozen RAD-DINO CXR embeddings, structured EHR features, GBDT baselines, deep fusion, and late fusion/stacking. The most important next step is not to add a larger neural network. The most important next step is to make the experimental design more rigorous, clinically grounded, and sample-efficient.

The recommended development priorities are:

1. **Add a strong clinical baseline.** Implement an AORTA-score-style EHR baseline using age, sex, height, weight, BMI/BSA, blood pressure, heart rate when available, and relevant comorbidities. All multimodal claims must be compared against this baseline.
2. **Audit and fix late-fusion leakage.** Late fusion currently looks strongest, but stacking can easily leak information if base models, imputers, PCA, scalers, calibrators, or meta-learners see outer-test patients.
3. **Stop relying only on the n=522 triple-modality training set.** Keep the triple-modality cohort as the fair evaluation set, but train modality-specific models on their larger eligible cohorts while excluding the outer-test patients.
4. **Add missing-modality support.** Implement fusion models that can train and predict with ECG-only, CXR-only, EHR-only, ECG+EHR, CXR+EHR, ECG+CXR, and ECG+CXR+EHR.
5. **Add body-size-indexed and residualized targets.** Root and ascending aortic diameters are heavily tied to body size, age, and sex. The model must show whether ECG/CXR add information beyond body-size physiology.
6. **Improve label quality and sensitivity analyses.** Add QC for echo label plausibility, measurement-site certainty, repeat measurements, and temporal alignment windows.
7. **Use smaller, better-regularized fusion architecture.** Use frozen encoders plus low-parameter projections, modality masks, gated fusion, and multi-task heads.
8. **Improve CXR modeling.** Separate PA vs AP/portable images, include view metadata, test mediastinum/aorta ROI crops, and add simple CXR measurement baselines.
9. **Improve ECG modeling.** Add ECG quality filtering, classical ECG features, and parameter-efficient fine-tuning or adaptation rather than training a large 1D model from scratch.
10. **Add clinical utility evaluation.** Report calibration, top-risk enrichment, sensitivity/specificity/PPV/NPV at clinically meaningful cutoffs, and number-needed-to-echo.

---

## 1. Non-Negotiable Experimental Principles

These rules should be enforced across the codebase.

### 1.1 Patient-Level Separation

All train/validation/test splits must be patient-level splits. A patient must never appear in both training and validation/test folds, even if they have multiple ECGs, CXRs, or echoes.

**Required checks:**

- `subject_id` overlap between outer train and outer test must be empty.
- If multiple rows per patient are used internally, all rows for a patient must be assigned to the same fold.
- Any cached features or predictions must include `subject_id` and `fold_id` so leakage can be audited.

### 1.2 Outer-Test Patients Must Be Completely Held Out

For every outer fold, no data from outer-test patients may be used for:

- base model fitting,
- meta-learner fitting,
- imputation,
- scaling,
- PCA,
- feature selection,
- calibration,
- threshold selection,
- hyperparameter tuning,
- early stopping,
- residualization models,
- label normalization,
- class weight estimation,
- embedding dimensionality reduction.

This is especially important for late fusion because base models may be trained on larger per-modality cohorts.

### 1.3 Pre-Echo-Only EHR Features

Structured EHR features should be available before or on the prediction index date. The default index date should be the echo date, but all feature timestamps must be checked.

**Allowed by default:**

- demographics available before echo,
- vitals/OMR values before echo,
- diagnoses before echo,
- medications before echo,
- prior measurements before echo.

**Not allowed unless explicitly labeled as a separate post-hoc analysis:**

- diagnoses entered after echo,
- procedures triggered by the echo result,
- notes that describe the echo finding,
- reports from imaging studies after the echo,
- future measurements.

### 1.4 Severe Endpoint Handling

If a cohort has zero positives for diameter >=5.0 cm, do not train or report AUROC/AUPRC for that endpoint. Mark it as **not estimable**.

For endpoints with very few positives, for example fewer than 20 positives:

- report event counts prominently,
- report wide confidence intervals,
- avoid making strong performance claims,
- do not select major model design decisions based on that endpoint alone.

### 1.5 Every Experimental Variant Must Be Reproducible

Every run should save:

- config YAML/JSON,
- git commit hash if available,
- random seed,
- cohort CSV path and hash,
- fold assignment path and hash,
- feature cache paths and hashes,
- hyperparameter grid,
- selected hyperparameters per fold,
- metrics per fold,
- out-of-fold predictions,
- bootstrap confidence intervals,
- environment information.

---

## 2. Recommended Implementation Roadmap

### 2.1 P0: Required Before Any Paper Submission

These items are essential.

#### P0.1 Add a Strong Clinical Baseline

Implement an AORTA-score-style EHR model and compare all multimodal approaches against it.

Deliverables:

- `ehr_clinical_baseline.py`
- `ehr_feature_builder.py`
- baseline metrics for root and ascending targets
- delta metrics for multimodal models vs EHR-only baseline

#### P0.2 Make Late Fusion Leakage-Proof

Refactor stacking so outer-test patients are excluded from every training/preprocessing/calibration step.

Deliverables:

- `leakage_safe_stacking.py`
- unit tests for outer-test exclusion
- per-fold train/test patient ID logs
- leakage audit report

#### P0.3 Add Calibration and Clinical Utility Metrics

AUROC and AUPRC are not enough for a screening model.

Deliverables:

- calibration curves
- Brier score
- calibration intercept and slope
- sensitivity/specificity/PPV/NPV at top 5%, 10%, 20% risk
- number-needed-to-echo
- decision curve analysis if feasible

#### P0.4 Store All OOF Predictions

Every model must save patient-level out-of-fold predictions for downstream comparison.

Required columns:

```text
subject_id
fold_id
site
endpoint
y_true_binary
y_true_diameter_cm
y_true_grade
pred_probability
pred_diameter_cm
pred_grade
model_name
modality_set
prediction_date
echo_date
ecg_date
cxr_date
has_ecg
has_cxr
has_ehr
```

---

### 2.2 P1: Strongly Recommended

#### P1.1 Train With Incomplete Modalities

Do not restrict training to only the n=522 complete-case cohort. Use all eligible data for each modality while keeping the complete-case cohort as the main fair evaluation set.

Recommended cohorts:

- EHR-only cohort
- ECG-only cohort
- CXR-only cohort
- ECG+EHR cohort
- CXR+EHR cohort
- ECG+CXR cohort
- ECG+CXR+EHR complete-case cohort
- any-modality cohort with missing-modality masks

#### P1.2 Add Body-Size-Indexed Targets

Add target variants:

- raw diameter in cm,
- diameter / BSA,
- diameter / height,
- residual diameter after clinical baseline adjustment,
- binary thresholds at 4.0 and 4.5 cm,
- ordinal grade: none, mild, moderate, severe.

Do not rely only on fixed diameter thresholds.

#### P1.3 Add Label QC and Sensitivity Analyses

Add label histograms, repeat-measurement checks, site certainty checks, and alignment-window sensitivity analyses.

Default windows to evaluate:

- 30 days,
- 90 days,
- 180 days,
- 365 days.

#### P1.4 Improve CXR Pipeline

Separate PA vs AP/portable images, add CXR view metadata, and test aorta/mediastinum ROI crops.

---

### 2.3 P2: Architecture and Representation Improvements

#### P2.1 Add Low-Parameter Gated Fusion

Use frozen encoders, small projection heads, modality masks, and gated fusion.

#### P2.2 Add Multi-Task Learning Across Sites

Train one model with shared representation and separate heads for root and ascending aorta.

#### P2.3 Add Residual Modeling

Use EHR clinical baseline to predict expected diameter, then train ECG/CXR to predict residual diameter.

#### P2.4 Add ECG and CXR Interpretability

Use permutation importance, SHAP for tabular models, Grad-CAM or attention rollout for CXR, and modality ablations.

---

## 3. Cohort, Splitting, and Data Handling

### 3.1 Core Entities

Use stable patient-level identifiers and dates throughout the pipeline.

Recommended columns for the master cohort table:

```text
subject_id
hadm_id                         # optional, if relevant
stay_id                         # optional, if relevant
echo_date
root_diameter_cm
ascending_diameter_cm
root_label_source
ascending_label_source
ecg_study_id
ecg_date
ecg_days_from_echo
cxr_study_id
cxr_date
cxr_days_from_echo
cxr_view_position
cxr_is_portable
has_ecg
has_cxr
has_ehr
age_at_echo
sex
height_cm
weight_kg
bmi
bsa
sbp
dbp
heart_rate                      # add if available
fold_id                         # assigned later
```

### 3.2 Cohort Definitions

Create explicit cohort flags rather than relying on implicit filtering.

```text
cohort_ehr_only                 = has_ehr
cohort_ecg_only                 = has_ecg
cohort_cxr_only                 = has_cxr
cohort_ecg_ehr                  = has_ecg and has_ehr
cohort_cxr_ehr                  = has_cxr and has_ehr
cohort_ecg_cxr                  = has_ecg and has_cxr
cohort_triple                   = has_ecg and has_cxr and has_ehr
cohort_any_modality             = has_ecg or has_cxr or has_ehr
```

The main fair comparison should still use `cohort_triple`, but base models can train on broader cohorts as long as outer-test patients are excluded.

### 3.3 Fold Assignment

Generate folds once and reuse them across all experiments.

Recommended strategy:

1. Start from the complete-case triple-modality cohort.
2. Stratify on a robust label such as `any_ad_4p0`, where:

```text
any_ad_4p0 = root_diameter_cm >= 4.0 or ascending_diameter_cm >= 4.0
```

3. If possible, also balance folds by sex, age group, and CXR view type.
4. Save `fold_assignments.csv` and treat it as immutable for the main experiments.

Required columns:

```text
subject_id
fold_id
any_ad_4p0
root_ad_4p0
root_ad_4p5
asc_ad_4p0
asc_ad_4p5
root_grade
asc_grade
```

### 3.4 Event-Count Table Per Fold

Generate this automatically for each run.

Example output:

```text
fold_id
n_total
n_root_ad_4p0
n_root_ad_4p5
n_root_ad_5p0
n_asc_ad_4p0
n_asc_ad_4p5
n_asc_ad_5p0
n_any_ad_4p0
```

Fail fast if an endpoint being evaluated has no positives or no negatives in any outer test fold.

### 3.5 Temporal Alignment

Default alignment currently uses 180 days. Keep this as the primary setting, but add sensitivity analyses.

Recommended implementation:

```text
max_days_offset in {30, 90, 180, 365}
```

For each modality:

- choose nearest study to echo date within the allowed window,
- save absolute and signed day offset,
- allow sensitivity analysis with only pre-echo modality measurements,
- allow sensitivity analysis with both pre-echo and post-echo measurements within window.

Important distinction:

```text
signed_days_from_echo = modality_date - echo_date
abs_days_from_echo    = abs(signed_days_from_echo)
```

For clinically prospective framing, pre-echo or same-day studies are preferable.

---

## 4. Clinical Baseline Implementation

A strong EHR/clinical baseline is mandatory. It is the reference model reviewers will use to judge whether ECG and CXR add value.

### 4.1 Minimum EHR Feature Set

The minimum baseline should include:

```text
age_at_echo
sex
height_cm
weight_kg
bmi
bsa
sbp
dbp
heart_rate              # if available before echo
```

### 4.2 Expanded EHR Feature Set

Add these if available with reliable timestamps before echo:

```text
hypertension
hyperlipidemia
diabetes
chronic_kidney_disease
smoking_status_or_proxy
bicuspid_aortic_valve
aortic_stenosis
aortic_regurgitation
connective_tissue_disease
marfan_syndrome
loeys_dietz_syndrome
ehlers_danlos_syndrome
family_history_aortic_disease     # only if structured and available
prior_aortic_diagnosis             # use cautiously; may be leakage-like
antihypertensive_medication_flag
beta_blocker_flag
acei_arb_flag
statin_flag
number_of_bp_measurements
median_sbp_last_1y
median_dbp_last_1y
max_sbp_last_1y
max_dbp_last_1y
```

### 4.3 Features to Avoid or Flag as High Leakage Risk

Do not include these in the main pre-echo model unless carefully justified:

```text
post_echo_diagnosis_codes
post_echo_procedure_codes
post_echo_notes
post_echo_imaging_reports
structured fields explicitly created from the target echo
known aneurysm diagnosis after the prediction date
surgical referral after echo
ct_or_mri_measurement_after_echo
```

Prior aortic diagnosis before echo can be included in a secondary analysis, but the cleanest screening model should report results without it.

### 4.4 Body Surface Area Calculation

Use a consistent BSA formula. The Mosteller formula is simple and common.

```text
bsa_m2 = sqrt(height_cm * weight_kg / 3600)
```

If height or weight is missing, impute within fold and include missingness indicators.

### 4.5 Preprocessing

For EHR features:

1. Winsorize extreme numeric values using training-fold limits only.
2. Impute missing numeric features using training-fold median.
3. Add missingness indicators for important variables.
4. Standardize numeric variables for linear/logistic models.
5. One-hot encode categorical variables such as sex and smoking status.
6. Fit all preprocessing objects inside the training fold only.

### 4.6 Baseline Models

Implement at least these:

#### 4.6.1 Logistic Regression for Binary Dilation

Use for endpoints:

```text
root_ad_4p0
root_ad_4p5
asc_ad_4p0
asc_ad_4p5
```

Recommended settings:

```text
penalty: l2
class_weight: balanced or None, selected by inner CV
C grid: [0.01, 0.03, 0.1, 0.3, 1, 3, 10]
solver: lbfgs or liblinear
max_iter: 5000
```

#### 4.6.2 Ridge or Elastic-Net Regression for Diameter

Use for:

```text
root_diameter_cm
ascending_diameter_cm
root_diameter_indexed_bsa
ascending_diameter_indexed_bsa
root_residual_diameter
ascending_residual_diameter
```

Recommended settings:

```text
ridge_alpha_grid: [0.1, 1, 3, 10, 30, 100]
elasticnet_l1_ratio_grid: [0.1, 0.5, 0.9]
```

#### 4.6.3 HistGradientBoosting Baseline

Use as nonlinear EHR baseline.

Recommended settings:

```text
learning_rate: [0.01, 0.02, 0.05, 0.1]
max_iter: [200, 500, 1000]
max_leaf_nodes: [7, 15, 31]
min_samples_leaf: [10, 20, 50]
l2_regularization: [0.0001, 0.001, 0.01, 0.1, 1.0]
early_stopping: true
```

### 4.7 Required EHR Baseline Comparisons

Report these explicitly:

```text
EHR clinical baseline only
ECG only
CXR only
ECG + CXR only
EHR + ECG
EHR + CXR
EHR + ECG + CXR
```

For each multimodal model, report:

```text
absolute_metric
metric_difference_vs_ehr
bootstrap_ci_for_difference
p_value_or_interval_if_available
```

Key question:

> Do ECG and CXR add incremental value beyond age, sex, body size, and blood pressure?

---

## 5. Leakage-Proof Late Fusion / Stacking

Late fusion is promising but must be implemented carefully. This section should be treated as a specification.

### 5.1 Definitions

```text
D_triple        = patients with ECG + CXR + EHR + echo label
D_ecg           = patients with ECG + echo label
D_cxr           = patients with CXR + echo label
D_ehr           = patients with EHR + echo label
outer_test_k    = D_triple patients assigned to fold k
outer_train_k   = D_triple patients not assigned to fold k
```

Base models may train on `D_ecg`, `D_cxr`, and `D_ehr`, but must exclude all `outer_test_k` patients.

### 5.2 Correct Outer-Fold Algorithm

For each outer fold `k`:

1. Define `outer_test_ids` from the complete-case triple cohort.
2. Define `outer_train_ids` from the remaining complete-case patients.
3. For each modality `m` in `{ECG, CXR, EHR}`:
   - Define eligible training cohort `D_m_train = D_m excluding outer_test_ids`.
   - Build all preprocessors using only `D_m_train`.
   - Generate out-of-fold predictions for `outer_train_ids` only using inner CV.
   - Train a final base model on `D_m_train`.
   - Predict the `outer_test_ids` with the final base model.
4. Fit the meta-learner using only base predictions for `outer_train_ids`.
5. Predict `outer_test_ids` using their base-model predictions.
6. Save all outer-test predictions.
7. Repeat for all outer folds.

### 5.3 Inner OOF Prediction Generation

For each modality and outer fold:

```python
for inner_train_ids, inner_valid_ids in inner_cv_split(outer_train_ids):
    # Training data for the base model can include broader per-modality patients,
    # but must exclude outer-test patients and inner validation patients.
    base_train_ids = all_modality_eligible_ids
    base_train_ids = base_train_ids - outer_test_ids
    base_train_ids = base_train_ids - inner_valid_ids

    fit_preprocessor(base_train_ids)
    fit_base_model(base_train_ids)
    pred_inner_valid = predict(inner_valid_ids)
    store_oof_predictions(inner_valid_ids, pred_inner_valid)
```

Important: `inner_valid_ids` should be complete-case patients from `outer_train_ids`, because the meta-learner is evaluated on the complete-case comparison cohort.

### 5.4 Meta-Learner Input Features

Default meta-learner features for binary endpoints:

```text
p_ecg
p_cxr
p_ehr
logit_p_ecg
logit_p_cxr
logit_p_ehr
has_ecg
has_cxr
has_ehr
```

Optional interaction features:

```text
p_ecg * p_cxr
p_ecg * p_ehr
p_cxr * p_ehr
max(p_ecg, p_cxr, p_ehr)
mean_available_probability
std_available_probability
```

Do not add too many features. The meta-learner must stay simple because positive counts are small.

Default meta-learner for binary endpoints:

```text
LogisticRegression
penalty: l2
class_weight: balanced or None, selected in inner CV
C: [0.01, 0.03, 0.1, 0.3, 1, 3, 10]
```

Default meta-learner for diameter regression:

```text
Ridge
alpha: [0.1, 1, 3, 10, 30, 100]
```

### 5.5 Calibration in Stacking

Calibration must also be leakage-safe.

Valid options:

1. **No calibration**, report raw probabilities and calibration metrics.
2. **Inner-fold Platt calibration**, where calibrators are fit only on inner-training/validation predictions inside the outer-training set.
3. **Isotonic calibration**, only if enough positives exist. Usually not recommended with very low event counts.

Do not fit a calibrator on the full dataset before outer-fold evaluation.

### 5.6 Stacking Unit Tests

Add automated tests:

```text
test_outer_test_not_in_base_training_ids
test_outer_test_not_in_preprocessor_fit_ids
test_outer_test_not_in_pca_fit_ids
test_outer_test_not_in_calibrator_fit_ids
test_outer_test_not_in_meta_training_ids
test_inner_valid_not_in_inner_base_training_ids
test_no_missing_oof_predictions_for_outer_train_ids
test_one_prediction_per_subject_per_endpoint_per_model
```

Each test should fail loudly if overlap is found.

---

## 6. Target Engineering

The current fixed-threshold labels are useful, but they do not fully account for body-size physiology. Add multiple target variants.

### 6.1 Raw Diameter Targets

```text
root_diameter_cm
ascending_diameter_cm
```

Primary regression metrics:

```text
MAE
RMSE
R2
Spearman correlation
Pearson correlation
```

### 6.2 Binary Threshold Targets

```text
root_ad_4p0 = root_diameter_cm >= 4.0
root_ad_4p5 = root_diameter_cm >= 4.5
root_ad_5p0 = root_diameter_cm >= 5.0
asc_ad_4p0  = ascending_diameter_cm >= 4.0
asc_ad_4p5  = ascending_diameter_cm >= 4.5
asc_ad_5p0  = ascending_diameter_cm >= 5.0
```

Only evaluate thresholds with at least one positive and one negative in every outer test fold. Preferably require at least 20 total positives for stable model comparison.

### 6.3 Ordinal Grade Targets

Default ordinal bins:

```text
grade 0: diameter < 4.0 cm
grade 1: 4.0 cm <= diameter < 4.5 cm
grade 2: 4.5 cm <= diameter < 5.0 cm
grade 3: diameter >= 5.0 cm
```

Metrics:

```text
quadratic_weighted_kappa
macro_mae_grade
confusion_matrix
```

If grade 3 has zero positives, keep the grade definition for consistency but state that severe-grade discrimination is not estimable.

### 6.4 Body-Size-Indexed Diameter Targets

Add indexed targets:

```text
root_diameter_per_bsa = root_diameter_cm / bsa_m2
asc_diameter_per_bsa  = ascending_diameter_cm / bsa_m2
root_diameter_per_height = root_diameter_cm / height_m
asc_diameter_per_height  = ascending_diameter_cm / height_m
```

Recommended metrics:

```text
MAE_indexed
R2_indexed
correlation_indexed
```

Add missingness indicators if height or weight was imputed.

### 6.5 Residualized Diameter Targets

Residualized targets help determine whether ECG/CXR predict aortic pathology beyond demographics and body size.

For each outer fold:

1. Fit a clinical baseline regression model on outer-training data only.
2. Predict expected diameter for outer-training and outer-test patients.
3. Compute residuals for outer-training patients:

```text
root_residual_train = root_diameter_cm_train - predicted_root_clinical_train
asc_residual_train  = ascending_diameter_cm_train - predicted_asc_clinical_train
```

4. Train ECG/CXR/fusion models to predict residuals.
5. For outer-test patients:

```text
final_predicted_diameter = predicted_clinical_diameter + predicted_residual
```

6. Evaluate both residual prediction and final reconstructed diameter prediction.

Critical rule: the clinical residualization model must be fit inside each outer fold. Do not residualize once using the full dataset.

### 6.6 Target Registry

Create a central target registry instead of hardcoding endpoints in scripts.

Example YAML:

```yaml
targets:
  root_diameter_cm:
    site: root
    type: regression
    source_column: root_diameter_cm
    primary_metric: mae

  asc_diameter_cm:
    site: ascending
    type: regression
    source_column: ascending_diameter_cm
    primary_metric: mae

  root_ad_4p0:
    site: root
    type: binary
    source_column: root_diameter_cm
    threshold: 4.0
    direction: ge
    primary_metric: auprc

  root_ad_4p5:
    site: root
    type: binary
    source_column: root_diameter_cm
    threshold: 4.5
    direction: ge
    primary_metric: auprc

  asc_ad_4p0:
    site: ascending
    type: binary
    source_column: ascending_diameter_cm
    threshold: 4.0
    direction: ge
    primary_metric: auprc

  asc_ad_4p5:
    site: ascending
    type: binary
    source_column: ascending_diameter_cm
    threshold: 4.5
    direction: ge
    primary_metric: auprc
```

---

## 7. Echo Label Quality Control

The label source determines the ceiling of model performance. Add explicit QC and sensitivity analyses.

### 7.1 Plausibility Rules

Current plausible clipping ranges are reasonable:

```text
root: 1.5 cm to 6.0 cm
ascending: 1.5 cm to 7.0 cm
```

Add a QC report that counts:

```text
n_raw_labels
n_missing_root
n_missing_ascending
n_root_below_range
n_root_above_range
n_ascending_below_range
n_ascending_above_range
n_after_qc
```

Do not silently drop or clip labels without logging.

### 7.2 Distribution Plots

Generate:

```text
root diameter histogram
ascending diameter histogram
root vs ascending scatter plot
root diameter by sex
ascending diameter by sex
root diameter by BSA quartile
ascending diameter by BSA quartile
```

Add vertical threshold lines at 4.0, 4.5, and 5.0 cm.

### 7.3 Repeat Echo Consistency

If multiple echo labels exist per patient:

- calculate within-patient variation,
- compare nearest echo only vs average of echoes in a window,
- check whether labels cross threshold over time,
- flag implausibly large changes over short intervals.

Suggested report:

```text
n_patients_with_multiple_echoes
median_time_between_echoes
median_abs_root_difference
median_abs_ascending_difference
n_crossing_root_4p0
n_crossing_ascending_4p0
```

### 7.4 Measurement-Site Certainty

If labels were parsed from reports, add a simple text audit if report text is available.

Potential root terms:

```text
sinus of valsalva
sinuses of valsalva
aortic root
root diameter
sinus diameter
```

Potential ascending terms:

```text
ascending aorta
proximal ascending aorta
mid ascending aorta
tubular ascending aorta
```

Potential ambiguity terms:

```text
not well visualized
poorly visualized
limited visualization
technically difficult
not measured
not seen
```

Create flags:

```text
root_site_explicit
ascending_site_explicit
ascending_visualization_limited
measurement_ambiguous
```

Then run a sensitivity analysis excluding ambiguous labels.

### 7.5 Temporal Sensitivity

For ECG and CXR alignment, repeat the main evaluation under:

```text
30 day window
90 day window
180 day window
365 day window
pre-echo only window
post-echo allowed window
```

Report how cohort size, event counts, and performance change.

---

## 8. Multimodal Architecture vNext

The current deep fusion model should be revised toward a smaller, more regularized, missing-modality-aware, multi-task design.

### 8.1 Design Goals

The fusion model should:

- use frozen ECG and CXR encoders by default,
- support missing modalities,
- train on incomplete-modality cohorts,
- learn separate but related root and ascending outputs,
- include a clinical baseline/residual option,
- be small enough for around 500 to a few thousand labeled examples,
- avoid full ViT fine-tuning unless sample size is expanded substantially.

### 8.2 Recommended Input Representations

```text
ECG embedding: PCLR, 320 dimensions
CXR embedding: RAD-DINO CLS token, 768 dimensions
EHR vector: structured clinical features, approximately 12 to 100 dimensions depending on feature expansion
modality mask: has_ecg, has_cxr, has_ehr
```

Optional additional features:

```text
ecg_quality_score
cxr_view_position
cxr_is_portable
days_ecg_from_echo
days_cxr_from_echo
height_missing
weight_missing
sbp_missing
dbp_missing
```

### 8.3 Low-Parameter Gated Fusion Model

Recommended architecture:

```text
ECG: 320 -> LayerNorm -> Linear(320, 64) -> GELU -> Dropout
CXR: 768 -> LayerNorm -> Linear(768, 64) -> GELU -> Dropout
EHR: d_ehr -> Linear(d_ehr, 64) -> GELU -> Dropout -> Linear(64, 64)
Mask embedding: 3 modality-presence bits -> Linear(3, 16)
Fusion input: concat(ecg_z, cxr_z, ehr_z, mask_z)
Gate: Linear(fusion_input, 3) -> softmax over available modalities
Weighted modality vector: sum_m gate_m * z_m
Final fused vector: concat(weighted_modality_vector, mask_z, optional_time_gap_features)
Trunk: Linear(..., 128) -> GELU -> Dropout -> Linear(128, 64) -> GELU
Heads: root regression, ascending regression, root ordinal, ascending ordinal, optional binary heads
```

### 8.4 Missing-Modality Handling

For a missing modality:

- set its embedding to zeros after projection,
- set corresponding mask bit to 0,
- ensure gate logits for missing modalities are masked before softmax,
- apply modality dropout during training to improve robustness.

Pseudo-code:

```python
z_ecg = project_ecg(ecg_embedding) if has_ecg else zeros_like_ecg_projection
z_cxr = project_cxr(cxr_embedding) if has_cxr else zeros_like_cxr_projection
z_ehr = project_ehr(ehr_vector) if has_ehr else zeros_like_ehr_projection

z_stack = torch.stack([z_ecg, z_cxr, z_ehr], dim=1)  # batch, modality, dim
mask = torch.tensor([has_ecg, has_cxr, has_ehr])     # batch, modality

gate_logits = gate_network(concat_all_features)
gate_logits = gate_logits.masked_fill(mask == 0, -1e9)
gate_weights = softmax(gate_logits, dim=1)

z_weighted = (gate_weights.unsqueeze(-1) * z_stack).sum(dim=1)
z_final = torch.cat([z_weighted, mask_features, time_gap_features], dim=-1)
```

### 8.5 Multi-Task Heads

Recommended heads:

```text
root_regression_head: outputs root diameter in cm
asc_regression_head: outputs ascending diameter in cm
root_ordinal_head: outputs cumulative logits for >=4.0, >=4.5, >=5.0
asc_ordinal_head: outputs cumulative logits for >=4.0, >=4.5, >=5.0
root_binary_heads: optional direct logits for >=4.0 and >=4.5
asc_binary_heads: optional direct logits for >=4.0 and >=4.5
```

Do not train direct >=5.0 binary heads if there are zero positives.

### 8.6 Multi-Task Loss

Recommended default:

```text
loss = w_root_reg * Huber(root_pred, root_diameter)
     + w_asc_reg  * Huber(asc_pred, asc_diameter)
     + w_root_ord * CORAL(root_ordinal_logits, root_grade)
     + w_asc_ord  * CORAL(asc_ordinal_logits, asc_grade)
     + w_root_bin * BCE_or_focal(root_binary_logits, root_binary_labels)
     + w_asc_bin  * BCE_or_focal(asc_binary_logits, asc_binary_labels)
```

Initial weights:

```text
w_root_reg = 1.0
w_asc_reg  = 1.0
w_root_ord = 0.5
w_asc_ord  = 0.5
w_root_bin = 0.5
w_asc_bin  = 0.5
```

Tune weights in inner CV or use uncertainty-based weighting. Do not tune based on outer test performance.

### 8.7 Recommended Deep Model Hyperparameters

```text
projection_dim: [32, 64, 128]
trunk_hidden_dim: [64, 128]
dropout: [0.3, 0.4, 0.5, 0.6]
weight_decay: [0.0001, 0.001, 0.01]
learning_rate_heads: [0.0001, 0.0003, 0.001]
batch_size: [16, 32, 64]
early_stopping_patience: [10, 20]
gradient_clip_norm: 1.0
max_epochs: 200
optimizer: AdamW
scheduler: ReduceLROnPlateau or cosine with warmup
```

### 8.8 Parameter-Efficient Fine-Tuning

Default should remain frozen encoders. If fine-tuning is attempted, use small trainable modules:

```text
linear probe
MLP probe
last block unfreeze
LoRA adapters
bias-only fine-tuning
adapter layers
```

Avoid full RAD-DINO ViT fine-tuning on around 520 images. It is likely to overfit.

### 8.9 Residual Fusion Model

Add an architecture mode:

```text
mode: residual_prediction
```

In this mode:

1. EHR clinical baseline predicts expected diameter.
2. ECG/CXR/fusion model predicts residual diameter.
3. Final prediction is clinical prediction plus residual prediction.

This is especially useful for checking whether ECG/CXR add information beyond age, sex, and body size.

---

## 9. ECG-Specific Implementation Changes

### 9.1 ECG Quality Control

Add ECG QC before embedding extraction or model training.

QC checks:

```text
all_12_leads_present
expected_sampling_rate
expected_signal_length
no_all_zero_leads
no_constant_leads
no_extreme_amplitude_values
low_missingness
acceptable_noise_metric
lead_order_valid
```

Recommended derived flags:

```text
ecg_qc_pass
ecg_num_bad_leads
ecg_noise_score
ecg_is_paced_rhythm
ecg_has_afib
ecg_has_bundle_branch_block
ecg_heart_rate
```

Paced rhythm and severe conduction abnormalities may carry useful information, so do not automatically exclude them from all analyses. Instead:

- run primary analysis with basic signal-quality exclusions,
- run sensitivity analysis excluding paced rhythms,
- include rhythm/conduction flags as optional features.

### 9.2 Classical ECG Features

Add classical features alongside PCLR embeddings.

Potential features:

```text
heart_rate
pr_interval
qrs_duration
qt_interval
qtc_interval
p_axis
qrs_axis
t_axis
rv5_voltage
sv1_voltage
sokolow_lyon_voltage
cornell_voltage
lvh_flag
rbbb_flag
lbbb_flag
paced_rhythm_flag
afib_flag
atrial_flutter_flag
first_degree_av_block_flag
```

These may be extractable from machine measurements or ECG reports. Ensure they are available before echo.

### 9.3 ECG Model Variants

Implement and compare:

```text
PCLR embedding + logistic/ridge
PCLR embedding + GBDT
PCLR embedding + classical ECG features
classical ECG features only
PCLR + EHR late fusion
PCLR + CXR + EHR late fusion
```

Optional advanced variants:

```text
PCLR embedding with small task-specific adapter
PCLR embedding with supervised contrastive fine-tuning
ResNet1D initialized from self-supervised pretraining
```

Avoid training a large 1D ResNet from scratch on the complete-case cohort as the main model.

### 9.4 ECG Fine-Tuning Strategy

If waveform-level fine-tuning is used:

1. Start with frozen backbone and train only the head.
2. Unfreeze only the last block or adapters.
3. Use low learning rate.
4. Use strong early stopping.
5. Use patient-level augmentation only within training folds.

Suggested hyperparameters:

```text
head_lr: [0.0001, 0.0003, 0.001]
backbone_lr: [0.000005, 0.00001, 0.00005]
weight_decay: [0.0001, 0.001, 0.01]
dropout: [0.3, 0.5]
batch_size: [16, 32, 64]
max_epochs: 100
patience: 10
```

### 9.5 ECG Augmentation

Use only physiologically plausible augmentations:

```text
small Gaussian noise
small amplitude scaling
time shift within a small range
random lead dropout during training only
baseline wander simulation
```

Avoid augmentations that distort clinically meaningful intervals or lead geometry unless validated.

---

## 10. CXR-Specific Implementation Changes

### 10.1 CXR View Handling

CXR projection strongly affects apparent aortic and mediastinal size. Add explicit view handling.

Required fields:

```text
cxr_view_position      # PA, AP, lateral, unknown
cxr_is_portable
cxr_study_date
cxr_study_time_if_available
cxr_days_from_echo
```

Recommended analyses:

```text
all CXR views
PA only
AP/portable only
PA vs AP subgroup metrics
view position as model covariate
```

Primary model should either:

- include view position as a covariate, or
- train/evaluate view-specific models.

### 10.2 CXR Image Preprocessing

Standardize preprocessing across runs.

Recommended steps:

```text
load image
convert to single-channel or three-channel consistently
resize to model input size
normalize using backbone-compatible normalization
preserve aspect ratio when possible
record original image dimensions
record crop or padding parameters
```

Do not use horizontal flipping. It changes anatomy.

Allowed augmentations for training:

```text
small rotation
small translation
small contrast/brightness jitter
mild random crop or resize crop
small Gaussian noise
```

### 10.3 ROI Cropping

Global CXR embeddings may dilute aortic information. Add ROI variants.

Minimum ROI experiments:

```text
whole image
central mediastinum crop
upper mediastinum/aortic knob crop
cardiac silhouette crop
multi-crop concatenation or averaging
```

Simple initial crop definitions can be image-coordinate heuristics:

```text
central mediastinum crop:
  x range: 25% to 75% of image width
  y range: 10% to 75% of image height

upper mediastinum crop:
  x range: 30% to 70% of image width
  y range: 10% to 55% of image height

cardiomediastinal crop:
  x range: 15% to 85% of image width
  y range: 20% to 90% of image height
```

These are only starting points. If segmentation tools are available, prefer anatomy-based crops.

### 10.4 CXR Measurement Baselines

Add simple interpretable image features. These can be manual, semi-automated, or model-derived.

Potential CXR features:

```text
aortic_knob_width
mediastinal_width
cardiothoracic_ratio
cardiac_silhouette_width
thoracic_width
predicted_aortic_contour_width
left_upper_mediastinal_prominence_score
```

Even if noisy, these features provide a strong sanity check. If a simple aortic knob or mediastinal width feature performs similarly to the embedding model, the paper can discuss interpretability and practical screening.

### 10.5 RAD-DINO Variants

Implement:

```text
RAD-DINO CLS whole image
RAD-DINO CLS mediastinum crop
RAD-DINO CLS upper mediastinum crop
RAD-DINO patch pooling over ROI
RAD-DINO + view metadata
RAD-DINO + simple measurement features
```

For patch pooling:

- extract patch tokens,
- average tokens inside ROI if mapping is available,
- compare CLS-only vs ROI-pooled features.

### 10.6 CXR Fine-Tuning Strategy

Default: frozen RAD-DINO.

If fine-tuning is attempted:

```text
linear probe first
MLP probe second
last transformer block unfreeze third
LoRA/adapters fourth
full fine-tuning only with much larger labeled data
```

Suggested hyperparameters:

```text
linear_probe_lr: [0.0003, 0.001, 0.003]
adapter_lr: [0.00001, 0.00003, 0.0001]
weight_decay: [0.0001, 0.001, 0.01]
dropout: [0.3, 0.5]
batch_size: [8, 16, 32]
max_epochs: 100
patience: 10
```

### 10.7 Optional CXR Report Baseline

If radiology reports are available before or near the CXR date, create a separate report-text baseline. Keep it separate from image-only claims.

Potential report terms:

```text
tortuous aorta
ectatic aorta
aortic atherosclerosis
aneurysm
aneurysmal dilation
widened mediastinum
prominent mediastinum
cardiomegaly
```

Important:

- label this as `CXR report text baseline`, not image-only AI,
- avoid post-echo reports,
- discuss that report text may encode radiologist prior knowledge.

---

## 11. EHR Feature Expansion

### 11.1 Structured Comorbidities

Add comorbidity flags from diagnosis codes before echo.

Recommended flags:

```text
hypertension
hyperlipidemia
diabetes
coronary_artery_disease
heart_failure
chronic_kidney_disease
atrial_fibrillation
stroke_or_tia
peripheral_vascular_disease
smoking_or_tobacco_use
bicuspid_aortic_valve
aortic_stenosis
aortic_regurgitation
mitral_valve_disease
connective_tissue_disease
marfan_syndrome
loeys_dietz_syndrome
ehlers_danlos_syndrome
turner_syndrome
coarctation_of_aorta
known_thoracic_aortic_aneurysm_prior
```

Use prior-only diagnosis windows:

```text
all history before echo
within 1 year before echo
within 5 years before echo
```

### 11.2 Medication Features

Add medication flags if prescriptions/administrations are available before echo.

Potential medication classes:

```text
beta_blocker
ace_inhibitor
arb
calcium_channel_blocker
diuretic
statin
anticoagulant
antiplatelet
```

Medication features can be proxies for hypertension and cardiovascular disease severity. They can also introduce confounding, so compare models with and without medication features.

### 11.3 Longitudinal Vitals

Instead of only nearest values, add summaries before echo:

```text
nearest_sbp
nearest_dbp
median_sbp_1y
median_dbp_1y
max_sbp_1y
max_dbp_1y
min_sbp_1y
min_dbp_1y
n_bp_measurements_1y
time_since_nearest_bp
```

Similar summaries can be added for weight and BMI.

### 11.4 Missingness as Signal

For each important EHR variable, add missingness indicators:

```text
height_missing
weight_missing
bmi_missing
bsa_missing
sbp_missing
dbp_missing
heart_rate_missing
smoking_missing
```

Missingness patterns may reflect care setting and should be available to models explicitly.

### 11.5 EHR Leakage Guardrails

Create a feature provenance table.

Required columns:

```text
feature_name
source_table
source_column
timestamp_column
allowed_time_relation_to_echo
uses_post_echo_data
leakage_risk_level
notes
```

Allowed values for `leakage_risk_level`:

```text
low
medium
high
exclude_main
```

The main model should use only low and carefully reviewed medium-risk features.

---

## 12. Model Families to Keep and Compare

### 12.1 Simple Linear/Logistic Models

Purpose:

- interpretable clinical baseline,
- stable in low-sample settings,
- useful for calibration.

Use for:

```text
EHR clinical baseline
late fusion meta-learner
residual diameter baseline
```

### 12.2 HistGradientBoosting Models

Purpose:

- strong tabular/embedding baseline,
- handles nonlinear effects,
- often stronger than small deep models in low-data settings.

Use for:

```text
ECG embedding only
CXR embedding only
EHR only
concatenated ECG+CXR+EHR
PCA-reduced embeddings
residual targets
```

### 12.3 Late Fusion / Stacking

Purpose:

- allows each modality to train on larger modality-specific cohorts,
- strong in missing-modality settings,
- interpretable modality contribution.

This should likely be one of the primary models.

### 12.4 Low-Parameter Deep Fusion

Purpose:

- test whether learned multimodal interactions help,
- support missing-modality masks,
- support multi-task learning.

Do not overemphasize deep fusion if it does not outperform simpler baselines. A negative result is still useful.

### 12.5 Residual Models

Purpose:

- test whether ECG/CXR add signal beyond clinical/body-size baseline,
- improve scientific interpretability.

Recommended variants:

```text
EHR predicts raw diameter
ECG predicts residual after EHR
CXR predicts residual after EHR
ECG+CXR predicts residual after EHR
EHR + residual ECG/CXR final prediction
```

---

## 13. Training and Hyperparameter Search

### 13.1 Nested Cross-Validation

Use nested CV for model selection when feasible.

Outer loop:

- fixed 5-fold patient-level split on complete-case cohort,
- used only for final performance estimation.

Inner loop:

- model selection,
- hyperparameter tuning,
- calibration,
- early stopping,
- feature selection,
- PCA dimension selection.

Do not select hyperparameters using outer-test performance.

### 13.2 HistGradientBoosting Hyperparameters

Recommended grid:

```yaml
hgbdt:
  learning_rate: [0.01, 0.02, 0.05, 0.1]
  max_iter: [200, 500, 1000]
  max_leaf_nodes: [7, 15, 31]
  min_samples_leaf: [10, 20, 50]
  l2_regularization: [0.0001, 0.001, 0.01, 0.1, 1.0]
  early_stopping: [true]
```

For very low positive counts, favor:

```text
smaller max_leaf_nodes
larger min_samples_leaf
stronger l2_regularization
```

### 13.3 PCA for Embeddings

Embedding dimensions are high relative to event count. Test PCA inside the fold.

Recommended grid:

```yaml
pca:
  enabled: [true, false]
  n_components: [16, 32, 64, 128]
  whiten: [false]
```

Important: PCA must be fit only on training data inside each fold.

### 13.4 Logistic Regression Hyperparameters

```yaml
logistic_regression:
  penalty: [l2]
  C: [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
  class_weight: [null, balanced]
  solver: [lbfgs]
  max_iter: [5000]
```

### 13.5 Ridge Regression Hyperparameters

```yaml
ridge:
  alpha: [0.1, 1.0, 3.0, 10.0, 30.0, 100.0]
```

### 13.6 Deep Fusion Hyperparameters

```yaml
deep_fusion:
  projection_dim: [32, 64, 128]
  trunk_hidden_dim: [64, 128]
  dropout: [0.3, 0.4, 0.5, 0.6]
  weight_decay: [0.0001, 0.001, 0.01]
  learning_rate: [0.0001, 0.0003, 0.001]
  batch_size: [16, 32, 64]
  max_epochs: [200]
  early_stopping_patience: [10, 20]
  gradient_clip_norm: [1.0]
  optimizer: [AdamW]
```

### 13.7 Loss Functions for Imbalance

For binary heads, test:

```text
class-weighted BCE
focal loss with gamma = 1
focal loss with gamma = 2
class-balanced loss using effective number of samples
```

Default focal loss settings:

```yaml
focal_loss:
  gamma: [1.0, 2.0]
  alpha: auto_from_training_prevalence
```

Do not compute class weights using full data. Compute them inside each training fold.

### 13.8 Early Stopping

For deep models:

- use inner-validation loss or primary validation metric,
- patience 10 to 20 epochs,
- restore best checkpoint,
- save validation curves.

For endpoints with few positives, validation AUROC may be unstable. Prefer a composite validation objective:

```text
validation_loss + auxiliary regression performance
```

or use repeated inner folds.

---

## 14. Evaluation Plan

### 14.1 Binary Metrics

For each binary endpoint:

```text
AUROC
AUPRC
prevalence
Brier score
calibration intercept
calibration slope
sensitivity at selected thresholds
specificity at selected thresholds
PPV
NPV
F1
balanced accuracy
```

AUPRC should always be reported with prevalence.

### 14.2 Regression Metrics

For diameter regression:

```text
MAE
RMSE
R2
Pearson correlation
Spearman correlation
mean error
median absolute error
calibration of predicted diameter if relevant
```

Add Bland-Altman-style plots for predicted vs echo diameter if useful.

### 14.3 Ordinal Metrics

```text
quadratic_weighted_kappa
mean_absolute_grade_error
ordinal confusion matrix
```

If severe grade has zero positives, clearly state that the 4-grade metric is underpowered and severe discrimination is not estimable.

### 14.4 Bootstrap Confidence Intervals

Use patient-level bootstrap on OOF predictions.

Recommended:

```text
n_bootstrap: 2000
resample_unit: subject_id
ci: 2.5th and 97.5th percentiles
```

Also bootstrap metric differences between models:

```text
AUROC_model_A - AUROC_model_B
AUPRC_model_A - AUPRC_model_B
MAE_model_A - MAE_model_B
R2_model_A - R2_model_B
```

### 14.5 Calibration

Generate for each binary endpoint:

```text
calibration curve
Brier score
calibration intercept
calibration slope
expected calibration error if implemented
```

Recommended binning:

```text
quantile bins: 5 or 10 depending on event count
```

For low event counts, use fewer bins.

### 14.6 Clinical Utility Metrics

Report risk-enrichment performance.

For each binary endpoint:

```text
top 5% risk: event rate, sensitivity, PPV, number-needed-to-echo
top 10% risk: event rate, sensitivity, PPV, number-needed-to-echo
top 20% risk: event rate, sensitivity, PPV, number-needed-to-echo
threshold for 90% sensitivity: specificity, PPV, NPV
threshold for 90% specificity: sensitivity, PPV, NPV
```

Number-needed-to-echo:

```text
number_needed_to_echo = 1 / PPV
```

If PPV is 0, report as not defined.

### 14.7 Decision Curve Analysis

If implemented, decision curve analysis should report net benefit across plausible threshold probabilities.

Suggested threshold probability grid:

```text
0.02 to 0.30 for AD >=4.0
0.005 to 0.10 for AD >=4.5
```

Compare:

```text
treat none
screen all
EHR baseline
best multimodal model
```

### 14.8 Subgroup Analyses

Minimum subgroups:

```text
sex
age group
BSA quartile
CXR view: PA vs AP/portable
time gap quartile between CXR and echo
time gap quartile between ECG and echo
inpatient vs outpatient if available
```

Optional subgroups:

```text
race/ethnicity if available and sufficiently complete
hypertension status
known valve disease status
image quality strata
ECG rhythm strata
```

Report sample size and event count for every subgroup. Do not overinterpret small subgroups.

### 14.9 Main Tables and Figures

Required outputs:

```text
cohort flow diagram
label distribution histograms
event count table by endpoint and fold
main model comparison table
EHR incremental value table
modality ablation table
calibration plots
clinical utility table
subgroup performance table
CXR view-specific performance table
alignment-window sensitivity table
```

---

## 15. Recommended Configuration Structure

Use config-driven experiments so agents can run reproducible variants.

Example:

```yaml
project:
  name: aortic_dilation_multimodal
  seed: 42
  output_dir: runs/experiment_name

data:
  master_cohort_csv: data/processed/master_cohort.csv
  fold_assignments_csv: data/processed/fold_assignments.csv
  max_days_offset: 180
  require_pre_echo_modalities: false
  cohort: triple

modalities:
  ecg:
    enabled: true
    embedding_path: data/features/pclr_embeddings.pt
    include_classical_features: true
    qc_required: true
  cxr:
    enabled: true
    embedding_path: data/features/rad_dino_embeddings.pt
    include_view_metadata: true
    roi_mode: whole_image
  ehr:
    enabled: true
    feature_set: expanded_pre_echo
    include_missingness_indicators: true

targets:
  - root_ad_4p0
  - root_ad_4p5
  - asc_ad_4p0
  - asc_ad_4p5
  - root_diameter_cm
  - asc_diameter_cm

splitting:
  outer_cv_folds: 5
  inner_cv_folds: 5
  stratify_on: any_ad_4p0
  patient_id_column: subject_id

model:
  family: late_fusion
  base_model_family: hgbdt
  meta_model_family: logistic_regression
  leakage_safe: true
  calibration: none

preprocessing:
  imputation: median
  scaling: standard_for_linear_only
  pca:
    enabled: true
    n_components: 64

training:
  nested_cv: true
  bootstrap_iterations: 2000
  save_oof_predictions: true
  save_fold_models: true

evaluation:
  binary_metrics: [auroc, auprc, brier, calibration_slope, calibration_intercept]
  regression_metrics: [mae, rmse, r2, pearson, spearman]
  ordinal_metrics: [qwk, mean_absolute_grade_error]
  clinical_utility: true
  subgroup_analysis: true
```

---

## 16. Required Output Artifacts

Each run should produce a standardized output directory.

```text
runs/<experiment_name>/
  config.yaml
  run_manifest.json
  cohort_summary.csv
  fold_event_counts.csv
  selected_hyperparameters.csv
  metrics_by_fold.csv
  metrics_summary.csv
  metrics_bootstrap_ci.csv
  metric_differences_vs_ehr.csv
  oof_predictions.csv
  calibration_metrics.csv
  clinical_utility.csv
  subgroup_metrics.csv
  leakage_audit.json
  plots/
    cohort_flow.png
    root_diameter_histogram.png
    ascending_diameter_histogram.png
    root_calibration.png
    ascending_calibration.png
    model_comparison_auprc.png
    model_comparison_auroc.png
    clinical_utility_top_risk.png
  models/
    fold_0/
    fold_1/
    fold_2/
    fold_3/
    fold_4/
```

### 16.1 Run Manifest

`run_manifest.json` should include:

```json
{
  "experiment_name": "example",
  "timestamp": "YYYY-MM-DDTHH:MM:SS",
  "seed": 42,
  "git_commit": "unknown_if_not_available",
  "data_hashes": {
    "master_cohort_csv": "...",
    "fold_assignments_csv": "..."
  },
  "n_subjects": 522,
  "modalities": ["ecg", "cxr", "ehr"],
  "model_family": "late_fusion",
  "targets": ["root_ad_4p0", "asc_ad_4p0"],
  "outer_cv_folds": 5,
  "inner_cv_folds": 5
}
```

### 16.2 OOF Prediction File

`oof_predictions.csv` should be the central artifact for analysis.

Required columns:

```text
subject_id
fold_id
model_name
model_family
modality_set
site
endpoint
target_type
y_true
pred_value
pred_probability
pred_diameter_cm
pred_grade
echo_date
ecg_date
cxr_date
ecg_days_from_echo
cxr_days_from_echo
has_ecg
has_cxr
has_ehr
cxr_view_position
cxr_is_portable
```

### 16.3 Leakage Audit File

`leakage_audit.json` should include for every fold:

```json
{
  "fold_id": 0,
  "outer_train_n": 418,
  "outer_test_n": 104,
  "outer_train_test_overlap_n": 0,
  "base_model_train_overlap_with_outer_test": {
    "ecg": 0,
    "cxr": 0,
    "ehr": 0
  },
  "preprocessor_fit_overlap_with_outer_test": 0,
  "pca_fit_overlap_with_outer_test": 0,
  "meta_train_overlap_with_outer_test": 0,
  "status": "pass"
}
```

---

## 17. Testing and Quality Assurance

### 17.1 Data Split Tests

```text
assert no subject_id overlap between outer train and outer test
assert no subject_id overlap between inner train and inner validation
assert every subject has exactly one outer fold
assert fold labels are stable across runs with the same seed
```

### 17.2 Preprocessing Leakage Tests

For each preprocessor object, save the IDs used for fitting.

Tests:

```text
assert outer_test_ids not in imputer.fit_ids
assert outer_test_ids not in scaler.fit_ids
assert outer_test_ids not in pca.fit_ids
assert outer_test_ids not in feature_selector.fit_ids
assert outer_test_ids not in calibrator.fit_ids
```

### 17.3 Stacking Tests

```text
assert base_model_train_ids do not include outer_test_ids
assert inner_valid_ids do not appear in their corresponding inner base training IDs
assert meta_learner_train_ids are exactly outer_train complete-case IDs
assert meta_learner_test_ids are exactly outer_test complete-case IDs
assert every outer_train patient has OOF base predictions before meta training
assert every outer_test patient has base predictions from models that excluded that patient
```

### 17.4 Target Tests

```text
assert binary labels match threshold definitions
assert ordinal labels match bin definitions
assert no AUROC/AUPRC computed when y_true has one class
assert severe endpoint marked not estimable when zero positives
assert residualization model fit only on training fold
```

### 17.5 Date Tests

```text
assert abs(ecg_days_from_echo) <= max_days_offset when ECG is used
assert abs(cxr_days_from_echo) <= max_days_offset when CXR is used
assert pre_echo_only mode has modality_date <= echo_date
assert EHR feature timestamps are <= echo_date for main model
```

### 17.6 Metric Tests

```text
assert AUPRC baseline prevalence is reported
assert bootstrap resamples by subject_id, not rows
assert confidence intervals are finite or explicitly marked unavailable
assert metric differences use paired predictions on the same subjects
```

---

## 18. Specific Implementation Backlog

### 18.1 Data and Cohort

- [ ] Create `master_cohort.csv` with explicit modality flags and dates.
- [ ] Create immutable `fold_assignments.csv` using patient-level stratification.
- [ ] Add fold event-count table generation.
- [ ] Add support for 30, 90, 180, and 365 day windows.
- [ ] Add pre-echo-only sensitivity mode.
- [ ] Add cohort flags for EHR-only, ECG-only, CXR-only, pairwise, triple, and any-modality cohorts.

### 18.2 EHR Baseline

- [ ] Implement minimum clinical feature set.
- [ ] Add BSA calculation and missingness indicators.
- [ ] Add expanded comorbidity flags from prior diagnosis codes.
- [ ] Add medication class flags if feasible.
- [ ] Add longitudinal BP summaries.
- [ ] Implement logistic, ridge, and HGBDT EHR baselines.
- [ ] Add metric deltas vs EHR baseline.

### 18.3 Late Fusion

- [ ] Refactor stacking into outer-fold-safe pipeline.
- [ ] Add broader per-modality training cohorts excluding outer-test IDs.
- [ ] Generate inner OOF predictions for meta-learner training.
- [ ] Save base predictions and meta predictions separately.
- [ ] Add leakage audit output.
- [ ] Add unit tests for all leakage risks.

### 18.4 Targets

- [ ] Add central target registry.
- [ ] Add indexed diameter targets.
- [ ] Add residualized target mode.
- [ ] Add ordinal labels for both sites.
- [ ] Add automatic endpoint validity checks.

### 18.5 Deep Fusion

- [ ] Implement low-parameter gated fusion model.
- [ ] Add missing-modality masks.
- [ ] Add modality dropout.
- [ ] Add multi-task root/ascending heads.
- [ ] Add Huber + CORAL + BCE/focal multi-task loss.
- [ ] Add residual prediction mode.
- [ ] Add training curves and checkpoint saving.

### 18.6 ECG

- [ ] Add ECG QC flags.
- [ ] Add classical ECG features if available.
- [ ] Add PCLR + classical feature variants.
- [ ] Add ECG rhythm/conduction sensitivity analyses.
- [ ] Add adapter or last-block fine-tuning only if enough data is available.

### 18.7 CXR

- [ ] Add CXR view metadata.
- [ ] Add PA-only and AP/portable-only analyses.
- [ ] Add ROI crop feature extraction.
- [ ] Add simple CXR measurement baselines.
- [ ] Add RAD-DINO ROI embedding variants.
- [ ] Add optional CXR report-text baseline as a separate analysis.

### 18.8 Evaluation

- [ ] Add bootstrap CIs for every metric.
- [ ] Add paired metric difference CIs vs EHR baseline.
- [ ] Add calibration metrics and curves.
- [ ] Add top-risk clinical utility tables.
- [ ] Add number-needed-to-echo.
- [ ] Add decision curve analysis if feasible.
- [ ] Add subgroup analysis.
- [ ] Add alignment-window sensitivity table.

---

## 19. Recommended Main Experiment Matrix

The following experiment matrix should be implemented first.

### 19.1 Primary Complete-Case Evaluation

Evaluation cohort: ECG + CXR + EHR complete-case patients.

Models:

```text
EHR logistic/ridge clinical baseline
ECG PCLR GBDT
CXR RAD-DINO GBDT
ECG + CXR concatenated GBDT
ECG + EHR concatenated GBDT
CXR + EHR concatenated GBDT
ECG + CXR + EHR concatenated GBDT
leakage-safe late fusion
low-parameter gated deep fusion
residual ECG/CXR after EHR baseline
```

Targets:

```text
root_ad_4p0
root_ad_4p5
asc_ad_4p0
asc_ad_4p5
root_diameter_cm
asc_diameter_cm
root_diameter_per_bsa
asc_diameter_per_bsa
root_residual_diameter
asc_residual_diameter
```

### 19.2 Larger-Cohort Training, Complete-Case Testing

Evaluation cohort: same complete-case outer folds.

Base training cohorts:

```text
ECG base model trained on all ECG-labeled patients excluding outer-test IDs
CXR base model trained on all CXR-labeled patients excluding outer-test IDs
EHR base model trained on all EHR-labeled patients excluding outer-test IDs
```

Meta-learner:

```text
trained only on complete-case outer-train OOF predictions
```

This is the key fair way to use larger single-modality cohorts.

### 19.3 Missing-Modality Evaluation

Evaluation cohorts:

```text
EHR only
ECG + EHR
CXR + EHR
ECG + CXR + EHR
any modality
```

Models:

```text
late fusion with missingness masks
gated fusion with missingness masks
EHR baseline
```

Goal:

- determine whether the model is useful when not every patient has all modalities.

### 19.4 Sensitivity Analyses

Run main model under:

```text
max_days_offset = 30
max_days_offset = 90
max_days_offset = 180
max_days_offset = 365
pre_echo_only = true
pre_echo_only = false
PA_CXR_only
AP_portable_CXR_only
exclude_ambiguous_echo_labels
exclude_paced_ECG
```

---

## 20. Suggested Code Organization

Recommended package structure:

```text
src/
  data/
    build_master_cohort.py
    build_fold_assignments.py
    ehr_features.py
    ecg_features.py
    cxr_features.py
    echo_label_qc.py
    cohort_filters.py
  targets/
    target_registry.py
    target_builders.py
    residualization.py
  models/
    ehr_baselines.py
    gbdt_models.py
    late_fusion.py
    gated_fusion.py
    losses.py
    calibration.py
  training/
    nested_cv.py
    train_gbdt.py
    train_late_fusion.py
    train_deep_fusion.py
  evaluation/
    metrics.py
    bootstrap.py
    calibration_metrics.py
    clinical_utility.py
    subgroup_analysis.py
    plots.py
  utils/
    leakage_checks.py
    hashing.py
    logging.py
    seed.py
configs/
  baseline_ehr.yaml
  gbdt_concat.yaml
  late_fusion.yaml
  gated_fusion.yaml
  residual_model.yaml
scripts/
  run_experiment.py
  run_all_primary_experiments.sh
  make_tables.py
  make_figures.py
```

---

## 21. Definition of Done

A model experiment is complete only when all of the following are true:

- [ ] Uses immutable patient-level fold assignments.
- [ ] Saves full config and run manifest.
- [ ] Passes leakage checks.
- [ ] Saves OOF predictions.
- [ ] Reports event counts per fold.
- [ ] Reports AUROC and AUPRC with bootstrap CIs for binary endpoints.
- [ ] Reports MAE and R2 with bootstrap CIs for regression endpoints.
- [ ] Reports calibration metrics for binary endpoints.
- [ ] Reports clinical utility metrics.
- [ ] Reports metric differences vs EHR clinical baseline.
- [ ] Marks non-estimable endpoints correctly.
- [ ] Produces plots and tables in a standardized output directory.
- [ ] Can be rerun from a single config file.

A paper-ready result set is complete only when the following are also true:

- [ ] AORTA-style EHR baseline is included.
- [ ] Late fusion is leakage-safe and audited.
- [ ] Larger per-modality training cohorts are used correctly.
- [ ] Missing-modality evaluation is included.
- [ ] Body-size-indexed and residualized targets are included.
- [ ] Label QC and alignment-window sensitivity analyses are included.
- [ ] CXR view-specific and ROI analyses are included.
- [ ] ECG QC and classical feature analysis are included.
- [ ] Subgroup analysis is included.
- [ ] All claims are supported by confidence intervals and event counts.

---

## 22. Recommended Reporting Emphasis

The implementation should support the following scientific conclusions if validated by results:

1. Whether ECG and CXR add incremental predictive value beyond clinical variables.
2. Whether late fusion is more reliable than deep fusion in a small, low-event multimodal cohort.
3. Whether root and ascending aorta prediction rely on different modality signals.
4. Whether the model provides clinically meaningful risk enrichment even if diagnostic performance is not yet sufficient for deployment.
5. Whether performance is robust to CXR view type, temporal alignment window, and label QC restrictions.

Avoid claims that the model replaces echo. The stronger and safer framing is risk enrichment or screening support.

---

## 23. Immediate Next Steps for Coding Agents

Start with this exact order:

1. Build immutable fold assignments and fold event-count reports.
2. Implement and validate EHR clinical baseline.
3. Refactor late fusion into a leakage-proof outer-fold pipeline.
4. Add OOF prediction saving and paired bootstrap comparison vs EHR.
5. Add calibration and clinical utility metrics.
6. Add target registry with indexed and residualized targets.
7. Add missing-modality cohorts and masks.
8. Add CXR view-specific preprocessing and ROI feature extraction.
9. Add ECG QC and classical ECG features.
10. Implement low-parameter gated fusion and residual fusion models.

This order prioritizes validity and publishability before architectural complexity.
