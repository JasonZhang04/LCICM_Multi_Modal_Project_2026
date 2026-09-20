# Multimodal Aortic-Dilation Model: Proposed Design and Repository Audit Brief

## Purpose

Use this document to audit the current implementation of the aortic-dilation project and determine which proposed improvements are appropriate. First inspect and describe the existing code and data flow. Do **not** assume the proposed design is already implemented, and do not modify the repository until the audit is complete and the user approves a patch plan.

The central modeling question is:

> After the model has already used the strongest modality—currently CXR—do ECG and EHR explain additional held-out variation or correct clinically important CXR errors?

Equal contribution from all modalities is not required. A useful multimodal model may remain mostly CXR-driven while ECG or EHR improves predictions for a smaller subset of difficult episodes.

## Current project context to verify

- Targets: continuous aortic-root and ascending-aorta diameters, plus clinically relevant dilation thresholds such as 4.0 cm and 4.5 cm.
- Label source: recorded echocardiographic diameter measurements. Raw echocardiogram images are not required when a valid diameter measurement is available.
- Inputs: CXR, 12-lead ECG, and EHR variables.
- Unit of modeling: an episode anchored to an aortic-diameter measurement event. A patient may contribute multiple episodes.
- Previously reported encoders: RAD-DINO for CXR and PCLR for ECG, with a separate EHR model. Verify what is currently used, whether encoders are frozen, and whether this has changed.
- Observed pattern: CXR currently drives most of the predictive performance.
- The cohort is no longer restricted to the earlier 522-patient complete-case cohort. Determine the current episode counts, unique-patient counts, target prevalence, and modality-overlap counts directly from the implementation/data summaries.

## Core proposed design

### 1. Episode construction and leakage control

Treat each echocardiographic measurement event as the episode anchor. Associate ECG, CXR, and EHR information using the prespecified temporal windows.

Non-negotiable safeguards:

1. Split data by **patient ID**, never by episode. All episodes, ECGs, CXRs, and echo measurements belonging to one patient must remain in one train/validation/test partition.
2. Do not treat every ECG × CXR combination around one echo as an independent example. If multiple studies belong to one episode, pool their predictions or sample one study per training iteration.
3. Check whether the same ECG or CXR is reused across multiple episodes. Reuse may be acceptable within a training partition, but it must be quantified and must never cross patient partitions.
4. Weight or sample episodes so that patients with many encounters do not dominate the loss.
5. Bootstrap uncertainty and confidence intervals by patient, not by episode.
6. Include modality-to-echo time differences as model features or perform sensitivity analyses across temporal windows.
7. For a screening interpretation, EHR features must be available before the prediction time. Exclude the target echo measurement, echo-report text containing the measurement, and downstream aortic diagnoses or interventions.

### 2. Label harmonization and quality control

Verify that the data pipeline distinguishes:

- aortic root;
- sinus of Valsalva;
- sinotubular junction;
- proximal/ascending aorta.

Also check units, plausible ranges, duplicate measurements from the same echo, measurement timestamps, and measurement provenance. Determine whether the presence of a recorded diameter is selective across sites, years, report templates, or patient characteristics.

For CXR, retain or recover acquisition metadata when available, especially PA versus AP/portable view, patient position, pixel spacing, and source-to-image distance. Projection and magnification can strongly affect apparent mediastinal and aortic width.

### 3. Train strong modality-specific predictors first

Train each modality-specific model on every eligible episode for which that modality and target label exist, rather than restricting all branch training to complete trimodal episodes.

Suggested branch outputs:

- predicted aortic-root diameter;
- predicted ascending-aorta diameter;
- optional uncertainty estimates;
- compact task-specific embeddings for fusion.

Suggested starting points:

- **CXR:** RAD-DINO with a small supervised adapter. Compare a full-image input with a mediastinal/aortic-region crop. Avoid unrestricted end-to-end fine-tuning until learning curves show it is justified.
- **ECG:** PCLR plus a target-specific adapter; compare against an aorta-specific or stronger ECG encoder if compatible official weights are available.
- **EHR:** a regularized shallow model using clinically relevant, strictly pre-index features. Complex EHR architectures should have to outperform this baseline.

All modality-specific predictions used to train the fusion model must be **out-of-fold (OOF)** predictions. For a held-out patient, the modality model producing that patient's fusion input must not have been trained on any episode from that patient.

### 4. Primary fusion proposal: regularized residual late fusion

Because CXR is currently strongest, use CXR as the anatomical base prediction and ask ECG/EHR to learn only the remaining correction.

For continuous diameter:

```text
final_prediction = cxr_prediction
                 + correction(ecg_prediction,
                              ehr_prediction,
                              cxr_prediction,
                              modality_quality,
                              time_gaps,
                              modality_presence)
```

For binary dilation:

```text
final_log_odds = cxr_log_odds
               + correction(ecg_features,
                            ehr_features,
                            quality/time/missingness metadata)
```

Start with a strongly regularized correction model such as Bayesian ridge, elastic net, or a small MLP. Shrink the correction toward zero so ECG/EHR are used only when they improve held-out performance.

Use separate target-specific correction heads or gates for the aortic root and ascending aorta. The relative value of CXR, ECG, and EHR may differ by anatomical target.

This residual architecture should be compared against:

- simple weighted averaging of calibrated unimodal predictions;
- regularized linear stacking;
- the current late-fusion implementation;
- CXR-only and all pairwise modality combinations.

### 5. Missing-modality training

Do not discard otherwise valid episodes only because one modality is missing. The model should accept available modality subsets through:

- modality-presence masks;
- modality-specific prediction heads;
- balanced sampling across modality combinations;
- modest modality dropout during training;
- a fallback fusion rule when CXR is absent.

Modality dropout and auxiliary unimodal losses can prevent the joint model from ignoring ECG/EHR. However, confirm that dropout improves complete-modality validation performance rather than merely improving robustness to missing inputs.

### 6. Recommended multitask objective

Continuous diameter should be the primary learning signal because it uses information from every labeled episode. Binary thresholds discard within-category information and may have substantially fewer positive examples.

A reasonable objective is:

```text
L_total = L_root_regression
        + L_ascending_regression
        + lambda_40 * L_threshold_4.0
        + lambda_45 * L_threshold_4.5
        + alpha * sum(L_unimodal_auxiliary)
```

Recommendations:

- Use Huber or Student-t loss for continuous diameters to reduce sensitivity to noisy/outlying echo measurements.
- Use target-availability masks so episodes with only root or only ascending measurements remain usable.
- If separate threshold heads are used, enforce `P(diameter >= 4.5) <= P(diameter >= 4.0)`.
- Alternatively, predict a continuous mean and variance and derive threshold probabilities from the predictive distribution.
- Use modest class weighting or balanced sampling for rare thresholds and recalibrate probabilities afterward.
- Do not adopt focal loss without demonstrating improved held-out AUPRC and acceptable calibration.

## Assessment of advanced ideas

### Bayesian integration

Recommended as a **fusion-layer** method, not initially as a fully Bayesian end-to-end neural network.

Potential uses:

- shrink weak modality contributions toward zero;
- incorporate uncertainty from each unimodal model;
- model differences associated with CXR view, site, scanner, calendar period, or label provenance;
- account for episode clustering during model estimation.

Naive precision weighting or naive Bayes is not sufficient because modality errors are likely correlated and the modalities are not conditionally independent.

### Cooperative learning

Potentially useful once all unimodal branches are independently competent. A cooperative objective adds a modest agreement penalty between branch predictions while retaining the fused prediction loss.

Audit recommendation:

- tune the agreement coefficient from zero upward;
- include zero as a serious candidate;
- monitor whether ECG/EHR standalone performance deteriorates during joint training;
- avoid forcing the indirect ECG/EHR signals to imitate CXR so strongly that complementary information disappears.

### Cross-attention

Treat as a second-stage experiment. Cross-attention is most meaningful when it operates on CXR patch tokens and ECG temporal/lead tokens. Attention over only three global modality vectors is primarily a complicated gating mechanism.

If tested:

- begin with frozen or mostly frozen encoders;
- project modalities into a small shared dimension;
- use one small cross-attention block;
- include quality, time-gap, and missingness information;
- compare against a parameter-matched gated MLP;
- justify it only if it improves patient-held-out performance consistently.

### Teacher-student learning

Most promising use:

1. Train modality-specific teachers on their maximum eligible cohorts.
2. Generate leakage-free OOF soft predictions.
3. Train a small multimodal student using both true labels and teacher predictions.

A multimodal teacher can also distill to unimodal students for deployment when modalities are missing. Avoid simply forcing ECG to reproduce the CXR teacher, because that may strengthen shared CXR-like signal without learning ECG's unique residual information.

### Masked self-supervised learning

Lower priority unless a large local unlabeled pool is available. RAD-DINO and PCLR are already self-supervised encoders. Additional masked reconstruction on a small local cohort may learn acquisition patterns, ribs, lungs, or devices rather than aortic structure.

Before adopting new self-supervised pretraining, compare it with:

- frozen foundation-model features;
- a small supervised adapter;
- partial last-block fine-tuning;
- CXR region-of-interest adaptation;
- task-specific ECG pretraining.

## Required complementarity diagnostics

Use identical patient-held-out folds for all comparisons.

1. Train CXR-, ECG-, and EHR-only models.
2. Generate OOF predictions and residuals for every branch.
3. Measure whether ECG/EHR predictions explain CXR residuals for root and ascending targets separately.
4. Measure correlations between unimodal residuals. Highly correlated errors imply limited fusion benefit.
5. Compare CXR with CXR+ECG, CXR+EHR, and CXR+ECG+EHR.
6. Evaluate modality ablation or permutation at test time to determine whether the fusion model actually uses each modality.
7. Stratify incremental value by CXR view/quality, temporal gap, target, and CXR confidence.
8. Plot learning curves against the number of **unique patients**, not merely episodes.

The multimodal model should be judged by incremental changes in:

- MAE and R-squared for continuous diameter;
- AUPRC and AUROC for dilation;
- sensitivity/specificity at the intended screening operating point;
- Brier score and calibration;
- patient-clustered confidence intervals for differences from CXR-only.

## Prioritized experiment sequence

| Priority | Experiment | Purpose |
|---|---|---|
| 1 | Patient-grouped split and leakage audit | Establish trustworthy evaluation |
| 2 | Label, temporal, view, and episode-duplication audit | Remove avoidable noise and pseudo-replication |
| 3 | Strong unimodal models trained on maximum eligible cohorts | Ensure each branch is competent |
| 4 | OOF residual and error-correlation analysis | Determine whether complementarity exists |
| 5 | Linear stacking and regularized residual fusion | Extract incremental signal with low overfitting risk |
| 6 | Multitask continuous + threshold objective | Use labels more efficiently |
| 7 | Missing-modality training, auxiliary losses, and modality dropout | Prevent branch neglect and expand usable data |
| 8 | Target-specific gated fusion | Learn patient- and target-specific modality reliability |
| 9 | Cooperative-learning ablation | Test whether modest agreement improves shared signal |
| 10 | Small cross-attention model | Test fine-grained interactions only after simpler baselines |
| 11 | Bayesian uncertainty layer or teacher-student deployment models | Improve calibration or missing-modality deployment |
| 12 | New masked/contrastive pretraining | Attempt only with sufficient unlabeled data and a clear ablation |

## Claude Code repository audit request

Please inspect the repository and return the following before making changes:

### A. Current implementation map

- Identify files responsible for cohort construction, episode matching, feature extraction, splitting, training, fusion, evaluation, and configuration.
- Summarize the actual end-to-end data flow.
- State which recommendations in this document are already implemented.

### B. Cohort and leakage audit

Report, if derivable without exposing patient-level data:

- total episodes;
- unique patients;
- unique echo measurement events;
- median and range of episodes per patient;
- root and ascending label availability;
- positive counts/prevalence at each threshold;
- counts for each modality combination;
- number of reused ECGs/CXRs across episodes;
- train/validation/test overlap checks by patient and source study ID.

Verify whether splitting occurs before or after episode generation and whether any preprocessing, normalization, feature selection, imputation, or dimensionality reduction is fit outside the training fold.

### C. Model audit

- Identify exact CXR, ECG, and EHR encoders and versions.
- State which layers are frozen or trainable.
- Report embedding dimensions and any normalization/dimensionality reduction.
- Describe the current early/late-fusion implementation precisely.
- Determine whether fusion inputs are truly OOF.
- Check whether each modality has its own auxiliary prediction loss.
- Determine how multiple studies and missing modalities are handled.
- Check whether the model has a straightforward path to ignore weak modalities.

### D. Loss and evaluation audit

- Identify regression/classification losses, class weighting, sampling, and calibration.
- Check whether root and ascending targets share information through multitask learning.
- Verify threshold consistency.
- Verify that model selection and hyperparameter tuning are nested within training data.
- Check whether metrics and confidence intervals account for patient clustering.
- Confirm that every modality comparison uses the same held-out patient population.

### E. Final report format

Return:

1. A concise description of the current implementation.
2. A table of deviations from this proposal, labeled `critical`, `high-value`, `optional`, or `not applicable`.
3. Any confirmed or suspected leakage risks, with file and line references.
4. Evidence for whether ECG and EHR contain conditional signal beyond CXR.
5. The five highest-value improvements ranked by expected benefit, implementation effort, and risk.
6. A minimal patch plan listing the exact files/functions that would change.
7. Any parts of this proposed design that should **not** be adopted, based on the actual cohort or code.

Do not assume a more complex model is preferable. Recommend cross-attention, cooperative learning, Bayesian components, or additional self-supervision only when the current data scale and diagnostics justify them.

## Relevant primary references

- Hayat et al., *MedFuse: Multi-modal fusion with clinical time-series data and chest X-ray images*: https://proceedings.mlr.press/v182/hayat22a.html
- Ding et al., *Cooperative Learning for Multi-view Analysis*: https://arxiv.org/abs/2112.12337
- Wu et al., *Characterizing and Overcoming the Greedy Nature of Learning in Multi-modal Deep Neural Networks*: https://proceedings.mlr.press/v162/wu22d.html
- Peng et al., *Balanced Multimodal Learning via On-the-fly Gradient Modulation*: https://openaccess.thecvf.com/content/CVPR2022/html/Peng_Balanced_Multimodal_Learning_via_On-the-Fly_Gradient_Modulation_CVPR_2022_paper.html
- Hayat et al., *DrFuse: Learning Disentangled Representation for Clinical Multi-Modal Fusion*: https://ojs.aaai.org/index.php/AAAI/article/view/29578
- Perez-Garcia et al., *RAD-DINO*: https://arxiv.org/abs/2401.10815
- Diamant et al., *Patient Contrastive Learning (PCLR)*: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009862
- Demarais et al., *Estimating Ascending Aortic Diameter From the Electrocardiogram*: https://pubmed.ncbi.nlm.nih.gov/41369619/
- Lee et al., *Deep learning-based detection of ascending aortic dilatation on chest radiographs*: https://doi.org/10.1016/j.ejrad.2025.112380

