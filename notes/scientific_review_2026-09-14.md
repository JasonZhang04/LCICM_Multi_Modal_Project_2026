# Scientific and implementation review — 14 September 2026

Discussion draft. No training, cohort, or evaluation production code was changed for this review. Findings below distinguish verified defects, limitations of the evidence, and hypotheses. The accompanying [improvement roadmap](model_improvement_roadmap_2026-09-14.md) is a proposal, not a record of completed experiments.

## Executive assessment

The project has made real progress: a substantially larger, correctly study-linked TTE cohort; patient-grouped evaluation; a raw-waveform ECG branch; multi-image CXR aggregation; and partial fine-tuning of RAD-DINO. The strongest recent result is the fine-tuned CXR model, particularly for ascending-aortic diameter. The latest saved multimodal model reports root/ascending R² of **0.331/0.334** and AUROC for diameter ≥4 cm of **0.838/0.830**.

However, the current evidence is **not ready to support the report's claims of leakage-free multimodal superiority or untouched temporal validation**. I verified an indirect target-leakage path through globally precomputed out-of-fold features, incompatible ECG embedding coordinate systems across folds, use of the purported holdout in development, an incorrect temporal interpretation of MIMIC anchor groups, and inclusion of some CXRs acquired after the recorded echo time. These are reasons to repair the evaluation before interpreting small performance differences—not evidence that the underlying task is impossible or that all unimodal results are worthless.

There is also a strategically important result missing from the headline: on exactly matched episodes, fine-tuned CXR alone slightly exceeds the full model's ≥4 cm AUROC at both sites. The full model adds a small root R² increment, but not a clear ascending R² increment. A publishable multimodal contribution must therefore be established against the strongest current CXR baseline, not just EHR or the older frozen-CXR stack.

Recommended discussion order: **A1–A6 validation and timing → A7 strongest comparator → A8–A10 design/provenance → revised experiments and framing**.

## 1. What the project actually is now

### Scope and chronology

I reconciled the current code and saved results with `PROGRESS_REPORT.md`, the August 4/5/30 updates, relevant design notes, git history through `f7e528b`, and the accessible Claude Code project session and memory. The main relevant session is `deb51f59-d3dd-43ee-8178-2871631e702d.jsonl` under the user's Claude project history. I used that history as provenance, not as an authority over executable code or your current request.

The history resolves an important scope ambiguity: on September 1 you explicitly clarified that the targets remain **aortic root and ascending aorta**, not the arch and descending aorta. The later-looking four-segment script/output is an unwanted side branch, not the new project objective. Your two intended contributions were (1) a rigorous modality-value study and (2) a clinically evaluated predictive model; label reliability was supporting work.

| Period | What changed / how to interpret it |
|---|---|
| Earlier prototype | Small, echo-DICOM-gated cohort and problematic historical label anchoring. Its numbers should not be compared directly with the current cohort. |
| July–August 5 | Study-specific TTE labels, episode cohort, frozen RAD-DINO/segmentation geometry, multi-instance CXR, EHR baseline, ECG machine-measurement analyses. |
| August 5–30 | Raw ECG waveform model, fusion comparisons, anatomical/lag analyses, repeat-echo reliability analysis, nominal temporal split. Several interpretations in the August 30 report are too strong; see below. |
| August 30–31 | CXR PCA sweep, last-two-block RAD-DINO fine-tuning, revised final stack, separate holdout blend, broader ECG training experiment. |
| September 1 | Root/ascending clinical metrics extended to ≥4.5 and ≥5 cm. Latest requested improvement brainstorming was not followed by a new completed improvement program in the inspected history. |

The README and generic transformer/echo-image training infrastructure do not describe the active final model. Current execution is primarily the episode-specific scripts.

### Cohort and prediction target

The current cohort contains **20,682 episodes from 16,464 patients**. Labels are TTE `sinus_diam` (root) and `ascending_diam`, medians within the same subject/study/timestamp, with a 1.5–7 cm plausibility filter. There are 20,429 root-labeled and 18,510 ascending-labeled episodes; 18,257 have both. Retained echo episodes are greedily separated by at least 180 days before checking CXR eligibility.

CXR eligibility is frontal AP/PA within 365 calendar days before/on the echo date. The instance table has 64,516 image–episode rows and 60,092 distinct images, of which 39,413 are AP and 20,679 PA; 60,090 have the frozen embedding artifact. A single episode can contain up to 90 images. ECG is optional, nominally within 180 days; the waveform cohort has 18,221 episodes/14,831 patients. EHR consists of age, sex, height, weight, BMI, BSA, systolic and diastolic BP, plus four missingness indicators.

The outcome is a **recorded TTE measurement**, not CT/MRI ground truth, a future dissection event, or a surgical indication. MIMIC-IV-Echo describes these structured measurements as originating from echo systems, with timestamps referring to the echo examination; they are not LLM-extracted diagnoses.[^1]

### Actual final model

```text
Each eligible frontal CXR
  ├─ frozen RAD-DINO CLS + aorta/mediastinum pool + heart pool
  │    → training-fold PCA-128 per block + 17 geometry features
  │    → image HGB regression → episode mean diameter
  └─ RAD-DINO, last 2 blocks fine-tuned + linear root/ascending head
       → image predictions → episode mean diameter

ECG waveform → supervised 1D ResNet → 256-dimensional hidden embedding → PCA-32
EHR → 12 raw/missingness features

Two CXR diameter scalars + EHR + ECG embedding
  → scaled, regularized ridge, separate model per aortic site
```

This is a **hybrid stacked/additive model**, not end-to-end cross-modal fusion. Ridge does not introduce explicit cross-modal interactions. Both CXR branches already aggregate multiple images; “try averaging multiple CXRs” is not a new proposal. Missing ECG becomes a zero standardized embedding block without an explicit ECG-availability indicator.

## 2. Latest saved results, not the older headline

Sources: [`final_model_episode/results.json`](../outputs/final_model_episode/results.json), [`clinical_metrics.json`](../outputs/final_model_episode/clinical_metrics.json), and their OOF predictions. The final saved run uses the **single immutable five-fold seed-42 partition**. The JSON still lists seeds 1/2/3, but that list is not the number of evaluated partitions in immutable mode.

### Main endpoint: ≥4 cm

| Metric | Root | Ascending |
|---|---:|---:|
| Labeled episodes | 20,429 | 18,510 |
| Positive episodes / positive patients | 927 / 823 | 827 / 721 |
| Diameter R², 95% patient-cluster CI | 0.331 [0.320, 0.342] | 0.334 [0.322, 0.347] |
| AUROC, 95% CI | 0.838 [0.826, 0.851] | 0.830 [0.815, 0.845] |
| AUPRC, implemented as average precision | 0.202 [0.180, 0.229] | 0.238 [0.209, 0.273] |
| MAE | 0.288 cm | 0.285 cm |
| Sensitivity if the highest-scoring 5% receive echo | 28.6% | 31.8% |
| PPV in that highest-scoring 5% | 26.0% | 28.4% |
| Number receiving echo per positive in that simulation | 3.85 | 3.52 |

These are exploratory retrospective numbers under the limitations below. In particular, ranking the top 5% concentrates positives but still misses about 68–71% of ≥4 cm episodes in this cohort. A reassuring overall MAE does not mean accurate measurement in the dilated tail: mean prediction error among observed ≥4 cm cases is −0.725/−0.752 cm. This conditional error also reflects regression toward the mean with noisy extreme labels; it is not by itself proof of a particular biological failure.

### Higher thresholds

| Site / endpoint | Positive episodes | AUROC | AUPRC | Sensitivity / PPV, top 5% |
|---|---:|---:|---:|---:|
| Root ≥4.5 cm | 90 | 0.882 | 0.046 | 45.6% / 4.02% |
| Ascending ≥4.5 cm | 124 | 0.885 | 0.097 | 51.6% / 6.91% |
| Root ≥5 cm | 9 | 0.796 | 0.002 | 11.1% / 0.098% |
| Ascending ≥5 cm | 22 | 0.890 | 0.048 | 54.5% / 1.30% |

For root ≥5 cm, the top-5% strategy captures **one of nine** positive patients. These severe-threshold results should be descriptive, with counts and uncertainty, not headline evidence for a safety-critical detector. The ≥4/4.5/5 cm endpoints should not be described as universal treatment thresholds; aortic segment, body size, etiology, and other clinical factors matter.[^2]

### Strongest current unimodal comparison — newly recomputed

I merged the full-model and fine-tuned CXR predictions by episode, patient and site, verified identical labels, and computed 2,000 paired patient-cluster bootstrap replicates using the project's bootstrap implementation.

| Metric | Fine-tuned CXR only | Full model | Full minus CXR, 95% CI |
|---|---:|---:|---:|
| Root R² | 0.31896 | 0.33090 | +0.01195 [+0.00782, +0.01614] |
| Ascending R² | 0.33281 | 0.33415 | +0.00134 [−0.00276, +0.00549] |
| Root AUROC ≥4 cm | 0.83961 | 0.83824 | −0.00137 [−0.00602, +0.00318] |
| Ascending AUROC ≥4 cm | 0.83284 | 0.83029 | −0.00254 [−0.00669, +0.00175] |

These intervals describe the saved scores, conditional on the existing training and selection process. They neither repair leakage nor include the uncertainty from retraining/model selection. The conclusion is **no demonstrated ≥4 cm AUROC advantage from the full model over fine-tuned CXR**, not proof of exact equivalence or proof that ECG/EHR can never help.

The separate [`holdout_final/results.json`](../outputs/holdout_final/results.json) reports root/ascending R² 0.330/0.321, AUROC 0.838/0.838 and AUPRC 0.174/0.192. Do not present these as external validation or as a clean temporal confirmation of the final ridge: both the split and the evaluated model differ from that description.

## 3. Issues requiring revision

### A1 — Critical, confirmed: the full stack is not nested despite its name

Evidence: [`train_final_model_episode.py`](../scripts/train_final_model_episode.py), especially `linear_early_oof` and the call to `gs.cxr_base_oof`; [`train_geometry_stack_episode.py`](../scripts/train_geometry_stack_episode.py); precomputed fine-tuned CXR and supervised ECG artifacts.

Every patient's base prediction/embedding is generated without that patient, which is good. But one global OOF feature matrix is then reused to cross-validate the second-stage model. That does **not** exclude the second-stage test labels from construction of its training features:

```text
Evaluating final model on fold 0:
  A fusion-training row in fold 1 gets a base feature from a model excluding fold 1.
  That base model was trained using fold 0 labels.
  Therefore: fold-0 labels → base model → fusion-training features → fitted fusion model.
```

Matching immutable fold assignments does not break this path. It is especially relevant when judging small increments over simpler baselines. The amount and direction of bias cannot be inferred from the existence of the path alone. Older scalar stacks and other fusion comparisons using the same construction also require re-evaluation; zero patient overlap is necessary but insufficient. The older geometry stack additionally tunes ridge using episode-level `KFold`, not patient-grouped inner folds.

**Revision:** For every outer test fold, isolate all its patients before any supervised feature creation. Generate inner-OOF base predictions using only outer-training patients; train the fusion on those predictions; refit the base learners on outer-training data; predict the outer test set and apply the learned fusion. Tune transformations and model choices within the appropriate training/validation boundaries. Cross-validated ensemble construction is established methodology, not a novelty claim.[^3]

**Acceptance test:** Log and assert the training-patient ancestry of every fitted component. Perturb outer-test labels and verify that training features, selected hyperparameters, fitted models and predictions remain unchanged. Score computation is the only stage allowed to use those labels.

### A2 — Critical, confirmed design problem: ECG embeddings do not share a stable basis

Evidence: [`train_ecg_waveform_episode.py`](../scripts/train_ecg_waveform_episode.py) writes the penultimate embeddings from five independently trained CNNs into one `ecg_embeddings.npy`; the final model fits one PCA/ridge across them.

Unlike a scalar diameter in cm, hidden coordinate 17 from one independently trained neural network has no guaranteed correspondence to coordinate 17 from another. Fusion training pools embeddings from four networks; testing uses the fifth. PCA does not align these learned coordinate systems. This is a separate issue from A1.

A diagnostic supports a severe fold signature: a nearest-centroid classifier on standardized 256-D embeddings predicts their source fold with **99.9817% accuracy**, versus 20% chance, on 5,459 patient-disjoint held-out embedding rows. This is evidence of representation/batch separation, not evidence that fold ID is clinically meaningful. It does not isolate how much comes from arbitrary coordinate orientation versus other between-training-run differences.

**Revision:** Start with properly nested scalar ECG stacking, where units align. For feature fusion, use a common externally pretrained frozen encoder, or an outer-fold-specific encoder applied consistently to both its fusion-training and test patients under a valid training scheme. Joint training inside each outer fold or additional sample splitting can avoid constructing one matrix from incompatible independently supervised bases. Merely regenerating global OOF embeddings with another seed is not a repair.

**Acceptance test:** Document which exact encoder produces each feature matrix, verify common feature semantics within each fusion fit, and compare the repaired embedding approach against nested scalar fusion. Re-run the modality-value claims.

### A3 — Critical, confirmed: the nominal holdout was not quarantined

Evidence: [`build_temporal_holdout.py`](../scripts/build_temporal_holdout.py) writes assignments, but normal training scripts load all episodes without excluding them. All **6,220 labeled root and 5,632 labeled ascending holdout episodes** appear in the development final-model OOF file. The development/ablation program therefore used this patient group; many development fits also trained on its labels.

The separate holdout run can still fit base models without those patients. That does not undo their prior use in development/model selection. “Touch once,” “which no model has seen,” and “set aside before the ablation program” are not supported by the actual workflow.

**Revision:** Relabel the existing experiment as an exploratory era-associated patient split. Do not retroactively call these patients untouched. Freeze a corrected development protocol and seek a genuinely unused evaluation cohort, ideally another institution. A new split of already extensively analyzed data can improve internal evaluation but cannot erase previous exposure. Report that history transparently, in line with prediction-model reporting/risk-of-bias guidance.[^4][^5]

### A4 — High, confirmed: `anchor_year_group` is not the episode's calendar era

Evidence: [`build_temporal_holdout.py`](../scripts/build_temporal_holdout.py) holds out the 2014–2016 *anchor* group. MIMIC maps `anchor_year` to an approximate real-year group; another shifted year for that patient must be interpreted relative to that anchor.[^6]

In 12,863 of 20,682 episodes, the echo year differs from the patient's anchor year. Applying the corresponding year offset to the anchor interval shows that **5,196 training episodes have an implied earliest year of 2014 or later**, while **20 nominal holdout episodes have an implied latest year before 2014**. These are interval-based year inferences, not recovered exact calendar dates. The split also leaves six patients' episodes from later anchor groups in training because only equality to the chosen group defines holdout.

**Revision:** Use interval-aware event-year logic with an explicit temporal guard band and patient-disjoint assignment, or describe the split without a chronological-generalization claim. If a patient straddles the boundary, predefine exclusion or another conservative patient-level rule. Do not sort patients by their fictitious absolute shifted dates.

**Acceptance test:** Under the intended temporal claim, the latest plausible training event precedes the earliest plausible evaluation event, with no patient crossing. If this costs too much sample size, use an honestly labeled grouped internal evaluation and prioritize external testing instead.

### A5 — High, confirmed: some “pre-index” CXRs are after the recorded echo time

Evidence: [`build_episode_cohort.py`](../scripts/build_episode_cohort.py) loads `StudyDate` without `StudyTime` and truncates both modalities and echo anchors to whole days. Reconstructing full CXR timestamps identifies **1,931 image–episode rows across 1,572 episodes** whose CXR follows the recorded echo timestamp. All examined CXR timestamps parsed, and none lacked `StudyTime`.

This violates the stated pre-echo ordering; its measured performance effect is unknown. More fundamentally, “everything before echo” is not necessarily “everything available when the CXR-based screening decision would be made.” EHR or ECG collected after an old CXR but before the later echo cannot be supplied to a model claiming to operate at that earlier CXR acquisition.

**Revision:** First specify the intended prediction time, then enforce full-timestamp availability of all inputs. For CXR-acquisition triage, freeze the inputs at that CXR time and use an appropriately near subsequent TTE as the outcome. For pre-TTE estimation at a later encounter, state that different use case explicitly. Check cross-dataset clock alignment and timestamp meaning. Date-only OMR entries on the same day need a conservative rule or sensitivity analysis. Harmonize ECG eligibility with the actual waveform timestamp rule; do not treat the day-level eligibility count as the number of usable waveforms.

**Acceptance test:** No input timestamp exceeds the defined decision time; report attrition and rerun matched sensitivity analyses. Do not silently drop affected cases and claim unchanged performance.

### A6 — High, confirmed: “final-model validation” uses a different fusion model

Evidence: [`train_holdout_final.py`](../scripts/train_holdout_final.py) takes an equal, NaN-aware mean of four scalar diameter predictions: frozen CXR, fine-tuned CXR, EHR HGB and ECG CNN. It does **not** use the final ridge, raw EHR covariates or ECG embedding. A comment still describes standardized blending, but the executable blend now averages raw cm. The history records correction of a prior scale error after inspecting its failed output.

Consequently, stable holdout numbers do not validate the actual selected final model. Nor is equal averaging mathematically equivalent to fitted ridge simply because predictors correlate. Missing ECG also changes the number and relative weight of components being averaged.

**Revision:** Evaluate the exact saved pipeline selected on development data. Treat the fixed-average model as a separate baseline. Save its per-patient predictions as well as metrics; the current holdout ensemble retains aggregate results only.

### A7 — High, confirmed: the strongest baseline changes the scientific conclusion

The matched comparison in Section 2 must be central. The paper cannot infer multimodal superiority from improvement over EHR alone or an older CXR representation when fine-tuned CXR alone matches/exceeds its discrimination. The total full-versus-CXR increment also does not isolate the contribution of ECG: it adds EHR and an extra CXR branch as well.

**Revision:** After A1/A2, run a same-cohort, same-split factorial comparison including fine-tuned CXR; CXR+EHR; CXR+ECG; and CXR+EHR+ECG, with the same CXR backbone and comparable tuning budgets. Include the frozen-CXR branch as a separately tested addition. Evaluate both the all-CXR cohort with optional ECG and an ECG-present matched subset. Report paired AP, AUROC, R² and practical operating-point differences. A negative incremental-value result is acceptable; an overstated positive claim is not.

### A8 — High, confirmed confounding: the broader-ECG experiment is not a clean A/B

Evidence: [`final_model_episode_broadecg`](../outputs/final_model_episode_broadecg/results.json), paired prediction files and Claude launch history. The reference launch explicitly set `K_PCA=128,K_ECG=32`; the broader-ECG launch omitted those settings, while the script defaults CXR `K_PCA` to 32. Independently of environment inheritance, the saved non-ECG `pred_geom_stack` changes for **every** matched root and ascending episode, with maximum differences 0.171/0.269 cm. Thus more than ECG changed.

The broader waveform cohort contains 62,559 episodes/46,230 patients. Its 37,050 always-training episodes come from 31,029 patients absent from the CXR cohort; the audit found no patient overlap there. On the exact small-cohort evaluation cases, broader training improves standalone ECG R² from **0.1414 to 0.1729** for root and **0.1499 to 0.1805** for ascending. These are recomputed point estimates, not a new causal proof of downstream benefit.

**Revision:** Reopen the broader-ECG question. Fix all non-ECG artifacts/configuration, repair A1/A2, and compare on identical evaluation patients. The prior “broad ECG hurts, close this avenue” conclusion is unsupported.

### A9 — High, confirmed mismatch; benefit of changing it remains a hypothesis: RAD-DINO inputs

Evidence: [`default_config.py`](../src/multimodal_aorta/configs/default_config.py), [`preprocessing.py`](../src/multimodal_aorta/data/preprocessing.py), [`extract_cxr_image_cache.py`](../scripts/extract_cxr_image_cache.py), and the local RAD-DINO processor configuration. Current inputs are square-resized 224×224 with ImageNet channel-specific normalization. The released checkpoint specifies 518-pixel resize/center crop, bicubic interpolation and equal grayscale-channel mean/std of approximately 0.5307/0.2583.[^7]

The encoder can technically accept 224 pixels through positional interpolation; this is not a tensor-shape bug. But the normalization, resolution and geometry differ from its pretrained input distribution. In particular, identical grayscale channels become differently normalized RGB channels. This affects frozen features and fine-tuning. Existing evidence does not tell us how much changing it will help.

**Revision:** Separate normalization, resolution and aspect-ratio interventions in controlled development experiments; compare the checkpoint processor with a full-field, geometry-preserving alternative. Version/rebuild caches accordingly. Do not silently introduce a center crop that clips the anatomy you intend to measure.

An important non-bug: this checkpoint uses patch size **14**, so 224 gives a **16×16** patch grid. Current dynamic grid handling is not broken merely because comments call it ViT-B/16. However, the hand-coded mask pooling should be replaced or checked before switching to a 37×37 grid at 518: floor-division pooling from a 512 mask can discard border pixels. All spatial masks must undergo the exact same resize/crop transforms as images.

### A10 — High transparency/generalization limitation: pretraining saw most study images

The released RAD-DINO includes MIMIC-CXR in its pretraining provenance.[^7][^8] Exact-ID comparison against the local `training_images.csv` finds **58,272 of the 60,092 referenced unique CXRs (97.0%)** in that pretraining manifest.

This is not evidence of aortic-label leakage: the pretraining is self-supervised and distinct from the target task. It is evidence that most evaluation images are not genuinely image-unseen to the encoder. A downstream patient split does not erase that exposure.

**Revision:** Disclose image/patient pretraining overlap and distinguish “unseen target labels” from “unseen inputs.” For stronger transportability evidence, evaluate on another institution or data demonstrably absent from pretraining; alternatively include a backbone with appropriately disjoint provenance. Do not exclude 97% of the current cohort and call the tiny remainder an adequate external validation without checking sample/event support.

### A11 — Moderate/high: sample weighting and missingness need a defined estimand

The frozen multi-instance HGB uses inverse image-count weights, giving each episode comparable training influence. The fine-tuning loader instead samples image rows uniformly, and its loss, early-stopping score and target standardization are image-weighted. With 1–90 images per episode, these are different objectives. This is not automatically mathematically invalid, but it may overemphasize high-utilization patients and makes comparisons less controlled. Repeated episodes also give some patients up to seven times the episode-level influence.

EHR values are pre-echo in recorded day offsets, but height/weight/BSA are missing in 28.4%/23.7%/28.5% and SBP in 33.2%. There is no maximum age for the last observation: 2,149 height values and 803 BP values are over a year old, with maxima above eight years. Missing ECG is encoded as the mean standardized embedding, conflating missingness and an average representation.

**Revision:** Choose an episode- or patient-level estimand, then align sampling, losses, validation and uncertainty with it. Test episode-balanced CXR bags; use modality-availability masks; consider observation-age covariates and predeclared staleness limits. Do not replace all missing data with complete-case restriction: that changes the population. Separate performance when ECG is genuinely absent from artificial modality dropout experiments. Missing-modality handling has precedent in MedFuse; it is a design requirement, not sufficient novelty by itself.[^9]

### A12 — Moderate/high: clinical evaluation is not yet deployable threshold validation

[`eval_episode.py`](../scripts/eval_episode.py) and [`clinical_metrics.py`](../src/multimodal_aorta/training/clinical_metrics.py) report useful quantities. However, choosing a Youden or 90%-sensitivity threshold using the same evaluation labels reports an optimistically selected operating point. The “top 5%” policy, if its capacity is fixed in advance, does not use labels to select its cutoff and is a legitimate retrospective ranking simulation, but its PPV/number-needed-to-echo apply to this echo-selected cohort—not automatically to all patients receiving CXR.

Predicted diameter is not a calibrated probability of dilation. No completed probability-calibration evaluation of the final model was found. Repeated-episode bootstrap is appropriately patient-clustered, but uncertainty conditional on already selected predictions does not cover the full search over models.

**Revision:** Select operating thresholds and any diameter-to-risk calibrator entirely on development data; freeze them for evaluation. Report calibration, average precision, sensitivity/specificity/PPV with counts and CIs, and decision-curve net benefit over clinically agreed probability thresholds.[^10] Sensitivity and recall are the same quantity. For ≥5 cm, retain descriptive counts and explicitly acknowledge the nine/22 positive-patient limitation.

### A13 — Important interpretive revisions to the progress report

These statements should be softened now, even before rerunning experiments:

| Existing interpretation | Evidence-supported replacement |
|---|---|
| “ρ=0 proves the modalities are redundant / have no complementary signal.” | The tested agreement penalty was not selected. Cooperative-learning theory does not make this a redundancy test; agreement can help views sharing latent signal. The two-view early/late relationships hold under the specified formulation, not as a general rule for every fusion model.[^11] |
| “Linear ridge lets modalities interact.” | Ridge combines inputs additively unless interaction features are explicitly supplied. Feature concatenation alone is not an interaction model. |
| “Cross-attention does not help.” | The particular small three-token/PCA attention configuration did not show a benefit under the existing evaluation. This does not settle spatial, EHR-conditioned, end-to-end fusion, and the experiment inherits representation/validation concerns. |
| “The CXR reads the aorta, not body size.” | Restricted features retain signal beyond recorded EHR. The aorta pool includes mediastinum; final ViT tokens are globally contextual; geometry uses other anatomical structures. Output-token masking does not prove the input information originated only inside the aorta. Complete-EHR analyses do not eliminate all habitus confounding. |
| “Flat lag performance proves stable anatomy.” | Different lag groups had similar point estimates. They differ in selection/case mix, and mean bag lag obscures individual images. The implemented integer lag bins also omit 120 episodes with fractional mean lags between boundaries. |
| “The ECG-measurement null is airtight; ±0.01 was prespecified.” | Effects were small for the tested measurements/models. I found no timestamped prespecification preceding the observed effects; the margin appears in later supporting analysis. Call it a post-hoc equivalence sensitivity analysis unless earlier provenance exists. Use a consistent practical-benefit criterion for waveform gains of similar magnitude. |
| “PCA saturation proves fine-tuning is the only remaining lever.” | Increasing PCA dimensions did not improve that particular downstream learner. Pooling, normalization, spatial information, loss design and cohort quality remain possible levers. |
| “R² is at a measured hard echo-noise ceiling / model beats echo repeatability.” | The ceiling estimate depends on assumptions about stable latent anatomy and independent, representative measurement error. Model-versus-one-echo and echo-versus-echo errors contain different noise terms; their raw magnitudes do not establish superiority to echocardiography. |

The repeat-echo analysis may remain useful supporting work. But short-interval repeats are selected, reader/protocol errors may correlate, and true change/intervention is not ruled out. Do not average later echoes into an index outcome without explicitly changing the target and defining the allowable window.

### A14 — Publication-critical limitations and reproducibility work

**Selection and endpoint definition.** This is an echo-labeled, hospital-linked CXR cohort, not an unselected population sample. Patients without TTE have unobserved outcomes, not negative labels. Known dilation, prior repair, valve disease, devices and referral context could make the task partly detection of established disease. Define whether the aim is occult-disease triage or estimation in an already investigated population. Report a complete inclusion/exclusion flow and label availability by subgroup; PROBAST+AI is useful for structuring these concerns.[^5]

**Label QC.** I streamed the original 2.6 GB structured-measurement table: the two TTE target fields are in cm, and the study/timestamp-specific filtered medians reproduce the current root and ascending labels exactly. That is reassuring. It does not prove every source measurement is anatomically accurate. Review extreme values and study conventions with a clinician; do not delete credible large aortas solely because they hurt loss. MIMIC-IV-Echo 1.0.1 documents DICOM identifier corrections, not a blanket change to this structured-label schema; do not assume a dataset update fixes these labels.[^1]

**No complete saved inference model.** The inspected final CXR/ECG trainers keep best weights in memory but save predictions/embeddings, not the selected network states. The staged PCA/HGB/ridge fits are likewise not packaged into an inference bundle. There is therefore no reproducible deployable “final model” artifact established by these scripts, despite the report's wording. The interpretability note correctly anticipates this problem; it is still a plan.

**Revision:** Save checkpoint states, feature order, preprocessing and target scalers, PCA/HGB/ridge objects, split manifests, data/model versions, commit and dirty diff, every relevant environment variable, training seed, and patient-level predictions. Use immutable run directories. Assert matching configuration for paired ablations and add leakage/time-order tests. Do not call the immutable five-fold run “three seeds.” Existing tests are not a substitute for end-to-end temporal and feature-ancestry assertions.

## 4. Literature and novelty assessment, current to 14 September 2026

This is a focused primary-source review, not a preregistered systematic review. Published papers, dataset/model documentation and clearly identified preprints are separated. I did not infer that inaccessible final articles contain every numerical detail of their preprints.

| Closest work | What is established | Consequence for this project |
|---|---|---|
| Pirruccello et al., *JAMA*, 2022: clinical AORTA score[^18] | Clinical-variable prediction of ascending diameter, with external CT-based Framingham and TTE-based Mass General Brigham validation. | Include an established clinical-score comparator where inputs can be faithfully reconstructed; beating a locally chosen EHR baseline alone is less convincing. TTE-based external evaluation is not itself new. |
| Lee et al., *European Journal of Radiology*, 2025: DeepCatch X Aorta[^12] | CXR-only ascending-dilation detection; two **external** PA-CXR/CT test cohorts, n=336 and 190, AUROC 0.89/0.88 at ≥4 cm. | Not merely a small internal study. We cannot claim first CXR aortic screening. AP inclusion, TTE reference and lower observed prevalence distinguish the setting, not automatically the scientific contribution. |
| Demarais, Olgin and Pirruccello, *JACC*, January 2026; accessible 2025 preprint[^13] | ECG-based ascending diameter estimation already exists. The preprint reports 69,173 ECG/MRI participants and held-out R² about 0.31. | Update the report's “2025 preprint” description. The task is feasible from ECG elsewhere; our small conditional increment is not evidence of no ECG anatomical signal. |
| ECG+CXR cooperative learning for aortic stenosis, *Frontiers in Radiology*, 2025[^14] | Multimodal ECG/CXR modeling for a related but different valve-disease endpoint; reported combined AUROC 0.812 versus ECG 0.775/CXR 0.755. | Neither multimodal cardiology nor agreement-based fusion is new. Aortic stenosis is not aortic dilation; transfer the experimental idea, not its endpoint or expected gain. |
| MedFuse, MLHC 2022; CareBench, February 2026 preprint[^9][^15] | EHR/CXR fusion, incomplete modalities, and task-dependent robustness/benefit already have substantial methodological precedent. | A generic “multimodal helps/does not help” story is insufficiently distinct. Need task-specific, well-controlled incremental utility and transportability. |
| EchoNext, *Nature*, 2025, with public data/weights[^16] | Large-scale ECG structural-heart-disease modeling, external cohorts and prospective silent evaluation. Its clinical trial recruited using related ValveNet; EchoNext analysis of that trial was post hoc. | Useful inspiration for ECG transfer and validation design. Its composite endpoint is not aortic diameter; do not treat the public resource as a ready-made diameter-validation dataset. |
| AnyECG-Echo, June 2026 preprint[^17] | Echo-report-supervised single-lead ECG includes ascending-aortic dilation among 13 structural subtypes. Reported aortic-dilation AUROC drops from 0.685 internal to 0.635 external. | Another direct endpoint neighbor and a useful training-supervision idea. It is a preprint, a different task definition, and not evidence that this method will outperform our model. |

Do not explain another study's higher AUROC solely by prevalence. With fixed class-conditional score distributions AUROC is prevalence-invariant; reference standard, disease severity/spectrum, selection and site shift can change it. AUPRC/PPV are directly prevalence-sensitive. Our and their published numbers are not a valid head-to-head experiment.

**Most defensible prospective framing:** a reproducible evaluation of the incremental value and clinical trade-offs of routinely available CXR, ECG waveform and EHR for root and ascending-aortic measurement/triage in a clearly defined hospital population, with strong unimodal comparators, missing-modality handling, strict information timing, and independent validation. That combination is a candidate contribution, not a verified “first.” A new architecture is optional; a trustworthy answer about when extra modalities actually help may be more valuable.

**Claims not currently supported:** first AI aortic-dilation detector; first ECG diameter estimator; first ECG+CXR cardiology model; proven biological redundancy; direct CT/MRI measurement accuracy; general-population screening effectiveness; external/untouched temporal validation; deployable saved model; clinical safety at ≥5 cm.

## 5. Proposed next decision

I recommend **repairing the evaluation before committing to an architecture or another large sweep**. First agree on the decision time and intended clinical population. Next fix the nested training and embedding semantics, reproduce the strongest baselines, and establish a realistic independent-validation plan. Only then decide whether the best paper is a positive multimodal model, a selective-use model, or a rigorous finding of little incremental value beyond CXR.

The [roadmap](model_improvement_roadmap_2026-09-14.md) specifies staged, literature-grounded improvements, including preprocessing, cohort construction, balanced multi-image training, broader/pretrained ECG, conditional fusion and clinically aligned losses. These are hypotheses with explicit comparators—not promises of improvement.

## 6. Audit trail and limits of this review

The read-only [audit script](../analysis/audit_project_2026_09_14.py) checks current cohort counts, patient/image grouping, recorded timestamp ordering, anchor-year intervals, holdout presence in development, raw label reproduction/units, EHR staleness, pretraining overlap, broader-ECG comparisons, embedding fold signatures, matched final-versus-CXR scores and optional paired bootstrap intervals.

Example commands from the project root:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 /scratch4/rsteven1/your_env_name/bin/python3.10 analysis/audit_project_2026_09_14.py --raw-labels
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 /scratch4/rsteven1/your_env_name/bin/python3.10 analysis/audit_project_2026_09_14.py --bootstrap
```

Both audit modes were run successfully during this review. Outputs are aggregate only; the script does not train or alter source data. Patient-fold overlap, image-fold overlap and patient overlap across the assigned holdout were all zero. The broader ECG always-training population also had zero patient overlap with the CXR cohort. These checks are positives, but do not negate the stage-wise leakage in A1.

I inspected the active cohort, feature extraction, training, fusion, evaluation and relevant supporting-analysis paths, and reconciled saved artifacts with reports/history. I did **not** retrain the GPU models, review every source image/echo manually, prove external generalizability, or exhaustively execute every legacy/experimental script. Conclusions about numerical improvements after repairs remain unknown. Existing unrelated worktree changes were preserved.

## Sources

[^1]: MIMIC-IV-Echo official documentation, [version 1.0](https://physionet.org/content/mimic-iv-echo/1.0/) and [version 1.0.1 release notes](https://physionet.org/content/mimic-iv-echo/1.0.1/). Structured measurements, timestamps, dataset scope and DICOM corrections.
[^2]: Isselbacher et al. [2022 ACC/AHA Guideline for the Diagnosis and Management of Aortic Disease](https://www.jacc.org/doi/10.1016/j.jacc.2022.08.004). Clinical measurement conventions and size-/context-dependent interpretation; not a claim that a fixed research endpoint is intrinsically invalid.
[^3]: Polley and van der Laan. [Super Learner in Prediction](https://biostats.bepress.com/ucbbiostat/paper266/), 2010. Cross-validated ensemble methodology; the specific leakage finding here comes from tracing the local implementation.
[^4]: Collins et al. [TRIPOD+AI statement](https://www.bmj.com/content/385/bmj-2023-078378), *BMJ*, 2024. Reporting guidance for clinical prediction models using regression or machine learning.
[^5]: Moons et al. [PROBAST+AI](https://www.bmj.com/content/388/bmj-2024-082505), *BMJ*, 2025. Quality, risk-of-bias and applicability assessment.
[^6]: Johnson et al. [MIMIC-IV, a freely accessible electronic health record dataset](https://pmc.ncbi.nlm.nih.gov/articles/PMC9810617/), *Scientific Data*, 2023; [official MIMIC-IV 3.1 documentation](https://physionet.org/content/mimiciv/3.1/). Patient-specific date shifts and anchor-year interpretation.
[^7]: Microsoft. [RAD-DINO model card](https://huggingface.co/microsoft/rad-dino) and [checkpoint preprocessor configuration](https://huggingface.co/microsoft/rad-dino/blob/fed8c449f7f64034766023af129fc58c0aa758b8/preprocessor_config.json). Released-checkpoint provenance and input specification; overlap count was independently computed locally.
[^8]: Pérez-García et al. [Exploring scalable medical image encoders beyond text supervision](https://arxiv.org/abs/2401.10815), RAD-DINO paper, 2024. Distinguish paper experiments from the later released checkpoint.
[^9]: Hayat, Geras and Shamout. [MedFuse: Multi-modal fusion with clinical time-series data and chest X-ray images](https://proceedings.mlr.press/v182/hayat22a.html), MLHC, 2022.
[^10]: Vickers and Elkin. [Decision Curve Analysis: A Novel Method for Evaluating Prediction Models](https://journals.sagepub.com/doi/10.1177/0272989X06295361), *Medical Decision Making*, 2006.
[^11]: Ding, Li, Narasimhan and Tibshirani. [Cooperative learning for multiview analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC9499553/), *PNAS*, 2022; [author preprint](https://arxiv.org/abs/2112.12337).
[^12]: Lee et al. [Deep learning-based detection of ascending aortic dilatation on chest radiographs: A diagnostic study](https://pubmed.ncbi.nlm.nih.gov/40907419/), *European Journal of Radiology* 192:112380, 2025. DOI 10.1016/j.ejrad.2025.112380. Numerical comparison here is from the indexed abstract, not an independent reanalysis.
[^13]: Demarais, Olgin and Pirruccello. [Estimating Ascending Aortic Diameter From the Electrocardiogram](https://pubmed.ncbi.nlm.nih.gov/41369619/), *JACC* 87(1):105–108, January 6, 2026, DOI 10.1016/j.jacc.2025.10.045. Detailed sample size/R² above are from the [accessible 2025 preprint record](https://pubmed.ncbi.nlm.nih.gov/40894145/), not assumed to be independently verified in the final full text.
[^14]: [Multimodal deep learning model for enhanced early detection of aortic stenosis integrating ECG and chest x-ray with cooperative learning](https://www.frontiersin.org/journals/radiology/articles/10.3389/fradi.2025.1698680/full), *Frontiers in Radiology*, 2025. DOI 10.3389/fradi.2025.1698680.
[^15]: Yin et al. [When Does Multimodal Learning Help in Healthcare? A Benchmark on EHR and Chest X-Ray Fusion](https://arxiv.org/abs/2602.23614), CareBench, February 2026 preprint. Evidence concerning the tested EHR/CXR fusion settings, not a theorem about multimodal redundancy.
[^16]: Poterucha et al. [Detecting structural heart disease from electrocardiograms using AI](https://www.nature.com/articles/s41586-025-09227-0), *Nature*, 2025; [EchoNext public data and models](https://physionet.org/content/echonext/1.1.0/).
[^17]: He et al. [Wearable Single-Lead ECG Detects Fine-Grained Structural Heart Disease Through Echo-Report Supervision](https://arxiv.org/html/2606.09332v1), June 8, 2026 preprint. Aortic-dilation-specific results, rather than the higher AUROCs for other structural subtypes, inform the comparison here.
[^18]: Pirruccello et al. [Development of a Prediction Model for Ascending Aortic Diameter Among Asymptomatic Individuals](https://jamanetwork.com/journals/jama/fullarticle/2798406), *JAMA* 328(19):1935–1944, 2022. DOI 10.1001/jama.2022.19701.
