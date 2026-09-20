# Model improvement and paper-design roadmap — 14 September 2026

Proposal for discussion, not implemented changes. Read the [scientific review](scientific_review_2026-09-14.md) first: validation defects currently prevent confident ranking of several existing approaches. This roadmap preserves the intended **root and ascending-aorta** scope. Literature informs the hypotheses; none of the papers establishes that these changes will improve this specific dataset.

## 1. Choose the question before the model

My preferred starting question is: **Among a clearly defined group receiving routine frontal CXR, can information already available at that time identify patients who should receive confirmatory echo for root/ascending dilation, and when does adding ECG or EHR improve that decision?**

That is an intended-use hypothesis, not a claim that the current echo-selected dataset already estimates population-screening performance. It requires fixing the decision time, outcome window, prior-disease exclusions, and validation population. An alternative is diameter estimation among patients already referred for TTE; that is easier to study with current labels but a narrower clinical use case. We should not move between these questions when interpreting favorable metrics.

Retain absolute diameter regression plus a prespecified ≥4 cm discrimination endpoint, with ≥4.5 cm secondary and ≥5 cm descriptive until enough positive patients support inference. Consider indexed diameter as a secondary analysis for body-size extremes, using a clinically agreed definition; it should supplement rather than silently replace the existing endpoint. Measurement and clinical decision thresholds are context dependent.[^1]

## 2. A staged experimental program

| Stage | Main question | Prerequisite | Relative burden |
|---|---|---|---|
| 0 | Can we evaluate the entire pipeline without information leakage? | Agree on time zero and cohort | Engineering + repeated base-model fitting |
| 1 | What is the strongest credible current baseline? | Stage 0 | CPU fits plus necessary CXR refits |
| 2 | Does matching CXR preprocessing and balancing image bags help? | Fixed development protocol | Moderate GPU work |
| 3 | Does more/better ECG training add conditional value? | Valid feature basis and fixed CXR baseline | Moderate GPU work |
| 4 | Does modest conditional fusion beat additive fusion? | Stages 1–3 | Small heads initially; joint fine-tuning only if warranted |
| 5 | Can loss/calibration better match the clinical endpoint? | Stable representation and sufficient events | Small heads, controlled refits |
| 6 | Does the selected model transport and support the proposed claims? | Model locked; independent data available | Data coordination/clinical review, not just compute |

Do not run a full Cartesian product of all choices. Use one-factor experiments for normalization, sampling and ECG data scale; then test only promising combinations. All decisions belong inside development. Repeatedly selecting winners from outer-test scores turns those folds into development data too.

### Stage 0 — Rebuild the evaluation contract

Implement whole-pipeline outer patient splits, with inner OOF predictions for stacking and no mixed independently trained embedding bases. Write a fit/predict interface that saves all components, and make the outer-test-label perturbation check a required test. Correct full-timestamp CXR ordering and ensure EHR/ECG are available by the selected decision time. Keep the previously exposed “holdout” explicitly exploratory.

Set aside an independent cohort if one is realistically available; otherwise be candid that corrected grouped CV is internal evaluation, not restored untouched external evidence. The external-validation design in the direct CXR aorta study is a stronger standard than its small sample size might suggest.[^2] TRIPOD+AI and PROBAST+AI provide reporting and bias-assessment frameworks, not shortcuts around the data requirements.[^3][^4]

The first successful deliverable is not a higher AUROC. It is a run that passes patient ancestry, modality timestamp, fitted-transform isolation, label-unit and artifact-reload prediction checks.

### Stage 1 — Establish the comparator that every addition must beat

Use the same evaluation episodes and strong CXR representation for the following comparisons:

| Arm | Purpose |
|---|---|
| EHR alone: regularized linear/GAM-style model and HGB | Transparent body-size/demographic baseline and a flexible clinical baseline |
| Fine-tuned CXR alone | Current strongest image-only comparator |
| Fine-tuned CXR + EHR | Practical low-overhead baseline for incremental ECG value |
| Fine-tuned CXR + ECG | Separates ECG value from EHR availability |
| Fine-tuned CXR + EHR + ECG | Tests the complete multimodal proposal |
| Each relevant arm ± frozen geometry/CXR branch | Determines whether retaining a second CXR model justifies complexity |

For the paper's modality-value objective, add ECG-only, ECG+EHR, and machine-summary-versus-waveform controls. Report both all-CXR optional-ECG performance and matched ECG-present performance. Match tuning opportunity, target scaling, early stopping, folds and preprocessing; architecture comparisons with weaker unimodal encoders answer the wrong question.

Also assess the published clinical AORTA score, which has external CT/TTE validation, where its inputs can be faithfully reconstructed.[^14] Distinguish the original fixed score from a locally recalibrated version; recalibration must use development data only. A stronger EHR arm could test pre-index heart rate and documented hypertension/diabetes alongside the current size/BP variables. Do not use post-encounter discharge diagnoses as if they were available at CXR acquisition.

Primary adoption criteria should include paired change in AP and sensitivity/PPV at a fixed feasible echo capacity, alongside AUROC and R². Agree on a practically meaningful gain before looking at revised results; a post-hoc ±0.01 R² band is not automatically a clinical minimum. Do not add two individually estimated gains together and treat the sum as an observed combined effect.

### Stage 2 — Highest-priority representation work: CXR preprocessing and multi-image learning

**Experiment C1: normalization before resolution.** Compare the existing 224-square/ImageNet-normalized pipeline with the same size/geometry using checkpoint-specific grayscale normalization. Then compare 224 with a higher-resolution, field-of-view-preserving input and the checkpoint's official 518 processor. Keep interpolation, cropping and masks explicit. RAD-DINO's published/self-supervised image representation is a sensible starting point, but the released processor and training-image provenance must be respected and reported.[^5]

This isolates an identifiable mismatch instead of attributing every gain from a large pipeline change to “resolution.” Recompute caches under immutable names; run a small image/mask overlay check before fitting models. AP/PA aspect and magnification differences make anatomical geometry especially sensitive to arbitrary square resizing. A center crop may lose peripheral anatomy, so official preprocessing is a benchmark rather than an unquestioned final choice.

**Experiment C2: balance the training unit.** First change only fine-tuning sampling/loss to give equal episode influence. Compare random one-image-per-episode training, a small sampled bag with mean pooling, and a capped bag with view/quality/recency-aware pooling. Validate at the same episode level. Then compare the current mean aggregator with a small attention-based multiple-instance model; permutation-invariant bag learning has established precedent.[^6]

The current model already averages multiple image predictions. The untested idea is *learning how to combine complementary or inconsistent images*, while preventing 90-image episodes from dominating. Limit bag size initially; stratify image selection by AP/PA and recency relative to the decision time. Never supply time-to-a-future-echo as an inference feature when that time is unknown at deployment.

**Experiment C3: fine-tuning depth and spatial representation.** After C1/C2, compare frozen, last-two-block and last-four-block training with discriminative learning rates (smaller backbone LR, larger head LR), matched stopping and a modest parameter budget. Test CLS against global+regional features before the diameter bottleneck. LoRA/adapters are optional engineering variants, not inherently novel or guaranteed improvements. Do not start with full-backbone fine-tuning plus a large fusion transformer on this effective sample size.

**What would persuade me:** paired improvements over the corrected fine-tuned CXR baseline across both AP/PA and recent-image subgroups, with no material calibration degradation. If only the frozen branch benefits, update that branch without claiming end-to-end superiority.

### Stage 3 — Reopen ECG data scale, then test transfer learning

**Experiment E1: controlled small-versus-broad training.** Repeat the existing broader-ECG comparison with identical CXR/EHR artifacts, PCA settings, fusion design and evaluation patients. Broader training already improves the saved standalone ECG predictions on matched cases; downstream benefit remains unresolved because the earlier A/B changed the non-ECG branch. Start with nested scalar ECG predictions to avoid the embedding-basis defect, then test a valid common-basis feature approach.

**Experiment E2: ECG pretraining versus training from scratch.** Test a public structural-heart-disease encoder such as EchoNext, after verifying its license, input/lead requirements, available weights and training-data overlap. Compare frozen representation plus a small head, partial fine-tuning and the existing scratch ResNet. EchoNext supports structural-supervision transfer as a plausible direction; it does not validate aortic-diameter transfer directly.[^7] The direct ECG/MRI aortic work also cautions against interpreting our current small incremental effect as an absence of ECG signal.[^8]

**Experiment E3: denser training-only cardiac supervision.** Within outer-training patients only, add masked auxiliary echo measurements/phenotypes—e.g., chamber size or LV structure—to encourage anatomical representation learning, with separate target normalization and controlled loss weights. Compare to the existing root/ascending/heart-rate objectives. Echo-report supervision in AnyECG-Echo provides related inspiration, but it is a preprint and its external aortic-dilation performance is weaker than its headline results for other subtypes.[^9]

These echo variables are **training targets only**, not inputs available at CXR/ECG inference. The root/ascending evaluation labels, including labels on other visits from outer-test patients, must remain excluded from all supervised pretraining. A joint teacher/student or contrastive model must obey the same restriction. Do not broaden the paper's output scope to arch/descending measurements without your approval.

**Waveform QC before augmentation sweeps.** The active cache preserves mV amplitudes and canonicalizes lead order, which is appropriate to retain potentially useful voltage information. It does not explicitly assert every record's sampling frequency/units, and failed records can remain as all-zero rows. Audit failures, nonfinite/flat leads, gain, length and frequency; use a quality/missingness mask rather than silently treating failed signals as valid. I have not established that the current cache contains such failures. Avoid automatically adopting the legacy per-record z-scoring loader, which would change amplitude information. Any pretrained model must receive its expected preprocessing; compare it fairly rather than forcing every encoder through the old cache.

### Stage 4 — Low-capacity, genuinely conditional fusion

The current CXR diameter scalars discard most image information before EHR/ECG can affect their interpretation. A plausible hypothesis is that body size, AP/PA projection or cardiac context changes which image features should matter. That calls for a constrained conditional model, not automatically a large attention network.

Start with one small EHR-conditioned affine/gating layer on a CXR feature vector or regional tokens, followed by a root/ascending head. DAFT provides precedent for conditioning image features on tabular information, demonstrated in a different imaging domain; transfer to aortic CXR is our hypothesis, not an established benefit.[^10] An even cheaper control is a prespecified set of interactions between CXR features and age/sex/BSA/projection, with regularization.

Compare additive fusion, conditional fusion and a capacity-matched image-only head. Only after a useful signal appears should we consider joint CXR/ECG feature fine-tuning or spatial cross-attention. The existing three-token attention null does not answer this narrower conditional-spatial hypothesis.

Use explicit modality masks, modality dropout during training and a well-defined missing-ECG path. MedFuse and CareBench show that missingness and task-specific fusion behavior deserve direct evaluation.[^11][^12] Distinguish robustness to artificial removal from actual clinical absence, which may carry selection information. Consider selective ECG use only if a subgroup/availability policy is learned on development data and frozen before evaluation; subgroup hunting on test errors would manufacture a contribution.

**Novelty caution:** FiLM/DAFT, attention pooling, cooperative learning, missing-modality gates and contrastive supervision are existing methods. A defensible methodological contribution would require a specific new formulation, rationale and ablation-supported advantage—not merely renaming their combination.

### Stage 5 — Align optimization with the tail and clinical decision

The current squared-error diameter objective emphasizes the abundant normal range. The observed negative error among dilated cases motivates an experiment, but regression toward the mean and noisy extreme labels also explain some of it. Blindly oversampling every large recorded diameter could amplify measurement errors.

Compare the regression baseline with a shared representation and two modest heads: continuous diameter plus an ordinal/threshold risk head. Enforce monotonicity across ≥4 and ≥4.5 cm probabilities, or use an ordered distributional model. Keep ≥5 cm descriptive initially; nine positive root patients do not support a reliable independent rare-event training/tuning program.

Evaluate unweighted regression, a robust loss and a carefully bounded tail-sensitive objective with the same architecture. If using event oversampling or class weights, calibrate on development data at the natural target prevalence. Test a simple monotone calibration of existing diameter scores before adding another deep head: ranking might already be adequate for a more useful probability output.

Report Brier score/calibration slope and intercept, reliability plots with enough events per bin, AP, and fixed-threshold sensitivity/specificity/PPV. Use decision curves to quantify consequences of extra confirmatory echoes under agreed benefit/harm assumptions, rather than treating a tiny AUROC gain as clinical utility.[^13] A scalar calibration cannot increase AUROC if it preserves ranking; any apparent claim to that effect needs scrutiny.

### Stage 6 — Cohort and outcome quality may matter more than another encoder

**Time-window design:** after fixing time zero, compare the latest eligible CXR, all recent CXRs and the longer history on the *same eligible patients* where possible. Also report the broader cohort's attrition. Shortening the window changes both label alignment and sample selection; separate those effects. Use contiguous lag bins, not integer gaps applied to fractional mean lags.

**Episode selection:** compare the current “separate echo dates, then require CXR” rule against “identify eligible paired episodes, then apply separation.” The current order can discard an otherwise useful eligible episode because an earlier ineligible echo was retained first. Include first-eligible-episode-per-patient as a sensitivity analysis and report patient-level performance. More visits are not more independent patients.

**Occult versus established disease:** define a primary or secondary cohort without documented prior aortic dilation/repair before the decision time. Examine device/procedure markers and prior echo utilization, but use only available pre-index information for predictive covariates. No prior diagnosis is not proof of no pre-existing disease, and a negative code search is not a validated exclusion by itself.

**Label quality:** a clinician-reviewed subset near the 4/4.5 cm thresholds and in the extremes could be disproportionately useful. Check anatomical convention, source measurements and report consistency, ideally with blinded repeated measurement where source imaging is available. The structured label table is much broader than the echo-DICOM subset, so raw images for every current target should not be assumed available. CT/MRI confirmation, if obtainable, is a distinct reference-standard study and needs its own selection analysis.

For training only, a stable short-interval repeat-label model could reduce noise, but prespecify stability criteria and avoid patients with intervening repair or major clinical change. A latent-measurement model with uncertainty is preferable to declaring a universal hard R² ceiling. Keep an independently adjudicated test target if the claim is improved anatomical accuracy; agreement with a smoothed training target is not sufficient.

**Generalization and subgroups:** validate at another institution or on truly unused, pretraining-unseen data. Report AP/PA, age, sex, body size, care setting, ECG availability and prior-disease strata with event counts and uncertainty. Avoid intersectional tables too sparse for interpretation. Describe transportability limits and screening verification bias explicitly; patients without echo cannot be relabeled as controls.

## 3. What the paper could become

Three defensible paths depend on the repaired results:

1. **Positive multimodal study:** full or selective fusion meaningfully improves over fine-tuned CXR+EHR, with valid patient/timing controls and independent confirmation. Contribution: demonstrated conditional utility for this aortic task, plus a rigorously justified model.
2. **Simpler-model study:** ECG adds little practical benefit despite a strong ECG encoder and fair fusion. Contribution: a careful, reproducible modality-value result and a simpler deployable CXR±EHR model. This needs stronger controls and external evidence than simply reporting a nonsignificant coefficient.
3. **Measurement/triage study:** model is useful for prioritizing confirmatory echo but not accurate enough for diameter substitution. Contribution: explicit uncertainty, calibrated risk and realistic resource trade-offs, without claiming CT-equivalent measurement or safe exclusion of severe dilation.

None is automatically novel merely because root and ascending targets are combined. The literature already contains direct CXR aortic detection, ECG aortic estimation, multimodal cardiac models and EHR/CXR benchmarks. The most valuable distinct contribution may be the *answer*: which available modalities improve a specific clinical decision, for whom, under what information timing, and whether that answer survives a change of population.

## 4. Practical stopping and adoption rules

Before launching the next substantial run, agree on the primary endpoint/site or multiplicity plan, clinically meaningful incremental benefit, intended echo capacity, permitted data availability, and independent evaluation strategy. Preserve all unsuccessful runs and immutable configs.

Stop expanding an architecture family when its well-tuned, valid comparison cannot establish a practically useful gain over the strongest simpler baseline. Do not interpret “CI includes zero” as equivalence; design equivalence margins prospectively if that is the scientific question. Conversely, do not retain ECG solely because a tiny R² increase is statistically detectable if it adds no useful ranking/calibration benefit and increases deployment burden.

My first three performance experiments, **after evaluation repair**, would be: (1) CXR checkpoint-normalization/geometry controls, (2) episode-balanced fine-tuning with a small multi-image pooling comparison, and (3) the controlled broader-ECG re-test. These are motivated by concrete mismatches or unresolved local evidence. Conditional fusion and auxiliary supervision come next, not first.

## Sources

[^1]: Isselbacher et al. [2022 ACC/AHA Aortic Disease Guideline](https://www.jacc.org/doi/10.1016/j.jacc.2022.08.004).
[^2]: Lee et al. [Deep learning-based detection of ascending aortic dilatation on chest radiographs: A diagnostic study](https://pubmed.ncbi.nlm.nih.gov/40907419/), *European Journal of Radiology*, 2025.
[^3]: Collins et al. [TRIPOD+AI](https://www.bmj.com/content/385/bmj-2023-078378), *BMJ*, 2024.
[^4]: Moons et al. [PROBAST+AI](https://www.bmj.com/content/388/bmj-2024-082505), *BMJ*, 2025.
[^5]: Microsoft [RAD-DINO checkpoint documentation](https://huggingface.co/microsoft/rad-dino); Pérez-García et al. [Exploring scalable medical image encoders beyond text supervision](https://arxiv.org/abs/2401.10815), 2024.
[^6]: Ilse, Tomczak and Welling. [Attention-based Deep Multiple Instance Learning](https://proceedings.mlr.press/v80/ilse18a.html), ICML, 2018. General bag-pooling precedent; not a CXR-aorta validation study.
[^7]: Poterucha et al. [Detecting structural heart disease from electrocardiograms using AI](https://www.nature.com/articles/s41586-025-09227-0), *Nature*, 2025; [EchoNext public resource](https://physionet.org/content/echonext/1.1.0/).
[^8]: Demarais, Olgin and Pirruccello. [Estimating Ascending Aortic Diameter From the Electrocardiogram](https://pubmed.ncbi.nlm.nih.gov/41369619/), *JACC*, 2026; [accessible preprint](https://pubmed.ncbi.nlm.nih.gov/40894145/), 2025.
[^9]: He et al. [Wearable Single-Lead ECG Detects Fine-Grained Structural Heart Disease Through Echo-Report Supervision](https://arxiv.org/html/2606.09332v1), June 2026 preprint.
[^10]: Pölsterl, Wolf and Wachinger. [Combining 3D Image and Tabular Data via the Dynamic Affine Feature Map Transform](https://arxiv.org/abs/2107.05990), MICCAI, 2021. DAFT was demonstrated in a different imaging application; the proposed adaptation is a hypothesis.
[^11]: Hayat, Geras and Shamout. [MedFuse](https://proceedings.mlr.press/v182/hayat22a.html), MLHC, 2022.
[^12]: Yin et al. [When Does Multimodal Learning Help in Healthcare? A Benchmark on EHR and Chest X-Ray Fusion](https://arxiv.org/abs/2602.23614), CareBench, February 2026 preprint.
[^13]: Vickers and Elkin. [Decision Curve Analysis](https://journals.sagepub.com/doi/10.1177/0272989X06295361), *Medical Decision Making*, 2006.
[^14]: Pirruccello et al. [Development of a Prediction Model for Ascending Aortic Diameter Among Asymptomatic Individuals](https://jamanetwork.com/journals/jama/fullarticle/2798406), *JAMA*, 2022.
