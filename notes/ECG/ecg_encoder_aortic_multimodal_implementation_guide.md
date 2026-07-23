# ECG Encoder Selection and Implementation Plan for Multimodal Aortic Diameter Prediction

**Project:** Prediction of aortic root and ascending aorta diameter from ECG, chest X-ray, and EHR data  
**Primary constraint:** Approximately 522 patients have all modalities and echocardiographic diameter labels  
**Prepared:** 2026-07-16  
**Intended audience:** Research engineers and coding agents implementing and evaluating the ECG branch and multimodal late-fusion system

---

## 1. Executive summary

The complete multimodal cohort is too small to support unrestricted end-to-end training of a large ECG encoder together with the chest X-ray, EHR, and fusion networks. The ECG component should therefore be handled as a transfer-learning and representation-extraction problem.

The recommended design is:

1. Use a publicly released pretrained ECG encoder.
2. Reproduce that encoder's required preprocessing exactly.
3. Initially freeze the encoder and extract ECG representations.
4. Reduce or regularize the representation before fusion.
5. Keep the current late-fusion architecture fixed while comparing ECG encoders.
6. Use repeated patient-level cross-validation and evaluate the incremental benefit of ECG beyond EHR and chest X-ray.
7. Use the much larger ECG-only pool for in-domain self-supervised adaptation, provided leakage is controlled.
8. Use every available ECG–echocardiogram pair for supervised ECG-specific training, even when chest X-ray is missing. The 522 complete cases should limit only the final multimodal fusion stage, not ECG representation learning.

### Recommended accessible encoders

For immediate implementation, use this order:

1. **xECG** — easiest modern model to download and use as a frozen feature extractor.
2. **ECG-CPC** — compact public checkpoint and particularly relevant because recent benchmarking found it strong on cardiac structure and regression-type tasks.
3. **ECGFounder** — accessible official weights and code; useful clinical CNN baseline, but preprocessing must be followed exactly.
4. **ECG-FM** — accessible weights and strong representation model, but substantially heavier and tied to the Fairseq Signals ecosystem.
5. **ECG-JEPA** — scientifically attractive for low-label settings; weights are available through a benchmark repository, but the current Google Drive distribution is less convenient for automated cluster deployment.
6. **PCLR** — retain as the established baseline.

The practical first experiment should compare frozen PCLR, xECG, ECG-CPC, and ECGFounder under the same folds, fusion model, and preprocessing-quality controls.

---

## 2. Scientific framing

An ECG does not directly measure aortic diameter. The aorta does not generate the surface electrical signal in the way myocardium does. Nevertheless, an ECG can contain information statistically associated with aortic dimensions through several pathways:

- Cardiac position and electrical-axis changes
- Ventricular hypertrophy or remodeling
- Aortic regurgitation or stenosis
- Hypertension and altered afterload
- Atrial enlargement
- Body geometry and thoracic anatomy
- Age, sex, height, body size, and chronic disease
- Conduction and repolarization changes accompanying structural cardiovascular disease

This means the ECG branch may learn a combination of:

1. **Potentially diameter-related physiological information**
2. **Cardiac remodeling associated with the same disease process**
3. **Stable demographic and anthropometric proxies**
4. **Institution-specific acquisition or population patterns**

The key scientific question is therefore not merely:

> Can ECG predict aortic diameter?

It is:

> Does ECG improve held-out prediction after accounting for the information already contained in EHR and chest X-ray?

The evaluation must explicitly measure this incremental contribution.

---

## 3. Why the 522-patient cohort changes the modeling strategy

With 522 complete multimodal observations, a high-dimensional end-to-end fusion system can overfit even if every individual encoder is pretrained.

Common risks include:

- Thousands of embedding dimensions relative to the number of patients
- Unstable validation estimates
- Learning demographic shortcuts
- Fold-to-fold variation that is larger than the difference between encoders
- Overfitting during full encoder fine-tuning
- Leakage through repeated ECGs or repeated imaging from the same patient
- Model selection overfitting from testing many architectures on the same folds

Therefore, the multimodal stage should be deliberately small.

A strong default is:

```text
ECG waveform -> frozen pretrained encoder -> 32–128 dimensional bottleneck
CXR image    -> frozen or separately trained encoder -> 32–128 dimensional bottleneck
EHR features -> regularized tabular model or small MLP -> 16–64 dimensional bottleneck

Concatenate bottlenecks + modality masks
    -> LayerNorm
    -> small MLP
    -> root diameter and ascending-aorta diameter
```

An even safer alternative is prediction-level stacking:

```text
ECG-only prediction
CXR-only prediction
EHR-only prediction
    -> regularized linear or shallow nonlinear fusion model
```

Prediction-level stacking may lose some cross-modal latent interactions, but it is statistically attractive for a cohort of 522 patients.

---

## 4. Publicly accessible ECG encoder weights

## 4.1 Accessibility matrix

| Encoder | Public weights | Access route | Approximate integration burden | Main reason to test |
|---|---:|---|---:|---|
| **xECG** | Yes | Hugging Face | Low to moderate | Easy pooled embedding and explicit support for signal-level regression heads |
| **ECG-CPC** | Yes | Figshare checkpoint + official benchmark code | Moderate | Compact and strong on structure/regression tasks in broad benchmarking |
| **ECGFounder** | Yes | Hugging Face + official GitHub | Moderate | Large-scale supervised ECG pretraining and straightforward CNN-style backbone |
| **ECG-FM** | Yes | Hugging Face + official GitHub | High | Large open wav2vec-style model with pretrained and fine-tuned checkpoints |
| **ECG-JEPA** | Yes, but less conveniently packaged | Google Drive link in official benchmark repository | Moderate to high | Strong candidate for low-label transfer; JEPA objective may retain useful general structure |
| **PCLR** | Yes | GitHub | Low | Existing baseline trained on more than three million ECGs |
| **Aortic-diameter-specific model from the direct paper** | No verified public checkpoint located | Architecture must be reproduced | High | Most target-specific architecture, but no drop-in pretrained weights currently verified |

---

## 4.2 xECG

### Why it is a strong first implementation

xECG is currently the most convenient modern alternative to PCLR for this project.

The public model page provides:

- A downloadable pretrained checkpoint
- A model class with a `from_pretrained` interface
- Patch-level representations
- A pooled signal-level representation
- Downstream classes intended for classification or regression
- An explicit required 12-lead order

The expected lead order is:

```python
["i", "ii", "iii", "avr", "avl", "avf",
 "v1", "v2", "v3", "v4", "v5", "v6"]
```

The Hugging Face model card reports approximately 57 million parameters. This is not a problem if the encoder is frozen during representation extraction.

### Official resources

- Model: https://huggingface.co/riccardolunelli/xECG_base_model_v1
- Code: https://github.com/dlaskalab/bench-xecg
- Paper: https://arxiv.org/abs/2509.10151

### Suggested cluster setup

```bash
git clone --branch release https://github.com/dlaskalab/bench-xecg.git
cd bench-xecg

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install huggingface_hub

hf download riccardolunelli/xECG_base_model_v1 \
  --local-dir checkpoints/xecg
```

The coding agent should confirm the repository branch and requirements at implementation time.

### First evaluation mode

1. Freeze all xECG parameters.
2. Extract the pooled representation for every ECG.
3. Save embeddings to disk keyed by patient ID and ECG timestamp.
4. Test:
   - Ridge regression
   - Elastic net
   - A one-hidden-layer MLP
5. Use the same frozen embeddings in the late-fusion model.
6. Do not begin with full fine-tuning.

### Important caution

The Hugging Face page supplies a custom model implementation rather than a standard `transformers.AutoModel` workflow. The agent should vendor or import the provided `xECG.py` implementation exactly as instructed by the model card.

---

## 4.3 ECG-CPC

### Why it is highly relevant

ECG-CPC is a compact foundation model proposed and evaluated in a recent broad ECG-foundation-model benchmark. In that benchmark, it was especially strong outside routine diagnostic-label classification, including cardiac structure and other regression-oriented tasks.

This is particularly relevant because aortic diameter is:

- A continuous target
- A structural target
- Likely represented by subtle distributed morphology rather than one explicit ECG diagnosis

### Public checkpoint

The public checkpoint package contains:

- Model checkpoint files
- A configuration file
- A stated intended use as a transfer-learning feature extractor

The Figshare package is approximately 34 MB, making it much easier to store and move than some of the larger foundation models.

### Official resources

- Benchmark repository: https://github.com/AI4HealthUOL/ecg-fm-benchmarking
- ECG-CPC checkpoint page: https://figshare.com/articles/dataset/ECG-CPC_Checkpoint_zip/30192604
- Direct checkpoint downloader: https://figshare.com/ndownloader/files/58173919
- Benchmark paper: https://arxiv.org/abs/2509.25095
- OpenReview: https://openreview.net/forum?id=xXRqWpt3Xr

### Suggested cluster download

```bash
mkdir -p checkpoints/ecg_cpc
wget -O checkpoints/ecg_cpc/ECGCPCCheckpoint.zip \
  https://figshare.com/ndownloader/files/58173919

unzip checkpoints/ecg_cpc/ECGCPCCheckpoint.zip \
  -d checkpoints/ecg_cpc
```

### First evaluation mode

Use both:

1. A linear or ridge head
2. A small nonlinear head

A safe nonlinear head is:

```text
ECG-CPC embedding
    -> LayerNorm
    -> Linear(d, 64)
    -> GELU
    -> Dropout(0.2–0.5)
    -> Linear(64, output dimension)
```

For two continuous targets, the output dimension can be two, with separate losses or a masked multitask loss.

### Why a nonlinear head matters

Some pretrained objectives organize useful information across token-level or nonlinear feature combinations. A model can appear weak under pure linear probing while performing well with a very small nonlinear head. Since the complete cohort is small, the nonlinear head must remain tiny and strongly regularized.

---

## 4.4 ECGFounder

### Why it is worth testing

ECGFounder was trained on more than 10 million ECGs with a large supervised diagnostic label set. It may encode clinically meaningful features relevant to aortic dimensions, including:

- Ventricular hypertrophy
- Axis abnormalities
- Chamber enlargement
- Conduction disease
- Repolarization abnormalities
- Aortic-valve-related electrical patterns

It is not specifically pretrained for diameter regression, but it is a useful contrast to self-supervised encoders.

### Public weights

The official Hugging Face repository contains:

- `12_lead_ECGFounder.pth`
- `1_lead_ECGFounder.pth`

The official code repository includes downstream fine-tuning examples.

### Official resources

- Model: https://huggingface.co/PKUDigitalHealth/ECGFounder
- Code: https://github.com/PKUDigitalHealth/ECGFounder
- Paper: https://arxiv.org/abs/2410.04133

### Suggested cluster download

```bash
git clone https://github.com/PKUDigitalHealth/ECGFounder.git
cd ECGFounder

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install huggingface_hub

hf download PKUDigitalHealth/ECGFounder \
  12_lead_ECGFounder.pth \
  config.json \
  --local-dir checkpoint
```

### Critical preprocessing warning

The authors explicitly state that users must follow the repository's preprocessing implementation, including filtering and z-score normalization. The coding agent should treat `dataset.py` as the source of truth.

Do not replace this pipeline with the PCLR pipeline or a generic ECG pipeline and assume the pretrained weights will remain valid.

### Voltage-preservation issue

Record-wise z-score normalization can remove absolute amplitude information. Absolute amplitude may contain useful information related to ventricular hypertrophy, body habitus, chest geometry, low-voltage states, or lead placement.

Therefore, test ECGFounder in two ways:

```text
A. ECGFounder embedding only
B. ECGFounder embedding + engineered raw-voltage features
```

Engineered raw-voltage features should be computed before record-wise normalization.

---

## 4.5 ECG-FM

### Why it may be useful

ECG-FM is an open wav2vec 2.0-style ECG foundation model with approximately 90.9 million parameters. The official Hugging Face repository provides both:

- A pretrained checkpoint
- A checkpoint fine-tuned on MIMIC-IV-ECG diagnostic tasks

For representation extraction, use the pretrained checkpoint first, because the diagnostic fine-tuning may narrow the representation toward the published label set.

### Public files

The Hugging Face model repository includes:

- `mimic_iv_ecg_physionet_pretrained.pt`
- `mimic_iv_ecg_physionet_pretrained.yaml`
- `mimic_iv_ecg_finetuned.pt`
- `mimic_iv_ecg_finetuned.yaml`

The model repository is approximately 2.17 GB.

### Official resources

- Model: https://huggingface.co/wanglab/ecg-fm
- Code: https://github.com/bowang-lab/ECG-FM
- Paper: https://arxiv.org/abs/2408.05178

### Suggested cluster download

```bash
git clone https://github.com/bowang-lab/ECG-FM.git
cd ECG-FM

pip install huggingface_hub

hf download wanglab/ecg-fm \
  mimic_iv_ecg_physionet_pretrained.pt \
  mimic_iv_ecg_physionet_pretrained.yaml \
  --local-dir checkpoints/ecg_fm
```

### Integration burden

ECG-FM uses the Fairseq Signals framework. This makes it less suitable as the first replacement for PCLR because:

- Environment setup is heavier
- Checkpoints are large
- The inference path is less direct than xECG
- Reproducible preprocessing and configuration require more care

It remains a valuable later comparison, especially if the team already has experience with Fairseq Signals.

---

## 4.6 ECG-JEPA

### Scientific motivation

ECG-JEPA learns by predicting masked latent representations rather than reconstructing every waveform sample. This can produce useful representations without forcing the model to reproduce nuisance noise.

JEPA-style learning is scientifically attractive for the current problem because:

- The target is subtle and structural
- The multimodal label set is small
- A general latent representation may transfer better than a diagnosis-specific representation
- Recent benchmarking reported favorable low-label behavior

### Accessibility reality

The official implementation is public, but its repository does not currently present a simple official Hugging Face checkpoint download.

A separate official ECG-foundation-model benchmark repository provides an ECG-JEPA checkpoint link hosted on Google Drive.

### Resources

- Official code: https://github.com/kweimann/ECG-JEPA
- Checkpoint link exposed by the benchmark repository:  
  https://drive.google.com/file/d/1gMOT4xjQQg0GZkY1iE6NuDzua4ALw00l/view
- Benchmark code that references the checkpoint:  
  https://github.com/AI4HealthUOL/ecg-fm-benchmarking
- JEPA paper: https://arxiv.org/abs/2410.13867

### Practical limitation

Google Drive can be inconvenient on a compute cluster because:

- Downloads may require browser interaction
- Automated downloads can be blocked by confirmation pages
- Reproducibility is weaker than a versioned Hugging Face or Figshare asset

ECG-JEPA is therefore a good scientific experiment but not the easiest first engineering experiment.

---

## 4.7 PCLR

PCLR should remain in the comparison as a serious baseline.

The public repository provides:

- Keras weights
- PyTorch weights
- A 12-lead model
- A representation extraction pipeline
- A 320-dimensional representation

The model expects a 10-second, 12-lead ECG interpolated to 4,096 samples in a specific lead order.

### Resources

- Practical implementation: https://github.com/mit-ccrg/PCLR
- Original model location: https://github.com/broadinstitute/ml4h/tree/master/model_zoo/PCLR
- Paper: https://arxiv.org/abs/2104.04569

PCLR may perform well because it captures stable patient characteristics. The core scientific comparison is whether newer encoders provide greater improvement beyond EHR and chest X-ray.

---

## 5. Recommended experimental priority

## Phase 1: frozen-encoder comparison

Implement:

1. PCLR
2. xECG
3. ECG-CPC
4. ECGFounder

Keep all other components fixed.

For each encoder, extract ECG embeddings and evaluate:

```text
ECG-only:
    embedding -> ridge
    embedding -> elastic net
    embedding -> small MLP

Multimodal:
    ECG embedding + fixed CXR embedding + fixed EHR representation
        -> same late-fusion model
```

Do not change the CXR encoder, EHR feature set, split, loss, or fusion architecture during the encoder comparison.

## Phase 2: add ECG-JEPA and ECG-FM

Add these after the first four pipelines are stable.

This avoids spending engineering time on heavier dependencies before determining whether changing the ECG representation materially affects the result.

## Phase 3: local adaptation

Take the best one or two encoders and continue self-supervised learning on the institutional ECG collection.

## Phase 4: supervised ECG–echo training

Use all available patients with ECG and echocardiographic aortic labels, even when chest X-ray is missing.

## Phase 5: limited parameter-efficient adaptation

Only after frozen encoders have been evaluated should the team consider:

- Unfreezing the final encoder block
- Layer-wise learning-rate decay
- Small adapter layers
- Low-rank adaptation when technically appropriate
- Updating normalization layers only

Full end-to-end fine-tuning on 522 patients should not be the initial approach.

---

## 6. Preprocessing protocol

## 6.1 Build a canonical raw ECG representation

Before model-specific preprocessing, standardize every ECG into a canonical object containing:

```text
patient_id
ecg_id
acquisition_timestamp
original_sampling_rate
duration_seconds
lead_names
waveform_in_millivolts
gain_metadata
filter_metadata, if available
quality_flags
```

Use the canonical lead order:

```text
I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
```

Do not rely only on positional channel indices from source files. Validate channel names.

## 6.2 Preserve an untouched waveform

Store:

1. The original calibrated waveform
2. A canonical resampled waveform
3. Model-specific preprocessed versions

Never overwrite the raw calibrated waveform.

## 6.3 Unit conversion

Convert digital values to millivolts using the original gain and baseline metadata.

Quality checks should flag:

- Implausibly large amplitudes
- Almost-flat channels
- Clipping
- Constant channels
- NaNs or infinities
- Missing calibration metadata
- Suspected unit errors

## 6.4 Resampling

Each encoder must receive the sampling rate expected by its implementation.

Use polyphase or another anti-aliased resampling method. Avoid naive sample dropping.

Record:

```text
source_sampling_rate
target_sampling_rate
resampling_method
```

## 6.5 Duration handling

When an encoder expects less than the full 10 seconds:

- Extract multiple deterministic windows, such as 0–5 seconds and 5–10 seconds.
- Compute an embedding for each window.
- Average or concatenate the embeddings.
- Report window-to-window agreement as a quality signal.

When the encoder expects 10 seconds:

- Prefer the complete recording.
- Pad only when necessary.
- Supply a valid-length mask if supported.

## 6.6 Missing leads

Do not silently insert zeros without tracking the missing lead.

For models that require all 12 leads:

1. Fill a missing lead according to the model's recommended policy.
2. Create a 12-dimensional lead-presence mask.
3. Include the mask as an auxiliary input to the downstream head or fusion model.
4. Report performance for complete-lead and incomplete-lead subsets.

## 6.7 Filtering

Follow the pretrained model's original pipeline.

Do not impose one universal filtering pipeline across all encoders. A model trained on minimally processed waveforms may degrade if aggressive filtering is added, while a model trained on filtered and normalized waveforms may fail if preprocessing is omitted.

## 6.8 Normalization

Maintain both:

1. A checkpoint-compatible normalized waveform
2. Raw-voltage features from the calibrated waveform

Potential raw-voltage features include:

- Peak-to-peak amplitude per lead
- RMS amplitude per lead
- QRS RMS amplitude
- Maximum absolute voltage
- Cornell voltage
- Sokolow–Lyon voltage
- Frontal QRS axis
- P-wave and T-wave axes
- PR interval
- QRS duration
- QT and QTc
- Heart rate
- RR variability
- R-wave progression features

These features should be generated with validated ECG processing software rather than fragile custom peak detection whenever possible.

## 6.9 Quality-control variables

Generate and retain:

```text
flat_lead_count
missing_lead_count
clipping_fraction
nan_fraction
signal_range_by_lead
baseline_wander_estimate
high_frequency_noise_estimate
powerline_noise_estimate
estimated_heart_rate
number_of_detected_beats
window_embedding_consistency
```

Quality variables can be used:

- To exclude unusable ECGs
- As adjustment variables
- For sensitivity analysis
- To diagnose model failure

Do not use quality thresholds selected by examining test-set outcomes.

---

## 7. Representation extraction

## 7.1 Freeze first

For the initial benchmark:

```python
model.eval()
for parameter in model.parameters():
    parameter.requires_grad = False
```

Use `torch.inference_mode()` or `torch.no_grad()` during embedding extraction.

## 7.2 Save embeddings offline

Do not repeatedly run large encoders during every fusion experiment.

Store a table such as:

```text
patient_id
ecg_id
timestamp
encoder_name
encoder_version
preprocessing_version
embedding
quality_flags
```

Recommended formats:

- NumPy `.npy` or `.npz`
- HDF5
- Parquet with vector support
- PyTorch tensors plus a metadata table

## 7.3 Version every representation

The embedding artifact name should encode:

```text
encoder
checkpoint hash
preprocessing version
sampling rate
window strategy
pooling strategy
```

Example:

```text
xecg_v1_sr500_full10s_meanpool_preproc_v3.pt
```

## 7.4 Pooling strategies

Evaluate:

- Model-provided pooled embedding
- Mean of patch/token embeddings
- Attention pooling trained only on the training fold
- Concatenation of final and penultimate layer pooled features
- Mean of embeddings from two 5-second windows

Do not test a large number of pooling variants without nested model selection. With 522 patients, excessive experimentation can overfit the cross-validation process.

---

## 8. Dimensionality reduction before fusion

The ECG embedding dimension should usually be reduced before fusion.

Test:

```text
No reduction
PCA to 16, 32, 64, or 128 dimensions
Supervised linear bottleneck
Small autoencoder trained only on training-fold embeddings
```

PCA must be fitted inside each training fold.

A strong default is:

```text
embedding
    -> LayerNorm
    -> Linear(d, 64)
    -> GELU
    -> Dropout(0.3)
```

Alternatively, use ridge regression directly on the full embedding for the unimodal benchmark.

---

## 9. Fusion design

## 9.1 Latent late fusion

```text
z_ecg = ECG bottleneck
z_cxr = CXR bottleneck
z_ehr = EHR bottleneck

z = concatenate(
    z_ecg,
    z_cxr,
    z_ehr,
    modality_presence_mask,
    modality_quality_features
)

prediction = small_fusion_mlp(z)
```

Recommended fusion-head size:

```text
Input
 -> LayerNorm
 -> Linear(input_dim, 64)
 -> GELU
 -> Dropout(0.3–0.5)
 -> Linear(64, 16)
 -> GELU
 -> Linear(16, 2)
```

Two outputs:

1. Aortic root diameter
2. Ascending-aorta diameter

## 9.2 Prediction-level stacking

Train unimodal models and generate strictly out-of-fold predictions:

```text
root_ecg, ascending_ecg
root_cxr, ascending_cxr
root_ehr, ascending_ehr
```

Then fit a regularized meta-model on those predictions.

This approach has major advantages:

- Very low fusion dimensionality
- Easier interpretation
- Lower overfitting risk
- Natural handling of modality contribution
- Direct estimation of incremental value

## 9.3 Modality dropout

During training, randomly mask one modality with modest probability. This can reduce dependence on one dominant branch and improve robustness.

However, because the current scientific analysis focuses on complete cases, always report complete-case performance separately.

## 9.4 Gating

A small gate may learn modality weights:

```text
weights = softmax(gating_network(summary_features))
prediction = sum(weights_m * prediction_m)
```

Use gating only after simple concatenation and stacking baselines. A complex gate may overfit 522 observations.

---

## 10. Target design

Model the aortic root and ascending aorta separately, even if they share an encoder.

A useful multitask setup is:

```text
Shared ECG representation
    -> root-specific head
    -> ascending-specific head
```

Possible targets:

1. Absolute diameter in millimeters
2. Body-size-indexed diameter
3. Age-, sex-, and body-size-adjusted residual diameter
4. Dilation classification
5. Joint continuous regression and dilation classification

A robust loss is:

```text
L = L_huber(root) + L_huber(ascending)
```

If labels are missing for one anatomic location, use a masked loss.

A multi-objective version is:

```text
L =
    L_huber(root_mm)
  + L_huber(ascending_mm)
  + lambda_root * L_binary(root_dilated)
  + lambda_asc  * L_binary(ascending_dilated)
```

Clinical dilation thresholds must be defined with clinician input and should account for whether indexed or unindexed measurements are used.

---

## 11. Exploiting the larger ECG collection

## 11.1 In-domain self-supervised adaptation

Continue the original self-supervised objective on the institutional ECG collection.

Potential benefits:

- Adaptation to local machines
- Adaptation to local filters and gain conventions
- Adaptation to local patient demographics
- Better robustness to local noise
- Reduced domain shift

Preferred order:

```text
public pretrained checkpoint
 -> local self-supervised adaptation
 -> frozen local encoder
 -> downstream diameter head
 -> multimodal fusion
```

Use a low learning rate and limit training duration.

## 11.2 Leakage control during self-supervised adaptation

The strictest design is:

- In each outer cross-validation fold, adapt only on ECGs from patients not in the outer test fold.

A practical alternative is:

- Use a separate institutional ECG pool with no patient overlap with the complete multimodal cohort.

Do not assume unlabeled test ECGs are harmless. Self-supervised exposure to held-out patients can make the evaluation transductive.

## 11.3 Use all ECG–echo pairs

Search the full database for patients with:

```text
ECG + echocardiographic root diameter
ECG + echocardiographic ascending-aorta diameter
```

They do not need to have chest X-ray.

Train an ECG-specific model on this expanded paired cohort, then freeze it before the 522-patient fusion stage.

This is likely more valuable than full fine-tuning on only 522 complete cases.

## 11.4 Time matching

For each ECG–echo pair, record the absolute interval between acquisitions.

Evaluate progressively stricter windows:

```text
<= 7 days
<= 30 days
<= 90 days
<= 180 days
```

Model performance as a function of interval. This helps distinguish stable patient features from time-varying physiological signal.

## 11.5 Auxiliary institutional labels

Potential auxiliary tasks include:

- Heart rate
- PR interval
- QRS duration
- QT and QTc
- P/QRS/T axes
- Automated ECG statements
- Left-ventricular-hypertrophy criteria
- Atrial enlargement
- Conduction abnormalities
- Paced rhythm
- Valve disease labels
- Ejection fraction category, if available
- Other echocardiographic structural variables

These tasks may encourage the encoder to represent cardiac geometry and remodeling before diameter-specific training.

---

## 12. Measuring incremental ECG value

## 12.1 Baseline models

Train and evaluate:

```text
EHR only
CXR only
ECG only
EHR + CXR
EHR + ECG
CXR + ECG
EHR + CXR + ECG
```

## 12.2 Primary comparison

The primary test should be:

```text
EHR + CXR
versus
EHR + CXR + ECG
```

Use identical outer folds and compare paired out-of-fold predictions.

## 12.3 Residual learning experiment

Generate strictly cross-fitted EHR+CXR predictions:

\[
r_i = y_i - \hat{y}_{EHR+CXR,i}
\]

Train the ECG branch to predict the residual:

\[
\hat{r}_i = f_{ECG}(x_{ECG,i})
\]

Then:

\[
\hat{y}_{combined,i}
=
\hat{y}_{EHR+CXR,i}
+
\hat{r}_i
\]

This directly tests whether the ECG contains complementary information.

Do not compute residuals using in-sample baseline predictions.

## 12.4 Demographic residualization

A second scientific analysis is:

```text
diameter ~ age + sex + height + weight/BSA + blood pressure
```

Use out-of-fold predictions from this clinical model and train ECG to predict the remaining residual.

This tests whether ECG captures more than the most obvious body-size and demographic relationships.

---

## 13. Cross-validation and statistical evaluation

## 13.1 Splitting

Use patient-level splitting.

All records from the same patient must remain in one fold, including:

- Multiple ECGs
- Multiple chest X-rays
- Multiple echocardiograms
- Repeated EHR snapshots

If possible, evaluate a temporal or site-based holdout.

## 13.2 Recommended cross-validation

A practical design is:

- 5-fold grouped cross-validation
- Repeated with multiple random seeds
- Identical folds for every encoder
- Inner validation or nested cross-validation for hyperparameter selection

Avoid selecting an encoder on one fold assignment and reporting that same estimate as final.

## 13.3 Regression metrics

Report:

- Mean absolute error in millimeters
- Root mean squared error
- \(R^2\)
- Spearman correlation
- Calibration intercept
- Calibration slope
- Error by target range
- Bland–Altman-style summaries when appropriate

## 13.4 Dilation metrics

If dilation classification is included:

- AUROC
- AUPRC
- Sensitivity
- Specificity
- Positive predictive value
- Negative predictive value
- Calibration curve
- Brier score

## 13.5 Confidence intervals

Use patient-level bootstrap confidence intervals on the pooled out-of-fold predictions.

For comparing two models, bootstrap the paired difference:

```text
metric(EHR+CXR+ECG) - metric(EHR+CXR)
```

## 13.6 Subgroups

Report at least:

- Sex
- Age group
- Body-size group
- Hypertension status
- Valve disease
- Normal versus enlarged diameter
- ECG–echo interval
- ECG quality
- Complete versus incomplete lead set

Subgroup results will have wide uncertainty. They should be labeled exploratory.

---

## 14. Avoiding common failure modes

### Failure mode 1: Comparing encoders with different folds

**Fix:** Persist one patient-level split file and use it for every experiment.

### Failure mode 2: Fitting PCA before cross-validation

**Fix:** Fit PCA on the training fold only.

### Failure mode 3: Using all patients for feature normalization

**Fix:** Fit all scalers on the training fold only.

### Failure mode 4: Selecting the closest ECG using future information inconsistently

**Fix:** Define the matching rule before modeling and apply it identically to every patient.

### Failure mode 5: Full encoder fine-tuning on 522 patients

**Fix:** Freeze first, then consider final-block or adapter tuning only after a stable frozen benchmark.

### Failure mode 6: Discarding amplitude through normalization

**Fix:** Retain raw-voltage engineered features and test them alongside normalized embeddings.

### Failure mode 7: Letting EHR shortcuts dominate

**Fix:** Evaluate residual prediction and incremental performance.

### Failure mode 8: Excessive hyperparameter search

**Fix:** Use a small prespecified grid and nested validation.

### Failure mode 9: Calling a model better based on one random split

**Fix:** Use repeated grouped cross-validation and paired confidence intervals.

### Failure mode 10: Treating root and ascending aorta as identical

**Fix:** Use separate heads, metrics, and subgroup analyses.

---

## 15. Concrete implementation work packages for coding agents

## Work package A: data audit

Deliverables:

```text
ecg_inventory.parquet
echo_inventory.parquet
cxr_inventory.parquet
ehr_inventory.parquet
paired_cohort_summary.md
```

Tasks:

- Count complete multimodal patients.
- Count ECG–echo patients without CXR.
- Count CXR–echo patients without ECG.
- Measure ECG–echo time intervals.
- Detect repeated patients and repeated studies.
- Audit lead order, sampling rates, duration, and units.

## Work package B: canonical ECG loader

Deliverables:

```text
src/ecg/io.py
src/ecg/canonicalize.py
src/ecg/qc.py
tests/test_ecg_io.py
```

Requirements:

- Return calibrated millivolt waveform.
- Return canonical lead order.
- Support variable source sampling rates.
- Preserve source metadata.
- Produce reproducible quality flags.

## Work package C: model-specific preprocessors

Deliverables:

```text
src/ecg/preprocess/pclr.py
src/ecg/preprocess/xecg.py
src/ecg/preprocess/ecg_cpc.py
src/ecg/preprocess/ecgfounder.py
src/ecg/preprocess/ecg_fm.py
src/ecg/preprocess/ecg_jepa.py
```

Each preprocessor must expose:

```python
def preprocess(
    waveform_mv,
    sampling_rate,
    lead_names,
    metadata
):
    ...
```

Each module must document:

- Expected sampling rate
- Expected duration
- Lead order
- Filtering
- Normalization
- Padding/cropping
- Output tensor shape

## Work package D: encoder wrappers

Deliverables:

```text
src/ecg/encoders/base.py
src/ecg/encoders/pclr.py
src/ecg/encoders/xecg.py
src/ecg/encoders/ecg_cpc.py
src/ecg/encoders/ecgfounder.py
src/ecg/encoders/ecg_fm.py
src/ecg/encoders/ecg_jepa.py
```

Required interface:

```python
class ECGEncoder:
    name: str
    output_dim: int

    def load_checkpoint(self, checkpoint_path: str) -> None:
        ...

    def encode(self, batch):
        # Return pooled embeddings and optional token embeddings.
        ...
```

## Work package E: offline embedding extraction

Deliverables:

```text
scripts/extract_ecg_embeddings.py
configs/embedding_extraction/*.yaml
artifacts/ecg_embeddings/
```

The script must:

- Run in inference mode
- Use deterministic preprocessing
- Save checkpoint hash
- Save preprocessing version
- Save patient and ECG identifiers
- Save quality variables
- Support restart/resume
- Avoid recomputing existing valid embeddings

## Work package F: unimodal benchmark

Deliverables:

```text
scripts/run_ecg_unimodal_benchmark.py
results/ecg_encoder_comparison.csv
results/ecg_encoder_comparison.md
```

Models:

- Ridge
- Elastic net
- Small MLP

Targets:

- Root diameter
- Ascending-aorta diameter

Metrics:

- MAE
- RMSE
- \(R^2\)
- Spearman correlation

## Work package G: fixed late-fusion benchmark

Deliverables:

```text
scripts/run_multimodal_encoder_ablation.py
results/multimodal_encoder_ablation.csv
```

Rules:

- Keep CXR and EHR branches fixed.
- Change only the ECG encoder.
- Use identical folds.
- Record parameter counts.
- Record fusion input dimension.
- Record training time and GPU memory.

## Work package H: residual-information analysis

Deliverables:

```text
scripts/run_residual_ecg_analysis.py
results/residual_ecg_results.csv
```

Baselines:

- EHR only
- EHR+CXR

Evaluate how well each ECG encoder predicts the cross-fitted residual.

## Work package I: local self-supervised adaptation

Deliverables:

```text
scripts/adapt_ecg_encoder_ssl.py
configs/ssl_adaptation/*.yaml
checkpoints/local_adaptation/
```

Start with the top one or two frozen encoders only.

## Work package J: experiment report

The final report should contain:

- Cohort flow diagram
- Data availability by modality
- Encoder accessibility table
- Preprocessing table
- Unimodal results
- Multimodal incremental results
- Paired confidence intervals
- Subgroup results
- Failure analysis
- Recommendation for the final ECG encoder

---

## 16. Minimal experiment configuration

```yaml
project:
  target_names:
    - aortic_root_diameter_mm
    - ascending_aorta_diameter_mm

splitting:
  strategy: grouped_kfold
  group_column: patient_id
  n_splits: 5
  repeats: 5
  seed_list: [11, 23, 37, 51, 79]

ecg:
  encoders:
    - pclr
    - xecg
    - ecg_cpc
    - ecgfounder
  freeze_encoder: true
  pooling:
    - model_default
  bottleneck_dims:
    - 32
    - 64

heads:
  ridge_alphas: [0.1, 1.0, 10.0, 100.0]
  mlp_hidden_dim: 64
  mlp_dropout: 0.3
  weight_decay: 0.001

fusion:
  type:
    - prediction_stacking
    - latent_concatenation
  hidden_dim: 64
  dropout: 0.4

training:
  loss: huber
  batch_size: 32
  max_epochs: 300
  early_stopping_patience: 30

evaluation:
  bootstrap_samples: 2000
  primary_metric: mae
  secondary_metrics:
    - rmse
    - r2
    - spearman
```

The grid should remain small. Do not expand it without nested validation.

---

## 17. Final recommendation

### Best first model for accessibility

**xECG** is the best first model to try when the objective is to move quickly from PCLR to a modern, downloadable encoder.

Reasons:

- Public Hugging Face weights
- Official code
- Pooled representation
- Regression-compatible downstream class
- Straightforward frozen-feature workflow
- No need to train the full encoder on 522 cases

### Best second model for scientific relevance

**ECG-CPC** should be the next model.

Reasons:

- Public compact checkpoint
- Official benchmark implementation
- Strong evidence on cardiac-structure and regression-oriented tasks
- Small storage footprint
- Suitable for frozen extraction and small nonlinear heads

### Best supervised-pretraining comparison

**ECGFounder** should be included as the main supervised foundation-model comparison.

### Model to defer

**ECG-FM** is accessible but should be deferred until the simpler pipelines are complete because of its size and Fairseq Signals dependency.

### Scientifically attractive but less convenient

**ECG-JEPA** remains worth testing, particularly given the small labeled cohort. Its checkpoint distribution is currently less convenient for automated cluster workflows than xECG, ECG-CPC, ECGFounder, or ECG-FM.

### Most important strategy beyond model choice

The highest-value improvement may not be replacing PCLR. It may be:

1. Use the large local ECG pool for self-supervised domain adaptation.
2. Use all available ECG–echo pairs for supervised diameter training.
3. Freeze the resulting ECG encoder.
4. Use the 522 complete patients only for the final multimodal fusion stage.

---

## 18. Verified resources

### xECG

- https://huggingface.co/riccardolunelli/xECG_base_model_v1
- https://github.com/dlaskalab/bench-xecg
- https://arxiv.org/abs/2509.10151

### ECG-CPC and benchmark

- https://github.com/AI4HealthUOL/ecg-fm-benchmarking
- https://figshare.com/articles/dataset/ECG-CPC_Checkpoint_zip/30192604
- https://figshare.com/ndownloader/files/58173919
- https://arxiv.org/abs/2509.25095
- https://openreview.net/forum?id=xXRqWpt3Xr

### ECGFounder

- https://huggingface.co/PKUDigitalHealth/ECGFounder
- https://github.com/PKUDigitalHealth/ECGFounder
- https://arxiv.org/abs/2410.04133

### ECG-FM

- https://huggingface.co/wanglab/ecg-fm
- https://github.com/bowang-lab/ECG-FM
- https://arxiv.org/abs/2408.05178

### ECG-JEPA

- https://github.com/kweimann/ECG-JEPA
- https://drive.google.com/file/d/1gMOT4xjQQg0GZkY1iE6NuDzua4ALw00l/view
- https://arxiv.org/abs/2410.13867

### PCLR

- https://github.com/mit-ccrg/PCLR
- https://github.com/broadinstitute/ml4h/tree/master/model_zoo/PCLR
- https://arxiv.org/abs/2104.04569

---

## 19. Reproducibility note

Public checkpoint availability and repositories were rechecked on 2026-07-16. Before freezing the production environment, record:

- Git commit SHA
- Model-repository revision
- Checkpoint filename
- Checkpoint SHA-256 hash
- Python version
- CUDA version
- PyTorch version
- Complete environment lockfile
- Preprocessing configuration
- Cohort split file checksum

These records are essential because repository defaults and model files can change over time.
