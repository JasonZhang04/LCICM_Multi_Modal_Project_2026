# Progress update — evaluation repairs, what survived, and where the multimodal question actually stands

**Companion to `PROGRESS_UPDATE_2026-08-30.md`.** Reporting period: 14 September 2026 (one intensive session). Written to be readable without a machine-learning background; technical terms are explained the first time they appear, and a glossary is at the end of Part 1.

This document has two parts:

- **Part 1 — What changed today and what the results are.** A second reviewer (Codex) audited the project and raised fourteen issues. I re-verified each major one against the actual code and data before changing anything, fixed the ones that were real, and re-ran the pipeline. The short version: the model's *performance numbers survived* a much stricter evaluation, but the *story we were telling about them* did not survive in its original form.
- **Part 2 — Where the project should go next.** My assessment after grounding Codex's second document (`notes/Aortic_Multimodal_Design_Audit_Brief.md`) in what has already been tried, plus new diagnostics run today specifically to answer the question "is there complementary signal we are failing to harness?"

---

## Part 1 — What changed today, and the results

### 1.0 The one-paragraph summary

Before today, the headline was: *a multimodal model (CXR + ECG + EHR) predicts aortic diameter with R² ≈ 0.33 and detects ≥4 cm dilation with AUROC ≈ 0.84, beating simpler baselines.* After today, the numbers are essentially unchanged — but they now rest on an evaluation that would survive peer review, and we know something we did not know before: **fine-tuned CXR alone already achieves that AUROC.** Whether ECG and EHR add anything on top is the question Part 2 is about, and the answer is more nuanced (and more interesting) than a flat "no."

### 1.1 What "leakage" means, and why it mattered here

Almost every issue Codex raised is a form of **information leakage**: the model, or the process that selected the model, got to see something about the test cases that it would not have in real life. Leakage does not make a model *wrong*; it makes its measured performance *optimistic* by an unknown amount. When a paper's claims rest on differences of ~0.01 (as ours do), an unknown optimism of ~0.01 is fatal, because you can no longer tell whether a "gain" is real.

The standard defence is **cross-validation (CV)**: split the patients into (say) five groups called **folds**; train on four, test on the fifth; rotate so every patient is tested exactly once while being held out of training. Predictions made this way are called **out-of-fold (OOF)** predictions. Our project already did this, with the important refinement that folds are **patient-grouped** — every episode of a patient sits in one fold, so a patient can never be both trained on and tested.

### 1.2 The defects that were real, in plain language

I checked each of Codex's major findings myself before acting. **All of them reproduced exactly.** This table is the verification log; the sections after it explain each one.

| Codex finding | Verified? | What I measured |
|---|---|---|
| A1 — the model stack is "not nested" | Yes | Traced the code path; wrote a test that catches it |
| A2 — ECG features from different folds are incompatible | Yes, worse than reported | A classifier recovers which fold produced each ECG feature vector with **100.0%** accuracy (chance = 20%) |
| A3 — the "untouched" holdout had been used in development | Yes | 6,220 root / 5,632 ascending "holdout" episodes appear in the development predictions |
| A4 — the "temporal" split was not temporal | Yes | 12,863 of 20,682 episodes occur in a different year from the patient's anchor year |
| A5 — some CXRs were taken *after* the echo they predict | Yes | 1,931 image rows across 1,572 episodes, by up to 22.7 hours |
| A6 — the holdout evaluated a different model than the one selected | Yes | It averaged four numbers; the selected model is a fitted ridge |

#### A1 — "Nested" cross-validation

**The defect.** Our model is a *stack*: a first-stage model (e.g. the CXR model) produces a predicted diameter, and a second-stage model (a **ridge regression** — a linear formula with a penalty that keeps the weights small) combines those first-stage predictions with EHR and ECG. The first-stage predictions were OOF, which sounds safe. But the second-stage model was then cross-validated *on top of one shared table of those predictions*. Trace what that means for test fold 0: the second stage trains on rows from folds 1–4, and each of those rows' CXR feature came from a first-stage model that was trained on folds {0,2,3,4}, {0,1,3,4}, … — every one of which *includes fold 0's labels*. So fold 0's answers reached the second-stage training data through the back door.

An analogy: a student is examined on questions from set A. Their study notes were written by a tutor who had read the answer key to set A. The student never saw the key — but the exam is still not clean.

**"Nested" is the fix.** For each test fold, remove those patients *first*, then do everything — including building the first-stage predictions — using only the remaining patients. Only at the very end are the removed patients predicted. Every fitted component can then be traced to training patients only. I built this as a reusable harness (`src/multimodal_aorta/training/nested.py`).

**The acceptance test.** The decisive check is a *perturbation test*: scramble the labels of the test fold, rerun, and require that the test fold's predictions come out *bit-identical*. If any label information leaked into anything that was fitted, the predictions move. I wrote this test, and — crucially — also wrote a guard-rail test that runs the *old* design through the same probe. The probe catches the old design and passes the new one. Without that guard rail, a passing test would prove nothing.

**A second, smaller defect in the same code:** the old stack tuned its penalty strength using folds that split episodes rather than patients, so a patient's repeat echoes could sit on both sides of that inner split. Fixed in the harness (all inner splits are patient-grouped).

#### A2 — The ECG feature matrix mixed five incompatible "languages"

The ECG branch is a neural network trained from scratch, five times (once per fold). Each network produces a 256-number internal summary of the ECG (an **embedding**). The five embeddings were written into one table and the fusion model was trained on that table as if all 256 columns meant the same thing in every row. They do not: column 17 of network A has no relationship to column 17 of network B. Each independently trained network invents its own internal coordinate system.

An analogy: five translators independently invent their own shorthand, and their notes are shuffled together. Each set is internally consistent; the mixture is gibberish.

The measurement: a simple classifier can tell *which network* produced a row with **100% accuracy** on patients it never saw. A valid feature table would give ~20% (chance). I turned this into a permanent regression test that will flip from "known-failing" to "passing" once the ECG branch is rebuilt on a common basis (Part 2 explains how).

**Why this matters for the multimodal question:** the +0.007 R² "ECG increment" reported on 30 August was measured through this broken table. Part 2 shows what happens when the ECG's contribution is measured cleanly.

#### A3 and A4 — The "temporal holdout" was neither temporal nor a holdout

A **holdout** (or **quarantine**) is a set of patients sealed away before any modelling, evaluated exactly once at the end, so the final number cannot have been influenced by choices made while looking at it. A **temporal** holdout is one drawn from a later era, to show the model generalises forward in time.

*Not a holdout:* the script that defined it only wrote a list; the training scripts never read that list unless a flag was set. So all 6,220 / 5,632 of those episodes were used throughout development. Those patients are "burned" — they cannot be described as untouched, ever.

*Not temporal:* MIMIC protects privacy by shifting each patient's dates by a random offset, and reports only a 3-year **anchor group** (e.g. "2014–2016") for one anchor year. The old split treated that group as the era of every episode. But an echo five years after a patient's anchor year happened five years later than the anchor group. Doing the arithmetic correctly, **5,196 "training" episodes are definitely from 2014 or later** — the era supposedly held out.

**The fix** (`scripts/build_temporal_holdout_v2.py`): compute each episode's *implied* year interval correctly, assign whole patients, and write three honest labels: `develop` (14,192 episodes), `quarantine` (1,072 episodes from **968 patients that have never been used**, now locked), and `exploratory_era` (the old split, kept only so it can be described honestly). We chose cutoff 2016 with no guard band after seeing that a strict 1-year guard band would have discarded 72% of the data — MIMIC's 3-year intervals make strict temporal ordering very expensive. The quarantine is small (40 root / 38 ascending positives): it supports a descriptive check, not fine inference.

#### A5 — Some "pre-echo" CXRs were taken after the echo

The cohort builder read only the CXR's *date*, and compared it to the echo's *date*. A CXR at 4 pm on the day of a 10 am echo therefore counted as "before." Using the full timestamps (which exist and parse for every image), **1,931 image rows in 1,572 episodes violate the ordering**, by up to 22.7 hours. Fixed in the builder; a rebuilt cohort has **zero** violations at a cost of 2.3% of episodes. The rebuilt cohort is in a separate directory and is *not yet promoted* — see §1.5 for why that is deliberate.

#### A6 — The holdout run evaluated a different model

The script that reported "holdout" performance did not run the selected model. It took an equal average of four predicted diameters. That is a legitimate *baseline*, but it is not validation of the ridge model we selected, and averaging is not equivalent to a fitted combination just because the inputs are correlated. The script now labels its output honestly and saves per-patient predictions (it previously kept only aggregates).

### 1.3 The three results that matter

#### Result 1 — The leak was real but nearly inert; the performance numbers survived

Same folds, same episodes, old (leaky) construction versus the properly nested one:

| | Old (global-OOF) | **Nested** | Nested, refittable components only |
|---|---:|---:|---:|
| Root R² | 0.331 | **0.332** [0.322, 0.344] | 0.326 |
| Ascending R² | 0.334 | **0.337** [0.325, 0.350] | 0.318 |
| Root AUROC ≥4 cm | 0.838 | **0.838** [0.826, 0.851] | 0.830 |
| Ascending AUROC ≥4 cm | 0.830 | **0.831** [0.816, 0.846] | 0.816 |

(*R²* = fraction of the variation in diameter the model explains; *AUROC* = probability that a random ≥4 cm case is ranked above a random <4 cm case, 0.5 = coin flip. Square brackets are 95% confidence intervals from a **patient-clustered bootstrap** — resampling *patients*, not episodes, so repeat visits are not treated as independent evidence.)

The right column is the run with *no* GPU-trained components (those cannot be refit on CPU inside the nesting), so it is the only run for which every component is provably clean. The nested harness records this honestly: any run using a precomputed GPU branch is flagged `exact_nesting: false` in its output and cannot be reported as fully clean by accident.

#### Result 2 — The multimodal model does not beat fine-tuned CXR alone on discrimination (A7)

This is the finding that changes the paper. Codex pointed out that we had been comparing against weak baselines (EHR alone; the older frozen-CXR model). The right comparator is our *strongest* single-modality arm, the fine-tuned CXR. On identical episodes and identical nested folds, with 2,000 paired patient-clustered bootstrap replicates:

| Site | Arm | R² | AUROC ≥4 cm |
|---|---|---:|---:|
| Root | fine-tuned CXR alone | 0.319 | **0.840** |
| Root | full multimodal | 0.332 | 0.838 |
| Asc | fine-tuned CXR alone | 0.333 | **0.833** |
| Asc | full multimodal | 0.337 | 0.831 |

| Site | Full − CXR: ΔR² | Full − CXR: ΔAUROC |
|---|---:|---:|
| Root | **+0.013** [+0.009, +0.017] | −0.001 [−0.006, +0.003] |
| Asc | +0.004 [−0.000, +0.008] | −0.002 [−0.006, +0.002] |

Read plainly: adding ECG and EHR *as currently built* buys a small root R² gain and **no discrimination gain** at either site. The frozen-CXR branch we also carry is *worse* than fine-tuned CXR on root AUROC (−0.011 [−0.019, −0.003]). **Part 2 argues this is a null for the current design, not for the hypothesis** — and shows why.

#### Result 3 — RAD-DINO's input normalisation was wrong, and fixing it changed nothing

**What normalisation is.** Before an image enters a neural network, each pixel value is rescaled: subtract a mean and divide by a standard deviation, so inputs sit in the range the network was trained on. RAD-DINO (the CXR encoder) was trained on greyscale X-rays with one mean/std applied equally to its three colour channels. Our pipeline applied *ImageNet* statistics — three *different* means and stds designed for colour photographs. Because a CXR's three channels are identical copies, dividing them by three different numbers injects an artificial colour gradient. I measured it: the three channel means came out at 0.05 / 0.18 / 0.40 (a spread of **0.35**) under the old pipeline versus **0.00** under the correct one.

**The controlled experiment.** I added a switchable preprocessing mode, verified the old mode reproduces the existing cache bit-for-bit (so nothing prior is invalidated), built a corrected image cache (60,090 images, identical set), and re-ran fine-tuning with *only* normalisation changed. I wrote down the interpretation rule *before* seeing the result (a difference under ±0.01 R² is inconclusive at one seed).

| | legacy | corrected | paired ΔR² | paired ΔAUROC |
|---|---:|---:|---:|---:|
| Root | 0.319 | 0.319 | **0.000** [−0.000, 0.001] | 0.000 [−0.000, 0.001] |
| Asc | 0.333 | 0.333 | **0.000** [−0.000, 0.001] | 0.000 [−0.000, 0.001] |

A clean null. The network's first layer is a learned linear map over the three channels, so a fixed per-channel rescaling is trivially absorbed — especially with the last two blocks fine-tuned. **This does not mean preprocessing is irrelevant:** only normalisation was tested. Resolution and aspect ratio (we squash every image to a 224×224 square; the encoder was trained at 518 on the shortest edge) are the *geometric* part of the mismatch and remain untested. For an aortic-silhouette measurement, aspect ratio is the plausible lever; channel scaling was not.

### 1.4 Two bugs found that the review had not flagged

- **Horizontal flip augmentation was on** (`cxr_aug_hflip_p = 0.5`). *Augmentation* means randomly perturbing training images (rotating, brightening) so the model does not memorise them. Mirroring a chest X-ray puts the heart and aortic arch on the patient's right, which is anatomically wrong and destroys the very left-mediastinal silhouette we measure. Saved results were unaffected only because fine-tuning happens to read a pre-computed, un-augmented cache — any future training run would have been corrupted. Now off.
- **The fold file could be silently overwritten.** The script that builds fold assignments writes over `episode_fold_assignments.csv` in place. Every saved prediction is joined to folds by episode ID with no record of *which* fold definition — so regenerating that file (as promoting the corrected cohort would require) would have made every existing result quietly non-comparable. I added a **guard**: the script now refuses to overwrite unless explicitly forced, and archives the old file with a timestamp when it is.

### 1.5 Things deliberately *not* done, and why

- **The timestamp-corrected cohort is built but not promoted.** Promoting it means regenerating folds, image features, EHR features, ECG binding, and every model — a whole-pipeline pass. It should be done *once*, after the preprocessing and fusion questions in Part 2 are settled, not twice. A runbook (`notes/cohort_v2_promotion_runbook.md`) records the safe order; a half-promoted state is worse than either consistent state.
- **The ECG branch and the fine-tuned CXR were not retrained inside the nesting.** Both are GPU jobs; the honest flag in the results covers the gap until they are.

### 1.6 Files added or changed today

| File | Purpose |
|---|---|
| `src/multimodal_aorta/training/nested.py` | Nested CV harness with ancestry guarantees and honesty flags |
| `src/multimodal_aorta/tests/test_nested_leakage.py` | Perturbation acceptance test + guard rail (4/4 pass) |
| `src/multimodal_aorta/tests/test_embedding_basis.py` | ECG basis probe (calibrated; currently xfail by design) |
| `src/multimodal_aorta/tests/test_preprocessing_and_timing.py` | Locks A5/A9/hflip invariants (6/6 pass) |
| `scripts/train_final_model_nested.py`, `slurm_final_model_nested.sh` | Nested rebuild of the final model + CXR-only comparator |
| `scripts/compare_a7_baselines.py` | Full vs fine-tuned CXR vs frozen CXR, paired |
| `scripts/compare_cxr_preproc.py` | Normalisation experiment, prespecified verdict |
| `scripts/build_temporal_holdout_v2.py` | Interval-aware split; old builder now refuses to run |
| `scripts/build_episode_cohort.py` | Full-timestamp ordering |
| `scripts/eval_supporting_episode.py` | Contiguous lag bins |
| `scripts/train_holdout_final.py` | Honest labelling + per-patient predictions |
| `scripts/extract_cxr_image_cache.py`, `train_cxr_finetune_episode.py`, `configs/default_config.py`, `data/preprocessing.py` | Mode-tagged preprocessing; hflip off |
| `scripts/build_episode_folds.py` | Overwrite guard |
| `scripts/diag_complementarity.py`, `diag_complementarity_followup.py` | Today's complementarity diagnostics (Part 2) |
| `notes/repairs_2026-09-14.md` | Full technical log with every number |

### 1.7 Glossary

- **Fold / cross-validation / OOF** — see §1.1. *Patient-grouped* means a patient is never split across folds.
- **Nested** — the test patients are removed before *any* model in the pipeline is fitted, including first-stage models that feed later stages.
- **Stack / late fusion / early fusion** — *late*: combine each modality's *predicted diameter*. *Early*: combine each modality's *raw features* before predicting. Our final model is a hybrid: CXR enters as a predicted diameter, EHR and ECG as features. **Full architecture walkthrough: §1.9.**
- **Base learner / second stage (fusion / meta-learner)** — the individual per-modality models, and the small model that combines their outputs. See §1.9.
- **Ridge** — linear regression with a penalty that shrinks weights toward zero; the penalty strength is a tuning knob chosen by inner cross-validation.
- **Embedding** — a neural network's internal numeric summary of an input (here 256 numbers per ECG, 768 per CXR).
- **PCA** — compresses an embedding to fewer numbers (we use 128 for CXR, 32 for ECG) keeping the directions of largest variation.
- **Normalisation** — rescaling inputs to the range a network expects (§1.3, Result 3).
- **Augmentation / hflip** — random training-time image perturbations; hflip = horizontal mirror.
- **Holdout / quarantine / temporal** — §1.2, A3/A4.
- **Anchor year / anchor group** — MIMIC's privacy-preserving date shift; only a 3-year bucket is real.
- **Bootstrap CI (patient-clustered)** — confidence interval from resampling patients with replacement; *paired* means both arms are scored on the same resample, so their *difference* gets a proper interval.
- **AUROC / R² / AUPRC / MAE** — ranking quality for a binary threshold / explained variance / precision-recall area (prevalence-sensitive, stricter for rare positives) / mean absolute error in cm.
- **Guard** — a check in a script that refuses a destructive action unless explicitly forced.
- **Residual** — what the model got wrong: true value minus predicted value. "Does ECG explain the CXR residual?" asks whether ECG knows something about the cases where CXR erred.

---

### 1.8 Training curves — the tool you asked for, and what it immediately revealed

**What a training curve is.** A neural network is trained in *steps*; at each step it sees a batch of examples and nudges its weights. Every few hundred steps we pause and measure the error on a *validation* set the model is not training on. Plotting that validation error against the step count is the training curve. It tells you three things you cannot get from the final score: whether the model is still learning (curve falling), has stopped (flat), or has started memorising the training data (curve rising again — **overfitting**). We were not recording this for any model.

**What was added.**
- `src/multimodal_aorta/training/curves.py` — a small logger now wired into both GPU trainers (fine-tuned CXR, ECG waveform). Every evaluation interval it writes one row (fold, step, mean train loss over the window, validation loss, best-so-far, learning rate, elapsed) to `outputs/<run>/training_curves.csv`. Future runs get train *and* validation curves.
- `scripts/plot_training_curves.py` — draws one panel per fold for every run it can find, overlays named runs for comparison, and writes a summary table. It can also **parse the old SLURM logs**, so every past run is viewable retroactively (validation loss only — train loss was never logged). Output in `figures/training_curves/`: one PNG + CSV per run, `compare__*.png` overlays, `summary.md`.

**What the curves show — this is important.**

*Fine-tuned CXR* (`compare__cxr_finetune_episode__vs__cxr_finetune_episode_ckpt_norm.png`):

| fold | best validation loss | step at best | stopped at |
|---|---:|---:|---:|
| 0 | 0.745 | 1,200 | 3,000 |
| 1 | 0.730 | 900 | 2,700 |
| 2 | 0.705 | 1,200 | 3,000 |
| 3 | 0.720 | 600 | 2,400 |
| 4 | 0.756 | 1,200 | 3,000 |

With a batch of 32, the best point at 600–1,200 steps corresponds to **19,000–38,000 images seen — less than one pass over the ~46,000 training images in a fold.** After that the validation loss *rises* by 10–15% until early stopping halts the run. Two conclusions:

1. **The CXR arm is overfitting almost immediately, not saturating.** The validation loss at the very first checkpoint (step 300) is already within ~2–3% of the best it ever reaches; fine-tuning the last two blocks adds only that much before the model begins memorising. This is consistent with the training setup: there is effectively **no augmentation** (the trainer reads a fixed pre-computed image cache and adds a little Gaussian noise), a single learning rate for backbone and head, and images are sampled uniformly so 90-image episodes dominate. Each of those is a standard, low-risk fix — and this is the most evidence-backed "stage we can revise to improve performance" in the project right now, because the CXR arm carries everything else.
2. **The two normalisation runs are superimposed** at every step of every fold — the null result in §1.3 was not a coincidence of the final number.

*ECG waveform* (`ecg_waveform_episode_broad.png`): the opposite picture — validation loss falls steadily from ~0.9 to ~0.78–0.83 over 5,000–10,000 steps, but with large spikes (fold 1 jumps from 0.83 to 0.94 and back). The model is still learning when it stops and the learning rate is not decayed after warm-up, so the noise never settles. The small-cohort ECG run varies widely across folds (best 0.76 to 0.91). Both suggest the ECG arm is data-limited and would benefit from more data (the broad cohort already helps) and a decaying learning-rate schedule.

*Reading the y-axis:* loss is mean-squared error on **standardised** diameter, so roughly `1 − R²` at the image level; 0.72 ≈ R² 0.28 per image, which averages up to ~0.32 per episode.

### 1.9 What "the stack" actually is — the current architecture, end to end

"Stack" is jargon for a specific, mundane idea: **instead of one model that reads
everything at once, we train several separate models and then train one more small
model whose only job is to combine their answers.** The word comes from "stacked
generalisation" (Wolpert, 1992). The combiner is called the *second stage*, *meta-learner*,
or *fusion model*; the models it combines are the *base learners*.

An analogy: three specialists each independently estimate the patient's aortic diameter —
a radiologist reading the chest X-ray, a cardiologist reading the ECG, a clinician reading
the chart. A stack is the resident who hears all three estimates and produces a final
number, having learned from past cases how much to trust each specialist. Crucially the
resident never sees the X-ray — only the radiologist's *number*.

#### The pieces, concretely

There is one complete model per site (root and ascending), trained identically.

**Base learner 1 — multi-instance frozen CXR** (`cxr_multiinstance`, CPU, refit inside the
harness). RAD-DINO (a vision transformer pretrained on chest X-rays) is run *frozen* over
each X-ray and emits three 768-number embeddings per image — a whole-image summary (`cls`)
plus two region-pooled summaries over the aorta and heart (`aortapool`, `heartpool`). Each
block is compressed by PCA to 128 numbers, and 17 hand-measured geometry features are
appended (cardiothoracic ratio, aortic knob position, mediastinal ratios, …). A gradient-
boosted tree regressor predicts diameter *per image*; because an episode can have several
X-rays, the per-image predictions are averaged up to one number per episode (each image
weighted 1/n so an episode with 5 films does not dominate training). **Output: 1 number.**

**Base learner 2 — EHR** (`ehr_hgb`, CPU, refit inside the harness). A gradient-boosted tree
on 12 tabular variables: age, sex, height, weight, BMI, BSA, systolic/diastolic BP, and four
missingness indicators. **Output: 1 number.**

**Base learner 3 — fine-tuned CXR** (`cxr_finetuned`, GPU, *precomputed*). The same RAD-DINO,
but the last 2 of its 12 transformer blocks are unfrozen and trained end-to-end with a linear
2-output head predicting [root, asc] directly from pixels. This is the strongest single arm
and the one whose training curve shows the overfitting problem (§1.8). **Output: 1 number.**

**Side inputs, not base learners.** Two blocks of raw features go *straight into the
combiner* without first being turned into a predicted diameter:
- the 12 standardised EHR variables (so the combiner can adjust for body size directly);
- the ECG: a 1-D ResNet (12 leads × 5000 samples → residual conv stages 64/128/256 channels
  → a self-attention block → average pooling) trained on [root, asc, heart rate]. Its 256-number
  penultimate embedding is compressed by PCA to 32, **plus one binary "ECG present" flag**
  so the combiner can tell a missing ECG from an average one.

**The combiner — ridge regression.** All of the above are concatenated into one feature row:

```
[ cxr_multiinstance ] [ ehr_hgb ] [ cxr_finetuned ]  ← 3 base predictions (cm)
[ 12 standardised EHR variables ]                     ← raw side features
[ 32 ECG PCA components ] [ ECG-present flag ]        ← raw side features
                       ↓
        ridge regression (one linear formula)
                       ↓
              final predicted diameter (cm)
```

Ridge is ordinary linear regression with a penalty that shrinks the coefficients toward
zero; the penalty strength (`alpha`) is picked by patient-grouped inner cross-validation
from {0.3, 1, 3, 10, 30, 100}. The chosen values in the current run were alpha=3 for the
root (fold 0 picked 100) and alpha=100 for every ascending fold — heavy shrinkage, which is
itself a hint that most of the extra columns are carrying little independent information.

**So the honest one-line description of the architecture is: *hybrid late/early fusion —
three model predictions plus two raw feature blocks, combined by a penalised linear model,
wrapped in nested patient-grouped cross-validation.*** It is deliberately simple. The
complexity lives in the base learners and in the evaluation discipline, not in the fusion.

#### Why a stack rather than one joint network

Three reasons, all still valid:
1. **Missing modalities.** Only a subset of episodes have an ECG. A stack handles this with
   a zero-fill plus an availability flag; a jointly trained network needs modality dropout and
   a masking scheme to avoid collapsing when an input is absent.
2. **Sample size.** ~20,000 episodes with ~900 positives is far too few to train a joint
   multimodal transformer from scratch without it memorising.
3. **Interpretability for the paper.** Each arm's standalone performance is directly readable,
   which is exactly what an incremental-value claim requires.

#### Where the defects sat in this picture

This diagram also locates everything in Part 1. **A1** was in the *combiner*: it was fitted on
base predictions built from one global table, so test-fold labels reached it through the base
learners. **A2** was in the *ECG side input*: the 256-number embeddings came from five
separately trained networks (one per fold), so column 7 meant something different depending
on which fold a row came from — like pooling five questionnaires that all have a question 7
but ask different questions. **A6** was a reporting bug where the holdout script averaged the
four arms equally instead of applying the fitted ridge.

#### One caveat that is still open

`exact_nesting: false` in `results.json`. The two GPU arms (fine-tuned CXR, ECG) cannot be
refit inside every inner fold on CPU within the compute budget, so they enter as precomputed
per-fold-OOF vectors. They are leakage-free with respect to the *outer* test patients, but
their inner-OOF and outer-test values come from the same artifact, so the nesting is not
exact for those two columns. The run records this explicitly rather than claiming clean
nesting. Setting `CXR_FT=0 ECG=0` produces a fully exact run of the CPU-refittable part.

---

## Part 2 — Where the project should go next

### 2.0 The question, stated the right way

Codex's second document puts the central question better than our own reports did:

> After the model has already used the strongest modality (CXR), do ECG and EHR explain additional held-out variation, or correct clinically important CXR errors?

Two things about that framing matter. It does **not** require every modality to contribute equally — a model that is 95% CXR-driven but rescues a specific subset of cases is a legitimate multimodal result. And it is answerable with diagnostics we can run today, before building anything new. Your instinct — that ECG and EHR *should* carry information a CXR cannot, and that the current design may not be harnessing it — is a hypothesis, not a conclusion, and the Part 1 result (A7) was a null *for one design*, not for the hypothesis. So today I ran the diagnostics the hypothesis needs, on existing predictions, before writing a single recommendation.

### 2.1 What the new diagnostics say

All of this uses the fine-tuned CXR as the base, the ECG waveform model's *predicted diameter* (a single number per episode, so the A2 basis problem does not apply), and the EHR model's predicted diameter, combined with a nested, patient-grouped ridge on the 17,986 root / 16,261 ascending episodes that have an ECG. Scripts: `diag_complementarity.py`, `diag_complementarity_followup.py`; outputs in `outputs/diag_complementarity/`.

**(a) Do the arms make *different* mistakes?** Only slightly. Their residuals correlate at 0.83 (CXR–ECG) and 0.91 (CXR–EHR) on the root. Part of that is unavoidable — every arm's residual contains the same echo measurement noise, which by itself would produce a correlation of roughly 0.55–0.6 — but 0.83 is well above that floor. The arms are largely seeing the same underlying thing (plausibly heart/body size).

**(b) How much of what CXR gets wrong can the others explain?**

| Explains the CXR residual (R²) | Root | Ascending |
|---|---:|---:|
| ECG alone | 0.006 [0.003, 0.009] | 0.008 [0.005, 0.011] |
| EHR alone | −0.001 | 0.002 |
| ECG (broad-cohort model) | 0.009 [0.005, 0.012] | 0.011 [0.007, 0.014] |
| CXR + ECG + EHR, main effects | **0.029** [0.023, 0.034] | **0.018 [0.013, 0.022]** |
| … + interaction terms (ECG×CXR, EHR×CXR, BSA×CXR) | 0.029 [0.024, 0.034] | 0.018 [0.013, 0.023] |
| Control: CXR's own prediction alone | **0.000** [−0.002, 0.001] | **-0.000 [-0.001, 0.000]** |
| Pre-index clinical-history flags alone | 0.000 | 0.001 [-0.002, 0.003] |
| CXR + history flags | 0.002 [−0.000, 0.005] | 0.001 [-0.002, 0.003] |


Three readings. First, once the CXR prediction is in the model the others explain ~3% of the residual (root) — the model is using the part of ECG/EHR that is *not* already in CXR. Second, the control row matters: the CXR's own prediction explains none of its residual, so this is not a recalibration artefact. Third, **interactions add nothing** (0.029 → 0.029): whatever complementary signal exists is *additive*. A "conditional" or gated fusion — where what the CXR features mean depends on the ECG or body size — has no support in these numbers. The CXR residual is also uncorrelated with age, sex, BSA, blood pressure, AP/PA view, and image lag (all |r| ≤ 0.06), so there is no simple covariate structure for a gate to exploit.

How big could this ever be? Echo measurement noise (σ ≈ 0.30 cm) accounts for roughly 44% of label variance and about 70% of the CXR residual variance; at most ~30% of the residual is explainable by *any* information. ECG+EHR explain ~3%, i.e. about a tenth of what is theoretically available.

**(c) The clean incremental-value table — what the paper should actually report.** Same episodes, paired patient-clustered bootstrap versus fine-tuned CXR alone:

| Root (n = 17,986; 815 ≥4 cm) | R² | AUROC | ΔR² vs CXR | ΔAUROC vs CXR |
|---|---:|---:|---:|---:|
| fine-tuned CXR | 0.314 | 0.835 | — | — |
| + EHR | 0.326 | 0.837 | +0.011 [0.008, 0.014] | +0.002 [−0.002, 0.005] |
| + ECG | 0.326 | 0.841 | +0.012 [0.009, 0.015] | **+0.006 [0.002, 0.009]** |
| + EHR + ECG | 0.334 | 0.841 | +0.020 [0.016, 0.024] | **+0.006 [0.002, 0.011]** |
| + EHR + ECG (broad-cohort ECG) | 0.337 | 0.844 | +0.023 [0.019, 0.027] | **+0.009 [0.004, 0.014]** |

| Ascending (n = 16,261; 729 ≥4 cm) | R² | AUROC | ΔR² vs CXR | ΔAUROC vs CXR |
|---|---:|---:|---:|---:|
| fine-tuned CXR | 0.326 | 0.826 | — | — |
| + EHR | 0.331 | 0.824 | +0.005 [0.003, 0.008] | −0.002 [−0.005, 0.000] |
| + ECG | 0.336 | 0.832 | +0.010 [0.007, 0.013] | **+0.006 [0.003, 0.009]** |
| + EHR + ECG | 0.338 | 0.830 | +0.012 [0.009, 0.015] | +0.004 [0.000, 0.007] |
| + EHR + ECG (broad-cohort ECG) | 0.341 | 0.832 | +0.015 [0.012, 0.019] | **+0.006 [0.003, 0.010]** |

This is a different picture from Part 1's A7 table, and the difference is diagnostic. **The ECG waveform adds a small but statistically clear discrimination increment on top of fine-tuned CXR at both sites** (+0.006 AUROC; +0.009 root with the broad-cohort ECG). The full model in Part 1 showed *none* — because its ECG entered through the 256-number embedding table with the incompatible-bases defect (A2), while here the ECG enters as one clean number. In other words: **the earlier null was partly a design artefact, and your intuition that the design was not harnessing the signal is correct.** EHR (age/sex/size/BP) adds R² on the root but no discrimination anywhere: the CXR already sees body size.

**(d) But does it change any decision?** At a fixed echo capacity — the clinically realistic operating point — no:

| Send the top 5% to echo | CXR alone sensitivity | Full stack sensitivity | Positives gained / lost |
|---|---:|---:|---:|
| Root | 26.4% | 26.6% (Δ +0.002 [−0.018, +0.025]) | 43 / 41 |
| Ascending | 30.7% | 30.6% (Δ −0.001 [−0.019, +0.010]) | 16 / 17 |

**AUPRC — which weights the top of the ranking — does not move at all**: root Δ 0.001 [-0.007, 0.010], ascending Δ -0.002 [-0.008, 0.005]. The AUROC gain comes from re-ordering cases in the *middle and bottom* of the ranking — among the ~10,000 episodes the CXR calls clearly normal, the stack lifts the few hidden positives (within-band ΔAUROC +0.04 at both sites; take that number with caution, since stratifying on the CXR's own output mechanically favours the other arms). It does not add positives at the top of the list, and it does not touch the systematic under-prediction of large aortas (mean error among true ≥4 cm cases: −0.72 cm for every arm). The gain is also somewhat larger in patients under 50 and in the lowest body-size tertile (root ΔAUROC +0.020 and +0.016, CIs exclude zero) — an interesting hypothesis (dilation in the young is more often valvular or genetic, where ECG may carry more) but not a finding yet.

**(e) Clinical history.** The strongest version of "EHR should contain different information" is diagnoses, not body size. I built pre-index inpatient diagnosis flags from MIMIC (only admissions *discharged before* the echo; 66% of episodes have at least one). They are strongly associated with the outcome in raw prevalence — a prior aortic-aneurysm code triples the ≥4 cm rate, bicuspid valve quadruples it, aortic regurgitation nearly doubles it — yet **on top of the fine-tuned CXR they add ≈ 0.002 residual R² on the root**. The CXR already "knows." (The prior-aneurysm flag is also the least interesting one clinically: it marks *already-diagnosed* disease, which a screening model should not lean on.) In the factorial, CXR + history versus CXR alone: root ΔR² 0.002 [-0.000, 0.003], ΔAUROC 0.001 [-0.001, 0.003], ΔAUPRC 0.003 [-0.003, 0.009]; ascending ΔR² 0.001 [-0.001, 0.002], ΔAUROC 0.005 [0.002, 0.007], ΔAUPRC 0.005 [-0.000, 0.010]. Adding history to CXR + EHR + ECG: root ΔAUROC 0.008 [0.003, 0.012] (versus 0.006 [0.002, 0.011] without), ascending 0.009 [0.005, 0.013]. Real but tiny — and the one flag that moves the CXR residual most is prior aneurysm (CXR under-predicts by a further 0.05 cm in those patients), i.e. known disease.

### 2.2 Codex's suggestions, triaged against what has actually been tried

Codex did not know the project's history. This table grounds each of its proposals; "done" means completed with the result noted, not merely attempted.

| Codex proposal | Status here | My assessment |
|---|---|---|
| Patient-grouped splits; patient-clustered bootstrap; OOF fusion inputs; pre-index windows | Done, and as of today properly nested | Agree; closed |
| Quantify ECG/CXR reuse across episodes; don't treat modality combinations as independent examples | Done (images pooled per episode; patient grouping covers reuse) | Closed |
| Weight episodes so heavy users don't dominate | Done in the frozen branch (1/n weights); **not** in fine-tuning (uniform image sampling) | Real gap — see Stage A |
| Label harmonisation (root vs sinus vs STJ), units, duplicates | Done: `sinus_diam` = root, medians within study, units verified against raw table | Closed |
| Retain PA/AP, pixel spacing, positioning metadata | Not used as features | Residual is uncorrelated with view (r = 0.01); low priority as a *fusion* input. Worth carrying as a covariate in CXR training |
| Train each branch on its maximum eligible cohort | Done (CXR on all 20,682; ECG on 18k or the 62k broad cohort) | Broad ECG is better — adopt as default |
| CXR: RAD-DINO + small adapter; compare full image vs mediastinal crop; avoid unrestricted fine-tuning | All done (frozen+HGB, ROI crops, last-2-block fine-tuning) | Fine-tuning won; but curves show it is under-regularised — Stage A |
| ECG: PCLR + adapter | Outdated: PCLR was superseded by a from-scratch waveform model that beats the ECG summary | The *idea* of a pretrained encoder is still right, for a different reason — it fixes A2 (Stage B) |
| EHR: regularised shallow model on strictly pre-index features | Done (HGB and logistic on 12 features) | Now extended with history flags: adds ≈ nothing |
| **Residual late fusion** (CXR base + shrunk correction) | Effectively what today's diagnostics implement | Adopt the *formulation* for the paper; numerically ≈ linear stacking |
| Compare against weighted average / linear stack / current fusion / CXR-only / pairwise | Done today (factorial) | This is the incremental-value table |
| Missing-modality masks, modality dropout, fallback rules | Mask added today; dropout only matters for joint training | Fine as is for a stack |
| Multitask continuous + threshold heads, monotone constraints | Root/asc multitask done; threshold heads not | Worth one small test on the CXR arm (Stage E); label flips (70% at 4.0 cm on repeat echo) cap what it can do |
| **Huber / Student-t loss for noisy labels** | Not done | **Do not adopt blindly.** Our clinically important errors are *under-predictions of large aortas*; a robust loss down-weights exactly those cases and would shrink the tail further |
| Class weighting / balanced sampling for rare thresholds | Not done | Cautious: ≥4 cm positives are the noisiest labels; oversampling amplifies noise. Test only with calibration reported |
| Bayesian shrinkage at the fusion layer | Ridge already shrinks; uncertainty-weighting untested | Defer; low expected value given additive, small signal |
| Cooperative learning, ρ tuned from zero | Done; ρ = 0 selected everywhere | Closed. (Codex's own caveat — "include zero as a serious candidate" — is what happened) |
| Cross-attention on patch × lead tokens, frozen encoders, parameter-matched control | 3-token version done (null); token-level not | Defer: no interaction signal at the scalar level, and the cheaper additive gain must be locked in first |
| Teacher–student distillation | Not done | Defer; useful for deployment when ECG is missing, not for discovering signal |
| Masked self-supervised pretraining | Not done | Reject for now — Codex itself ranks it last |
| Learning curves vs. unique patients | Not done | Cheap with the nested harness; tells us whether the CXR arm is data-limited. Stage A |
| Stratify incremental value by view, lag, target, CXR confidence | Done today | Uniform across view/lag; larger in young/small-bodied; artefact-prone by CXR band |

### 2.3 The recommended program, in order, and why each step is where it is

The ordering principle: **improve the base arms first, then re-measure the increment, then decide the architecture.** The diagnostics say the complementary signal is small and additive, so architecture is unlikely to be the lever; the training curves say the base arms are the lever.

**Stage A — Fix the CXR arm's training (highest value, cheapest, best-evidenced).**
The curves in §1.8 show best validation loss before one epoch, then steep overfitting. Every standard remedy is untried: real augmentation (small rotations, scale/translation, contrast — *no* horizontal flip), episode-balanced sampling (one image per episode per pass, or 1/n weights, matching the frozen branch), discriminative learning rates (backbone ≈ 1e‑5, head ≈ 1e‑3, instead of one 1e‑4), cosine decay, and possibly four unfrozen blocks under that lower rate. Success criterion: best validation later than one epoch and lower than 0.70, and paired improvement over the current CXR arm on the same folds. This also strengthens the comparator the paper hinges on. Run on `develop` only. Log curves.

**Stage B — Rebuild the ECG branch on a valid basis (fixes A2; the multimodal question hinges on it).**
Two routes, both inside the nested harness so the fusion sees one coordinate system:
1. *Frozen pretrained encoder* — a public structural-heart-disease ECG model (EchoNext's released weights) gives a single embedding space by construction and needs no GPU training; verify licence, lead order, sampling rate and any MIMIC overlap first.
2. *Per-outer-fold encoder used consistently* — train one waveform CNN per outer fold and use *its* embedding for both the inner-OOF fusion training and the outer test patients.
Also: adopt the broad-cohort training (already +0.003 AUROC downstream and +0.03 standalone R²), and add learning-rate decay — the ECG curves are still falling and noisy when they stop. Then test scalar vs embedding honestly; the earlier +0.007 embedding increment was measured through the defect and cannot be cited.

**Stage C — Re-run the incremental-value factorial with the improved arms, nested, on `develop`.**
Report, for CXR / +EHR / +ECG / +both: R², AUROC, AUPRC, sensitivity and PPV at a fixed top‑5% and top‑10% capacity, calibration, and paired CIs. Include the current 4-branch model as an arm so the paper can say explicitly that the frozen-CXR branch is redundant (it is worse than fine-tuned CXR on root AUROC, −0.011 [−0.019, −0.003]). The candidate final model is likely **fine-tuned CXR + broad ECG scalar (+ EHR for diameter)** — simpler than what we have.

**Stage D — Cohort promotion and validation.**
Promote the timestamp-corrected cohort (runbook in `notes/`), evaluate the locked model once on the 968-patient quarantine, and pursue an external cohort. The quarantine is too small for fine claims; external data is the only route to a transportability claim, and A7 makes transportability of the *CXR* arm the paper's central risk.

**Stage E — Two prespecified hypotheses, run only after A–C, each with a written success criterion.**
1. *Selective value in the young / small-bodied*: does ECG's increment concentrate where dilation is more often valvular or genetic? Test on `develop`, freeze the subgroup definition, confirm on quarantine. If it holds, it is a genuine "corrects a specific subset" claim.
2. *Threshold head on the CXR arm*: continuous + monotone ≥4/≥4.5 heads; adopt only if AUPRC and calibration both improve.

**Explicitly deprioritised:** token-level cross-attention, cooperative learning revisits, Bayesian fusion layers, distillation, new self-supervision, robust losses. Each is either already answered, unsupported by the additivity result, or aimed at a problem we do not have.

### 2.4 How I would frame the paper now

Neither "multimodal beats CXR" nor "multimodal adds nothing." Something closer to:

> A fine-tuned CXR foundation model estimates aortic root and ascending diameter at roughly 58% of the echo label-noise ceiling and detects ≥4 cm dilation with AUROC ≈ 0.84. The 12‑lead ECG waveform adds a small, statistically robust, *additive* increment (≈ +0.006–0.009 AUROC) that does not change the number of dilated aortas found at realistic echo capacity; routine EHR variables and pre-index diagnoses add no discrimination beyond CXR. An apparent absence of ECG value in an earlier design was traced to an invalid feature basis — a leakage-adjacent defect that other stacked multimodal pipelines may share.

That is an honest incremental-value study with a positive methodological lesson, and it survives review. Stages A–B are designed to test whether the ECG increment becomes clinically meaningful once the base arms are properly trained and the ECG basis is valid — the program is built to *answer* that, not to assume it either way.

**What would change my mind toward a stronger multimodal claim:** an ECG increment ≥ +0.02 AUROC after Stage B; a fixed-capacity sensitivity gain whose CI excludes zero; or a confirmed subgroup where the increment is several times the average. **What would close the question:** the increment stays ≈ +0.006 after both arms are strengthened.

### 2.5 Immediate next actions (this week)

1. Stage A run: augmentation + episode-balanced sampling + discriminative LR on the CXR fine-tuning, on `develop`, curves logged, paired against the current arm.
2. Stage B route 1 feasibility: EchoNext weights, licence, input spec, MIMIC overlap.
3. ~~Fill the pending ascending-site and AUPRC rows in §2.1~~ — done (`outputs/diag_complementarity/followup.json`).
4. Reconcile `PROGRESS_UPDATE_2026-08-30.md`'s claims with Part 1 (a short erratum section) before anyone cites it.
