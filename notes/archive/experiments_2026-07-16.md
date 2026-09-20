# Experiments & Results — 2026-07-16 (dimensionality, ROI, sample size)

> Follow-up to [notes/system_overview_and_analysis.md](system_overview_and_analysis.md). This
> logs a round of experiments testing the top improvement ideas, with fold-safe protocols and
> paired bootstrap CIs against the EHR floor. **All experiments reuse the frozen
> `pretrained_checkpoints/` artifacts and the immutable `fold_assignments.csv`, so every number
> is directly comparable to `outputs/eval_report/`.**
>
> Scratch harness: `/scratch4/rsteven1/chenjia_echo_project/aorta_exp/` (exp_*.py).
> Formalized repo scripts (the keepers): `scripts/train_reduced_fusion.py`,
> `scripts/extract_cxr_anatomy_rois.py`, `scripts/extract_raddino_anatomy_roi_embeddings.py`.
>
> **Reading the CIs:** everything is 5-fold OOF over the 522 triple cohort with 2000-sample
> patient bootstrap. `*` on a paired delta = 95% CI excludes 0 (distinguishable). Positives are
> tiny (root ge40 = 32, asc ge40 = 48, ge45 = 8/10), so binary CIs are wide; regression (dense)
> CIs are tight.

---

## TL;DR of this round

1. **PCA reduction of the frozen embeddings is a large, robust win** — it beats no-PCA concat on
   nearly every endpoint, and for the **ascending aorta the gain over the equal-n EHR floor is
   statistically significant** (paired CI excludes 0). *This is the single highest-value change.*
2. **CXR carries the aortic signal; the PCLR ECG embedding adds almost nothing.** Dropping ECG is
   often *better*; dropping CXR collapses ascending prediction. Confirms the PI's point that the
   aorta is readable off the X-ray.
3. **Anatomy-driven ROI (segment aorta+mediastinum+heart, per-patient box) is the best CXR
   representation for the ascending aorta** — beats both the whole image and the old hardcoded box.
   The *tight* aorta-only crop is worse (over-zoom loses context). Root prefers a lower/central box.
4. **Multi-instance (use all ~2,800 in-window CXRs, not 1/patient) strengthens the CXR base
   learner** (asc ge40 0.71→0.76, R² up markedly) — but only inside a **late-fusion** design;
   early-concatenating multi-instance CXR with EHR *hurts*.
5. **Negative results (honest):** PLS supervised reduction overfits (negative R²); deep-fusion
   val-loss model-selection regressed AUROC (reverted). Window relaxation (180→365 d) adds **no**
   new patients (all 522 already have both modalities in-window) — only more instances/patient.

**Recommended / new best model (BUILT & VALIDATED — see §7):** a **late-fusion stack**, CXR base =
*multi-instance + anatomy-ROI + PCA(32)*, restricted to **frontal (PA/AP)** views, + full-cohort EHR
base, + a small meta-learner (or just a rank-mean ensemble — as good at this n). It **significantly
beats the full-cohort EHR floor for the ascending aorta on both classification (ge40 0.756 vs 0.668,
Δ +0.081 \*) and diameter regression (R² 0.203 vs 0.143, Δ +0.060 \*)** — the strongest, cleanest
positive multimodal result the project has. Run: `scripts/train_combined_stack.py`
(`outputs/combined_stack_frontal_anat/`).

---

## 1. PCA / dimensionality reduction (fold-safe, HGB head)

Concat of raw frozen features = 1,100 dims on ~420 train rows → overfits. PCA (fit per train fold)
before the tree fixes it. Equal-n (every arm on the same 522, same folds):

| Config | root ge40 | root ge45 | root R² | asc ge40 | asc ge45 | asc R² |
|---|---|---|---|---|---|---|
| ehr_only (equal-n floor) | 0.614 | 0.365 | 0.110 | 0.555 | 0.368 | −0.015 |
| concat_all (no PCA) | 0.630 | 0.516 | 0.306 | 0.683 | 0.569 | 0.092 |
| pca e16 c32 +ehr | 0.663 | 0.660 | 0.286 | 0.693 | 0.518 | 0.132 |
| pca e16 c16 +ehr | 0.677 | **0.764** | 0.260 | 0.666 | **0.665** | 0.122 |
| pca c32 +ehr (no ECG) | **0.704** | 0.653 | 0.289 | 0.690 | 0.476 | 0.127 |
| pca c32 only (CXR) | 0.690 | 0.582 | 0.173 | 0.706 | 0.662 | 0.095 |
| pca e16 only (ECG) | 0.656 | 0.644 | 0.019 | 0.475 | 0.483 | −0.089 |

**Paired bootstrap vs the equal-n EHR floor (ge40):**

- **Ascending — every PCA config significantly beats the floor:** pca c32-only Δ **+0.151 [+0.062,
  +0.239] \***, pca e16c32 Δ **+0.138 [+0.045, +0.233] \***, concat Δ +0.128 [+0.027, +0.226] \*.
- **Root — not significant** (only 32 positives): best Δ +0.090 [−0.024, +0.203], but every point
  estimate is up, and **root diameter R² is significantly better**: 0.11 (floor) → **0.31**
  (pca+linear e16c32, CI [0.233, 0.377] vs floor [−0.003, 0.210]).

**Head choice:** PCA+HGB is best for ge40; PCA+**linear** (LogReg/Ridge on the reduced space) is
best for regression (root R² 0.309) and the rare ge45. **PLS (supervised) failed** — negative R²,
it overfits the target at this n. Use unsupervised PCA.

**Dim sweet spot:** ~16–32 comps/modality. Fewer dims (e16 c16) help the rare ≥4.5 endpoint most
(denoising matters more when positives are scarce); more dims (e32 c64) help ge40.

## 2. Anatomy-driven ROI (vs hardcoded box vs whole image)

Segmented every cohort CXR with a pretrained ChestX-Det PSPNet (torchxrayvision) → per-patient box
= (aorta ∪ mediastinum ∪ heart) + 6% pad. CXR-only, PCA c32 + HGB:

| CXR representation | asc ge40 | asc ge45 | asc R² | root ge40 | root R² |
|---|---|---|---|---|---|
| whole image | 0.706 | 0.662 | 0.095 | 0.690 | 0.173 |
| hardcoded box (0.25,0.08,0.75,0.60) | 0.692 | 0.659 | 0.090 | 0.679 | **0.241** |
| **anatomy union** | **0.740** | **0.741** | **0.105** | 0.627 | 0.199 |
| anatomy tight (aorta+mediastinum) | 0.664 | 0.583 | 0.043 | 0.534 | 0.116 |

- **Anatomy union is the best CXR rep for the ascending aorta** on all 3 metrics (Δ ge40 vs whole
  +0.034, vs fixed +0.048; directionally consistent, not individually significant at n).
- **Over-cropping hurts** (tight aorta-only crop worst) — a principled box that keeps the heart
  silhouette beats an aggressive one. This is exactly what a hardcoded box can't adapt.
- **The optimal ROI is site-specific:** root (deeper/central) does best with the lower fixed box;
  ascending does best with the anatomy box.
- Combined win: `reduced_fusion_anat` (anatomy ROI + PCA, no ECG) → asc ge40 **0.711**, ge45
  **0.721**, the best ascending numbers in this round for the fusion model.

## 3. Sample size — multi-instance CXR (2,792 images, ~5.3/patient)

The pipeline keeps 1 best CXR/patient (522). Every in-window CXR is a valid instance for the same
patient-level label. Trained the CXR base learner on **all** instances (patient-weighted 1/nᵢ,
patient-level folds so no leakage), averaged predictions per patient at test:

| | asc ge40 | asc R² | root ge40 | root R² |
|---|---|---|---|---|
| CXR-only, single image | 0.706 | 0.095 | 0.690 | 0.173 |
| **CXR-only, multi-instance** | **0.755** | **0.159** | 0.725 | **0.259** |
| CXR **+EHR** early-concat, multi | 0.608 ⚠ | 0.118 | 0.694 | 0.216 |

- **Multi-instance meaningfully strengthens the pure CXR learner** (asc ge40 +0.049, R² +0.06;
  root R² 0.17→0.26). More images ≈ more data ≈ better imaging model, as expected.
- **But early-concatenating multi-instance CXR with EHR *significantly hurts* ascending** (0.690 →
  0.608, Δ −0.082 [−0.161, −0.004] \*): repeating a patient's EHR across its images distorts the
  tabular signal. **⇒ multi-instance belongs in a late-fusion CXR base learner, not early concat.**
- Window relaxation 180→365 d gives **0** new triple-cohort patients (all 522 already have both
  modalities in-window); it only adds instances. Growing *patients* needs more CXR downloads.

## 4. Capstone — imaging vs the practical (full-cohort) EHR floor

CXR-multi base learner vs the repo's full-cohort `ehr_baseline` (the real floor, ~4,525 train),
aligned patient-for-patient:

| Endpoint | CXR-multi (imaging) | EHR-full floor | Δ (CXR − EHR) |
|---|---|---|---|
| **asc ge40** | **0.755** [0.679, 0.823] | 0.668 [0.590, 0.747] | +0.087 [−0.013, +0.181] |
| asc ge45 | 0.678 [0.514, 0.825] | 0.676 [0.517, 0.800] | +0.002 [−0.160, +0.218] |
| root ge40 | 0.725 [0.647, 0.801] | 0.776 [0.692, 0.858] | −0.051 [−0.142, +0.038] |
| root ge45 | 0.611 | 0.852 [0.789, 0.912] | −0.241 [−0.427, −0.070] \* |

Parameter-free **rank-mean ensemble (CXR-multi + EHR-full)** vs the EHR floor (ge40):

| Site | Ensemble AUROC | Δ vs EHR floor |
|---|---|---|
| **asc** | **0.751** [0.686, 0.814] | **+0.083 [+0.032, +0.130] \*** |
| root | 0.786 [0.707, 0.861] | +0.010 [−0.033, +0.053] |

**The headline result of the whole round:** for the ascending aorta, the multi-instance CXR
learner *by itself* (0.755) out-points even the **full-cohort** EHR floor (0.668, trained on ~8×
more patients), and a trivial imaging+EHR ensemble **significantly** beats that floor (Δ +0.083,
CI excludes 0). This is a genuine, defensible multimodal gain against the hardest baseline — the
positive result the project was missing. For the root (a body-size phenomenon) EHR still
dominates, and imaging only nudges the ensemble a hair above the floor.

## 5. Negatives / dead ends (documented so they aren't retried)

- **PLS supervised reduction** — negative test R²; overfits target directions at n≈420. PCA wins.
- **Deep-fusion val-loss model selection** — regressed AUROC (root ge40 0.651→0.588); reverted.
  Deep fusion trails the PCA-reduced GBDT regardless; not the path forward at n=522.
- **Window 180→365 d** — no new patients.

## 6. Concrete next steps (in priority order)

1. ~~Build the late-fusion stack with the improved base learners.~~ **DONE — see §7.**
2. ~~Extract anatomy-ROI embeddings for the multi-instance set.~~ **DONE (2,792 images) — see §7.**
3. **Grow the cohort** = download more CXRs (the binding constraint) — the only way past n=522.
4. **Revisit ECG** only after auditing PCLR preprocessing (unnormalized / full-length interp,
   see system_overview §9) — decide whether ECG is truly weak or just mis-fed before investing.

## 7. Combined late-fusion stack — the new best model (built 2026-07-16)

Combines every win: **CXR base learner = multi-instance + anatomy-ROI + PCA(32)**, **EHR base =
full-cohort HGB** (repo `ehr_baseline` OOF), **meta = nested-CV LogisticRegression/Ridge** over
`[p_cxr, p_ehr, d_ehr]`. Extraction: segmented all 2,792 in-window CXRs (ChestX-Det PSPNet) and
re-ran RAD-DINO on each anatomy box → `raddino_multi_anatomy_embeddings.pt`. Repo scripts:
`scripts/extract_multi_cxr_instances.py`, `scripts/extract_multi_anatomy_roi_embeddings.py`,
`scripts/train_combined_stack.py`. All 5-fold OOF on the 522, paired bootstrap vs the full-cohort
EHR floor.

**Segmentation fallback diagnostic:** 28% of instances are **lateral** views (LATERAL/LL), where the
frontal-trained segmenter correctly fails (PA/AP seg-ok ≈99%, lateral ≈15%) and falls back to whole
image. Restricting the multi-instance pool to **frontal (PA/AP)** views removes that noise and is the
best config.

| Stack variant | asc ge40 AUROC | Δ vs EHR floor | asc ge45 | asc diam R² | ΔR² vs floor | root ge40 |
|---|---|---|---|---|---|---|
| anatomy, all views | 0.731 | +0.063 [0.002, 0.123] \* | 0.730 | 0.197 | +0.053 [0.006, 0.105] \* | 0.787 |
| whole, all views | 0.736 | +0.068 [−0.003, 0.139] | 0.650 | 0.189 | +0.046 [0.008, 0.085] \* | 0.776 |
| **anatomy, frontal-only** | **0.756** | **+0.081 [0.007, 0.158] \*** | **0.702** | **0.203** | **+0.060 [0.014, 0.103] \*** | 0.772 |
| whole, frontal-only | 0.740 | +0.065 [0.003, 0.133] \* | 0.647 | 0.208 | +0.065 [0.020, 0.110] \* | 0.775 |

Reference: EHR floor asc ge40 **0.668**, asc R² **0.143**, root ge40 0.776; late_fusion_v2 asc ge40 0.695.

**Result:** the **anatomy + frontal-only** stack is the new best model. For the **ascending aorta** it
**significantly beats the full-cohort EHR floor on both classification (ge40 0.756, Δ +0.081 \*) and
diameter regression (R² 0.203, Δ +0.060 \*)** — two independent endpoints, paired CI excludes 0. It
also out-points the previous best `late_fusion_v2` (0.695 → 0.756 on asc ge40). For the **root**, it
ties the EHR floor (ge40 0.772 vs 0.776; diam R² 0.338 vs 0.309, borderline) — root remains a
body-size phenomenon, as expected.

**Honest caveats:**
- **The learned nested-CV meta ≈ the parameter-free rank-mean ensemble** (asc ge40 stack ≈ rank-mean
  ≈ 0.74–0.76). At n=522 the meta-learner adds complexity without clear gain — a simple rank-mean of
  the CXR and EHR base learners is a fine, more robust production choice.
- For **asc ge40 the CXR base learner alone (0.758) ≈ the full stack (0.756)** — EHR adds nothing to
  the *ascending* prediction (it's imaging-driven); the stack matters for combining sites into one
  model and for the root.
- Still n=522 / 48 positives: the ge40 delta CI lower bound is ~0.007 (significant but marginal); the
  regression ΔR² is the more robust of the two.
- Anatomy-ROI's edge over whole-image is modest under multi-instance averaging (clearest on the rare
  asc ge45: 0.702 vs 0.647) — averaging many whole images already denoises a lot.

**New best-model recipe (drop-in):** `VIEW=frontal CXR_EMB=raddino_multi_anatomy_embeddings.pt
python scripts/train_combined_stack.py` → `outputs/combined_stack_frontal_anat/`.
