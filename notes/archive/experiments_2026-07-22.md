# Experiments & Results — 2026-07-22 (geometry, regression-derived scoring, fusion limits)

> Follow-up to [experiments_2026-07-16.md](experiments_2026-07-16.md). Same frozen artifacts,
> same 522 triple cohort. **Every headline number here is validated by repeated cross-validation
> over 5 independent fold seeds**, not a single split — see §5 for why that mattered.
>
> Formalized repo scripts: `scripts/train_geometry_stack.py` (the model),
> `scripts/extract_cxr_geometry_features.py`, `scripts/extract_raddino_patchpool.py`.
> Scratch harness: `/scratch4/rsteven1/chenjia_echo_project/aorta_exp/`.

---

## TL;DR — new best model, validated

| Endpoint | Previous best (2026-07-16) | **Now** (mean ± SD over 5 fold seeds) |
|---|---|---|
| **Root ge40 AUROC** | 0.772 | **0.809 ± 0.007** (0.800–0.818) |
| **Asc ge40 AUROC** | 0.756 | **0.790 ± 0.014** (0.771–0.804) |
| Root ge45 AUROC | 0.906* | 0.900 ± 0.020 |
| Asc ge45 AUROC | 0.859* | 0.869 ± 0.022 |
| Root diameter R² | 0.344 | 0.354 ± 0.010 |
| Asc diameter R² | 0.203 | 0.221 ± 0.013 |

**The model:** multi-instance FRONTAL CXRs → per-image `[cls | aortapool | heartpool]` RAD-DINO
views (PCA-32 each, fit per training fold) **+ 17 engineered geometric features** → HGB
**regression on diameter** (patient-weighted, averaged per patient) → ridge stack with the
full-cohort EHR diameter → **rank by predicted diameter** as the ge40/ge45 score.

Two mechanisms produced the gain, and both are cases of *the same principle*: **add orthogonal
information, and don't throw label information away.**

---

## 1. Engineered geometric CXR features (complementary to the embedding)

From the ChestX-Det (torchxrayvision) masks we compute what a radiologist actually reads off a
frontal film — 17 features incl. **mediastinal width ratios (upper/mid/lower)**, **cardiothoracic
ratio**, **aortic-knob lateralization** relative to the spine midline, and normalized areas.
(`scripts/extract_cxr_geometry_features.py`; 1,798 frontal instances, 1 bad seg, 0 failures.)

Sanity: medians are anatomically correct — CTR **0.531** (MIMIC is ICU-heavy with AP portable
films, which magnify the heart), upper mediastinum **narrower** than lower (0.186 vs 0.268).
The geometry math was additionally unit-tested against synthetic masks.

Ascending, CXR-only:

| CXR features | ge40 AUROC | R² |
|---|---|---|
| embedding only | 0.758 | 0.157 |
| geometry only | 0.696 | 0.102 |
| **embedding + geometry** | **0.797** | **0.205** |

**Geometry alone is worse than the embedding, yet the two together beat either** — they are
orthogonal (ViT texture/appearance vs explicit measurements). Root is unchanged: geometry-only
root = 0.585, because **the aortic root is not visible on a frontal film** (it sits behind the
heart/sternum) — only the ascending aorta and arch form the mediastinal border.

## 2. Regression-derived classification (the biggest single win)

**Stop training a classifier on 32–48 binary positives. Regress the continuous diameter (all 522
graded labels) and rank patients by predicted diameter.**

| Site / endpoint | Direct classifier | Regression-derived | Δ (paired) |
|---|---|---|---|
| Root ge40 (32 pos) | 0.767 | **0.809** | +0.042 [−0.005, 0.091] |
| Root ge45 (8 pos) | 0.872 | **0.906** | +0.033 [−0.012, 0.085] |
| Asc ge40 (48 pos) | 0.777 | 0.784 | +0.006 |
| Asc ge45 (10 pos) | 0.699 | **0.859** | **+0.160 [0.034, 0.298] \*** |

Per-modality the effect is dramatic (CXR asc-ge45 0.681 → **0.901**; CXR root-ge40 0.669 → 0.775).

**Why:** the binary label discards most of the information — 3.9 cm and 2.5 cm are both
"negative". The decisive evidence that this is the mechanism: **the gain scales inversely with
positive count** (+0.160 at 10 positives, +0.006 at 48). ⇒ **improving diameter R² is now the
direct lever on AUROC.**

## 3. Patch-token pooling (modest, and only in combination)

`scripts/extract_raddino_patchpool.py` runs RAD-DINO on the whole image and pools the 196 patch
tokens with the segmentation mask as weights (a *soft* ROI: spatially selective without cropping
away context). Emits `cls`, `aortapool` (aorta ∪ mediastinum), `heartpool`.

- **aortapool ALONE is worse than the anatomy-ROI crop** (asc R² 0.183 vs 0.205).
- **Combining views wins**: `cls+aortapool+heartpool` → asc R² 0.205 → **0.234**, ge45 → 0.916.

Same complementarity principle as §1: several different summaries of one image beat any single one.

## 4. Fusion: what it can and cannot do (answers "can late fusion beat the best modality?")

- The meta-learner **is** per-site (fit separately for root/asc), so it already learns site-specific
  weights; the inner-CV blend weight makes this explicit and interpretable — it chose
  **w_CXR ≈ 0.6–0.9 for ascending** and favored EHR for root, exactly as anatomy predicts.
- **But fusion here ≈ the best single modality and does not exceed it**, because per site the
  modalities are largely *redundant*, not complementary (asc: CXR 0.797 / EHR 0.668; root: EHR
  0.776 / CXR 0.669). Late fusion only beats its best input when inputs carry complementary signal.
- Worse, at ~48 positives a learned LR meta **dilutes** a strong base: asc 0.797 → 0.777.
- **Inner-CV selection of the fusion strategy backfired on root** (0.776 → 0.718, significantly
  worse than just using EHR). With ~32 positives split into 3 inner folds, each inner-val holds
  ~2 positives, so the selector is choosing on noise. The *same* selector was perfectly stable
  (5/5 folds) on the **dense regression** target. ⇒ **never do discrete model selection on the
  sparse binary endpoint at this n.**

## 5. Why repeated CV was essential (methodological lesson)

After testing ~6 representation variants × 2 sites on one fixed fold assignment, all deltas were
non-significant — a textbook setup for fitting the cross-validation itself. Re-running two locked
configs under 5 seeds:

| Config | Root ge40 | Asc ge40 |
|---|---|---|
| A: anat_roi + geometry | 0.799 ± 0.013 | 0.774 ± 0.012 |
| **B: cls+aorta+heart + geometry** | **0.809 ± 0.007** | **0.790 ± 0.014** |

On the **single** split, A looked better for root (0.809 vs 0.797) and we would have shipped it.
Across seeds, **B is better at both sites with lower variance** — the single-split comparison had
it backwards. **Lesson: at n=522, do not believe any variant choice from one fold assignment.**

## 6. Negative results (documented so they aren't retried)

- **LoRA fine-tuning of RAD-DINO underperforms the frozen features.** Trained against the *dense*
  diameter target (1,798 instances, 0.30M trainable params, patient-level folds, early stopping —
  i.e. the favourable setup): root R² 0.152 vs frozen 0.287, ge40 0.708 vs 0.784; asc R² 0.105 vs
  0.234. Replicates the v2 finding that fine-tuning does not pay at this cohort size.
  *Caveat:* not perfectly matched (frozen path feeds 3 views + geometry to a GBDT; LoRA emits one
  2-dim prediction). `aorta_exp/train_cxr_lora.py`.
- **No horizontal-flip augmentation.** The aortic arch/knob is left-sided; flipping destroys the
  laterality signal the geometry features rely on. A "standard" augmentation that would actively hurt.
- **ECG remains redundant** (see 2026-07-16 round): expanding to 2,874 patients made ECG-only root
  0.577 → 0.680, but it adds ~0 beyond EHR+CXR with either PCLR or xECG.

## 7. Next steps

1. **Diameter R² is now the lever on AUROC** — anything that improves the regression improves the
   screen. Asc R² (0.221) is the weaker of the two and the more imaging-driven, so it has headroom.
2. **Cross-site transfer**: root and ascending diameters correlate, and the ascending aorta *is*
   visible; feeding the predicted ascending diameter as an input to the root model is cheap and untested.
3. **More patients** remains the only way past the n=522 ceiling (relaxing the window to ±365 d
   would add ~120 tri-modal patients at some temporal-coherence cost).
4. Treat 0.85 as ambitious: at R² ≈ 0.35 the implied AUROC ceiling for a threshold on an indirect
   measurement is roughly where we now are.
