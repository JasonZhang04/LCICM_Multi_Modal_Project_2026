# Progress update — model rigor, the raw-waveform ECG arm, and the fusion architecture study

**Companion to `PROGRESS_UPDATE_2026-08-05.md`. Read that first for the episode cohort, the two-modality (CXR+EHR) headline, and the ECG-measurement null. This document does not repeat any of it.**
Reporting period: 5–30 August 2026

---

## 0. What this document covers

The 5 August update established the two-modality (CXR+EHR) model and a *measurement-level* ECG null. Since then, prompted by a publishability review against two competing papers, the work has (1) hardened the two core scientific claims with targeted ablations, (2) built a **raw-waveform ECG model** to close the biggest hole in the ECG story, and (3) run a **fusion architecture study** that settles how the modalities should be combined. Framing was also sharpened into two objectives, below.

Nothing here changes the cohort or the evaluation protocol from the previous reports; it adds experiments on top of them.

---

## 1. Positioning (two objectives, and how we differ from prior art)

Two directly relevant papers appeared:

- **DeepCatch X Aorta** ([Lee et al., *Eur J Radiol* 2025](https://pubmed.ncbi.nlm.nih.gov/40907419/)) — a commercial CXR-only device that reaches AUROC 0.88 for ascending ≥4.0 cm. But its labels are **CT** (near-gold-standard), its cohort is **~50% prevalence** (patients who received a CT), it is **PA-only**, and it is **n=526**. Our setting is TTE labels, **4.5% prevalence**, unselected, PA+AP. The numbers are not on comparable footing — theirs is a high-prevalence CT-confirmation task, ours is low-prevalence opportunistic screening on the noisier reference standard.
- **ECGAI-TAA** ([Pirruccello et al., medRxiv 2025](https://www.medrxiv.org/content/10.1101/2025.08.19.25333786v1)) — an ECG-only model (MRI labels, healthy UK Biobank, R²=0.31) whose predictions were *not* explained by clinical factors. Weights are not publicly released (confirmed by search).

Neither is multimodal, and both are single-modality in curated cohorts. Our contribution is therefore framed as **two objectives**:

1. **A comprehensive modality-value ablation** for aortic-diameter estimation from a single large real-world dataset — which of CXR, ECG, EHR carry signal, whether they are complementary or redundant, and how best to fuse them.
2. **A multimodal model built and evaluated** with a full clinical operating picture.

Label-reliability (the echo noise ceiling) is kept as a *supporting* note, not a headline. This positioning does not require beating a commercial CT-labeled detector; it stakes out the multimodal-value question no single-modality paper answers.

---

## 2. Hardening the two core claims (Tier-0 ablations)

### 2.1 The ECG-measurement null is now airtight (Problem A)

The concern was that late-fusion-of-scalars could make a near-zero ECG increment structurally inevitable. Three tests that bypass that:

- **Early fusion** (one HGB over all concatenated features — can model interactions): ECG-measurement increment +0.002 [0.000, 0.004] (root) / +0.003 [0.001, 0.006] (asc) R².
- **Residual fit** (can the ECG explain what CXR+EHR gets wrong?): R² ≈ **−0.02** — it explains none of the residual.
- **TOST equivalence** against a pre-specified ±0.01 MCID: all four increments' 90% CIs fall inside the band → **formally equivalent to no difference**, not merely "small CI."

The measurement-level null does not depend on the fusion design. (The *waveform* question is separate — §3.)

### 2.2 The CXR reads the aorta, not body size (Problem B)

The reviewer's sharpest question. Two ablations, both favorable at both sites:

| | floor R² | CXR-full increment | aorta-only increment |
|---|---|---|---|
| root, all | 0.265 | R² +0.062 | +0.051 |
| root, complete-EHR | 0.283 | R² +0.061 | +0.052 |
| asc, all | 0.202 | R² +0.117 | +0.093 |
| asc, complete-EHR | 0.209 | R² +0.116 | +0.094 |

The increment barely moves when restricted to episodes with complete height/weight/BP (so it is **not** body-size imputation), and it survives stripping the CXR branch to **aorta-pool + aorta geometry only** (so it is **anatomically localized**). This vindicates the root, which we had worried was habitus.

### 2.3 Supporting analyses

- **Lag sensitivity** — performance is flat across CXR-to-index lag (root 0.812 at 0–7 d vs 0.843 at 91–365 d; asc ~0.805 throughout). The model reads **stable anatomy**, supporting the opportunistic-screening framing and the pre-index window.
- **Temporal holdout — quarantined.** The echo cohort clusters in 2008–2016, so we locked away **2014–2016** (6,291 episodes, 262 root+ / 228 asc+; train pool 14,391) as a touch-once temporal test set, set aside before the ablation program.

---

## 3. The raw-waveform ECG arm

The 5 August null was about the ECG *summary* (`machine_measurements.csv`), not the signal. To close that, we built a purpose-built waveform model — the ECGAI-TAA architecture reimplemented from scratch on MIMIC-IV-ECG (10 s / 500 Hz / 12-lead), trained against our own TTE labels (no MRI/UK-Biobank domain gap; the paper's weights are not public).

- Nested-in-fold OOF (one 3.05M-param 1D-ResNet per outer fold), audited leakage-free (0 patients span folds; embeddings out-of-fold).
- **Waveform-alone beats the summary:** R² **0.141 / 0.150** (root/asc) and AUROC **0.716 / 0.713**, versus the summary's 0.085 / 0.113 and 0.67. The raw signal carries ~50–60% more aortic signal than the machine's numbers — the summary was discarding it.

Whether that unique signal survives *on top of CXR+EHR* is answered in §4.

---

## 4. Fusion architecture study — the main new result

We compared how to combine the modalities, at the **feature level** (using the ECG CNN's 256-d penultimate embedding, not its scalar — avoiding the bottleneck), on identical patient-grouped folds with cluster-bootstrap CIs. Arms: **late** (ridge over per-modality diameter scalars), **early** (one HGB over concatenated features), and **cooperative learning** (the Ding & Tibshirani agreement penalty used by the AS paper — a ρ knob that interpolates early↔late, tuned by inner CV).

> **Methods review caught two blockers first.** A dedicated review pass found (i) a cooperative-learning intercept bug that silently collapsed its predictions for ρ>0, and (ii) an ECG-block NaN crash. Both were fixed and verified before any result was trusted. The numbers below are post-fix.

**Fusion ranking (R² on the 18,221 ECG-having episodes, 3 seeds):**

| arm | root | asc |
|---|---|---|
| **linear early fusion** | **0.339** | **0.327** |
| HGB early fusion | 0.328 | 0.319 |
| late fusion (ridge over scalars) | 0.323 | 0.313 |

- early-vs-late: +0.005 [0.002, 0.009] (root), +0.006 [0.002, 0.009] (asc) — early beats late.
- linear-early vs HGB-early: **+0.011 [0.007, 0.015]** (root), **+0.008 [0.002, 0.013]** (asc) — the regularized linear model beats the tree (which overfits slightly at the label-noise ceiling).

**The decisive finding: the cooperative-learning agreement penalty was never used — ρ=0 was selected in all 15 folds, at both sites.** The framework *adaptively tested* whether the modalities complement each other and answered **no**: they are redundant, so there is no complementary signal for an interaction layer to exploit. At ρ=0, cooperative learning reduces to a regularized **linear early fusion** — and that is the winner. This is consistent with the broader evidence that cross-modal fusion helps for complementary, not redundant, modalities ([CareBench, MIMIC EHR+CXR](https://arxiv.org/html/2602.23614v1); [Ding & Tibshirani, PNAS 2022](https://arxiv.org/abs/2112.12337)). The practical conclusion is the one the design brief asked for: **the best fusion is the simplest one that lets the modalities interact at the feature level — a regularized linear early fusion — not an elaborate cross-modal architecture.**

**The waveform ECG earns a small but real place.** Feature-level increment of adding the waveform embedding to CXR+EHR (same episodes): root R² **+0.007 [0.005, 0.010]**, asc **+0.005 [0.003, 0.008]** — statistically nonzero, and **more than the ECG measurements added** (which were TOST-equivalent to zero). The AUROC increment is borderline (~+0.003–0.004), consistent with the signal living in the continuous-diameter headroom. So the raw waveform recovers a little unique signal the summary discarded; the arm was worth building.

---

## 5. The final integrated model

Combining the three ablation winners — the **multi-instance CXR** representation, **regularized linear early fusion**, and the **waveform-ECG embedding** — into one model (`train_final_model_episode.py`, 3 seeds, all 20,682 episodes):

| | Final model | vs EHR floor | vs geometry-stack (5-Aug headline) |
|---|---|---|---|
| root ge40 | **0.832 [0.819, 0.844]** | +0.047 [0.039, 0.056] \* | **+0.005 [0.002, 0.009]** \* |
| root diam R² | **0.326 [0.314, 0.337]** | +0.055 [0.049, 0.061] \* | **+0.009 [0.006, 0.012]** \* |
| asc ge40 | **0.814 [0.799, 0.830]** | +0.089 [0.076, 0.103] \* | **+0.007 [0.003, 0.011]** \* |
| asc diam R² | **0.314 [0.301, 0.326]** | +0.106 [0.097, 0.115] \* | **+0.008 [0.005, 0.011]** \* |

\* paired cluster-bootstrap CI excludes zero.

The integrated model **significantly beats the previous headline** (geometry-stack, late fusion of CXR+EHR) at both sites on both metrics — modestly (+0.005–0.009) but with CIs that exclude zero. The gain decomposes cleanly into the three winners: linear early over late fusion (~+0.006), the waveform-ECG increment (~+0.005–0.007), and the retained multi-instance CXR. It reaches **~54–58% of the label-noise ceiling** (R² ~0.56–0.58), up from ~53–57%.

This is the deployable model, and it is *simple*: multi-instance CXR predicted diameter → a regularized linear ridge over that scalar plus the raw EHR features and the ECG-waveform embedding → rank by predicted diameter. No cross-modal attention, no cooperative-learning penalty (which the data rejected), no end-to-end joint training. Every component earned its place in a controlled ablation, which is the story the PI asked for: clear, empirically grounded, and hard to overfit.

---

## 6. Follow-up: four design questions (PCA, leakage, fusion completeness, fine-tuning)

A review of the model raised four questions; each was answered directly.

**PCA dimensionality (is 32 a bottleneck?) — yes, mildly, and fixed for free.** A sweep of the CXR base learner's OOF diameter R² over PCA dim (single seed) shows it rises then plateaus:

| PCA dim | 16 | 32 | 64 | 128 | 768 (none) |
|---|---|---|---|---|---|
| root CXR-alone R² | 0.294 | 0.298 | 0.307 | **0.309** | 0.307 |
| asc CXR-alone R² | 0.287 | 0.292 | 0.302 | **0.304** | 0.303 |

At n=522 PCA-32 was essential overfitting control; at ~16k it under-uses the embedding. Moving to **K=128** lifts the final model from R² 0.326→**0.334** (root) and 0.314→**0.325** (asc), ge40 0.832→0.835 / 0.814→0.818 — a free gain now adopted. That no-PCA (768) does *not* beat 128 also means the frozen RAD-DINO representation **saturates around 128 effective dimensions** — so the only way past that ceiling is to update the backbone (fine-tuning, below).

**Leakage between the ECG model and the multimodal model — none.** The ECG waveform model is trained under the same patient-grouped nested CV: each patient's ECG embedding is produced by a model that never saw that patient, so it is a pre-computed leakage-free feature (like the frozen CXR embeddings). An independent methods review confirmed the embedding is out-of-fold and that 0 patients span folds. (The two core fusion scripts were also review-audited; a cooperative-learning intercept bug and an ECG-NaN crash were caught and fixed before any result was trusted — §4.)

**Cross-attention — tested directly, and it does not help.** A small regularized cross-attention block over the three modality tokens, run head-to-head against linear early fusion on the same folds, is statistically indistinguishable from it: root R² −0.002 [−0.005, 0.001], asc −0.001 [−0.005, 0.003] (both CIs include zero; AUROC likewise). So the claim that letting the modalities attend to each other adds nothing over simple linear fusion now rests on **direct evidence**, not just the cooperative-learning ρ=0 inference — as predicted for redundant modalities.

**Fine-tuning — the ECG was trained from scratch; the CXR was not.** The waveform ECG model is trained end-to-end on our task (not a frozen embedding). RAD-DINO, however, is frozen — and the PCA saturation above shows that caps the CXR contribution. Partial fine-tuning of RAD-DINO (unfreeze the last blocks + a regression head, image-level, nested-in-fold) is now under way as the main performance lever, feasible at 16k patients / 60k images where it was not at 522. Result to follow.

---

## 7. Where this leaves the paper

- **Objective 1 (modality-value ablation) is essentially complete and coherent:** CXR carries most of the aortic signal, anatomically localized; EHR contributes body-size context; the ECG *summary* is redundant (TOST-equivalent), while the raw *waveform* adds a small unique R² increment; and the modalities are redundant rather than complementary, so simple linear early fusion is optimal (cooperative learning's ρ=0 confirms it).
- **Objective 2 (the multimodal model)** is delivered as the integrated model in §5 — it significantly improves on the 5-Aug headline and has the clinical operating picture (AUPRC, confusion matrices, calibration, NNE) already in place from the 5 August work (to be regenerated on the final model's OOF).
- **Remaining before a draft:** evaluate the quarantined temporal holdout once, on the final model only; regenerate the figure suite on the episode cohort; CONSORT diagram; and the small hygiene items (pytest fixtures, GitHub-token rotation).
