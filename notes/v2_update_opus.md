# Multimodal Aortic Diameter Model — Integrated Redesign Plan

> Synthesizes this conversation's analysis with the prior [notes/v2_update.md](2026 Multi-Modal Project/notes/v2_update.md) draft from another agent. Where the two diverged, the rationale for the chosen path is recorded inline.

## Context

The v1 run is collapsing to the population mean. From [outputs/multimodal_aorta/train.log](2026 Multi-Modal Project/outputs/multimodal_aorta/train.log):

- Best val MAE = **0.394 cm at epoch 2** (population std = 0.497 cm → R² ≈ 0.07).
- Val loss starts diverging at epoch 3; early-stop fires at epoch 12.
- Epoch-2 per-modality ablation: **both=0.393, ECG-only=0.402, CXR-only=0.404** → fusion is **not fusing**; either modality alone matches the joint prediction.

The user's two diagnoses are correct:

1. **ECG ResNet1D-34 (~30 M params, random init) on ~1,400 train ECGs** is hopeless for a subtle regression task. Cold-start ECG branch learns "predict the mean."
2. **Modality imbalance is extreme**: 522 both-modality / 2,352 ECG-only / 28 CXR-only / 1,630 dropped. 82 % of batches have no CXR; CXR mask-token gets most of the gradient.

This plan is a **6-step rollout**, each step independently runnable and evaluable against the 0.394 cm baseline. Adopt the recommended order; don't skip Step 1.

---

## Diagnosis — what is currently wrong

| Symptom | Root cause |
|---|---|
| Val MAE plateaus at 0.39 cm = ~population std | Model predicts mean. No useful gradient signal. |
| Both ≈ ECG-only ≈ CXR-only ablation at epoch 2 | Fusion ignores both inputs. Mask token dominates. |
| Train loss drops, val diverges by epoch 5 | 89 M trainable params on 2 k samples → overfit. |
| Huber δ=0.5 cm with target std=0.5 cm | ~84 % of errors land in the quadratic regime → Huber buys nothing over MSE. |
| Targets in raw cm units (~3.2 mean) | Gradient scale mismatched to net activations; output head dominates early training. |
| ECG encoder cold-start | 30 M params on 1,400 patients with no pretraining → can't learn aortic-relevant features. |
| Spatial info in CXR collapsed to a single 768-dim CLS | Aortic dilation has localized image signal; CLS token is too coarse. |

---

## The 6-step rollout

Each step changes **≤ 4 files** to existing code, leaves the rest of the pipeline runnable, and produces a number we can compare against the v1 baseline. Run a 2-epoch debug job first, then a full SLURM job, then compare on **(val MAE, val R², ablation gap)** per target.

### Step 1 — Target standardization + MSE *(highest impact / lowest effort)*

**Problem:** Huber δ=0.5 cm is essentially MSE on this label distribution (target std=0.5). Raw-cm targets (mean 3.2) blow up the regression head's gradient relative to encoder gradients. This alone likely explains the post-epoch-2 divergence.

**Change:**
- Compute `(root_mean, root_std, asc_mean, asc_std)` on the train split only; cache in [data/dataset.py](2026 Multi-Modal Project/src/multimodal_aorta/data/dataset.py).
- Subtract mean / divide by std inside the dataset; train against z-scored targets with plain MSE.
- De-normalize at eval time so reported MAE/R²/AUROC stay in human-readable cm.

**Files:** [training/losses.py](2026 Multi-Modal Project/src/multimodal_aorta/training/losses.py), [data/dataset.py](2026 Multi-Modal Project/src/multimodal_aorta/data/dataset.py), [configs/default_config.py](2026 Multi-Modal Project/src/multimodal_aorta/configs/default_config.py), [training/evaluate.py](2026 Multi-Modal Project/src/multimodal_aorta/training/evaluate.py).

**Expected:** Val MAE stops diverging after epoch 2. R² lifts to ~0.15-0.20 from this fix alone.

---

### Step 2 — Self-supervised ECG pretraining (SimCLR on MIMIC-IV-ECG)

This step **adopts the proposal in [notes/v2_update.md](2026 Multi-Modal Project/notes/v2_update.md)** over my initial recommendation of ECG-FM. Reasoning:

| | ECG-FM (wanglab) | SimCLR self-pretrain *(chosen)* |
|---|---|---|
| Pretraining corpus | MIMIC-IV-ECG + PhysioNet 2021 (~1.5 M) | MIMIC-IV-ECG (800 k records, 161 k patients) |
| Model capacity | 90 M params, wav2vec 2.0 | 30 M params, ResNet1D-34 (existing) |
| Integration risk | **High** — must reimplement wav2vec 2.0 + state-dict remap; `fairseq_signals` is not pip-installable in our env (torch 2.5.1 + transformers 4.39.3) | **Low** — same encoder we already have; only adds a projection head + a new dataset + new train loop |
| Expected gain | Largest (in theory) | Moderate but reliable |
| Implementation time | ~1 week including debugging | ~3-4 h coding + ~3 h GPU |

**Decision:** Ship SimCLR first as Step 2 (low risk, ships in a day, gets us *some* pretraining benefit immediately). Defer ECG-FM to **Step 6 (optional)** — only attempt if SimCLR isn't enough after Steps 1-5 are in.

**Architecture during pretraining:**
```
ResNet1DEncoder (same 30 M-param arch) → h ∈ R^768
ProjectionHead: Linear(768 → 512) → BN → GELU → Linear(512 → 128) → z ∈ R^128
```
Projection head discarded after pretraining. Only encoder weights are transferred.

**Augmentations (two independent views per ECG):**

| Aug | Range | Notes |
|---|---|---|
| Time crop | Keep 4000/5000 samples, zero-pad | partial recording |
| Amplitude scale | U[0.7, 1.3]× per lead | electrode-contact variation |
| Gaussian noise | σ = U[0, 0.02] × signal std | measurement noise |
| Lead masking | Zero out 1-2 random leads | lead detachment (MIMIC has these) |
| Baseline wander | Sinusoid f < 1 Hz, amp < 0.1× | common artifact |

**Loss:** NT-Xent (SimCLR), τ = 0.07, **batch size 512** (large batch is critical — 1022 negatives per positive).

**Pretraining config:** AdamW, LR=3e-4, weight-decay=0.01, cosine decay, 5 warmup epochs, 100 total epochs (~3 h on A100). Save `pretrained_checkpoints/ecg_pretrain.pt` (encoder weights only).

**New files:**
| File | Purpose |
|---|---|
| [src/multimodal_aorta/data/ecg_pretrain_dataset.py](2026 Multi-Modal Project/src/multimodal_aorta/data/ecg_pretrain_dataset.py) | Dataset returning (view1, view2) augmented pairs over all 800 k MIMIC-IV-ECG records |
| [src/multimodal_aorta/training/pretrain_ecg.py](2026 Multi-Modal Project/src/multimodal_aorta/training/pretrain_ecg.py) | ProjectionHead, nt_xent_loss(), pretrain loop |
| [src/pretrain_ecg.py](2026 Multi-Modal Project/src/pretrain_ecg.py) | CLI entry (--epochs, --batch_size, --output_ckpt) |
| [scripts/slurm_pretrain_ecg.sh](2026 Multi-Modal Project/scripts/slurm_pretrain_ecg.sh) | SLURM job (24 h, A100) |

**Files modified:**
| File | Change |
|---|---|
| [models/ecg_encoder.py](2026 Multi-Modal Project/src/multimodal_aorta/models/ecg_encoder.py) | Add `ResNet1DEncoder.load_pretrained_weights(ckpt_path)` |
| [configs/default_config.py](2026 Multi-Modal Project/src/multimodal_aorta/configs/default_config.py) | Add `ModelConfig.ecg_pretrain_ckpt: Optional[str] = None` |
| [src/run_training.py](2026 Multi-Modal Project/src/run_training.py) | `--ecg_pretrain_ckpt` arg → passed through to encoder build |
| [models/full_model.py](2026 Multi-Modal Project/src/multimodal_aorta/models/full_model.py) | Use `lr_ecg_encoder=1e-5` (down from 5e-5) when pretrained to avoid catastrophic forgetting |

**Verification:** (a) pretrain loss decreases from `log(512) ≈ 6.2` toward ~2-3; (b) encoder loads cleanly into a fresh `ResNet1DEncoder` and produces a `(B, 768)` tensor; (c) supervised debug job starts cleanly with `--ecg_pretrain_ckpt`.

---

### Step 3 — Multi-task auxiliary heads *(fixes modality imbalance)*

This step picks **multi-task auxiliary heads** over the v2_update.md "two-stage training" proposal. Reasoning:

| | Two-stage (per v2_update.md) | Multi-task aux heads *(chosen)* |
|---|---|---|
| Training loop | Two loops, two checkpoint formats | One loop, one checkpoint |
| Stage-2 cohort | 522 patients fine-tune a fusion+CXR path → high overfit risk | All 522 contribute to fusion path; full 2,874 contribute to ECG path; full 550 to CXR path |
| Implementation complexity | New CLI, new SLURM script, new training module | Edit `full_model.py` + `losses.py` only |
| Regularization signal | Loses modality-dropout signal in stage 2 | Preserves modality dropout |

Both approaches solve the same underlying problem (ECG branch starved of data). Multi-task aux heads achieve the same goal with one training loop and no checkpoint management.

**Change:** Add two small auxiliary regression heads next to the fusion head:
```
                  ┌── head_ecg   (reads ECG embedding)   — trained on 2,874 ECG-bearing patients
ECG  ─┐           ├── head_cxr   (reads CXR embedding)   — trained on   550 CXR-bearing patients
      ├─ fusion ──┤
CXR  ─┘           └── head_fused (existing)              — trained on the joint signal
```
Loss = `w_f · L_fused + w_e · L_ecg + w_c · L_cxr` with per-sample masking by `has_ecg`/`has_cxr`. Suggested weights `(1.0, 0.5, 0.25)`.

**Files:** [models/full_model.py](2026 Multi-Modal Project/src/multimodal_aorta/models/full_model.py), [models/regression_head.py](2026 Multi-Modal Project/src/multimodal_aorta/models/regression_head.py), [training/losses.py](2026 Multi-Modal Project/src/multimodal_aorta/training/losses.py), [training/train.py](2026 Multi-Modal Project/src/multimodal_aorta/training/train.py).

**Expected:** ECG-only branch gets a real number on 2,874 patients (this is the "predict aorta from ECG" baseline we actually want). Fusion improves modestly because the ECG encoder no longer starves.

---

### Step 4 — Demographics token *(possibly the single biggest gain)*

**Data source (confirmed by inspection):** `aortic_labels.csv` has only `subject_id, ascending_diam_cm, sinus_diam_cm, any_dilation` — no demographics. [Data/patients.csv](Data/patients.csv) has `subject_id, gender, anchor_age, anchor_year` for every MIMIC-IV patient. **Merge by `subject_id`** at cohort-build time.

**Change:**
- In [data/dataset.py](2026 Multi-Modal Project/src/multimodal_aorta/data/dataset.py)::`build_cohort`, left-join `patients.csv` on `subject_id`. Add `DataConfig.patients_csv` pointing at `/scratch4/rsteven1/chenjia_echo_project/Data/patients.csv`.
- Compute `age_at_echo = anchor_age + (echo_year - anchor_year)`. Z-score age on the train split.
- Surface `(age_scaled, sex_M, sex_F)` as a 3-dim feature in `__getitem__`.
- Project to 768-dim and add as a 4th token in fusion: `[CLS, ECG, CXR, DEMO]`. `DEMO` token is always present (no mask token needed).

**Files:** [data/dataset.py](2026 Multi-Modal Project/src/multimodal_aorta/data/dataset.py), [models/fusion.py](2026 Multi-Modal Project/src/multimodal_aorta/models/fusion.py), [models/full_model.py](2026 Multi-Modal Project/src/multimodal_aorta/models/full_model.py), [configs/default_config.py](2026 Multi-Modal Project/src/multimodal_aorta/configs/default_config.py).

**Expected:** Probably the **biggest single jump in the whole rollout**. A linear model on age + sex alone often hits R² ≈ 0.15-0.25 for aortic root. Run a demographics-only ablation (drop ECG + CXR tokens) to bound how much signal is just age/sex.

---

### Step 5 — Cross-attention fusion *(adopted from v2_update.md)*

This step **adopts the cross-attention fusion idea** from v2_update.md — it's a genuine architectural upgrade I had missed.

**Problem:** Current fusion uses `[CLS, ECG_cls, CXR_cls]` — CXR contributes a single 768-dim CLS pool. Aortic dilation has **localized image signal** (the silhouette of the great vessels in the upper mediastinum); the CLS pool throws that away.

**Change:** Let the ECG embedding query over all **197 RAD-DINO patch tokens** so the model can learn which chest regions correlate with which ECG patterns.

```
ECG encoder → ecg_emb ∈ R^(B, 768)
CXR encoder → cxr_tokens ∈ R^(B, 197, 768)     [expose ALL patch tokens, not just CLS]
DEMO token  → demo_emb ∈ R^(B, 768)

CrossAttentionFusion:
  Query:   ecg_proj(ecg_emb).unsqueeze(1)        # (B, 1, 768)
  Key/Val: cxr_proj(cxr_tokens)                  # (B, 197, 768)
  Cross-attn out: (B, 1, 768) → squeeze → fused_ecg ∈ R^(B, 768)
  Final repr: LayerNorm(fused_ecg + ecg_emb + cxr_cls + demo_emb)
  → RegressionHead(768 → 2)
```

**Files:**
- [models/cxr_encoder.py](2026 Multi-Modal Project/src/multimodal_aorta/models/cxr_encoder.py) — add `forward_tokens()` returning `(B, 197, 768)` last hidden state (currently returns only CLS).
- New `models/cross_attention_fusion.py` — `CrossAttentionFusion` module.
- [models/full_model.py](2026 Multi-Modal Project/src/multimodal_aorta/models/full_model.py) — support `fusion_type="transformer"` (existing) or `"cross_attention"` (new) via config.
- [configs/default_config.py](2026 Multi-Modal Project/src/multimodal_aorta/configs/default_config.py) — `ModelConfig.fusion_type: str = "transformer"`, switch to `"cross_attention"` for this step.

Keep `FusionTransformer` intact for backward compatibility with earlier steps' checkpoints.

**Expected:** Genuine fusion (the both-vs-ECG-only ablation gap finally widens beyond noise). Spatial information from CXR is preserved.

---

### Step 6 — ECG-FM swap *(optional, only if Steps 1-5 aren't enough)*

If after Steps 1-5 the model is still bottlenecked on ECG representation quality, swap SimCLR-pretrained ResNet1D for [wanglab/ecg-fm](https://huggingface.co/wanglab/ecg-fm) (90 M params, wav2vec 2.0, pretrained on MIMIC-IV-ECG + PhysioNet 2021 ~ 1.5 M ECGs).

**Why it's last, not first:** ECG-FM is a strict upgrade *in expectation* (more capacity, more pretraining data, better SSL objective), but the integration is risky — `fairseq_signals` won't install cleanly in our env, and a hand-rolled wav2vec 2.0 module + state-dict remap is ~250 LoC of careful work. Land it on top of a *stabilized harness* so we can attribute any regression cleanly.

**Plan for [models/ecg_encoder.py](2026 Multi-Modal Project/src/multimodal_aorta/models/ecg_encoder.py)::ECGFMEncoder:**
- Hand-rolled wav2vec 2.0 module (7-layer conv feature extractor + 12-layer Transformer, d=768, 12 heads, ffn=3072). New helper file `models/ecgfm_state_dict_remap.py` for fairseq → local key remap.
- **Input contract:** ECG-FM trained on 5 s @ 500 Hz = 2,500 samples. We feed 10 s = 5,000 samples. Crop centre 5 s inside `forward()` — don't change `DataConfig.ecg_target_length` (keeps SimCLR/ResNet baseline runnable for ablations).
- **Pooling:** masked mean-pool over time of the final transformer layer (no [CLS] in ECG-FM). Output 768-dim, matches `d_model`.
- **LLRD freeze schedule:** freeze conv feature extractor permanently; freeze transformer blocks 0-7 for epochs 0-4; unfreeze with `lr=1e-5` (blocks 0-7) and `lr=5e-5` (blocks 8-11).
- **Two sub-runs:** (6a) ECG-FM fully frozen — apples-to-apples vs SimCLR-pretrained ResNet1D, also de-risks the load path; (6b) ECG-FM with LLRD unfreeze.

**Files:** [models/ecg_encoder.py](2026 Multi-Modal Project/src/multimodal_aorta/models/ecg_encoder.py), [models/full_model.py](2026 Multi-Modal Project/src/multimodal_aorta/models/full_model.py) (param groups), [configs/default_config.py](2026 Multi-Modal Project/src/multimodal_aorta/configs/default_config.py), new `models/ecgfm_state_dict_remap.py`.

---

## Creative / future ideas

These are tracked for follow-up papers, not the immediate rollout:

- **Auxiliary clinical targets** *(from v2_update.md Step 4)* — multi-task with LV mass index, diastolic dysfunction grade, or tricuspid regurgitation velocity. Requires first inspecting `structured-measurement.csv::measurement_description` for coverage in the 2,902 cohort. The novelty value here is real (LVEF from ECG is over-studied; these targets aren't).
- **ECG augmentation at supervised fine-tuning time** — random crop, lead dropout, baseline wander as light regularization (separate from SimCLR augmentations).
- **CXR Mixup at the embedding level** between same-target-bin patients.
- **Cross-modal CLIP-style contrastive pretraining** on the 522 paired patients before regression fine-tuning.
- **Test-time augmentation:** average over N ECG crops × M CXR augmentations.
- **Heteroscedastic head:** output `(μ, log σ²)` per target, Gaussian NLL loss. Gives calibrated uncertainty and improves AUROC for the dilation classification proxy.
- **Simplify fusion when overfitting persists:** drop fusion transformer to 1 layer / 4 heads / ffn=1024 if Step 5's cross-attention adds parameter pressure on the 522-patient joint cohort.

---

## Sequencing summary

| Step | Description | Files touched | Effort | Expected MAE delta |
|---|---|---|---|---|
| 1 | Target z-score + MSE | 4 | ~30 LoC | -0.04 to -0.08 cm |
| 2 | SimCLR pretrain ResNet1D | 4 new + 4 edits | 3-4 h + 3 h GPU | -0.02 to -0.05 cm |
| 3 | Multi-task aux heads | 4 edits | ~80 LoC | -0.02 to -0.04 cm |
| 4 | Demographics token | 4 edits | ~50 LoC | -0.05 to -0.10 cm *(possibly biggest)* |
| 5 | Cross-attention fusion | 2 new + 3 edits | ~150 LoC | -0.02 to -0.05 cm; widens ablation gap |
| 6 *(opt)* | ECG-FM swap | 1 new + 3 edits | ~1 week | -0.02 to -0.05 cm if Step 2 wasn't enough |

**Acceptance target after Steps 1-5:** val MAE_root ≤ 0.30 cm; both-vs-ECG-only ablation gap ≥ 0.03 cm; val R² ≥ 0.30.

Each step is committed on its own branch (`step1-zscore`, `step2-simclr`, …) so any regression can be bisected.

---

## Decisions resolved

1. **Sequencing**: 1 → 2 → 3 → 4 → 5; Step 6 only if needed (user confirmed recommended order).
2. **Demographics source**: merge from [Data/patients.csv](Data/patients.csv) by `subject_id` (file inspected; has `gender, anchor_age, anchor_year`).
3. **ECG encoder strategy**: **SimCLR self-pretrain first (Step 2), ECG-FM as optional Step 6**. The v2_update.md proposal wins on near-term risk/reward; the ECG-FM swap remains available as a follow-up if SimCLR isn't enough.
4. **Fusion approach**: keep transformer fusion through Steps 2-4; upgrade to cross-attention over patch tokens in Step 5.
5. **Two-stage vs multi-task aux heads**: multi-task aux heads (one training loop, full-cohort gradient on the ECG branch).

---

## Verification per step

1. Run a 2-epoch debug job on a small subset → forward/backward + checkpoint I/O.
2. Submit full 50-epoch run via `sbatch scripts/slurm_train.sh`.
3. Report three numbers per target: **val MAE, val R², ablation gap (both - ECG_only)**.
4. Commit each step on its own branch; tag the best-val-MAE checkpoint for later loading.
