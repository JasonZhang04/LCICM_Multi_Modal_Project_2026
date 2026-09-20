# Multimodal Aorta v2+ — Architectural Improvements Plan

## Context

Baseline run (v1) completed. Best val MAE = 0.394 cm, R² ≈ 0.07 — essentially predicting the mean.
Three fundamental problems identified:

1. **ECG encoder trains from scratch** on ~1,400 patients. No pretraining. This is the primary bottleneck.
2. **Modality imbalance**: 81% ECG-only training batches; CXR branch barely trains.
3. **Fusion architecture**: CXR CLS token is a coarse summary; cross-attention over patch tokens would give richer spatial interaction.

Plan covers **four improvement steps** implemented in order, with v1 preserved as baseline.

---

## Step 1 — Self-supervised ECG Pre-training (THIS STEP)

### Why
`fairseq-signals` is not pip-installable. Instead, pre-train the existing ResNet1DEncoder on all
800K MIMIC-IV-ECG records (161K patients) using SimCLR-style contrastive learning. Pre-training
and fine-tuning data come from the same institution and hardware — perfect distribution match.

### Architecture during pre-training
```
ResNet1DEncoder (same 16M-param architecture) → h ∈ R^768
ProjectionHead: Linear(768→512) → BN → GELU → Linear(512→128) → z ∈ R^128
```
Projection head is discarded after pre-training. Only encoder weights are transferred.

### ECG Augmentations (two independent views per ECG)
| Augmentation | Range | Notes |
|---|---|---|
| Time crop | Keep 4000/5000 samples, zero-pad | simulates partial recording |
| Amplitude scale | U[0.7, 1.3]× per lead | simulates electrode contact variation |
| Gaussian noise | σ = U[0, 0.02] × signal std | simulates noise |
| Lead masking | Zero out 1–2 random leads | simulates lead detachment |
| Baseline wander | Add sinusoid f < 1 Hz, amp < 0.1× | common artifact |

### Loss — NT-Xent (SimCLR)
Temperature τ = 0.07. Batch size = 512. Large batches are critical for contrastive learning —
512 gives 1022 negatives per positive pair.

### Training config
- Optimizer: AdamW, LR=3e-4, weight decay=0.01, cosine decay, warmup 5 epochs
- 100 epochs on all 800K ECGs → ~2–3h on A100
- Save: `pretrained_checkpoints/ecg_pretrain.pt` (encoder weights only, not projection head)

### New files to create
| File | Purpose |
|---|---|
| `src/multimodal_aorta/data/ecg_pretrain_dataset.py` | Dataset returning (view1, view2) augmented pairs |
| `src/multimodal_aorta/training/pretrain_ecg.py` | ProjectionHead, nt_xent_loss(), pretrain() loop |
| `src/pretrain_ecg.py` | CLI entry point (--epochs, --batch_size, --output_ckpt) |
| `scripts/slurm_pretrain_ecg.sh` | SLURM job (24h, A100) |

### Files to modify
| File | Change |
|---|---|
| `models/ecg_encoder.py` | Add `load_pretrained_weights(ckpt_path)` to ResNet1DEncoder |
| `configs/default_config.py` | Add `ecg_pretrain_ckpt: Optional[str] = None` to ModelConfig |
| `src/run_training.py` | Add `--ecg_pretrain_ckpt` arg; pass to build_ecg_encoder() |
| `models/full_model.py` | Pass pretrain_ckpt through to build_ecg_encoder() |

### Integration with fine-tuning
After pre-training:
```bash
sbatch scripts/slurm_train.sh --ecg_pretrain_ckpt pretrained_checkpoints/ecg_pretrain.pt \
    --output_dir outputs/multimodal_aorta_pretrained_ecg
```
The encoder weights are loaded and **fine-tuned** (not frozen) during supervised training.
Use `lr_ecg_encoder=1e-5` (lower than default 5e-5) for pretrained encoder to avoid catastrophic forgetting.

### Verification
```python
# 1. Check pre-training loss decreases from ~log(512)≈6.2 toward ~2-3
# 2. Check pretrain checkpoint encoder loads cleanly:
enc = ResNet1DEncoder()
enc.load_pretrained_weights("pretrained_checkpoints/ecg_pretrain.pt")
out = enc(torch.randn(4, 12, 5000))
assert out.shape == (4, 768)
# 3. Run fine-tuning debug job with --ecg_pretrain_ckpt and confirm it starts without error
```

---

## Step 2 — Two-Stage Training (separate comparison experiment)

### Why
81% of training batches have no CXR. The mask token for absent CXR carries no information.
Two-stage training fixes this by separating the two learning problems:

- **Stage 1**: Train ECG encoder + unimodal ECG head on all 2,902 patients → optimal ECG representations
- **Stage 2**: Freeze ECG encoder, train CXR encoder + fusion + fusion head on 522 both-modality patients only

Results saved to `outputs/two_stage/` — does not overwrite v1 or pretrain outputs.

### New files
| File | Purpose |
|---|---|
| `src/multimodal_aorta/training/train_two_stage.py` | Two-stage training loop |
| `src/run_two_stage.py` | CLI entry point |
| `scripts/slurm_two_stage.sh` | SLURM job |

### Architecture change
Add a second head to `AortaModel`: `ecg_head` (same LayerNorm→256→2 as main head).
Stage 1 trains `ecg_encoder + ecg_head`. Stage 2 freezes `ecg_encoder`, trains `cxr_encoder + fusion + head`.

---

## Step 3 — Cross-Attention Fusion

### Why
Current fusion: `[CLS, ECG_cls, CXR_cls]` — the CXR contributes a single pooled 768-dim vector.
Cross-attention fusion: ECG embedding attends over all 197 RAD-DINO patch tokens → richer spatial
interaction (model can learn which chest regions correlate with which ECG patterns).

### Architecture
```
ECG encoder → ecg_emb ∈ R^(B,768)
CXR encoder → cxr_tokens ∈ R^(B,197,768)   [new: expose all patch tokens, not just CLS]

CrossAttentionFusion:
  Query:  ecg_proj(ecg_emb).unsqueeze(1)       # (B, 1, 768)
  Key/Val: cxr_proj(cxr_tokens)                # (B, 197, 768)
  Cross-attn output: (B, 1, 768) → squeeze → fused_ecg ∈ R^(B,768)
  Final repr: LayerNorm(fused_ecg + ecg_emb + cxr_cls)  # residual + CXR CLS
  → RegressionHead(768→2)
```

### New/modified files
| File | Change |
|---|---|
| `models/cxr_encoder.py` | Add `forward_tokens()` returning (B,197,768) last hidden state |
| `models/cross_attention_fusion.py` | New `CrossAttentionFusion` module |
| `models/full_model.py` | Support `fusion_type="transformer"` or `"cross_attention"` via config |
| `configs/default_config.py` | Add `fusion_type: str = "transformer"` to ModelConfig |

Keep original `FusionTransformer` intact for v1/v2 backward compatibility.

---

## Step 4 — Multi-task Learning (deferred)

Predicting LVEF from ECG is well-studied (established literature). For novelty, consider:
- LV mass index (less studied in ECG+CXR multimodal context)
- Diastolic dysfunction grade (categorical, clinically impactful)
- Tricuspid regurgitation velocity (proxy for pulmonary pressures, underexplored)

Requires first inspecting `structured-measurement.csv` `measurement_description` column to see
which measurements have sufficient coverage in the 2,902-patient cohort. Defer until Steps 1–3
are run and compared.

---

## Implementation Order for This Session

1. **Step 1 only** — ECG pre-training. Expected to take ~3–4h to implement + ~3h A100 GPU time.
2. Submit pre-training job, then immediately submit fine-tuning job with `--ecg_pretrain_ckpt`.
3. Compare pretrained-ECG vs random-init ECG using same v2 hyperparameters.
4. Steps 2–4 in subsequent sessions.
