# Sub-task 3: Model Architecture Plan (Updated)

## Context
Sub-tasks 1 + 2 done. Implementing model architecture now. Key design decision from user:
- ECG encoder uses an **abstract interface** with two concrete implementations
- Implement **ResNet1DEncoder** now (baseline, trains from scratch)
- **ECGFMEncoder** (ECG-FM with fairseq_signals) to be added later as drop-in
- Both encoders must output the same shape so downstream code is encoder-agnostic

---

## What Gets Downloaded

### RAD-DINO — `microsoft/rad-dino` (~330 MB)
ViT-B/16 pretrained on 882K chest X-rays via DINOv2 (Microsoft Research).
Outputs **768-dim** CLS token. Cleaner to use than BioViL-T (whose HuggingFace
package only exposes the text encoder, not vision).

### ECG encoder — No download now
ResNet1DEncoder trains from random init. ECG-FM will be downloaded in a future
session when we implement ECGFMEncoder.

---

## Key Dimension Decision

Both encoders output **768-dim** vectors:
- RAD-DINO CLS token → 768
- ResNet1DEncoder final layer → **768** (not 512 as original spec — aligns with ECG-FM
  which also outputs 768, making the interface truly drop-in)
- ECGFMEncoder (future) → 768

Fusion transformer `d_model` = **768** (avoids an unnecessary 768→512 bottleneck
projection; both encoders already output 768 so the projection layers become
768→768 learnable alignment transforms, which is still useful but doesn't compress).

`ModelConfig` changes from spec:
- `ecg_out_dim`: 512 → **768**  
- `d_model`: 512 → **768**
- `nhead`: 8 (768 / 8 = 96 — valid)
- `dim_feedforward`: 1024 → **2048** (maintain ~2.7× ratio of spec)

Regression head stays: 768 → 256 → 2.

---

## Files to Implement

### `models/ecg_encoder.py`

```
ECGEncoder (abstract nn.Module)
  └── output contract: (batch, 12, 5000) → (batch, 768)

ResNet1DEncoder(ECGEncoder)   ← implement now
  Stem: Conv1d(12→64, k=15, s=2) + BN + GELU + MaxPool2
  Stage 1: ResBlock(64→64,  k=7) ×3            (ResNet-34 depth)
  Stage 2: ResBlock(64→128, k=7) ×4, stride=2
  Stage 3: ResBlock(128→256, k=7) ×6, stride=2
  Stage 4: ResBlock(256→768, k=7) ×3, stride=2
  GlobalAvgPool → (batch, 768)

ECGFMEncoder(ECGEncoder)      ← stub only, raises NotImplementedError
  (loads fairseq_signals checkpoint, to be implemented later)
```

ResNet-34 depth (3-4-6-3 blocks) gives enough capacity without over-parameterizing
for a from-scratch ECG model. Output dim 768 matches the interface contract.

### `models/cxr_encoder.py`

```
CXREncoder(nn.Module)
  - Loads microsoft/rad-dino (ViT-B/16)
  - forward(pixel_values) → CLS token → (batch, 768)
  - freeze_blocks(n): freezes first n transformer blocks
  - unfreeze_all(): unfreezes everything
  
Note on input resolution: RAD-DINO is DINOv2-based and supports any resolution
via interpolated position embeddings. We pass 224×224 (already resized in preprocessing)
— no need to change to 518×518. The model handles this gracefully.
```

### `models/fusion.py`

```
FusionTransformer(nn.Module)
  Learned parameters:
    ecg_proj:   Linear(768 → 768)   # learnable alignment, not just identity
    cxr_proj:   Linear(768 → 768)
    cls_token:  Parameter(1, 768)   # learned aggregator token
    ecg_type_emb: Parameter(1, 768) # modality tag
    cxr_type_emb: Parameter(1, 768)
    cls_type_emb: Parameter(1, 768)
    ecg_mask_token: Parameter(1, 768)  # substituted when ECG missing
    cxr_mask_token: Parameter(1, 768)  # substituted when CXR missing

  forward(ecg_emb, cxr_emb, has_ecg, has_cxr):
    1. Replace missing modalities with mask tokens (per-sample in batch)
    2. Apply linear projections
    3. Add modality type embeddings
    4. Prepend CLS token
    5. TransformerEncoder (3 layers, nhead=8, d_ff=2048, pre-norm, dropout=0.1)
    6. Return CLS token output: (batch, 768)
```

### `models/regression_head.py`

```
RegressionHead(nn.Module)
  LayerNorm(768) → Linear(768→256) → GELU → Dropout(0.2) → Linear(256→2)
  Output: (batch, 2) raw predictions in cm [root_diam, asc_diam]
```

### `models/full_model.py`

```
AortaModel(nn.Module)
  __init__(model_cfg, train_cfg):
    self.ecg_encoder = ResNet1DEncoder(out_dim=768)
    self.cxr_encoder = CXREncoder(model_name=cfg.cxr_model_name)
    self.fusion = FusionTransformer(d_model=768, ...)
    self.head = RegressionHead(in_dim=768, ...)
    self.ecg_mask_token, self.cxr_mask_token  # (inside FusionTransformer)

  forward(ecg, cxr, has_ecg, has_cxr, p_drop_ecg=0.15, p_drop_cxr=0.15):
    # Training-time modality dropout (applied on top of real has_ecg/has_cxr)
    if self.training:
        randomly zero-out has_ecg or has_cxr with prob p_drop_*
    ecg_emb = self.ecg_encoder(ecg)           # (B, 768)
    cxr_emb = self.cxr_encoder(cxr)           # (B, 768)
    fused = self.fusion(ecg_emb, cxr_emb, has_ecg, has_cxr)  # (B, 768)
    return self.head(fused)                    # (B, 2)

  get_param_groups(lr_ecg, lr_cxr, lr_fusion):
    # returns list of dicts for AdamW with per-component LRs

  set_cxr_encoder_frozen(epoch):
    # freeze blocks 0–5 if epoch < unfreeze_epoch, else unfreeze all
```

### Config changes in `configs/default_config.py`
```python
ModelConfig:
  d_model: 768          # was 512
  ecg_out_dim: 768      # was 512
  cxr_out_dim: 768      # unchanged (RAD-DINO output)
  dim_feedforward: 2048 # was 1024
  cxr_model_name: "microsoft/rad-dino"   # was BiomedVLP-BioViL-T
```

---

## Verification
Single-batch forward pass with both modalities and missing CXR:
```python
model = AortaModel(model_cfg, train_cfg)
ecg = torch.randn(4, 12, 5000)
cxr = torch.randn(4, 3, 224, 224)
has_ecg = torch.tensor([True, True, True, True])
has_cxr = torch.tensor([True, False, True, False])  # 2 patients lack CXR
out = model(ecg, cxr, has_ecg, has_cxr)
assert out.shape == (4, 2)
```
