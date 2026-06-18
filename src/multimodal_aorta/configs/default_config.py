"""
All hyperparameters and path configuration in one place.
Import and override individual keys as needed before passing to training.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataConfig:
    # --- Paths ---
    echo_labels_path: str = (
        "/scratch4/rsteven1/chenjia_echo_project/2026 Multi-Modal Project/data/echo/aortic_labels.csv"
    )
    echo_structured_path: str = (
        "/scratch4/rsteven1/chenjia_echo_project/2026 Multi-Modal Project/data/echo/structured-measurement.csv"
    )
    ecg_root: str = (
        "/scratch4/rsteven1/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0"
    )
    # Primary CXR source: your complete MIMIC-CXR-JPG download (p10–p19, all patients).
    # Checks for .png then .jpg then .jpeg under this root.
    # Path structure: <cxr_png_root>/p<partition>/p<subject_id>/s<study_id>/<dicom_id>.<ext>
    cxr_png_root: str = "/scratch4/rsteven1/MIMIC_CXR_JPG_cohort/files"
    # Secondary raster root — set to None once the full download above is complete.
    # Only needed if you want to fall back to the old partial MIMIC_CXR_GS_512 during download.
    cxr_extra_png_root: Optional[str] = None
    # Fallback CXR source: original DICOMs (used when no raster image is found).
    # Path structure: <cxr_dicom_root>/files/p<partition>/p<subject_id>/s<study_id>/<dicom_id>.dcm
    cxr_dicom_root: str = "/scratch4/rsteven1/MIMIC_CXR_GS"

    # Shared index CSVs (same for both formats)
    cxr_record_list: str = "/scratch4/rsteven1/MIMIC_CXR_GS/cxr-record-list.csv"

    # Cache for CXR file metadata (ViewPosition, StudyDate, resolved path, format).
    # Built once on first run, reused on subsequent runs.
    cxr_metadata_cache: str = (
        "/scratch4/rsteven1/chenjia_echo_project/2026 Multi-Modal Project/data/cxr_metadata_cache.csv"
    )

    # Official MIMIC-CXR metadata CSV (mimic-cxr-2.0.0-metadata.csv.gz).
    # Provides ViewPosition + StudyDate for all 377K images, including JPG-only files.
    # Download from PhysioNet: physionet.org/content/mimic-cxr-jpg/2.1.0/
    cxr_metadata_csv: str = (
        "/scratch4/rsteven1/MIMIC_CXR_JPG_cohort/mimic-cxr-2.0.0-metadata.csv.gz"
    )

    # --- Column names (derived from inspecting aortic_labels.csv) ---
    # sinus_diam_cm  = aortic root (sinus of Valsalva)
    # ascending_diam_cm = ascending aorta
    col_root: str = "sinus_diam_cm"
    col_asc: str = "ascending_diam_cm"

    # --- Temporal alignment ---
    max_days_offset: int = 180  # ECG/CXR must be within this many days of the echo
                                # 180 days balances sample size vs temporal coherence;
                                # use --max_days_offset 365 to expand if needed

    # --- Label QC clip ranges (cm) ---
    root_range: tuple = (1.5, 6.0)
    asc_range: tuple = (1.5, 7.0)

    # --- PCLR precomputed embeddings ---
    # Path to {subject_id: tensor(320,)} .pt file produced by extract_pclr_embeddings.py.
    # When set, ecg_encoder is automatically treated as "pclr_frozen".
    pclr_embeddings_path: Optional[str] = None

    # --- RAD-DINO precomputed CXR embeddings ---
    # Path to {subject_id: tensor(768,)} .pt file produced by
    # extract_raddino_embeddings.py. When set, the CXR branch uses frozen
    # precomputed embeddings (cxr_encoder="raddino_frozen") instead of running
    # the 44M-param ViT end-to-end. This makes the trainable model tiny and is
    # the recommended setup given only ~520 patients have a CXR.
    cxr_embeddings_path: Optional[str] = None

    # --- Target normalization (Step 1 redesign) ---
    # When True, targets are z-scored using stats computed on the train split,
    # and the model trains against z-scored targets with plain MSE. Reported
    # metrics (MAE / R² / AUROC) are de-normalized back to cm.
    target_normalize: bool = True

    # --- Splits ---
    train_frac: float = 0.70
    val_frac: float = 0.15
    # test_frac is implicit: 1 - train_frac - val_frac = 0.15
    split_seed: int = 42
    stratify_bins: int = 3  # equal-frequency bins on aortic root for stratification
                            # 3 bins avoids the "1-member class" failure on small cohorts

    # --- ECG preprocessing ---
    ecg_target_length: int = 5000   # 10 s × 500 Hz
    ecg_target_fs: int = 500        # Hz
    ecg_n_leads: int = 12
    ecg_norm_clip: float = 4.0      # clip to ±N std before z-scoring

    # --- CXR preprocessing ---
    cxr_image_size: int = 224
    cxr_imagenet_mean: tuple = (0.485, 0.456, 0.406)
    cxr_imagenet_std: tuple = (0.229, 0.224, 0.225)
    # Training augmentations
    cxr_aug_hflip_p: float = 0.5
    cxr_aug_rotate_deg: float = 10.0
    cxr_aug_brightness: float = 0.1
    cxr_aug_contrast: float = 0.1


@dataclass
class ModelConfig:
    # Common embedding dimension — both encoders output this dim.
    # 768 matches RAD-DINO and future ECG-FM, avoiding a lossy bottleneck projection.
    d_model: int = 768

    # ECG encoder
    # "resnet1d"    — 1D ResNet-34, random init (or SimCLR-pretrained via ecg_pretrain_ckpt)
    # "pclr_frozen" — frozen PCLR embeddings + learnable Linear(320→768) projection
    # "ecgfm"       — future ECG-FM integration
    ecg_encoder: str = "resnet1d"
    ecg_out_dim: int = 768          # output dim of ECGEncoder (must match d_model)
    # Path to SimCLR-pretrained encoder weights (output of pretrain_ecg.py).
    # When set, AortaModel loads these weights and uses lr_ecg_pretrained LR.
    ecg_pretrain_ckpt: Optional[str] = None

    # CXR encoder
    # "rad_dino"        — RAD-DINO ViT-B/16 run end-to-end (frozen as feature
    #                     extractor by default; see cxr_freeze_blocks below)
    # "raddino_frozen"  — frozen precomputed RAD-DINO embeddings + learnable
    #                     Linear(768→d_model). Set via DataConfig.cxr_embeddings_path.
    cxr_encoder: str = "rad_dino"
    # Local snapshot of microsoft/rad-dino on /scratch4 (downloaded once via
    # huggingface_hub.snapshot_download). Pointed at a local dir rather than the
    # hub id so loads work offline on compute nodes and survive ~/.cache wipes
    # (the home HF cache was cleared for quota; do not depend on it). To re-fetch:
    #   HF_HOME=/scratch4/rsteven1/chenjia_echo_project/hf_home python -c \
    #     "from huggingface_hub import snapshot_download; \
    #      snapshot_download('microsoft/rad-dino', local_dir='<this path>')"
    cxr_model_name: str = (
        "/scratch4/rsteven1/chenjia_echo_project/2026 Multi-Modal Project/"
        "pretrained_checkpoints/rad-dino"
    )
    cxr_out_dim: int = 768          # RAD-DINO CLS token dim
    # RAD-DINO is kept FULLY FROZEN as a feature extractor. Fine-tuning a
    # 44M-param ViT on the ~520 patients who have a CXR overfit catastrophically
    # in earlier runs (val loss jumped the moment blocks unfroze). 12 = all
    # ViT-B blocks frozen; unfreeze epoch set far beyond training horizon.
    cxr_freeze_blocks: int = 12     # freeze ALL transformer blocks (ViT-B has 12)
    cxr_unfreeze_epoch: int = 10_000  # effectively never unfreeze

    # Fusion transformer — kept small: only 2 tokens to fuse over ~520 dual-modality
    # patients, so a deep/wide transformer just memorizes. 1 layer + narrow FFN.
    nhead: int = 8                  # 768 / 8 = 96 per head
    num_fusion_layers: int = 1
    dim_feedforward: int = 512      # narrow FFN to limit capacity
    fusion_dropout: float = 0.2

    # Regression head
    head_hidden_dim: int = 256
    head_dropout: float = 0.3

    n_targets: int = 2              # [aortic_root_cm, ascending_aorta_cm]


@dataclass
class TrainConfig:
    batch_size: int = 32
    # Earlier runs overfit hard: val loss bottomed by epoch ~4 then climbed for
    # 20 more epochs. Short horizon + tight patience selects the real best model.
    num_epochs: int = 20
    early_stop_patience: int = 8

    # Per-parameter-group learning rates
    # ECG encoder trains from scratch — keep LR modest to avoid early overfitting
    lr_ecg_encoder: float = 5e-5
    # When a SimCLR pretrained checkpoint is loaded, this lower LR is used for
    # the ECG encoder to prevent catastrophic forgetting of pretrained features.
    lr_ecg_pretrained: float = 1e-5
    lr_cxr_encoder: float = 2e-5
    lr_fusion: float = 1e-4         # covers projection layers, fusion transformer, head

    weight_decay: float = 0.1    # stronger regularization for the small dataset
    grad_clip_norm: float = 1.0
    warmup_epochs: int = 3       # short warmup to match the 20-epoch horizon

    # Loss
    # Note: as of Step 1, training uses plain MSE on z-scored targets.
    # huber_delta is retained for backward compatibility with old checkpoints
    # but is no longer used by the loss.
    huber_delta: float = 0.5

    # Modality dropout (training only)
    p_drop_ecg: float = 0.15
    p_drop_cxr: float = 0.15

    # I/O
    num_workers: int = 4
    pin_memory: bool = True

    output_dir: str = "outputs/multimodal_aorta"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
