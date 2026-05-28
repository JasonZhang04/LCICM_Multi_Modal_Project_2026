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
    ecg_encoder: str = "resnet1d"   # "resnet1d" | "ecgfm" (future)
    ecg_out_dim: int = 768          # output dim of ECGEncoder (must match d_model)

    # CXR encoder (RAD-DINO: ViT-B/16 pretrained on 882K chest X-rays)
    cxr_model_name: str = "microsoft/rad-dino"
    cxr_out_dim: int = 768          # RAD-DINO CLS token dim
    cxr_freeze_blocks: int = 6      # freeze first N ViT blocks for first cxr_unfreeze_epoch epochs
    cxr_unfreeze_epoch: int = 10    # unfreeze all blocks after warmup is fully settled

    # Fusion transformer
    nhead: int = 8                  # 768 / 8 = 96 per head
    num_fusion_layers: int = 3
    dim_feedforward: int = 2048     # ~2.7× d_model
    fusion_dropout: float = 0.1

    # Regression head
    head_hidden_dim: int = 256
    head_dropout: float = 0.2

    n_targets: int = 2              # [aortic_root_cm, ascending_aorta_cm]


@dataclass
class TrainConfig:
    batch_size: int = 32
    num_epochs: int = 100
    early_stop_patience: int = 20

    # Per-parameter-group learning rates
    # ECG encoder trains from scratch — keep LR modest to avoid early overfitting
    lr_ecg_encoder: float = 5e-5
    lr_cxr_encoder: float = 2e-5
    lr_fusion: float = 1e-4         # covers projection layers, fusion transformer, head

    weight_decay: float = 0.05
    grad_clip_norm: float = 1.0
    warmup_epochs: int = 10

    # Loss
    huber_delta: float = 0.5

    # Modality dropout (training only)
    p_drop_ecg: float = 0.15
    p_drop_cxr: float = 0.15

    # I/O
    num_workers: int = 4
    pin_memory: bool = True

    output_dir: str = (
        "/scratch4/rsteven1/chenjia_echo_project/2026 Multi-Modal Project/outputs/multimodal_aorta"
    )


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
