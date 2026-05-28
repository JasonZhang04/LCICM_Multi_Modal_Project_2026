"""
Preprocessing functions for ECG waveforms and chest X-ray DICOMs.

Each function is stateless and deterministic (except augmentation, which is
controlled by is_train). They are passed as transform callables to AortaDataset.
"""
import logging
import os
from typing import TYPE_CHECKING

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

if TYPE_CHECKING:
    from multimodal_aorta.configs.default_config import DataConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ECG preprocessing
# ---------------------------------------------------------------------------

def load_ecg(path: str, cfg: "DataConfig") -> torch.Tensor:
    """
    Load a 12-lead ECG waveform from a MIMIC IV-ECG path prefix (.hea/.dat).

    Steps:
      1. Read with wfdb.
      2. Resample to cfg.ecg_target_fs (500 Hz) if needed.
      3. Select or pad/crop to cfg.ecg_target_length samples.
      4. Robust per-lead z-score normalisation (clip ±cfg.ecg_norm_clip std).
      5. Fill missing leads with zeros.

    Returns
    -------
    torch.Tensor of shape (12, 5000), dtype float32.
    """
    import wfdb
    from scipy.signal import resample

    record = wfdb.rdrecord(path)
    signal = record.p_signal  # (n_samples, n_leads), may be None for some leads
    fs = record.fs
    n_leads_present = signal.shape[1] if signal is not None else 0

    # --- Resample to target fs ---
    target_len = cfg.ecg_target_length
    if fs != cfg.ecg_target_fs and signal is not None:
        n_target = int(round(signal.shape[0] * cfg.ecg_target_fs / fs))
        signal = resample(signal, n_target, axis=0)

    # --- Pad / crop to target_len ---
    if signal is not None:
        n_samples = signal.shape[0]
        if n_samples < target_len:
            pad = np.zeros((target_len - n_samples, signal.shape[1]), dtype=np.float32)
            signal = np.concatenate([signal, pad], axis=0)
        elif n_samples > target_len:
            # Take central window
            start = (n_samples - target_len) // 2
            signal = signal[start : start + target_len]

    # --- Build (12, target_len) array, filling missing leads with zeros ---
    out = np.zeros((cfg.ecg_n_leads, target_len), dtype=np.float32)
    lead_mask = np.zeros(cfg.ecg_n_leads, dtype=bool)

    if signal is not None:
        n_leads = min(signal.shape[1], cfg.ecg_n_leads)
        for i in range(n_leads):
            lead = signal[:, i].astype(np.float32)
            # Replace NaN (missing samples) with 0 before normalising
            nan_mask = np.isnan(lead)
            if nan_mask.all():
                continue  # completely missing lead — leave as zeros
            if nan_mask.any():
                lead[nan_mask] = 0.0
            lead_mask[i] = True

            # Robust z-score: clip outliers before computing stats
            mu = np.nanmean(lead)
            std = np.nanstd(lead) + 1e-8
            lead_clipped = np.clip(lead, mu - cfg.ecg_norm_clip * std, mu + cfg.ecg_norm_clip * std)
            mu2 = lead_clipped.mean()
            std2 = lead_clipped.std() + 1e-8
            out[i] = (lead - mu2) / std2

    return torch.from_numpy(out)  # (12, 5000) float32


# ---------------------------------------------------------------------------
# CXR preprocessing
# ---------------------------------------------------------------------------

def _load_pil_from_path(path: str) -> "Image.Image":
    """
    Load a CXR image as a PIL grayscale image from either a PNG or DICOM file.
    Returns a PIL Image in mode 'L' (8-bit grayscale).
    """
    from PIL import Image
    ext = os.path.splitext(path)[1].lower()

    if ext in (".png", ".jpg", ".jpeg"):
        img = Image.open(path).convert("L")
        return img

    # DICOM path (.dcm)
    import pydicom
    ds = pydicom.dcmread(path)
    pixel_array = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    pixel_array = pixel_array * slope + intercept
    photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
    if photometric == "MONOCHROME1":
        pixel_array = pixel_array.max() - pixel_array
    lo, hi = pixel_array.min(), pixel_array.max()
    if hi > lo:
        pixel_array = (pixel_array - lo) / (hi - lo) * 255.0
    pixel_array = pixel_array.clip(0, 255).astype(np.uint8)
    return Image.fromarray(pixel_array, mode="L")


def load_cxr(path: str, cfg: "DataConfig", is_train: bool = False) -> torch.Tensor:
    """
    Load a chest X-ray (PNG or DICOM), convert to 3-channel float32 tensor.

    Steps:
      1. Load as grayscale PIL Image (handles both .png and .dcm).
      2. Replicate to 3 channels (grayscale → RGB).
      3. Resize to cfg.cxr_image_size × cfg.cxr_image_size.
      4. Apply training augmentations or val/test normalization.
      5. Apply ImageNet normalization.

    Returns
    -------
    torch.Tensor of shape (3, H, W), dtype float32.
    """
    import os
    from PIL import Image

    img = _load_pil_from_path(path)
    # Replicate to RGB (BioViL-T expects 3-channel input)
    img = img.convert("RGB")

    # --- Transforms ---
    size = cfg.cxr_image_size
    mean = list(cfg.cxr_imagenet_mean)
    std = list(cfg.cxr_imagenet_std)

    if is_train:
        transform = T.Compose([
            T.Resize((size, size)),
            T.RandomHorizontalFlip(p=cfg.cxr_aug_hflip_p),
            T.RandomRotation(degrees=cfg.cxr_aug_rotate_deg),
            T.ColorJitter(
                brightness=cfg.cxr_aug_brightness,
                contrast=cfg.cxr_aug_contrast,
            ),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    else:
        transform = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    return transform(img)  # (3, H, W) float32


# ---------------------------------------------------------------------------
# Collate function for DataLoader (handles NaN in targets)
# ---------------------------------------------------------------------------

def collate_fn(batch):
    """
    Custom collate that stacks tensors and preserves NaN in targets.
    Also converts the modality_mask list-of-dicts to a dict-of-tensors.
    """
    ecg = torch.stack([b["ecg"] for b in batch])
    cxr = torch.stack([b["cxr"] for b in batch])
    target = torch.stack([b["target"] for b in batch])
    subject_ids = [b["subject_id"] for b in batch]

    has_ecg = torch.tensor([b["modality_mask"]["ecg"] for b in batch], dtype=torch.bool)
    has_cxr = torch.tensor([b["modality_mask"]["cxr"] for b in batch], dtype=torch.bool)

    return {
        "ecg": ecg,
        "cxr": cxr,
        "target": target,
        "has_ecg": has_ecg,
        "has_cxr": has_cxr,
        "subject_id": subject_ids,
    }
