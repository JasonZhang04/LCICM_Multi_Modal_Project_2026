"""
PyTorch Dataset for the JHU echocardiogram dataset.

Key design:
- Each DICOM video clip is a training sample
- All US DICOMs in a study share the same label
- SR DICOMs and single-frame DICOMs filtered out
- WeightedRandomSampler ensures balanced batches
- Study-level splitting prevents data leakage
- PanEcho preprocessing: 16 frames, 224x224, ImageNet-normalized
"""

import os
import json
import logging
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split

from data.dicom_utils import extract_frames_from_dicom

logger = logging.getLogger(__name__)


def filter_us_dicoms(dicom_paths: list[str]) -> list[str]:
    """Filter DICOM paths to keep only US (ultrasound) files."""
    us_paths = [
        p for p in dicom_paths
        if any(part.startswith("US_") for part in p.replace("\\", "/").split("/"))
    ]

    if len(us_paths) == 0 and len(dicom_paths) > 0:
        logger.warning(
            f"No US_ folders found, keeping all {len(dicom_paths)} DICOMs. "
            f"Sample: {dicom_paths[0]}"
        )
        return dicom_paths

    return us_paths


def filter_multiframe_dicoms(dicom_paths: list[str], min_frames: int = 4) -> list[str]:
    """Filter to only DICOMs with multi-frame video data."""
    video_paths = []
    for path in dicom_paths:
        try:
            dcm = pydicom.dcmread(path, stop_before_pixels=True)
            n_frames = int(dcm.get("NumberOfFrames", 1))
            if n_frames >= min_frames:
                video_paths.append(path)
        except Exception:
            continue
    return video_paths


def build_samples_dataframe(
    labels_csv: str,
    accession_index_path: str,
    min_frames_per_dicom: int = 4,
    max_dicoms_per_study: int = 10,
    accession_filter: set = None,
) -> pd.DataFrame:
    """
    Build the expanded samples DataFrame: one row per DICOM clip.

    This is separated from the Dataset class so that train/val/test
    splits can share the same expansion logic but filter by accession.

    Args:
        labels_csv: Path to labels CSV.
        accession_index_path: Path to accession index JSON.
        min_frames_per_dicom: Skip DICOMs with fewer frames.
        max_dicoms_per_study: Cap DICOMs per study.
        accession_filter: If provided, only include these accessions.

    Returns:
        DataFrame with columns [accession, label, dicom_path].
    """
    labels_df = pd.read_csv(labels_csv)
    labels_df["accession"] = labels_df["accession"].astype(str)
    labels_df = labels_df.dropna(subset=["label"])

    with open(accession_index_path) as f:
        acc_index = json.load(f)

    rows = []
    studies_used = 0

    for _, row in labels_df.iterrows():
        acc = str(row["accession"])
        label = int(row["label"])

        # Skip if not in the requested split
        if accession_filter is not None and acc not in accession_filter:
            continue

        dicom_paths = acc_index.get(acc, [])
        if not dicom_paths:
            continue

        dicom_paths = filter_us_dicoms(dicom_paths)
        if not dicom_paths:
            continue

        dicom_paths = filter_multiframe_dicoms(dicom_paths, min_frames=min_frames_per_dicom)
        if not dicom_paths:
            continue

        if len(dicom_paths) > max_dicoms_per_study:
            rng = np.random.RandomState(hash(acc) % 2**31)
            dicom_paths = list(rng.choice(dicom_paths, max_dicoms_per_study, replace=False))

        studies_used += 1
        for dpath in dicom_paths:
            rows.append({"accession": acc, "label": label, "dicom_path": dpath})

    df = pd.DataFrame(rows)

    if len(df) > 0:
        n_pos = (df["label"] == 1).sum()
        n_neg = (df["label"] == 0).sum()
        logger.info(
            f"Built samples: {len(df)} clips from {studies_used} studies "
            f"(normal: {n_neg}, dilated: {n_pos}, ratio: {n_neg/max(n_pos,1):.1f}:1)"
        )
    else:
        logger.warning("No samples found!")

    return df


class JHUEchoDataset(Dataset):
    """
    Dataset where each sample is a single DICOM video clip.

    Takes a pre-built samples DataFrame (from build_samples_dataframe)
    so that the Dataset itself is simple and the split logic lives
    outside.
    """

    def __init__(
        self,
        samples_df: pd.DataFrame,
        num_frames: int = 16,
        frame_size: int = 224,
        normalize_mean: list[float] = None,
        normalize_std: list[float] = None,
        augment: bool = False,
    ):
        self.samples = samples_df.reset_index(drop=True)
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.augment = augment

        mean = normalize_mean or [0.485, 0.456, 0.406]
        std = normalize_std or [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean=mean, std=std)

        if augment:
            self.spatial_aug = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(
                    degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)
                ),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ])
        else:
            self.spatial_aug = None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        row = self.samples.iloc[idx]
        dicom_path = row["dicom_path"]
        label = int(row["label"])
        accession = str(row["accession"])

        frames = extract_frames_from_dicom(dicom_path)

        if frames is None or len(frames) == 0:
            video = torch.zeros(3, self.num_frames, self.frame_size, self.frame_size)
            return {"video": video, "label": label, "accession": accession}

        video = self._sample_and_preprocess(frames)
        return {"video": video, "label": label, "accession": accession}

    def _sample_and_preprocess(self, frames: np.ndarray) -> torch.Tensor:
        n_total = len(frames)

        if n_total >= self.num_frames:
            indices = np.linspace(0, n_total - 1, self.num_frames, dtype=int)
        else:
            indices = list(range(n_total))
            while len(indices) < self.num_frames:
                indices.append(n_total - 1)
            indices = np.array(indices)

        sampled = frames[indices]
        resized = np.stack([
            cv2.resize(frame, (self.frame_size, self.frame_size))
            for frame in sampled
        ])

        tensor = torch.from_numpy(resized).float() / 255.0
        tensor = tensor.permute(0, 3, 1, 2)  # (T, C, H, W)

        if self.spatial_aug is not None:
            seed = np.random.randint(0, 2**32)
            augmented = []
            for t in range(tensor.shape[0]):
                torch.manual_seed(seed)
                augmented.append(self.spatial_aug(tensor[t]))
            tensor = torch.stack(augmented)

        normalized = torch.stack([
            self.normalize(tensor[t]) for t in range(tensor.shape[0])
        ])

        # (C, T, H, W) — PanEcho format
        return normalized.permute(1, 0, 2, 3)


def build_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler that guarantees balanced batches.

    The weights are directly aligned with the dataset indices because
    we pass in the labels array from the already-built dataset (not
    from a separate source that could be misaligned).

    How it works:
    - Each sample gets a weight = 1 / (count of its class)
    - So if there are 4000 normal and 400 dilated clips:
      - Each normal clip gets weight 1/4000 = 0.00025
      - Each dilated clip gets weight 1/400 = 0.0025 (10x higher)
    - The sampler draws samples proportional to their weight
    - Result: each batch has roughly 50/50 class balance

    With replacement=True, dilated clips get resampled multiple times
    per epoch. This is expected and necessary — it's the only way to
    balance a 10:1 ratio without throwing away 90% of normal data.

    Args:
        labels: 1D numpy array of integer labels, one per sample,
                indexed 0 to N-1 matching the dataset.

    Returns:
        WeightedRandomSampler ready to pass to DataLoader.
    """
    labels = labels.astype(int)
    class_counts = np.bincount(labels, minlength=2)

    if class_counts[1] == 0:
        logger.warning("No positive samples in this split! Sampler will be uniform.")
        sample_weights = torch.ones(len(labels))
    else:
        # Weight per class = total_samples / (num_classes * class_count)
        # This is a standard formula that produces balanced expected
        # sampling rates for each class
        n_samples = len(labels)
        n_classes = len(class_counts)
        class_weights = n_samples / (n_classes * class_counts.astype(float))

        # Map class weight to each individual sample
        sample_weights = torch.tensor(
            [class_weights[label] for label in labels],
            dtype=torch.float64,
        )

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True,
    )

    logger.info(
        f"WeightedRandomSampler built: "
        f"class_0={class_counts[0]} (w={n_samples/(n_classes*class_counts[0]):.4f}), "
        f"class_1={class_counts[1]} (w={n_samples/(n_classes*class_counts[1]):.4f}), "
        f"expected ~{100/n_classes:.0f}% each class per batch"
    )

    return sampler


def create_dataloaders(
    labels_csv: str,
    accession_index_path: str,
    config: dict,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
) -> tuple:
    """
    Create train/val/test DataLoaders.

    Pipeline:
    1. Load all labeled accessions
    2. Stratified split at STUDY level (prevents data leakage)
    3. For each split, expand studies -> DICOM clips
    4. Build separate Dataset per split (clean index alignment)
    5. Train gets WeightedRandomSampler; val/test get sequential

    By building separate datasets per split (instead of one big dataset
    with Subset), we guarantee that each dataset's internal indices
    (0 to N-1) perfectly align with the sampler weights. This was the
    bug in the previous version — Subset remaps indices, which broke
    the sampler weight alignment.
    """
    # --- Step 1: Load study-level labels ---
    labels_df = pd.read_csv(labels_csv)
    labels_df["accession"] = labels_df["accession"].astype(str)
    labels_df = labels_df.dropna(subset=["label"])

    with open(accession_index_path) as f:
        acc_index = json.load(f)

    # Keep only studies that have DICOMs
    labels_df = labels_df[labels_df["accession"].isin(acc_index.keys())]
    accessions = labels_df["accession"].values
    labels = labels_df["label"].astype(int).values

    # --- Step 2: Stratified split at study level ---
    acc_train_val, acc_test, lab_train_val, _ = train_test_split(
        accessions, labels, test_size=test_split, stratify=labels, random_state=seed
    )
    relative_val = val_split / (1 - test_split)
    acc_train, acc_val, _, _ = train_test_split(
        acc_train_val, lab_train_val, test_size=relative_val,
        stratify=lab_train_val, random_state=seed
    )

    logger.info(f"Study-level split — train: {len(acc_train)}, val: {len(acc_val)}, test: {len(acc_test)}")

    # Log class balance per split
    for name, accs in [("train", acc_train), ("val", acc_val), ("test", acc_test)]:
        split_labels = labels_df[labels_df["accession"].isin(accs)]["label"].astype(int)
        n_pos = (split_labels == 1).sum()
        n_neg = (split_labels == 0).sum()
        logger.info(f"  {name}: {n_neg} normal, {n_pos} dilated studies")

    # --- Step 3: Build separate samples DataFrame per split ---
    common_kwargs = dict(
        labels_csv=labels_csv,
        accession_index_path=accession_index_path,
        min_frames_per_dicom=config.get("min_frames_per_dicom", 4),
        max_dicoms_per_study=config.get("max_dicoms_per_study", 10),
    )

    train_samples = build_samples_dataframe(**common_kwargs, accession_filter=set(acc_train))
    val_samples = build_samples_dataframe(**common_kwargs, accession_filter=set(acc_val))
    test_samples = build_samples_dataframe(**common_kwargs, accession_filter=set(acc_test))

    # --- Step 4: Build Dataset per split ---
    ds_kwargs = dict(
        num_frames=config.get("num_frames", 16),
        frame_size=config.get("frame_size", 224),
        normalize_mean=config.get("normalize_mean"),
        normalize_std=config.get("normalize_std"),
    )

    train_ds = JHUEchoDataset(train_samples, augment=True, **ds_kwargs)
    val_ds = JHUEchoDataset(val_samples, augment=False, **ds_kwargs)
    test_ds = JHUEchoDataset(test_samples, augment=False, **ds_kwargs)

    logger.info(
        f"Clip-level counts — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}"
    )

    # --- Step 5: Build DataLoaders ---
    batch_size = config.get("batch_size", 8)
    num_workers = config.get("num_workers", 4)

    # Build sampler from the TRAIN dataset's own labels
    # Since train_ds has indices 0..N-1 matching its internal samples,
    # the sampler weights are perfectly aligned
    train_labels = train_ds.samples["label"].values.astype(int)
    train_sampler = build_weighted_sampler(train_labels)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader