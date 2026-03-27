"""
PyTorch Dataset for the JHU echocardiogram dataset.

Key features:
- Each DICOM video clip is a training sample
- All US DICOMs in a study share the same label
- SR DICOMs filtered out, single-frame DICOMs filtered out
- WeightedRandomSampler ensures balanced batches (50/50 normal/dilated)
- Study-level splitting prevents data leakage
- PanEcho-compatible preprocessing: 16 frames, 224x224, ImageNet-normalized
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
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split

from data.dicom_utils import extract_frames_from_dicom

logger = logging.getLogger(__name__)


def filter_us_dicoms(dicom_paths: list[str]) -> list[str]:
    """Filter DICOM paths to keep only US (ultrasound) files."""
    us_paths = []
    for path in dicom_paths:
        parts = path.replace("\\", "/").split("/")
        is_us = any(part.startswith("US_") for part in parts)
        if is_us:
            us_paths.append(path)

    if len(us_paths) == 0 and len(dicom_paths) > 0:
        logger.warning(
            f"No US_ folders found in paths, keeping all {len(dicom_paths)} DICOMs. "
            f"Sample path: {dicom_paths[0]}"
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


class JHUEchoDataset(Dataset):
    """
    Dataset where each sample is a single DICOM video clip.
    All clips from a study share the study's aortic dilation label.
    """

    def __init__(
        self,
        labels_csv: str,
        accession_index_path: str,
        num_frames: int = 16,
        frame_size: int = 224,
        min_frames_per_dicom: int = 4,
        max_dicoms_per_study: int = 50,
        normalize_mean: list[float] = None,
        normalize_std: list[float] = None,
        augment: bool = False,
    ):
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.augment = augment

        labels_df = pd.read_csv(labels_csv)
        labels_df["accession"] = labels_df["accession"].astype(str)
        labels_df = labels_df.dropna(subset=["label"])

        with open(accession_index_path) as f:
            acc_index = json.load(f)

        # Expand: one row per DICOM clip
        rows = []
        studies_used = 0

        for _, row in labels_df.iterrows():
            acc = str(row["accession"])
            label = int(row["label"])
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
                rows.append({
                    "accession": acc,
                    "label": label,
                    "dicom_path": dpath,
                })

        self.samples = pd.DataFrame(rows)

        n_pos = (self.samples["label"] == 1).sum()
        n_neg = (self.samples["label"] == 0).sum()
        logger.info(
            f"JHUEchoDataset: {len(self.samples)} clips from {studies_used} studies "
            f"(normal: {n_neg}, dilated: {n_pos}, ratio: {n_neg/max(n_pos,1):.1f}:1)"
        )

        # Normalization — ImageNet defaults for PanEcho
        mean = normalize_mean or [0.485, 0.456, 0.406]
        std = normalize_std or [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean=mean, std=std)

        if augment:
            self.spatial_aug = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
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
        tensor = tensor.permute(0, 3, 1, 2)

        if self.spatial_aug is not None:
            seed = np.random.randint(0, 2**32)
            augmented = []
            for t in range(tensor.shape[0]):
                torch.manual_seed(seed)
                augmented.append(self.spatial_aug(tensor[t]))
            tensor = torch.stack(augmented)

        normalized = torch.stack([self.normalize(tensor[t]) for t in range(tensor.shape[0])])
        video = normalized.permute(1, 0, 2, 3)

        return video

    def get_labels(self) -> np.ndarray:
        """Get all labels as numpy array (for building sampler)."""
        return self.samples["label"].values.astype(int)


def build_weighted_sampler(dataset: Dataset, indices: list[int]) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler that balances classes in each batch.

    Instead of seeing 10:1 normal:dilated ratio, the sampler ensures
    roughly 50/50 by oversampling dilated clips and undersampling
    normal clips. This gives the model a strong gradient signal for
    the minority class from the very first batch.

    Args:
        dataset: The full dataset.
        indices: The subset indices (e.g., train split).

    Returns:
        WeightedRandomSampler for the DataLoader.
    """
    all_labels = dataset.get_labels()
    subset_labels = all_labels[indices]

    # Count classes in this subset
    class_counts = np.bincount(subset_labels, minlength=2)
    # Weight = inverse frequency
    class_weights = 1.0 / class_counts.astype(float)
    # Assign weight to each sample
    sample_weights = class_weights[subset_labels]
    sample_weights = torch.from_numpy(sample_weights).float()

    # num_samples = length of subset, replacement=True for oversampling
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(indices),
        replacement=True,
    )

    n_pos = (subset_labels == 1).sum()
    n_neg = (subset_labels == 0).sum()
    logger.info(
        f"WeightedRandomSampler: {n_neg} normal (weight={class_weights[0]:.6f}), "
        f"{n_pos} dilated (weight={class_weights[1]:.6f})"
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
    Create train/val/test DataLoaders with:
    - Study-level splitting (no data leakage)
    - WeightedRandomSampler for balanced training batches
    """
    labels_df = pd.read_csv(labels_csv)
    labels_df["accession"] = labels_df["accession"].astype(str)
    labels_df = labels_df.dropna(subset=["label"])

    with open(accession_index_path) as f:
        acc_index = json.load(f)

    labels_df = labels_df[labels_df["accession"].isin(acc_index.keys())]

    accessions = labels_df["accession"].values
    labels = labels_df["label"].astype(int).values

    # Study-level stratified split
    acc_train_val, acc_test, lab_train_val, lab_test = train_test_split(
        accessions, labels, test_size=test_split, stratify=labels, random_state=seed
    )
    relative_val = val_split / (1 - test_split)
    acc_train, acc_val, _, _ = train_test_split(
        acc_train_val, lab_train_val, test_size=relative_val,
        stratify=lab_train_val, random_state=seed
    )

    acc_train_set = set(acc_train)
    acc_val_set = set(acc_val)
    acc_test_set = set(acc_test)

    logger.info(
        f"Study-level split — train: {len(acc_train)}, val: {len(acc_val)}, test: {len(acc_test)}"
    )

    common_kwargs = dict(
        labels_csv=labels_csv,
        accession_index_path=accession_index_path,
        num_frames=config.get("num_frames", 16),
        frame_size=config.get("frame_size", 224),
        min_frames_per_dicom=config.get("min_frames_per_dicom", 4),
        max_dicoms_per_study=config.get("max_dicoms_per_study", 50),
        normalize_mean=config.get("normalize_mean"),
        normalize_std=config.get("normalize_std"),
    )

    train_ds = JHUEchoDataset(**common_kwargs, augment=True)
    val_ds = JHUEchoDataset(**common_kwargs, augment=False)
    test_ds = JHUEchoDataset(**common_kwargs, augment=False)

    train_indices = [i for i, row in train_ds.samples.iterrows() if row["accession"] in acc_train_set]
    val_indices = [i for i, row in val_ds.samples.iterrows() if row["accession"] in acc_val_set]
    test_indices = [i for i, row in test_ds.samples.iterrows() if row["accession"] in acc_test_set]

    logger.info(
        f"Clip-level counts — train: {len(train_indices)}, "
        f"val: {len(val_indices)}, test: {len(test_indices)}"
    )

    batch_size = config.get("batch_size", 8)
    num_workers = config.get("num_workers", 4)

    # Build WeightedRandomSampler for balanced training
    train_sampler = build_weighted_sampler(train_ds, train_indices)

    # Use SubsetRandomSampler-like behavior: we need to map sampler to subset
    # Since WeightedRandomSampler works on the full dataset, we use Subset
    train_subset = Subset(train_ds, train_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        sampler=train_sampler,  # balanced sampling instead of shuffle
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        Subset(val_ds, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        Subset(test_ds, test_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader