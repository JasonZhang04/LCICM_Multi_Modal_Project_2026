"""
DICOM utilities for the JHU echo dataset.

Handles reading DICOM files, extracting accession numbers for linking
to pmap metadata, and extracting video frames from echo DICOMs.
"""

import os
import glob
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pydicom

logger = logging.getLogger(__name__)


def get_accession_number(dicom_path: str) -> Optional[str]:
    """
    Extract AccessionNumber from a DICOM file without loading pixel data.

    Args:
        dicom_path: Path to a .dcm file.

    Returns:
        AccessionNumber string, or None if not found.
    """
    try:
        dcm = pydicom.dcmread(dicom_path, stop_before_pixels=True)
        accession = dcm.get("AccessionNumber")
        return str(accession) if accession else None
    except Exception as e:
        logger.warning(f"Failed to read accession from {dicom_path}: {e}")
        return None


def extract_frames_from_dicom(dicom_path: str) -> Optional[np.ndarray]:
    """
    Extract video frames from an echocardiogram DICOM file.

    Args:
        dicom_path: Path to a .dcm file containing echo video.

    Returns:
        numpy array of shape (num_frames, H, W, C) with uint8 values,
        or None if extraction fails.
    """
    try:
        dcm = pydicom.dcmread(dicom_path)

        # Check if pixel data exists
        if not hasattr(dcm, "pixel_array"):
            logger.warning(f"No pixel data in {dicom_path}")
            return None

        pixel_array = dcm.pixel_array

        # Handle different array shapes:
        # (frames, H, W)       -> grayscale video
        # (frames, H, W, 3)    -> color video
        # (H, W) or (H, W, 3) -> single frame
        if pixel_array.ndim == 2:
            # Single grayscale frame -> expand to (1, H, W, 1)
            frames = pixel_array[np.newaxis, :, :, np.newaxis]
            frames = np.repeat(frames, 3, axis=-1)
        elif pixel_array.ndim == 3:
            if pixel_array.shape[-1] == 3:
                # Single color frame (H, W, 3) -> (1, H, W, 3)
                frames = pixel_array[np.newaxis, ...]
            else:
                # Grayscale video (frames, H, W) -> (frames, H, W, 3)
                frames = pixel_array[:, :, :, np.newaxis]
                frames = np.repeat(frames, 3, axis=-1)
        elif pixel_array.ndim == 4:
            # Color video (frames, H, W, 3) — already correct
            frames = pixel_array
        else:
            logger.warning(f"Unexpected pixel array shape {pixel_array.shape} in {dicom_path}")
            return None

        return frames.astype(np.uint8)

    except Exception as e:
        logger.warning(f"Failed to extract frames from {dicom_path}: {e}")
        return None


def find_dicom_files(root_dir: str, extension: str = ".dcm") -> list[str]:
    """
    Recursively find all DICOM files under a directory.

    Args:
        root_dir: Root directory to search.
        extension: File extension to look for.

    Returns:
        List of absolute paths to DICOM files.
    """
    pattern = os.path.join(root_dir, "**", f"*{extension}")
    files = glob.glob(pattern, recursive=True)
    logger.info(f"Found {len(files)} DICOM files under {root_dir}")
    return sorted(files)


def build_accession_index(dicom_dir: str) -> dict[str, list[str]]:
    """
    Build a mapping from AccessionNumber -> list of DICOM file paths.

    This is needed because the JHU dataset stores DICOMs in a nested
    structure and we need accession numbers to link to pmap metadata.

    Args:
        dicom_dir: Root directory containing DICOM files.

    Returns:
        Dict mapping accession numbers to lists of DICOM paths.
    """
    dicom_files = find_dicom_files(dicom_dir)
    index: dict[str, list[str]] = {}

    for i, fpath in enumerate(dicom_files):
        if i % 500 == 0:
            logger.info(f"Indexing DICOM {i}/{len(dicom_files)}...")

        accession = get_accession_number(fpath)
        if accession:
            index.setdefault(accession, []).append(fpath)

    logger.info(f"Built accession index: {len(index)} unique accessions")
    return index


def get_dicom_metadata(dicom_path: str) -> dict:
    """
    Extract useful metadata fields from a DICOM file.

    Returns dict with fields like PatientID, StudyDate, Modality, etc.
    """
    try:
        dcm = pydicom.dcmread(dicom_path, stop_before_pixels=True)
        return {
            "AccessionNumber": str(dcm.get("AccessionNumber", "")),
            "PatientID": str(dcm.get("PatientID", "")),
            "StudyDate": str(dcm.get("StudyDate", "")),
            "Modality": str(dcm.get("Modality", "")),
            "Manufacturer": str(dcm.get("Manufacturer", "")),
            "Rows": int(dcm.get("Rows", 0)),
            "Columns": int(dcm.get("Columns", 0)),
            "NumberOfFrames": int(dcm.get("NumberOfFrames", 1)),
            "SOPClassUID": str(dcm.get("SOPClassUID", "")),
        }
    except Exception as e:
        logger.warning(f"Failed to read metadata from {dicom_path}: {e}")
        return {}
