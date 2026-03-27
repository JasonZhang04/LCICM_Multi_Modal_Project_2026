"""
Video preprocessing utilities for echocardiogram data.

Provides functions for frame extraction, resizing, cropping,
and quality checks specific to echo videos.
"""

import cv2
import numpy as np


def crop_echo_region(frame: np.ndarray, threshold: int = 10) -> np.ndarray:
    """
    Crop the ultrasound cone region from an echo frame.

    Echocardiogram frames often have black borders and UI elements.
    This attempts to crop to just the ultrasound image region.

    Args:
        frame: (H, W, C) uint8 image.
        threshold: Pixel intensity threshold for detecting non-black regions.

    Returns:
        Cropped frame.
    """
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame

    # Find rows/cols with sufficient intensity
    row_means = gray.mean(axis=1)
    col_means = gray.mean(axis=0)

    rows_active = np.where(row_means > threshold)[0]
    cols_active = np.where(col_means > threshold)[0]

    if len(rows_active) == 0 or len(cols_active) == 0:
        return frame  # can't crop, return as-is

    r_start, r_end = rows_active[0], rows_active[-1] + 1
    c_start, c_end = cols_active[0], cols_active[-1] + 1

    return frame[r_start:r_end, c_start:c_end]


def check_video_quality(frames: np.ndarray, min_std: float = 5.0) -> bool:
    """
    Basic quality check for echo video frames.

    Rejects videos that are essentially blank or have very low variance
    (likely corrupted or non-diagnostic).

    Args:
        frames: (N, H, W, C) uint8 array.
        min_std: Minimum standard deviation across frames.

    Returns:
        True if video passes quality check.
    """
    if frames is None or len(frames) == 0:
        return False

    # Check overall variance
    if frames.std() < min_std:
        return False

    # Check that we have temporal variation (not a static image)
    if len(frames) > 1:
        frame_diffs = np.diff(frames.astype(float), axis=0)
        if np.abs(frame_diffs).mean() < 1.0:
            return False  # essentially a still image

    return True


def resize_and_pad(frame: np.ndarray, target_size: int = 224) -> np.ndarray:
    """
    Resize frame to target size, preserving aspect ratio with padding.

    Args:
        frame: (H, W, C) uint8 image.
        target_size: Output will be (target_size, target_size, C).

    Returns:
        Resized and padded frame.
    """
    h, w = frame.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to square
    pad_h = target_size - new_h
    pad_w = target_size - new_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT, value=0,
    )
    return padded