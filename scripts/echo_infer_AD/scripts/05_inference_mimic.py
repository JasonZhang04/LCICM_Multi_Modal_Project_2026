#!/usr/bin/env python3
"""
Step 5: Run inference on MIMIC-IV echocardiograms.

Uses the fine-tuned model to generate aortic dilation pseudo-labels
for MIMIC-IV echo data (which lacks free-text reports).

These pseudo-labels can then be used in the multi-modal pipeline
(ECG + CXR + Echo) for aortic dilation detection.

Usage:
    python scripts/05_inference_mimic.py \
        --config configs/default.yaml \
        --checkpoint /path/to/best_model.pt \
        --mimic_dir /path/to/mimic_iv_echos \
        --output /path/to/mimic_pseudo_labels.csv
"""

import os
import sys
import argparse
import logging

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dicom_utils import find_dicom_files, extract_frames_from_dicom, get_accession_number
from data.jhu_dataset import JHUEchoDataset
from models.echo_classifier import AorticDilationClassifier
from utils.training_utils import load_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@torch.no_grad()
def run_inference(model, video_tensor, device):
    """Run model inference on a single video tensor."""
    model.eval()
    video = video_tensor.unsqueeze(0).to(device)  # add batch dim
    logits = model(video)
    probs = torch.softmax(logits, dim=1)
    pred = logits.argmax(dim=1).item()
    confidence = probs.max().item()
    return pred, confidence, probs.squeeze().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="MIMIC-IV echo inference")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mimic_dir", required=True, help="Path to MIMIC-IV echo DICOMs")
    parser.add_argument("--output", default="mimic_iv_pseudo_labels.csv")
    parser.add_argument("--batch_limit", type=int, default=None, help="Max files to process (for testing)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load model
    model_cfg = config["model"]
    model_cfg["label_mode"] = config["labels"]["mode"]
    model = AorticDilationClassifier(model_cfg).to(device)
    load_checkpoint(model, None, args.checkpoint, device)

    # Create a dummy dataset instance just for preprocessing parameters
    video_cfg = config["video"]
    from torchvision import transforms
    normalize = transforms.Normalize(
        mean=video_cfg.get("normalize_mean", [0.485, 0.456, 0.406]),
        std=video_cfg.get("normalize_std", [0.229, 0.224, 0.225]),
    )

    # Find DICOM files
    dicom_files = find_dicom_files(args.mimic_dir)
    if args.batch_limit:
        dicom_files = dicom_files[:args.batch_limit]
    logger.info(f"Found {len(dicom_files)} DICOM files to process")

    # Process each file
    results = []
    for fpath in tqdm(dicom_files, desc="Inference"):
        accession = get_accession_number(fpath)
        frames = extract_frames_from_dicom(fpath)

        if frames is None or len(frames) == 0:
            continue

        # Preprocess (same as JHUEchoDataset._sample_and_preprocess)
        import cv2
        n_frames = video_cfg.get("num_frames", 32)
        frame_size = video_cfg.get("frame_size", 224)

        n_total = len(frames)
        if n_total >= n_frames:
            indices = np.linspace(0, n_total - 1, n_frames, dtype=int)
        else:
            indices = list(range(n_total))
            while len(indices) < n_frames:
                indices.append(n_total - 1)
            indices = np.array(indices)

        sampled = frames[indices]
        resized = np.stack([cv2.resize(f, (frame_size, frame_size)) for f in sampled])

        tensor = torch.from_numpy(resized).float() / 255.0
        tensor = tensor.permute(0, 3, 1, 2)  # (T, C, H, W)
        normalized = torch.stack([normalize(tensor[t]) for t in range(tensor.shape[0])])
        video_tensor = normalized.permute(1, 0, 2, 3)  # (C, T, H, W)

        # Inference
        pred, confidence, probs = run_inference(model, video_tensor, device)

        results.append({
            "dicom_path": fpath,
            "accession": accession,
            "predicted_label": pred,
            "confidence": confidence,
            "prob_normal": probs[0],
            "prob_dilated": probs[1] if len(probs) > 1 else None,
        })

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    logger.info(f"\nPseudo-labels saved to {args.output}")
    logger.info(f"Processed: {len(df)} echocardiograms")
    logger.info(f"Predicted dilated: {(df['predicted_label'] == 1).sum()}")
    logger.info(f"Predicted normal: {(df['predicted_label'] == 0).sum()}")
    logger.info(f"Mean confidence: {df['confidence'].mean():.3f}")


if __name__ == "__main__":
    main()