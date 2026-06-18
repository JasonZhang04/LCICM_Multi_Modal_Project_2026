"""
One-time script: extract PanEcho embeddings for all aortic cohort patients.

Loads each patient's echo study (same temporal matching as training), runs the
frozen PanEcho encoder on 8 uniformly-sampled clips, mean-pools to a single
768-dim vector, and saves the result.

Saves: pretrained_checkpoints/panecho_embeddings.pt  →  {subject_id: tensor(768,)}

Runtime: ~3-6 hours on a GPU node (bottleneck is DICOM I/O, not GPU).
Submit via SLURM:
    sbatch scripts/slurm_extract_panecho.sh
"""

import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

OUTPUT_PATH = "pretrained_checkpoints/panecho_embeddings.pt"
BATCH_SIZE  = 8    # patients per GPU batch (each = 8 clips × (3,16,224,224))
N_CLIPS     = 8
N_FRAMES    = 16
IMG_SIZE    = 224


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_dir = os.path.join(project_root, "src")
    sys.path.insert(0, src_dir)

    import torch
    from torch.utils.data import DataLoader

    from echo_baseline.data.echo_dataset import build_echo_cohort, EchoDataset
    from echo_baseline.models.echo_model import PanEchoEncoder
    from echo_baseline.training.train_echo import EchoTrainConfig

    cfg = EchoTrainConfig()
    out_path = os.path.join(project_root, OUTPUT_PATH)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # ── 1. Build cohort (all patients, no split needed) ───────────────────
    log.info("Building echo cohort...")
    cohort = build_echo_cohort(
        echo_labels_path     = cfg.echo_labels_path,
        echo_structured_path = cfg.echo_structured_path,
        echo_study_list      = cfg.echo_study_list,
        echo_record_list     = cfg.echo_record_list,
    )
    log.info("Cohort: %d patients", len(cohort))

    dataset = EchoDataset(
        cohort,
        echo_root  = cfg.echo_root,
        n_clips    = N_CLIPS,
        n_frames   = N_FRAMES,
        img_size   = IMG_SIZE,
        target_stats = None,   # raw labels, not needed for embedding
    )
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    log.info("DataLoader: %d batches of %d patients", len(loader), BATCH_SIZE)

    # ── 2. Load PanEcho encoder ────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Loading PanEcho encoder on %s...", device)
    encoder = PanEchoEncoder().to(device).eval()
    log.info("PanEcho encoder ready.")

    # ── 3. Extract embeddings ──────────────────────────────────────────────
    embeddings = {}
    n_zero = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            video = batch["video"].to(device, non_blocking=True)  # (B, N, C, T, H, W)
            sids  = batch["subject_id"]                            # (B,)

            B, N, C, T, H, W = video.shape
            feat = encoder(video.view(B * N, C, T, H, W))         # (B*N, 768)
            feat = feat.view(B, N, -1).mean(dim=1)                 # (B, 768)

            for j in range(B):
                sid = int(sids[j])
                # Skip patients whose video was entirely zero (all clips failed)
                if video[j].abs().sum().item() < 1e-6:
                    n_zero += 1
                    continue
                embeddings[sid] = feat[j].cpu().float()

            if (i + 1) % 50 == 0 or (i + 1) == len(loader):
                log.info(
                    "  %d / %d batches  |  %d embeddings saved  |  %d all-zero skipped",
                    i + 1, len(loader), len(embeddings), n_zero,
                )

    log.info("Done: %d embeddings extracted, %d patients skipped (all-zero clips)",
             len(embeddings), n_zero)

    # ── 4. Save ────────────────────────────────────────────────────────────
    torch.save(embeddings, out_path)
    log.info("Saved → %s", out_path)


if __name__ == "__main__":
    main()
