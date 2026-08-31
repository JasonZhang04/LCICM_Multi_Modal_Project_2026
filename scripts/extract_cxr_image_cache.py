"""
Cache the preprocessed 224x224 CXR tensors into one memmap so fine-tuning reads from
RAM instead of decoding+resizing a JPG per image per epoch (the ~4 img/s bottleneck).

For each unique on-disk frontal dicom in cxr_instances_episode.csv, run the exact
val-path preprocessing (load_cxr is_train=False: grayscale->3ch, resize 224, ImageNet
normalize) and store as float16. Fine-tuning then indexes this by dicom row.
Parallelized across worker processes (single-threaded load_cxr is ~150 ms/image).

Saves:
  pretrained_checkpoints/cxr_image_cache.npy        (N, 3, 224, 224) float16
  pretrained_checkpoints/cxr_image_cache_index.csv  dicom_id -> row (only successful rows)

Note: caches the val-path (no augmentation) tensor; partial fine-tuning (last 2 blocks)
is conservative enough that dropping train-time augmentation is acceptable for v1.
Run: sbatch scripts/slurm_extract_cxr_image_cache.sh
"""
import logging, os, sys
import multiprocessing as mp
import numpy as np, pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PC = os.path.join(ROOT, "pretrained_checkpoints")
S = 224
_CFG = None


def _init():
    global _CFG
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from multimodal_aorta.configs.default_config import Config
    _CFG = Config()


def _load(args):
    row, path = args
    from multimodal_aorta.data.preprocessing import load_cxr
    try:
        return row, load_cxr(path, _CFG.data, is_train=False).numpy().astype(np.float16)
    except Exception:  # noqa: BLE001
        return row, None


def main():
    inst = pd.read_csv(os.path.join(PC, "cxr_instances_episode.csv"))
    inst = inst[inst.on_disk.fillna(False).astype(bool)].drop_duplicates("dicom_id").reset_index(drop=True)
    n = len(inst)
    log.info("caching %d unique on-disk CXRs -> (%d,3,%d,%d) float16 (%.1f GB)",
             n, n, S, S, n * 3 * S * S * 2 / 1e9)

    out_path = os.path.join(PC, "cxr_image_cache.npy")
    arr = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16, shape=(n, 3, S, S))
    ok = np.zeros(n, bool)
    nfail = done = 0
    nproc = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    with mp.Pool(nproc, initializer=_init) as pool:
        for row, x in pool.imap_unordered(_load, list(enumerate(inst.cxr_path)), chunksize=16):
            done += 1
            if x is None:
                nfail += 1
            else:
                arr[row] = x; ok[row] = True
            if done % 5000 == 0:
                arr.flush(); log.info("  %d/%d (fail=%d)", done, n, nfail)
    arr.flush()
    inst.assign(row=np.arange(n))[ok][["dicom_id", "row"]].to_csv(
        os.path.join(PC, "cxr_image_cache_index.csv"), index=False)
    log.info("saved %s + index (%d ok, %d fail) with %d workers", out_path, int(ok.sum()), nfail, nproc)


if __name__ == "__main__":
    main()
