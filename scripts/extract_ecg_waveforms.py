"""
Cache the raw 12-lead ECG waveforms for the waveform cohort into one memmap array,
so training reads from RAM instead of ~5 ms/file WFDB reads (millions of reads/epoch).

For each unique study in ecg_waveform_cohort.csv:
  - wfdb.rdrecord -> (5000, 12) mV signal
  - reorder columns to the canonical lead order (records vary, e.g. aVF/aVL swapped)
  - NaN -> 0, cast float16
Saves:
  pretrained_checkpoints/ecg_waveforms.npy         (N, 12, 5000) float16
  pretrained_checkpoints/ecg_waveform_index.csv    study_id -> row index

Left in mV scale with NO other preprocessing, matching the ECGAI-TAA description.
Run: sbatch scripts/slurm_extract_ecg_waveforms.sh
"""
import logging, os, sys
import numpy as np, pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PC = os.path.join(ROOT, "pretrained_checkpoints")

CANONICAL = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
L = 5000


def main():
    import wfdb
    coh = pd.read_csv(os.path.join(PC, "ecg_waveform_cohort.csv"))
    studies = coh.drop_duplicates("study_id")[["study_id", "ecg_path"]].reset_index(drop=True)
    n = len(studies)
    log.info("caching %d unique ECG waveforms", n)

    arr = np.zeros((n, 12, L), dtype=np.float16)
    nfail = 0
    for i, r in enumerate(studies.itertuples(index=False)):
        try:
            rec = wfdb.rdrecord(r.ecg_path)
            sig = rec.p_signal.astype(np.float32)           # (5000, 12)
            name = list(rec.sig_name)
            idx = [name.index(l) if l in name else -1 for l in CANONICAL]
            out = np.zeros((L, 12), np.float32)
            for c, j in enumerate(idx):
                if j >= 0:
                    col = sig[:, j]
                    out[:min(L, len(col)), c] = col[:L]
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
            arr[i] = out.T.astype(np.float16)               # (12, 5000)
        except Exception as e:  # noqa: BLE001
            nfail += 1
            if nfail <= 5:
                log.warning("fail %s: %s", r.study_id, e)
        if (i + 1) % 2000 == 0:
            log.info("  %d/%d (fail=%d)", i + 1, n, nfail)

    np.save(os.path.join(PC, "ecg_waveforms.npy"), arr)
    studies.assign(row=np.arange(n))[["study_id", "row"]].to_csv(
        os.path.join(PC, "ecg_waveform_index.csv"), index=False)
    log.info("saved ecg_waveforms.npy %s (%.1f GB) + index | fail=%d",
             arr.shape, arr.nbytes / 1e9, nfail)


if __name__ == "__main__":
    main()
