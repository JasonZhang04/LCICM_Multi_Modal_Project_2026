"""
Per-EPISODE ECG measurement features (episode-level rebuild of the v7 ECG-measurement idea).

We only ever fed ECG to the model as a learned embedding (PCLR/xECG) and found it
redundant with EHR+CXR. On the CXR side the win came from adding EXPLICIT clinical
measurements alongside the embedding. MIMIC-IV-ECG ships machine_measurements.csv
(cart-computed intervals/axes + the machine's diagnostic statements), never used at
episode scale. This builds those features per episode so the "does ECG add anything?"
question can be answered on ~18,500 episodes instead of 522.

Per ECG, from rr/p/qrs/t fiducials:
    heart_rate, rr_interval, pr_interval, qrs_duration, qt, qtc_bazett, qtc_frid,
    p_duration, p_axis, qrs_axis, t_axis, qrs_t_angle
Plus binary flags parsed from the report text (mechanistically relevant to aortic
dilation): lvh, lae, afib, rbbb, lbbb, st_t, mi, paced.

Aggregated over each EPISODE's PRE-INDEX ECGs (ecg_time in [index-180d, index]):
    mean(numeric)  +  sd(numeric)  (temporal variability)  +  max(flags)  +  n_records.
Causal by construction (only ECGs at or before the index echo) — matches the rebuild's
EHR/CXR windows, unlike the v7 scratch version which used |delta| and the patient's
first-echo date.

Output: pretrained_checkpoints/ecg_features_episode.csv  (keyed by episode_id)
Run: sbatch scripts/slurm_extract_ecg_episode.sh
"""

import logging
import os
import sys

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PC = os.path.join(PROJ, "pretrained_checkpoints")
OUT = os.path.join(PC, "ecg_features_episode.csv")
MM = ("/scratch4/rsteven1/mimic-iv-ecg-diagnostic-electrocardiogram-"
      "matched-subset-1.0/machine_measurements.csv")
WINDOW = int(os.environ.get("ECG_WINDOW", "180"))

NUM = ["heart_rate", "rr_interval", "pr_interval", "qrs_duration", "qt", "qtc_bazett",
       "qtc_frid", "p_duration", "p_axis", "qrs_axis", "t_axis", "qrs_t_angle"]
FLAGS = {
    "lvh":  r"left ventricular hypertrophy|LVH",
    "lae":  r"left atrial (enlargement|abnormality)",
    "afib": r"atrial fibrillation|atrial flutter",
    "rbbb": r"right bundle branch block",
    "lbbb": r"left bundle branch block",
    "st_t": r"ST (depression|elevation)|T wave (abnormal|inversion)|repolarization abnormalit",
    "mi":   r"myocardial infarct",
    "paced": r"paced|pacemaker",
}
REP_COLS = [f"report_{i}" for i in range(18)]


def per_ecg_features(mm: pd.DataFrame) -> pd.DataFrame:
    """Derive intervals/axes/flags for each ECG row (same math as the v7 reference)."""
    rr = pd.to_numeric(mm.rr_interval, errors="coerce")
    mm = mm.copy()
    mm["rr_interval"] = rr
    mm["heart_rate"] = 60000.0 / rr.replace(0, np.nan)
    mm["pr_interval"] = pd.to_numeric(mm.qrs_onset, errors="coerce") - pd.to_numeric(mm.p_onset, errors="coerce")
    mm["qrs_duration"] = pd.to_numeric(mm.qrs_end, errors="coerce") - pd.to_numeric(mm.qrs_onset, errors="coerce")
    mm["qt"] = pd.to_numeric(mm.t_end, errors="coerce") - pd.to_numeric(mm.qrs_onset, errors="coerce")
    mm["p_duration"] = pd.to_numeric(mm.p_end, errors="coerce") - pd.to_numeric(mm.p_onset, errors="coerce")
    rr_s = rr / 1000.0
    mm["qtc_bazett"] = mm.qt / np.sqrt(rr_s.replace(0, np.nan))
    mm["qtc_frid"] = mm.qt / np.cbrt(rr_s.replace(0, np.nan))
    for c in ("p_axis", "qrs_axis", "t_axis"):
        mm[c] = pd.to_numeric(mm[c], errors="coerce")
    ang = (mm.qrs_axis - mm.t_axis).abs()
    mm["qrs_t_angle"] = np.minimum(ang, 360 - ang)
    for c, lo, hi in [("heart_rate", 20, 250), ("pr_interval", 50, 400), ("qrs_duration", 40, 250),
                      ("qt", 200, 700), ("qtc_bazett", 300, 700), ("qtc_frid", 300, 700),
                      ("p_duration", 30, 200), ("rr_interval", 240, 3000)]:
        mm.loc[(mm[c] < lo) | (mm[c] > hi), c] = np.nan
    txt = mm[[c for c in REP_COLS if c in mm.columns]].fillna("").agg(" | ".join, axis=1).str.lower()
    for f, pat in FLAGS.items():
        mm[f"flag_{f}"] = txt.str.contains(pat, regex=True, case=False, na=False).astype(int)
    return mm


def main():
    sys.path.insert(0, os.path.join(PROJ, "src"))
    from multimodal_aorta.data.episodes import load_episodes

    ep = load_episodes(PC, require_ecg=False)[["episode_id", "subject_id", "echo_dt"]].copy()
    subs = set(ep.subject_id.astype(int))
    log.info("episodes %d / patients %d", len(ep), len(subs))

    use = (["subject_id", "ecg_time", "rr_interval", "p_onset", "p_end",
            "qrs_onset", "qrs_end", "t_end", "p_axis", "qrs_axis", "t_axis"] + REP_COLS)
    log.info("reading machine_measurements ...")
    mm = pd.read_csv(MM, usecols=lambda c: c in use, low_memory=False)
    mm = mm[mm.subject_id.isin(subs)].copy()
    mm["ecg_time"] = pd.to_datetime(mm.ecg_time, errors="coerce")
    mm = mm[mm.ecg_time.notna()]
    log.info("ECG rows for cohort subjects: %d", len(mm))
    mm = per_ecg_features(mm)

    keep = ["subject_id", "ecg_time"] + NUM + [f"flag_{f}" for f in FLAGS]
    mm = mm[keep]

    # Episode x ECG on subject, keep PRE-INDEX window, aggregate per episode.
    ecg_by_subj = {s: g for s, g in mm.groupby("subject_id")}
    rows = []
    for e in ep.itertuples(index=False):
        g = ecg_by_subj.get(int(e.subject_id))
        if g is None or pd.isna(e.echo_dt):
            continue
        delta = (e.echo_dt - g.ecg_time).dt.days
        w = g[(delta >= 0) & (delta <= WINDOW)]
        if w.empty:
            continue
        rec = {"episode_id": e.episode_id, "subject_id": int(e.subject_id),
               "ecg_n_records": len(w)}
        for c in NUM:
            rec[f"ecg_{c}_mean"] = float(w[c].mean())
            rec[f"ecg_{c}_sd"] = float(w[c].std()) if len(w) > 1 else 0.0
        for f in FLAGS:
            rec[f"ecg_flag_{f}"] = int(w[f"flag_{f}"].max())
        rows.append(rec)

    feat = pd.DataFrame(rows)
    feat.to_csv(OUT, index=False)
    log.info("Saved %d episodes x %d features -> %s (%.1f%% of episodes have a pre-index ECG)",
             len(feat), feat.shape[1] - 2, OUT, 100 * len(feat) / len(ep))
    numcols = [c for c in feat.columns if c.startswith("ecg_") and c.endswith(("_mean", "_sd"))]
    log.info("numeric coverage (non-null %%): %.0f", feat[numcols].notna().mean().mean() * 100)
    log.info("flag prevalence:\n%s",
             feat[[c for c in feat.columns if c.startswith("ecg_flag")]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
