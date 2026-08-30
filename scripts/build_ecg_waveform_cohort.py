"""
Cohort for the raw-waveform ECG arm: episode -> its closest PRE-INDEX 12-lead ECG.

For each episode (patient, echo study) that has a 12-lead ECG in [index-180d, index],
pick the ECG closest to the index date and record its study_id + WFDB path + labels
(root/asc diameter, and HR from the cart rr_interval as an auxiliary target) + the
episode's immutable CV fold. This is the label table the waveform CNN trains on; the
waveforms themselves are cached separately by extract_ecg_waveforms.py.

v1 uses the CXR-cohort episodes (so d_ecg_waveform slots straight into the multimodal
stack with the existing folds). Scaling to the broader ECG-only cohort is a later step.

Out: pretrained_checkpoints/ecg_waveform_cohort.csv
Run: python scripts/build_ecg_waveform_cohort.py
"""
import logging, os, sys
import numpy as np, pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PC = os.path.join(ROOT, "pretrained_checkpoints")
sys.path.insert(0, os.path.join(ROOT, "src"))
ECG_ROOT = "/scratch4/rsteven1/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0"
WINDOW = 180


def main():
    from multimodal_aorta.data.episodes import load_episodes
    from multimodal_aorta.data.splits import load_episode_fold_id_map

    ep = load_episodes(PC, require_ecg=False)
    fold = load_episode_fold_id_map(os.path.join(PC, "episode_fold_assignments.csv"))
    ep["fold_id"] = ep.episode_id.astype(str).map(fold)

    rl = pd.read_csv(os.path.join(ECG_ROOT, "record_list.csv"),
                     usecols=["subject_id", "study_id", "ecg_time", "path"])
    rl["ecg_time"] = pd.to_datetime(rl.ecg_time, errors="coerce")
    rl = rl[rl.ecg_time.notna()]
    rl_by_subj = {s: g.sort_values("ecg_time") for s, g in rl.groupby("subject_id")}

    # HR aux target from cart rr_interval (ms) -> bpm
    mm = pd.read_csv(os.path.join(ECG_ROOT, "machine_measurements.csv"),
                     usecols=["study_id", "rr_interval"], low_memory=False)
    mm["hr"] = 60000.0 / pd.to_numeric(mm.rr_interval, errors="coerce").replace(0, np.nan)
    hr_of = dict(zip(mm.study_id, mm.hr))

    rows = []
    for e in ep.itertuples(index=False):
        g = rl_by_subj.get(int(e.subject_id))
        if g is None or pd.isna(e.echo_dt):
            continue
        delta = (e.echo_dt - g.ecg_time).dt.days
        w = g[(delta >= 0) & (delta <= WINDOW)]
        if w.empty:
            continue
        r = w.iloc[-1]                       # closest pre-index (latest ecg_time <= index)
        hr = hr_of.get(r.study_id, np.nan)
        rows.append({
            "episode_id": e.episode_id, "subject_id": int(e.subject_id), "fold_id": int(e.fold_id),
            "study_id": int(r.study_id), "ecg_path": os.path.join(ECG_ROOT, r.path),
            "days_before": int((e.echo_dt - r.ecg_time).days),
            "root_cm": e.target_root, "asc_cm": e.target_asc,
            "hr": float(hr) if pd.notna(hr) and 20 < hr < 250 else np.nan,
        })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(PC, "ecg_waveform_cohort.csv"), index=False)
    log.info("Wrote ecg_waveform_cohort.csv: %d episodes / %d patients / %d unique ECGs | HR present %.0f%%",
             len(out), out.subject_id.nunique(), out.study_id.nunique(), 100 * out.hr.notna().mean())
    log.info("root labelled %d | asc labelled %d | days_before median %.0f",
             int(out.root_cm.notna().sum()), int(out.asc_cm.notna().sum()), out.days_before.median())


if __name__ == "__main__":
    main()
