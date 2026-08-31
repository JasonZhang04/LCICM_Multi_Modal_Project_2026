"""
BROADER ECG-waveform cohort: every echo episode with a pre-index ECG, regardless of CXR.

The tri-modal ECG cohort is only the ~18k CXR-overlap episodes; there are far more patients
with an ECG + aortic echo but no CXR. Training the waveform CNN on all of them gives it
much more data (closer to the JACC model's 69k) and a more robust ECG representation, while
staying leakage-free for the fusion.

Derivation (reuses build_episode_cohort): all TTE aortic studies, median-aggregated,
[1.5,7.0] cm, >=180d separation -> ~136k episodes; keep those with a 12-lead ECG in
[index-180d, index], take the closest.

FOLD ASSIGNMENT (leakage-critical):
  - A patient in the CXR cohort keeps their IMMUTABLE fold id (0-4). Their ECG OOF /
    embedding for a CXR-cohort episode will be predicted by a model trained on folds != k,
    matching the fusion's immutable folds exactly.
  - A patient NOT in the CXR cohort gets fold_id = -1 ("always-train"): they augment
    training for every fold and are NEVER in a test fold, so they cannot leak into any
    CXR-cohort episode's OOF prediction.
  - Because folds are per-PATIENT, ALL of a patient's episodes (CXR-eligible or not) share
    one fold — a CXR-cohort patient's extra (non-CXR) episodes sit in their own fold and are
    excluded from training when that patient is the held-out fold.

Out: pretrained_checkpoints/ecg_waveform_cohort_broad.csv
     (episode_id, subject_id, fold_id, study_id, ecg_path, days_before, root_cm, asc_cm, hr,
      in_cxr_cohort)
Run: sbatch scripts/slurm_ecg_cohort_broad.sh
"""
import logging, os, sys
import numpy as np, pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PC = os.path.join(ROOT, "pretrained_checkpoints")
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ECG_ROOT = "/scratch4/rsteven1/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0"
WINDOW = 180
import build_episode_cohort as bec          # reuse load_aortic_rows / build_episodes


def main():
    from multimodal_aorta.data.splits import load_episode_fold_id_map
    from multimodal_aorta.data.episodes import make_episode_id

    log.info("[1/4] scanning aortic measurements ...")
    rows = bec.load_aortic_rows(bec.STRUCT)
    log.info("[2/4] building episodes (>=180d separation) ...")
    ep = bec.build_episodes(rows, 180)                    # all TTE aortic episodes (~136k)
    ep["episode_id"] = [make_episode_id(s, m) for s, m in zip(ep.subject_id, ep.measurement_id)]

    log.info("[3/4] matching closest pre-index ECG ...")
    rl = pd.read_csv(os.path.join(ECG_ROOT, "record_list.csv"),
                     usecols=["subject_id", "study_id", "ecg_time", "path"])
    rl["ecg_time"] = pd.to_datetime(rl.ecg_time, errors="coerce"); rl = rl[rl.ecg_time.notna()]
    rl_by_subj = {s: g.sort_values("ecg_time") for s, g in rl.groupby("subject_id")}
    mm = pd.read_csv(os.path.join(ECG_ROOT, "machine_measurements.csv"),
                     usecols=["study_id", "rr_interval"], low_memory=False)
    mm["hr"] = 60000.0 / pd.to_numeric(mm.rr_interval, errors="coerce").replace(0, np.nan)
    hr_of = dict(zip(mm.study_id, mm.hr))

    fold_map = load_episode_fold_id_map(os.path.join(PC, "episode_fold_assignments.csv"))
    # a patient is in the CXR cohort iff any of their episodes has a fold assignment
    cxr_fold_of_subject = {}
    for eid, k in fold_map.items():
        sid = int(str(eid).split("_")[0]); cxr_fold_of_subject[sid] = k   # patient-grouped: one fold/patient

    out_rows = []
    for e in ep.itertuples(index=False):
        g = rl_by_subj.get(int(e.subject_id))
        if g is None or pd.isna(e.echo_dt):
            continue
        delta = (e.echo_dt - g.ecg_time).dt.days
        w = g[(delta >= 0) & (delta <= WINDOW)]
        if w.empty:
            continue
        r = w.iloc[-1]
        sid = int(e.subject_id)
        fold = cxr_fold_of_subject.get(sid, -1)          # -1 = broader-only patient (always-train)
        hr = hr_of.get(r.study_id, np.nan)
        out_rows.append({
            "episode_id": e.episode_id, "subject_id": sid, "fold_id": int(fold),
            "study_id": int(r.study_id), "ecg_path": os.path.join(ECG_ROOT, r.path),
            "days_before": int((e.echo_dt - r.ecg_time).days),
            "root_cm": e.root_cm, "asc_cm": e.asc_cm,   # build_episodes already renames sites
            "hr": float(hr) if pd.notna(hr) and 20 < hr < 250 else np.nan,
            "in_cxr_cohort": int(fold >= 0),
        })
    out = pd.DataFrame(out_rows)
    out.to_csv(os.path.join(PC, "ecg_waveform_cohort_broad.csv"), index=False)
    log.info("[4/4] Wrote ecg_waveform_cohort_broad.csv: %d episodes / %d patients / %d unique ECGs",
             len(out), out.subject_id.nunique(), out.study_id.nunique())
    log.info("  CXR-cohort episodes (fold 0-4): %d / %d patients | broader-only (fold -1): %d / %d patients",
             int((out.fold_id >= 0).sum()), out[out.fold_id >= 0].subject_id.nunique(),
             int((out.fold_id == -1).sum()), out[out.fold_id == -1].subject_id.nunique())
    log.info("  vs the tri-modal ECG cohort (18,221). root labelled %d | asc labelled %d",
             int(out.root_cm.notna().sum()), int(out.asc_cm.notna().sum()))
    # leakage self-check: every patient has exactly one fold
    span = out.groupby("subject_id").fold_id.nunique()
    log.info("  leakage check: patients spanning >1 fold = %d (must be 0)", int((span > 1).sum()))


if __name__ == "__main__":
    main()
