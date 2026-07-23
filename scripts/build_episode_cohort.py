"""
Build the episode-level (echo-study-level) cohort for the multimodal aorta project.

This replaces the patient-level cohort in data/dataset.py::build_cohort, which had
two defects that this script exists to fix:

  1. Labels came from analysis/explore_structured_measurements.py, which selected
     the FIRST row inside each patient's WORST severity tier across their entire
     echo history and then discarded the date. The cohort separately anchored on
     min(measurement_datetime) over ALL measurement types. The two rarely referred
     to the same study: ~25% of labels were >180d after the anchor (max ~10 years),
     and 62% of ">=4.0" positives had a patient median <4.0 -- i.e. the binary label
     was largely produced by taking a max over a noisy repeated-measures series.

  2. Labels were gated on the MIMIC-IV-ECHO DICOM subset (4,579 patients) because
     an earlier design used echo videos. The current pipeline uses none, so that
     filter discarded ~24.5k label-eligible patients for no reason.

Here, one row = one (patient, echo study) episode, anchored on that study's own
measurement_datetime, with imaging matched strictly BEFORE the anchor.

Design decisions (all overridable by flag, defaults are the prespecified analysis):
  --sep-days 180   minimum separation between retained episodes for one patient.
                   Two echoes weeks apart are one clinical event, not two, and
                   would otherwise contribute near-duplicate rows.
  --cxr-win 365    frontal (PA/AP) CXR must fall in [anchor-365d, anchor].
  --ecg-win 180    ECG must fall in [anchor-180d, anchor].
  TTE only         'stress' and 'tee' studies measure differently and are dropped.
  median           duplicate measurements within a study are aggregated by median,
                   not max, so a single mis-measure cannot define the label.

Outputs (to pretrained_checkpoints/):
  episodes.csv                 one row per episode: ids, anchor date, diameters, counts
  episode_cxr_instances.csv    (episode, dicom_id, days_before_echo) for multi-instance
  cxr_download_manifest.csv    images referenced by the cohort that are not yet on disk

Run on a compute node -- the structured-measurement scan is ~2.6 GB:
    sbatch scripts/slurm_build_episode_cohort.sh
"""

import argparse
import os

import numpy as np
import pandas as pd

PROJ = "/scratch4/rsteven1/chenjia_echo_project/2026 Multi-Modal Project"
STRUCT = f"{PROJ}/data/echo/structured-measurement.csv"
PC = f"{PROJ}/pretrained_checkpoints"
CXR_META = "/scratch4/rsteven1/MIMIC_CXR_JPG_cohort/mimic-cxr-2.0.0-metadata.csv.gz"
CXR_RECS = "/scratch4/rsteven1/MIMIC_CXR_GS/cxr-record-list.csv"
JPG_ROOT = "/scratch4/rsteven1/MIMIC_CXR_JPG_cohort/files"
ECG_RECS = ("/scratch4/rsteven1/mimic-iv-ecg-diagnostic-electrocardiogram-"
            "matched-subset-1.0/record_list.csv")

SITES = {"sinus_diam": "root_cm", "ascending_diam": "asc_cm"}
DIAM_RANGE = (1.5, 7.0)   # cm; same plausibility bounds the v3 pipeline used


def load_aortic_rows(path: str, chunksize: int = 2_000_000) -> pd.DataFrame:
    """Stream the 2.6 GB EAV table, keeping only usable TTE aortic diameter rows."""
    keep, n_raw = [], 0
    cols = ["subject_id", "measurement_id", "measurement_datetime",
            "test_type", "measurement", "result"]
    for ch in pd.read_csv(path, usecols=cols, chunksize=chunksize):
        n_raw += len(ch)
        ch = ch[ch["measurement"].isin(SITES) & (ch["test_type"] == "tte")]
        if len(ch):
            keep.append(ch)
    df = pd.concat(keep, ignore_index=True)
    print(f"  scanned {n_raw:,} rows -> {len(df):,} TTE aortic rows")

    df["value"] = pd.to_numeric(df["result"], errors="coerce")
    df["echo_dt"] = pd.to_datetime(df["measurement_datetime"], errors="coerce")
    df = df[df["value"].notna() & df["echo_dt"].notna()]
    df = df[df["value"].between(*DIAM_RANGE)]
    print(f"  after value/date/range QC {DIAM_RANGE} cm: {len(df):,} rows")
    return df


def build_episodes(rows: pd.DataFrame, sep_days: int) -> pd.DataFrame:
    """One row per (patient, study); median-aggregate duplicates; enforce separation."""
    ep = rows.pivot_table(index=["subject_id", "measurement_id", "echo_dt"],
                          columns="measurement", values="value",
                          aggfunc="median").reset_index()
    ep = ep.rename(columns=SITES)
    for c in SITES.values():
        if c not in ep:
            ep[c] = np.nan
    # provenance: did both sites come from this same study?
    ep["both_sites"] = ep["root_cm"].notna() & ep["asc_cm"].notna()
    ep = ep.sort_values(["subject_id", "echo_dt"]).reset_index(drop=True)
    print(f"  TTE aortic studies: {len(ep):,} across {ep.subject_id.nunique():,} patients")

    # Greedy earliest-first: keep a study only if >= sep_days after the last kept one.
    keep, last = np.zeros(len(ep), bool), {}
    for i, (sid, d) in enumerate(zip(ep.subject_id.values, ep.echo_dt.values)):
        if sid not in last or (d - last[sid]) / np.timedelta64(1, "D") >= sep_days:
            keep[i] = True
            last[sid] = d
    ep = ep[keep].reset_index(drop=True)
    print(f"  after >={sep_days}d separation: {len(ep):,} episodes / "
          f"{ep.subject_id.nunique():,} patients ({len(ep)/ep.subject_id.nunique():.2f} per patient)")
    return ep


def load_modality_indexes() -> tuple[dict, dict]:
    """Per-subject arrays of (dates, dicom_ids) for frontal CXR, and dates for ECG."""
    cx = pd.read_csv(CXR_META, usecols=["subject_id", "dicom_id", "ViewPosition", "StudyDate"])
    cx = cx[cx["ViewPosition"].isin(["PA", "AP"])].copy()
    cx["date"] = pd.to_datetime(cx["StudyDate"].astype("Int64").astype(str),
                                format="%Y%m%d", errors="coerce")
    cx = cx[cx["date"].notna()]
    eg = pd.read_csv(ECG_RECS, usecols=["subject_id", "ecg_time"])
    eg["date"] = pd.to_datetime(eg["ecg_time"], errors="coerce")
    eg = eg[eg["date"].notna()]
    print(f"  frontal CXRs {len(cx):,} / {cx.subject_id.nunique():,} pts | "
          f"ECGs {len(eg):,} / {eg.subject_id.nunique():,} pts")

    cxr_idx = {s: (g["date"].values.astype("datetime64[D]").astype(int),
                   g["dicom_id"].values) for s, g in cx.groupby("subject_id")}
    ecg_idx = {s: g["date"].values.astype("datetime64[D]").astype(int)
               for s, g in eg.groupby("subject_id")}
    return cxr_idx, ecg_idx


def match_modalities(ep, cxr_idx, ecg_idx, cxr_win, ecg_win):
    """Attach pre-index imaging counts to each episode; emit episode-instance pairs."""
    ep_rows, inst_rows = [], []
    for sid, g in ep.groupby("subject_id", sort=False):
        got = cxr_idx.get(sid)
        if got is None:
            continue                       # no frontal CXR ever -> cannot enter cohort
        cdates, cids = got
        edates = ecg_idx.get(sid)
        for r in g.itertuples(index=False):
            anchor = np.datetime64(r.echo_dt, "D").astype(int)
            lag = cdates - anchor                      # <=0 means CXR precedes echo
            sel = (lag <= 0) & (lag >= -cxr_win)
            if not sel.any():
                continue
            n_ecg = 0
            if edates is not None:
                el = edates - anchor
                n_ecg = int(((el <= 0) & (el >= -ecg_win)).sum())
            ep_rows.append((sid, r.measurement_id, r.echo_dt, r.root_cm, r.asc_cm,
                            bool(r.both_sites), int(sel.sum()), n_ecg))
            for did, l in zip(cids[sel], lag[sel]):
                inst_rows.append((sid, r.measurement_id, did, int(-l)))

    episodes = pd.DataFrame(ep_rows, columns=[
        "subject_id", "measurement_id", "echo_dt", "root_cm", "asc_cm",
        "both_sites", "n_cxr", "n_ecg"])
    instances = pd.DataFrame(inst_rows, columns=[
        "subject_id", "measurement_id", "dicom_id", "days_before_echo"])
    return episodes, instances


def write_manifest(instances: pd.DataFrame, out_path: str) -> None:
    """Resolve dicom_ids to JPG paths and list the ones not yet downloaded."""
    rec = pd.read_csv(CXR_RECS, usecols=["subject_id", "study_id", "dicom_id", "path"])
    rec["dicom_id"] = rec["dicom_id"].astype(str)
    man = pd.DataFrame({"dicom_id": instances["dicom_id"].astype(str).unique()}).merge(
        rec, on="dicom_id", how="left")
    unresolved = man["path"].isna().sum()
    man = man[man["path"].notna()].copy()
    man["jpg_rel"] = man["path"].str.replace(r"\.dcm$", ".jpg", regex=True)
    local = JPG_ROOT + "/" + man["jpg_rel"].str.replace(r"^files/", "", regex=True)
    have = np.fromiter((os.path.exists(p) for p in local), bool, len(man))

    print(f"  unique images referenced : {len(man):,} (unresolved paths: {unresolved})")
    print(f"  already on disk          : {int(have.sum()):,}")
    print(f"  to download              : {int((~have).sum()):,} "
          f"(~{1.54*int((~have).sum())/1024:.0f} GB)")
    man[~have][["subject_id", "study_id", "dicom_id", "jpg_rel"]].to_csv(out_path, index=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sep-days", type=int, default=180)
    p.add_argument("--cxr-win", type=int, default=365)
    p.add_argument("--ecg-win", type=int, default=180)
    p.add_argument("--out-dir", default=PC)
    a = p.parse_args()

    print("[1/5] loading aortic measurements ...")
    rows = load_aortic_rows(STRUCT)
    print("[2/5] building episodes ...")
    ep = build_episodes(rows, a.sep_days)
    print("[3/5] loading modality indexes ...")
    cxr_idx, ecg_idx = load_modality_indexes()
    print("[4/5] matching pre-index imaging ...")
    episodes, instances = match_modalities(ep, cxr_idx, ecg_idx, a.cxr_win, a.ecg_win)

    tri = episodes[episodes["n_ecg"] > 0]
    print(f"\n  CXR-eligible : {len(episodes):,} episodes / {episodes.subject_id.nunique():,} patients"
          f"  root>=4.0 {(episodes.root_cm>=4.0).sum():,}  asc>=4.0 {(episodes.asc_cm>=4.0).sum():,}")
    print(f"  tri-modal    : {len(tri):,} episodes / {tri.subject_id.nunique():,} patients"
          f"  root>=4.0 {(tri.root_cm>=4.0).sum():,}  asc>=4.0 {(tri.asc_cm>=4.0).sum():,}")
    print(f"  ECG binding cost: {len(episodes)-len(tri):,} episodes "
          f"({100*(1-len(tri)/len(episodes)):.1f}%)")

    os.makedirs(a.out_dir, exist_ok=True)
    episodes.to_csv(f"{a.out_dir}/episodes.csv", index=False)
    instances.to_csv(f"{a.out_dir}/episode_cxr_instances.csv", index=False)
    print("[5/5] writing download manifest ...")
    write_manifest(instances, f"{a.out_dir}/cxr_download_manifest.csv")
    print(f"\nwrote episodes.csv ({len(episodes):,}), episode_cxr_instances.csv "
          f"({len(instances):,}), cxr_download_manifest.csv")


if __name__ == "__main__":
    main()
