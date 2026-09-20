"""Read-only aggregate audit of existing artifacts; no training or patient-level output.

Run with the project's Python environment. Optional --bootstrap enables paired
patient-cluster intervals for final versus fine-tuned CXR (existing scores only).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PC = ROOT / "pretrained_checkpoints"
sys.path.insert(0, str(ROOT / "src"))


def emit(name, value):
    print(json.dumps({"check": name, "value": value}, default=lambda x: x.item(), allow_nan=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--raw-labels", action="store_true", help="Stream the 2.6 GB original EAV table")
    args = parser.parse_args()
    ep = pd.read_csv(PC / "episodes.csv", parse_dates=["echo_dt"])
    ep["episode_id"] = ep.subject_id.astype(str) + "_" + ep.measurement_id.astype(str)
    folds = pd.read_csv(PC / "episode_fold_assignments.csv")
    hold = pd.read_csv(PC / "episode_temporal_holdout.csv")
    inst = pd.read_csv(PC / "episode_cxr_instances.csv")
    inst["episode_id"] = inst.subject_id.astype(str) + "_" + inst.measurement_id.astype(str)
    emit("cohort", {"episodes": len(ep), "patients": ep.subject_id.nunique(),
                    "duplicate_episode_ids": ep.episode_id.duplicated().sum(),
                    "patients_spanning_folds": int((folds.groupby("subject_id").fold_id.nunique() > 1).sum()),
                    "patients_spanning_holdout": int((hold.groupby("subject_id").holdout.nunique() > 1).sum()),
                    "cxr_instance_rows": len(inst), "unique_cxr": inst.dicom_id.nunique(),
                    "cxr_same_day_rows": int((inst.days_before_echo == 0).sum()),
                    "cxr_lag_range": [inst.days_before_echo.min(), inst.days_before_echo.max()],
                    "max_images_per_episode": int(inst.groupby("episode_id").size().max()),
                    "max_episodes_per_patient": int(ep.groupby("subject_id").size().max())})
    cxrf = inst.merge(folds[["episode_id", "fold_id"]], on="episode_id", validate="many_to_one")
    emit("image_fold_overlap", {"images_spanning_folds": int((cxrf.groupby("dicom_id").fold_id.nunique() > 1).sum()),
                                "images_reused_across_episodes": int((inst.groupby("dicom_id").episode_id.nunique() > 1).sum())})
    meta = pd.read_csv("/scratch4/rsteven1/MIMIC_CXR_JPG_cohort/mimic-cxr-2.0.0-metadata.csv.gz",
                       usecols=["dicom_id", "StudyDate", "StudyTime", "ViewPosition"])
    timed = inst.merge(ep[["episode_id", "echo_dt"]], on="episode_id", validate="many_to_one").merge(
        meta, on="dicom_id", validate="many_to_one")
    time_text = timed.StudyTime.fillna(0).map(lambda t: f"{t:013.6f}")
    cxr_time = pd.to_datetime(timed.StudyDate.astype(str) + time_text,
                             format="%Y%m%d%H%M%S.%f", errors="coerce")
    after = cxr_time > timed.echo_dt
    emit("cxr_timestamp_order", {"rows_after_echo_timestamp": int(after.sum()),
        "episodes_with_after_echo_cxr": timed.loc[after, "episode_id"].nunique(),
        "missing_study_time": int(timed.StudyTime.isna().sum()),
        "unparsed_cxr_timestamp": int(cxr_time.isna().sum()),
        "unique_image_view_counts": timed.drop_duplicates("dicom_id").ViewPosition.value_counts().to_dict()})
    lag = inst.groupby("episode_id").days_before_echo.mean()
    covered = sum(((lag >= lo) & (lag <= hi)).sum() for lo, hi in [(0, 7), (8, 30), (31, 90), (91, 365)])
    emit("lag_bin_gaps", {"episodes_dropped_between_integer_bins": len(lag) - covered})
    pat = pd.read_csv("/scratch4/rsteven1/physionet.org/files/mimiciv/3.1/hosp/patients.csv.gz",
                      usecols=["subject_id", "anchor_year", "anchor_year_group"])
    era = ep.merge(pat, on="subject_id", validate="many_to_one").merge(
        hold[["episode_id", "holdout"]], on="episode_id", validate="one_to_one")
    delta = era.echo_dt.dt.year - era.anchor_year
    era["earliest_year"] = era.anchor_year_group.str[:4].astype(int) + delta
    era["latest_year"] = era.anchor_year_group.str[-4:].astype(int) + delta
    emit("anchor_era", {"episode_counts": era.anchor_year_group.value_counts().to_dict(),
                        "episode_year_differs_from_anchor": int((delta != 0).sum()),
                        "train_definitely_2014_or_later": int(((era.holdout == 0) & (era.earliest_year >= 2014)).sum()),
                        "holdout_definitely_before_2014": int(((era.holdout == 1) & (era.latest_year < 2014)).sum()),
                        "train_may_2014_or_later": int(((era.holdout == 0) & (era.latest_year >= 2014)).sum()),
                        "holdout_may_before_2014": int(((era.holdout == 1) & (era.earliest_year < 2014)).sum())})
    final = pd.read_csv(ROOT / "outputs/final_model_episode/oof_predictions.csv")
    emit("holdout_in_development_oof", final.merge(hold[["episode_id", "holdout"]], on="episode_id").query(
        "holdout == 1").groupby("site").size().to_dict())
    if args.raw_labels:
        parts = []
        for chunk in pd.read_csv(ROOT / "data/echo/structured-measurement.csv", chunksize=1_000_000,
                                 usecols=["subject_id", "measurement_id", "measurement_datetime",
                                          "measurement", "test_type", "result", "unit"], low_memory=False):
            keep = chunk.measurement.isin(["sinus_diam", "ascending_diam"]) & (chunk.test_type == "tte")
            parts.append(chunk.loc[keep])
        raw = pd.concat(parts, ignore_index=True)
        raw["value"] = pd.to_numeric(raw.result, errors="coerce")
        emit("raw_target_units", raw.groupby(["measurement", "unit"], dropna=False).size().rename(
            "rows").reset_index().to_dict(orient="records"))
        emit("raw_target_ranges", {s: {"rows": len(g), "numeric": int(g.value.notna().sum()),
            "below_1_5": int((g.value < 1.5).sum()), "above_7": int((g.value > 7).sum())}
            for s, g in raw.groupby("measurement")})
        raw["echo_dt"] = pd.to_datetime(raw.measurement_datetime, errors="coerce")
        valid = raw[raw.value.between(1.5, 7) & raw.echo_dt.notna()]
        labels = valid.pivot_table(index=["subject_id", "measurement_id", "echo_dt"], columns="measurement",
                                   values="value", aggfunc="median").reset_index()
        compare = ep.merge(labels, on=["subject_id", "measurement_id", "echo_dt"], how="left", validate="one_to_one")
        emit("raw_target_reproduction", {s: bool(np.allclose(compare[col], compare[raw_col], equal_nan=True))
            for s, col, raw_col in [("root", "root_cm", "sinus_diam"), ("asc", "asc_cm", "ascending_diam")]})
    ehr = pd.read_csv(PC / "ehr_features_episode.csv")
    emit("ehr_missing", ehr[["height_cm", "weight_kg", "bsa", "sbp", "dbp"]].isna().mean().to_dict())
    emit("ehr_offsets", {c: {"negative": int((ehr[c] < 0).sum()), "over_year": int((ehr[c] > 365).sum()),
                            "max": float(ehr[c].max())} for c in ehr if c.startswith("qc_")})
    manifest = pd.read_csv(PC / "rad-dino/training_images.csv")
    train_ids = set(manifest.loc[manifest.rel_image_path.str.contains("MIMIC", case=False), "image_id"].astype(str))
    # Use filename stems as a second exact-ID check, without fuzzy matching.
    train_ids |= set(manifest.loc[manifest.rel_image_path.str.contains("MIMIC", case=False),
                                "rel_image_path"].map(lambda p: Path(p).stem))
    used = set(inst.dicom_id.astype(str))
    emit("rad_dino_pretraining_overlap", {"unique_used_images": len(used), "exact_id_overlap": len(used & train_ids)})
    for broad in [False, True]:
        stem = "ecg_waveform_cohort" + ("_broad" if broad else "")
        ecg = pd.read_csv(PC / (stem + ".csv"))
        emit(stem, {"episodes": len(ecg), "patients": ecg.subject_id.nunique(),
                    "patients_spanning_folds": int((ecg.groupby("subject_id").fold_id.nunique() > 1).sum()),
                    "evaluation_episodes": int((ecg.fold_id >= 0).sum()),
                    "always_train_patient_overlap": len(set(ecg.loc[ecg.fold_id == -1, "subject_id"]) & set(ep.subject_id))})
    small = pd.read_csv(ROOT / "outputs/ecg_waveform_episode/oof_predictions.csv")
    broad = pd.read_csv(ROOT / "outputs/ecg_waveform_episode_broad/oof_predictions.csv")
    matched = small.merge(broad, on=["episode_id", "subject_id", "site"], suffixes=("_small", "_broad"), validate="one_to_one")
    emit("broad_ecg_matched_standalone", {s: {"n": len(g), "small_r2": r2_score(g.diam_true_small, g.pred_ecg_waveform_small),
        "broad_r2": r2_score(g.diam_true_small, g.pred_ecg_waveform_broad)} for s, g in matched.groupby("site")})
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.neighbors import NearestCentroid
    from sklearn.preprocessing import StandardScaler
    eix = pd.read_csv(ROOT / "outputs/ecg_waveform_episode/ecg_embedding_index.csv")
    cohort = pd.read_csv(PC / "ecg_waveform_cohort.csv")
    eix = eix[["episode_id"]].merge(cohort[["episode_id", "subject_id", "fold_id"]],
                                   on="episode_id", validate="one_to_one")
    emb = np.load(ROOT / "outputs/ecg_waveform_episode/ecg_embeddings.npy")
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=.3, random_state=42).split(emb, groups=eix.subject_id))
    scaler = StandardScaler().fit(emb[a])
    classifier = NearestCentroid().fit(scaler.transform(emb[a]), eix.fold_id.iloc[a])
    emit("embedding_fold_identity", {"patient_disjoint_test_n": len(b), "chance": .2,
        "nearest_centroid_accuracy": classifier.score(scaler.transform(emb[b]), eix.fold_id.iloc[b])})
    geom_broad = pd.read_csv(ROOT / "outputs/final_model_episode_broadecg/oof_predictions.csv")
    ab = final.merge(geom_broad, on=["episode_id", "site"], suffixes=("_small", "_broad"), validate="one_to_one")
    emit("broad_ab_other_branch_changed", {s: {"n": len(g),
        "geom_predictions_changed": int((np.abs(g.pred_geom_stack_small - g.pred_geom_stack_broad) > 1e-10).sum()),
        "geom_max_abs_diff": float(np.max(np.abs(g.pred_geom_stack_small - g.pred_geom_stack_broad)))} for s, g in ab.groupby("site")})
    ft = pd.read_csv(ROOT / "outputs/cxr_finetune_episode/oof_predictions.csv")
    joined = final.merge(ft, on=["episode_id", "subject_id", "site"], suffixes=("_final", "_ft"), validate="one_to_one")
    for site, g in joined.groupby("site"):
        y = g.diam_true_final.to_numpy(); p = g.pred_final.to_numpy(); q = g.pred_cxr_ft.to_numpy()
        emit("final_vs_cxr_ft_" + site, {"n": len(g), "labels_identical": bool(np.allclose(y, g.diam_true_ft)),
            "final_r2": r2_score(y, p), "ft_r2": r2_score(y, q),
            "final_auroc40": roc_auc_score(y >= 4, p), "ft_auroc40": roc_auc_score(y >= 4, q)})
        for cut in [4, 4.5, 5]:
            yy = y >= cut
            top = p >= np.sort(p)[-max(1, round(len(p) * .05))]
            emit(f"final_{site}_ge{cut}", {"n": len(g), "positive_episodes": int(yy.sum()),
                "positive_patients": g.loc[yy, "subject_id"].nunique(), "auroc": roc_auc_score(yy, p),
                "average_precision": average_precision_score(yy, p), "top5_sensitivity": float(yy[top].sum() / yy.sum()),
                "top5_ppv": float(yy[top].mean()), "mean_error_positive_cm": float((p[yy]-y[yy]).mean())})
        if args.bootstrap:
            from multimodal_aorta.training.bootstrap import paired_cluster_bootstrap_diff, auroc, r2
            groups = g.subject_id.to_numpy()
            emit("paired_final_minus_ft_" + site, {
                "r2_ci": paired_cluster_bootstrap_diff(y, p, q, groups, r2, need_both_classes=False),
                "auroc40_ci": paired_cluster_bootstrap_diff((y >= 4).astype(float), p, q, groups, auroc)})


if __name__ == "__main__":
    main()
