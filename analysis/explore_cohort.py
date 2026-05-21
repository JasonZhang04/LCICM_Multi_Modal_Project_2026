#!/scratch4/rsteven1/your_env_name/bin/python
"""
explore_cohort.py

Dataset overview and cohort overlap analysis for the 2026 Multi-Modal
Aortic Dilation project. Covers three modalities:
  1. Aortic Labels  — from MIMIC-IV-Echo structured measurements
  2. Chest X-ray    — MIMIC-CXR-GS (512px PNG)
  3. ECG            — MIMIC-IV-ECG (12-lead waveforms)

Figures saved to: outputs/cohort_exploration/

Usage:
    /scratch4/rsteven1/your_env_name/bin/python explore_cohort.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ── Paths ─────────────────────────────────────────────────────────────────────
LABELS_CSV     = "/scratch4/rsteven1/chenjia_echo_project/2026 Multi-Modal Project/data/echo/aortic_labels.csv"
CXR_STUDY_CSV  = "/scratch4/rsteven1/MIMIC_CXR_GS/cxr-study-list.csv"
CXR_RECORD_CSV = "/scratch4/rsteven1/MIMIC_CXR_GS/cxr-record-list.csv"
CXR_512_ROOT   = "/scratch4/rsteven1/MIMIC_CXR_GS_512/files"
ECG_RECORD_CSV = "/scratch4/rsteven1/MIMIC_ECG/physionet.org/files/mimic-iv-ecg/1.0/record_list.csv"
ECG_MEAS_CSV   = "/scratch4/rsteven1/MIMIC_ECG/physionet.org/files/mimic-iv-ecg/1.0/machine_measurements.csv"
OUT_DIR        = "/scratch4/rsteven1/chenjia_echo_project/2026 Multi-Modal Project/outputs/cohort_exploration"

# ── Constants ─────────────────────────────────────────────────────────────────
DILATION_ORDER = ["normal", "mildly_dilated", "moderately_dilated", "severely_dilated"]
DILATION_COLORS = {
    "normal":             "#4CAF50",
    "mildly_dilated":     "#FFC107",
    "moderately_dilated": "#FF5722",
    "severely_dilated":   "#B71C1C",
    "missing":            "#9E9E9E",
    "unknown":            "#9E9E9E",
}
SEP = "=" * 70


# ── Helpers ───────────────────────────────────────────────────────────────────

def savefig(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def ordered_bar(ax, counts, order, title, ylabel="Subjects"):
    cats   = [c for c in order if c in counts.index] + \
             [c for c in counts.index if c not in order]
    vals   = [counts.get(c, 0) for c in cats]
    colors = [DILATION_COLORS.get(c, "#9E9E9E") for c in cats]
    bars   = ax.bar(range(len(cats)), vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, rotation=20, ha="right", fontsize=8)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=10)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.01,
                str(val), ha="center", va="bottom", fontsize=8)


# ── Section 1: Aortic Labels ──────────────────────────────────────────────────

def section_labels():
    print(f"\n{SEP}")
    print("SECTION 1 — AORTIC LABELS  (from MIMIC-IV-Echo structured measurements)")
    print(SEP)

    df = pd.read_csv(LABELS_CSV)

    print(f"Subjects:  {len(df):,}")
    print(f"Columns:   {df.columns.tolist()}")
    print(f"\nMissing values:")
    print(df.isnull().sum().to_string())
    print(f"\nAscending aorta diameter (cm):")
    print(f"  {df['ascending_diam_cm'].describe().loc[['min','mean','50%','max']].to_dict()}")
    print(f"\nAortic root / sinus diameter (cm):")
    print(f"  {df['sinus_diam_cm'].describe().loc[['min','mean','50%','max']].to_dict()}")
    print(f"\nascending_diam_label:")
    print(df["ascending_diam_label"].fillna("missing").value_counts().to_string())
    print(f"\nsinus_diam_label:")
    print(df["sinus_diam_label"].fillna("missing").value_counts().to_string())
    print(f"\nany_dilation (ascending OR sinus ≥ mildly_dilated):")
    print(df["any_dilation"].value_counts().to_string())

    # Fig 1a: diameter histograms
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Aortic Diameter Distributions (full echo cohort, n=4,579)", fontsize=13)
    for ax, col, title, threshold, color in [
        (axes[0], "ascending_diam_cm", "Ascending Aorta Diameter", 4.0, "#1976D2"),
        (axes[1], "sinus_diam_cm",     "Aortic Root (Sinus) Diameter", 3.7, "#7B1FA2"),
    ]:
        data = df[col].dropna()
        ax.hist(data, bins=60, color=color, alpha=0.75, edgecolor="white", linewidth=0.4)
        ax.axvline(threshold, color="red", linestyle="--", linewidth=1.5,
                   label=f"Dilation threshold ({threshold} cm)")
        ax.set_xlabel("Diameter (cm)", fontsize=11)
        ax.set_ylabel("Number of subjects", fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
    fig.tight_layout()
    savefig(fig, "fig1a_diameter_distributions.png")

    # Fig 1b: dilation severity bar charts
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Dilation Severity Label Distribution (full echo cohort)", fontsize=13)
    for ax, col, title in [
        (axes[0], "ascending_diam_label", "Ascending Aorta"),
        (axes[1], "sinus_diam_label",     "Aortic Root (Sinus)"),
    ]:
        counts = df[col].fillna("missing").value_counts()
        ordered_bar(ax, counts, DILATION_ORDER, title)
    fig.tight_layout()
    savefig(fig, "fig1b_dilation_labels.png")

    return df


# ── Section 2: Chest X-ray ────────────────────────────────────────────────────

def section_cxr():
    print(f"\n{SEP}")
    print("SECTION 2 — CHEST X-RAY  (MIMIC-CXR-GS)")
    print(SEP)

    studies = pd.read_csv(CXR_STUDY_CSV)
    records = pd.read_csv(CXR_RECORD_CSV)

    print(f"Dataset root (DICOM):   {os.path.dirname(CXR_STUDY_CSV)}")
    print(f"Dataset root (512 PNG): {CXR_512_ROOT}")
    print(f"\nUnique subjects: {studies['subject_id'].nunique():,}")
    print(f"Unique studies:  {studies['study_id'].nunique():,}")
    print(f"Total images:    {len(records):,}")

    per_subj  = studies.groupby("subject_id")["study_id"].count()
    per_study = records.groupby("study_id")["dicom_id"].count()

    print(f"\nStudies per subject:  median={per_subj.median():.0f}  "
          f"mean={per_subj.mean():.1f}  max={per_subj.max()}")
    print(f"Images per study:     median={per_study.median():.0f}  "
          f"mean={per_study.mean():.1f}  max={per_study.max()}")
    print(f"\nFile format (512px PNG): {CXR_512_ROOT}/p<partition>/p<subject_id>/s<study_id>/<dicom_id>.png")

    # Fig 2: studies/subject and images/study
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("MIMIC-CXR-GS Dataset Distributions", fontsize=13)

    axes[0].hist(per_subj.clip(upper=20), bins=20, color="#1976D2", alpha=0.8,
                 edgecolor="white", linewidth=0.4)
    axes[0].set_xlabel("Studies per subject (clipped at 20)", fontsize=10)
    axes[0].set_ylabel("Number of subjects", fontsize=10)
    axes[0].set_title(f"Studies per Subject  (total {studies['subject_id'].nunique():,} subjects)", fontsize=10)

    axes[1].hist(per_study.clip(upper=8), bins=8, color="#0288D1", alpha=0.8,
                 edgecolor="white", linewidth=0.4)
    axes[1].set_xlabel("Images per study (clipped at 8)", fontsize=10)
    axes[1].set_ylabel("Number of studies", fontsize=10)
    axes[1].set_title(f"Images per Study  (total {len(records):,} images)", fontsize=10)

    fig.tight_layout()
    savefig(fig, "fig2_cxr_distributions.png")

    return studies


# ── Section 3: ECG ────────────────────────────────────────────────────────────

def section_ecg():
    print(f"\n{SEP}")
    print("SECTION 3 — ECG  (MIMIC-IV-ECG v1.0)")
    print(SEP)

    records = pd.read_csv(ECG_RECORD_CSV)
    meas    = pd.read_csv(ECG_MEAS_CSV,
                          usecols=["subject_id", "study_id", "rr_interval", "qrs_axis"])

    print(f"Dataset root: {os.path.dirname(ECG_RECORD_CSV)}")
    print(f"\nUnique subjects: {records['subject_id'].nunique():,}")
    print(f"Unique studies:  {records['study_id'].nunique():,}")

    per_subj = records.groupby("subject_id")["study_id"].count()
    print(f"\nStudies per subject:  median={per_subj.median():.0f}  "
          f"mean={per_subj.mean():.1f}  max={per_subj.max()}")
    print(f"\nWaveform format: .dat / .hea  (WFDB, 12-lead, 500 Hz)")
    print(f"Path pattern: files/p<partition>/p<subject_id>/s<study_id>/<study_id>.dat")

    rr = pd.to_numeric(meas["rr_interval"], errors="coerce")
    print(f"\nRR interval (ms):  {rr.describe().loc[['min','mean','50%','max']].to_dict()}")

    # Fig 3: studies/subject and RR interval distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("MIMIC-IV-ECG Dataset Distributions", fontsize=13)

    axes[0].hist(per_subj.clip(upper=20), bins=20, color="#388E3C", alpha=0.8,
                 edgecolor="white", linewidth=0.4)
    axes[0].set_xlabel("Studies per subject (clipped at 20)", fontsize=10)
    axes[0].set_ylabel("Number of subjects", fontsize=10)
    axes[0].set_title(f"ECG Studies per Subject  (total {records['subject_id'].nunique():,} subjects)", fontsize=10)

    rr_clipped = rr[(rr >= 300) & (rr <= 1500)].dropna()
    axes[1].hist(rr_clipped, bins=60, color="#43A047", alpha=0.8,
                 edgecolor="white", linewidth=0.4)
    axes[1].set_xlabel("RR Interval (ms)", fontsize=10)
    axes[1].set_ylabel("Number of ECGs", fontsize=10)
    axes[1].set_title("RR Interval Distribution (300–1500 ms)", fontsize=10)

    fig.tight_layout()
    savefig(fig, "fig3_ecg_distributions.png")

    return records


# ── Section 4: Cohort Overlap ─────────────────────────────────────────────────

def section_overlap(labels_df, cxr_studies, ecg_records):
    print(f"\n{SEP}")
    print("SECTION 4 — COHORT OVERLAP")
    print(SEP)

    S_label = set(labels_df["subject_id"])
    S_cxr   = set(cxr_studies["subject_id"])
    S_ecg   = set(ecg_records["subject_id"])

    only_label    = S_label - S_cxr - S_ecg
    only_cxr      = S_cxr - S_label - S_ecg
    only_ecg      = S_ecg - S_label - S_cxr
    label_cxr     = (S_label & S_cxr) - S_ecg
    label_ecg     = (S_label & S_ecg) - S_cxr
    cxr_ecg       = (S_cxr & S_ecg) - S_label
    label_cxr_ecg = S_label & S_cxr & S_ecg

    print(f"Aortic label subjects:       {len(S_label):,}")
    print(f"CXR subjects:                {len(S_cxr):,}")
    print(f"ECG subjects:                {len(S_ecg):,}")
    print()
    print(f"Labels ∩ CXR (no ECG):       {len(label_cxr):,}")
    print(f"Labels ∩ ECG (no CXR):       {len(label_ecg):,}")
    print(f"CXR ∩ ECG (no Labels):       {len(cxr_ecg):,}")
    print(f"Labels ∩ CXR ∩ ECG (cohort): {len(label_cxr_ecg):,}  ← training cohort")

    cohort = labels_df[labels_df["subject_id"].isin(label_cxr_ecg)].copy()
    print(f"\nTraining cohort label distribution:")
    print(f"  any_dilation: {cohort['any_dilation'].value_counts().to_dict()}")
    print(f"  ascending_diam_label:")
    print(cohort["ascending_diam_label"].fillna("missing").value_counts().to_string())
    print(f"  sinus_diam_label:")
    print(cohort["sinus_diam_label"].fillna("missing").value_counts().to_string())

    # ── Fig 4a: Venn diagram ──
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Subject Overlap Across Modalities  (MIMIC-IV)", fontsize=13, pad=15)

    cx_L, cy_L = 3.8, 6.2
    cx_C, cy_C = 6.2, 6.2
    cx_E, cy_E = 5.0, 4.0
    radius = 2.2

    for (cx, cy), color in [(( cx_L, cy_L), "#1976D2"),
                             ((cx_C, cy_C), "#F57C00"),
                             ((cx_E, cy_E), "#388E3C")]:
        ax.add_patch(Circle((cx, cy), radius, color=color, alpha=0.30, zorder=1))
        ax.add_patch(Circle((cx, cy), radius, fill=False, edgecolor=color,
                            linewidth=1.5, zorder=2))

    region_texts = [
        (cx_L - 1.4, cy_L + 1.1, f"Labels only\n{len(only_label):,}",       "#1976D2"),
        (cx_C + 1.3, cy_C + 1.1, f"CXR only\n{len(only_cxr):,}",            "#F57C00"),
        (cx_E,       cy_E - 1.7, f"ECG only\n{len(only_ecg):,}",            "#388E3C"),
        (5.0,        cy_L + 0.3, f"L∩C\n{len(label_cxr):,}",               "#555"),
        (cx_L - 0.2, 5.0,        f"L∩E\n{len(label_ecg):,}",               "#555"),
        (cx_C + 0.1, 5.0,        f"C∩E\n{len(cxr_ecg):,}",                 "#555"),
        (5.0,        5.45,       f"L∩C∩E\n{len(label_cxr_ecg):,}",         "black"),
    ]
    for x, y, txt, color in region_texts:
        ax.text(x, y, txt, ha="center", va="center", fontsize=9,
                fontweight="bold", color=color, zorder=3)

    ax.text(cx_L, cy_L + radius + 0.3, "Aortic Labels\n(Echo)",
            ha="center", fontsize=10, color="#1976D2", fontweight="bold")
    ax.text(cx_C, cy_C + radius + 0.3, "Chest X-ray\n(CXR)",
            ha="center", fontsize=10, color="#F57C00", fontweight="bold")
    ax.text(cx_E, cy_E - radius - 0.55, "ECG",
            ha="center", fontsize=10, color="#388E3C", fontweight="bold")

    fig.tight_layout()
    savefig(fig, "fig4a_venn_overlap.png")

    # ── Fig 4b: bar chart of label-cohort subsets ──
    fig, ax = plt.subplots(figsize=(9, 5))
    labels_text = [
        "Labels\nonly", "Labels ∩ CXR\n(no ECG)",
        "Labels ∩ ECG\n(no CXR)", "Labels ∩ CXR ∩ ECG\n(training cohort)",
    ]
    values = [len(only_label), len(label_cxr), len(label_ecg), len(label_cxr_ecg)]
    colors = ["#90CAF9", "#FFB74D", "#81C784", "#1976D2"]
    bars = ax.bar(range(len(labels_text)), values, color=colors,
                  edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(labels_text)))
    ax.set_xticklabels(labels_text, fontsize=10)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                f"{val:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Number of subjects", fontsize=11)
    ax.set_title("Subject Counts by Modality Availability (within label cohort)", fontsize=12)
    ax.set_ylim(0, max(values) * 1.15)
    fig.tight_layout()
    savefig(fig, "fig4b_overlap_bars.png")

    # ── Fig 4c: training cohort label breakdown ──
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(f"Training Cohort Label Distribution  (n={len(cohort):,})", fontsize=13)

    # Panel 1: binary pie chart (explicit order: Normal first, Dilated second)
    pie_vals   = [int((~cohort["any_dilation"]).sum()), int(cohort["any_dilation"].sum())]
    pie_labels = [f"Normal\n({pie_vals[0]:,})", f"Dilated\n({pie_vals[1]:,})"]
    axes[0].pie(pie_vals, labels=pie_labels, colors=["#66BB6A", "#EF5350"],
                autopct="%1.1f%%", startangle=90, textprops={"fontsize": 10})
    axes[0].set_title("Any Dilation (binary)", fontsize=11)

    # Panel 2: ascending severity
    ordered_bar(axes[1],
                cohort["ascending_diam_label"].fillna("missing").value_counts(),
                DILATION_ORDER, "Ascending Aorta Label")

    # Panel 3: sinus severity
    ordered_bar(axes[2],
                cohort["sinus_diam_label"].fillna("missing").value_counts(),
                DILATION_ORDER, "Aortic Root (Sinus) Label")

    fig.tight_layout()
    savefig(fig, "fig4c_cohort_label_breakdown.png")

    # ── Fig 4d: diameter overlay (normal vs dilated) in training cohort ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Diameter Distributions in Training Cohort  (n={len(cohort):,})", fontsize=13)

    for ax, col, title, threshold in [
        (axes[0], "ascending_diam_cm", "Ascending Aorta Diameter", 4.0),
        (axes[1], "sinus_diam_cm",     "Aortic Root (Sinus) Diameter", 3.7),
    ]:
        normal  = cohort.loc[~cohort["any_dilation"], col].dropna()
        dilated = cohort.loc[ cohort["any_dilation"], col].dropna()
        all_vals = pd.concat([normal, dilated])
        bins = np.linspace(all_vals.min(), all_vals.max(), 50)
        ax.hist(normal,  bins=bins, color="#66BB6A", alpha=0.7,
                label=f"Normal (n={len(normal):,})")
        ax.hist(dilated, bins=bins, color="#EF5350", alpha=0.7,
                label=f"Dilated (n={len(dilated):,})")
        ax.axvline(threshold, color="black", linestyle="--", linewidth=1.2,
                   label=f"Threshold ({threshold} cm)")
        ax.set_xlabel("Diameter (cm)", fontsize=10)
        ax.set_ylabel("Subjects", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8)

    fig.tight_layout()
    savefig(fig, "fig4d_cohort_diameter_overlay.png")

    return cohort


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    labels_df   = section_labels()
    cxr_studies = section_cxr()
    ecg_records = section_ecg()
    cohort      = section_overlap(labels_df, cxr_studies, ecg_records)

    print(f"\n{SEP}")
    print("SUMMARY")
    print(SEP)
    print(f"  Full label cohort:            {len(labels_df):,} subjects")
    print(f"  CXR dataset:                  {cxr_studies['subject_id'].nunique():,} subjects")
    print(f"  ECG dataset:                  {ecg_records['subject_id'].nunique():,} subjects")
    print(f"  Training cohort (L∩CXR∩ECG):  {len(cohort):,} subjects")
    print(f"    → Dilated:                  {int(cohort['any_dilation'].sum()):,}  "
          f"({100*cohort['any_dilation'].mean():.1f}%)")
    print(f"    → Normal:                   {int((~cohort['any_dilation']).sum()):,}  "
          f"({100*(~cohort['any_dilation']).mean():.1f}%)")
    print(f"\nFigures saved to: {OUT_DIR}")
