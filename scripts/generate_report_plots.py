"""
Publication figures for PROGRESS_REPORT.md.

Reads the frozen results JSONs under outputs/ and renders every figure into
figures/ as 300-dpi PNG + SVG. No model is retrained here; this is a pure
reporting pass, so it is safe to run on the login node.

Run:  /scratch4/rsteven1/your_env_name/bin/python3.10 scripts/generate_report_plots.py
"""
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

# ---------------------------------------------------------------- design tokens
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"   # validated categorical slots 1-3
DEEMPH = "#c9c8c2"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8.5,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "text.color": INK,
    "axes.titlecolor": INK,
    "legend.frameon": False,
    "figure.dpi": 110,
})


def parse_ci(s):
    """'0.809 [0.723, 0.886]' -> (0.809, 0.723, 0.886). Floats pass through."""
    if isinstance(s, (int, float)):
        return float(s), None, None
    m = re.match(r"\s*(-?[\d.]+)\s*(?:\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\])?", str(s))
    g = m.groups()
    return float(g[0]), (float(g[1]) if g[1] else None), (float(g[2]) if g[2] else None)


def load(name):
    with open(os.path.join(OUT, name)) as f:
        return json.load(f)


def style_axes(ax, xgrid=True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.grid(axis="x" if xgrid else "y", color=GRID, linewidth=0.8, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def px_x(ax, px):
    inv = ax.transData.inverted()
    return inv.transform((px, 0))[0] - inv.transform((0, 0))[0]


def px_y(ax, px):
    inv = ax.transData.inverted()
    return inv.transform((0, px))[1] - inv.transform((0, 0))[1]


def hbar(ax, x0, value, ycen, height, color, r_px=4, alpha=1.0):
    """Horizontal bar: square at the baseline, 4px rounded data-end."""
    cap = abs(px_y(ax, 24))
    height = min(height, cap)
    r = min(abs(px_x(ax, r_px)), abs(value - x0) * 0.5, height * 0.5)
    y0, y1 = ycen - height / 2, ycen + height / 2
    x1 = value
    verts = [(x0, y0), (x1 - r, y0), (x1, y0), (x1, y0 + r),
             (x1, y1 - r), (x1, y1), (x1 - r, y1), (x0, y1), (x0, y0)]
    codes = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3,
             Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, edgecolor="none",
                           alpha=alpha, zorder=3))


def vbar(ax, y0, value, xcen, width, color, r_px=4):
    """Vertical column: square at the baseline, 4px rounded cap."""
    cap = abs(px_x(ax, 24))
    width = min(width, cap)
    r = min(abs(px_y(ax, r_px)), abs(value - y0) * 0.5, width * 0.5)
    x0, x1 = xcen - width / 2, xcen + width / 2
    y1 = value
    verts = [(x0, y0), (x0, y1 - r), (x0, y1), (x0 + r, y1),
             (x1 - r, y1), (x1, y1), (x1, y1 - r), (x1, y0), (x0, y0)]
    codes = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3,
             Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, edgecolor="none",
                           zorder=3))


def tip_label(ax, value, ycen, text, pad_px=5):
    ax.text(value + px_x(ax, pad_px), ycen, text, va="center", ha="left",
            fontsize=7.6, color=INK2, zorder=4)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".svg"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# =============================================================== FIGURE 1
# Model evolution: AUROC (>=4.0 cm) and diameter R^2, per site, vs the EHR floor.
EVOLUTION = [
    # label,               date,      root_auroc, asc_auroc, root_r2, asc_r2
    ("GBDT early fusion",  "Jun 18",  0.630, 0.683, 0.306, 0.092),
    ("Deep transformer",   "Jun 18",  0.651, 0.669, 0.242, 0.082),
    ("Late fusion v1",     "Jun 18",  0.728, 0.678, 0.329, 0.184),
    ("Late fusion v2",     "Jul 09",  0.769, 0.695, 0.326, 0.187),
    ("Combined stack",     "Jul 16",  0.772, 0.756, 0.344, 0.203),
    ("Geometry stack",     "Jul 22",  0.809, 0.790, 0.354, 0.221),
    ("+ cross-site + ECG", "Jul 22",  0.815, 0.806, 0.358, 0.235),
]
FLOOR = {"root_auroc": 0.776, "asc_auroc": 0.668, "root_r2": 0.309, "asc_r2": 0.143}


def figure1():
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.4))
    panels = [
        (axes[0, 0], 2, S1, "a  Aortic root — AUROC (≥4.0 cm)", 0.5, 0.88, "AUROC", "root_auroc"),
        (axes[0, 1], 3, S2, "b  Ascending aorta — AUROC (≥4.0 cm)", 0.5, 0.88, "AUROC", "asc_auroc"),
        (axes[1, 0], 4, S1, "c  Aortic root — diameter R²", 0.0, 0.42, "R²", "root_r2"),
        (axes[1, 1], 5, S2, "d  Ascending aorta — diameter R²", 0.0, 0.42, "R²", "asc_r2"),
    ]
    labels = [f"{n}\n{d}" for n, d, *_ in EVOLUTION]
    ys = np.arange(len(EVOLUTION))[::-1]

    for ax, col, color, title, x0, x1, xlab, key in panels:
        vals = [row[col] for row in EVOLUTION]
        ax.set_xlim(x0, x1)
        ax.set_ylim(-0.7, len(EVOLUTION) - 0.3)
        ax.set_yticks(ys)
        ax.set_yticklabels(labels, fontsize=7.4)
        style_axes(ax)
        fig.canvas.draw()
        for y, v in zip(ys, vals):
            best = v == max(vals)
            hbar(ax, x0, v, y, 0.5, color, alpha=1.0 if best else 0.72)
            tip_label(ax, v, y, f"{v:.3f}")
        f = FLOOR[key]
        ax.axvline(f, color=MUTED, linewidth=1.0, zorder=2)
        ax.text(f, len(EVOLUTION) - 0.42, f"  EHR floor {f:.3f}", color=MUTED,
                fontsize=7.2, va="bottom", ha="left")
        ax.set_title(title, fontsize=9.2, loc="left", pad=14, weight="semibold")
        ax.set_xlabel(xlab, fontsize=8)

    fig.suptitle("Model evolution, 18 Jun – 22 Jul 2026 · 5-fold OOF, n = 522 tri-modal cohort",
                 fontsize=10.5, x=0.012, ha="left", y=1.005, weight="semibold")
    fig.text(0.012, -0.018,
             "Bars grow from chance (AUROC 0.5) / zero (R²). Grey rule = full-cohort EHR baseline "
             "(root n≈4,525). Darkest bar = best in panel.",
             fontsize=7.2, color=MUTED, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    save(fig, "fig1_model_evolution")


# =============================================================== FIGURE 2
def figure2():
    rd = load("reg_derived/results.json")["sites"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.0))
    for ax, site, color, title in [
        (axes[0], "root", S1, "a  Aortic root"),
        (axes[1], "asc", S2, "b  Ascending aorta"),
    ]:
        d = rd[site]["ge40"]
        rows = [("Chest X-ray only", parse_ci(d["REG_DERIVED_cxr"])),
                ("EHR only (full cohort)", parse_ci(d["REG_DERIVED_ehr"])),
                ("CXR + EHR stack", parse_ci(d["REG_DERIVED_stack"]))]
        ys = np.arange(len(rows))[::-1]
        ax.set_xlim(0.5, 1.0)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_yticks(ys)
        ax.set_yticklabels([r[0] for r in rows], fontsize=8)
        style_axes(ax)
        fig.canvas.draw()
        for y, (_, (v, lo, hi)) in zip(ys, rows):
            hbar(ax, 0.5, v, y, 0.46, color)
            ax.plot([lo, hi], [y, y], color=INK2, linewidth=1.2, zorder=5,
                    solid_capstyle="butt")
            tip_label(ax, hi, y, f"{v:.3f}")
        ax.set_title(title, fontsize=9.2, loc="left", pad=10, weight="semibold")
        ax.set_xlabel("AUROC, dilation ≥4.0 cm", fontsize=8)

    fig.suptitle("Where the signal lives: the informative modality is site-specific",
                 fontsize=10.5, x=0.012, ha="left", y=1.06, weight="semibold")
    fig.text(0.012, -0.10,
             "Regression-derived scores, 5-fold OOF on n = 522. Thin rules = 95% patient-bootstrap CI. "
             "Root is body-size driven (EHR ≈ CXR); the ascending aorta forms the mediastinal border, "
             "so the X-ray dominates (0.805 vs 0.667).",
             fontsize=7.2, color=MUTED, ha="left")
    fig.tight_layout()
    save(fig, "fig2_modality_contribution")


# =============================================================== FIGURE 3
def figure3():
    rd = load("reg_derived/results.json")["sites"]
    eps = [("Root ≥4.0 cm", "root", "ge40"), ("Root ≥4.5 cm", "root", "ge45"),
           ("Asc ≥4.0 cm", "asc", "ge40"), ("Asc ≥4.5 cm", "asc", "ge45")]
    rows = []
    for lab, site, ep in eps:
        d = rd[site][ep]
        rows.append((lab, d["pos"], parse_ci(d["direct_clf_stack"])[0],
                     parse_ci(d["REG_DERIVED_stack"])[0],
                     parse_ci(d["delta_regderived_vs_directclf"])))

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4), gridspec_kw={"width_ratios": [1.35, 1]})

    ax = axes[0]
    ys = np.arange(len(rows))[::-1].astype(float)
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(-0.75, len(rows) - 0.25)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r[0]}\n{r[1]} positives" for r in rows], fontsize=7.6)
    style_axes(ax)
    fig.canvas.draw()
    for y, r in zip(ys, rows):
        hbar(ax, 0.5, r[2], y + 0.20, 0.34, DEEMPH)
        hbar(ax, 0.5, r[3], y - 0.20, 0.34, S1)
        tip_label(ax, r[2], y + 0.20, f"{r[2]:.3f}")
        tip_label(ax, r[3], y - 0.20, f"{r[3]:.3f}")
    ax.set_title("a  Direct classifier vs regression-derived score", fontsize=9.2,
                 loc="left", pad=10, weight="semibold")
    ax.set_xlabel("AUROC", fontsize=8)
    h = [plt.Line2D([], [], color=c, linewidth=6, solid_capstyle="butt")
         for c in (DEEMPH, S1)]
    ax.legend(h, ["Direct binary classifier", "Rank by predicted diameter"],
              fontsize=7.6, loc="upper center", bbox_to_anchor=(0.5, -0.20),
              ncol=2, labelcolor=INK2, handlelength=1.4)

    ax = axes[1]
    npos = np.array([r[1] for r in rows], dtype=float)
    dv = np.array([r[4][0] for r in rows])
    ax.set_xlim(0, 58)
    ax.set_ylim(-0.03, 0.20)
    style_axes(ax, xgrid=False)
    ax.axhline(0, color=AXIS, linewidth=0.8, zorder=1)
    ax.scatter(npos, dv, s=70, color=S1, zorder=4, edgecolor=SURFACE, linewidth=2)
    # Placed by hand: the two rare endpoints (8 and 10 positives) sit close together.
    offsets = {8: (11, -19, "left"), 10: (12, -3, "left"),
               32: (11, 5, "left"), 48: (-11, 8, "right")}
    for r, x, y in zip(rows, npos, dv):
        dx, dy, ha = offsets[r[1]]
        ax.annotate(f"{r[0]}\nΔ {y:+.3f}", (x, y), textcoords="offset points",
                    xytext=(dx, dy), fontsize=7.2, color=INK2, ha=ha)
    ax.set_title("b  Gain scales inversely with positive count", fontsize=9.2,
                 loc="left", pad=10, weight="semibold")
    ax.set_xlabel("Positive cases at endpoint", fontsize=8)
    ax.set_ylabel("Δ AUROC (regression-derived − direct)", fontsize=8)

    fig.suptitle("Regression-derived classification: the largest single gain of the period",
                 fontsize=10.5, x=0.012, ha="left", y=1.05, weight="semibold")
    fig.text(0.012, -0.20,
             "Regressing the continuous diameter and ranking by the prediction, instead of fitting a "
             "classifier to 8–48 positives. The inverse scaling with positive count identifies the "
             "mechanism: the binary label was discarding information.",
             fontsize=7.2, color=MUTED, ha="left")
    fig.tight_layout()
    save(fig, "fig3_regression_derived")


# =============================================================== FIGURE 4/5
def _window_table():
    rows = []
    for fn in sorted(os.listdir(os.path.join(OUT, "window_experiments"))):
        if fn.endswith(".json"):
            rows.append(load(f"window_experiments/{fn}"))
    return rows


def figure4():
    rows = _window_table()
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.4), sharex=True)
    titles = {("root", "own"): "a  Root — own cohort",
              ("root", "common"): "b  Root — common 381 patients",
              ("asc", "own"): "c  Ascending — own cohort",
              ("asc", "common"): "d  Ascending — common 381 patients"}
    for i, site in enumerate(["root", "asc"]):
        for j, ev in enumerate(["own", "common"]):
            ax = axes[i, j]
            ends = []
            for direction, color, name in [("pre", S1, "Pre-echo only"),
                                           ("sym", S2, "Symmetric ±W")]:
                sub = sorted([r for r in rows if r["direction"] == direction],
                             key=lambda r: r["cxr_w"])
                x = [r["cxr_w"] for r in sub]
                y = [r["sites"][site][ev]["ge40"]["mean"] for r in sub]
                e = [r["sites"][site][ev]["ge40"]["sd"] for r in sub]
                ax.errorbar(x, y, yerr=e, color=color, linewidth=2, marker="o",
                            markersize=7, markeredgecolor=SURFACE, markeredgewidth=2,
                            capsize=0, elinewidth=1.2, zorder=3, label=name,
                            solid_capstyle="round")
                ends.append((y[-1], x[-1]))
            # Separate the two end-labels vertically so they never overlap when the
            # series converge (they differ by <0.005 in several panels).
            ends.sort(reverse=True)
            for (yv, xv), dy in zip(ends, (9, -13)):
                ax.annotate(f"{yv:.3f}", (xv, yv), textcoords="offset points",
                            xytext=(11, dy), fontsize=7.2, color=INK2)
            ax.set_xlim(150, 505)
            ax.set_ylim(0.75, 0.85)
            ax.set_xticks([180, 270, 365, 450])
            style_axes(ax, xgrid=False)
            ax.set_title(titles[(site, ev)], fontsize=9.2, loc="left", pad=8,
                         weight="semibold")
            if i == 1:
                ax.set_xlabel("Chest X-ray window (days from echo)", fontsize=8)
            if j == 0:
                ax.set_ylabel("AUROC, ≥4.0 cm", fontsize=8)
    axes[0, 0].legend(fontsize=7.8, loc="lower right", labelcolor=INK2)

    fig.suptitle("Temporal window design: symmetric gains are case mix, pre-only costs almost nothing",
                 fontsize=10.5, x=0.012, ha="left", y=1.0, weight="semibold")
    fig.text(0.012, -0.02,
             "Mean ± SD over 5 fold seeds. ECG window fixed at ±180 d. Left column scores each "
             "design on its own cohort; right column scores every design on the same 381 patients, which "
             "separates 'more data helped' from 'an easier cohort'.",
             fontsize=7.2, color=MUTED, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    save(fig, "fig4_window_designs")


def figure5():
    rows = _window_table()
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    w = [180, 270, 365, 450]
    xs = np.arange(len(w), dtype=float)
    ax.set_xlim(-0.6, len(w) - 0.4)
    ax.set_ylim(0, 700)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{v} d" for v in w])
    style_axes(ax, xgrid=False)
    fig.canvas.draw()
    for direction, color, off in [("pre", S1, -0.19), ("sym", S2, +0.19)]:
        sub = sorted([r for r in rows if r["direction"] == direction],
                     key=lambda r: r["cxr_w"])
        for x, r in zip(xs, sub):
            vbar(ax, 0, r["n"], x + off, 0.34, color)
            ax.text(x + off, r["n"] + 12, str(r["n"]), ha="center", fontsize=7.4,
                    color=INK2)
            ax.text(x + off, 18, f"{r['asc_pos']}+", ha="center", fontsize=6.9,
                    color=SURFACE, weight="bold")
    handles = [plt.Line2D([], [], color=c, linewidth=6, solid_capstyle="butt")
               for c in (S1, S2)]
    ax.legend(handles, ["Pre-echo only", "Symmetric ±W"], fontsize=7.8,
              loc="upper left", labelcolor=INK2)
    ax.set_xlabel("Chest X-ray window", fontsize=8)
    ax.set_ylabel("Patients in cohort", fontsize=8)
    ax.set_title("Cohort size by window design", fontsize=9.6, loc="left", pad=10,
                 weight="semibold")
    fig.text(0.012, -0.06,
             "White figure inside each bar = ascending-aorta positives (≥4.0 cm). Loosening the "
             "X-ray window is the only lever that adds patients; tightening the ECG window is nearly free.",
             fontsize=7.2, color=MUTED, ha="left")
    fig.tight_layout()
    save(fig, "fig5_window_cohort_sizes")


# =============================================================== FIGURE 6
def figure6():
    st = load("stability/results.json")
    cfgs = [("A_anat+geom", "A: anatomy-ROI crop\n+ geometry", DEEMPH),
            ("B_all3+geom", "B: cls + aorta-pool + heart-pool\n+ geometry", S1)]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 2.9), sharex=True)
    for ax, site, title in [(axes[0], "root", "a  Aortic root"),
                            (axes[1], "asc", "b  Ascending aorta")]:
        ys = np.array([1.0, 0.0])
        for y, (key, lab, color) in zip(ys, cfgs):
            d = st[key][site]["ge40"]
            seeds = d["per_seed"]
            ax.scatter(seeds, [y] * len(seeds), s=52, color=color, zorder=3,
                       edgecolor=SURFACE, linewidth=2)
            ax.plot([d["mean"]] * 2, [y - 0.22, y + 0.22], color=INK, linewidth=1.8,
                    zorder=4, solid_capstyle="butt")
            ax.text(d["mean"], y + 0.30, f"{d['mean']:.3f} ± {d['sd']:.3f}",
                    ha="center", fontsize=7.4, color=INK2)
        ax.set_yticks(ys)
        ax.set_yticklabels([c[1] for c in cfgs], fontsize=7.6)
        ax.set_ylim(-0.6, 1.75)
        ax.set_xlim(0.75, 0.83)
        style_axes(ax, xgrid=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_title(title, fontsize=9.2, loc="left", pad=8, weight="semibold")
        ax.set_xlabel("AUROC, ≥4.0 cm", fontsize=8)

    fig.suptitle("Why repeated cross-validation was necessary",
                 fontsize=10.5, x=0.012, ha="left", y=1.07, weight="semibold")
    fig.text(0.012, -0.13,
             "Each dot = one of 5 independent fold assignments; black rule = mean. On the single original "
             "split, config A led at the root (0.809 vs 0.797) and would have shipped; across seeds B wins "
             "at both sites with lower variance.",
             fontsize=7.2, color=MUTED, ha="left")
    fig.tight_layout()
    save(fig, "fig6_repeated_cv_stability")


# =============================================================== FIGURE 7
ROI_ASC = [("Whole image", 0.706), ("Hard-coded box", 0.692),
           ("Anatomy union (best)", 0.740), ("Tight aorta crop", 0.664)]


def figure7():
    cg = load("cxr_geometry/results.json")["sites"]["asc"]
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 2.9))

    ax = axes[0]
    ys = np.arange(len(ROI_ASC))[::-1]
    ax.set_xlim(0.5, 0.80)
    ax.set_ylim(-0.6, len(ROI_ASC) - 0.4)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in ROI_ASC], fontsize=7.8)
    style_axes(ax)
    fig.canvas.draw()
    best = max(r[1] for r in ROI_ASC)
    for y, (_, v) in zip(ys, ROI_ASC):
        hbar(ax, 0.5, v, y, 0.46, S2, alpha=1.0 if v == best else 0.66)
        tip_label(ax, v, y, f"{v:.3f}")
    ax.set_title("a  CXR field of view", fontsize=9.2, loc="left", pad=10, weight="semibold")
    ax.set_xlabel("Ascending AUROC ≥4.0 cm", fontsize=8)

    feats = [("Embedding only", "emb"), ("Geometry only", "geom"),
             ("Embedding + geometry", "emb+geom")]
    for ax, metric, xmax, xlab, title in [
        (axes[1], "cxr_only_ge40", 0.90, "Ascending AUROC ≥4.0 cm", "b  Feature set — ranking"),
        (axes[2], "cxr_only_r2", 0.28, "Ascending diameter R²", "c  Feature set — regression"),
    ]:
        vals = [parse_ci(cg[f"{k}_{metric}"])[0] for _, k in feats]
        ys = np.arange(len(feats))[::-1]
        x0 = 0.5 if "ge40" in metric else 0.0
        ax.set_xlim(x0, xmax)
        ax.set_ylim(-0.6, len(feats) - 0.4)
        ax.set_yticks(ys)
        ax.set_yticklabels([f[0] for f in feats], fontsize=7.8)
        style_axes(ax)
        fig.canvas.draw()
        for y, v in zip(ys, vals):
            hbar(ax, x0, v, y, 0.46, S2, alpha=1.0 if v == max(vals) else 0.66)
            tip_label(ax, v, y, f"{v:.3f}")
        ax.set_title(title, fontsize=9.2, loc="left", pad=10, weight="semibold")
        ax.set_xlabel(xlab, fontsize=8)

    fig.suptitle("Chest X-ray representation: a principled crop beats both extremes, and geometry is orthogonal",
                 fontsize=10.5, x=0.012, ha="left", y=1.07, weight="semibold")
    fig.text(0.012, -0.13,
             "Chest-X-ray-only learners, ascending aorta. Panel a: over-cropping to the aorta alone is worse "
             "than the whole image. Panels b–c: 17 engineered geometric measurements are weaker than the "
             "RAD-DINO embedding alone, yet the two together beat either.",
             fontsize=7.2, color=MUTED, ha="left")
    fig.tight_layout()
    save(fig, "fig7_cxr_representation")


# =============================================================== FIGURE 8
PCA_TABLE = [
    ("EHR only (equal-n floor)", 0.614, 0.555, True),
    ("Concat, no PCA (1,100 dims)", 0.630, 0.683, False),
    ("PCA e16 + c32 + EHR", 0.663, 0.693, False),
    ("PCA e16 + c16 + EHR", 0.677, 0.666, False),
    ("PCA c32 + EHR (no ECG)", 0.704, 0.690, False),
    ("PCA c32 only (CXR)", 0.690, 0.706, False),
    ("PCA e16 only (ECG)", 0.656, 0.475, False),
]


def figure8():
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    for ax, idx, color, title in [(axes[0], 1, S1, "a  Aortic root"),
                                  (axes[1], 2, S2, "b  Ascending aorta")]:
        ys = np.arange(len(PCA_TABLE))[::-1]
        vals = [r[idx] for r in PCA_TABLE]
        ax.set_xlim(0.45, 0.78)
        ax.set_ylim(-0.6, len(PCA_TABLE) - 0.4)
        ax.set_yticks(ys)
        ax.set_yticklabels([r[0] for r in PCA_TABLE], fontsize=7.6)
        style_axes(ax)
        fig.canvas.draw()
        best = max(vals)
        for y, r in zip(ys, PCA_TABLE):
            v = r[idx]
            c = DEEMPH if r[3] else color
            hbar(ax, 0.45, v, y, 0.46, c, alpha=1.0 if (v == best and not r[3]) else 0.72)
            tip_label(ax, v, y, f"{v:.3f}")
        ax.set_title(title, fontsize=9.2, loc="left", pad=10, weight="semibold")
        ax.set_xlabel("AUROC, ≥4.0 cm", fontsize=8)

    fig.suptitle("Fold-safe PCA of the frozen embeddings, at equal n = 522",
                 fontsize=10.5, x=0.012, ha="left", y=1.05, weight="semibold")
    fig.text(0.012, -0.10,
             "Grey bar = the equal-n EHR baseline. Every PCA configuration beats it for the ascending aorta "
             "with a paired CI excluding zero. Dropping the ECG block often improves the root, which is the "
             "first evidence that the ECG embedding is redundant.",
             fontsize=7.2, color=MUTED, ha="left")
    fig.tight_layout()
    save(fig, "fig8_pca_reduction")


if __name__ == "__main__":
    figure1(); figure2(); figure3(); figure4()
    figure5(); figure6(); figure7(); figure8()
    print("\nAll figures in", FIG)
