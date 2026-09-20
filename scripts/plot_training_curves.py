"""
Training-curve viewer for the GPU trainers (fine-tuned CXR, ECG waveform CNN).

Two sources, newest preferred:
  1. outputs/<run>/training_curves.csv   written by CurveLogger (train + val loss, lr)
  2. logs/*.err SLURM logs               parsed for `fold k step N val X (best Y bad Z)`
                                          -- val loss only; lets every PAST run be viewed.

Produces, under figures/training_curves/:
  <run>.png            one panel per fold: validation loss vs step (+ train loss if logged),
                       best point marked, stop reason in the panel title
  <run>_curves.csv     the plotted numbers (table view)
  compare__A__vs__B.png  overlays of named runs, one panel per fold
  summary.md           best val / step-at-best / total steps / stop reason per run and fold

Run:  python scripts/plot_training_curves.py                 # everything found
      python scripts/plot_training_curves.py --compare cxr_finetune_episode cxr_finetune_episode_ckpt_norm
"""
import argparse, glob, json, os, re, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures", "training_curves")
LINE = re.compile(r"fold (\d+) step (\d+) val ([\d.]+) \(best ([\d.]+),? bad (\d+)\) (\d+)s")
SAVED = re.compile(r"Saved -> .*?outputs/([A-Za-z0-9_]+)")
MAXSTEPS = re.compile(r"MAX_STEPS=(\d+)")

# chart chrome (light surface) and the fixed categorical slot order -- never cycled
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e2"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def from_csv(path):
    d = pd.read_csv(path)
    meta_p = os.path.join(os.path.dirname(path), "training_curves_meta.json")
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    d["source"] = "csv"
    return d, meta


def from_log(path):
    rows, name, max_steps = [], None, None
    for ln in open(path, errors="ignore"):
        m = LINE.search(ln)
        if m:
            f, st, v, b, bad, el = m.groups()
            rows.append({"fold": int(f), "step": int(st), "val_loss": float(v), "best_val": float(b),
                         "bad": int(bad), "elapsed_s": int(el), "train_loss": np.nan, "lr": np.nan})
        m2 = SAVED.search(ln)
        if m2: name = m2.group(1)
        m3 = MAXSTEPS.search(ln)
        if m3: max_steps = int(m3.group(1))
    if not rows:
        return None, None, None
    d = pd.DataFrame(rows); d["source"] = "log"
    return d, name, {"max_steps": max_steps, "log": os.path.basename(path)}


def collect():
    runs = {}
    for p in glob.glob(os.path.join(ROOT, "outputs", "*", "training_curves.csv")):
        d, meta = from_csv(p); runs[os.path.basename(os.path.dirname(p))] = (d, meta)
    for p in sorted(glob.glob(os.path.join(ROOT, "logs", "*.err")), key=os.path.getmtime):
        d, name, meta = from_log(p)
        if d is None or len(d) < 5: continue          # smoke tests / crashes
        name = name or f"unsaved_{os.path.basename(p).split('.')[0]}"
        if name in runs and runs[name][0].source.iloc[0] == "csv": continue   # csv wins
        runs[name] = (d, meta)                       # later log of same name wins
    return runs


def stop_reason(fold_df, meta):
    last = fold_df.iloc[-1]
    ms = (meta or {}).get("max_steps")
    if ms and last.step >= ms: return f"hit max_steps={ms}"
    return f"early stop (patience) at step {int(last.step)}"


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=1, linestyle="-"); ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.xaxis.label.set_color(INK2); ax.yaxis.label.set_color(INK2)


def plot_run(name, d, meta):
    folds = sorted(d.fold.unique()); n = len(folds)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6), squeeze=False, facecolor=SURFACE)
    has_train = d.train_loss.notna().any()
    for ax, k in zip(axes[0], folds):
        f = d[d.fold == k].sort_values("step"); style(ax)
        ax.plot(f.step, f.val_loss, color=SLOTS[0], linewidth=2, solid_joinstyle="round", label="validation")
        if has_train:
            ax.plot(f.step, f.train_loss, color=SLOTS[1], linewidth=2, solid_joinstyle="round", label="train (window mean)")
        i = int(f.val_loss.idxmin()); bx, by = f.loc[i, "step"], f.loc[i, "val_loss"]
        ax.plot([bx], [by], marker="o", markersize=9, markerfacecolor=SLOTS[0], markeredgecolor=SURFACE, markeredgewidth=2, linestyle="none")
        ax.annotate(f"best {by:.3f}", (bx, by), textcoords="offset points", xytext=(6, 8), fontsize=9, color=INK)
        ax.set_title(f"fold {k}  ·  {stop_reason(f, meta)}", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("optimizer step")
    axes[0][0].set_ylabel("loss (MSE, standardized diameter)")
    if has_train: axes[0][0].legend(frameon=False, fontsize=9, labelcolor=INK2)
    sub = ", ".join(f"{k}={v}" for k, v in (meta or {}).items() if k in ("seed", "ft_blocks", "cxr_preproc", "lr", "batch", "log", "broad")) or ""
    fig.suptitle(f"{name}   —   validation loss per fold" + (f"   ({sub})" if sub else ""), fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(FIG, f"{name}.png"); fig.savefig(out, dpi=140, facecolor=SURFACE); plt.close(fig)
    d.sort_values(["fold", "step"]).to_csv(os.path.join(FIG, f"{name}_curves.csv"), index=False)
    return out


def plot_compare(names, runs):
    present = [n for n in names if n in runs]
    if len(present) < 2: return None
    folds = sorted(set.intersection(*[set(runs[n][0].fold.unique()) for n in present])); k = len(folds)
    fig, axes = plt.subplots(1, k, figsize=(4.2 * k, 3.6), squeeze=False, facecolor=SURFACE)
    for ax, fd in zip(axes[0], folds):
        style(ax)
        for si, n in enumerate(present):                     # slot colour follows the RUN, fixed order
            f = runs[n][0]; f = f[f.fold == fd].sort_values("step")
            ax.plot(f.step, f.val_loss, color=SLOTS[si], linewidth=2, solid_joinstyle="round", label=n)
            i = int(f.val_loss.idxmin())
            ax.plot([f.loc[i, "step"]], [f.loc[i, "val_loss"]], marker="o", markersize=9, markerfacecolor=SLOTS[si],
                    markeredgecolor=SURFACE, markeredgewidth=2, linestyle="none")
        ax.set_title(f"fold {fd}", fontsize=10, color=INK, loc="left"); ax.set_xlabel("optimizer step")
    axes[0][0].set_ylabel("validation loss (MSE, standardized diameter)")
    axes[0][0].legend(frameon=False, fontsize=9, labelcolor=INK2)
    fig.suptitle("validation loss per fold  —  " + "  vs  ".join(present), fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(FIG, "compare__" + "__vs__".join(present) + ".png"); fig.savefig(out, dpi=140, facecolor=SURFACE); plt.close(fig)
    return out


def summary(runs):
    rows = []
    for name, (d, meta) in runs.items():
        for k in sorted(d.fold.unique()):
            f = d[d.fold == k].sort_values("step"); i = int(f.val_loss.idxmin())
            rows.append({"run": name, "fold": k, "best_val": round(float(f.loc[i, "val_loss"]), 4),
                         "step_at_best": int(f.loc[i, "step"]), "last_step": int(f.step.iloc[-1]),
                         "stop": stop_reason(f, meta), "source": f.source.iloc[0]})
    s = pd.DataFrame(rows)
    cols = list(s.columns)                        # plain markdown table; no optional deps
    md = "| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    md += "".join("| " + " | ".join(str(v) for v in r) + " |\n" for r in s.itertuples(index=False))
    with open(os.path.join(FIG, "summary.md"), "w") as fh:
        fh.write("# Training-curve summary\n\n" + md)
    return s


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--compare", nargs="*", action="append", default=[])
    a = ap.parse_args(); os.makedirs(FIG, exist_ok=True)
    runs = collect()
    if not runs: sys.exit("no training curves found (no training_curves.csv and no parsable logs)")
    for name, (d, meta) in runs.items():
        print("wrote", plot_run(name, d, meta))
    compares = a.compare or [["cxr_finetune_episode", "cxr_finetune_episode_ckpt_norm"],
                             ["ecg_waveform_episode", "ecg_waveform_episode_broad"]]
    for c in compares:
        out = plot_compare(c, runs)
        if out: print("wrote", out)
    s = summary(runs); print("\n" + s.to_string(index=False)); print(f"\nsummary -> {os.path.join(FIG, 'summary.md')}")


if __name__ == "__main__":
    main()
