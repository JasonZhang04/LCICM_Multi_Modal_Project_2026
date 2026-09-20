"""
Structured training-curve logging for the GPU trainers.

Until 2026-09-14 the only record of a training run was the `fold k step N val X`
line in the SLURM stderr log: no train loss, no learning rate, nothing machine-
readable, nothing plottable. `CurveLogger` writes one CSV row per evaluation
interval so every run's curves can be inspected and compared afterwards with
`scripts/plot_training_curves.py`.

Usage inside a training loop:

    cl = CurveLogger(out_dir, run_name="cxr_finetune", meta={"seed": SEED, ...})
    ...
    cl.train_step(loss.item())                 # every optimizer step
    if step % VAL_EVERY == 0:
        vl = run_val()
        cl.eval_point(fold=k, step=step, val_loss=vl, best=best, bad=bad, lr=lr)
    ...
    cl.close()                                  # writes <out_dir>/training_curves.csv

The train loss recorded at each eval point is the MEAN over the steps since the
previous eval point (a running window), not the noisy single-step value.
"""
from __future__ import annotations

import csv, json, os, time
from typing import Optional


class CurveLogger:
    COLS = ["fold", "step", "train_loss", "val_loss", "best_val", "bad", "lr", "elapsed_s", "wall_time"]

    def __init__(self, out_dir: str, run_name: str = "", meta: Optional[dict] = None):
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, "training_curves.csv")
        self._f = open(self.path, "w", newline="")
        self._w = csv.writer(self._f); self._w.writerow(self.COLS)
        self._sum = 0.0; self._n = 0; self._t0 = time.time()
        with open(os.path.join(out_dir, "training_curves_meta.json"), "w") as f:
            json.dump({"run_name": run_name, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                       **(meta or {})}, f, indent=2)

    def train_step(self, loss: float) -> None:
        self._sum += float(loss); self._n += 1

    def new_fold(self) -> None:
        """Reset the wall clock and the train-loss window at the start of a fold."""
        self._sum = 0.0; self._n = 0; self._t0 = time.time()

    def eval_point(self, fold: int, step: int, val_loss: float, best: float, bad: int,
                   lr: Optional[float] = None) -> None:
        tl = self._sum / self._n if self._n else float("nan")
        self._w.writerow([fold, step, f"{tl:.6f}", f"{float(val_loss):.6f}", f"{float(best):.6f}",
                          int(bad), "" if lr is None else f"{lr:.3e}", f"{time.time() - self._t0:.0f}",
                          time.strftime("%Y-%m-%d %H:%M:%S")])
        self._f.flush()
        self._sum = 0.0; self._n = 0

    def close(self) -> None:
        self._f.close()
