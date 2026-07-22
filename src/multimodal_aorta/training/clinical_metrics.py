"""
M2 — calibration + clinical-utility metrics for the screening framing.

AUROC/AUPRC say how well a model ranks patients; they do NOT say whether the
probabilities are trustworthy (calibration) or whether the model would actually
enrich an echo-referral list (clinical utility). For a "reduce reliance on echo"
pitch, reviewers will expect both.

Everything here is numpy-only (no sklearn / statsmodels / matplotlib import cost)
so it runs cheaply and deterministically. All functions drop NaN rows first.
"""

from typing import Dict, List

import numpy as np


def _clean(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    m = ~(np.isnan(y) | np.isnan(p))
    return y[m], p[m]


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------

def brier_score(y, p) -> float:
    y, p = _clean(y, p)
    if len(y) == 0:
        return float("nan")
    return float(np.mean((p - y) ** 2))


def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def calibration_slope_intercept(y, p) -> Dict[str, float]:
    """
    Calibration slope + intercept via a logistic recalibration model:
        y ~ Bernoulli(sigmoid(intercept + slope * logit(p))).
    Perfect calibration -> slope=1, intercept=0. Slope<1 => over-confident.
    Fit with plain IRLS (Newton) + tiny ridge for numerical stability.
    """
    y, p = _clean(y, p)
    if len(y) < 5 or len(np.unique(y)) < 2:
        return {"slope": float("nan"), "intercept": float("nan")}
    x = _logit(p)
    X = np.column_stack([np.ones_like(x), x])
    b = np.zeros(2)
    for _ in range(100):
        eta = X @ b
        mu = 1.0 / (1.0 + np.exp(-eta))
        W = mu * (1 - mu) + 1e-8
        z = eta + (y - mu) / W
        XtW = X.T * W
        try:
            b_new = np.linalg.solve(XtW @ X + 1e-8 * np.eye(2), XtW @ z)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(b_new - b)) < 1e-9:
            b = b_new
            break
        b = b_new
    return {"intercept": float(b[0]), "slope": float(b[1])}


def reliability_curve(y, p, n_bins: int = 10) -> Dict[str, List[float]]:
    """Quantile-binned reliability curve: mean predicted vs observed frequency."""
    y, p = _clean(y, p)
    if len(y) == 0:
        return {"mean_pred": [], "obs_freq": [], "count": []}
    n_bins = min(n_bins, max(1, len(np.unique(p))))
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    mean_pred, obs_freq, count = [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() == 0:
            continue
        mean_pred.append(float(p[m].mean()))
        obs_freq.append(float(y[m].mean()))
        count.append(int(m.sum()))
    return {"mean_pred": mean_pred, "obs_freq": obs_freq, "count": count}


# --------------------------------------------------------------------------
# Clinical utility (risk enrichment)
# --------------------------------------------------------------------------

def _confusion(y, flagged):
    tp = int(np.sum(flagged & (y == 1)))
    fp = int(np.sum(flagged & (y == 0)))
    fn = int(np.sum(~flagged & (y == 1)))
    tn = int(np.sum(~flagged & (y == 0)))
    pos, neg = tp + fn, tn + fp
    sens = tp / pos if pos else float("nan")
    spec = tn / neg if neg else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    nne = 1.0 / ppv if ppv and not np.isnan(ppv) and ppv > 0 else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "sens": sens, "spec": spec, "ppv": ppv, "npv": npv,
            "nne": nne, "n_flagged": int(tp + fp)}


def metrics_at_topk(y, p, frac: float) -> Dict[str, float]:
    """
    Refer the top `frac` highest-risk patients to echo; report enrichment.
    number_needed_to_echo (nne) = 1 / PPV = echoes per true positive found.
    """
    y, p = _clean(y, p)
    if len(y) == 0:
        return {"frac": frac, "sens": float("nan"), "ppv": float("nan"), "nne": float("nan")}
    k = max(int(round(frac * len(y))), 1)
    thr = np.sort(p)[::-1][k - 1]           # k-th largest score
    out = _confusion(y, p >= thr)
    out["frac"] = frac
    out["threshold"] = float(thr)
    return out


def threshold_for_target(y, p, target: float, mode: str) -> Dict[str, float]:
    """
    Find the operating point meeting a target sensitivity or specificity, and
    report the full confusion profile there.

    mode="sensitivity": highest threshold with sens >= target (max specificity
        subject to the sensitivity floor — the screening-friendly choice).
    mode="specificity": lowest threshold with spec >= target.
    """
    y, p = _clean(y, p)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {"target": target, "mode": mode, "sens": float("nan"), "spec": float("nan")}
    cand = np.unique(p)
    best = None
    for thr in cand:
        c = _confusion(y, p >= thr)
        if mode == "sensitivity" and not np.isnan(c["sens"]) and c["sens"] >= target:
            # prefer the highest such threshold (best specificity)
            if best is None or thr > best[0]:
                best = (thr, c)
        elif mode == "specificity" and not np.isnan(c["spec"]) and c["spec"] >= target:
            if best is None or thr < best[0]:
                best = (thr, c)
    if best is None:
        return {"target": target, "mode": mode, "sens": float("nan"), "spec": float("nan")}
    thr, c = best
    c.update({"target": target, "mode": mode, "threshold": float(thr)})
    return c
