"""
Bootstrap confidence intervals for OOF metrics (v3 eval polish).

At n~522 with ~32-48 positives, single-number AUROC/AUPRC are noisy. This gives
percentile bootstrap CIs over patients so we can say whether differences
(GBDT vs deep vs late-fusion, CXR vs EHR) are actually distinguishable.

Standalone (depends only on numpy + sklearn); does not import or modify any
existing training/eval code.
"""

from typing import Callable, Dict, Tuple

import numpy as np


def bootstrap_ci(
    y: np.ndarray,
    p: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 2000,
    seed: int = 0,
    need_both_classes: bool = True,
) -> Tuple[float, float, float]:
    """
    Percentile bootstrap CI for metric_fn(y, p).

    NaNs in y or p are dropped first. Returns (point_estimate, lo2.5%, hi97.5%).
    Resamples patients with replacement; skips resamples lacking both classes
    (for classification metrics).
    """
    m = ~np.isnan(y) & ~np.isnan(p)
    y, p = y[m], p[m]
    if len(y) < 2:
        return float("nan"), float("nan"), float("nan")
    point = float(metric_fn(y, p))
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    vals = []
    for _ in range(n_boot):
        s = rng.choice(idx, len(idx), replace=True)
        if need_both_classes and len(np.unique(y[s])) < 2:
            continue
        try:
            vals.append(float(metric_fn(y[s], p[s])))
        except Exception:  # noqa: BLE001
            continue
    if not vals:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_bootstrap_diff(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 2000,
    seed: int = 0,
    need_both_classes: bool = True,
) -> Tuple[float, float, float]:
    """
    Percentile CI for the PAIRED difference metric_fn(y, p_a) - metric_fn(y, p_b).

    This is the core "does model A add value over model B (the EHR floor)?" test:
    both models are scored on the SAME patients in each bootstrap resample, so the
    CI reflects the paired difference (much tighter than differencing two
    independent CIs). Rows with NaN in y, p_a, or p_b are dropped up front so both
    models are compared on the identical patient set.

    Returns (point_difference, lo2.5%, hi97.5%). A CI excluding 0 means the
    difference is distinguishable from zero at ~95%.
    """
    m = ~np.isnan(y) & ~np.isnan(p_a) & ~np.isnan(p_b)
    y, p_a, p_b = y[m], p_a[m], p_b[m]
    if len(y) < 2:
        return float("nan"), float("nan"), float("nan")
    point = float(metric_fn(y, p_a) - metric_fn(y, p_b))
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    diffs = []
    for _ in range(n_boot):
        s = rng.choice(idx, len(idx), replace=True)
        if need_both_classes and len(np.unique(y[s])) < 2:
            continue
        try:
            diffs.append(float(metric_fn(y[s], p_a[s]) - metric_fn(y[s], p_b[s])))
        except Exception:  # noqa: BLE001
            continue
    if not diffs:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def cluster_bootstrap_ci(
    y: np.ndarray,
    p: np.ndarray,
    groups: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 2000,
    seed: int = 0,
    need_both_classes: bool = True,
) -> Tuple[float, float, float]:
    """
    Percentile bootstrap CI that resamples PATIENTS (clusters), not rows.

    In the episode-level cohort each patient contributes several correlated
    episodes, so resampling rows understates uncertainty (it treats an ~1.3x
    inflated, correlated sample as independent). Here we resample whole patients
    with replacement and pool all episodes of each drawn patient, which is the
    standard cluster bootstrap. When every row is its own group this reduces to
    the ordinary row bootstrap.

    `groups` is an array aligned with y/p giving each row's patient id.
    """
    m = ~np.isnan(y) & ~np.isnan(p)
    y, p, groups = y[m], p[m], np.asarray(groups)[m]
    if len(y) < 2:
        return float("nan"), float("nan"), float("nan")
    point = float(metric_fn(y, p))

    uniq = np.unique(groups)
    # Precompute row indices per patient so each resample is a cheap concat.
    rows_of = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        drawn = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([rows_of[g] for g in drawn])
        if need_both_classes and len(np.unique(y[idx])) < 2:
            continue
        try:
            vals.append(float(metric_fn(y[idx], p[idx])))
        except Exception:  # noqa: BLE001
            continue
    if not vals:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_cluster_bootstrap_diff(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    groups: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 2000,
    seed: int = 0,
    need_both_classes: bool = True,
) -> Tuple[float, float, float]:
    """
    Patient-clustered analogue of paired_bootstrap_diff: the paired A-vs-B
    difference test (e.g. multimodal vs the EHR floor) with whole patients drawn
    together. Rows with NaN in y, p_a, or p_b are dropped so both models are
    compared on the identical episode set.
    """
    m = ~np.isnan(y) & ~np.isnan(p_a) & ~np.isnan(p_b)
    y, p_a, p_b, groups = y[m], p_a[m], p_b[m], np.asarray(groups)[m]
    if len(y) < 2:
        return float("nan"), float("nan"), float("nan")
    point = float(metric_fn(y, p_a) - metric_fn(y, p_b))

    uniq = np.unique(groups)
    rows_of = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        drawn = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([rows_of[g] for g in drawn])
        if need_both_classes and len(np.unique(y[idx])) < 2:
            continue
        try:
            diffs.append(float(metric_fn(y[idx], p_a[idx]) - metric_fn(y[idx], p_b[idx])))
        except Exception:  # noqa: BLE001
            continue
    if not diffs:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


# Convenience metric fns ----------------------------------------------------

def auroc(y, p):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y.astype(int), p)


def auprc(y, p):
    from sklearn.metrics import average_precision_score
    return average_precision_score(y.astype(int), p)


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def r2(y, p):
    ss = np.sum((y - p) ** 2)
    tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss / tot) if tot > 0 else float("nan")


def fmt(ci: Tuple[float, float, float]) -> str:
    """'0.695 [0.61, 0.78]'"""
    pt, lo, hi = ci
    return f"{pt:.3f} [{lo:.3f}, {hi:.3f}]"
