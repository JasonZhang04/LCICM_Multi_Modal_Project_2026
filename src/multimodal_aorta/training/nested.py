"""
Nested cross-validation harness (review issue A1).

THE DEFECT THIS REPLACES
------------------------
The previous stacks built ONE global out-of-fold (OOF) base-feature matrix and then
cross-validated the second-stage model on top of it. Per-patient that looks safe --
every base prediction excludes its own patient -- but it is not nested:

    outer test fold k
      a fusion-TRAINING row in fold j != k carries a base feature produced by a
      base model trained on every fold except j -- which INCLUDES fold k.
      => fold-k labels -> base model -> fusion training features -> fitted fusion

Matching immutable fold ids does not break that path. It matters here because the
scientific claims rest on ~0.01 R^2 increments, the same order as the bias.

WHAT THIS DOES INSTEAD
----------------------
For each outer fold, the outer-test PATIENTS are removed before any supervised
fitting happens:

    1. split the outer-training patients into INNER folds
    2. inner-OOF base predictions: each inner-test block is predicted by a base
       model fit only on the other inner-training blocks (all inside outer-train)
    3. fit the fusion model on those inner-OOF features
    4. REFIT each base learner on all outer-training patients
    5. apply the refit base learners to the outer-test patients, then apply the
       fitted fusion

Every fitted object (PCA, scaler, HGB, ridge, alpha choice) therefore has an
ancestry containing only outer-training patients. Outer-test labels are used at
exactly one place: scoring.

Base learners are supplied as `BaseSpec` objects so the same harness serves the
CXR multi-instance branch, the EHR branch, and any scalar branch. Precomputed
deep-model predictions (fine-tuned CXR, ECG CNN) cannot be refit on CPU, so they
are passed as `PrecomputedSpec` and carry an explicit honesty flag -- see
`NestedResult.exact_nesting`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

Folds = Sequence[Tuple[np.ndarray, np.ndarray]]


# --------------------------------------------------------------------------
# base-learner specification
# --------------------------------------------------------------------------
@dataclass
class BaseSpec:
    """A refittable base learner producing one scalar feature per episode row.

    fit_predict(train_rows, predict_rows, y) -> array aligned to `predict_rows`.
    It must fit ONLY on `train_rows` (including any PCA/scaler/imputation) and must
    not read `y` at any index outside `train_rows`. The harness enforces the row
    discipline; the closure is responsible for not smuggling global state in.
    """
    name: str
    fit_predict: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    exact: bool = True


@dataclass
class PrecomputedSpec:
    """A base feature that cannot be refit inside the harness (e.g. a GPU model).

    `values` is a full-length per-row vector. Using it means the corresponding
    branch is NOT exactly nested: its inner-OOF and outer-test values come from the
    same globally-trained artifact. The harness propagates `exact=False` so the run
    is never silently reported as clean.
    """
    name: str
    values: np.ndarray
    exact: bool = False


@dataclass
class NestedResult:
    predictions: np.ndarray
    exact_nesting: bool
    inexact_components: List[str] = field(default_factory=list)
    chosen_alphas: List[float] = field(default_factory=list)
    n_outer: int = 0


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def inner_folds_by_patient(rows: np.ndarray, sid: np.ndarray, n_inner: int,
                           seed: int = 0) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Patient-grouped inner folds over a set of episode ROW indices.

    Patients are partitioned, never episodes, so repeated episodes of one patient
    cannot straddle an inner boundary (the old ridge_stack tuned alpha with plain
    episode-level KFold, which did straddle -- see review A1).
    """
    pts = np.unique(sid[rows])
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pts))
    assign = {int(pts[order[i]]): i % n_inner for i in range(len(pts))}
    out = []
    who = np.array([assign[int(s)] for s in sid[rows]])
    for k in range(n_inner):
        te = rows[who == k]
        tr = rows[who != k]
        if len(te) and len(tr):
            out.append((tr, te))
    return out


def assert_patient_disjoint(train_rows: np.ndarray, test_rows: np.ndarray,
                            sid: np.ndarray, where: str = "") -> None:
    overlap = set(sid[train_rows].tolist()) & set(sid[test_rows].tolist())
    if overlap:
        raise AssertionError(
            f"patient leak{' in ' + where if where else ''}: "
            f"{len(overlap)} patients on both sides (e.g. {sorted(overlap)[:5]})")


# --------------------------------------------------------------------------
# the harness
# --------------------------------------------------------------------------
def nested_stack(
    outer_folds: Folds,
    row_of: Dict[str, int],
    y: np.ndarray,
    sid: np.ndarray,
    bases: Sequence[BaseSpec | PrecomputedSpec],
    fusion_fit_predict: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, float]],
    extra_features: Optional[Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None,
    n_inner: int = 5,
    seed: int = 0,
) -> NestedResult:
    """Run a properly nested stacked model and return per-row outer-test predictions.

    Parameters
    ----------
    outer_folds : (train_episode_ids, test_episode_ids) per outer fold.
    row_of      : episode_id -> row index into y/sid and the feature arrays.
    y           : per-row target (NaN where unlabelled).
    sid         : per-row subject_id, used for every grouping assertion.
    bases       : refittable BaseSpec and/or non-refittable PrecomputedSpec learners.
    fusion_fit_predict : (F_train, y_train, groups_train, F_test) -> (pred_test, alpha).
                  Fits the second stage on inner-OOF base features only.
    extra_features : optional (train_rows, predict_rows) -> (block_train, block_predict)
                  for non-stacked raw covariates (EHR, ECG embedding). It is called
                  separately for the inner-OOF fit and for the outer refit, so any
                  scaler/PCA inside it is fit on the appropriate training rows only.
    """
    n = len(y)
    out = np.full(n, np.nan)
    inexact = [b.name for b in bases if not getattr(b, "exact", True)]
    alphas: List[float] = []

    for k, (tr_eids, te_eids) in enumerate(outer_folds):
        tr = np.array([row_of[e] for e in map(str, tr_eids) if e in row_of], dtype=int)
        te = np.array([row_of[e] for e in map(str, te_eids) if e in row_of], dtype=int)
        if len(tr) < 50 or not len(te):
            logger.warning("outer fold %d too small (train %d test %d) -- skipped", k, len(tr), len(te))
            continue
        assert_patient_disjoint(tr, te, sid, where=f"outer fold {k}")

        tr_lab = tr[~np.isnan(y[tr])]
        if len(tr_lab) < 50:
            continue

        # ---- 1-2. inner-OOF base features, built only from outer-training patients
        inner = inner_folds_by_patient(tr_lab, sid, n_inner=n_inner, seed=seed + k)
        F_tr = np.full((len(tr_lab), len(bases)), np.nan)
        pos_of = {int(r): i for i, r in enumerate(tr_lab)}
        for bi, b in enumerate(bases):
            if isinstance(b, PrecomputedSpec):
                F_tr[:, bi] = b.values[tr_lab]
                continue
            for itr, ite in inner:
                assert_patient_disjoint(itr, ite, sid, where=f"inner fold, outer {k}, base {b.name}")
                pred = b.fit_predict(itr, ite, y)
                for r, v in zip(ite, np.asarray(pred, dtype=float)):
                    F_tr[pos_of[int(r)], bi] = v

        # ---- 4. refit each base on ALL outer-training patients, predict outer test
        F_te = np.full((len(te), len(bases)), np.nan)
        for bi, b in enumerate(bases):
            if isinstance(b, PrecomputedSpec):
                F_te[:, bi] = b.values[te]
            else:
                F_te[:, bi] = np.asarray(b.fit_predict(tr_lab, te, y), dtype=float)

        # ---- optional raw covariate blocks, fit on the same training rows
        if extra_features is not None:
            Etr_in, _ = extra_features(tr_lab, tr_lab)
            _, Ete = extra_features(tr_lab, te)
            F_tr = np.column_stack([F_tr, Etr_in])
            F_te = np.column_stack([F_te, Ete])

        # ---- 3+5. fit fusion on inner-OOF features; apply to refit outer-test features
        ok = ~np.isnan(F_tr).any(1)
        if ok.sum() < 50:
            logger.warning("outer fold %d: only %d usable fusion rows -- skipped", k, int(ok.sum()))
            continue
        te_ok = ~np.isnan(F_te).any(1)
        if not te_ok.any():
            continue
        pred, alpha = fusion_fit_predict(F_tr[ok], y[tr_lab][ok], sid[tr_lab][ok], F_te[te_ok])
        out[te[te_ok]] = pred
        alphas.append(float(alpha))

    return NestedResult(predictions=out, exact_nesting=len(inexact) == 0,
                        inexact_components=inexact, chosen_alphas=alphas,
                        n_outer=len(outer_folds))
