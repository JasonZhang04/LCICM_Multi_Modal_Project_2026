"""
Acceptance test for review issue A1: outer-test labels must not influence anything
except the score.

The decisive check is a PERTURBATION test. Corrupt the labels of the outer-test
patients and rerun. A correctly nested pipeline produces bit-identical predictions,
because no fitted object ever saw those labels. The previous global-OOF stack does
not: its fusion-training features are produced by base models trained on the outer
test fold, so corrupting those labels moves the predictions.

Run:
  /scratch4/rsteven1/your_env_name/bin/python3.10 -m pytest \
      src/multimodal_aorta/tests/test_nested_leakage.py -q
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from multimodal_aorta.training.nested import (      # noqa: E402
    BaseSpec, nested_stack, inner_folds_by_patient, assert_patient_disjoint)


# ---------------------------------------------------------------- fixtures
def _synth(n_pt=300, seed=0):
    """Synthetic episodes with a real signal, patient-grouped, some repeats."""
    rng = np.random.default_rng(seed)
    sids, rows = [], []
    for p in range(n_pt):
        for _ in range(rng.integers(1, 3)):
            sids.append(p)
            rows.append(rng.standard_normal(4))
    sid = np.array(sids)
    X = np.array(rows)
    y = X[:, 0] * 1.5 + X[:, 1] * 0.5 + rng.standard_normal(len(X)) * 0.4 + 3.5
    eids = [f"e{i}" for i in range(len(X))]
    row_of = {e: i for i, e in enumerate(eids)}
    return X, y, sid, np.array(eids), row_of


def _outer_folds(eids, sid, n=5, seed=0):
    pts = np.unique(sid)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pts))
    assign = {int(pts[order[i]]): i % n for i in range(len(pts))}
    who = np.array([assign[int(s)] for s in sid])
    return [(eids[who != k], eids[who == k]) for k in range(n)]


def _ridge_fusion(Ftr, ytr, gtr, Fte):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Ftr)
    m = Ridge(alpha=1.0).fit(sc.transform(Ftr), ytr)
    return m.predict(sc.transform(Fte)), 1.0


def _base_from(X):
    from sklearn.ensemble import HistGradientBoostingRegressor
    def fp(train_rows, predict_rows, y):
        m = HistGradientBoostingRegressor(max_iter=40, random_state=0)
        m.fit(X[train_rows], y[train_rows])
        return m.predict(X[predict_rows])
    return BaseSpec(name="cxr_like", fit_predict=fp)


# ------------------------------------------------- the old (leaky) construction
def _global_oof(folds, X, y, sid, row_of):
    """Reproduces the previous design: ONE global OOF matrix reused for stacking."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    out = np.full(len(y), np.nan)
    for tr_e, te_e in folds:
        tr = np.array([row_of[e] for e in tr_e]); te = np.array([row_of[e] for e in te_e])
        m = HistGradientBoostingRegressor(max_iter=40, random_state=0).fit(X[tr], y[tr])
        out[te] = m.predict(X[te])
    return out


def _legacy_stack(folds, X, y, sid, row_of):
    d = _global_oof(folds, X, y, sid, row_of)
    out = np.full(len(y), np.nan)
    for tr_e, te_e in folds:
        tr = np.array([row_of[e] for e in tr_e]); te = np.array([row_of[e] for e in te_e])
        F = np.column_stack([d, X[:, 2]])
        p, _ = _ridge_fusion(F[tr], y[tr], sid[tr], F[te])
        out[te] = p
    return out


# ---------------------------------------------------------------- the tests
def test_nested_predictions_are_invariant_to_outer_test_labels():
    """THE acceptance test: corrupting outer-test labels must change nothing."""
    X, y, sid, eids, row_of = _synth()
    folds = _outer_folds(eids, sid, seed=1)
    base = _base_from(X)

    def run(yy):
        return nested_stack(folds, row_of, yy, sid, [base], _ridge_fusion,
                            n_inner=4, seed=0).predictions

    p_clean = run(y)

    rng = np.random.default_rng(99)
    for k, (_, te_e) in enumerate(folds):
        te = np.array([row_of[e] for e in te_e])
        y_bad = y.copy()
        y_bad[te] = rng.standard_normal(len(te)) * 10 + 50      # wildly corrupted
        p_bad = run(y_bad)
        # Compare fold k's OWN predictions. Those rows are the outer test set for
        # fold k, so nothing fitted to produce them may have seen their labels.
        # (Rows outside fold k legitimately move: for their own outer folds these
        # perturbed rows are training data.)
        both = ~np.isnan(p_clean[te]) & ~np.isnan(p_bad[te])
        assert both.any(), f"fold {k} produced no predictions"
        assert np.allclose(p_clean[te][both], p_bad[te][both], atol=1e-10), (
            f"outer-test labels of fold {k} influenced its own predictions -- not nested")


def test_legacy_global_oof_stack_is_detectably_leaky():
    """Guard rail: the test above is only meaningful if it can fail. The previous
    construction must be caught by it."""
    X, y, sid, eids, row_of = _synth()
    folds = _outer_folds(eids, sid, seed=1)

    p_clean = _legacy_stack(folds, X, y, sid, row_of)
    rng = np.random.default_rng(99)
    te = np.array([row_of[e] for e in folds[0][1]])
    y_bad = y.copy()
    y_bad[te] = rng.standard_normal(len(te)) * 10 + 50
    p_bad = _legacy_stack(folds, X, y_bad, sid, row_of)

    # Fold 0's own predictions must be invariant under a correct design. Under the
    # global-OOF construction they are not: fold-0 labels reached the base models
    # that generated fold 0's fusion-training features.
    ok = ~np.isnan(p_clean[te]) & ~np.isnan(p_bad[te])
    assert not np.allclose(p_clean[te][ok], p_bad[te][ok], atol=1e-10), (
        "legacy stack appeared clean -- the perturbation probe is not sensitive")


def test_inner_folds_never_split_a_patient():
    X, y, sid, eids, row_of = _synth()
    rows = np.arange(len(y))
    for tr, te in inner_folds_by_patient(rows, sid, n_inner=5, seed=0):
        assert_patient_disjoint(tr, te, sid, where="inner")


def test_assert_patient_disjoint_catches_a_real_leak():
    sid = np.array([1, 1, 2, 2, 3])
    assert_patient_disjoint(np.array([0, 1]), np.array([2, 3]), sid)
    try:
        assert_patient_disjoint(np.array([0, 2]), np.array([1, 3]), sid)
    except AssertionError:
        return
    raise AssertionError("patient overlap was not detected")
