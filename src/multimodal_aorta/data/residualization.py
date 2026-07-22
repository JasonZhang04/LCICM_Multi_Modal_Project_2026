"""
M3 — leakage-safe residualized-diameter modeling.

Central scientific question of the project: do ECG/CXR add information about aortic
size BEYOND age, sex, and body size? Residualization answers it directly:

  1. Fit a clinical (EHR-only) regressor on the outer-TRAIN patients.
  2. Predict expected diameter for everyone; residual = actual - expected.
  3. Train an imaging (ECG+CXR) model on the outer-train RESIDUALS.
  4. Outer-test reconstruction = clinical_expected + imaging_residual.

If the reconstructed diameter beats the clinical-only prediction (paired, on the
same held-out patients), imaging adds size-independent signal. If not, it doesn't.

CRITICAL leakage rule: the clinical model is refit INSIDE every outer fold on that
fold's training patients only — never once on the full data. Both clinical and
imaging models here are NaN-native (HistGradientBoosting), so EHR missingness needs
no imputation.
"""

from typing import Callable, List, Tuple

import numpy as np


def residual_oof_regression(
    diam: np.ndarray,
    ehr_X: np.ndarray,
    img_X: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    idx_of: dict,
    clinical_factory: Callable,
    imaging_factory: Callable,
    min_train: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Out-of-fold reconstructed diameter and clinical-only diameter.

    Parameters
    ----------
    diam : (N,) actual diameter in cm (NaN for missing-site patients).
    ehr_X : (N, d_ehr) clinical features (the "expected size" model input).
    img_X : (N, d_img) imaging features (ECG+CXR) that predict the residual.
    folds : list of (train_ids, test_ids) subject_id arrays (immutable outer folds).
    idx_of : {subject_id -> row index into diam/ehr_X/img_X}.
    clinical_factory, imaging_factory : callables returning fresh regressors.

    Returns
    -------
    oof_final : (N,) clinical_expected + imaging_residual on held-out patients.
    oof_clinical : (N,) clinical_expected only (the within-protocol EHR floor).
    """
    N = len(diam)
    oof_final = np.full(N, np.nan)
    oof_clinical = np.full(N, np.nan)
    for tr_ids, te_ids in folds:
        tr = [idx_of[s] for s in tr_ids if s in idx_of and not np.isnan(diam[idx_of[s]])]
        te = [idx_of[s] for s in te_ids if s in idx_of and not np.isnan(diam[idx_of[s]])]
        if len(tr) < min_train:
            continue
        tr = np.asarray(tr)
        te = np.asarray(te)
        # 1-2) clinical model on outer-train -> expected diameter + residuals
        clin = clinical_factory().fit(ehr_X[tr], diam[tr])
        resid_tr = diam[tr] - clin.predict(ehr_X[tr])
        # 3) imaging model predicts the residual
        img = imaging_factory().fit(img_X[tr], resid_tr)
        # 4) held-out reconstruction
        clin_te = clin.predict(ehr_X[te])
        oof_clinical[te] = clin_te
        oof_final[te] = clin_te + img.predict(img_X[te])
    return oof_final, oof_clinical
