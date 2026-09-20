"""
Review issue A2: a feature matrix must have ONE coordinate system.

`ecg_embeddings.npy` was assembled from the penultimate layers of five independently
trained CNNs (one per fold). Hidden coordinate j of fold 0's network has no defined
relationship to coordinate j of fold 1's; PCA cannot align them. The fusion model then
trains on four networks' coordinates and tests on a fifth.

The diagnostic: if a classifier can recover which network produced a row, the rows do
not share a basis. Measured on the real artifact this hits 100% on patient-disjoint
held-out rows, versus 20% chance.

These tests are skipped when the artifacts are absent (e.g. a fresh clone).
"""
import os, sys
import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
EMB = os.path.join(ROOT, "outputs", "ecg_waveform_episode", "ecg_embeddings.npy")
IDX = os.path.join(ROOT, "outputs", "ecg_waveform_episode", "ecg_embedding_index.csv")
FOLDS = os.path.join(ROOT, "pretrained_checkpoints", "episode_fold_assignments.csv")

CHANCE = 0.20
TOLERATED = 0.35        # a shared basis should be near chance; well below this


def fold_signature_accuracy(X, fold, group, seed=0):
    """Nearest-centroid recovery of the source fold, patient-disjoint."""
    Z = (X - X.mean(0)) / (X.std(0) + 1e-8)
    rng = np.random.default_rng(seed)
    pts = np.unique(group); rng.shuffle(pts)
    hold = set(pts[: int(len(pts) * 0.3)].tolist())
    h = np.array([g in hold for g in group])
    ks = sorted(np.unique(fold))
    cent = np.vstack([Z[~h][fold[~h] == k].mean(0) for k in ks])
    d = ((Z[h][:, None, :] - cent[None]) ** 2).sum(2)
    pred = np.array(ks)[d.argmin(1)]
    return float((pred == fold[h]).mean()), int(h.sum())


def _load():
    if not (os.path.exists(EMB) and os.path.exists(IDX) and os.path.exists(FOLDS)):
        pytest.skip("ECG embedding artifacts not present")
    E = np.load(EMB)
    ix = pd.read_csv(IDX); ix["episode_id"] = ix.episode_id.astype(str)
    fa = pd.read_csv(FOLDS); fa["episode_id"] = fa.episode_id.astype(str)
    ix = ix.merge(fa[["episode_id", "fold_id"]], on="episode_id", how="left")
    ok = ix.fold_id.notna().to_numpy()
    return E[ok], ix.fold_id[ok].to_numpy(int), ix.subject_id[ok].to_numpy(int)


def test_the_probe_is_calibrated_on_a_shared_basis():
    """Guard rail: on rows that DO share a basis the probe must sit near chance,
    otherwise a failure below would be meaningless."""
    rng = np.random.default_rng(0)
    n = 4000
    group = rng.integers(0, 1200, n)
    fold = rng.integers(0, 5, n)
    X = rng.standard_normal((n, 64))          # one basis, fold ids unrelated to it
    acc, _ = fold_signature_accuracy(X, fold, group)
    assert acc < TOLERATED, f"probe is not calibrated: {acc:.3f} on a shared basis"


@pytest.mark.xfail(reason="review A2: known defect in the current saved artifact; "
                          "this test must PASS once the ECG branch is rebuilt on a "
                          "common basis (nested scalar stacking or a shared encoder)",
                   strict=False)
def test_ecg_embeddings_share_one_basis():
    X, fold, group = _load()
    acc, n = fold_signature_accuracy(X, fold, group)
    assert acc < TOLERATED, (
        f"source fold recovered with {acc:.4%} accuracy on {n} patient-disjoint rows "
        f"(chance {CHANCE:.0%}). The embedding matrix pools incompatible coordinate "
        f"systems from independently trained networks -- it is not a valid feature "
        f"matrix for a single PCA/ridge fit.")
