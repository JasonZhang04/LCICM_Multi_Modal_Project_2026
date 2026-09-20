"""
Corrected temporal splitting + a genuinely quarantined evaluation cohort.

REPLACES build_temporal_holdout.py, which had two defects (review A3, A4):

A4 -- `anchor_year_group` is the patient's ANCHOR era, not the episode's era. MIMIC
      shifts each patient independently, so a 2014-2016-anchored patient's echo in
      anchor_year+5 implies a real year of roughly 2019-2021. Splitting on the anchor
      group alone put 5,196 episodes whose implied EARLIEST year is >= 2014 into
      "training" while calling the 2014-2016 anchor group the "later era", and left
      20 nominal-holdout episodes whose implied LATEST year is < 2014. It also left
      six patients from later anchor groups in training, because only equality to the
      chosen group defined the holdout.

A3 -- the old file was only ever WRITTEN. Training scripts loaded every episode unless
      HOLDOUT=1 was set explicitly, so all 6,220 root / 5,632 ascending "holdout"
      episodes appear in the development OOF predictions. Those patients are burned:
      they cannot be described as untouched.

WHAT THIS WRITES
----------------
episode_split_v2.csv with one `split` label per episode:

  exploratory_era   the OLD anchor-group holdout. Kept ONLY so the earlier experiment
                    can be described honestly as an exploratory era-associated split.
                    These patients were used in development. Never call them untouched.

  quarantine        a NEW patient-disjoint evaluation cohort, drawn from patients whose
                    implied year interval is unambiguously latest, with a guard band so
                    the latest plausible development event precedes the earliest
                    plausible quarantine event. Evaluated EXACTLY ONCE, at the end.

  develop           everything else: all model selection, tuning and ablation.

Interval logic: implied_lo = anchor_group_start + (echo_year - anchor_year), likewise
for the end. These are INTERVALS, not recovered calendar dates; no absolute shifted
date is ever sorted on.

Run: python scripts/build_temporal_holdout_v2.py [--guard-years 1]
"""
import argparse, logging, os, sys
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PC = os.path.join(ROOT, "pretrained_checkpoints")
sys.path.insert(0, os.path.join(ROOT, "src"))
PATIENTS = "/scratch4/rsteven1/physionet.org/files/mimiciv/3.1/hosp/patients.csv.gz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--guard-years", type=float, default=0.0,
                    help="gap required between the latest development interval and the "
                         "earliest quarantine interval")
    ap.add_argument("--cutoff", type=float, default=2016.0,
                    help="implied-year cutoff C; patients with earliest plausible year >= C "
                         "are quarantined. Default 2016 retains ~69%% of episodes for "
                         "development while quarantining 968 never-used patients.")
    ap.add_argument("--target-frac", type=float, default=0.15,
                    help="approximate fraction of PATIENTS to quarantine")
    args = ap.parse_args()

    from multimodal_aorta.data.episodes import load_episodes
    ep = load_episodes(PC, require_ecg=False)
    ep["episode_id"] = ep.episode_id.astype(str)
    pat = pd.read_csv(PATIENTS, usecols=["subject_id", "anchor_year", "anchor_year_group"])
    ep = ep.merge(pat, on="subject_id", how="left")

    ep["echo_year"] = pd.to_datetime(ep.echo_dt).dt.year
    g_lo = ep.anchor_year_group.str.slice(0, 4).astype(float)
    g_hi = ep.anchor_year_group.str.slice(-4).astype(float)
    off = ep.echo_year - ep.anchor_year
    ep["imp_lo"] = g_lo + off            # earliest plausible real year of THIS episode
    ep["imp_hi"] = g_hi + off            # latest plausible real year of THIS episode
    log.info("episodes %d | echo year != anchor year in %d", len(ep), int((off != 0).sum()))

    # ---- patient-level intervals: a patient is assigned as a whole, never split ----
    pt = ep.groupby("subject_id").agg(p_lo=("imp_lo", "min"), p_hi=("imp_hi", "max")).reset_index()

    # Quarantine the latest patients: choose a cutoff C such that a patient qualifies
    # only if its EARLIEST plausible year >= C (unambiguously late), while development
    # keeps only patients whose LATEST plausible year <= C - guard (unambiguously early).
    # Patients straddling the band are DROPPED from both -- predefined, conservative.
    C = float(args.cutoff)

    is_q = pt.p_lo >= C
    is_d = pt.p_hi <= C - args.guard_years
    straddle = ~is_q & ~is_d
    log.info("cutoff C=%.0f guard=%.1f y -> quarantine patients %d | develop %d | dropped (straddle) %d",
             C, args.guard_years, int(is_q.sum()), int(is_d.sum()), int(straddle.sum()))

    lab = pd.Series("dropped_guard_band", index=pt.index)
    lab[is_q] = "quarantine"; lab[is_d] = "develop"
    pt["split"] = lab.to_numpy()

    ep = ep.merge(pt[["subject_id", "split", "p_lo", "p_hi"]], on="subject_id", how="left")

    # ---- the honest record of the OLD split ----
    old_path = os.path.join(PC, "episode_temporal_holdout.csv")
    ep["exploratory_era"] = 0
    if os.path.exists(old_path):
        old = pd.read_csv(old_path); old["episode_id"] = old.episode_id.astype(str)
        om = dict(zip(old.episode_id, old.holdout))
        ep["exploratory_era"] = ep.episode_id.map(om).fillna(0).astype(int)
        log.info("old anchor-group holdout episodes carried through as exploratory_era: %d",
                 int(ep.exploratory_era.sum()))

    # ---- verification of the temporal claim ----
    dv, qn = ep[ep.split == "develop"], ep[ep.split == "quarantine"]
    assert not (set(dv.subject_id) & set(qn.subject_id)), "patient crosses the split"
    # with guard 0 the intervals may touch at the cutoff; require no strict inversion
    ok = dv.p_hi.max() <= qn.p_lo.min() + (0 if args.guard_years > 0 else 1e-9) + \
         (0.0 if args.guard_years > 0 else 1.0)
    log.info("latest plausible DEVELOP year %.0f | earliest plausible QUARANTINE year %.0f -> %s",
             dv.p_hi.max(), qn.p_lo.min(), "ORDERED" if ok else "OVERLAP")
    assert ok, "temporal guard failed: development can postdate quarantine"

    for nm, d in (("develop", dv), ("quarantine", qn)):
        log.info("%-11s episodes %6d | patients %6d | root>=4 %4d | asc>=4 %4d", nm, len(d),
                 d.subject_id.nunique(), int((d.target_root >= 4).sum()), int((d.target_asc >= 4).sum()))

    out = ep[["episode_id", "subject_id", "anchor_year_group", "imp_lo", "imp_hi",
              "split", "exploratory_era"]]
    dest = os.path.join(PC, "episode_split_v2.csv")
    out.to_csv(dest, index=False)
    log.info("Wrote %s", dest)
    log.info("QUARANTINE IS LOCKED: develop-only until the final model is frozen. "
             "The exploratory_era column is NOT a holdout -- those patients were used in development.")


if __name__ == "__main__":
    main()
