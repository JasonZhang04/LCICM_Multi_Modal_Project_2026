"""
Quarantine a temporal held-out test set on anchor_year_group.

MIMIC date-shifts every patient into a 3-year anchor_year_group bucket, so calendar
dates are fictional and the year-group is the only valid temporal axis. We hold out
the LATEST era as a locked test set (train on earlier eras, test on the latest =
temporal generalization), set it aside NOW before the ablation program, and evaluate
it exactly ONCE on the final chosen model.

This script only ASSIGNS and reports; it does not train or evaluate anything. It writes
episode_temporal_holdout.csv (episode_id, subject_id, anchor_year_group, holdout 0/1)
and prints per-era event counts so we can confirm the held-out era is large enough.

Run: python scripts/build_temporal_holdout.py
"""
import logging, os, sys
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PC = os.path.join(ROOT, "pretrained_checkpoints")
sys.path.insert(0, os.path.join(ROOT, "src"))
PATIENTS = "/scratch4/rsteven1/physionet.org/files/mimiciv/3.1/hosp/patients.csv.gz"


def main():
    from multimodal_aorta.data.episodes import load_episodes
    ep = load_episodes(PC, require_ecg=False)
    pat = pd.read_csv(PATIENTS, usecols=["subject_id", "anchor_year_group"])
    ep = ep.merge(pat, on="subject_id", how="left")

    g = ep.assign(rpos=ep.target_root >= 4.0, apos=ep.target_asc >= 4.0).groupby("anchor_year_group")
    tab = g.agg(episodes=("episode_id", "size"), patients=("subject_id", "nunique"),
                root_pos=("rpos", "sum"), asc_pos=("apos", "sum")).reset_index()
    log.info("Per-era distribution:\n%s", tab.to_string(index=False))

    # The echo cohort clusters in 2008-2016; 2017+ buckets are near-empty. Hold out
    # the latest era with enough episodes to be a usable test set.
    MIN_EP = 1000
    viable = tab[tab.episodes >= MIN_EP].sort_values("anchor_year_group")
    holdout_era = viable.anchor_year_group.iloc[-1]
    ep["holdout"] = (ep.anchor_year_group == holdout_era).astype(int)
    log.info("HELD-OUT era = %s | holdout episodes %d / patients %d | root+ %d asc+ %d",
             holdout_era, int(ep.holdout.sum()), ep[ep.holdout == 1].subject_id.nunique(),
             int((ep[ep.holdout == 1].target_root >= 4.0).sum()),
             int((ep[ep.holdout == 1].target_asc >= 4.0).sum()))
    log.info("TRAIN pool (earlier eras): episodes %d / patients %d",
             int((ep.holdout == 0).sum()), ep[ep.holdout == 0].subject_id.nunique())

    out = ep[["episode_id", "subject_id", "anchor_year_group", "holdout"]]
    out.to_csv(os.path.join(PC, "episode_temporal_holdout.csv"), index=False)
    log.info("Wrote episode_temporal_holdout.csv — QUARANTINED, do not evaluate until the final model.")


if __name__ == "__main__":
    main()
