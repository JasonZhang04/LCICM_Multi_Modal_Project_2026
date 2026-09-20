"""
Regression tests for review issues A5 (information timing) and A9 (encoder inputs).

These lock in invariants that were silently violated before and would be easy to
reintroduce: a CXR must precede its echo by the recorded TIME (not merely the date),
and a grayscale X-ray's three identical channels must stay identical after
normalization.
"""
import os
import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PC = os.path.join(ROOT, "pretrained_checkpoints")
CXR_META = "/scratch4/rsteven1/MIMIC_CXR_JPG_cohort/mimic-cxr-2.0.0-metadata.csv.gz"


# ------------------------------------------------------------------ A9
def _cfg(mode):
    import sys
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from multimodal_aorta.configs.default_config import Config
    c = Config(); c.data.cxr_preproc = mode
    return c


def _a_real_cxr():
    p = os.path.join(PC, "cxr_instances_episode.csv")
    if not os.path.exists(p):
        pytest.skip("cxr_instances_episode.csv not present")
    inst = pd.read_csv(p)
    inst = inst[inst.on_disk.fillna(False).astype(bool)]
    if not len(inst):
        pytest.skip("no on-disk CXR available")
    return inst.cxr_path.iloc[0]


@pytest.mark.parametrize("mode", ["ckpt_norm", "ckpt_full"])
def test_checkpoint_modes_keep_grayscale_channels_identical(mode):
    """A CXR's 3 channels are copies of one grayscale plane. RAD-DINO's checkpoint uses
    equal per-channel statistics, so they must stay identical after normalization."""
    import sys; sys.path.insert(0, os.path.join(ROOT, "src"))
    from multimodal_aorta.data.preprocessing import load_cxr
    x = load_cxr(_a_real_cxr(), _cfg(mode).data, is_train=False)
    assert np.allclose(x[0].numpy(), x[1].numpy(), atol=1e-6)
    assert np.allclose(x[1].numpy(), x[2].numpy(), atol=1e-6)


def test_legacy_mode_is_the_known_defect():
    """Documents WHY the modes exist: ImageNet per-channel stats split identical
    channels apart. If this ever stops failing, legacy has silently changed."""
    import sys; sys.path.insert(0, os.path.join(ROOT, "src"))
    from multimodal_aorta.data.preprocessing import load_cxr
    x = load_cxr(_a_real_cxr(), _cfg("legacy").data, is_train=False)
    spread = float(max(x[i].mean() for i in range(3)) - min(x[i].mean() for i in range(3)))
    assert spread > 0.1, (
        "legacy no longer shows the ImageNet channel split; if that was intentional, "
        "retire this test and the legacy mode together")


def test_unknown_preproc_mode_is_rejected():
    import sys; sys.path.insert(0, os.path.join(ROOT, "src"))
    from multimodal_aorta.data.preprocessing import load_cxr
    with pytest.raises(ValueError):
        load_cxr(_a_real_cxr(), _cfg("not_a_mode").data, is_train=False)


def test_horizontal_flip_is_off_by_default():
    """Mirroring a CXR puts the heart and aortic arch on the wrong side."""
    assert _cfg("legacy").data.cxr_aug_hflip_p == 0.0


# ------------------------------------------------------------------ A5
@pytest.mark.parametrize("cohort_dir", ["cohort_v2_timefix"])
def test_every_cxr_precedes_its_echo_timestamp(cohort_dir):
    """The rebuilt cohort must contain zero post-echo CXRs (was 1,931 rows)."""
    d = os.path.join(PC, cohort_dir)
    if not (os.path.exists(os.path.join(d, "episodes.csv")) and os.path.exists(CXR_META)):
        pytest.skip(f"{cohort_dir} or CXR metadata not present")
    cx = pd.read_csv(CXR_META, usecols=["dicom_id", "StudyDate", "StudyTime"])
    cx["dicom_id"] = cx.dicom_id.astype(str)
    cx["ts"] = pd.to_datetime(
        cx.StudyDate.astype("Int64").astype(str) + " " +
        cx.StudyTime.astype(float).map(lambda t: "%06d" % int(t)),
        format="%Y%m%d %H%M%S", errors="coerce")
    inst = pd.read_csv(os.path.join(d, "episode_cxr_instances.csv"))
    inst["dicom_id"] = inst.dicom_id.astype(str)
    ep = pd.read_csv(os.path.join(d, "episodes.csv"))
    ep["echo_ts"] = pd.to_datetime(ep.echo_dt)
    m = (inst.merge(cx[["dicom_id", "ts"]], on="dicom_id", how="left")
             .merge(ep[["subject_id", "measurement_id", "echo_ts"]],
                    on=["subject_id", "measurement_id"], how="left"))
    bad = int((m.ts >= m.echo_ts).sum())
    assert bad == 0, f"{bad} CXR rows are not strictly before their echo timestamp"
    assert m.days_before_echo.min() > 0, "a lag of exactly 0 means same-instant or after"
