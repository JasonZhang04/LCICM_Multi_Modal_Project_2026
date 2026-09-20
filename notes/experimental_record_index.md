# outputs/ — experimental record

Not in git (`/outputs/` is gitignored). This file is the index; it *is* tracked,
via `notes/`, so the record survives even if the directory is cleared.

## The one rule

Results from different **cohort eras are not comparable**. Check the era before
quoting any number.

| Era | Cohort | Status |
|---|---|---|
| **A — end-to-end** (2026-05 → 06) | ~1.4k patients | Abandoned. Models collapsed to the population mean. |
| **B — 522 triple cohort** (2026-07-02 → 07-22) | 522 patients | **Superseded.** The 2026-07-23 audit found a DICOM gate capping n at 522. Findings may hold directionally; the numbers must not be quoted. |
| **C — episode cohort** (2026-07-23 → present) | 20,682 ep / 16,464 pt | **Current.** These are the comparable numbers. |

## Era C — the live record (quote from here)

| Directory | What it holds |
|---|---|
| `final_model_nested_exact/` | **The headline number.** Nested CV with `exact_nesting`; the leak-free estimate. |
| `final_model_nested_full/`, `_smoke/` | Companion nested runs (full harness; smoke test). |
| `a7_comparison/` | Factorial modality comparison. **Measured through the broken ECG basis — Stage D re-runs it.** |
| `diag_complementarity/` | Modality complementarity; the evidence that CXR/ECG residuals are near-independent. Argues against CLIP-style alignment. |
| `cxr_preproc_comparison/`, `cxr_finetune_episode_ckpt_norm/` | The two CXR preprocessing/normalisation arms. |
| `cxr_finetune_episode/` | CXR partial fine-tuning, the strongest single arm. Its curves show pre-one-epoch overfitting → **Stage A**. |
| `ecg_waveform_episode_broad/` | ECG on the 62,559-episode broad cohort. Embeddings + index → **Stage B**. |
| `ecg_waveform_episode/` | ECG on the matched cohort. The per-fold encoders with the **100% fold-recoverable basis defect**. |
| `*_holdout/`, `holdout_final/` | Temporal holdout (2014–2016). **Quarantined — opened once, at Stage E.** |
| `final_model_episode*`, `fusion_episode/`, `pca_sweep_episode/`, `modality_value_episode/`, `cxr_ablation_episode/`, `ecg_increment_episode/`, `geometry_stack_episode/`, `ehr_baseline_episode/`, `eval_supporting_episode/`, `crossattn_episode/` | Supporting era-C runs: EHR floor, PCA sweep, ablations, cross-attention (did not beat linear early fusion). |

## Eras A and B — kept for provenance only

`multimodal_aorta*`, `multimodal_simclr/`, `multimodal_pclr*`, `echo_*`
(era A) and `combined_stack*`, `reduced_fusion*`, `late_fusion*`,
`gbdt_fusion/`, `deep_fusion/`, `residual_fusion/`, `geometry_stack/`,
`cxr_geometry/`, `cxr_roi_compare/`, `patchpool/`, `crosssite/`,
`ecg_measurements/`, `ecg_expand/`, `reg_derived/`, `stability/`,
`window_experiments/`, `fusion_fix/`, `ehr_baseline/`, `eval_report/`,
`cohort_exploration/` (era B).

Their `results.json`, `train.log` and `training_log.csv` are the reason to keep
them: they record what was tried and what happened. Era-B findings (PCA
dimensionality, anatomy ROIs, multi-view patch pooling, geometry features,
window design) were largely re-derived at episode scale; take the conclusion
from era C.

## Model weights

The era-A/B runs additionally hold ~15 GB of `best_model.pt` / `final_model.pt`
from abandoned architectures, and `scripts/echo_infer_AD/results/checkpoints/`
holds ~8 GB more. **No current script loads any of them** and none are
reproducible inputs to the plan — they are outputs of superseded code.
Deleting the `.pt` files while keeping every log and metric file would reclaim
~23 GB and lose nothing that the record above depends on.
