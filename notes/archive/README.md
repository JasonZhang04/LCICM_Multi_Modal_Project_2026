# Archived notes

These documents are **superseded**. They are kept for provenance — to trace how a
decision was reached — not as guidance. Do not use them to plan work.

The current design of record is
[`../DESIGN_CONSOLIDATION_2026-09-20.md`](../DESIGN_CONSOLIDATION_2026-09-20.md).

## Why each was archived

| Document | Era | Superseded because |
|---|---|---|
| `v2_update.md`, `v2_update_opus.md` | 2026-05 | v1/v2 end-to-end fusion collapsed to the population mean. That architecture was abandoned; the project moved to frozen encoders + stacking. |
| `multimodal_v3_plan.md` | 2026-06 | The v3 plan as drafted; its decisions were implemented and have since been revised by the 2026-09 review. |
| `aortic_dilation_multimodal_implementation_guidance.md` | 2026-07 | Written for the **522-patient triple cohort**, replaced by the ~15k-patient episode cohort (2026-07-23 audit). Cohort-size-dependent advice no longer applies. |
| `system_overview_and_analysis.md` | 2026-07-16 | Pipeline walk-through of the 522-cohort system, at commit `39b8c75`. Both the cohort and the code have moved on. |
| `experiments_2026-07-16.md`, `experiments_2026-07-22.md`, `experiments_2026-07-22b_windows.md` | 2026-07 | Results measured on the 522 triple cohort. The numbers are not comparable to current episode-level results and must not be quoted in the paper.|

## One caveat worth knowing

The 2026-07 experiment notes record findings (PCA dimensionality, anatomy ROIs,
multi-view patch pooling, geometry features, window design) that were *real* at
522 patients. Several were re-derived at episode scale and survive in current
results. If you want the conclusion, take it from the current documents; these
files only tell you what was believed at the time.

Note that `ECG/ecg_encoder_aortic_multimodal_implementation_guide.md` was
deliberately **not** archived. Its cohort framing is stale, but its survey of
public pretrained ECG encoders (ECGFounder, ECG-FM, xECG, ECG-CPC) is the
reference for Stage B/C of the current plan, which needs exactly one frozen
pretrained encoder.
