# Window-design experiments — 2026-07-22 (temporal framing & robustness)

> 8 cohort designs (2 directions x 4 CXR windows), ECG window fixed at 180 d, both modalities
> required. Each runs the validated model (multi-view patch-pool + geometry + cross-site +
> ECG measurements, regression-derived scoring) over 5 fold seeds.
> Feature extraction was done ONCE over a +/-450 d superset (2,698 frontal CXRs / 684 patients,
> 99.9% segmentation success); every design is a filter on it.

## How the window is anchored (verified in code)

The **echo is the sole anchor**. ECG and CXR are each matched to the echo *independently*
(`|t - echo| <= W`), not to each other. In practice ECG<->CXR separation is **median 1 day**
(90th pct 92 d; only 1% >180 d apart), so the unconstrained pairing is a non-issue.
Under the symmetric window ~half the data is post-echo (ECG 49% pre, CXR 44% pre).

`pre` = data must precede the echo (delta <= 0, same-day allowed); `sym` = |delta| <= W.

## Results

| design | n | root+ | asc+ | inst/pt | ROOT ge40 own | ROOT ge40 **common** | ROOT R² | ASC ge40 own | ASC ge40 **common** | ASC R² |
|---|---|---|---|---|---|---|---|---|---|---|
| pre/180 | 381 | 19 | 36 | 1.9 | 0.821±0.019 | 0.821 | 0.327 | 0.776±0.011 | 0.776 | 0.238 |
| pre/270 | 403 | 20 | 37 | 2.0 | 0.818±0.013 | 0.827 | 0.334 | 0.777±0.020 | 0.775 | 0.240 |
| **pre/365** | **427** | 21 | 38 | 2.0 | **0.828±0.011** | **0.830** | 0.341 | **0.789±0.020** | 0.786 | 0.262 |
| pre/450 | 439 | 22 | 39 | 2.1 | 0.829±0.008 | 0.832 | 0.350 | 0.788±0.018 | 0.788 | 0.259 |
| sym/180 | 509 | 32 | 45 | 3.5 | 0.807±0.012 | 0.822 | 0.357 | 0.803±0.007 | 0.772 | 0.230 |
| sym/270 | 552 | 35 | 47 | 3.7 | 0.820±0.007 | 0.821 | 0.356 | 0.797±0.014 | 0.775 | 0.232 |
| sym/365 | 602 | 40 | 53 | 3.9 | 0.828±0.005 | 0.819 | 0.375 | 0.810±0.017 | 0.795 | 0.263 |
| sym/450 | 630 | 40 | 60 | 4.1 | 0.832±0.008 | 0.823 | 0.386 | 0.800±0.012 | 0.780 | 0.244 |

**common** = evaluated only on the 381 patients present in the strict pre/180 design, so all
designs are scored on the SAME patients. This separates "more training data helped" from
"the looser window admitted an easier case mix".

## Three findings

**1. The model is robust across every cohort definition.** Over 8 designs spanning 381–630
patients, root ge40 stays in **0.807–0.832** and asc in **0.776–0.810**. Nothing collapses.
This is the strongest robustness evidence the project has: the design is not tuned to one cohort.

**2. Symmetric windows' apparent gains are mostly CASE MIX, not a better model.** Root, symmetric:
own-cohort rises 0.807 → 0.832 as the window loosens (+0.025), but on the **common** patients it is
flat: 0.822 → 0.823 (+0.001). Loosening a symmetric window mainly changes *who is in the cohort*.
By contrast, pre-only rises on the common set too (0.821 → 0.832), i.e. the extra prior imaging
genuinely improved the model. Without the common-subset evaluation we would have mis-read this.

**3. Pre-only costs almost nothing.** pre/365 gives root **0.828** vs sym/365 0.828 (identical) and
asc 0.789 vs 0.810 (−0.02). So the medically legitimate, screening-valid design is **not** a
performance sacrifice — the trade-off we feared does not exist for the root, and is small for the
ascending aorta.

## Recommended design

**Primary: pre-only, ECG ±180 d, CXR ≤365 d before the echo** (n=427; root ge40 0.828±0.011,
asc 0.789±0.020, root R² 0.341).
- Every input strictly precedes the echo -> supports a genuine *early screening* claim.
- 365 d for CXR is clinically defensible: dilated aortas grow ~0.1 cm/yr, so worst-case label
  drift is ~0.1 cm against a 4.0 cm threshold. 450 d adds only 12 patients for more drift.
- Tighter ECG (180 d) is nearly free because ECGs are plentiful; the CXR is the binding modality.

**Sensitivity analyses to report alongside:** sym/365 (the permissive framing, asc 0.810) and
pre/180 (the strictest, root 0.821). Also report performance vs |delta t|.

## Caveats

- Differences between adjacent designs are mostly within 1 SD; the trustworthy claims are the
  three findings above, not the ranking of individual cells.
- Pre-only cohorts have ~2 CXRs/patient vs ~3.5 symmetric, so multi-instance averaging is weaker.
- Root positives are 19–22 in the pre-only arms -> wide CIs; diameter R² is the more stable metric
  there (and, thanks to regression-derived scoring, it is what drives the AUROC anyway).
