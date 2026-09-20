# Design Consolidation — answers, verified facts, and a proposed plan
**2026-09-20** · supersedes ad-hoc decisions; written to be read start-to-finish

Everything numeric below was re-derived from the code and artifacts today, not
recalled. Commands are given so you can re-run any of it.

---

## Part A — Your questions, answered

### A1. "A classifier can tell which network produced a row with 100% accuracy" — what does that mean?

You have the setup right: 5 folds, 5 independently trained ECG networks, and every
episode gets a 256-number embedding from whichever network didn't train on it. My
sentence was compressed to the point of being cryptic. Unpacking it:

**"A row"** = one episode's 256-number ECG embedding. One row of the feature matrix.

**"Which network produced it"** = the fold index 0–4. Each row was produced by exactly
one of the 5 networks, so each row carries a hidden label: *which network made me*.

**The probe**: hide that label, hand a simple classifier only the 256 numbers, and ask
it to guess. If the 5 networks had produced genuinely comparable numbers, guessing
would be impossible — 20% accuracy, pure chance.

I re-ran it today:

```
FOLD RECOVERY ACC = 1.0000 on 5508 held-out rows (chance 0.20)
```

`src/multimodal_aorta/tests/test_embedding_basis.py` — nearest-centroid, patients
held out, so it is not memorisation.

**Why 100% is damning.** Each network's embedding carries a network-specific
fingerprint so strong it swamps everything else. Coordinate #7 means
"lateral-lead QRS energy" in network 0 and something entirely unrelated in network 1 —
neural networks have no reason to order their hidden units the same way twice.

The consequence for the stack: PCA runs over all 18,221 rows at once and finds, as its
loudest directions, *which network made this row* — not *how dilated is this aorta*.
Those top components go to the ridge, which fits coefficients on four networks'
fingerprints and then meets a fifth at test time. The ECG's real signal was there; it
was buried under an artifact of how we assembled the matrix.

**This is why I resist "multimodality adds nothing."** The A7 null was measured through
this broken channel. It is evidence the *plumbing* was broken, not that the ECG is
uninformative — and the corroboration is that when ECG enters as a single scalar
prediction (no basis problem: a predicted diameter means the same thing regardless of
which network produced it), it *does* add: +0.006 AUROC at both sites, CIs excluding zero.

**Two clean fixes, both eliminate the defect by construction:**
1. **One frozen encoder for all folds** — a single pretrained ECG model, never trained
   on our labels, so all rows come from the same network. One basis, trivially.
2. **Scalar-only** — pass the predicted diameter, never the embedding.

Option 1 is strictly better if a good pretrained encoder exists: it keeps the 256
dimensions of richness instead of collapsing to 1 number.

---

### A2. Anchor years, anchor groups, and why we can't just sort by echo date

This is genuinely confusing and the confusion is MIMIC's fault, not yours.

**What MIMIC does.** To de-identify, MIMIC shifts every patient's dates into the
future by a random offset — *a different offset per patient*. It then tells you two
things:

- `anchor_year`: one integer per patient, in the **shifted** timeline.
- `anchor_year_group`: a 3-year window — `"2011 - 2013"`, `"2014 - 2016"` etc. — saying
  where that patient's `anchor_year` sat in **real** time.

So for a patient with `anchor_year = 2150` and group `"2014 - 2016"`: their shifted
year 2150 was really somewhere in 2014–2016. The 3-year window is the deliberate
blurring — MIMIC will not tell you the exact real year, only a 3-year bracket. That is
what "MIMIC's 3-year interval" means: **the irreducible uncertainty in any real date**.

**Your intuition about offsets — this is the important part.** You said you thought the
shift preserves differences between patients. It preserves differences *within* a
patient but **not** between patients. Patient A shifted +130 years and patient B
shifted +7 years both have shifted dates, but the gap between A's echo and B's echo in
the shifted timeline is meaningless. **Sorting all episodes by `echo_dt` sorts by a
random number.** That is exactly why we cannot just use echo dates — and it is the
single fact that makes this whole area subtle.

**What we can recover.** Within a patient, offsets are constant, so the *difference*
between an echo date and the anchor is real:

```python
off    = echo_year - anchor_year          # real elapsed years, exact
imp_lo = group_start + off                # earliest plausible real year
imp_hi = group_end   + off                # latest plausible real year
```
(`scripts/build_temporal_holdout_v2.py:73-77`)

Each episode gets a real-time **interval** 3 years wide. Not a date — an interval.

**The bug this fixed (A4).** The old split used `anchor_year_group` alone — the
*patient's* era, ignoring that the echo may be years after the anchor. A patient
anchored 2014–2016 with an echo 5 years later is really a 2019–2021 event. Result:
**5,196 episodes whose earliest plausible year was ≥2014 sat in "training"** while
the 2014–2016 group was being called the later era. The split was partly inverted.

**"Anchor group for one anchor year"** was my shorthand for: one `anchor_year` value
maps to a 3-year *group*, so a single patient-year is only ever known to 3-year
resolution. Poor phrasing on my part; the mechanism is above.

**Why a guard band, and why we chose none.** Because intervals are 3 years wide, two
episodes can have overlapping intervals and an unknowable true order. The code keeps
a patient only if *unambiguously* late (`imp_lo >= C`) or *unambiguously* early
(`imp_hi <= C - guard`); straddlers are dropped. A guard band would widen that gap and
buy certainty at the cost of data. You chose `C=2016, guard=0` — intervals may touch at
the boundary but never invert, which the script asserts. That remains a defensible
choice; it just needs stating in the paper as a stated limitation.

---

### A3. How we use the ECG vs. how ECGAI-TAA uses it

Their recipe, and ours, line by line (`scripts/train_ecg_waveform_episode.py`,
`extract_ecg_waveforms.py`, `src/multimodal_aorta/models/ecg_resnet.py`):

| | ECGAI-TAA | Ours | Match |
|---|---|---|---|
| Input | 10 s, 12-lead, 500 Hz | 10 s (5000 samples), 12-lead, 500 Hz | ✅ |
| Preprocessing | none, raw mV | none, raw mV | ✅ |
| Stem | kernel 15 | kernel 15 | ✅ |
| Body | 3 residual stages, group norm, downsampling | 3 stages (64,128,256), GroupNorm, stride-2 | ✅ |
| Dilation | alternating blocks | `dil = 2 if bi % 2 else 1` | ✅ |
| Attention | residual multi-head | `AttnBlock(256, heads=4)` | ✅ |
| Head | GAP → dropout → linear | GAP → dropout 0.3 → `Linear(256, 3)` | ✅ |
| Targets | asc diameter + HR | root + asc + HR | ✅ (we add root) |
| Loss | MSE on normalised; val on diameter only | masked MSE; val on diameters only | ✅ |
| Augmentation | noise per lead | `x + randn * 0.01` | ✅ |
| Batch / LR | 64; warmup 2000 → flat 1e-3 | 64; warmup 2000 → flat 1e-3 | ✅ |
| Steps | 30,000 minibatches | **12,000** | ❌ **40% of theirs** |

**We already implement their architecture faithfully.** One real gap: we train 12,000
steps to their 30,000. Cheap to test.

**On the 2-second segment** — that's the *other* multimodal paper, and you're right to
be sceptical. Aortic dilation is a static anatomical property: every beat carries the
same evidence, so random 2 s crops are just augmentation. At 10 s we get ~10–12 beats
and the attention layer can pool across them. Random cropping is worth trying *as
augmentation* (train on 5 s crops, test on full 10 s), but not as a replacement.

**The bigger miss is not the architecture — it's the cohort.** We train on 18,221
episodes because we restricted to episodes that also have a CXR. The ECG-only cohort
is already built and 3.4× larger:

```
matched cohort: 18,221 episodes | 14,831 patients | 815 root ≥4 cm
broad cohort:   62,559 episodes | 46,230 patients | 3,058 root ≥4 cm
```

**3.75× the positives.** Deep networks are positive-count-limited here. Training the
ECG encoder on the broad cohort and applying it to the matched cohort is the highest
value-per-GPU-hour move available, and it **simultaneously fixes A2**: train once on
the broad cohort with CXR-cohort patients excluded, freeze it, apply to all 5 folds →
one basis, no leakage, no per-fold networks. Two problems, one job.

---

### A4. Augmentation — what we actually do

**ECG**: Gaussian noise `σ=0.01` on the raw mV signal (`train_ecg_waveform_episode.py:78`).
Close to the paper's "±0.1 mV random noise". Your suggested baseline-zeroing is not
done. Worth adding; the SimCLR pipeline in
`src/multimodal_aorta/data/ecg_pretrain_dataset.py` already implements baseline wander,
amplitude scaling, lead masking and time cropping — written, tested, **never run**.

**CXR — this is a real finding.** We do essentially *nothing*:

```python
x = x + torch.randn_like(x) * 0.02    # train_cxr_finetune_episode.py:113
```

Gaussian pixel noise. No rotation, no contrast, no scaling. And the reason is
structural, recorded in `extract_cxr_image_cache.py:14-16`: we cache **pre-resized
224×224 tensors** for speed (JPEG decode was the 4 img/s bottleneck). The cache stores
the *validation* transform. Geometric augmentation needs the original image, so caching
it away foreclosed augmentation. The comment says partial fine-tuning is "conservative
enough that dropping augmentation is acceptable for v1" — **§1.8's overfitting curves
show that assumption was wrong.** Best val at step 600–1,200, then 10–15% degradation.

Your proposal (±10° rotation, contrast at 25% probability, no h-flip) is exactly right,
and your h-flip reasoning is correct and worth stating in the paper: the aortic knob is
a *left*-sided structure; mirroring creates anatomically impossible images and
destroys the laterality the model needs. (`cxr_aug_hflip_p` is already 0.0 with a test
pinning it there — `test_preprocessing_and_timing.py:71`.)

Rotation and contrast can be applied **on the cached tensor** on GPU — no re-decode, no
lost throughput. That makes this cheap, and it is the best-evidenced fix we have.

**Patient-level split — yes, done throughout, and verified today:**
```
episodes 20682 | patients 16464 | patients appearing in >1 fold: 0
```
`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, grouped on
`subject_id`, stratified on episode-level any-AD, frozen to a file so every trainer
shares one definition. Our 16,464 patients exceed most comparable published cohorts,
so patient-level splitting is affordable and correct.

---

### A5. 1-D ResNet + Transformer across cardiac cycles

Worth understanding what we already have. Our `AttnBlock(256, heads=4)` sits after the
conv stages, on a sequence downsampled 8× from 5000 → ~625 positions. It is
multi-head self-attention over time, so **we already attend across cardiac cycles.**

What their design adds is a **class token** — a learned slot that aggregates the
sequence, instead of global average pooling. A CLS token can learn *selective* pooling
("weight the T-wave region more") where GAP weights everything equally.

Honest assessment: a **modest, cheap** experiment. It is a small architectural delta —
add a CLS token, replace GAP with the CLS output. I'd expect little on its own, because
our attention block already provides cross-cycle mixing, and the binding constraint is
815 positives, not architectural capacity. **Do it as an arm of the ECG sweep, not as a
standalone effort, and only after the cohort expansion.** Ordering matters: 3.75× the
positives will move things more than a pooling change.

---

### A6. Self-supervised contrastive learning — the honest evaluation

You asked me to be objective, including about whether this is worth it. There are
three distinct ideas here and they differ sharply in value.

**Idea 1 — single-modality SSL pretraining (ECG SimCLR on 800k MIMIC ECGs).**
*Verdict: valuable, and it's mostly built.* The dataset class exists and is tested;
only the training loop was never run. Pretraining on 800k unlabelled ECGs and
fine-tuning on our 3,058 positives is the standard answer to positive-count scarcity.
**And it fixes A2 for free** — one encoder, one basis. This is the well-supported idea.

**Idea 2 — cross-modal CXR↔ECG contrastive (CLIP-style).** This is the one you were
most drawn to, so let me be careful. *Verdict: scientifically interesting, but the
evidence in our own results argues against it as a performance lever.*

Feasibility is fine — 18,221 episodes have both modalities, comparable to many CLIP-style
medical papers. The problem is what it optimises *for*. Cross-modal contrastive learning
pulls together representations of the same patient across modalities — it learns the
**shared** subspace. But our complementarity diagnostics say the shared part is already
captured: residual correlation is low, the CXR residual is uncorrelated with age, sex,
body size, view and lag, and the ECG's contribution is small and *additive*. **The
value in the ECG is precisely the part the CXR does not see — the non-shared part.**
Cross-modal alignment maximises exactly the component we already have and discards the
one we need. It could plausibly make things *worse*.

**Idea 3 — supervised contrastive on dilation severity.** Pull together patients with
similar diameters, push apart dissimilar, across modalities. This keeps the
label-relevant structure that Idea 2 throws away and is a more defensible framing:
"aligned on the clinical phenotype, not on patient identity."

**On the publication-venue motivation** — worth being direct. Chasing novelty for venue
fit is how good clinical papers get weakened. But the good news: **you already have the
computational novelty, and it's better than a bolted-on contrastive module.** A
rigorous multimodal incremental-value study, with a nested-CV harness that provably
prevents the leak, an executable probe that *detects a broken embedding basis*, a
label-noise ceiling that bounds what any model could achieve, and a quarantined
temporal cohort opened once — that is a methods contribution. The `exact_nesting` flag
and the fold-recovery probe are genuinely novel evaluation instruments; most papers in
this space would not have caught either defect. **Frame the rigour as the contribution.**
If SSL pretraining (Idea 1) also improves results, it strengthens the story — but it
should earn its place on results, not be inserted for venue fit.

---

## Part B — Proposed plan

Ordered by evidence strength, not by interest. Every stage writes a JSON result.

### Stage A — CXR augmentation *(highest confidence)*
The §1.8 curves show the strongest arm overfits before one epoch with no augmentation.
Add GPU-side rotation (±10°), contrast (p=0.25), slight scaling; no h-flip. Cosine
schedule, discriminative LR. Re-run 5-fold.
*Why first:* direct evidence of the failure; cheap; touches the strongest arm.

### Stage B — ECG on the broad cohort with a frozen shared encoder *(highest ceiling)*
Train one ECGResNet on the broad cohort (62,559 episodes / 3,058 positives) with
CXR-cohort patients excluded; freeze; apply to all folds. **Fixes A2 by construction
and gives 3.75× positives.** Then a small sweep: 30k steps (their recipe), CLS token,
crop augmentation.
*Why second:* biggest expected gain, and it retires a known defect.

### Stage C — SSL pretraining, if B leaves headroom
Run the built-but-never-run SimCLR loop on 800k ECGs; fine-tune. Only if Stage B shows
the encoder is still data-limited.

### Stage D — re-run the factorial nested comparison on `develop`
With a fixed CXR arm and a valid ECG basis, the A7 comparison becomes meaningful for
the first time. **This is the number that goes in the paper.**

### Stage E — quarantine, once
Freeze everything, then open the quarantined cohort. Once.

### Deferred
Supervised-contrastive cross-modal (Idea 3) as an *exploratory* arm after D, reported
whatever the result. Cross-modal CLIP-style (Idea 2) not planned, for the reason above.

### Not on the list, but needed
An erratum for `PROGRESS_UPDATE_2026-08-30.md` (the A6 reporting bug), and a standing
rule that every GPU run writes `training_curves.csv` — now automatic via `CurveLogger`.

---

## Part C — Facts verified today

| Claim | Value | Source |
|---|---|---|
| ECG fold-ID recoverability | **100.00%** on 5,508 patient-disjoint rows (chance 20%) | probe re-run today |
| Patients in >1 fold | **0** of 16,464 | `episode_fold_assignments.csv` |
| Matched ECG cohort | 18,221 ep / 14,831 pt / 815 root ≥4 | `ecg_waveform_cohort.csv` |
| Broad ECG cohort | 62,559 ep / 46,230 pt / 3,058 root ≥4 | `ecg_waveform_cohort_broad.csv` |
| Episodes with both modalities | 18,221 | join, computed today |
| CXR augmentation | Gaussian noise σ=0.02 only | `train_cxr_finetune_episode.py:113` |
| ECG augmentation | Gaussian noise σ=0.01 | `train_ecg_waveform_episode.py:78` |
| ECG training steps | 12,000 vs paper's 30,000 | `train_ecg_waveform_episode.py:33` |
| h-flip probability | 0.0, pinned by a test | `default_config.py:134` |
| Images per episode | mean 3.12, median 2, max 90 | `episode_cxr_instances.csv` |
| SimCLR ECG pretraining | implemented, tested, **never run** | no output dir |
