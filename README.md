# Multi-Modal Aortic Dilation Prediction

Estimates aortic root and ascending aorta diameter from **12-lead ECG** (MIMIC-IV-ECG)
and **chest X-ray** (MIMIC-CXR-JPG), with structured EHR as the baseline floor, against
echocardiogram-derived labels (MIMIC-IV-Echo). The clinical framing is screening: can
cheap, ubiquitous tests flag dilation that today needs an echo to find?

**Current design of record: [`notes/DESIGN_CONSOLIDATION_2026-09-20.md`](notes/DESIGN_CONSOLIDATION_2026-09-20.md).**
Read it before planning work. It supersedes earlier plans.

## Where things stand

The architecture is **not** end-to-end fusion. That was tried (v1/v2, 2026-05) and
collapsed to the population mean. The current system is frozen/partially-fine-tuned
per-modality encoders feeding a **stacked linear model**, evaluated with nested
cross-validation.

- **Unit of analysis:** the *episode* (one echo + its temporally-matched ECG/CXR),
  not the patient. 20,682 episodes / 16,464 patients.
- **Splitting:** `StratifiedGroupKFold(5)` grouped on `subject_id`. Zero patients
  appear in more than one fold.
- **Headline estimate:** `outputs/final_model_nested_exact/` (leak-free nested CV).
- **Temporal holdout (2014–2016) is quarantined** and opens exactly once, at Stage E.

Two known defects are being repaired, both documented in the design note: the ECG
embedding basis is 100% fold-recoverable (per-fold encoders → the stack fits network
fingerprints), and CXR fine-tuning runs with essentially no augmentation and overfits
before one epoch.

## Project structure

```
.
├── src/multimodal_aorta/       # Main package
│   ├── configs/default_config.py   # All paths and hyperparameters
│   ├── data/                       # Episode construction, preprocessing, splits
│   ├── models/                     # ECG ResNet, fusion heads
│   ├── training/                   # Nested CV harness, curve logging
│   └── tests/                      # Leakage + basis + preprocessing regression tests
├── scripts/                    # Cohort building, extraction, training, eval (+ SLURM)
├── analysis/                   # Standalone exploration and audit scripts
├── notes/                      # Design docs, reviews, roadmaps
│   ├── DESIGN_CONSOLIDATION_2026-09-20.md   # ← current design of record
│   ├── experimental_record_index.md         # what each outputs/ run means
│   ├── ECG/                                 # pretrained ECG encoder survey
│   └── archive/                             # superseded notes (provenance only)
├── figures/                    # Paper figures + training curves
├── data/                       # Small curated files (large data lives on scratch)
├── logs/                       # Run logs, by date
├── outputs/                    # Run results (gitignored; see notes/experimental_record_index.md)
└── pretrained_checkpoints/     # Embeddings, caches, cohort tables (gitignored)
```

`scripts/echo_infer_AD/` and `scripts/llm_filter_AD_MIMIC_no_use/` are **past attempts**,
kept locally and gitignored. Nothing current imports them.

## Data

Large datasets are **not** stored here. They live in shared lab scratch and are
referenced by absolute path in `src/multimodal_aorta/configs/default_config.py`.

| Dataset | Shared path |
|---|---|
| MIMIC-IV-ECG | `/scratch4/rsteven1/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0` |
| MIMIC-CXR-JPG | `/scratch4/rsteven1/MIMIC_CXR_JPG/files` |
| MIMIC-CXR DICOM | `/scratch4/rsteven1/MIMIC_CXR_GS` |

`data/cxr_metadata_cache.csv` is built on first run and gitignored.

## Setup

```bash
pip install -r requirements.txt
pip install -e src/   # install multimodal_aorta as an editable package
```

## Running on Rockfish (GPU)

Trainers and extractors each have a SLURM wrapper in `scripts/`:

```bash
sbatch scripts/slurm_train.sh                  # main trainer
sbatch scripts/slurm_final_model_nested.sh     # nested-CV final model
sbatch scripts/slurm_extract_cxr_image_cache.sh
```

Every GPU run writes `training_curves.csv` via `CurveLogger` — keep it that way; the
2026-09 review needed those curves to diagnose the overfitting.

**Interactive Jupyter session:**
```bash
# 1. On login node — request a GPU
srun --partition=a100 --account=rsteven1_gpu --gres=gpu:1 --cpus-per-task=6 \
     --mem=48G --time=4:00:00 --pty bash

# 2. On the GPU node — start Jupyter
jupyter notebook --no-browser --port=8888 --ip=$(hostname)

# 3. In a new local terminal — forward the port (replace gpu18 with your node)
ssh -J czhan182@login.rockfish.jhu.edu -L 8888:localhost:8888 czhan182@gpu18
```

## Tests

```bash
python -m pytest src/multimodal_aorta/tests/ -q
```

These are **defect regression tests**, not unit tests: they pin the no-leak nesting,
the embedding-basis probe, and the preprocessing/timing invariants. They load real
cohort artifacts, so they take minutes, not seconds.
