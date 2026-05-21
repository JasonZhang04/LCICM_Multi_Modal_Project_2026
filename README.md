# Multi-Modal Aortic Dilation Prediction

Predicts aortic root and ascending aorta diameter from paired echocardiogram labels,
12-lead ECG (MIMIC-IV-ECG), and chest X-ray (MIMIC-CXR-JPG) using a transformer-based
fusion model.

## Project Structure

```
.
├── src/multimodal_aorta/       # Main Python package
│   ├── configs/default_config.py   # All paths and hyperparameters
│   ├── data/                       # Dataset, preprocessing, splits
│   ├── models/                     # Model architecture (in progress)
│   └── training/                   # Training loop (in progress)
├── analysis/                   # Standalone exploration scripts
│   ├── explore_cohort.py           # Multi-modal cohort overlap analysis
│   └── explore_structured_measurements.py  # MIMIC-IV-Echo EAV exploration
├── data/                       # Small curated data files (large datasets not stored here)
│   ├── echo/
│   │   ├── aortic_labels.csv           # Aortic diameter labels (tracked in git)
│   │   └── structured-measurement.csv  # MIMIC-IV-Echo raw measurements (too large for git)
│   ├── echo_report/
│   │   ├── raw/                    # Raw PhysioNet / MIMIC-IV echo reports
│   │   └── processed/              # LLM-extracted aortic dilation labels
│   ├── ecg/                    # See note below
│   └── x_ray/                  # See note below
├── scripts/                    # Past-attempt pipelines (local reference only, not pushed)
│   ├── echo_infer_AD/              # EchoJEPA fine-tuning on JHU echo dataset
│   └── llm_filter_AD_MIMIC_no_use/ # Earlier LLM-based filtering pipeline
├── requirements.txt
└── outputs/                    # Generated figures (gitignored)
```

## Data

Large datasets are **not** stored in this directory. They live in the shared lab scratch
space and are referenced by absolute path in `src/multimodal_aorta/configs/default_config.py`.

| Dataset | Shared path |
|---|---|
| MIMIC-IV-ECG | `/scratch4/rsteven1/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0` |
| MIMIC-CXR-JPG | `/scratch4/rsteven1/MIMIC_CXR_JPG/files` |
| MIMIC-CXR DICOM | `/scratch4/rsteven1/MIMIC_CXR_GS` |

The CXR file metadata cache (`data/cxr_metadata_cache.csv`) is built automatically on
first run and stored in `data/`. It is gitignored.

## Setup

```bash
pip install -r requirements.txt
pip install -e src/   # install multimodal_aorta as an editable package
```

## Running on Rockfish (GPU)

**SLURM batch job:**
```bash
sbatch scripts/echo_infer_AD/slurm/train.sh
```

**Interactive Jupyter session:**
```bash
# 1. On login node — request a GPU
srun --partition=a100 --account=rsteven1_gpu --gres=gpu:1 --cpus-per-task=6 \
     --mem=48G --time=4:00:00 --pty bash

# 2. On the GPU node — start Jupyter
jupyter notebook --no-browser --port=8888 --ip=$(hostname)

# 3. In a new local terminal — forward the port (replace gpu18 with your node)
ssh -J czhan182@login.rockfish.jhu.edu -L 8888:localhost:8888 czhan182@gpu18

# 4. Open the URL printed by Jupyter in your browser
#    NOTE: make sure port 8888 is not occupied from a previous session
```
