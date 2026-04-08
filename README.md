# alljoined_vs_things

Comparisons with the AllJoined-1.6M and the THINGS-EEG2 datasets.

## Setup

```bash
conda create -n alljoined python=3.12 -y
conda activate alljoined
pip install neuralset scikit-learn matplotlib torch
```

## Data

The script expects the **Xu2025** study data to be available locally via
`neuralset`. You need to provide two paths:

| Argument | Description |
|---|---|
| `--study_dir` | Root directory containing the Xu2025 study (the folder that has a `Xu2025/` subdirectory with `download/` inside) |
| `--cache_dir` | Writable directory where extracted features (EEG, DINOv2 embeddings) are cached. First run will be slow; subsequent runs reuse the cache. |

## Usage

```bash
# 1 subject, 2000 images
python linear_decoding.py \
    --study_dir /path/to/studies \
    --cache_dir /tmp/alljoined_cache \
    --n_subjects 1 --n_images 2000

# 5 subjects, 2000 images each, custom window
python linear_decoding.py \
    --study_dir /path/to/studies \
    --cache_dir /tmp/alljoined_cache \
    --n_subjects 5 --n_images 2000 \
    --window_start -0.05 --window_duration 0.5

# All subjects, all images
python linear_decoding.py \
    --study_dir /path/to/studies \
    --cache_dir /tmp/alljoined_cache
```

Figures are saved to `--output_dir` (default: `results/`).

## All options

| Flag | Default | Description |
|---|---|---|
| `--study_dir` | *(required)* | Path to the studies root |
| `--cache_dir` | *(required)* | Cache directory for features |
| `--n_subjects` | all | Number of subjects to load |
| `--n_images` | all | Max image trials per subject |
| `--window_start` | -0.05 | Window start relative to stimulus (s) |
| `--window_duration` | 0.5 | Window length (s) |
| `--ridge_alpha` | 1.0 | Ridge regularisation strength |
| `--test_size` | 0.2 | Test set fraction |
| `--seed` | 42 | Random seed |
| `--output_dir` | results/ | Where to save plots |
