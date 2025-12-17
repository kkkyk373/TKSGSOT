## Overview

Selective transfer learning for commuting OD flow prediction with three model families:

- `run_selective_dgm.py` — Deep Gravity Model (PyTorch)
- `run_selective_rf.py`  — Random Forest (scikit-learn)
- `run_selective_svr.py` — Support Vector Regressor (scikit-learn)

Path handling is anonymized and defaults to repository-relative locations. Local and Slurm array runners are provided.

## Directory Layout (key)

- `ComOD-dataset/` — data root  
  - `data/` — per-area features + OD pairs  
  - `fgw_dist_matrice/` — FGW area ids (`fgw_area_ids.npy`) and distance matrices (`fgw_dist_<alpha>.dat`)
- `comod_source_target_lists/` — `targets_seed*.txt`, `sources_seed*.txt`
- `jobs/` — array runners  
  - `bash_dgm_array.sh`, `bash_svr_array.sh`, `bash_rf_array.sh` (local, no Slurm)  
  - `slurm_dgm_array.sh`, `slurm_classic_array.sh` (Slurm)
- `results/` — experiment outputs (`<model>/raw/...`)
- `outputs/` — saved models
- `logs/` — job stdout/stderr
- `analysis/` — aggregation and plotting utilities

## Environment Setup (uv, Python 3.10.12)

pip は使わず、uv の lock に従ってセットアップしてください。

```bash
# 1) Create venv (default name used by job scripts)
uv venv .venv --python 3.10.12

# 2) Install dependencies from pyproject/uv.lock
UV_PROJECT_ENVIRONMENT=.venv uv sync --frozen

# 3) Activate when running manually
source .venv/bin/activate
```

Dependencies are pinned in `pyproject.toml` / `uv.lock`. WandB is not used.

## Data Layout

Place the ComOD dataset under `ComOD-dataset/` (or override with env vars below):

```
ComOD-dataset/
├── data/<area_id>/{demos.npy, pois.npy, dis.npy, od.npy}
└── fgw_dist_matrice/
    ├── fgw_area_ids.npy
    └── fgw_dist_<alpha>.dat   # memory-mapped FGW distances
```

Optional environment variable overrides (also honored by job scripts):

```
DATA_DIR          default: ComOD-dataset/data
FGW_DIR           default: ComOD-dataset/fgw_dist_matrice
TARGETS_BASE      default: comod_source_target_lists
SOURCES_BASE      default: comod_source_target_lists
RESULTS_DIR       default: results
MODEL_OUTPUT_DIR  default: outputs
VENV_ACTIVATE     default: .venv/bin/activate
PYTHON_BIN        default: python
```

## Running Locally (direct Python)

Example (SVR, top-k):

```bash
PYTHONPATH=$(pwd) .venv/bin/python src/experiments/run_selective_svr.py \
  --data_dir "${DATA_DIR:-ComOD-dataset/data}" \
  --fgw_dir "${FGW_DIR:-ComOD-dataset/fgw_dist_matrice}" \
  --targets_path "${TARGETS_BASE:-comod_source_target_lists}/targets_seed0.txt" \
  --sources_path "${SOURCES_BASE:-comod_source_target_lists}/sources_seed0.txt" \
  --results_dir "${RESULTS_DIR:-results}" \
  --model_output_dir "${MODEL_OUTPUT_DIR:-outputs}" \
  --condition topk \
  --top_k 100 \
  --alpha 50 \
  --max_samples 50000 \
  --seed 0
```

Flags are shared across models; DGM also uses `--epochs/--batch_size/--lr`. `condition` ∈ {`topk`, `bottomk`, `random`, `all`}; for `all`/`random`, `alpha` is ignored.

## Local Array Runners (no Slurm)

All three sweep scripts share the same grid: seeds 0–9; `alpha` in {0,50,100} for `topk`/`bottomk`; `alpha=0` for `all`/`random`; total 80 runs.

- DGM: `bash jobs/bash_dgm_array.sh`
- SVR: `bash jobs/bash_svr_array.sh`
- RF : `bash jobs/bash_rf_array.sh`

Behavior:
- No args ⇒ run all 80.
- With indexes ⇒ run only those (e.g., `bash jobs/bash_dgm_array.sh 0 3 7`). Index ordering matches `PARAMS` in the scripts (topk→bottomk for each alpha/seed, then all/random).
- Logs: `logs/<model>_unified/<condition>/alpha<alpha>/localXX_seed<seed>_<timestamp>.out|.err`.
- Uses `.venv` by default; override via `VENV_ACTIVATE=...`.

## Slurm Array Jobs

- DGM: `jobs/slurm_dgm_array.sh` (`#SBATCH --array=0-79`)
- Classic SVR all-only: `jobs/slurm_classic_array.sh` (`#SBATCH --array=0-9`, condition=`all`, alpha dummy)

Submit normally, e.g. `sbatch jobs/slurm_dgm_array.sh`. The same env var overrides apply.

## Results & Models

- Results: `results/<model>/raw/<condition>/alpha<alpha>/seed<seed>/...json`
- Models: `outputs/<model>/<condition>/alpha<alpha>/seed<seed>/...`
- Logs: under `logs/` as noted above.

## Analysis

`analysis/` holds scripts to parse/aggregate and plot metrics. They consume the JSON outputs under `results/` and write figures to `figs/`. Refer to script docstrings for CLI options.
