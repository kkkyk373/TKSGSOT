## Overview

This repository hosts the reproducible subset of the selective transfer learning experiments for commuting origin–destination (OD) flow prediction. The code covers three model families:

* `src/experiments/run_selective_dgm.py` – Deep Gravity Model (PyTorch)
* `src/experiments/run_selective_rf.py` – Random Forest regressor (scikit-learn)
* `src/experiments/run_selective_svr.py` – Support Vector Regressor (scikit-learn)

All scripts share the same command-line interface so that model comparisons can be automated. Analysis helpers, pre-computed FGW distance matrices, and the source/target split lists required to reproduce the paper figures are included.

```
analysis/                     Plotting & aggregation utilities
comod_source_target_lists/    Source/target area ID splits (one pair per seed)
jobs/slurm_classic_array.sh   Example SLURM array submission (if you dont use, you may have to write other kinds of scripts.)
results/                      Saved experiment results as JSON files.
figs/                         Saved plottings by the analysis scripts
src/                          The main
```

## Environment Setup

1. Create a dedicated Python environment (Python ≥ 3.10) and install the dependencies:
   ```bash
   python -m venv env
   source env/bin/activate
   pip install -r requirements.txt
   ```

## Data Preparation

The scripts expect the [Commuting OD dataset](https://github.com/tsinghua-fib-lab/CommutingODGen-Dataset) to be available locally with the following layout relative to the repository root:

```
ComOD-dataset/
├── data/
│   └── <area_id>/
│       ├── demos.npy
│       ├── pois.npy
│       ├── dis.npy
│       └── od.npy
└── fgw_dist_matrice/
    ├── fgw_area_ids.npy
    └── fgw_dist_<alpha>.dat    # memory-mapped FGW distances
```

If you need to regenerate the FGW distances locally, run `src/experiments/fgw.py`. The script iterates over the `data/` subfolders, computes all pairwise FGW distances, and writes the outputs directly under `ComOD-dataset/fgw_dist_matrice/` (or a custom destination via the CLI flags):

```bash
python src/experiments/fgw.py \
  --data_dir ComOD-dataset/data \
  --alpha 0.5 \
  --n_graphs 100 \
  --ids_bin ComOD-dataset/fgw_dist_matrice/fgw_area_ids.npy \
  --dist_bin ComOD-dataset/fgw_dist_matrice/fgw_dist_50.dat
```

The resulting `fgw_area_ids.npy` captures the ordered list of areas used when building the matrix, and `fgw_dist_*.dat` stores the corresponding symmetric FGW distance matrix in memory-mapped form for the experiment scripts.

Set the environment variables below if you store the dataset elsewhere (the defaults point to the structure above):

```
export TKSGSOT_DATA_DIR=/path/to/data
export TKSGSOT_FGW_DIR=/path/to/fgw_dist_matrice
```

The source/target splits used throughout the experiments are provided in `comod_source_target_lists/targets_seed*.txt` and `comod_source_target_lists/sources_seed*.txt`.

## Running Experiments

Each experiment script shares the same set of CLI flags. Replace `<MODEL>` with `dgm`, `rf`, or `svr`:

```bash
python src/experiments/run_selective_<MODEL>.py \
  --data_dir "${TKSGSOT_DATA_DIR:-ComOD-dataset/data}" \
  --fgw_dir "${TKSGSOT_FGW_DIR:-ComOD-dataset/fgw_dist_matrice}" \
  --targets_path comod_source_target_lists/targets_seed0.txt \
  --sources_path comod_source_target_lists/sources_seed0.txt \
  --condition topk \
  --top_k 100 \
  --alpha 50 \
  --max_samples 5000 \
  --results_dir results \
  --model_output_dir outputs \
  --seed 0
```

* `condition`: `topk`, `bottomk`, `random`, or `all`.
* `top_k` / `bottom_k`: number of source areas to select for each target.
* `alpha`: FGW trade-off parameter (ignored for `all` and `random`).
* `max_samples`: maximum sampled OD pairs per training call (use `None` to disable subsampling).
* Additional DGM-only flags include `--epochs`, `--batch_size`, and `--lr`.

## Analysis & Reproduction

The `analysis/` directory contains self-contained utilities to:

* Parse legacy SVR/RF text logs into JSON (`parse_classic_json.py`).
* Aggregate micro-level (pair) and macro-level (graph) metrics across seeds (`aggregate_micro_results.py`, `aggregate_macro_results.py`).
* Render publication-ready comparison plots with Matplotlib (`plot_all_summaries_plt.py`) or Seaborn (`plot_all_summaries_sns.py`).
* Visualize FGW distance matrices or example geographic regions (`plot_fgw_dist.py`, `plot_geo_cases.py`).

All scripts read from the `results/` and `figs/` directories included in this repository. Refer to the docstrings for command-line usage.

## Validation Checklist

To recreate the released numbers:

1. Download and place the ComOD dataset plus FGW outputs under `ComOD-dataset/` (`fgw_dist_matrice/` holds the provided FGW files).
2. Install dependencies via `pip install -r requirements.txt`.
3. Run the desired experiment script (or the provided SLURM job) for the seeds of interest.
4. Post-process the raw JSON in `results/<model>/raw/...` with the aggregation utilities.
5. Use `analysis/plot_all_summaries_*.py` to regenerate the comparison figures in `figs/`.

The repository contains only anonymized, relative paths, so it can be published as-is once the dataset paths are configured on the target machine.
