#!/bin/bash
#SBATCH --job-name=svr_all_exp
#SBATCH --partition=cluster_long 
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=40:00:00
#SBATCH --array=0-9

# Resolve project paths regardless of submission directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Parameter definitions.
SEEDS=(0 1 2 3 4 5 6 7 8 9)
PARAMS=()
for seed in "${SEEDS[@]}"; do
    PARAMS+=("all 0 ${seed}")  # alpha is unused for the all condition
done

# Map the task ID to a parameter set.
read PARAM_COND PARAM_ALPHA PARAM_SEED <<< "${PARAMS[$SLURM_ARRAY_TASK_ID]}"

# Logging configuration.
LOG_DIR="${PROJECT_ROOT}/logs/svr_all_exp/${PARAM_COND}"
mkdir -p "${LOG_DIR}"
export OUT_FILE="${LOG_DIR}/${SLURM_JOB_ID}_seed${PARAM_SEED}.out"
export ERR_FILE="${LOG_DIR}/${SLURM_JOB_ID}_seed${PARAM_SEED}.err"
exec > "${OUT_FILE}" 2> "${ERR_FILE}"

echo "--- SVR 'all' condition experiment ---"
echo "Job ID: ${SLURM_JOB_ID}, Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Timestamp: $(date)"
echo "Parameters: condition=${PARAM_COND}, alpha=${PARAM_ALPHA} (dummy), seed=${PARAM_SEED}"
echo "----------------------"

# Paths can be overridden via environment variables to keep the script anonymized.
VENV_ACTIVATE="${VENV_ACTIVATE:-${PROJECT_ROOT}/env/bin/activate}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/ComOD-dataset/data}"
FGW_DIR="${FGW_DIR:-${PROJECT_ROOT}/ComOD-dataset/fgw_dist_matrice}"
TARGETS_BASE="${TARGETS_BASE:-${PROJECT_ROOT}/comod_source_target_lists}"
SOURCES_BASE="${SOURCES_BASE:-${PROJECT_ROOT}/comod_source_target_lists}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_ROOT}/results}"
MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR:-${PROJECT_ROOT}/outputs}"
PYTHON_BIN="${PYTHON_BIN:-python}"

TARGETS_PATH="${TARGETS_BASE}/targets_seed${PARAM_SEED}.txt"
SOURCES_PATH="${SOURCES_BASE}/sources_seed${PARAM_SEED}.txt"

if [ -f "${VENV_ACTIVATE}" ]; then
    # shellcheck disable=SC1090
    source "${VENV_ACTIVATE}"
else
    echo "[WARN] Virtual environment not found at ${VENV_ACTIVATE}. Continuing with system python."
fi

PYTHONPATH="${PROJECT_ROOT}" "${PYTHON_BIN}" src/experiments/run_selective_svr.py \
    --data_dir "${DATA_DIR}" \
    --fgw_dir "${FGW_DIR}" \
    --targets_path "${TARGETS_PATH}" \
    --sources_path "${SOURCES_PATH}" \
    --results_dir "${RESULTS_DIR}" \
    --model_output_dir "${MODEL_OUTPUT_DIR}" \
    --condition "${PARAM_COND}" \
    --alpha "${PARAM_ALPHA}" \
    --seed "${PARAM_SEED}" \
    --max_samples 50000

echo "--- Job Finished ---"
echo "Timestamp: $(date)"
