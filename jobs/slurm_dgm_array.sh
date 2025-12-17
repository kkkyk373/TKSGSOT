#!/bin/bash
#SBATCH --job-name=dgm_unified
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-79 # topk/bottomk × alpha × seed(10) + all + random

# Resolve project paths regardless of submission directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Parameter definitions (edit here to control the sweep).
CONDITIONS=("topk" "bottomk" "all" "random")
ALPHAS=(0 50 100)
SEEDS=({0..9})

# Pre-compute parameter combinations.
PARAMS=()
for seed in "${SEEDS[@]}"; do
    for alpha in "${ALPHAS[@]}"; do
        for cond in "topk" "bottomk"; do
            PARAMS+=("${cond} ${alpha} ${seed}")
        done
    done
done
for seed in "${SEEDS[@]}"; do
    PARAMS+=("all 0 ${seed}")
    PARAMS+=("random 0 ${seed}")
done

# Map the task ID to a parameter set.
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [ "${TASK_ID}" -lt 0 ] || [ "${TASK_ID}" -ge "${#PARAMS[@]}" ]; then
    echo "[ERROR] TASK_ID ${TASK_ID} is out of range (0-$(( ${#PARAMS[@]} - 1 )))." >&2
    exit 1
fi
read PARAM_COND PARAM_ALPHA PARAM_SEED <<< "${PARAMS[$TASK_ID]}"
JOB_ID="${SLURM_JOB_ID:-manual}"

# Logging configuration.
LOG_DIR="${PROJECT_ROOT}/logs/dgm_unified/${PARAM_COND}/alpha${PARAM_ALPHA}"
mkdir -p "${LOG_DIR}"
export OUT_FILE="${LOG_DIR}/${JOB_ID}_seed${PARAM_SEED}.out"
export ERR_FILE="${LOG_DIR}/${JOB_ID}_seed${PARAM_SEED}.err"
exec > "${OUT_FILE}" 2> "${ERR_FILE}"

echo "--- DGM Unified Experiment ---"
echo "Job ID: ${JOB_ID}, Array Task ID: ${TASK_ID}"
echo "Timestamp: $(date)"
echo "Parameters: condition=${PARAM_COND}, alpha=${PARAM_ALPHA}, seed=${PARAM_SEED}"
echo "----------------------"

# Paths can be overridden via environment variables to keep the script anonymized.
VENV_ACTIVATE="${VENV_ACTIVATE:-${PROJECT_ROOT}/.venv/bin/activate}"
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

PYTHONPATH="${PROJECT_ROOT}" "${PYTHON_BIN}" src/experiments/run_selective_dgm.py \
    --data_dir "${DATA_DIR}" \
    --fgw_dir "${FGW_DIR}" \
    --targets_path "${TARGETS_PATH}" \
    --sources_path "${SOURCES_PATH}" \
    --results_dir "${RESULTS_DIR}" \
    --model_output_dir "${MODEL_OUTPUT_DIR}" \
    --condition "${PARAM_COND}" \
    --alpha "${PARAM_ALPHA}" \
    --seed "${PARAM_SEED}" \
    --epochs 20 \
    --max_samples 50000 \
    --lr 0.001 \
    --batch_size 32

echo "--- Job Finished ---"
echo "Timestamp: $(date)"
