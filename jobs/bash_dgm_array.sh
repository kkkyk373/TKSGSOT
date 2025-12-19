#!/bin/bash

# Local bash runner for the DGM sweep (mirrors slurm_dgm_array.sh without Slurm).

# Resolve project paths regardless of invocation directory.
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

# Paths can be overridden via environment variables to keep the script anonymized.
VENV_ACTIVATE="${VENV_ACTIVATE:-${PROJECT_ROOT}/.venv/bin/activate}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/ComOD-dataset/data}"
FGW_DIR="${FGW_DIR:-${PROJECT_ROOT}/ComOD-dataset/fgw_dist_matrice}"
TARGETS_BASE="${TARGETS_BASE:-${PROJECT_ROOT}/comod_source_target_lists}"
SOURCES_BASE="${SOURCES_BASE:-${PROJECT_ROOT}/comod_source_target_lists}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_ROOT}/results}"
MODEL_OUTPUT_DIR_DEFAULT="${MODEL_OUTPUT_DIR:-${PROJECT_ROOT}/outputs}"
PYTHON_BIN="${PYTHON_BIN:-python}"

print_usage() {
    echo "Usage: $0 [--save-models] [index ...]"
    echo "  --save-models    Enable model checkpoint saving (default: off)"
    echo "  --no-save-models Disable model checkpoint saving"
    echo "  -h, --help       Show this help"
}

SAVE_MODELS=0
TASK_LIST=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --save-models)
            SAVE_MODELS=1
            shift
            ;;
        --no-save-models)
            SAVE_MODELS=0
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "[ERROR] Unknown option: $1" >&2
            print_usage
            exit 1
            ;;
        *)
            TASK_LIST+=("$1")
            shift
            ;;
    esac
done

if [ "$#" -gt 0 ]; then
    TASK_LIST+=("$@")
fi

run_task() {
    local idx="$1"
    if [ "${idx}" -lt 0 ] || [ "${idx}" -ge "${#PARAMS[@]}" ]; then
        echo "[WARN] Skipping invalid index ${idx} (valid: 0-$(( ${#PARAMS[@]} - 1 )))."
        return 0
    fi

    read PARAM_COND PARAM_ALPHA PARAM_SEED <<< "${PARAMS[$idx]}"

    local LOG_DIR="${PROJECT_ROOT}/logs/dgm_unified/${PARAM_COND}/alpha${PARAM_ALPHA}"
    mkdir -p "${LOG_DIR}"

    local JOB_ID="local$(printf '%02d' "${idx}")"
    local TS
    TS="$(date +%Y%m%d_%H%M%S)"
    local OUT_FILE="${LOG_DIR}/${JOB_ID}_seed${PARAM_SEED}_${TS}.out"
    local ERR_FILE="${LOG_DIR}/${JOB_ID}_seed${PARAM_SEED}_${TS}.err"

    local MODEL_OUTPUT_DIR_EFFECTIVE=""
    if [ "${SAVE_MODELS}" -eq 1 ]; then
        MODEL_OUTPUT_DIR_EFFECTIVE="${MODEL_OUTPUT_DIR_DEFAULT}"
    fi

    (
        exec >"${OUT_FILE}" 2>"${ERR_FILE}"
        echo "--- DGM Unified Experiment (local) ---"
        echo "Job ID: ${JOB_ID}, Index: ${idx}"
        echo "Timestamp: $(date)"
        echo "Parameters: condition=${PARAM_COND}, alpha=${PARAM_ALPHA}, seed=${PARAM_SEED}"
        echo "Save models: $([ "${SAVE_MODELS}" -eq 1 ] && echo yes || echo no)"
        echo "----------------------"

        if [ -f "${VENV_ACTIVATE}" ]; then
            # shellcheck disable=SC1090
            source "${VENV_ACTIVATE}"
        else
            echo "[WARN] Virtual environment not found at ${VENV_ACTIVATE}. Continuing with system python."
        fi

        PYTHONPATH="${PROJECT_ROOT}" "${PYTHON_BIN}" src/experiments/run_selective_dgm.py \
            --data_dir "${DATA_DIR}" \
            --fgw_dir "${FGW_DIR}" \
            --targets_path "${TARGETS_BASE}/targets_seed${PARAM_SEED}.txt" \
            --sources_path "${SOURCES_BASE}/sources_seed${PARAM_SEED}.txt" \
            --results_dir "${RESULTS_DIR}" \
            --model_output_dir "${MODEL_OUTPUT_DIR_EFFECTIVE}" \
            --condition "${PARAM_COND}" \
            --alpha "${PARAM_ALPHA}" \
            --seed "${PARAM_SEED}" \
            --epochs 20 \
            --max_samples 50000 \
            --lr 0.001 \
            --batch_size 32

        echo "--- Job Finished ---"
        echo "Timestamp: $(date)"
    )

    echo "[INFO] Finished index ${idx} (${PARAM_COND}, alpha=${PARAM_ALPHA}, seed=${PARAM_SEED}). Logs -> ${OUT_FILE}"
}

# If indexes are provided, run only those; otherwise, run all.
if [ "${#TASK_LIST[@]}" -eq 0 ]; then
    TASK_LIST=("${!PARAMS[@]}")
    echo "[INFO] Prepared ${#PARAMS[@]} runs."
else
    echo "[INFO] Selected ${#TASK_LIST[@]} runs out of ${#PARAMS[@]}."
fi

for idx in "${TASK_LIST[@]}"; do
    run_task "${idx}"
done
