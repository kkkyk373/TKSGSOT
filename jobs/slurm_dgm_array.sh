#!/bin/bash
#SBATCH --job-name=dgm_unified
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-7 # alpha × seed × top/bottomk + 1(all) + 1(random)

# --- パラメータ定義 (ここを編集して実験を制御) ---
CONDITIONS=("topk" "bottomk" "all" "random")
ALPHAS=(0 50 100)
SEEDS=(9)

# --- パラメータ組み合わせを事前に定義 ---
PARAMS=()
# topk, bottomk (alphaあり)
for seed in "${SEEDS[@]}"; do
    for alpha in "${ALPHAS[@]}"; do
        for cond in "topk" "bottomk"; do
            PARAMS+=("${cond} ${alpha} ${seed}")
        done
    done
done
# all, random (alphaなし)
for seed in "${SEEDS[@]}"; do
    PARAMS+=("all 0 ${seed}")
    PARAMS+=("random 0 ${seed}")
done

# --- SlurmタスクIDからパラメータを取得 ---
# SLURM_ARRAY_TASK_ID は 0 から始まる
read PARAM_COND PARAM_ALPHA PARAM_SEED <<< "${PARAMS[$SLURM_ARRAY_TASK_ID]}"

# --- ログ設定 ---
LOG_DIR="logs/dgm_unified/${PARAM_COND}/alpha${PARAM_ALPHA}"
mkdir -p ${LOG_DIR}
export OUT_FILE="${LOG_DIR}/${SLURM_JOB_ID}_seed${PARAM_SEED}.out"
export ERR_FILE="${LOG_DIR}/${SLURM_JOB_ID}_seed${PARAM_SEED}.err"
exec > "$OUT_FILE" 2> "$ERR_FILE"

# --- 環境設定と実行 ---
echo "--- DGM Unified Experiment ---"
echo "Job ID: ${SLURM_JOB_ID}, Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Timestamp: $(date)"
echo "Parameters: condition=${PARAM_COND}, alpha=${PARAM_ALPHA}, seed=${PARAM_SEED}"
echo "----------------------"

# 仮想環境のアクティベート
source /work/hideki-h/tgss/env/bin/activate

# Pythonスクリプトの実行
PYTHONPATH=. python src/experiments/run_selective_dgm.py \
    --data_dir "/work/hideki-h/tgss/ComOD-dataset/data" \
    --fgw_dir "/work/hideki-h/tgss/ComOD-dataset/fgw_dist_matrice" \
    --targets_path "comod_source_target_lists/targets_seed${PARAM_SEED}.txt" \
    --sources_path "comod_source_target_lists/sources_seed${PARAM_SEED}.txt" \
    --results_dir "results" \
    --model_output_dir "outputs" \
    --condition "${PARAM_COND}" \
    --alpha ${PARAM_ALPHA} \
    --seed ${PARAM_SEED} \
    --epochs 20 \
    --max_samples 50000 \
    --lr 0.001 \
    --batch_size 32

echo "--- Job Finished ---"
echo "Timestamp: $(date)"
