#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEEDS="${SEEDS:-43 44}"

if [[ -d "/kaggle/working" ]]; then
    DEFAULT_OUTPUT_ROOT="/kaggle/working/dico_nli_runs/modernbert_large_full_seeds43_44"
else
    DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/results/modernbert_large_full_seeds43_44"
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-${DEFAULT_OUTPUT_ROOT}}"

mkdir -p "${OUTPUT_ROOT}"

for seed in ${SEEDS}; do
    output_dir="${OUTPUT_ROOT}/seed_${seed}"
    echo "============================================================"
    echo "ModernBERT-large full fine-tuning | seed=${seed}"
    echo "Output: ${output_dir}"
    echo "============================================================"

    SEED="${seed}" \
    OUTPUT_DIR="${output_dir}" \
    SAVE_EACH_EPOCH="${SAVE_EACH_EPOCH:-0}" \
    PYTHON_BIN="${PYTHON_BIN:-python}" \
    MODEL_PATH="${MODEL_PATH:-answerdotai/ModernBERT-large}" \
    DATA_ROOT="${DATA_ROOT:-}" \
    EPOCHS="${EPOCHS:-5}" \
    MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}" \
    GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}" \
    MAX_LENGTH="${MAX_LENGTH:-128}" \
    LEARNING_RATE="${LEARNING_RATE:-2e-5}" \
    COMPUTE_DTYPE="${COMPUTE_DTYPE:-float16}" \
    bash "${SCRIPT_DIR}/run_modernbert_large_full.sh"
done
