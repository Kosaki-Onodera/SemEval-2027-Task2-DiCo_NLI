#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-answerdotai/ModernBERT-large}"
MAX_LENGTH="${MAX_LENGTH:-128}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
EPOCHS="${EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
COMPUTE_DTYPE="${COMPUTE_DTYPE:-float16}"

KAGGLE_DATA_ROOT="/kaggle/input/datasets/gongbaoxin/dico-nli-dataset/final_data"
if [[ -z "${DATA_ROOT:-}" ]]; then
    if [[ -d "${KAGGLE_DATA_ROOT}" ]]; then
        DATA_ROOT="${KAGGLE_DATA_ROOT}"
    else
        DATA_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)/final_data"
    fi
fi

if [[ -d "/kaggle/working" ]]; then
    DEFAULT_OUTPUT_DIR="/kaggle/working/dico_nli_runs/modernbert_large_full"
else
    DEFAULT_OUTPUT_DIR="${SCRIPT_DIR}/results/modernbert_large_full"
fi
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"

TRAIN_PATH="${DATA_ROOT}/train/dico_nli_train_track1_participant_labeled.csv"
DEV_PATH="${DATA_ROOT}/dev/dico_nli_dev_track1_participant_labeled.csv"
REFERENCE_PATH="${DATA_ROOT}/dev/dico_nli_dev_track1_reference.csv"

for data_path in "${TRAIN_PATH}" "${DEV_PATH}" "${REFERENCE_PATH}"; do
    if [[ ! -f "${data_path}" ]]; then
        echo "Required data file not found: ${data_path}" >&2
        exit 1
    fi
done

mkdir -p "${OUTPUT_DIR}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"${PYTHON_BIN}" -u "${SCRIPT_DIR}/main.py" \
    --model_path "${MODEL_PATH}" \
    --tokenizer_path "${MODEL_PATH}" \
    --do_train \
    --train_path "${TRAIN_PATH}" \
    --do_dev \
    --dev_path "${DEV_PATH}" \
    --no-do_test \
    --seed 42 \
    --dropout 0.1 \
    --warmup_pcrt 0.05 \
    --lr "${LEARNING_RATE}" \
    --batch_size "${MICRO_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --wd 1e-5 \
    --num_epochs "${EPOCHS}" \
    --no-is_optuna_trial \
    --max_length "${MAX_LENGTH}" \
    --no-use_lora \
    --gradient_checkpointing \
    --use_amp \
    --compute_dtype "${COMPUTE_DTYPE}" \
    --negative_class_weight 1.0 \
    --rdrop_alpha 0.0 \
    --scl_weight 0.0 \
    --pair_consistency_weight 0.0 \
    --no-deterministic \
    --results_dir "${OUTPUT_DIR}" \
    --reference_path "${REFERENCE_PATH}" \
    --save_path "${OUTPUT_DIR}" \
    --save_each_epoch
