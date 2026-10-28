#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-microsoft/deberta-v2-xxlarge}"
MAX_LENGTH="${MAX_LENGTH:-128}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
EPOCHS="${EPOCHS:-3}"
COMPUTE_DTYPE="${COMPUTE_DTYPE:-float16}"

KAGGLE_DATA_ROOT="/kaggle/input/datasets/gongbaoxin/dico-nli-dataset/final_data"
if [[ -z "${DATA_ROOT:-}" ]]; then
    if [[ -d "${KAGGLE_DATA_ROOT}" ]]; then
        DATA_ROOT="${KAGGLE_DATA_ROOT}"
    else
        DATA_ROOT="${REPO_ROOT}/final_data"
    fi
fi

if [[ -d "/kaggle/working" ]]; then
    DEFAULT_OUTPUT_DIR="/kaggle/working/dico_nli_runs/deberta_v2_xxlarge_lora_track1"
else
    DEFAULT_OUTPUT_DIR="${SCRIPT_DIR}/results/deberta_v2_xxlarge_lora_track1"
fi
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"

TRAIN_PATH="${DATA_ROOT}/train/dico_nli_train_track1_participant_labeled.csv"
DEV_PATH="${DATA_ROOT}/dev/dico_nli_dev_track1_participant_labeled.csv"
REFERENCE_PATH="${DATA_ROOT}/dev/dico_nli_dev_track1_reference.csv"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

for data_path in "${TRAIN_PATH}" "${DEV_PATH}" "${REFERENCE_PATH}"; do
    if [[ ! -f "${data_path}" ]]; then
        echo "Required data file not found: ${data_path}" >&2
        exit 1
    fi
done

mkdir -p "${OUTPUT_DIR}"

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
    --warmup_pcrt 0.03 \
    --lr 1e-4 \
    --batch_size "${MICRO_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --wd 1e-5 \
    --num_epochs "${EPOCHS}" \
    --no-is_optuna_trial \
    --max_length "${MAX_LENGTH}" \
    --use_lora \
    --gradient_checkpointing \
    --compute_dtype "${COMPUTE_DTYPE}" \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules "query_proj,value_proj" \
    --no-deterministic \
    --results_dir "${OUTPUT_DIR}" \
    --reference_path "${REFERENCE_PATH}" \
    --save_path "${OUTPUT_DIR}"
