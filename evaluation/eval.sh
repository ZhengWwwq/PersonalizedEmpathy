#!/usr/bin/env bash
set -euo pipefail

# Evaluate model predictions with fixed PERM criteria.
# Set paths and API credentials through environment variables before running.

: "${EVAL_DATASET_PATH:?Set EVAL_DATASET_PATH to the dataset JSON path}"
: "${EVAL_PREDICT_PATH:?Set EVAL_PREDICT_PATH to the prediction JSON path}"
: "${EVAL_API_KEY:?Set EVAL_API_KEY}"

EVAL_BASE_URL="${EVAL_BASE_URL:-https://api.deepseek.com}"
EVAL_MODEL="${EVAL_MODEL:-deepseek-chat}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.0}"
EVAL_NUM="${EVAL_NUM:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"

python eval.py \
    --dataset_path "${EVAL_DATASET_PATH}" \
    --predict_path "${EVAL_PREDICT_PATH}" \
    --criteria_path "${EVAL_CRITERIA_PATH:-}" \
    --eval_api_key "${EVAL_API_KEY}" \
    --eval_base_url "${EVAL_BASE_URL}" \
    --eval_model "${EVAL_MODEL}" \
    --eval_temperature "${EVAL_TEMPERATURE}" \
    --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --eval_num "${EVAL_NUM}"
