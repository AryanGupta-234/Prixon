#!/usr/bin/env bash
set -euo pipefail

# Run this from the Prixon repository root on a Linux cloud GPU instance.
# All tunable values remain in YAML/environment variables.

PYTHON_BIN="${PYTHON_BIN:-python3}"
REQUIREMENTS_FILE="${PRIXON_TRAIN_REQUIREMENTS:-training/requirements.txt}"
DATA_CONFIG="${PRIXON_DATA_CONFIG:-training/config/dataset.yaml}"
TRAIN_CONFIG="${PRIXON_TRAIN_CONFIG:-training/config/training.yaml}"

"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install -r "${REQUIREMENTS_FILE}"

"${PYTHON_BIN}" training/build_dataset.py --config "${DATA_CONFIG}"
"${PYTHON_BIN}" training/evaluate_dataset.py \
  "${PRIXON_TRAIN_FILE:-training/output/prixon_train.jsonl}" \
  "${PRIXON_VALIDATION_FILE:-training/output/prixon_validation.jsonl}" \
  "${PRIXON_EVAL_FILE:-training/output/prixon_eval.jsonl}"

TRAIN_ARGS=(--config "${TRAIN_CONFIG}")
if [[ "${PRIXON_RESUME:-false}" == "true" ]]; then
  TRAIN_ARGS+=(--resume)
fi

"${PYTHON_BIN}" training/train_qwen.py "${TRAIN_ARGS[@]}"

if [[ "${PRIXON_EXPORT:-true}" == "true" ]]; then
  ADAPTER_DIR="${PRIXON_TRAIN_OUTPUT:-training/runs/prixon-qwen}/adapter"
  "${PYTHON_BIN}" training/export_qwen.py --config "${TRAIN_CONFIG}" --adapter "${ADAPTER_DIR}"
fi
