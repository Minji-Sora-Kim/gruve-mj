#!/usr/bin/env bash
set -euo pipefail

# Grader one-command inference entrypoint
# - Uses final EDA-aligned inference pipeline
# - Runs prediction on unseen_data.csv
# - Writes outputs to outputs/predictions/

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="python"
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
fi

TRAIN_PATH="$ROOT_DIR/data/processed/train.csv"
MODEL_PATH="$ROOT_DIR/ag_models"
OUT_DIR="$ROOT_DIR/outputs/predictions"
OUT_CSV="$OUT_DIR/unseen_predictions.csv"
OUT_METRICS="$OUT_DIR/unseen_metrics.json"

if [[ -f "$ROOT_DIR/unseen_data.csv" ]]; then
  INPUT_PATH="$ROOT_DIR/unseen_data.csv"
elif [[ -f "$ROOT_DIR/data/processed/unseen_data.csv" ]]; then
  INPUT_PATH="$ROOT_DIR/data/processed/unseen_data.csv"
else
  echo "[ERROR] unseen_data.csv not found in project root or data/processed/"
  exit 1
fi

if [[ ! -f "$TRAIN_PATH" ]]; then
  echo "[ERROR] Missing train split: $TRAIN_PATH"
  exit 1
fi

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "[ERROR] Missing model artifacts directory: $MODEL_PATH"
  exit 1
fi

mkdir -p "$OUT_DIR"

echo "[INFO] Python: $("$PYTHON_BIN" --version)"
echo "[INFO] Input:  $INPUT_PATH"
echo "[INFO] Model:  $MODEL_PATH"
echo "[INFO] Output: $OUT_CSV"

"$PYTHON_BIN" "$ROOT_DIR/src/eda_infer.py" \
  --train-path "$TRAIN_PATH" \
  --input-path "$INPUT_PATH" \
  --output-path "$OUT_CSV" \
  --metrics-output "$OUT_METRICS" \
  --model-path "$MODEL_PATH" \
  --label "readmitted" \
  --id-column "encounter_id" \
  --include-input

echo "[DONE] Inference completed."
echo "[DONE] Predictions: $OUT_CSV"
