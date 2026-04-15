#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAINING_DIR="$REPO_ROOT/training/classifier"
CHESS_RTK="${CHESS_RTK:-/home/lennart/Code/chess-rtk}"
USB_ROOT="${USB_ROOT:-/media/lennart/USB STICK}"
DATA_NAME="${DATA_NAME:-usb-stacks-puzzle-focused}"
DATA_STEM="$TRAINING_DIR/data/$DATA_NAME/$DATA_NAME"
RUNS_DIR="$TRAINING_DIR/runs"
MODEL_OUT="${MODEL_OUT:-$REPO_ROOT/models/puzzle-classifier_21planes-6blocksx64-head32-logit1.bin}"

mkdir -p "$(dirname "$DATA_STEM")" "$RUNS_DIR" "$(dirname "$MODEL_OUT")"

if [[ ! -d "$USB_ROOT/1M_STACK" || ! -d "$USB_ROOT/2M_STACK" ]]; then
  echo "USB stack directories not found under: $USB_ROOT" >&2
  echo "Mount the USB or set USB_ROOT=/path/to/usb." >&2
  exit 1
fi

if [[ ! -f "$DATA_STEM.classifier.inputs.npy" || ! -f "$DATA_STEM.classifier.labels.npy" ]]; then
  echo "Exporting puzzle-focused classifier dataset to $DATA_STEM"
  echo "This scans all USB stacks, writes all positives, and caps negatives at MAX_NEGATIVES=${MAX_NEGATIVES:-3000000}."
  java -cp "$CHESS_RTK/out" application.Main record-to-classifier \
    -i "$USB_ROOT/1M_STACK" \
    -i "$USB_ROOT/2M_STACK" \
    --recursive \
    --max-negatives "${MAX_NEGATIVES:-3000000}" \
    -o "$DATA_STEM"
else
  echo "Reusing existing dataset at $DATA_STEM"
fi

weighted_loss_args=()
if [[ "${WEIGHTED_LOSS:-0}" == "1" ]]; then
  weighted_loss_args=(--weighted-loss)
fi

echo "Training classifier CNN with all real puzzle rows and sampled real non-puzzle rows"

python3 -u "$TRAINING_DIR/train_classifier.py" \
  --base "$DATA_STEM" \
  --out-dir "$RUNS_DIR" \
  --model cnn \
  --use-all-data \
  --epochs "${EPOCHS:-80}" \
  --batch-size "${BATCH_SIZE:-1024}" \
  --lr "${LR:-0.0003}" \
  --weight-decay "${WEIGHT_DECAY:-0.0001}" \
  --early-stopping-patience "${PATIENCE:-6}" \
  --early-stopping-min-delta "${MIN_DELTA:-0.0005}" \
  --device "${DEVICE:-auto}" \
  --cuda-device "${CUDA_DEVICE:-0}" \
  "${weighted_loss_args[@]}" \
  ${THREADS:+--threads "$THREADS"}

latest_run="$(find "$RUNS_DIR" -mindepth 1 -maxdepth 1 -type d -name '*-cnn' -printf '%T@ %p\n' | sort -nr | awk 'NR==1 {print $2}')"
if [[ -z "$latest_run" ]]; then
  echo "No CNN run directory found under $RUNS_DIR" >&2
  exit 1
fi

python3 "$TRAINING_DIR/export_classifier_bin.py" \
  --checkpoint "$latest_run/model.pt" \
  --output "$MODEL_OUT"

sha256sum "$MODEL_OUT" | tee "$latest_run/export.sha256"
echo "$MODEL_OUT" > "$latest_run/exported-model.txt"
echo "Exported $MODEL_OUT from $latest_run"
