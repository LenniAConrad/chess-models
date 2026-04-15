#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: ./train.sh

Runs the default puzzle-classifier training pipeline.

Common overrides:
  USB_ROOT="/media/lennart/USB STICK" ./train.sh
  DEVICE=auto ./train.sh
  DEVICE=cuda CUDA_DEVICE=0 ./train.sh
  MAX_NEGATIVES=3000000 ./train.sh
  EPOCHS=80 PATIENCE=6 BATCH_SIZE=1024 LR=0.0003 ./train.sh

Outputs:
  training/classifier/data/       ignored local training tensors
  training/classifier/runs/       ignored local run artifacts
  models/puzzle-classifier_21planes-6blocksx64-head32-logit1.bin
EOF
  exit 0
fi

exec "$ROOT/training/classifier/run_full_usb_pipeline.sh" "$@"
