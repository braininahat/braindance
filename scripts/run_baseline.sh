#!/usr/bin/env bash
# Run ENIGMA baseline (no pretraining) at full 15-min calibration
set -euo pipefail

CONFIG="${1:-configs/enigma_finetune.yaml}"

echo "=== ENIGMA Baseline (no pretraining, 15-min calibration) ==="
uv run python -m src.train_enigma --config "$CONFIG"
