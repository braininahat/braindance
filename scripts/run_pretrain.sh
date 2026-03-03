#!/usr/bin/env bash
# Run LeJEPA pretraining
set -euo pipefail

CONFIG="${1:-configs/lejepa_default.yaml}"

echo "=== LeJEPA Pretraining ==="
uv run python -m src.train_lejepa --config "$CONFIG"
