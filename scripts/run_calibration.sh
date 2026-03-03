#!/usr/bin/env bash
# Run calibration budget ablation: fine-tune with varying data budgets
# Compares pretrained vs from-scratch at 1, 2, 5, 10, 15 min calibration
set -euo pipefail

PRETRAINED_CKPT="${1:-checkpoints/lejepa/final.pt}"
BASE_CONFIG="configs/enigma_finetune.yaml"
BUDGETS=(1 2 5 10 15)

for budget in "${BUDGETS[@]}"; do
    echo "=== Calibration budget: ${budget} min ==="

    # From scratch
    echo "--- From scratch ---"
    uv run python -c "
import yaml
with open('$BASE_CONFIG') as f:
    cfg = yaml.safe_load(f)
cfg['calibration']['budget_minutes'] = $budget
cfg['pretrained_checkpoint'] = None
cfg['infra']['checkpoint_dir'] = 'checkpoints/calibration/scratch_${budget}min'
cfg['infra']['log_dir'] = 'logs/calibration/scratch_${budget}min'
with open('/tmp/cal_scratch_${budget}.yaml', 'w') as f:
    yaml.dump(cfg, f)
"
    uv run python -m src.train_enigma --config "/tmp/cal_scratch_${budget}.yaml"

    # Pretrained
    echo "--- Pretrained ---"
    uv run python -c "
import yaml
with open('$BASE_CONFIG') as f:
    cfg = yaml.safe_load(f)
cfg['calibration']['budget_minutes'] = $budget
cfg['pretrained_checkpoint'] = '$PRETRAINED_CKPT'
cfg['infra']['checkpoint_dir'] = 'checkpoints/calibration/pretrained_${budget}min'
cfg['infra']['log_dir'] = 'logs/calibration/pretrained_${budget}min'
with open('/tmp/cal_pretrained_${budget}.yaml', 'w') as f:
    yaml.dump(cfg, f)
"
    uv run python -m src.train_enigma --config "/tmp/cal_pretrained_${budget}.yaml"
done

echo "=== Calibration ablation complete ==="
