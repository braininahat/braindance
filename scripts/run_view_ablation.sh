#!/usr/bin/env bash
# Run view strategy ablation: pretrain with different view generation strategies
set -euo pipefail

STRATEGIES=(trial_repetition temporal_jitter channel_subset combined_ab combined_abc)
LEJEPA_CONFIG="configs/lejepa_default.yaml"
ENIGMA_CONFIG="configs/enigma_finetune.yaml"

for strategy in "${STRATEGIES[@]}"; do
    echo "=== View ablation: ${strategy} ==="

    # Generate configs
    uv run python -c "
import yaml
# LeJEPA config
with open('$LEJEPA_CONFIG') as f:
    cfg = yaml.safe_load(f)
cfg['views']['strategy'] = '$strategy'
cfg['infra']['checkpoint_dir'] = 'checkpoints/view_ablation/${strategy}/lejepa'
cfg['infra']['log_dir'] = 'logs/view_ablation/${strategy}/lejepa'
with open('/tmp/lejepa_${strategy}.yaml', 'w') as f:
    yaml.dump(cfg, f)
# ENIGMA config
with open('$ENIGMA_CONFIG') as f:
    cfg = yaml.safe_load(f)
cfg['pretrained_checkpoint'] = 'checkpoints/view_ablation/${strategy}/lejepa/final.pt'
cfg['infra']['checkpoint_dir'] = 'checkpoints/view_ablation/${strategy}/enigma'
cfg['infra']['log_dir'] = 'logs/view_ablation/${strategy}/enigma'
with open('/tmp/enigma_${strategy}.yaml', 'w') as f:
    yaml.dump(cfg, f)
"

    # Pretrain
    echo "--- LeJEPA pretraining (${strategy}) ---"
    uv run python -m src.train_lejepa --config "/tmp/lejepa_${strategy}.yaml"

    # Fine-tune
    echo "--- ENIGMA fine-tuning (${strategy}) ---"
    uv run python -m src.train_enigma --config "/tmp/enigma_${strategy}.yaml"
done

echo "=== View ablation complete ==="
