#!/usr/bin/env bash
# Run channel count ablation: pretrain and fine-tune at different channel counts
set -euo pipefail

MONTAGES=(ch16 ch24 ch32 ch63)
LEJEPA_CONFIG="configs/lejepa_default.yaml"
ENIGMA_CONFIG="configs/enigma_finetune.yaml"

for montage in "${MONTAGES[@]}"; do
    # Get channel count from montage name
    num_ch="${montage#ch}"

    echo "=== Channel ablation: ${montage} (${num_ch} channels) ==="

    # Generate configs with adjusted channel count
    uv run python -c "
import yaml
# LeJEPA config
with open('$LEJEPA_CONFIG') as f:
    cfg = yaml.safe_load(f)
cfg['data']['num_channels'] = $num_ch
cfg['infra']['checkpoint_dir'] = 'checkpoints/channel_ablation/${montage}/lejepa'
cfg['infra']['log_dir'] = 'logs/channel_ablation/${montage}/lejepa'
with open('/tmp/lejepa_${montage}.yaml', 'w') as f:
    yaml.dump(cfg, f)
# ENIGMA config
with open('$ENIGMA_CONFIG') as f:
    cfg = yaml.safe_load(f)
cfg['data']['num_channels'] = $num_ch
cfg['pretrained_checkpoint'] = 'checkpoints/channel_ablation/${montage}/lejepa/final.pt'
cfg['infra']['checkpoint_dir'] = 'checkpoints/channel_ablation/${montage}/enigma'
cfg['infra']['log_dir'] = 'logs/channel_ablation/${montage}/enigma'
with open('/tmp/enigma_${montage}.yaml', 'w') as f:
    yaml.dump(cfg, f)
"

    # Pretrain
    echo "--- LeJEPA pretraining (${montage}) ---"
    uv run python -m src.train_lejepa --config "/tmp/lejepa_${montage}.yaml"

    # Fine-tune
    echo "--- ENIGMA fine-tuning (${montage}) ---"
    uv run python -m src.train_enigma --config "/tmp/enigma_${montage}.yaml"
done

echo "=== Channel ablation complete ==="
