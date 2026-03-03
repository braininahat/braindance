# braindance

LeJEPA-EEG: Evaluating self-supervised pretraining (LeJEPA) for the ENIGMA EEG encoder.

## Research Questions

1. **Calibration efficiency**: Does LeJEPA pretraining reduce ENIGMA's 15-minute per-subject calibration requirement?
2. **Channel reduction**: Does SSL pretraining compensate for reduced EEG channels (16-channel OpenBCI Cyton+Daisy vs 64-channel research caps)?

## Method

- Pretrain ENIGMA's spatio-temporal CNN encoder on THINGS-EEG2 using LeJEPA (distribution-matching SSL)
- Fine-tune with varying calibration budgets (1, 2, 5, 10, 15 minutes)
- Ablate across channel counts (16, 24, 32, 63) and view strategies

## Setup

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/braininahat/braindance.git
cd braindance

# Install dependencies
uv sync
```

## Project Structure

```
src/                    # Main package
  views.py              # EEG view generation strategies for LeJEPA
  data.py               # THINGS-EEG2 dataset loading
  channel_subsample.py  # Channel subsampling utilities
  lejepa_eeg.py         # LeJEPA-ENIGMA wrapper
  train_lejepa.py       # LeJEPA pretraining script
  train_enigma.py       # ENIGMA fine-tuning script
  evaluate.py           # Evaluation metrics
  utils.py              # Shared utilities
configs/                # YAML configuration files
scripts/                # Shell scripts for experiment automation
upstream/               # Git submodules (ENIGMA, LeJEPA)
```

## References

- [ENIGMA](https://github.com/Alljoined/ENIGMA) — EEG-Driven Image Generation with Multi-modal Alignment
- [LeJEPA](https://github.com/galilai-group/lejepa) — Learning by Joint Embedding Predictive Architectures
- [THINGS-EEG2](https://openneuro.org/datasets/ds004212) — Large-scale EEG dataset
