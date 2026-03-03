# CLAUDE.md — braindance

## Project Overview

LeJEPA-EEG: Evaluating LeJEPA self-supervised pretraining for ENIGMA EEG encoder.
Primary question: Does SSL pretraining reduce the 15-minute calibration window?
Secondary: Does SSL compensate for reduced channels (16 vs 64)?

## Development Rules

- **Python tooling**: Use `uv` only. Never pip, conda, or poetry.
- **Package management**: All deps in `pyproject.toml`.
- **Imports**: Use `lejepa` (pip package) for LeJEPA imports. Reference upstream repos as `upstream.enigma` / `upstream.lejepa` in comments only.
- **Type hints**: Standard Python type hints throughout.
- **Docstrings**: Google style, brief.
- **Config**: YAML configs loaded with PyYAML. Training scripts accept `--config`.
- **Logging**: Python `logging` module, not print().
- **Device handling**: Support CUDA and CPU via `torch.device`.

## Key Architecture

- **ENIGMA encoder**: Spatio-temporal CNN, input (B, 63, 275), output (B, 184)
- **ENIGMA projector**: 184 → 1024 (CLIP ViT-H/14), total 2.38M params
- **LeJEPA API**: `lejepa.univariate.EppsPulley` + `lejepa.multivariate.SlicingUnivariateTest`
- **LeJEPA loss**: (1-λ)*pred + λ*SIGReg, λ=0.05
- **For CNN encoders**: global_views = all_views (no local/global distinction)

## Project Structure

```
upstream/          # git submodules (ENIGMA, lejepa)
src/               # main package
configs/           # YAML configs
scripts/           # shell automation scripts
```

## Testing

No test suite yet. Verify with:
- Python syntax: `python -c "import ast; ast.parse(open('file.py').read())"`
- YAML validity: `python -c "import yaml; yaml.safe_load(open('file.yaml'))"`
- Shell syntax: `bash -n script.sh`
