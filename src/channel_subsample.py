"""Channel subsampling utilities for EEG data.

Provides predefined channel montages and utilities for mapping between
full and reduced channel sets, enabling experiments at different channel counts.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Default path to channel montages config
_DEFAULT_MONTAGE_PATH = Path(__file__).parent.parent / "configs" / "channel_montages.yaml"


def load_montages(path: str | Path | None = None) -> dict[str, dict]:
    """Load channel montage definitions from YAML.

    Args:
        path: Path to montages YAML. Uses default if None.

    Returns:
        Dict mapping montage name to montage config.
    """
    path = Path(path) if path else _DEFAULT_MONTAGE_PATH
    with open(path) as f:
        return yaml.safe_load(f)


def get_channel_indices(
    montage_name: str,
    full_channel_list: list[str],
    montages: dict[str, dict] | None = None,
) -> list[int]:
    """Get indices of montage channels within the full channel list.

    Args:
        montage_name: Name of the montage (e.g., "ch16", "ch32").
        full_channel_list: Ordered list of all channel names in the dataset.
        montages: Pre-loaded montage definitions. Loaded from default if None.

    Returns:
        Sorted list of integer indices into full_channel_list.

    Raises:
        KeyError: If montage_name not found.
        ValueError: If a montage channel is not in full_channel_list.
    """
    if montages is None:
        montages = load_montages()
    if montage_name not in montages:
        raise KeyError(f"Montage '{montage_name}' not found. Available: {list(montages)}")

    subset_channels = montages[montage_name]["channels"]
    channel_to_idx = {ch: i for i, ch in enumerate(full_channel_list)}
    indices = []
    for ch in subset_channels:
        if ch not in channel_to_idx:
            raise ValueError(f"Channel '{ch}' from montage '{montage_name}' not in dataset.")
        indices.append(channel_to_idx[ch])

    return sorted(indices)


def subsample_eeg(
    eeg,
    channel_indices: list[int],
):
    """Subsample EEG channels using precomputed indices.

    Args:
        eeg: EEG tensor, shape (..., C, T).
        channel_indices: Indices of channels to keep.

    Returns:
        Subsampled EEG tensor, shape (..., len(channel_indices), T).
    """
    return eeg[..., channel_indices, :]
