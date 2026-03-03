"""THINGS-EEG2 dataset loading for LeJEPA pretraining and ENIGMA fine-tuning.

Loads preprocessed EEG epochs from THINGS-EEG2, supporting:
- Repetition-aware sampling (grouping multiple presentations of the same image)
- Channel subsampling via predefined montages
- Subject-specific or subject-pooled modes
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class ThingsEEG2Dataset(Dataset):
    """PyTorch Dataset for THINGS-EEG2 preprocessed EEG data.

    Expects preprocessed data in .npy format following ENIGMA's conventions:
    - Preprocessed with MNE: 0.5Hz high-pass, 60Hz notch, baseline correction,
      resampled to 250Hz, multivariate noise normalization.
    - Per-subject files: sub-{id}/split/eeg.npy (trials x channels x time)
    - Image condition labels: sub-{id}/split/labels.npy (trials,)
    """

    def __init__(
        self,
        data_dir: str | Path,
        subjects: list[str],
        split: str = "train",
        channel_indices: Optional[list[int]] = None,
        max_trials: Optional[int] = None,
    ):
        """Args:
            data_dir: Root directory containing preprocessed data.
            subjects: List of subject IDs to include (e.g., ["01", "02"]).
            split: Data split ("train", "val", "test").
            channel_indices: If provided, select only these channel indices.
            max_trials: If provided, limit trials per subject (for calibration budget).
        """
        self.data_dir = Path(data_dir)
        self.subjects = subjects
        self.split = split
        self.channel_indices = channel_indices

        self.eeg_data: torch.Tensor  # (N_total, C, T)
        self.labels: np.ndarray
        self.subject_ids: list[str] = []
        self.trial_indices: list[int] = []

        self._load_data(max_trials)

    def _load_data(self, max_trials: Optional[int] = None) -> None:
        """Load and concatenate data from all subjects into a single tensor."""
        all_eeg: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        for subj in self.subjects:
            subj_dir = self.data_dir / f"sub-{subj}" / self.split
            eeg_path = subj_dir / "eeg.npy"
            labels_path = subj_dir / "labels.npy"

            try:
                eeg = np.load(eeg_path)  # (trials, channels, time)
            except FileNotFoundError:
                logger.warning("Missing data for subject %s at %s", subj, eeg_path)
                continue

            try:
                labels = np.load(labels_path)
            except FileNotFoundError:
                labels = np.zeros(len(eeg), dtype=np.int64)

            if max_trials is not None:
                eeg = eeg[:max_trials]
                labels = labels[:max_trials]

            if self.channel_indices is not None:
                eeg = eeg[:, self.channel_indices, :]

            n_trials = len(eeg)
            all_eeg.append(eeg)
            all_labels.append(labels)
            self.subject_ids.extend([subj] * n_trials)
            self.trial_indices.extend(range(n_trials))

        if all_eeg:
            self.eeg_data = torch.from_numpy(np.concatenate(all_eeg, axis=0)).float()
            self.labels = np.concatenate(all_labels, axis=0).astype(np.int64)
        else:
            self.eeg_data = torch.empty(0)
            self.labels = np.array([], dtype=np.int64)

        logger.info(
            "Loaded %d trials from %d subjects (split=%s)",
            len(self.eeg_data), len(self.subjects), self.split,
        )

    def __len__(self) -> int:
        return len(self.eeg_data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int]:
        """Return a single trial.

        Returns:
            Dict with keys: "eeg" (C, T), "label" (int), "subject" (str),
            "trial_idx" (int).
        """
        return {
            "eeg": self.eeg_data[idx],
            "label": int(self.labels[idx]),
            "subject": self.subject_ids[idx],
            "trial_idx": self.trial_indices[idx],
        }


class RepetitionGroupedDataset(Dataset):
    """Wraps ThingsEEG2Dataset to group trials by stimulus image.

    For LeJEPA view strategy A (trial repetition): returns all repetitions
    of the same image stimulus as a group, enabling use as natural views.
    """

    def __init__(self, base_dataset: ThingsEEG2Dataset):
        """Args:
            base_dataset: Underlying THINGS-EEG2 dataset.
        """
        self.base = base_dataset
        self.groups: dict[int, list[int]] = {}
        for i, label in enumerate(base_dataset.labels):
            self.groups.setdefault(int(label), []).append(i)
        self.group_keys = list(self.groups.keys())

    def __len__(self) -> int:
        return len(self.group_keys)

    def __getitem__(self, idx: int) -> dict:
        """Return all repetitions for a given stimulus.

        Returns:
            Dict with keys: "eeg" (R, C, T) where R=number of repetitions,
            "label" (int), "subject_ids" (list[str]).
        """
        label = self.group_keys[idx]
        trial_indices = self.groups[label]
        trials = self.base.eeg_data[trial_indices]  # (R, C, T) via indexing
        subjects = [self.base.subject_ids[i] for i in trial_indices]
        return {
            "eeg": trials,
            "label": label,
            "subject_ids": subjects,
        }
