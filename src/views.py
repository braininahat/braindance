"""EEG view generation strategies for LeJEPA pretraining.

Each strategy produces multiple views of the same EEG trial for
self-supervised contrastive/distribution-matching learning.
"""

from __future__ import annotations

import inspect
import logging
from abc import ABC, abstractmethod

import torch

logger = logging.getLogger(__name__)


class ViewStrategy(ABC):
    """Base class for EEG view generation strategies."""

    @abstractmethod
    def __call__(self, eeg: torch.Tensor, **kwargs) -> list[torch.Tensor]:
        """Generate views from an EEG trial.

        Args:
            eeg: Single EEG trial, shape (C, T).

        Returns:
            List of view tensors, each (C', T') depending on strategy.
        """
        ...


class TrialRepetitionViews(ViewStrategy):
    """Strategy A: Use repeated presentations of the same image as views.

    THINGS-EEG2 shows each image multiple times. Different repetitions
    of the same stimulus serve as natural augmentations.
    """

    def __call__(self, eeg: torch.Tensor, **kwargs) -> list[torch.Tensor]:
        """Identity transform — repetition grouping handled by the dataset."""
        return [eeg]


class TemporalJitterViews(ViewStrategy):
    """Strategy B: Generate views via temporal jitter (random crop in time).

    Shifts the analysis window by a random offset within +/-jitter_samples,
    producing slightly different temporal views of the same event.
    """

    def __init__(self, jitter_samples: int = 12, num_views: int = 2):
        """Args:
            jitter_samples: Max jitter in samples (e.g., 12 @ 250Hz ~ 50ms).
            num_views: Number of jittered views to generate.
        """
        self.jitter_samples = jitter_samples
        self.num_views = num_views

    def __call__(self, eeg: torch.Tensor, **kwargs) -> list[torch.Tensor]:
        """Apply temporal jitter to produce multiple views.

        Args:
            eeg: EEG trial, shape (C, T). Must have T > 2*jitter_samples.

        Returns:
            List of num_views tensors, each shape (C, T - 2*jitter_samples).
        """
        C, T = eeg.shape
        margin = self.jitter_samples
        crop_len = T - 2 * margin
        views = []
        for _ in range(self.num_views):
            offset = torch.randint(0, 2 * margin + 1, (1,)).item()
            views.append(eeg[:, offset : offset + crop_len])
        return views


class ChannelSubsetViews(ViewStrategy):
    """Strategy C: Generate views by sampling different channel subsets.

    Each view uses a random subset of EEG channels, simulating
    reduced-channel setups and providing spatial augmentation.
    """

    def __init__(self, subset_size: int = 16, num_views: int = 2):
        """Args:
            subset_size: Number of channels per view.
            num_views: Number of channel-subset views to generate.
        """
        self.subset_size = subset_size
        self.num_views = num_views

    def __call__(self, eeg: torch.Tensor, **kwargs) -> list[torch.Tensor]:
        """Sample random channel subsets.

        Args:
            eeg: EEG trial, shape (C, T).

        Returns:
            List of num_views tensors, each shape (subset_size, T).
        """
        C, T = eeg.shape
        views = []
        for _ in range(self.num_views):
            idx = torch.randperm(C)[: self.subset_size].sort().values
            views.append(eeg[idx])
        return views


class CombinedABCViews(ViewStrategy):
    """Strategy E: Combine trial repetition + temporal jitter + channel subsets."""

    def __init__(
        self,
        jitter_samples: int = 12,
        subset_size: int = 16,
        num_views: int = 2,
    ):
        self.jitter = TemporalJitterViews(jitter_samples, num_views)
        self.subset_size = subset_size

    def __call__(self, eeg: torch.Tensor, **kwargs) -> list[torch.Tensor]:
        """Apply temporal jitter then channel subsampling.

        Args:
            eeg: EEG trial, shape (C, T).

        Returns:
            List of num_views tensors, each shape (subset_size, T').
        """
        jittered = self.jitter(eeg)
        C = eeg.size(0)
        return [
            v[torch.randperm(C)[: self.subset_size].sort().values]
            for v in jittered
        ]


STRATEGY_REGISTRY: dict[str, type[ViewStrategy]] = {
    "trial_repetition": TrialRepetitionViews,
    "temporal_jitter": TemporalJitterViews,
    "channel_subset": ChannelSubsetViews,
    "combined_ab": TemporalJitterViews,  # Same as temporal jitter; repetition grouping is in dataset
    "combined_abc": CombinedABCViews,
}


def build_view_strategy(
    name: str,
    jitter_samples: int = 12,
    subset_size: int = 16,
    num_views: int = 2,
) -> ViewStrategy:
    """Factory function to create a view strategy by name.

    Args:
        name: Strategy name (see STRATEGY_REGISTRY).
        jitter_samples: Temporal jitter in samples.
        subset_size: Channel subset size.
        num_views: Number of views to generate.

    Returns:
        Configured ViewStrategy instance.
    """
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown view strategy '{name}'. Choose from: {list(STRATEGY_REGISTRY)}")

    all_kwargs = dict(jitter_samples=jitter_samples, subset_size=subset_size, num_views=num_views)
    sig = inspect.signature(cls.__init__)
    valid_params = {k for k in sig.parameters if k != "self"}
    return cls(**{k: v for k, v in all_kwargs.items() if k in valid_params})
