"""Evaluation metrics for neural image reconstruction from EEG.

Implements standard metrics from the brain decoding literature:
PixCorr, SSIM, AlexNet(2/5), InceptionV3, CLIP, EfficientNet-B, SwAV-ResNet50,
and 2-way forced-choice comparisons.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def pixel_correlation(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Pixel-level Pearson correlation between predicted and target images.

    Args:
        pred: Predicted images, shape (N, C, H, W).
        target: Target images, shape (N, C, H, W).

    Returns:
        Mean pixel correlation across samples.
    """
    pred_flat = pred.reshape(pred.size(0), -1)
    target_flat = target.reshape(target.size(0), -1)

    pred_centered = pred_flat - pred_flat.mean(dim=1, keepdim=True)
    target_centered = target_flat - target_flat.mean(dim=1, keepdim=True)

    num = (pred_centered * target_centered).sum(dim=1)
    den = pred_centered.norm(dim=1) * target_centered.norm(dim=1) + 1e-8

    return (num / den).mean().item()


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    C1: float = 0.01**2,
    C2: float = 0.03**2,
) -> float:
    """Structural Similarity Index (SSIM).

    Simplified implementation using uniform window.

    Args:
        pred: Predicted images, shape (N, C, H, W), range [0, 1].
        target: Target images, shape (N, C, H, W), range [0, 1].
        window_size: Size of averaging window.
        C1: Stability constant for luminance.
        C2: Stability constant for contrast.

    Returns:
        Mean SSIM across samples.
    """
    pad = window_size // 2
    mu_pred = F.avg_pool2d(pred, window_size, stride=1, padding=pad)
    mu_target = F.avg_pool2d(target, window_size, stride=1, padding=pad)

    mu_pred_sq = mu_pred**2
    mu_target_sq = mu_target**2
    mu_cross = mu_pred * mu_target

    sigma_pred_sq = F.avg_pool2d(pred**2, window_size, stride=1, padding=pad) - mu_pred_sq
    sigma_target_sq = F.avg_pool2d(target**2, window_size, stride=1, padding=pad) - mu_target_sq
    sigma_cross = F.avg_pool2d(pred * target, window_size, stride=1, padding=pad) - mu_cross

    ssim_map = ((2 * mu_cross + C1) * (2 * sigma_cross + C2)) / (
        (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2)
    )
    return ssim_map.mean().item()


class FeatureExtractor(nn.Module):
    """Extract intermediate features from a pretrained model at specified layers."""

    def __init__(self, model: nn.Module, layer_names: list[str]):
        super().__init__()
        self.model = model
        self.layer_names = layer_names
        self._features: dict[str, torch.Tensor] = {}

        for name, module in self.model.named_modules():
            if name in layer_names:
                module.register_forward_hook(self._hook(name))

    def _hook(self, name: str):
        def fn(module, input, output):
            self._features[name] = output
        return fn

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self._features = {}
        self.model(x)
        return self._features


def feature_similarity(
    pred_features: torch.Tensor,
    target_features: torch.Tensor,
) -> float:
    """Cosine similarity between feature vectors, averaged across samples.

    Args:
        pred_features: Predicted features, shape (N, D).
        target_features: Target features, shape (N, D).

    Returns:
        Mean cosine similarity.
    """
    pred_norm = F.normalize(pred_features.flatten(1), dim=1)
    target_norm = F.normalize(target_features.flatten(1), dim=1)
    return (pred_norm * target_norm).sum(dim=1).mean().item()


def two_way_identification(
    pred_features: torch.Tensor,
    target_features: torch.Tensor,
) -> float:
    """2-way forced-choice identification accuracy.

    For each sample i, check if sim(pred_i, target_i) > sim(pred_i, target_j)
    for all j != i. Reports the fraction of pairwise comparisons won,
    matching the upstream ENIGMA evaluation protocol.

    Args:
        pred_features: Predicted features, shape (N, D).
        target_features: Target features, shape (N, D).

    Returns:
        Accuracy (fraction of pairwise comparisons correct).
    """
    N = pred_features.size(0)
    pred_norm = F.normalize(pred_features.flatten(1), dim=1)
    target_norm = F.normalize(target_features.flatten(1), dim=1)

    sim = pred_norm @ target_norm.T  # (N, N)
    diag = sim.diag().unsqueeze(1)  # (N, 1)
    # For each row i, count how many off-diagonal entries are less than sim[i,i]
    mask = ~torch.eye(N, dtype=torch.bool, device=sim.device)
    wins = (diag > sim).masked_select(mask).float().mean().item()
    return wins


@torch.no_grad()
def evaluate_reconstruction(
    pred_images: torch.Tensor,
    target_images: torch.Tensor,
    pred_clip_emb: Optional[torch.Tensor] = None,
    target_clip_emb: Optional[torch.Tensor] = None,
) -> dict[str, float]:
    """Compute all evaluation metrics for image reconstruction.

    Args:
        pred_images: Predicted images, shape (N, 3, H, W), range [0, 1].
        target_images: Target images, shape (N, 3, H, W), range [0, 1].
        pred_clip_emb: Optional precomputed CLIP embeddings for predictions.
        target_clip_emb: Optional precomputed CLIP embeddings for targets.

    Returns:
        Dict of metric name to value.
    """
    metrics = {}

    # Low-level metrics
    metrics["pixcorr"] = pixel_correlation(pred_images, target_images)
    metrics["ssim"] = ssim(pred_images, target_images)

    # CLIP metrics (if embeddings provided)
    if pred_clip_emb is not None and target_clip_emb is not None:
        metrics["clip_sim"] = feature_similarity(pred_clip_emb, target_clip_emb)
        metrics["clip_2way"] = two_way_identification(pred_clip_emb, target_clip_emb)

    logger.info("Evaluation metrics: %s", metrics)
    return metrics
