"""LeJEPA-EEG model wrapper connecting ENIGMA's encoder to LeJEPA loss.

Wraps the ENIGMA Spatio_Temporal_CNN encoder to work with LeJEPA's
distribution-matching self-supervised loss.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange

from lejepa.multivariate import SlicingUnivariateTest
from lejepa.univariate import EppsPulley

logger = logging.getLogger(__name__)


class SpatioTemporalCNN(nn.Module):
    """ENIGMA's spatio-temporal CNN encoder.

    Re-implemented to avoid coupling with ENIGMA's subject layers and projector.
    Architecture matches upstream/enigma/source/models.py::Spatio_Temporal_CNN
    including the Rearrange in projection for compatible embedding layout.
    """

    def __init__(
        self,
        num_channels: int = 63,
        sequence_length: int = 275,
        emb_size: int = 4,
        conv1_kernel: tuple[int, int] = (1, 5),
        pool1_kernel: tuple[int, int] = (1, 17),
        pool1_stride: tuple[int, int] = (1, 5),
    ):
        """Args:
            num_channels: Number of EEG channels.
            sequence_length: Temporal length of input (unused, kept for reference).
            emb_size: Embedding size per spatial position after projection.
            conv1_kernel: Temporal convolution kernel.
            pool1_kernel: Temporal pooling kernel.
            pool1_stride: Temporal pooling stride.
        """
        super().__init__()
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, conv1_kernel, stride=(1, 1)),
            nn.AvgPool2d(pool1_kernel, pool1_stride),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (num_channels, 1), stride=(1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )
        self.projection = nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: EEG input, shape (B, C, T).

        Returns:
            Flat embedding, shape (B, D) where D = T' * emb_size.
            Layout matches upstream ENIGMA: time-first via rearrange.
        """
        x = x.unsqueeze(1)  # (B, 1, C, T)
        x = self.tsconv(x)  # (B, 40, 1, T')
        x = self.projection(x)  # (B, emb_size, 1, T')
        # Match upstream Rearrange("b e (h) (w) -> b (h w) e") layout
        x = rearrange(x, "b e h w -> b (h w) e")  # (B, T', emb_size)
        x = x.contiguous().view(x.size(0), -1)  # (B, T'*emb_size)
        return x


class EEGLeJEPA(nn.Module):
    """LeJEPA wrapper for EEG self-supervised pretraining.

    Combines ENIGMA's spatio-temporal CNN encoder with LeJEPA's loss:
      loss = lambda * SIGReg(proj) + (1 - lambda) * invariance(proj)

    Where SIGReg is the Epps-Pulley Gaussianity test (via sliced projections)
    and invariance is the prediction/invariance term from the LeJEPA paper.
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_slices: int = 1024,
        num_points: int = 17,
        lambda_sigreg: float = 0.05,
    ):
        """Args:
            encoder: EEG encoder module (forward: (B,C,T) -> (B,D)).
            num_slices: Number of random projections for sliced test.
            num_points: Quadrature points for Epps-Pulley test.
            lambda_sigreg: Weight of SIGReg (lambda in the paper).
        """
        super().__init__()
        self.encoder = encoder
        self.lambda_sigreg = lambda_sigreg

        univariate_test = EppsPulley(n_points=num_points)
        self.multivariate_test = SlicingUnivariateTest(
            univariate_test, num_slices=num_slices
        )

    def forward(
        self,
        views: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute LeJEPA loss across views.

        Following MINIMAL.md from upstream LeJEPA:
          inv_loss = (proj.mean(0) - proj).square().mean()
          sigreg_loss = sigreg(proj)
          lejepa_loss = sigreg_loss * lambda + inv_loss * (1 - lambda)

        Args:
            views: List of EEG view tensors, each shape (B, C, T).
                For CNN encoders, all views are treated as global views.

        Returns:
            Dict with "loss", "inv_loss", "sigreg_loss", "embeddings".
        """
        embeddings = [self.encoder(v) for v in views]

        # Stack views: (num_views, B, D) — matches upstream's (V, B, proj_dim)
        proj = torch.stack(embeddings, dim=0)  # (V, B, D)

        # Invariance/prediction loss: embeddings should be similar across views
        inv_loss = (proj.mean(0) - proj).square().mean()

        # SIGReg: Epps-Pulley Gaussianity test on concatenated embeddings
        all_emb = torch.cat(embeddings, dim=0)  # (V*B, D)
        sigreg_loss = self.multivariate_test(all_emb)

        total_loss = self.lambda_sigreg * sigreg_loss + (1 - self.lambda_sigreg) * inv_loss

        return {
            "loss": total_loss,
            "inv_loss": inv_loss,
            "sigreg_loss": sigreg_loss,
            "embeddings": embeddings,
        }


def build_lejepa_model(
    num_channels: int = 63,
    sequence_length: int = 275,
    emb_size: int = 4,
    num_slices: int = 1024,
    num_points: int = 17,
    lambda_sigreg: float = 0.05,
    pretrained_path: Optional[str] = None,
    device: str | torch.device = "cpu",
) -> EEGLeJEPA:
    """Factory function to build a LeJEPA-EEG model.

    Args:
        num_channels: Number of EEG channels.
        sequence_length: Temporal length of input.
        emb_size: Embedding size per spatial position.
        num_slices: Number of random projections.
        num_points: Quadrature points for Epps-Pulley.
        lambda_sigreg: SIGReg weight (lambda in the paper).
        pretrained_path: Optional path to pretrained encoder checkpoint.
        device: Device to place model on.

    Returns:
        Configured EEGLeJEPA model.
    """
    encoder = SpatioTemporalCNN(
        num_channels=num_channels,
        sequence_length=sequence_length,
        emb_size=emb_size,
    )

    if pretrained_path is not None:
        from src.utils import load_encoder_weights
        load_encoder_weights(encoder, pretrained_path, device=device)

    model = EEGLeJEPA(
        encoder=encoder,
        num_slices=num_slices,
        num_points=num_points,
        lambda_sigreg=lambda_sigreg,
    )
    return model.to(device)
