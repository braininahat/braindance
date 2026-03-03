"""Shared utilities: checkpointing, logging, metrics, embedding diagnostics."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def setup_logging(log_dir: str | Path, name: str = "braindance") -> logging.Logger:
    """Configure logging to file and console.

    Args:
        log_dir: Directory for log files.
        name: Logger name.

    Returns:
        Configured logger.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log = logging.getLogger(name)
    log.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    log.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_dir / f"{name}.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    log.addHandler(fh)

    return log


def save_checkpoint(
    path: str | Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
    metrics: Optional[dict] = None,
) -> None:
    """Save a training checkpoint.

    Args:
        path: File path for the checkpoint.
        epoch: Current epoch number.
        model: Model to save.
        optimizer: Optimizer state.
        scheduler: Optional LR scheduler state.
        metrics: Optional dict of metric values to record.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    if metrics is not None:
        state["metrics"] = metrics
    torch.save(state, path)
    logger.info("Saved checkpoint to %s (epoch %d)", path, epoch)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str | torch.device = "cpu",
) -> dict:
    """Load a training checkpoint.

    Args:
        path: Path to checkpoint file.
        model: Model to load weights into.
        optimizer: Optional optimizer to restore state.
        scheduler: Optional scheduler to restore state.
        device: Device to load tensors to.

    Returns:
        Checkpoint dict (includes epoch, metrics, etc.).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    logger.info("Loaded checkpoint from %s (epoch %d)", path, ckpt.get("epoch", -1))
    return ckpt


def get_device(device_str: str = "cuda") -> torch.device:
    """Get torch device, falling back to CPU if CUDA unavailable.

    Args:
        device_str: Requested device ("cuda" or "cpu").

    Returns:
        torch.device instance.
    """
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable, falling back to CPU")
        return torch.device("cpu")
    return torch.device(device_str)


def embedding_diagnostics(embeddings: torch.Tensor) -> dict[str, float]:
    """Compute diagnostics on embedding distributions for Gaussianity checks.

    Useful for monitoring LeJEPA's SIGReg term — embeddings should
    approximate a Gaussian distribution for optimal performance.

    Args:
        embeddings: Embedding tensor, shape (N, D).

    Returns:
        Dict with per-dimension statistics aggregated:
        mean_kurtosis, mean_skewness, mean_std, collapse_ratio.
    """
    # Per-dimension statistics
    mean = embeddings.mean(dim=0)
    centered = embeddings - mean
    std = centered.std(dim=0)

    # Kurtosis (excess): 0 for Gaussian
    m4 = (centered**4).mean(dim=0)
    kurtosis = m4 / (std**4 + 1e-8) - 3.0

    # Skewness: 0 for Gaussian
    m3 = (centered**3).mean(dim=0)
    skewness = m3 / (std**3 + 1e-8)

    # Collapse ratio: fraction of dimensions with very low variance
    collapse_threshold = 0.01
    collapse_ratio = (std < collapse_threshold).float().mean().item()

    return {
        "mean_kurtosis": kurtosis.mean().item(),
        "mean_skewness": skewness.abs().mean().item(),
        "mean_std": std.mean().item(),
        "collapse_ratio": collapse_ratio,
    }


def load_encoder_weights(
    encoder: nn.Module,
    path: str | Path,
    device: str | torch.device = "cpu",
    prefix: str = "encoder.",
) -> None:
    """Load encoder weights from a checkpoint, stripping the prefix.

    Args:
        encoder: Encoder module to load weights into.
        path: Path to checkpoint file.
        device: Device to map tensors to.
        prefix: Key prefix to strip (e.g., "encoder.").
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    encoder_state = {
        k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)
    }
    if not encoder_state:
        logger.warning("No encoder weights found under prefix '%s' in %s", prefix, path)
        return
    encoder.load_state_dict(encoder_state)
    logger.info("Loaded pretrained encoder from %s", path)


def step_lr_with_warmup(
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    warmup_epochs: int,
    base_lr: float,
) -> None:
    """Step LR scheduler with linear warmup.

    Args:
        optimizer: Optimizer whose LR to adjust.
        scheduler: LR scheduler to step after warmup.
        epoch: Current epoch (0-indexed).
        warmup_epochs: Number of warmup epochs.
        base_lr: Target learning rate at end of warmup.
    """
    if epoch < warmup_epochs:
        warmup_factor = (epoch + 1) / warmup_epochs
        for pg in optimizer.param_groups:
            pg["lr"] = base_lr * warmup_factor
    else:
        scheduler.step()


class AverageMeter:
    """Running average tracker for training metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)
