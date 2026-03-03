"""ENIGMA fine-tuning script.

Fine-tunes the spatio-temporal CNN encoder (optionally from LeJEPA pretrained
checkpoint) with a projector head targeting CLIP ViT-H/14 embeddings.
Supports calibration budget experiments (varying training data per subject).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
import yaml
from torch.utils.data import DataLoader

from src.data import ThingsEEG2Dataset
from src.lejepa_eeg import SpatioTemporalCNN
from src.utils import (
    AverageMeter,
    get_device,
    load_checkpoint,
    load_encoder_weights,
    save_checkpoint,
    setup_logging,
    step_lr_with_warmup,
)

logger = logging.getLogger(__name__)


class ResidualAdd(nn.Module):
    """Residual connection wrapper matching upstream ENIGMA."""

    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class MLPProjector(nn.Sequential):
    """EEG -> CLIP projection head.

    Matches upstream/enigma/source/models.py::MLP_Projector exactly:
    Linear(emb->proj) -> ResidualAdd(GELU -> Linear(proj->proj) -> Dropout) -> LayerNorm
    """

    def __init__(self, embedding_dim: int = 184, proj_dim: int = 1024, drop_proj: float = 0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(proj_dim, proj_dim),
                    nn.Dropout(drop_proj),
                )
            ),
            nn.LayerNorm(proj_dim),
        )


class ENIGMAFinetune(nn.Module):
    """ENIGMA model for fine-tuning: encoder + subject alignment + projector."""

    def __init__(
        self,
        encoder: SpatioTemporalCNN,
        subjects: list[str],
        sequence_length: int = 275,
        embedding_dim: int = 184,
        proj_dim: int = 1024,
        drop_proj: float = 0.5,
    ):
        super().__init__()
        self.encoder = encoder
        self.subject_linear = nn.ModuleDict({
            subj: nn.Linear(sequence_length, sequence_length) for subj in subjects
        })
        self.projector = MLPProjector(embedding_dim, proj_dim, drop_proj)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(
        self, eeg: torch.Tensor, subject_ids: list[str]
    ) -> torch.Tensor:
        """Forward pass with grouped subject-wise alignment.

        Args:
            eeg: EEG input, shape (B, C, T).
            subject_ids: List of subject IDs for each sample in batch.

        Returns:
            CLIP-aligned projections, shape (B, proj_dim).
        """
        # Group by subject for efficient batched linear
        x = torch.empty_like(eeg)
        groups: dict[str, list[int]] = {}
        for i, subj in enumerate(subject_ids):
            groups.setdefault(subj, []).append(i)
        for subj, indices in groups.items():
            idx = torch.tensor(indices, device=eeg.device)
            x[indices] = self.subject_linear[subj](eeg[idx])

        # Encode and project
        x = self.encoder(x)
        x = self.projector(x)
        return x


def infonce_loss(
    eeg_proj: torch.Tensor,
    clip_proj: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """InfoNCE contrastive loss between EEG and CLIP embeddings.

    Args:
        eeg_proj: EEG projections, shape (B, D).
        clip_proj: CLIP projections, shape (B, D).
        logit_scale: Learnable temperature parameter.

    Returns:
        Scalar InfoNCE loss.
    """
    eeg_proj = nn.functional.normalize(eeg_proj, dim=-1)
    clip_proj = nn.functional.normalize(clip_proj, dim=-1)

    logits = logit_scale.exp() * eeg_proj @ clip_proj.T
    labels = torch.arange(len(logits), device=logits.device)
    loss_i = nn.functional.cross_entropy(logits, labels)
    loss_t = nn.functional.cross_entropy(logits.T, labels)
    return (loss_i + loss_t) / 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ENIGMA fine-tuning")
    parser.add_argument("--config", type=str, default="configs/enigma_finetune.yaml")
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    log = setup_logging(cfg["infra"]["log_dir"], "enigma_finetune")
    device = get_device(cfg["infra"]["device"])
    torch.manual_seed(cfg["infra"]["seed"])

    log.info("Config: %s", args.config)
    log.info("Device: %s", device)

    # Wandb
    try:
        wandb.init(project=cfg["infra"]["wandb_project"], config=cfg)
        use_wandb = True
    except Exception as e:
        log.warning("Wandb init failed: %s. Continuing without wandb.", e)
        use_wandb = False

    # Calibration budget: convert minutes to trial count
    budget_minutes = cfg["calibration"]["budget_minutes"]
    trials_per_min = cfg["calibration"]["trials_per_minute"]
    max_trials = int(budget_minutes * trials_per_min)
    log.info("Calibration budget: %d min -> %d trials/subject", budget_minutes, max_trials)

    # Build dataset
    subjects = [f"{i:02d}" for i in range(1, 11)]

    train_dataset = ThingsEEG2Dataset(
        data_dir=cfg["data"]["data_dir"],
        subjects=subjects,
        split="train",
        max_trials=max_trials,
    )
    val_dataset = ThingsEEG2Dataset(
        data_dir=cfg["data"]["data_dir"],
        subjects=subjects,
        split="val",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["infra"]["num_workers"],
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["infra"]["num_workers"],
    )

    # Build encoder (optionally from pretrained LeJEPA checkpoint)
    encoder = SpatioTemporalCNN(
        num_channels=cfg["data"]["num_channels"],
        sequence_length=cfg["data"]["sequence_length"],
        emb_size=cfg["model"]["encoder"]["emb_size"],
    )

    pretrained_path = cfg.get("pretrained_checkpoint")
    if pretrained_path:
        load_encoder_weights(encoder, pretrained_path, device=device)

    # Compute embedding dim from a dummy forward pass
    with torch.no_grad():
        dummy = torch.randn(1, cfg["data"]["num_channels"], cfg["data"]["sequence_length"])
        emb_dim = encoder(dummy).shape[1]
    log.info("Encoder embedding dim: %d", emb_dim)

    model = ENIGMAFinetune(
        encoder=encoder,
        subjects=subjects,
        sequence_length=cfg["data"]["sequence_length"],
        embedding_dim=emb_dim,
        proj_dim=cfg["model"]["proj_dim"],
        drop_proj=cfg["model"]["drop_proj"],
    ).to(device)
    log.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))

    # Load CLIP model for targets
    try:
        import open_clip
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            cfg["clip"]["model_name"], pretrained=cfg["clip"]["pretrained"]
        )
        clip_model = clip_model.to(device).eval()
        log.info("Loaded CLIP model: %s", cfg["clip"]["model_name"])
    except Exception as e:
        log.warning("Could not load CLIP model: %s. Using random targets.", e)
        clip_model = None

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["epochs"] - cfg["training"]["warmup_epochs"],
    )

    start_epoch = 0
    if args.resume:
        ckpt = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        start_epoch = ckpt["epoch"] + 1

    mse_weight = cfg["loss"]["mse_weight"]
    infonce_weight = cfg["loss"]["infonce_weight"]

    # Training loop
    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        model.train()
        meters = {k: AverageMeter() for k in ["loss", "mse", "infonce"]}

        for batch in train_loader:
            eeg = batch["eeg"].to(device)
            subject_ids = batch["subject"]

            eeg_proj = model(eeg, subject_ids)

            # TODO: Load precomputed CLIP embeddings from image features.
            # Currently using random targets as placeholder.
            clip_targets = torch.randn_like(eeg_proj)
            clip_targets = nn.functional.normalize(clip_targets, dim=-1)

            mse_loss = nn.functional.mse_loss(eeg_proj, clip_targets)
            nce_loss = infonce_loss(eeg_proj, clip_targets, model.logit_scale)
            loss = mse_weight * mse_loss + infonce_weight * nce_loss

            optimizer.zero_grad()
            loss.backward()
            if cfg["training"]["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
            optimizer.step()

            B = eeg.size(0)
            meters["loss"].update(loss.item(), B)
            meters["mse"].update(mse_loss.item(), B)
            meters["infonce"].update(nce_loss.item(), B)

        step_lr_with_warmup(
            optimizer, scheduler, epoch,
            cfg["training"]["warmup_epochs"],
            cfg["training"]["learning_rate"],
        )

        lr = optimizer.param_groups[0]["lr"]
        log.info(
            "Epoch %d/%d — loss: %.4f mse: %.4f infonce: %.4f lr: %.2e",
            epoch + 1, cfg["training"]["epochs"],
            meters["loss"].avg, meters["mse"].avg, meters["infonce"].avg, lr,
        )
        if use_wandb:
            wandb.log({
                "train/loss": meters["loss"].avg,
                "train/mse": meters["mse"].avg,
                "train/infonce": meters["infonce"].avg,
                "lr": lr,
                "epoch": epoch + 1,
            })

        # Save checkpoint
        if (epoch + 1) % cfg["infra"]["save_every"] == 0:
            save_checkpoint(
                Path(cfg["infra"]["checkpoint_dir"]) / f"epoch_{epoch+1:04d}.pt",
                epoch, model, optimizer, scheduler,
                {k: m.avg for k, m in meters.items()},
            )

    # Save final
    save_checkpoint(
        Path(cfg["infra"]["checkpoint_dir"]) / "final.pt",
        cfg["training"]["epochs"] - 1, model, optimizer, scheduler,
        {k: m.avg for k, m in meters.items()},
    )
    if use_wandb:
        wandb.finish()
    log.info("Training complete.")


if __name__ == "__main__":
    main()
