"""LeJEPA pretraining script for EEG encoder.

Subject-pooled self-supervised pretraining using LeJEPA's distribution-matching
loss (Epps-Pulley Gaussianity test + SIGReg regularization).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import wandb
import yaml
from torch.utils.data import DataLoader

from src.data import ThingsEEG2Dataset
from src.lejepa_eeg import build_lejepa_model
from src.utils import (
    AverageMeter,
    embedding_diagnostics,
    get_device,
    load_checkpoint,
    save_checkpoint,
    setup_logging,
    step_lr_with_warmup,
)
from src.views import build_view_strategy

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LeJEPA EEG pretraining")
    parser.add_argument("--config", type=str, default="configs/lejepa_default.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    return parser.parse_args()


def collate_views(batch: list[dict], view_strategy) -> list[torch.Tensor]:
    """Collate a batch of EEG trials into view pairs.

    Args:
        batch: List of dataset items with "eeg" key (C, T).
        view_strategy: ViewStrategy instance.

    Returns:
        List of view tensors, each (B, C', T').
    """
    all_views: list[list[torch.Tensor]] = []
    for item in batch:
        views = view_strategy(item["eeg"])
        all_views.append(views)

    num_views = len(all_views[0])
    batched_views = []
    for v_idx in range(num_views):
        batched_views.append(torch.stack([all_views[b][v_idx] for b in range(len(batch))]))
    return batched_views


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    view_strategy,
    device: torch.device,
    grad_clip: float = 1.0,
) -> dict[str, float]:
    """Train for one epoch.

    Args:
        model: EEGLeJEPA model.
        dataloader: Training data loader.
        optimizer: Optimizer.
        view_strategy: View generation strategy.
        device: Training device.
        grad_clip: Gradient clipping value.

    Returns:
        Dict of average metrics for the epoch.
    """
    model.train()
    meters = {k: AverageMeter() for k in ["loss", "inv_loss", "sigreg_loss"]}

    for batch in dataloader:
        views = collate_views(batch, view_strategy)
        views = [v.to(device) for v in views]

        output = model(views)
        loss = output["loss"]

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        B = views[0].size(0)
        for k in meters:
            meters[k].update(output[k].item(), B)

    return {k: m.avg for k, m in meters.items()}


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    log = setup_logging(cfg["infra"]["log_dir"], "lejepa_pretrain")
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

    # Build view strategy
    view_cfg = cfg["views"]
    jitter_samples = int(view_cfg.get("temporal_jitter_ms", 50) * cfg["data"]["sample_rate"] / 1000)
    view_strategy = build_view_strategy(
        name=view_cfg["strategy"],
        jitter_samples=jitter_samples,
        subset_size=view_cfg.get("channel_subset_size") or cfg["data"]["num_channels"],
    )
    log.info("View strategy: %s", view_cfg["strategy"])

    # Build dataset
    # THINGS-EEG2 subjects (10 subjects, IDs 01-10)
    subjects = [f"{i:02d}" for i in range(1, 11)]
    dataset = ThingsEEG2Dataset(
        data_dir=cfg["data"]["data_dir"],
        subjects=subjects,
        split="train",
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["infra"]["num_workers"],
        drop_last=True,
        collate_fn=lambda batch: batch,  # Custom collation in collate_views
    )

    # Build model
    model = build_lejepa_model(
        num_channels=cfg["data"]["num_channels"],
        sequence_length=cfg["data"]["sequence_length"],
        emb_size=cfg["model"]["encoder"]["emb_size"],
        num_slices=cfg["lejepa"]["num_slices"],
        num_points=cfg["lejepa"]["n_points"],
        lambda_sigreg=cfg["lejepa"]["lambda_sigreg"],
        device=device,
    )
    log.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))

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
        log.info("Resumed from epoch %d", start_epoch)

    # Training loop
    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        metrics = train_one_epoch(
            model, dataloader, optimizer, view_strategy, device,
            grad_clip=cfg["training"]["grad_clip"],
        )

        step_lr_with_warmup(
            optimizer, scheduler, epoch,
            cfg["training"]["warmup_epochs"],
            cfg["training"]["learning_rate"],
        )

        lr = optimizer.param_groups[0]["lr"]
        log.info(
            "Epoch %d/%d — loss: %.4f inv: %.4f sigreg: %.4f lr: %.2e",
            epoch + 1, cfg["training"]["epochs"],
            metrics["loss"], metrics["inv_loss"], metrics["sigreg_loss"], lr,
        )
        if use_wandb:
            wandb.log({
                "train/loss": metrics["loss"],
                "train/inv_loss": metrics["inv_loss"],
                "train/sigreg_loss": metrics["sigreg_loss"],
                "lr": lr,
                "epoch": epoch + 1,
            })

        # Embedding diagnostics every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                sample_batch = next(iter(dataloader))
                sample_views = collate_views(sample_batch, view_strategy)
                sample_views = [v.to(device) for v in sample_views]
                out = model(sample_views)
                diag = embedding_diagnostics(out["embeddings"][0])
                log.info("Embedding diagnostics: %s", diag)
                if use_wandb:
                    wandb.log({f"diag/{k}": v for k, v in diag.items()})
            model.train()

        # Save checkpoint
        if (epoch + 1) % cfg["infra"]["save_every"] == 0:
            save_checkpoint(
                Path(cfg["infra"]["checkpoint_dir"]) / f"epoch_{epoch+1:04d}.pt",
                epoch, model, optimizer, scheduler, metrics,
            )

    # Save final checkpoint
    save_checkpoint(
        Path(cfg["infra"]["checkpoint_dir"]) / "final.pt",
        cfg["training"]["epochs"] - 1, model, optimizer, scheduler, metrics,
    )
    if use_wandb:
        wandb.finish()
    log.info("Training complete.")


if __name__ == "__main__":
    main()
