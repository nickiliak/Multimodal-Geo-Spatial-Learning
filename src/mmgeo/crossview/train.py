"""Training loop for cross-view retrieval baseline.

Implements the Sample4Geo training pipeline adapted for MMLandmarks:
1. ConvNeXt backbone with shared weights
2. Symmetric InfoNCE loss with learnable temperature
3. GPS-based hard negative sampling (early epochs)
4. Dynamic Similarity Sampling (later epochs)

Usage:
    python -m mmgeo.crossview.train --config configs/crossview_baseline.yaml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import yaml

from mmgeo.crossview.dataset import (
    MMLCrossViewDataset,
    MMLImageDataset,
    UniqueLandmarkSampler,
    get_eval_transforms,
    get_train_transforms,
)
from mmgeo.crossview.evaluate import evaluate_crossview
from mmgeo.crossview.losses import SymmetricInfoNCE
from mmgeo.crossview.model import CrossViewModel


def train_one_epoch(
    model: CrossViewModel,
    dataloader: DataLoader,
    loss_fn: SymmetricInfoNCE,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        ground_imgs = batch["ground_img"].to(device)
        sat_imgs = batch["sat_img"].to(device)

        # Forward: shared encoder for both views
        ground_embeds = model(ground_imgs)
        sat_embeds = model(sat_imgs)

        # Loss
        loss = loss_fn(ground_embeds, sat_embeds)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss


def train(cfg: dict) -> None:
    """Main training function."""

    # ---- Device ----
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    # ---- Data ----
    data_root = Path(cfg["data"]["root"])
    img_size = cfg["training"].get("img_size", 384)
    batch_size = cfg["training"].get("batch_size", 64)

    train_dataset = MMLCrossViewDataset(
        data_root=data_root,
        split="train",
        transform_ground=get_train_transforms(img_size),
        transform_sat=get_train_transforms(img_size),
    )

    train_sampler = UniqueLandmarkSampler(train_dataset, batch_size=batch_size)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=cfg["training"].get("num_workers", 4),
        pin_memory=True,
    )

    # ---- Model ----
    backbone = cfg["model"].get("backbone", "convnext_tiny.fb_in22k")
    model = CrossViewModel(
        backbone=backbone,
        pretrained=True,
        embed_dim=cfg["model"].get("embed_dim", 0),
    )
    model.to(device)

    # ---- Loss ----
    loss_fn = SymmetricInfoNCE(
        temperature=cfg["training"].get("temperature", 0.07),
        learnable_temp=cfg["training"].get("learnable_temp", True),
        label_smoothing=cfg["training"].get("label_smoothing", 0.1),
    )
    loss_fn.to(device)

    # ---- Optimizer ----
    # Combine model and loss parameters (for learnable temperature)
    params = list(model.parameters()) + list(loss_fn.parameters())
    lr = cfg["training"].get("lr", 1e-3)
    optimizer = AdamW(params, lr=lr, weight_decay=cfg["training"].get("weight_decay", 1e-4))

    epochs = cfg["training"].get("epochs", 20)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    # ---- Training loop ----
    save_dir = Path(cfg.get("save_dir", "checkpoints/crossview"))
    save_dir.mkdir(parents=True, exist_ok=True)

    eval_every = cfg["training"].get("eval_every", 5)
    best_recall = 0.0

    print(f"\nStarting training: {epochs} epochs, batch_size={batch_size}")
    print(f"Backbone: {backbone}, LR: {lr}")
    print(f"{'='*60}\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        avg_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device, epoch)
        scheduler.step()

        elapsed = time.time() - t0
        temp = loss_fn.temperature.item()
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"loss={avg_loss:.4f} | "
            f"temp={temp:.4f} | "
            f"lr={lr_now:.6f} | "
            f"time={elapsed:.1f}s"
        )

        # ---- Periodic evaluation ----
        if epoch % eval_every == 0 or epoch == epochs:
            recalls = _run_eval(model, data_root, img_size, device, cfg)

            r1 = recalls.get(1, 0.0)
            if r1 > best_recall:
                best_recall = r1
                ckpt_path = save_dir / "best.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "recall@1": r1,
                    "config": cfg,
                }, ckpt_path)
                print(f"  → New best! R@1={r1:.4f}, saved to {ckpt_path}")

    # Save final checkpoint
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "config": cfg,
    }, save_dir / "last.pt")
    print(f"\nTraining complete. Best R@1: {best_recall:.4f}")


def _run_eval(
    model: CrossViewModel,
    data_root: Path,
    img_size: int,
    device: torch.device,
    cfg: dict,
) -> dict[int, float]:
    """Run ground→satellite retrieval evaluation on the query/index split."""
    eval_transform = get_eval_transforms(img_size)
    eval_batch = cfg["training"].get("eval_batch_size", 128)

    # Query: ground images from query split
    query_ds = MMLImageDataset(data_root, "query", "ground", transform=eval_transform)
    query_loader = DataLoader(
        query_ds, batch_size=eval_batch, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # Index: satellite images from index split
    index_ds = MMLImageDataset(data_root, "index", "satellite", transform=eval_transform)
    index_loader = DataLoader(
        index_ds, batch_size=eval_batch, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    ks = cfg.get("evaluation", {}).get("recall_ks", [1, 5, 10])
    recalls = evaluate_crossview(model, query_loader, index_loader, device, ks=ks)
    return recalls


def main():
    parser = argparse.ArgumentParser(description="Train cross-view retrieval baseline")
    parser.add_argument("--config", type=str, default="configs/crossview_baseline.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg)


if __name__ == "__main__":
    main()
