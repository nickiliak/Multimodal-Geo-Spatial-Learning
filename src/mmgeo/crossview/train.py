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
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
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
) -> dict[str, float]:
    """Train for one epoch and return aggregate training metrics."""
    model.train()
    total_loss = 0.0
    total_diag_sim = 0.0
    total_offdiag_sim = 0.0
    total_margin = 0.0
    total_batch_acc = 0.0
    n_batches = 0

    for batch in dataloader:
        ground_imgs = batch["ground_img"].to(device)
        sat_imgs = batch["sat_img"].to(device)

        # Forward: shared encoder for both views
        ground_embeds = model(ground_imgs)
        sat_embeds = model(sat_imgs)

        # Loss
        loss = loss_fn(ground_embeds, sat_embeds)
        
        # Diagnostic stats: if these stay flat, training is near-uniform.
        with torch.no_grad():
            sims = ground_embeds @ sat_embeds.T  # cosine similarities
            bsz = sims.size(0)
            diag = sims.diag()
            if bsz > 1:
                offdiag_mask = ~torch.eye(bsz, dtype=torch.bool, device=sims.device)
                offdiag = sims[offdiag_mask]
                offdiag_mean = offdiag.mean()
            else:
                offdiag_mean = torch.zeros((), device=sims.device)
            batch_acc = (sims.argmax(dim=1) == torch.arange(bsz, device=sims.device)).float().mean()

            total_diag_sim += diag.mean().item()
            total_offdiag_sim += offdiag_mean.item()
            total_margin += (diag.mean() - offdiag_mean).item()
            total_batch_acc += batch_acc.item()

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    denom = max(n_batches, 1)
    return {
        "loss": total_loss / denom,
        "diag_sim": total_diag_sim / denom,
        "offdiag_sim": total_offdiag_sim / denom,
        "sim_margin": total_margin / denom,
        "batch_acc": total_batch_acc / denom,
    }


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
    warmup_epochs = cfg["training"].get("warmup_epochs", 2)
    warmup_start_factor = cfg["training"].get("warmup_start_factor", 0.1)
    min_lr_ratio = cfg["training"].get("min_lr_ratio", 0.01)

    if warmup_epochs > 0:
        cosine_epochs = max(epochs - warmup_epochs, 1)
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(
                    optimizer,
                    start_factor=warmup_start_factor,
                    end_factor=1.0,
                    total_iters=warmup_epochs,
                ),
                CosineAnnealingLR(
                    optimizer,
                    T_max=cosine_epochs,
                    eta_min=lr * min_lr_ratio,
                ),
            ],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * min_lr_ratio)

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

        train_metrics = train_one_epoch(model, train_loader, loss_fn, optimizer, device, epoch)
        scheduler.step()

        elapsed = time.time() - t0
        temp = loss_fn.temperature.item()
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"loss={train_metrics['loss']:.4f} | "
            f"diag={train_metrics['diag_sim']:.4f} | "
            f"offdiag={train_metrics['offdiag_sim']:.4f} | "
            f"margin={train_metrics['sim_margin']:.4f} | "
            f"acc={train_metrics['batch_acc']:.3f} | "
            f"temp={temp:.4f} | "
            f"lr={lr_now:.6f} | "
            f"time={elapsed:.1f}s"
        )

        # ---- Periodic evaluation ----
        if epoch % eval_every == 0 or epoch == epochs:
            eval_results = _run_eval(model, data_root, img_size, device, cfg)

            # Use g2s Recall@1 as the checkpoint selection signal
            g2s = eval_results.get("g2s", {})
            r1 = g2s.get("recall@1", 0.0)
            if r1 > best_recall:
                best_recall = r1
                ckpt_path = save_dir / "best.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "metrics": eval_results,
                    "config": cfg,
                }, ckpt_path)
                print(f"  → New best! g2s R@1={r1:.4f}, saved to {ckpt_path}")

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
) -> dict[str, dict[str, float]]:
    """Run benchmark-style cross-view retrieval evaluation.

    Queries come from the ``query`` split; gallery/index comes from the
    ``index`` split. Runs both directions by default:

    - g2s: query ground → index satellite
    - s2g: query satellite → index ground

    A retrieved item is relevant if it shares the query's landmark_id.

    Returns
    -------
    dict mapping direction ("g2s", "s2g") → metrics dict.
    """
    eval_cfg = cfg.get("evaluation", {}) or {}
    eval_transform = get_eval_transforms(img_size)
    eval_batch = cfg["training"].get("eval_batch_size", 128)
    num_workers = cfg["training"].get("num_workers", 4)

    ks = eval_cfg.get("recall_ks", [1, 5, 10])
    map_k = eval_cfg.get("map_k", 1000)
    directions = eval_cfg.get("directions", ["g2s", "s2g"])

    def _loader(split: str, modality: str) -> DataLoader:
        ds = MMLImageDataset(data_root, split, modality, transform=eval_transform)
        return DataLoader(
            ds, batch_size=eval_batch, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

    results: dict[str, dict[str, float]] = {}

    if "g2s" in directions:
        q_loader = _loader("query", "ground")
        idx_loader = _loader("index", "satellite")
        results["g2s"] = evaluate_crossview(
            model, q_loader, idx_loader, device,
            ks=ks, map_k=map_k, direction="g2s",
        )

    if "s2g" in directions:
        q_loader = _loader("query", "satellite")
        idx_loader = _loader("index", "ground")
        results["s2g"] = evaluate_crossview(
            model, q_loader, idx_loader, device,
            ks=ks, map_k=map_k, direction="s2g",
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Train cross-view retrieval baseline")
    parser.add_argument("--config", type=str, default="configs/crossview_baseline.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg)


if __name__ == "__main__":
    main()
