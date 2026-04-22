"""Training loop for cross-view retrieval baseline.

Implements a Sample4Geo-style training pipeline adapted for MMLandmarks:

1. ConvNeXt backbone with shared weights across ground and satellite views.
2. Symmetric InfoNCE loss with learnable temperature.
3. Hard-negative batch sampling (when ``hard_negatives.enabled``):
   a. Early epochs (``gps_epochs``): batches built from GPS-nearest neighbors.
   b. Later epochs: Dynamic Similarity Sampling — the neighbor table is
      rebuilt every ``dss_refresh_every`` epochs from the current model's
      landmark embeddings (mean of ground + satellite view per landmark).
4. Fallback: if ``hard_negatives.enabled`` is false, uses
   :class:`UniqueLandmarkSampler` (random unique-landmark batches).

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
from torch.utils.data import ConcatDataset, DataLoader
import yaml

from mmgeo.crossview.dataset import (
    MMLCrossViewDataset,
    MMLImageDataset,
    UniqueLandmarkSampler,
    get_eval_transforms,
    get_train_transforms,
)
from mmgeo.crossview.evaluate import evaluate_crossview
from mmgeo.crossview.logging_utils import RunLogger
from mmgeo.crossview.losses import SymmetricInfoNCE
from mmgeo.crossview.model import CrossViewModel
from mmgeo.crossview.sampling import (
    HardNegativeBatchSampler,
    build_gps_neighbors,
    build_similarity_neighbors,
    compute_landmark_embeddings,
)


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


def train(cfg: dict, resume: str | None = None) -> None:
    """Main training function.

    Parameters
    ----------
    cfg : dict
        Parsed YAML config.
    resume : str | None
        Optional path to a ``best.pt`` / ``last.pt`` checkpoint to resume from.
        Model weights are restored and the LR scheduler is fast-forwarded so
        the cosine decay continues from the correct position.
    """

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

    # ---- Hard-negative sampler setup (Sample4Geo-style) ----
    hn_cfg = cfg.get("hard_negatives", {}) or {}
    hn_enabled = bool(hn_cfg.get("enabled", False))
    num_workers = cfg["training"].get("num_workers", 4)

    train_sampler: object
    hn_state: dict = {}
    if hn_enabled:
        neighbor_pool = int(hn_cfg.get("neighbor_pool", 20))
        pool_size = int(hn_cfg.get("pool_size", neighbor_pool))
        print(f"\n[HardNeg] Precomputing GPS neighbors (k={neighbor_pool})...")
        coords = train_dataset.get_all_coords()  # (N, 2) degrees
        gps_neighbors = build_gps_neighbors(coords, k=neighbor_pool)
        print(f"[HardNeg] GPS neighbor table: {gps_neighbors.shape}")

        iters_per_epoch_cfg = hn_cfg.get("iters_per_epoch")
        train_sampler = HardNegativeBatchSampler(
            neighbor_table=gps_neighbors,
            batch_size=batch_size,
            pool_size=pool_size,
            iters_per_epoch=iters_per_epoch_cfg,
            seed=cfg.get("seed"),
        )
        print(f"[HardNeg] iters_per_epoch={train_sampler.iters_per_epoch}")
        hn_state = {
            "gps_neighbors": gps_neighbors,
            "neighbor_pool": neighbor_pool,
            "pool_size": pool_size,
            "gps_epochs": int(hn_cfg.get("gps_epochs", 3)),
            "dss_refresh_every": int(hn_cfg.get("dss_refresh_every", 1)),
            "dss_embed_batch": int(hn_cfg.get("dss_embed_batch", 128)),
        }
    else:
        train_sampler = UniqueLandmarkSampler(train_dataset, batch_size=batch_size)

    persistent_workers = bool(cfg["training"].get("persistent_workers", True)) and num_workers > 0
    prefetch_factor = cfg["training"].get("prefetch_factor", 2) if num_workers > 0 else None

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    # Separate dataset copy with eval transforms for DSS embedding passes
    dss_dataset = None
    if hn_enabled:
        dss_dataset = MMLCrossViewDataset(
            data_root=data_root,
            split="train",
            transform_ground=get_eval_transforms(img_size),
            transform_sat=get_eval_transforms(img_size),
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

    # ---- Resume: load checkpoint weights + fast-forward scheduler ----
    start_epoch = 1
    if resume:
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        print(f"[Resume] Loaded checkpoint from epoch {ckpt['epoch']} ({resume})")
        print(f"[Resume] Resuming training from epoch {start_epoch}")
        # Fast-forward scheduler so cosine decay continues from the right position
        for _ in range(ckpt["epoch"]):
            scheduler.step()
        print(f"[Resume] LR after fast-forward: {optimizer.param_groups[0]['lr']:.6f}")

    # ---- Run directory + logger ----
    logging_cfg = cfg.get("logging", {}) or {}
    runs_root = Path(logging_cfg.get("runs_root", cfg.get("save_dir", "checkpoints/crossview")))
    run_tag = logging_cfg.get("run_tag")
    logger = RunLogger(
        root_dir=runs_root,
        cfg=cfg,
        run_prefix=logging_cfg.get("run_prefix"),
        run_tag=run_tag,
        selection_metric=logging_cfg.get("selection_metric", "g2s_recall@1"),
    )
    save_dir = logger.run_dir

    eval_every = cfg["training"].get("eval_every", 5)

    print(f"\nStarting training: {epochs} epochs, batch_size={batch_size}")
    print(f"Backbone: {backbone}, LR: {lr}")
    if resume:
        print(f"Resuming from epoch {start_epoch} / {epochs}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()

        # ---- Hard-negative phase switch / DSS refresh ----
        if hn_enabled:
            phase = _maybe_refresh_hard_negatives(
                epoch=epoch,
                sampler=train_sampler,  # type: ignore[arg-type]
                hn_state=hn_state,
                model=model,
                dss_dataset=dss_dataset,
                device=device,
                num_workers=num_workers,
            )
            print(f"[HardNeg] Epoch {epoch}: phase={phase}")

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

        logger.log_train(
            epoch=epoch,
            metrics=train_metrics,
            extra={"temperature": temp, "lr": lr_now, "epoch_seconds": elapsed},
        )

        # ---- Periodic evaluation ----
        if epoch % eval_every == 0 or epoch == epochs:
            eval_results = _run_eval(model, data_root, img_size, device, cfg)

            prev_best_epoch = logger.best["epoch"]
            logger.log_eval(epoch, eval_results)

            # Save best checkpoint whenever the logger advanced the best epoch
            if logger.best["epoch"] == epoch and prev_best_epoch != epoch:
                ckpt_path = save_dir / "best.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "metrics": eval_results,
                    "config": cfg,
                }, ckpt_path)
                print(f"  → New best! {logger.selection_metric}={logger.best['score']:.4f}, saved to {ckpt_path}")

        # Always overwrite last.pt so a wall-time kill leaves a usable resume point
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "config": cfg,
        }, save_dir / "last.pt")

    logger.finalize()
    best_score = logger.best["score"] if logger.best["epoch"] is not None else float("nan")
    print(f"\nTraining complete. Best {logger.selection_metric}: {best_score:.4f}")


def _maybe_refresh_hard_negatives(
    epoch: int,
    sampler: HardNegativeBatchSampler,
    hn_state: dict,
    model: CrossViewModel,
    dss_dataset: MMLCrossViewDataset | None,
    device: torch.device,
    num_workers: int,
) -> str:
    """Switch neighbor source between GPS and DSS as training progresses.

    Phase rules:
    - ``epoch <= gps_epochs``: GPS neighbors (set once at init, no work here).
    - ``epoch > gps_epochs`` and ``(epoch - gps_epochs - 1) % dss_refresh_every == 0``:
      recompute DSS neighbors from current model embeddings and swap into sampler.
    - Otherwise: reuse previous table.

    Returns
    -------
    str
        ``"gps"`` or ``"dss"`` for logging.
    """
    gps_epochs = hn_state["gps_epochs"]
    if epoch <= gps_epochs:
        return "gps"

    refresh_every = hn_state["dss_refresh_every"]
    steps_into_dss = epoch - gps_epochs - 1
    if steps_into_dss % refresh_every != 0:
        return "dss"

    assert dss_dataset is not None, "dss_dataset must be provided when hard negatives are enabled"
    print(f"[HardNeg] Rebuilding DSS neighbor table (epoch {epoch})...")
    embeds = compute_landmark_embeddings(
        model,
        dss_dataset,
        device,
        batch_size=hn_state["dss_embed_batch"],
        num_workers=num_workers,
        seed=epoch,  # different deterministic pick each refresh
    )
    sim_neighbors = build_similarity_neighbors(embeds, k=hn_state["neighbor_pool"])
    sampler.set_neighbors(sim_neighbors, pool_size=hn_state["pool_size"])
    print(f"[HardNeg] DSS neighbor table: {sim_neighbors.shape}")
    return "dss"


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
    include_index = bool(eval_cfg.get("include_index", True))

    def _single_loader(split: str, modality: str) -> DataLoader:
        ds = MMLImageDataset(data_root, split, modality, transform=eval_transform)
        return DataLoader(
            ds, batch_size=eval_batch, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

    def _gallery_loader(modality: str) -> DataLoader:
        # Gallery = query-side (labeled positives) + index-side (unlabeled
        # distractors, landmark_id = -1). The ``index`` split has no
        # landmark_id column, so those rows never match any query and serve
        # purely as hard distractors for a benchmark-style retrieval setup.
        query_ds = MMLImageDataset(data_root, "query", modality, transform=eval_transform)
        if include_index:
            index_ds = MMLImageDataset(data_root, "index", modality, transform=eval_transform)
            ds = ConcatDataset([query_ds, index_ds])
            print(
                f"[Eval] gallery {modality}: query={len(query_ds)} + "
                f"index={len(index_ds)} = {len(ds)}"
            )
        else:
            ds = query_ds
        return DataLoader(
            ds, batch_size=eval_batch, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

    results: dict[str, dict[str, float]] = {}

    if "g2s" in directions:
        q_loader = _single_loader("query", "ground")
        idx_loader = _gallery_loader("satellite")
        results["g2s"] = evaluate_crossview(
            model, q_loader, idx_loader, device,
            ks=ks, map_k=map_k, direction="g2s",
        )

    if "s2g" in directions:
        q_loader = _single_loader("query", "satellite")
        idx_loader = _gallery_loader("ground")
        results["s2g"] = evaluate_crossview(
            model, q_loader, idx_loader, device,
            ks=ks, map_k=map_k, direction="s2g",
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Train cross-view retrieval baseline")
    parser.add_argument("--config", type=str, default="configs/crossview_baseline.yaml")
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to a checkpoint (best.pt or last.pt) to resume training from. "
             "The model weights and epoch number are restored; the LR scheduler is "
             "fast-forwarded to match.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
