"""Hybrid pipeline finetuning.

Trains the Sample4Geo cross-view model with plain (ground, sat) symmetric /
multi-positive InfoNCE — exactly like ``crossview/train.py`` — but evaluates
with the hybrid pipeline (GeoCLIP -> radius narrowing -> Sample4Geo rerank).

GeoCLIP is loaded frozen and only used at eval time. Training-time forward
passes do not use GeoCLIP. The model is initialised from a pretrained
Sample4Geo checkpoint (``model.pretrained_ckpt`` in the config, default
``models/finetuned.pt``) and finetuned for a small number of additional
epochs.

Usage
-----
    python -m mmgeo.train_pipeline --config configs/hybrid.yaml
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
    UniqueLandmarkSampler,
    get_eval_transforms,
    get_train_transforms,
)
from mmgeo.crossview.logging_utils import RunLogger
from mmgeo.crossview.losses import MultiPositiveInfoNCE, SymmetricInfoNCE
from mmgeo.crossview.model import CrossViewModel as Sample4Geo
from mmgeo.crossview.sampling import (
    HardNegativeBatchSampler,
    build_gps_neighbors,
    build_similarity_neighbors,
    compute_landmark_embeddings,
)
from mmgeo.geolocalizations.geoclip.geoclip_baseline import (
    GeoClipBaseline,
    load_gallery_coords,
)
from mmgeo.inference import (
    SatelliteGallery,
    _build_ground_index,
    _build_index_sat_gallery,
    _build_query_sat_gallery,
    _embed_hex_ids,
    _haversine_matrix_km,
    embed_ground_queries,
    predict_rough_gps,
)


# ---------------------------------------------------------------------------
# Train one epoch (mirrors crossview/train.py)
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: Sample4Geo,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    """Train Sample4Geo for one epoch on (ground, sat) pairs.

    Plain InfoNCE — GeoCLIP is not used here. Identical structure to
    ``mmgeo.crossview.train.train_one_epoch``.
    """
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

        B = sat_imgs.shape[0]
        K = 1
        if ground_imgs.dim() == 5:
            B, K, C, H, W = ground_imgs.shape
            ground_imgs = ground_imgs.view(B * K, C, H, W)

        ground_embeds = model(ground_imgs)
        sat_embeds = model(sat_imgs)

        loss = loss_fn(ground_embeds, sat_embeds)

        with torch.no_grad():
            if K > 1:
                D = ground_embeds.shape[-1]
                ground_diag = torch.nn.functional.normalize(
                    ground_embeds.view(B, K, D).mean(dim=1), dim=-1
                )
            else:
                ground_diag = ground_embeds
            sims = ground_diag @ sat_embeds.T
            diag = sims.diag()
            if B > 1:
                offdiag_mask = ~torch.eye(B, dtype=torch.bool, device=sims.device)
                offdiag_mean = sims[offdiag_mask].mean()
            else:
                offdiag_mean = torch.zeros((), device=sims.device)
            batch_acc = (sims.argmax(dim=1) == torch.arange(B, device=sims.device)).float().mean()

            total_diag_sim += diag.mean().item()
            total_offdiag_sim += offdiag_mean.item()
            total_margin += (diag.mean() - offdiag_mean).item()
            total_batch_acc += batch_acc.item()

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


# ---------------------------------------------------------------------------
# Hybrid evaluation
# ---------------------------------------------------------------------------

def _build_satellite_gallery_in_memory(
    model: Sample4Geo,
    data_root: Path,
    index_mode: str,
    device: torch.device,
    img_size: int,
    batch_size: int,
    num_workers: int,
) -> SatelliteGallery:
    """Build a satellite gallery from the *current* model weights.

    Always recomputes (no disk cache) because Sample4Geo weights change every
    epoch — a stale cache would give wrong embeddings. The cost is one full
    sat-gallery embedding pass per evaluation.
    """
    if index_mode == "query":
        hex_ids, lids, coords = _build_query_sat_gallery(data_root)
        embeds = _embed_hex_ids(
            model, data_root, "query", hex_ids, device,
            batch_size=batch_size, num_workers=num_workers, desc="embed query-sat",
        )
    elif index_mode == "full":
        q_hex, q_lids, q_coords = _build_query_sat_gallery(data_root)
        i_hex, i_lids, i_coords = _build_index_sat_gallery(data_root)
        q_emb = _embed_hex_ids(
            model, data_root, "query", q_hex, device,
            batch_size=batch_size, num_workers=num_workers, desc="embed query-sat",
        )
        i_emb = _embed_hex_ids(
            model, data_root, "index", i_hex, device,
            batch_size=batch_size, num_workers=num_workers, desc="embed index-sat",
        )
        hex_ids = q_hex + i_hex
        lids = np.concatenate([q_lids, i_lids])
        coords = np.concatenate([q_coords, i_coords], axis=0)
        embeds = torch.cat([q_emb, i_emb], dim=0)
    else:
        raise ValueError(f"index_mode must be 'query' or 'full', got {index_mode!r}")

    return SatelliteGallery(coords=coords, landmark_ids=lids, embeds=embeds, hex_ids=hex_ids)


def _hybrid_rerank(
    query_embeds: torch.Tensor,
    query_lids: np.ndarray,
    rough_gps: np.ndarray,
    gallery: SatelliteGallery,
    radius_km: float,
    device: torch.device,
    query_batch: int,
    recall_ks: list[int],
    map_k: int,
) -> dict:
    """Rerank Sample4Geo against radius-narrowed gallery, ``fallback_full`` policy.

    Empty-candidate queries (no sat within ``radius_km`` of GeoCLIP's predicted
    GPS) fall back to reranking against the full gallery — equivalent to
    radius=inf for those queries only. This is the ``fallback_full`` policy
    from ``mmgeo.inference._rerank_and_score``.
    """
    idx_embeds = gallery.embeds.to(device)
    idx_lids = gallery.landmark_ids
    idx_coords = gallery.coords

    n_q = len(query_embeds)
    n_i = len(idx_embeds)
    effective_map_k = min(map_k, n_i)
    top_k_needed = max(max(recall_ks), effective_map_k)

    unique_idx, counts = np.unique(idx_lids, return_counts=True)
    lid_to_count = dict(zip(unique_idx.tolist(), counts.tolist()))

    recall_hits = {k: 0 for k in recall_ks}
    ap_sum = 0.0
    ap_count = 0
    empty_count = 0
    candidate_counts: list[int] = []

    for start in range(0, n_q, query_batch):
        end = min(start + query_batch, n_q)
        q_emb = query_embeds[start:end].to(device)
        q_rough = rough_gps[start:end]

        d_km = _haversine_matrix_km(
            q_rough[:, 0], q_rough[:, 1], idx_coords[:, 0], idx_coords[:, 1]
        )
        mask = d_km <= radius_km

        per_q = mask.sum(axis=1)
        candidate_counts.extend(per_q.tolist())

        # Fallback: empty rows -> unmask everything for that row (radius=inf).
        empty_rows = per_q == 0
        if empty_rows.any():
            mask[empty_rows, :] = True
        empty_count += int(empty_rows.sum())

        sims = q_emb @ idx_embeds.T
        neg_inf = torch.finfo(sims.dtype).min
        mask_t = torch.from_numpy(mask).to(device)
        sims = sims.masked_fill(~mask_t, neg_inf)
        topk = sims.topk(top_k_needed, dim=1).indices.cpu().numpy()

        for i in range(end - start):
            q_lid = int(query_lids[start + i])
            retrieved = idx_lids[topk[i]]
            relevance = (retrieved == q_lid)

            for k in recall_ks:
                if relevance[:k].any():
                    recall_hits[k] += 1

            total_rel = lid_to_count.get(q_lid, 0)
            if total_rel == 0:
                continue
            rel_top = relevance[:effective_map_k].astype(np.float64)
            if rel_top.sum() == 0:
                ap_count += 1
                continue
            ranks = np.arange(1, effective_map_k + 1, dtype=np.float64)
            cum = np.cumsum(rel_top)
            ap = ((cum / ranks) * rel_top).sum() / min(total_rel, effective_map_k)
            ap_sum += ap
            ap_count += 1

    return {
        **{f"g2s_R@{k}": recall_hits[k] / max(n_q, 1) for k in recall_ks},
        f"g2s_mAP@{effective_map_k}": (ap_sum / ap_count) if ap_count > 0 else 0.0,
        "empty_rate": empty_count / max(n_q, 1),
        "mean_candidates": float(np.mean(candidate_counts)) if candidate_counts else 0.0,
        "median_candidates": float(np.median(candidate_counts)) if candidate_counts else 0.0,
        "total_queries": n_q,
    }


def run_hybrid_eval(
    model: Sample4Geo,
    geoclip: GeoClipBaseline,
    geoclip_gallery_coords: np.ndarray,
    data_root: Path,
    device: torch.device,
    cfg: dict,
) -> dict:
    """Run a single hybrid eval pass: GeoCLIP -> radius -> Sample4Geo rerank.

    Returns a flat dict of metrics suitable for CSV logging. ``fallback_full``
    is hard-coded — when no sat is within the radius of GeoCLIP's predicted
    GPS, the query falls back to the full gallery (i.e. plain Sample4Geo
    reranking for that query).
    """
    eval_cfg = cfg["hybrid_eval"]
    img_size = cfg["training"].get("img_size", 384)
    eval_batch = cfg["training"].get("eval_batch_size", 128)
    num_workers = cfg["training"].get("num_workers", 4)

    model.eval()

    # Embed satellite gallery with current model weights.
    t_gal = time.perf_counter()
    gallery = _build_satellite_gallery_in_memory(
        model, data_root,
        index_mode=eval_cfg["index_mode"],
        device=device,
        img_size=img_size,
        batch_size=eval_batch,
        num_workers=num_workers,
    )
    gallery_s = time.perf_counter() - t_gal
    print(f"[HybridEval] gallery ({eval_cfg['index_mode']}): {len(gallery.hex_ids)} sat, {gallery_s:.1f}s")

    # Build ground query set.
    paths, q_lids, _q_true_coords = _build_ground_index(data_root, eval_cfg["query_mode"])
    print(f"[HybridEval] ground queries ({eval_cfg['query_mode']}): {len(paths)}")

    # GeoCLIP rough GPS.
    rough_gps, geoclip_s = predict_rough_gps(
        geoclip, paths, geoclip_gallery_coords,
        batch_size=int(eval_cfg.get("geoclip_batch_size", 64)),
    )

    # Sample4Geo ground embeddings.
    q_embeds, s4g_s = embed_ground_queries(
        model, paths, device,
        batch_size=eval_batch, num_workers=num_workers,
    )

    metrics = _hybrid_rerank(
        q_embeds, q_lids, rough_gps, gallery,
        radius_km=float(eval_cfg["radius_km"]),
        device=device,
        query_batch=int(eval_cfg.get("query_batch", 64)),
        recall_ks=list(eval_cfg.get("recall_ks", [1, 5, 10])),
        map_k=int(eval_cfg.get("map_k", 1000)),
    )

    metrics["radius_km"] = float(eval_cfg["radius_km"])
    metrics["index_mode"] = eval_cfg["index_mode"]
    metrics["query_mode"] = eval_cfg["query_mode"]
    metrics["geoclip_s_total"] = geoclip_s
    metrics["s4g_embed_s_total"] = s4g_s
    metrics["gallery_embed_s"] = gallery_s

    map_k_eff = min(int(eval_cfg.get("map_k", 1000)), len(gallery.embeds))
    map_val = metrics.get(f"g2s_mAP@{map_k_eff}", 0.0)
    print(
        f"[HybridEval] R@1={metrics.get('g2s_R@1', 0):.4f} "
        f"R@5={metrics.get('g2s_R@5', 0):.4f} "
        f"R@10={metrics.get('g2s_R@10', 0):.4f} "
        f"mAP@{map_k_eff}={map_val:.4f} "
        f"empty={metrics['empty_rate']:.3f} "
        f"mean_cand={metrics['mean_candidates']:.1f}"
    )
    return metrics


# ---------------------------------------------------------------------------
# Hard-negative phase switch (copied from crossview/train.py)
# ---------------------------------------------------------------------------

def _maybe_refresh_hard_negatives(
    epoch: int,
    sampler: HardNegativeBatchSampler,
    hn_state: dict,
    model: Sample4Geo,
    dss_dataset: MMLCrossViewDataset | None,
    device: torch.device,
    num_workers: int,
) -> str:
    gps_epochs = hn_state["gps_epochs"]
    if epoch <= gps_epochs:
        return "gps"
    refresh_every = hn_state["dss_refresh_every"]
    steps_into_dss = epoch - gps_epochs - 1
    if steps_into_dss % refresh_every != 0:
        return "dss"

    assert dss_dataset is not None
    print(f"[HardNeg] Rebuilding DSS neighbor table (epoch {epoch})...")
    embeds = compute_landmark_embeddings(
        model, dss_dataset, device,
        batch_size=hn_state["dss_embed_batch"],
        num_workers=num_workers, seed=epoch,
    )
    sim_neighbors = build_similarity_neighbors(embeds, k=hn_state["neighbor_pool"])
    sampler.set_neighbors(sim_neighbors, pool_size=hn_state["pool_size"])
    print(f"[HardNeg] DSS neighbor table: {sim_neighbors.shape}")
    return "dss"


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def _load_pretrained_sample4geo(ckpt_path: Path, model: Sample4Geo, device: torch.device) -> int:
    """Load Sample4Geo weights from a checkpoint into ``model``. Returns ckpt epoch."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[Init] loaded Sample4Geo from {ckpt_path} (epoch={ckpt.get('epoch')})")
    return int(ckpt.get("epoch", 0))


def train(cfg: dict, resume: str | None = None) -> None:
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    # ---- Data ----
    data_root = Path(cfg["data"]["root"])
    img_size = cfg["training"].get("img_size", 384)
    batch_size = cfg["training"].get("batch_size", 24)
    n_ground = cfg["training"].get("n_ground", 1)

    train_dataset = MMLCrossViewDataset(
        data_root=data_root, split="train",
        transform_ground=get_train_transforms(img_size),
        transform_sat=get_train_transforms(img_size),
        n_ground=n_ground,
    )

    # ---- Hard-negative sampler ----
    hn_cfg = cfg.get("hard_negatives", {}) or {}
    hn_enabled = bool(hn_cfg.get("enabled", False))
    num_workers = cfg["training"].get("num_workers", 4)

    train_sampler: object
    hn_state: dict = {}
    if hn_enabled:
        neighbor_pool = int(hn_cfg.get("neighbor_pool", 20))
        pool_size = int(hn_cfg.get("pool_size", neighbor_pool))
        coords = train_dataset.get_all_coords()
        gps_neighbors = build_gps_neighbors(coords, k=neighbor_pool)
        train_sampler = HardNegativeBatchSampler(
            neighbor_table=gps_neighbors,
            batch_size=batch_size,
            pool_size=pool_size,
            iters_per_epoch=hn_cfg.get("iters_per_epoch"),
            seed=cfg.get("seed"),
        )
        hn_state = {
            "gps_neighbors": gps_neighbors,
            "neighbor_pool": neighbor_pool,
            "pool_size": pool_size,
            "gps_epochs": int(hn_cfg.get("gps_epochs", 2)),
            "dss_refresh_every": int(hn_cfg.get("dss_refresh_every", 1)),
            "dss_embed_batch": int(hn_cfg.get("dss_embed_batch", 128)),
        }
    else:
        train_sampler = UniqueLandmarkSampler(train_dataset, batch_size=batch_size)

    persistent_workers = bool(cfg["training"].get("persistent_workers", True)) and num_workers > 0
    prefetch_factor = cfg["training"].get("prefetch_factor", 2) if num_workers > 0 else None

    train_loader = DataLoader(
        train_dataset, batch_sampler=train_sampler,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=persistent_workers, prefetch_factor=prefetch_factor,
    )

    dss_dataset = None
    if hn_enabled:
        dss_dataset = MMLCrossViewDataset(
            data_root=data_root, split="train",
            transform_ground=get_eval_transforms(img_size),
            transform_sat=get_eval_transforms(img_size),
        )

    # ---- Model: Sample4Geo (trainable) ----
    backbone = cfg["model"].get("backbone", "convnext_base.fb_in22k")
    model = Sample4Geo(
        backbone=backbone, pretrained=False,  # weights will come from ckpt
        embed_dim=cfg["model"].get("embed_dim", 0),
    )
    pretrained_ckpt = cfg["model"].get("pretrained_ckpt")
    if pretrained_ckpt and not resume:
        _load_pretrained_sample4geo(Path(pretrained_ckpt), model, device)
    model.to(device)

    # ---- Model: GeoCLIP (frozen, eval only) ----
    print("[Init] loading frozen GeoCLIP...")
    geoclip = GeoClipBaseline(device=str(device))
    for p in geoclip.model.parameters():
        p.requires_grad = False
    geoclip.model.eval()
    geoclip_gallery_coords = load_gallery_coords(
        data_root, source=cfg["hybrid_eval"].get("geoclip_gallery_source", "paper"),
    )
    geoclip.build_gallery(geoclip_gallery_coords)
    print(f"[Init] GeoCLIP gallery: {len(geoclip_gallery_coords)} coords")

    # ---- Loss ----
    loss_kwargs = dict(
        temperature=cfg["training"].get("temperature", 0.07),
        learnable_temp=cfg["training"].get("learnable_temp", True),
        label_smoothing=cfg["training"].get("label_smoothing", 0.1),
    )
    if n_ground > 1:
        loss_fn = MultiPositiveInfoNCE(**loss_kwargs)
        print(f"[Loss] MultiPositiveInfoNCE (n_ground={n_ground})")
    else:
        loss_fn = SymmetricInfoNCE(**loss_kwargs)
        print(f"[Loss] SymmetricInfoNCE (n_ground=1)")
    loss_fn.to(device)

    # ---- Optimizer (Sample4Geo + temperature only — GeoCLIP frozen) ----
    params = list(model.parameters()) + list(loss_fn.parameters())
    lr = cfg["training"].get("lr", 5e-5)
    optimizer = AdamW(params, lr=lr, weight_decay=cfg["training"].get("weight_decay", 1e-4))

    epochs = cfg["training"].get("epochs", 10)
    warmup_epochs = cfg["training"].get("warmup_epochs", 1)
    warmup_start_factor = cfg["training"].get("warmup_start_factor", 0.1)
    min_lr_ratio = cfg["training"].get("min_lr_ratio", 0.01)

    if warmup_epochs > 0:
        cosine_epochs = max(epochs - warmup_epochs, 1)
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=warmup_start_factor,
                         end_factor=1.0, total_iters=warmup_epochs),
                CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=lr * min_lr_ratio),
            ],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * min_lr_ratio)

    # ---- Resume ----
    start_epoch = 1
    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        print(f"[Resume] from epoch {ckpt['epoch']} ({resume})")
        for _ in range(ckpt["epoch"]):
            scheduler.step()

    # ---- Run dir + logger ----
    logging_cfg = cfg.get("logging", {}) or {}
    runs_root = Path(logging_cfg.get("runs_root", "checkpoints/hybrid"))
    logger = RunLogger(
        root_dir=runs_root, cfg=cfg,
        run_prefix=logging_cfg.get("run_prefix", "hybrid_"),
        run_tag=logging_cfg.get("run_tag"),
        selection_metric=logging_cfg.get("selection_metric", "g2s_R@1"),
    )
    save_dir = logger.run_dir
    eval_every = cfg["training"].get("eval_every", 4)

    print(f"\nStarting training: {epochs} epochs, batch_size={batch_size}, LR={lr}")
    print(f"Backbone: {backbone}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()

        if hn_enabled:
            phase = _maybe_refresh_hard_negatives(
                epoch=epoch, sampler=train_sampler,  # type: ignore[arg-type]
                hn_state=hn_state, model=model,
                dss_dataset=dss_dataset, device=device, num_workers=num_workers,
            )
            print(f"[HardNeg] Epoch {epoch}: phase={phase}")

        train_metrics = train_one_epoch(model, train_loader, loss_fn, optimizer, device, epoch)
        scheduler.step()

        elapsed = time.time() - t0
        temp = loss_fn.temperature.item()
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d}/{epochs} | loss={train_metrics['loss']:.4f} | "
            f"diag={train_metrics['diag_sim']:.4f} | offdiag={train_metrics['offdiag_sim']:.4f} | "
            f"margin={train_metrics['sim_margin']:.4f} | acc={train_metrics['batch_acc']:.3f} | "
            f"temp={temp:.4f} | lr={lr_now:.6f} | time={elapsed:.1f}s"
        )
        logger.log_train(
            epoch=epoch, metrics=train_metrics,
            extra={"temperature": temp, "lr": lr_now, "epoch_seconds": elapsed},
        )

        # Hybrid eval: every ``eval_every`` epochs and always on the last epoch.
        do_eval = (epoch % eval_every == 0) or (epoch == epochs)
        if do_eval:
            hybrid_metrics = run_hybrid_eval(
                model, geoclip, geoclip_gallery_coords,
                data_root=data_root, device=device, cfg=cfg,
            )
            # RunLogger expects {direction: {metric: value}} — wrap so the
            # selection_metric key flattens to the configured name.
            map_k = int(cfg["hybrid_eval"].get("map_k", 1000))
            map_key = f"g2s_mAP@{map_k}"
            recall_ks = list(cfg["hybrid_eval"].get("recall_ks", [1, 5, 10]))
            wrapped = {
                "g2s": {  # logger flattens to "g2s_<key>"
                    **{f"R@{k}": hybrid_metrics[f"g2s_R@{k}"] for k in recall_ks},
                    f"mAP@{map_k}": hybrid_metrics.get(map_key, 0.0),
                },
                "hybrid": {  # extra side-info, prefixed "hybrid_"
                    "empty_rate": hybrid_metrics["empty_rate"],
                    "mean_candidates": hybrid_metrics["mean_candidates"],
                    "median_candidates": hybrid_metrics["median_candidates"],
                    "radius_km": hybrid_metrics["radius_km"],
                },
            }

            prev_best_epoch = logger.best["epoch"]
            logger.log_eval(epoch, wrapped)

            if logger.best["epoch"] == epoch and prev_best_epoch != epoch:
                ckpt_path = save_dir / "best.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "metrics": wrapped,
                    "config": cfg,
                }, ckpt_path)
                print(f"  -> New best! {logger.selection_metric}={logger.best['score']:.4f} -> {ckpt_path}")

        # Always save last.pt for resume.
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "config": cfg,
        }, save_dir / "last.pt")

    logger.finalize()
    best_score = logger.best["score"] if logger.best["epoch"] is not None else float("nan")
    print(f"\nTraining complete. Best {logger.selection_metric}: {best_score:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid pipeline finetuning")
    parser.add_argument("--config", type=str, default="configs/hybrid.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
