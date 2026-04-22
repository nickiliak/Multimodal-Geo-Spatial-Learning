"""Evaluation utilities for cross-view retrieval.

Metrics follow ILIAS / MMLandmarks / Sample4Geo conventions:
- Recall@K: fraction of queries with a correct match in top-K results
- mAP@K: mean Average Precision restricted to the top-K retrieved items
- A match is correct if the retrieved image has the same landmark_id as the query

Two evaluation directions:
- Ground → Satellite: query ground images, retrieve from satellite index
- Satellite → Ground: query satellite images, retrieve from ground index
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


DEFAULT_RECALL_KS = [1, 5, 10]
DEFAULT_MAP_K = 1000


@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, np.ndarray]:
    """Extract embeddings and landmark IDs from a dataset.

    Parameters
    ----------
    model : nn.Module
        Image encoder (should return L2-normalized embeddings).
    dataloader : DataLoader
        Yields (images, landmark_ids) batches.
    device : torch.device

    Returns
    -------
    embeddings : torch.Tensor, shape (N, D)
    landmark_ids : np.ndarray, shape (N,)
    """
    model.eval()
    all_embeds = []
    all_lids = []

    for images, lids in tqdm(dataloader, desc="Extracting embeddings", unit="batch"):
        images = images.to(device)
        embeds = model(images)  # (B, D), already L2-normalized
        all_embeds.append(embeds.cpu())
        if isinstance(lids, torch.Tensor):
            all_lids.append(lids.numpy())
        else:
            all_lids.append(np.array(lids))

    embeddings = torch.cat(all_embeds, dim=0)
    landmark_ids = np.concatenate(all_lids, axis=0)
    return embeddings, landmark_ids


def compute_retrieval_metrics(
    query_embeds: torch.Tensor,
    query_lids: np.ndarray,
    index_embeds: torch.Tensor,
    index_lids: np.ndarray,
    recall_ks: list[int] | None = None,
    map_k: int = DEFAULT_MAP_K,
    batch_size: int = 256,
) -> dict[str, float]:
    """Compute Recall@K and mAP@K for retrieval.

    For each query, ranks all index images by cosine similarity (descending).
    A retrieved item is relevant if it shares the query's landmark_id.

    Recall@K: 1 if any relevant item appears in top-K, else 0 (averaged over queries).
    mAP@K: Average Precision restricted to top-K per query, using the standard
        AP@K = (1 / min(R, K)) * Σ_{i=1..K} P(i) * rel(i)
        where R is the total number of relevant items for the query and P(i) is
        precision at rank i. Queries with R = 0 are skipped.

    Parameters
    ----------
    query_embeds : torch.Tensor, shape (Q, D)
    query_lids : np.ndarray, shape (Q,)
    index_embeds : torch.Tensor, shape (I, D)
    index_lids : np.ndarray, shape (I,)
    recall_ks : list of int
    map_k : int
        Truncation depth for mAP. Clamped to the index size.
    batch_size : int
        Process queries in batches to bound memory use.

    Returns
    -------
    dict with keys:
        "recall@k" for each k in recall_ks (float, 0..1)
        f"map@{map_k}" (float, 0..1)
    """
    if recall_ks is None:
        recall_ks = DEFAULT_RECALL_KS

    n_queries = len(query_embeds)
    n_index = len(index_embeds)
    effective_map_k = min(map_k, n_index)
    top_k_needed = max(max(recall_ks), effective_map_k)

    # Precompute per-landmark relevant counts (for AP denominator)
    # relevant_counts[query_lid] = total index items with that landmark_id
    unique_idx_lids, idx_counts = np.unique(index_lids, return_counts=True)
    lid_to_count = dict(zip(unique_idx_lids.tolist(), idx_counts.tolist()))

    recall_hits = {k: 0 for k in recall_ks}
    ap_sum = 0.0
    ap_count = 0  # queries with at least one relevant item

    for start in range(0, n_queries, batch_size):
        end = min(start + batch_size, n_queries)
        q_batch = query_embeds[start:end]  # (B, D)
        q_lids = query_lids[start:end]     # (B,)

        # Cosine similarity (embeddings are L2-normalized)
        sims = q_batch @ index_embeds.T  # (B, I)

        topk_indices = sims.topk(top_k_needed, dim=1).indices.cpu().numpy()  # (B, top_k_needed)

        for i in range(len(q_batch)):
            q_lid = int(q_lids[i])
            retrieved_lids = index_lids[topk_indices[i]]  # (top_k_needed,)
            relevance = (retrieved_lids == q_lid)

            # Recall@K
            for k in recall_ks:
                if relevance[:k].any():
                    recall_hits[k] += 1

            # mAP@K
            total_relevant = lid_to_count.get(q_lid, 0)
            if total_relevant == 0:
                continue

            rel_top = relevance[:effective_map_k].astype(np.float64)
            if rel_top.sum() == 0:
                ap_count += 1  # contributes 0 AP to the average
                continue

            ranks = np.arange(1, effective_map_k + 1, dtype=np.float64)
            cum_hits = np.cumsum(rel_top)
            precision_at_ranks = cum_hits / ranks  # P(i) for i=1..K
            ap = (precision_at_ranks * rel_top).sum() / min(total_relevant, effective_map_k)
            ap_sum += ap
            ap_count += 1

    metrics: dict[str, float] = {
        f"recall@{k}": recall_hits[k] / max(n_queries, 1) for k in recall_ks
    }
    metrics[f"map@{effective_map_k}"] = (ap_sum / ap_count) if ap_count > 0 else 0.0
    return metrics


def pool_embeddings_by_landmark(
    embeddings: torch.Tensor,
    landmark_ids: np.ndarray,
) -> tuple[torch.Tensor, np.ndarray]:
    """Mean-pool per-image embeddings into one L2-normalized embedding per landmark.

    Images with landmark_id == -1 (unlabeled index distractors) are kept as-is
    since they have no group to merge into.

    Parameters
    ----------
    embeddings : torch.Tensor, shape (N, D)
        L2-normalized per-image embeddings.
    landmark_ids : np.ndarray, shape (N,)
        Landmark ID for each image. -1 = unlabeled distractor.

    Returns
    -------
    pooled_embeds : torch.Tensor, shape (M, D)
        One L2-normalized embedding per unique landmark (labeled) + one per
        unlabeled image, M ≤ N.
    pooled_lids : np.ndarray, shape (M,)
        Corresponding landmark IDs.
    """
    labeled_mask = landmark_ids != -1
    if not labeled_mask.any():
        # Nothing to pool (all distractors or all unlabeled)
        return embeddings, landmark_ids

    labeled_lids = landmark_ids[labeled_mask]
    labeled_embeds = embeddings[labeled_mask]

    unique_lids = np.unique(labeled_lids)
    pooled_list = []
    for lid in unique_lids:
        mask = labeled_lids == lid
        mean_e = labeled_embeds[mask].mean(dim=0)
        pooled_list.append(F.normalize(mean_e, dim=0))

    pooled_embeds = torch.stack(pooled_list, dim=0)  # (M, D)
    pooled_lids = unique_lids

    # Append any unlabeled rows unchanged (should not appear on query side
    # but kept for completeness)
    unlabeled_mask = ~labeled_mask
    if unlabeled_mask.any():
        pooled_embeds = torch.cat([pooled_embeds, embeddings[unlabeled_mask]], dim=0)
        pooled_lids = np.concatenate([pooled_lids, landmark_ids[unlabeled_mask]])

    return pooled_embeds, pooled_lids


def compute_recall_at_k(
    query_embeds: torch.Tensor,
    query_lids: np.ndarray,
    index_embeds: torch.Tensor,
    index_lids: np.ndarray,
    ks: list[int] | None = None,
    batch_size: int = 256,
) -> dict[int, float]:
    """Backward-compatible Recall@K wrapper around compute_retrieval_metrics."""
    if ks is None:
        ks = DEFAULT_RECALL_KS
    metrics = compute_retrieval_metrics(
        query_embeds, query_lids, index_embeds, index_lids,
        recall_ks=ks, map_k=0, batch_size=batch_size,
    )
    return {k: metrics[f"recall@{k}"] for k in ks}


def evaluate_crossview(
    model: torch.nn.Module,
    query_loader: DataLoader,
    index_loader: DataLoader,
    device: torch.device,
    ks: list[int] | None = None,
    map_k: int = DEFAULT_MAP_K,
    direction: str = "g2s",
    pool_queries: bool = True,
) -> dict[str, float]:
    """Full cross-view retrieval evaluation pipeline.

    Parameters
    ----------
    model : nn.Module
        Shared image encoder.
    query_loader : DataLoader
        Query images (ground for g2s, satellite for s2g).
    index_loader : DataLoader
        Index/gallery images (satellite for g2s, ground for s2g).
    device : torch.device
    ks : list of int
        Recall thresholds.
    map_k : int
        Truncation depth for mAP (default 1000).
    direction : str
        ``"g2s"`` (ground→satellite) or ``"s2g"`` (satellite→ground).
    pool_queries : bool
        If True (default), mean-pool all per-image embeddings that share the
        same landmark_id into a single L2-normalized query embedding before
        retrieval. This follows the MMLandmarks benchmark protocol (all ground
        images per landmark are averaged). Has no effect when each landmark
        already has exactly one query image (e.g. s2g satellite queries).

    Returns
    -------
    dict with keys ``"recall@k"`` for each k, and ``"map@<map_k>"``.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {direction.upper()} retrieval")
    print(f"{'='*60}")

    print("Embedding queries...")
    q_embeds, q_lids = extract_embeddings(model, query_loader, device)
    n_raw = len(q_embeds)
    n_unique = len(np.unique(q_lids[q_lids != -1]))
    print(f"  → {n_raw} query embeddings, {n_unique} unique landmarks")

    if pool_queries and n_raw > n_unique:
        q_embeds, q_lids = pool_embeddings_by_landmark(q_embeds, q_lids)
        print(f"  → pooled to {len(q_embeds)} landmark embeddings (mean of {n_raw/max(n_unique,1):.1f} imgs/landmark)")

    print("Embedding index...")
    idx_embeds, idx_lids = extract_embeddings(model, index_loader, device)
    print(f"  → {len(idx_embeds)} index embeddings, {len(np.unique(idx_lids))} unique landmarks")

    print("Computing metrics...")
    metrics = compute_retrieval_metrics(
        q_embeds, q_lids, idx_embeds, idx_lids,
        recall_ks=ks, map_k=map_k,
    )

    for name, v in metrics.items():
        print(f"  {name}: {v:.4f} ({v*100:.2f}%)")

    return metrics
