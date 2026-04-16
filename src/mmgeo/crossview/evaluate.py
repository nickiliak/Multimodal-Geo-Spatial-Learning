"""Evaluation utilities for cross-view retrieval.

Metrics follow ILIAS/Sample4Geo conventions:
- Recall@K: fraction of queries with a correct match in top-K results
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


def compute_recall_at_k(
    query_embeds: torch.Tensor,
    query_lids: np.ndarray,
    index_embeds: torch.Tensor,
    index_lids: np.ndarray,
    ks: list[int] | None = None,
    batch_size: int = 256,
) -> dict[int, float]:
    """Compute Recall@K for retrieval.

    For each query, finds the top-K nearest index images by cosine similarity
    and checks if any share the same landmark_id.

    Parameters
    ----------
    query_embeds : torch.Tensor, shape (Q, D)
    query_lids : np.ndarray, shape (Q,)
    index_embeds : torch.Tensor, shape (I, D)
    index_lids : np.ndarray, shape (I,)
    ks : list of int
        Recall thresholds.
    batch_size : int
        Process queries in batches to avoid OOM on large index sets.

    Returns
    -------
    dict mapping k -> recall (0.0 to 1.0)
    """
    if ks is None:
        ks = DEFAULT_RECALL_KS
    max_k = max(ks)

    n_queries = len(query_embeds)
    correct = {k: 0 for k in ks}

    # Process in batches to handle large similarity matrices
    for start in range(0, n_queries, batch_size):
        end = min(start + batch_size, n_queries)
        q_batch = query_embeds[start:end]  # (B, D)
        q_lids = query_lids[start:end]     # (B,)

        # Cosine similarity (embeddings are already L2-normalized)
        sims = q_batch @ index_embeds.T  # (B, I)

        # Top-K indices
        topk_indices = sims.topk(max_k, dim=1).indices.numpy()  # (B, max_k)

        for i in range(len(q_batch)):
            query_lid = q_lids[i]
            retrieved_lids = index_lids[topk_indices[i]]

            for k in ks:
                if query_lid in retrieved_lids[:k]:
                    correct[k] += 1

    return {k: correct[k] / n_queries for k in ks}


def evaluate_crossview(
    model: torch.nn.Module,
    query_loader: DataLoader,
    index_loader: DataLoader,
    device: torch.device,
    ks: list[int] | None = None,
    direction: str = "g2s",
) -> dict[int, float]:
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
    direction : str
        ``"g2s"`` (ground→satellite) or ``"s2g"`` (satellite→ground).

    Returns
    -------
    dict mapping k -> recall value
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {direction.upper()} retrieval")
    print(f"{'='*60}")

    print("Embedding queries...")
    q_embeds, q_lids = extract_embeddings(model, query_loader, device)
    print(f"  → {len(q_embeds)} query embeddings, {len(np.unique(q_lids))} unique landmarks")

    print("Embedding index...")
    idx_embeds, idx_lids = extract_embeddings(model, index_loader, device)
    print(f"  → {len(idx_embeds)} index embeddings, {len(np.unique(idx_lids))} unique landmarks")

    print("Computing recall...")
    recalls = compute_recall_at_k(q_embeds, q_lids, idx_embeds, idx_lids, ks=ks)

    for k, v in recalls.items():
        print(f"  Recall@{k}: {v:.4f} ({v*100:.2f}%)")

    return recalls
