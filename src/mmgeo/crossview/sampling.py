"""Hard-negative batch sampling for Sample4Geo-style training.

Two neighbor sources for building hard batches:

1. **GPS neighbors** (used in early epochs): landmarks that are
   geographically close are likely visually similar in satellite view,
   so they make natural hard negatives without needing a trained model.

2. **Dynamic Similarity Sampling (DSS)** (used in later epochs): once the
   model has learned a reasonable representation, its own embedding space
   is queried to find the current hardest negatives per landmark. The
   similarity neighbor table is rebuilt every ``dss_refresh_every`` epochs.

Both produce an (N, K) integer neighbor table indexed by dataset index
(which equals the position in ``MMLCrossViewDataset.landmark_ids``), with
self excluded. The :class:`HardNegativeBatchSampler` then constructs
batches of unique landmarks by pairing each seed with sampled neighbors.
"""

from __future__ import annotations

import math
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler


# ---------------------------------------------------------------------------
# GPS (haversine) neighbors
# ---------------------------------------------------------------------------

def build_gps_neighbors(
    coords_deg: np.ndarray,
    k: int,
    chunk: int = 512,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Find K nearest GPS neighbors for each landmark via haversine distance.

    Parameters
    ----------
    coords_deg : np.ndarray, shape (N, 2)
        [lat, lon] in degrees.
    k : int
        Number of neighbors to return per landmark (self excluded).
    chunk : int
        Query-side chunk size. Keeps peak memory bounded.
    device : str | torch.device
        Device for the pairwise computation. CPU is usually fine for
        N ≈ 20k since the operation is O(N²) but runs once.

    Returns
    -------
    neighbors : np.ndarray, shape (N, k), int64
        Neighbor landmark indices, sorted from closest to farthest.
    """
    coords_rad = torch.from_numpy(np.deg2rad(coords_deg.astype(np.float64))).to(device)
    lat = coords_rad[:, 0]  # (N,)
    lon = coords_rad[:, 1]  # (N,)
    n = lat.shape[0]

    k_eff = min(k, n - 1)
    out = torch.empty((n, k_eff), dtype=torch.long, device=device)

    cos_lat = torch.cos(lat)  # (N,)

    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        q_lat = lat[start:end].unsqueeze(1)        # (B, 1)
        q_lon = lon[start:end].unsqueeze(1)        # (B, 1)
        q_cos = cos_lat[start:end].unsqueeze(1)    # (B, 1)

        dlat = q_lat - lat.unsqueeze(0)            # (B, N)
        dlon = q_lon - lon.unsqueeze(0)            # (B, N)

        a = torch.sin(dlat / 2) ** 2 + q_cos * cos_lat.unsqueeze(0) * torch.sin(dlon / 2) ** 2
        # Angular distance on unit sphere; multiplying by Earth radius is
        # unnecessary for ranking. Clamp for numerical safety.
        dist = 2 * torch.asin(torch.clamp(torch.sqrt(a), max=1.0))

        # Exclude self by setting diagonal to +inf
        row_idx = torch.arange(start, end, device=device)
        dist[torch.arange(end - start, device=device), row_idx] = float("inf")

        _, idx = torch.topk(dist, k_eff, dim=1, largest=False)
        out[start:end] = idx

    return out.cpu().numpy().astype(np.int64)


# ---------------------------------------------------------------------------
# Landmark embeddings (for DSS)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_landmark_embeddings(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
    batch_size: int = 128,
    num_workers: int = 4,
    seed: int = 0,
) -> torch.Tensor:
    """Embed each landmark once by averaging its ground and satellite views.

    Iterates ``dataset`` deterministically (with a fixed numpy seed for the
    random image pick inside ``__getitem__``) and returns a tensor of
    L2-normalized (ground + satellite) / 2 embeddings in dataset order.

    Parameters
    ----------
    model : nn.Module
        Shared image encoder returning L2-normalized embeddings.
    dataset : Dataset
        An instance of :class:`MMLCrossViewDataset` (returns dict with
        ``ground_img``, ``sat_img``, ``landmark_id``).
    device : torch.device
    batch_size : int
    num_workers : int
    seed : int
        Seed for the numpy RNG so image choice is deterministic within a call.

    Returns
    -------
    embeds : torch.Tensor, shape (N, D)
        L2-normalized landmark embeddings on CPU.
    """
    was_training = model.training
    model.eval()

    # Deterministic image picks: seed numpy used inside MMLCrossViewDataset.__getitem__
    np.random.seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    all_embeds = []
    for batch in loader:
        g = batch["ground_img"].to(device, non_blocking=True)
        s = batch["sat_img"].to(device, non_blocking=True)
        g_e = model(g)
        s_e = model(s)
        # Mean of both views, renormalized. Captures "where is this landmark
        # in the shared embedding space" from both sides.
        mean_e = F.normalize(g_e + s_e, dim=-1)
        all_embeds.append(mean_e.cpu())

    if was_training:
        model.train()

    return torch.cat(all_embeds, dim=0)


# ---------------------------------------------------------------------------
# Similarity neighbors (for DSS)
# ---------------------------------------------------------------------------

def build_similarity_neighbors(
    embeddings: torch.Tensor,
    k: int,
    chunk: int = 512,
) -> np.ndarray:
    """Find K most similar landmarks for each landmark by cosine similarity.

    Parameters
    ----------
    embeddings : torch.Tensor, shape (N, D)
        L2-normalized landmark embeddings.
    k : int
        Neighbors per landmark (self excluded).
    chunk : int
        Query-side chunk size.

    Returns
    -------
    neighbors : np.ndarray, shape (N, k), int64
    """
    n = embeddings.shape[0]
    k_eff = min(k, n - 1)
    out = torch.empty((n, k_eff), dtype=torch.long)

    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        sims = embeddings[start:end] @ embeddings.T  # (B, N)
        # Exclude self: set diagonal to -inf
        row_idx = torch.arange(start, end)
        sims[torch.arange(end - start), row_idx] = float("-inf")
        _, idx = torch.topk(sims, k_eff, dim=1, largest=True)
        out[start:end] = idx

    return out.numpy().astype(np.int64)


# ---------------------------------------------------------------------------
# Hard-negative batch sampler
# ---------------------------------------------------------------------------

class HardNegativeBatchSampler(Sampler):
    """Yield batches where all landmarks are mutual hard negatives.

    Each epoch, every landmark appears exactly once as the "seed" of some
    batch. The rest of the batch is drawn (without replacement) from the
    seed's top-``pool_size`` neighbors in ``neighbor_table``. If a seed's
    pool is too small, the shortfall is filled by random landmarks so every
    batch has exactly ``batch_size`` items.

    The neighbor table may be swapped in place via :meth:`set_neighbors`
    (used to switch from GPS to DSS and to refresh DSS between epochs)
    without recreating the DataLoader.

    Parameters
    ----------
    neighbor_table : np.ndarray, shape (N, K), int64
        Per-landmark neighbor indices (self excluded).
    batch_size : int
    pool_size : int | None
        Only sample from the closest ``pool_size`` neighbors per seed.
        Defaults to K. Smaller pool = harder negatives (but less diverse).
    seed : int | None
        RNG seed. If None, a fresh nondeterministic RNG is used.
    """

    def __init__(
        self,
        neighbor_table: np.ndarray,
        batch_size: int,
        pool_size: int | None = None,
        seed: int | None = None,
    ) -> None:
        self._check_table(neighbor_table)
        self.neighbor_table = neighbor_table
        self.batch_size = batch_size
        self.pool_size = min(pool_size or neighbor_table.shape[1], neighbor_table.shape[1])
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _check_table(table: np.ndarray) -> None:
        if table.ndim != 2:
            raise ValueError(f"neighbor_table must be 2D, got shape {table.shape}")

    def set_neighbors(self, new_table: np.ndarray, pool_size: int | None = None) -> None:
        """Swap the neighbor table in place (no DataLoader recreation needed)."""
        self._check_table(new_table)
        if new_table.shape[0] != self.neighbor_table.shape[0]:
            raise ValueError(
                f"new_table rows {new_table.shape[0]} != existing {self.neighbor_table.shape[0]}"
            )
        self.neighbor_table = new_table
        if pool_size is not None:
            self.pool_size = min(pool_size, new_table.shape[1])
        else:
            self.pool_size = min(self.pool_size, new_table.shape[1])

    def __iter__(self) -> Iterator[list[int]]:
        n = self.neighbor_table.shape[0]
        need = self.batch_size - 1
        pool_size = self.pool_size
        all_indices = np.arange(n)
        seed_order = self.rng.permutation(n)

        for seed in seed_order:
            pool = self.neighbor_table[seed, :pool_size]
            # Guard against accidental self-inclusion
            pool = pool[pool != seed]
            used = {int(seed)}
            chosen: list[int] = []

            if len(pool) >= need:
                picks = self.rng.choice(pool, size=need, replace=False)
                chosen = [int(x) for x in picks]
                used.update(chosen)
            else:
                chosen = [int(x) for x in pool]
                used.update(chosen)
                # Top up with random non-duplicate landmarks
                while len(chosen) < need:
                    cand = int(self.rng.integers(0, n))
                    if cand not in used:
                        chosen.append(cand)
                        used.add(cand)

            yield [int(seed)] + chosen

    def __len__(self) -> int:
        return self.neighbor_table.shape[0]
