"""Sample4Geo implementation skeleton.

Paper: "Sample4Geo: Hard Negative Sampling For Cross-View Geo-Localisation"
       Deuser, Habel, Oswald — arXiv:2303.11851v2

Work through the milestones in order. Each TODO block tells you what to build,
which paper section to read, and how to verify your work before moving on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms

from mmgeo.geolocalizations.sample4geo.datasets import MMLDataset, get_transforms


# ---------------------------------------------------------------------------
# Milestone 2 — Model Architecture
# ---------------------------------------------------------------------------
# Read: paper §3.2
# Dependency: pip install timm for ConvNeXt backbones
# ---------------------------------------------------------------------------


class Sample4Geo(nn.Module):
    """Siamese ConvNeXt encoder with shared weights.

    Both views (ground and satellite) pass through the same backbone.
    Outputs are mean-pooled feature vectors, L2-normalised.

    Paper §3.2
    ----------
    - Single encoder for both views (weight sharing)
    - No special aggregation modules — plain mean pooling
    - ConvNeXt-B (88M params) is the default

    Hint: `timm.create_model(model_name, pretrained=pretrained, num_classes=0)`
    returns a model whose forward() gives pooled features directly when
    num_classes=0. Check `model.num_features` for the embedding dimension.

    Verify: model(torch.randn(4, 3, 384, 384)) → shape (4, D), each row has L2-norm ≈ 1.
    """

    def __init__(self, model_name: str = "convnext_base", pretrained: bool = True) -> None:
        # TODO (Milestone 2)
        # 1. super().__init__()
        # 2. Build the backbone with timm.create_model(...)
        # 3. Store self.embed_dim = backbone.num_features
        super().__init__()
        

        model_name = "convnext_base"
        pretrained = True
        import timm
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.embed_dim = self.backbone.num_features
        


    def forward(self, images: Tensor) -> Tensor:
        """Encode a batch of images → L2-normalised embeddings.

        Parameters
        ----------
        images : Tensor, shape (B, 3, H, W)

        Returns
        -------
        Tensor, shape (B, embed_dim), unit L2-norm per row
        """
        # TODO (Milestone 2)
        # 1. features = self.backbone(images)         → (B, D)
        # 2. return F.normalize(features, dim=-1)
        features = self.backbone(images)
        return F.normalize(features, dim=-1)


# ---------------------------------------------------------------------------
# Milestone 3 — Symmetric InfoNCE Loss
# ---------------------------------------------------------------------------
# Read: paper §3.1, equation (1)
# ---------------------------------------------------------------------------


class SymmetricInfoNCE(nn.Module):
    """Symmetric InfoNCE loss with learnable temperature and label smoothing.

    Paper §3.1
    ----------
    - Temperature τ is a *learnable* scalar (initialise log_tau = log(1/0.07) ≈ 2.66,
      same as CLIP).  Declare it as nn.Parameter so the optimiser updates it.
    - Loss:
        S  = query_emb @ ref_emb.T * exp(log_tau)     # shape (B, B)
        L  = 0.5 * [CE(S, arange(B)) + CE(S.T, arange(B))]
    - Label smoothing 0.1 (pass directly to F.cross_entropy).

    Verify
    ------
    - Random unit-norm embeddings with B=128: loss ≈ log(128) ≈ 4.85
    - query_emb == ref_emb (diagonal dominant): loss ≈ 0
    """

    def __init__(self, label_smoothing: float = 0.1) -> None:
        # TODO (Milestone 3)
        super().__init__()
        self.log_tau = nn.Parameter(torch.tensor(2.6592)) #fixed init tau
        self.label_smoothing = label_smoothing
        self.embed_dim = None  # will be set when we see the first batch of embeddings



    def forward(self, query_emb: Tensor, ref_emb: Tensor) -> Tensor:
        # TODO (Milestone 3)
        # Step 1: S = query_emb @ ref_emb.T * self.log_tau.exp()
        # Step 2: labels = torch.arange(B, device=query_emb.device)
        # Step 3: return 0.5 * (F.cross_entropy(S, labels, label_smoothing=...)
        #                     + F.cross_entropy(S.T, labels, label_smoothing=...))
        S = query_emb @ ref_emb.T * self.log_tau.exp()
        labels = torch.arange(query_emb.shape[0], device=query_emb.device)
        #set variable
        if self.embed_dim is None:
            self.embed_dim = query_emb.shape[1]
        loss = 0.5 * (F.cross_entropy(S, labels, label_smoothing=self.label_smoothing)
                        + F.cross_entropy(S.T, labels, label_smoothing=self.label_smoothing))
        return loss
        
# ---------------------------------------------------------------------------
# Milestone 4 — GPS Sampler
# ---------------------------------------------------------------------------
# Read: paper §3.3
# Dependency: haversine from mmgeo.geolocalizations.geoclip.evaluate
# ---------------------------------------------------------------------------


def build_gps_neighbours(lats: np.ndarray, lons: np.ndarray, k: int) -> np.ndarray:
    """For each sample, find the k nearest neighbours by great-circle distance.

    Parameters
    ----------
    lats, lons : shape (N,), decimal degrees
    k          : number of neighbours per sample (not counting self)

    Returns
    -------
    neighbour_indices : shape (N, k), dtype int
        Row i contains indices of the k closest landmarks to landmark i.

    Hint: import haversine from mmgeo.geolocalizations.geoclip.evaluate.
    For N < 50k, scipy.spatial.distance.cdist on radian coords with
    metric='haversine' * 6371 gives a full (N, N) distance matrix.
    Then use np.argsort per row, skipping the diagonal (self).
    For larger N: sklearn.neighbors.BallTree with haversine metric.

    Verify: for landmark i, all returned neighbours should be within
    a few hundred km of (lats[i], lons[i]).
    """
    
    from mmgeo.geolocalizations.geoclip.evaluate import haversine

    N = len(lats)
    # Broadcast: repeat each lat/lon N times vs tile N times → (N*N,) pairwise distances
    dist_flat = haversine(
        np.repeat(lats, N), np.repeat(lons, N),
        np.tile(lats, N), np.tile(lons, N),
    )
    dist_matrix = dist_flat.reshape(N, N)
    np.fill_diagonal(dist_matrix, np.inf)  # exclude self
    neighbour_indices = np.argsort(dist_matrix, axis=1)[:, :k]
    return neighbour_indices


class GPSSampler(Sampler):
    """Batch sampler that groups geographically close samples.

    Paper §3.3: nearby locations share visual properties (vegetation, road type)
    → contrasting them in one batch gives useful hard negatives from epoch 1,
    before any embeddings have been computed.

    Algorithm per batch
    -------------------
    1. Pick a random anchor index i from a shuffled queue of all indices.
    2. Fill the batch with [i] + its k precomputed GPS neighbours.
    3. Clip to batch_size; mark indices used so they are not repeated this epoch.

    Verify: look up (lat, lon) for all indices in one batch — they should
    all cluster within the same geographic region.
    """

    def __init__(self, neighbour_indices: np.ndarray, batch_size: int) -> None:
        # TODO (Milestone 4)
        # Store neighbour_indices (N, k) and batch_size.
        super().__init__()
        self.neighbour_indices = neighbour_indices
        self.batch_size = batch_size
        self.num_samples = neighbour_indices.shape[0]
        self.indices = np.arange(self.num_samples)
        self.used = np.zeros(self.num_samples, dtype=bool)  # track used indices

    def __iter__(self):
        # TODO (Milestone 4)
        # Shuffle all anchor indices at the start of each epoch.
        np.random.shuffle(self.indices)
        self.used[:] = False
        for i in self.indices:
            if not self.used[i]:
                batch = [i]
                neighbours = self.neighbour_indices[i]
                for n in neighbours:
                    if not self.used[n]:
                        batch.append(n)
                    if len(batch) >= self.batch_size:
                        break
                self.used[batch] = True
                yield batch

    def __len__(self) -> int:
        # TODO (Milestone 4)
        return (self.num_samples + self.batch_size - 1) // self.batch_size


# ---------------------------------------------------------------------------
# Milestone 5 — Dynamic Similarity Sampler (DSS)
# ---------------------------------------------------------------------------
# Read: paper §3.4
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_all_embeddings(
    model: Sample4Geo,
    loader: DataLoader,
    device: torch.device | str,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a full inference pass over the training set to collect query embeddings.

    Parameters
    ----------
    model  : Sample4Geo in eval mode
    loader : DataLoader over MMLDataset — encode the ground images only

    Returns
    -------
    query_embs : np.ndarray, shape (N, D), float32, L2-normalised
    sample_ids : np.ndarray, shape (N,), int  — original dataset indices in order

    Hint: model.eval(); iterate loader; call model(ground_imgs) each batch;
    accumulate results and track sequential indices.
    """
    model.eval()
    all_embs = []
    idx_offset = 0
    with torch.no_grad():
        for batch in loader:
            ground_imgs, satellite_imgs, lat, lon, landmark_id = batch
            ground_imgs = ground_imgs.to(device)
            emb = model(ground_imgs).cpu().numpy()
            all_embs.append(emb)
            idx_offset += emb.shape[0]
    query_embs = np.concatenate(all_embs, axis=0)
    sample_ids = np.arange(query_embs.shape[0])
    return query_embs, sample_ids


def build_similarity_neighbours(query_embs: np.ndarray, K: int = 128) -> np.ndarray:
    """For each sample, find the K nearest neighbours by cosine similarity.

    Parameters
    ----------
    query_embs : shape (N, D), L2-normalised  (cosine sim = dot product)
    K          : pool size per sample (paper uses K=128)

    Returns
    -------
    neighbour_indices : shape (N, K), sorted descending by similarity (nearest first),
                        self excluded.

    Hint: S = query_embs @ query_embs.T  → (N, N).
    Fill diagonal with -inf, then torch.topk(S, K, dim=1).indices or np.argsort.
    For large N (>50k) consider torch.topk on GPU or faiss.
    """
    
    S = query_embs @ query_embs.T
    np.fill_diagonal(S, -np.inf)
    neighbour_indices = np.argsort(-S, axis=1)[:, :K]
    return neighbour_indices


class DSSSampler(Sampler):
    """Dynamic Similarity Sampler — hard-negative batches from visual similarity.

    Paper §3.4 algorithm per batch
    ------------------------------
    Given K pre-ranked neighbours for anchor i:
    - Take the k//2 nearest (hardest negatives).
    - Randomly sample k//2 from the remaining K − (k//2) neighbours (diversity).
    - Combine into a batch of size k (+ anchor).

    Default hyperparameters (paper): k=64, K=128.
    The caller rebuilds neighbour_indices every 4 epochs by calling
    compute_all_embeddings + build_similarity_neighbours.

    Why the random half? Pure hard-negative batches can cause model collapse
    (paper §3.4, citing [26]). The random half ensures enough diversity.

    Verify: average intra-batch cosine similarity should exceed that of a
    random batch drawn from the same dataset.
    """

    def __init__(
        self,
        neighbour_indices: np.ndarray,
        batch_size: int,
        k: int = 64,
    ) -> None:
        super().__init__()
        self.neighbour_indices = neighbour_indices
        self.batch_size = batch_size
        self.k = k
        self.num_samples = neighbour_indices.shape[0]
        self.indices = np.arange(self.num_samples)
        self.used = np.zeros(self.num_samples, dtype=bool)  # track used indices

    def __iter__(self):
        np.random.shuffle(self.indices)
        self.used[:] = False
        for i in self.indices:
            if not self.used[i]:
                neighbours = self.neighbour_indices[i]
                hard_negatives = neighbours[:self.k // 2]
                remaining = neighbours[self.k // 2:]
                random_negatives = np.random.choice(remaining, size=self.k // 2, replace=False)
                batch = [i] + list(hard_negatives) + list(random_negatives)
                batch = batch[:self.batch_size]  # clip to batch_size
                self.used[batch] = True
                yield batch

    def __len__(self) -> int:
        return (self.num_samples + self.batch_size - 1) // self.batch_size

# ---------------------------------------------------------------------------
# Milestone 6 — Training Loop
# ---------------------------------------------------------------------------
# Read: paper §4.2 (optimiser, scheduler, GPS→DSS transition)
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: Sample4Geo,
    loss_fn: SymmetricInfoNCE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device | str,
) -> float:
    """Run one training epoch; return mean loss over all batches.

    Hint: standard PyTorch loop — zero_grad → forward both views → loss → backward → step.
    Call scheduler.step() once per batch (OneCycleLR) or once per epoch (CosineAnnealingLR).
    """
    from tqdm import tqdm
    model.train()
    total_loss = 0.0
    num_batches = 0
    for batch in tqdm(loader, desc="Training", leave=False):
        ground_imgs, satellite_imgs, lat, lon, landmark_id = batch
        ground_imgs = ground_imgs.to(device)
        satellite_imgs = satellite_imgs.to(device)
        optimizer.zero_grad()
        query_embs = model(ground_imgs)
        ref_embs = model(satellite_imgs)
        loss = loss_fn(query_embs, ref_embs)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
        num_batches += 1
    mean_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return mean_loss


def train(
    data_root: Path,
    epochs: int = 40,
    batch_size: int = 128,
    lr: float = 1e-3,
    gps_epochs: int = 10,
    dss_refresh_every: int = 4,
    device: str = "cuda",
) -> Sample4Geo:
    """Full training procedure with GPS → DSS sampler transition.

    Paper §4.2 recipe
    -----------------
    - AdamW optimiser, initial lr=0.001
    - Cosine LR scheduler with 1-epoch linear warmup
      (torch.optim.lr_scheduler.OneCycleLR handles warmup + cosine decay in one call)
    - Epochs 1 .. gps_epochs:     GPSSampler
    - Epochs gps_epochs+1 .. end: DSSSampler; rebuild every dss_refresh_every epochs via
                                   compute_all_embeddings + build_similarity_neighbours

    Steps
    -----
    1. Build MMLDataset (train split), get GPS coords for GPSSampler.
    2. build_gps_neighbours → GPSSampler.
    3. For each epoch:
       a. If epoch > gps_epochs and epoch % dss_refresh_every == 0:
          recompute embeddings and switch to / refresh DSSSampler.
       b. train_one_epoch(...)
       c. Log loss.
    4. Return trained model.
    """
    # 1. Build dataset and get GPS coords
    ground_transform = get_transforms("ground", "train")
    satellite_transform = get_transforms("satellite", "train")
    dataset = MMLDataset(data_root, split="train",
                         ground_transform=ground_transform,
                         satellite_transform=satellite_transform)

    lats = dataset.data["lat"].values
    lons = dataset.data["lon"].values

    # 2. Initial GPS sampler + loader
    gps_neighbours = build_gps_neighbours(lats, lons, k=batch_size - 1)
    sampler = GPSSampler(gps_neighbours, batch_size=batch_size)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=4)

    # 3. Model, loss, optimizer
    model = Sample4Geo().to(device)
    loss_fn = SymmetricInfoNCE().to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(loss_fn.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr,
        total_steps=epochs * len(loader) + 100,
        pct_start=0.025,
    )

    for epoch in range(epochs):
        if epoch >= gps_epochs and epoch % dss_refresh_every == 0:
            query_embs, sample_ids = compute_all_embeddings(model, loader, device)
            neighbour_indices = build_similarity_neighbours(query_embs, K=128)
            sampler = DSSSampler(neighbour_indices, batch_size=batch_size, k=64)
            loader = DataLoader(dataset, batch_sampler=sampler, num_workers=4)
        mean_loss = train_one_epoch(model, loss_fn, loader, optimizer, scheduler, device)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {mean_loss:.4f}")
    return model

# ---------------------------------------------------------------------------
# Milestone 7 — Evaluation
# ---------------------------------------------------------------------------
# Read: paper §4.1 (metrics: R@1, R@5, R@10, R@1%)
# Reuse: accuracy_at_thresholds, haversine from geoclip/evaluate.py
# ---------------------------------------------------------------------------


@torch.no_grad()
def build_gallery(
    model: Sample4Geo,
    index_loader: DataLoader,
    device: torch.device | str,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode all index satellite images into an embedding gallery.

    The index CSV is `data/MML_Data/index/mml_index_satellite.csv`
    with columns: images, lat, lon, year.

    Returns
    -------
    gallery_embs   : shape (M, D), L2-normalised
    gallery_coords : shape (M, 2), [[lat, lon], ...]
    """
    model.eval()
    all_embs = []
    all_coords = []
    with torch.no_grad():
        for batch in index_loader:
            satellite_imgs = batch['satellite_image'].to(device)
            emb = model(satellite_imgs).cpu().numpy()
            all_embs.append(emb)
            coords = np.stack([batch['lat'].numpy(), batch['lon'].numpy()], axis=1)
            all_coords.append(coords)
    gallery_embs = np.concatenate(all_embs, axis=0)
    gallery_coords = np.concatenate(all_coords, axis=0)
    return gallery_embs, gallery_coords


@torch.no_grad()
def evaluate(
    model: Sample4Geo,
    query_loader: DataLoader,
    gallery_embs: np.ndarray,
    gallery_coords: np.ndarray,
    device: torch.device | str,
) -> dict[str, float]:
    """Retrieve top-1 satellite per query and compute geo-localisation metrics.

    Algorithm
    ---------
    1. Encode each query ground image → query_emb  (shape B × D).
    2. scores = query_emb @ gallery_embs.T          (shape B × M).
    3. top1_idx = scores.argmax(dim=-1).
    4. pred_coords = gallery_coords[top1_idx].
    5. errors_km = haversine(pred_lat, pred_lon, true_lat, true_lon).
    6. Return accuracy_at_thresholds(errors_km) and median error.

    Returns a dict, e.g.:
        {"r@1km": 0.12, "r@25km": 0.45, "r@200km": 0.71,
         "r@750km": 0.88, "r@2500km": 0.95, "median_km": 312.4}

    Hint: from mmgeo.geolocalizations.geoclip.evaluate import (
              haversine, accuracy_at_thresholds, median_error)
    """
    from mmgeo.geolocalizations.geoclip.evaluate import haversine, accuracy_at_thresholds, median_error
    model.eval()
    all_embs = []
    all_coords = []
    with torch.no_grad():
        for batch in query_loader:
            query_imgs = batch['query_image'].to(device)
            emb = model(query_imgs).cpu().numpy()
            all_embs.append(emb)
            coords = np.stack([batch['lat'].numpy(), batch['lon'].numpy()], axis=1)
            all_coords.append(coords)
    query_embs = np.concatenate(all_embs, axis=0)
    query_coords = np.concatenate(all_coords, axis=0)

    scores = query_embs @ gallery_embs.T
    top1_idx = scores.argmax(axis=-1)
    pred_coords = gallery_coords[top1_idx]
    errors_km = haversine(pred_coords[:, 0], pred_coords[:, 1], query_coords[:, 0], query_coords[:, 1])
    metrics = accuracy_at_thresholds(errors_km)
    metrics["median_km"] = median_error(errors_km)
    return metrics
