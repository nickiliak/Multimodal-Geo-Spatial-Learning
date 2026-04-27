"""Hybrid two-stage retrieval pipeline.

Stage 1: GeoCLIP predicts a rough GPS for each ground query.
Stage 2: gps_to_satellite_image_pipe narrows the satellite gallery to a
         radius around the rough GPS.
Stage 3: Sample4Geo reranks the ground query against only the narrowed subset.

For each (index_mode, query_mode) combination, we sweep a list of radii and
report retrieval metrics (mAP@1k, R@1/5/10), per-stage timings, and narrowing
statistics (mean/median candidates, empty-candidate rate) for two fallback
policies:
    * ``fail``          — empty-candidate queries score 0 AP / 0 recall.
    * ``fallback_full`` — empty-candidate queries rerank against the full
                          gallery (so narrowing never hurts).

We also include ``radius=inf`` as a no-narrowing control: pure Sample4Geo
reranking over the full gallery.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from mmgeo.crossview.dataset import MMLImageDataset, _hex_path, get_eval_transforms
from mmgeo.crossview.model import CrossViewModel
from mmgeo.geolocalizations.geoclip.evaluate import (
    accuracy_at_thresholds,
    haversine,
)
from mmgeo.geolocalizations.geoclip.geoclip_baseline import (
    GeoClipBaseline,
    load_gallery_coords,
)

QueryMode = Literal["one_per_landmark", "all"]
IndexMode = Literal["query", "full"]
FallbackPolicy = Literal["fail", "fallback_full"]

DEFAULT_RADII_KM = [5.0, 25.0, 100.0, 500.0, 2000.0, float("inf")]
DEFAULT_RECALL_KS = [1, 5, 10]
DEFAULT_MAP_K = 1000
DEFAULT_DIST_THRESHOLDS_KM = [1, 10, 25, 200, 750, 2500]


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


# ---------------------------------------------------------------------------
# Sample4Geo loader
# ---------------------------------------------------------------------------

def load_sample4geo(ckpt_path: Path, device: torch.device) -> CrossViewModel:
    """Load the trained cross-view model from ``best1.pt``."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    backbone = cfg.get("model", {}).get("backbone", "convnext_base.fb_in22k")
    embed_dim = cfg.get("model", {}).get("embed_dim", 0)

    model = CrossViewModel(backbone=backbone, pretrained=False, embed_dim=embed_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"[Sample4Geo] loaded {ckpt_path} (epoch={ckpt.get('epoch')}, backbone={backbone})")
    return model


# ---------------------------------------------------------------------------
# Satellite gallery
# ---------------------------------------------------------------------------

@dataclass
class SatelliteGallery:
    coords: np.ndarray        # (N, 2) lat, lon
    landmark_ids: np.ndarray  # (N,) int, -1 for unlabeled index sats
    embeds: torch.Tensor      # (N, D) L2-normalized, on CPU
    hex_ids: list[str]


def _query_sat_coord_map(data_root: Path) -> dict[int, tuple[float, float]]:
    master = pd.read_csv(data_root / "query" / "mml_query.csv")
    return {int(r["landmark_id"]): (float(r["lat"]), float(r["lon"])) for _, r in master.iterrows()}


def _build_query_sat_gallery(data_root: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Query satellite split: coords come from the landmark's row in mml_query.csv."""
    lid_to_coord = _query_sat_coord_map(data_root)
    df = pd.read_csv(data_root / "query" / "mml_query_satellite.csv")

    hex_ids: list[str] = []
    lids: list[int] = []
    coords: list[tuple[float, float]] = []
    for _, row in df.iterrows():
        lid = int(row["landmark_id"])
        c = lid_to_coord[lid]
        for hid in str(row["images"]).split():
            hex_ids.append(hid)
            lids.append(lid)
            coords.append(c)
    return hex_ids, np.array(lids, dtype=np.int64), np.array(coords, dtype=np.float64)


def _build_index_sat_gallery(data_root: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Index satellite split: every row is one image with its own lat/lon."""
    df = pd.read_csv(data_root / "index" / "mml_index_satellite.csv")
    hex_ids = df["images"].astype(str).tolist()
    lids = np.full(len(df), -1, dtype=np.int64)
    coords = df[["lat", "lon"]].to_numpy(dtype=np.float64)
    return hex_ids, lids, coords


@torch.no_grad()
def _embed_hex_ids(
    model: CrossViewModel,
    data_root: Path,
    split: str,
    hex_ids: list[str],
    device: torch.device,
    batch_size: int = 128,
    num_workers: int = 4,
    desc: str = "embed sat",
) -> torch.Tensor:
    """Embed satellite images by hex id using an ad-hoc dataset."""
    transform = get_eval_transforms(384)

    class _HexDataset(torch.utils.data.Dataset):
        def __init__(self, hex_ids: list[str]) -> None:
            self.hex_ids = hex_ids

        def __len__(self) -> int:
            return len(self.hex_ids)

        def __getitem__(self, idx: int) -> torch.Tensor:
            from PIL import Image
            h = self.hex_ids[idx]
            path = _hex_path(data_root, split, "satellite", h)
            img = Image.open(path).convert("RGB")
            return transform(img)

    loader = DataLoader(
        _HexDataset(hex_ids), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    out = []
    for imgs in tqdm(loader, desc=desc, unit="batch"):
        imgs = imgs.to(device, non_blocking=True)
        out.append(model(imgs).cpu())
    return torch.cat(out, dim=0)


def build_satellite_gallery(
    model: CrossViewModel,
    data_root: Path,
    index_mode: IndexMode,
    device: torch.device,
    cache_dir: Path,
    batch_size: int = 128,
    num_workers: int = 4,
) -> tuple[SatelliteGallery, float]:
    """Build (and cache) the satellite gallery for the requested index mode.

    Returns the gallery and the wall-clock seconds spent embedding (0 on cache hit).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"sat_gallery_{index_mode}.pt"
    if cache_path.exists():
        blob = torch.load(cache_path, map_location="cpu", weights_only=False)
        print(f"[Gallery] cache hit: {cache_path}")
        return SatelliteGallery(**blob), 0.0

    if index_mode == "query":
        hex_ids, lids, coords = _build_query_sat_gallery(data_root)
        embeds_all = []
        t0 = time.perf_counter()
        embeds_all.append(_embed_hex_ids(
            model, data_root, "query", hex_ids, device,
            batch_size=batch_size, num_workers=num_workers, desc="embed query-sat",
        ))
        elapsed = time.perf_counter() - t0
        embeds = torch.cat(embeds_all, dim=0)
    elif index_mode == "full":
        # Gallery = query sats (labeled) + index sats (distractors, lid=-1).
        q_hex, q_lids, q_coords = _build_query_sat_gallery(data_root)
        i_hex, i_lids, i_coords = _build_index_sat_gallery(data_root)
        t0 = time.perf_counter()
        q_emb = _embed_hex_ids(
            model, data_root, "query", q_hex, device,
            batch_size=batch_size, num_workers=num_workers, desc="embed query-sat",
        )
        i_emb = _embed_hex_ids(
            model, data_root, "index", i_hex, device,
            batch_size=batch_size, num_workers=num_workers, desc="embed index-sat",
        )
        elapsed = time.perf_counter() - t0
        hex_ids = q_hex + i_hex
        lids = np.concatenate([q_lids, i_lids])
        coords = np.concatenate([q_coords, i_coords], axis=0)
        embeds = torch.cat([q_emb, i_emb], dim=0)
    else:
        raise ValueError(f"Unknown index_mode {index_mode!r}")

    gallery = SatelliteGallery(coords=coords, landmark_ids=lids, embeds=embeds, hex_ids=hex_ids)
    torch.save(gallery.__dict__, cache_path)
    print(f"[Gallery] built {index_mode} in {elapsed:.1f}s, saved to {cache_path}")
    return gallery, elapsed


# ---------------------------------------------------------------------------
# Ground queries
# ---------------------------------------------------------------------------

@dataclass
class GroundQueries:
    image_paths: list[Path]
    landmark_ids: np.ndarray  # (Q,)
    true_coords: np.ndarray   # (Q, 2) lat, lon
    embeds: torch.Tensor      # (Q, D) Sample4Geo embeddings


def _build_ground_index(data_root: Path, query_mode: QueryMode) -> tuple[list[Path], np.ndarray, np.ndarray]:
    lid_to_coord = _query_sat_coord_map(data_root)
    df = pd.read_csv(data_root / "query" / "mml_query_ground.csv")

    paths: list[Path] = []
    lids: list[int] = []
    coords: list[tuple[float, float]] = []
    for _, row in df.iterrows():
        lid = int(row["landmark_id"])
        hex_list = str(row["images"]).split()
        hexes = hex_list[:1] if query_mode == "one_per_landmark" else hex_list
        for h in hexes:
            paths.append(_hex_path(data_root, "query", "ground", h))
            lids.append(lid)
            coords.append(lid_to_coord[lid])
    return paths, np.array(lids, dtype=np.int64), np.array(coords, dtype=np.float64)


@torch.no_grad()
def embed_ground_queries(
    model: CrossViewModel,
    image_paths: list[Path],
    device: torch.device,
    batch_size: int = 128,
    num_workers: int = 4,
) -> tuple[torch.Tensor, float]:
    """Embed ground queries with Sample4Geo. Returns (embeds_cpu, seconds)."""
    transform = get_eval_transforms(384)

    class _PathDataset(torch.utils.data.Dataset):
        def __init__(self, paths: list[Path]) -> None:
            self.paths = paths

        def __len__(self) -> int:
            return len(self.paths)

        def __getitem__(self, idx: int) -> torch.Tensor:
            from PIL import Image
            img = Image.open(self.paths[idx]).convert("RGB")
            return transform(img)

    loader = DataLoader(
        _PathDataset(image_paths), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    out = []
    t0 = time.perf_counter()
    for imgs in tqdm(loader, desc="embed ground (s4g)", unit="batch"):
        imgs = imgs.to(device, non_blocking=True)
        out.append(model(imgs).cpu())
    _sync(device)
    elapsed = time.perf_counter() - t0
    return torch.cat(out, dim=0), elapsed


# ---------------------------------------------------------------------------
# GeoCLIP rough-GPS stage
# ---------------------------------------------------------------------------

def predict_rough_gps(
    geoclip: GeoClipBaseline,
    image_paths: list[Path],
    gallery_coords: np.ndarray,
    batch_size: int = 64,
) -> tuple[np.ndarray, float]:
    """Run GeoCLIP → rough GPS per query. Returns ((Q,2) coords, seconds)."""
    geoclip.build_gallery(gallery_coords)
    t0 = time.perf_counter()
    preds = geoclip.predict_batch(image_paths, batch_size=batch_size)
    _sync(geoclip.device)
    elapsed = time.perf_counter() - t0
    return preds, elapsed


# ---------------------------------------------------------------------------
# Hybrid eval core (radius sweep, fixed query set)
# ---------------------------------------------------------------------------

def _haversine_matrix_km(a_lat: np.ndarray, a_lon: np.ndarray,
                         b_lat: np.ndarray, b_lon: np.ndarray) -> np.ndarray:
    """Full pairwise haversine in km. a: (Q,), b: (I,) -> (Q, I)."""
    a_lat_r = np.radians(a_lat)[:, None]
    a_lon_r = np.radians(a_lon)[:, None]
    b_lat_r = np.radians(b_lat)[None, :]
    b_lon_r = np.radians(b_lon)[None, :]
    dlat = b_lat_r - a_lat_r
    dlon = b_lon_r - a_lon_r
    h = np.sin(dlat / 2) ** 2 + np.cos(a_lat_r) * np.cos(b_lat_r) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(h))


@dataclass
class RadiusResult:
    radius_km: float
    fallback: str
    map_at_k: float
    recall: dict[int, float]
    dist_at_km: dict[int, float]
    empty_rate: float
    mean_candidates: float
    median_candidates: float
    total_queries: int
    # Timings in ms/query averaged over queries (rerank dominates)
    rerank_ms_per_query: float


def _rerank_and_score(
    query_embeds: torch.Tensor,    # (Q, D) cpu
    query_lids: np.ndarray,        # (Q,)
    query_true_coords: np.ndarray, # (Q, 2)
    rough_gps: np.ndarray,         # (Q, 2)
    gallery: SatelliteGallery,
    radius_km: float,
    fallback: FallbackPolicy,
    device: torch.device,
    query_batch: int = 64,
    recall_ks: list[int] = DEFAULT_RECALL_KS,
    map_k: int = DEFAULT_MAP_K,
    dist_thresholds: list[int] = DEFAULT_DIST_THRESHOLDS_KM,
) -> RadiusResult:
    """Rerank each query against the gallery with an optional radius mask.

    Any gallery index whose coord is > radius from the query's rough GPS is
    masked to ``-inf``. For queries with 0 candidates, ``fallback_full`` drops
    the mask for that query; ``fail`` keeps the full mask (all -inf) so the
    top-K is arbitrary and lid-match probability is essentially 0.
    """
    idx_embeds = gallery.embeds.to(device)  # (I, D)
    idx_lids = gallery.landmark_ids
    idx_coords = gallery.coords

    n_q = len(query_embeds)
    n_i = len(idx_embeds)
    effective_map_k = min(map_k, n_i)
    top_k_needed = max(max(recall_ks), effective_map_k)

    # Per-landmark total relevant count for AP denominator
    unique_idx, counts = np.unique(idx_lids, return_counts=True)
    lid_to_count = dict(zip(unique_idx.tolist(), counts.tolist()))

    recall_hits = {k: 0 for k in recall_ks}
    ap_sum = 0.0
    ap_count = 0
    empty_count = 0
    candidate_counts: list[int] = []

    pred_coords_top1 = np.zeros((n_q, 2), dtype=np.float64)
    rerank_total_s = 0.0

    for start in range(0, n_q, query_batch):
        end = min(start + query_batch, n_q)
        q_emb = query_embeds[start:end].to(device)
        q_rough = rough_gps[start:end]

        # Haversine (B, I) in km
        d_km = _haversine_matrix_km(q_rough[:, 0], q_rough[:, 1],
                                    idx_coords[:, 0], idx_coords[:, 1])
        mask = d_km <= radius_km  # (B, I)

        per_q_candidates = mask.sum(axis=1)
        candidate_counts.extend(per_q_candidates.tolist())

        # Empty-candidate handling
        empty_rows = per_q_candidates == 0
        if fallback == "fallback_full":
            mask[empty_rows, :] = True
        empty_count += int(empty_rows.sum())

        # Sim + mask + top-K on GPU for speed
        _sync(device)
        t0 = time.perf_counter()
        sims = q_emb @ idx_embeds.T  # (B, I)
        neg_inf = torch.finfo(sims.dtype).min
        mask_t = torch.from_numpy(mask).to(device)
        sims = sims.masked_fill(~mask_t, neg_inf)
        topk = sims.topk(top_k_needed, dim=1).indices.cpu().numpy()
        _sync(device)
        rerank_total_s += time.perf_counter() - t0

        for i in range(end - start):
            q_lid = int(query_lids[start + i])
            retrieved = idx_lids[topk[i]]
            relevance = (retrieved == q_lid)

            # Top-1 predicted GPS
            pred_coords_top1[start + i] = idx_coords[topk[i][0]]

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
            cum_hits = np.cumsum(rel_top)
            p_at_r = cum_hits / ranks
            ap = (p_at_r * rel_top).sum() / min(total_rel, effective_map_k)
            ap_sum += ap
            ap_count += 1

    dist = accuracy_at_thresholds(
        pred_coords_top1[:, 0], pred_coords_top1[:, 1],
        query_true_coords[:, 0], query_true_coords[:, 1],
        thresholds_km=dist_thresholds,
    )

    return RadiusResult(
        radius_km=radius_km,
        fallback=fallback,
        map_at_k=(ap_sum / ap_count) if ap_count > 0 else 0.0,
        recall={k: recall_hits[k] / max(n_q, 1) for k in recall_ks},
        dist_at_km={int(t): v for t, v in dist.items()},
        empty_rate=empty_count / max(n_q, 1),
        mean_candidates=float(np.mean(candidate_counts)),
        median_candidates=float(np.median(candidate_counts)),
        total_queries=n_q,
        rerank_ms_per_query=1000.0 * rerank_total_s / max(n_q, 1),
    )


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

@dataclass
class PipelineTimings:
    geoclip_s: float = 0.0
    pipe_s: float = 0.0        # haversine mask — measured inside rerank, tiny
    s4g_embed_s: float = 0.0   # ground embedding
    gallery_embed_s: float = 0.0
    total_queries: int = 0

    def per_query_ms(self) -> dict[str, float]:
        n = max(self.total_queries, 1)
        return {
            "geoclip_ms": 1000.0 * self.geoclip_s / n,
            "s4g_embed_ms": 1000.0 * self.s4g_embed_s / n,
            # rerank/pipe is reported per-radius in RadiusResult
        }


class HybridPipeline:
    """Two-stage ground→satellite pipeline with radius narrowing."""

    def __init__(
        self,
        data_root: Path,
        sample4geo_ckpt: Path,
        device: torch.device,
        cache_dir: Path,
        geoclip_gallery_source: str = "index",
        num_workers: int = 4,
        batch_size: int = 128,
    ) -> None:
        self.data_root = Path(data_root)
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.num_workers = num_workers
        self.batch_size = batch_size

        print("[HybridPipeline] loading GeoCLIP...")
        self.geoclip = GeoClipBaseline(device=str(device))
        self.geoclip_gallery = load_gallery_coords(self.data_root, source=geoclip_gallery_source)
        print(f"[HybridPipeline] GeoCLIP gallery source='{geoclip_gallery_source}', size={len(self.geoclip_gallery)}")

        print("[HybridPipeline] loading Sample4Geo...")
        self.sample4geo = load_sample4geo(Path(sample4geo_ckpt), device)

    def run(
        self,
        index_mode: IndexMode,
        query_mode: QueryMode,
        radii_km: list[float],
        fallback_policies: list[FallbackPolicy],
    ) -> dict:
        # 1. Build satellite gallery (cached)
        gallery, gal_s = build_satellite_gallery(
            self.sample4geo, self.data_root, index_mode,
            self.device, self.cache_dir,
            batch_size=self.batch_size, num_workers=self.num_workers,
        )

        # 2. Build ground query set
        paths, q_lids, q_coords = _build_ground_index(self.data_root, query_mode)
        print(f"[HybridPipeline] {query_mode} ground queries: {len(paths)}")

        # 3. GeoCLIP → rough GPS
        rough_gps, geoclip_s = predict_rough_gps(
            self.geoclip, paths, self.geoclip_gallery, batch_size=64,
        )

        # 4. Sample4Geo → ground embeddings
        q_embeds, s4g_s = embed_ground_queries(
            self.sample4geo, paths, self.device,
            batch_size=self.batch_size, num_workers=self.num_workers,
        )

        timings = PipelineTimings(
            geoclip_s=geoclip_s,
            s4g_embed_s=s4g_s,
            gallery_embed_s=gal_s,
            total_queries=len(paths),
        )

        # 5. Sweep radii × fallback policies
        runs: list[dict] = []
        for r_km in radii_km:
            for fb in fallback_policies:
                result = _rerank_and_score(
                    q_embeds, q_lids, q_coords, rough_gps, gallery,
                    radius_km=r_km, fallback=fb, device=self.device,
                    query_batch=64,
                )
                row = {
                    "index_mode": index_mode,
                    "query_mode": query_mode,
                    "radius_km": r_km,
                    "fallback": fb,
                    "mAP@1k": result.map_at_k,
                    **{f"R@{k}": v for k, v in result.recall.items()},
                    **{f"dist@{t}km": v for t, v in result.dist_at_km.items()},
                    "empty_rate": result.empty_rate,
                    "mean_candidates": result.mean_candidates,
                    "median_candidates": result.median_candidates,
                    "rerank_ms_per_query": result.rerank_ms_per_query,
                    "geoclip_ms_per_query": timings.per_query_ms()["geoclip_ms"],
                    "s4g_embed_ms_per_query": timings.per_query_ms()["s4g_embed_ms"],
                    "total_ms_per_query": (
                        timings.per_query_ms()["geoclip_ms"]
                        + timings.per_query_ms()["s4g_embed_ms"]
                        + result.rerank_ms_per_query
                    ),
                    "total_queries": result.total_queries,
                }
                runs.append(row)
                r_str = "inf" if r_km == float("inf") else f"{r_km:g}"
                print(
                    f"  [{index_mode}/{query_mode}] radius={r_str}km fb={fb:>13s} "
                    f"mAP@1k={result.map_at_k:.4f} R@1={result.recall[1]:.4f} "
                    f"R@5={result.recall[5]:.4f} R@10={result.recall[10]:.4f} "
                    f"empty={result.empty_rate:.3f} mean_cand={result.mean_candidates:.1f} "
                    f"rerank={result.rerank_ms_per_query:.2f}ms/q"
                )

        return {
            "index_mode": index_mode,
            "query_mode": query_mode,
            "gallery_size": len(gallery.hex_ids),
            "gallery_embed_s": gal_s,
            "geoclip_s_total": geoclip_s,
            "s4g_embed_s_total": s4g_s,
            "runs": runs,
        }


def full_sweep(
    data_root: Path,
    sample4geo_ckpt: Path,
    device: torch.device,
    cache_dir: Path,
    output_json: Path,
    radii_km: list[float] = DEFAULT_RADII_KM,
    index_modes: list[IndexMode] = ("query", "full"),
    query_modes: list[QueryMode] = ("one_per_landmark", "all"),
    fallback_policies: list[FallbackPolicy] = ("fail", "fallback_full"),
    num_workers: int = 4,
    batch_size: int = 128,
) -> dict:
    """Run the full matrix and write a single JSON report."""
    pipeline = HybridPipeline(
        data_root=data_root,
        sample4geo_ckpt=sample4geo_ckpt,
        device=device,
        cache_dir=cache_dir,
        num_workers=num_workers,
        batch_size=batch_size,
    )

    all_runs: list[dict] = []
    per_config: list[dict] = []
    for im in index_modes:
        for qm in query_modes:
            block = pipeline.run(im, qm, list(radii_km), list(fallback_policies))
            per_config.append({k: v for k, v in block.items() if k != "runs"})
            all_runs.extend(block["runs"])

    report = {
        "device": str(device),
        "sample4geo_ckpt": str(sample4geo_ckpt),
        "data_root": str(data_root),
        "geoclip_gallery_source": "paper",
        "radii_km": [r for r in radii_km],
        "per_config": per_config,
        "runs": all_runs,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(report, f, indent=2, default=_json_default)

    csv_path = output_json.with_suffix(".csv")
    pd.DataFrame(all_runs).to_csv(csv_path, index=False)
    print(f"\n[full_sweep] wrote {output_json}")
    print(f"[full_sweep] wrote {csv_path}")
    return report


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if o == float("inf"):
        return "inf"
    raise TypeError(f"Not JSON serializable: {type(o)}")
