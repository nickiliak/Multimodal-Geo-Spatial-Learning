from __future__ import annotations

from typing import Protocol

import numpy as np
from click import Path


class _LandmarkDataset(Protocol):
    landmark_ids: list[int]
    coords: dict[int, tuple[float, float]]


# ---------------------------------------------------------------------------

def _hex_path(data_root: Path, split: str, modality: str, hex_id: str) -> Path:
    """Build hex-sharded path: {split}/{modality}/{h[0]}/{h[1]}/{h[2]}/{h}.{ext}"""
    ext = "jpg" if modality == "ground" else "png"
    return data_root / split / modality / hex_id[0] / hex_id[1] / hex_id[2] / f"{hex_id}.{ext}"


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two (lat, lon) points."""
    R = 6_371_000.0
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    d_lat, d_lon = lat2 - lat1, lon2 - lon1
    a = np.sin(d_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(d_lon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def gps_to_satellite_image_pipe(lat: float, lon: float, radius_m: float, dataset: _LandmarkDataset) -> list[int]:
    """Return IDs of all landmark satellite images within radius_m metres of (lat, lon).

    Parameters
    ----------
    lat : float
        Predicted GPS latitude.
    lon : float
        Predicted GPS longitude.
    radius_m : float
        Search radius in metres.
    dataset : Load_satelite
        Loaded dataset providing landmark coordinates.

    Returns
    -------
    list[int]
        Landmark IDs whose GPS coords fall within the given radius.
    """
    closest_landmark_ids = [
        lid for lid in dataset.landmark_ids
        if _haversine_m(lat, lon, *dataset.coords[lid]) <= radius_m
    ]

    return closest_landmark_ids
