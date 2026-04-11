"""Evaluation utilities for GPS prediction tasks."""

from __future__ import annotations

import numpy as np

DEFAULT_THRESHOLDS_KM = [1, 25, 200, 750, 2500]

_EARTH_RADIUS_KM = 6371.0


def haversine(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """Compute great-circle distances in km between paired coordinates.

    Parameters
    ----------
    lat1, lon1 : array-like, shape (n,)
        First set of GPS coordinates in degrees.
    lat2, lon2 : array-like, shape (n,)
        Second set of GPS coordinates in degrees.

    Returns
    -------
    np.ndarray, shape (n,)
        Distances in kilometers.
    """
    lat1, lon1, lat2, lon2 = (
        np.radians(np.asarray(v, dtype=np.float64)) for v in (lat1, lon1, lat2, lon2)
    )
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def accuracy_at_thresholds(
    pred_lat: np.ndarray,
    pred_lon: np.ndarray,
    true_lat: np.ndarray,
    true_lon: np.ndarray,
    thresholds_km: list[float] | None = None,
) -> dict[float, float]:
    """Compute fraction of predictions within each distance threshold.

    Returns
    -------
    dict mapping threshold_km -> accuracy (0.0 to 1.0)
    """
    if thresholds_km is None:
        thresholds_km = DEFAULT_THRESHOLDS_KM
    distances = haversine(pred_lat, pred_lon, true_lat, true_lon)
    return {t: float(np.mean(distances <= t)) for t in thresholds_km}


def median_error(
    pred_lat: np.ndarray,
    pred_lon: np.ndarray,
    true_lat: np.ndarray,
    true_lon: np.ndarray,
) -> float:
    """Median localization error in km."""
    return float(np.median(haversine(pred_lat, pred_lon, true_lat, true_lon)))
