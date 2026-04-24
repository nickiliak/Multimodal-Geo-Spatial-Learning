"""Test gps_to_satellite_image_pipe: load query split, query a fixed point, plot results."""

from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from mmgeo.pipe_helpers import _hex_path, gps_to_satellite_image_pipe

DATA_ROOT = Path(__file__).parents[1] / "data" / "MML_Data"
SPLIT = "query"

# Sample query point and radius
QUERY_LAT = 40.7128   # New York City
QUERY_LON = -74.0060
RADIUS_M = 50_000     # 50 km


def _build_dataset(data_root: Path, split: str) -> SimpleNamespace:
    """Build a minimal dataset object from the real CSVs."""
    master_df = pd.read_csv(data_root / split / f"mml_{split}.csv")
    sat_df = pd.read_csv(data_root / split / f"mml_{split}_satellite.csv")

    sat_images: dict[int, list[str]] = {}
    for _, row in sat_df.iterrows():
        lid = int(row["landmark_id"])
        sat_images[lid] = str(row["images"]).split()

    coords: dict[int, tuple[float, float]] = {}
    landmark_ids: list[int] = []
    for _, row in master_df.iterrows():
        lid = int(row["landmark_id"])
        if lid in sat_images:
            landmark_ids.append(lid)
            coords[lid] = (float(row["lat"]), float(row["lon"]))

    return SimpleNamespace(
        landmark_ids=landmark_ids,
        coords=coords,
        sat_images=sat_images,
        data_root=data_root,
        split=split,
    )


def test_gps_to_satellite_image_pipe_and_plot():
    dataset = _build_dataset(DATA_ROOT, SPLIT)

    candidate_ids = gps_to_satellite_image_pipe(QUERY_LAT, QUERY_LON, RADIUS_M, dataset)

    print(f"Found {len(candidate_ids)} landmarks within {RADIUS_M/1000:.0f} km of ({QUERY_LAT}, {QUERY_LON})")
    assert isinstance(candidate_ids, list), "Expected a list of landmark IDs"

    ids_to_plot = candidate_ids[:10]

    n = len(ids_to_plot)
    if n == 0:
        print("No candidates found — nothing to plot.")
        return

    cols = min(n, 5)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = [axes] if n == 1 else list(axes.flat) if rows > 1 else list(axes)

    fig.suptitle(
        f"Closest satellite landmarks\nQuery: ({QUERY_LAT}°N, {QUERY_LON}°E)  radius={RADIUS_M/1000:.0f} km",
        fontsize=13,
    )

    for ax, lid in zip(axes, ids_to_plot):
        hex_id = dataset.sat_images[lid][0]
        img_path = _hex_path(dataset.data_root, dataset.split, "satellite", hex_id)
        img = Image.open(img_path).convert("RGB")
        lat, lon = dataset.coords[lid]
        ax.imshow(img)
        ax.set_title(f"ID {lid}\n({lat:.4f}, {lon:.4f})", fontsize=8)
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("test_helper_output.png", dpi=150)
    plt.show()
    print("Plot saved to test_helper_output.png")


if __name__ == "__main__":
    test_gps_to_satellite_image_pipe_and_plot()
