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

# ---------------------------------------------------------------------------
# Milestone 1 — Dataset & Transforms
# ---------------------------------------------------------------------------
# Read: paper §4.2 (input sizes, augmentations)
# Reference for data loading: geoclip/geoclip_baseline.py::load_train_data
# ---------------------------------------------------------------------------


def _hex_image_path(data_root: Path, split: str, view: str, hex_id: str) -> Path:
    """Return the sharded image path for a given hex image ID.

    Pattern: <data_root>/<split>/<view>/<h[0]>/<h[1]>/<h[2]>/<hex_id>.<ext>
    This is the same 3-level hex-prefix sharding used by the GeoClip baseline.
    Ground images are .jpg, satellite images are .png.
    """
    ext = "png" if view == "satellite" else "jpg"
    return data_root / split / view / hex_id[0] / hex_id[1] / hex_id[2] / f"{hex_id}.{ext}"


def get_transforms(view: str, split: str = "train") -> transforms.Compose:
    """Build torchvision transforms for a given view and split.

    Parameters
    ----------
    view : "ground" | "satellite"
    split : "train" | "val"

    Paper §4.2 target sizes
    -----------------------
    - Ground:    140 × 768  (H × W)
    - Satellite: 384 × 384

    Paper §4.2 train augmentations (apply to BOTH views synchronously)
    -------------------------------------------------------------------
    - Horizontal flip
    - Rotation  (satellite only — street-view shift is equivalent)
    - Grid / coarse dropout
    - Colour jitter

    Hint: for val, use only Resize + CenterCrop + ToTensor + Normalize.
    ImageNet stats: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]

    Verify: check output tensor shape matches target H×W.
    """
    if view == "ground":
        target_size = (140, 768)
    elif view == "satellite":
        target_size = (384, 384)
    else:
        raise ValueError(f"Invalid view: {view}")

    if split == "train":
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15) if view == "satellite" else transforms.Lambda(lambda x: x),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomApply([transforms.RandomErasing()], p=0.5),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    return transform    


    


class MMLDataset(Dataset):
    """Paired ground-view / satellite-view dataset for MML Landmarks.

    Each item is one landmark, represented by one ground image and one satellite image.

    CSV files used
    --------------
    train/mml_train.csv            — landmark_id, lat, lon
    train/mml_train_ground.csv     — landmark_id, images  (space-separated hex IDs) 
    train/mml_train_satellite.csv  — landmark_id, images

    Same structure for split="query" (use mml_query_*.csv files).

    Hint: the `images` column contains *space-separated* hex IDs — take the first one,
    exactly like geoclip_baseline.py::load_train_data does for ground images.
    """

    def __init__(
        self,
        data_root: Path,
        split: str = "train",
        ground_transform: transforms.Compose | None = None,
        satellite_transform: transforms.Compose | None = None,
    ) -> None:
        # TODO (Milestone 1)
        # 1. Load the three CSVs and merge on landmark_id.
        # 2. Store one (ground_hex_id, satellite_hex_id, lat, lon, landmark_id) row per item.
        # 3. Store data_root, split, and transforms as instance attributes.
        self.data_root = data_root
        self.split = split
        self.ground_transform = ground_transform
        self.satellite_transform = satellite_transform
        self.data = pd.DataFrame()  # Placeholder for merged CSV data
        data_root = Path(data_root)

        query_df = pd.read_csv(data_root / split / f"mml_{split}.csv")
        ground_df = pd.read_csv(data_root / split / f"mml_{split}_ground.csv")
        satellite_df = pd.read_csv(data_root / split / f"mml_{split}_satellite.csv")  

        merged = query_df.merge(ground_df, on="landmark_id").merge(satellite_df, on="landmark_id")
        self.data = merged[["landmark_id", "lat", "lon", "images_x", "images_y"]].copy()
        self.data.columns = ["landmark_id", "lat", "lon", "ground_hex_id", "satellite_hex_id"]

    def __len__(self) -> int:
        
        """Return the number of items in the dataset."""
        return len(self.data)
    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, float, float, int]:
        """Return (ground_image, satellite_image, lat, lon, landmark_id).

        Hint: open images with PIL.Image.open(...).convert("RGB"), then apply transforms.
        Use _hex_image_path() to resolve the file path.
        """
        row = self.data.iloc[idx]
        landmark_id = row["landmark_id"]
        lat = row["lat"]
        lon = row["lon"]
        ground_hex_id = row["ground_hex_id"].split()[0]  # Take the first hex ID
        satellite_hex_id = row["satellite_hex_id"].split()[0]

        ground_image_path = _hex_image_path(self.data_root, self.split, "ground", ground_hex_id)
        satellite_image_path = _hex_image_path(self.data_root, self.split, "satellite", satellite_hex_id)

        ground_image = Image.open(ground_image_path).convert("RGB")
        satellite_image = Image.open(satellite_image_path).convert("RGB")

        if self.ground_transform:
            ground_image = self.ground_transform(ground_image)
        if self.satellite_transform:
            satellite_image = self.satellite_transform(satellite_image)

        return ground_image, satellite_image, lat, lon, landmark_id

if __name__ == "__main__":
    # Quick test: instantiate the dataset and check an item
    data_root = Path(__file__).resolve().parents[4] / "data" / "MML_Data"  # symlink to /dtu/blackhole/02/137570/MML
    ground_transform = get_transforms("ground", "train")
    satellite_transform = get_transforms("satellite", "train")
    dataset = MMLDataset(data_root, split="train", ground_transform=ground_transform, satellite_transform=satellite_transform)
    print(f"Dataset size: {len(dataset)}")
    ground_image, satellite_image, lat, lon, landmark_id = dataset[0]
    print(f"Ground image shape: {ground_image.shape}, Satellite image shape: {satellite_image.shape}, Lat: {lat}, Lon: {lon}, Landmark ID: {landmark_id}")