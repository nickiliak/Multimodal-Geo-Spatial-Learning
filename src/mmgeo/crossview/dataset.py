"""MMLandmarks cross-view dataset and sampling strategies.

Adapts Sample4Geo's approach to MMLandmarks' instance-level structure:
- Each landmark has multiple ground images and multiple satellite images
- Training pairs are (ground, satellite) from the SAME landmark
- Custom samplers ensure no duplicate landmarks per batch (required for InfoNCE)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms


# ---------------------------------------------------------------------------
# Image path helpers (match existing repo convention from geoclip_baseline.py)
# ---------------------------------------------------------------------------

def _hex_path(data_root: Path, split: str, modality: str, hex_id: str) -> Path:
    """Build hex-sharded path: {split}/{modality}/{h[0]}/{h[1]}/{h[2]}/{h}.{ext}"""
    ext = "jpg" if modality == "ground" else "png"
    return data_root / split / modality / hex_id[0] / hex_id[1] / hex_id[2] / f"{hex_id}.{ext}"


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_train_transforms(img_size: int = 384) -> transforms.Compose:
    """Training augmentations for both ground and satellite images."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),                                        # ← move ToTensor before RandomErasing
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),         # ← RandomErasing after ToTensor
    ])


def get_eval_transforms(img_size: int = 384) -> transforms.Compose:
    """Deterministic transforms for evaluation."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MMLCrossViewDataset(Dataset):
    """Cross-view training dataset for MMLandmarks.

    Each sample returns a (ground_image, satellite_image) pair from the same
    landmark. Images are randomly sampled per landmark each epoch, providing
    natural augmentation across epochs.

    Parameters
    ----------
    data_root : Path
        Root data directory (``data/MML_Data``).
    split : str
        Dataset split: ``"train"`` or ``"query"``.
    transform_ground : callable, optional
        Transform for ground images.
    transform_sat : callable, optional
        Transform for satellite images.
    """

    def __init__(
        self,
        data_root: Path,
        split: str = "train",
        transform_ground: transforms.Compose | None = None,
        transform_sat: transforms.Compose | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split

        self.transform_ground = transform_ground or get_train_transforms()
        self.transform_sat = transform_sat or get_train_transforms()

        # Load master CSV for landmark metadata
        master_df = pd.read_csv(self.data_root / split / f"mml_{split}.csv")
        ground_df = pd.read_csv(self.data_root / split / f"mml_{split}_ground.csv")
        sat_df = pd.read_csv(self.data_root / split / f"mml_{split}_satellite.csv")

        # Build per-landmark image lists
        # CSV format: landmark_id | images (space-separated hex IDs)
        self.landmark_ids: list[int] = []
        self.ground_images: dict[int, list[str]] = {}
        self.sat_images: dict[int, list[str]] = {}
        self.coords: dict[int, tuple[float, float]] = {}

        # Parse ground images per landmark
        for _, row in ground_df.iterrows():
            lid = int(row["landmark_id"])
            hex_ids = str(row["images"]).split()
            self.ground_images[lid] = hex_ids

        # Parse satellite images per landmark
        for _, row in sat_df.iterrows():
            lid = int(row["landmark_id"])
            hex_ids = str(row["images"]).split()
            self.sat_images[lid] = hex_ids

        # Store GPS coords and build final landmark list
        # Only include landmarks that have BOTH ground and satellite images
        for _, row in master_df.iterrows():
            lid = int(row["landmark_id"])
            if lid in self.ground_images and lid in self.sat_images:
                self.landmark_ids.append(lid)
                self.coords[lid] = (float(row["lat"]), float(row["lon"]))

        print(
            f"[MMLCrossViewDataset] split={split}, "
            f"landmarks={len(self.landmark_ids)}, "
            f"ground_imgs={sum(len(v) for v in self.ground_images.values())}, "
            f"sat_imgs={sum(len(v) for v in self.sat_images.values())}"
        )

    def __len__(self) -> int:
        return len(self.landmark_ids)

    def __getitem__(self, idx: int) -> dict:
        """Return a ground-satellite pair from the same landmark.

        Returns dict with keys: ground_img, sat_img, landmark_id, lat, lon
        """
        lid = self.landmark_ids[idx]

        # Randomly pick one ground and one satellite image
        g_hex = np.random.choice(self.ground_images[lid])
        s_hex = np.random.choice(self.sat_images[lid])

        g_path = _hex_path(self.data_root, self.split, "ground", g_hex)
        s_path = _hex_path(self.data_root, self.split, "satellite", s_hex)

        g_img = Image.open(g_path).convert("RGB")
        s_img = Image.open(s_path).convert("RGB")

        g_img = self.transform_ground(g_img)
        s_img = self.transform_sat(s_img)

        lat, lon = self.coords[lid]
        return {
            "ground_img": g_img,
            "sat_img": s_img,
            "landmark_id": lid,
            "lat": lat,
            "lon": lon,
        }

    def get_all_coords(self) -> np.ndarray:
        """Return (N, 2) array of [lat, lon] for all landmarks, in order."""
        return np.array([self.coords[lid] for lid in self.landmark_ids])


# ---------------------------------------------------------------------------
# Evaluation dataset (loads all images, not just one per landmark)
# ---------------------------------------------------------------------------

class MMLImageDataset(Dataset):
    """Load all images from a single modality for embedding extraction.

    Used at evaluation time to embed all query/index images.

    Parameters
    ----------
    data_root : Path
        Root data directory.
    split : str
        ``"query"`` or ``"index"``.
    modality : str
        ``"ground"`` or ``"satellite"``.
    transform : callable, optional
    """

    def __init__(
        self,
        data_root: Path,
        split: str,
        modality: str,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.modality = modality
        self.transform = transform or get_eval_transforms()

        csv_path = self.data_root / split / f"mml_{split}_{modality}.csv"
        df = pd.read_csv(csv_path)

        # Explode space-separated hex IDs into individual rows
        self.image_ids: list[str] = []
        self.landmark_ids: list[int] = []

        for _, row in df.iterrows():
            lid = int(row["landmark_id"])
            for hex_id in str(row["images"]).split():
                self.image_ids.append(hex_id)
                self.landmark_ids.append(lid)

        self.landmark_ids_array = np.array(self.landmark_ids)
        self._split = split
        print(
            f"[MMLImageDataset] split={split}, modality={modality}, "
            f"images={len(self.image_ids)}, landmarks={len(df)}"
        )

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        hex_id = self.image_ids[idx]
        path = _hex_path(self.data_root, self._split, self.modality, hex_id)
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, self.landmark_ids[idx]


# ---------------------------------------------------------------------------
# Unique-landmark batch sampler (prevents duplicate landmarks in a batch)
# ---------------------------------------------------------------------------

class UniqueLandmarkSampler(Sampler):
    """Ensures each batch contains at most one sample per landmark.

    Required for InfoNCE: if two samples share a landmark_id, the loss
    would incorrectly treat a true positive as a negative.

    Since MMLandmarks has 17,557 train landmarks and typical batch sizes
    are 64-128, collisions are unlikely with random shuffling, but this
    sampler guarantees correctness.
    """

    def __init__(self, dataset: MMLCrossViewDataset, batch_size: int) -> None:
        self.n = len(dataset)
        self.batch_size = batch_size

    def __iter__(self):
        # Simple random permutation — with 17K landmarks and batch_size 128,
        # each batch of consecutive indices maps to unique landmarks
        perm = torch.randperm(self.n).tolist()
        # Yield batches (drop last incomplete batch)
        for i in range(0, self.n - self.batch_size + 1, self.batch_size):
            yield perm[i : i + self.batch_size]

    def __len__(self) -> int:
        return self.n // self.batch_size
