"""Zero-shot GeoClip baseline: gallery construction and batch inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from geoclip import GeoCLIP
from PIL import Image
from tqdm import tqdm


def _patch_image_encoder(encoder: torch.nn.Module) -> None:
    """Fix for transformers>=5 where CLIP.get_image_features returns an object."""
    original_forward = encoder.forward

    def patched_forward(x: torch.Tensor) -> torch.Tensor:
        clip_out = encoder.CLIP.get_image_features(pixel_values=x)
        if not isinstance(clip_out, torch.Tensor):
            clip_out = clip_out.pooler_output
        return encoder.mlp(clip_out)

    encoder.forward = patched_forward


class GeoClipBaseline:
    """Zero-shot GeoClip inference against a custom GPS gallery.

    Parameters
    ----------
    device : str
        Torch device string (``"cuda"`` or ``"cpu"``).
    """

    def __init__(self, device: str = "cuda") -> None:
        self.device = torch.device(device)
        self.model = GeoCLIP(from_pretrained=True)
        _patch_image_encoder(self.model.image_encoder)
        self.model.to(self.device)
        self.model.eval()
        self._gallery_features: torch.Tensor | None = None
        self._gallery_coords: np.ndarray | None = None

    def build_gallery(self, coords: np.ndarray) -> None:
        """Precompute location embeddings for the GPS gallery.

        Parameters
        ----------
        coords : np.ndarray, shape (m, 2)
            Gallery GPS coordinates as ``[[lat, lon], ...]``.
        """
        self._gallery_coords = coords
        gps_tensor = torch.tensor(coords, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            self._gallery_features = self.model.location_encoder(gps_tensor)
            self._gallery_features = F.normalize(self._gallery_features, dim=-1)

    def predict_batch(
        self,
        image_paths: list[Path],
        batch_size: int = 64,
    ) -> np.ndarray:
        """Predict GPS for a list of images against the prebuilt gallery.

        Returns
        -------
        np.ndarray, shape (n, 2)
            Predicted ``[[lat, lon], ...]`` for each image.
        """
        assert self._gallery_features is not None, "Call build_gallery() first"

        all_preds: list[np.ndarray] = []
        for start in tqdm(
            range(0, len(image_paths), batch_size),
            desc="Predicting",
            unit="batch",
        ):
            batch_paths = image_paths[start : start + batch_size]
            batch_tensors = torch.cat(
                [self._load_and_preprocess(p) for p in batch_paths], dim=0
            ).to(self.device)

            with torch.no_grad():
                img_features = self.model.image_encoder(batch_tensors)
                img_features = F.normalize(img_features, dim=-1)

            sims = img_features @ self._gallery_features.T
            top1_indices = sims.argmax(dim=1).cpu().numpy()
            all_preds.append(self._gallery_coords[top1_indices])

        return np.concatenate(all_preds, axis=0)

    def _load_and_preprocess(self, image_path: Path) -> torch.Tensor:
        """Load a single image and return the preprocessed tensor."""
        img = Image.open(image_path).convert("RGB")
        return self.model.image_encoder.preprocess_image(img)


def load_gallery_coords(
    data_root: Path,
    include_index: bool = False,
) -> np.ndarray:
    """Load GPS gallery coordinates from train (and optionally index) CSVs.

    Parameters
    ----------
    data_root : Path
        Root data directory (``data/MML_Data``).
    include_index : bool
        If ``True``, append 101,302 index satellite GPS points.

    Returns
    -------
    np.ndarray, shape (m, 2)
    """
    train_df = pd.read_csv(data_root / "train" / "mml_train.csv")
    coords = train_df[["lat", "lon"]].values

    if include_index:
        index_df = pd.read_csv(data_root / "index" / "mml_index_satellite.csv")
        index_coords = index_df[["lat", "lon"]].values
        coords = np.concatenate([coords, index_coords], axis=0)

    return coords


def load_query_data(
    data_root: Path,
) -> tuple[list[Path], np.ndarray, np.ndarray]:
    """Load query image paths, ground-truth coordinates, and landmark IDs.

    Picks the first ground image per query landmark. Image paths use the
    3-level hex-prefix sharding scheme: ``ground/{h[0]}/{h[1]}/{h[2]}/{h}.jpg``.

    Returns
    -------
    image_paths : list[Path]
        One image path per query landmark.
    true_coords : np.ndarray, shape (n, 2)
        Ground-truth ``[[lat, lon], ...]``.
    landmark_ids : np.ndarray, shape (n,)
    """
    query_df = pd.read_csv(data_root / "query" / "mml_query.csv")
    ground_df = pd.read_csv(data_root / "query" / "mml_query_ground.csv")
    merged = query_df.merge(ground_df, on="landmark_id")

    true_coordsmerged = merged[["lat", "lon"]].values
    image_paths: list[Path] = []
    true_coords = np.zeros((18688,2))
    for j, row in merged.iterrows():
        
        for i in range(len(str(row["images"]).split())):
            hex_id = str(row["images"]).split()[i]
            path = (
                data_root
                / "query"
                / "ground"
                / hex_id[0]
                / hex_id[1]
                / hex_id[2]
                / f"{hex_id}.jpg"
            )
            image_paths.append(path)
            true_coords[len(image_paths)-1,:] = true_coordsmerged[j,:]

    landmark_ids = merged["landmark_id"].values
    return image_paths, true_coords, landmark_ids


def load_train_data(
    data_root: Path,
) -> tuple[list[Path], np.ndarray, np.ndarray]:
    """Load train image paths, ground-truth coordinates, and landmark IDs.

    Picks the first ground image per train landmark. Image paths use the
    3-level hex-prefix sharding scheme: ``ground/{h[0]}/{h[1]}/{h[2]}/{h}.jpg``.

    Returns
    -------
    image_paths : list[Path]
        One image path per train landmark.
    true_coords : np.ndarray, shape (n, 2)
        Ground-truth ``[[lat, lon], ...]``.
    landmark_ids : np.ndarray, shape (n,)
    """
    train_df = pd.read_csv(data_root / "train" / "mml_train.csv")
    ground_df = pd.read_csv(data_root / "train" / "mml_train_ground.csv")
    merged = train_df.merge(ground_df, on="landmark_id")

    true_coordsmerged = merged[["lat", "lon"]].values
    image_paths: list[Path] = []
    true_coords = np.zeros((18688,2))
    for j, row in merged.iterrows():
        
        for i in range(len(str(row["images"]).split())):
            hex_id = str(row["images"]).split()[i]
            path = (
                data_root
                / "train"
                / "ground"
                / hex_id[0]
                / hex_id[1]
                / hex_id[2]
                / f"{hex_id}.jpg"
            )
            image_paths.append(path)
            true_coords[len(image_paths)-1,:] = true_coordsmerged[j,:]

    landmark_ids = merged["landmark_id"].values
    return image_paths, true_coords, landmark_ids