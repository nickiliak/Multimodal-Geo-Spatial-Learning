"""Zero-shot GeoClip baseline: gallery construction and batch inference.

Uses GeoCLIP's own functions where possible:
- `img_val_transform` for the training-matched ImageNet-normalized preprocessing
- `model.forward` for the image↔gallery similarity computation
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from geoclip import GeoCLIP
from geoclip.train.dataloader import img_val_transform
from PIL import Image
from tqdm import tqdm


def _patch_image_encoder(encoder: torch.nn.Module) -> None:
    """Unwrap ``BaseModelOutputWithPooling`` from newer transformers so the
    downstream MLP receives a tensor."""
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
        self._transform = img_val_transform()
        self._gallery_tensor: torch.Tensor | None = None
        self._gallery_coords: np.ndarray | None = None

    def build_gallery(self, coords: np.ndarray) -> None:
        """Register the GPS gallery for inference.

        The model re-encodes the gallery on every `forward` call, matching
        `GeoCLIP.predict`. We only cache the tensor.
        """
        self._gallery_coords = coords
        self._gallery_tensor = torch.tensor(coords, dtype=torch.float32).to(self.device)

    def predict_batch(
        self,
        image_paths: list[Path],
        batch_size: int = 64,
    ) -> np.ndarray:
        """Predict GPS for a list of images against the prebuilt gallery."""
        assert self._gallery_tensor is not None, "Call build_gallery() first"

        all_preds: list[np.ndarray] = []
        for start in tqdm(
            range(0, len(image_paths), batch_size),
            desc="Predicting",
            unit="batch",
        ):
            batch_paths = image_paths[start : start + batch_size]
            batch_tensors = torch.stack(
                [self._load_and_preprocess(p) for p in batch_paths]
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(batch_tensors, self._gallery_tensor)
                top1 = logits.softmax(dim=-1).argmax(dim=-1).cpu().numpy()
            all_preds.append(self._gallery_coords[top1])

        return np.concatenate(all_preds, axis=0)

    def _load_and_preprocess(self, image_path: Path) -> torch.Tensor:
        """Load a single image through GeoCLIP's training-matched transform."""
        img = Image.open(image_path).convert("RGB")
        return self._transform(img)


_TRAIN_CSV = Path("train") / "mml_train.csv"
_INDEX_CSV = Path("index") / "mml_index_satellite.csv"


def load_gallery_coords(data_root: Path, source: str = "index") -> np.ndarray:
    """Load GPS gallery coordinates as ``[[lat, lon], ...]``.

    ``source``:
    - ``"train"`` — 17,557 train-landmark GPS from ``train/mml_train.csv``.
    - ``"index"`` — ~100k index-satellite GPS from ``index/mml_index_satellite.csv``
      (paper Sec 5.2 protocol, default).
    - ``"both"`` — train concatenated with index (~118k).
    """
    def _load(rel: Path) -> np.ndarray:
        return pd.read_csv(data_root / rel)[["lat", "lon"]].values

    if source == "train":
        return _load(_TRAIN_CSV)
    if source == "index":
        return _load(_INDEX_CSV)
    if source == "both":
        return np.concatenate([_load(_TRAIN_CSV), _load(_INDEX_CSV)], axis=0)
    raise ValueError(f"source must be 'train' | 'index' | 'both', got {source!r}")


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
    n_images = sum(len(str(r).split()) for r in merged["images"])
    true_coords = np.zeros((n_images, 2))
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
    n_images = sum(len(str(r).split()) for r in merged["images"])
    true_coords = np.zeros((n_images, 2))
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