from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class MMLDataset(Dataset):
    """Image + GPS dataset for MMlandmarks ground images.

    Loads and preprocesses each image in ``__getitem__`` so that
    ``DataLoader(num_workers>0)`` can prefetch ahead of GPU compute.

    Parameters
    ----------
    paths : list[Path]
        Absolute paths to ground images.
    coords : np.ndarray, shape (n, 2)
        Corresponding ``[[lat, lon], ...]`` coordinates.
    transform : Callable
        Preprocessing function applied to each PIL image, e.g.
        ``baseline.model.image_encoder.preprocess_image``.
    """

    def __init__(
        self,
        paths: list[Path],
        coords: np.ndarray,
        transform: Callable,
    ) -> None:
        self.paths = [str(p) for p in paths]
        self.coords = coords.astype(np.float32)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        img = Image.open(self.paths[idx]).convert("RGB")
        img_tensor = self.transform(img).squeeze(0)
        return img_tensor, torch.from_numpy(self.coords[idx])
