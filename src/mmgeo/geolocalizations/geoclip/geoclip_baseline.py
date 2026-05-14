"""Zero-shot GeoClip baseline: gallery construction and batch inference.

Uses GeoCLIP's own functions where possible:
- `model.image_encoder.preprocess_image` (CLIPProcessor) for preprocessing
  — matches what the pretrained weights were actually trained with;
  `img_val_transform` (ImageNet norm) was tested and is strictly worse.
- `model.forward` for the image↔gallery similarity computation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from geoclip import GeoCLIP
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

import torch.nn as nn
import torch.nn.functional as F

class newGeoCLIP(GeoCLIP):
    """GeoCLIP variant that uses a transformer architecture to encode multiple images per landmark"""

    def __init__(self, device: str = "cuda", transformer=True,load_weights=True):
        super().__init__(from_pretrained=load_weights)
        self.device = torch.device(device)
        self.to(self.device)

        if transformer:
            #Transformer with 4 layers and 8 heads
            encoder_layer = torch.nn.TransformerEncoderLayer(d_model=512, nhead=8,device=self.device)
            self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=4)
            self.cls_token = nn.Parameter(torch.randn(1, 512, device=self.device)) # CLS token for aggregating image features)
        else:
            #We combine them based on how similar they are
            #We calculate a score based on cosine similarity to the mean, 
            # then softmax, and then weight the embeddings accordingly
            pass
    
    def forward(self, image, location, landmark_id):
        """ GeoCLIP's forward pass

        Args:
            image (torch.Tensor): Image tensor of shape (n, 3, 224, 224)
            location (torch.Tensor): GPS location tensor of shape (m, 2)
            landmark_id (torch.Tensor): Landmark ID tensor of shape (n,)

        Returns:
            logits_per_landmark (torch.Tensor): Logits per landmark of shape (num_landmarks, m)
        """

        raw_features = self.image_encoder(image)
        unique_landmark_ids,inverse_indices = torch.unique(landmark_id,return_inverse=True)
        num_landmarks = unique_landmark_ids.size(0)
        counts = torch.bincount(inverse_indices).view(-1,1) # (Num_Landmarks, 1)
        max_batch_images = counts.max().item() # Max number of images for any landmark ID in the batch / Masked attention max

        #Combine images features here using either transformer or similarity-based weighting
        if hasattr(self, 'transformer'):
            # Build a padded (num_landmarks, max_batch_images, 512) tensor by scattering
            # raw_features into their landmark's slot. position_in_group gives each image
            # a 0..count-1 index within its landmark group.
            position_in_group = torch.zeros_like(inverse_indices)
            running = torch.zeros(num_landmarks, dtype=torch.long, device=self.device)
            for i in range(inverse_indices.size(0)):
                lid = inverse_indices[i]
                position_in_group[i] = running[lid]
                running[lid] += 1

            padded = torch.zeros((num_landmarks, max_batch_images, 512), device=self.device)
            padded[inverse_indices, position_in_group] = raw_features

            # Prepend CLS token: (num_landmarks, 1+max_batch_images, 512)
            cls = self.cls_token.unsqueeze(0).expand(num_landmarks, -1, -1)
            seq = torch.cat([cls, padded], dim=1)

            # key_padding_mask: True where token should be ignored.
            # CLS is always valid; image slot j is valid iff j < counts[landmark].
            arange = torch.arange(max_batch_images, device=self.device).unsqueeze(0)
            image_pad_mask = arange >= counts  # (num_landmarks, max_batch_images)
            cls_mask = torch.zeros((num_landmarks, 1), dtype=torch.bool, device=self.device)
            key_padding_mask = torch.cat([cls_mask, image_pad_mask], dim=1)

            # TransformerEncoderLayer defaults to (S, N, E); transpose accordingly.
            seq = seq.transpose(0, 1)
            out = self.transformer(seq, src_key_padding_mask=key_padding_mask)
            image_features = out[0]  # CLS output, (num_landmarks, 512)

        else:
            #Calculate the mean embedding for each landmark ID
            group_sum = torch.zeros((num_landmarks, 512), device=self.device)
            group_sum.index_add_(0, inverse_indices, raw_features)
            group_means = group_sum / counts # (Num_Landmarks, 512)
            expanded_means = group_means[inverse_indices] # (n, 512)

            #Calculate cosine similarity of each image embedding to the mean embedding of its landmark ID
            similarities = F.cosine_similarity(raw_features, expanded_means, dim=1)

            #Calculate a weight for each image embedding based on the cosine similarity (e.g. softmax)
            tau = 0.1  # softmax temperature; low → sharp weighting (sweep candidate)
            exp_sim = torch.exp(similarities / tau)
            sum_exp = torch.zeros((num_landmarks,), device=self.device)
            sum_exp.index_add_(0, inverse_indices, exp_sim)
            weights = exp_sim / sum_exp[inverse_indices]

            #Weight the image embeddings accordingly and sum to get a single embedding per landmark ID
            weighted_features = raw_features * weights.unsqueeze(1)
            image_features = torch.zeros((num_landmarks, 512), device=self.device)
            image_features.index_add_(0, inverse_indices, weighted_features)

        location_features = self.location_encoder(location)
        logit_scale = self.logit_scale.exp()
        
        # Normalize features
        image_features = F.normalize(image_features, dim=1)
        location_features = F.normalize(location_features, dim=1)
        
        # Cosine similarity (Image Features & Location Features)
        logits_per_landmark = logit_scale * (image_features @ location_features.t())

        return logits_per_landmark



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
        """Load a single image through GeoCLIP's CLIPProcessor preprocessing."""
        img = Image.open(image_path).convert("RGB")
        return self.model.image_encoder.preprocess_image(img).squeeze(0)


class NewGeoClipBaseline:
    """Inference wrapper around `newGeoCLIP` that aggregates multiple ground
    images per landmark via a transformer encoder before scoring against a
    GPS gallery.

    Assumes `image_paths` and `landmark_ids` are sorted by landmark, so each
    landmark's images form a contiguous run. Batches are formed along landmark
    boundaries — a single landmark is never split across batches.

    One prediction is returned per *unique* landmark, in the order landmarks
    first appear in the input.
    """

    def __init__(self, device: str = "cuda", transformer: bool = True) -> None:
        self.device = torch.device(device)
        self.model = newGeoCLIP(device=device, transformer=transformer)
        _patch_image_encoder(self.model.image_encoder)
        self.model.to(self.device)
        self.model.eval()
        self._gallery_tensor: torch.Tensor | None = None
        self._gallery_coords: np.ndarray | None = None

    def build_gallery(self, coords: np.ndarray) -> None:
        self._gallery_coords = coords
        self._gallery_tensor = torch.tensor(coords, dtype=torch.float32).to(self.device)

    def predict_batch(
        self,
        image_paths: list[Path],
        landmark_ids: np.ndarray,
        batch_size: int = 64,
    ) -> np.ndarray:
        """Predict GPS for a list of images grouped by landmark.

        Returns one ``[lat, lon]`` per unique landmark, in first-appearance order.
        """
        assert self._gallery_tensor is not None, "Call build_gallery() first"
        assert len(image_paths) == len(landmark_ids), (
            "image_paths and landmark_ids must align"
        )

        # Find contiguous landmark group boundaries.
        lids = np.asarray(landmark_ids)
        change = np.flatnonzero(np.r_[True, lids[1:] != lids[:-1]])
        group_starts = np.r_[change, len(lids)]  # n_groups + 1

        # Pack groups into batches without splitting a group.
        batches: list[tuple[int, int]] = []  # (start_idx, end_idx) into image_paths
        i = 0
        while i < len(change):
            start = change[i]
            j = i
            while (
                j + 1 < len(change)
                and group_starts[j + 1] - start <= batch_size
            ):
                j += 1
            end = group_starts[j + 1]
            batches.append((start, end))
            i = j + 1

        all_preds: list[np.ndarray] = []
        for start, end in tqdm(batches, desc="Predicting", unit="batch"):
            batch_paths = image_paths[start:end]
            batch_lids = torch.as_tensor(
                lids[start:end], dtype=torch.long, device=self.device
            )
            batch_tensors = torch.stack(
                [self._load_and_preprocess(p) for p in batch_paths]
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(
                    batch_tensors, self._gallery_tensor, batch_lids
                )  # (n_landmarks_in_batch, n_gallery)
                top1 = logits.softmax(dim=-1).argmax(dim=-1).cpu().numpy()
            all_preds.append(self._gallery_coords[top1])

        return np.concatenate(all_preds, axis=0)

    def _load_and_preprocess(self, image_path: Path) -> torch.Tensor:
        img = Image.open(image_path).convert("RGB")
        return self.model.image_encoder.preprocess_image(img).squeeze(0)


_TRAIN_CSV = Path("train") / "mml_train.csv"
_INDEX_CSV = Path("index") / "mml_index_satellite.csv"
_QUERY_CSV = Path("query") / "mml_query.csv"


def load_gallery_coords(data_root: Path, source: str = "paper") -> np.ndarray:
    """Load GPS gallery coordinates as ``[[lat, lon], ...]``.

    ``source``:
    - ``"train"`` — 17,557 train-landmark GPS.
    - ``"index"`` — ~100k index-satellite GPS (honest "in-the-wild" gallery).
    - ``"paper"`` — ~100k index + 1,000 query-landmark GPS = ~101k. Matches the
      camera-ready MML paper Sec 5.2 protocol and reproduces the 21.37 % @1 km
      number. The query GPS being in the gallery makes this an *upper bound*.
    - ``"both"`` — train + index (~118k).
    """
    def _load(rel: Path) -> np.ndarray:
        return pd.read_csv(data_root / rel)[["lat", "lon"]].values

    if source == "train":
        return _load(_TRAIN_CSV)
    if source == "index":
        return _load(_INDEX_CSV)
    if source == "paper":
        return np.concatenate([_load(_INDEX_CSV), _load(_QUERY_CSV)], axis=0)
    if source == "both":
        return np.concatenate([_load(_TRAIN_CSV), _load(_INDEX_CSV)], axis=0)
    raise ValueError(
        f"source must be 'train' | 'index' | 'paper' | 'both', got {source!r}"
    )


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
    landmark_idsmerged = merged["landmark_id"].values
    image_paths: list[Path] = []
    n_images = sum(len(str(r).split()) for r in merged["images"])
    true_coords = np.zeros((n_images, 2))
    landmark_ids = np.zeros((n_images,), dtype=int)
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
            landmark_ids[len(image_paths)-1] = landmark_idsmerged[j]

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
    landmark_idsmerged = merged["landmark_id"].values
    image_paths: list[Path] = []
    n_images = sum(len(str(r).split()) for r in merged["images"])
    true_coords = np.zeros((n_images, 2))
    landmark_ids = np.zeros((n_images,), dtype=int)
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
            landmark_ids[len(image_paths)-1] = landmark_idsmerged[j]

    return image_paths, true_coords, landmark_ids