"""Training script for newGeoCLIP's transformer aggregator.

Freezes all GeoCLIP weights (image encoder, location encoder, logit scale)
and trains only the transformer encoder + CLS token that aggregates
multiple ground images per landmark into a single embedding.

The contrastive loss matches each landmark's aggregated image embedding
against its ground-truth GPS, with other landmarks in the batch as negatives.

Run from repo root:
    uv run --no-sync python -m mmgeo.geolocalizations.geoclip.train_new_geo
or via scripts/train_new_geo.sh.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

from mmgeo.geolocalizations.geoclip.geoclip_baseline import (
    NewGeoClipBaseline,
    _patch_image_encoder,
    load_gallery_coords,
    load_query_data,
    load_train_data,
    newGeoCLIP,
)


class LandmarkImageDataset(Dataset):
    """Yields (image, coord, landmark_id) per ground image."""

    def __init__(self, paths, coords, landmark_ids, transform):
        self.paths = [str(p) for p in paths]
        self.coords = coords.astype(np.float32)
        self.landmark_ids = np.asarray(landmark_ids, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        img_tensor = self.transform(img).squeeze(0)
        return (
            img_tensor,
            torch.from_numpy(self.coords[idx]),
            int(self.landmark_ids[idx]),
        )


class LandmarkGroupSampler(Sampler[list[int]]):
    """Yields batches of indices grouped by landmark.

    Each batch contains every image of `landmarks_per_batch` randomly chosen
    landmarks. Landmark order is reshuffled every epoch; within a landmark,
    image order is preserved (the transformer is permutation-invariant).
    """

    def __init__(self, landmark_ids: np.ndarray, landmarks_per_batch: int, shuffle: bool = True):
        self.landmarks_per_batch = landmarks_per_batch
        self.shuffle = shuffle
        self._groups: list[np.ndarray] = []
        unique = np.unique(landmark_ids)
        for lid in unique:
            self._groups.append(np.flatnonzero(landmark_ids == lid))

    def __iter__(self) -> Iterator[list[int]]:
        order = np.arange(len(self._groups))
        if self.shuffle:
            np.random.shuffle(order)
        for start in range(0, len(order) - self.landmarks_per_batch + 1, self.landmarks_per_batch):
            chosen = order[start : start + self.landmarks_per_batch]
            batch: list[int] = []
            for g in chosen:
                batch.extend(int(i) for i in self._groups[g])
            yield batch

    def __len__(self) -> int:
        return len(self._groups) // self.landmarks_per_batch


def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    coords = torch.stack([b[1] for b in batch])
    lids = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return imgs, coords, lids


def freeze_backbone(model: newGeoCLIP) -> tuple[int, int]:
    """Freeze everything except the transformer aggregator + CLS token.

    Returns (trainable_params, total_params).
    """
    for p in model.parameters():
        p.requires_grad = False
    if hasattr(model, "transformer"):
        for p in model.transformer.parameters():
            p.requires_grad = True
    if hasattr(model, "cls_token"):
        model.cls_token.requires_grad = True

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def train_step(model: newGeoCLIP, imgs, coords, lids, device) -> torch.Tensor:
    """One contrastive step.

    `forward` aggregates per landmark and returns logits of shape
    (num_landmarks, num_images_in_batch). The diagonal positives are at
    indices given by sorted-unique-landmark-id order, so we sort the
    per-landmark GPS the same way before passing it in.
    """
    imgs = imgs.to(device, non_blocking=True)
    lids = lids.to(device, non_blocking=True)
    coords = coords.to(device, non_blocking=True)

    # Per-landmark GPS in the same order torch.unique will produce inside forward.
    unique_lids, first_idx = np.unique(lids.cpu().numpy(), return_index=True)
    landmark_coords = coords[torch.as_tensor(first_idx, device=device)]

    logits = model(imgs, landmark_coords, lids)  # (num_landmarks, num_landmarks)
    targets = torch.arange(logits.size(0), device=device)

    # Symmetric InfoNCE: image -> location and location -> image.
    loss = 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))
    return loss


def main() -> None:
    cfg_path = Path("configs/geoclip_new_train.yaml")
    if not cfg_path.exists():
        cfg_path = Path("configs/geoclip_train.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    tcfg = cfg["training"]
    num_epochs = tcfg["num_epochs"]
    landmarks_per_batch = tcfg.get("landmarks_per_batch", 16)
    lr = tcfg["lr"]
    num_workers = tcfg.get("num_workers", 4)
    checkpoint_path = Path(tcfg.get("checkpoint_path", "models/best_new_geoclip.pth"))
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    data_root = Path(cfg["data"]["root"])
    assert data_root.exists(), f"DATA_ROOT not found: {data_root}"

    device = cfg["inference"]["device"] if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Data root: {data_root.resolve()}")

    baseline = NewGeoClipBaseline(device=device, transformer=True)
    model = baseline.model
    _patch_image_encoder(model.image_encoder)

    trainable, total = freeze_backbone(model)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}  ({100 * trainable / total:.2f}%)")

    train_paths, train_coords, train_lids = load_train_data(data_root)
    print(f"Train images: {len(train_paths):,} across {len(np.unique(train_lids)):,} landmarks")

    train_ds = LandmarkImageDataset(
        train_paths, train_coords, train_lids, model.image_encoder.preprocess_image
    )
    sampler = LandmarkGroupSampler(train_ds.landmark_ids, landmarks_per_batch, shuffle=True)
    train_loader = DataLoader(
        train_ds,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
    )

    optim = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )

    # Eval setup: predict per-landmark GPS for query, compare to GT.
    gallery_coords = load_gallery_coords(data_root, source=cfg["gallery"]["source"])
    query_paths, query_true_coords, query_lids = load_query_data(data_root)
    print(f"Query: {len(query_paths):,} images / {len(np.unique(query_lids)):,} landmarks")

    from mmgeo.geolocalizations.geoclip.evaluate import (
        accuracy_at_thresholds,
        haversine,
    )
    thresholds = cfg["evaluation"]["thresholds_km"]

    def evaluate() -> dict:
        model.eval()
        baseline.build_gallery(gallery_coords)
        per_landmark_pred = baseline.predict_batch(
            query_paths, query_lids, batch_size=cfg["inference"]["batch_size"]
        )
        # Expand per-landmark predictions back to per-image to match true_coords.
        _, counts = np.unique(query_lids, return_counts=True)
        preds = np.repeat(per_landmark_pred, counts, axis=0)
        plat, plon = preds[:, 0], preds[:, 1]
        tlat, tlon = query_true_coords[:, 0], query_true_coords[:, 1]
        acc = accuracy_at_thresholds(plat, plon, tlat, tlon, thresholds)
        d = haversine(plat, plon, tlat, tlon)
        return {"acc": acc, "median_km": float(np.median(d)), "mean_km": float(d.mean())}

    print("\n=== Zero-shot eval (untrained transformer) ===")
    z = evaluate()
    for t, a in z["acc"].items():
        print(f"  Acc@{t}km: {a*100:.2f}%")
    print(f"  median {z['median_km']:.1f} km | mean {z['mean_km']:.1f} km")

    best_acc25 = z["acc"].get(25, 0.0)
    last_path = checkpoint_path.with_name("last_" + checkpoint_path.name.removeprefix("best_"))

    for epoch in range(num_epochs):
        model.train()
        t0 = time.time()
        loss_sum, n = 0.0, 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{num_epochs}", unit="batch")
        for imgs, coords, lids in pbar:
            optim.zero_grad()
            loss = train_step(model, imgs, coords, lids, device)
            loss.backward()
            optim.step()
            loss_sum += loss.item()
            n += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg = loss_sum / max(n, 1)
        print(f"epoch {epoch+1} | avg loss {avg:.4f} | {time.time()-t0:.1f}s")

        m = evaluate()
        for t, a in m["acc"].items():
            print(f"  val Acc@{t}km: {a*100:.2f}%")
        print(f"  val median {m['median_km']:.1f} km | mean {m['mean_km']:.1f} km")

        torch.save(model.state_dict(), last_path)
        if m["acc"].get(25, 0.0) > best_acc25:
            best_acc25 = m["acc"][25]
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> new best Acc@25km {best_acc25*100:.2f}% saved to {checkpoint_path}")

    print(f"\nDone. Best Acc@25km: {best_acc25*100:.2f}%")


if __name__ == "__main__":
    main()
