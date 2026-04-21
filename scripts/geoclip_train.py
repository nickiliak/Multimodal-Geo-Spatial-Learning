from pathlib import Path
import time

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from mmgeo.geolocalizations.geoclip.evaluate import (
    accuracy_at_thresholds,
    haversine,
    median_error,
)
from mmgeo.geolocalizations.geoclip.geoclip_baseline import (
    GeoClipBaseline,
    load_gallery_coords,
    load_query_data,
    load_train_data,
)

NUM_EPOCHS = 2
TRAIN_BATCH_SIZE = 32
LR = 1e-4
CHECKPOINT_DIR = Path("models")
CHECKPOINT_PATH = CHECKPOINT_DIR / "best_geoclip_baseline.pth"


class _PathCoordDataset(Dataset):
    def __init__(self, paths: list[Path], coords: np.ndarray):
        self.paths = [str(p) for p in paths]
        self.coords = coords.astype(np.float32)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        return self.paths[idx], self.coords[idx]


def _collate(batch):
    paths = [b[0] for b in batch]
    coords = torch.from_numpy(np.stack([b[1] for b in batch], axis=0))
    return paths, coords


def _print_metrics(tag: str, m: dict) -> None:
    df = pd.DataFrame(
        [{"Threshold (km)": t, "Accuracy (%)": f"{a * 100:.2f}"} for t, a in m["acc"].items()]
    )
    print(f"\n[{tag}]")
    print(df.to_string(index=False))
    print(f"Median error: {m['median_km']:.1f} km | Mean error: {m['mean_km']:.1f} km")


class GeoClipLitModule(L.LightningModule):
    def __init__(
        self,
        baseline: GeoClipBaseline,
        gallery_coords: np.ndarray,
        query_paths: list[Path],
        query_true_coords: np.ndarray,
        thresholds_km: list[float],
        lr: float = LR,
        inference_batch_size: int = 64,
        checkpoint_path: Path = CHECKPOINT_PATH,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["baseline", "gallery_coords", "query_paths", "query_true_coords"])
        self.baseline = baseline
        self.model = baseline.model
        self.gallery_coords = gallery_coords
        self.query_paths = query_paths
        self.query_true_coords = query_true_coords
        self.thresholds_km = thresholds_km
        self.checkpoint_path = Path(checkpoint_path)
        self.best_acc25: float | None = None
        self._epoch_t0: float | None = None
        self._epoch_loss_sum: float = 0.0
        self._epoch_loss_n: int = 0

    def configure_optimizers(self):
        trainable = [p for p in self.baseline.model.parameters() if p.requires_grad]
        return torch.optim.Adam(trainable, lr=self.hparams.lr)

    def training_step(self, batch, batch_idx):
        paths, coords = batch
        coords = coords.to(self.device)

        imgs = torch.cat(
            [
                self.baseline.model.image_encoder.preprocess_image(
                    Image.open(p).convert("RGB")
                )
                for p in paths
            ],
            dim=0,
        ).to(self.device)

        img_emb = F.normalize(self.baseline.model.image_encoder(imgs), dim=-1)
        loc_emb = F.normalize(self.baseline.model.location_encoder(coords), dim=-1)

        logits = (img_emb @ loc_emb.T) * self.baseline.model.logit_scale.exp()
        targets = torch.arange(logits.size(0), device=self.device)
        loss = 0.5 * (
            F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)
        )
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self._epoch_loss_sum += loss.detach().item()
        self._epoch_loss_n += 1
        return loss

    def on_train_epoch_start(self) -> None:
        self._epoch_t0 = time.time()
        self._epoch_loss_sum = 0.0
        self._epoch_loss_n = 0

    def on_train_epoch_end(self) -> None:
        avg_loss = self._epoch_loss_sum / max(self._epoch_loss_n, 1)
        logit_scale = self.baseline.model.logit_scale.item()
        dt = time.time() - (self._epoch_t0 or time.time())
        print(
            f"Epoch {self.current_epoch + 1}/{self.trainer.max_epochs} | "
            f"train contrastive loss: {avg_loss:.4f} | "
            f"logit_scale: {logit_scale:.4f} | {dt:.1f}s"
        )

        self.baseline.build_gallery(self.gallery_coords)
        metrics = self.evaluate_on_query()
        _print_metrics(f"epoch {self.current_epoch + 1} val", metrics)

        self.log_dict(
            {f"val_acc@{t}": a for t, a in metrics["acc"].items()}
            | {"val_median_km": metrics["median_km"], "val_mean_km": metrics["mean_km"]},
        )

        acc25 = metrics["acc"][25]
        if self.best_acc25 is None or acc25 > self.best_acc25:
            self.best_acc25 = acc25
            self.checkpoint_path.parent.mkdir(exist_ok=True)
            torch.save(self.baseline.model.state_dict(), self.checkpoint_path)
            print(f"  -> saved checkpoint (Acc@25km {acc25 * 100:.2f}%)")

    def evaluate_on_query(self) -> dict:
        self.baseline.model.eval()
        pred_coords = self.baseline.predict_batch(
            self.query_paths, batch_size=self.hparams.inference_batch_size
        )
        pred_lat, pred_lon = pred_coords[:, 0], pred_coords[:, 1]
        tlat, tlon = self.query_true_coords[:, 0], self.query_true_coords[:, 1]
        acc = accuracy_at_thresholds(pred_lat, pred_lon, tlat, tlon, self.thresholds_km)
        med = median_error(pred_lat, pred_lon, tlat, tlon)
        dists = haversine(pred_lat, pred_lon, tlat, tlon)
        return {"acc": acc, "median_km": med, "mean_km": float(dists.mean())}


def main() -> None:
    with open("configs/geoclip_baseline.yaml") as f:
        cfg = yaml.safe_load(f)

    data_root = Path(cfg["data"]["root"])
    assert data_root.exists(), f"DATA_ROOT not found: {data_root}"

    device = cfg["inference"]["device"] if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Data root: {data_root.resolve()}")

    baseline = GeoClipBaseline(device=device)

    total_params = sum(p.numel() for p in baseline.model.parameters())
    trainable_params = sum(p.numel() for p in baseline.model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    gallery_coords = load_gallery_coords(
        data_root, include_index=cfg["gallery"]["include_index"]
    )
    print(f"Gallery size: {len(gallery_coords):,} GPS points")
    print(f"Lat range: [{gallery_coords[:, 0].min():.2f}, {gallery_coords[:, 0].max():.2f}]")
    print(f"Lon range: [{gallery_coords[:, 1].min():.2f}, {gallery_coords[:, 1].max():.2f}]")

    baseline.build_gallery(gallery_coords)
    print("Gallery embeddings computed.")

    image_paths, true_coords, _ = load_query_data(data_root)
    print(f"Query landmarks: {len(image_paths)}")
    print(image_paths[0])

    missing = [p for p in image_paths if not p.exists()]
    if missing:
        print(f"WARNING: {len(missing)} images not found. First missing: {missing[0]}")
    else:
        print("All query images found.")

    lit = GeoClipLitModule(
        baseline=baseline,
        gallery_coords=gallery_coords,
        query_paths=image_paths,
        query_true_coords=true_coords,
        thresholds_km=cfg["evaluation"]["thresholds_km"],
        lr=LR,
        inference_batch_size=cfg["inference"]["batch_size"],
        checkpoint_path=CHECKPOINT_PATH,
    )

    t0 = time.time()
    zeroshot = lit.evaluate_on_query()
    print(f"Zero-shot inference in {time.time() - t0:.1f}s")
    _print_metrics("zero-shot", zeroshot)
    lit.best_acc25 = zeroshot["acc"][25]
    print(f"Zero-shot Acc@25km baseline: {lit.best_acc25 * 100:.2f}%")

    print(f"Initial logit_scale: {baseline.model.logit_scale.item():.4f}")

    train_paths, train_coords, _ = load_train_data(data_root)
    print(f"\nTrain landmarks: {len(train_paths)}")
    print(train_paths[0])

    train_loader = DataLoader(
        _PathCoordDataset(train_paths, train_coords),
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=_collate,
        drop_last=False,
    )

    trainer = L.Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
    )
    trainer.fit(lit, train_dataloaders=train_loader)

    if CHECKPOINT_PATH.exists():
        baseline.model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        baseline.build_gallery(gallery_coords)
        final = lit.evaluate_on_query()
        _print_metrics("final (best checkpoint)", final)
    else:
        print("\nNo checkpoint beat zero-shot; skipping final load.")


if __name__ == "__main__":
    main()
