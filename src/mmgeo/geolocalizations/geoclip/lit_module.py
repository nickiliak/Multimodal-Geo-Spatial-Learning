from __future__ import annotations

import time
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from mmgeo.geolocalizations.geoclip.evaluate import (
    accuracy_at_thresholds,
    haversine,
    median_error,
)
from mmgeo.geolocalizations.geoclip.geoclip_baseline import GeoClipBaseline


def print_metrics(tag: str, m: dict) -> None:
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
        lr: float = 1e-4,
        inference_batch_size: int = 64,
        checkpoint_path: Path = Path("models/best_geoclip_baseline.pth"),
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
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.Adam(trainable, lr=self.hparams.lr)

    def training_step(self, batch, batch_idx):
        imgs, coords = batch
        imgs = imgs.to(self.device)
        coords = coords.to(self.device)

        img_emb = F.normalize(self.model.image_encoder(imgs), dim=-1)
        loc_emb = F.normalize(self.model.location_encoder(coords), dim=-1)

        logits = (img_emb @ loc_emb.T) * self.model.logit_scale.exp()
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
        logit_scale = self.model.logit_scale.item()
        dt = time.time() - (self._epoch_t0 or time.time())
        print(
            f"Epoch {self.current_epoch + 1}/{self.trainer.max_epochs} | "
            f"train contrastive loss: {avg_loss:.4f} | "
            f"logit_scale: {logit_scale:.4f} | {dt:.1f}s"
        )

        self.baseline.build_gallery(self.gallery_coords)
        metrics = self.evaluate_on_query()
        print_metrics(f"epoch {self.current_epoch + 1} val", metrics)

        self.log_dict(
            {f"val_acc@{t}": a for t, a in metrics["acc"].items()}
            | {"val_median_km": metrics["median_km"], "val_mean_km": metrics["mean_km"]},
        )

        acc25 = metrics["acc"][25]
        if self.best_acc25 is None or acc25 > self.best_acc25:
            self.best_acc25 = acc25
            self.checkpoint_path.parent.mkdir(exist_ok=True)
            torch.save(self.model.state_dict(), self.checkpoint_path)
            print(f"  -> saved checkpoint (Acc@25km {acc25 * 100:.2f}%)")

    def evaluate_on_query(self) -> dict:
        self.model.eval()
        pred_coords = self.baseline.predict_batch(
            self.query_paths, batch_size=self.hparams.inference_batch_size
        )
        pred_lat, pred_lon = pred_coords[:, 0], pred_coords[:, 1]
        tlat, tlon = self.query_true_coords[:, 0], self.query_true_coords[:, 1]
        acc = accuracy_at_thresholds(pred_lat, pred_lon, tlat, tlon, self.thresholds_km)
        med = median_error(pred_lat, pred_lon, tlat, tlon)
        dists = haversine(pred_lat, pred_lon, tlat, tlon)
        return {"acc": acc, "median_km": med, "mean_km": float(dists.mean())}
