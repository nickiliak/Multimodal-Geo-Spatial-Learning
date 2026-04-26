from pathlib import Path
import time

import lightning as L
import torch
import yaml
from torch.utils.data import DataLoader

from mmgeo.geolocalizations.geoclip.dataset import MMLDataset
from mmgeo.geolocalizations.geoclip.geoclip_baseline import (
    GeoClipBaseline,
    load_gallery_coords,
    load_query_data,
    load_train_data,
)
from mmgeo.geolocalizations.geoclip.lit_module import GeoClipLitModule, print_metrics


def main() -> None:
    with open("configs/geoclip_train.yaml") as f:
        cfg = yaml.safe_load(f)

    tcfg = cfg["training"]
    num_epochs = tcfg["num_epochs"]
    train_batch_size = tcfg["batch_size"]
    lr = tcfg["lr"]
    num_workers = tcfg.get("num_workers", 4)
    checkpoint_path = Path(tcfg["checkpoint_path"])

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

    gallery_coords = load_gallery_coords(data_root, source=cfg["gallery"]["source"])
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
        lr=lr,
        inference_batch_size=cfg["inference"]["batch_size"],
        checkpoint_path=checkpoint_path,
    )

    t0 = time.time()
    zeroshot = lit.evaluate_on_query()
    print(f"Zero-shot inference in {time.time() - t0:.1f}s")
    print_metrics("zero-shot", zeroshot)
    lit.best_acc25 = zeroshot["acc"][25]
    print(f"Zero-shot Acc@25km baseline: {lit.best_acc25 * 100:.2f}%")

    print(f"Initial logit_scale: {baseline.model.logit_scale.item():.4f}")

    train_paths, train_coords, _ = load_train_data(data_root)
    print(f"\nTrain landmarks: {len(train_paths)}")
    print(train_paths[0])

    train_loader = DataLoader(
        MMLDataset(train_paths, train_coords, baseline.model.image_encoder.preprocess_image),
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    trainer = L.Trainer(
        max_epochs=num_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
    )
    trainer.fit(lit, train_dataloaders=train_loader)

    last_path = checkpoint_path.with_name(
        "last_" + checkpoint_path.name.removeprefix("best_")
    )
    load_path = (
        checkpoint_path if checkpoint_path.exists()
        else (last_path if last_path.exists() else None)
    )
    if load_path is not None:
        baseline.model.load_state_dict(torch.load(load_path, map_location=device))
        baseline.build_gallery(gallery_coords)
        final = lit.evaluate_on_query()
        tag = "best" if load_path == checkpoint_path else "last"
        print_metrics(f"final ({tag} checkpoint)", final)
    else:
        print("\nNo checkpoint on disk; skipping final load.")


if __name__ == "__main__":
    main()
