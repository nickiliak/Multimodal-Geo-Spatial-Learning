from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

with open("configs/geoclip_baseline.yaml") as f:
    cfg = yaml.safe_load(f)

DATA_ROOT = Path(cfg["data"]["root"])
assert DATA_ROOT.exists(), f"DATA_ROOT not found: {DATA_ROOT}"

device = cfg["inference"]["device"] if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print(f"Data root: {DATA_ROOT.resolve()}")

NUM_EPOCHS = 2
TRAIN_BATCH_SIZE = 32
LR = 1e-4
CHECKPOINT_PATH = "best_geoclip_baseline.pth"

from mmgeo.geolocalizations.geoclip.geoclip_baseline import (
    GeoClipBaseline,
    load_gallery_coords,
    load_query_data,
    load_train_data,
)
from mmgeo.geolocalizations.geoclip.evaluate import (
    accuracy_at_thresholds,
    median_error,
    haversine,
)

baseline = GeoClipBaseline(device=device)

total_params = sum(p.numel() for p in baseline.model.parameters())
trainable_params = sum(p.numel() for p in baseline.model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# ---------- GALLERY AND QUERY SETUP ----------
gallery_coords = load_gallery_coords(
    DATA_ROOT, include_index=cfg["gallery"]["include_index"]
)
print(f"Gallery size: {len(gallery_coords):,} GPS points")
print(f"Lat range: [{gallery_coords[:, 0].min():.2f}, {gallery_coords[:, 0].max():.2f}]")
print(f"Lon range: [{gallery_coords[:, 1].min():.2f}, {gallery_coords[:, 1].max():.2f}]")

baseline.build_gallery(gallery_coords)
print("Gallery embeddings computed.")

image_paths, true_coords, landmark_ids = load_query_data(DATA_ROOT)
print(f"Query landmarks: {len(image_paths)}")
print(image_paths[0])

missing = [p for p in image_paths if not p.exists()]
if missing:
    print(f"WARNING: {len(missing)} images not found. First missing: {missing[0]}")
else:
    print("All query images found.")


def evaluate_on_query() -> dict:
    baseline.model.eval()
    pred_coords = baseline.predict_batch(
        image_paths, batch_size=cfg["inference"]["batch_size"]
    )
    pred_lat, pred_lon = pred_coords[:, 0], pred_coords[:, 1]
    tlat, tlon = true_coords[:, 0], true_coords[:, 1]
    acc = accuracy_at_thresholds(pred_lat, pred_lon, tlat, tlon, cfg["evaluation"]["thresholds_km"])
    med = median_error(pred_lat, pred_lon, tlat, tlon)
    dists = haversine(pred_lat, pred_lon, tlat, tlon)
    return {"acc": acc, "median_km": med, "mean_km": float(dists.mean())}


def print_metrics(tag: str, m: dict) -> None:
    df = pd.DataFrame(
        [{"Threshold (km)": t, "Accuracy (%)": f"{a * 100:.2f}"} for t, a in m["acc"].items()]
    )
    print(f"\n[{tag}]")
    print(df.to_string(index=False))
    print(f"Median error: {m['median_km']:.1f} km | Mean error: {m['mean_km']:.1f} km")


# ---------- BASELINE (ZERO-SHOT) ----------
t0 = time.time()
zeroshot = evaluate_on_query()
print(f"Zero-shot inference in {time.time() - t0:.1f}s")
print_metrics("zero-shot", zeroshot)


# ---------- TRAINING (CONTRASTIVE) ----------
train_paths, train_coords, train_ids = load_train_data(DATA_ROOT)
print(f"\nTrain landmarks: {len(train_paths)}")
print(train_paths[0])

optimizer = torch.optim.Adam(
    [p for p in baseline.model.parameters() if p.requires_grad], lr=LR
)

logit_scale_init = baseline.model.logit_scale.item()
print(f"Initial logit_scale: {logit_scale_init:.4f}")

best_acc25 = zeroshot["acc"][25]
print(f"Zero-shot Acc@25km baseline: {best_acc25 * 100:.2f}%")

for epoch in range(NUM_EPOCHS):
    baseline.model.train()
    perm = np.random.permutation(len(train_paths))
    running_loss = 0.0
    n_batches = 0
    t_epoch = time.time()

    for start in range(0, len(train_paths), TRAIN_BATCH_SIZE):
        idx = perm[start : start + TRAIN_BATCH_SIZE]
        if len(idx) < 2:
            continue
        batch_paths = [train_paths[i] for i in idx]
        batch_coords = torch.tensor(train_coords[idx], dtype=torch.float32, device=device)

        imgs = torch.cat(
            [
                baseline.model.image_encoder.preprocess_image(
                    Image.open(p).convert("RGB")
                )
                for p in batch_paths
            ],
            dim=0,
        ).to(device)

        optimizer.zero_grad()
        img_emb = F.normalize(baseline.model.image_encoder(imgs), dim=-1)
        loc_emb = F.normalize(baseline.model.location_encoder(batch_coords), dim=-1)

        logits = (img_emb @ loc_emb.T) * baseline.model.logit_scale.exp()
        targets = torch.arange(logits.size(0), device=device)
        batch_loss = 0.5 * (
            F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)
        )
        batch_loss.backward()
        optimizer.step()

        running_loss += batch_loss.item()
        n_batches += 1

    avg_loss = running_loss / max(n_batches, 1)
    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS} | train contrastive loss: {avg_loss:.4f} "
        f"| logit_scale: {baseline.model.logit_scale.item():.4f} "
        f"| {time.time() - t_epoch:.1f}s"
    )

    baseline.build_gallery(gallery_coords)
    metrics = evaluate_on_query()
    print_metrics(f"epoch {epoch + 1} val", metrics)

    if metrics["acc"][25] > best_acc25:
        best_acc25 = metrics["acc"][25]
        torch.save(baseline.model.state_dict(), CHECKPOINT_PATH)
        print(f"  -> saved checkpoint (Acc@25km {best_acc25 * 100:.2f}%)")


# ---------- FINAL EVAL USING BEST CHECKPOINT ----------
if Path(CHECKPOINT_PATH).exists():
    baseline.model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    baseline.build_gallery(gallery_coords)
    final = evaluate_on_query()
    print_metrics("final (best checkpoint)", final)
else:
    print("\nNo checkpoint beat zero-shot; skipping final load.")
