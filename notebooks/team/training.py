from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import yaml

plt.rcParams.update({"figure.dpi": 120})

# Load config
with open("../../configs/geoclip_baseline.yaml") as f:
    cfg = yaml.safe_load(f)

DATA_ROOT = Path("../../") / cfg["data"]["root"]
assert DATA_ROOT.exists(), f"DATA_ROOT not found: {DATA_ROOT}"

device = cfg["inference"]["device"] if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print(f"Data root: {DATA_ROOT.resolve()}")

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

#----------GALLERY AND QUERY SETUP------------
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

# Verify all image files exist
missing = [p for p in image_paths if not p.exists()]
if missing:
    print(f"WARNING: {len(missing)} images not found. First missing: {missing[0]}")
else:
    print("All query images found.")

    

#-----------BASE INFERENCE------------
t0 = time.time()
pred_coords = baseline.predict_batch(
    image_paths, batch_size=cfg["inference"]["batch_size"]
)

elapsed = time.time() - t0
print(f"Inference complete: {len(pred_coords)} predictions in {elapsed:.1f}s")

#---------EVALUATION------------
pred_lat, pred_lon = pred_coords[:, 0], pred_coords[:, 1]
true_lat, true_lon = true_coords[:, 0], true_coords[:, 1]


thresholds = cfg["evaluation"]["thresholds_km"]
results = accuracy_at_thresholds(pred_lat, pred_lon, true_lat, true_lon, thresholds)
med_err = median_error(pred_lat, pred_lon, true_lat, true_lon)
distances = haversine(pred_lat, pred_lon, true_lat, true_lon)

results_df = pd.DataFrame(
    [{"Threshold (km)": t, "Accuracy (%)": f"{acc * 100:.2f}"} for t, acc in results.items()]
)
print(results_df.to_string(index=False))
print(f"\nMedian error: {med_err:.1f} km")
print(f"Mean error: {distances.mean():.1f} km")


#---------TRAINING------------
train_paths, train_coords, train_ids = load_train_data(DATA_ROOT)
print(f"Train landmarks: {len(train_paths)}")
print(train_paths[0])

baseline = GeoClipBaseline(device=device)
optimizer = torch.optim.Adam(baseline.model.parameters(), lr=1e-4)
loss = torch.nn.MSELoss()


best_val_loss = float("inf")
for epoch in range(2):
    for batch in range(len(train_paths) // 32):
        optimizer.zero_grad()
        batch_paths = train_paths[batch * 32 : (batch + 1) * 32]
        batch_coords = train_coords[batch * 32 : (batch + 1) * 32]
        pred_coords = baseline.predict_batch(batch_paths, batch_size=32)
        batch_loss = loss(torch.tensor(pred_coords), torch.tensor(batch_coords))
        batch_loss.backward()
        optimizer.step()
    
    #output validation loss after each epoch
    val_loss = []
    for batch in range(len(image_paths) // 32):
        batch_paths = image_paths[batch * 32 : (batch + 1) * 32]
        batch_coords = true_coords[batch * 32 : (batch + 1) * 32]
        pred_coords = baseline.predict_batch(batch_paths, batch_size=32)
        val_loss.append(loss(torch.tensor(pred_coords), torch.tensor(batch_coords)).item())
    print(f"Epoch {epoch+1}: Mean Validation MSE Loss = {sum(val_loss) / len(val_loss):.4f}")
    #Save model if validation loss improves
    if sum(val_loss) / len(val_loss) < best_val_loss:
        best_val_loss = sum(val_loss) / len(val_loss)
        torch.save(baseline.model.state_dict(), "best_geoclip_baseline.pth")



#---------INFERENCE AND EVALUATE AGAIN----------
t0 = time.time()
pred_coords = baseline.predict_batch(
    image_paths, batch_size=cfg["inference"]["batch_size"]
)

elapsed = time.time() - t0
print(f"Inference complete: {len(pred_coords)} predictions in {elapsed:.1f}s")

pred_lat, pred_lon = pred_coords[:, 0], pred_coords[:, 1]
true_lat, true_lon = true_coords[:, 0], true_coords[:, 1]


thresholds = cfg["evaluation"]["thresholds_km"]
results = accuracy_at_thresholds(pred_lat, pred_lon, true_lat, true_lon, thresholds)
med_err = median_error(pred_lat, pred_lon, true_lat, true_lon)
distances = haversine(pred_lat, pred_lon, true_lat, true_lon)

results_df = pd.DataFrame(
    [{"Threshold (km)": t, "Accuracy (%)": f"{acc * 100:.2f}"} for t, acc in results.items()]
)
print(results_df.to_string(index=False))
print(f"\nMedian error: {med_err:.1f} km")
print(f"Mean error: {distances.mean():.1f} km")
