# Hybrid Pipeline — GeoCLIP + Sample4Geo Finetuning

## 1. What this is

A two-stage cross-view retrieval pipeline:

1. **GeoCLIP (frozen)** predicts a rough GPS for each ground query.
2. **`gps_to_satellite_image_pipe`** narrows the satellite gallery to a radius around that predicted GPS.
3. **Sample4Geo (trainable)** reranks the ground query against the narrowed sat candidates and returns top-K.

This document covers the **finetuning** of Sample4Geo for a small number of additional epochs on top of an existing checkpoint (`models/finetuned.pt`), evaluated end-to-end with the hybrid pipeline. The goal is to measure whether a few more epochs of plain InfoNCE finetuning improve the hybrid pipeline metrics over the starting checkpoint.

GeoCLIP is **never updated** — only Sample4Geo's parameters (and the InfoNCE temperature) receive gradients.

---

## 2. Quick results summary

Filled in after the run completes. R@K and mAP@1k are for ground -> satellite retrieval (g2s) under the hybrid pipeline at radius=25 km with `fallback_full` (queries with no sat in radius fall back to full-gallery rerank).

| Model                      | Source                        | g2s R@1 | g2s R@5 | g2s R@10 | g2s mAP@1k |
|----------------------------|-------------------------------|---------|---------|----------|------------|
| Sample4Geo only (no hybrid)| `models/finetuned.pt`         | TBD     | TBD     | TBD      | TBD        |
| Hybrid @ 25 km (start)     | `models/finetuned.pt`         | TBD     | TBD     | TBD      | TBD        |
| Hybrid @ 25 km (epoch 4)   | finetuned + 4 epochs          | TBD     | TBD     | TBD      | TBD        |
| Hybrid @ 25 km (epoch 8)   | finetuned + 8 epochs          | TBD     | TBD     | TBD      | TBD        |
| **Hybrid @ 25 km (epoch 10, best)** | **finetuned + 10 epochs (best.pt)** | **TBD** | **TBD** | **TBD** | **TBD**    |

> Selection metric: `g2s_R@1` under the hybrid eval config in `configs/hybrid.yaml`. The `best.pt` checkpoint is the epoch that maximised this.

---

## 3. Architecture

### Stage 1 — GeoCLIP (frozen)
- `geoclip.GeoCLIP` from the public `geoclip` package, loaded with pretrained weights.
- Gallery source = `paper` (~100k index sat coords + 1k query landmark coords). Matches the hybrid inference sweep config.
- All parameters set to `requires_grad=False` and put in `eval()` mode.

### Stage 2 — `gps_to_satellite_image_pipe` (haversine radius mask)
- Implemented in [pipe_helpers.py](../../../src/mmgeo/pipe_helpers.py).
- Given GeoCLIP's predicted (lat, lon) and a radius in km, returns indices of all sat-gallery items whose GPS is within the radius.
- The training pipeline uses a vectorised matrix form (`_haversine_matrix_km` in [inference.py](../../../src/mmgeo/inference.py)) so a whole batch of queries can be masked at once.

### Stage 3 — Sample4Geo (trainable)
- `CrossViewModel` from [crossview/model.py](../../../src/mmgeo/crossview/model.py): shared-weight ConvNeXt-Base encoder, L2-normalized output.
- Backbone: `convnext_base.fb_in22k_ft_in1k_384` (22k -> 1k fine-tune at 384px).
- Initialised from `models/finetuned.pt` (multi-positive InfoNCE checkpoint with `n_ground=3`).

### Loss
- `MultiPositiveInfoNCE` with `n_ground=3` (matches the source checkpoint's training regime). Reduces to `SymmetricInfoNCE` when `n_ground=1`.
- Learnable log-space temperature, init 0.07.
- Label smoothing 0.1.

### Empty-candidate fallback
- Built into `_hybrid_rerank` in [train_pipeline.py](../../../src/mmgeo/train_pipeline.py): if a query has zero sat candidates within the radius, the radius mask is dropped for that query (= rerank against the full gallery). Equivalent to "if no matches in 25 km, fall back to radius=infinity for that query."

---

## 4. Training setup

Config file: [configs/hybrid.yaml](../../../configs/hybrid.yaml)

| Setting          | Value                                       | Notes                                       |
|------------------|---------------------------------------------|---------------------------------------------|
| Backbone         | `convnext_base.fb_in22k_ft_in1k_384`        | matches finetuned.pt source                 |
| Image size       | 384 px                                      | matches backbone resolution                 |
| Batch size       | 24                                          | 384px on 32GB V100                          |
| n_ground         | 3                                           | multi-positive InfoNCE                      |
| Epochs           | 10                                          | additional finetuning on top of finetuned.pt|
| LR               | 5e-5                                        | lower than v3 (1e-4) — continuation         |
| Weight decay     | 1e-4                                        |                                             |
| LR schedule      | 1ep warmup -> cosine                        |                                             |
| Hard negatives   | GPS (2 ep) -> DSS (8 ep)                    | shorter GPS warmup (model already trained)  |
| Eval frequency   | every 4 epochs + final epoch (4, 8, 10)     | hybrid eval is expensive                    |
| Eval batch size  | 192                                         |                                             |
| GPU              | V100 32GB                                   | both Sample4Geo and GeoCLIP loaded          |
| Init checkpoint  | `models/finetuned.pt`                       | `model.pretrained_ckpt` in config           |

### What happens during training

Plain (ground, sat) InfoNCE — same as `crossview/train.py`:
1. Sample a batch of unique landmarks (GPS-hard or DSS-hard).
2. For each landmark, sample 3 ground images and 1 satellite image.
3. Forward both through the shared Sample4Geo encoder.
4. Multi-positive InfoNCE loss; symmetric.
5. Backprop; update Sample4Geo + temperature only.

GeoCLIP is **not** in the training graph.

### What happens at evaluation

The hybrid eval at epochs 4, 8, 10:
1. Re-embed the satellite gallery with the **current** Sample4Geo weights (no disk cache — weights change every epoch). Cost: one full sat-gallery embedding pass per eval.
2. Run GeoCLIP on all 18,688 ground queries -> predicted GPS.
3. Embed all 18,688 ground queries with current Sample4Geo.
4. For each query, mask sat gallery to a 25 km radius; if empty, drop the mask (`fallback_full`).
5. Cosine-similarity top-K -> compute R@1/5/10 and mAP@1k.

### Checkpoints
- `best.pt` written whenever the hybrid `g2s_R@1` improves.
- `last.pt` overwritten every epoch — safe resume after wall-time kill.
- Both are saved into a versioned run dir under `checkpoints/hybrid/hybrid_<timestamp>/`.

---

## 5. Evaluation setup

### Hybrid eval config (during training)
| Setting       | Value           | Notes                                       |
|---------------|-----------------|---------------------------------------------|
| index_mode    | `full`          | sat gallery = query-sat + index-sat (~100k) |
| query_mode    | `all`           | all 18,688 ground images                    |
| radius_km     | 25.0            | primary narrowing                           |
| fallback      | `fallback_full` | empty -> full-gallery rerank                |
| recall_ks     | [1, 5, 10]      |                                             |
| map_k         | 1000            |                                             |

### Gallery composition
Same as v2/v3 cross-view baseline:

| Direction | Queries                                    | Gallery                                      |
|-----------|--------------------------------------------|----------------------------------------------|
| g2s       | 18,688 ground images (1,000 landmarks)     | 1,000 query-sat + 99,539 index-sat = 100,539 |

The hybrid pipeline's narrowing reduces the effective gallery size *per query* (mean candidates after 25 km mask is logged in the eval CSV).

### Metrics
- **R@K** (K ∈ {1, 5, 10}): fraction of queries whose top-K contains an item with the same `landmark_id`.
- **mAP@1000**: mean average precision truncated at rank 1000.
- **empty_rate**: fraction of queries that triggered the radius fallback (no sat in 25 km of GeoCLIP's prediction).
- **mean/median_candidates**: pre-fallback narrowing statistics.

---

## 6. Key files

| File                                          | Description                                                    |
|-----------------------------------------------|----------------------------------------------------------------|
| [configs/hybrid.yaml](../../../configs/hybrid.yaml) | Training + hybrid-eval config                            |
| [src/mmgeo/train_pipeline.py](../../../src/mmgeo/train_pipeline.py) | Main training loop (this file)                |
| [src/mmgeo/pipe_helpers.py](../../../src/mmgeo/pipe_helpers.py) | `gps_to_satellite_image_pipe`, haversine        |
| [src/mmgeo/inference.py](../../../src/mmgeo/inference.py) | Reused: gallery building, GeoCLIP rough-GPS step      |
| [src/mmgeo/crossview/model.py](../../../src/mmgeo/crossview/model.py) | `CrossViewModel` (Sample4Geo)                 |
| [src/mmgeo/geolocalizations/geoclip/geoclip_baseline.py](../../../src/mmgeo/geolocalizations/geoclip/geoclip_baseline.py) | `GeoClipBaseline`, `load_gallery_coords` |
| [scripts/train_hybrid.sh](../../../scripts/train_hybrid.sh) | LSF job script                                          |
| `models/finetuned.pt`                         | Source Sample4Geo checkpoint (gitignored)                      |

---

## 7. How to run

### Train on HPC
```bash
bsub < scripts/train_hybrid.sh
```
Edit `RESUME` in the script to resume from a checkpoint, or leave empty to start fresh from `models/finetuned.pt`.

### Run with a different config or eval radius
Override values in `configs/hybrid.yaml`:
```yaml
hybrid_eval:
  radius_km: 10.0   # tighter narrowing
  query_mode: "one_per_landmark"  # fast eval (1k queries)
```

### Run plain hybrid inference (no training, sweep radii)
The existing inference sweep (separate from this training pipeline):
```bash
bsub < scripts/run_hybrid_inference.sh
```

---

## 8. Comparison to v2/v3 cross-view baseline

| Aspect             | v2/v3 cross-view                  | Hybrid pipeline (this doc)                    |
|--------------------|-----------------------------------|-----------------------------------------------|
| Training loss      | Symmetric / Multi-positive InfoNCE| Same                                          |
| Models trained     | Sample4Geo only                   | Sample4Geo only (GeoCLIP loaded but frozen)   |
| Eval at training   | Pure Sample4Geo retrieval         | GeoCLIP -> radius -> Sample4Geo rerank        |
| Selection metric   | `g2s_recall@1` (full gallery)     | `g2s_R@1` (radius-narrowed gallery)           |
| Init weights       | ImageNet-22k (v2) / 22k+1k@384(v3)| `models/finetuned.pt` (already trained)       |
| Epochs             | 35                                | 10                                            |
| Use case           | establish image-only baseline     | measure benefit of hybrid eval pipeline       |

Per [crossview_baseline_v2_tasks.md](../Mateusz/crossview_baseline_v2_tasks.md), v2 (ConvNeXt-Base, 35 epochs) reached g2s R@1 = 7.21% unpooled. Hybrid eval should *raise* this number for the same Sample4Geo weights because GeoCLIP narrowing eliminates a large fraction of distractors before reranking.

---

## 9. Limitations

- **Frozen GeoCLIP**: any GeoCLIP error (wrong rough GPS) cannot be corrected by training Sample4Geo — the radius mask still hides the true sat. The fallback policy partially mitigates this (when no candidates remain, drop the mask).
- **Single radius during training**: 25 km only. A radius sweep is left to the standalone hybrid inference (`scripts/run_hybrid_inference.sh`) on the final `best.pt`.
- **Plain InfoNCE training**: training does not condition on GeoCLIP's predictions. The model is not directly optimised for "rerank within radius"; it is optimised for general cross-view matching, then evaluated under narrowing. A GeoCLIP-conditioned training loss is left to future work.
- **Gallery cache**: not used during training (model weights change every epoch). One full sat-gallery embedding per eval is unavoidable.
