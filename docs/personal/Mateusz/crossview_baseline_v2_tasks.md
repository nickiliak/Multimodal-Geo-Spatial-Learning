# Cross-View Retrieval Baseline — v1, v2 & v3 Documentation

## 1. What this is

This document describes the **image-only cross-view retrieval baseline** built on MMLandmarks. The method is inspired by [Sample4Geo](https://arxiv.org/abs/2303.11913): a shared ConvNeXt encoder trained with symmetric InfoNCE loss to match ground-level photos to satellite imagery and vice versa.

This baseline serves as the **image-only comparison point** in the final project report, against which multimodal approaches (GeoClip, text+image methods) will be measured.

---

## 2. Quick results summary

| Model | Backbone | Training | g2s R@1 | g2s R@5 | g2s R@10 | g2s mAP@1k | Protocol |
|-------|----------|----------|---------|---------|---------|-----------|---------|
| v1 | ConvNeXt-Tiny | MMLandmarks train | 9.73% | 27.11% | 37.62% | — | pooled, query-only gallery |
| **v2 (pooled)** | **ConvNeXt-Base** | **MMLandmarks train** | **17.60%** | **46.66%** | **60.96%** | **25.50%** | pooled — NOT paper-comparable |
| **v2 (unpooled)** | **ConvNeXt-Base** | **MMLandmarks train** | **7.21%** | **18.52%** | **25.31%** | **13.10%** | **unpooled — paper-comparable** |
| ConvNeXt-Base zero-shot | ConvNeXt-Base | ImageNet-22k only | 0.25% | 0.76% | 1.13% | 0.59% | unpooled, no MMLandmarks training |
| **v3 (unpooled)** | **ConvNeXt-Base (384px)** | **MMLandmarks train** | **TBD** | — | — | — | **unpooled, multi-positive InfoNCE** |
| MMCLIP (paper) | CLIP ViT-L | **zero-shot** | 20.5% | — | — | — | unpooled |
| GeoClip (paper) | CLIP ViT-L | **zero-shot** | 21.1% | — | — | — | unpooled |

> **Critical note:** MMCLIP and GeoClip numbers are **zero-shot** — those models never trained on MMLandmarks data. Our v2 model DID train on MMLandmarks. See Section 6 (pooling) and Section 9 (zero-shot distinction).  
> **Best checkpoint:** epoch 30 (training peaked at ep30; ep33 was marginally lower). All v2 numbers above are from `cv_v2_base_20260422_230539/best.pt`.

---

## 3. Architecture

### Shared encoder (`CrossViewModel`)
- **Backbone:** `convnext_base.fb_in22k` via [timm](https://github.com/huggingface/pytorch-image-models) — 88M parameters, pretrained on ImageNet-22k
- **Weight sharing:** the same encoder is used for both ground and satellite images (no separate tower)
- **Projection head:** optional linear projection to a target `embed_dim`; in v2 `embed_dim=0` keeps the native 1024-dim features
- **Output:** L2-normalized embeddings per image

### Loss
- **Symmetric InfoNCE** with a learnable (log-space) temperature parameter
- Initialized at `τ = 0.07`, learned end-to-end during training
- Label smoothing: none (0.0) — found not to help in v2

---

## 4. Training setup (v2)

Config file: `configs/crossview_convnext_base.yaml`

| Setting | Value | Notes |
|---------|-------|-------|
| Backbone | `convnext_base.fb_in22k` | 88M params |
| Image size | 224 px | 256 px caused OOM on 32GB V100 |
| Batch size | 64 | per GPU |
| Epochs | 35 | |
| Learning rate | 1e-4 | AdamW |
| Weight decay | 1e-4 | |
| LR schedule | Cosine with warmup | 3 warmup epochs from 0.1× LR, min LR = 1% |
| Hard negatives | GPS → DSS | see below |
| Eval frequency | Every 3 epochs | |
| Eval batch size | 384 | inference-only, fits 32GB easily |
| GPU | V100 32GB | `select[gpu32gb]` in LSF |
| Wall time used | ~8.5 hours | training + eval |

### Hard-negative sampling (Sample4Geo-style)

Training uses a two-phase sampling strategy:

**Phase 1 — GPS-based negatives (epochs 1–4)**  
Negatives are selected from geographically nearby landmarks. This forces the model to learn fine-grained visual differences between nearby locations rather than trivially separating distant ones.

**Phase 2 — Dynamic Similarity Sampling / DSS (epochs 5–35)**  
Negatives are selected based on embedding similarity (i.e., the hardest in feature space). The embedding index is refreshed every epoch. This is the core Sample4Geo technique and is what drives metric gains after the initial GPS warmup.

### Training data
- **Split used:** dedicated `train` split — 17,557 landmarks with ground + satellite images (separate from the `query`/`index` eval splits)
- **Ground images:** one random ground image sampled per landmark per iteration
- **Satellite images:** one random satellite image sampled per landmark per iteration
- **UniqueLandmarkSampler** ensures no duplicate landmark IDs within one batch

> The model is **not zero-shot** — it fine-tunes on MMLandmarks training data. MMCLIP and GeoClip in Table 2 of the paper are zero-shot. See Section 9 for the comparison implications.

### Checkpoints
| Run dir | Best epoch | g2s R@1 (pooled) | g2s R@1 (unpooled) | Notes |
|---------|-----------|-----------------|-------------------|-------|
| `checkpoints/crossview/cv_v2_base_20260420_120027/` | 19 | ~7% | — | wall-time killed at epoch 22 |
| `checkpoints/crossview/cv_v2_base_20260422_230539/` | **30** | **17.60%** | **7.21%** | full 35-epoch run (resumed from above); training peaked at ep30 |

---

## 5. Evaluation setup

### Gallery composition
Evaluation uses the **full index + query satellite gallery** for ground-to-satellite retrieval (g2s), matching the MMLandmarks benchmark protocol:

| Direction | Queries | Gallery |
|-----------|---------|---------|
| g2s (ground → satellite) | 18,688 ground images (1,000 landmarks) | 1,000 query-sat + 99,539 index-sat = **100,539 images** |
| s2g (satellite → ground) | 1,000 satellite images | 18,688 query-ground + 714,554 index-ground = **733,242 images** |

Retrieval is by cosine similarity of L2-normalized embeddings. A retrieved item is **relevant** if it shares the same `landmark_id` as the query.

### Metrics
- **Recall@K** (K = 1, 5, 10): fraction of queries with at least one correct match in the top-K results
- **mAP@1000**: mean Average Precision truncated at rank 1000

---

## 6. Pooled vs unpooled evaluation — important distinction

This is the most important protocol note for report writing.

### What pooling means here
Each of the 1,000 test landmarks has ~18 ground-level photos. "Pooling" means: before retrieval, all ground embeddings for the same landmark are **mean-pooled and L2-renormalized** into a single landmark embedding. This yields 1,000 query vectors instead of 18,688.

### Why this matters
- **Pooled:** 1,000 queries. Numbers are higher and arguably unfair — the model effectively sees a "consensus" view of the landmark.
- **Unpooled:** 18,688 queries. Each ground photo is queried individually, including difficult/atypical images. This is harder and gives lower numbers.

### What the MMLandmarks paper uses
After checking with the paper authors (Oskar Ahlén, email April 2026): **Table 2 in the MMLandmarks paper uses unpooled evaluation** — each of the 18,689 ground images is a separate query. The numbers in Table 2 (MMCLIP 20.5%, GeoClip 21.1%) are unpooled.

Our v2 pooled R@1 = **17.60%** is therefore NOT directly comparable to Table 2. To get a fair comparison number, we need to run the standalone eval script with `--no-pool`.

### Unpooled eval (pending HPC job)
The standalone eval script `src/mmgeo/crossview/eval.py` was built specifically for this. Submit:
```bash
bsub < scripts/eval_crossview.sh
```
This runs g2s and s2g with 18,689 individual query images (no pooling) and saves results to `logs/eval_nopooled_<timestamp>.json`.

Expected unpooled g2s R@1: roughly **10–14%** (harder, but paper-comparable).

---

## 7. Key files

| File | Description |
|------|-------------|
| `configs/crossview_convnext_base.yaml` | Training config (backbone, img_size, LR, hard-neg settings) |
| `src/mmgeo/crossview/model.py` | `CrossViewModel` — shared encoder + optional projection head |
| `src/mmgeo/crossview/train.py` | Full training loop: GPS/DSS hard-neg, LR schedule, eval, checkpointing, `--resume` support |
| `src/mmgeo/crossview/evaluate.py` | Eval utilities: `extract_embeddings`, `compute_retrieval_metrics`, `pool_embeddings_by_landmark`, `evaluate_crossview` |
| `src/mmgeo/crossview/eval.py` | **Standalone eval script** — `--checkpoint` for trained eval, `--pretrained-only` for zero-shot, `--pool`/`--no-pool` |
| `src/mmgeo/crossview/dataset.py` | `MMLImageDataset`, `MMLCrossViewDataset` (with `n_ground` parameter), `get_eval_transforms` |
| `src/mmgeo/crossview/losses.py` | `SymmetricInfoNCE`, `MultiPositiveInfoNCE` (K=1 reduces to symmetric) |
| `scripts/run_crossview_convnext_base.sh` | LSF job script for v2 training (with resume support) |
| `scripts/eval_crossview.sh` | LSF job script for trained eval (`--no-pool`, paper-comparable) |
| `scripts/eval_crossview_zeroshot.sh` | LSF job script for zero-shot eval (`--pretrained-only --no-pool`) |
| `configs/crossview_convnext_base.yaml` | v2 config (224px, single-positive, pooled eval) |
| `configs/crossview_convnext_base_384_zeroshot.yaml` | 384px ConvNeXt zero-shot config |

---

## 8. How to run

### Train (or resume) on HPC
```bash
bsub < scripts/run_crossview_convnext_base.sh
```
Edit `RESUME` in the script to point to an existing checkpoint, or leave empty to start fresh.

For v3 (multi-positive InfoNCE, 384px), copy the v2 yaml and edit:
`backbone.name=convnext_base.fb_in22k_ft_in1k_384`, `dataset.image_size=384`,
`training.batch_size=24`, `loss.type=multi_positive`, `dataset.n_ground=3`,
`evaluation.pool_queries=false`.

### Standalone eval on HPC (paper-comparable, no pooling)
```bash
bsub < scripts/eval_crossview.sh
```
Results are printed to the job log and saved as JSON in `logs/`.


### Standalone eval locally (pooled, quick sanity check)
```bash
python -m mmgeo.crossview.eval \
    --config configs/crossview_convnext_base.yaml \
    --checkpoint checkpoints/crossview/cv_v2_base_20260422_230539/best.pt \
    --pool
```

### Standalone eval locally (unpooled, paper-comparable)
```bash
python -m mmgeo.crossview.eval \
    --config configs/crossview_convnext_base.yaml \
    --checkpoint checkpoints/crossview/cv_v2_base_20260422_230539/best.pt \
    --no-pool \
    --output results/eval_nopooled.json
```

### Zero-shot eval on HPC (ImageNet-22k weights only, no MMLandmarks training)
```bash
bsub < scripts/eval_crossview_zeroshot.sh
```

### Zero-shot eval locally
```bash
python -m mmgeo.crossview.eval \
    --config configs/crossview_convnext_base.yaml \
    --pretrained-only \
    --no-pool \
    --output results/eval_zeroshot.json
```

---

## 9. Comparison to related work

### Zero-shot vs trained — critical distinction for the report

**Zero-shot** means the model has never seen MMLandmarks data. It uses its pretrained features directly for retrieval.  
**Trained** means the model was fine-tuned on the MMLandmarks `train` split (17,557 landmarks).

These are different evaluation conditions and should be presented separately in the report.

**Zero-shot methods (no MMLandmarks training):**

| Method | Backbone | g2s R@1 | g2s R@5 | g2s R@10 | g2s mAP@1k | Notes |
|--------|----------|---------|---------|---------|-----------|-------|
| ConvNeXt-Base zero-shot | ConvNeXt-Base 88M | 0.25% | 0.76% | 1.13% | 0.59% | ImageNet-22k only |
| MMCLIP | CLIP ViT-L ~300M | 20.5% | — | — | — | from MMLandmarks Table 2 |
| GeoClip | CLIP ViT-L ~300M | 21.1% | — | — | — | CLIP + geo-contrastive pretraining |

**Trained on MMLandmarks (our work, unpooled, paper-comparable):**

| Method | Backbone | g2s R@1 | g2s R@5 | g2s R@10 | g2s mAP@1k | Notes |
|--------|----------|---------|---------|---------|-----------|-------|
| v1 | ConvNeXt-Tiny | — | — | — | — | pooled, old gallery — not paper-comparable |
| **v2 (ep30)** | **ConvNeXt-Base 88M** | **7.21%** | **18.52%** | **25.31%** | **13.10%** | 35 epochs, GPS+DSS |

**Sample4Geo (reference, different dataset):**  
~27% on CVUSA/CVACT — not directly comparable (different dataset, different protocol).

### Key takeaway for the report

The zero-shot ConvNeXt-Base scores 0.25% g2s R@1. After fine-tuning on MMLandmarks, the same backbone reaches 7.21% — a **29× improvement** from domain training alone. Despite this, CLIP-based zero-shot methods still outperform our trained model (20.5% vs 7.21%), because CLIP ViT-L was pretrained on orders of magnitude more data including geo-tagged imagery.

This sets up a clear narrative: cross-view retrieval strongly benefits from domain-specific training, but the backbone and pretraining matter enormously. A CLIP-initialized backbone fine-tuned on MMLandmarks would likely close most of this gap — and that is what GeoClip and multimodal methods pursue.

**s2g results (our v2, trained, unpooled):**
- R@1: 5.20% | R@5: 14.40% | R@10: 18.70% | mAP@1k: 2.66%  
  (s2g is harder because the ground gallery is 733,242 images vs 100,539 for g2s)

---

## 10. v2 → v3 changes summary

| Area | v2 | v3 |
|------|----|----|
| Backbone | `convnext_base.fb_in22k` (ImageNet-22k only) | `convnext_base.fb_in22k_ft_in1k_384` (22k → 1k fine-tune at 384px) |
| Image size | 224 px | 384 px (matches backbone resolution) |
| Batch size | 64 | 24 (reduced for 384px memory) |
| Eval batch size | 384 | 192 (reduced for 384px) |
| Label smoothing | 0.0 | 0.1 |
| Loss | `SymmetricInfoNCE` (1 ground / landmark) | `MultiPositiveInfoNCE` (3 ground / landmark) |
| n_ground | 1 | 3 |
| Pool queries (training eval) | True (pooled) | **False** (unpooled, paper-comparable from start) |
| Run prefix | `cv_v2_base_` | `cv_v3_base_` |

### Multi-positive InfoNCE — what changed

**`src/mmgeo/crossview/losses.py`** — new `MultiPositiveInfoNCE` class:
- Receives `ground_embeds (B×K, D)` and `sat_embeds (B, D)`
- s2g direction: each satellite has K soft positives (uniform 1/K each), smoothed with label_smoothing
- g2s direction: each of the B×K ground images has one hard satellite positive (`j // K`)
- K=1 reduces exactly to `SymmetricInfoNCE`

**`src/mmgeo/crossview/dataset.py`** — `MMLCrossViewDataset` gains `n_ground` parameter:
- `n_ground=1` (default): backward-compatible, `ground_img` is `(3, H, W)` as before
- `n_ground=K`: `ground_img` is `(K, 3, H, W)`, K images sampled randomly per landmark

**`src/mmgeo/crossview/train.py`**:
- `train_one_epoch`: detects 5D ground tensor, reshapes to `(B×K, 3, H, W)` before forward; diagnostics use mean-pooled per-landmark embedding
- `train()`: reads `n_ground` from config, passes to train dataset; DSS embedding dataset keeps `n_ground=1` (needs single embeddings per landmark)
- `_run_eval`: `pool_queries` reads from `cfg["evaluation"]["pool_queries"]` (default True; pass `pool_queries=False` in config for the paper-comparable v3 protocol).

---

## 11. v1 → v2 changes summary

| Area | v1 | v2 |
|------|----|----|
| Backbone | ConvNeXt-Tiny | ConvNeXt-Base (88M params) |
| Embed dim | 256 | 1024 (native) |
| Image size | 224 px | 224 px |
| Epochs | 20 | 35 |
| Hard negatives | GPS only (partial) | GPS (4 ep) → DSS (31 ep) |
| Gallery | query-only satellite | full index + query satellite (100,539 / 733,242) |
| Metrics | R@1/5/10 | R@1/5/10 + mAP@1k |
| Query pooling | none | optional (pool or no-pool) |
| Eval speed | batch=128, every epoch | batch=384, every 3 epochs |
| Resume support | none | `--resume` from any `.pt` |
| Per-epoch checkpoint | only at end | `last.pt` saved after every epoch |
| Experiment logging | basic | RunLogger: versioned run dirs, metrics CSV, config copy |
| Best g2s R@1 | 9.73% | 17.60% (pooled) |

---

## 12. Limitations and what's next (Task 7)


### Known limitations
- **Single positive per batch step:** training uses one ground + one satellite image per landmark per iteration. MMLandmarks has ~18 ground images per landmark; we don't use all of them simultaneously in one loss computation (multi-positive InfoNCE). This is a valid simplification but not the maximum the data allows.
- **No text/tag modality:** this is a pure image-image baseline. Other team members are exploring text+image and GPS-aware models.
- **224px resolution:** 256px exceeded GPU memory at batch=64. Gradient checkpointing or mixed precision could unlock higher resolution in a future run.

### Eval jobs — completed ✓
- **Unpooled trained eval** (job 28291418): g2s R@1=7.21%, mAP@1k=13.10% — results in `logs/eval_nopooled_20260426_025336.json`
- **Zero-shot eval** (job 28291442): g2s R@1=0.25%, mAP@1k=0.59% — results in `logs/eval_zeroshot_20260426_025656.json`

All TBD rows in the tables above have been filled in. The baseline is fully evaluated.

### Task 7 — Report integration
For the final report, the baseline section should clearly state:

1. **Method:** shared-encoder contrastive retrieval, symmetric InfoNCE, Sample4Geo-style GPS+DSS hard-negative sampling
2. **Backbone:** ConvNeXt-Base pretrained on ImageNet-22k, fine-tuned on MMLandmarks
3. **Training data:** MMLandmarks `train` split — 17,557 landmarks (ground + satellite pairs)
4. **Evaluation protocol:** unpooled (18,689 individual ground queries), full 100,539-image satellite gallery — same as MMLandmarks Table 2
5. **Results:** R@1, R@5, R@10, mAP@1k for both g2s and s2g directions
6. **Zero-shot reference:** also report ConvNeXt-Base zero-shot to show the contribution of MMLandmarks training
7. **Comparison note:** MMCLIP and GeoClip are zero-shot; our trained model is a separate experimental condition
8. **Limitations:** image-only, single positive per step, 224px, ConvNeXt-Base (not CLIP backbone)
