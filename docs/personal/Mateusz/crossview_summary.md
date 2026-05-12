# Cross-View Geo-Localisation — Experiment Summary

## What is this task?

Given a **ground-level photo** (taken by a person on the street), retrieve the matching **satellite image** of the same location from a large database — and vice versa. This is called **cross-view geo-localisation** or cross-view retrieval.

It is hard because the two views look completely different: one is a street-level photo with buildings and trees in the foreground; the other is a top-down aerial image of the same spot. The model must learn an embedding space where the ground and satellite representations of the same location are close together, and far from all other locations.

---

## Dataset — MMLandmarks

MMLandmarks is a large geo-tagged dataset with three non-overlapping splits:

| Split | Landmarks | Ground images | Satellite images | Purpose |
|-------|-----------|--------------|-----------------|---------|
| Train | 17,557 | 310,661 | 186,574 | Model training |
| Query | 1,000 | 18,688 | 1,000 | Evaluation queries |
| Index | ~99,500 | ~714,500 | 99,539 | Evaluation gallery (distractors) |

Each landmark has multiple ground photos (~18 on average) and at least one satellite image. Training and evaluation use completely separate landmarks — no data leakage.

---

## Model — CrossViewModel

A **single shared encoder** processes both ground and satellite images. There is no separate tower for each modality — the same weights handle both, forcing the model to learn modality-agnostic location features.

- **Backbone:** ConvNeXt (Tiny in v1, Base 88M params in v2/v3), pretrained on ImageNet
- **Output:** L2-normalised embedding vector per image (1024-dim in v2/v3)
- **No separate projection head in v2/v3** — the backbone's native features are used directly

The model is **not zero-shot** — it is fine-tuned on MMLandmarks training data.

---

## Training — How it works

### Loss function: InfoNCE (contrastive learning)

Within each batch of B (ground, satellite) pairs, the loss pushes each ground embedding close to its matching satellite embedding, and far from all B−1 other satellites in the same batch. This is called **InfoNCE** (also known as NT-Xent or contrastive loss with in-batch negatives).

In v3 we extended this to **Multi-Positive InfoNCE**: each satellite sees K=2 ground images per step instead of 1, so the loss distributes probability equally across both positives. More signal per step, less variance.

### Hard-negative sampling (Sample4Geo-style)

Random batches give easy negatives (far-away locations that are trivially dissimilar). To force the model to learn fine distinctions, we use a two-phase sampling strategy:

**Phase 1 — GPS-based negatives (epochs 1–4):**  
Batches are built from geographically nearby landmarks. Forces learning of fine visual differences between nearby locations.

**Phase 2 — Dynamic Similarity Sampling / DSS (epochs 5–end):**  
Batches are built from the most similar pairs in *embedding space* (refreshed every epoch). These are the hardest possible negatives — locations that currently look similar to the model but are actually different. This is what drives the main metric gains.

### Learning rate schedule
Cosine annealing with a 3-epoch linear warmup (starting at 10% LR). Minimum LR = 1% of peak. AdamW optimiser.

### Learnable temperature
The InfoNCE temperature τ starts at 0.07 and is learned during training. It controls how "sharp" the similarity distribution is — lower τ = harder assignments.

---

## Evaluation protocol

Evaluation is **unpooled** (paper-comparable): each of the 18,688 ground images is an individual query — including difficult/atypical views of each landmark. The gallery contains all index + query satellite images (100,539 total for g2s).

**Metrics:** Recall@1, Recall@5, Recall@10, mAP@1000  
**Directions:** g2s (ground → satellite) and s2g (satellite → ground)

> **Note on pooling:** an earlier v2 variant averaged all ~18 ground images per landmark into one "consensus" embedding before querying (pooled). This gives artificially high numbers (17.60% vs 7.21%) and is **not** what the MMLandmarks paper uses. All numbers below are **unpooled**.

---

## Experiments

### Zero-shot baseline
Before any training, we ran ConvNeXt-Base with pure ImageNet-22k weights directly on the eval set. No exposure to MMLandmarks data whatsoever.

**Purpose:** quantify how much of the performance comes from domain-specific training vs. general visual features.

**Result:** g2s R@1 = **0.25%** — essentially random. Cross-view retrieval requires domain training; general visual features are not sufficient.

---

### v1 — Proof of concept
First end-to-end run. Smaller backbone (ConvNeXt-Tiny), 224px, 20 epochs, partial hard-negative setup. Evaluated with a smaller pooled gallery (not paper-comparable).

**Purpose:** validate the pipeline works end-to-end.

**Result:** g2s R@1 = ~9.73% (pooled, query-only gallery — not directly comparable to later experiments).

---

### v2 — Stronger backbone, full pipeline
Upgraded to ConvNeXt-Base (88M params, 8× the capacity of Tiny), 224px, 35 epochs, full GPS→DSS hard-negative schedule on the complete gallery.

**What changed from v1:** backbone size, full DSS sampling, 35 epochs, complete index gallery, proper metrics (mAP@1k added), resumable training, per-epoch checkpointing.

**Purpose:** establish a solid baseline with the full Sample4Geo pipeline.

**Best checkpoint:** epoch 30.

---

### v3 — Higher resolution, multi-positive loss, AMP
Upgraded the backbone to a version additionally fine-tuned at 384px on ImageNet-1k (stronger starting point). Increased image resolution to 384px to match. Added Multi-Positive InfoNCE (K=2 ground images per satellite per step). Added AMP (fp16 mixed precision) to fit larger batches at 384px on a 32GB GPU.

**What changed from v2:**

| | v2 | v3 |
|--|--|--|
| Backbone | ConvNeXt-Base, IN-22k only | ConvNeXt-Base, IN-22k → IN-1k at 384px |
| Image size | 224 px | 384 px |
| Loss | Symmetric InfoNCE (1 positive) | Multi-Positive InfoNCE (K=2 positives) |
| Label smoothing | 0.0 | 0.1 |
| Mixed precision | no | fp16 (AMP) |
| Batch size | 64 | 16 (memory constraint at 384px) |

**Why these changes:**
- 384px matches the backbone's fine-tuning resolution — higher fidelity features
- Multi-positive loss uses more of the available ground images per step
- AMP was required to fit 48 images (16 sat + 32 ground) at 384px into 32GB GPU memory

**Status:** ✅ Complete — 36 epochs, 16.5 hours. Best checkpoint: epoch 36.

---

## Results (unpooled, paper-comparable)

### Ground → Satellite (g2s)

| Model | Training | R@1 | R@5 | R@10 | mAP@1k | Notes |
|-------|----------|-----|-----|------|--------|-------|
| ConvNeXt-Base zero-shot | ImageNet-22k only | 0.25% | 0.76% | 1.13% | 0.59% | No MMLandmarks training |
| v2 | MMLandmarks (35 ep) | 7.21% | 18.52% | 25.31% | 13.10% | ep30 best |
| **v3** | **MMLandmarks (36 ep)** | **8.58%** | **18.13%** | **22.29%** | **13.25%** | **ep36 best — final** |
| MMCLIP (paper) | **zero-shot** (no MMLandmarks) | 20.5% | — | — | — | CLIP ViT-L ~300M |
| GeoClip (paper) | **zero-shot** (no MMLandmarks) | 21.1% | — | — | — | CLIP ViT-L + geo pretraining |

### Satellite → Ground (s2g)

| Model | R@1 | R@5 | R@10 | mAP@1k |
|-------|-----|-----|------|--------|
| v2 (ep30) | 5.20% | 14.40% | 18.70% | 2.66% |
| **v3 (ep36)** | **5.40%** | **11.10%** | **13.70%** | **1.85%** |

> **Important:** MMCLIP and GeoClip are *zero-shot* — they never trained on MMLandmarks. Our models *did* train on MMLandmarks. These are different experimental conditions and should not be directly compared as competing methods.

### v3 training progression (evals at ep 9, 18, 27, 36)

| Epoch | g2s R@1 | g2s R@5 | g2s R@10 | s2g R@1 | Batch acc | Margin |
|-------|---------|---------|---------|---------|-----------|--------|
| 9 | 6.61% | 14.92% | 20.18% | 4.30% | 51.1% | 0.119 |
| 18 | 8.32% | 18.41% | 23.51% | 5.00% | 81.3% | 0.179 |
| 27 | 8.42% | 18.01% | 22.49% | 5.50% | 92.0% | 0.201 |
| **36** | **8.58%** | **18.13%** | **22.29%** | **5.40%** | **93.5%** | **0.209** |

Gains were front-loaded: most improvement happened by epoch 18 (+1.71% g2s R@1 vs ep9). The final 18 epochs added only +0.26%, suggesting the model converged well before epoch 36.

---

## Key takeaways

1. **Domain training is essential.** Zero-shot ConvNeXt-Base = 0.25% R@1. After fine-tuning on MMLandmarks: 8.58%. A ~34× improvement from training alone.

2. **v3 beats v2 by +1.37% R@1** (8.58% vs 7.21%). The gains come from two combined changes: higher resolution (384px vs 224px) matching the backbone's pretraining, and multi-positive InfoNCE providing richer signal per step.

3. **Hard negatives drive the gains.** Batch accuracy climbs from ~49% (end of GPS phase) to 93.5% by epoch 36. The jump from GPS to DSS hard negatives (epoch 5) initially causes a loss spike as the task suddenly gets harder — then the model adapts and improves steadily.

4. **CLIP backbone dominates even zero-shot.** MMCLIP/GeoClip reach 20–21% with no MMLandmarks training. The gap vs. our trained ConvNeXt (8.58%) is almost entirely explained by backbone quality — CLIP ViT-L was pretrained on far more data, including geo-tagged imagery.

5. **The clearest path to closing the CLIP gap** is to initialise the encoder from CLIP weights and fine-tune — which is exactly what GeoClip does, and what multimodal extensions of this work pursue.
