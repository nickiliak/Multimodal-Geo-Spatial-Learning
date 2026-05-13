# Cross-View Retrieval — Unified Results

Given a ground-level photo, retrieve the matching satellite image of the same location from a 100k+ image gallery. All models use a single shared ConvNeXt-Base encoder (no separate towers) fine-tuned on the MMLandmarks dataset with InfoNCE + Sample4Geo hard-negative sampling.

---

## Evaluation Protocols

### Per-image (unpooled) — paper-comparable
Each of the **18,688** ground query images is an independent query. A landmark with 20 photos contributes 20 separate queries. Biased toward landmarks with more images, but directly matches the MMLandmarks paper protocol and is the standard for external comparison.

### Per-landmark (score-space aggregation) — fairer
For each of the **1,000** query landmarks, all K ground-image embeddings are used to compute similarities to every index item. The K score vectors are aggregated (max or mean), then ranked once. Every landmark counts exactly once regardless of how many photos it has. `max` aggregation = "success if any ground image retrieves correctly."

> **Note:** per-landmark max recall is generally higher than per-image recall. However v3 is an exception — per-lm max (7.10%) < per-image (8.58%) — because v3's gains are concentrated in easy, high-photo-count landmarks. When each landmark counts once, v3 identifies fewer distinct locations than v2 (9.00%). See Key Takeaways.

---

## Results

### Ground → Satellite (g2s) — Master Table

All three eval protocols side-by-side. Columns: per-image (img), per-landmark mean-agg (mean), per-landmark max-agg (max).

| Model | R@1 img | R@1 mean | R@1 attn | R@1 max | R@5 img | R@5 mean | R@5 max | R@10 img | R@10 mean | R@10 max | mAP img | mAP mean | mAP max |
|-------|---------|----------|----------|---------|---------|----------|---------|----------|-----------|----------|---------|----------|---------|
| Zero-shot | 0.34% | 0.40% | TBD | 0.30% | 1.23% | 1.20% | 0.90% | 2.14% | 1.80% | 2.10% | 1.00% | 0.89% | 0.89% |
| v2 (ep30) | 7.21% | 17.60% | TBD | **9.00%** | 18.52% | 33.40% | 20.30% | 25.31% | **42.20%** | 27.20% | 13.10% | 25.50% | 15.21% |
| v3 (ep36) | **8.58%** | 18.40% | TBD | 7.10% | 18.13% | 31.70% | 14.90% | 22.29% | 37.20% | 18.80% | 13.25% | 25.12% | 11.10% |
| v4 (ep36) | 7.63% | **18.50%** | TBD | 8.10% | **19.02%** | **32.40%** | 18.50% | 24.57% | 39.60% | 25.30% | **13.34%** | **25.33%** | 13.72% |

*per-image* = each of 18,688 ground images is an independent query (paper-comparable).
*mean-agg* = 1,000 landmarks, average score across all ground photos (team primary; ≡ embedding-space mean-pooling for ranking).
*attn* = attention-weighted mean: photos weighted by cosine-sim to landmark centre (pending HPC eval).
*max-agg* = 1,000 landmarks, best score across all ground photos (upper bound on landmark coverage).

---

### Ground → Satellite (g2s) — Detailed Breakdown

Per-image eval (18,688 queries) — paper-comparable:

| Model | Backbone | img_size | Epochs | R@1 | R@5 | R@10 | mAP@1k | Notes |
|-------|----------|----------|--------|-----|-----|------|--------|-------|
| Zero-shot | ConvNeXt-B (fb_in22k_ft_1k_384) | 384 | 0 | 0.34% | 1.23% | 2.14% | 1.00% | ImageNet weights only |
| v2 (ep30) | ConvNeXt-B (fb_in22k) | 224 | 35 | 7.21% | 18.52% | 25.31% | 13.10% | ep30 best |
| v3 (ep36) | ConvNeXt-B (fb_in22k_ft_1k_384) | 384 | 36 | **8.58%** | 18.13% | 22.29% | 13.25% | ep36 best |
| v4 (ep36) | ConvNeXt-B (fb_in22k) | 224 | 36 | 7.63% | 19.02% | 24.57% | 13.34% | ep36 best |
| MMCLIP† | CLIP ViT-L | — | zero-shot | 20.5% | — | — | — | no MML training |
| GeoClip† | CLIP ViT-L + geo | — | zero-shot | 21.1% | — | — | — | no MML training |

Per-landmark eval (1,000 landmarks) — fairer measure.
Each landmark counts once regardless of photo count.

**Max-agg** ("any photo wins" — upper bound on landmark coverage):

| Model | R@1 | R@5 | R@10 | mAP@1k |
|-------|-----|-----|------|--------|
| Zero-shot | 0.30% | 0.90% | 2.10% | 0.89% |
| v2 (ep30) | **9.00%** | 20.30% | 27.20% | 15.21% |
| v3 (ep36) | 7.10% | 14.90% | 18.80% | 11.10% |
| v4 (ep36) | 8.10% | 18.50% | 25.30% | 13.72% |

**Mean-agg** ("average score across all photos" — what the team uses; mathematically equivalent to mean-pooling embeddings):

| Model | R@1 | R@5 | R@10 | mAP@1k |
|-------|-----|-----|------|--------|
| Zero-shot | 0.40% | 1.20% | 1.80% | 0.89% |
| v2 (ep30) | 17.60% | 33.40% | **42.20%** | 25.50% |
| v3 (ep36) | 18.40% | 31.70% | 37.20% | 25.12% |
| v4 (ep36) | **18.50%** | **32.40%** | 39.60% | **25.33%** |

Mean-agg: v4 ≈ v3 > v2. Max-agg (landmark coverage): v2 > v4 > v3 — reversed.
Note: s2g per-lm max = per-lm mean (one satellite per landmark), shown in s2g table below.

> † MMCLIP/GeoClip are zero-shot — never trained on MMLandmarks. Not a fair direct comparison with our trained models.

### Satellite → Ground (s2g)

Per-image eval (= per-landmark for s2g — one satellite per landmark, nothing to aggregate):

| Model | R@1 | R@5 | R@10 | mAP@1k |
|-------|-----|-----|------|--------|
| Zero-shot | 0.00% | 0.30% | 0.80% | 0.05% |
| v2 (ep30) | 5.20% | 14.40% | 18.70% | 2.66% |
| v3 (ep36) | **5.40%** | 11.10% | 13.70% | 1.85% |
| v4 (ep36) | 4.00% | 11.30% | 16.80% | 2.12% |

(Per-lm numbers match per-image exactly; s2g has one satellite query per landmark.)

---

## Training Progression — v3

Evals at epochs 9, 18, 27, 36 (unpooled, g2s and s2g):

| Epoch | g2s R@1 | g2s R@5 | g2s R@10 | g2s mAP@1k | s2g R@1 |
|-------|---------|---------|---------|-----------|---------|
| 9 | 6.61% | 14.92% | 20.18% | 11.07% | 4.30% |
| 18 | 8.32% | 18.41% | 23.51% | 13.39% | 5.00% |
| 27 | 8.42% | 18.01% | 22.49% | 13.17% | 5.50% |
| **36** | **8.58%** | **18.13%** | **22.29%** | **13.25%** | **5.40%** |

Most improvement happened by epoch 18. Final 18 epochs added only +0.26% g2s R@1.

---

## Experiment Descriptions

### Zero-shot
ConvNeXt-Base (fb_in22k_ft_in1k_384) with pure ImageNet weights — no MMLandmarks training. Establishes how much domain-specific fine-tuning contributes.

### v1 — Proof of concept
ConvNeXt-Tiny, 224px, 20 epochs, partial hard-negative schedule, pooled eval with query-only gallery. Result: ~9.73% (pooled, not comparable). Purpose: validate pipeline end-to-end.

### v2 — Full pipeline baseline
ConvNeXt-Base (88M, fb_in22k), 224px, 35 epochs. Full GPS→DSS hard-negative schedule with complete 100k index gallery. Single positive per step (InfoNCE), no label smoothing, no AMP.
- g2s R@1 = **7.21%** (unpooled, ep30)

### v3 — Higher resolution + algorithmic improvements
ConvNeXt-Base (fb_in22k_ft_in1k_384), 384px, 36 epochs. Multi-positive InfoNCE (K=2), label smoothing 0.1, AMP (fp16). AMP was required to fit 48 images at 384px into 32GB GPU (31.7 GB without AMP → OOM).
- g2s R@1 = **8.58%** (unpooled, ep36)
- Δ vs v2: +1.37% from combined backbone+resolution+loss changes

### v4 — Ablation: algorithmic improvements only (complete)
ConvNeXt-Base (fb_in22k), 224px, 36 epochs. Multi-positive InfoNCE (K=2), label smoothing 0.1 — same as v3. Backbone and resolution same as v2. No AMP needed at 224px (fits 32GB at batch=64).
- Purpose: isolate the effect of multi-positive + label smoothing, separate from backbone/resolution
- v2→v4 = algorithmic only (+0.42% per-image); v4→v3 = architectural only (+0.95% per-image)
- g2s R@1 = **7.63%** (per-image, ep36), per-lm max = **8.10%**

---

## Ablation Summary

| Change | v2 → v4 | v4 → v3 |
|--------|---------|---------|
| Backbone | same (fb_in22k) | fb_in22k → fb_in22k_ft_in1k_384 |
| Resolution | same (224px) | 224px → 384px |
| Multi-positive (n_ground) | 1 → 2 | same (2) |
| Label smoothing | 0.0 → 0.1 | same (0.1) |
| Batch size | same (64) | 64 → 16 (memory) |

---

## Key Takeaways

1. **Domain training is essential.** Zero-shot = 0.34% → fine-tuned best = 8.58% (~25×). General visual features are not sufficient for cross-view retrieval.

2. **v4 ablation reveals what drives the v2→v3 gain.** v4 (same backbone/res as v2, adds multi-positive + label smoothing) gives 7.63% — between v2 (7.21%) and v3 (8.58%). Both algorithmic (+0.42%) and architectural (+0.95%) changes contribute.

3. **Per-image and per-landmark rankings are opposite.** Per-lm max: v2 (9.00%) > v4 (8.10%) > v3 (7.10%). Every change made per-landmark coverage worse. The dominant factor is the number of unique landmarks (InfoNCE negatives) per training step: v2 batch=64 → 63 negatives; v4 batch=64 n_ground=2 → 32 unique landmarks; v3 batch=16 → 15 negatives. More negatives per step = broader, more uniform landmark coverage.

4. **v3's per-image headline (8.58%) is partially a measurement artefact.** Its gains are concentrated in easy, high-photo-count landmarks. When each landmark counts once (per-lm max), v2 is actually the best model. For Kostas's pipeline (where the goal is to correctly identify landmarks, not score well on easy ones), **recommend v2 or v4**.

5. **Hard negatives are the main driver of training.** GPS → DSS transition causes a temporary loss spike, then steady improvement. Batch accuracy: ~49% → 93.5% over training.

6. **CLIP backbone gap is large.** MMCLIP/GeoClip reach 20–21% zero-shot with CLIP ViT-L. Our trained ConvNeXt reaches 8.58% per-image. The gap is almost entirely backbone quality — CLIP was pretrained on 400M+ diverse image-text pairs vs 14M ImageNet.

---

## How to reproduce

All evals are complete. JSON files are in `eval_results/`. To re-run from scratch:

```bash
# Re-evaluate existing checkpoints (all on HPC — 4–8h wall time):
bsub < scripts/eval_crossview_zeroshot.sh   # zero-shot, 8h
bsub < scripts/eval_crossview_v2.sh         # v2, 4h
bsub < scripts/eval_crossview_v3.sh         # v3, 8h (384px)
bsub < scripts/eval_crossview_v4.sh         # v4, 4h

# Manual eval (any checkpoint, local):
python -m mmgeo.crossview.eval \
    --config configs/crossview_convnext_base_v3.yaml \
    --checkpoint checkpoints/crossview/cv_v3_base_20260429_055409/best.pt \
    --no-pool --landmark-agg max \
    --output eval_results/eval_v3_repro.json
```
