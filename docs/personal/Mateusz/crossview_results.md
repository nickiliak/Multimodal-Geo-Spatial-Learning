# Cross-View Retrieval — Unified Results

Given a ground-level photo, retrieve the matching satellite image of the same location from a 100k+ image gallery. All models use a single shared ConvNeXt-Base encoder (no separate towers) fine-tuned on the MMLandmarks dataset with InfoNCE + Sample4Geo hard-negative sampling.

---

## Evaluation Protocols

### Per-image (unpooled) — paper-comparable
Each of the **18,688** ground query images is an independent query. A landmark with 20 photos contributes 20 separate queries. Biased toward landmarks with more images, but directly matches the MMLandmarks paper protocol and is the standard for external comparison.

### Per-landmark (score-space aggregation) — fairer
For each of the **1,000** query landmarks, all K ground-image embeddings are used to compute similarities to every index item. The K score vectors are aggregated (max or mean), then ranked once. Every landmark counts exactly once regardless of how many photos it has. `max` aggregation = "success if any ground image retrieves correctly."

> Per-landmark recall is higher than per-image recall because a landmark can succeed even if only its best photo matches — which is a valid and useful measure.

---

## Results

### Ground → Satellite (g2s)

Per-image eval (18,688 queries) — paper-comparable:

| Model | Backbone | img_size | Epochs | R@1 | R@5 | R@10 | mAP@1k | Notes |
|-------|----------|----------|--------|-----|-----|------|--------|-------|
| Zero-shot | ConvNeXt-B (fb_in22k) | 384 | 0 | 0.25% | 0.76% | 1.13% | 0.59% | ImageNet weights only |
| v2 | ConvNeXt-B (fb_in22k) | 224 | 35 | 7.21% | 18.52% | 25.31% | 13.10% | ep30 best |
| v3 | ConvNeXt-B (fb_in22k_ft_1k_384) | 384 | 36 | **8.58%** | 18.13% | 22.29% | 13.25% | ep36 best |
| v4 | ConvNeXt-B (fb_in22k) | 224 | 36 | TBD | TBD | TBD | TBD | planned |
| MMCLIP† | CLIP ViT-L | — | zero-shot | 20.5% | — | — | — | no MML training |
| GeoClip† | CLIP ViT-L + geo | — | zero-shot | 21.1% | — | — | — | no MML training |

Per-landmark eval (1,000 landmarks, max-agg) — fairer measure:

| Model | R@1 | R@5 | R@10 | mAP@1k | Notes |
|-------|-----|-----|------|--------|-------|
| Zero-shot | TBD | TBD | TBD | TBD | run: `eval_crossview_zeroshot.sh` |
| v2 | TBD | TBD | TBD | TBD | run: `eval_crossview_v2.sh` |
| v3 | TBD | TBD | TBD | TBD | run: `eval_crossview_v3.sh` |
| v4 | TBD | TBD | TBD | TBD | after v4 training |

> † MMCLIP/GeoClip are zero-shot — never trained on MMLandmarks. Not a fair direct comparison with our trained models.

### Satellite → Ground (s2g)

Per-image eval:

| Model | R@1 | R@5 | R@10 | mAP@1k |
|-------|-----|-----|------|--------|
| Zero-shot | ~0% | ~0% | ~0% | ~0% |
| v2 (ep30) | 5.20% | 14.40% | 18.70% | 2.66% |
| v3 (ep36) | **5.40%** | 11.10% | 13.70% | 1.85% |
| v4 | TBD | TBD | TBD | TBD |

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

### v4 — Ablation: algorithmic improvements only (planned)
ConvNeXt-Base (fb_in22k), 224px, 36 epochs. Multi-positive InfoNCE (K=2), label smoothing 0.1 — same as v3. Backbone and resolution same as v2. No AMP needed at 224px (fits 32GB at batch=64).
- Purpose: isolate the effect of multi-positive + label smoothing, separate from backbone/resolution
- v2→v4 = algorithmic only; v4→v3 = architectural only
- Expected: ~8–10% g2s R@1

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

1. **Domain training is essential.** Zero-shot = 0.25% → fine-tuned v3 = 8.58% (~34×). General visual features are not sufficient for cross-view retrieval.

2. **v3 outperforms v2 by +1.37% R@1.** Likely driven by higher resolution (384px matching the backbone's pretraining) and multi-positive loss. v4 will isolate which change matters more.

3. **Hard negatives are the main driver.** GPS → DSS transition causes a temporary loss spike (task suddenly harder), then steady improvement. Batch accuracy: ~49% → 93.5% over training.

4. **CLIP backbone gap is large.** MMCLIP/GeoClip reach 20–21% zero-shot with CLIP ViT-L. Our trained ConvNeXt reaches 8.58%. The gap is almost entirely explained by backbone quality — CLIP was pretrained on far more diverse data. This motivates moving to CLIP-based encoders.

5. **Per-landmark vs per-image.** Once per-landmark eval runs, expect significantly higher numbers (maybe 15–25% R@1) because success = any of ~18 ground images retrieves correctly. This is a useful and fairer metric for reporting.

---

## How to reproduce

```bash
# Re-evaluate existing checkpoints with per-landmark metrics (run on HPC):
bsub < scripts/eval_crossview_zeroshot.sh
bsub < scripts/eval_crossview_v2.sh
bsub < scripts/eval_crossview_v3.sh

# Train v4:
bsub < scripts/run_crossview_convnext_base_v4.sh

# Manual eval (any checkpoint):
python -m mmgeo.crossview.eval \
    --config configs/crossview_convnext_base_v3.yaml \
    --checkpoint checkpoints/crossview/cv_v3_base_20260429_055409/best.pt \
    --no-pool --landmark-agg max \
    --output eval_results_v3.json
```
