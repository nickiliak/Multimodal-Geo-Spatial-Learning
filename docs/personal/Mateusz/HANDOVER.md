# Agent Handover — Cross-View Geo-Localisation (Mateusz)

This document is a cold-start brief for the next agent continuing work on this branch.
Everything needed to pick up immediately is here — no session memory required.

---

## Project in One Paragraph

Cross-view geo-localisation on MMLandmarks: given a ground-level photo, retrieve the
matching satellite image from a 100k+ gallery (and vice versa). A single shared
ConvNeXt-Base encoder (88M params, identical weights for both modalities) is trained
with InfoNCE loss and Sample4Geo-style two-phase hard-negative sampling (GPS neighbours
→ DSS). This model is one component in a larger team pipeline: GeoClip (colleague Kostas)
first narrows the search space to landmarks within ~20km of a GPS estimate, then this
model does fine-grained retrieval within that limited candidate set. We hand off `.pt`
checkpoint files to Kostas. The training and evaluation code lives in
`src/mmgeo/crossview/`.

---

## Repository Key Files

| File | Purpose |
|------|---------|
| `src/mmgeo/crossview/train.py` | Training loop, `_run_eval()`, AMP, hard-negative schedule |
| `src/mmgeo/crossview/evaluate.py` | `extract_embeddings`, `compute_retrieval_metrics`, `compute_per_landmark_retrieval_metrics`, `evaluate_crossview` |
| `src/mmgeo/crossview/eval.py` | Standalone eval CLI (`python -m mmgeo.crossview.eval`) |
| `src/mmgeo/crossview/losses.py` | `SymmetricInfoNCE`, `MultiPositiveInfoNCE` |
| `src/mmgeo/crossview/dataset.py` | `MMLCrossViewDataset` (landmark-indexed), `MMLImageDataset`, `UniqueLandmarkSampler` |
| `src/mmgeo/crossview/hard_negatives.py` | `build_gps_neighbors`, `compute_landmark_embeddings`, `build_similarity_neighbors`, `HardNegativeBatchSampler` |
| `src/mmgeo/crossview/model.py` | `CrossViewModel` — shared ConvNeXt backbone, L2-norm output |
| `configs/crossview_convnext_base.yaml` | v2 config (restored to true v2 settings) |
| `configs/crossview_convnext_base_v3.yaml` | v3 config |
| `configs/crossview_convnext_base_v4.yaml` | v4 ablation config |
| `configs/crossview_convnext_base_384_zeroshot.yaml` | Zero-shot eval config |
| `checkpoints/crossview/cv_v2_base_20260422_230539/` | v2 checkpoint — best=ep30 |
| `checkpoints/crossview/cv_v3_base_20260429_055409/` | v3 checkpoint — best=ep36 |
| `checkpoints/crossview/cv_v4_base_20260513_013400/` | v4 checkpoint — best=ep36 |
| `eval_results/` | Eval JSONs for all models (zero-shot, v2, v3, v4) |
| `docs/personal/Mateusz/crossview_results.md` | Unified results table |
| `docs/personal/Mateusz/sample4geo_explainer.md` | Full technical explainer for teammates/exam |
| `scripts/plot_crossview_results.py` | Generates figures — PER_IMAGE and PER_LANDMARK dicts need filling |
| `docs/personal/Mateusz/figures/` | Output directory for plots |

---

## Code Changes Made (completed, needs committing)

### `src/mmgeo/crossview/evaluate.py`
- Added `compute_per_landmark_retrieval_metrics()` — score-space aggregation, one result
  per landmark regardless of how many ground images it has.
- Updated `evaluate_crossview()`: loops over both `["max", "mean"]` when
  `landmark_agg is not None`. Keys: `lm_max_recall@k` and `lm_mean_recall@k`.
  Raw per-image embeddings saved as `q_embeds_raw` before any pooling.

### `src/mmgeo/crossview/train.py`
- `_run_eval()`: added `landmark_agg: str | None = None`, passed to both
  `evaluate_crossview()` calls. During training, called with `landmark_agg=None` for
  speed — per-landmark metrics only appear in standalone eval runs.
- AMP: `GradScaler`, `torch.autocast`, scaler state saved in checkpoints as
  `"scaler_state_dict"`.

### `src/mmgeo/crossview/eval.py`
- `--landmark-agg max|mean` / `--no-landmark-agg` flags added.
- `landmark_agg=args.landmark_agg` passed to `_run_eval()`.

### Configs
- `configs/crossview_convnext_base.yaml`: restored to true v2 settings
  (fb_in22k, 224px, batch=64, label_smoothing=0.0, pool_queries=true).
- `configs/crossview_convnext_base_v4.yaml`: new — fb_in22k, 224px, batch=64,
  n_ground=2, label_smooth=0.1, pool_queries=false.

### Scripts
- `scripts/run_crossview_convnext_base_v4.sh` — v4 training (complete)
- `scripts/eval_crossview_v2/v3/v4/zeroshot.sh` — eval scripts, all with `--landmark-agg max`
- Wall time bumped to 8h for 384px evals (v3, zero-shot), 4h for v4

---

## Current Status — All Jobs Complete

All training and evaluation is finished. Nothing pending on HPC.

| Run | Status | Best epoch | Checkpoint |
|-----|--------|-----------|-----------|
| v2 | ✅ Complete | ep30 | `cv_v2_base_20260422_230539/best.pt` |
| v3 | ✅ Complete | ep36 | `cv_v3_base_20260429_055409/best.pt` |
| v4 | ✅ Complete | ep36 | `cv_v4_base_20260513_013400/best.pt` |
| Zero-shot eval | ✅ Complete | — | `eval_results/eval_zeroshot_*.json` |
| v2 eval | ✅ Complete | — | `eval_results/eval_v2_*.json` |
| v3 eval | ✅ Complete | — | `eval_results/eval_v3_*.json` |
| v4 eval | ✅ Complete | — | `eval_results/eval_v4_*.json` |

---

## Final Results

### g2s — Master Table (all protocols)

| Model | R@1 img | R@1 mean | R@1 attn | R@1 max | R@5 img | R@5 mean | R@5 max | R@10 img | R@10 mean | R@10 max | mAP img | mAP mean | mAP max |
|-------|---------|----------|----------|---------|---------|----------|---------|----------|-----------|----------|---------|----------|---------|
| Zero-shot | 0.34% | 0.40% | 0.30% | 0.30% | 1.23% | 1.20% | 0.90% | 2.14% | 1.80% | 2.10% | 1.00% | 0.89% | 0.89% |
| v2 (ep30) | 7.21% | 17.60% | 17.50% | **9.00%** | 18.52% | 33.40% | 20.30% | 25.31% | **42.20%** | 27.20% | 13.10% | 25.50% | 15.21% |
| v3 (ep36) | **8.58%** | 18.40% | 18.20% | 7.10% | 18.13% | 31.70% | 14.90% | 22.29% | 37.20% | 18.80% | 13.25% | 25.12% | 11.10% |
| v4 (ep36) | 7.63% | **18.50%** | 18.20% | 8.10% | **19.02%** | **32.40%** | 18.50% | 24.57% | 39.60% | 25.30% | **13.34%** | **25.33%** | 13.72% |

img = per-image (18,688 queries, paper-comparable) | mean = per-lm mean-agg (team primary) | attn = per-lm attn-weighted mean | max = per-lm max-agg (coverage)

**Key finding: attn < mean** (−0.1% to −0.3% R@1). Simple mean wins — diverse viewpoints are not noise.

### g2s — Per-image (18,688 queries)

| Model | R@1 | R@5 | R@10 | mAP@1k |
|-------|-----|-----|------|--------|
| Zero-shot | 0.34% | 1.23% | 2.14% | 1.00% |
| v2 (ep30) | 7.21% | 18.52% | 25.31% | 13.10% |
| v3 (ep36) | **8.58%** | 18.13% | 22.29% | 13.25% |
| v4 (ep36) | 7.63% | 19.02% | 24.57% | 13.34% |

### g2s — Per-landmark max-agg (1,000 landmarks, "any photo wins")

| Model | R@1 | R@5 | R@10 | mAP@1k |
|-------|-----|-----|------|--------|
| Zero-shot | 0.30% | 0.90% | 2.10% | 0.89% |
| v2 (ep30) | **9.00%** | 20.30% | 27.20% | 15.21% |
| v3 (ep36) | 7.10% | 14.90% | 18.80% | 11.10% |
| v4 (ep36) | 8.10% | 18.50% | 25.30% | 13.72% |

### g2s — Per-landmark mean-agg (1,000 landmarks, "average score" — team primary)

| Model | R@1 | R@5 | R@10 | mAP@1k |
|-------|-----|-----|------|--------|
| Zero-shot | 0.40% | 1.20% | 1.80% | 0.89% |
| v2 (ep30) | 17.60% | 33.40% | 42.20% | 25.50% |
| v3 (ep36) | 18.40% | 31.70% | 37.20% | 25.12% |
| v4 (ep36) | **18.50%** | **32.40%** | **39.60%** | **25.33%** |

Mean-agg R@1 ranking: v4 ≈ v3 > v2. Max-agg R@1 ranking: v2 > v4 > v3 (reversed).

### g2s — Per-landmark attn-weighted mean (experimental; consistently ≤ mean)

| Model | R@1 | R@5 | R@10 | mAP@1k |
|-------|-----|-----|------|--------|
| Zero-shot | 0.30% | 1.40% | 1.80% | 0.85% |
| v2 (ep30) | 17.50% | 33.10% | 42.10% | 25.21% |
| v3 (ep36) | 18.20% | 32.00% | 37.20% | 24.85% |
| v4 (ep36) | 18.20% | 32.10% | 39.20% | 25.21% |

Attn: −0.1% to −0.3% vs mean. Use lm_mean_recall@1 as the primary metric.

### s2g — Per-image (= per-landmark; one satellite per landmark, nothing to aggregate)

| Model | R@1 | R@5 | R@10 | mAP@1k |
|-------|-----|-----|------|--------|
| Zero-shot | 0.00% | 0.30% | 0.80% | 0.05% |
| v2 (ep30) | 5.20% | 14.40% | 18.70% | 2.66% |
| v3 (ep36) | **5.40%** | 11.10% | 13.70% | 1.85% |
| v4 (ep36) | 4.00% | 11.30% | 16.80% | 2.12% |

---

## Remaining Tasks

1. **Commit everything** (docs updated, figures generated, eval JSONs already committed):
   ```bash
   git add docs/personal/Mateusz/ scripts/plot_crossview_results.py
   git commit
   ```

2. **Share with Kostas:** send him `eval_results/eval_v4_*.json` (or v2/v3). The keys:
   - `g2s.lm_mean_recall@1` — mean-agg, what the team uses
   - `g2s.lm_max_recall@1` — max-agg, "any photo wins"
   - `g2s.recall@1` — per-image (paper-comparable)
   
   **Checkpoint recommendation:**
   - For **mean-agg / pipeline use**: v4 is marginally best (lm_mean R@1 18.50% vs v3 18.40% vs v2 17.60%)
   - For **landmark coverage (max-agg)**: v2 is best (9.00% vs v4 8.10% vs v3 7.10%)
   - Practical choice: **v4** (same size/speed as v2, better mean-agg, nearly as good max-agg)
   Use v3 if the per-image headline number is more important (8.58%).

---

## Technical Gotchas — Read Before Touching Anything

**v2 pooled vs unpooled confusion (most common mistake):**
`checkpoints/crossview/cv_v2_base_20260422_230539/best_metrics.json` shows R@1=17.6%.
This is POOLED. The unpooled v2 number is 7.21%. Never compare 17.6% directly to v3's
8.58% (which is unpooled). Note: score-space mean-agg (17.60%) = embedding-space pooling
mathematically — they are equivalent for L2-normalized vectors.

**v3 per-landmark max (7.10%) is LOWER than per-image (8.58%) — this is not a bug:**
v3's improvements are concentrated in easy, high-photo-count landmarks. When each landmark
gets one vote, v3 actually identifies fewer distinct locations than v2 (9.00%). By the
fairest metric, v2 is the best model. This is the key finding from the ablation study.

**Per-landmark key names:**
`lm_max_recall@k` and `lm_mean_recall@k` — both computed in one eval pass.
Old logs/JSONs from before the code change used `lm_recall@k` (max only, now obsolete).

**During-training eval vs standalone eval:**
`_run_eval()` inside the training loop uses `landmark_agg=None` for speed — the
`eval_curves.csv` files only contain per-image metrics. Per-landmark numbers only
come from standalone eval JSON files.

**Dataset is already landmark-uniform:**
`MMLCrossViewDataset.__len__()` returns 17,557 (landmarks, not images). Each epoch,
each landmark gets exactly one slot. No custom weighting needed.

**AMP:** Required at 384px/batch=16 (v3). Not needed at 224px/batch=64 (v2, v4).

**Normalization:** Use `timm.data.resolve_data_config({}, model=model.backbone)` —
do NOT use standard ImageNet mean/std directly.

**384px eval wall time:** s2g index has 733k ground images at 384px — takes ~4h to
embed. All 384px eval scripts now set to 8h wall time. v4 (224px) set to 4h.
