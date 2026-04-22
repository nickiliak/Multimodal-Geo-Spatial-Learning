# GeoClip — Current Approach & Benchmarks

Team-facing reference for the GeoClip branch of the project. For paper deep-dive,
design rationale, and future extensions, see
[docs/personal/Nikos/geoclip_approach.md](../personal/Nikos/geoclip_approach.md).

## What it is

GeoClip treats geo-localization as **image → GPS retrieval** in a shared 512-dim embedding
space. A frozen CLIP ViT/L-14 backbone + a trainable linear head encodes the image; an
Equal Earth projection + hierarchical Random Fourier Features + MLPs encode GPS. At
inference we embed all gallery GPS points once and argmax cosine similarity per query
image. We start from the authors' pretrained weights — zero-shot is the current baseline.

## Code layout

| File | Role |
|---|---|
| [src/mmgeo/geolocalizations/geoclip/geoclip_baseline.py](../../src/mmgeo/geolocalizations/geoclip/geoclip_baseline.py) | `GeoClipBaseline` — gallery build + batch inference; CSV loaders |
| [src/mmgeo/geolocalizations/geoclip/dataset.py](../../src/mmgeo/geolocalizations/geoclip/dataset.py) | `MMLDataset` — PIL→tensor+GPS pairs for the training `DataLoader` |
| [src/mmgeo/geolocalizations/geoclip/lit_module.py](../../src/mmgeo/geolocalizations/geoclip/lit_module.py) | Lightning wrapper — symmetric InfoNCE, per-epoch eval, save-best-Acc@25km |
| [src/mmgeo/geolocalizations/geoclip/evaluate.py](../../src/mmgeo/geolocalizations/geoclip/evaluate.py) | `haversine`, `accuracy_at_thresholds`, `median_error` |
| [scripts/geoclip_train.py](../../scripts/geoclip_train.py) | Fine-tuning entrypoint (Lightning `Trainer.fit`) |
| [configs/geoclip_baseline.yaml](../../configs/geoclip_baseline.yaml), [configs/geoclip_train.yaml](../../configs/geoclip_train.yaml) | Inference and training configs |
| [notebooks/team/03_geoclip_zeroshot.ipynb](../../notebooks/team/03_geoclip_zeroshot.ipynb) | Zero-shot eval notebook |
| [notebooks/team/04_geoclip_finetuned.ipynb](../../notebooks/team/04_geoclip_finetuned.ipynb) | Fine-tuned eval notebook (reads `models/best_geoclip_baseline.pth`) |

## Evaluation protocol

- **Gallery:** 17,557 train-landmark GPS points from `train/mml_train.csv`. Toggling
  `gallery.include_index: true` extends it with 101,302 index-satellite points
  (118,859 total).
- **Queries:** all 18,688 query ground images (multiple images per landmark, each scored
  against the landmark's ground-truth GPS).
- **Metric:** Haversine distance → Accuracy @ {1, 25, 200, 750, 2500} km + median / mean
  error in km.
- **Params:** 438M total, 10.4M trainable (linear image head + full location encoder);
  the CLIP ViT/L-14 vision tower is frozen.

## Zero-shot benchmark

Run on a V100 against the 17,557-point gallery
(source: [Output_28246313.out](../../Output_28246313.out)):

| Threshold (km) | Accuracy (%) |
|---:|---:|
| 1    | 19.22 |
| 25   | 34.56 |
| 200  | 46.84 |
| 750  | 71.26 |
| 2500 | 91.33 |

Median error 249.3 km · Mean error 686.1 km · ~7.5 min GPU inference.

### Paper contrast

| Method | Dataset | Gallery | @1 km | @25 km | @200 km | @750 km | @2500 km |
|---|---|---:|---:|---:|---:|---:|---:|
| GeoClip (own paper) | Im2GPS3k (global) | 100K | 14.11 | 34.47 | 50.65 | 69.67 | 83.82 |
| Off-shelf GeoClip (MML paper) | MMlandmarks (US) | 17,557 | 21.37 | 36.44 | 48.57 | 71.45 | 91.50 |
| **Ours (zero-shot)** | MMlandmarks (US) | 17,557 | **19.22** | **34.56** | **46.84** | **71.26** | **91.33** |

We beat the GeoClip paper's own Im2GPS3k number because the US-only task is easier and
the MP-16 training set overlaps with US landmarks. Against the MML paper row — same
dataset, same gallery — we sit ~2 points lower at 1 km. The most plausible explanations
are evaluation add-ons we don't apply: TenCrop augmentation at test time and LLaVA-based
indoor-image filtering.

## Fine-tuning — status: not yet beating zero-shot

Training is wired end-to-end in [scripts/geoclip_train.py](../../scripts/geoclip_train.py):
symmetric InfoNCE with the pretrained `logit_scale`, Adam `lr=1e-4`, batch size 32,
10-epoch target. After each epoch we re-embed the gallery and evaluate on the full query
set; a checkpoint is written **only** when Acc@25km improves over the zero-shot baseline
(34.56%).

First run on V100: epoch 1 Acc@1km **12.97%** (regressed from zero-shot's 19.22%),
epoch 2 partially recovered to **15.10%**, then the job hit the 4-hour LSF wall-clock
limit mid-epoch-3. No epoch cleared the Acc@25km gate, so no checkpoint was saved — the
"fine-tuned" notebook still shows `_TBD_`. Work in progress.

## How to run

```bash
# Zero-shot evaluation
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/team/03_geoclip_zeroshot.ipynb

# Fine-tune (HPC, GPU; writes models/best_geoclip_baseline.pth when a checkpoint improves)
uv run python scripts/geoclip_train.py

# Fine-tuned evaluation (requires checkpoint)
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/team/04_geoclip_finetuned.ipynb
```
