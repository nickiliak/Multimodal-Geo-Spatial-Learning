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

- **Gallery:** 99,539 index-satellite GPS points from `index/mml_index_satellite.csv`
  (config `gallery.source: "index"`). Matches the MML paper Sec 5.2 protocol.
  `load_gallery_coords` also supports `source: "train"` (17,557 train-landmark GPS,
  cluster-luck ablation) and `source: "both"` (~118k train + index).
- **Queries:** all 18,688 query ground images (multiple images per landmark, each scored
  against the landmark's ground-truth GPS). Paper reports 18,689 — one image likely
  delisted since their internal snapshot.
- **Metric:** Haversine distance → Accuracy @ {1, 25, 200, 750, 2500} km + median / mean
  error in km.
- **Params:** 438M total, 10.4M trainable (linear image head + full location encoder);
  the CLIP ViT/L-14 vision tower is frozen.

## Zero-shot benchmark

Current headline run on V100, `gallery.source: "index"` (100k index-satellite gallery,
paper protocol):

| Threshold (km) | Accuracy (%) |
|---:|---:|
| 1    |  6.67 |
| 25   | 28.79 |
| 200  | 44.48 |
| 750  | 69.07 |
| 2500 | 91.07 |

Median error 294.3 km · Mean error 724.2 km · ~8 min GPU inference.

**Ablation — train-landmark gallery (not a fair comparison to the paper):**

| Threshold (km) | Accuracy (%) |
|---:|---:|
| 1    | 19.22 |
| 25   | 34.56 |
| 200  | 46.84 |
| 750  | 71.26 |
| 2500 | 91.33 |

This number looks higher than the index-gallery result but is inflated by cluster-luck —
the 17,557 train landmarks cluster tightly in tourist cities where the 1,000 query
landmarks also live, so the nearest train-landmark GPS is often coincidentally <1 km
from a query (e.g. two museums in the same Manhattan block). The 100k index gallery is
designed to eliminate this by being offset >500 m from every train landmark (paper
Sec 3.2) — it measures real retrieval precision, not landmark co-location.

### Paper contrast

| Method | Dataset | Gallery | @1 km | @25 km | @200 km | @750 km | @2500 km |
|---|---|---:|---:|---:|---:|---:|---:|
| GeoClip (own paper) | Im2GPS3k (global) | 100k | 14.11 | 34.47 | 50.65 | 69.67 | 83.82 |
| Off-shelf GeoClip (MML paper) | MMlandmarks (US) | 100k index | **21.37** | **36.44** | 48.57 | 71.45 | 91.50 |
| **Ours (zero-shot, paper protocol)** | MMlandmarks (US) | 100k index | **6.67** | **28.79** | 44.48 | 69.07 | 91.07 |

We sit ~14 points below the MML paper at @1 km on the same dataset, same gallery, same
off-the-shelf `geoclip` PyPI weights, same Haversine metric. That gap is **currently
unexplained**.

What we ruled out in investigation:
- Wrong gallery (professor confirmed `mml_index_satellite.csv` is correct).
- Wrong query set (18,688 vs paper's 18,689 — off by one image, negligible).
- Bug in our retrieval path — our `predict_batch` matches GeoCLIP's native
  `model.predict(top_k=1)` 500/500 on a sanity subset.
- Wrong preprocessing — tested both `CLIPProcessor` (CLIP mean/std, current) and
  `img_val_transform` (ImageNet mean/std); CLIPProcessor is the better of the two.
- Wrong distance metric — Haversine matches paper Sec 5.2 and agrees with
  `geopy.distance.geodesic` to within 0.3 %.
- Sparse-gallery floor — theoretical @1 km ceiling on this gallery is 69.69 %, well
  above both our 6.67 % and the paper's 21.37 %.

Open question: does the MML team's GeoCLIP eval use a specific checkpoint / package
version / gallery-swap convention we haven't matched? Email sent to the first author
(Oskar Kristoffersen) asking for the exact eval script. This doc will be updated when
the gap is resolved.

## Fine-tuning — stale, needs re-run

⚠️ **The numbers below assumed the old train-landmark zero-shot baseline (19.22 % / 34.56 %).
With the current index-gallery baseline (6.67 % / 28.79 %) the Acc@25km save gate shifts,
so everything in this section needs to be redone.**

Training is wired end-to-end in [scripts/geoclip_train.py](../../scripts/geoclip_train.py):
symmetric InfoNCE with the pretrained `logit_scale`, Adam `lr=1e-4`, batch size 32,
10-epoch target. After each epoch we re-embed the gallery and evaluate on the full query
set; a checkpoint is written **only** when Acc@25km improves over the zero-shot baseline.

Historical run on V100 (stale — old baseline): epoch 1 Acc@1km 12.97 %, epoch 2 15.10 %,
then the job hit the 4-hour LSF wall-clock limit mid-epoch-3. No epoch cleared the
Acc@25km gate. Work in progress.

## How to run

```bash
# Zero-shot evaluation
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/team/03_geoclip_zeroshot.ipynb

# Fine-tune (HPC, GPU; writes models/best_geoclip_baseline.pth when a checkpoint improves)
uv run python scripts/geoclip_train.py

# Fine-tuned evaluation (requires checkpoint)
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/team/04_geoclip_finetuned.ipynb
```
