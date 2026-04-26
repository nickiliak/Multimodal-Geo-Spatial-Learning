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

- **Gallery:** configurable via `gallery.source`:
  - `"paper"` (default) — 100,539 GPS = 99,539 index-satellite + 1,000 query-landmark
    coords. **Matches the camera-ready MML paper Sec 5.2** ("combined satellite index
    and query sets, 101k"). Reproduces their 21.37 % @1 km. Because every query's GT
    GPS is *in* the gallery, this is an **upper bound**, not an in-the-wild number.
  - `"index"` — 99,539 index-satellite only, honest in-the-wild gallery (~6.67 %).
  - `"train"` — 17,557 train-landmark GPS (cluster-luck ablation).
  - `"both"` — train + index (~118k).
- **Queries:** all 18,688 query ground images (multiple images per landmark, each scored
  against the landmark's ground-truth GPS). Matches paper Sec 5.2 exactly.
- **Metric:** Haversine distance → Accuracy @ {1, 25, 200, 750, 2500} km + median / mean
  error in km.
- **Params:** 438M total, 10.4M trainable (linear image head + full location encoder);
  the CLIP ViT/L-14 vision tower is frozen.

## Zero-shot benchmark

V100, off-the-shelf `geoclip` PyPI weights, `gallery.source: "paper"` — **reproduces
the MML paper's 21.37 / 36.44 row** within rounding:

| Threshold (km) | Accuracy (%) | Paper (Table 3) |
|---:|---:|---:|
| 1    | 21.35 | 21.37 |
| 25   | 36.44 | 36.44 |
| 200  | 48.61 | 48.57 |
| 750  | 71.41 | 71.45 |
| 2500 | 91.52 | 91.50 |

### Why 21 % is an upper bound, not a realistic number

The paper's gallery is `index ∪ query_landmarks`. Since we evaluate on the exact same
query landmarks, every query's ground-truth GPS is already in the gallery — the model
just has to retrieve the right one out of 100,539. Per Oskar (first author): *"6.67 %
is a more realistic result of what could be achievable for geolocalization in the wild,
and 21 % is then a geolocalization upper limit."*

### Honest in-the-wild result — `gallery.source: "index"`

Same model, gallery stripped to the 99,539 index-satellite coords only (query GT
removed):

| Threshold (km) | Accuracy (%) |
|---:|---:|
| 1    |  6.67 |
| 25   | 28.79 |
| 200  | 44.48 |
| 750  | 69.07 |
| 2500 | 91.07 |

This is the number to reference when talking about "how well does off-the-shelf GeoCLIP
actually localize US landmarks". Anything above this is gallery leakage.

### Ablation — train-landmark gallery (17,557 points)

| Threshold (km) | Accuracy (%) |
|---:|---:|
| 1    | 19.22 |
| 25   | 34.56 |
| 200  | 46.84 |
| 750  | 71.26 |
| 2500 | 91.33 |

Inflated by cluster-luck — train landmarks cluster tightly in tourist cities where
query landmarks also live, so the nearest train-landmark GPS is often coincidentally
<1 km from a query. Keep this around as an ablation, not a headline.

## Fine-tuning — needs re-run against new baseline

Training is wired end-to-end in [scripts/geoclip_train.py](../../scripts/geoclip_train.py):
symmetric InfoNCE with the pretrained `logit_scale`, Adam `lr=1e-4`, batch size 32,
10-epoch target. After each epoch we re-embed the gallery and evaluate on the full query
set; a checkpoint is written only when Acc@25km improves over the zero-shot baseline.

With `gallery.source: "paper"` the zero-shot Acc@25km gate is **36.44 %**, which is an
upper bound and hard to beat by fine-tuning alone. For meaningful progress it makes more
sense to run fine-tuning against `gallery.source: "index"` (honest 28.79 % baseline) and
report improvement there. Work in progress.

Historical run on V100 (stale — pre-refactor baseline): epoch 1 Acc@1km 12.97 %,
epoch 2 15.10 %, then the job hit the 4-hour LSF wall-clock limit mid-epoch-3.
No epoch cleared the gate. Needs a fresh run after this refactor.

## How to run

```bash
# Zero-shot evaluation
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/team/03_geoclip_zeroshot.ipynb

# Fine-tune (HPC, GPU; writes models/best_geoclip_baseline.pth when a checkpoint improves)
uv run python scripts/geoclip_train.py

# Fine-tuned evaluation (requires checkpoint)
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/team/04_geoclip_finetuned.ipynb
```
