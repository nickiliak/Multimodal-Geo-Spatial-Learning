# Multimodal Geo-Spatial Learning

DTU 02501 *Advanced Deep Learning in Computer Vision* group project. Predicts the
geographic location of a US landmark from a ground-level photograph by combining
GeoCLIP (ground → GPS) with Sample4Geo-style cross-view retrieval (ground ↔ satellite)
on the **MMLandmarks** dataset (ground photos · aerial tiles · Wikipedia text · GPS).

## Goals

1. **Shrink the satellite search space.** Use a cheap GPS prior to filter the 101K-tile
   index down to ~10² candidates per query.
2. **Improve ground → GPS accuracy.** Re-rank the candidates with a high-capacity
   ground↔satellite matcher so the final prediction inherits the satellite tile's
   geolocation resolution (meters, not kilometers).

Full design rationale lives in [docs/team/pipeline_design.md](docs/team/pipeline_design.md).

## Pipeline (intended)

```
ground image
   │
   ▼
[1] GeoCLIP  ───────────►  raw_gps           (CLIP ViT-L/14 + location encoder)
   │
   ▼
[2] GPS-based narrowing ─►  C(raw_gps)       (~100 satellite tiles; hard-radius / top-K / soft)
   │
   ▼
[3] Sample4Geo  ─────────►  best tile        (Siamese ConvNeXt-B, symmetric InfoNCE)
   │
   ▼
refined GPS = GPS of top-1 satellite tile
```

Stages 1 and 3 are implemented; stages 2 and 4 (joint loss `α·L_gps + β·L_sat`) are
not yet wired. See [pipeline_design.md §7](docs/team/pipeline_design.md) for the
status table and grading-priority list.

## Current numbers

**GeoCLIP zero-shot** — 18,688 query images, off-the-shelf weights:

| Gallery | Acc@1km | Acc@25km | Acc@200km | Acc@750km | Acc@2500km |
|---|---:|---:|---:|---:|---:|
| `paper` (100,539, query GT in gallery) | **21.35 %** | 36.44 % | 48.61 % | 71.41 % | 91.52 % |
| `index` (99,539, honest in-the-wild)   |  **6.67 %** | 28.79 % | 44.48 % | 69.07 % | 91.07 % |

The 21.35 % row reproduces the MMLandmarks paper's Table 3 within rounding and is an
**upper bound** (every query's GT GPS sits in the gallery). The 6.67 % row is the
honest baseline to beat. Details in [docs/team/geoclip.md](docs/team/geoclip.md).

**Sample4Geo standalone** — ConvNeXt-Base, 35 epochs, ground → satellite:
R@1 **17.60 %**, R@5 33.00 %, R@10 41.00 %, mAP@1000 25.46 %.

## Repository layout

```
src/mmgeo/
  geolocalizations/geoclip/   # GeoCLIP baseline, dataset, Lightning module, eval
  crossview/                  # Sample4Geo-style ground↔satellite retrieval
  pipe_helpers.py             # GPS → satellite-tile helpers (stage-2 plumbing)
configs/                      # YAML configs for both tracks
scripts/                      # Training entrypoints + LSF submission scripts
notebooks/team/               # EDA + zero-shot / fine-tuned eval notebooks
docs/team/                    # Shared design docs, GeoCLIP reference, data setup
docs/personal/                # Per-author scratch notes
tests/                        # Unit tests for the helpers
```

## Setup

Requires Python ≥ 3.11 and [uv](https://github.com/astral-sh/uv). Install dependencies:

```bash
uv sync
```

The dataset lives on DTU HPC at `/dtu/blackhole/02/137570/MML`. On HPC, create the
symlink:

```bash
bash scripts/setup_data.sh        # creates data/MML_Data → /dtu/blackhole/02/137570/MML
```

Off-HPC the script aborts; data-loading code will fail unless guarded by a path check.
See [docs/team/getting-started-data.md](docs/team/getting-started-data.md).

## How to run

```bash
# GeoCLIP — zero-shot eval
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/team/03_geoclip_zeroshot.ipynb

# GeoCLIP — fine-tune (HPC, GPU; writes models/best_geoclip_baseline.pth on improvement)
uv run python scripts/geoclip_train.py

# GeoCLIP — fine-tuned eval
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/team/04_geoclip_finetuned.ipynb

# Cross-view (Sample4Geo) — submit on HPC
bsub < scripts/run_crossview_convnext_base.sh
```

## Documentation

- [docs/team/pipeline_design.md](docs/team/pipeline_design.md) — two-stage pipeline
  design, status table, open questions, grade-raising priorities.
- [docs/team/geoclip.md](docs/team/geoclip.md) — GeoCLIP track: code map, evaluation
  protocol, zero-shot benchmarks, fine-tuning notes.
- [docs/team/getting-started-data.md](docs/team/getting-started-data.md) — HPC data
  setup.
- [docs/personal/](docs/personal/) — per-author notes (not authoritative).

## References

- *Sample4Geo: Hard Negative Sampling For Cross-View Geo-Localisation* — Deuser et
  al., ICCV 2023 (`papers/sample4geo.pdf`).
- *MultiModalGeolocalization* — Kristoffersen, project brief
  (`MultiModalGeolocalization.pdf`).
- *GeoCLIP* — Vivanco Cepeda et al., NeurIPS 2023.
