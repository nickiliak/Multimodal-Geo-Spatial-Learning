# Bridging Cross-View Retrieval with GPS

**A GeoCLIP + Sample4Geo Hybrid for Landmark Geolocalisation**
DTU 02501 *Advanced Deep Learning in Computer Vision* — Spring 2026, Group 4.

> *"Can we use GeoCLIP's GPS guess to narrow Sample4Geo's satellite search and
> improve cross-view retrieval?"*

Authors: Mateusz Zbyslaw · Edvin Smajlovic · Konstantinos Papadopoulos · Nikolaos Iliakis.
Supervisor: Oskar Kristoffersen. Benchmark: [MMLandmarks](MultiModalGeolocalization.pdf)
(17,557 train landmarks · 1,000 query landmarks · 100k satellite gallery · 4 modalities:
ground / satellite / GPS / Wikipedia text).

The poster summarising the project lives at [report/Poster.pdf](report/Poster.pdf).
Detailed write-ups: [report/sample4geo_results/sample4geo_explainer.md](report/sample4geo_results/sample4geo_explainer.md),
[report/sample4geo_results/crossview_results.md](report/sample4geo_results/crossview_results.md).

## Idea

1. **GeoCLIP (ground → GPS).** A single ground photo → coarse GPS estimate
   (CLIP ViT-L image encoder + location encoder, Equal-Earth → RFF → hierarchical MLP).
2. **Multi-image GeoCLIP (new).** Aggregate the K ground photos of a landmark with a
   small Transformer encoder + `[CLS]` token to produce one landmark embedding —
   reduces mean GPS error from ~650 km to ~400 km.
3. **GPS-based narrowing.** Keep only satellite tiles inside a 25 km radius around
   the predicted GPS — typical candidate set 50–500 tiles.
4. **Sample4Geo (ground ↔ satellite).** Shared ConvNeXt-Base encoder, symmetric
   InfoNCE, GPS → DSS hard-negative curriculum. Re-ranks the radius-filtered candidates;
   final prediction = GPS of the top-1 satellite tile.

```
ground image(s)
   │
   ▼
[1] (multi-image) GeoCLIP ─►  raw_gps
   │
   ▼
[2] 25 km radius filter   ─►  C(raw_gps)         (~50–500 satellite tiles)
   │
   ▼
[3] Sample4Geo            ─►  best tile          (ConvNeXt-B, InfoNCE + Sample4Geo HN)
   │
   ▼
refined GPS = GPS of top-1 tile
```

## Headline numbers

All metrics **per-landmark** (each of the 1,000 test landmarks counts once,
regardless of how many ground photos it has — fair to single- and multi-image models).

**Sample4Geo standalone** (ground → satellite, weighted-mean aggregation over photos):

| Model | R@1 | R@10 | R@25 | mAP@1k | Notes |
|---|---:|---:|---:|---:|---|
| Zero-shot ConvNeXt-B 384px | 0.16% | 1.25% | 2.36% | 0.55% | pure ImageNet — no MML training |
| **v2** (batch 64, K=1, InfoNCE)   | 6.45% | 22.29% | 30.92% | 11.50% | full HN curriculum |
| v3 (384px backbone, batch 16, AMP, K=2, label-smooth) | 6.95% | 17.45% | 22.29% | 10.57% | per-image headline best |
| v4 (batch 64, K=2, label-smooth — ablation) | 6.71% | 20.53% | 28.26% | 11.31% | algorithmic-only |
| **v2 weighted-mean (poster)** | **17.5%** | **42.1%** | **51.0%** | **25.2%** | mean-agg over photos |

**Hybrid pipeline** (GeoCLIP narrows → Sample4Geo re-ranks, 25 km radius):

| Pipeline | R@1 | R@10 | R@25 | mAP@1k |
|---|---:|---:|---:|---:|
| Finetuned Hybrid (single image) | 9.1% | 15.9% | 20.6% | 22.1% |
| **Transformer-GeoCLIP + Weighted-mean S4G** | **23.30%** | **33.30%** | **40.60%** | **23.70%** |

Multi-image hybrid (R@1 23.30%) is a ~50% relative improvement over single-image
Sample4Geo (v2 17.5%). Full breakdown — per-image vs per-landmark, max- vs mean-agg,
ablations, training progression — in
[report/sample4geo_results/crossview_results.md](report/sample4geo_results/crossview_results.md).

Pipeline radius sweeps and best/worst qualitative panels:
[report/pipeline_results/](report/pipeline_results/).

## Lessons (from poster)

- **Domain training is essential for Sample4Geo.** Zero-shot ConvNeXt → ~0%;
  fine-tuned reaches 17.5% (mean-agg). Generic ImageNet features do not transfer
  across the ground/satellite viewpoint gap.
- **CLIP backbone dominated GeoCLIP.** Fine-tuning GeoCLIP on MMLandmarks barely
  moved the needle — 17k US landmarks vs the 4M+ photos CLIP was pretrained on.
- **Fair per-landmark evaluation matters.** Per-image numbers are biased toward
  landmarks with many photos; per-image and per-landmark rankings can flip
  (v3 wins per-image at 8.58%, loses per-landmark-max to v2's 9.00%).
- **Multi-image aggregation cuts uncertainty.** Transformer-`[CLS]` over K ground
  embeds: mean GPS error 650 km → 400 km.
- **The hybrid works.** Single-image hybrid: +50% over fine-tuned Sample4Geo.
  Multi-image hybrid extends the gain further.

## Repository layout

```
src/mmgeo/
  geolocalizations/geoclip/   # GeoCLIP baseline + newGeoCLIP (transformer aggregator)
    geoclip_baseline.py       # newGeoCLIP model + data loaders
    train_new_geo.py          # transformer-aggregator training
    lit_module.py, dataset.py # baseline Lightning module + dataset
    evaluate.py               # haversine-threshold + recall metrics
  crossview/                  # Sample4Geo-style cross-view retrieval
    model.py, losses.py       # CrossViewModel, SymmetricInfoNCE, MultiPositiveInfoNCE
    sampling.py               # GPS → DSS hard-negative sampler
    train.py, eval.py         # training + per-image / per-landmark evaluation
  pipe_helpers.py             # GPS → satellite-tile narrowing (radius / top-K)
  train_pipeline.py           # joint hybrid fine-tune
  inference.py                # end-to-end pipeline inference
configs/                      # YAML configs per track
  geoclip_baseline.yaml geoclip_train.yaml geoclip_new_train.yaml
  crossview_baseline.yaml crossview_convnext_base.yaml crossview_convnext_base_384_zeroshot.yaml
  hybrid.yaml
scripts/                      # LSF submission + Python entrypoints
notebooks/team/               # EDA + zero-shot / fine-tuned / pipeline notebooks
docs/team/                    # Design docs, GeoCLIP reference, data setup
report/                       # Poster + final-report assets (results, viz)
tests/                        # Unit tests for helpers
```

## Setup

Python ≥ 3.11 and [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

Dataset on DTU HPC at `/dtu/blackhole/02/137570/MML`. Create the symlink:

```bash
bash scripts/setup_data.sh   # creates data/MML_Data → /dtu/blackhole/02/137570/MML
```

Off-HPC the script aborts. See [docs/team/getting-started-data.md](docs/team/getting-started-data.md).

## How to run

```bash
# GeoCLIP — zero-shot eval (notebook)
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/team/03_geoclip_zeroshot.ipynb

# GeoCLIP — fine-tune (HPC)
bsub < scripts/train_geoclip.sh

# newGeoCLIP — train the transformer aggregator (frozen GeoCLIP)
bsub < scripts/train_new_geo.sh

# Sample4Geo — train / fine-tune
bsub < scripts/run_crossview_convnext_base.sh
# Sample4Geo — evaluate a checkpoint
bsub < scripts/eval_crossview.sh

# Hybrid — joint fine-tune (GeoCLIP + S4G)
bsub < scripts/train_hybrid.sh

# Hybrid — pipeline inference sweep (radii, index/query modes, fallbacks)
bsub < scripts/run_hybrid_inference.sh

# Pipeline notebook (interactive)
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/team/05_pipeline.ipynb
```

## Documentation

- [docs/team/pipeline_design.md](docs/team/pipeline_design.md) — pipeline design,
  status table, open questions.
- [docs/team/geoclip.md](docs/team/geoclip.md) — GeoCLIP track: code map, evaluation
  protocol, zero-shot benchmarks.
- [docs/team/getting-started-data.md](docs/team/getting-started-data.md) — HPC data setup.
- [report/sample4geo_results/sample4geo_explainer.md](report/sample4geo_results/sample4geo_explainer.md)
  — exam-level Sample4Geo technical write-up (architecture, losses, hard-negative
  curriculum, three eval protocols, all ablations, design-decision Q&A).
- [report/sample4geo_results/crossview_results.md](report/sample4geo_results/crossview_results.md)
  — unified cross-view results across all model versions and eval protocols.
- [report/pipeline_results/](report/pipeline_results/) — hybrid pipeline sweeps
  (`results_combined.csv`) and best/worst qualitative panels per radius.

## References

- *Sample4Geo: Hard Negative Sampling For Cross-View Geo-Localisation* — Deuser et al.,
  ICCV 2023 ([papers/sample4geo.pdf](papers/sample4geo.pdf)).
- *GeoCLIP: Clip-Inspired Alignment between Locations and Images for Effective Worldwide
  Geo-localization* — Vivanco Cepeda et al., NeurIPS 2023.
- *MMLandmarks* — Kristoffersen et al., project brief
  ([MultiModalGeolocalization.pdf](MultiModalGeolocalization.pdf)).

## AI disclaimer (from poster)

Claude / ChatGPT used for: code writing & debugging, data processing, summarising
branches, paper discussion, math sanity-checks, and poster wording. All
experiments, model decisions, results, and conclusions were made and checked by
the team; any AI-generated content was reviewed and verified before use.
