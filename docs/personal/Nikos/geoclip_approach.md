# GeoClip Baseline for MMlandmarks Geolocalization

## 1. Paper Summary

**GeoClip: Clip-Inspired Alignment between Locations and Images for Effective Worldwide Geo-localization**
(Vivanco et al., NeurIPS 2023 — github.com/vicentevivan/geoclip)

### 1.1 Problem Framing

GeoClip treats worldwide geo-localization as an **image-to-GPS retrieval** problem rather
than classification. Given a query image, the model retrieves the most likely GPS coordinate
from a gallery by finding the closest match in a shared embedding space.

### 1.2 Architecture

```
Image ──► CLIP ViT/L-14 (frozen) ──► linear h0, h1 ──► v ∈ R^512
                                      (trainable, 768→512)

GPS ───► Equal Earth Projection ──► Random Fourier Features (3 hierarchies) ──► L ∈ R^512
         (lat, lon → EEP coords)     (MLP per level, concat)
```

**Image Encoder:**
- Backbone: OpenAI CLIP ViT/L-14 — frozen during training
- Trainable linear head layers: 768 → 1024 → 512

**Location Encoder (the novel contribution):**
1. **Equal Earth Projection (EEP)**: Remaps GPS (lat, lon) to minimize area distortion.
   Rescales longitude to [−1, 1], latitude proportionally.
2. **Random Fourier Features (RFF)**: Spectral positional encoding capturing spatial
   frequencies at multiple scales. σ parameter controls resolution: σ = 2^4 = 16 optimal.
3. **Hierarchical MLPs (M=3 levels)**: Three independent MLP branches with different σ
   values; their outputs are concatenated → 512-dim GPS embedding.

### 1.3 Training

| Component | Value |
|-----------|-------|
| Dataset | MP-16 (16M geotagged Flickr images) |
| Loss | Contrastive (SimCLR-style), temperature τ learnable (init 0.07) |
| Dynamic queue | 4096 GPS negatives, noise σ_q = 1000m |
| Batch GPS noise | σ_b = 150m (augments positive GPS labels) |
| Optimizer | Adam, LR = 3×10⁻⁴, weight decay = 1×10⁻⁶ |
| Scheduler | StepLR, γ = 0.7, step = 1 epoch |
| Batch size | 512 |
| Epochs | ~10 (convergence) |
| Hardware | 12× NVIDIA A100 |

### 1.4 Inference

At test time, GeoClip computes cosine similarity between the query image embedding and all
GPS embeddings in a precomputed **gallery**:

```
sim(V_query, L_i) = V_query · L_i / (||V_query|| ||L_i||)
predicted GPS = argmax_i sim(V_query, L_i)
```

Gallery sizes tested: 21K (default), 100K, 1M global coordinates. Larger galleries improve
street-level (1km) accuracy: 21K → 11.88%, 1M → 13.98% on Im2GPS3k.

### 1.5 Results on Im2GPS3k

| Method | 1 km | 25 km | 200 km | 750 km | 2500 km |
|--------|------|-------|--------|--------|---------|
| GeoDecoder (prev. SOTA) | 10.1 | 23.9 | 34.1 | 49.6 | 69.0 |
| **GeoClip** | **14.11** | **34.47** | **50.65** | **69.67** | **83.82** |

---

## 2. Environment & HPC Setup

**Package manager:** `uv` — all commands use `uv run`.

```bash
# Install dependencies
uv sync

# Install package in editable mode (required for src/ imports)
uv pip install -e .

# Execute a notebook in-place on HPC
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/team/<notebook>.ipynb
```

**Data location:** DTU HPC only — `/dtu/blackhole/02/137570/MML`. Local machines have no access.

```bash
# First-time HPC setup — creates data/MML_Data symlink
bash scripts/setup_data.sh
```

All data access goes through `data/MML_Data/`. Any code reading data must run on HPC or be
guarded with a path existence check.

---

## 3. MMlandmarks Dataset — Relevant Characteristics

| Property | Value |
|----------|-------|
| Train landmarks | 17,557 (all 4 modalities: ground, satellite, text, GPS) |
| Query landmarks | 1,000 (disjoint from train, used for evaluation) |
| Index images | 101,302 (satellite images with GPS, for gallery extension) |
| Ground images | JPEG, 800×600 px, ~11.9 per landmark (train median 9) |
| Satellite images | PNG, 800×800 px, ~12.8 per landmark |
| Text descriptions | JSON (Wikipedia/Commons), 1 per landmark |
| GPS coverage | 100% (lat/lon for every landmark) |
| Geographic scope | Primarily USA (lat ~18–50°N, lon ~−157° to −67°W) |
| Category imbalance | Gini ~0.5; top-5 categories ≈ 40% of landmarks |

**Key splits for baseline** (all under `data/MML_Data/`)**:**
- `train/mml_train.csv` — 17,557 landmarks with lat/lon + image IDs (GPS gallery + training labels)
- `query/mml_query.csv` — 1,000 query landmarks with ground-truth GPS
- `query/mml_query_ground.csv` — query image IDs (input to model)
- `index/mml_index_satellite.csv` — 101,302 index images with GPS (optional gallery extension)
- `train/ground/`, `train/satellite/` — image directories

---

## 4. Baseline Strategy

### 4.1 Why GeoClip fits here

- GeoClip's image encoder is CLIP ViT/L-14 — pretrained on internet images including
  landmark photos. Zero-shot performance on our domain is plausible.
- The retrieval framing maps naturally to our dataset: train GPS points become the gallery,
  query images are the probes.
- The US-centric geography means fine-tuning on our domain should yield measurable gains.

### 4.2 Two-Phase Plan

#### Phase 1 — Zero-Shot Baseline (pretrained GeoClip, no fine-tuning)

**Goal:** Measure out-of-the-box performance as lower bound.

1. Add the geoclip package: `uv add geoclip` (then `uv sync` to install)
2. Build GPS gallery from `data/MML_Data/train/mml_train.csv` (17,557 coordinates).
   Optionally add `data/MML_Data/index/mml_index_satellite.csv` (101,302 coords) for denser coverage.
3. For each of the 1,000 query landmarks:
   - Pick one ground image from `data/MML_Data/query/ground/` (first listed in `data/MML_Data/query/mml_query_ground.csv`)
   - Get image embedding via pretrained GeoClip image encoder
   - Retrieve top-1 GPS from gallery by cosine similarity
4. Compute accuracy @ {1, 25, 200, 750, 2500} km (Haversine distance).

**Expected behavior:** Pretrained weights were trained globally, so the model may predict
globally distributed GPS even for USA-only queries. Street-level (1km) accuracy likely < 5%.
Country-level (750km) should be reasonable if the model recognizes US landmarks.

#### Phase 2 — Fine-Tuned GeoClip (domain adaptation)

**Goal:** Improve performance by adapting Location Encoder (and linear image head) to our
landmark distribution.

1. Build a PyTorch Dataset from `data/MML_Data/train/mml_train.csv` + `data/MML_Data/train/ground/`:
   - Each sample: (image path, lat, lon)
   - One random ground image per landmark per epoch
   - CLIP preprocessing: resize to 224×224, normalize with CLIP stats
2. Train with the same contrastive loss as the paper:
   - Batch GPS noise σ_b = 150m (positive augmentation)
   - Dynamic queue of 4096 GPS negatives, σ_q = 1000m
   - Temperature τ initialized to 0.07, learnable
3. Optimizer: Adam, LR=3×10⁻⁴, StepLR γ=0.7
4. Train ~10 epochs; validate on a 10% holdout of train.
5. Re-run Phase 1 evaluation with fine-tuned checkpoint.
6. Report delta: zero-shot vs fine-tuned per threshold.

---

## 5. Proposed Code Structure

```
src/mmgeo/geolocalizations/
├── __init__.py
└── geoclip/
    ├── __init__.py
    ├── dataset.py              # MMLandmarksDataset (image + GPS, ground modality)
    ├── geoclip_baseline.py     # Zero-shot inference: build gallery, predict, batch query
    ├── evaluate.py             # haversine(), accuracy_at_thresholds()
    └── train_geoclip.py        # Fine-tuning loop with contrastive loss + dynamic queue

configs/
└── geoclip_baseline.yaml   # Paths, hyperparameters, gallery choice

notebooks/team/
└── 03_geoclip_baseline.ipynb   # End-to-end: setup → zero-shot → fine-tune → results table
```

---

## 6. Evaluation Protocol

**Metric:** Accuracy @ k km — fraction of query landmarks where the predicted GPS is within
k km of ground truth.

**Distance function:** Haversine formula (great-circle distance).

**Thresholds:** {1, 25, 200, 750, 2500} km (same as GeoClip paper).

**US context note:** The continent threshold (2500km) will be near-trivially satisfied
(the US fits within ~4000km). The most informative thresholds for our dataset are:
- **25 km (city-level)** — distinguishes between nearby cities
- **200 km (regional)** — state/region level
- **750 km (country)** — cross-continental US

**Gallery choice for evaluation:**
- Phase 1 default: 17,557 train GPS points (closed-world)
- Ablation: add 101,302 index GPS points (open-world variant)

---

## 7. Key Implementation Notes

- **CLIP frozen**: Only the linear image head and Location Encoder are trainable. This
  keeps training fast (~few hours on a single GPU for 17K samples).
- **Gallery precomputation**: Embed all gallery GPS points once before evaluation.
  At 17K points × 512 dims = 34MB — trivially fits in RAM/VRAM.
- **Batch inference**: Process all 1,000 queries in batches; compute similarity against
  precomputed gallery matrix via `torch.nn.functional.cosine_similarity` or matrix multiply.
- **Image selection**: For Phase 1 evaluation, use the first ground image per query
  landmark (deterministic). For fine-tuning, sample randomly per epoch.
- **No satellite / text yet**: These modalities are deferred to future experiments. The
  ground image modality directly matches GeoClip's training domain.

---

## 8. Future Extensions (post-baseline)

| Extension | What changes |
|-----------|-------------|
| Satellite images | Swap image encoder input; CLIP may need fine-tuning |
| Text + GPS | Add text encoder branch; contrastive loss over triplets |
| Multimodal fusion | Late fusion: average embeddings from ground + text |
| Larger gallery | Use global 1M GPS + train GPS for open-world evaluation |
| Metric: MedErr | Median localization error (km) as additional metric |

---

## 9. References

- Vivanco et al., "GeoClip: Clip-Inspired Alignment between Locations and Images for
  Effective Worldwide Geo-localization", NeurIPS 2023.
  arXiv: 2309.16020 | GitHub: github.com/vicentevivan/geoclip
- Radford et al., "Learning Transferable Visual Models From Natural Language Supervision"
  (CLIP), ICML 2021.
- MMlandmarks dataset paper: `papers/MMMLandmarks_paper.pdf`
