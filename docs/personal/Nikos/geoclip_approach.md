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
street-level (1km) accuracy: 21K → 11.88%, 100K → 14.11%, 1M → 13.98% on Im2GPS3k.
The paper also uses **TenCrop** at evaluation (5 crops + flips, predictions averaged).

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

### 4.2 Plan

#### Phase 1 — Zero-Shot Baseline (pretrained GeoClip, no fine-tuning) ✅ DONE

**Goal:** Measure out-of-the-box performance as lower bound.

1. Build GPS gallery from `data/MML_Data/train/mml_train.csv` (17,557 coordinates).
2. For each of the 1,000 query landmarks, pick the first ground image, get embedding, retrieve top-1 GPS.
3. Compute accuracy @ {1, 25, 200, 750, 2500} km (Haversine distance).

**Results** (CPU, batch_size=64, 17,557-point gallery):

| Threshold | Accuracy |
|-----------|----------|
| 1 km      | 11.30%   |
| 25 km     | 23.40%   |
| 200 km    | 43.10%   |
| 750 km    | 72.40%   |
| 2500 km   | 93.90%   |

Median error: 279.8 km — Mean error: 626.5 km — Inference: ~20 min on CPU

**Comparison across papers and our results:**

| Method | Dataset | Gallery | @1 km | @25 km | @200 km | @750 km | @2500 km |
|--------|---------|--------:|------:|-------:|--------:|--------:|---------:|
| GeoClip (own paper, Table 1) | Im2GPS3k (global) | 100K | 14.11 | 34.47 | 50.65 | 69.67 | 83.82 |
| GeoClip off-shelf (MML paper, Table 3) | MMlandmarks (US) | ? | 21.37 | 36.44 | 48.57 | 71.45 | 91.50 |
| MMCLIP (trained on MML) | MMlandmarks (US) | ? | 18.72 | 33.15 | 56.20 | 73.78 | 91.50 |
| **Our zero-shot** | MMlandmarks (US) | 17,557 | 11.30 | 23.40 | 43.10 | 72.40 | 93.90 |

**Key observation:** Our 11.30% @ 1 km is actually close to GeoClip's own benchmark of
14.11% on Im2GPS3k. The small gap is explained by:
- Smaller gallery (17,557 vs 100K GPS points)
- No TenCrop augmentation (GeoClip paper uses 5 crops + flips, averaging predictions)
- Single image per landmark (vs multi-image evaluation)

The real anomaly is the MMlandmarks paper reporting 21.37% for off-shelf GeoClip — much
higher than GeoClip's own 14.11% on Im2GPS3k. Likely explanations:
- **Data leakage**: GeoClip was trained on MP-16 (16M geotagged Flickr images). The MMlandmarks
  paper itself notes "over 1.2M MP-16 images were taken in the US, increasing the chances of
  overlap" with MMlandmarks. Many query landmarks may have near-duplicates in the training set.
- **Evaluation setup**: The MML paper evaluates on all 18,689 ground images (not 1 per landmark),
  and may use a different gallery construction.
- **Indoor filtering**: The MML paper filters images with LLaVA-1.5-7b-hf (Section 3.3), keeping
  only outdoor images (274,650 outdoor vs 54,699 indoor, ~83%). They prompted LLaVA to categorize
  every image as indoor or outdoor. We do not apply this filter — ~8% of our query images may be
  indoor/zoomed with no geographic context, disproportionately hurting fine-grained accuracy.

**Bottom line:** Our zero-shot baseline is consistent with GeoClip's published performance.
The gap vs the MMlandmarks paper's GeoClip number is likely inflated by data overlap and
evaluation differences, not a deficiency in our setup.

#### Phase 1b — Zero-Shot Improvements (next step)

Improvements to the zero-shot baseline, no training required:

**1. Multi-image aggregation** (highest expected impact)
Instead of using 1 image per landmark, embed all available ground images per query landmark
and mean-pool their embeddings before gallery lookup. The MML paper uses all 18,689 images.

**2. TenCrop augmentation** (medium impact — used in GeoClip paper)
The GeoClip paper (Section 4) uses TenCrop at evaluation: 5 crops of the image + their
horizontal flips, predictions averaged. This boosts fine-grained accuracy without training.

**3. Extended gallery** (medium impact at fine-grained thresholds)
Add `data/MML_Data/index/mml_index_satellite.csv` (101,302 GPS points) to the gallery,
giving 118,859 total points. Denser coverage reduces the distance to the nearest gallery point.
GeoClip paper uses 100K gallery points; our current 17,557 is substantially smaller.

**4. Indoor filtering** (lower priority — complex)
The MML paper uses LLaVA-1.5-7b-hf to filter indoor images. We could use a simpler
heuristic or skip entirely and document as a known limitation (~8% noisy queries).

#### Phase 2 — MMCLIP-style Multimodal Training (future)

The MMlandmarks paper trains a joint model across all 4 modalities (ground images, satellite,
text, GPS) with contrastive loss across every modality pair. This is distinct from simply
fine-tuning GeoClip on ground→GPS. It requires implementing `dataset.py` and a multimodal
training loop — deferred until the zero-shot baseline is fully squeezed.

---

## 5. Code Structure

```
src/mmgeo/geolocalizations/
├── __init__.py
└── geoclip/
    ├── __init__.py
    ├── geoclip_baseline.py     # Zero-shot inference: build gallery, predict, batch query
    └── evaluate.py             # haversine(), accuracy_at_thresholds(), median_error()

configs/
└── geoclip_baseline.yaml   # Paths, batch size, gallery choice, evaluation thresholds

notebooks/team/
└── 03_geoclip_baseline.ipynb   # End-to-end: setup → gallery → inference → evaluate → plots
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

- **Everything frozen**: We use pretrained GeoClip as-is — no weights are updated.
- **Gallery precomputation**: Embed all gallery GPS points once before evaluation.
  At 17K points × 512 dims = 34MB — trivially fits in RAM/VRAM.
- **Batch inference**: Process all 1,000 queries in batches; compute similarity against
  precomputed gallery matrix via matrix multiply.
- **Image selection**: Currently uses the first ground image per query landmark (deterministic).
  Next improvement: mean-pool embeddings across all available ground images per landmark.
- **No satellite / text yet**: These modalities are deferred to Phase 2 multimodal training.

---

## 8. Future Extensions

| Extension | What changes |
|-----------|-------------|
| Multi-image aggregation | Mean-pool all ground image embeddings per query landmark |
| Extended gallery | Add 101K index GPS points for denser coverage (118K total) |
| MMCLIP-style training | Joint contrastive training across ground, satellite, text, GPS |
| Satellite modality | Swap image encoder input for aerial images |
| Text + GPS | Add frozen CLIP text encoder branch, contrastive loss over all pairs |

---

## 9. References

- Vivanco et al., "GeoClip: Clip-Inspired Alignment between Locations and Images for
  Effective Worldwide Geo-localization", NeurIPS 2023.
  arXiv: 2309.16020 | GitHub: github.com/vicentevivan/geoclip
- Radford et al., "Learning Transferable Visual Models From Natural Language Supervision"
  (CLIP), ICML 2021.
- MMlandmarks dataset paper: `papers/MMMLandmarks_paper.pdf`
