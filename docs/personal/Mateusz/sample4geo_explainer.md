# Sample4Geo Cross-View Retrieval — Technical Explainer

For teammates working on GeoClip and the pipeline. Covers everything from conceptual
to exam-level detail. Read this before the exam.

---

## 1. What Is the Task?

**Cross-view geo-localisation**: given a ground-level photo (street, tourist, any
viewpoint), retrieve the matching satellite image of the same location from a gallery
of ~100k satellite images. Also evaluated in reverse: satellite → ground.

Why is it hard? The two views are radically different. A ground photo shows a building
façade, trees, sky. A satellite image of the same spot is a top-down aerial view showing
rooftops, roads, green patches. There are no shared low-level features — the model must
learn high-level, location-specific representations that are viewpoint-invariant.

This is a **retrieval problem**, not classification. The model never directly predicts
a location — it produces embeddings and ranks a gallery by similarity.

---

## 2. Dataset — MMLandmarks

| Split | Landmarks | Ground images | Satellite images | Purpose |
|-------|-----------|--------------|-----------------|---------|
| Train | 17,557 | 310,661 | 186,574 | Fine-tuning |
| Query | 1,000 | 18,688 | 1,000 | Eval queries |
| Index | — | ~714,500 | 99,539 | Eval gallery |

Each landmark has on average ~18 ground photos and 1–few satellite images. Train, query,
and index landmarks are completely disjoint — zero data leakage. During evaluation, the
query ground images are matched against the full index+query satellite gallery (100,539
items for g2s). The model has never seen any query or index landmark during training.

---

## 3. Architecture — CrossViewModel

**One shared encoder for both modalities.** The same ConvNeXt-Base weights process both
ground and satellite images. No separate towers, no modality-specific layers.

```
ground image (H, W, 3)  ──┐
                           ├──► ConvNeXt-Base (88M params) ──► L2-norm ──► embedding (1024-dim)
satellite image (H, W, 3) ─┘          (shared weights)
```

**Backbone options used:**
- `convnext_base.fb_in22k` — pretrained on ImageNet-22k (14M images, 22k classes)
- `convnext_base.fb_in22k_ft_in1k_384` — same, additionally fine-tuned at 384px on ImageNet-1k

**Output:** L2-normalized 1024-dim vector. Similarity between two embeddings =
cosine similarity = dot product (since L2-normalized). No projection head needed
(`embed_dim=0` uses native backbone output).

**Why shared weights?** Forces the model to learn features that are meaningful for
*both* viewpoints — effectively learning location fingerprints that survive viewpoint
change. Trade-off: cannot learn modality-specific features (satellite textures, ground
perspective distortion). Separate encoders (like CLIP's dual-tower) would be more
expressive but double the parameters.

**Why ConvNeXt, not ViT?** ConvNeXt is a pure CNN modernized to match ViT performance,
with better inductive biases for dense visual features and lower memory at high resolution.
The `fb_in22k` checkpoint has proven strong transfer performance on visual retrieval tasks.
CLIP ViT-L (used by GeoClip) is larger and was pretrained on more diverse data — that's
why the zero-shot CLIP gap is large (see Section 9).

---

## 4. Loss Function — InfoNCE

### SymmetricInfoNCE (used in v1, v2)

Within each batch of B (ground, satellite) pairs from B different landmarks:

```
logits[i, j] = cosine_sim(ground_i, sat_j) / τ
             = (ground_i · sat_j) / τ          # L2-normalized, so cosine = dot product

labels = [0, 1, 2, ..., B-1]                  # diagonal = positives

loss_g2s = CrossEntropy(logits,   labels)      # each ground → its satellite
loss_s2g = CrossEntropy(logits.T, labels)      # each satellite → its ground
loss = (loss_g2s + loss_s2g) / 2
```

This pushes each ground embedding close to its paired satellite and far from all B-1
other satellites in the batch. More negatives per batch = better gradient signal =
harder to cheat.

**Temperature τ:** learnable, initialized at 0.07 (stored internally as `log_τ`).
Lower τ → sharper softmax distribution → harder assignments → more discriminative learning.
Jointly optimized with the encoder via backprop.

**Label smoothing (v3, v4):** replace hard target (1.0 on positive) with soft target
(0.9 on positive, 0.1/(B-1) spread across negatives). Prevents the model from becoming
overconfident. Acts as regularisation, especially useful when negatives are hard and
similar to positives.

### MultiPositiveInfoNCE (used in v3, v4 — K=2 ground images per satellite per step)

Instead of one ground image per landmark per step, sample K=2. The batch layout becomes:
```
ground_embeds: (B*K, D)   — K sequential views per landmark
                             [lm0_view0, lm0_view1, lm1_view0, lm1_view1, ...]
sat_embeds:    (B, D)     — one satellite per landmark
```

```
sims = sat_embeds @ ground_embeds.T / τ        # (B, B*K)

# s2g: satellite i has K ground positives at columns [i*K, ..., i*K+K-1]
targets[i, i*K : i*K+K] = 1/K                 # soft uniform over K positives
targets elsewhere = 0
loss_s2g = -(softened_targets * log_softmax(sims)).sum().mean()

# g2s: ground j maps to satellite j // K (hard labels)
sat_labels = [0, 0, 1, 1, 2, 2, ...]          # K copies each
loss_g2s = CrossEntropy(sims.T, sat_labels, label_smoothing=0.1)

loss = (loss_g2s + loss_s2g) / 2
```

When K=1 this reduces exactly to SymmetricInfoNCE. The benefit: more gradient signal per
step (two views of the same landmark must both align with the satellite), and the model
learns that different photos of the same place share a common representation.

---

## 5. Hard Negative Sampling — Sample4Geo Style

Random batches give easy negatives — landmarks sampled randomly are usually very
different visually. The model quickly saturates. We use a two-phase curriculum to
continuously find harder negatives.

### Phase 1 — GPS-based (epochs 1–4)

Build a neighbor table using **Haversine distance** (great-circle distance on Earth's
surface) between all training landmark GPS coordinates:

```python
# For each landmark i, find its k nearest landmarks by GPS distance
# Haversine formula:
a = sin²(Δlat/2) + cos(lat_i) * cos(lat_j) * sin²(Δlon/2)
dist_ij = 2 * arcsin(√a) * R_earth
```

Config: `neighbor_pool=64` (store 64 GPS-nearest), `pool_size=32` (sample from top 32).

**Batch construction:** pick a seed landmark, sample (batch_size-1) neighbours from
its top-32 GPS-nearest pool. Shuffle within pool to vary difficulty.

**Why GPS?** Nearby locations share similar satellite appearance (same neighbourhood,
similar vegetation, similar street layout). These are hard negatives for a model that
hasn't learned fine-grained features yet.

### Phase 2 — DSS / Dynamic Similarity Sampling (epochs 5+)

Build neighbor table using **cosine similarity in the current model's embedding space**:

```python
# Step 1: Embed all 17,557 training landmarks with current model
mean_embed = L2_normalize(ground_embed + sat_embed)   # fused representation

# Step 2: Find k most similar landmarks in embedding space
sims = mean_embeds @ mean_embeds.T                     # (N, N) cosine similarities
# top-k per row (excluding self)
```

Same batch construction as Phase 1, just with a different neighbor table. Refreshed every
`dss_refresh_every=1` epoch (re-embed full dataset, rebuild table). Transition from GPS
to DSS is done by swapping the table in-place (`set_neighbors()`) — no DataLoader restart.

**Why DSS?** The hardest possible negatives for the *current model state*. Two landmarks
that currently look similar to the model but are actually different — these are the
examples that force the most learning. As the model improves, so do its hardest negatives.
This is a self-adversarial curriculum.

**What you observe in training:** the transition at epoch 5 (GPS→DSS) causes a temporary
loss spike and batch accuracy drop — the task suddenly becomes much harder. Then the model
adapts and improves steadily.

### Batch Guarantee

`HardNegativeBatchSampler` guarantees every landmark appears at most once per batch.
This is required for InfoNCE correctness — if two samples share a `landmark_id`, the
loss would treat a true positive as a negative, corrupting gradients.

---

## 6. Training Schedule

| Parameter | Value | Why |
|-----------|-------|-----|
| Optimizer | AdamW | Weight decay regularisation, standard for ViT/ConvNeXt fine-tuning |
| Learning rate | 1e-4 | Standard for backbone fine-tuning (not training from scratch) |
| Weight decay | 1e-4 | L2 regularisation |
| Warmup | 3 epochs, linear 1e-5 → 1e-4 | Prevents large gradient updates before embeddings stabilise |
| Main schedule | CosineAnnealingLR, 1e-4 → 1e-6 | Smooth decay, final fine-tuning at low LR |
| Temperature | 0.07 (learnable) | Starting point from CLIP literature |
| Label smoothing | 0.0 (v2), 0.1 (v3/v4) | Regularisation for hard-negative setting |

**Landmark-uniform sampling:** `MMLCrossViewDataset.__len__()` returns 17,557 (number of
landmarks, not images). Each `__getitem__(idx)` maps an index to a landmark and randomly
samples ground/satellite images for it. Result: every landmark appears exactly once per
epoch, regardless of how many photos it has. A 1-photo landmark and a 15-photo landmark
both get one training slot per epoch. Sampling is already fair — no custom weighting needed.

---

## 7. AMP — Automatic Mixed Precision (v3 only)

At 384px with batch=16, a forward pass uses ~32GB GPU memory without AMP — hitting the
V100-32GB limit exactly.

**How AMP works:**
- `torch.autocast("cuda")`: runs the forward pass and loss in **float16** — activations
  use half the memory (~2× effective batch capacity)
- Weights stay in **float32** — numerically stable for optimizer updates
- `GradScaler`: multiplies the loss by a large scale factor before backward (prevents
  fp16 gradient underflow), then divides before the optimizer step. Auto-adjusts the
  scale if inf/NaN gradients are detected.
- Scaler state is saved in checkpoints (`scaler_state_dict`) so training can resume
  with the correct scale factor.

AMP is **not needed** at 224px/batch=64 (v2, v4) — those fit ~21GB without it.

---

## 8. Evaluation Protocols

Four protocols are used. Numbers are **not** comparable across protocols.
All per-landmark protocols are computed in one pass; each of the 1,000 landmarks counts once.

---

### Protocol 1 — Per-image (`recall@k`) — paper-comparable

**Simple terms:** Each of the 18,688 ground photos is its own independent query. A landmark
with 18 photos gets 18 separate shots at finding the right satellite. Like grading 18,688
individual exam answers — a student who submits 18 answers has 18 chances to get one right.

**Math:**
```
score(sat_j) = e_query · sat_j          (dot product of L2-normalised embeddings)
R@K = (# queries where correct sat is in top-K) / 18,688
```

**Used for:** comparison with the MMLandmarks paper, MMCLIP, GeoClip.

**Limitation:** biased toward landmarks with more photos. Easy landmarks with 18 clear
photos dominate the count. Hard landmarks with 2 blurry photos barely register.

**Key numbers (g2s R@1):** Zero-shot 0.34% → v2 7.21% → v4 7.63% → v3 8.58%.

---

### Protocol 2 — Per-landmark max-agg (`lm_max_recall@k`)

**Simple terms:** For each landmark, run ALL its photos as queries. Does the single BEST
photo find the right satellite? Each landmark gets one pass/fail regardless of photo count.
Like asking: "did this student get at least one question right out of their 18 attempts?"

**Math:**
```
sims[k, j] = e_k · sat_j               (K × N_gallery similarity matrix)
agg_score[j] = max_k sims[k, j]        (best score across K photos for gallery item j)
R@K = (# landmarks where correct sat is in top-K of agg_score) / 1,000
```

**Upper bound:** a landmark succeeds if even one photo happens to match. Can be fooled if
one photo is very confidently wrong (high similarity to the wrong satellite), pushing
the wrong satellite above the correct one via max.

**Key numbers (g2s R@1):** v3 7.10% < v4 8.10% < v2 9.00% — reversed from per-image!

---

### Protocol 3 — Per-landmark mean-agg (`lm_mean_recall@k`) — **team primary metric**

**Simple terms:** For each landmark, average the similarity scores of ALL photos to each
gallery satellite. No single photo dominates — it's the class average. Each landmark gets
one vote. The most robust metric.

**Math:**
```
agg_score[j] = (1/K) Σ_k sims[k, j] = avg_embed · sat_j
R@K = (# landmarks where correct sat is in top-K of agg_score) / 1,000
```

**Equivalence proof (mean-agg = embedding-space mean pooling):**
```
score-space mean:     agg_score[j] = avg_embed · sat_j
embedding-space pool: agg_score[j] = normalize(avg_embed) · sat_j
```
These differ only by `||avg_embed||` — a positive constant, the same for every gallery
item. So the **ranking is identical** → same R@K and mAP. Our `lm_mean` numerically equals
what you get from averaging the embeddings first and then querying with the L2-normalised mean.

**Key numbers (g2s R@1):** v2 17.60% → v3 18.40% → v4 **18.50%** (best overall).

---

### Protocol 4 — Per-landmark attention-weighted mean (`lm_attn_recall@k`)

**Simple terms:** Like mean-agg, but photos that look more "typical" for the landmark
get more weight, and unusual-looking photos get less weight. Idea: downweight the outlier
photos that might be pulling the result toward the wrong satellite.

**Math:**
```
mean_embed = normalize((1/K) Σ_k e_k)          (1, D)  — L2-normalised centroid
w_k = softmax(e_k · mean_embed)                 (K,)    — weight = cosine sim to centroid
attn_embed = normalize(Σ_k w_k e_k)             (1, D)  — weighted sum, re-normalised
agg_score[j] = attn_embed · sat_j
```

**Why genuinely different from mean-agg:** non-uniform weights w_k mean
`Σ w_k e_k ≠ c · avg_embed` in general, so the scalar-equivalence argument does not apply.
Attention can produce a genuinely different ranking.

**Empirical result — attn is slightly WORSE than mean:**

| Model | mean R@1 | attn R@1 | Δ |
|-------|----------|----------|---|
| v2 | 17.60% | 17.50% | −0.10% |
| v3 | 18.40% | 18.20% | −0.20% |
| v4 | 18.50% | 18.20% | −0.30% |

With K≈18 diverse photos, the mean is already stable — attention adds no benefit.
The photos that look "atypical" (low cosine-sim to the centroid) are not noise; they
are genuine photos from unusual angles that sometimes align better with the satellite
(top-down view). Downweighting them discards useful evidence. Simple mean treats all
views equally and wins. Attention would help if the dataset contained corrupted or
off-topic images, but MMLandmarks is clean.

---

### Protocol 5 — Pooled (legacy v2 — do not use for reporting)

Mean-pool all K ground embeddings into one vector, L2-normalise, query once. Reduces
18,688 queries to 1,000. Not paper-comparable; gives inflated numbers (v2: 17.6% pooled
vs 7.21% unpooled). **Only v2 was ever evaluated this way.** The 17.6% in
`best_metrics.json` is this pooled number — identical to lm_mean (same math, see above).

---

### Quick reference

| Protocol | # queries | Each landmark counts | Key metric | Best model |
|----------|-----------|---------------------|-----------|-----------|
| Per-image | 18,688 | ×(# photos) | Paper comparison | v3 (8.58%) |
| lm_max | 1,000 | ×1 | Landmark coverage | v2 (9.00%) |
| lm_mean | 1,000 | ×1 | **Team primary** | v4 (18.50%) |
| lm_attn | 1,000 | ×1 | Experimental | v3/v4 (18.20%) |

**Note on expected numbers:** per-lm max is usually higher than per-image because each
landmark gets its best shot. However v3 is an exception (per-image 8.58% > per-lm max
7.10%) — v3's improvements are concentrated in easy, high-photo-count landmarks. When
each landmark counts once, v2 identifies more distinct locations. Per-landmark metrics
cannot be inflated this way — they are the fairer measure.

---

## 9. Experiments — What We Tried and Why

### Zero-shot
**Config:** ConvNeXt-Base (fb_in22k_ft_in1k_384), no MMLandmarks training.
**Purpose:** baseline for "how much do ImageNet features help?"

| Metric | g2s | s2g |
|--------|-----|-----|
| Per-image R@1 | 0.34% | 0.00% |
| Per-image R@5 | 1.23% | 0.30% |
| Per-image R@10 | 2.14% | 0.80% |
| Per-image mAP@1k | 1.00% | — |
| Per-lm max R@1 | 0.30% | 0.00% |
| Per-lm mean R@1 | 0.40% | 0.00% |

Essentially random. General visual features do not transfer to cross-view retrieval without
domain-specific training. Note: g2s R@1 = 0.34% is just above the random baseline for a
1,000-image candidate set (expected 0.1% if uniform). ImageNet pretraining adds a tiny
signal but not enough to be useful.

### v1 — Proof of concept
**Config:** ConvNeXt-Tiny (28M params), 224px, 20 epochs, partial hard-negative setup.
Evaluated with pooled protocol on query-only gallery (not paper-comparable).
**Result:** ~9.73% (pooled, small gallery). Purpose: validate the full pipeline runs.

### v2 — Full pipeline baseline
**Config:** ConvNeXt-Base (88M, `fb_in22k`, ImageNet-22k only), 224px, batch=64, 35
epochs, n_ground=1, label_smooth=0.0, GPS(4ep)→DSS, pool_queries=True (selected on).

**Key decisions:**
- `fb_in22k` (not fine-tuned on ImageNet-1k): aggressive pretraining on 22k classes
  gives diverse features; the fine-tuned version expects 384px inputs
- batch=64: 63 negatives per step — strong InfoNCE signal
- Single positive (K=1): simpler, no risk of OOM

**Best checkpoint:** epoch 30.

| Metric | Pooled | Unpooled |
|--------|--------|---------|
| g2s R@1 | **17.6%** | **7.21%** |
| g2s R@5 | 33.4% | 18.52% |
| g2s mAP@1k | 25.5% | 13.10% |
| s2g R@1 | 5.2% | 5.2% |

**Critical note:** `best_metrics.json` records 17.6% — this was the selection metric
and it is POOLED. When comparing to v3 (unpooled 8.58%), use 7.21%. Never cite 17.6%
without the "(pooled)" qualifier.

### v3 — Higher resolution, multi-positive, AMP
**Config:** ConvNeXt-Base (`fb_in22k_ft_in1k_384`, fine-tuned at 384px on ImageNet-1k),
384px, batch=16, 36 epochs, n_ground=2 (multi-positive InfoNCE), label_smooth=0.1,
use_amp=True.

**What changed from v2 and why:**

| Change | v2 → v3 | Reason |
|--------|---------|--------|
| Backbone | fb_in22k → fb_in22k_ft_in1k_384 | Backbone was fine-tuned at 384px → better features at 384px |
| Resolution | 224px → 384px | Match backbone pretraining resolution |
| n_ground | 1 → 2 | More gradient signal per step, multi-positive InfoNCE |
| label_smooth | 0.0 → 0.1 | Regularisation for harder negatives |
| AMP | off → on | Required: 48 images at 384px = ~32GB without AMP |
| batch_size | 64 → 16 | Forced by GPU memory at 384px (16 sat + 32 ground = 48 images) |

**Best checkpoint:** epoch 36.

| Metric | g2s | s2g |
|--------|-----|-----|
| Per-image R@1 | **8.58%** | 5.40% |
| Per-image R@5 | 18.13% | 11.10% |
| Per-image R@10 | 22.29% | 13.70% |
| Per-image mAP@1k | 13.25% | — |
| Per-lm max R@1 | 7.10% | 5.40% |
| Per-lm max R@5 | 14.90% | 11.10% |
| Per-lm max R@10 | 18.80% | 13.70% |
| Per-lm mean R@1 | 18.40% | 5.40% |
| Per-lm mean R@5 | 31.70% | 11.10% |
| Per-lm mean R@10 | 37.20% | 13.70% |

Note: s2g per-lm max = per-lm mean (one satellite per landmark, nothing to aggregate over).

**Training progression (g2s R@1):**

| Epoch | g2s R@1 | s2g R@1 | Note |
|-------|---------|---------|------|
| 9 | 6.61% | 4.30% | End of GPS phase → DSS begins |
| 18 | 8.32% | 5.00% | Most of the gain happens here |
| 27 | 8.42% | 5.50% | Plateau beginning |
| **36** | **8.58%** | **5.40%** | Best checkpoint |

Most improvement by epoch 18. Final 18 epochs added only +0.26% g2s R@1.

**Critical finding — per-lm max 7.10% < per-image 8.58%:** This is the opposite of what
we expect (and the opposite of v2 and v4 where per-lm max > per-image). It means v3's
correct retrievals are concentrated in easy landmarks — those with many diverse photos,
at least one of which is highly distinctive. When each of the 1,000 landmarks gets exactly
one vote (per-lm max), v3 identifies fewer distinct locations than v2 (9.00%). The
per-image headline improvement (+1.37%) is partially a measurement artefact: v3 does
better on the same few easy landmarks, not across the board. **By the fairest metric,
v2 is actually the best model for geographic coverage.**

**Confirmed limitation:** batch=16 gives only 15 negatives per InfoNCE step vs 63 in v2.
The v4 ablation confirmed this is the dominant factor: despite a stronger backbone and
higher resolution, v3 covers fewer distinct landmarks than v2. More negatives per step
forces the model to learn location features broadly; fewer negatives lets it specialise
on the easy/frequent cases.

### v4 — Ablation (complete)
**Config:** same backbone and resolution as v2 (fb_in22k, 224px, batch=64) + v3
algorithmic improvements (n_ground=2, label_smooth=0.1). No AMP needed.
**Best checkpoint:** epoch 36.

**Purpose:** disentangle what caused the v2→v3 improvement.

| Change | v2 → v4 | v4 → v3 |
|--------|---------|---------|
| Backbone | same | fb_in22k → fb_in22k_ft_in1k_384 |
| Resolution | same (224px) | 224px → 384px |
| n_ground | 1 → 2 | same (2) |
| label_smooth | 0.0 → 0.1 | same (0.1) |
| batch_size | same (64) | 64 → 16 |

| Metric | g2s | s2g |
|--------|-----|-----|
| Per-image R@1 | 7.63% | 4.00% |
| Per-image R@5 | 19.02% | 11.30% |
| Per-image R@10 | 24.57% | 16.80% |
| Per-image mAP@1k | 13.34% | — |
| Per-lm max R@1 | 8.10% | 4.00% |
| Per-lm mean R@1 | 18.50% | 4.00% |

**Ablation findings:**

| Metric | v2 | v4 | v3 | v4−v2 | v3−v4 |
|--------|----|----|----|----|-----|
| g2s per-image R@1 | 7.21% | 7.63% | 8.58% | +0.42% | +0.95% |
| g2s per-lm max R@1 | **9.00%** | 8.10% | 7.10% | −0.90% | −1.00% |
| s2g per-image R@1 | 5.20% | 4.00% | 5.40% | −1.20% | +1.40% |

**Interpretations:**
- **v4 vs v2 (+0.42% per-image):** multi-positive training (n_ground=2) and label
  smoothing both help modestly when the batch size and backbone are unchanged.
- **v3 vs v4 (+0.95% per-image):** upgrading the backbone (fb_in22k→ft_in1k_384) and
  resolution (224→384px) adds further improvement. Both algorithmic and architectural
  changes contribute.
- **Per-lm max tells the opposite story:** v2 (9.00%) > v4 (8.10%) > v3 (7.10%). Every
  single change made per-landmark coverage worse. The cause is the batch size reduction:
  v4 has the same batch=64 as v2 but still shows slightly worse per-lm max, meaning
  multi-positive actually hurts landmark coverage (the batch sees only 32 unique landmarks
  instead of 64). v3's further drop to batch=16 (15 negatives/step) amplifies this.
- **Conclusion:** the dominant factor for per-landmark coverage is the number of unique
  landmarks in each InfoNCE batch (= negative count). More negatives → more diverse
  gradient signal → broader landmark coverage. The backbone upgrade does not compensate.
- **s2g anomaly:** v4 s2g R@1 (4.00%) is notably weaker than both v2 (5.20%) and v3
  (5.40%). Multi-positive training (which adds more ground queries per satellite) did not
  help the satellite-as-query direction — the satellite embeddings may be less well-trained
  when the gradient from s2g is diluted by multiple ground positives.

---

## 10. Results Summary

### g2s — Master Table (all protocols side-by-side)

Columns: per-image (img), per-landmark mean-agg (mean, team primary), per-landmark max-agg (max), attention-weighted mean (attn).

| Model | R@1 img | R@1 mean | R@1 max | R@1 attn | R@5 img | R@5 mean | R@5 max | R@10 img | R@10 mean | R@10 max | mAP img | mAP mean | mAP max |
|-------|---------|----------|---------|----------|---------|----------|---------|----------|-----------|----------|---------|----------|---------|
| Zero-shot | 0.34% | 0.40% | 0.30% | 0.30% | 1.23% | 1.20% | 0.90% | 2.14% | 1.80% | 2.10% | 1.00% | 0.89% | 0.89% |
| v2 (ep30) | 7.21% | 17.60% | **9.00%** | 17.50% | 18.52% | 33.40% | 20.30% | 25.31% | **42.20%** | 27.20% | 13.10% | 25.50% | 15.21% |
| v3 (ep36) | **8.58%** | 18.40% | 7.10% | 18.20% | 18.13% | 31.70% | 14.90% | 22.29% | 37.20% | 18.80% | 13.25% | 25.12% | 11.10% |
| v4 (ep36) | 7.63% | **18.50%** | 8.10% | 18.20% | **19.02%** | **32.40%** | 18.50% | 24.57% | 39.60% | 25.30% | **13.34%** | **25.33%** | 13.72% |

*attn = attention-weighted mean (photos weighted by softmax cosine-sim to centroid). Finding: attn < mean for all trained models (−0.1% to −0.3%).*

### Per-image (unpooled, paper-comparable) — g2s

| Model | Backbone | img_size | Epochs | R@1 | R@5 | mAP@1k |
|-------|----------|----------|--------|-----|-----|--------|
| Zero-shot | fb_in22k_ft_1k_384 | 384 | 0 | 0.34% | 1.23% | 1.00% |
| v2 (ep30) | fb_in22k | 224 | 35 | 7.21% | 18.52% | 13.10% |
| v3 (ep36) | fb_in22k_ft_1k_384 | 384 | 36 | **8.58%** | 18.13% | 13.25% |
| v4 (ep36) | fb_in22k | 224 | 36 | 7.63% | 19.02% | 13.34% |
| MMCLIP† | CLIP ViT-L | — | 0* | 20.5% | — | — |
| GeoClip† | CLIP ViT-L+geo | — | 0* | 21.1% | — | — |

† zero-shot on MMLandmarks — never trained on this dataset. Not a direct comparison.

### Per-landmark max-agg — g2s ("any photo wins", upper bound on landmark coverage)

| Model | R@1 | R@5 | R@10 | mAP@1k |
|-------|-----|-----|------|--------|
| Zero-shot | 0.30% | 0.90% | 2.10% | 0.89% |
| v2 (ep30) | **9.00%** | 20.30% | 27.20% | 15.21% |
| v3 (ep36) | 7.10% | 14.90% | 18.80% | 11.10% |
| v4 (ep36) | 8.10% | 18.50% | 25.30% | 13.72% |

**Max-agg ranking: v2 > v4 > v3 — opposite to per-image.**
v3 has the best per-image headline but identifies the fewest distinct locations.

### Per-landmark mean-agg — g2s ("average score across all photos" — team primary metric)

Mathematically equivalent to embedding-space mean-pooling for L2-normalized vectors.

| Model | R@1 | R@5 | R@10 | mAP@1k |
|-------|-----|-----|------|--------|
| Zero-shot | 0.40% | 1.20% | 1.80% | 0.89% |
| v2 (ep30) | 17.60% | 33.40% | 42.20% | 25.50% |
| v3 (ep36) | 18.40% | 31.70% | 37.20% | 25.12% |
| v4 (ep36) | **18.50%** | **32.40%** | **39.60%** | **25.33%** |

**Mean-agg ranking: v4 ≈ v3 > v2.** v4 is marginally the best on mean-agg (18.50% vs 18.40%).
This is the metric to report to the team and for the poster.

### s2g — Per-image (= per-landmark; one satellite per landmark)

| Model | R@1 | R@5 | R@10 | mAP@1k |
|-------|-----|-----|------|--------|
| Zero-shot | 0.00% | 0.30% | 0.80% | 0.05% |
| v2 (ep30) | 5.20% | 14.40% | 18.70% | 2.66% |
| v3 (ep36) | **5.40%** | 11.10% | 13.70% | 1.85% |
| v4 (ep36) | 4.00% | 11.30% | 16.80% | 2.12% |

### Mean-agg equivalence to pooled

`lm_mean_recall@k` (score-space aggregation) is mathematically equivalent to embedding-
space mean-pooling followed by L2-normalization, when all vectors are L2-normalized. This
is why v2 `lm_mean_recall@1` = 17.60% = the old "pooled v2 R@1" = 17.60%. The pooled
number was not inflated or wrong — it was measuring mean-agg all along. The distinction
is only in protocol (pooled = single query vector; mean-agg = K vectors, averaged scores).

---

## 11. Critical Design Decisions — Expect Exam Questions

**Q: Why a single shared encoder instead of separate ground and satellite encoders?**
Shared weights force the model to learn modality-agnostic location features — the
representation must capture what makes a place unique regardless of viewpoint. It also
halves the parameter count. The risk is that modality-specific nuances are ignored.
In practice, with enough hard negatives, the model learns what it needs. Separate
encoders (CLIP-style) would be more expressive but require twice as many parameters
and more training data.

**Q: Why InfoNCE, not triplet loss?**
InfoNCE uses all B-1 negatives in a batch simultaneously (contrastive over a full
softmax). Triplet loss uses only one negative per anchor. With hard-negative sampling,
InfoNCE gets much stronger gradient signal. Also easier to scale — add more samples to
the batch, get more negatives for free.

**Q: Why two-phase hard-negative sampling (GPS then DSS)?**
GPS phase acts as curriculum — geographic neighbours look visually similar in satellite
view, so they're meaningful hard negatives even before the model has learned anything.
Jumping straight to DSS would give random hard negatives (the model doesn't know what's
similar yet). DSS then takes over once the model has learned basic features and can
meaningfully identify its own failure cases.

**Q: Why does the GPS→DSS transition cause a performance dip?**
The model was trained on GPS-hard negatives and suddenly faces similarity-hard negatives
— a completely different difficulty distribution. This is like changing the exam while
the student is mid-study. The model temporarily regresses, then adapts.

**Q: Why AMP in v3 but not v2?**
Resolution is the bottleneck. At 384px, one image takes ~4× the activation memory vs
224px (resolution squared). With batch=16 and n_ground=2, forward pass = 48 images at
384px. Without AMP: ~32GB → OOM. With AMP (fp16 activations): ~16GB → fits. v2/v4 at
224px use ~21GB without AMP.

**Q: Why is per-landmark eval fairer than per-image?**
Per-image: a landmark with 18 photos contributes 18 queries. If 15 of them are easy
views, the landmark drags up the average unfairly. Per-landmark: each landmark counts
once — success requires at least one photo to match (max-agg) or consistent evidence
across all photos (mean-agg). Exams are graded per student, not per answer.

**Q: Why is v2's 17.6% not comparable to v3's 8.58%?**
17.6% is pooled (mean-average all 18 ground embeddings → one query vector). Pooling
reduces 18 noisy views into one clean consensus representation, making retrieval much
easier. 8.58% is unpooled (each of 18 photos is an independent query). These measure
different things. Unpooled is paper-comparable; pooled is not.

**Q: Why is there a big gap between our trained model (8.58%) and zero-shot CLIP (20.5%)?**
Almost entirely backbone quality. CLIP ViT-L was pretrained on 400M+ diverse image-text
pairs, including geo-tagged content. ConvNeXt-Base was pretrained on ImageNet (14M
images). More data + more diverse pretraining = stronger features. The solution is to
fine-tune from CLIP weights — exactly what GeoClip does, and what the extended pipeline
explores. Our contribution is demonstrating what ConvNeXt can achieve with careful
training, and providing the pipeline integration.

**Q: v3 has a better per-image R@1 than v2 (8.58% vs 7.21%) but a worse per-landmark
max R@1 (7.10% vs 9.00%). How is that possible?**
v3's correct retrievals are concentrated in easy landmarks — those with many diverse
ground photos including at least one clear, distinctive shot. When you count all 18,688
individual photos (per-image), these easy landmarks contribute many correct answers and
raise the overall rate. When you give each of the 1,000 test landmarks exactly one vote
(per-lm max), v3 identifies fewer distinct locations than v2. This means v3's headline
improvement is partially a measurement artefact, not a genuine improvement in geographic
coverage. Think of it as: v3 gets more questions right, but the questions it gets right
are mostly the same easy ones it was already getting right — it does not learn to handle
new, harder landmarks.

The root cause is the batch size. v3 uses batch=16 (15 negatives per InfoNCE step) vs v2's
batch=64 (63 negatives). With fewer negatives per step, training focuses on the same easy,
similar-looking landmarks over and over. With 63 negatives, the model is forced to
distinguish a much broader diversity of landmark pairs per step, which builds more uniform
coverage across all 1,000 test landmarks.

**Q: What did the v4 ablation reveal about the v2→v3 improvement?**
v4 uses the same backbone and resolution as v2 (fb_in22k, 224px, batch=64) but adds the
v3 algorithmic changes (n_ground=2, label_smooth=0.1). Result: v4 per-image R@1 = 7.63%
— between v2 (7.21%) and v3 (8.58%). So both algorithmic changes and backbone/resolution
changes contribute to the per-image headline improvement.

However per-lm max tells a different story: v2 (9.00%) > v4 (8.10%) > v3 (7.10%). Every
change made landmark coverage worse. Even v4's algorithmic changes (same batch=64 as v2)
slightly reduced per-lm max because n_ground=2 means the batch only sees 32 unique
landmarks per step instead of 64. The dominant factor for landmark coverage is the number
of unique landmarks (= InfoNCE negatives) per training step — not the backbone quality,
not the resolution, not multi-positive training.

**Q: Is averaging scores (score-space mean) the same as averaging embeddings (embedding-space mean pool)?**
For ranking metrics (R@K, mAP) — yes, identical. Score-space mean gives
`score(s_j) = avg_embed · s_j`; embedding-space mean pool gives
`score(s_j) = normalize(avg_embed) · s_j`. They differ by `||avg_embed||`, which is
a positive constant the same for every gallery item, so the ranking and all top-K metrics
are identical. Our `lm_mean` result IS what you would get from embedding-space mean pooling.

**Q: Why does attention-weighted mean differ from simple mean, if both produce a single query embedding?**
Simple mean weights all K photos equally: `embed = normalize(Σ e_k / K)`. Because the
weight is uniform (1/K for every photo), the weighted sum is just a scaled version of
the mean embedding, and the ranking is identical to score-space mean.
Attention breaks this symmetry: `w_k = softmax(e_k · mean_embed)`, so representative
photos get higher weight and outlier photos get lower weight. Now `Σ w_k e_k` is NOT
proportional to the mean embedding in general, and the ranking can genuinely change.

**Q: Why does attention-weighted mean perform WORSE than simple mean in our experiments?**
Empirically: attn R@1 = 17.50% (v2), 18.20% (v3), 18.20% (v4) vs mean R@1 = 17.60%, 18.40%,
18.50%. Attention is consistently −0.1% to −0.3% below mean.

The cause: with K≈18 diverse photos, the mean is already stable (averaging 18 independent
noisy views converges well). The photos that look "atypical" — low cosine-similarity to the
landmark centroid — are not noise. They are genuine photos of the same place from unusual
angles or lighting. Some of those angles actually align better with the satellite top-down view
(which is itself an "unusual" viewpoint). By down-weighting them, attention discards useful
evidence. Simple mean treats all ground viewpoints equally and wins.

Attention would likely help only if a landmark had genuinely bad/unrelated photos mixed in
(true noise, not just diverse viewpoints). MMLandmarks images are generally clean, so attention
finds no noise to suppress — it only suppresses diversity.

---

## 12. How Our Model Integrates Into the Pipeline

**What we hand off:** `.pt` checkpoint files containing `model_state_dict`.

**How to load:**
```python
from mmgeo.crossview.model import CrossViewModel
import torch, timm.data

model = CrossViewModel(backbone="convnext_base.fb_in22k_ft_in1k_384", pretrained=False)
ckpt = torch.load("best.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Get correct normalization — do NOT use standard ImageNet mean/std
data_cfg = timm.data.resolve_data_config({}, model=model.backbone)
norm_mean = data_cfg["mean"]   # e.g. [0.485, 0.456, 0.406]
norm_std  = data_cfg["std"]
```

**What the model outputs:** L2-normalized 1024-dim embedding. Cosine similarity =
`query_embed @ gallery_embeds.T`. No further normalization needed.

**Single-image query:** embed the photo → compute similarity against all candidate
satellite embeddings → argmax.

**Multi-image query (multiple photos of same landmark):**
Embed each photo independently → get N embeddings. Compute similarity for each →
N×M_gallery similarity matrix. Aggregate row-wise (max or mean) → one score vector →
argmax. **Do not average embeddings before computing similarity** — average scores instead.

**Pipeline context:** GeoClip first estimates GPS (lat, lon) from a single photo with
±20km accuracy. We take all landmarks within 20km → typically 50–500 candidates.
Our model re-ranks within this candidate set. If GeoClip's GPS estimate is off by more
than 20km, the correct landmark is excluded and we cannot recover. This error mode is
not captured in our standalone eval numbers (which test against all 100k satellite images).
