# Hybrid pipeline results

Output of `scripts/run_hybrid_inference.py`. Each row in `results_*.csv` is one
evaluation pass over the query set for a single
`(index_mode, query_mode, radius_km, fallback)` configuration.

## Pipeline recap

```
ground_img ──► GeoCLIP ──► rough_gps ──► gps_to_sat_pipe(radius) ──► candidates
     │                                                                     │
     └──────────► Sample4Geo embed ──► cosine vs candidates (rest = -inf) ──┘
                                                │
                                                ▼
                                        top-K satellites
```

GeoCLIP gallery: paper-protocol `index ∪ query` (~101k coords).
Sample4Geo: trained checkpoint loaded from `models/best1.pt`.

## Sweep matrix

| Field | Values |
|---|---|
| `index_mode` | `query` (1k labelled sats) · `full` (1k labelled + ~100k distractors) |
| `query_mode` | `one_per_landmark` (1k queries) · `all` (~18.7k queries, paper protocol) |
| `radius_km` | 5 · 25 · 100 · 500 · 2000 · `inf` (no narrowing — control) |
| `fallback` | `fail` (empty candidate set ⇒ query scores 0) · `fallback_full` (empty ⇒ rerank against full gallery) |

48 rows total = 2 × 2 × 6 × 2.

## Column reference

### Identifiers
- `index_mode`, `query_mode`, `radius_km`, `fallback` — see matrix above.
- `total_queries` — number of ground queries used for this row.

### Retrieval metrics (paper Table 2 style)
- `mAP@1k` — mean Average Precision at top-1000, landmark-id match.
- `R@1`, `R@5`, `R@10` — Recall@K. Fraction of queries where any top-K result
  shares the query's landmark_id.

### GPS-prediction metrics (paper Tables 3/4 style)
Computed by taking the top-1 satellite's coordinates as the predicted GPS.
- `dist@{1,25,200,750,2500}km` — fraction of queries whose predicted GPS is
  within X km of the ground-truth landmark coord.

### Narrowing diagnostics
- `empty_rate` — fraction of queries where the radius produced 0 candidates.
- `mean_candidates`, `median_candidates` — gallery indices kept by the radius
  filter, averaged over queries.

### Per-query timings (ms)
Reported as wall-clock time / query (batched amortized; CUDA-synced).
- `geoclip_ms_per_query` — GeoCLIP forward + gallery softmax.
- `s4g_embed_ms_per_query` — Sample4Geo ground-image embedding.
- `rerank_ms_per_query` — radius mask + cosine sim + top-K.
- `total_ms_per_query` — sum of the three stages above.

Gallery embedding (one-time; cached in `outputs/hybrid_cache/`) is not counted.

## How to read a row

`query / all / 25km / fallback_full`:
- 18,688 ground queries, gallery = 1,000 query-split satellites.
- Each query keeps satellites whose coords are ≤ 25 km from GeoCLIP's
  predicted GPS; if the filter empties, the full gallery is used for that query.
- Scores: `mAP@1k=0.199`, `R@1=0.142`, `R@10=0.311`, `empty_rate=22 %`,
  `mean_candidates≈21`, `total≈35.7 ms/query`.

## Headline observations

- **Narrowing helps.** For every `(index_mode, query_mode)` block the
  `radius=inf` control is the worst row — Sample4Geo over the full gallery is
  beaten by every reasonable radius.
- **Sweet spot depends on gallery size.**
  - `index_mode=query` (1k sats): best around 500 km for `one_per_landmark`,
    around 5–25 km for `all`.
  - `index_mode=full` (101k sats): best at the tightest radius (5 km) — every
    extra distractor only hurts.
- **`fallback_full` is a clean win at small radii.** When ~40 % of queries
  empty the 5 km filter, falling back to the full gallery recovers most of the
  signal you'd otherwise lose. At larger radii (≥ 100 km) the two policies
  collapse to the same number.
- **Stage cost.** GeoCLIP dominates per-query latency (~27–41 ms),
  Sample4Geo embedding adds ~9–11 ms, the rerank itself is sub-ms — so the
  narrowing is essentially free at inference time.

## Re-running

```bash
# full HPC sweep
bsub < scripts/run_hybrid_inference.sh

# local subset
uv run python scripts/run_hybrid_inference.py \
    --index-modes query --query-modes one_per_landmark \
    --radii-km 25,100,inf --fallbacks fail
```

Each run writes `results_<JOBID>.{json,csv}` here. The JSON also contains the
per-config gallery sizes and one-time embedding wall times under
`per_config[]`.
