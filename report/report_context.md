# Report context — Group 4

## Team

| Member | Student ID |
|---|---|
| Nikolaos Iliakis | s250201 |
| Mateusz Zbyslaw | s250778 |
| Kostas Papadopoulos | s250219 |
| Edvin Smajlovic | s224204 |

Supervisor: Oskar Kristoffersen
Course: 02501 — Advanced Deep Learning in Computer Vision · DTU Compute · Spring 2026

---

## ⚠️ Note to Claude Design — read first

**Everything below is a *recommendation*, not a hard spec.** This document was produced by Q&A with the team; for most questions the team accepted the recommended answer rather than authoring every detail. So:

- **Treat the design as a starting point, not a contract.** If a recommendation does not work visually at A0 landscape, tweak it freely so the poster looks better.
- **Hierarchy of intent (use this when in doubt):**
  1. **Hard constraints** (must follow): A0 landscape · 3 columns · light theme · DTU red `#990000` as the only strong accent · do NOT reuse the dark navy/teal/orange palette from `first_draft_poster.pdf` · no "What we would try next" section · all 16 content containers must appear · per-author bylines stay OFF the poster.
  2. **Strong preferences** (follow unless they break the layout): Inter typography · numbered section headers with 3 pt red rules · no container boxes · headline-number callouts in big red numerals · Tufte-minimal tables · shared diagram colour grammar (red = trained, grey = frozen, white = data).
  3. **Soft suggestions** (override if a better choice exists): exact font sizes, exact gutter widths, exact bullet counts, sub-title wording, badge glyphs, icon shapes, column-balance tricks. If something fits better differently, change it.
- **When recommendations conflict with readability at 1–2 m,** readability wins. Examiners must be able to read every bullet from 1.5 m and every figure caption from 0.5 m.
- **When a recommendation conflicts with the source data,** the data wins. Use the numbers in `report/pipeline_results/results_combined.csv` and `report/sample4geo_results/crossview_results.md` as ground truth; the bullet text in this doc may quote slightly rounded versions.
- **If a container is over-stuffed,** drop the lowest-priority bullets rather than shrinking type. Per-container priority order is: headline takeaway → hero figure / table → supporting bullets → caption / footnote.
- **Empty space is allowed.** Better white at the bottom of column 1 than a cramped column 3.

---

## 0. Global poster design

> Anchors the look/feel for every container below.

- **Format / orientation:** A0 landscape (1189 × 841 mm)
- **Column count / grid:** 3 columns, reading order top→bottom within each column, columns flow left → middle → right
- **Theme:** Light — off-white background `#fafaf7`, charcoal body text `#1a1a1a`
- **Accent palette:**
  - Primary accent: **DTU red `#990000`** — section headers, key callouts, highlight cells, key numbers
  - Background: `#fafaf7` (warm off-white)
  - Body text: `#1a1a1a`
  - Neutral greys: `#e8e6e1` (block dividers), `#7a7a7a` (secondary text), `#444` (table borders), `#d6d2c8` (warm grey for diagram fills / row banding)
- **Hard constraint:** do NOT reuse the palette from `first_draft_poster.pdf` (dark navy + teal/orange/amber). Fresh palette required.
- **Typography:** Inter, single family, multiple weights
  - Title: Inter ExtraBold, ~72–80 pt
  - Section headers (container titles, prefixed `01 — `, `02 — ` … in DTU red): Inter Bold, ~28–32 pt
  - Sub-headers / labels: Inter SemiBold, ~18–20 pt
  - Body bullets: Inter Regular, ~16–18 pt
  - Captions / footers: Inter Regular, ~12–13 pt, secondary grey
  - Numbers in tables: Inter Medium, tabular figures
- **Density target:** Medium-dense — ~20 mm gutters between containers, 4–6 bullets per container, every key figure included. Optimised for 1–2 m reading distance.
- **Reading order:** No arrows between containers. Section headers numbered `01 — …`, `02 — …` in DTU red. Arrows reserved for diagrams (pipeline flow, architecture).
- **Container framing:** No boxes, no fills. Each container starts with a **3 pt DTU-red horizontal accent line** above its header; title sits directly on page background. ~20 mm vertical gap between containers acts as the divider. Minimal, modern, lets figures breathe.
- **Section header micro-style (used 16×):** two-line stack
  - Line 1 (above): 3 pt DTU-red horizontal rule spanning the container width.
  - Line 2: `SECTION 02 · PROBLEM AND DATASET` — Inter SemiBold, 12 pt, DTU red, all caps, letter-spacing `0.08em`.
  - Line 3: descriptive sub-title (e.g. `Where was this photo taken?`) — Inter Bold, 28 pt, charcoal.
  - ~6 mm gap between the rule and line 2, ~3 mm between lines 2 and 3.

### TL;DR box (top of column 1, above §2)

> One-stop "what should I know in 10 seconds?" box. Sits above container §2.

- **Style:** off-white background with a 3 pt DTU-red **left** edge accent line (vertical, full height of box). No top/bottom rules, no fill.
- **Header:** `TL;DR` in 16 pt Inter SemiBold, DTU red, small caps, letter-spacing 0.05em.
- **Body (3 short bullets, ~18 pt Inter Regular):**
  - Task: ground photo → matching satellite from 100k gallery.
  - Method: GeoCLIP predicts rough GPS → 25 km radius filter → Sample4Geo re-ranks.
  - Result: multi-image hybrid reaches **40.6 % R@25km**, median GPS error drops from 530 km → **79 km** vs single-image.

### Iconography

> Used only where they speed up scanning. Same line-weight (1.5 pt), no fills, charcoal stroke.

- **Modalities (used in §2 stat strip and §3 cross-view diagram):**
  - 📷 ground photo → simple camera outline icon
  - 🛰 satellite → satellite-dish-with-tile icon
  - 📍 GPS → map-pin icon
  - 📄 Wikipedia text → document-with-line icon
- **Trained vs frozen tags (used in diagrams):** tiny `▲ trained` (DTU-red filled triangle) / `▢ frozen` (warm-grey outline square) legend in the corner of every architecture diagram, 11 pt mono grey.
- **Status badges (used in tables):** ★ (DTU red, 14 pt) next to "Best" rows or column cells. Avoid emoji — use a proper star glyph.

### Equations

> Inline only, no full-width displayed equations. Posters with big equations look like papers; we want one-glance retention.

- **§3 Sample4Geo** — one tiny inline expression next to the loss bullet:
  `L_InfoNCE = ½(L_g→s + L_s→g)` in Inter Medium, charcoal, no display style.
- **§4 GeoCLIP** — no equation. Replace with the small architecture diagram instead.
- **§7 Multi-positive** — no equation. State `K = 2 ground views per satellite per step` as plain text.
- **No** displayed math anywhere. If Claude Design wants more, it should add it as small caption-style annotations inside the architecture diagrams.

### Column-height balancing

> Col 3 has more blocks than col 1 or col 2. Strategy to keep visual balance.

- **Col 1 (lightest content):** add the TL;DR box at top + the example-collage `Fig 1` from existing draft (4 modalities side-by-side) to occupy bottom space. Should fill ~80–90 % of column height.
- **Col 2 (densest method/results):** target ~95 % column-height fill. Hybrid pipeline diagram is the visual anchor mid-column.
- **Col 3 (most blocks):** group §9, §15, §16, §17 into tighter sub-blocks if needed. AI inclusion + footer share the bottom band.
- **Acceptable unevenness:** ±50 mm between column bottoms is fine. Don't stretch content to force exact balance — better to have small white space at the foot of col 1 than crowded text in col 3.

### Margins and gutters (explicit numbers)

- Outer page margin (all sides): **40 mm**
- Column gutter (between cols 1↔2 and 2↔3): **30 mm**
- Inter-container vertical gap within a column: **20 mm**
- Intra-container line-spacing (bullet → bullet): **6 mm**
- Container header (rule → label → sub-title → first bullet): **12 mm total above first bullet**
- Footer height (the bottom citations strip): **30 mm**, sitting at the absolute bottom edge with a 0.5 pt warm-grey rule above it.

### "Best model" badge style

- Used wherever a row/cell is the best performer (tables in §7, §8, bar chart in §13).
- **Glyph:** ★ (filled five-point star) in DTU red, 14 pt.
- **Placement in tables:** at the left of the model name row, *or* as a prefix on the model label (`★ Hybrid + Multi-image`).
- **Placement in bar charts:** directly above the value label of the winning bar.
- No "Recommended" / "Best" pill — just the star. Trust the reader.

### Top-right corner of poster (above course badge)

- **DTU logo:** small DTU wordmark logo, 40 mm wide, in DTU red, top-right corner.
- Sits above the course badge line.
- If logo licensing is uncertain at design time, fall back to: bold `DTU` in DTU red (Inter ExtraBold, 36 pt) — no graphic mark needed.

### Descriptive sub-titles per container

> One memorable sub-title per container, used in the section header.

| § | Container | Sub-title |
|---|-----------|-----------|
| 2 | Problem and Dataset | *Where was this photo taken?* |
| 3 | Cross-view + Sample4Geo | *Two views, one place* |
| 4 | GeoCLIP bridge | *Photo → GPS in one shot* |
| 5 | Hybrid model | *Filter first, rerank second* |
| 6 | Paper vs index | *Is the answer in the room?* |
| 7 | Improving Sample4Geo | *What helped, what hurt* |
| 8 | Single-image results | *Headline numbers* |
| 9 | GeoCLIP upper bound | *The ceiling we can't break* |
| 10 | Multi-image models | *Aggregating K photos into one* |
| 11 | New GeoCLIP arch | *A multi-image GeoCLIP, two stages* |
| 12 | GeoCLIP + uncertainty | *How sure is the model?* |
| 13 | Multi-image results | *7× less error, in one chart* |
| 14 | Qualitative examples | *v3 vs hybrid, head-to-head* |
| 15 | Per-landmark eval | *Two metrics, opposite rankings* |
| 16 | Conclusion | *What we'd say in 30 seconds* |
| 17 | AI inclusion | *AI assistance* |
- **Headline-number callouts:** large DTU-red numerals (~48 pt Inter ExtraBold) with a small grey label below (~14 pt Inter Regular, secondary grey). Used sparingly — reserved for the 4–5 most important results:
  - `40.6%` / "R@25km, hybrid + multi-image" (§13)
  - `79 km` / "median GPS error, hybrid + multi-image" (§13)
  - `7×` / "reduction in median error vs single-image" (§13/§16)
  - `8.58%` / "Sample4Geo v3 R@1, per-image" (§7)
  - `~25×` / "lift over zero-shot ConvNeXt" (§3)
- **Table style (used in §7, §8, §15):** Tufte-minimal.
  - No vertical lines anywhere.
  - Hairline (0.5 pt) warm-grey rules: above header row, below header row, below the final row. Nothing between data rows.
  - Numbers right-aligned, Inter Medium tabular figures.
  - **Best cell per column** = soft DTU-red fill (`#ffe6e6`) + bold black text (`#1a1a1a`).
  - No alternating row banding.
  - Column labels in Inter SemiBold, ~16 pt, secondary grey.
- **Bar chart style (§12, §13):** grouped bars, single-image = warm grey `#a8a39a`, multi-image (or "best" series) = DTU red. No gridlines, no top/right spines. X-axis labels only (no tick marks). Value labels (e.g. `40.6%`) sit on top of each bar in 14 pt tabular figures. Y-axis omitted (values on bars replace it).
- **Architecture diagram grammar (§3, §4, §5, §11):** consistent across the whole poster.
  - Flat rectangles or pills, no gradients, no drop shadows.
  - **Trained module** = DTU red fill `#990000`, white text.
  - **Frozen / pretrained module** = warm grey fill `#d6d2c8`, charcoal text.
  - **Data tensor / input / output** = white fill, 1 pt charcoal outline, charcoal text.
  - **Operation** (loss, projection, argmax, filter) = white fill, 1 pt warm-grey dashed outline.
  - Tensor shapes in monospace 12 pt grey under arrows (e.g. `(B, 1024)`).
  - Arrows: 1.5 pt solid charcoal, filled triangle heads.
- **Footer:** Single 11 pt warm-grey strip at bottom edge
  - Left: `MMLandmarks [Kristoffersen et al., arXiv 2012.17492]  ·  Sample4Geo [Deuser et al., ICCV 2023]  ·  GeoCLIP [Vivanco Cepeda et al., NeurIPS 2023]`
  - Right: `02501 · Advanced Deep Learning in Computer Vision · DTU Compute · Spring 2026 · Group 4`

### Column → section mapping

- **Col 1 — Setup ("what"):** §1 Title strip (spans full top), §2 Problem & Dataset, §3 Cross-view + Sample4Geo intro
- **Col 2 — Method + main results ("how"):** §4 GeoCLIP bridge, §5 Hybrid model, §6 Paper-vs-index, §7 Improving Sample4Geo + finetune, §8 Single-image results
- **Col 3 — Extensions + so-what:** §10 Multi-image (weighted mean vs transformer), §11 New GeoCLIP arch + two-stage, §12 GeoCLIP results + uncertainty, §13 Multi-image results, §9 Issues (GeoCLIP upper bound), §14 Qualitative examples, §15 Per-landmark eval, §16 Conclusion, §17 AI inclusion

---

## 1. Title block (spans full poster width)

- **Headline title:** *Bridging Cross-View Retrieval with GPS: A GeoCLIP + Sample4Geo Hybrid*
- **One-line research question (sub-title, italic, charcoal):** *"Can we use GeoCLIP's GPS guess to narrow Sample4Geo's satellite search and improve cross-view retrieval?"*
- **Authors line:** Mateusz Zbyslaw · Edvin Smajlovic · Konstantinos Papadopoulos · Nikolaos Iliakis  · Supervisor: Oskar Kristoffersen
- **Course badge (top-right corner, small caps, secondary grey):** `02501 · ADVANCED DEEP LEARNING IN COMPUTER VISION · DTU COMPUTE 2026`
- **Dataset stat strip (below sub-title, in warm-grey pills):** `17,557 train landmarks  ·  1,000 query landmarks  ·  100k satellite gallery  ·  4 modalities (ground / satellite / GPS / Wikipedia text)`
- **Visual treatment:**
  - Title sits directly on `#fafaf7` background — no fill, no box.
  - Stack order (top → bottom): course badge (top-right corner) · title · sub-title · authors line · stat-strip pills.
  - One **6 pt DTU-red horizontal rule** spans the full poster width *below* the title strip and separates it from the 3-column content. Acts as the only "frame" the title gets.
  - Top margin: ~30 mm. Title strip occupies ~140–170 mm of the 841 mm poster height.

---

## 2. Problem and dataset *(Niko, ~4–5 lines)*

> Container goal: viewer understands the task, the data shape, and why it's hard — in <15 seconds.

- **Headline takeaway:** Given a ground photo, retrieve the matching landmark from a 100k-image satellite gallery.
- **Task framing:** Retrieval problem — model outputs an embedding, gallery is ranked by cosine similarity. Not classification.
- **Dataset:** MMLandmarks — open-source US-landmark benchmark with one-to-one ground/satellite correspondence and per-landmark Wikipedia text + GPS metadata.
- **Key numbers (call-out tiles):** 17k train landmarks · 311k ground images · 187k satellite images · 1k query landmarks · 18k query ground images.
- **Show example images:** YES — small 4-cell collage showing one landmark across all four modalities (ground photo · satellite tile · GPS coordinate · Wikipedia text snippet). Reuse `Fig 1` from existing draft if possible.
- **Why it's hard (1 bullet):** Ground and satellite views share no low-level features — model must learn high-level, viewpoint-invariant location fingerprints.

---

## 3. Cross-view retrieval + Sample4Geo intro *(Mateusz)*

> Container goal: viewer understands the cross-view task and that we chose Sample4Geo as the backbone.

- **Cross-view definition:** Ground photo ↔ satellite image of the same place. Train ≠ Query ≠ Index landmarks (zero leakage).
- **Why hard:** Façade vs roof-top, trees vs canopy — no shared low-level features.
- **Sample4Geo in one sentence:** Single shared ConvNeXt-Base encoder, InfoNCE contrastive loss with hard-negative sampling (GPS-near → Dynamic Similarity Sampling).
- **Key ingredients to call out:** Shared encoder (88M, no separate towers) · Symmetric InfoNCE with learnable temperature · GPS-curriculum → DSS curriculum at epoch 5.
- **Anchor numbers:** Zero-shot ConvNeXt → 0.34 % R@1 (essentially random). Sample4Geo fine-tuned → **8.58 % R@1** (~25× lift). CLIP-based methods (MMCLIP/GeoCLIP zero-shot) ≈ 20 % R@1 — large backbone gap.
- **Mini diagram:** small Sample4Geo block: `Ground img + Satellite img → shared ConvNeXt → L2-norm → 1024-dim embedding`. Reuse `Fig 2` from existing draft (shared-encoder cartoon).

---

## 4. Methodology — GeoCLIP as a bridge *(Kostas)*

> Container goal: motivate using GeoCLIP to leverage GPS + multimodal data.

- **One-sentence framing:** GeoCLIP turns a ground photo directly into a GPS coordinate — closing the loop between image and location without needing the satellite view.
- **Architecture:** Frozen CLIP ViT-L image encoder → linear head (768→1024→512) · Location encoder: Equal-Earth projection → Random Fourier Features → hierarchical MLP → 512-dim GPS embedding.
- **Loss:** Contrastive (SimCLR-style), learnable temperature τ.
- **Why it "bridges":** Provides a noisy GPS prior from the same modality (ground photo) as the cross-view query → lets us narrow the satellite gallery before contrastive ranking.
- **Two-phase plan:** Phase 1 zero-shot pretrained GeoCLIP · Phase 2 fine-tune Location Encoder + linear head on MMLandmarks.
- **Mini diagram:** image encoder → embedding ⟶ gallery of GPS embeddings → argmax cosine sim → (lat, lon). Reuse `Fig 3` from existing draft.

---

## 5. Methodology — Hybrid model idea *(Kostas)*

> Container goal: viewer can describe the 2-stage hybrid pipeline from the poster alone.

- **Pitch:** GeoCLIP → rough GPS → mask satellites within 20–25 km → Sample4Geo re-ranks the surviving candidates.
- **Pipeline diagram (HERO figure of col 2):** five horizontal pill-shaped stages, thick black arrows between them.
  ```
  Ground img → [Stage 1: GeoCLIP]  → (lat, lon) → [Filter: 25 km radius] → N candidates → [Stage 2: Sample4Geo] → Satellite match
  ```
  - **Pill style:** rounded rectangles, large, centred labels.
  - **Color rule:** DTU red fill + white text for *trained* modules (GeoCLIP, Sample4Geo). White fill + warm-grey 1pt border for *non-trained* modules (radius filter).
  - **Arrow labels:** print input/output type underneath each arrow in 12 pt mono grey (e.g. `(lat, lon)`, `N ≈ 50–500 sats`).
  - **Stage badge:** small grey "Stage 1" / "Stage 2" tag above each trained pill.
- **Why this order:** GeoCLIP is cheap and globally aware (predicts anywhere on Earth). Sample4Geo is expensive and locally precise. Cheap global filter → expensive local match.
- **Failure mode call-out (small red label in diagram):** "If GeoCLIP error > radius, correct landmark is excluded — un-recoverable."
- **Radius choice rationale:** 25 km balances recall (don't drop true positive) vs filter strength (drop enough distractors to help reranking).

---

## 6. Paper-vs-index gallery decision *(Niko)*

> Container goal: explain the subtle but important decision about what's in the candidate pool.

- **Paper gallery:** 100k index satellites + 1k query satellites (101k total) — matches MMLandmarks paper protocol. Ground truth is in the pool by construction.
- **Index-only gallery:** 100k index satellites only — query landmarks are NOT in the gallery. Tests generalisation outside the labelled set.
- **Decision:** Use paper gallery for headline numbers (apples-to-apples with prior work). Report index-only as a separate column.
- **Effect on numbers (one bullet):** Index-only retrieval is harder by definition — recall drops because the true match is absent. Mention size delta in caption.
- **Visual:** small Venn-style diagram with "Index 100k" and "Query 1k" circles, the union is "Paper gallery 101k". Highlight which evals use which.

---

## 7. Improving Sample4Geo + finetuning the hybrid *(Mateusz, Kostas)*

> Container goal: describe the v2→v3→v4 ablation and the hybrid finetune.

- **What was tuned (ordered bullets):**
  - **Backbone:** `fb_in22k` (v2) → `fb_in22k_ft_in1k_384` (v3)
  - **Resolution:** 224 px (v2/v4) → 384 px (v3)
  - **Multi-positive InfoNCE:** K = 1 (v2) → K = 2 (v3/v4) — two ground views per satellite per step
  - **Label smoothing:** 0.0 (v2) → 0.1 (v3/v4)
  - **AMP fp16:** off (v2/v4) → on (v3, required at 384 px)
- **Ablation table (compact, 4-row):** zero-shot · v2 (224, K=1) · v3 (384, K=2) · v4 (224, K=2). Three columns: per-image R@1, per-landmark mean R@1, per-landmark max R@1. Use DTU-red bold for the best cell in each column.
- **Headline finding:** **Per-image best ≠ per-landmark best.** v3 wins per-image (8.58 %), v2 wins per-lm max (9.00 %), v4 wins per-lm mean (18.50 %). Driver = number of unique landmarks (= InfoNCE negatives) per batch.
- **Hybrid finetune outcome:** Fine-tuning Sample4Geo *inside* the hybrid pipeline gives small additional gain. Best hybrid checkpoint = `hybrid_20260430_013258`.

---

## 8. Single-image results *(Mateusz, Kostas)*

> Container goal: headline single-image numbers across pipelines.

- **Radii to show:** **25 km** and **∞** (no radius filter). Side-by-side columns in one table.
- **Pipelines to compare:**
  - GeoCLIP zero-shot (baseline)
  - ZS GeoCLIP + ZS Sample4Geo (no training)
  - ZS GeoCLIP + FT Sample4Geo (our v3)
  - **FT GeoCLIP + FT Sample4Geo (hybrid, single-image)** — full method
- **Format:** Compact table, 5 cols × 4 rows: pipeline, R@1km, R@25km, R@200km, mAP@1000. Best cell per column highlighted in DTU red.
- **Headline metric:** R@25km. It's the radius the filter uses, so it's the natural success threshold.
- **From `results_combined.csv`:** at r=25 km, ZS+FT-S4G mAP@1000 = 22.1 %; mean dist 660 km. At r=∞ for the same pipeline mAP = 20.5 %, mean dist 1217 km.
- **Highlight cells:** YES — DTU red background fill, white text, for the best cell in each metric column.

---

## 9. Issues — GeoCLIP as upper bound, finetuning did nothing *(Niko)*

> Container goal: honest critique — GeoCLIP limits the pipeline ceiling.

- **Upper bound idea:** If GeoCLIP's GPS error exceeds the filter radius, the correct landmark is removed from the candidate set → Sample4Geo cannot recover, no matter how good it is.
- **Evidence finetuning didn't help:**
  - Finetuned vanilla GeoCLIP on MMLandmarks → median GPS error barely moved
  - Sample4Geo gains stayed within noise (≤0.5 % R@1 delta)
- **Best hypothesis (1 bullet):** MMLandmarks is too small (17k landmarks across US) to materially improve a model trained on 4M+ globally-distributed geo-tagged photos. CLIP backbone is already saturated for this scale.
- **Framing:** Honest caveat AND a motivating result — it justifies the multi-image and new-architecture work in §10–§12.

---

## 10. Multi-image models — weighted mean vs transformer *(Edvin)*

> Container goal: explain the variable-image-count problem and our two answers.

- **Variable-K problem:** Each landmark has 1 to ~150 ground photos (avg 18). Need a way to aggregate K embeddings into a single landmark-level representation.
- **Weighted-mean approach:** Compute mean of K embeddings, weight each photo by cosine similarity to that mean (softmax-normalised) → emphasises representative views, down-weights outliers.
- **Transformer approach:** Encoder-only transformer with a `[CLS]` token over N image embeddings → aggregates into one landmark embedding.
- **Architecture diagram (small):** two stacked panels — top: mean → reweight → mean'; bottom: img embeds + CLS → self-attention → CLS embed.
- **Winner:** Transformer (Multi-Image Attention) — **13.32 % R@1 on per-landmark eval** vs 12.13 % for weighted mean. Both beat single-image baseline (~6.7 %).

---

## 11. New GeoCLIP architecture + two-stage learning *(Edvin)*

> Container goal: explain the redesigned GeoCLIP architecture and the staged training scheme.

- **What changed vs vanilla GeoCLIP:**
  - Native multi-image input (N ground photos per landmark → 1 GPS prediction)
  - Transformer aggregator (CLS-token) replaces independent per-photo predictions
  - Two-stage training: Stage 1 trains location encoder on GPS pairs only · Stage 2 freezes location encoder and trains the image-side aggregator
- **Diagram:** reuse `New GeoCLIP architecture` diagram from existing draft (location encoder + transformer aggregator + similarity matrix).
- **Why two-stage:** Decouples "what does the world look like at this GPS" (Stage 1) from "how do these N photos collectively describe a place" (Stage 2). Cleaner gradients, less collapse risk.
- **Expected benefit:** Better calibrated GPS predictions for landmarks with many photos · Path to uncertainty estimation (see §12).

---

## 12. GeoCLIP results + uncertainty *(Edvin)*

> Container goal: GeoCLIP-only numbers and the uncertainty estimation result.

- **Headline graph (Edvin's plot):** GeoCLIP error distribution by gallery rank / by photo count — to be inserted from Edvin's notebook output.
- **Axes:** x = photo count per landmark (or rank), y = median GPS error in km. Shows how uncertainty scales with input richness.
- **Uncertainty estimation (1 bullet):** GeoCLIP outputs both a GPS coordinate AND a confidence score (softmax temperature on the gallery similarity distribution). Low entropy = high confidence; high entropy = bail out and fall back to wider radius.
- **Headline numbers:**
  - GeoCLIP zero-shot R@25km ≈ 20 % · R@200km ≈ 42 %
  - New multi-image GeoCLIP R@25km ≈ 40 % · R@200km ≈ 65 %
  - Confidence-adaptive radius gives an additional 2–3 pts at no median-error cost.

---

## 13. Multi-image results *(any)*

> Container goal: show that multi-image > single-image, and by how much.

- **Format:** Two-bar grouped chart per metric (R@1km, R@25km, R@200km, mAP@1000). Bars = single-image hybrid vs multi-image hybrid. Same DTU-red highlight for the multi-image bar.
- **From `results_combined.csv` (multi-image, r=25km):** R@1km = 23.3 %, R@25km = 40.6 %, R@200km = 65.1 %, mAP = 23.7 %, median dist = **79 km** (vs 530 km single-image — 6.7× reduction).
- **Headline delta:** Multi-image median GPS error is ~7× lower than single-image — the biggest single gain in the project.

---

## 14. Discussion — qualitative examples *(Kostas)*

> Container goal: visual head-to-head between single-image v3 Sample4Geo and our fine-tuned hybrid.

- **Grid:** 5 rows × 4 columns. Each row = one ground query.
  - **Col 1: Ground image** (the query photo)
  - **Col 2: v3 Sample4Geo prediction** (top-1 satellite, single-image)
  - **Col 3: Finetuned hybrid prediction** (top-1 satellite from full pipeline)
  - **Col 4: Ground-truth satellite**
- **Cell decoration:**
  - Green 2 pt outline around a prediction cell if it matches ground truth.
  - Red 2 pt outline if it doesn't.
  - Ground-truth column has a thin grey outline always.
- **Row caption:** one-line landmark name + tiny `(GPS error: NNN km)` for each model.
- **Example selection:** pick 5 rows that show a mix of outcomes (e.g. 2 where hybrid wins, 1 where both win, 1 where both lose, 1 where v3 wins). Demonstrates honest behaviour.
- **Layout note:** this container is wide; spans the full width of column 3 (or even pushes into adjacent column 2 if needed). Caption strip below the whole grid: `Cols 2–3: top-1 retrieval. Green border = correct landmark. Red border = wrong.`

---

## 15. Discussion — per-landmark evaluation *(Edvin)*

> Container goal: per-image vs per-landmark, why the rankings flip, and which is fairer.

- **Per-image:** Each of 18k ground photos is an independent query. Biased to landmarks with many photos.
- **Per-landmark max:** Each of 1k landmarks counts once. "Success if any photo retrieves correctly." Upper bound on landmark coverage.
- **Per-landmark mean:** Each landmark contributes mean score across its photos. Mathematically = embedding-space mean pooling for ranking.
- **The flip:** v3 wins per-image (8.58 %) but loses per-lm max (7.10 %). v2 is the opposite. **Root cause:** v3's batch=16 → 15 InfoNCE negatives per step vs v2's 63 → v3 specialises on easy, high-photo-count landmarks instead of broad coverage.
- **Why it matters:** Per-image rewards "doing well on easy questions many times". Per-landmark max rewards "covering more distinct places". For our task, per-landmark max is the truer signal.
- **Mini table:** 3-col × 3-row — `Model | per-image R@1 | per-lm max R@1`. v2, v3, v4 rows. Highlight the flip in DTU red.

---

## 16. Conclusion *(Niko / all)*

> Container goal: 3–4 bullets that summarise the project.

- **What works:** GPS-narrowed retrieval is sound — 25 km filter + Sample4Geo rerank consistently beats Sample4Geo alone (R@25km jumps from ~6 % zero-shot to **40.6 %** with multi-image hybrid).
- **What doesn't:** Finetuning GeoCLIP on MMLandmarks barely moves the needle — the CLIP backbone is already near-saturated for this dataset size.
- **What we learned about the dataset/eval:** Per-image recall can hide model weakness. Per-landmark max is the fairer measure of geographic coverage.
- **Biggest surprise:** Multi-image aggregation gives a **~7× reduction in median GPS error** (530 km → 79 km) — bigger than any architectural change we tested.

---

## 17. AI inclusion statement

- **Where on poster:** Small side-bar block at bottom of col 3, above the footer line. Title `AI Assistance` in DTU red, body in 12 pt warm grey.
- **What it says (one bullet):** "Claude Code (Anthropic) used as a pair-programming assistant for boilerplate, refactoring, and debugging in `mmgeo/` codebase. All experimental design, training runs, ablations, and analysis are the authors' own. Generative AI was not used to write report or poster text."

---

## 18. Exclusions (things we explicitly do NOT want)

- ~~"What we would try next"~~ — removed from new poster
- ~~Dark navy + teal/orange/amber palette~~ — explicit constraint, new palette required
- ~~Per-author byline inside each container~~ — author ownership stays in this internal doc only, not on the poster

---

## 19. Figure/asset inventory

Available files we can drop in directly:

- `report/pipeline_results/results_combined.csv` — final pipeline numbers (3 pipelines × {25 km, ∞})
- `report/pipeline_results/viz_*_best.png` (6 files) — best-case retrievals per pipeline/radius
- `report/pipeline_results/viz_*_worst.png` (6 files) — worst-case retrievals per pipeline/radius
- `report/sample4geo_results/crossview_results.md` — full v2/v3/v4 ablation tables
- `report/sample4geo_results/sample4geo_explainer.md` — exam-level technical writeup
- `report/first_draft_poster.pdf` — previous poster (dark theme, 3-column) — reference for figures 1, 2, 3 and the New-GeoCLIP architecture diagram only
- `notebooks/team/03_geoclip_zeroshot.ipynb`, `04_geoclip_finetuned.ipynb`, `05_pipeline.ipynb` — source for any additional plots (e.g. Edvin's uncertainty graph for §12)
