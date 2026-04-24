# Session summary — zero-shot GeoCLIP debugging

## ✅ Resolution (closed)

The gap was a **gallery definition mismatch**, not a code bug. The camera-ready MML paper Sec 5.2 uses `index ∪ query_landmarks` (100,539 GPS) — the query landmarks' GT coordinates are in the gallery itself. Once we added `source: "paper"` to `load_gallery_coords`, we reproduced the paper's Table 3 row to within 0.02 points (**21.35 / 36.44 / 48.61 / 71.41 / 91.52**). The arXiv Dec 2025 cut said "100k from satellite index set" which is what tripped us up; the camera-ready says "combined satellite index and query sets, 101k". First author Oskar Kristoffersen confirmed this and frames the two numbers as *"21 % is a geolocalization upper limit, 6.67 % is more realistic in the wild"*. See [docs/team/geoclip.md](../../team/geoclip.md) for current headline numbers.

Everything below is the **investigation paper trail** kept for reference — the "unresolved 14-point gap" narrative is out of date.

---

## Context

We spent an extended session trying to reproduce the MMLandmarks paper's **21.37 %** @1 km on the zero-shot GeoCLIP baseline. Our current run on the paper's gallery (100k `mml_index_satellite.csv`, index-only) stands at **6.67 %** @1 km. This document summarizes everything we verified, everything we changed, and what's still open.

## What's known to be right

1. **Gallery matches paper.** `gallery.source: index` loads exactly the 99,539 GPS points from `index/mml_index_satellite.csv`. Confirmed by prof's email. Ceiling calc (@1 km = 69.69 %) confirms the gallery is dense enough — 6.67 % is not a density floor.
2. **Our retrieval = native `model.predict`.** HPC debug script showed our `predict_batch` matches `model.predict` **500/500** on identical images. Retrieval path is not the bug. Our code now also routes everything through `self.model.forward` explicitly to guarantee parity.
3. **Distance metric matches paper.** MMLandmarks paper uses Haversine (Sec 5.2). GeoCLIP's own paper uses Geodesic (via geopy), but that's not the paper we're reproducing. Our `evaluate.py` uses Haversine — correct.
4. **`_patch_image_encoder` is mathematically lossless.** Verified via HF source: transformers 5.x returns `BaseModelOutputWithPooling` from `get_image_features`, whose `.pooler_output` field holds the exact same `visual_projection(pooled)` tensor older 4.x versions returned directly (HF commit 55dadb8, PR #42564).
5. **Training-set ceiling argument resolved.** The old 19.22 % on the 17k train gallery looked close to paper's 21.37 % only because of cluster-luck on co-located tourist landmarks — not a fair comparison. 6.67 % on the 100k index gallery is the honest number.

## What we changed

- `load_gallery_coords` simplified to one `source: "train" | "index" | "both"` arg.
- [configs/geoclip_baseline.yaml](configs/geoclip_baseline.yaml), [configs/geoclip_train.yaml](configs/geoclip_train.yaml) set `gallery.source: index`.
- [src/mmgeo/geolocalizations/geoclip/geoclip_baseline.py](src/mmgeo/geolocalizations/geoclip/geoclip_baseline.py):
  - `_patch_image_encoder` kept (needed for `transformers==5.5.3`).
  - Preprocessing switched from `self.model.image_encoder.preprocess_image` (CLIP mean/std via CLIPProcessor) to `img_val_transform()` from `geoclip.train.dataloader` (ImageNet mean/std). **This is the untested variable.**
  - Retrieval uses `self.model(imgs, gallery_tensor)` + `softmax` + `argmax` directly — no hand-rolled cosine.
  - `build_gallery` just caches raw GPS tensor; model re-encodes per call (matches native).
- Syntax bug (`"""Predict GP  ` unterminated docstring in `predict_batch`) fixed — that was the cause of the last HPC crash.
- Plan/history/debug scaffolding in `scripts/geoclip_debug.py`, `scripts/run_geo_debug.sh`.

## Open question — what's still untested

Will the preprocessing swap (ImageNet norm instead of CLIP norm) close the 6.67 → 21 gap?

- Stat difference: stds differ ~17 %; means nearly identical. Plausibly absorbed by ViT's first LayerNorm, so effect may be small.
- Structural difference: `img_val_transform` resizes 256 → center-crop 224 (keeps 32 px margin); `preprocess_image` resizes 224 → center-crop 224 (no margin). Different image content.
- Best evidence it's the fix: GeoCLIP's own train dataloader uses ImageNet norm, which is what the MLP head was trained to consume.
- Best evidence against: stds "washed away by LayerNorm" intuition suggests effect is small.

## Fallback plan if re-run lands near 6.67 %

Email the professor for his exact zero-shot eval script. Specific asks:
1. Which image preprocessing (`img_val_transform` ImageNet vs `preprocess_image` CLIP vs custom)?
2. Was `logit_scale` / temperature handled differently?
3. Any per-landmark aggregation before thresholding (e.g., averaging predictions for the N images of each landmark)?
4. What pretrained GeoCLIP snapshot was used (PyPI `geoclip==X.Y.Z` or a local checkpoint)?

## Memory saved during session

- [geoclip_preprocessing_inconsistency.md](../.claude/projects/-zhome-57-e-219332-Multimodal-Geo-Spatial-Learning/memory/geoclip_preprocessing_inconsistency.md) — upstream ImageNet-train / CLIP-predict inconsistency is a real upstream bug, not ours.
- [geoclip_baseline_cleanup_items.md](../.claude/projects/-zhome-57-e-219332-Multimodal-Geo-Spatial-Learning/memory/geoclip_baseline_cleanup_items.md) — six code-quality items in `geoclip_baseline.py` (misleading docstring, length-mismatched `landmark_ids` return, redundant `.split()`, `iterrows` anti-pattern, positional `j` index, missing shape/NaN guard in `build_gallery`). Not bugs.
- [geoclip_zeroshot_notebook_cleanup.md](../.claude/projects/-zhome-57-e-219332-Multimodal-Geo-Spatial-Learning/memory/geoclip_zeroshot_notebook_cleanup.md) — five items in `03_geoclip_zeroshot.ipynb`: stale intro markdown (wrong gallery size), stale summary with wrong paper-contrast table and debunked TenCrop/LLaVA speculation, "Query landmarks" label actually counts images, fragile `../../` DATA_ROOT path, stale cell output from crashed run.

Note: `MEMORY.md` index currently references only the first two; the third was written while plan mode was active so needs a one-line index addition next time we're out of plan mode.

## Critical files

- [src/mmgeo/geolocalizations/geoclip/geoclip_baseline.py](src/mmgeo/geolocalizations/geoclip/geoclip_baseline.py) — current active code.
- [configs/geoclip_baseline.yaml](configs/geoclip_baseline.yaml) — `gallery.source: index`.
- [notebooks/team/03_geoclip_zeroshot.ipynb](notebooks/team/03_geoclip_zeroshot.ipynb) — zero-shot eval, needs post-run markdown refresh.
- [docs/team/geoclip.md](docs/team/geoclip.md) — team-facing summary, also needs refresh.

## Immediate next action

Submit the zero-shot job on HPC (current code is clean, syntax verified with `py_compile`). Read `Output_<jobid>.out` for @1 km. Three branches:

| Outcome | Action |
|---|---|
| Near 21 % | Preprocessing was the fix. Update `docs/team/geoclip.md` numbers, update notebook markdown, add the MEMORY.md index entry for the notebook cleanup file, delete debug scaffolding. |
| Between 7 % and 20 % | Normalization is *part* of the fix but not all. Email professor for eval script. |
| Still 6.67 % | Normalization wasn't it. Email professor for eval script (same as above). |

## Q&A — session wrap-up

### What's different in the currently-queued HPC job?

Exactly **one behavioral change** vs the 6.67 % run, plus one refactor:

| Aspect | Previous (6.67 %) | Current job |
|---|---|---|
| Image preprocessing | `self.model.image_encoder.preprocess_image(img)` → CLIPProcessor → CLIP mean/std `(0.481, 0.458, 0.408) / (0.269, 0.261, 0.276)`, Resize 224 → CenterCrop 224 | **`img_val_transform()`** → torchvision → ImageNet mean/std `(0.485, 0.456, 0.406) / (0.229, 0.224, 0.225)`, **Resize 256 → CenterCrop 224** |
| Retrieval path | hand-rolled `F.normalize` + `img @ gallery.T` + argmax on pre-encoded gallery | `self.model(imgs, gallery_tensor)` + softmax + argmax; gallery re-encoded per batch. **Mathematically identical to the previous path** (already verified 500/500 on HPC). |
| Gallery | 100k index | 100k index (unchanged) |
| Patch | applied | applied (unchanged) |

So the single variable being tested is **ImageNet mean/std + Resize(256) vs CLIP mean/std + Resize(224)**. Everything else is bit-identical.

### Did we use `preprocess_image` before, or always `img_val_transform`?

Chronological truth:
1. **Run 1 (train gallery, 17,557):** `preprocess_image` → 19.22 % @1 km.
2. **Run 2 (index gallery, 99,539):** `preprocess_image` → 6.67 % @1 km.
3. **Attempted run 3:** I switched to `img_val_transform` **and** mistakenly deleted `_patch_image_encoder`. Crashed with `BaseModelOutputWithPooling` TypeError. No number.
4. **Attempted run 4:** `img_val_transform` + patch reinstated, but I left an unterminated docstring in `predict_batch`. Crashed with SyntaxError. No number.
5. **Current queued run:** `img_val_transform` + patch + syntax fix. **First successful run with ImageNet normalization.**

So every number we've actually *recorded* so far used `preprocess_image`. The hypothesis that the paper authors used `img_val_transform` is **completely untested** until this job returns.

### If `img_val_transform` doesn't fix it, what else can we try without the professor's script?

Ordered by expected payoff ÷ effort:

1. **Per-landmark aggregation at eval.** Paper reports 18,689 images; if they averaged predictions across the ~19 images of each landmark before thresholding, the effective signal-to-noise is higher. Cheap to test: keep our per-image predictions, group by landmark_id, take the median/mean GPS per landmark, then threshold. Expected gain: moderate — the paper says *"18,689 ground images"* (not "1,000 landmarks") so they likely didn't aggregate, but worth a sanity check.
2. **Different `logit_scale` handling.** The shipped `logit_scale` is initialised to `ln(1/0.07)` ≈ 2.659 and fine-tuned during training. If the paper authors reset it to exactly 0.07 (or to `0.0`) at inference, softmax behaves differently — though argmax would be unchanged. Unlikely.
3. **Switch to Geodesic distance.** Paper uses Haversine per Sec 5.2, so not actually a candidate here. (Skip.)
4. **TenCrop or simple augmentation averaging.** Professor said no; skip.
5. **Fine-tune the MLP head on MML train split ground images**, then re-evaluate zero-shot on query. Changes the "zero-shot" definition but might close the gap if the upstream weights were trained with slightly different data.

Beyond these, the remaining plausible explanations require the professor's code:
- Different pretrained checkpoint snapshot (see next section).
- Different CLIP backbone init (e.g., loading a *patched* CLIP weights file, or using a different HF revision).
- Custom text or satellite pre-filtering at eval we don't know about.

### What other GeoCLIP checkpoints could the paper authors have used?

The PyPI `geoclip` package ships **one** set of weights in `geoclip/model/weights/`:
- `image_encoder_mlp_weights.pth` — the MLP head on top of frozen CLIP.
- `location_encoder_weights.pth` — the 3-scale RFF+MLP location tower.
- `logit_scale_weights.pth` — the scalar temperature.

That's the "official" pretrained model shipped since late 2023. Possible alternative weights the paper authors might have used:
- **A different PyPI version.** Check pinned version in their repo if they publish one. We're on whatever `uv sync` resolves today — could be newer weights than the MML paper used. Easy to check with `pip show geoclip`.
- **The GeoCLIP GitHub repo directly** ([github.com/VicenteVivan/geo-clip](https://github.com/VicenteVivan/geo-clip)) — may have extra checkpoint files not shipped via PyPI (e.g. per-dataset variants: Im2GPS3k, YFCC26k, GWS15k). The GeoCLIP paper reports on multiple datasets in Section 4 — they may have released per-dataset weights.
- **A *retrained* GeoCLIP** by the MML paper authors on MP-16 with their exact training recipe. Would technically still be "off-shelf" if they used the original author's code but re-ran training. Unlikely given their "off-shelf GeoCLIP" language implies pretrained.
- **A HuggingFace model hub upload.** Sometimes authors upload checkpoints separately from PyPI. Search `huggingface.co/models?search=geoclip` if this thread is still open.

**Concrete diagnostic to add to the professor email:** *"Which `geoclip` package version and which checkpoint file exactly? Was it the PyPI default, or a specific GitHub release?"*

## Save target

Copy this plan file to [docs/personal/Nikos/](docs/personal/Nikos/) with filename `geoclip_zeroshot_debugging.md` so it survives outside `.claude/plans/`. Plan mode prevents this edit in-place; will do after ExitPlanMode.

---

# (Historical) Match paper's zero-shot GeoCLIP number by using the 100k index gallery

## Context

`docs/team/geoclip.md` reports our zero-shot GeoCLIP as **19.22 % @1 km / 34.56 % @25 km**; the MMLandmarks paper (Table 3, Ground → GPS row) reports **21.37 / 36.44** on the same query set. Root cause confirmed by reading the paper: the paper uses a **100k-point GPS gallery from the satellite index set** (Sec 5.2, page 7), while our current run uses the **17,557 train-landmark GPS** only — see `Gallery size: 17,557` in [Output_28262499.out:33](Output_28262499.out#L33) and `include_index: false` in [configs/geoclip_baseline.yaml:7](configs/geoclip_baseline.yaml#L7).

Why it matters: GeoCLIP predicts *one of the gallery points*. Query landmarks are disjoint from train landmarks, so the true GPS is never in our gallery. 17.5k points across the US ≈ ~20–25 km average spacing, which caps @1 km accuracy. The paper's 100k index gallery is dense *and* deliberately offset >500 m from every training landmark (Sec 3.2), designed exactly to be a coverage grid.

A secondary cleanup: [docs/team/geoclip.md:65-67](docs/team/geoclip.md#L65-L67) speculates the gap is due to TenCrop test-time augmentation and LLaVA indoor filtering. The paper uses **neither** at evaluation — TenCrop is never mentioned; LLaVA filtering is training-only and Sec 3.3 explicitly says *"Unless stated otherwise, both indoor and outdoor images are used"*. That speculation is wrong and should be removed.

## Decision

Go with **option B**: re-run zero-shot using only the ~101,302 index-satellite GPS coordinates as the gallery. This exactly matches the paper's Sec 5.2 protocol.

Not option A (current, 17,557 train-only) — too sparse.
Not option C (train + index, 118k) — strictly denser than paper so would over-report versus the paper's published row; option B gives the cleanest apples-to-apples comparison.

## Where the index is consumed (safety audit)

Grep across `*.py / *.yaml / *.ipynb / *.md` for `mml_index_satellite | include_index | index_only`:

**GeoCLIP pipeline (the one we're changing):**
- [src/mmgeo/geolocalizations/geoclip/geoclip_baseline.py:102-127](src/mmgeo/geolocalizations/geoclip/geoclip_baseline.py#L102-L127) — **the only place** `index/mml_index_satellite.csv` is actually read. Currently only two modes (train-only / train+index).
- [scripts/geoclip_train.py:44-45](scripts/geoclip_train.py#L44-L45) — calls `load_gallery_coords(data_root, include_index=cfg["gallery"]["include_index"])` for per-epoch eval during fine-tune.
- [notebooks/team/03_geoclip_zeroshot.ipynb:442](notebooks/team/03_geoclip_zeroshot.ipynb#L442) and [notebooks/team/04_geoclip_finetuned.ipynb:117](notebooks/team/04_geoclip_finetuned.ipynb#L117) — same call signature.
- [configs/geoclip_baseline.yaml:7](configs/geoclip_baseline.yaml#L7), [configs/geoclip_train.yaml:5](configs/geoclip_train.yaml#L5) — both have `include_index: false` today.

**Cross-view pipeline (unrelated — separate flag, separate data path):**
- [src/mmgeo/crossview/train.py:407,422](src/mmgeo/crossview/train.py#L407-L422) — its `include_index` flag controls loading of `MMLImageDataset(data_root, "index", modality)`, i.e. **image files**, as hard distractors for Ground↔Satellite retrieval. It never reads `mml_index_satellite.csv`.
- [configs/crossview_baseline.yaml:35](configs/crossview_baseline.yaml#L35), [configs/crossview_convnext_base.yaml:34](configs/crossview_convnext_base.yaml#L34) — isolated to cross-view.

**Conclusion:** the `mml_index_satellite.csv` GPS coords are consumed only by `load_gallery_coords` in the GeoCLIP baseline module. No training path uses them as learning targets (training config has `include_index: false`); no validation split depends on them; cross-view doesn't touch this CSV. Switching the zero-shot gallery to index-only is safe — no leakage risk, no collateral effects.

One knock-on: the fine-tune script ([scripts/geoclip_train.py:44](scripts/geoclip_train.py#L44)) builds its per-epoch eval gallery from the same config key. For consistency with the new zero-shot protocol, also flip [configs/geoclip_train.yaml:5](configs/geoclip_train.yaml#L5) to `index_only: true` so the per-epoch "Acc@25km gate" is measured against the same gallery as zero-shot. Without this, the fine-tune gate would compare apples (17k gallery) to the new zero-shot oranges (100k).

## Changes

### 1. `load_gallery_coords` — add `index_only` switch
[src/mmgeo/geolocalizations/geoclip/geoclip_baseline.py:102-127](src/mmgeo/geolocalizations/geoclip/geoclip_baseline.py#L102-L127)

Today the function can load *train only* or *train + index*; there is no path to *index only*. Add a third mode:

```python
def load_gallery_coords(
    data_root: Path,
    include_index: bool = False,
    index_only: bool = False,
) -> np.ndarray:
    if index_only:
        index_df = pd.read_csv(data_root / "index" / "mml_index_satellite.csv")
        return index_df[["lat", "lon"]].values
    ...  # existing logic
```

Guard against both flags being true at once (raise `ValueError`).

### 2. Configs — flip to index-only
[configs/geoclip_baseline.yaml:6-7](configs/geoclip_baseline.yaml#L6-L7) **and** [configs/geoclip_train.yaml:5](configs/geoclip_train.yaml#L5)

```yaml
gallery:
  include_index: false
  index_only: true      # match paper Sec 5.2: 100k index-satellite GPS
```

Updating both keeps zero-shot and fine-tune's per-epoch eval on the same gallery.

### 3. Call sites — thread the new flag
Three places read `cfg["gallery"]["include_index"]`; each needs to also pass `index_only`:

- [scripts/geoclip_train.py:44-45](scripts/geoclip_train.py#L44-L45)
- [notebooks/team/03_geoclip_zeroshot.ipynb:442](notebooks/team/03_geoclip_zeroshot.ipynb#L442)
- [notebooks/team/04_geoclip_finetuned.ipynb:117](notebooks/team/04_geoclip_finetuned.ipynb#L117)

Pattern: `load_gallery_coords(DATA_ROOT, include_index=cfg["gallery"]["include_index"], index_only=cfg["gallery"].get("index_only", False))`.

Then re-execute zero-shot on HPC:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/team/03_geoclip_zeroshot.ipynb
```

### 4. Update docs/team/geoclip.md
[docs/team/geoclip.md](docs/team/geoclip.md)

- **Line 31** (evaluation protocol): change gallery description to *"101,302 index-satellite GPS points from `index/mml_index_satellite.csv` — matches paper Sec 5.2. Train-landmark gallery (17,557) retained as a sparser ablation."*
- **Lines 43–53** (Zero-shot benchmark table): replace with the new numbers from step 3.
- **Lines 55–67** (Paper contrast): update "Ours" row with the new numbers; **delete the TenCrop/LLaVA speculation** (wrong — paper uses neither at evaluation) and replace with a one-sentence note that our setup now matches the paper's gallery protocol, so any residual gap is model/randomness, not evaluation add-ons.

## Verification

1. `Gallery size: 101302 GPS points` appears in the job's stdout (confirms the new flag wired end-to-end).
2. Zero-shot Acc@1 km lands close to **21.37 %** and Acc@25 km close to **36.44 %** (within ~0.5 pts). If off by more, revisit whether GeoCLIP's `preprocess_image` or the frozen `_patch_image_encoder` in [geoclip_baseline.py:16-26](src/mmgeo/geolocalizations/geoclip/geoclip_baseline.py#L16-L26) is a subtle culprit.
3. Committed diff of [docs/team/geoclip.md](docs/team/geoclip.md) no longer contains "TenCrop" or "LLaVA" as speculative causes.
