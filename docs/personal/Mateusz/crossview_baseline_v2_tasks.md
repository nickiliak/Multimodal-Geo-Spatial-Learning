# Cross-View Baseline v1 -> v2 Tasks

## Purpose
This note documents the current state of the **Sample4Geo-like cross-view baseline on MMLandmarks**, what was originally intended when implementing it, what was actually achieved in baseline v1, and what should be done next in baseline v2.

It is written for:
- teammates working on other project directions,
- later report writing,
- future continuation by me,
- possible AI-agent support.

---

# 1. Baseline v1 summary

## Goal of v1
The goal of baseline v1 was to build a **working image-image cross-view retrieval baseline** for MMLandmarks, inspired by Sample4Geo:
- shared encoder for ground and satellite images,
- contrastive training with symmetric InfoNCE,
- landmark-level pairing instead of fixed one-to-one image pairing,
- simple retrieval evaluation to verify that the pipeline learns.

The intention was **not yet** to reproduce the full final benchmark protocol perfectly, but to first get a clean, trainable, inspectable baseline running end-to-end.

## What was implemented
Baseline v1 currently contains:
- **Shared-weight ConvNeXt encoder** (`CrossViewModel`)
- **Symmetric InfoNCE loss** with learnable temperature
- **MMLandmarks landmark-based training pairs**: one ground + one satellite image sampled from the same landmark
- **UniqueLandmarkSampler** to avoid duplicate landmark IDs inside one batch
- **Recall@1 / Recall@5 / Recall@10 evaluation**
- Basic training diagnostics:
  - train loss
  - diagonal similarity
  - off-diagonal similarity
  - similarity margin
  - batch accuracy
  - temperature

## Important design decision in v1
Because MMLandmarks has **multiple ground images and multiple satellite images per landmark**, training was adapted to the landmark identity level.

This means a training sample is **not** one permanently fixed image pair.
Instead, for landmark `L`, each epoch may sample:
- one random ground image from `L`
- one random satellite image from `L`

This was intentional and is the right direction for MMLandmarks.

---

# 2. What v1 did well

## 2.1 The pipeline works end-to-end
The model trains successfully, runs on HPC, evaluates during training, and saves checkpoints.

## 2.2 The model is actually learning
Training did not collapse. The metrics improved steadily across the run.

### Final baseline v1 result (from `baseline_v1_final.out`)
Best observed result:
- **Recall@1 = 0.0973 (9.73%)**
- **Recall@5 = 0.2711 (27.11%)**
- **Recall@10 = 0.3762 (37.62%)**

### Training trend
Compared with early epochs, the run showed:
- lower loss,
- higher batch accuracy,
- larger diagonal-vs-offdiagonal similarity margin,
- better retrieval scores.

So v1 is a valid starting baseline, not a broken experiment.

## 2.3 The training logic matches MMLandmarks better than fixed-pair logic
The current design already respects the fact that MMLandmarks is **instance-based**.
That is an important adaptation relative to simpler one-pair-per-location thinking.

---

# 3. What v1 does NOT yet match well enough

This section is the most important one for baseline v2.

## 3.1 Evaluation protocol is currently too simplified
### Current state
Evaluation currently does:
- **query = ground images from the `query` split**
- **index = satellite images from the `query` split**

So retrieval is currently **query ground -> query satellite**.

### Original intention
The intention was to have a quick and simple retrieval check during training, so that we could confirm the model learns before implementing the full benchmark protocol.

### Problem
This is **not yet the same as the proper benchmark-style retrieval setup**.
The papers and the MMLandmarks benchmark logic expect retrieval against a larger and proper **index/gallery split**, not only against the satellite images from `query`.

### Consequence
The current v1 numbers are useful for:
- debugging,
- tracking learning progress,
- comparing small internal changes.

But they should **not yet** be treated as directly benchmark-comparable paper numbers.

---

## 3.2 Metrics are incomplete for MMLandmarks-style benchmarking
### Current state
v1 reports only:
- Recall@1
- Recall@5
- Recall@10

### Original intention
The intention was to follow the common retrieval practice used in cross-view papers like Sample4Geo and keep evaluation simple at first.

### Problem
For an instance-level benchmark like MMLandmarks, this is incomplete.
A stronger and more paper-aligned setup should also report **mAP@1k**.

### Consequence
Right now, the baseline captures whether the correct landmark appears near the top, but does **not** fully describe ranking quality across many valid positives.

---

## 3.3 The code claims Sample4Geo-style hard-negative logic, but v1 is mostly the simpler version
### Current state
The training file says the baseline implements:
1. GPS-based hard negative sampling (early)
2. Dynamic Similarity Sampling (later)

But in practice, the visible implemented pipeline is still centered around:
- random landmark sampling,
- unique landmarks per batch,
- standard symmetric InfoNCE.

### Original intention
The intention was to make the implementation evolve toward the full Sample4Geo-style pipeline.

### Problem
As of v1, this looks more like a **Sample4Geo-inspired baseline** than a full hard-negative mining implementation.

### Consequence
Methodologically, v1 can be described as:
- **aligned with the main Sample4Geo direction**,
- but **not yet a full reproduction of its sampling strategy**.

---

## 3.4 Training still uses one sampled positive pair per landmark per batch
### Current state
For each landmark in a batch, we currently sample:
- one ground image,
- one satellite image,
- and treat this as the positive pair.

### Original intention
This was done because it keeps training simple and compatible with standard diagonal InfoNCE.

### Problem
MMLandmarks contains **multiple valid positives per landmark**. v1 does not explicitly exploit multiple same-landmark positives in one loss computation.

### Consequence
This is not wrong, but it is still a simplification of the full instance-level setting.

---

## 3.5 Backbone / setup is a practical baseline, not the strongest paper-style configuration
### Current state
v1 uses a smaller/faster setup intended to get working results quickly.

### Original intention
Fast iteration first, stronger model later.

### Problem
This means v1 is more of a **sanity-check baseline** than a final strong benchmark run.

### Consequence
If performance is lower than expected, it may partly be because the current setup is intentionally conservative.

---

# 4. What can and cannot be compared to papers

## 4.1 What v1 can be compared to
### Sample4Geo
v1 can be compared to Sample4Geo in terms of:
- overall method idea,
- shared encoder,
- symmetric contrastive retrieval training,
- the role of hard negatives as a missing next step.

This is a **methodological comparison**, not a fair raw-score comparison.

### MMLandmarks paper
v1 can eventually be compared to the MMLandmarks paper **if** the evaluation protocol is aligned better.
This is the most relevant benchmark source because it is:
- the same dataset,
- the same task family,
- the same benchmark tables.

## 4.2 What v1 should NOT be directly compared to
The current v1 scores should **not** be directly compared numerically to:
- Sample4Geo benchmark numbers,
- GLDv2 numbers,
- ILIAS numbers,
- or any paper that uses a different dataset/protocol.

Reason: the dataset, query/index setup, and the notion of positives are different.

---

# 5. Baseline v2 tasks

This section lists the concrete next steps.

---

## Task 1 — Replace simplified evaluation with official query->index retrieval

### What to do
Change evaluation from:
- `query ground -> query satellite`

to proper benchmark-style retrieval using:
- `query ground -> index satellite`

And ideally also the reverse direction:
- `query satellite -> index ground`

### Why
This is the single most important step for making the baseline results closer to paper-style benchmarking.

Without this change, the current numbers are only internal progress indicators.

### How
1. In `_run_eval`, stop building the retrieval index from `split="query"`.
2. Use `MMLImageDataset(data_root, "index", modality, ...)` for the gallery side.
3. Keep query images from `query`.
4. Run:
   - ground-to-satellite evaluation
   - satellite-to-ground evaluation
5. Store both result directions clearly.

### Expected outcome
- Results become much more meaningful.
- We can compare much more honestly to MMLandmarks benchmark tables.

---

## Task 2 — Add mAP@1k

### What to do
Extend evaluation to compute **mAP@1k** in addition to Recall@1/5/10.

### Why
MMLandmarks is instance-level and may have multiple relevant items for the same landmark.
mAP@1k is therefore a better ranking metric and aligns better with paper reporting.

### How
1. Add a function similar to `compute_recall_at_k`, but computing average precision per query.
2. Use the same relevance definition:
   - retrieved item is relevant if it shares the same `landmark_id`.
3. Restrict ranking evaluation to top-1000 retrieved items.
4. Report:
   - mAP@1k
   - Recall@1
   - Recall@5
   - Recall@10

### Expected outcome
- Evaluation becomes closer to the MMLandmarks paper.
- Retrieval quality is measured more completely.

---

## Task 3 — Make the Sample4Geo-related hard-negative story true in code

### What to do
Implement the missing parts needed to honestly describe the method as having Sample4Geo-style hard-negative logic.

### Why
Right now the training file/docstring promises more than the visible implementation fully delivers.
That makes the method description stronger than the actual code.

### How
Two possible levels:

#### Option A — minimal honest fix
If hard-negative mining is not implemented soon:
- update comments/docstrings/readme to describe the method honestly as
  **Sample4Geo-inspired shared-encoder InfoNCE baseline**.

#### Option B — actual implementation
Implement:
1. **GPS-based negative selection early in training**
2. **Dynamic Similarity Sampling later in training**

This likely requires:
- precomputing or efficiently accessing coordinates,
- building candidate hard-negative landmark sets,
- refreshing similarity-based neighbors across epochs.

### Expected outcome
- Better alignment between code and method description.
- Potentially better retrieval performance.

---

## Task 4 — Add stronger evaluation logging and versioned experiment outputs

### What to do
Make experiment outputs easier to compare and easier to use later in the report.

### Why
Right now there is enough information to inspect one run, but it is still inconvenient for:
- comparing versions,
- writing the report later,
- handing over to teammates.

### How
For each run, save:
- config file copy,
- final metrics JSON/YAML,
- best metrics JSON/YAML,
- checkpoint path,
- training curves CSV,
- short run summary markdown or text file.

Suggested fields:
- model backbone
- image size
- batch size
- loss settings
- evaluation split logic
- metrics by epoch
- best epoch
- runtime

### Expected outcome
- Easier ablation tracking
- Better report writing later
- Less chance of forgetting what changed between runs

---

## Task 5 — Explore stronger training setups after evaluation is fixed

### What to do
After fixing the protocol, run stronger but controlled experiments.

### Why
There is little value in scaling up if evaluation is still not benchmark-like.
Once evaluation is fixed, stronger runs become more informative.

### How
Possible upgrades:
1. Move from smaller backbone to stronger backbone
2. Increase image resolution if memory allows
3. Tune batch size / LR accordingly
4. Compare shorter-fast vs stronger-slower setups

### Expected outcome
- Stronger baseline
- More informative comparison point for multimodal methods later

---

## Task 6 — Consider whether to stay with single-positive InfoNCE or extend it

### What to do
Decide whether baseline v2 should remain a simple one-positive-per-landmark contrastive baseline, or whether it should explicitly use multiple positives per landmark.

### Why
The current approach is valid and simple, but MMLandmarks naturally supports multiple positives.
This decision matters for how close the method should be to the instance-level nature of the dataset.

### How
Two options:

#### Option A — keep current simple formulation
Keep one sampled ground and one sampled satellite image per landmark per batch.

Use this if the goal is:
- strong baseline simplicity,
- easier debugging,
- clean comparison with Sample4Geo-style training.

#### Option B — extend to multi-positive training
Explore losses or batch logic that allow multiple same-landmark positives without treating them as negatives.

Use this only after the simpler benchmark-aligned baseline is stable.

### Expected outcome
- Clearer scope for v2
- Better decision on whether v2 is a paper-aligned baseline or an instance-tailored extension

---

## Task 7 — Prepare the baseline for fair comparison in the final report

### What to do
Turn baseline v2 into the official comparison point for later approaches.

### Why
The project will likely compare:
- image-image baseline,
- multimodal methods,
- other teammate approaches.

So baseline v2 must be:
- reproducible,
- well-described,
- fairly evaluated.

### How
For the final baseline description, clearly state:
1. **Dataset logic**
   - training pairs are sampled from the same landmark
2. **Loss logic**
   - symmetric InfoNCE
3. **Backbone**
   - shared ConvNeXt
4. **Evaluation logic**
   - query->index retrieval
   - same-landmark relevance
   - metrics used
5. **Limitations**
   - whether hard-negative mining is implemented
   - whether multi-positive training is used

### Expected outcome
- A clean baseline section for the report
- Fairer comparison to teammate methods
- Less confusion later when writing methodology/results

---

# 6. Suggested execution order

Recommended order for baseline v2:

1. **Fix evaluation split logic**
2. **Add mAP@1k**
3. **Make method description honest / or implement hard negatives**
4. **Improve run logging and saved outputs**
5. **Run stronger model variants**
6. **Only then decide on multi-positive extension**

This order gives the fastest path toward a benchmark-comparable and report-ready baseline.

---

# 7. One-paragraph summary

Baseline v1 successfully established a working Sample4Geo-like cross-view retrieval model for MMLandmarks. The model trains, improves over time, and reaches a valid first result. However, v1 still uses a simplified evaluation protocol and incomplete benchmark metrics, so its numbers are best treated as internal baseline indicators rather than final comparable results. Baseline v2 should therefore first fix evaluation to the official query->index retrieval setting, add mAP@1k, clarify or implement the hard-negative strategy, and improve experiment logging. After that, stronger runs and possible multi-positive extensions will be much more meaningful.
