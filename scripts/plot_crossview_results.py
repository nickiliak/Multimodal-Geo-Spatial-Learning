"""Plot cross-view retrieval results.

Generates three figures saved to docs/personal/Mateusz/figures/:

Figure 1 — Model comparison bar chart (g2s Recall@1, per-image)
  Two groups: MMLandmarks-trained models vs. zero-shot CLIP-based models.
  Annotated to flag the different training conditions.

Figure 2 — v3 training progression (g2s and s2g Recall@1 vs epoch)
  4 eval checkpoints: epoch 9, 18, 27, 36.
  Dashed reference line at v2 unpooled R@1 = 7.21%.

Figure 3 — Per-image vs Per-landmark comparison (all three: per-image, max, mean)
  Shows the reversal: v3 wins per-image but v2 wins max-agg;
  v4 ≈ v3 on mean-agg (team primary metric).

Usage
-----
python scripts/plot_crossview_results.py

# Save to a different output directory:
python scripts/plot_crossview_results.py --out-dir path/to/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Data (update per-landmark fields once eval jobs complete)
# ---------------------------------------------------------------------------

# Per-image Recall@1 (unpooled, g2s)
PER_IMAGE = {
    "Zero-shot": 0.34,
    "v2 (ep30)": 7.21,
    "v3 (ep36)": 8.58,
    "v4 (ep36)": 7.63,
}

# Per-landmark max-agg Recall@1 (g2s) — "any photo wins"
PER_LM_MAX = {
    "Zero-shot": 0.30,
    "v2 (ep30)": 9.00,
    "v3 (ep36)": 7.10,
    "v4 (ep36)": 8.10,
}

# Per-landmark mean-agg Recall@1 (g2s) — team primary metric (= embedding-space mean pool)
PER_LM_MEAN = {
    "Zero-shot": 0.40,
    "v2 (ep30)": 17.60,
    "v3 (ep36)": 18.40,
    "v4 (ep36)": 18.50,
}

# Per-landmark attention-weighted mean Recall@1 (g2s)
# Finding: attn < mean for all models — simple mean wins with K≈18 diverse photos
PER_LM_ATTN = {
    "Zero-shot": 0.30,
    "v2 (ep30)": 17.50,
    "v3 (ep36)": 18.20,
    "v4 (ep36)": 18.20,
}

# Keep legacy alias for backward compat
PER_LANDMARK = PER_LM_MAX

# Paper zero-shot baselines (CLIP-based, never trained on MMLandmarks)
PAPER_BASELINES = {
    "MMCLIP†": 20.5,
    "GeoClip†": 21.1,
}

# v3 training curve (from eval_curves.csv)
V3_EPOCHS = [9, 18, 27, 36]
V3_G2S = [6.61, 8.32, 8.42, 8.58]
V3_S2G = [4.30, 5.00, 5.50, 5.40]

# Reference line: v2 unpooled per-image R@1
V2_BASELINE = 7.21

# ---------------------------------------------------------------------------
# Figure 1 — Comparison bar chart
# ---------------------------------------------------------------------------

def fig1_comparison(out_dir: Path) -> None:
    our_models = [k for k, v in PER_IMAGE.items() if v is not None]
    our_vals = [PER_IMAGE[k] for k in our_models]
    paper_models = list(PAPER_BASELINES.keys())
    paper_vals = list(PAPER_BASELINES.values())

    fig, ax = plt.subplots(figsize=(10, 5))

    x_our = np.arange(len(our_models))
    x_paper = np.arange(len(our_models) + 1.5, len(our_models) + 1.5 + len(paper_models))

    bar_our = ax.bar(x_our, our_vals, color="#4C72B0", edgecolor="white", width=0.7, label="Trained on MMLandmarks")
    bar_paper = ax.bar(x_paper, paper_vals, color="#DD8452", edgecolor="white", width=0.7, label="Zero-shot, no MML training†")

    # Value labels on bars
    for bar, val in zip(bar_our, our_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.2, f"{val:.2f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color="#4C72B0")
    for bar, val in zip(bar_paper, paper_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.2, f"{val:.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color="#DD8452")

    # Group labels below x-axis
    all_x = list(x_our) + list(x_paper)
    all_labels = our_models + paper_models
    ax.set_xticks(all_x)
    ax.set_xticklabels(all_labels, fontsize=11)

    # Group brackets
    mid_our = (x_our[0] + x_our[-1]) / 2
    mid_paper = (x_paper[0] + x_paper[-1]) / 2
    ax.annotate("MMLandmarks trained", xy=(mid_our, -3.5), fontsize=9, ha="center",
                xycoords=("data", "axes fraction"), annotation_clip=False, color="#4C72B0")
    ax.annotate("Zero-shot (CLIP-based)†", xy=(mid_paper, -3.5), fontsize=9, ha="center",
                xycoords=("data", "axes fraction"), annotation_clip=False, color="#DD8452")

    # Dashed separator between groups
    ax.axvline(len(our_models) - 0.25, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_ylabel("Recall@1 (%)", fontsize=12)
    ax.set_title("Cross-View Retrieval — g2s Recall@1 (per-image, unpooled)", fontsize=13, pad=12)
    ax.set_ylim(0, max(paper_vals) * 1.15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    fig.text(0.99, 0.01,
             "† Zero-shot models never trained on MMLandmarks — different experimental condition.",
             ha="right", va="bottom", fontsize=7, color="gray", style="italic")

    fig.subplots_adjust(bottom=0.18)
    out_path = out_dir / "fig1_model_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 2 — v3 training progression
# ---------------------------------------------------------------------------

def fig2_v3_progression(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(V3_EPOCHS, V3_G2S, "o-", color="#4C72B0", linewidth=2, markersize=7, label="g2s R@1")
    ax.plot(V3_EPOCHS, V3_S2G, "s--", color="#55A868", linewidth=2, markersize=7, label="s2g R@1")

    # Annotate each point
    for ep, g, s in zip(V3_EPOCHS, V3_G2S, V3_S2G):
        ax.annotate(f"{g:.2f}%", (ep, g), textcoords="offset points", xytext=(5, 5),
                    fontsize=8, color="#4C72B0")
        ax.annotate(f"{s:.2f}%", (ep, s), textcoords="offset points", xytext=(5, -12),
                    fontsize=8, color="#55A868")

    # v2 reference line
    ax.axhline(V2_BASELINE, color="#C44E52", linestyle=":", linewidth=1.5, alpha=0.8)
    ax.text(V3_EPOCHS[-1] + 0.3, V2_BASELINE + 0.1, f"v2 baseline ({V2_BASELINE}%)",
            color="#C44E52", fontsize=8, va="bottom")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Recall@1 (%)", fontsize=12)
    ax.set_title("v3 Training Progression — g2s and s2g Recall@1", fontsize=13, pad=10)
    ax.set_xticks(V3_EPOCHS)
    ax.set_ylim(0, max(V3_G2S) * 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=10)

    plt.tight_layout()
    out_path = out_dir / "fig2_v3_progression.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 3 — Per-image vs Per-landmark max vs Per-landmark mean
# ---------------------------------------------------------------------------

def fig3_per_landmark(out_dir: Path) -> None:
    """Bar chart: per-image / per-lm max / per-lm mean / per-lm attn Recall@1 (g2s).

    Highlights the reversal: v3 leads per-image; v2 leads max-agg;
    v4 ≈ v3 leads mean-agg (team primary). Attn bar shown when available.
    """
    models = list(PER_IMAGE.keys())
    img_vals  = [PER_IMAGE[m]   for m in models]
    max_vals  = [PER_LM_MAX[m]  for m in models]
    mean_vals = [PER_LM_MEAN[m] for m in models]

    has_attn = all(PER_LM_ATTN.get(m) is not None for m in models)
    attn_vals = [PER_LM_ATTN[m] for m in models] if has_attn else None

    n_groups = 4 if has_attn else 3
    width = 0.8 / n_groups
    offsets = np.linspace(-(n_groups - 1) / 2, (n_groups - 1) / 2, n_groups) * width

    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(11, 5))

    bars1 = ax.bar(x + offsets[0], img_vals,  width, label="Per-image (18,688 queries)",          color="#4C72B0", edgecolor="white")
    bars2 = ax.bar(x + offsets[1], max_vals,  width, label="Per-lm max (any photo wins)",          color="#DD8452", edgecolor="white")
    bars3 = ax.bar(x + offsets[2], mean_vals, width, label="Per-lm mean (team primary)",           color="#55A868", edgecolor="white")
    all_bars = list(bars1) + list(bars2) + list(bars3)
    all_vals = img_vals + max_vals + mean_vals

    if has_attn:
        bars4 = ax.bar(x + offsets[3], attn_vals, width, label="Per-lm attn-weighted mean",       color="#8172B3", edgecolor="white")
        all_bars += list(bars4)
        all_vals += attn_vals

    for bar, val in zip(all_bars, all_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.3, f"{val:.1f}%",
                ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("g2s Recall@1 (%)", fontsize=12)
    title = "Per-Image vs Per-Landmark Recall@1 (g2s) — All Eval Protocols"
    if not has_attn:
        title += " (attn pending HPC)"
    ax.set_title(title, fontsize=12, pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)

    fig.subplots_adjust(bottom=0.12)
    out_path = out_dir / "fig3_per_landmark_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot cross-view retrieval results")
    parser.add_argument(
        "--out-dir", default="docs/personal/Mateusz/figures",
        help="Directory to save figures (default: docs/personal/Mateusz/figures)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig1_comparison(out_dir)
    fig2_v3_progression(out_dir)
    fig3_per_landmark(out_dir)

    print(f"\nAll figures saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
