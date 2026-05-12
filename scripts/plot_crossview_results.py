"""Plot cross-view retrieval results.

Generates two figures saved to docs/personal/Mateusz/figures/:

Figure 1 — Model comparison bar chart (g2s Recall@1, per-image)
  Two groups: MMLandmarks-trained models vs. zero-shot CLIP-based models.
  Annotated to flag the different training conditions.

Figure 2 — v3 training progression (g2s and s2g Recall@1 vs epoch)
  4 eval checkpoints: epoch 9, 18, 27, 36.
  Dashed reference line at v2 unpooled R@1 = 7.21%.

Usage
-----
# After running eval jobs, update PER_LANDMARK dict below and re-run:
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
    "Zero-shot": 0.25,
    "v2": 7.21,
    "v3": 8.58,
    "v4": None,       # fill in after v4 training + eval
}

# Per-landmark Recall@1 (max-agg, g2s) — update after eval jobs
PER_LANDMARK = {
    "Zero-shot": None,   # fill from eval_results_zeroshot.json
    "v2":        None,   # fill from eval_results_v2.json
    "v3":        None,   # fill from eval_results_v3.json
    "v4":        None,   # fill after v4 training + eval
}

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

    plt.tight_layout()
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
# Figure 3 — Per-image vs Per-landmark comparison (fill in once eval runs)
# ---------------------------------------------------------------------------

def fig3_per_landmark(out_dir: Path) -> None:
    """Bar chart comparing per-image vs per-landmark Recall@1.

    Only rendered once per-landmark numbers are available; skipped otherwise.
    """
    models_with_both = [k for k in PER_IMAGE if PER_IMAGE.get(k) is not None and PER_LANDMARK.get(k) is not None]
    if not models_with_both:
        print("Skipping Figure 3: per-landmark eval results not yet available. "
              "Update PER_LANDMARK dict and re-run.")
        return

    img_vals = [PER_IMAGE[m] for m in models_with_both]
    lm_vals = [PER_LANDMARK[m] for m in models_with_both]

    x = np.arange(len(models_with_both))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))

    bars1 = ax.bar(x - width / 2, img_vals, width, label="Per-image (18,688 queries)", color="#4C72B0", edgecolor="white")
    bars2 = ax.bar(x + width / 2, lm_vals, width, label="Per-landmark (1,000 landmarks, max-agg)", color="#DD8452", edgecolor="white")

    for bar, val in zip(list(bars1) + list(bars2), img_vals + lm_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.3, f"{val:.1f}%",
                ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(models_with_both, fontsize=11)
    ax.set_ylabel("g2s Recall@1 (%)", fontsize=12)
    ax.set_title("Per-Image vs Per-Landmark Evaluation (g2s R@1)", fontsize=13, pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)

    plt.tight_layout()
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
    print("To update per-landmark numbers: edit PER_LANDMARK dict in this script and re-run.")


if __name__ == "__main__":
    main()
