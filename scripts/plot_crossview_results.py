"""Plot cross-view retrieval results.

Generates figures saved to docs/personal/Mateusz/figures/:

  fig1_model_comparison.png        — g2s R@1 per-image, our models vs paper baselines
  fig2_v3_progression.png          — v3 training curve (g2s + s2g R@1 vs epoch)
  fig3_per_landmark_comparison.png — R@1: per-image / max / mean / attn (clean labels)
  fig4_mean_recall_curve.png       — per-lm mean R@1 / R@10 / R@25 grouped bars
  fig5_g2s_vs_s2g.png              — g2s vs s2g R@1 side-by-side
  fig6_aggregation_benefit.png     — aggregation lift: per-image vs lm_img vs lm_mean

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
import numpy as np

# ---------------------------------------------------------------------------
# Data — from newest eval JSONs (2026-05-15), metric set: R@1 / R@10 / R@25
# ---------------------------------------------------------------------------

MODELS = ["Zero-shot", "v2", "v3", "v4"]

# g2s per-image (18,688 queries, paper-comparable)
PER_IMAGE_R1  = {"Zero-shot": 0.34, "v2": 7.21, "v3": 8.58, "v4": 7.63}
PER_IMAGE_R10 = {"Zero-shot": 2.14, "v2": 25.32, "v3": 22.29, "v4": 24.58}
PER_IMAGE_R25 = {"Zero-shot": 3.88, "v2": 34.35, "v3": 27.47, "v4": 33.13}

# Legacy alias
PER_IMAGE = PER_IMAGE_R1

# g2s per-landmark max-agg ("any photo wins")
PER_LM_MAX_R1  = {"Zero-shot": 0.30, "v2": 9.00, "v3": 7.10, "v4": 8.10}
PER_LM_MAX_R10 = {"Zero-shot": 2.10, "v2": 27.20, "v3": 18.80, "v4": 25.30}
PER_LM_MAX_R25 = {"Zero-shot": 3.30, "v2": 37.00, "v3": 25.40, "v4": 33.40}

PER_LM_MAX   = PER_LM_MAX_R1
PER_LANDMARK = PER_LM_MAX_R1  # backward compat

# g2s per-landmark mean-agg (team primary; equiv. embedding-space mean pool)
PER_LM_MEAN_R1  = {"Zero-shot": 0.40, "v2": 17.60, "v3": 18.40, "v4": 18.50}
PER_LM_MEAN_R10 = {"Zero-shot": 1.80, "v2": 42.20, "v3": 37.20, "v4": 39.60}
PER_LM_MEAN_R25 = {"Zero-shot": 3.50, "v2": 51.50, "v3": 43.60, "v4": 50.10}

PER_LM_MEAN = PER_LM_MEAN_R1

# g2s per-landmark attn-weighted mean
PER_LM_ATTN_R1  = {"Zero-shot": 0.30, "v2": 17.50, "v3": 18.20, "v4": 18.20}
PER_LM_ATTN_R10 = {"Zero-shot": 1.80, "v2": 42.10, "v3": 37.20, "v4": 39.20}
PER_LM_ATTN_R25 = {"Zero-shot": 3.50, "v2": 51.00, "v3": 43.30, "v4": 49.60}

PER_LM_ATTN = PER_LM_ATTN_R1

# g2s per-landmark per-image (lm_img) — per-landmark average of per-image hit rates
# Fairer than global per-image (de-biases high-photo-count landmarks), no aggregation benefit
PER_LM_IMG_R1  = {"Zero-shot": 0.16, "v2": 6.45, "v3": 6.95, "v4": 6.71}
PER_LM_IMG_R10 = {"Zero-shot": 1.25, "v2": 22.29, "v3": 17.45, "v4": 20.53}
PER_LM_IMG_R25 = {"Zero-shot": 2.36, "v2": 30.92, "v3": 22.29, "v4": 28.26}

# s2g per-image (= per-landmark for s2g — one satellite per landmark)
S2G_R1  = {"Zero-shot": 0.00, "v2": 5.30, "v3": 5.50, "v4": 4.00}
S2G_R10 = {"Zero-shot": 0.80, "v2": 18.70, "v3": 13.70, "v4": 16.80}
S2G_R25 = {"Zero-shot": 1.60, "v2": 28.10, "v3": 20.50, "v4": 25.50}

# Paper zero-shot baselines (CLIP-based, never trained on MMLandmarks)
PAPER_BASELINES = {
    "MMCLIP†": 20.5,
    "GeoClip†": 21.1,
}

# v3 training curve (per-image g2s / s2g R@1 at eval checkpoints)
V3_EPOCHS = [9, 18, 27, 36]
V3_G2S    = [6.61, 8.32, 8.42, 8.58]
V3_S2G    = [4.30, 5.00, 5.50, 5.40]

V2_BASELINE = 7.21  # reference line on training-progression figure

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------

C_BLUE   = "#4C72B0"
C_ORANGE = "#DD8452"
C_GREEN  = "#55A868"
C_PURPLE = "#8172B3"
C_RED    = "#C44E52"
C_BROWN  = "#937860"


# ---------------------------------------------------------------------------
# Figure 1 — Model comparison bar chart (g2s R@1 per-image)
# ---------------------------------------------------------------------------

def fig1_comparison(out_dir: Path) -> None:
    our_models = MODELS
    our_vals   = [PER_IMAGE_R1[m] for m in our_models]
    paper_models = list(PAPER_BASELINES.keys())
    paper_vals   = list(PAPER_BASELINES.values())

    fig, ax = plt.subplots(figsize=(10, 5))

    x_our   = np.arange(len(our_models))
    x_paper = np.arange(len(our_models) + 1.5, len(our_models) + 1.5 + len(paper_models))

    bar_our   = ax.bar(x_our,   our_vals,   color=C_BLUE,   edgecolor="white", width=0.7, label="Trained on MMLandmarks")
    bar_paper = ax.bar(x_paper, paper_vals, color=C_ORANGE, edgecolor="white", width=0.7, label="Zero-shot, no MML training†")

    for bar, val in zip(bar_our, our_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.2, f"{val:.2f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color=C_BLUE)
    for bar, val in zip(bar_paper, paper_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.2, f"{val:.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color=C_ORANGE)

    all_x      = list(x_our) + list(x_paper)
    all_labels = our_models + paper_models
    ax.set_xticks(all_x)
    ax.set_xticklabels(all_labels, fontsize=11)

    mid_our   = (x_our[0]   + x_our[-1])   / 2
    mid_paper = (x_paper[0] + x_paper[-1]) / 2
    ax.annotate("MMLandmarks trained", xy=(mid_our, -3.5), fontsize=9, ha="center",
                xycoords=("data", "axes fraction"), annotation_clip=False, color=C_BLUE)
    ax.annotate("Zero-shot (CLIP-based)†", xy=(mid_paper, -3.5), fontsize=9, ha="center",
                xycoords=("data", "axes fraction"), annotation_clip=False, color=C_ORANGE)

    ax.axvline(len(our_models) - 0.25, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_ylabel("R@1 (%)", fontsize=12)
    ax.set_title("Cross-View Retrieval — g2s R@1 (per-image, unpooled)", fontsize=13, pad=12)
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

    ax.plot(V3_EPOCHS, V3_G2S, "o-",  color=C_BLUE,  linewidth=2, markersize=7, label="g2s R@1")
    ax.plot(V3_EPOCHS, V3_S2G, "s--", color=C_GREEN, linewidth=2, markersize=7, label="s2g R@1")

    for ep, g, s in zip(V3_EPOCHS, V3_G2S, V3_S2G):
        ax.annotate(f"{g:.2f}%", (ep, g), textcoords="offset points", xytext=(5, 5),
                    fontsize=8, color=C_BLUE)
        ax.annotate(f"{s:.2f}%", (ep, s), textcoords="offset points", xytext=(5, -12),
                    fontsize=8, color=C_GREEN)

    ax.axhline(V2_BASELINE, color=C_RED, linestyle=":", linewidth=1.5, alpha=0.8)
    ax.text(V3_EPOCHS[-1] + 0.3, V2_BASELINE + 0.1, f"v2 baseline ({V2_BASELINE}%)",
            color=C_RED, fontsize=8, va="bottom")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("R@1 (%)", fontsize=12)
    ax.set_title("v3 Training Progression — g2s and s2g R@1", fontsize=13, pad=10)
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
# Figure 3 — Per-image vs Per-landmark R@1 (clean labels, no parentheticals)
# ---------------------------------------------------------------------------

def fig3_per_landmark(out_dir: Path) -> None:
    models    = MODELS
    img_vals  = [PER_IMAGE_R1[m]   for m in models]
    max_vals  = [PER_LM_MAX_R1[m]  for m in models]
    mean_vals = [PER_LM_MEAN_R1[m] for m in models]
    attn_vals = [PER_LM_ATTN_R1[m] for m in models]

    n_groups = 4
    width    = 0.8 / n_groups
    offsets  = np.linspace(-(n_groups - 1) / 2, (n_groups - 1) / 2, n_groups) * width
    x = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(10, 5))

    bars1 = ax.bar(x + offsets[0], img_vals,  width, label="Per-image",   color=C_BLUE,   edgecolor="white")
    bars2 = ax.bar(x + offsets[1], max_vals,  width, label="Per-lm max",  color=C_ORANGE, edgecolor="white")
    bars3 = ax.bar(x + offsets[2], mean_vals, width, label="Per-lm mean", color=C_GREEN,  edgecolor="white")
    bars4 = ax.bar(x + offsets[3], attn_vals, width, label="Per-lm attn", color=C_PURPLE, edgecolor="white")

    all_bars = list(bars1) + list(bars2) + list(bars3) + list(bars4)
    all_vals = img_vals + max_vals + mean_vals + attn_vals

    for bar, val in zip(all_bars, all_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.3, f"{val:.1f}%",
                ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("g2s R@1 (%)", fontsize=12)
    ax.set_title("Per-Image vs Per-Landmark R@1 (g2s) — All Eval Protocols", fontsize=12, pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)

    fig.subplots_adjust(bottom=0.12)
    out_path = out_dir / "fig3_per_landmark_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 4 — Per-landmark mean: R@1 / R@10 / R@25 grouped bars
# ---------------------------------------------------------------------------

def fig4_mean_recall_curve(out_dir: Path) -> None:
    """Full recall depth for the team's primary metric."""
    models   = MODELS
    r1_vals  = [PER_LM_MEAN_R1[m]  for m in models]
    r10_vals = [PER_LM_MEAN_R10[m] for m in models]
    r25_vals = [PER_LM_MEAN_R25[m] for m in models]

    n_groups = 3
    width    = 0.8 / n_groups
    offsets  = np.linspace(-(n_groups - 1) / 2, (n_groups - 1) / 2, n_groups) * width
    x = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(10, 5))

    bars1 = ax.bar(x + offsets[0], r1_vals,  width, label="R@1",  color=C_BLUE, edgecolor="white", alpha=1.0)
    bars2 = ax.bar(x + offsets[1], r10_vals, width, label="R@10", color=C_BLUE, edgecolor="white", alpha=0.65)
    bars3 = ax.bar(x + offsets[2], r25_vals, width, label="R@25", color=C_BLUE, edgecolor="white", alpha=0.40)

    for bar, val in zip(list(bars1) + list(bars2) + list(bars3),
                        r1_vals + r10_vals + r25_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5, f"{val:.1f}%",
                ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("g2s Recall (%)", fontsize=12)
    ax.set_title("Per-Landmark Mean-Agg Recall — R@1 / R@10 / R@25 (team primary metric)", fontsize=12, pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=10)

    fig.subplots_adjust(bottom=0.12)
    out_path = out_dir / "fig4_mean_recall_curve.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 5 — g2s vs s2g R@1 side-by-side
# ---------------------------------------------------------------------------

def fig5_g2s_vs_s2g(out_dir: Path) -> None:
    """Compares both retrieval directions (per-image R@1)."""
    models   = MODELS
    g2s_vals = [PER_IMAGE_R1[m] for m in models]
    s2g_vals = [S2G_R1[m]       for m in models]

    width = 0.35
    x = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(9, 5))

    bars_g = ax.bar(x - width / 2, g2s_vals, width, label="g2s (ground → satellite)", color=C_BLUE,  edgecolor="white")
    bars_s = ax.bar(x + width / 2, s2g_vals, width, label="s2g (satellite → ground)", color=C_GREEN, edgecolor="white")

    for bar, val in zip(list(bars_g) + list(bars_s), g2s_vals + s2g_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.1, f"{val:.2f}%",
                ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("R@1 (%)", fontsize=12)
    ax.set_title("Retrieval Direction Comparison — g2s vs s2g R@1 (per-image)", fontsize=12, pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=10)
    ax.set_ylim(0, max(g2s_vals) * 1.25)

    plt.tight_layout()
    out_path = out_dir / "fig5_g2s_vs_s2g.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 6 — Aggregation benefit: per-image vs lm_img vs lm_mean
# ---------------------------------------------------------------------------

def fig6_aggregation_benefit(out_dir: Path) -> None:
    """Shows the benefit of aggregating K ground images per landmark.

    per-image (global)     — biased toward high-photo-count landmarks
    per-lm per-image       — de-biased (lm_img: avg per-image hit rate per landmark)
    per-lm mean-agg        — aggregation of K embeddings → big lift (~2.5×)
    """
    models     = MODELS
    img_vals   = [PER_IMAGE_R1[m]   for m in models]
    lmimg_vals = [PER_LM_IMG_R1[m]  for m in models]
    mean_vals  = [PER_LM_MEAN_R1[m] for m in models]

    n_groups = 3
    width    = 0.8 / n_groups
    offsets  = np.linspace(-(n_groups - 1) / 2, (n_groups - 1) / 2, n_groups) * width
    x = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(10, 5))

    bars1 = ax.bar(x + offsets[0], img_vals,   width, label="Per-image (global)",      color=C_ORANGE, edgecolor="white")
    bars2 = ax.bar(x + offsets[1], lmimg_vals, width, label="Per-lm per-image (fair)", color=C_BROWN,  edgecolor="white")
    bars3 = ax.bar(x + offsets[2], mean_vals,  width, label="Per-lm mean-agg",         color=C_GREEN,  edgecolor="white")

    for bar, val in zip(list(bars1) + list(bars2) + list(bars3),
                        img_vals + lmimg_vals + mean_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.3, f"{val:.1f}%",
                ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("g2s R@1 (%)", fontsize=12)
    ax.set_title("Aggregation Benefit — Per-Image vs Per-Lm Per-Image vs Per-Lm Mean", fontsize=12, pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)

    fig.subplots_adjust(bottom=0.12)
    out_path = out_dir / "fig6_aggregation_benefit.png"
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
    fig4_mean_recall_curve(out_dir)
    fig5_g2s_vs_s2g(out_dir)
    fig6_aggregation_benefit(out_dir)

    print(f"\nAll figures saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
