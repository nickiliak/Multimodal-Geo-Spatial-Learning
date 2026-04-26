"""Standalone evaluation script for cross-view retrieval checkpoints.

Loads a saved checkpoint (best.pt / last.pt) and runs the full benchmark
evaluation without any training. Useful for:
- Getting paper-comparable (unpooled) numbers after a training run
- Comparing pooled vs unpooled metrics on the same checkpoint
- Quick re-evaluation after changing eval protocol

Usage
-----
# Paper-comparable protocol (18,689 individual queries, no pooling):
python -m mmgeo.crossview.eval \\
    --config configs/crossview_convnext_base.yaml \\
    --checkpoint checkpoints/crossview/cv_v2_base_20260422_230539/best.pt \\
    --no-pool

# Pooled protocol (1,000 landmark embeddings, mean of ~18 imgs each):
python -m mmgeo.crossview.eval \\
    --config configs/crossview_convnext_base.yaml \\
    --checkpoint checkpoints/crossview/cv_v2_base_20260422_230539/best.pt \\
    --pool
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from mmgeo.crossview.dataset import (
    MMLImageDataset,
    get_eval_transforms,
)
from mmgeo.crossview.model import CrossViewModel
from mmgeo.crossview.train import _run_eval
from torch.utils.data import ConcatDataset, DataLoader


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a cross-view retrieval checkpoint"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config used during training",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to checkpoint file (best.pt or last.pt)",
    )
    pool_group = parser.add_mutually_exclusive_group()
    pool_group.add_argument(
        "--pool", dest="pool_queries", action="store_true", default=False,
        help="Pool all ground images per landmark before retrieval "
             "(1,000 landmark queries). Gives higher numbers but differs "
             "from the MMLandmarks paper protocol.",
    )
    pool_group.add_argument(
        "--no-pool", dest="pool_queries", action="store_false",
        help="Use each ground image as a separate query (18,689 queries). "
             "Directly comparable to Table 2 of the MMLandmarks paper. "
             "[default]",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Optional path to save results as JSON (e.g. eval_results.json)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    # Load model
    backbone = cfg["model"].get("backbone", "convnext_tiny.fb_in22k")
    model = CrossViewModel(
        backbone=backbone,
        pretrained=False,   # weights come from checkpoint
        embed_dim=cfg["model"].get("embed_dim", 0),
    )
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} ({args.checkpoint})")
    print(f"Pool queries: {args.pool_queries}")
    print(f"{'='*60}")

    data_root = Path(cfg["data"]["root"])
    img_size = cfg["training"].get("img_size", 224)

    results = _run_eval(
        model=model,
        data_root=data_root,
        img_size=img_size,
        device=device,
        cfg=cfg,
        pool_queries=args.pool_queries,
    )

    print(f"\n{'='*60}")
    print(f"SUMMARY  |  pool_queries={args.pool_queries}  |  epoch={ckpt['epoch']}")
    print(f"{'='*60}")
    for direction, metrics in results.items():
        print(f"\n  {direction.upper()}:")
        for k, v in metrics.items():
            print(f"    {k}: {v*100:.2f}%")

    if args.output:
        out = {
            "checkpoint": args.checkpoint,
            "epoch": ckpt["epoch"],
            "pool_queries": args.pool_queries,
            "results": results,
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
