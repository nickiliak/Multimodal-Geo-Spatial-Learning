"""CLI entry point for the hybrid GeoCLIP + Sample4Geo pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mmgeo.inference import (
    DEFAULT_RADII_KM,
    full_sweep,
)


def _parse_radii(raw: str) -> list[float]:
    out: list[float] = []
    for tok in raw.split(","):
        tok = tok.strip().lower()
        if tok in ("inf", "infinity"):
            out.append(float("inf"))
        else:
            out.append(float(tok))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid GeoCLIP + Sample4Geo inference sweep")
    parser.add_argument("--data-root", type=Path, default=Path("data/MML_Data"))
    parser.add_argument("--ckpt", type=Path, default=Path("models/best1.pt"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/hybrid_cache"))
    parser.add_argument("--output", type=Path, default=Path("outputs/hybrid/results.json"))
    parser.add_argument(
        "--radii-km", type=_parse_radii,
        default=DEFAULT_RADII_KM,
        help="Comma-separated radii in km. Use 'inf' for no narrowing.",
    )
    parser.add_argument(
        "--index-modes", nargs="+", choices=["query", "full"],
        default=["query", "full"],
    )
    parser.add_argument(
        "--query-modes", nargs="+", choices=["one_per_landmark", "all"],
        default=["one_per_landmark", "all"],
    )
    parser.add_argument(
        "--fallbacks", nargs="+", choices=["fail", "fallback_full"],
        default=["fail", "fallback_full"],
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[run_hybrid_inference] device={device}")
    print(f"[run_hybrid_inference] radii_km={args.radii_km}")
    print(f"[run_hybrid_inference] index_modes={args.index_modes}")
    print(f"[run_hybrid_inference] query_modes={args.query_modes}")
    print(f"[run_hybrid_inference] fallbacks={args.fallbacks}")

    full_sweep(
        data_root=args.data_root,
        sample4geo_ckpt=args.ckpt,
        device=device,
        cache_dir=args.cache_dir,
        output_json=args.output,
        radii_km=args.radii_km,
        index_modes=tuple(args.index_modes),
        query_modes=tuple(args.query_modes),
        fallback_policies=tuple(args.fallbacks),
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
