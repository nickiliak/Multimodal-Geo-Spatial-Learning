"""Experiment logging for the cross-view baseline.

Each training run gets its own versioned directory containing:

- ``config.yaml``         — exact config used (copy of the resolved dict)
- ``train_curves.csv``    — per-epoch training stats
- ``eval_curves.csv``     — per-epoch evaluation metrics (flattened across directions)
- ``best.pt`` / ``last.pt`` — checkpoints (written by the training loop)
- ``best_metrics.json``   — best eval metrics + epoch (selection = g2s recall@1)
- ``final_metrics.json``  — last eval metrics
- ``summary.md``          — human-readable run summary for the report

A run dir is created as ``<root_dir>/<timestamp>[_<tag>]``.
"""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def _flatten_eval(eval_results: dict[str, dict[str, float]]) -> dict[str, float]:
    """{'g2s': {'recall@1': x, ...}} -> {'g2s_recall@1': x, ...}."""
    flat: dict[str, float] = {}
    for direction, metrics in eval_results.items():
        for name, value in metrics.items():
            flat[f"{direction}_{name}"] = float(value)
    return flat


class RunLogger:
    """Writes config, per-epoch curves, best/final metrics, and a summary.md.

    Parameters
    ----------
    root_dir : Path
        Parent dir that will contain the versioned run dir.
    cfg : dict
        Resolved config dict to snapshot into the run dir.
    run_prefix : str | None
        Optional prefix prepended to the dir name (e.g. ``"cv_v2_"`` to
        distinguish model families). Applied before the timestamp.
    run_tag : str | None
        Optional suffix appended to the timestamped dir name.
    selection_metric : str
        Flat eval metric key used to pick the "best" epoch
        (default: ``"g2s_recall@1"``).
    """

    def __init__(
        self,
        root_dir: Path,
        cfg: dict,
        run_prefix: str | None = None,
        run_tag: str | None = None,
        selection_metric: str = "g2s_recall@1",
    ) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = ts if not run_tag else f"{ts}_{run_tag}"
        if run_prefix:
            name = f"{run_prefix}{name}"
        self.run_dir = Path(root_dir) / name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.cfg = cfg
        self.selection_metric = selection_metric
        self._start_time = time.time()

        with open(self.run_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        self.train_csv = self.run_dir / "train_curves.csv"
        self.eval_csv = self.run_dir / "eval_curves.csv"
        self._train_header: list[str] | None = None
        self._eval_header: list[str] | None = None

        self.best: dict[str, Any] = {
            "epoch": None,
            "score": float("-inf"),
            "metrics": None,
            "ckpt_path": None,
        }
        self.last_eval: dict[str, dict[str, float]] | None = None
        self.last_epoch: int | None = None

        print(f"[RunLogger] run dir: {self.run_dir}")

    # ------------------------------------------------------------------ paths
    def ckpt_path(self, name: str) -> Path:
        return self.run_dir / name

    # ------------------------------------------------------------------ train
    def log_train(self, epoch: int, metrics: dict[str, float], extra: dict[str, Any] | None = None) -> None:
        row: dict[str, Any] = {"epoch": epoch}
        row.update({k: float(v) for k, v in metrics.items()})
        if extra:
            row.update(extra)
        self._append_row(self.train_csv, row, "_train_header")

    # ------------------------------------------------------------------ eval
    def log_eval(self, epoch: int, eval_results: dict[str, dict[str, float]]) -> None:
        flat = _flatten_eval(eval_results)
        row: dict[str, Any] = {"epoch": epoch, **flat}
        self._append_row(self.eval_csv, row, "_eval_header")

        self.last_eval = eval_results
        self.last_epoch = epoch

        score = flat.get(self.selection_metric)
        if score is not None and score > self.best["score"]:
            self.best = {
                "epoch": epoch,
                "score": float(score),
                "metrics": eval_results,
                "ckpt_path": str(self.ckpt_path("best.pt")),
            }

    # ------------------------------------------------------------------ finalize
    def finalize(self) -> None:
        runtime = time.time() - self._start_time

        if self.last_eval is not None:
            with open(self.run_dir / "final_metrics.json", "w", encoding="utf-8") as f:
                json.dump(
                    {"epoch": self.last_epoch, "metrics": self.last_eval},
                    f, indent=2,
                )

        if self.best["epoch"] is not None:
            with open(self.run_dir / "best_metrics.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "epoch": self.best["epoch"],
                        "selection_metric": self.selection_metric,
                        "score": self.best["score"],
                        "metrics": self.best["metrics"],
                        "ckpt_path": self.best["ckpt_path"],
                    },
                    f, indent=2,
                )

        with open(self.run_dir / "summary.md", "w", encoding="utf-8") as f:
            f.write(self._render_summary(runtime))

        print(f"[RunLogger] finalized run: {self.run_dir}")

    # ------------------------------------------------------------------ helpers
    def _append_row(self, path: Path, row: dict[str, Any], header_attr: str) -> None:
        header = getattr(self, header_attr)
        write_header = header is None or not path.exists()
        if header is None:
            header = list(row.keys())
            setattr(self, header_attr, header)
        else:
            # Extend header if new keys appear (shouldn't normally happen)
            new_keys = [k for k in row.keys() if k not in header]
            if new_keys:
                header.extend(new_keys)
                # Rewrite file with new header — rare path
                if path.exists():
                    existing = list(csv.DictReader(path.open()))
                    with path.open("w", newline="", encoding="utf-8") as f:
                        w = csv.DictWriter(f, fieldnames=header)
                        w.writeheader()
                        for r in existing:
                            w.writerow(r)
                    write_header = False

        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header)
            if write_header:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in header})

    def _render_summary(self, runtime_sec: float) -> str:
        cfg = self.cfg
        model_cfg = cfg.get("model", {})
        train_cfg = cfg.get("training", {})
        eval_cfg = cfg.get("evaluation", {})
        hn_cfg = cfg.get("hard_negatives", {}) or {}

        lines = [
            f"# Run Summary — {self.run_dir.name}",
            "",
            "## Config",
            f"- **Backbone:** `{model_cfg.get('backbone', 'n/a')}`",
            f"- **Embed dim:** {model_cfg.get('embed_dim', 'native')}",
            f"- **Image size:** {train_cfg.get('img_size', 'n/a')}",
            f"- **Batch size:** {train_cfg.get('batch_size', 'n/a')}",
            f"- **Epochs:** {train_cfg.get('epochs', 'n/a')}",
            f"- **LR:** {train_cfg.get('lr', 'n/a')} (weight_decay={train_cfg.get('weight_decay', 'n/a')})",
            f"- **Temperature:** init={train_cfg.get('temperature', 'n/a')}, learnable={train_cfg.get('learnable_temp', True)}",
            f"- **Label smoothing:** {train_cfg.get('label_smoothing', 0.0)}",
            f"- **Hard negatives:** enabled={hn_cfg.get('enabled', False)}"
            + (
                f", gps_epochs={hn_cfg.get('gps_epochs')}, pool_size={hn_cfg.get('pool_size')}, "
                f"dss_refresh_every={hn_cfg.get('dss_refresh_every')}"
                if hn_cfg.get("enabled") else ""
            ),
            f"- **Eval directions:** {eval_cfg.get('directions', ['g2s', 's2g'])}",
            f"- **Eval metrics:** recall_ks={eval_cfg.get('recall_ks', [1,5,10])}, map_k={eval_cfg.get('map_k', 1000)}",
            "",
            "## Runtime",
            f"- Total wall time: **{runtime_sec/60:.1f} min** ({runtime_sec:.0f} s)",
        ]

        if self.best["epoch"] is not None:
            best = self.best
            lines += [
                "",
                "## Best Epoch",
                f"- Epoch: **{best['epoch']}**  (selected on `{self.selection_metric}`)",
                f"- Score: **{best['score']:.4f}**",
            ]
            lines += self._render_metrics_block(best["metrics"])

        if self.last_eval is not None and self.last_epoch is not None:
            lines += [
                "",
                "## Final Epoch",
                f"- Epoch: **{self.last_epoch}**",
            ]
            lines += self._render_metrics_block(self.last_eval)

        lines += [
            "",
            "## Files",
            "- `config.yaml` — resolved config used for this run",
            "- `train_curves.csv` — per-epoch training stats",
            "- `eval_curves.csv` — per-epoch evaluation metrics",
            "- `best.pt`, `last.pt` — checkpoints",
            "- `best_metrics.json`, `final_metrics.json` — metrics snapshots",
            "",
        ]
        return "\n".join(lines)

    @staticmethod
    def _render_metrics_block(eval_results: dict[str, dict[str, float]]) -> list[str]:
        out: list[str] = []
        for direction, metrics in eval_results.items():
            out.append(f"- **{direction.upper()}**: " + ", ".join(
                f"{k}={v:.4f}" for k, v in metrics.items()
            ))
        return out
