#!/usr/bin/env python3
"""S3: plot the real V0 training curve from the run artifacts.

Usage:
    python3 tools/s3_make_figures.py --config agro_jepa/config/s3_v0.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s3_v0.yaml")
    ap.add_argument("--run-summary", default=None)
    ap.add_argument("--results", default="reports/s3_v0_results.json")
    ap.add_argument("--out", default="reports/figures/s3_v0_training_curve.png")
    args = ap.parse_args()

    import yaml

    cfg = yaml.safe_load(open(args.config))
    summary_path = Path(
        args.run_summary or (Path(cfg["training"]["output_dir"]) / "run_summary.json")
    )
    run = json.loads(summary_path.read_text())
    hist = run.get("history", [])
    if not hist:
        raise SystemExit(f"{summary_path} has no epoch history")

    epochs = [h["epoch"] for h in hist]
    train = [h["train_loss"] for h in hist]
    val = [h["val"]["mse_pred"] for h in hist]
    persist = [h["val"]["mse_persist"] for h in hist]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(epochs, train, marker="o", ms=3, label="train loss")
    ax.plot(epochs, val, marker="o", ms=3, label="validation MSE (V0)")
    ax.plot(epochs, persist, marker="o", ms=3, linestyle="--",
            label="validation MSE (persistence)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE in frozen DINOv2 token space")
    ax.set_title("FIP-DINO-WM V0: training and validation")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1]
    rel = [
        (p - v) / p * 100.0 if p else 0.0
        for v, p in zip(val, persist)
    ]
    ax.plot(epochs, rel, marker="o", ms=3, color="tab:green")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE reduction vs persistence (%)")
    ax.set_title("Improvement over the persistence baseline")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
