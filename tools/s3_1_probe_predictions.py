#!/usr/bin/env python3
"""S3.1: regenerate E2 probe per-sample validation predictions.

The S3 E2 report stored only aggregate metrics, so a plot-level cluster
bootstrap of the A -> B difference was impossible from the saved artifacts.
This tool re-fits the exact same readouts (same pooled features, same seeds,
same budget) and writes one row per validation sample and seed.

It does not overwrite any S3 artifact.

Usage
-----
    python3 tools/s3_1_probe_predictions.py \
        --config agro_jepa/config/s3_v0.yaml \
        --probe-samples reports/s3_probe_samples.parquet \
        --pooled reports/s3_probe_pooled.npz \
        --out reports/s3_1_probe_val_predictions.parquet \
        --summary reports/s3_1_probe_reproduction.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agro_jepa" / "eval"))

from agro_jepa.eval.eval_v0 import fit_probe, pool_matrix, set_seed  # noqa: E402


def scope_frames(probe_df: pd.DataFrame, trait: str) -> dict[str, pd.DataFrame]:
    """Mirror the exact S3 scope construction in ``eval_v0.e2_probes``.

    The saved probe parquet already contains one row per
    ``(plot_uid, image_stem_b, pair_type)``.  ``overall`` further de-duplicates
    to one row per ``(plot_uid, image_stem_b)`` keeping the smallest delta, so
    the regenerated per-sample predictions cover the same rows as the reported
    S3 aggregates.
    """
    scopes: dict[str, pd.DataFrame] = {}
    overall = probe_df.sort_values(
        ["plot_uid", "image_stem_b", "delta_days", "pair_type", "pair_id"],
        kind="mergesort",
    ).drop_duplicates(["plot_uid", "image_stem_b"], keep="first").reset_index(drop=True)
    scopes["overall"] = overall[overall[f"{trait}_mask_b"]]
    for span in ("span7", "span14"):
        scopes[span] = probe_df[
            (probe_df.pair_type == span) & probe_df[f"{trait}_mask_b"]
        ].reset_index(drop=True)
    return scopes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s3_v0.yaml")
    ap.add_argument("--probe-samples", default="reports/s3_probe_samples.parquet")
    ap.add_argument("--pooled", default="reports/s3_probe_pooled.npz")
    ap.add_argument("--out", default="reports/s3_1_probe_val_predictions.parquet")
    ap.add_argument("--summary", default="reports/s3_1_probe_reproduction.json")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    probe_df = pd.read_parquet(args.probe_samples)
    # the saved parquet dropped "_pool_index"; the pooled matrices were stored
    # after the same de-duplication and are aligned with the file row order.
    probe_df["_pool_index"] = np.arange(len(probe_df))
    pooled = np.load(args.pooled)
    pooled_a, pooled_b = pooled["pool_a"], pooled["pool_b"]
    set_seed(0)

    rows: list[dict] = []
    reproduction: dict = {"device": str(device), "scopes": {}}
    t0 = time.time()
    for trait in cfg["data"]["traits"]:
        reproduction["scopes"][trait] = {}
        for scope, frame in scope_frames(probe_df, trait).items():
            tr = frame[frame.split == "train"]
            va = frame[frame.split == "validation"].reset_index(drop=True)
            if len(tr) < 100 or len(va) < 30:
                continue
            reproduction["scopes"][trait][scope] = {
                "n_train": int(len(tr)),
                "n_val": int(len(va)),
                "n_val_plots": int(va.plot_uid.nunique()),
                "conditions": {},
            }
            for cond, col in (("A_current", "pool_a"), ("B_predicted", "pool_b")):
                x_tr = pool_matrix(tr, col, pooled_a, pooled_b)
                x_va = pool_matrix(va, col, pooled_a, pooled_b)
                y_tr = tr[f"{trait}_b"].to_numpy(np.float32)
                y_va = va[f"{trait}_b"].to_numpy(np.float32)
                runs = []
                for seed in cfg["eval"]["probe_seeds"]:
                    metrics = fit_probe(
                        x_tr, y_tr, x_va, y_va, cfg, int(seed), device,
                        return_predictions=True,
                    )
                    pred = metrics.pop("predictions")
                    runs.append(metrics)
                    for i in range(len(va)):
                        rows.append({
                            "trait": trait,
                            "scope": scope,
                            "condition": cond,
                            "seed": int(seed),
                            "pair_id": va.pair_id.iloc[i],
                            "plot_uid": va.plot_uid.iloc[i],
                            "image_stem_b": va.image_stem_b.iloc[i],
                            "pair_type": va.pair_type.iloc[i],
                            "delta_days": int(va.delta_days.iloc[i]),
                            "lag_days": float(va[f"{trait}_b_lag"].iloc[i]),
                            "y_true": float(y_va[i]),
                            "y_pred": float(pred[i]),
                        })
                reproduction["scopes"][trait][scope]["conditions"][cond] = {
                    "n_seeds": len(runs),
                    "mae_mean": float(np.mean([r["mae"] for r in runs])),
                    "rmse_mean": float(np.mean([r["rmse"] for r in runs])),
                    "r2_mean": float(np.mean([r["r2"] for r in runs])),
                    "runs": runs,
                }
            print(f"{trait} {scope}: train={len(tr)} val={len(va)}", flush=True)

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(args.out, index=False)
    reproduction["n_rows"] = int(len(out_df))
    reproduction["elapsed_seconds"] = round(time.time() - t0, 1)
    reproduction["command"] = " ".join(sys.argv)
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary, "w") as fh:
        json.dump(reproduction, fh, indent=2)
    print(f"wrote {args.out} ({len(out_df)} rows) and {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
