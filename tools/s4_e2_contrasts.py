#!/usr/bin/env python3
"""S4A E2 pairwise contrasts (weather/GDD vs time-only, on the headline scope).

The primary E2 run reports each condition against ``A_current`` (current
features).  That answers "do predicted features help?" but not "does weather
add anything beyond Δt?" -- both are B conditions measured against the same A.
This tool takes the stored per-seed E2 validation predictions, averages them
over the three probe seeds, and bootstraps the *pairwise* ΔR² / ΔMAE / ΔRMSE
over plot clusters.  No model is retrained; no new data is read.

Usage
-----
    python3 tools/s4_e2_contrasts.py --config agro_jepa/config/s4_pilot.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from s3_1_diagnostics import cluster_bootstrap  # noqa: E402

CONTRASTS = [
    ("B_b3_delta", "B_b2_no_time"),
    ("B_b5_delta_weather_P", "B_b3_delta"),
    ("B_b5_delta_weather_R", "B_b3_delta"),
    ("B_b4_delta_gdd_P", "B_b3_delta"),
    ("B_b4_delta_gdd_R", "B_b3_delta"),
    ("B_b4_delta_gdd_R_tbase5", "B_b4_delta_gdd_R"),
    ("B_b4_delta_gdd_P", "B_b2_no_time"),
    ("B_b5_delta_weather_P", "B_b2_no_time"),
    ("B_b5_delta_weather_R", "B_b5_delta_weather_P"),
]


def _metrics(t, hi: str, lo: str) -> dict:
    y = t["y"]
    eh = t[hi] - y
    el = t[lo] - y
    ss = float(np.sum((y - y.mean()) ** 2))
    return {
        "delta_mae": float(np.mean(np.abs(eh)) - np.mean(np.abs(el))),
        "delta_rmse": float(np.sqrt(np.mean(eh ** 2)) - np.sqrt(np.mean(el ** 2))),
        "delta_r2": (-np.sum(eh ** 2) + np.sum(el ** 2)) / ss,
    }


def _r2(y, p):
    ss = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - float(np.sum((p - y) ** 2)) / ss


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s4_pilot.yaml")
    ap.add_argument("--preds", default="reports/s4_e2_probe_val_predictions.parquet")
    ap.add_argument("--out", default="reports/s4_e2_contrasts.json")
    ap.add_argument("--n-resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261024)
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    df = pd.read_parquet(args.preds)
    out = {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "source_predictions": args.preds,
        "scope": "primary_lag7 / overall (the only scope stored per-pair)",
        "seed_rule": "per-pair prediction averaged over probe seeds; bootstrap over plots",
        "traits": {},
    }
    for trait in ("cc", "ph"):
        d = df[df.trait == trait]
        if d.empty:
            continue
        base = d[["pair_id", "plot_uid"]].drop_duplicates().sort_values("pair_id")
        base = base.reset_index(drop=True)
        wide = base.copy()
        y = None
        for cond in sorted(d.condition.unique()):
            sub = (d[d.condition == cond]
                   .groupby("pair_id", as_index=False)
                   .agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean")))
            sub = base[["pair_id"]].merge(sub, on="pair_id", how="left")
            if sub.y_pred.isna().any():
                raise RuntimeError(f"{trait}/{cond}: missing pairs")
            wide[cond] = sub.y_pred.to_numpy(np.float64)
            y = sub.y_true.to_numpy(np.float64)
        wide["y"] = y
        plots = wide.plot_uid.to_numpy()
        tables = {"y": y}
        for cond in sorted(d.condition.unique()):
            tables[cond] = wide[cond].to_numpy(np.float64)

        block = {"n_pairs": int(len(wide)), "n_plots": int(
            wide.plot_uid.nunique()), "r2": {}, "contrasts": {}}
        for cond in sorted(d.condition.unique()):
            block["r2"][cond] = _r2(y, tables[cond])
        for hi, lo in CONTRASTS:
            block["contrasts"][f"{hi}_minus_{lo}"] = {
                k: cluster_bootstrap(tables, plots, lambda t, k=k, hi=hi, lo=lo:
                                     _metrics(t, hi, lo)[k],
                                     args.n_resamples, args.seed)
                for k in ("delta_mae", "delta_rmse", "delta_r2")
            }
        out["traits"][trait] = block

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)

    for trait, b in out["traits"].items():
        print(f"[{trait}] n={b['n_pairs']} plots={b['n_plots']}")
        for cond, r2 in b["r2"].items():
            print(f"  R2 {cond:32s} {r2:.5f}")
        for key, m in b["contrasts"].items():
            x = m["delta_r2"]
            print(f"  {key:56s} dR2 {x['mean']:+.5f} "
                  f"[{x['ci95_low']:+.5f},{x['ci95_high']:+.5f}]  "
                  f"dMAE {m['delta_mae']['mean']:+.5f} "
                  f"[{m['delta_mae']['ci95_low']:+.5f},{m['delta_mae']['ci95_high']:+.5f}]")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
