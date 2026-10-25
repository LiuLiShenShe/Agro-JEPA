#!/usr/bin/env python3
"""S4A generalization split: seen vs unseen genotype, using only stored E1 outputs.

The official FIP split holds out plots, never years: every validation harvest
year also appears in the training set, but 44 of the 183 validation genotypes
(in the S3 pair index) never occur on a training plot.  This tool takes the
per-pair E1 predictions that already exist on disk and splits them by that
genotype axis, so the question "does the gain survive on unseen genotypes?"
is answered without any new forward pass.

Input   reports/s4_e1_val_predictions.parquet   (long format, one row/pair/condition)
        the S3 pair index (train rows define the seen-genotype set)
Output  reports/s4_generalization.json

Usage
-----
    python3 tools/s4_generalization_split.py --config agro_jepa/config/s4_pilot.yaml
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

LABELS = {
    "b2_no_time": "B2 no-time",
    "b3_delta": "B3 delta",
    "b4_gdd_R": "B4 GDD(R)",
    "b4_delta_gdd_R": "B4 delta+GDD(R)",
    "b4_delta_gdd_R_tbase5": "B4 delta+GDD(R,Tb5)",
    "b4_delta_gdd_P": "B4 delta+GDD(P)",
    "b5_delta_weather_P": "B5 delta+weather(P)",
    "b5_delta_weather_R": "B5 delta+weather(R)",
}

CONTRASTS = [
    ("b5_delta_weather_P", "b3_delta"),
    ("b5_delta_weather_R", "b3_delta"),
    ("b4_delta_gdd_P", "b3_delta"),
    ("b4_delta_gdd_R", "b3_delta"),
    ("b3_delta", "b2_no_time"),
]


def _summary(g: pd.DataFrame) -> dict:
    mp = float(g.mse_pred.mean())
    ma = float(g.mse_persist.mean())
    return {
        "n_pairs": int(len(g)),
        "n_plots": int(g.plot_uid.nunique()),
        "n_genotypes": int(g.genotype_id.nunique()),
        "mse_pred": mp,
        "mse_persist": ma,
        "relative_improvement_vs_persist": (ma - mp) / ma if ma > 0 else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s4_pilot.yaml")
    ap.add_argument("--preds", default="reports/s4_e1_val_predictions.parquet")
    ap.add_argument("--out", default="reports/s4_generalization.json")
    ap.add_argument("--n-resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261024)
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    index = pd.read_parquet(cfg["data"]["index"])
    seen = set(index.loc[index.split == "train", "genotype_id"])
    df = pd.read_parquet(args.preds)
    if df.split.nunique() != 1 or df.split.iloc[0] != "validation":
        raise RuntimeError("E1 predictions must be the official validation split")
    df = df.copy()
    df["genotype_group"] = np.where(df.genotype_id.isin(seen),
                                    "seen_genotype", "unseen_genotype")

    # one row per pair per condition -> wide by condition
    conds = sorted(df.condition.unique())
    wide_idx = df[df.condition == conds[0]][["pair_id"]].reset_index(drop=True)
    wide = wide_idx.copy()
    meta_cols = ["pair_id", "plot_uid", "genotype_id", "genotype_group",
                 "harvest_year", "pair_type", "mse_persist"]
    base = df[df.condition == conds[0]][meta_cols].reset_index(drop=True)
    wide = base.copy()
    for c in conds:
        sub = df[df.condition == c]
        if not np.array_equal(sub.pair_id.to_numpy(), wide.pair_id.to_numpy()):
            raise RuntimeError(f"pair rows not aligned for {c}")
        wide[f"mse::{c}"] = sub.mse_pred.to_numpy(np.float64)

    out = {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "source_predictions": args.preds,
        "split_used": "official validation (never used for selection)",
        "genotype_definition": (
            "seen = genotype_id occurs on at least one official-train plot in the "
            "S3 pair index; unseen = no training plot has this genotype"
        ),
        "n_train_genotypes": int(len(seen)),
        "n_val_genotypes": int(df.genotype_id.nunique()),
        "n_val_genotypes_unseen": int(df.loc[df.genotype_group == "unseen_genotype",
                                             "genotype_id"].nunique()),
        "n_pairs_unseen_genotype": int((df.genotype_group == "unseen_genotype").sum()
                                       // len(conds)),
        "groups": {}, "contrasts": {},
    }

    plots = wide.plot_uid.to_numpy()
    for grp, g in wide.groupby("genotype_group", sort=True):
        out["groups"][grp] = {"conditions": {}}
        for c in conds:
            g = g.copy()
            g["mse_pred"] = g[f"mse::{c}"]
            out["groups"][grp]["conditions"][c] = _summary(g)

    # bootstrap: improvement vs persistence, and pairwise MSE contrasts
    for grp, g in wide.groupby("genotype_group", sort=True):
        gp = g.plot_uid.to_numpy()
        tables = {"persist": g.mse_persist.to_numpy(np.float64)}
        for c in conds:
            tables[c] = g[f"mse::{c}"].to_numpy(np.float64)

        def improve(t, key):
            m = float(np.mean(t["persist"]))
            return (m - float(np.mean(t[key]))) / m

        blk = {"improvement_vs_persist": {}, "contrasts": {}}
        for c in conds:
            blk["improvement_vs_persist"][c] = cluster_bootstrap(
                tables, gp, lambda t, c=c: improve(t, c),
                args.n_resamples, args.seed)
        for hi, lo in CONTRASTS:
            def delta(t, hi=hi, lo=lo):
                return float(np.mean(t[hi]) - np.mean(t[lo]))
            blk["contrasts"][f"{hi}_minus_{lo}"] = cluster_bootstrap(
                tables, gp, delta, args.n_resamples, args.seed)
        out["contrasts"][grp] = blk

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"val genotypes {out['n_val_genotypes']}, unseen "
          f"{out['n_val_genotypes_unseen']} "
          f"({out['n_pairs_unseen_genotype']} pairs)")
    for grp, blk in out["groups"].items():
        print(f"[{grp}]")
        for c in conds:
            s = blk["conditions"][c]
            ci = out["contrasts"][grp]["improvement_vs_persist"][c]
            print(f"  {LABELS.get(c, c):24s} mse={s['mse_pred']:.5f} "
                  f"impr={s['relative_improvement_vs_persist']*100:5.2f}% "
                  f"[{ci['ci95_low']*100:5.2f},{ci['ci95_high']*100:5.2f}]")
    for grp, blk in out["contrasts"].items():
        print(f"[{grp}] contrasts (MSE)")
        for k, v in blk["contrasts"].items():
            print(f"  {k:44s} {v['mean']:+.5f} "
                  f"[{v['ci95_low']:+.5f},{v['ci95_high']:+.5f}]")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
