#!/usr/bin/env python3
"""S1: split / label / timing statistics from the audited manifest CSVs.

Pure post-processing of data/manifests/fip1_plots.csv and
data/manifests/fip1_observations.csv; no parquet access.

Emits reports/s1_split_statistics.json with:
  - split sizes and cross-split overlap for plot_uid / genotype_id /
    yearsite_uid / harvest_year
  - per-year, per-split label availability
  - image-date vs phenotype-date lag distributions and coverage at
    +/-3 d and +/-7 d
  - weather-window definition check: (t_a, t_b] yields exactly dt days
  - candidate pair counts for consecutive pairs and for target spans

Usage:
    python3 tools/s1_dataset_stats.py --manifests data/manifests \
        --report reports/s1_split_statistics.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ["train", "validation", "test_plot", "test_genotype",
          "test_environment", "test_genotype_environment"]
TRAITS = ["cc", "ph", "whc", "sen"]


def pct(x, n):
    return float(x) / float(n) if n else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", default="data/manifests")
    ap.add_argument("--report", default="reports/s1_split_statistics.json")
    args = ap.parse_args()
    md = Path(args.manifests)

    plots = pd.read_csv(md / "fip1_plots.csv", dtype={"plot_uid": str, "genotype_id": str})
    obs_path = md / "fip1_observations.csv"
    if not obs_path.exists():
        obs_path = md / "fip1_observations.csv.gz"
    obs = pd.read_csv(obs_path, dtype={"plot_uid": str})
    out = {"n_plots": int(len(plots)), "n_observations": int(len(obs))}

    # ---------- split sizes ----------
    out["split_sizes"] = {
        s: int((plots.split == s).sum()) for s in SPLITS
    }
    out["split_year_matrix"] = {
        s: {str(y): int(((plots.split == s) & (plots.harvest_year == y)).sum())
            for y in sorted(plots.harvest_year.unique())}
        for s in SPLITS
    }

    # ---------- cross-split overlap ----------
    def overlap(col):
        sets = {s: set(plots.loc[plots.split == s, col].dropna()) for s in SPLITS}
        pairs = {}
        for i, a in enumerate(SPLITS):
            for b in SPLITS[i + 1:]:
                inter = sets[a] & sets[b]
                pairs[f"{a}__{b}"] = {
                    "n_shared": len(inter),
                    "examples": sorted(inter)[:5],
                }
        return {"n_unique_per_split": {s: len(v) for s, v in sets.items()}, "pairwise": pairs}

    out["overlap_plot_uid"] = overlap("plot_uid")
    out["overlap_genotype_id"] = overlap("genotype_id")
    out["overlap_yearsite_uid"] = overlap("yearsite_uid")

    # ---------- label availability (per plot row, from raw arrays) ----------
    label_by_year_split = {}
    for s in SPLITS:
        sub = plots[plots.split == s]
        per_year = {}
        for y in sorted(sub.harvest_year.unique()):
            sy = sub[sub.harvest_year == y]
            row = {"n_plots": int(len(sy))}
            for tr, col in (("cc", "cc_n"), ("ph", "ph_n"), ("whc", "whc_n"), ("sen", "sen_n")):
                row[f"{tr}_plots_with_any"] = int((sy[col] > 0).sum())
                row[f"{tr}_observations"] = int(sy[col].sum())
            row["n_images"] = int(sy.n_images.sum())
            per_year[str(y)] = row
        label_by_year_split[s] = per_year
    out["label_availability"] = label_by_year_split

    # ---------- image-date vs label-date lag ----------
    lag_stats = {}
    for tr in TRAITS:
        lag = pd.to_numeric(obs[f"{tr}_lag_days"], errors="coerce")
        val = pd.to_numeric(obs[f"{tr}_value"], errors="coerce")
        lag_stats[tr] = {
            "n_observations": int(len(obs)),
            "n_matched_any": int(lag.notna().sum()),
            "n_unmatched": int(lag.isna().sum()),
            "n_valid_value": int(val.notna().sum()),
            "n_zero_value": int((val == 0).sum()),
            "lag_median_days": float(lag.median()) if lag.notna().any() else None,
            "lag_p90_days": float(lag.quantile(0.9)) if lag.notna().any() else None,
            "frac_lag_le_3": pct((lag.abs() <= 3).sum(), len(obs)),
            "frac_lag_le_7": pct((lag.abs() <= 7).sum(), len(obs)),
            "value_min": float(val.min()) if val.notna().any() else None,
            "value_max": float(val.max()) if val.notna().any() else None,
        }
    out["label_timing"] = lag_stats

    # ---------- candidate pairs ----------
    obs["image_date"] = pd.to_datetime(obs["image_date"])
    pair_rows = []
    for puid, g in obs.groupby("plot_uid", sort=False):
        g = g.sort_values("image_date")
        dates = g.image_date.tolist()
        n = len(g)
        for i in range(n - 1):
            da = dates[i]
            db = dates[i + 1]
            dt = (db - da).days
            ra = g.iloc[i]
            rb = g.iloc[i + 1]
            rec = {"manifest_id": "fip1-s1-v1",
                   "plot_uid": puid, "split": ra.split, "harvest_year": int(ra.harvest_year),
                   "i": int(ra.obs_index), "j": int(rb.obs_index),
                   "image_a": ra["inner_plot_image"], "image_b": rb["inner_plot_image"],
                   "date_a": da.date().isoformat(), "date_b": db.date().isoformat(),
                   "delta_days": int(dt),
                   "window_start_exclusive": da.date().isoformat(),
                   "window_end_inclusive": db.date().isoformat(),
                   "window_n_days": int(dt)}
            for tr in TRAITS:
                rec[f"{tr}_a"] = ra[f"{tr}_value"]
                rec[f"{tr}_b"] = rb[f"{tr}_value"]
                rec[f"{tr}_a_lag"] = ra[f"{tr}_lag_days"]
                rec[f"{tr}_b_lag"] = rb[f"{tr}_lag_days"]
                rec[f"{tr}_a_date"] = ra[f"{tr}_date"]
                rec[f"{tr}_b_date"] = rb[f"{tr}_date"]
            pair_rows.append(rec)
    pairs = pd.DataFrame(pair_rows)
    pairs.to_parquet(md / "fip1_pairs_consecutive.parquet", index=False)

    bins = [0, 3, 7, 14, 21, 28, 35, 45, 60, 1000]
    labels = ["0-3", "4-7", "8-14", "15-21", "22-28", "29-35", "36-45", "46-60", "60+"]
    hist = np.histogram(pairs.delta_days, bins=bins)[0].tolist()
    out["consecutive_pairs"] = {
        "n": int(len(pairs)),
        "delta_days": {
            "min": int(pairs.delta_days.min()),
            "median": float(pairs.delta_days.median()),
            "p90": float(pairs.delta_days.quantile(0.9)),
            "max": int(pairs.delta_days.max()),
            "histogram": dict(zip(labels, hist)),
        },
        "by_split": {s: int((pairs.split == s).sum()) for s in SPLITS},
    }
    # valid-pair counts: label present at both ends within +/-7 d
    for tr in TRAITS:
        la = pd.to_numeric(pairs[f"{tr}_a_lag"], errors="coerce")
        lb = pd.to_numeric(pairs[f"{tr}_b_lag"], errors="coerce")
        va = pd.to_numeric(pairs[f"{tr}_a"], errors="coerce")
        vb = pd.to_numeric(pairs[f"{tr}_b"], errors="coerce")
        mask = (la.abs() <= 7) & (lb.abs() <= 7) & va.notna() & vb.notna()
        out["consecutive_pairs"][f"{tr}_valid_both_le7"] = int(mask.sum())
        out["consecutive_pairs"][f"{tr}_valid_both_le7_by_split"] = {
            s: int((mask & (pairs.split == s)).sum()) for s in SPLITS}

    # target-span candidate counts: for every anchor image pick the nearest
    # later image to the target span (tolerance +/-2 d). This is the sampling
    # rule frozen for the S1 manifest; it does not require consecutive images.
    spans = [7, 14, 21, 28, 35, 42]
    span_counts = {str(sp): {"n_pairs": 0, "by_split": {s: 0 for s in SPLITS},
                             "cc_valid_both_le7": 0, "ph_valid_both_le7": 0,
                             "cc_valid_both_le7_by_split": {s: 0 for s in SPLITS},
                             "ph_valid_both_le7_by_split": {s: 0 for s in SPLITS}}
                   for sp in spans}
    for puid, g in obs.groupby("plot_uid", sort=False):
        dates = g.image_date.values.astype("datetime64[D]")
        n = len(dates)
        if n < 2:
            continue
        for i in range(n - 1):
            ra = g.iloc[i]
            for sp in spans:
                target = dates[i] + np.timedelta64(sp, "D")
                j = int(np.searchsorted(dates, target, side="left"))
                cand = [c for c in (j - 1, j) if i < c < n]
                if not cand:
                    continue
                jj = min(cand, key=lambda c: abs((dates[c] - dates[i]).astype(int) - sp))
                dt = int((dates[jj] - dates[i]).astype(int))
                if abs(dt - sp) > 2 or dt <= 0:
                    continue
                rb = g.iloc[jj]
                rec = span_counts[str(sp)]
                rec["n_pairs"] += 1
                rec["by_split"][ra.split] += 1
                for tr in ("cc", "ph"):
                    la = abs(float(ra[f"{tr}_lag_days"])) if pd.notna(ra[f"{tr}_lag_days"]) else 1e9
                    lb = abs(float(rb[f"{tr}_lag_days"])) if pd.notna(rb[f"{tr}_lag_days"]) else 1e9
                    va, vb = ra[f"{tr}_value"], rb[f"{tr}_value"]
                    if la <= 7 and lb <= 7 and pd.notna(va) and pd.notna(vb):
                        rec[f"{tr}_valid_both_le7"] += 1
                        rec[f"{tr}_valid_both_le7_by_split"][ra.split] += 1
    out["span_candidates"] = span_counts

    # ---------- image path sanity ----------
    bad = obs[~obs.inner_plot_image.str.contains(r"inner_plot_image/", na=False)]
    fname = obs.inner_plot_image.str.rsplit("/", n=1).str[-1]
    stem_ok = pd.Series(
        [isinstance(f, str) and f.startswith(str(p))
         for f, p in zip(fname, obs.plot_uid)],
        index=obs.index, dtype=bool)
    # filename embeds YYYYMMDD; compare with the recorded image_date
    fdate = fname.str.extract(r"_(\d{8})_")[0]
    rec_date = obs.image_date.dt.strftime("%Y%m%d")
    date_ok = (fdate == rec_date)
    out["image_paths"] = {
        "n": int(len(obs)),
        "n_missing_path": int(obs.inner_plot_image.isna().sum()),
        "n_without_inner_plot_image_segment": int(len(bad)),
        "n_filename_plot_uid_mismatch": int((~stem_ok).sum()),
        "n_filename_date_mismatch": int((~date_ok).sum()),
        "example": obs.inner_plot_image.iloc[0],
    }

    with open(args.report, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps({k: out[k] for k in
                      ("split_sizes", "consecutive_pairs")}, indent=2)[:3000])
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
