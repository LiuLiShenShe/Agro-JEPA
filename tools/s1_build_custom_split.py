#!/usr/bin/env python3
"""S1: build the project's *custom* cross-year split manifest.

This is deliberately separate from the official FIP splits (train /
validation / test_plot / test_genotype / test_environment /
test_genotype_environment) which are frozen in data/manifests/fip1_plots.csv.
The official assignment is never modified.

Custom protocol (from the project handbook, section 2.3):

  * Train/Val/ID-Test years: 2016, 2017, 2019, 2021
      - split *by genotype* inside each year: 80% / 10% / 10%
      - fixed seed 42; genotype sets are sorted before permutation so the
        assignment is reproducible
      - ID-Test is "seen years, unseen genotypes" (not a typical random IID test)
  * Cross-Year Test: 2018, 2022 -- the whole year is held out

Output:
    data/manifests/custom_crossyear_splits.csv
    reports/s1_custom_split_statistics.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
RULE_VERSION = "custom-crossyear-v1"
ID_YEARS = (2016, 2017, 2019, 2021)
OOD_YEARS = (2018, 2022)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plots", default="data/manifests/fip1_plots.csv")
    ap.add_argument("--out", default="data/manifests/custom_crossyear_splits.csv")
    ap.add_argument("--report", default="reports/s1_custom_split_statistics.json")
    args = ap.parse_args()

    plots = pd.read_csv(args.plots, dtype={"plot_uid": str, "genotype_id": str})
    rows = []
    rng = np.random.RandomState(SEED)
    for y in sorted(plots.harvest_year.unique()):
        sub = plots[plots.harvest_year == y]
        if y in OOD_YEARS:
            for _, r in sub.iterrows():
                rows.append({**r.to_dict(), "custom_split": "cross_year_test"})
            continue
        if y not in ID_YEARS:
            raise SystemExit(f"year {y} has no rule")
        genotypes = sorted(sub.genotype_id.dropna().unique())
        perm = rng.permutation(len(genotypes))
        shuffled = [genotypes[i] for i in perm]
        n = len(shuffled)
        n_tr = int(round(0.8 * n))
        n_va = int(round(0.1 * n))
        assign = {}
        for i, g in enumerate(shuffled):
            assign[g] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "id_test")
        for _, r in sub.iterrows():
            rows.append({**r.to_dict(), "custom_split": assign.get(r.genotype_id, "train")})

    out = pd.DataFrame(rows)
    keep = ["plot_uid", "harvest_year", "genotype_id", "official_split", "custom_split",
            "n_images", "cc_n", "ph_n", "whc_n", "sen_n"]
    out = out.rename(columns={"split": "official_split"})[keep]
    out["rule_version"] = RULE_VERSION
    out["seed"] = SEED
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    stats = {
        "rule_version": RULE_VERSION,
        "seed": SEED,
        "id_years": list(ID_YEARS),
        "cross_year_years": list(OOD_YEARS),
        "plot_counts": out.custom_split.value_counts().to_dict(),
        "plot_counts_by_year": {
            str(y): out[out.harvest_year == y].custom_split.value_counts().to_dict()
            for y in sorted(out.harvest_year.unique())
        },
        "genotype_counts": {
            "train": int(out[out.custom_split == "train"].genotype_id.nunique()),
            "val": int(out[out.custom_split == "val"].genotype_id.nunique()),
            "id_test": int(out[out.custom_split == "id_test"].genotype_id.nunique()),
            "cross_year_test": int(out[out.custom_split == "cross_year_test"].genotype_id.nunique()),
        },
        "genotype_overlap": {},
    }
    # genotype overlap between custom splits
    sets = {s: set(out.loc[out.custom_split == s, "genotype_id"].dropna())
            for s in out.custom_split.unique()}
    names = sorted(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            inter = sets[a] & sets[b]
            if inter:
                stats["genotype_overlap"][f"{a}__{b}"] = {
                    "n_shared": len(inter), "examples": sorted(inter)[:3]}
    stats["note"] = (
        "ID-Test is 'seen years, unseen genotypes' by construction; "
        "cross_year_test genotypes may also appear in other years (2018/2022 "
        "are held out as years, not as genotypes).")
    with open(args.report, "w") as fh:
        json.dump(stats, fh, indent=2)
    print(json.dumps(stats, indent=2))
    print(f"wrote {args.out} ({len(out)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
