#!/usr/bin/env python3
"""S4B.3B -- freeze the CC/PH readout protocol before any test label is read.

This tool reads **only** the official-train pair table and the frozen S1 plots
manifest.  It never opens a test phenotype label.  It fixes, on disk:

* the inner 80/20 plot split of **all 2,930 official-train plots** (seed
  20261025, stratified by harvest_year) and its SHA256;
* the label-matching rule (value present AND |label_lag_days| <= 7, missing ->
  invalid, never 0);
* the readout normalisation, epoch-selection rule and metrics;
* the condition set (A_current / B2 / B2-C / B3) and the four test splits.

The split file and protocol JSON are committed before ``s4b3_test_e2.py`` is
allowed to run.  That ordering is what makes the E2 test numbers pre-registered.

Usage
-----
    python3 tools/s4b3_e2_protocol.py \
        --config agro_jepa/config/s4b3_official_test.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agro_jepa.datasets.s4_dataset import build_role_map  # noqa: E402
from s3_1_diagnostics import git, sha256_file  # noqa: E402

SPLIT_CSV = "reports/s4b3_e2_plot_split.csv"
PROTOCOL_PRE = "reports/s4b3_e2_protocol_prefix.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s4b3_official_test.yaml")
    ap.add_argument("--out-json", default="reports/s4b3_e2_protocol.json")
    ap.add_argument("--out-md", default="reports/s4b3_e2_protocol.md")
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    t0 = time.time()
    e2 = cfg["e2"]

    # ---- official-train pair table (labels stay on disk; not read here) --
    train_full = pd.read_parquet(ROOT / cfg["data"]["full_train_pairs"],
                                 columns=["pair_id", "plot_uid", "harvest_year",
                                          "split", "pair_type", "delta_days"])
    if set(train_full.split.unique()) != {"train"}:
        raise SystemExit("full-train table contains non-train rows")
    n_plots = int(train_full.plot_uid.nunique())
    if n_plots != 2930:
        raise SystemExit(f"expected 2,930 official-train plots, got {n_plots}")

    # ---- frozen inner 80/20 split over ALL official-train plots ---------
    role_map, split_stats = build_role_map(
        train_full, int(e2["plot_split_seed"]), float(e2["inner_train_fraction"]))
    split = pd.DataFrame({
        "plot_uid": list(role_map.keys()),
        "role": list(role_map.values()),
    })
    year = train_full.groupby("plot_uid").harvest_year.first()
    split["harvest_year"] = split.plot_uid.map(year).astype(int)
    split = split.sort_values(["harvest_year", "plot_uid"]).reset_index(drop=True)
    if split.plot_uid.duplicated().any():
        raise SystemExit("inner split contains duplicated plots")
    split_path = ROOT / SPLIT_CSV
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(split_path, index=False)
    split_sha = sha256_file(split_path)

    # ---- verify the split against the frozen rule ------------------------
    in_tr = set(split.loc[split.role == "inner_train", "plot_uid"])
    in_dev = set(split.loc[split.role == "inner_dev", "plot_uid"])
    if in_tr & in_dev:
        raise SystemExit("inner-train and inner-dev overlap")
    if len(in_tr) + len(in_dev) != n_plots:
        raise SystemExit("inner split does not cover every official-train plot")
    # official validation and every test plot must be absent from the split
    train_plots = set(train_full.plot_uid.astype(str))
    if not (in_tr | in_dev) <= train_plots:
        raise SystemExit("inner split contains a non-train plot")

    # ---- per-year role counts -------------------------------------------
    by_year = {
        str(y): {
            "plots": int(len(g)),
            "inner_train": int((g.role == "inner_train").sum()),
            "inner_dev": int((g.role == "inner_dev").sum()),
        }
        for y, g in split.groupby("harvest_year")
    }

    checks = []

    def record(cid, ok, detail):
        checks.append({"id": cid, "pass": bool(ok), "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'}  {cid}: {detail}", flush=True)

    record("E1_all_2930_train_plots",
           n_plots == 2930 and len(in_tr) + len(in_dev) == 2930,
           f"{n_plots} train plots -> {len(in_tr)} inner-train / {len(in_dev)} inner-dev")
    record("E2_disjoint_and_complete", not (in_tr & in_dev) and
           len(in_tr | in_dev) == 2930,
           "inner-train and inner-dev are disjoint and jointly cover all train plots")
    record("E3_no_validation_or_test_plot",
           not (in_tr | in_dev) - train_plots,
           "official validation and every test plot are excluded")
    record("E4_seed_and_fraction",
           split_stats["seed"] == int(e2["plot_split_seed"])
           and abs(split_stats["inner_train_fraction"] - float(e2["inner_train_fraction"])) < 1e-12,
           f"seed {e2['plot_split_seed']}, inner-train fraction "
           f"{e2['inner_train_fraction']}")

    protocol = {
        "stage": "S4B.3",
        "task": "E2 CC/PH phenotype readout protocol (frozen before test labels)",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": " ".join(sys.argv),
        "config": args.config,
        "config_sha256": sha256_file(ROOT / args.config),
        "git_head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "train_scope": {
            "source": cfg["data"]["full_train_pairs"],
            "source_sha256": sha256_file(ROOT / cfg["data"]["full_train_pairs"]),
            "n_plots": n_plots,
            "n_pairs": int(len(train_full)),
            "note": ("all 2,930 official-train plots (NOT the S3.2 1,000-plot "
                     "pilot); official validation and test plots excluded"),
        },
        "inner_split": {
            **split_stats,
            "file": SPLIT_CSV,
            "sha256": split_sha,
            "by_year": by_year,
            "rule": ("80/20 by plot_uid, stratified by harvest_year, "
                     "numpy.default_rng(seed) inside each year"),
        },
        "conditions": {
            "A_current": "mean(z_a) current frozen DINOv2 feature",
            "B2": "mean(z_hat_b) from b2_no_time",
            "B2-C": "mean(z_hat_b) from b2_no_time_constant_delta",
            "B3": "mean(z_hat_b) from b3_delta (real delta_days)",
        },
        "label_matching": {
            "source": "frozen S2 span7/span14 pair table joined on pair_id",
            "value_column": "{trait}_b",
            "lag_column": "{trait}_b_lag",
            "date_column": "{trait}_b_date",
            "validity": ("value finite AND date finite AND |label_lag_days| <= "
                         "lag_max_abs_days; a missing label is INVALID and is "
                         "never filled with 0"),
            "primary_lag_max_abs_days": int(e2["lag_variants"]["primary_lag7"]),
            "sensitivity_variants": {
                "strict_lag3": int(e2["lag_variants"]["strict_lag3"]),
                "same_day_lag0": int(e2["lag_variants"]["same_day_lag0"]),
            },
            "scope": "overall (one row per plot_uid, image_stem_b)",
        },
        "readout": {
            "architecture": e2["architecture"],
            "pooling": e2["pooling"],
            "input_dim": 768,
            "normalisation": ("x standardised by inner-train mean/std; y by "
                              "inner-train mean/std"),
            "optimiser": "AdamW + cosine annealing anchored to the candidate budget",
            "candidate_epochs": int(e2["candidate_epochs"]),
            "patience": int(e2["patience"]),
            "batch_size": int(e2["batch_size"]),
            "lr": float(e2["lr"]),
            "weight_decay": float(e2["weight_decay"]),
            "seeds": [int(s) for s in e2["seeds"]],
            "selection": ("minimum inner-dev MSE over the candidate budget; "
                          "median selected epoch across probe seeds"),
            "final_refit": ("refit on all rows of the full official-train set at "
                            "the selected epoch; evaluated once per test split"),
            "official_validation_used_for_selection": False,
            "test_labels_used_for_training_or_selection": False,
        },
        "metrics": ["mae", "rmse", "r2", "effective_n", "effective_plots"],
        "bootstrap": {
            "unit": "plot_uid",
            "n_resamples": int(e2["bootstrap_resamples"]),
            "seed": int(e2["bootstrap_seed"]),
            "interval": "95% percentile",
        },
        "test_splits": list(cfg["test_splits"].keys()),
        "verification": {
            "n_checks": len(checks),
            "n_pass": sum(1 for c in checks if c["pass"]),
            "all_pass": all(c["pass"] for c in checks),
            "checks": checks,
        },
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(protocol, fh, indent=2)
    print(f"wrote {out_json}")

    L = []
    L.append("# S4B.3 E2 -- CC/PH Readout Protocol (frozen before test labels)")
    L.append("")
    L.append(f"- Generated: {protocol['generated_at']}")
    L.append(f"- Git HEAD: `{protocol['git_head']}` ({protocol['branch']})")
    L.append(f"- Config: `{args.config}` (sha `{protocol['config_sha256'][:16]}`)")
    L.append("")
    L.append("## Train scope")
    L.append("")
    L.append(f"- All **{n_plots:,} official-train plots** ({len(train_full):,} pairs).")
    L.append("- Official validation and every test plot are excluded from training "
             "and from epoch/model selection.")
    L.append("")
    L.append("## Frozen inner split (seed "
             f"{e2['plot_split_seed']}, 80/20, stratified by harvest_year)")
    L.append("")
    L.append(f"- File: `{SPLIT_CSV}` (sha `{split_sha}`)")
    L.append(f"- inner-train: {len(in_tr)} plots / inner-dev: {len(in_dev)} plots")
    L.append("")
    L.append("| harvest_year | plots | inner-train | inner-dev |")
    L.append("|---|---:|---:|---:|")
    for y, b in by_year.items():
        L.append(f"| {y} | {b['plots']} | {b['inner_train']} | {b['inner_dev']} |")
    L.append("")
    L.append("## Conditions")
    L.append("")
    for k, v in protocol["conditions"].items():
        L.append(f"- **{k}**: {v}")
    L.append("")
    L.append("## Label matching")
    L.append("")
    lm = protocol["label_matching"]
    L.append(f"- Source: {lm['source']}")
    L.append(f"- Primary window: `{lm['validity']}` "
             f"(lag <= {lm['primary_lag_max_abs_days']} d).")
    L.append("- Sensitivity: strict-lag3 (lag <= 3 d) and same-day (lag == 0).")
    L.append("- Missing labels are invalid, never 0.")
    L.append("")
    L.append("## Readout")
    L.append("")
    rd = protocol["readout"]
    L.append(f"- {rd['architecture']}; {rd['pooling']}.")
    L.append(f"- {rd['normalisation']}.")
    L.append(f"- {rd['optimiser']}; candidate {rd['candidate_epochs']} epochs, "
             f"patience {rd['patience']}, batch {rd['batch_size']}, "
             f"lr {rd['lr']}, wd {rd['weight_decay']}, seeds {rd['seeds']}.")
    L.append(f"- Selection: {rd['selection']}.")
    L.append(f"- Final: {rd['final_refit']}.")
    L.append("- Test labels are never used for training, epoch choice or model "
             "selection.")
    L.append("")
    L.append("## Protocol checks")
    L.append("")
    L.append("| check | pass | detail |")
    L.append("|---|---|---|")
    for c in checks:
        L.append(f"| {c['id']} | {'PASS' if c['pass'] else 'FAIL'} | {c['detail']} |")
    L.append("")
    L.append(f"**all_pass = {protocol['verification']['all_pass']}**")
    Path(args.out_md).write_text("\n".join(L) + "\n")
    print(f"wrote {args.out_md}")
    return 0 if protocol["verification"]["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
