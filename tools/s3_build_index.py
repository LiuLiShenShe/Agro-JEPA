#!/usr/bin/env python3
"""S3: build the frozen train/validation pair index (no S1/S2 file is modified).

Usage
-----
    python3 tools/s3_build_index.py --config agro_jepa/config/s3_v0.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agro_jepa.datasets.s3_pairs import (  # noqa: E402
    S3IndexRule,
    build_s3_index,
    index_sha256,
    load_download_selection,
    load_feature_index,
    load_s2_pairs,
)
from agro_jepa.models.time_embedding import DeltaNormStats  # noqa: E402


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s3_v0.yaml")
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    d = cfg["data"]

    t0 = time.time()
    pairs = load_s2_pairs(d["pairs"], pair_types=d["pair_types"], splits=d["splits"])
    selection = load_download_selection(d["selection"])
    features = load_feature_index(d["feature_index"])
    rule = S3IndexRule(
        pair_types=tuple(d["pair_types"]),
        splits=tuple(d["splits"]),
        dedupe_physical_pairs=bool(d.get("dedupe_physical_pairs", True)),
        require_cache=True,
    )
    index, stats = build_s3_index(pairs, selection, features, rule)

    # time normalisation: train split only, never validation
    train_delta = index.loc[index.split == "train", "delta_days"]
    if train_delta.empty:
        raise SystemExit("the index has no train pairs; check downloads/features")
    norm = DeltaNormStats.fit(train_delta)
    stats["delta_normalization"] = norm.as_dict()
    stats["index_sha256"] = index_sha256(index)
    stats["manifest_id"] = cfg["manifest_id"]
    stats["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    stats["elapsed_seconds"] = round(time.time() - t0, 1)
    stats["inputs"] = {
        "pairs": d["pairs"],
        "selection": d["selection"],
        "feature_index": d["feature_index"],
    }

    # compact, committed selection rule (the full selection manifest stays in
    # external_data because it carries ~50k per-image SHA records)
    with open(d["selection"]) as fh:
        sel_meta = json.load(fh)
    sel_df = pd.DataFrame(sel_meta["images"])
    rule_path = Path(d.get("selection_rule_out", "data/manifests/s3_selection_rule.json"))
    rule_path.parent.mkdir(parents=True, exist_ok=True)
    rule = {
        "manifest_id": cfg["manifest_id"],
        "selection_manifest": d["selection"],
        "selection_manifest_sha256": sha256_file(d["selection"]),
        "selection": sel_meta.get("selection"),
        "n_images": int(len(sel_df)),
        "n_plots": int(sel_df.plot_uid.nunique()),
        "n_failed": int(sel_meta.get("n_failed", 0)),
        "by_split": {str(k): int(v) for k, v in sel_df.split.value_counts().items()},
        "by_year": {
            str(k): int(v) for k, v in
            sel_df.harvest_year.value_counts().sort_index().items()
        },
        "plots_per_year_and_split": {
            f"{s}|{int(y)}": int(v)
            for (s, y), v in sel_df.groupby(["split", "harvest_year"]).plot_uid.nunique().items()
        },
        "plots": sorted(sel_df.plot_uid.unique().tolist()),
        "rule": (
            "all official validation plots + a fixed-seed per-year sample of "
            "official train plots; every audited image of every selected plot"
        ),
    }
    with open(rule_path, "w") as fh:
        json.dump(rule, fh, indent=2)

    out = Path(d["index_out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    index.to_parquet(out, index=False)
    sample_path = Path(d.get("index_sample_csv", "reports/s3_index_sample.csv"))
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_cols = [
        "pair_id", "plot_uid", "genotype_id", "split", "pair_type", "span_days",
        "delta_days", "date_a", "date_b", "image_stem_a", "image_stem_b",
        "cc_a", "cc_b", "cc_mask_a", "cc_mask_b", "cc_b_lag", "cc_b_date",
        "ph_a", "ph_b", "ph_mask_a", "ph_mask_b", "ph_b_lag", "ph_b_date",
        "ph_negative_a", "ph_negative_b",
    ]
    index.sample(n=min(200, len(index)), random_state=42)[sample_cols].to_csv(
        sample_path, index=False
    )
    report = Path(d["index_stats"])
    report.parent.mkdir(parents=True, exist_ok=True)
    with open(report, "w") as fh:
        json.dump(stats, fh, indent=2)

    print(json.dumps({
        "n_pairs": stats["n_pairs"],
        "train": {k: stats.get("train", {}).get(k) for k in
                  ("n_pairs", "n_unique_images", "n_plots", "n_years")},
        "validation": {k: stats.get("validation", {}).get(k) for k in
                       ("n_pairs", "n_unique_images", "n_plots", "n_years")},
        "delta_normalization": norm.as_dict(),
        "index_sha256": stats["index_sha256"],
        "selection_rule_file": str(rule_path),
        "selection_manifest_sha256": rule["selection_manifest_sha256"],
    }, indent=2), flush=True)
    print(f"wrote {out} and {report}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
