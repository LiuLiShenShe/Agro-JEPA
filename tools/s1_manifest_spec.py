#!/usr/bin/env python3
"""S1: write the frozen manifest specification with content hashes.

Reads the manifest artifacts under data/manifests, records their SHA256 and
the exact rules used to build them, and writes data/manifests/manifest_spec.json.

Usage:
    python3 tools/s1_manifest_spec.py --manifests data/manifests
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

ARTIFACTS = [
    "fip1_plots.csv",
    "fip1_observations.csv.gz",
    "fip1_pairs_consecutive.parquet",
    "custom_crossyear_splits.csv",
    "weather_daily_first_plot_per_year.csv",
]
DATASET_REVISION = "11cdd84bc0ba8d5c6641ea7b53550b1658527e1c"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", default="data/manifests")
    ap.add_argument("--out", default="data/manifests/manifest_spec.json")
    args = ap.parse_args()
    md = Path(args.manifests)

    files = {}
    for name in ARTIFACTS:
        p = md / name
        if p.exists():
            files[name] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
        else:
            files[name] = {"missing": True}

    spec = {
        "manifest_id": "fip1-s1-v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset": {
            "name": "mikeboss/FIP1",
            "revision": DATASET_REVISION,
            "license": "CC-BY-4.0",
            "parquet_shards": 13,
            "parquet_total_bytes": 1375668510,
        },
        "official_splits": {
            "names": ["train", "validation", "test_plot", "test_genotype",
                      "test_environment", "test_genotype_environment"],
            "source": "row membership of the HuggingFace parquet shards",
            "note": ("test_environment and test_genotype_environment contain "
                     "only 2019 plots; test_plot is 'seen genotypes, seen "
                     "environment, unseen plots'; test_genotype is unseen "
                     "genotypes in seen environments."),
        },
        "custom_split": {
            "rule_version": "custom-crossyear-v1",
            "seed": 42,
            "id_years": [2016, 2017, 2019, 2021],
            "cross_year_years": [2018, 2022],
            "rule": ("within each ID year, genotypes are sorted, permuted with "
                     "numpy RandomState(42), and assigned 80/10/10 to "
                     "train/val/id_test; 2018 and 2022 are held out entirely"),
            "note": "ID-Test = seen years, unseen genotypes.",
        },
        "sample_definition": {
            "unit": "one image pair from the same plot_uid",
            "anchor_image": "inner_plot_images entry on date t_a",
            "target_image": "another inner_plot_images entry on date t_b",
            "required": "t_b > t_a; both images belong to the same plot_uid",
            "Δt": "(t_b - t_a).days",
            "consecutive_pairs": "fip1_pairs_consecutive.parquet (Δt unrestricted)",
            "span_pairs": ("for each span in {7,14,21,28,35,42} d, for each "
                           "anchor image, choose the nearest later image with "
                           "|Δt - span| <= 2 d; counts in "
                           "reports/s1_split_statistics.json"),
        },
        "weather_window": {
            "convention": "(t_a, t_b] -- left open, right closed",
            "n_daily_steps": "Δt",
            "raw_records": "hourly, 24 records per day, Sep 1 -> Aug 31",
            "missingness": "0 NaN in every checked year; identical across plots of a year",
            "left_join": "daily aggregates are built in S2; S1 provides window bounds",
        },
        "label_matching": {
            "rule": ("for every image date, match the nearest measurement date "
                     "of each trait; record the signed lag label_date - image_date"),
            "validity_masks": ("downstream uses |lag| <= 3 d (strict) or <= 7 d "
                               "(primary); the exact thresholds and counts are "
                               "in reports/s1_label_timing.md"),
            "missing_vs_zero": ("a trait is missing when there is no measurement "
                                "date in the plot record; a measured 0 is kept "
                                "as 0 and is never imputed"),
        },
        "units": {
            "canopy_cover": "raw 0-1 fraction (trait metadata says '%'; values are fractions)",
            "height": "m",
            "spike_count": "count (dimensionless)",
            "senescence": "0-10 score",
            "yield": "t ha-1",
            "protein": "%",
            "weather": "temperature degC, humidity %, irradiance W m-2, precipitation raw",
        },
        "artifacts": files,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(spec, fh, indent=2)
    print(json.dumps(files, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())