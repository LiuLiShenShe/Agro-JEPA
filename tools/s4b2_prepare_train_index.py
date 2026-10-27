#!/usr/bin/env python3
"""S4B.2: build the dedicated full-official-train index for final training.

Reads the frozen S4B.1 full-train pair table and joins it to the S4B.1 feature
index by exact ``image_filename_stem`` (uppercased), never by row position.  It
keeps only official ``train`` rows and drops every phenotype-label / label-mask
/ measurement-date column, so the derived training index carries no label.

Outputs
-------
external_data/s4b2/train_index.parquet
reports/s4b2_data_audit.json
reports/s4b2_data_audit.md

Usage
-----
    python3 tools/s4b2_prepare_train_index.py \
        --config agro_jepa/config/s4b2_full_train.yaml
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

TRAITS = ("cc", "ph", "whc", "sen")
# metadata columns the model/dataset actually needs
KEEP = [
    "pair_id", "pair_type", "span_days", "plot_uid", "genotype_id",
    "harvest_year", "split", "image_stem_a", "image_stem_b", "date_a", "date_b",
    "delta_days", "window_n_days", "image_a", "image_b",
]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def index_sha256(df: pd.DataFrame) -> str:
    cols = sorted(df.columns)
    return hashlib.sha256(df[cols].to_csv(index=False).encode()).hexdigest()


def strip_label_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    drop = [c for c in df.columns
            if any(c == t or c.startswith(t + "_") for t in TRAITS)]
    return df.drop(columns=drop), drop


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s4b2_full_train.yaml")
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    d = cfg["data"]
    t0 = time.time()

    pairs = pd.read_parquet(d["full_train_pairs"])
    feat = pd.read_parquet(d["feature_index"])
    feat["_stem"] = feat.image_filename_stem.astype(str).str.upper()
    if feat._stem.duplicated().any():
        raise SystemExit("feature index has duplicate image_filename_stem")
    fmap = feat.drop_duplicates("_stem").set_index("_stem")

    # official train only
    pairs = pairs[pairs.split == "train"].copy()
    pairs["image_stem_a"] = pairs.image_stem_a.astype(str).str.upper()
    pairs["image_stem_b"] = pairs.image_stem_b.astype(str).str.upper()

    ca = pairs.image_stem_a.map(fmap["cache_file"])
    cb = pairs.image_stem_b.map(fmap["cache_file"])
    missing = int(ca.isna().sum() + cb.isna().sum())
    if missing:
        raise SystemExit(f"{missing} train endpoints have no feature cache row")

    pairs["cache_file_a"] = ca.astype(str)
    pairs["cache_file_b"] = cb.astype(str)
    pairs["feature_sha256_a"] = pairs.image_stem_a.map(fmap["sha256"])
    pairs["feature_sha256_b"] = pairs.image_stem_b.map(fmap["sha256"])
    pairs["feature_split_a"] = pairs.image_stem_a.map(fmap["split"])
    pairs["feature_split_b"] = pairs.image_stem_b.map(fmap["split"])

    # strip labels before preserving order
    stripped, dropped = strip_label_columns(pairs)
    for c in KEEP:
        if c not in stripped.columns:
            raise SystemExit(f"required column {c!r} missing after label strip")
    keep_cols = KEEP + ["cache_file_a", "cache_file_b", "feature_sha256_a",
                        "feature_sha256_b", "feature_split_a", "feature_split_b"]
    index = stripped[keep_cols].reset_index(drop=True)  # preserve original order

    # ---- validation ------------------------------------------------------
    cache_paths = pd.concat([index.cache_file_a, index.cache_file_b]).unique()
    missing_files = [p for p in cache_paths if not Path(p).is_file()]
    # shape check on a deterministic sample of all unique caches
    rng = np.random.RandomState(0)
    sample = rng.choice(len(cache_paths), size=min(64, len(cache_paths)), replace=False)
    bad_shape = []
    for i in sample:
        try:
            if np.load(cache_paths[i], mmap_mode="r").shape != (256, 768):
                bad_shape.append(str(cache_paths[i]))
        except Exception as e:  # noqa: BLE001
            bad_shape.append(f"{cache_paths[i]}: {e}")

    train_only = bool((index.split == "train").all())
    feat_splits_ok = bool(
        (index.feature_split_a == "train").all()
        and (index.feature_split_b == "train").all()
    )

    audit = {
        "stage": "S4B.2",
        "task": "dedicated full-official-train index for final training",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": args.config,
        "config_sha256": sha256_file(args.config),
        "inputs": {
            "full_train_pairs": d["full_train_pairs"],
            "full_train_pairs_sha256": sha256_file(d["full_train_pairs"]),
            "feature_index": d["feature_index"],
            "feature_index_sha256": sha256_file(d["feature_index"]),
        },
        "n_pairs": int(len(index)),
        "n_plots": int(index.plot_uid.nunique()),
        "split_values": sorted(index.split.unique().tolist()),
        "train_only": train_only,
        "n_unique_endpoints": int(pd.unique(pd.concat(
            [index.image_stem_a, index.image_stem_b])).size),
        "n_by_pair_type": {str(k): int(v) for k, v in index.pair_type.value_counts().items()},
        "n_by_year": {str(int(k)): int(v) for k, v in
                      index.groupby("harvest_year").size().items()},
        "dropped_label_columns": dropped,
        "demographic_cols_kept": KEEP,
        "cache_coverage": {
            "n_unique_cache_files": int(len(cache_paths)),
            "missing_files": len(missing_files),
            "shape_checked": int(len(sample)),
            "bad_shape": len(bad_shape),
            "feature_split_all_train": feat_splits_ok,
        },
        "index_sha256": index_sha256(index),
        "output": d["train_index"],
        "output_file_sha256": None,
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    out = Path(d["train_index"])
    out.parent.mkdir(parents=True, exist_ok=True)
    index.to_parquet(out, index=False)
    audit["output_file_sha256"] = sha256_file(out)

    ok = (train_only and feat_splits_ok and len(index) == 179823
          and index.plot_uid.nunique() == 2930 and not missing_files
          and not bad_shape)
    audit["checks"] = {
        "n_pairs_179823": len(index) == 179823,
        "plots_2930": int(index.plot_uid.nunique()) == 2930,
        "train_only": train_only,
        "feature_split_all_train": feat_splits_ok,
        "all_cache_files_exist": not missing_files,
        "sample_shapes_ok": not bad_shape,
        "labels_stripped": len(dropped) > 0,
    }
    audit["all_pass"] = ok

    jpath = Path(cfg["reports"]["data_audit_json"])
    jpath.parent.mkdir(parents=True, exist_ok=True)
    with open(jpath, "w") as fh:
        json.dump(audit, fh, indent=2)

    md = [
        "# S4B.2 Data Audit",
        "",
        f"- Generated: {audit['generated_at']}",
        f"- Pairs: **{audit['n_pairs']:,}**  |  plots: **{audit['n_plots']:,}**",
        f"- split values in index: {audit['split_values']}",
        f"- unique endpoints: {audit['n_unique_endpoints']:,}",
        f"- pair types: {audit['n_by_pair_type']}",
        f"- labels stripped ({len(dropped)} cols): {', '.join(dropped[:6])}...",
        f"- cache coverage: {audit['cache_coverage']['n_unique_cache_files']:,} unique files, "
        f"{audit['cache_coverage']['missing_files']} missing, "
        f"{audit['cache_coverage']['bad_shape']} bad shapes / "
        f"{audit['cache_coverage']['shape_checked']} sampled",
        f"- feature split all train: {audit['cache_coverage']['feature_split_all_train']}",
        f"- index sha256: `{audit['index_sha256']}`",
        f"- output: `{audit['output']}` (file sha `{audit['output_file_sha256'][:16]}...`)",
        "",
        f"**all_pass = {ok}**",
    ]
    (Path(cfg["reports"]["data_audit_md"])).write_text("\n".join(md) + "\n")

    print(json.dumps({
        "n_pairs": audit["n_pairs"], "n_plots": audit["n_plots"],
        "train_only": train_only, "cache_missing": len(missing_files),
        "bad_shape": len(bad_shape), "labels_stripped": len(dropped),
        "index_sha256": audit["index_sha256"], "all_pass": ok,
    }, indent=2), flush=True)
    print(f"wrote {out} and {jpath}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
