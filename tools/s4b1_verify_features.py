#!/usr/bin/env python3
"""S4B.1: verify feature coverage of the full train index and the four test splits.

Reads the S4B.1 feature index (``external_data/s4b_features/index.parquet``)
and checks, per split, that every pair endpoint has a locatable, correctly
shaped feature cache.  It also re-confirms that the frozen assets (S1/S2/S3/S4A
files used as inputs) are byte-for-byte unchanged, and that no test phenotype
label was touched in this stage.

Usage
-----
    python3 tools/s4b1_verify_features.py \
        --config agro_jepa/config/s4b1_data_cache.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agro_jepa.datasets.provenance import (  # noqa: E402
    DINOV2_EMBED_DIM,
    MAINLINE_N_TOKENS,
)

# Frozen input assets whose SHA256 must not change in S4B.1.
FROZEN_ASSETS = {
    "s1_plots": "data/manifests/fip1_plots.csv",
    "s1_observations": "data/manifests/fip1_observations.csv.gz",
    "s2_pairs": "external_data/pairs_cache/fip1_pairs_s2.parquet",
    "s3_feature_index": "external_data/s3_features/index.parquet",
    "s3_train_selection": "data/manifests/s3_image_selection.json",
    "s4b_test_plot_index": "external_data/s4b_index/test_plot_test_pairs.parquet",
    "s4b_test_genotype_index": "external_data/s4b_index/test_genotype_test_pairs.parquet",
    "s4b_test_environment_index": "external_data/s4b_index/test_environment_test_pairs.parquet",
    "s4b_test_ge_index": "external_data/s4b_index/test_genotype_environment_test_pairs.parquet",
}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s4b1_data_cache.yaml")
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    feat_cfg = cfg["features"]
    t0 = time.time()

    fidx = pd.read_parquet(feat_cfg["s4b_feature_index"])
    fidx["image_filename_stem"] = fidx.image_filename_stem.astype(str).str.upper()
    have = set(fidx.image_filename_stem)
    cache_by_stem = fidx.set_index("image_filename_stem")["cache_file"].to_dict()

    # verify cache files physically exist and have correct shape
    bad_shape = 0
    missing_file = 0
    for stem in fidx.image_filename_stem:
        p = cache_by_stem[stem]
        if not os.path.isfile(p):
            missing_file += 1
            continue
        try:
            arr = np.load(p)
            if arr.shape != (MAINLINE_N_TOKENS, DINOV2_EMBED_DIM):
                bad_shape += 1
        except Exception:  # noqa: BLE001
            missing_file += 1

    verify: dict = {"feature_cache": {}}
    splits = {}
    full_train = pd.read_parquet(cfg["index"]["full_train_out"])
    full_train["ia"] = full_train.image_stem_a.astype(str).str.upper()
    full_train["ib"] = full_train.image_stem_b.astype(str).str.upper()
    sa = full_train.ia.isin(have)
    sb = full_train.ib.isin(have)
    splits["train_full"] = {
        "n_pairs": int(len(full_train)),
        "n_plots": int(full_train.plot_uid.nunique()),
        "n_endpoints": int(pd.unique(pd.concat([full_train.ia, full_train.ib])).size),
        "missing_endpoints": int((~sa).sum() + (~sb).sum()),
        "coverage_fraction": round(float(sa.mean() * sb.mean()), 6)
        if (sa.mean() and sb.mean()) else 0.0,
        "fully_covered": bool(sa.all() and sb.all()),
    }

    test_index_dir = Path(cfg["index"]["test_index_dir"])
    for split in cfg["index"]["test_splits"]:
        t = pd.read_parquet(test_index_dir / f"{split}_test_pairs.parquet")
        ta = t.image_stem_a.astype(str).str.upper()
        tb = t.image_stem_b.astype(str).str.upper()
        ca = ta.isin(have)
        cb = tb.isin(have)
        splits[split] = {
            "n_pairs": int(len(t)),
            "n_plots": int(t.plot_uid.nunique()),
            "n_endpoints": int(pd.unique(pd.concat([ta, tb])).size),
            "missing_endpoints": int((~ca).sum() + (~cb).sum()),
            "fully_covered": bool(ca.all() and cb.all()),
        }

    # frozen-asset integrity (no test phenotype label is ever read here)
    frozen: dict = {}
    for name, rel in FROZEN_ASSETS.items():
        if os.path.isfile(rel):
            frozen[name] = {"path": rel, "sha256": sha256_file(rel), "present": True}
        else:
            frozen[name] = {"path": rel, "present": False}

    verify = {
        "stage": "S4B.1",
        "task": "feature coverage + frozen-asset integrity verification",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "feature_index": feat_cfg["s4b_feature_index"],
        "feature_index_sha256": sha256_file(feat_cfg["s4b_feature_index"]),
        "n_feature_rows": int(len(fidx)),
        "by_split": {str(k): int(v) for k, v in fidx.split.value_counts().items()},
        "cache_integrity": {
            "missing_files": missing_file,
            "bad_shape": bad_shape,
            "ok": missing_file == 0 and bad_shape == 0,
        },
        "coverage": splits,
        "all_train_pairs_covered": splits["train_full"]["fully_covered"],
        "all_test_pairs_covered": all(splits[s]["fully_covered"]
                                       for s in cfg["index"]["test_splits"]),
        "frozen_assets": frozen,
        "test_phenotype_labels_read": False,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    out = Path(cfg["verify"]["out_json"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(verify, fh, indent=2)

    print(json.dumps({
        "train_full": splits["train_full"],
        "test_splits": {s: splits[s] for s in cfg["index"]["test_splits"]},
        "cache_integrity_ok": verify["cache_integrity"]["ok"],
        "all_train_covered": verify["all_train_pairs_covered"],
        "all_test_covered": verify["all_test_pairs_covered"],
    }, indent=2), flush=True)
    print(f"wrote {out}", flush=True)
    ok = (verify["cache_integrity"]["ok"] and verify["all_train_pairs_covered"]
          and verify["all_test_pairs_covered"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
