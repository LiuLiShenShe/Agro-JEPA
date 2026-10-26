#!/usr/bin/env python3
"""S4B.1: build the full official-train span7/span14 pair index.

This tool only reads the frozen S2 pair table (``fip1-s2-v1``) plus the frozen
S1 plots/observations manifests.  It never re-derives the pairing rule: the
span7/span14 semantics are exactly the ones S3 consumed, because S3 and S4B.1
both filter the same ``external_data/pairs_cache/fip1_pairs_s2.parquet``.

What it produces
----------------
* ``external_data/s4b_index/full_train_pairs.parquet`` -- every span7/span14
  pair of every official-train plot (2,930 plots), deduplicated on
  ``(plot_uid, image_stem_a, image_stem_b)`` with the longest span kept.  No
  image-cache or feature-cache filter is applied (S3 required both; that is a
  limitation of the frozen pilot, not of the pairing rule).
* ``reports/s4b1_full_train_stats.json`` -- counts, delta-day distribution,
  year/genotype coverage, rule version and index SHA256.
* ``external_data/s4b_index/s4b1_desired_endpoints.parquet`` -- the union of
  unique image endpoints needed by (a) the full train index and (b) the four
  frozen test pair indexes, with split/plot/date/year metadata.

The four official test splits are *not* rebuilt here: S4A.1 already froze them
in ``external_data/s4b_index/<split>_test_pairs.parquet`` and that file is read
read-only.

Usage
-----
    python3 tools/s4b1_build_full_train_index.py \
        --config agro_jepa/config/s4b1_data_cache.yaml
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

from agro_jepa.datasets.s3_pairs import (  # noqa: E402
    S3IndexRule,
    index_sha256,
    index_statistics,
    load_s2_pairs,
)

TRAITS = ("cc", "ph", "whc", "sen")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dedupe_physical(pairs: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Identical dedupe as ``agro_jepa.datasets.s3_pairs.build_s3_index``."""
    n_before = len(pairs)
    pairs = pairs.sort_values(
        ["plot_uid", "image_stem_a", "image_stem_b", "span_days", "pair_id"],
        kind="mergesort",
    )
    pairs = pairs.drop_duplicates(
        subset=["plot_uid", "image_stem_a", "image_stem_b"], keep="last"
    )
    return pairs, {"rows_before_dedupe": int(n_before), "rows_after_dedupe": int(len(pairs))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s4b1_data_cache.yaml")
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    d = cfg["data"]
    rule_cfg = cfg["pair_rule"]
    t0 = time.time()

    # ---------------- frozen inputs (read-only) ---------------------------
    plots = pd.read_csv(d["plots"], dtype={"plot_uid": str})
    plots["plot_uid"] = plots.plot_uid.astype(str)
    obs = pd.read_csv(
        d["observations"], dtype={"plot_uid": str, "split": str}, low_memory=False
    )
    obs["image_date"] = pd.to_datetime(obs["image_date"])
    obs = obs.drop_duplicates("image_filename_stem").copy()
    obs["image_filename_stem"] = obs.image_filename_stem.astype(str).str.upper()

    # ---------------- full official-train span index ---------------------
    pairs = load_s2_pairs(
        d["s2_pairs"],
        pair_types=tuple(rule_cfg["pair_types"]),
        splits=("train",) if rule_cfg.get("train_split_only", True)
        else ("train", "validation"),
    )
    n_s2_span_train = len(pairs)
    pairs, dedupe_stats = dedupe_physical(pairs)

    # official-train plot coverage (must be complete)
    official_train_plots = set(plots.loc[plots.split == "train", "plot_uid"])
    index_plots = set(pairs.plot_uid.astype(str))
    missing_train_plots = sorted(official_train_plots - index_plots)
    extra_plots = sorted(index_plots - official_train_plots)

    rule = S3IndexRule(
        pair_types=tuple(rule_cfg["pair_types"]),
        splits=("train",),
        dedupe_physical_pairs=bool(rule_cfg.get("dedupe_physical_pairs", True)),
        require_cache=False,
        manifest_id=cfg["manifest_id"],
    )
    stats = index_statistics(pairs, rule)
    stats["rule"]["require_image_cache"] = False
    stats["rule"]["source_pair_table"] = d["s2_pairs"]
    stats["rule"]["source_pair_manifest_id"] = "fip1-s2-v1"
    stats["pairing_note"] = (
        "span7/span14 rows taken verbatim from the frozen S2 table; the same "
        "table S3 filtered.  Span7 delta_days lies in 5-9 d, span14 in 12-16 d, "
        "so the S3 max_delta=16 bound excludes nothing for these families."
    )
    stats["dedupe"] = dedupe_stats
    stats["n_s2_span_train_rows"] = int(n_s2_span_train)
    stats["coverage"] = {
        "official_train_plots": int(len(official_train_plots)),
        "plots_in_index": int(len(index_plots)),
        "missing_train_plots": missing_train_plots,
        "extra_non_train_plots": extra_plots,
        "covers_all_2930": len(missing_train_plots) == 0 and not extra_plots,
        "n_unique_endpoints": int(
            pd.unique(pd.concat([pairs.image_stem_a, pairs.image_stem_b])).size
        ),
        "n_by_pair_type": {
            "span7": int((pairs.pair_type == "span7").sum()),
            "span14": int((pairs.pair_type == "span14").sum()),
        },
        "n_plots_by_year": {
            str(int(k)): int(v)
            for k, v in pairs.groupby("harvest_year").plot_uid.nunique().items()
        },
        "n_plots_by_genotype": int(pairs.genotype_id.nunique()),
    }
    stats["index_sha256"] = index_sha256(pairs)
    stats["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    stats["elapsed_seconds"] = round(time.time() - t0, 1)
    stats["inputs"] = {
        "plots": d["plots"],
        "plots_sha256": sha256_file(d["plots"]),
        "observations": d["observations"],
        "observations_sha256": sha256_file(d["observations"]),
        "s2_pairs": d["s2_pairs"],
        "s2_pairs_sha256": sha256_file(d["s2_pairs"]),
    }

    # ---------------- write full train index -----------------------------
    out = Path(cfg["index"]["full_train_out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    pairs = pairs.sort_values(
        ["split", "pair_type", "delta_days", "pair_id"], kind="mergesort"
    ).reset_index(drop=True)
    pairs.to_parquet(out, index=False)

    sample_path = Path(cfg["index"]["full_train_sample_csv"])
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "pair_id", "plot_uid", "genotype_id", "split", "pair_type", "span_days",
        "delta_days", "date_a", "date_b", "image_stem_a", "image_stem_b",
    ]
    pairs.sample(n=min(200, len(pairs)), random_state=42)[cols].to_csv(
        sample_path, index=False
    )

    # ---------------- desired endpoints (train + 4 test splits) ----------
    meta = obs.set_index("image_filename_stem")
    ep_rows: list[dict] = []

    def add_endpoints(stems, split_label):
        for stem in sorted(set(stems)):
            if stem not in meta.index:
                raise RuntimeError(f"endpoint {stem} absent from observations manifest")
            r = meta.loc[stem]
            ep_rows.append({
                "image_filename_stem": stem,
                "plot_uid": str(r.plot_uid),
                "split": split_label,
                "harvest_year": int(r.harvest_year),
                "image_date": pd.Timestamp(r.image_date).strftime("%Y-%m-%d"),
                "inner_plot_image": str(r.inner_plot_image),
                "source": "train_full_index" if split_label == "train"
                else f"{split_label}_test_index",
            })

    train_stems = pd.concat([pairs.image_stem_a, pairs.image_stem_b]).astype(str).str.upper()
    add_endpoints(train_stems, "train")

    test_meta = {
        "test_plot": {},
        "test_genotype": {},
        "test_environment": {},
        "test_genotype_environment": {},
    }
    test_index_dir = Path(cfg["index"]["test_index_dir"])
    for split in cfg["index"]["test_splits"]:
        p = test_index_dir / f"{split}_test_pairs.parquet"
        if not p.exists():
            raise SystemExit(f"missing frozen test index {p}; cannot prepare cache")
        tdf = pd.read_parquet(p)
        if not (tdf.split == split).all():
            raise SystemExit(f"{p} contains rows outside split {split!r}")
        stems = pd.concat([tdf.image_stem_a, tdf.image_stem_b]).astype(str).str.upper()
        add_endpoints(stems, split)
        test_meta[split] = {
            "index_file": str(p),
            "index_sha256": sha256_file(p),
            "n_pairs": int(len(tdf)),
            "n_unique_endpoints": int(stems.nunique()),
        }

    ep = pd.DataFrame(ep_rows).drop_duplicates("image_filename_stem")
    # a stem must belong to exactly one split
    dup = ep[ep.image_filename_stem.duplicated(keep=False)]
    if len(dup):
        raise RuntimeError(
            f"{len(dup)} stems map to multiple splits: "
            f"{dup[['image_filename_stem','split']].head().to_dict('records')}"
        )
    ep_out = Path(cfg["index"]["endpoints_out"])
    ep_out.parent.mkdir(parents=True, exist_ok=True)
    ep = ep.sort_values(["split", "plot_uid", "image_date", "image_filename_stem"],
                        kind="mergesort").reset_index(drop=True)
    ep.to_parquet(ep_out, index=False)

    stats["test_splits"] = test_meta
    stats["desired_endpoints"] = {
        "total": int(len(ep)),
        "by_split": {str(k): int(v) for k, v in ep.split.value_counts().items()},
        "by_year": {str(int(k)): int(v) for k, v in ep.harvest_year.value_counts().sort_index().items()},
        "endpoints_file": str(ep_out),
        "endpoints_sha256": sha256_file(ep_out),
    }

    stats_path = Path(cfg["index"]["full_train_stats"])
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as fh:
        json.dump(stats, fh, indent=2)

    print(json.dumps({
        "n_pairs": stats["n_pairs"],
        "n_plots": stats["coverage"]["plots_in_index"],
        "covers_all_2930": stats["coverage"]["covers_all_2930"],
        "n_unique_endpoints": stats["coverage"]["n_unique_endpoints"],
        "n_by_pair_type": stats["coverage"]["n_by_pair_type"],
        "desired_endpoints_total": stats["desired_endpoints"]["total"],
        "test_splits": {k: v["n_unique_endpoints"] for k, v in test_meta.items()},
        "index_sha256": stats["index_sha256"],
    }, indent=2), flush=True)
    print(f"wrote {out}, {ep_out}, {stats_path}", flush=True)
    return 0 if stats["coverage"]["covers_all_2930"] else 1


if __name__ == "__main__":
    sys.exit(main())
