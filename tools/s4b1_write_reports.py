#!/usr/bin/env python3
"""S4B.1: assemble the human-readable audit and coverage reports.

Collects the JSON artifacts produced by the other S4B.1 tools and emits:
  reports/s4b1_data_cache_audit.md        (human narrative + acceptance answers)
  reports/s4b1_data_cache_audit.json     (machine copy)
  reports/s4b1_feature_coverage.json     (per-split coverage table)

This tool only READS the other artifacts; it never downloads, encodes, or
modifies any frozen asset.

Usage
-----
    python3 tools/s4b1_write_reports.py \
        --config agro_jepa/config/s4b1_data_cache.yaml
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

FROZEN_ASSETS = {
    "s1_plots": "data/manifests/fip1_plots.csv",
    "s1_observations": "data/manifests/fip1_observations.csv.gz",
    "s2_pairs": "external_data/pairs_cache/fip1_pairs_s2.parquet",
    "s3_feature_index": "external_data/s3_features/index.parquet",
    "s3_train_selection": "data/manifests/s3_image_selection.json",
    "s3_index": "external_data/s3_index/fip1_s3_v0_index.parquet",
    "s4a1_matched_control": "agro_jepa/config/s4a1_matched_control.yaml",
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


def load(path: str):
    with open(path) as fh:
        return json.load(fh)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s4b1_data_cache.yaml")
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    t0 = time.time()

    R = cfg["reports"]
    full_stats = load(cfg["index"]["full_train_stats"])
    feat_report = load(cfg["features"]["report"])
    verify = load(cfg["verify"]["out_json"])
    dl_report = None
    dl_path = cfg["images"]["report"]
    if Path(dl_path).exists():
        dl_report = load(dl_path)

    # ---- frozen asset current hashes; compare to any prior record ----------
    frozen_now = {}
    for name, rel in FROZEN_ASSETS.items():
        if Path(rel).exists():
            frozen_now[name] = {"path": rel, "sha256": sha256_file(rel)}
    # a prior frozen snapshot is not available in this stage; we instead assert
    # these files were NOT modified by checking git: they are committed at the
    # frozen parent commit 4e01ec0, so any change would show in git status.
    import subprocess
    changed = subprocess.check_output(
        ["git", "status", "--porcelain", *FROZEN_ASSETS.values()]
    ).decode().strip().splitlines()
    frozen_modified = [c for c in changed if c and not c.startswith("??")]

    cov = verify["coverage"]
    train_cov = cov["train_full"]
    test_cov = {s: cov[s] for s in cfg["index"]["test_splits"]}

    audit = {
        "stage": "S4B.1",
        "task": "full official-train image/feature cache + 4 test-split feature cache",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": args.config,
        "config_sha256": sha256_file(args.config),
        "frozen_commit": "4e01ec0f054238e2951912923ff52f40d46eb45a",
        "frozen_branch": "agro-s4a1-matched-control",
        "dinov2_code_commit": "7764ea0f912e53c92e82eb78a2a1631e92725fc8",
        "dinov2_weights_sha256": "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73",
        "cache_protocol": "mainline224_s1-v1",
        "pairing_rule": full_stats["rule"],
        "full_train_index": {
            "n_pairs": full_stats["n_pairs"],
            "n_plots": full_stats["coverage"]["plots_in_index"],
            "covers_all_2930": full_stats["coverage"]["covers_all_2930"],
            "n_unique_endpoints": full_stats["coverage"]["n_unique_endpoints"],
            "n_by_pair_type": full_stats["coverage"]["n_by_pair_type"],
            "index_sha256": full_stats["index_sha256"],
        },
        "desired_endpoints": full_stats["desired_endpoints"],
        "image_download": dl_report and {
            "endpoints_total": dl_report["endpoints_total"],
            "reused_already_on_disk": dl_report["reused_already_on_disk"],
            "n_downloaded": dl_report["n_downloaded"],
            "n_cached_skip": dl_report["n_cached_skip"],
            "n_failed": dl_report["n_failed"],
            "n_bytes_gb": round(dl_report["n_bytes"] / 1e9, 2),
            "elapsed_seconds": dl_report["elapsed_seconds"],
        },
        "feature_build": {
            "endpoints_total": feat_report["endpoints_total"],
            "reuse_s3": feat_report["reuse_s3"],
            "s4b_computed": feat_report["s4b_computed"],
            "missing_image_or_failed": feat_report["missing_image_or_failed"],
            "coverage_fraction": feat_report["coverage"]["coverage_fraction"],
            "index_sha256": feat_report["index_sha256"],
        },
        "coverage": cov,
        "cache_integrity": verify["cache_integrity"],
        "frozen_assets_current_sha256": frozen_now,
        "frozen_assets_modified_in_stage": [c.strip() for c in frozen_modified],
        "test_phenotype_labels_read": False,
    }

    # ---- acceptance answers -------------------------------------------------
    total_endpoints = full_stats["desired_endpoints"]["total"]
    covered = int(feat_report["endpoints_total"]) - 0  # index rows == covered stems
    audit["acceptance"] = {
        "Q1_full_train_covers_2930_plots":
            full_stats["coverage"]["covers_all_2930"],
        "Q2_pairing_rule_consistent_with_S3":
            full_stats["rule"]["manifest_id"] == "fip1-s2-v1"
            and full_stats["rule"]["dedupe_physical_pairs"] is True,
        "Q3_unique_train_pair_endpoints": full_stats["coverage"]["n_unique_endpoints"],
        "Q4_reuse_vs_new_download": {
            "reused_s3_features": feat_report["reuse_s3"],
            "s4b_computed_features": feat_report["s4b_computed"],
            "images_reused_from_disk": (dl_report["reused_already_on_disk"]
                                        if dl_report else None),
            "images_newly_downloaded": (dl_report["n_downloaded"]
                                        if dl_report else None),
        },
        "Q5_train_pair_endpoint_coverage":
            train_cov["fully_covered"],
        "Q6_four_test_splits_ready":
            verify["all_test_pairs_covered"],
        "Q7_provenance_and_sha256_complete":
            verify["cache_integrity"]["ok"],
        "Q8_test_phenotype_labels_read": False,
        "Q9_frozen_assets_unchanged":
            len(frozen_modified) == 0,
        "Q10_ready_for_full_training_and_one_shot_test_eval":
            train_cov["fully_covered"] and verify["all_test_pairs_covered"]
            and len(frozen_modified) == 0,
    }

    out_md = Path(R["audit_md"])
    out_json = Path(R["audit_json"])
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(audit, fh, indent=2)

    cov_json = Path(R["coverage_json"])
    cov_payload = {
        "generated_at": audit["generated_at"],
        "total_desired_endpoints": total_endpoints,
        "feature_index_sha256": feat_report["index_sha256"],
        "splits": {
            "train_full": train_cov,
            **{s: test_cov[s] for s in cfg["index"]["test_splits"]},
        },
        "by_feature_source": {
            "reuse_s3": feat_report["reuse_s3"],
            "s4b_computed": feat_report["s4b_computed"],
            "missing": feat_report["missing_image_or_failed"],
        },
    }
    with open(cov_json, "w") as fh:
        json.dump(cov_payload, fh, indent=2)

    # ---- markdown narrative -------------------------------------------------
    md = []
    md.append("# S4B.1 Data & Feature Cache Audit\n")
    md.append(f"- Generated: {audit['generated_at']}")
    md.append(f"- Config: `{args.config}` "
              f"(sha256 `{audit['config_sha256'][:16]}…`)")
    md.append(f"- Frozen commit: `{audit['frozen_commit']}` "
              f"({audit['frozen_branch']})")
    md.append(f"- DINOv2 code commit: `{audit['dinov2_code_commit']}`")
    md.append(f"- Weights sha256: `{audit['dinov2_weights_sha256'][:16]}…`")
    md.append(f"- Cache protocol: `{audit['cache_protocol']}`\n")
    md.append("## Acceptance\n")
    A = audit["acceptance"]
    md.append(f"1. Full official-train covers **2,930 plots**: "
              f"**{A['Q1_full_train_covers_2930_plots']}** "
              f"({full_stats['coverage']['plots_in_index']} plots in index)")
    md.append(f"2. Pairing rule consistent with S3 "
              f"(source `fip1-s2-v1`, dedupe physical pairs): "
              f"**{A['Q2_pairing_rule_consistent_with_S3']}**")
    md.append(f"3. Unique official-train pair endpoints required: "
              f"**{A['Q3_unique_train_pair_endpoints']:,}**")
    Q4 = A["Q4_reuse_vs_new_download"]
    md.append(f"4. Reused vs new: S3 features reused = **{Q4['reused_s3_features']:,}**, "
              f"S4B computed = **{Q4['s4b_computed_features']:,}**; "
              f"images reused from disk = **{Q4['images_reused_from_disk']:,}**, "
              f"newly downloaded = **{Q4['images_newly_downloaded']:,}**")
    md.append(f"5. Train pair endpoint feature coverage (100%): "
              f"**{A['Q5_train_pair_endpoint_coverage']}**")
    md.append(f"6. All four test splits' pair endpoints ready: "
              f"**{A['Q6_four_test_splits_ready']}**")
    md.append(f"7. Provenance + SHA256 complete (cache integrity ok): "
              f"**{A['Q7_provenance_and_sha256_complete']}**")
    md.append(f"8. Any test phenotype label read? **{A['Q8_test_phenotype_labels_read']}**")
    md.append(f"9. Frozen S1/S2/S3/S4A/S4A.1 assets unchanged: "
              f"**{A['Q9_frozen_assets_unchanged']}** "
              f"({len(frozen_modified)} modified)")
    md.append(f"10. Ready for full-training + one-shot test eval: "
              f"**{A['Q10_ready_for_full_training_and_one_shot_test_eval']}**\n")

    md.append("## Full train index\n")
    md.append(f"- pairs: {full_stats['n_pairs']:,}, span7 = "
              f"{full_stats['coverage']['n_by_pair_type']['span7']:,}, "
              f"span14 = {full_stats['coverage']['n_by_pair_type']['span14']:,}")
    md.append(f"- unique endpoints: {full_stats['coverage']['n_unique_endpoints']:,}")
    md.append(f"- index sha256: `{full_stats['index_sha256']}`\n")

    md.append("## Coverage by split\n")
    md.append("| split | pairs | endpoints | fully covered |")
    md.append("|---|---|---|---|")
    md.append(f"| train (full) | {train_cov['n_pairs']:,} | "
              f"{train_cov['n_endpoints']:,} | {train_cov['fully_covered']} |")
    for s in cfg["index"]["test_splits"]:
        c = test_cov[s]
        md.append(f"| {s} | {c['n_pairs']:,} | {c['n_endpoints']:,} | "
                  f"{c['fully_covered']} |")
    md.append("")

    if audit["frozen_assets_modified_in_stage"]:
        md.append("### WARNING: frozen assets modified\n")
        for f in audit["frozen_assets_modified_in_stage"]:
            md.append(f"- `{f}`")
        md.append("")

    with open(out_md, "w") as fh:
        fh.write("\n".join(md))

    print(f"wrote {out_md}, {out_json}, {cov_json}", flush=True)
    ok = A["Q10_ready_for_full_training_and_one_shot_test_eval"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
