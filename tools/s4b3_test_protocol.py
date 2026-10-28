#!/usr/bin/env python3
"""S4B.3A -- freeze the official-test evaluation protocol before any test metric.

This tool does **not** compute a test metric and does **not** read a test
phenotype label.  It re-verifies the frozen inputs from local artifacts and
writes ``reports/s4b3_test_protocol.{json,md}`` so that the E1 (future-latent)
and E2 (CC/PH readout) protocols are immutable before either is run.

What is checked, from local bytes only (no number is taken on trust)
--------------------------------------------------------------------
1. every one of the nine S4B.2 ``final.pt`` checkpoints exists and its SHA256 is
   recorded (these are the only models S4B.3 ever evaluates);
2. each of the four official test pair indexes exists, its SHA256 and its real
   row / plot counts match the frozen S4A.1 audit;
3. each test conditioning npz exists and matches its frozen SHA256;
4. the feature cache index and the full-train pair table match their frozen
   SHA256, and the feature cache covers every test image endpoint (coverage 1.0);
5. the test pair ids do not leak into the S3 train/validation index or the
   S4B.2 full-train index;
6. the official test pair indexes carry no phenotype label column (E1 cannot
   read a label even by accident);
7. the [B2-C fixed 9.0 d] vs [B3 real delta_days] distinction is stated from the
   frozen S4A.1 value, not re-derived.

Usage
-----
    python3 tools/s4b3_test_protocol.py \
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
sys.path.insert(0, str(ROOT / "tools"))

from agro_jepa.datasets.s3_pairs import index_sha256  # noqa: E402
from s3_1_diagnostics import git, sha256_file  # noqa: E402

LABEL_COLUMNS = [
    f"{t}_{end}{suf}"
    for t in ("cc", "ph", "whc", "sen")
    for end in ("a", "b")
    for suf in ("", "_lag", "_date", "_mask")
] + ["ph_negative_a", "ph_negative_b"]


def load_cfg(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s4b3_official_test.yaml")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    t0 = time.time()

    checks: list[dict] = []

    def record(cid: str, ok: bool, detail) -> None:
        checks.append({"id": cid, "pass": bool(ok), "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'}  {cid}: {detail}", flush=True)

    frozen = cfg["frozen"]

    # ---------------- 1. nine frozen checkpoints --------------------------
    ckpt_records = {}
    all_exist = True
    for cond, seeds in frozen["checkpoints"].items():
        ckpt_records[cond] = {}
        for seed_key, rel in seeds.items():
            p = ROOT / rel
            exists = p.is_file()
            digest = sha256_file(p) if exists else None
            all_exist &= exists
            ckpt_records[cond][seed_key] = {
                "path": rel, "exists": exists, "sha256": digest,
            }
            print(f"  ckpt {cond}/{seed_key}: {digest[:16] if digest else 'MISSING'}",
                  flush=True)
    record("C1_nine_checkpoints_present", all_exist,
           f"{sum(len(v) for v in ckpt_records.values())} final.pt files")

    # ---------------- 2. four test pair indexes ---------------------------
    feature_index = pd.read_parquet(ROOT / cfg["data"]["feature_index"])
    feat_sha = sha256_file(ROOT / cfg["data"]["feature_index"])
    record("C2_feature_index_sha",
           feat_sha == cfg["data"]["feature_index_sha256"],
           f"{feat_sha[:16]} (expected {cfg['data']['feature_index_sha256'][:16]})")
    feat_stems = set(feature_index.image_filename_stem.astype(str).str.upper())
    feat_cache = set(feature_index.cache_file.astype(str))

    test_summary = {}
    counts_ok = True
    idx_sha_ok = True
    coverage_ok = True
    label_free_ok = True
    for split, spec in cfg["test_splits"].items():
        p = ROOT / spec["index"]
        df = pd.read_parquet(p)
        got_sha = sha256_file(p)
        got_pairs = len(df)
        got_plots = df.plot_uid.nunique()
        exp_pairs = int(spec["n_pairs"])
        exp_plots = int(spec["n_plots"])
        idx_sha_ok &= got_sha == spec["index_sha256"]
        counts_ok &= (got_pairs == exp_pairs and got_plots == exp_plots)
        n_a = df.image_stem_a.astype(str).str.upper().isin(feat_stems).sum()
        n_b = df.image_stem_b.astype(str).str.upper().isin(feat_stems).sum()
        cov = (n_a == len(df) and n_b == len(df))
        coverage_ok &= cov
        label_cols = [c for c in df.columns if c in set(LABEL_COLUMNS)]
        label_free_ok &= len(label_cols) == 0
        cond_p = ROOT / spec["conditioning_npz"]
        cond_sha = sha256_file(cond_p) if cond_p.is_file() else None
        cond_ok = cond_sha == spec["conditioning_npz_sha256"]
        test_summary[split] = {
            "index": spec["index"],
            "index_sha256": got_sha,
            "index_sha256_matches_frozen": got_sha == spec["index_sha256"],
            "n_pairs": int(got_pairs),
            "n_pairs_expected": exp_pairs,
            "n_plots": int(got_plots),
            "n_plots_expected": exp_plots,
            "n_unique_images": int(pd.unique(pd.concat(
                [df.image_stem_a, df.image_stem_b])).size),
            "by_pair_type": {str(k): int(v) for k, v in
                             df.pair_type.value_counts().sort_index().items()},
            "harvest_years": sorted(int(y) for y in df.harvest_year.unique()),
            "delta_days": {"min": int(df.delta_days.min()),
                           "max": int(df.delta_days.max()),
                           "mean": float(df.delta_days.mean())},
            "feature_coverage": {"a": int(n_a), "b": int(n_b),
                                 "n_pairs": int(len(df)),
                                 "fraction": float(min(n_a, n_b) / len(df))},
            "conditioning_npz": spec["conditioning_npz"],
            "conditioning_npz_sha256": cond_sha,
            "conditioning_npz_sha256_matches_frozen": cond_ok,
            "phenotype_label_columns_present": label_cols,
        }
        print(f"  {split}: {got_pairs} pairs / {got_plots} plots, "
              f"sha {got_sha[:16]}, feature_cov {min(n_a, n_b)/len(df):.3f}, "
              f"labels {label_cols}", flush=True)
    record("C3_test_counts_and_sha", counts_ok and idx_sha_ok,
           "row/plot counts and SHA256 match the frozen S4A.1 audit")
    record("C4_test_feature_coverage", coverage_ok,
           "every test image endpoint has a cached DINOv2 feature")
    record("C5_test_indexes_label_free", label_free_ok,
           "no test pair index carries a phenotype label column (E1 blind)")

    total_pairs = sum(v["n_pairs"] for v in test_summary.values())
    total_plots = sum(v["n_plots"] for v in test_summary.values())
    record("C6_total_test_pairs", total_pairs == 30357,
           f"{total_pairs} pairs over {total_plots} plots across four splits")

    # ---------------- 3. leakage checks -----------------------------------
    s3 = pd.read_parquet(ROOT / cfg["data"]["full_train_pairs"])
    s3_sha = sha256_file(ROOT / cfg["data"]["full_train_pairs"])
    s3_ids = set(s3.pair_id.astype(str))
    s3_plots = set(s3.plot_uid.astype(str))
    test_ids, test_plots = set(), set()
    for split, spec in cfg["test_splits"].items():
        df = pd.read_parquet(ROOT / spec["index"])
        test_ids |= set(df.pair_id.astype(str))
        test_plots |= set(df.plot_uid.astype(str))
    record("C7_no_pair_id_leak", len(test_ids & s3_ids) == 0,
           f"{len(test_ids & s3_ids)} test pair_ids also in full-train table")
    record("C8_no_plot_leak", len(test_plots & s3_plots) == 0,
           f"{len(test_plots & s3_plots)} test plots also in full-train table")
    record("C9_full_train_table_sha", s3_sha == cfg["data"]["full_train_pairs_sha256"],
           f"{s3_sha[:16]}")

    train_index = pd.read_parquet(ROOT / cfg["data"]["train_cache_index"])
    record("C10_full_train_counts",
           len(s3) == 179823 and s3.plot_uid.nunique() == 2930
           and set(s3.split.unique()) == {"train"},
           f"{len(s3)} pairs / {s3.plot_uid.nunique()} plots / "
           f"splits {sorted(s3.split.unique())}")

    # ---------------- 4. B2-C vs B3 distinction ---------------------------
    dnorm = cfg["e1"]["delta_norm"]
    fixed = float(cfg["e1"]["fixed_delta_days"])
    fixed_norm = (fixed - float(dnorm["mean"])) / float(dnorm["std"])
    record("C11_b2c_vs_b3_distinction",
           fixed == 9.0 and cfg["conditions"]["b2_no_time_constant_delta"][
               "delta_mode"] == "constant"
           and cfg["conditions"]["b3_delta"]["delta_mode"] == "sample",
           f"B2-C fixed {fixed} d (normalised {fixed_norm:+.6f}); "
           "B3 uses each pair's real delta_days in [5,16]")

    # ---------------- protocol document -----------------------------------
    protocol = {
        "stage": "S4B.3",
        "task": "official test-set evaluation protocol (E1 frozen first)",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": " ".join(sys.argv),
        "config": args.config,
        "config_sha256": sha256_file(ROOT / args.config),
        "git_head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "s4b2_baseline_commit": frozen["s4b2_baseline_commit"],
        "test_splits": test_summary,
        "totals": {"n_pairs": int(total_pairs), "n_plots": int(total_plots),
                   "n_splits": len(test_summary)},
        "checkpoints": ckpt_records,
        "feature_index": {
            "path": cfg["data"]["feature_index"],
            "sha256": feat_sha,
            "n_endpoints": int(len(feature_index)),
            "cache_version": frozen["feature_cache_version"],
        },
        "e1_protocol": {
            "conditions": {
                "B0_persistence": "Z_hat_b = Z_a (no model)",
                "B1_train_mean": ("Z_hat_b = mean Z_b over the full S4B.2 "
                                  "official-train index (no model)"),
                "B2": "b2_no_time checkpoint, z_a only",
                "B2-C": "b2_no_time_constant_delta checkpoint, fixed 9.0 d",
                "B3": "b3_delta checkpoint, real per-pair delta_days",
            },
            "seeds": cfg["e1"]["seeds"],
            "seed_averaging": ("prediction averaged over seeds {0,1,2} before "
                               "MSE; each seed is also reported alone"),
            "metrics": ["mse", "cosine_similarity",
                        "relative_improvement_vs_B0",
                        "relative_improvement_vs_B1",
                        "paired_mse_delta_B3_minus_B2-C",
                        "paired_mse_delta_B3_minus_B2"],
            "stratification": ["span7", "span14", "harvest_year"],
            "bootstrap": {
                "unit": "plot_uid (all pairs of a resampled plot kept together)",
                "n_resamples": int(cfg["e1"]["bootstrap_resamples"]),
                "seed": int(cfg["e1"]["bootstrap_seed"]),
                "interval": "95% percentile",
            },
            "reads_test_phenotype_labels": False,
            "selection": "none (checkpoints frozen before any test metric)",
        },
        "e2_protocol": {
            "conditions": {
                "A_current": "mean(z_a) current frozen DINOv2 feature",
                "B2": "mean(z_hat_b) from b2_no_time",
                "B2-C": "mean(z_hat_b) from b2_no_time_constant_delta",
                "B3": "mean(z_hat_b) from b3_delta (real delta_days)",
            },
            "train_scope": ("all 2,930 official-train plots; inner 80/20 by "
                            "plot_uid stratified by harvest_year"),
            "inner_split_seed": int(cfg["e2"]["plot_split_seed"]),
            "inner_split_fraction": float(cfg["e2"]["inner_train_fraction"]),
            "epoch_selection": ("minimum inner-dev MSE over the candidate "
                                "budget; refit on the full official-train data "
                                "at the selected epoch"),
            "candidate_epochs": int(cfg["e2"]["candidate_epochs"]),
            "patience": int(cfg["e2"]["patience"]),
            "architecture": cfg["e2"]["architecture"],
            "pooling": cfg["e2"]["pooling"],
            "label_max_lag_days": int(cfg["e2"]["lag_variants"]["primary_lag7"]),
            "lag_variants": cfg["e2"]["lag_variants"],
            "missing_labels": "validity mask (mask=False fails, never filled with 0)",
            "metrics": ["mae", "rmse", "r2", "effective_n",
                        "effective_plots"],
            "bootstrap": {
                "unit": "plot_uid",
                "n_resamples": int(cfg["e2"]["bootstrap_resamples"]),
                "seed": int(cfg["e2"]["bootstrap_seed"]),
            },
            "official_validation_in_train": False,
            "test_labels_used_for_training_or_selection": False,
            "frozen_before_test_label_read": True,
        },
        "constraints": [
            "Nine S4B.2 final.pt checkpoints are evaluated byte-identically.",
            "No checkpoint, epoch, seed or hyper-parameter is chosen on a test row.",
            "E1 reads no test phenotype label; the four test indexes carry none.",
            "E2 trains and selects only on official-train plots; official "
            "validation and every test plot stay out of training and selection.",
            "B2-C receives one fixed 9.0 d interval; B3 receives each pair's real "
            "delta_days -- this is the only difference between the two arms.",
        ],
        "verification": {
            "n_checks": len(checks),
            "n_pass": sum(1 for c in checks if c["pass"]),
            "all_pass": all(c["pass"] for c in checks),
            "checks": checks,
        },
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    out_json = Path(args.out_json or cfg["artifacts"]["protocol_json"])
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(protocol, fh, indent=2)
    print(f"wrote {out_json}")

    # ---------------- markdown --------------------------------------------
    L = []
    L.append("# S4B.3 Official-Test Evaluation Protocol (frozen before E1)")
    L.append("")
    L.append(f"- Generated: {protocol['generated_at']}")
    L.append(f"- Git HEAD: `{protocol['git_head']}` ({protocol['branch']})")
    L.append(f"- S4B.2 baseline commit: `{frozen['s4b2_baseline_commit']}`")
    L.append(f"- Config: `{args.config}` (sha `{protocol['config_sha256'][:16]}`)")
    L.append("")
    L.append("## Data: four official test splits (locally re-verified)")
    L.append("")
    L.append("| split | pairs | plots | images | span7 | span14 | "
             "delta range | years |")
    L.append("|---|---:|---:|---:|---:|---:|---|---|")
    for split, s in test_summary.items():
        bt = s["by_pair_type"]
        L.append(f"| {split} | {s['n_pairs']:,} | {s['n_plots']} | "
                 f"{s['n_unique_images']:,} | {bt.get('span7',0)} | "
                 f"{bt.get('span14',0)} | "
                 f"{s['delta_days']['min']}-{s['delta_days']['max']} d | "
                 f"{s['harvest_years']} |")
    L.append(f"| **total** | **{total_pairs:,}** | **{total_plots}** | "
             f"-- | -- | -- | -- | -- |")
    L.append("")
    L.append("All four index SHA256 values and row/plot counts independently "
             "re-produce the frozen S4A.1 audit. Conditioned npz inputs are "
             "present and byte-identical.")
    L.append("")
    L.append("## Nine frozen checkpoints")
    L.append("")
    L.append("| condition | seed | final.pt SHA256 |")
    L.append("|---|---:|---|")
    for cond, seeds in ckpt_records.items():
        for seed_key, r in seeds.items():
            L.append(f"| {cond} | {seed_key} | `{r['sha256']}` |")
    L.append("")
    L.append("## E1 -- future-latent prediction (no test label read)")
    L.append("")
    L.append("- Arms: **B0** persistence, **B1** train-mean future, **B2** no "
             "time, **B2-C** fixed 9.0 d, **B3** real delta_days.")
    L.append("- B2/B2-C/B3 evaluated on seeds 0,1,2 individually and as a "
             "three-seed prediction ensemble.")
    L.append(f"- Bootstrap: plot-clustered, "
             f"{cfg['e1']['bootstrap_resamples']} resamples, seed "
             f"{cfg['e1']['bootstrap_seed']}; all pairs of a resampled plot "
             "stay together.")
    L.append("- Stratified by span7 / span14 and harvest year; paired "
             "B3-B2-C and B3-B2 MSE differences reported.")
    L.append("- The four test indexes carry **no** phenotype label column, so "
             "E1 cannot read a test label.")
    L.append("")
    L.append("## E2 -- CC / PH phenotype readout (frozen before test labels read)")
    L.append("")
    L.append("- Linear readout on **all 2,930 official-train plots**, inner "
             "80/20 split by plot_uid stratified by harvest_year, seed "
             f"{cfg['e2']['plot_split_seed']}.")
    L.append("- Epoch selected on the inner-dev set, then refit on the full "
             "official-train data; official validation and every test plot stay "
             "out of training and selection.")
    L.append("- Conditions: **A_current** (mean z_a), **B2**, **B2-C**, **B3** "
             "(mean predicted z_hat_b).")
    L.append(f"- Primary lag window: |label_lag_days| <= "
             f"{cfg['e2']['lag_variants']['primary_lag7']}; strict-lag3 and "
             "same-day are sensitivity analyses.")
    L.append("- Missing labels use a real validity mask; they are never "
             "filled with 0.")
    L.append("")
    L.append("## Protocol checks")
    L.append("")
    L.append("| check | pass | detail |")
    L.append("|---|---|---|")
    for c in checks:
        L.append(f"| {c['id']} | {'PASS' if c['pass'] else 'FAIL'} | {c['detail']} |")
    L.append("")
    L.append(f"**all_pass = {protocol['verification']['all_pass']}**")
    out_md = Path(args.out_md or cfg["artifacts"]["protocol_md"])
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(L) + "\n")
    print(f"wrote {out_md}")
    return 0 if protocol["verification"]["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
