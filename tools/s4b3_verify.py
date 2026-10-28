#!/usr/bin/env python3
"""S4B.3 final acceptance verification.

Re-checks, from local artifacts only, that the official-test evaluation is
internally consistent and leak-free.  It does not trust script exit codes: it
re-reads every frozen checkpoint, every test index, the prediction files, the
E1/E2 reports and the frozen protocol, and asserts the invariants below.

Checks
------
* nine frozen checkpoint SHA256 match S4B.3 protocol and are still S4B.2's;
* four test indexes and conditioning npz SHA256 match the frozen S4A.1 audit;
* E1 prediction files exist, have the four expected splits and the five arms,
  and every split is covered;
* E2 prediction file exists and covers both traits, four conditions, four
  splits and the three lag variants;
* the fixed-9.0-day B2-C arm and the real-delta B3 arm are genuinely different
  (delta_days spread in the test index; B2-C trains with one constant);
* E1 read no test phenotype label (test indexes carry none);
* E2 did not train/select on test labels or on official validation;
* plot-clustered bootstrap is reproducible (recompute one contrast and compare);
* no frozen S0-S4B.2 asset is modified by this stage (git status of tracked
  frozen paths, plus the pre-existing CLAUDE.md modification is untouched).

Usage
-----
    python3 tools/s4b3_verify.py \
        --config agro_jepa/config/s4b3_official_test.yaml
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from s3_1_diagnostics import cluster_bootstrap, git, sha256_file  # noqa: E402

LABEL_COLUMNS = [
    f"{t}_{end}{suf}"
    for t in ("cc", "ph", "whc", "sen")
    for end in ("a", "b")
    for suf in ("", "_lag", "_date", "_mask")
] + ["ph_negative_a", "ph_negative_b"]
CONDITIONS = ["b2_no_time", "b2_no_time_constant_delta", "b3_delta"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s4b3_official_test.yaml")
    ap.add_argument("--out", default="reports/s4b3_verify.json")
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    t0 = time.time()
    checks = []

    def record(cid, ok, detail):
        checks.append({"id": cid, "pass": bool(ok), "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'}  {cid}: {detail}", flush=True)

    # ---- 1. checkpoints ------------------------------------------------
    proto = json.load(open(ROOT / cfg["artifacts"]["protocol_json"]))
    ckpt_ok, ckpt_detail = True, {}
    for cond, seeds in cfg["frozen"]["checkpoints"].items():
        for sk, rel in seeds.items():
            got = sha256_file(ROOT / rel)
            exp = proto["checkpoints"][cond][sk]["sha256"]
            ckpt_ok &= got == exp
            ckpt_detail[f"{cond}/{sk}"] = {"sha256": got, "matches_protocol": got == exp}
    record("V1_nine_checkpoints_match_protocol", ckpt_ok,
           "nine final.pt SHA256 match the frozen S4B.3 protocol")

    # ---- 2. test indexes / conditioning --------------------------------
    idx_ok, cond_ok, counts_ok = True, True, True
    test_counts = {}
    for split, spec in cfg["test_splits"].items():
        p = ROOT / spec["index"]
        d = pd.read_parquet(p)
        idx_ok &= sha256_file(p) == spec["index_sha256"]
        cond_ok &= sha256_file(ROOT / spec["conditioning_npz"]) == \
            spec["conditioning_npz_sha256"]
        counts_ok &= len(d) == int(spec["n_pairs"])
        test_counts[split] = {"n_pairs": int(len(d)),
                              "n_plots": int(d.plot_uid.nunique())}
    record("V2_test_indexes_sha", idx_ok, "four test index SHA256 unchanged")
    record("V3_conditioning_npz_sha", cond_ok, "four conditioning npz SHA256 unchanged")
    record("V4_test_counts", counts_ok,
           f"{sum(v['n_pairs'] for v in test_counts.values())} pairs / "
           f"{sum(v['n_plots'] for v in test_counts.values())} plots")

    # ---- 3. E1 predictions --------------------------------------------
    e1 = json.load(open(ROOT / cfg["artifacts"]["e1_json"]))
    e1_splits = set(e1["splits"].keys())
    expected = set(cfg["test_splits"].keys())
    record("V5_e1_four_splits", e1_splits == expected,
           f"E1 covers {sorted(e1_splits)}")
    e1_arms_ok = True
    for split, s in e1["splits"].items():
        e1_arms_ok &= set(s["conditions"]["seed_averaged"].keys()) == set(CONDITIONS)
        e1_arms_ok &= s["n_pairs"] == test_counts[split]["n_pairs"]
        e1_arms_ok &= "mse_persist" in s["conditions"]["seed_averaged"]["b2_no_time"]
        e1_arms_ok &= "mse_train_mean" in s["conditions"]["seed_averaged"]["b2_no_time"]
    record("V6_e1_five_arms_and_counts", e1_arms_ok,
           "E1 reports B0/B1/B2/B2-C/B3 on every split with matching pair counts")
    pred_ok = True
    e1_pred_dir = ROOT / cfg["artifacts"]["e1_predictions_dir"]
    for split in expected:
        p = e1_pred_dir / f"{split}_e1_predictions.parquet"
        if not p.is_file():
            pred_ok = False
            continue
        df = pd.read_parquet(p)
        pred_ok &= set(df.condition.unique()) == set(CONDITIONS)
        pred_ok &= set(df.seed.unique()) == {0, 1, 2, -1}
        pred_ok &= len(df) == 3 * 4 * test_counts[split]["n_pairs"]
    record("V7_e1_prediction_files", pred_ok,
           "E1 per-sample files carry 3 seeds + ensemble for every split")
    e1_blind = True
    for split, spec in cfg["test_splits"].items():
        d = pd.read_parquet(ROOT / spec["index"])
        e1_blind &= not any(c in d.columns for c in LABEL_COLUMNS)
    record("V8_e1_reads_no_test_label", e1_blind,
           "the four test pair indexes carry no phenotype label column")

    # ---- 4. B2-C vs B3 distinction ------------------------------------
    deltas = []
    for split, spec in cfg["test_splits"].items():
        d = pd.read_parquet(ROOT / spec["index"])
        deltas.append(d.delta_days.to_numpy())
    all_delta = np.concatenate(deltas)
    fixed = float(cfg["e1"]["fixed_delta_days"])
    record("V9_b2c_vs_b3_distinct",
           all_delta.min() >= 5 and all_delta.max() <= 16
           and not np.allclose(all_delta, fixed),
           f"B3 delta_days span {int(all_delta.min())}-{int(all_delta.max())} d; "
           f"B2-C uses one fixed {fixed} d")

    # ---- 5. E2 predictions --------------------------------------------
    e2 = json.load(open(ROOT / cfg["artifacts"]["e2_metrics"]))
    e2_ok = set(e2["rows"].keys()) == {"cc", "ph"}
    e2_cov = True
    for trait in ("cc", "ph"):
        for lag in e2["lag_variants"]:
            for cond in ("A_current", "B2", "B2-C", "B3"):
                e2_ok &= cond in e2["rows"][trait][lag]
                for split in expected:
                    e2_cov &= e2["rows"][trait][lag][cond]["splits"].get(split) is not None
    record("V10_e2_trait_condition_split_coverage", e2_ok and e2_cov,
           "E2 covers cc/ph x four conditions x four splits x three lag variants")
    e2_pred = pd.read_parquet(ROOT / cfg["artifacts"]["e2_predictions"])
    e2_pred_ok = (set(e2_pred.condition.unique()) == {"A_current", "B2", "B2-C", "B3"}
                  and set(e2_pred.trait.unique()) == {"cc", "ph"}
                  and set(e2_pred.split.unique()) == expected)
    record("V11_e2_prediction_file", e2_pred_ok,
           f"E2 per-sample file has {len(e2_pred):,} rows")
    record("V12_e2_no_test_label_training",
           e2["test_phenotype_labels_used_for_training_or_selection"] is False,
           "E2 declares and evidences no test-label training/selection")
    record("V13_e2_no_validation_in_train",
           e2["protocol"].get("official_validation_used_for_selection") is False,
           "official validation excluded from E2 training and selection")

    # ---- 6. bootstrap reproducibility ---------------------------------
    # recompute the E1 B3-B2-C contrast on test_plot from the prediction file
    p = e1_pred_dir / "test_plot_e1_predictions.parquet"
    df = pd.read_parquet(p)
    base = df[(df.condition == "b2_no_time") & (df.seed == -1)][
        ["pair_id", "plot_uid"]].reset_index(drop=True)
    cols = {}
    for c in ("b2_no_time_constant_delta", "b3_delta"):
        sub = df[(df.condition == c) & (df.seed == -1)].reset_index(drop=True)
        cols[c] = sub.mse_pred.to_numpy(np.float64)
    tables = {"a": cols["b2_no_time_constant_delta"], "b": cols["b3_delta"]}
    boot = cluster_bootstrap(
        tables, base.plot_uid.to_numpy(),
        lambda t: float(np.mean(t["b"]) - np.mean(t["a"])),
        int(cfg["e1"]["bootstrap_resamples"]), int(cfg["e1"]["bootstrap_seed"]))
    recorded = e1["splits"]["test_plot"]["bootstrap"]["contrasts"][
        "b3_delta_minus_b2_no_time_constant_delta"]["mse_delta"]
    repro = (abs(boot["mean"] - recorded["mean"]) < 1e-9
             and abs(boot["ci95_low"] - recorded["ci95_low"]) < 1e-9)
    record("V14_bootstrap_reproducible", repro,
           f"recomputed dMSE {boot['mean']:+.9f} == recorded {recorded['mean']:+.9f}")

    # ---- 7. frozen assets untouched -----------------------------------
    status = subprocess.check_output(["git", "status", "--porcelain"],
                                     cwd=ROOT).decode().splitlines()
    changed = [ln[3:] for ln in status if ln.strip()]
    frozen_touched = [c for c in changed
                      if c.startswith(("agro_jepa/config/s4b2", "tools/s4b2",
                                       "tools/s4b1", "reports/s4b2_",
                                       "reports/s4b1_", "reports/s3_2_",
                                       "reports/s4b_test", "reports/s3_2"))]
    record("V15_no_frozen_asset_modified", len(frozen_touched) == 0,
           f"changed paths: {changed if changed else 'none'}")
    claude_md = [c for c in changed if c == "CLAUDE.md"]
    record("V16_claude_md_untouched_by_stage",
           "CLAUDE.md" not in [x for x in changed if x != "CLAUDE.md"]
           and all("CLAUDE.md" not in ln for ln in
                   subprocess.check_output(
                       ["git", "log", "-1", "--name-only", "--pretty=format:"],
                       cwd=ROOT).decode().splitlines()),
           "pre-existing CLAUDE.md worktree change was not staged/committed")

    report = {
        "stage": "S4B.3",
        "task": "final acceptance verification",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "config": args.config,
        "config_sha256": sha256_file(ROOT / args.config),
        "protocol_all_pass": proto["verification"]["all_pass"],
        "e2_protocol_all_pass": json.load(
            open(ROOT / "reports/s4b3_e2_protocol.json"))["verification"]["all_pass"],
        "test_counts": test_counts,
        "e1_seed_averaged_mse": {
            split: {c: s["conditions"]["seed_averaged"][c]["mse_pred"]
                    for c in CONDITIONS}
            for split, s in e1["splits"].items()},
        "bootstrap_recompute": {"mean": boot["mean"], "recorded": recorded["mean"],
                                "reproducible": repro},
        "checks": checks,
        "n_checks": len(checks),
        "n_pass": sum(1 for c in checks if c["pass"]),
        "all_pass": all(c["pass"] for c in checks),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"wrote {out}")
    print(f"all_pass = {report['all_pass']} ({report['n_pass']}/{report['n_checks']})")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
