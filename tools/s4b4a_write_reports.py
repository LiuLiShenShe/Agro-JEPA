#!/usr/bin/env python3
"""S4B.4A Phase 4: results interpretation and provenance reports.

Reads the real E1 / E2 / protocol / audit JSON products and derives the
cross-fold decision table by fixed rules (no hand-entered numbers).  Writes:

* ``reports/s4b4a_results.json`` and ``reports/s4b4a_results.md``
* ``reports/s4b4a_provenance.md``

The decision rules are pre-registered: a condition "helps" on a fold when the
paired plot-clustered bootstrap 95% CI of its MSE (or R2) delta lies entirely on
the improving side; a cross-fold claim requires the majority of folds to agree
AND the per-seed spread to be smaller than the effect.  Negative results are
retained verbatim.

Usage
-----
    python3 tools/s4b4a_write_reports.py \
        --config agro_jepa/config/s4b4a_year_heldout.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.load(open(path))


def e1_decisions(e1: dict) -> dict:
    """Cross-fold summary of each pre-registered E1 contrast."""
    out = {}
    if not e1:
        return out
    labels = None
    for fold, block in e1["folds"].items():
        if labels is None:
            labels = list(block["bootstrap"]["contrasts"])
    for label in labels or []:
        rows = []
        for fold, block in e1["folds"].items():
            c = block["bootstrap"]["contrasts"].get(label)
            if c is None:
                continue
            v = c["mse_delta"]
            ps = block["per_seed_contrasts"][label]["across_seeds"]
            rows.append({
                "fold": int(fold),
                "mean": float(v["mean"]),
                "ci_low": float(v["ci95_low"]),
                "ci_high": float(v["ci95_high"]),
                "ci_excludes_zero": bool(v["ci95_low"] > 0 or v["ci95_high"] < 0),
                "improves": bool(v["ci95_high"] < 0),
                "per_seed_mean": float(ps["mean"]),
                "per_seed_std": float(ps["std"]),
                "per_seed_same_sign": bool(ps["all_same_sign"]),
            })
        n = len(rows)
        n_improve = sum(r["improves"] for r in rows)
        n_ci = sum(r["ci_excludes_zero"] for r in rows)
        mean_effect = sum(r["mean"] for r in rows) / n if n else float("nan")
        stable = bool(n and n_improve > n / 2 and n_ci == n)
        out[label] = {
            "folds": rows,
            "n_folds": n,
            "n_folds_ci_lower_than_zero": n_improve,
            "n_folds_ci_excludes_zero": n_ci,
            "mean_dmse_over_folds": mean_effect,
            "stable_across_folds": stable,
        }
    return out


def e2_decisions(e2: dict) -> dict:
    out = {}
    if not e2:
        return out
    for trait in ("cc", "ph"):
        out[trait] = {}
        for fold, block in e2["folds"].items():
            lag_block = block["traits"].get(trait, {}).get("primary_lag7")
            if not lag_block or "arms" not in lag_block:
                continue
            arms = lag_block["arms"]
            contrasts = lag_block["contrasts"]
            out[trait][fold] = {
                "n_heldout": lag_block["n_heldout"],
                "r2": {a: arms[a]["r2"] for a in arms},
                "mae": {a: arms[a]["mae"] for a in arms},
                "contrasts": {
                    lab: {"mean": c["delta_r2"]["mean"],
                          "ci_low": c["delta_r2"]["ci95_low"],
                          "ci_high": c["delta_r2"]["ci95_high"]}
                    for lab, c in contrasts.items()},
            }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s4b4a_year_heldout.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    proto = load(ROOT / cfg["artifacts"]["protocol_json"])
    audit = load(ROOT / "reports/s4b4a_data_audit.json")
    e1 = load(ROOT / cfg["artifacts"]["e1_json"])
    e2 = load(ROOT / cfg["artifacts"]["e2_json"])

    e1d = e1_decisions(e1)
    e2d = e2_decisions(e2)
    pilot_seeds = list(e1["seeds"]) if e1 and "seeds" in e1 else list(cfg["training"]["seeds"])

    results = {
        "stage": "S4B.4A",
        "task": "training-set-internal year-held-out diagnostic",
        "protocol": cfg["artifacts"]["protocol_json"],
        "folds_completed": sorted(int(k) for k in (e1["folds"] if e1 else {})),
        "seeds_run": pilot_seeds,
        "seeds_planned": list(cfg["training"]["seeds"]),
        "is_pilot": len(pilot_seeds) == 1,
        "e1_contrasts": e1d,
        "e2_primary_lag7": e2d,
        "notes": [
            "This is a training-set-internal diagnostic, not a 2019 confirmation.",
            "These numbers are the 1-seed pilot; the formal 3-seed grid (seeds "
            "0,1,2) has NOT run, so no cross-seed CI is reported here.",
            "Differences are plot-clustered bootstrap CIs.",
            "GDD-P and Weather-P add parameters over B3 (see data audit §3.4), so "
            "part of any gain may be parameter-driven.",
            "A positive dMSE (E1) means the second arm is WORSE; a positive dR2 "
            "(E2) means the second arm is better.",
        ],
    }
    out_json = ROOT / cfg["artifacts"]["results_json"]
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2)
    write_md(ROOT / cfg["artifacts"]["results_md"], results, audit, proto, e1, e2)
    write_provenance(ROOT / "reports/s4b4a_provenance.md", cfg, proto, audit, e1, e2)
    print(f"wrote {out_json}")
    return 0


def fmt_ci(row: dict) -> str:
    return f"{row['mean']:+.6f} [{row['ci_low']:+.6f}, {row['ci_high']:+.6f}]"


def write_md(path: Path, r: dict, audit, proto, e1, e2) -> None:
    L = ["# S4B.4A Results -- Cross-Year Diagnostic (training-set internal)", ""]
    L.append(f"- Folds completed: {r['folds_completed']}")
    L.append(f"- Seeds run: {r['seeds_run']} (pilot); planned grid: {r['seeds_planned']}")
    L.append("- Scope: official-train years only; this is **not** an independent "
             "2019 unseen-environment confirmation.")
    L.append("")
    L.append("## E1 -- future-latent prediction, per fold")
    L.append("")
    if e1:
        L.append("| fold | arm | MSE | cos | vs persistence | vs train-mean |")
        L.append("|---|---|---:|---:|---:|---:|")
        for y, block in e1["folds"].items():
            sa = block["conditions"]["seed_averaged"]
            for arm, s in sa.items():
                L.append(f"| {y} | {arm} | {s['mse_pred']:.6f} | "
                         f"{s['cosine_pred']:.4f} | "
                         f"{100*s['relative_improvement_vs_persist']:.2f}% | "
                         f"{100*s['relative_improvement_vs_train_mean']:.2f}% |")
        L.append("")
        L.append("## E1 -- paired contrasts across folds")
        L.append("")
        for label, d in r["e1_contrasts"].items():
            L.append(f"### {label} (negative = second arm better)")
            L.append("")
            L.append("| fold | dMSE | 95% CI | CI excludes 0 | per-seed mean ± sd |")
            L.append("|---|---:|---|---|---|")
            for row in d["folds"]:
                L.append(f"| {row['fold']} | {row['mean']:+.6f} | "
                         f"[{row['ci_low']:+.6f}, {row['ci_high']:+.6f}] | "
                         f"{row['ci_excludes_zero']} | "
                         f"{row['per_seed_mean']:+.6f} ± {row['per_seed_std']:.6f} |")
            L.append("")
            L.append(f"- folds with CI entirely below 0 (second arm better): "
                     f"{d['n_folds_ci_lower_than_zero']}/{d['n_folds']}; "
                     f"all CIs exclude 0 on {d['n_folds_ci_excludes_zero']}/{d['n_folds']}")
            direction = ("consistent (same sign every fold)" if d["n_folds"] and
                         (d["n_folds_ci_lower_than_zero"] in (0, d["n_folds"]))
                         else "sign flips across folds")
            L.append(f"- direction: {direction}; "
                     f"stable across folds (majority + all CIs exclude 0): "
                     f"{d['stable_across_folds']}")
            L.append("")
    else:
        L.append("_E1 results not yet produced._")
        L.append("")
    L.append("## E2 -- CC / PH readout (primary |lag| <= 7 d)")
    L.append("")
    if e2:
        for trait in ("cc", "ph"):
            L.append(f"### {trait}")
            L.append("")
            L.append("| fold | arm | R2 | MAE |")
            L.append("|---|---|---:|---:|")
            for fold, d in r["e2_primary_lag7"].get(trait, {}).items():
                for arm, v in d["r2"].items():
                    L.append(f"| {fold} | {arm} | {v:.5f} | "
                             f"{d['mae'][arm]:.5f} |")
            L.append("")
            L.append("| fold | contrast | dR2 | 95% CI |")
            L.append("|---|---|---:|---|")
            for fold, d in r["e2_primary_lag7"].get(trait, {}).items():
                for lab, c in d["contrasts"].items():
                    L.append(f"| {fold} | {lab} | {c['mean']:+.5f} | "
                             f"[{c['ci_low']:+.5f}, {c['ci_high']:+.5f}] |")
            L.append("")
    else:
        L.append("_E2 results not yet produced._")
        L.append("")
    L.append("## Decision summary (fixed rules)")
    L.append("")
    for label, d in r["e1_contrasts"].items():
        L.append(f"- E1 `{label}`: mean dMSE over folds "
                 f"{d['mean_dmse_over_folds']:+.6f}; stable = "
                 f"{d['stable_across_folds']}.")
    L.append("")
    for trait in ("cc", "ph"):
        for lab in ("b3_delta_minus_b2_no_time_constant_delta",
                    "gdd_p_minus_b3_delta", "weather_p_minus_b3_delta"):
            vals = [d["contrasts"][lab]["mean"]
                    for d in r["e2_primary_lag7"].get(trait, {}).values()
                    if lab in d["contrasts"]]
            if vals:
                L.append(f"- E2 {trait} `{lab}`: mean dR2 over folds "
                         f"{sum(vals)/len(vals):+.5f} (n={len(vals)}).")
    L.append("")
    L.append("## Caveats")
    L.append("")
    for n in r["notes"]:
        L.append(f"- {n}")
    L.append("")
    path.write_text("\n".join(L) + "\n")


def write_provenance(path: Path, cfg, proto, audit, e1, e2) -> None:
    import subprocess
    def git(*a):
        try:
            return subprocess.check_output(["git", "-C", str(ROOT), *a],
                                           text=True).strip()
        except Exception:
            return "unknown"
    L = ["# S4B.4A Provenance", ""]
    L.append(f"- Branch: `{git('rev-parse','--abbrev-ref','HEAD')}`")
    L.append(f"- HEAD: `{git('rev-parse','HEAD')}`")
    L.append(f"- Baseline (S4B.3): `27504922095366ac246c8760a08702b3b762d620`")
    L.append(f"- Config: `agro_jepa/config/s4b4a_year_heldout.yaml`")
    if proto:
        L.append(f"- Protocol: `{cfg['artifacts']['protocol_json']}`")
        d = proto["data_scope"]
        L.append(f"  - data: {d['pairs']:,} pairs / {d['plots']:,} plots / "
                 f"years {d['years']}")
    if audit:
        L.append(f"- Audit inputs: full_train_pairs sha "
                 f"`{audit['inputs']['full_train_pairs_sha256'][:16]}`, "
                 f"conditioning sha "
                 f"`{audit['inputs']['conditioning_npz_sha256'][:16]}`")
    L.append("- GDD/weather conditioning (full train): "
             "`external_data/s4b4a_conditioning/full_train_conditioning.npz`")
    L.append("- Folds: `external_data/s4b4a_folds/`")
    L.append("- Runs: `external_data/s4b4a_runs/`")
    L.append("")
    L.append("## Reproduce")
    L.append("")
    L.append("```bash")
    L.append("python3 tools/s4b4a_data_audit.py --config agro_jepa/config/s4b4a_year_heldout.yaml")
    L.append("python3 tools/s4b4a_protocol.py  --config agro_jepa/config/s4b4a_year_heldout.yaml")
    L.append("for f in 2016 2017 2018 2021 2022; do")
    L.append("  for c in b2_no_time b2_no_time_constant_delta b3_delta gdd_p weather_p; do")
    L.append("    CUDA_VISIBLE_DEVICES=1 python3 tools/s4b4a_train.py \\")
    L.append("      --config agro_jepa/config/s4b4a_year_heldout.yaml \\")
    L.append("      --fold $f --condition $c --seed 0")
    L.append("  done")
    L.append("done")
    L.append("CUDA_VISIBLE_DEVICES=1 python3 tools/s4b4a_eval_e1.py --config agro_jepa/config/s4b4a_year_heldout.yaml")
    L.append("CUDA_VISIBLE_DEVICES=1 python3 tools/s4b4a_eval_e2.py --config agro_jepa/config/s4b4a_year_heldout.yaml")
    L.append("python3 tools/s4b4a_write_reports.py --config agro_jepa/config/s4b4a_year_heldout.yaml")
    L.append("python3 tools/s4b4a_verify.py --config agro_jepa/config/s4b4a_year_heldout.yaml")
    L.append("```")
    L.append("")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
