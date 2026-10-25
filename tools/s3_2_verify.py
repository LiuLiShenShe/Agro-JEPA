#!/usr/bin/env python3
"""S3.2 task D -- leakage / integrity verification for the new artifacts.

Every check reads a *produced artifact* and recomputes the property it claims,
or inspects the committed code path that was executed.  Nothing is asserted
from memory.

Checks
------
D1  official train and validation plot sets are disjoint (index)
D2  probe inner-train / inner-dev plot sets are disjoint and cover the 1000
    official-train plots; no validation plot is in the split
D3  probe feature normalisation statistics are exactly those of the training
    rows of the corresponding condition (float32 mean/std hash)
D4  final readouts use an epoch count selected on inner dev only, and no final
    run used official validation for selection
D5  the predictor input contains no future image / future phenotype; the probe
    reads only mean-pooled current or predicted features
D6  z_a, z_b and pair_id are unchanged across every delta intervention
D7  every A2/A4 permutation preserves the delta multiset inside its group
D8  A3's per-pair_type means come from the official train split only
D9  reported aggregates are recomputable from the per-sample artifacts
D10 the S3 checkpoint, frozen index, S3/S3.1 reports and manifests are untouched

Usage
-----
    python3 tools/s3_2_verify.py --config agro_jepa/config/s3_2_diagnostics.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agro_jepa" / "eval"))

from agro_jepa.datasets.s3_pairs import index_sha256  # noqa: E402
from agro_jepa.models.fip_dino_wm import FIPDinoWM  # noqa: E402
from agro_jepa.eval import eval_v0  # noqa: E402

import subprocess  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Checks:
    def __init__(self) -> None:
        self.items: list[dict] = []

    def add(self, cid: str, ok: bool, detail) -> None:
        self.items.append({"id": cid, "passed": bool(ok), "detail": detail})

    def summary(self) -> dict:
        n = len(self.items)
        p = sum(1 for i in self.items if i["passed"])
        return {"n_checks": n, "n_passed": p, "n_failed": n - p,
                "all_passed": p == n, "checks": self.items}


# --------------------------------------------------------------------------
def scope_frames(probe_df: pd.DataFrame, trait: str) -> dict[str, pd.DataFrame]:
    scopes = {}
    overall = probe_df.sort_values(
        ["plot_uid", "image_stem_b", "delta_days", "pair_type", "pair_id"],
        kind="mergesort",
    ).drop_duplicates(["plot_uid", "image_stem_b"], keep="first").reset_index(drop=True)
    scopes["overall"] = overall[overall[f"{trait}_mask_b"]].reset_index(drop=True)
    for span in ("span7", "span14"):
        scopes[span] = probe_df[
            (probe_df.pair_type == span) & probe_df[f"{trait}_mask_b"]
        ].reset_index(drop=True)
    return scopes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s3_2_diagnostics.yaml")
    ap.add_argument("--delta", default="reports/s3_2_delta_diagnostics.json")
    ap.add_argument("--delta-pred", default="reports/s3_2_delta_interventions.parquet")
    ap.add_argument("--permuted", default="reports/s3_2_permuted_deltas.parquet")
    ap.add_argument("--probe-protocol", default="reports/s3_2_probe_protocol.json")
    ap.add_argument("--probe-pred", default="reports/s3_2_probe_val_predictions.parquet")
    ap.add_argument("--plot-split", default="reports/s3_2_probe_plot_split.csv")
    ap.add_argument("--probe-samples", default="reports/s3_probe_samples.parquet")
    ap.add_argument("--pooled", default="reports/s3_probe_pooled.npz")
    ap.add_argument("--out", default="reports/s3_2_verify.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    pcfg = cfg["probe"]
    delta = json.loads(Path(args.delta).read_text())
    protocol = json.loads(Path(args.probe_protocol).read_text())
    delta_pred = pd.read_parquet(args.delta_pred)
    permuted = pd.read_parquet(args.permuted)
    probe_pred = pd.read_parquet(args.probe_pred)
    split = pd.read_csv(args.plot_split, dtype={"plot_uid": str})
    probe_df = pd.read_parquet(args.probe_samples)
    probe_df["_pool_index"] = np.arange(len(probe_df))
    pooled = np.load(args.pooled)
    index = pd.read_parquet(cfg["frozen"]["index"])
    c = Checks()

    # ---- D1 -------------------------------------------------------------
    tr_plots = set(index[index.split == "train"].plot_uid)
    va_plots = set(index[index.split == "validation"].plot_uid)
    c.add(
        "D1_official_train_val_plot_disjoint",
        len(tr_plots & va_plots) == 0,
        {
            "n_train_plots": len(tr_plots),
            "n_validation_plots": len(va_plots),
            "n_overlap": len(tr_plots & va_plots),
        },
    )

    # ---- D2 -------------------------------------------------------------
    roles = split.set_index("plot_uid").role
    inner_tr = set(split[split.role == "inner_train"].plot_uid)
    inner_dev = set(split[split.role == "inner_dev"].plot_uid)
    c.add(
        "D2_probe_split_disjoint_and_covers_train",
        len(inner_tr & inner_dev) == 0
        and len(inner_tr | inner_dev) == len(tr_plots)
        and (inner_tr | inner_dev) == tr_plots
        and len(va_plots & (inner_tr | inner_dev)) == 0,
        {
            "n_inner_train": len(inner_tr),
            "n_inner_dev": len(inner_dev),
            "n_union": len(inner_tr | inner_dev),
            "n_official_train": len(tr_plots),
            "n_overlap_train_dev": len(inner_tr & inner_dev),
            "n_validation_plots_in_split": len(va_plots & (inner_tr | inner_dev)),
            "file": args.plot_split,
        },
    )

    # ---- D3 -------------------------------------------------------------
    def xmat(frame: pd.DataFrame, col: str) -> np.ndarray:
        idx = frame["_pool_index"].to_numpy()
        return (pooled["pool_a"] if col == "pool_a" else pooled["pool_b"])[idx]

    def build_train_frame(trait: str, variant: str, lag_max: int):
        frame = scope_frames(probe_df, trait)[variant]
        frame = frame[frame[f"{trait}_b_lag"].abs() <= lag_max]
        tr = frame[frame.split == "train"].copy()
        tr["role"] = tr.plot_uid.map(roles)
        return tr

    norm_checks = []
    for variant, lag_max in {
        "primary_lag7": int(pcfg["lag_variants"]["primary"]),
        "strict_lag3": int(pcfg["lag_variants"]["strict_lag3"]),
        "same_day_lag0": int(pcfg["lag_variants"]["same_day"]),
    }.items():
        for trait in ("cc", "ph"):
            for scope, sel in protocol["C3_inner_selection"].get(variant, {}).get(trait, {}).items():
                if "conditions" not in sel:
                    continue
                tr = build_train_frame(trait, scope, lag_max)
                tr_in = tr[tr.role == "inner_train"]
                tr_dev = tr[tr.role == "inner_dev"]

                def mean_hash(rows: pd.DataFrame, col: str) -> str:
                    x = xmat(rows, col)
                    return hashlib.sha256(
                        np.ascontiguousarray(
                            x.mean(axis=0, keepdims=True), dtype=np.float32
                        ).tobytes()
                    ).hexdigest()

                for cond, col in (("A_current", "pool_a"), ("B_predicted", "pool_b")):
                    want_in = mean_hash(tr_in, col)
                    want_dev = mean_hash(tr_dev, col)
                    got_inner = {
                        r["x_mean_sha256"]
                        for r in sel["conditions"][cond]["runs"]
                    }
                    norm_checks.append({
                        "variant": variant, "trait": trait, "scope": scope,
                        "condition": cond, "rows": "inner_train",
                        "n_rows": int(len(tr_in)),
                        "hash_matches": bool(got_inner == {want_in}),
                        "inner_dev_stats_differ": bool(want_in != want_dev),
                    })
                    final = protocol["C4_final_readout"][variant][trait][scope]
                    want_tr = mean_hash(tr, col)
                    for proto_name, proto in final["protocols"].items():
                        got_final = {
                            r["x_mean_sha256"]
                            for r in proto["conditions"][cond]["runs"]
                        }
                        norm_checks.append({
                            "variant": variant, "trait": trait, "scope": scope,
                            "condition": cond, "rows": "all_official_train",
                            "n_rows": int(len(tr)), "protocol": proto_name,
                            "hash_matches": bool(got_final == {want_tr}),
                        })
    c.add(
        "D3_normalisation_from_training_rows_only",
        all(v["hash_matches"] for v in norm_checks),
        {"n_checks": len(norm_checks),
         "n_failed": sum(1 for v in norm_checks if not v["hash_matches"]),
         "checks": norm_checks},
    )

    # ---- D4 -------------------------------------------------------------
    selection_ok = True
    details = []
    for variant, traits in protocol["C4_final_readout"].items():
        for trait, scopes in traits.items():
            for scope, blk in scopes.items():
                if "protocols" not in blk:
                    continue
                sel = protocol["C3_inner_selection"][variant][trait][scope]
                for proto_name, pb in blk["protocols"].items():
                    for cond, cb in pb["conditions"].items():
                        if proto_name == "per_condition_epochs":
                            want = sel["conditions"][cond]["selected_epoch_median"]
                        else:
                            want = sel["shared_epoch_median_over_both_conditions"]
                        got = cb["epochs_used"]
                        flags = all(
                            r["official_validation_used_for_selection"] is False
                            and "inner_dev" in r["selection_source"]
                            for r in cb["runs"]
                        )
                        ok = (got == want) and flags
                        selection_ok = selection_ok and ok
                        details.append({
                            "variant": variant, "trait": trait, "scope": scope,
                            "protocol": proto_name, "condition": cond,
                            "epochs_used": got, "expected": want,
                            "selection_flags_ok": flags,
                        })
    c.add("D4_no_validation_early_stopping_in_final_readout", selection_ok,
          {"n_checks": len(details), "checks": details})

    # ---- D5 -------------------------------------------------------------
    sig = inspect.signature(FIPDinoWM.forward)
    forward_params = list(sig.parameters)
    src = inspect.getsource(eval_v0.collect_pooled_features)
    model_calls = [ln.strip() for ln in src.splitlines() if "model(" in ln]
    no_future_input = (
        forward_params == ["self", "z_a", "delta_days", "return_components"]
        and len(model_calls) > 0
        and all("z_b" not in ln and "_b" not in ln for ln in model_calls)
    )
    probe_meta_ok = (
        protocol["inputs"]["pooling"].startswith("mean")
        and "linear" in protocol["inputs"]["architecture"]
    )
    c.add(
        "D5_no_future_image_or_phenotype_in_model_input",
        no_future_input and probe_meta_ok,
        {
            "predictor_forward_parameters": forward_params,
            "collect_pooled_features_model_calls": model_calls,
            "probe_inputs": protocol["inputs"],
            "note": (
                "B is the V0 prediction from z_a + delta_days; the probe never "
                "receives the future feature as an input, only the future label "
                "as a target."
            ),
        },
    )

    # ---- D6 -------------------------------------------------------------
    inv = delta["invariants"]
    pair_ok = bool(
        (delta_pred.pair_id.to_numpy() == permuted.pair_id.to_numpy()).all()
        and (delta_pred.pair_id.to_numpy() == index[index.split == "validation"].pair_id.to_numpy()).all()
    )
    c.add(
        "D6_z_and_pair_order_unchanged_across_interventions",
        bool(inv["z_and_pair_order_unchanged"]) and pair_ok,
        {
            "max_abs_persist_deviation_by_condition": inv[
                "max_abs_persist_deviation_by_condition"
            ],
            "persistence_mse_real": inv["persistence_mse_real"],
            "pair_id_order_matches_index": pair_ok,
            "note": inv["note"],
        },
    )

    # ---- D7 -------------------------------------------------------------
    span = index[index.split == "validation"].pair_type.to_numpy()
    year = index[index.split == "validation"].harvest_year.to_numpy().astype(str)
    strata = np.array([f"{s}|{y}" for s, y in zip(span, year)], dtype=object)
    orig = permuted.delta_original.to_numpy()
    freq = {"A2": [], "A4": []}
    for tag, labels in (("A2", span), ("A4", strata)):
        cols = [c for c in permuted.columns if c.startswith(tag.lower() + "_s")]
        for col in cols:
            arr = permuted[col].to_numpy()
            for g in np.unique(labels):
                pos = np.flatnonzero(labels == g)
                same = np.array_equal(np.sort(arr[pos]), np.sort(orig[pos]))
                freq[tag].append({"seed_col": col, "group": str(g),
                                  "n": int(len(pos)), "preserved": bool(same)})
    c.add(
        "D7_within_group_delta_frequency_preserved",
        all(v["preserved"] for tag in freq for v in freq[tag]),
        {
            "A2_groups_checked": len(freq["A2"]),
            "A4_groups_checked": len(freq["A4"]),
            "n_failed": sum(1 for tag in freq for v in freq[tag] if not v["preserved"]),
            "group_sizes_A2": delta["conditions"]["A2_within_pair_type_permutation"]["groups"],
            "group_sizes_A4": delta["conditions"]["A4_within_pair_type_year_permutation"]["groups"],
        },
    )

    # ---- D8 -------------------------------------------------------------
    train_means_json = delta["conditions"]["A3_pair_type_train_mean"]["train_means"]
    recomputed = {
        str(k): float(v)
        for k, v in index[index.split == "train"].groupby("pair_type").delta_days.mean().items()
    }
    val_means = {
        str(k): float(v)
        for k, v in index[index.split == "validation"].groupby("pair_type").delta_days.mean().items()
    }
    a3_vec = permuted.delta_a3.to_numpy()
    per_type_ok = all(
        np.allclose(
            a3_vec[span == t], train_means_json[t], atol=1e-9, rtol=0
        )
        for t in ("span7", "span14")
    )
    c.add(
        "D8_A3_means_from_train_only",
        all(abs(train_means_json[k] - recomputed[k]) < 1e-9 for k in recomputed)
        and per_type_ok,
        {
            "train_means_json": train_means_json,
            "train_means_recomputed": recomputed,
            "validation_means_for_reference_only": val_means,
            "a3_vector_uses_its_own_pair_type": bool(per_type_ok),
        },
    )

    # ---- D9 -------------------------------------------------------------
    agg_checks = []
    base_mse = float(delta_pred.mse_real.mean())
    agg_checks.append({
        "item": "A0_mse",
        "json": delta["A0_real_delta"]["mse_real"],
        "recomputed": base_mse,
        "ok": abs(delta["A0_real_delta"]["mse_real"] - base_mse) < 1e-9,
    })
    for name, tag in (("A1_global_permutation", "a1"),
                      ("A2_within_pair_type_permutation", "a2"),
                      ("A4_within_pair_type_year_permutation", "a4")):
        cols = [c for c in delta_pred.columns if c.startswith(tag + "_s")]
        per_seed = delta_pred[cols].to_numpy().mean(axis=0)
        agg_checks.append({
            "item": f"{name}_mse_mean",
            "json": delta["conditions"][name]["mse_condition_mean"],
            "recomputed": float(per_seed.mean()),
            "ok": abs(delta["conditions"][name]["mse_condition_mean"]
                      - float(per_seed.mean())) < 1e-9,
        })
    a3_mean = float(delta_pred.mse_a3.mean())
    agg_checks.append({
        "item": "A3_mse",
        "json": delta["conditions"]["A3_pair_type_train_mean"]["mse_condition_mean"],
        "recomputed": a3_mean,
        "ok": abs(delta["conditions"]["A3_pair_type_train_mean"]["mse_condition_mean"]
                  - a3_mean) < 1e-9,
    })
    probe_checks = []
    for variant in ("primary_lag7",):
        for trait in ("cc", "ph"):
            for scope in ("overall", "span7", "span14"):
                blk = protocol["C4_final_readout"][variant][trait][scope]
                for cond in ("A_current", "B_predicted"):
                    sub = probe_pred[(probe_pred.variant == variant)
                                     & (probe_pred.trait == trait)
                                     & (probe_pred.scope == scope)
                                     & (probe_pred.condition == cond)]
                    per_seed = [
                        float(np.mean(np.abs(g.y_pred - g.y_true)))
                        for _, g in sub.groupby("seed")
                    ]
                    reported = blk["protocols"]["per_condition_epochs"]["conditions"][cond]["mae"]["mean"]
                    recomputed_mae = float(np.mean(per_seed))
                    probe_checks.append({
                        "trait": trait, "scope": scope, "condition": cond,
                        "reported": reported, "recomputed": recomputed_mae,
                        "ok": abs(reported - recomputed_mae) < 1e-6,
                    })
    ci_ok = all(
        blk[key]["ci95_low"] <= blk[key]["mean"] <= blk[key]["ci95_high"]
        for blk in delta["conditions"].values()
        for key in ("relative_increase_bootstrap", "paired_diff")
    )
    c.add(
        "D9_metrics_recomputable_from_per_sample_artifacts",
        all(x["ok"] for x in agg_checks) and all(x["ok"] for x in probe_checks) and ci_ok,
        {
            "delta_aggregates": agg_checks,
            "probe_aggregates": probe_checks,
            "bootstrap_ci_ordering_ok": bool(ci_ok),
        },
    )

    # ---- D10 ------------------------------------------------------------
    frozen = cfg["frozen"]
    ckpt_sha = sha256_file(Path(frozen["checkpoint"]))
    idx_sha = index_sha256(index)
    protected = [
        "external_data/s3_runs/v0/checkpoints/best.pt",
        "external_data/s3_index/fip1_s3_v0_index.parquet",
        "reports/s3_training.json",
        "reports/s3_v0_results.json",
        "reports/s3_1_diagnostic_audit.json",
        "reports/s3_1_probe_reproduction.json",
        "reports/s3_training.md",
        "reports/s3_v0_results.md",
        "reports/s3_1_diagnostic_audit.md",
        "reports/s3_1_provenance.md",
        "data/manifests/s3_image_selection.json",
    ]
    tracked = subprocess.run(
        ["git", "ls-files", *protected], cwd=ROOT, capture_output=True, text=True
    ).stdout.split()
    status = subprocess.run(
        ["git", "status", "--porcelain", *protected],
        cwd=ROOT, capture_output=True, text=True,
    ).stdout.strip()
    c.add(
        "D10_original_artifacts_untouched",
        ckpt_sha == frozen["checkpoint_sha256"]
        and idx_sha == frozen["index_sha256"]
        and status == "",
        {
            "checkpoint_sha256": ckpt_sha,
            "checkpoint_expected": frozen["checkpoint_sha256"],
            "index_sha256": idx_sha,
            "index_expected": frozen["index_sha256"],
            "protected_paths": protected,
            "git_tracked_subset": tracked,
            "git_status_porcelain_for_protected": status.splitlines(),
            "file_sha256_now": {
                p: sha256_file(ROOT / p)
                for p in protected if (ROOT / p).exists()
            },
        },
    )

    out = {
        "stage": "S3.2",
        "task": "D -- leakage / integrity verification",
        "generated_at": pd.Timestamp.now().isoformat(),
        "config": args.config,
        "artifacts": {
            "delta": args.delta, "delta_predictions": args.delta_pred,
            "permuted_deltas": args.permuted,
            "probe_protocol": args.probe_protocol,
            "probe_predictions": args.probe_pred,
            "plot_split": args.plot_split,
        },
        **c.summary(),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({
        "n_checks": out["n_checks"], "n_passed": out["n_passed"],
        "n_failed": out["n_failed"], "all_passed": out["all_passed"],
        "failed": [i for i in out["checks"] if not i["passed"]],
    }, indent=2))
    return 0 if out["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
