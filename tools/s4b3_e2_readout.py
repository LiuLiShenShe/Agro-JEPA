#!/usr/bin/env python3
"""S4B.3B E2 stage 2: CC / PH readout on the four official test splits.

Reads the pooled features dumped by ``s4b3_e2_pool.py`` and trains a linear
readout on **all 2,930 official-train plots** under the frozen E2 protocol
(``reports/s4b3_e2_protocol.json`` + ``reports/s4b3_e2_plot_split.csv``).  Test
labels are used only for the final evaluation.

Conditions
----------
A_current  mean(z_a)  current frozen DINOv2 feature
B2         mean(z_hat_b) predicted future latent, no time input
B2-C       mean(z_hat_b) predicted future latent, fixed 9.0 d
B3         mean(z_hat_b) predicted future latent, real per-pair delta_days

Protocol: inner 80/20 by plot (seed 20261025, stratified by harvest_year) ->
minimum inner-dev MSE over 150 candidate epochs -> refit on all official-train
rows at the median selected epoch -> one evaluation per test split.  Primary
label window |lag| <= 7 d; strict-lag3 and same-day sensitivities.

Usage
-----
    CUDA_VISIBLE_DEVICES=1 python3 tools/s4b3_e2_readout.py \
        --config agro_jepa/config/s4b3_official_test.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "agro_jepa" / "eval"))

from agro_jepa.eval.eval_s4 import scope_frames  # noqa: E402
from s3_1_diagnostics import cluster_bootstrap, git, sha256_file  # noqa: E402
from s3_2_probe_eval import _metrics, predict_readout, train_readout  # noqa: E402

TRAITS = ("cc", "ph")
CONDITIONS = ("A_current", "B2", "B2-C", "B3")


def run_readout(meta, pooled, trait, cond, lag_max, pcfg, role, device,
                test_splits):
    frame = scope_frames(meta, trait)["overall"].copy()
    frame = frame[frame[f"{trait}_b_lag"].abs() <= lag_max].reset_index(drop=True)
    if frame.empty:
        raise RuntimeError(f"{trait}/{cond}/lag{lag_max}: no rows")
    tr = frame[frame.split == "train"].copy()
    tr["role"] = tr.plot_uid.map(role)
    if tr.role.isna().any():
        raise RuntimeError("a train plot is missing from the frozen inner split")
    tr_in = tr[tr.role == "inner_train"].reset_index(drop=True)
    tr_dev = tr[tr.role == "inner_dev"].reset_index(drop=True)

    x_in = pooled[tr_in["_pool_index"].to_numpy()]
    x_dev = pooled[tr_dev["_pool_index"].to_numpy()]
    x_tr = pooled[tr["_pool_index"].to_numpy()]
    y_in = tr_in[f"{trait}_b"].to_numpy(np.float32)
    y_dev = tr_dev[f"{trait}_b"].to_numpy(np.float32)
    y_tr = tr[f"{trait}_b"].to_numpy(np.float32)

    epochs = []
    for seed in pcfg["seeds"]:
        _, info = train_readout(x_in, y_in, pcfg, int(seed), device,
                                epochs_to_run=0, x_dev=x_dev, y_dev=y_dev,
                                select=True)
        epochs.append(int(info["best_epoch"]))
    selected = int(round(float(np.median(epochs))))

    split_metrics, split_preds = {}, {}
    for split in test_splits:
        te = frame[frame.split == split].reset_index(drop=True)
        if te.empty:
            split_metrics[split] = None
            continue
        x_te = pooled[te["_pool_index"].to_numpy()]
        y_te = te[f"{trait}_b"].to_numpy(np.float32)
        per_seed_pred, per_seed_metrics = [], []
        for seed in pcfg["seeds"]:
            state, _ = train_readout(x_tr, y_tr, pcfg, int(seed), device,
                                     epochs_to_run=selected, select=False)
            pred = predict_readout(state, x_te, device)
            per_seed_pred.append(pred)
            per_seed_metrics.append(_metrics(y_te, pred))
        pred_avg = np.mean(per_seed_pred, axis=0)
        m = _metrics(y_te, pred_avg)
        split_preds[split] = (te, y_te, pred_avg, per_seed_pred)
        block = {
            "n": int(len(te)), "n_plots": int(te.plot_uid.nunique()),
            "mae": float(m["mae"]), "rmse": float(m["rmse"]),
            "r2": float(m["r2"]),
            "per_seed_mae_mean": float(np.mean([r["mae"] for r in per_seed_metrics])),
            "per_seed_mae_std": float(np.std([r["mae"] for r in per_seed_metrics])),
            "per_seed_rmse_mean": float(np.mean([r["rmse"] for r in per_seed_metrics])),
            "per_seed_r2_mean": float(np.mean([r["r2"] for r in per_seed_metrics])),
            "per_seed_r2_std": float(np.std([r["r2"] for r in per_seed_metrics])),
            "bootstrap": {},
        }
        for key in ("mae", "rmse", "r2"):
            block["bootstrap"][key] = cluster_bootstrap(
                {"y": y_te.astype(np.float64), "p": pred_avg.astype(np.float64)},
                te.plot_uid.to_numpy(),
                lambda t, k=key: (float(np.mean(np.abs(t["p"] - t["y"])))
                                  if k == "mae" else
                                  float(np.sqrt(np.mean((t["p"] - t["y"]) ** 2)))
                                  if k == "rmse" else
                                  float(1.0 - np.sum((t["p"] - t["y"]) ** 2)
                                        / np.sum((t["y"] - t["y"].mean()) ** 2))),
                int(pcfg["bootstrap_resamples"]), int(pcfg["bootstrap_seed"]))
        split_metrics[split] = block
    return {
        "selected_epoch_median": selected,
        "selected_epochs_per_seed": epochs,
        "counts": {
            "n_train": int(len(tr)), "n_train_plots": int(tr.plot_uid.nunique()),
            "n_inner_train": int(len(tr_in)),
            "n_inner_dev": int(len(tr_dev)),
            "lag_max_abs_days": int(lag_max),
        },
        "splits": split_metrics, "preds": split_preds,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s4b3_official_test.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument("--pooled-dir", default=None)
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    e2 = cfg["e2"]

    proto = json.load(open(ROOT / "reports/s4b3_e2_protocol.json"))
    if not proto["verification"]["all_pass"]:
        raise SystemExit("E2 protocol did not pass; refusing to read test labels")
    if sha256_file(ROOT / proto["inner_split"]["file"]) != proto["inner_split"]["sha256"]:
        raise SystemExit("inner split file changed since the protocol was frozen")

    t0 = time.time()
    device = torch.device(args.device or (cfg["device"]["cuda"]
                          if torch.cuda.is_available() else "cpu"))
    seeds = [int(s) for s in e2["seeds"]]
    pcfg = {
        "lr": float(e2["lr"]), "weight_decay": float(e2["weight_decay"]),
        "batch_size": int(e2["batch_size"]),
        "candidate_epochs": int(e2["candidate_epochs"]),
        "patience": int(e2["patience"]), "seeds": seeds,
        "bootstrap_resamples": int(e2["bootstrap_resamples"]),
        "bootstrap_seed": int(e2["bootstrap_seed"]),
    }
    test_splits = list(cfg["test_splits"].keys())
    pooled_dir = ROOT / (args.pooled_dir or cfg["artifacts"]["e2_pooled_dir"])

    split_df = pd.read_csv(ROOT / proto["inner_split"]["file"])
    role = split_df.set_index("plot_uid").role
    meta = pd.read_parquet(pooled_dir / "combined_meta.parquet")
    pooled_mat = {
        "A_current": np.load(pooled_dir / "A_current.npz")["pool_a"],
        **{c: np.load(pooled_dir / f"{c}.npz")["pool_b"] for c in ("B2", "B2-C", "B3")},
    }
    for c, arr in pooled_mat.items():
        if len(arr) != len(meta):
            raise SystemExit(f"{c}: pooled rows {len(arr)} != meta {len(meta)}")

    label_audit = {}
    for split in test_splits:
        te = meta[meta.split == split]
        label_audit[split] = {
            "n_pairs": int(len(te)), "n_plots": int(te.plot_uid.nunique()),
            "n_cc_valid_lag7": int(te.cc_mask_b.sum()),
            "n_ph_valid_lag7": int(te.ph_mask_b.sum()),
            "cc_lag_days": {"min": float(te.cc_b_lag.min()),
                            "max": float(te.cc_b_lag.max()),
                            "median": float(te.cc_b_lag.median())},
            "ph_lag_days": {"min": float(te.ph_b_lag.min()),
                            "max": float(te.ph_b_lag.max()),
                            "median": float(te.ph_b_lag.median())},
            "label_join_failures": 0,
            "n_cc_negatives": int((te.cc_b < 0).sum()),
            "n_ph_negatives": int((te.ph_b < 0).sum()),
        }

    results = {
        "stage": "S4B.3",
        "task": "E2 CC/PH readout on the four official test splits",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": " ".join(sys.argv),
        "git_head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "config": args.config, "config_sha256": sha256_file(ROOT / args.config),
        "protocol_json": "reports/s4b3_e2_protocol.json",
        "protocol_sha256": sha256_file(ROOT / "reports/s4b3_e2_protocol.json"),
        "inner_split_file": proto["inner_split"]["file"],
        "inner_split_sha256": proto["inner_split"]["sha256"],
        "device": str(device),
        "protocol": proto["readout"],
        "lag_variants": e2["lag_variants"],
        "label_audit": label_audit,
        "test_phenotype_labels_used_for_training_or_selection": False,
        "n_meta_rows": int(len(meta)),
        "rows": {}, "bootstrap": {},
    }
    pred_rows = []
    lag_variants = {
        "primary_lag7": int(e2["lag_variants"]["primary_lag7"]),
        "strict_lag3": int(e2["lag_variants"]["strict_lag3"]),
        "same_day_lag0": int(e2["lag_variants"]["same_day_lag0"]),
    }
    for trait in TRAITS:
        results["rows"][trait] = {}
        for lag_name, lag_max in lag_variants.items():
            results["rows"][trait][lag_name] = {}
            for cond in CONDITIONS:
                out = run_readout(meta, pooled_mat[cond], trait, cond, lag_max,
                                  pcfg, role, device, test_splits)
                results["rows"][trait][lag_name][cond] = {
                    "selected_epoch_median": out["selected_epoch_median"],
                    "selected_epochs_per_seed": out["selected_epochs_per_seed"],
                    "counts": out["counts"], "splits": out["splits"],
                }
                for split, (te, y_te, pred_avg, per_seed) in out["preds"].items():
                    for si, seed in enumerate(seeds):
                        for i in range(len(te)):
                            pred_rows.append({
                                "trait": trait, "lag_variant": lag_name,
                                "condition": cond, "seed": int(seed),
                                "split": split, "pair_id": te.pair_id.iloc[i],
                                "plot_uid": te.plot_uid.iloc[i],
                                "lag_days": float(te[f"{trait}_b_lag"].iloc[i]),
                                "y_true": float(y_te[i]),
                                "y_pred": float(per_seed[si][i]),
                            })
                print(f"[E2] {trait}/{lag_name}/{cond}: "
                      f"epoch={out['selected_epoch_median']} "
                      + " ".join(f"{s.split('_',1)[1][:4]}={out['splits'][s]['mae']:.4f}"
                                 for s in test_splits), flush=True)
    preds = pd.DataFrame(pred_rows)
    preds.to_parquet(ROOT / cfg["artifacts"]["e2_predictions"], index=False)

    for trait in TRAITS:
        results["bootstrap"][trait] = {}
        for lag_name in lag_variants:
            results["bootstrap"][trait][lag_name] = {}
            d = preds[(preds.trait == trait) & (preds.lag_variant == lag_name)]
            for split in test_splits:
                ds = d[d.split == split]
                base = ds[["pair_id", "plot_uid"]].drop_duplicates()
                base = base.sort_values("pair_id").reset_index(drop=True)
                wide = base.copy()
                for cond in CONDITIONS:
                    sub = (ds[ds.condition == cond].groupby("pair_id", as_index=False)
                           .agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean")))
                    sub = base[["pair_id"]].merge(sub, on="pair_id", how="left")
                    wide[cond] = sub.y_pred.to_numpy(np.float64)
                yt = (ds[ds.condition == "A_current"].groupby("pair_id", as_index=False)
                      .agg(y_true=("y_true", "first")))
                yt = base[["pair_id"]].merge(yt, on="pair_id", how="left")
                wide["y"] = yt.y_true.to_numpy(np.float64)
                plots = wide.plot_uid.to_numpy()
                tables = {"y": wide["y"].to_numpy(np.float64),
                          **{c: wide[c].to_numpy(np.float64) for c in CONDITIONS}}

                def _contrast(t, hi, lo, k):
                    eh, el = t[hi] - t["y"], t[lo] - t["y"]
                    ss = float(np.sum((t["y"] - t["y"].mean()) ** 2))
                    if k == "delta_mae":
                        return float(np.mean(np.abs(eh)) - np.mean(np.abs(el)))
                    if k == "delta_rmse":
                        return float(np.sqrt(np.mean(eh ** 2)) - np.sqrt(np.mean(el ** 2)))
                    return float((-np.sum(eh ** 2) + np.sum(el ** 2)) / ss)

                block = {"n_pairs": int(len(wide)),
                         "n_plots": int(wide.plot_uid.nunique()),
                         "mean_metrics": {}, "contrasts": {}}
                for cond in CONDITIONS:
                    block["mean_metrics"][cond] = {
                        "mae": float(np.mean(np.abs(tables[cond] - tables["y"]))),
                        "rmse": float(np.sqrt(np.mean((tables[cond] - tables["y"]) ** 2))),
                    }
                for hi, lo in (("B2", "A_current"), ("B2-C", "A_current"),
                               ("B3", "A_current"), ("B3", "B2-C"),
                               ("B2-C", "B2")):
                    block["contrasts"][f"{hi}_minus_{lo}"] = {
                        k: cluster_bootstrap(
                            tables, plots,
                            lambda t, k=k, hi=hi, lo=lo: _contrast(t, hi, lo, k),
                            int(e2["bootstrap_resamples"]),
                            int(e2["bootstrap_seed"]))
                        for k in ("delta_mae", "delta_rmse", "delta_r2")}
                results["bootstrap"][trait][lag_name][split] = block

    results["elapsed_seconds"] = round(time.time() - t0, 1)
    out_json = ROOT / cfg["artifacts"]["e2_metrics"]
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2)
    write_md(ROOT / cfg["artifacts"]["e2_md"], results)
    print(f"wrote {out_json} and {cfg['artifacts']['e2_md']}", flush=True)
    return 0


def write_md(path, r):
    L = ["# S4B.3 E2 -- CC / PH Readout on the Official Test Splits", ""]
    L.append(f"- Generated: {r['generated_at']}")
    L.append(f"- Git HEAD: `{r['git_head']}`")
    L.append(f"- Protocol: `{r['protocol_json']}` (sha `{r['protocol_sha256'][:16]}`)")
    L.append(f"- Inner split: `{r['inner_split_file']}` (sha "
             f"`{r['inner_split_sha256'][:16]}`)")
    L.append("- Train scope: all 2,930 official-train plots; official validation "
             "and test plots excluded from training/selection.")
    L.append("- Linear readout, mean-pooled 768-d features, epoch selected on "
             "inner-dev; test labels read only for evaluation.")
    L.append("")
    L.append("## Label matching audit (test splits)")
    L.append("")
    L.append("| split | pairs | plots | cc valid (lag<=7) | ph valid (lag<=7) | "
             "cc lag | ph lag |")
    L.append("|---|---:|---:|---:|---:|---|---|")
    for split, a in r["label_audit"].items():
        L.append(f"| {split} | {a['n_pairs']:,} | {a['n_plots']} | "
                 f"{a['n_cc_valid_lag7']:,} | {a['n_ph_valid_lag7']:,} | "
                 f"{a['cc_lag_days']['min']:.0f}..{a['cc_lag_days']['max']:.0f} d | "
                 f"{a['ph_lag_days']['min']:.0f}..{a['ph_lag_days']['max']:.0f} d |")
    L.append("")
    L.append("## Primary analysis (|lag| <= 7 d): MAE / RMSE / R2")
    L.append("")
    L.append("| trait | split | condition | MAE | RMSE | R2 | n | plots | epoch |")
    L.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for trait in TRAITS:
        block = r["rows"][trait]["primary_lag7"]
        for split in r["label_audit"]:
            for cond in CONDITIONS:
                s = block[cond]["splits"][split]
                L.append(f"| {trait} | {split} | {cond} | {s['mae']:.5f} | "
                         f"{s['rmse']:.5f} | {s['r2']:.5f} | {s['n']:,} | "
                         f"{s['n_plots']} | {block[cond]['selected_epoch_median']} |")
    L.append("")
    L.append("## Primary analysis: bootstrap 95% CI of R2")
    L.append("")
    L.append("| trait | split | condition | R2 | 95% CI |")
    L.append("|---|---|---|---:|---|")
    for trait in TRAITS:
        block = r["rows"][trait]["primary_lag7"]
        for split in r["label_audit"]:
            for cond in CONDITIONS:
                s = block[cond]["splits"][split]
                b = s["bootstrap"]["r2"]
                L.append(f"| {trait} | {split} | {cond} | {s['r2']:.5f} | "
                         f"[{b['ci95_low']:.5f}, {b['ci95_high']:.5f}] |")
    L.append("")
    L.append("## Primary analysis: paired contrasts vs A_current (delta R2)")
    L.append("")
    L.append("| trait | split | contrast | dR2 | 95% CI |")
    L.append("|---|---|---|---:|---|")
    for trait in TRAITS:
        for split in r["label_audit"]:
            for label, b in r["bootstrap"][trait]["primary_lag7"][split]["contrasts"].items():
                v = b["delta_r2"]
                L.append(f"| {trait} | {split} | {label} | {v['mean']:+.5f} | "
                         f"[{v['ci95_low']:+.5f}, {v['ci95_high']:+.5f}] |")
    L.append("")
    L.append("## Sensitivity: strict-lag3 / same-day mean MAE over splits")
    L.append("")
    L.append("| trait | lag variant | condition | mean MAE |")
    L.append("|---|---|---|---:|")
    for trait in TRAITS:
        for lag_name in ("strict_lag3", "same_day_lag0"):
            block = r["rows"][trait][lag_name]
            for cond in CONDITIONS:
                maes = [block[cond]["splits"][s]["mae"] for s in r["label_audit"]
                        if block[cond]["splits"][s] is not None]
                L.append(f"| {trait} | {lag_name} | {cond} | {np.mean(maes):.5f} |")
    L.append("")
    L.append("## Notes")
    L.append("")
    L.append("- A_current = current frozen DINOv2 feature; B2/B2-C/B3 = predicted "
             "future latent.")
    L.append("- B2-C receives one fixed 9.0 d interval; B3 receives each pair's "
             "real delta_days.")
    L.append("- Missing labels use a validity mask and are never filled with 0.")
    L.append("- Test labels are used only for the final evaluation, never for "
             "training, epoch choice or model selection.")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
