#!/usr/bin/env python3
"""S4A.1 E2: CC / PH phenotype readout for the matched time control.

Reuses the S3.2 corrected probe protocol exactly (inner-dev selection -> refit on
all official-train plots -> one evaluation on official validation -> plot-cluster
bootstrap).  Compared conditions:

    A_current                              current DINOv2 feature
    B_b2_no_time                           no time input at all
    B_b2_no_time_constant_delta            fixed time embedding (parameter matched)
    B_b3_delta                             real per-sample delta_days

Reported per condition: MAE, RMSE, R2, effective sample size, selected epoch, and
the paired B-C / B3 differences with a plot-cluster 95% CI.  The E2 result is
auxiliary: a worse E1 for B3 does not automatically imply a worse phenotype
readout, and the report says so explicitly.

Usage
-----
    CUDA_VISIBLE_DEVICES=1 python3 tools/s4a1_e2_eval.py \
        --config agro_jepa/config/s4a1_matched_control.yaml
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

from agro_jepa.datasets.s3_pairs import index_sha256  # noqa: E402
from agro_jepa.datasets.s4_dataset import (  # noqa: E402
    build_role_map,
    load_conditioning,
)
from agro_jepa.eval.eval_s4 import (  # noqa: E402
    condition_spec,
    load_condition_models,
    run_dir,
    scope_frames,
)
from agro_jepa.eval.s4_common import collect_pooled  # noqa: E402
from s3_1_diagnostics import cluster_bootstrap, git, sha256_file  # noqa: E402
from s3_2_probe_eval import _metrics, predict_readout, train_readout  # noqa: E402

CONTROL = "b2_no_time_constant_delta"
MATCHED = "b3_delta"
NO_TIME = "b2_no_time"
CONDITIONS = [NO_TIME, CONTROL, MATCHED]
CONTRASTS = [
    ("B_b3_delta", "B_b2_no_time_constant_delta"),
    ("B_b3_delta", "B_b2_no_time"),
    ("B_b2_no_time_constant_delta", "B_b2_no_time"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s4a1_matched_control.yaml")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    t0 = time.time()
    pcfg = cfg["probe"]
    device = torch.device(cfg["eval"]["device"] if torch.cuda.is_available() else "cpu")

    index = pd.read_parquet(cfg["data"]["index"])
    cond = load_conditioning(cfg["data"]["conditioning_npz"], index)
    traits = tuple(cfg["data"]["traits"])

    pooled_dir = Path(cfg["artifacts"]["e2_pooled_dir"])
    pooled_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    loaded = {}
    base_meta = None
    for name in CONDITIONS:
        models, _ = load_condition_models(cfg, name, device)
        spec = condition_spec(cfg, name)
        meta, pool_a, pool_b = collect_pooled(
            models, index, cond, spec, traits, device,
            int(cfg["eval"]["batch_size"]))
        path = pooled_dir / f"{name}.npz"
        np.savez_compressed(path, pair_id=meta.pair_id.to_numpy().astype(str),
                            pool_a=pool_a, pool_b=pool_b)
        meta.to_parquet(pooled_dir / f"{name}_meta.parquet", index=False)
        manifest[name] = {
            "npz": str(path), "sha256": sha256_file(path),
            "n_rows": int(len(meta)),
            "n_train": int((meta.split == "train").sum()),
            "n_validation": int((meta.split == "validation").sum()),
        }
        loaded[name] = (meta, pool_a, pool_b)
        if base_meta is None:
            base_meta = meta
        elif not np.array_equal(meta.pair_id.to_numpy(), base_meta.pair_id.to_numpy()):
            raise RuntimeError(f"pooled rows are not aligned for {name}")
        print(f"[E2 collect] {name}: {len(meta)} rows", flush=True)

    role_map, split_stats = build_role_map(
        index, int(pcfg["plot_split_seed"]), float(pcfg["inner_train_fraction"]))
    role = pd.Series(role_map)

    results = {
        "stage": "S4A.1",
        "task": "E2 CC/PH readout (S3.2 corrected protocol)",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": " ".join(sys.argv),
        "git_head": git("rev-parse", "HEAD"),
        "device": str(device),
        "index_sha256": index_sha256(index),
        "split": split_stats,
        "pooled": manifest,
        "protocol": {
            "architecture": "single linear layer (PooledReadout, 768 -> 1)",
            "pooling": "mean over 256 patch tokens, identical for A and B",
            "candidate_epochs": int(pcfg["candidate_epochs"]),
            "patience": int(pcfg["patience"]),
            "seeds": [int(s) for s in pcfg["seeds"]],
            "selection": "minimum inner-dev MSE; refit on all official-train plots",
            "official_validation_used_for_selection": False,
            "bootstrap": {"unit": "plot_uid",
                          "n_resamples": int(pcfg["bootstrap_resamples"]),
                          "seed": int(pcfg["bootstrap_seed"])},
            "lag_variants": {"primary_lag7": int(pcfg["lag_variants"]["primary"])},
        },
        "interpretation_limit": (
            "E2 is auxiliary. A better E1 latent MSE does not guarantee a better "
            "phenotype readout, and a B3 E2 gain does not by itself prove that the "
            "sample-level delta improves phenotype prediction."
        ),
        "rows": {},
        "bootstrap": {},
    }

    pred_rows = []
    lag_max = int(pcfg["lag_variants"]["primary"])
    for trait in cfg["data"]["traits"]:
        frames = scope_frames(base_meta, trait)
        frame = frames["overall"]
        frame = frame[frame[f"{trait}_b_lag"].abs() <= lag_max]
        tr = frame[frame.split == "train"].copy()
        va = frame[frame.split == "validation"].copy().reset_index(drop=True)
        tr["role"] = tr.plot_uid.map(role)
        tr_in = tr[tr.role == "inner_train"]
        tr_dev = tr[tr.role == "inner_dev"]
        counts = {
            "lag_max_abs_days": lag_max,
            "n_train": int(len(tr)), "n_train_plots": int(tr.plot_uid.nunique()),
            "n_inner_train": int(len(tr_in)),
            "n_inner_train_plots": int(tr_in.plot_uid.nunique()),
            "n_inner_dev": int(len(tr_dev)),
            "n_inner_dev_plots": int(tr_dev.plot_uid.nunique()),
            "n_val": int(len(va)), "n_val_plots": int(va.plot_uid.nunique()),
        }
        y_in = tr_in[f"{trait}_b"].to_numpy(np.float32)
        y_dev = tr_dev[f"{trait}_b"].to_numpy(np.float32)
        y_tr = tr[f"{trait}_b"].to_numpy(np.float32)
        y_va = va[f"{trait}_b"].to_numpy(np.float32)

        def xmat(df, which, name):
            _, pa, pb = loaded[name]
            return (pa if which == "a" else pb)[df["_pool_index"].to_numpy()]

        block = {"counts": counts, "conditions": {}}
        plan = [("A_current", "a", CONDITIONS[0])] + [
            (f"B_{n}", "b", n) for n in CONDITIONS]
        for label, which, name in plan:
            src = name
            x_in = xmat(tr_in, which, src)
            x_dev = xmat(tr_dev, which, src)
            x_tr = xmat(tr, which, src)
            x_va = xmat(va, which, src)
            runs, epochs_sel = [], []
            for seed in pcfg["seeds"]:
                state, info = train_readout(
                    x_in, y_in, pcfg, int(seed), device,
                    epochs_to_run=0, x_dev=x_dev, y_dev=y_dev, select=True)
                epochs_sel.append(int(info["best_epoch"]))
                runs.append(info)
            selected = int(round(float(np.median(epochs_sel))))
            final_runs = []
            for seed in pcfg["seeds"]:
                state, info = train_readout(
                    x_tr, y_tr, pcfg, int(seed), device,
                    epochs_to_run=selected, select=False)
                pred = predict_readout(state, x_va, device)
                info["metrics"] = _metrics(y_va, pred)
                info["selected_epoch"] = selected
                final_runs.append(info)
                for i in range(len(va)):
                    pred_rows.append({
                        "trait": trait, "condition": label, "seed": int(seed),
                        "pair_id": va.pair_id.iloc[i],
                        "plot_uid": va.plot_uid.iloc[i],
                        "lag_days": float(va[f"{trait}_b_lag"].iloc[i]),
                        "y_true": float(y_va[i]), "y_pred": float(pred[i]),
                    })
            agg = {}
            for k in ("mae", "rmse", "r2"):
                vals = np.array([r["metrics"][k] for r in final_runs], float)
                agg[k] = {"mean": float(vals.mean()), "std": float(vals.std()),
                          "values": vals.tolist()}
            agg["selected_epoch_median"] = selected
            agg["selected_epochs_per_seed"] = epochs_sel
            agg["n_val"] = int(len(va))
            agg["n_val_plots"] = int(va.plot_uid.nunique())
            agg["n_train"] = int(len(tr))
            agg["role"] = ("A_current_feature" if label == "A_current"
                           else "B_predicted")
            block["conditions"][label] = agg
        a = block["conditions"]["A_current"]
        for label, b in block["conditions"].items():
            if label == "A_current":
                continue
            block.setdefault("B_minus_A", {})[label] = {
                k: {"mean_delta": b[k]["mean"] - a[k]["mean"],
                    "a_mean": a[k]["mean"], "b_mean": b[k]["mean"]}
                for k in ("mae", "rmse", "r2")}
        results["rows"][trait] = block
        print(f"[E2] {trait} done (n_val={len(va)})", flush=True)

    preds = pd.DataFrame(pred_rows)
    Path(cfg["artifacts"]["e2_predictions"]).parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(cfg["artifacts"]["e2_predictions"], index=False)

    # plot-cluster bootstrap of the pairwise contrasts (predictions averaged
    # over the probe seeds, exactly as in the S3.2/S4A contrast tool)
    for trait in cfg["data"]["traits"]:
        d = preds[preds.trait == trait]
        base = d[["pair_id", "plot_uid"]].drop_duplicates().sort_values("pair_id")
        base = base.reset_index(drop=True)
        wide = base.copy()
        y = None
        for cond in sorted(d.condition.unique()):
            sub = (d[d.condition == cond]
                   .groupby("pair_id", as_index=False)
                   .agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean")))
            sub = base[["pair_id"]].merge(sub, on="pair_id", how="left")
            if sub.y_pred.isna().any():
                raise RuntimeError(f"{trait}/{cond}: missing pairs")
            wide[cond] = sub.y_pred.to_numpy(np.float64)
            y = sub.y_true.to_numpy(np.float64)
        wide["y"] = y
        plots = wide.plot_uid.to_numpy()
        tables = {"y": y}
        for cond in sorted(d.condition.unique()):
            tables[cond] = wide[cond].to_numpy(np.float64)
        block = {"n_pairs": int(len(wide)), "n_plots": int(wide.plot_uid.nunique()),
                 "r2": {}, "contrasts": {}}

        def _r2(y, p):
            ss = float(np.sum((y - y.mean()) ** 2))
            return 1.0 - float(np.sum((p - y) ** 2)) / ss

        for cond in sorted(d.condition.unique()):
            block["r2"][cond] = _r2(y, tables[cond])

        def _metrics_fn(t, hi, lo):
            y = t["y"]
            eh = t[hi] - y
            el = t[lo] - y
            ss = float(np.sum((y - y.mean()) ** 2))
            return {
                "delta_mae": float(np.mean(np.abs(eh)) - np.mean(np.abs(el))),
                "delta_rmse": float(np.sqrt(np.mean(eh ** 2)) - np.sqrt(np.mean(el ** 2))),
                "delta_r2": (-np.sum(eh ** 2) + np.sum(el ** 2)) / ss,
            }

        for hi, lo in CONTRASTS:
            block["contrasts"][f"{hi}_minus_{lo}"] = {
                k: cluster_bootstrap(
                    tables, plots, lambda t, k=k, hi=hi, lo=lo: _metrics_fn(t, hi, lo)[k],
                    int(pcfg["bootstrap_resamples"]), int(pcfg["bootstrap_seed"]))
                for k in ("delta_mae", "delta_rmse", "delta_r2")}
        results["bootstrap"][trait] = block

    results["elapsed_seconds"] = round(time.time() - t0, 1)
    out = Path(args.out or cfg["artifacts"]["e2_metrics"])
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"wrote {out}", flush=True)
    for trait in cfg["data"]["traits"]:
        for label, m in results["rows"][trait]["conditions"].items():
            print(f"  {trait} {label:34s} R2={m['r2']['mean']:.5f} "
                  f"MAE={m['mae']['mean']:.5f} RMSE={m['rmse']['mean']:.5f} "
                  f"epoch={m['selected_epoch_median']}", flush=True)
        for key, m in results["bootstrap"][trait]["contrasts"].items():
            x = m["delta_r2"]
            print(f"  {trait} {key:52s} dR2 {x['mean']:+.5f} "
                  f"[{x['ci95_low']:+.5f},{x['ci95_high']:+.5f}]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
