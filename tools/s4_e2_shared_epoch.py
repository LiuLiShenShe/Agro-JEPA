#!/usr/bin/env python3
"""S4A E2 shared-epoch control (the S3.2 protocol's second readout regime).

The primary E2 readout lets each condition pick its own inner-dev epoch.  The
selected epochs can differ a lot (a linear readout on a better future feature
sometimes prefers a longer fit), so S3.2 also reported a control where **every**
condition uses one shared epoch: the median of all inner-dev selections across
all conditions and probe seeds.

This tool re-runs only that control on the headline scope
(``primary_lag7 / overall``), reusing the selection epochs already stored in
``reports/s4_e2_probe_metrics.json`` -- no selection is repeated and the
official validation split is used for evaluation only.

Usage
-----
    CUDA_VISIBLE_DEVICES=1 python3 tools/s4_e2_shared_epoch.py \
        --config agro_jepa/config/s4_pilot.yaml
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

from agro_jepa.datasets.s4_dataset import build_role_map  # noqa: E402
from agro_jepa.eval.eval_s4 import scope_frames  # noqa: E402
from s3_1_diagnostics import cluster_bootstrap  # noqa: E402
from s3_2_probe_eval import _metrics, predict_readout, train_readout  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s4_pilot.yaml")
    ap.add_argument("--e2", default="reports/s4_e2_probe_metrics.json")
    ap.add_argument("--out", default="reports/s4_e2_shared_epoch.json")
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    pcfg = cfg["probe"]
    with open(args.e2) as fh:
        e2 = json.load(fh)
    device = torch.device(cfg["eval"]["device"] if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    pooled_dir = Path(cfg["artifacts"]["e2_pooled_dir"])
    conditions = list(cfg["e2_conditions"])
    base_meta = None
    loaded = {}
    for name in conditions:
        z = np.load(pooled_dir / f"{name}.npz", allow_pickle=True)
        meta = pd.read_parquet(pooled_dir / f"{name}_meta.parquet")
        if base_meta is None:
            base_meta = meta
        elif not np.array_equal(meta.pair_id.to_numpy(), base_meta.pair_id.to_numpy()):
            raise RuntimeError(f"pooled rows are not aligned for {name}")
        loaded[name] = (np.asarray(z["pool_a"]), np.asarray(z["pool_b"]))
    index = pd.read_parquet(cfg["data"]["index"])
    role_map, _ = build_role_map(
        index, int(pcfg["plot_split_seed"]), float(pcfg["inner_train_fraction"])
    )
    role = pd.Series(role_map)

    out = {"stage": "S4A", "task": "E2 shared-epoch control (primary_lag7 / overall)",
           "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "command": " ".join(sys.argv),
           "selection_source": "stored inner-dev selections from the primary E2 run",
           "official_validation_used_for_selection": False,
           "protocol": {
               "shared_epoch_rule": (
                   "median of all inner-dev selected epochs across A and every B/C "
                   "condition and every probe seed"
               ),
               "refit": "all official-train plots",
               "evaluation": "official validation, once",
           },
           "traits": {}}

    for trait in ("cc", "ph"):
        frame = scope_frames(base_meta, trait)["overall"]
        frame = frame[frame[f"{trait}_b_lag"].abs() <= int(pcfg["lag_variants"]["primary"])]
        tr = frame[frame.split == "train"].copy()
        va = frame[frame.split == "validation"].copy().reset_index(drop=True)
        tr["role"] = tr.plot_uid.map(role)
        tr_in = tr[tr.role == "inner_train"]
        tr_dev = tr[tr.role == "inner_dev"]
        y_in = tr_in[f"{trait}_b"].to_numpy(np.float32)
        y_dev = tr_dev[f"{trait}_b"].to_numpy(np.float32)
        y_tr = tr[f"{trait}_b"].to_numpy(np.float32)
        y_va = va[f"{trait}_b"].to_numpy(np.float32)

        def xmat(df, which, name):
            pa, pb = loaded[name]
            return (pa if which == "a" else pb)[df["_pool_index"].to_numpy()]

        # collect the stored selections
        blk = e2["rows"]["primary_lag7"][trait]["overall"]["conditions"]
        epochs = []
        for label, m in blk.items():
            epochs.extend(int(e) for e in m["selected_epochs_per_seed"])
        shared = int(round(float(np.median(epochs))))

        results = {}
        preds = {}
        for label in blk:
            if label == "A_current":
                src = conditions[0]
                x_tr = xmat(tr, "a", src)
                x_va = xmat(va, "a", src)
            else:
                name = label[2:] if label.startswith("B_") else label
                x_tr = xmat(tr, "b", name)
                x_va = xmat(va, "b", name)
            runs = []
            seed_preds = []
            for seed in pcfg["seeds"]:
                state, info = train_readout(
                    x_tr, y_tr, pcfg, int(seed), device,
                    epochs_to_run=shared, select=False)
                pred = predict_readout(state, x_va, device)
                info["metrics"] = _metrics(y_va, pred)
                runs.append(info)
                seed_preds.append(pred)
            agg = {}
            for k in ("mae", "rmse", "r2"):
                vals = np.array([r["metrics"][k] for r in runs], float)
                agg[k] = {"mean": float(vals.mean()), "std": float(vals.std()),
                          "values": vals.tolist()}
            agg["n_val"] = int(len(va))
            agg["n_val_plots"] = int(va.plot_uid.nunique())
            agg["n_train"] = int(len(tr))
            results[label] = agg
            preds[label] = np.mean(seed_preds, axis=0)
            print(f"[shared {trait}] {label}: R2={agg['r2']['mean']:.5f}", flush=True)

        # bootstrap B-A on the shared-epoch predictions
        plots = va.plot_uid.to_numpy()
        boot = {}
        a = preds["A_current"]
        for label, pb in preds.items():
            if label == "A_current":
                continue
            tables = {"y": y_va.astype(np.float64), "pa": a.astype(np.float64),
                      "pb": pb.astype(np.float64)}

            def metrics_delta(t):
                ea = t["pa"] - t["y"]
                eb = t["pb"] - t["y"]
                ss = float(np.sum((t["y"] - t["y"].mean()) ** 2))
                return {
                    "delta_mae": float(np.mean(np.abs(eb)) - np.mean(np.abs(ea))),
                    "delta_rmse": float(np.sqrt(np.mean(eb ** 2))
                                        - np.sqrt(np.mean(ea ** 2))),
                    "delta_r2": ((-np.sum(eb ** 2) + np.sum(ea ** 2)) / ss)
                    if ss > 0 else float("nan"),
                }

            boot[label] = {
                k: cluster_bootstrap(
                    tables, plots, lambda t, k=k: metrics_delta(t)[k],
                    int(pcfg["bootstrap_resamples"]), int(pcfg["bootstrap_seed"]))
                for k in ("delta_mae", "delta_rmse", "delta_r2")
            }
        out["traits"][trait] = {
            "shared_epoch": shared,
            "n_selections": int(len(epochs)),
            "selected_epochs": epochs,
            "conditions": results,
            "bootstrap_B_minus_A": boot,
        }

    out["elapsed_seconds"] = round(time.time() - t0, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
