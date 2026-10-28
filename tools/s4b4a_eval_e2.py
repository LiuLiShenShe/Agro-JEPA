#!/usr/bin/env python3
"""S4B.4A E2: CC / PH readout on each fold's held-out year.

Per fold and per trait:

* feature = mean-pooled 768-d vector: ``A_current`` uses the current frozen
  DINOv2 feature ``z_a``; every other arm uses the seed-averaged predicted
  future latent ``z_hat_b``;
* a linear readout is trained on the fold-train rows, its epoch is chosen on the
  fold-dev rows, and the whole held-out year is evaluated ONCE;
* label windows: primary |lag| <= 7 d, plus strict-lag3 and same-day
  sensitivities; missing labels use a validity mask and are never filled with 0;
* paired contrasts (delta MAE / RMSE / R2) against ``A_current`` and against
  ``B2-C`` on the identical held-out rows, with plot-clustered bootstrap CIs.

Held-out labels are read only for the final evaluation -- never for readout
training, epoch choice or hyper-parameter tuning.

Usage
-----
    CUDA_VISIBLE_DEVICES=1 python3 tools/s4b4a_eval_e2.py \
        --config agro_jepa/config/s4b4a_year_heldout.yaml
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
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from agro_jepa.datasets.s4_dataset import ConditioningSpec  # noqa: E402
from agro_jepa.models.s4_predictor import conditioning_spec, has_fixed_delta  # noqa: E402
from s3_1_diagnostics import cluster_bootstrap, git, sha256_file  # noqa: E402
from s3_2_probe_eval import _metrics, predict_readout, train_readout  # noqa: E402

# E2 arm name -> training-condition key (None = current feature)
ARM_TO_COND = {
    "A_current": None,
    "b2_no_time": "b2_no_time",
    "b2_no_time_constant_delta": "b2_no_time_constant_delta",
    "b3_delta": "b3_delta",
    "gdd_p": "gdd_p",
    "weather_p": "weather_p",
}


class PoolDataset(Dataset):
    """One row -> current feature plus every conditioning array any arm needs.

    The frame is a superset: arms that do not use GDD / weather simply ignore
    the extra tensors, so a single DataLoader pass can pool every arm.
    """

    def __init__(self, frame: pd.DataFrame, cond_arrays: dict):
        self.rows = frame.to_dict("records")
        self.cond = cond_arrays
        self._gdd_keys = tuple(f"gdd_past{w}_base0" for w in (7, 14, 28))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        return {
            "z_a": torch.from_numpy(np.load(r["cache_file_a"]).astype(np.float32)),
            "delta_days": torch.tensor(float(r["delta_days"]), dtype=torch.float32),
            "gdd": torch.tensor([float(self.cond[k][i]) for k in self._gdd_keys],
                                dtype=torch.float32),
            "weather": torch.from_numpy(
                np.asarray(self.cond["weather_past"][i], dtype=np.float32)),
            "weather_mask": torch.from_numpy(
                np.asarray(self.cond["weather_past_mask"][i], dtype=bool)),
        }


def slice_conditioning(cond_full: dict, orig_rows: np.ndarray) -> dict:
    n_ref = cond_full["delta_days"].shape[0]
    return {k: (v[orig_rows] if isinstance(v, np.ndarray) and v.ndim >= 1
                and v.shape[0] == n_ref else v) for k, v in cond_full.items()}


@torch.no_grad()
def pool_all_arms(arm_models, frame, cond_arrays, device, batch_size) -> dict:
    """ONE pass over ``frame``; returns {arm: (n, 768) pooled feature array}.

    ``arm_models`` maps arm -> list of models (empty list = current feature).
    """
    ds = PoolDataset(frame, cond_arrays)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=8,
                        pin_memory=True)
    sums = {arm: [] for arm in arm_models}
    for batch in loader:
        z_a = batch["z_a"].to(device, non_blocking=True).float()
        for arm, models in arm_models.items():
            if not models:
                sums[arm].append(z_a.mean(dim=1).cpu().numpy())
                continue
            parts = getattr(models[0], "parts", [])
            kw = {}
            if "delta" in parts and not has_fixed_delta(models[0]):
                kw["delta_days"] = batch["delta_days"].to(device, non_blocking=True)
            if "gdd" in parts:
                kw["gdd"] = batch["gdd"].to(device, non_blocking=True)
            if "weather" in parts:
                kw["weather"] = batch["weather"].to(device, non_blocking=True)
                kw["weather_mask"] = batch["weather_mask"].to(device, non_blocking=True)
            pred = None
            for m in models:
                out = m(z_a, **kw).float()
                pred = out if pred is None else pred + out
            sums[arm].append((pred / len(models)).mean(dim=1).cpu().numpy())
    return {arm: np.concatenate(v, axis=0).astype(np.float32)
            for arm, v in sums.items()}


def run_readout(x_tr, y_tr, x_dev, y_dev, x_te, pcfg, seeds, device):
    epochs = []
    for seed in seeds:
        _, info = train_readout(x_tr, y_tr, pcfg, int(seed), device,
                                epochs_to_run=0, x_dev=x_dev, y_dev=y_dev,
                                select=True)
        epochs.append(int(info["best_epoch"]))
    selected = int(round(float(np.median(epochs))))
    preds = []
    for seed in seeds:
        state, _ = train_readout(x_tr, y_tr, pcfg, int(seed), device,
                                 epochs_to_run=selected, select=False)
        preds.append(predict_readout(state, x_te, device))
    return np.mean(preds, axis=0), selected, epochs


def _metric_fn(key):
    if key == "mae":
        return lambda t: float(np.mean(np.abs(t["p"] - t["y"])))
    if key == "rmse":
        return lambda t: float(np.sqrt(np.mean((t["p"] - t["y"]) ** 2)))
    return lambda t: float(1.0 - np.sum((t["p"] - t["y"]) ** 2)
                           / np.sum((t["y"] - t["y"].mean()) ** 2))


def _contrast_fn(hi, lo, key):
    def f(t):
        eh, el = t[hi] - t["y"], t[lo] - t["y"]
        ss = float(np.sum((t["y"] - t["y"].mean()) ** 2))
        if key == "delta_mae":
            return float(np.mean(np.abs(eh)) - np.mean(np.abs(el)))
        if key == "delta_rmse":
            return float(np.sqrt(np.mean(eh ** 2)) - np.sqrt(np.mean(el ** 2)))
        return float((-np.sum(eh ** 2) + np.sum(el ** 2)) / ss)
    return f


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s4b4a_year_heldout.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument("--folds", default=None)
    ap.add_argument("--seeds", default=None, help="model seeds for pooling")
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    from s4b4a_eval_e1 import FEATURE_SPEC, load_model

    t0 = time.time()
    device = torch.device(args.device or (cfg["eval"]["device"]
                          if torch.cuda.is_available() else "cpu"))
    seeds = [int(s) for s in (args.seeds.split(",") if args.seeds
                              else cfg["training"]["seeds"])]
    fold_years = ([int(s) for s in args.folds.split(",")] if args.folds
                  else [int(y) for y in cfg["folds"]["heldout_year_per_fold"]])
    fold_dir = ROOT / cfg["artifacts"]["fold_dir"]
    cond_full = dict(np.load(ROOT / cfg["data"]["conditioning_npz"], allow_pickle=True))
    e2 = cfg["e2"]
    e2_seeds = [int(s) for s in e2["seeds"]]
    pcfg = {"lr": float(e2["lr"]), "weight_decay": float(e2["weight_decay"]),
            "batch_size": int(e2["batch_size"]),
            "candidate_epochs": int(e2["candidate_epochs"]),
            "patience": int(e2["patience"]), "seeds": e2_seeds,
            "bootstrap_resamples": int(e2["bootstrap_resamples"]),
            "bootstrap_seed": int(e2["bootstrap_seed"])}
    lag_variants = {k: int(v) for k, v in e2["lag_variants"].items()}
    arms = list(cfg["e2_conditions"])
    contrasts = [("b2_no_time", "A_current"),
                 ("b2_no_time_constant_delta", "A_current"),
                 ("b3_delta", "A_current"),
                 ("b3_delta", "b2_no_time_constant_delta"),
                 ("gdd_p", "b3_delta"),
                 ("weather_p", "b3_delta")]

    report = {
        "stage": "S4B.4A",
        "task": "E2 CC/PH readout on each fold's held-out year",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": " ".join(sys.argv),
        "git_head": git("rev-parse", "HEAD"),
        "config": args.config,
        "config_sha256": sha256_file(ROOT / args.config),
        "device": str(device),
        "model_seeds": seeds,
        "readout_seeds": e2_seeds,
        "lag_variants": lag_variants,
        "note": ("readout trained on fold-train, epoch chosen on fold-dev, "
                 "held-out year evaluated once; held-out labels never used for "
                 "training/selection"),
        "folds": {},
    }

    for fold_year in fold_years:
        df_train = pd.read_parquet(fold_dir / f"fold{fold_year}_fold_train.parquet")
        df_dev = pd.read_parquet(fold_dir / f"fold{fold_year}_fold_dev.parquet")
        df_held = pd.read_parquet(fold_dir / f"fold{fold_year}_heldout.parquet")
        frames = []
        for f, role in ((df_train, "fold_train"), (df_dev, "fold_dev"),
                        (df_held, "heldout")):
            g = f.copy()
            g["role"] = role
            frames.append(g)
        combined = pd.concat(frames, ignore_index=True).reset_index(drop=True)
        cond_arrays = slice_conditioning(cond_full,
                                         combined["_orig_row"].to_numpy(np.int64))
        roles = combined["role"].to_numpy()
        idx_train = np.where(roles == "fold_train")[0]
        idx_dev = np.where(roles == "fold_dev")[0]
        idx_held = np.where(roles == "heldout")[0]
        print(f"[fold {fold_year}] train {len(idx_train)} dev {len(idx_dev)} "
              f"heldout {len(idx_held)}", flush=True)

        arm_models = {}
        for arm in arms:
            cond_key = ARM_TO_COND[arm]
            if cond_key is None:
                arm_models[arm] = []
                continue
            spec = conditioning_spec(cfg["conditions"][cond_key])
            models = []
            for seed in seeds:
                ck_p = (Path(cfg["training"]["output_root"]) / f"fold{fold_year}"
                        / cond_key / f"seed{seed}" / "checkpoints" / "best.pt")
                if not ck_p.is_file():
                    raise SystemExit(f"missing checkpoint {ck_p}")
                m, _ = load_model(ck_p, device)
                models.append(m)
            arm_models[arm] = models
        pooled = pool_all_arms(arm_models, combined, cond_arrays, device,
                               int(e2["batch_size"]))
        del arm_models
        torch.cuda.empty_cache()
        print(f"[fold {fold_year}] pooled {len(pooled)} arms", flush=True)

        fold_block = {"fold_year": fold_year, "n_train": int(len(idx_train)),
                      "n_dev": int(len(idx_dev)), "n_heldout": int(len(idx_held)),
                      "traits": {}}
        for trait in cfg["data"]["traits"]:
            lag_col = combined[f"{trait}_b_lag"].to_numpy(np.float64)
            y_all = combined[f"{trait}_b"].to_numpy(np.float64)
            mask_all = combined[f"{trait}_mask_b"].to_numpy(bool)
            fold_block["traits"][trait] = {}
            for lag_name, lag_max in lag_variants.items():
                ok = mask_all & np.isfinite(y_all) & (np.abs(lag_col) <= lag_max)
                tr = idx_train[ok[idx_train]]
                dv = idx_dev[ok[idx_dev]]
                he = idx_held[ok[idx_held]]
                block = {"n_train": int(len(tr)), "n_dev": int(len(dv)),
                         "n_heldout": int(len(he)),
                         "n_heldout_plots": int(combined.iloc[he].plot_uid.nunique()),
                         "arms": {}, "contrasts": {}}
                if len(tr) < 10 or len(dv) < 5 or len(he) < 5:
                    block["note"] = "too few valid rows"
                    fold_block["traits"][trait][lag_name] = block
                    continue
                y_he = y_all[he]
                plots = combined.iloc[he].plot_uid.to_numpy()
                preds = {}
                for arm in arms:
                    pred, selected, epochs = run_readout(
                        pooled[arm][tr], y_all[tr], pooled[arm][dv], y_all[dv],
                        pooled[arm][he], pcfg, e2_seeds, device)
                    preds[arm] = pred
                    m = _metrics(y_he, pred)
                    boot = {k: cluster_bootstrap(
                        {"y": y_he.astype(np.float64), "p": pred.astype(np.float64)},
                        plots, _metric_fn(k), int(e2["bootstrap_resamples"]),
                        int(e2["bootstrap_seed"])) for k in ("mae", "rmse", "r2")}
                    block["arms"][arm] = {
                        "selected_epoch_median": selected,
                        "selected_epochs_per_seed": epochs,
                        "mae": float(m["mae"]), "rmse": float(m["rmse"]),
                        "r2": float(m["r2"]), "n": int(m["n"]),
                        "bootstrap": boot}
                tables = {"y": y_he.astype(np.float64),
                          **{a: preds[a].astype(np.float64) for a in arms}}
                for hi, lo in contrasts:
                    if hi not in preds or lo not in preds:
                        continue
                    block["contrasts"][f"{hi}_minus_{lo}"] = {
                        k: cluster_bootstrap(tables, plots, _contrast_fn(hi, lo, k),
                                             int(e2["bootstrap_resamples"]),
                                             int(e2["bootstrap_seed"]))
                        for k in ("delta_mae", "delta_rmse", "delta_r2")}
                fold_block["traits"][trait][lag_name] = block
                print(f"[fold {fold_year}] {trait}/{lag_name}: "
                      + " ".join(f"{a}={block['arms'][a]['r2']:.4f}" for a in arms),
                      flush=True)
        report["folds"][str(fold_year)] = fold_block
        del pooled
        torch.cuda.empty_cache()

    report["elapsed_seconds"] = round(time.time() - t0, 1)
    out_json = ROOT / cfg["artifacts"]["e2_json"]
    with open(out_json, "w") as fh:
        json.dump(report, fh, indent=2)
    write_md(ROOT / cfg["artifacts"]["e2_md"], report)
    print(f"wrote {out_json}", flush=True)
    return 0


def write_md(path: Path, r: dict) -> None:
    L = ["# S4B.4A E2 -- CC / PH Readout by Held-Out Year", ""]
    L.append(f"- Generated: {r['generated_at']}")
    L.append(f"- Git HEAD: `{r['git_head']}`")
    L.append(f"- Model seeds pooled: {r['model_seeds']}; readout seeds: "
             f"{r['readout_seeds']}")
    L.append("- Readout trained on fold-train, epoch on fold-dev, held-out year "
             "evaluated once; held-out labels never used for training/selection.")
    L.append("")
    for y, e in r["folds"].items():
        L.append(f"## Fold {y} (held-out year)")
        L.append("")
        for trait, lags in e["traits"].items():
            for lag_name, b in lags.items():
                if "arms" not in b:
                    continue
                L.append(f"### {trait} / {lag_name} "
                         f"(n_heldout={b['n_heldout']}, plots={b['n_heldout_plots']})")
                L.append("")
                L.append("| arm | MAE | RMSE | R2 | R2 95% CI | epoch |")
                L.append("|---|---:|---:|---:|---|---:|")
                for arm, v in b["arms"].items():
                    ci = v["bootstrap"]["r2"]
                    L.append(f"| {arm} | {v['mae']:.5f} | {v['rmse']:.5f} | "
                             f"{v['r2']:.5f} | [{ci['ci95_low']:.5f}, "
                             f"{ci['ci95_high']:.5f}] | {v['selected_epoch_median']} |")
                L.append("")
                L.append("| contrast | dR2 | 95% CI |")
                L.append("|---|---:|---|")
                for label, c in b["contrasts"].items():
                    v = c["delta_r2"]
                    L.append(f"| {label} | {v['mean']:+.5f} | "
                             f"[{v['ci95_low']:+.5f}, {v['ci95_high']:+.5f}] |")
                L.append("")
        L.append("")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
