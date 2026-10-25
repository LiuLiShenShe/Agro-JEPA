#!/usr/bin/env python3
"""S4A evaluation: E1 future-latent baselines and E2 CC/PH phenotype readout.

E1
    Every trained condition is evaluated on the official validation split with
    predictions averaged over the fixed seed set.  B0 persistence and B1
    train-mean are computed directly (no training).  Per-sample rows are written
    to parquet so every aggregate can be audited and bootstrapped.

E2
    The S3.2 corrected readout protocol, unchanged: selection on an inner dev
    set of official-train plots, refit on all official-train plots, one
    evaluation on official validation, plot-cluster bootstrap.  Condition A is
    the current DINOv2 feature; each B is a condition's predicted future
    feature.  Predicted features are averaged over the fixed seed set.

Usage
-----
    CUDA_VISIBLE_DEVICES=1 python3 agro_jepa/eval/eval_s4.py --mode e1
    CUDA_VISIBLE_DEVICES=1 python3 agro_jepa/eval/eval_s4.py --mode e2
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
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from agro_jepa.datasets.s3_pairs import index_sha256  # noqa: E402
from agro_jepa.datasets.s4_dataset import (  # noqa: E402
    build_role_map,
    conditioning_rows,
    load_conditioning,
    split_rows,
)
from agro_jepa.eval.s4_common import (  # noqa: E402
    collect_pooled,
    forward_batch,
    load_s4_model,
    make_dataset,
    mean_future_feature,
)
from agro_jepa.models.s4_predictor import conditioning_spec  # noqa: E402
from s3_1_diagnostics import cluster_bootstrap, git, sha256_file  # noqa: E402
from s3_2_probe_eval import (  # noqa: E402
    _metrics,
    predict_readout,
    train_readout,
)


def sha256_file_local(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_dir(cfg: dict, condition: str, seed: int) -> Path:
    return Path(cfg["training"]["output_root"]) / condition / f"seed{seed}"


def load_condition_models(cfg: dict, condition: str, device: torch.device):
    models, ckpts = [], []
    for seed in cfg["training"]["seeds"]:
        ckpt = run_dir(cfg, condition, int(seed)) / "checkpoints" / "best.pt"
        if not ckpt.exists():
            raise FileNotFoundError(
                f"missing checkpoint for {condition} seed{seed}: {ckpt}. "
                "Run tools/s4_train_pilot.py first."
            )
        m, c = load_s4_model(str(ckpt), device)
        models.append(m)
        ckpts.append(c)
    return models, ckpts


def condition_spec(cfg: dict, condition: str):
    return conditioning_spec(cfg["conditions"][condition])


# --------------------------------------------------------------------------
# E1
# --------------------------------------------------------------------------
@torch.no_grad()
def e1_condition(
    models: list,
    frame: pd.DataFrame,
    cond: dict,
    spec,
    traits,
    device: torch.device,
    batch_size: int,
    train_mean: torch.Tensor,
) -> pd.DataFrame:
    ds = make_dataset(frame, cond, spec, traits)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4,
                        pin_memory=True)
    z_mean = train_mean.to(device)
    rows = []
    for batch in loader:
        z_a = batch["z_a"].to(device, non_blocking=True).float()
        z_b = batch["z_b"].to(device, non_blocking=True).float()
        z_pred = None
        for m in models:
            out = forward_batch(m, batch, device).float()
            z_pred = out if z_pred is None else z_pred + out
        z_pred = z_pred / len(models)
        mse_pred = (z_pred - z_b).pow(2).mean(dim=(1, 2))
        mse_persist = (z_a - z_b).pow(2).mean(dim=(1, 2))
        mse_mean = (z_mean.unsqueeze(0) - z_b).pow(2).mean(dim=(1, 2))
        cos_pred = torch.nn.functional.cosine_similarity(
            z_pred.flatten(1), z_b.flatten(1), dim=-1)
        cos_persist = torch.nn.functional.cosine_similarity(
            z_a.flatten(1), z_b.flatten(1), dim=-1)
        for i in range(z_a.shape[0]):
            rows.append({
                "pair_id": batch["pair_id"][i],
                "plot_uid": batch["plot_uid"][i],
                "genotype_id": batch["genotype_id"][i],
                "split": batch["split"][i],
                "pair_type": batch["pair_type"][i],
                "span_days": int(batch["span_days"][i]),
                "delta_days": int(float(batch["delta_days"][i])),
                "harvest_year": int(batch["harvest_year"][i]),
                "date_a": batch["date_a"][i],
                "date_b": batch["date_b"][i],
                "mse_pred": float(mse_pred[i]),
                "mse_persist": float(mse_persist[i]),
                "mse_train_mean": float(mse_mean[i]),
                "cosine_pred": float(cos_pred[i]),
                "cosine_persist": float(cos_persist[i]),
            })
    return pd.DataFrame(rows)


def _summarise_e1(df: pd.DataFrame) -> dict:
    def fin(g: pd.DataFrame) -> dict:
        mp = float(g.mse_pred.mean())
        ma = float(g.mse_persist.mean())
        mm = float(g.mse_train_mean.mean())
        return {
            "n_samples": int(len(g)),
            "n_plots": int(g.plot_uid.nunique()),
            "mse_pred": mp,
            "mse_persist": ma,
            "mse_train_mean": mm,
            "relative_improvement_vs_persist": (ma - mp) / ma if ma > 0 else None,
            "relative_improvement_vs_train_mean": (mm - mp) / mm if mm > 0 else None,
            "cosine_pred": float(g.cosine_pred.mean()),
            "cosine_persist": float(g.cosine_persist.mean()),
        }

    out = fin(df)
    out["by_span"] = {
        str(k): fin(g) for k, g in df.groupby("pair_type", sort=True)
    }
    out["by_year"] = {
        str(k): fin(g) for k, g in df.groupby("harvest_year", sort=True)
    }
    out["by_delta"] = {
        str(k): fin(g) for k, g in df.groupby("delta_days", sort=True)
    }
    return out


def bootstrap_e1(wide: pd.DataFrame, conditions: list[str], n: int, seed: int) -> dict:
    """Plot-cluster bootstrap for each condition and for key pairwise contrasts."""
    plots = wide.plot_uid.to_numpy()
    y_persist = wide["mse_persist"].to_numpy(np.float64)
    y_mean = wide["mse_train_mean"].to_numpy(np.float64)
    tables = {"persist": y_persist, "train_mean": y_mean}
    for c in conditions:
        tables[c] = wide[f"mse::{c}"].to_numpy(np.float64)

    def improve(t, key):
        m = float(np.mean(t["persist"]))
        return (m - float(np.mean(t[key]))) / m

    def improve_mean(t, key):
        m = float(np.mean(t["train_mean"]))
        return (m - float(np.mean(t[key]))) / m

    out = {"unit": "plot_uid", "n_resamples": int(n), "seed": int(seed),
           "per_condition": {}, "contrasts": {}}
    for c in conditions:
        out["per_condition"][c] = {
            "rel_improvement_vs_persist": cluster_bootstrap(
                tables, plots, lambda t, k=c: improve(t, k), n, seed),
            "rel_improvement_vs_train_mean": cluster_bootstrap(
                tables, plots, lambda t, k=c: improve_mean(t, k), n, seed),
        }

    def contrast(t, a, b):
        return float(np.mean(t[b]) - np.mean(t[a]))

    pairs = [
        ("b2_no_time", "b3_delta"),
        ("b3_delta", "b4_delta_gdd_R"),
        ("b3_delta", "b4_delta_gdd_P"),
        ("b3_delta", "b5_delta_weather_P"),
        ("b3_delta", "b5_delta_weather_R"),
        ("b4_delta_gdd_R", "b4_delta_gdd_R_tbase5"),
        ("b4_delta_gdd_P", "b4_delta_gdd_R"),
        ("b5_delta_weather_P", "b5_delta_weather_R"),
    ]
    for a, b in pairs:
        if a in conditions and b in conditions:
            out["contrasts"][f"{b}_minus_{a}"] = {
                "mse_delta": cluster_bootstrap(
                    tables, plots, lambda t, a=a, b=b: contrast(t, a, b), n, seed),
            }

    # per-harvest-year contrasts: a within-year weather effect cannot be a
    # pure year-identity effect, so these are the honest test of question (7).
    out["by_year"] = {}
    for year in sorted(wide.harvest_year.unique()):
        m = (wide.harvest_year == year).to_numpy()
        sub = {k: v[m] for k, v in tables.items()}
        sub_plots = plots[m]
        block = {"n_samples": int(m.sum()),
                 "n_plots": int(len(np.unique(sub_plots))), "contrasts": {}}
        for a, b in pairs:
            if a in conditions and b in conditions:
                block["contrasts"][f"{b}_minus_{a}"] = cluster_bootstrap(
                    sub, sub_plots, lambda t, a=a, b=b: contrast(t, a, b), n, seed)
        out["by_year"][str(year)] = block
    return out


def run_e1(cfg: dict, device: torch.device) -> dict:
    t0 = time.time()
    index = pd.read_parquet(cfg["data"]["index"])
    idx_sha = index_sha256(index)
    if idx_sha != cfg["frozen"]["index_sha256"]:
        raise SystemExit("frozen index digest mismatch")
    cond = load_conditioning(cfg["data"]["conditioning_npz"], index)
    role_map, split_stats = build_role_map(
        index, int(cfg["split"]["seed"]), float(cfg["split"]["inner_train_fraction"])
    )
    traits = tuple(cfg["data"]["traits"])
    val = index[index.split == "validation"].copy()
    vl_rows = index.index[index.split == "validation"].to_numpy()
    val["_orig_row"] = vl_rows
    val = val.reset_index(drop=True)

    inner_train = split_rows(index, role_map, "inner_train")
    t_mean = mean_future_feature(
        inner_train, cond, device, int(cfg["eval"]["batch_size"]))
    print(f"[B1] train-mean from {len(inner_train)} inner-train pairs", flush=True)

    trained = [
        name for name, c in cfg["conditions"].items() if c.get("mode") == "trained"
    ]
    if cfg.get("_only"):
        trained = [n for n in trained if n in cfg["_only"]]
    frames = {}
    ckpt_meta = {}
    for name in trained:
        models, ckpts = load_condition_models(cfg, name, device)
        spec = condition_spec(cfg, name)
        df = e1_condition(
            models, val, cond, spec, traits, device,
            int(cfg["eval"]["batch_size"]), t_mean,
        )
        df["condition"] = name
        frames[name] = df
        ckpt_meta[name] = {
            "checkpoint": str(run_dir(cfg, name, 0) / "checkpoints" / "best.pt"),
            "checkpoint_sha256": [
                sha256_file_local(run_dir(cfg, name, int(s)) / "checkpoints" / "best.pt")
                for s in cfg["training"]["seeds"]
            ],
            "n_trainable_parameters": int(ckpts[0]["n_trainable_parameters"]),
            "n_condition_parameters": int(ckpts[0]["n_condition_parameters"]),
            "predictor_input_dim": int(ckpts[0]["predictor_input_dim"]),
            "conditioning_spec": ckpts[0]["conditioning_spec"],
            "protocol": cfg["conditions"][name].get("protocol"),
            "seeds": [int(s) for s in cfg["training"]["seeds"]],
        }
        print(f"[E1] {name}: mse_pred={df.mse_pred.mean():.5f} "
              f"persist={df.mse_persist.mean():.5f}", flush=True)

    long = pd.concat(frames.values(), ignore_index=True)
    long.to_parquet(cfg["artifacts"]["e1_predictions"], index=False)

    # wide frame for bootstrap (identical rows across conditions)
    base = frames[trained[0]][
        ["pair_id", "plot_uid", "pair_type", "harvest_year", "delta_days",
         "mse_persist", "mse_train_mean"]
    ].copy()
    for name, df in frames.items():
        if not np.array_equal(df.pair_id.to_numpy(), base.pair_id.to_numpy()):
            raise RuntimeError(f"validation rows are not aligned for {name}")
        base[f"mse::{name}"] = df.mse_pred.to_numpy()

    per_condition = {}
    for name, df in frames.items():
        per_condition[name] = _summarise_e1(df)
    # trivial baselines
    b0 = base.mse_persist
    b1 = base.mse_train_mean
    per_condition["b0_persistence"] = {
        "mse_pred": float(b0.mean()),
        "mse_persist": float(b0.mean()),
        "relative_improvement_vs_persist": 0.0,
        "n_samples": int(len(base)),
        "n_plots": int(base.plot_uid.nunique()),
        "cosine_pred": None,
        "note": "Z_hat_b = Z_a",
    }
    per_condition["b1_train_mean"] = {
        "mse_pred": float(b1.mean()),
        "mse_train_mean": float(b1.mean()),
        "relative_improvement_vs_persist": float(
            (b0.mean() - b1.mean()) / b0.mean()),
        "n_samples": int(len(base)),
        "n_plots": int(base.plot_uid.nunique()),
        "note": "Z_hat_b = mean z_b over the inner-train pairs of the same model split",
    }

    boot = bootstrap_e1(
        base, trained,
        int(cfg["eval"]["bootstrap_resamples"]),
        int(cfg["eval"]["bootstrap_seed"]),
    )
    boot["b0_persistence"] = {
        "mse": float(b0.mean()),
        "rel_improvement_vs_persist": 0.0,
    }
    boot["b1_train_mean"] = {
        "mse": float(b1.mean()),
        "rel_improvement_vs_persist": float((b0.mean() - b1.mean()) / b0.mean()),
    }
    with open(cfg["artifacts"]["e1_bootstrap"], "w") as fh:
        json.dump(boot, fh, indent=2)

    return {
        "stage": "S4A",
        "task": "E1 future-latent prediction on the official validation split",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": " ".join(sys.argv),
        "git_head": git("rev-parse", "HEAD"),
        "device": str(device),
        "n_val_pairs": int(len(val)),
        "n_val_plots": int(val.plot_uid.nunique()),
        "n_val_images": int(pd.unique(pd.concat(
            [val.image_stem_a, val.image_stem_b])).size),
        "val_years": sorted(int(y) for y in val.harvest_year.unique()),
        "split": split_stats,
        "index_sha256": idx_sha,
        "conditioning_sha256": sha256_file_local(cfg["data"]["conditioning_npz"]),
        "conditions": per_condition,
        "checkpoints": ckpt_meta,
        "bootstrap": {"path": cfg["artifacts"]["e1_bootstrap"],
                      "n_resamples": int(cfg["eval"]["bootstrap_resamples"]),
                      "seed": int(cfg["eval"]["bootstrap_seed"])},
        "elapsed_seconds": round(time.time() - t0, 1),
    }


# --------------------------------------------------------------------------
# E2
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


def collect_e2(cfg: dict, device: torch.device, out_dir: Path) -> dict:
    index = pd.read_parquet(cfg["data"]["index"])
    cond = load_conditioning(cfg["data"]["conditioning_npz"], index)
    traits = tuple(cfg["data"]["traits"])
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name in cfg["e2_conditions"]:
        models, _ = load_condition_models(cfg, name, device)
        spec = condition_spec(cfg, name)
        meta, pool_a, pool_b = collect_pooled(
            models, index, cond, spec, traits, device,
            int(cfg["eval"]["batch_size"]),
        )
        path = out_dir / f"{name}.npz"
        np.savez_compressed(
            path,
            pair_id=meta.pair_id.to_numpy().astype(str),
            pool_a=pool_a,
            pool_b=pool_b,
        )
        meta.to_parquet(out_dir / f"{name}_meta.parquet", index=False)
        manifest[name] = {
            "npz": str(path),
            "sha256": sha256_file_local(path),
            "n_rows": int(len(meta)),
            "n_train": int((meta.split == "train").sum()),
            "n_validation": int((meta.split == "validation").sum()),
        }
        print(f"[E2 collect] {name}: {len(meta)} rows", flush=True)
        del pool_a, pool_b, meta
    return manifest


def run_e2(cfg: dict, device: torch.device, pooled_dir: Path) -> dict:
    t0 = time.time()
    pcfg = cfg["probe"]
    index = pd.read_parquet(cfg["data"]["index"])
    role_map, split_stats = build_role_map(
        index, int(pcfg["plot_split_seed"]), float(pcfg["inner_train_fraction"])
    )
    role = pd.Series(role_map)
    conditions = list(cfg["e2_conditions"])
    env_set = set(cfg.get("e2_environment_conditions", []))

    loaded = {}
    base_meta = None
    for name in conditions:
        z = np.load(pooled_dir / f"{name}.npz", allow_pickle=True)
        meta = pd.read_parquet(pooled_dir / f"{name}_meta.parquet")
        if base_meta is None:
            base_meta = meta
        else:
            if not np.array_equal(meta.pair_id.to_numpy(), base_meta.pair_id.to_numpy()):
                raise RuntimeError(f"pooled rows are not aligned for {name}")
        loaded[name] = (meta, z["pool_a"], z["pool_b"])
        print(f"[E2] loaded {name}", flush=True)

    variants = {
        "primary_lag7": int(pcfg["lag_variants"]["primary"]),
        "strict_lag3": int(pcfg["lag_variants"]["strict_lag3"]),
        "same_day_lag0": int(pcfg["lag_variants"]["same_day"]),
    }
    scope_plan = {
        "primary_lag7": ["overall", "span7", "span14"],
        "strict_lag3": ["overall"],
        "same_day_lag0": ["overall"],
    }

    results = {"stage": "S4A", "task": "E2 CC/PH readout (S3.2 corrected protocol)",
               "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "command": " ".join(sys.argv), "git_head": git("rev-parse", "HEAD"),
               "device": str(device), "split": split_stats,
               "protocol": {
                   "architecture": "single linear layer (PooledReadout, 768 -> 1)",
                   "pooling": "mean over 256 patch tokens, identical for A and B",
                   "candidate_epochs": int(pcfg["candidate_epochs"]),
                   "patience": int(pcfg["patience"]),
                   "seeds": [int(s) for s in pcfg["seeds"]],
                   "selection": "minimum inner-dev MSE; refit on all official-train plots",
                   "official_validation_used_for_selection": False,
                   "bootstrap": {
                       "unit": "plot_uid",
                       "n_resamples": int(pcfg["bootstrap_resamples"]),
                       "seed": int(pcfg["bootstrap_seed"]),
                   },
                   "lag_variants": variants,
               },
               "rows": {}, "bootstrap": {}}

    # per-sample predictions for the primary overall scope
    pred_rows = []
    for variant, lag_max in variants.items():
        results["rows"][variant] = {}
        for trait in cfg["data"]["traits"]:
            results["rows"][variant][trait] = {}
            frames = scope_frames(base_meta, trait)
            for scope in scope_plan[variant]:
                frame = frames[scope]
                frame = frame[frame[f"{trait}_b_lag"].abs() <= lag_max]
                tr = frame[frame.split == "train"].copy()
                va = frame[frame.split == "validation"].copy().reset_index(drop=True)
                tr["role"] = tr.plot_uid.map(role)
                tr_in = tr[tr.role == "inner_train"]
                tr_dev = tr[tr.role == "inner_dev"]
                if len(tr_in) < 100 or len(tr_dev) < 30 or len(va) < 30:
                    results["rows"][variant][trait][scope] = {
                        "skipped": True, "n_inner_train": int(len(tr_in)),
                        "n_inner_dev": int(len(tr_dev)), "n_val": int(len(va))}
                    continue
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
                    meta, pa, pb = loaded[name]
                    idx = df["_pool_index"].to_numpy()
                    return (pa if which == "a" else pb)[idx]

                # conditions: A_current + each predicted condition
                cond_plan = [("A_current", "a", None)] + [
                    (f"B_{n}", "b", n) for n in conditions
                ]
                block = {"counts": counts, "conditions": {}}
                for label, which, name in cond_plan:
                    src = name if name is not None else conditions[0]
                    if label == "A_current":
                        x_in = xmat(tr_in, "a", src)
                        x_dev = xmat(tr_dev, "a", src)
                        x_tr = xmat(tr, "a", src)
                        x_va = xmat(va, "a", src)
                    else:
                        x_in = xmat(tr_in, "b", name)
                        x_dev = xmat(tr_dev, "b", name)
                        x_tr = xmat(tr, "b", name)
                        x_va = xmat(va, "b", name)
                    runs = []
                    epochs_sel = []
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
                        metric = _metrics(y_va, pred)
                        info["metrics"] = metric
                        info["selected_epoch"] = selected
                        final_runs.append(info)
                        if variant == "primary_lag7" and scope == "overall":
                            for i in range(len(va)):
                                pred_rows.append({
                                    "variant": variant, "trait": trait,
                                    "scope": scope, "condition": label,
                                    "seed": int(seed),
                                    "pair_id": va.pair_id.iloc[i],
                                    "plot_uid": va.plot_uid.iloc[i],
                                    "pair_type": va.pair_type.iloc[i],
                                    "delta_days": int(va.delta_days.iloc[i]),
                                    "lag_days": float(va[f"{trait}_b_lag"].iloc[i]),
                                    "y_true": float(y_va[i]),
                                    "y_pred": float(pred[i]),
                                })
                    agg = {}
                    for k in ("mae", "rmse", "r2"):
                        vals = np.array([r["metrics"][k] for r in final_runs], float)
                        agg[k] = {"mean": float(vals.mean()),
                                  "std": float(vals.std()),
                                  "values": vals.tolist()}
                    agg["selected_epoch_median"] = selected
                    agg["selected_epochs_per_seed"] = epochs_sel
                    agg["n_val"] = int(len(va))
                    agg["n_val_plots"] = int(va.plot_uid.nunique())
                    agg["n_train"] = int(len(tr))
                    agg["role"] = (
                        "A_current_feature" if label == "A_current"
                        else "C_environment_predicted" if name in env_set
                        else "B_predicted"
                    )
                    block["conditions"][label] = agg
                a = block["conditions"]["A_current"]
                for label in block["conditions"]:
                    if label == "A_current":
                        continue
                    b = block["conditions"][label]
                    block.setdefault("B_minus_A", {})[label] = {
                        k: {"mean_delta": b[k]["mean"] - a[k]["mean"],
                            "a_mean": a[k]["mean"], "b_mean": b[k]["mean"]}
                        for k in ("mae", "rmse", "r2")
                    }
                results["rows"][variant][trait][scope] = block
                print(f"[E2] {variant}|{trait}|{scope} done "
                      f"(n_val={len(va)})", flush=True)

    Path(cfg["artifacts"]["e2_predictions"]).parent.mkdir(parents=True, exist_ok=True)
    preds = pd.DataFrame(pred_rows)
    preds.to_parquet(cfg["artifacts"]["e2_predictions"], index=False)

    # plot-cluster bootstrap of B-A on the primary overall scope
    boot = {"unit": "plot_uid", "n_resamples": int(pcfg["bootstrap_resamples"]),
            "seed": int(pcfg["bootstrap_seed"]), "scopes": {}}
    for trait in cfg["data"]["traits"]:
        sub = preds[(preds.variant == "primary_lag7") & (preds.scope == "overall")
                    & (preds.trait == trait)]
        if sub.empty:
            continue
        agg = sub.groupby(["condition", "pair_id", "plot_uid"], as_index=False).agg(
            y_true=("y_true", "first"), y_pred=("y_pred", "mean"))
        a = agg[agg.condition == "A_current"].set_index("pair_id")
        y = a.y_true.to_numpy(np.float64)
        pa = a.y_pred.to_numpy(np.float64)
        plots = a.plot_uid.to_numpy()
        block = {"n_samples": int(len(y)), "n_plots": int(len(np.unique(plots)))}
        for label in sorted(agg.condition.unique()):
            if label == "A_current":
                continue
            b = agg[agg.condition == label].set_index("pair_id").loc[a.index]
            pb = b.y_pred.to_numpy(np.float64)
            tables = {"y": y, "pa": pa, "pb": pb}

            def metrics_delta(t):
                ea = t["pa"] - t["y"]
                eb = t["pb"] - t["y"]
                ss = float(np.sum((t["y"] - t["y"].mean()) ** 2))
                return {
                    "delta_mae": float(np.mean(np.abs(eb)) - np.mean(np.abs(ea))),
                    "delta_rmse": float(np.sqrt(np.mean(eb ** 2)) - np.sqrt(np.mean(ea ** 2))),
                    "delta_r2": (-np.sum(eb ** 2) / ss) - (-np.sum(ea ** 2) / ss)
                    if ss > 0 else float("nan"),
                }

            block[label] = {
                k: cluster_bootstrap(
                    tables, plots, lambda t, k=k: metrics_delta(t)[k],
                    int(pcfg["bootstrap_resamples"]), int(pcfg["bootstrap_seed"]))
                for k in ("delta_mae", "delta_rmse", "delta_r2")
            }
        boot["scopes"][trait] = block
    results["bootstrap"] = boot
    results["elapsed_seconds"] = round(time.time() - t0, 1)
    with open(cfg["artifacts"]["e2_metrics"], "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"wrote {cfg['artifacts']['e2_metrics']}", flush=True)
    return results


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s4_pilot.yaml")
    ap.add_argument("--mode", choices=["e1", "e2", "e2-collect", "e2-probe", "all"],
                    default="all")
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict E1/E2 to these trained conditions")
    ap.add_argument("--seeds", nargs="*", type=int, default=None,
                    help="override the seed set (defaults to the config set)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    if args.only:
        cfg["_only"] = list(args.only)
        cfg["e2_conditions"] = [c for c in cfg["e2_conditions"] if c in args.only]
    if args.seeds:
        cfg["training"]["seeds"] = [int(s) for s in args.seeds]
        cfg["_seeds_overridden"] = True
    device = torch.device(
        cfg["eval"]["device"] if torch.cuda.is_available() else "cpu")
    pooled_dir = Path(cfg["artifacts"]["e2_pooled_dir"])

    report = {"stage": "S4A", "manifest_id": cfg["manifest_id"],
              "config": args.config,
              "config_sha256": sha256_file_local(args.config),
              "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "command": " ".join(sys.argv)}
    if args.mode in ("e1", "all"):
        report["E1"] = run_e1(cfg, device)
    if args.mode in ("e2-collect", "all"):
        report["E2_collect"] = collect_e2(cfg, device, pooled_dir)
    if args.mode in ("e2", "e2-probe", "all"):
        if not args.mode == "all":
            report["E2"] = run_e2(cfg, device, pooled_dir)
        else:
            report["E2"] = run_e2(cfg, device, pooled_dir)

    out = Path(args.out or cfg["artifacts"]["results_json"])
    out.parent.mkdir(parents=True, exist_ok=True)
    # Merge with any previously written sections so running E1 and E2 as
    # separate commands yields one combined results file.
    merged = {}
    if out.exists():
        try:
            merged = json.loads(out.read_text())
        except json.JSONDecodeError:
            merged = {}
    merged.update(report)
    merged["sections_present"] = sorted(
        k for k in merged if k.startswith("E") or k in ("E1", "E2", "E2_collect")
    )
    with open(out, "w") as fh:
        json.dump(merged, fh, indent=2)
    print(f"wrote {out} (sections: {merged['sections_present']})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
