#!/usr/bin/env python3
"""S4A.1 E1: future-latent prediction on the official validation split.

Same evaluation set and same metric definitions as S4A (``eval_s4.run_e1``):
future-latent MSE, cosine similarity, improvement over persistence and over the
inner-train mean future feature, plot-cluster bootstrap.  Two additions are
required by the S4A.1 brief:

1. **Per-seed metrics** for every trained seed, so training-seed variability is
   reported separately from the plot bootstrap interval.  The bootstrap CI is
   *not* a training-randomness CI and is never presented as one.
2. **Paired seed-level contrast** B3 - B2-C, alongside the plot-bootstrap CI of
   the same contrast on the seed-averaged predictions.

Usage
-----
    CUDA_VISIBLE_DEVICES=1 python3 tools/s4a1_e1_eval.py \
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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from agro_jepa.datasets.s3_pairs import index_sha256  # noqa: E402
from agro_jepa.datasets.s4_dataset import (  # noqa: E402
    build_role_map,
    load_conditioning,
    split_rows,
)
from agro_jepa.eval.eval_s4 import condition_spec, load_condition_models, run_dir  # noqa: E402
from agro_jepa.eval.s4_common import (  # noqa: E402
    forward_batch,
    make_dataset,
    mean_future_feature,
)
from s3_1_diagnostics import cluster_bootstrap, git, sha256_file  # noqa: E402

CONTROL = "b2_no_time_constant_delta"
MATCHED = "b3_delta"
NO_TIME = "b2_no_time"
CONDITIONS = [NO_TIME, CONTROL, MATCHED]
CONTRASTS = [
    (CONTROL, MATCHED, "b3_delta_minus_b2_no_time_constant_delta"),
    (NO_TIME, MATCHED, "b3_delta_minus_b2_no_time"),
    (NO_TIME, CONTROL, "b2_no_time_constant_delta_minus_b2_no_time"),
]


@torch.no_grad()
def eval_one_seed(models, frame, cond, spec, traits, device, batch_size, z_mean):
    ds = make_dataset(frame, cond, spec, traits)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4,
                        pin_memory=True)
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
                "pair_type": batch["pair_type"][i],
                "delta_days": int(float(batch["delta_days"][i])),
                "harvest_year": int(batch["harvest_year"][i]),
                "mse_pred": float(mse_pred[i]),
                "mse_persist": float(mse_persist[i]),
                "mse_train_mean": float(mse_mean[i]),
                "cosine_pred": float(cos_pred[i]),
                "cosine_persist": float(cos_persist[i]),
            })
    return pd.DataFrame(rows)


def summarise(g: pd.DataFrame) -> dict:
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


def bootstrap_block(wide: pd.DataFrame, n: int, seed: int) -> dict:
    plots = wide.plot_uid.to_numpy()
    tables = {
        "persist": wide.mse_persist.to_numpy(np.float64),
        "train_mean": wide.mse_train_mean.to_numpy(np.float64),
        **{c: wide[f"mse::{c}"].to_numpy(np.float64) for c in CONDITIONS},
    }
    out = {"unit": "plot_uid", "n_resamples": int(n), "seed": int(seed),
           "per_condition": {}, "contrasts": {}}
    for c in CONDITIONS:
        m_p = float(np.mean(tables["persist"]))
        m_m = float(np.mean(tables["train_mean"]))
        out["per_condition"][c] = {
            "rel_improvement_vs_persist": cluster_bootstrap(
                tables, plots,
                lambda t, k=c: (m_p - float(np.mean(t[k]))) / m_p, n, seed),
            "rel_improvement_vs_train_mean": cluster_bootstrap(
                tables, plots,
                lambda t, k=c: (m_m - float(np.mean(t[k]))) / m_m, n, seed),
        }
    for lo_name, hi_name, label in CONTRASTS:
        out["contrasts"][label] = {"mse_delta": cluster_bootstrap(
            tables, plots,
            lambda t, a=lo_name, b=hi_name: float(np.mean(t[b]) - np.mean(t[a])),
            n, seed)}

    # stratified contrasts (question 4: does the conclusion depend on span/year?)
    out["stratified"] = {}
    for key, col in (("by_span", "pair_type"), ("by_year", "harvest_year")):
        out["stratified"][key] = {}
        for value in sorted(wide[col].unique()):
            m = (wide[col] == value).to_numpy()
            sub = {k: v[m] for k, v in tables.items()}
            sub_plots = plots[m]
            block = {"n_samples": int(m.sum()),
                     "n_plots": int(len(np.unique(sub_plots))), "contrasts": {}}
            for lo_name, hi_name, label in CONTRASTS:
                block["contrasts"][label] = cluster_bootstrap(
                    sub, sub_plots,
                    lambda t, a=lo_name, b=hi_name:
                        float(np.mean(t[b]) - np.mean(t[a])), n, seed)
            out["stratified"][key][str(value)] = block
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s4a1_matched_control.yaml")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    t0 = time.time()
    device = torch.device(cfg["eval"]["device"] if torch.cuda.is_available() else "cpu")
    bs = int(cfg["eval"]["batch_size"])
    seeds = [int(s) for s in cfg["training"]["seeds"]]

    index = pd.read_parquet(cfg["data"]["index"])
    idx_sha = index_sha256(index)
    if idx_sha != cfg["frozen"]["index_sha256"]:
        raise SystemExit("frozen index digest mismatch")
    cond = load_conditioning(cfg["data"]["conditioning_npz"], index)
    role_map, split_stats = build_role_map(
        index, int(cfg["split"]["seed"]), float(cfg["split"]["inner_train_fraction"]))
    traits = tuple(cfg["data"]["traits"])

    val = index[index.split == "validation"].copy()
    val["_orig_row"] = index.index[index.split == "validation"].to_numpy()
    val = val.reset_index(drop=True)

    inner_train = split_rows(index, role_map, "inner_train")
    z_mean = mean_future_feature(inner_train, cond, device, bs).to(device)
    print(f"[B1] inner-train mean future feature from {len(inner_train)} pairs",
          flush=True)

    frames: list[pd.DataFrame] = []
    meta: dict = {}
    for name in CONDITIONS:
        models, ckpts = load_condition_models(cfg, name, device)
        spec = condition_spec(cfg, name)
        for k, seed in enumerate(seeds):
            df = eval_one_seed([models[k]], val, cond, spec, traits, device, bs, z_mean)
            df["condition"] = name
            df["seed"] = seed
            frames.append(df)
            print(f"[E1] {name} seed{seed}: mse={df.mse_pred.mean():.5f}", flush=True)
        df = eval_one_seed(models, val, cond, spec, traits, device, bs, z_mean)
        df["condition"] = name
        df["seed"] = -1
        frames.append(df)
        meta[name] = {
            "checkpoints": [
                {"seed": s,
                 "path": str(run_dir(cfg, name, s) / "checkpoints" / "best.pt"),
                 "sha256": sha256_file(
                     run_dir(cfg, name, s) / "checkpoints" / "best.pt")}
                for s in seeds
            ],
            "n_trainable_parameters": int(ckpts[0]["n_trainable_parameters"]),
            "n_condition_parameters": int(ckpts[0]["n_condition_parameters"]),
            "predictor_input_dim": int(ckpts[0]["predictor_input_dim"]),
            "conditioning_spec": ckpts[0]["conditioning_spec"],
            "fixed_delta": ckpts[0].get("fixed_delta"),
        }

    long = pd.concat(frames, ignore_index=True)
    out_dir = Path(cfg["artifacts"]["e1_predictions"]).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    long.to_parquet(cfg["artifacts"]["e1_predictions"], index=False)

    # per-seed and seed-averaged summaries
    summary = {"per_seed": {}, "seed_averaged": {}}
    for name in CONDITIONS:
        summary["per_seed"][name] = {}
        for seed in seeds:
            sub = long[(long.condition == name) & (long.seed == seed)]
            summary["per_seed"][name][str(seed)] = summarise(sub)
        avg = long[(long.condition == name) & (long.seed == -1)]
        summary["seed_averaged"][name] = summarise(avg)
        summary["seed_averaged"][name]["by_span"] = {
            str(k): summarise(g) for k, g in avg.groupby("pair_type", sort=True)}
        summary["seed_averaged"][name]["by_year"] = {
            str(k): summarise(g) for k, g in avg.groupby("harvest_year", sort=True)}

    # wide frame on seed-averaged predictions (plot bootstrap)
    base = long[(long.condition == NO_TIME) & (long.seed == -1)][
        ["pair_id", "plot_uid", "pair_type", "harvest_year", "delta_days",
         "mse_persist", "mse_train_mean"]].reset_index(drop=True)
    for name in CONDITIONS:
        sub = long[(long.condition == name) & (long.seed == -1)].reset_index(drop=True)
        if not np.array_equal(sub.pair_id.to_numpy(), base.pair_id.to_numpy()):
            raise RuntimeError(f"validation rows are not aligned for {name}")
        base[f"mse::{name}"] = sub.mse_pred.to_numpy()
    boot = bootstrap_block(
        base, int(cfg["eval"]["bootstrap_resamples"]),
        int(cfg["eval"]["bootstrap_seed"]))

    # per-seed paired contrast: training randomness, not plot sampling
    per_seed_contrasts = {}
    for lo_name, hi_name, label in CONTRASTS:
        per_seed_contrasts[label] = {}
        for seed in seeds:
            a = long[(long.condition == lo_name) & (long.seed == seed)].reset_index(drop=True)
            b = long[(long.condition == hi_name) & (long.seed == seed)].reset_index(drop=True)
            if not np.array_equal(a.pair_id.to_numpy(), b.pair_id.to_numpy()):
                raise RuntimeError("per-seed rows are not aligned")
            d = b.mse_pred.to_numpy() - a.mse_pred.to_numpy()
            per_seed_contrasts[label][str(seed)] = {
                "mse_delta_mean": float(d.mean()),
                "mse_delta_relative_to_low_arm": float(d.mean() / a.mse_pred.mean()),
                "n_samples": int(len(d)),
            }
        vals = [v["mse_delta_mean"] for v in per_seed_contrasts[label].values()]
        per_seed_contrasts[label]["across_seeds"] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "n_seeds": len(vals),
            "all_same_sign": bool(all(v < 0 for v in vals) or all(v > 0 for v in vals)),
            "note": ("standard deviation across training seeds; not a plot "
                     "bootstrap interval"),
        }

    b2c_over_b3 = (
        summary["seed_averaged"][CONTROL]["mse_pred"]
        / summary["seed_averaged"][MATCHED]["mse_pred"] - 1.0)

    report = {
        "stage": "S4A.1",
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
        "conditioning_sha256": sha256_file(cfg["data"]["conditioning_npz"]),
        "seeds": seeds,
        "seed_rule": ("seed -1 = prediction averaged over the seed set, then MSE; "
                      "each other seed = that checkpoint alone"),
        "parameters": meta,
        "conditions": summary,
        "bootstrap": {
            "path": cfg["artifacts"]["e1_bootstrap"],
            "unit": "plot_uid",
            "n_resamples": int(cfg["eval"]["bootstrap_resamples"]),
            "seed": int(cfg["eval"]["bootstrap_seed"]),
            "results": boot,
        },
        "per_seed_contrasts": per_seed_contrasts,
        "control_vs_matched_relative_gap": float(b2c_over_b3),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with open(cfg["artifacts"]["e1_bootstrap"], "w") as fh:
        json.dump(boot, fh, indent=2)
    out = Path(args.out or cfg["artifacts"]["results_json"])
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"wrote {out} and {cfg['artifacts']['e1_bootstrap']}", flush=True)
    for name in CONDITIONS:
        s = summary["seed_averaged"][name]
        print(f"  {name:32s} MSE={s['mse_pred']:.5f} cos={s['cosine_pred']:.4f} "
              f"vs persist={100*s['relative_improvement_vs_persist']:.2f}%", flush=True)
    for label, block in per_seed_contrasts.items():
        a = block["across_seeds"]
        print(f"  {label}: per-seed dMSE {a['mean']:+.5f} ± {a['std']:.5f} "
              f"(all same sign={a['all_same_sign']})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
