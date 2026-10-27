#!/usr/bin/env python3
"""S4B.2 E1: future-latent prediction on the official validation split.

Runs only **after** all nine full-train checkpoints have been frozen.  It reads
the frozen S3 validation split (no test row, no inner split, no selection) and,
for each of the three conditions and each of seeds {0,1,2}, rebuilds the exact
S4B.2 predictor from its ``final.pt`` and predicts ``z_b`` from ``z_a``.

Baselines and arms
------------------
* **B0** persistence      -- ``z_a`` against ``z_b`` (no model);
* **B1** train-mean future -- mean ``z_b`` over the full S4B.2 official-train
  index, against ``z_b`` (no model);
* **B2** ``b2_no_time``           -- z_a only, no time channel;
* **B2-C** ``b2_no_time_constant_delta`` -- parameter-matched, fixed 9.0 d;
* **B3** ``b3_delta``             -- z_a + real per-pair ``delta_days``.

Metrics: MSE (mean over token*dim), cosine similarity, relative improvement
over B0 and over B1, stratified by span (span7/span14) and by harvest year,
the paired ``B3 - B2-C`` MSE contrast, and a **plot-clustered** bootstrap 95 %
CI over ``plot_uid`` (all pairs of a resampled plot kept together).

This is a diagnostic only: no checkpoint, epoch, seed or hyper-parameter is
chosen here, and nothing about the official test split is read or touched.

Usage
-----
    CUDA_VISIBLE_DEVICES=1 python3 tools/s4b2_validation_e1.py \
        --config agro_jepa/config/s4b2_full_train.yaml
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
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from agro_jepa.datasets.s3_pairs import index_sha256  # noqa: E402
from agro_jepa.datasets.s4_dataset import ConditioningSpec, S4PairDataset  # noqa: E402
from agro_jepa.models.fip_dino_wm import move_attention_masks  # noqa: E402
from agro_jepa.models.s4_predictor import (  # noqa: E402
    S4ModelConfig,
    build_s4_predictor,
    conditioning_spec,
)
from agro_jepa.models.time_embedding import DeltaNormStats  # noqa: E402
from s3_1_diagnostics import cluster_bootstrap, git, sha256_file  # noqa: E402

NO_TIME = "b2_no_time"
CONTROL = "b2_no_time_constant_delta"
MATCHED = "b3_delta"
CONDITIONS = [NO_TIME, CONTROL, MATCHED]
CONTRASTS = [
    (CONTROL, MATCHED, "b3_delta_minus_b2_no_time_constant_delta"),
    (NO_TIME, MATCHED, "b3_delta_minus_b2_no_time"),
    (NO_TIME, CONTROL, "b2_no_time_constant_delta_minus_b2_no_time"),
]


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_s4b2_model(ckpt_path: str, device: torch.device):
    """Rebuild an S4B.2 predictor from its ``final.pt``; returns (model, ckpt)."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ck.get("format") != "agro-jepa-s4b2-state-dict-v1":
        raise ValueError(f"{ckpt_path} is not an S4B.2 state-dict checkpoint")
    cs = ck["conditioning_spec"]
    spec = conditioning_spec({
        "use_delta": cs["use_delta"],
        "delta_mode": cs.get("delta_mode", "sample"),
        "fixed_delta_days": cs.get("fixed_delta_days"),
        "gdd_mode": cs["gdd_mode"],
        "gdd_base": cs.get("gdd_base", 0.0),
        "weather_mode": cs["weather_mode"],
    })
    dn = ck.get("delta_norm")
    delta_norm = DeltaNormStats(
        mean=dn["mean"], std=dn["std"], min=dn["min"], max=dn["max"],
        n_samples=dn["n_samples"],
    ) if dn else None
    model = build_s4_predictor(
        S4ModelConfig.from_dict(ck["model_config"]), spec,
        delta_norm=delta_norm if spec.use_delta else None,
    )
    model.load_state_dict(ck["model"])
    model.to(device)
    move_attention_masks(model.predictor, device)
    model.eval()
    return model, ck


def forward_batch(model, batch: dict, device: torch.device) -> torch.Tensor:
    from agro_jepa.models.s4_predictor import has_fixed_delta

    parts = getattr(model, "parts", [])
    kwargs = {}
    if "delta" in parts and not has_fixed_delta(model):
        kwargs["delta_days"] = batch["delta_days"].to(device, non_blocking=True)
    return model(batch["z_a"].to(device, non_blocking=True), **kwargs)


def run_dir(cfg: dict, cond: str, seed: int) -> Path:
    return Path(cfg["training"]["output_root"]) / cond / f"seed{seed}"


# --------------------------------------------------------------------------
# passes
# --------------------------------------------------------------------------
@torch.no_grad()
def eval_one(models, ds, device, batch_size, z_mean) -> pd.DataFrame:
    """One pass; predictions averaged over ``models`` (one or many seeds)."""
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


@torch.no_grad()
def train_mean_feature(train_index: pd.DataFrame, device, batch_size=512) -> torch.Tensor:
    """B1: mean z_b over the full S4B.2 official-train index."""
    spec = conditioning_spec({"use_delta": True, "gdd_mode": "none",
                              "weather_mode": "none"})
    ds = S4PairDataset(train_index, {}, spec, traits=(), load_features=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=8,
                        pin_memory=True)
    total, n = None, 0
    for batch in loader:
        z_b = batch["z_b"].to(device, non_blocking=True).float()
        s = z_b.sum(dim=0)
        total = s if total is None else total + s
        n += z_b.shape[0]
    return (total / max(n, 1)).cpu()


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
    out = {"unit": "plot_uid (all pairs of a resampled plot kept together)",
           "n_resamples": int(n), "seed": int(seed),
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
            "mse_pred": cluster_bootstrap(
                tables, plots, lambda t, k=c: float(np.mean(t[k])), n, seed),
        }
    for lo_name, hi_name, label in CONTRASTS:
        out["contrasts"][label] = {"mse_delta": cluster_bootstrap(
            tables, plots,
            lambda t, a=lo_name, b=hi_name: float(np.mean(t[b]) - np.mean(t[a])),
            n, seed)}
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


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def write_md(path: Path, report: dict, long: pd.DataFrame) -> None:
    L = []
    L.append("# S4B.2 E1 Validation Diagnostics")
    L.append("")
    L.append(f"- Generated: {report['generated_at']}")
    L.append(f"- Git HEAD: `{report['git_head']}`")
    L.append(f"- Validation pairs: **{report['n_val_pairs']:,}** over "
             f"**{report['n_val_plots']:,}** plots ({report['n_val_images']:,} images)")
    L.append(f"- Years: {report['val_years']}")
    L.append(f"- S3 index sha256: `{report['index_sha256']}`")
    L.append(f"- Bootstrap: plot-clustered, {report['bootstrap']['n_resamples']} "
             f"resamples, seed {report['bootstrap']['seed']}")
    L.append("")
    L.append("## Frozen checkpoints")
    L.append("")
    L.append("| condition | seed | final.pt sha256 (16) | params | input dim |")
    L.append("|---|---:|---|---:|---:|")
    for c in CONDITIONS:
        for s in report["seeds"]:
            e = report["parameters"][c]["checkpoints"][s]
            L.append(f"| {c} | {s} | `{e['sha256'][:16]}` | "
                     f"{report['parameters'][c]['n_trainable_parameters']:,} | "
                     f"{report['parameters'][c]['predictor_input_dim']} |")
    L.append("")
    L.append("## Seed-averaged metrics on validation")
    L.append("")
    L.append("| arm | MSE | cos | rel.vs B0(persist) | rel.vs B1(train-mean) |")
    L.append("|---|---:|---:|---:|---:|")
    base0 = report["conditions"]["seed_averaged"][NO_TIME]
    for name, label in (("B0_persist", "B0 persistence"),
                        ("B1_train_mean", "B1 train-mean")):
        if name == "B0_persist":
            L.append(f"| {label} | {base0['mse_persist']:.6f} | "
                     f"{base0['cosine_persist']:.4f} | 0 | -- |")
        else:
            L.append(f"| {label} | {base0['mse_train_mean']:.6f} | -- | -- | 0 |")
    for c in CONDITIONS:
        s = report["conditions"]["seed_averaged"][c]
        L.append(f"| {c} | {s['mse_pred']:.6f} | {s['cosine_pred']:.4f} | "
                 f"{100*s['relative_improvement_vs_persist']:.2f}% | "
                 f"{100*s['relative_improvement_vs_train_mean']:.2f}% |")
    L.append("")
    L.append("## Bootstrap 95% CI (plot-clustered)")
    L.append("")
    L.append("| arm | rel.vs persist [95% CI] | rel.vs train-mean [95% CI] |")
    L.append("|---|---|---|")
    for c in CONDITIONS:
        b = report["bootstrap"]["results"]["per_condition"][c]
        rp, rm = b["rel_improvement_vs_persist"], b["rel_improvement_vs_train_mean"]
        L.append(f"| {c} | {100*rp['mean']:.2f}% "
                 f"[{100*rp['ci95_low']:.2f}, {100*rp['ci95_high']:.2f}] | "
                 f"{100*rm['mean']:.2f}% "
                 f"[{100*rm['ci95_low']:.2f}, {100*rm['ci95_high']:.2f}] |")
    L.append("")
    L.append("## Paired B3 - B2-C contrast")
    L.append("")
    b = report["bootstrap"]["results"]["contrasts"][
        "b3_delta_minus_b2_no_time_constant_delta"]["mse_delta"]
    L.append(f"- seed-averaged: dMSE = {b['mean']:+.6f} "
             f"[{b['ci95_low']:+.6f}, {b['ci95_high']:+.6f}] "
             f"(negative = B3 better)")
    L.append("- per training seed (training randomness, *not* a plot bootstrap):")
    for s in report["seeds"]:
        e = report["per_seed_contrasts"][
            "b3_delta_minus_b2_no_time_constant_delta"][str(s)]
        L.append(f"  - seed{s}: dMSE = {e['mse_delta_mean']:+.6f}")
    a = report["per_seed_contrasts"][
        "b3_delta_minus_b2_no_time_constant_delta"]["across_seeds"]
    L.append(f"  - across seeds: {a['mean']:+.6f} +/- {a['std']:.6f} "
             f"(all same sign = {a['all_same_sign']})")
    L.append("")
    L.append("## Stratified paired contrasts (dMSE B3 - B2-C)")
    L.append("")
    L.append("| stratum | value | n pairs | dMSE | 95% CI |")
    L.append("|---|---|---:|---:|---|")
    for key in ("by_span", "by_year"):
        for value, block in report["bootstrap"]["results"]["stratified"][key].items():
            b = block["contrasts"]["b3_delta_minus_b2_no_time_constant_delta"]
            L.append(f"| {key} | {value} | {block['n_samples']} | "
                     f"{b['mean']:+.6f} | [{b['ci95_low']:+.6f}, "
                     f"{b['ci95_high']:+.6f}] |")
    L.append("")
    L.append("## Notes")
    L.append("")
    L.append("- B0 = persistence (z_a), B1 = mean z_b over the full S4B.2 "
             "official-train index.")
    L.append("- Seed-averaged = prediction averaged over seeds {0,1,2} before MSE.")
    L.append("- The plot bootstrap captures validation-sample sampling only; it is "
             "not a training-seed interval (per-seed spread is reported separately).")
    L.append("- No test row, no test label and no test metric is used anywhere.")
    path.write_text("\n".join(L) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s4b2_full_train.yaml")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--bootstrap-resamples", type=int, default=2000)
    ap.add_argument("--bootstrap-seed", type=int, default=20261024)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-predictions", default=None)
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    t0 = time.time()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in cfg["training"]["seeds"]]
    bs = int(args.batch_size)

    # ---- frozen inputs ----------------------------------------------------
    s3 = pd.read_parquet(cfg["validation_e1"]["s3_index"])
    idx_sha = index_sha256(s3)
    expected_sha = cfg["validation_e1"]["split_s3_index_sha256"]
    if idx_sha != expected_sha:
        raise SystemExit(f"S3 index digest mismatch: {idx_sha} != {expected_sha}")

    val = s3[s3.split == cfg["validation_e1"]["split"]].copy().reset_index(drop=True)
    missing = 0
    for col in ("cache_file_a", "cache_file_b"):
        for p in val[col].unique():
            if not Path(p).is_file():
                missing += 1
    if missing:
        raise SystemExit(f"{missing} validation feature caches are missing")
    val_ds = S4PairDataset(
        val, {},
        conditioning_spec({"use_delta": True, "gdd_mode": "none",
                           "weather_mode": "none"}),
        traits=(), load_features=True,
    )
    print(f"[E1] validation: {len(val)} pairs, {val.plot_uid.nunique()} plots, "
          f"years {sorted(int(y) for y in val.harvest_year.unique())}", flush=True)

    # ---- B1 train-mean feature -------------------------------------------
    train_index = pd.read_parquet(cfg["data"]["train_index"])
    mean_cache = Path(cfg["training"]["output_root"]) / "e1_train_mean.pt"
    if mean_cache.is_file():
        z_mean = torch.load(mean_cache, map_location="cpu")
        print(f"[B1] loaded cached train-mean feature {mean_cache}", flush=True)
    else:
        z_mean = train_mean_feature(train_index, device, bs)
        mean_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(z_mean, mean_cache)
        print(f"[B1] mean future feature over {len(train_index)} official-train pairs "
              f"(cached -> {mean_cache})", flush=True)
    z_mean = z_mean.float().to(device)

    # ---- per-condition evaluation ----------------------------------------
    frames: list[pd.DataFrame] = []
    meta: dict = {}
    for name in CONDITIONS:
        models, ckpts = [], []
        for seed in seeds:
            ckpt = run_dir(cfg, name, seed) / "checkpoints" / "final.pt"
            if not ckpt.is_file():
                raise SystemExit(f"missing checkpoint {ckpt}")
            m, ck = load_s4b2_model(str(ckpt), device)
            models.append(m)
            ckpts.append((seed, ckpt, ck))
        for k, seed in enumerate(seeds):
            df = eval_one([models[k]], val_ds, device, bs, z_mean)
            df["condition"] = name
            df["seed"] = seed
            frames.append(df)
            print(f"[E1] {name} seed{seed}: mse={df.mse_pred.mean():.6f} "
                  f"cos={df.cosine_pred.mean():.4f}", flush=True)
        df = eval_one(models, val_ds, device, bs, z_mean)
        df["condition"] = name
        df["seed"] = -1
        frames.append(df)
        meta[name] = {
            "checkpoints": {
                s: {"path": str(p), "sha256": sha256_file(p)}
                for s, p, _ in ckpts
            },
            "n_trainable_parameters": int(ckpts[0][2]["n_trainable_parameters"]),
            "n_condition_parameters": int(ckpts[0][2]["n_condition_parameters"]),
            "predictor_input_dim": int(ckpts[0][2]["predictor_input_dim"]),
            "conditioning_spec": ckpts[0][2]["conditioning_spec"],
            "fixed_delta": ckpts[0][2].get("fixed_delta"),
        }
        del models
        torch.cuda.empty_cache()

    long = pd.concat(frames, ignore_index=True)
    out_pred = Path(args.out_predictions) if args.out_predictions else (
        Path(cfg["training"]["output_root"]) / "e1_predictions.parquet")
    out_pred.parent.mkdir(parents=True, exist_ok=True)
    long.to_parquet(out_pred, index=False)

    # ---- summaries --------------------------------------------------------
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

    base = long[(long.condition == NO_TIME) & (long.seed == -1)][
        ["pair_id", "plot_uid", "pair_type", "harvest_year", "delta_days",
         "mse_persist", "mse_train_mean"]].reset_index(drop=True)
    for name in CONDITIONS:
        sub = long[(long.condition == name) & (long.seed == -1)].reset_index(drop=True)
        if not np.array_equal(sub.pair_id.to_numpy(), base.pair_id.to_numpy()):
            raise RuntimeError(f"validation rows are not aligned for {name}")
        base[f"mse::{name}"] = sub.mse_pred.to_numpy()
    boot = bootstrap_block(base, int(args.bootstrap_resamples),
                           int(args.bootstrap_seed))

    # per-seed paired contrasts (training randomness, not plot sampling)
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

    b2c_over_b3 = (summary["seed_averaged"][CONTROL]["mse_pred"]
                   / summary["seed_averaged"][MATCHED]["mse_pred"] - 1.0)

    report = {
        "stage": "S4B.2",
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
        "index_sha256": idx_sha,
        "index_role": {
            "validation_pairs": int(len(val)),
            "b1_mean_source": "full S4B.2 official-train index",
            "selection": "none (checkpoints frozen before this pass)",
            "test_used": False,
        },
        "seeds": seeds,
        "seed_rule": ("seed -1 = prediction averaged over the seed set, then MSE; "
                      "each other seed = that checkpoint alone"),
        "parameters": meta,
        "conditions": summary,
        "bootstrap": {
            "path": str(out_pred),
            "unit": "plot_uid",
            "n_resamples": int(args.bootstrap_resamples),
            "seed": int(args.bootstrap_seed),
            "results": boot,
        },
        "per_seed_contrasts": per_seed_contrasts,
        "control_vs_matched_relative_gap": float(b2c_over_b3),
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    out_json = Path(args.out_json or cfg["reports"]["validation_e1_json"])
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(report, fh, indent=2)
    out_md = Path(cfg["reports"]["validation_e1_md"])
    write_md(out_md, report, long)

    print(f"wrote {out_json} and {out_md}", flush=True)
    for name in CONDITIONS:
        s = summary["seed_averaged"][name]
        print(f"  {name:32s} MSE={s['mse_pred']:.6f} cos={s['cosine_pred']:.4f} "
              f"vs persist={100*s['relative_improvement_vs_persist']:.2f}%", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
