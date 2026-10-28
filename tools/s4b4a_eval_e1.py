#!/usr/bin/env python3
"""S4B.4A E1: future-latent prediction on each fold's held-out year.

For every fold it loads that fold's held-out pairs (never used in training or
epoch selection) ONCE, runs every condition's per-fold checkpoints on the same
batches, and reports:

* MSE, cosine similarity, relative improvement over persistence (z_a) and over
  the fold-train-mean future feature;
* paired MSE contrasts B3-B2-C, GDD-P-B3, Weather-P-B3 on the identical sample
  set, with plot-clustered bootstrap 95% CIs;
* span7 / span14 strata and the per-seed spread (training randomness).

The held-out features are read exactly once and reused for every condition
(single-pass, all-conditions evaluation) to keep I/O bounded.

Usage
-----
    CUDA_VISIBLE_DEVICES=1 python3 tools/s4b4a_eval_e1.py \
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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from agro_jepa.datasets.s4_dataset import S4PairDataset, ConditioningSpec  # noqa: E402
from agro_jepa.models.s4_predictor import (  # noqa: E402
    S4ModelConfig,
    build_s4_predictor,
    conditioning_spec,
)
from agro_jepa.models.s4_conditioning import VectorNormStats  # noqa: E402
from agro_jepa.models.time_embedding import DeltaNormStats  # noqa: E402
from agro_jepa.models.fip_dino_wm import move_attention_masks  # noqa: E402
from s3_1_diagnostics import cluster_bootstrap, git, sha256_file  # noqa: E402


# Superset spec: the loader materialises delta + every GDD/weather array any
# arm may need, so one pass serves all conditions.  Each model receives only the
# tensors its own ``parts`` list declares.
FEATURE_SPEC = ConditioningSpec(use_delta=True, delta_mode="sample",
                                gdd_mode="past", gdd_base=0.0,
                                weather_mode="past")
MINIMAL_SPEC = ConditioningSpec(use_delta=False, gdd_mode="none", gdd_base=0.0,
                                weather_mode="none")


def slice_conditioning(cond_full: dict, orig_rows: np.ndarray) -> dict:
    n_ref = cond_full["delta_days"].shape[0]
    return {k: (v[orig_rows] if isinstance(v, np.ndarray) and v.ndim >= 1
                and v.shape[0] == n_ref else v) for k, v in cond_full.items()}


def load_model(ckpt_path: Path, device: torch.device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ck.get("format") != "agro-jepa-s4b4a-state-dict-v1":
        raise ValueError(f"{ckpt_path} is not an S4B.4A checkpoint")
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
    delta_norm = DeltaNormStats(mean=dn["mean"], std=dn["std"], min=dn["min"],
                                max=dn["max"], n_samples=dn["n_samples"]) if dn else None
    gn = ck.get("gdd_norm")
    gdd_norm = VectorNormStats(mean=tuple(gn["mean"]), std=tuple(gn["std"]),
                               n_samples=gn.get("n_samples", 0)) if gn else None
    wn = ck.get("weather_norm")
    weather_norm = VectorNormStats(mean=tuple(wn["mean"]), std=tuple(wn["std"]),
                                   n_samples=wn.get("n_samples", 0)) if wn else None
    model = build_s4_predictor(S4ModelConfig.from_dict(ck["model_config"]), spec,
                               delta_norm=delta_norm, gdd_norm=gdd_norm,
                               weather_norm=weather_norm)
    model.load_state_dict(ck["model"])
    model.to(device)
    move_attention_masks(model.predictor, device)
    model.eval()
    return model, ck


def forward_batch(model, batch, device):
    from agro_jepa.models.s4_predictor import has_fixed_delta
    parts = getattr(model, "parts", [])
    kw = {}
    if "delta" in parts and not has_fixed_delta(model):
        kw["delta_days"] = batch["delta_days"].to(device, non_blocking=True)
    if "gdd" in parts:
        kw["gdd"] = batch["gdd"].to(device, non_blocking=True)
    if "weather" in parts:
        kw["weather"] = batch["weather"].to(device, non_blocking=True)
        kw["weather_mask"] = batch["weather_mask"].to(device, non_blocking=True)
    return model(batch["z_a"].to(device, non_blocking=True), **kw)


@torch.no_grad()
def train_mean_feature(train_frame, device, batch_size=512, cache=None):
    if cache is not None and Path(cache).is_file():
        return torch.load(cache, map_location="cpu")
    ds = S4PairDataset(train_frame, {}, MINIMAL_SPEC, traits=(), load_features=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=8,
                        pin_memory=True)
    total, n = None, 0
    for batch in loader:
        z_b = batch["z_b"].to(device).float()
        s = z_b.sum(dim=0)
        total = s if total is None else total + s
        n += z_b.shape[0]
    z_mean = (total / max(n, 1)).cpu()
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        torch.save(z_mean, cache)
    return z_mean


@torch.no_grad()
def eval_all_conditions(models_by_cond, seeds, ds, device, batch_size, z_mean):
    """ONE pass over the held-out loader; evaluate every condition and seed."""
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4,
                        pin_memory=True)
    rows = []
    for batch in loader:
        z_a = batch["z_a"].to(device).float()
        z_b = batch["z_b"].to(device).float()
        mse_persist = (z_a - z_b).pow(2).mean(dim=(1, 2))
        mse_mean = (z_mean.unsqueeze(0) - z_b).pow(2).mean(dim=(1, 2))
        cos_persist = torch.nn.functional.cosine_similarity(
            z_a.flatten(1), z_b.flatten(1), dim=-1)
        n = z_a.shape[0]
        per = {c: {"preds": {}, "mse": {}, "cos": {}} for c in models_by_cond}
        for cond, seed_models in models_by_cond.items():
            stack = []
            for seed in seeds:
                m = seed_models[seed]
                out = forward_batch(m, batch, device).float()
                stack.append(out)
                per[cond]["preds"][seed] = out
            avg = torch.stack(stack, 0).mean(0)
            per[cond]["preds"]["avg"] = avg
        for i in range(n):
            base = {
                "pair_id": batch["pair_id"][i], "plot_uid": batch["plot_uid"][i],
                "pair_type": batch["pair_type"][i],
                "delta_days": int(float(batch["delta_days"][i])),
                "harvest_year": int(batch["harvest_year"][i]),
                "mse_persist": float(mse_persist[i]),
                "mse_train_mean": float(mse_mean[i]),
                "cosine_persist": float(cos_persist[i]),
            }
            for cond in models_by_cond:
                for key in list(seeds) + ["avg"]:
                    p = per[cond]["preds"][key][i]
                    base[f"mse::{cond}::{key}"] = float(
                        (p - z_b[i]).pow(2).mean())
                    base[f"cos::{cond}::{key}"] = float(
                        torch.nn.functional.cosine_similarity(
                            p.flatten(), z_b[i].flatten(), dim=0))
            rows.append(base)
    return pd.DataFrame(rows)


def summarise(g: pd.DataFrame, mse_col: str, cos_col: str) -> dict:
    mp = float(g[mse_col].mean())
    ma = float(g.mse_persist.mean())
    mm = float(g.mse_train_mean.mean())
    return {
        "n_samples": int(len(g)), "n_plots": int(g.plot_uid.nunique()),
        "mse_pred": mp, "mse_persist": ma, "mse_train_mean": mm,
        "relative_improvement_vs_persist": (ma - mp) / ma if ma > 0 else None,
        "relative_improvement_vs_train_mean": (mm - mp) / mm if mm > 0 else None,
        "cosine_pred": float(g[cos_col].mean()),
        "cosine_persist": float(g.cosine_persist.mean()),
    }


def bootstrap_block(wide, conditions, contrasts, n, seed, suffix) -> dict:
    plots = wide.plot_uid.to_numpy()
    tables = {"persist": wide.mse_persist.to_numpy(np.float64),
              "train_mean": wide.mse_train_mean.to_numpy(np.float64),
              **{c: wide[f"mse::{c}::{suffix}"].to_numpy(np.float64)
                 for c in conditions}}
    out = {"unit": "plot_uid", "n_resamples": int(n), "seed": int(seed),
           "per_condition": {}, "contrasts": {}, "stratified": {}}
    for c in conditions:
        m_p = float(np.mean(tables["persist"]))
        m_m = float(np.mean(tables["train_mean"]))
        out["per_condition"][c] = {
            "rel_improvement_vs_persist": cluster_bootstrap(
                tables, plots, lambda t, k=c: (m_p - float(np.mean(t[k]))) / m_p,
                n, seed),
            "rel_improvement_vs_train_mean": cluster_bootstrap(
                tables, plots, lambda t, k=c: (m_m - float(np.mean(t[k]))) / m_m,
                n, seed)}
    for lo, hi, label in contrasts:
        out["contrasts"][label] = {"mse_delta": cluster_bootstrap(
            tables, plots,
            lambda t, a=lo, b=hi: float(np.mean(t[b]) - np.mean(t[a])), n, seed)}
    out["stratified"]["by_span"] = {}
    for value in sorted(wide["pair_type"].unique()):
        mask = (wide["pair_type"] == value).to_numpy()
        sub = {k: v[mask] for k, v in tables.items()}
        sub_plots = plots[mask]
        block = {"n_samples": int(mask.sum()),
                 "n_plots": int(len(np.unique(sub_plots))), "contrasts": {}}
        for lo, hi, label in contrasts:
            block["contrasts"][label] = cluster_bootstrap(
                sub, sub_plots,
                lambda t, a=lo, b=hi: float(np.mean(t[b]) - np.mean(t[a])),
                n, seed)
        out["stratified"]["by_span"][str(value)] = block
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s4b4a_year_heldout.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--folds", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    t0 = time.time()
    device = torch.device(args.device or (cfg["eval"]["device"]
                          if torch.cuda.is_available() else "cpu"))
    seeds = [int(s) for s in (args.seeds.split(",") if args.seeds
                              else cfg["training"]["seeds"])]
    fold_years = ([int(s) for s in args.folds.split(",")] if args.folds
                  else [int(y) for y in cfg["folds"]["heldout_year_per_fold"]])
    bs = int(args.batch_size or cfg["eval"]["batch_size"])
    n_res = int(cfg["eval"]["bootstrap_resamples"])
    bseed = int(cfg["eval"]["bootstrap_seed"])
    conditions = list(cfg["e1_conditions"])
    contrasts = [tuple(c) for c in cfg["e1_contrasts"]]
    fold_dir = ROOT / cfg["artifacts"]["fold_dir"]
    runs_dir = ROOT / cfg["artifacts"]["runs_dir"]
    cond_full = dict(np.load(ROOT / cfg["data"]["conditioning_npz"],
                             allow_pickle=True))

    report = {
        "stage": "S4B.4A",
        "task": "E1 future-latent prediction on each fold's held-out year",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": " ".join(sys.argv),
        "git_head": git("rev-parse", "HEAD"),
        "config": args.config, "config_sha256": sha256_file(ROOT / args.config),
        "device": str(device), "seeds": seeds,
        "heldout_labels_read": False, "test_used": False,
        "folds": {},
    }

    for fold_year in fold_years:
        held = pd.read_parquet(fold_dir / f"fold{fold_year}_heldout.parquet")
        train = pd.read_parquet(fold_dir / f"fold{fold_year}_fold_train.parquet")
        cond_arrays = slice_conditioning(
            cond_full, held["_orig_row"].to_numpy(np.int64))
        ds = S4PairDataset(held, cond_arrays, FEATURE_SPEC, traits=(),
                           load_features=True)
        print(f"[fold {fold_year}] heldout {len(held)} pairs", flush=True)

        z_mean = train_mean_feature(
            train, device, bs,
            cache=runs_dir / f"fold{fold_year}" / "e1_train_mean.pt").to(device)

        models_by_cond = {}
        ckpt_info = {}
        for cond in conditions:
            models_by_cond[cond] = {}
            ckpt_info[cond] = {}
            for seed in seeds:
                ck_p = (Path(cfg["training"]["output_root"]) / f"fold{fold_year}"
                        / cond / f"seed{seed}" / "checkpoints" / "best.pt")
                if not ck_p.is_file():
                    raise SystemExit(f"missing checkpoint {ck_p}")
                m, _ = load_model(ck_p, device)
                models_by_cond[cond][seed] = m
                ckpt_info[cond][str(seed)] = {"path": str(ck_p),
                                              "sha256": sha256_file(ck_p)}
        long = eval_all_conditions(models_by_cond, seeds, ds, device, bs, z_mean)
        torch.cuda.empty_cache()
        (runs_dir / f"fold{fold_year}").mkdir(parents=True, exist_ok=True)
        long.to_parquet(runs_dir / f"fold{fold_year}" / "e1_predictions.parquet",
                        index=False)

        summary = {"seed_averaged": {}, "per_seed": {}}
        for cond in conditions:
            summary["seed_averaged"][cond] = summarise(
                long, f"mse::{cond}::avg", f"cos::{cond}::avg")
            summary["seed_averaged"][cond]["by_span"] = {
                str(k): summarise(g, f"mse::{cond}::avg", f"cos::{cond}::avg")
                for k, g in long.groupby("pair_type", sort=True)}
            summary["per_seed"][cond] = {
                str(s): summarise(long, f"mse::{cond}::{s}", f"cos::{cond}::{s}")
                for s in seeds}

        boot = bootstrap_block(long, conditions, contrasts, n_res, bseed, "avg")
        per_seed_contrasts = {}
        for lo, hi, label in contrasts:
            per_seed_contrasts[label] = {}
            for seed in seeds:
                d = (long[f"mse::{hi}::{seed}"].to_numpy(float)
                     - long[f"mse::{lo}::{seed}"].to_numpy(float))
                per_seed_contrasts[label][str(seed)] = {
                    "mse_delta_mean": float(d.mean()), "n_samples": int(len(d))}
            vals = [v["mse_delta_mean"] for v in per_seed_contrasts[label].values()]
            per_seed_contrasts[label]["across_seeds"] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "all_same_sign": bool(all(v < 0 for v in vals)
                                      or all(v > 0 for v in vals)),
                "n_seeds": len(vals)}

        report["folds"][str(fold_year)] = {
            "n_pairs": int(len(held)), "n_plots": int(held.plot_uid.nunique()),
            "n_span7": int((held.pair_type == "span7").sum()),
            "n_span14": int((held.pair_type == "span14").sum()),
            "delta_days": {"min": int(held.delta_days.min()),
                           "max": int(held.delta_days.max()),
                           "mean": float(held.delta_days.mean())},
            "checkpoints": ckpt_info,
            "conditions": summary, "bootstrap": boot,
            "per_seed_contrasts": per_seed_contrasts,
        }
        for cond in conditions:
            del models_by_cond[cond]
        print(f"[fold {fold_year}] done", flush=True)

    report["elapsed_seconds"] = round(time.time() - t0, 1)
    out_json = Path(args.out_json or cfg["artifacts"]["e1_json"])
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(report, fh, indent=2)
    write_md(Path(args.out_md or cfg["artifacts"]["e1_md"]), report)
    print(f"wrote {out_json}", flush=True)
    return 0


def write_md(path: Path, r: dict) -> None:
    L = ["# S4B.4A E1 -- Future-Latent Prediction by Held-Out Year", ""]
    L.append(f"- Generated: {r['generated_at']}")
    L.append(f"- Git HEAD: `{r['git_head']}`")
    L.append(f"- Seeds: {r['seeds']}; held-out labels read: "
             f"{r['heldout_labels_read']}; test used: {r['test_used']}")
    L.append("")
    L.append("| fold | pairs | plots | span7 | span14 | delta mean [min,max] |")
    L.append("|---|---:|---:|---:|---:|---|")
    for y, e in r["folds"].items():
        d = e["delta_days"]
        L.append(f"| {y} | {e['n_pairs']:,} | {e['n_plots']} | {e['n_span7']:,} | "
                 f"{e['n_span14']:,} | {d['mean']:.2f} [{d['min']},{d['max']}] |")
    L.append("")
    for y, e in r["folds"].items():
        L.append(f"## Fold {y} (held-out year)")
        L.append("")
        L.append("| arm | MSE | cos | vs persist | vs train-mean |")
        L.append("|---|---:|---:|---:|---:|")
        for cond, s in e["conditions"]["seed_averaged"].items():
            L.append(f"| {cond} | {s['mse_pred']:.6f} | {s['cosine_pred']:.4f} | "
                     f"{100*s['relative_improvement_vs_persist']:.2f}% | "
                     f"{100*s['relative_improvement_vs_train_mean']:.2f}% |")
        L.append("")
        L.append("| contrast | dMSE | 95% CI | per-seed mean ± sd |")
        L.append("|---|---:|---|---|")
        for label, b in e["bootstrap"]["contrasts"].items():
            v = b["mse_delta"]
            ps = e["per_seed_contrasts"][label]["across_seeds"]
            L.append(f"| {label} | {v['mean']:+.6f} | "
                     f"[{v['ci95_low']:+.6f}, {v['ci95_high']:+.6f}] | "
                     f"{ps['mean']:+.6f} ± {ps['std']:.6f} |")
        L.append("")
        L.append("Span strata (paired dMSE [95% CI]):")
        L.append("")
        labels = list(e["bootstrap"]["contrasts"].keys())
        L.append("| span | n | " + " | ".join(labels) + " |")
        L.append("|---|---:|" + "---:|" * len(labels))
        for sp, blk in e["bootstrap"]["stratified"]["by_span"].items():
            cells = []
            for lab in labels:
                v = blk["contrasts"][lab]
                cells.append(f"{v['mean']:+.6f} "
                             f"[{v['ci95_low']:+.6f},{v['ci95_high']:+.6f}]")
            L.append(f"| {sp} | {blk['n_samples']:,} | " + " | ".join(cells) + " |")
        L.append("")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
