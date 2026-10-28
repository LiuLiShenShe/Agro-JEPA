#!/usr/bin/env python3
"""S4B.3A E1: future-latent prediction on the four official test splits.

Runs only after ``reports/s4b3_test_protocol.{json,md}`` is frozen.  It rebuilds
each of the nine S4B.2 ``final.pt`` predictors exactly as S4B.2 did, predicts
``z_b`` from ``z_a`` on each official test pair, and compares five arms:

* **B0** persistence      -- ``z_a`` against ``z_b`` (no model);
* **B1** train-mean future -- mean ``z_b`` over the full S4B.2 official-train
  index (cached ``e1_train_mean.pt``), against ``z_b`` (no model);
* **B2** ``b2_no_time``  -- no time input;
* **B2-C** ``b2_no_time_constant_delta`` -- parameter matched, fixed 9.0 d;
* **B3** ``b3_delta``    -- ``z_a`` + real per-pair ``delta_days``.

The model reconstruction, preprocessing, delta normalisation, metric definitions,
seed rule and plot-clustered bootstrap are copied from
``tools/s4b2_validation_e1.py`` unchanged.  Only the data source changes: the
frozen S3 validation split becomes the four official test splits.

E1 reads **no test phenotype label**: the four test pair indexes carry none, and
this tool only touches feature vectors, dates and ``delta_days``.

Outputs
-------
* ``external_data/s4b3_runs/e1_predictions/<split>_e1_predictions.parquet``
* ``reports/s4b3_test_e1.json`` and ``reports/s4b3_test_e1.md``

Usage
-----
    CUDA_VISIBLE_DEVICES=1 python3 tools/s4b3_test_e1.py \
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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "agro_jepa" / "eval"))

from agro_jepa.datasets.s4_dataset import S4PairDataset  # noqa: E402
from agro_jepa.models.s4_predictor import conditioning_spec  # noqa: E402
from s3_1_diagnostics import cluster_bootstrap, git, sha256_file  # noqa: E402
from s4b2_validation_e1 import (  # noqa: E402
    CONDITIONS,
    CONTRASTS,
    NO_TIME,
    bootstrap_block,
    forward_batch,
    load_s4b2_model,
    summarise,
)

TEST_SPEC = conditioning_spec({"use_delta": True, "delta_mode": "sample",
                              "gdd_mode": "none", "weather_mode": "none"})
LABEL_COLUMNS = [
    f"{t}_{end}{suf}"
    for t in ("cc", "ph", "whc", "sen")
    for end in ("a", "b")
    for suf in ("", "_lag", "_date", "_mask")
] + ["ph_negative_a", "ph_negative_b"]


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def add_cache_columns(index: pd.DataFrame, feature_index: pd.DataFrame) -> pd.DataFrame:
    """Attach ``cache_file_a/b`` from the frozen S4B.1 feature index."""
    stem2cache = dict(zip(
        feature_index.image_filename_stem.astype(str).str.upper(),
        feature_index.cache_file.astype(str),
    ))
    out = index.copy()
    out["cache_file_a"] = out.image_stem_a.astype(str).str.upper().map(stem2cache)
    out["cache_file_b"] = out.image_stem_b.astype(str).str.upper().map(stem2cache)
    if out.cache_file_a.isna().any() or out.cache_file_b.isna().any():
        missing = int(out.cache_file_a.isna().sum() + out.cache_file_b.isna().sum())
        raise SystemExit(f"{missing} test endpoints have no cached feature")
    missing_files = [p for p in pd.unique(pd.concat([out.cache_file_a, out.cache_file_b]))
                     if not (ROOT / p).is_file()]
    if missing_files:
        raise SystemExit(f"{len(missing_files)} cached feature files are missing")
    return out.reset_index(drop=True)


@torch.no_grad()
def eval_one(models, ds, device, batch_size, z_mean) -> pd.DataFrame:
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


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def write_md(path: Path, report: dict) -> None:
    L = []
    L.append("# S4B.3 E1 -- Future-Latent Prediction on the Official Test Splits")
    L.append("")
    L.append(f"- Generated: {report['generated_at']}")
    L.append(f"- Git HEAD: `{report['git_head']}`")
    L.append(f"- Protocol: `reports/s4b3_test_protocol.json` "
             f"(all_pass = {report['protocol_all_pass']})")
    L.append(f"- Seeds: {report['seeds']}; seed rule: {report['seed_rule']}")
    L.append(f"- Bootstrap: plot-clustered, {report['bootstrap_resamples']} "
             f"resamples, seed {report['bootstrap_seed']}")
    L.append("- E1 reads **no** test phenotype label (the test indexes carry none).")
    L.append("")
    L.append("## Test splits")
    L.append("")
    L.append("| split | pairs | plots | years | span7 | span14 |")
    L.append("|---|---:|---:|---|---:|---:|")
    for split, s in report["splits"].items():
        L.append(f"| {split} | {s['n_pairs']:,} | {s['n_plots']} | "
                 f"{s['years']} | {s['n_span7']} | {s['n_span14']} |")
    L.append("")
    L.append("## Seed-averaged MSE / cosine / relative improvement")
    L.append("")
    L.append("| split | arm | MSE | cos | vs B0 | vs B1 |")
    L.append("|---|---|---:|---:|---:|---:|")
    for split, s in report["splits"].items():
        sa = s["conditions"]["seed_averaged"]
        b0 = sa[NO_TIME]
        L.append(f"| {split} | B0 persistence | {b0['mse_persist']:.6f} | "
                 f"{b0['cosine_persist']:.4f} | 0 | -- |")
        L.append(f"| | B1 train-mean | {b0['mse_train_mean']:.6f} | -- | -- | 0 |")
        for c in CONDITIONS:
            x = sa[c]
            L.append(f"| | {c} | {x['mse_pred']:.6f} | {x['cosine_pred']:.4f} | "
                     f"{100*x['relative_improvement_vs_persist']:.2f}% | "
                     f"{100*x['relative_improvement_vs_train_mean']:.2f}% |")
    L.append("")
    L.append("## Bootstrap 95% CI (plot-clustered) -- relative improvement")
    L.append("")
    L.append("| split | arm | vs B0 [95% CI] | vs B1 [95% CI] |")
    L.append("|---|---|---|---|")
    for split, s in report["splits"].items():
        for c in CONDITIONS:
            b = s["bootstrap"]["per_condition"][c]
            rp, rm = b["rel_improvement_vs_persist"], b["rel_improvement_vs_train_mean"]
            L.append(f"| {split} | {c} | {100*rp['mean']:.2f}% "
                     f"[{100*rp['ci95_low']:.2f}, {100*rp['ci95_high']:.2f}] | "
                     f"{100*rm['mean']:.2f}% "
                     f"[{100*rm['ci95_low']:.2f}, {100*rm['ci95_high']:.2f}] |")
    L.append("")
    L.append("## Paired B3 - B2-C and B3 - B2 (seed-averaged, plot bootstrap)")
    L.append("")
    L.append("| split | contrast | dMSE | 95% CI |")
    L.append("|---|---|---:|---|")
    for split, s in report["splits"].items():
        for label, b in s["bootstrap"]["contrasts"].items():
            v = b["mse_delta"]
            L.append(f"| {split} | {label} | {v['mean']:+.6f} | "
                     f"[{v['ci95_low']:+.6f}, {v['ci95_high']:+.6f}] |")
    L.append("")
    L.append("## Per-seed spread (training randomness, not a plot interval)")
    L.append("")
    L.append("| split | contrast | seed0 | seed1 | seed2 | mean +/- std | "
             "same sign |")
    L.append("|---|---|---:|---:|---:|---|---|")
    for split, s in report["splits"].items():
        for label, block in s["per_seed_contrasts"].items():
            vals = [block[str(seed)]["mse_delta_mean"] for seed in report["seeds"]]
            a = block["across_seeds"]
            L.append(f"| {split} | {label} | "
                     + " | ".join(f"{v:+.6f}" for v in vals)
                     + f" | {a['mean']:+.6f} +/- {a['std']:.6f} | "
                     f"{a['all_same_sign']} |")
    L.append("")
    L.append("## Span stratification (seed-averaged MSE)")
    L.append("")
    L.append("| split | arm | span7 MSE | span14 MSE | span7 vs B0 | "
             "span14 vs B0 |")
    L.append("|---|---|---:|---:|---:|---:|")
    for split, s in report["splits"].items():
        sa = s["conditions"]["seed_averaged"]
        for c in CONDITIONS:
            b7 = sa[c]["by_span"].get("span7", {})
            b14 = sa[c]["by_span"].get("span14", {})
            L.append(f"| {split} | {c} | {b7.get('mse_pred', float('nan')):.6f} | "
                     f"{b14.get('mse_pred', float('nan')):.6f} | "
                     f"{100*b7.get('relative_improvement_vs_persist', 0):.2f}% | "
                     f"{100*b14.get('relative_improvement_vs_persist', 0):.2f}% |")
    L.append("")
    L.append("## Notes")
    L.append("")
    L.append("- B0 = persistence (z_a), B1 = mean z_b over the full S4B.2 "
             "official-train index.")
    L.append("- Seed-averaged = prediction averaged over seeds {0,1,2} before MSE.")
    L.append("- Plot bootstrap captures test-sample sampling only; per-seed spread "
             "captures training randomness and is reported separately.")
    L.append("- B2-C receives one fixed 9.0 d interval; B3 receives each pair's "
             "real delta_days.")
    path.write_text("\n".join(L) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s4b3_official_test.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--split", default=None,
                    help="optional single split; default = all four")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    protocol = json.load(open(ROOT / cfg["artifacts"]["protocol_json"]))
    if not protocol["verification"]["all_pass"]:
        raise SystemExit("protocol checks did not pass; refusing to evaluate test")

    t0 = time.time()
    device = torch.device(args.device or cfg["device"]["cuda"]
                          if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in cfg["e1"]["seeds"]]
    bs = int(args.batch_size or cfg["e1"]["batch_size"])
    n_resamples = int(cfg["e1"]["bootstrap_resamples"])
    boot_seed = int(cfg["e1"]["bootstrap_seed"])

    feature_index = pd.read_parquet(ROOT / cfg["data"]["feature_index"])
    if sha256_file(ROOT / cfg["data"]["feature_index"]) != \
            cfg["data"]["feature_index_sha256"]:
        raise SystemExit("feature index digest mismatch")

    # B1 train-mean feature (identical to S4B.2; reuse the frozen cache)
    mean_cache = ROOT / "external_data/s4b2_runs/e1_train_mean.pt"
    if not mean_cache.is_file():
        raise SystemExit(f"missing frozen B1 train-mean cache {mean_cache}")
    z_mean = torch.load(mean_cache, map_location="cpu").float().to(device)
    print(f"[B1] loaded frozen train-mean feature {mean_cache}", flush=True)

    splits = ([args.split] if args.split else list(cfg["test_splits"].keys()))
    report = {
        "stage": "S4B.3",
        "task": "E1 future-latent prediction on the official test splits",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": " ".join(sys.argv),
        "git_head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "config": args.config,
        "config_sha256": sha256_file(ROOT / args.config),
        "protocol_json": cfg["artifacts"]["protocol_json"],
        "protocol_sha256": sha256_file(ROOT / cfg["artifacts"]["protocol_json"]),
        "protocol_all_pass": protocol["verification"]["all_pass"],
        "device": str(device),
        "seeds": seeds,
        "seed_rule": ("seed -1 = prediction averaged over the seed set, then MSE; "
                      "each other seed = that checkpoint alone"),
        "bootstrap_resamples": n_resamples,
        "bootstrap_seed": boot_seed,
        "b1_mean_source": "full S4B.2 official-train index (cached)",
        "test_phenotype_labels_read": False,
        "checkpoints": protocol["checkpoints"],
        "splits": {},
    }

    pred_dir = ROOT / cfg["artifacts"]["e1_predictions_dir"]
    pred_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        spec = cfg["test_splits"][split]
        idx_path = ROOT / spec["index"]
        if sha256_file(idx_path) != spec["index_sha256"]:
            raise SystemExit(f"{split}: test index digest mismatch")
        index = pd.read_parquet(idx_path)
        if any(c in index.columns for c in LABEL_COLUMNS):
            raise SystemExit(f"{split}: test index unexpectedly carries labels")
        index = add_cache_columns(index, feature_index)
        ds = S4PairDataset(index, {}, TEST_SPEC, traits=(), load_features=True)
        print(f"[{split}] {len(index)} pairs, {index.plot_uid.nunique()} plots, "
              f"years {sorted(int(y) for y in index.harvest_year.unique())}",
              flush=True)

        frames: list[pd.DataFrame] = []
        for name in CONDITIONS:
            models = []
            for seed in seeds:
                ckpt = ROOT / cfg["frozen"]["checkpoints"][name][f"seed{seed}"]
                m, _ = load_s4b2_model(str(ckpt), device)
                models.append(m)
            for k, seed in enumerate(seeds):
                df = eval_one([models[k]], ds, device, bs, z_mean)
                df["condition"] = name
                df["seed"] = seed
                frames.append(df)
                print(f"[{split}] {name} seed{seed}: "
                      f"mse={df.mse_pred.mean():.6f} "
                      f"cos={df.cosine_pred.mean():.4f}", flush=True)
            df = eval_one(models, ds, device, bs, z_mean)
            df["condition"] = name
            df["seed"] = -1
            frames.append(df)
            del models
            torch.cuda.empty_cache()

        long = pd.concat(frames, ignore_index=True)
        long.to_parquet(pred_dir / f"{split}_e1_predictions.parquet", index=False)

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
                raise RuntimeError(f"{split}: rows not aligned for {name}")
            base[f"mse::{name}"] = sub.mse_pred.to_numpy()
        boot = bootstrap_block(base, n_resamples, boot_seed)

        per_seed_contrasts = {}
        for lo_name, hi_name, label in CONTRASTS:
            per_seed_contrasts[label] = {}
            for seed in seeds:
                a = long[(long.condition == lo_name) & (long.seed == seed)].reset_index(drop=True)
                b = long[(long.condition == hi_name) & (long.seed == seed)].reset_index(drop=True)
                if not np.array_equal(a.pair_id.to_numpy(), b.pair_id.to_numpy()):
                    raise RuntimeError(f"{split}: per-seed rows not aligned")
                d = b.mse_pred.to_numpy() - a.mse_pred.to_numpy()
                per_seed_contrasts[label][str(seed)] = {
                    "mse_delta_mean": float(d.mean()),
                    "mse_delta_relative_to_low_arm": float(d.mean() / a.mse_pred.mean()),
                    "n_samples": int(len(d)),
                }
            vals = [v["mse_delta_mean"] for v in per_seed_contrasts[label].values()
                    if isinstance(v, dict) and "mse_delta_mean" in v]
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

        report["splits"][split] = {
            "index": spec["index"],
            "index_sha256": spec["index_sha256"],
            "n_pairs": int(len(index)),
            "n_plots": int(index.plot_uid.nunique()),
            "n_images": int(pd.unique(pd.concat(
                [index.image_stem_a, index.image_stem_b])).size),
            "years": sorted(int(y) for y in index.harvest_year.unique()),
            "n_span7": int((index.pair_type == "span7").sum()),
            "n_span14": int((index.pair_type == "span14").sum()),
            "delta_days": {"min": int(index.delta_days.min()),
                           "max": int(index.delta_days.max()),
                           "mean": float(index.delta_days.mean())},
            "conditions": summary,
            "bootstrap": boot,
            "per_seed_contrasts": per_seed_contrasts,
            "predictions": str((pred_dir / f"{split}_e1_predictions.parquet")),
        }
        sa = summary["seed_averaged"]
        print(f"[{split}] B3 vs B2-C dMSE = "
              f"{boot['contrasts']['b3_delta_minus_b2_no_time_constant_delta']['mse_delta']['mean']:+.6f}",
              flush=True)

    report["elapsed_seconds"] = round(time.time() - t0, 1)
    out_json = Path(args.out_json or cfg["artifacts"]["e1_json"])
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(report, fh, indent=2)
    out_md = Path(args.out_md or cfg["artifacts"]["e1_md"])
    write_md(out_md, report)
    print(f"wrote {out_json} and {out_md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
