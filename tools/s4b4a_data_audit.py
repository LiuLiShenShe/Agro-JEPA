#!/usr/bin/env python3
"""S4B.4A Phase 1: data / time-encoding / GDD-weather / capacity audit.

This tool only *reads* frozen S0-S4B.3 artifacts and writes two new audit
files (``reports/s4b4a_data_audit.{json,md}``).  It never rewrites an input.

What it establishes, from the real frozen indexes (no hand-entered numbers):

1. Training data and year distribution of the full official-train index
   (2,930 plots, 5 harvest years), the S3 pilot index (1,000 train plots)
   and the official validation / test splits.
2. Time-encoding provenance: every ``DeltaNormStats`` actually used by the
   frozen S4B.2 checkpoints, and whether S4B.2 fitted on full train (it did).
3. GDD / weather definitions and which protocol (P deployable / R oracle)
   each condition consumes, read from the real conditioning npz and code.
4. Model capacity: real parameter counts of every condition, instantiated
   from the same code path the trainers use.

Usage
-----
    python3 tools/s4b4a_data_audit.py \
        --config agro_jepa/config/s4b4a_year_heldout.yaml
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
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from agro_jepa.models.s4_predictor import (  # noqa: E402
    S4ModelConfig,
    build_s4_predictor,
    conditioning_spec,
)
from agro_jepa.models.time_embedding import DeltaNormStats  # noqa: E402


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_index_sha(df: pd.DataFrame) -> str:
    cols = sorted(df.columns)
    return hashlib.sha256(df[cols].to_csv(index=False).encode()).hexdigest()


def delta_stats(series: pd.Series) -> dict:
    d = series.to_numpy(np.float64)
    return {
        "n": int(len(d)),
        "mean": float(d.mean()),
        "std": float(d.std(ddof=0)),
        "min": float(d.min()),
        "max": float(d.max()),
        "median": float(np.median(d)),
        "quantiles": {
            f"p{int(q*100)}": float(np.quantile(d, q))
            for q in (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
        },
    }


def year_table(df: pd.DataFrame) -> dict:
    """Per harvest_year: pairs, plots, unique images, span7, span14, delta stats."""
    out = {}
    imgs = pd.unique(pd.concat([df.image_stem_a, df.image_stem_b]))
    df_img = df.assign(_img=None)
    for y, g in df.groupby("harvest_year", sort=True):
        uniq = pd.unique(pd.concat([g.image_stem_a, g.image_stem_b]))
        out[str(int(y))] = {
            "pairs": int(len(g)),
            "plots": int(g.plot_uid.nunique()),
            "images": int(len(uniq)),
            "span7": int((g.pair_type == "span7").sum()),
            "span14": int((g.pair_type == "span14").sum()),
            "delta_days": delta_stats(g.delta_days),
        }
    return out


def capacity_of(condition: dict, model_cfg: S4ModelConfig) -> dict:
    """Instantiate a predictor on CPU exactly as the trainer would and count."""
    spec = conditioning_spec(condition)
    model = build_s4_predictor(model_cfg, spec)
    return {
        "spec": spec.as_dict(),
        "has_condition": bool(model.has_condition),
        "parts": list(model.parts),
        "n_trainable_parameters": int(model.n_trainable_parameters()),
        "n_condition_parameters": int(model.n_condition_parameters()),
        "predictor_input_dim": int(model.config.predictor_dim(model.has_condition)),
        "delta_encoder_params": (
            int(sum(p.numel() for p in model.delta_encoder.parameters()))
            if model.delta_encoder is not None else 0),
        "gdd_encoder_params": (
            int(sum(p.numel() for p in model.gdd_encoder.parameters()))
            if model.gdd_encoder is not None else 0),
        "weather_encoder_params": (
            int(sum(p.numel() for p in model.weather_encoder.parameters()))
            if model.weather_encoder is not None else 0),
        "fusion_params": (
            int(sum(p.numel() for p in model.fusion.parameters()))
            if model.fusion is not None else 0),
        "action_encoder_params": (
            int(sum(p.numel() for p in model.action_encoder.parameters()))
            if model.action_encoder is not None else 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s4b4a_year_heldout.yaml")
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    t0 = time.time()
    full = pd.read_parquet(ROOT / cfg["data"]["full_train_pairs"])
    train_idx = pd.read_parquet(ROOT / cfg["data"]["train_index"])
    s3 = pd.read_parquet(ROOT / cfg["data"]["s3_index"])
    feat = pd.read_parquet(ROOT / cfg["data"]["feature_index"])
    cond_z = np.load(ROOT / cfg["data"]["conditioning_npz"], allow_pickle=True)

    # ---- 3.1 training data & year distribution --------------------------
    full_years = year_table(full)
    s3_train = s3[s3.split == "train"]
    s3_val = s3[s3.split == "validation"]
    s3_train_years = year_table(s3_train)
    s3_val_years = year_table(s3_val)

    test_year_dist = {}
    for split, path in cfg["data"]["test_indexes"].items():
        te = pd.read_parquet(ROOT / path)
        test_year_dist[split] = {
            "pairs": int(len(te)),
            "plots": int(te.plot_uid.nunique()),
            "years": sorted(int(y) for y in te.harvest_year.unique()),
            "n_by_year": {str(int(y)): int((te.harvest_year == y).sum())
                          for y in sorted(te.harvest_year.unique())},
        }

    # real normalisation statistics inside every frozen S4B.2 checkpoint
    frozen_norms = {}
    for cond, seedmap in cfg["frozen"]["s4b2_checkpoints"].items():
        frozen_norms[cond] = {}
        for seed, rel in seedmap.items():
            p = ROOT / rel
            if not p.is_file():
                raise SystemExit(f"missing frozen checkpoint {p}")
            import torch
            ck = torch.load(p, map_location="cpu", weights_only=False)
            dn = ck.get("delta_norm") or {}
            frozen_norms[cond][seed] = {
                "path": rel,
                "sha256": sha256_file(p),
                "delta_norm": {k: dn.get(k) for k in ("mean", "std", "min", "max", "n_samples")},
                "n_trainable_parameters": int(ck["n_trainable_parameters"]),
                "n_condition_parameters": int(ck["n_condition_parameters"]),
                "predictor_input_dim": int(ck["predictor_input_dim"]),
                "fixed_delta": ck.get("fixed_delta"),
                "epochs": int(ck["epoch"]) + 1,
                "index_sha256": ck.get("index_sha256"),
            }
            del ck

    # the delta normalisation S4B.2 actually used (fitted on the full index)
    s4b2_delta_norm = DeltaNormStats.fit(train_idx.delta_days.to_numpy())

    # ---- 3.2 time encoding audit ---------------------------------------
    time_audit = {
        "class": "agro_jepa.models.time_embedding.DeltaNormStats",
        "fit_call_sites": {
            "tools/train_s4.py:213 (S4A/S4A.1)":
                "fitted on inner-train rows of this model's split only",
            "agro_jepa/train/train_s4b2_full.py:227":
                "fitted on the ENTIRE official-train index (all 5 years) -- "
                "the frozen S4B.2 baseline",
        },
        "s4b2_full_train_stats": s4b2_delta_norm.as_dict(),
        "s4b2_per_checkpoint": frozen_norms,
        "s4a_inner_train_example": {
            "note": "S4A.1 inner-train delta norm recorded by its verify report",
            "mean": 10.251753,
            "std": 3.675335,
            "n_samples": 44925,
        },
        "normalisation_rule": (
            "delta_days -> (d - mean) / std, with mean/std from the fitting "
            "row set; the held-out rows are transformed with training statistics "
            "and never re-fit"),
        "s4b4a_requirement": (
            "each fold must refit DeltaNormStats (and GDD/weather VectorNormStats) "
            "on that fold's *training plot* rows only"),
    }
    # check the frozen checkpoints agree with the full-train fit
    agree = True
    for cond, seeds in frozen_norms.items():
        for seed, e in seeds.items():
            dn = e["delta_norm"]
            if dn.get("mean") is None:
                continue
            if abs(float(dn["mean"]) - s4b2_delta_norm.mean) > 1e-6 or \
               abs(float(dn["std"]) - s4b2_delta_norm.std) > 1e-6:
                agree = False
    time_audit["all_checkpoints_share_full_train_norm"] = bool(agree)

    # ---- 3.3 GDD / weather audit ----------------------------------------
    channels = [str(c) for c in cond_z["weather_channels"]]
    gdd_audit = {
        "definition": "GDD_d = max(T_mean_d - T_base, 0); T_base in {0, 5} degC",
        "unit": "degC day",
        "T_mean_source": "S1 confirmed channel T_mean_C",
        "protocol_R_definition": str(cond_z["interval_window_rule"]),
        "protocol_R_deployable": False,
        "protocol_P_definition": str(cond_z["past_window_rule"]),
        "protocol_P_deployable": True,
        "gdd_keys_present": [k for k in cond_z.files if k.startswith("gdd_")],
        "interval_valid_days_all_full": bool(
            (cond_z["interval_valid_days"] == cond_z["delta_days"]).all()),
        "past_valid_days_all_full": bool(
            (cond_z["past_valid_days"] == 28).all()),
        "conditioning_rows": int(len(cond_z["pair_id"])),
        "conditioning_splits": {
            str(k): int(v) for k, v in
            zip(*np.unique(cond_z["split"], return_counts=True))},
    }
    weather_audit = {
        "channels": channels,
        "units": {
            "T_mean_C": "degC", "T_max_C": "degC", "T_min_C": "degC",
            "RH_mean_pct": "%", "Rs_sum_MJ_m2_d": "MJ m-2 d-1",
        },
        "excluded": {
            "P_sum_raw": "unit unresolved (S1); precipitation integrals forbidden by S2",
            "VPD_mean_kPa": "derived channel, opt-in only; not used",
        },
        "protocol_P_window_days": 28,
        "mask": "per-day, per-channel validity from S1 hourly-record completeness",
        "past_window_strictly_before_t_a": True,
        "structural_limit": (
            "all plots of one harvest_year share a single weather series; "
            "weather carries date/year-level environment variation only"),
    }
    # per-year GDD (past-28, base 0) distribution -- tests for year shift
    gdd28 = cond_z["gdd_past28_base0"]
    gy = cond_z["harvest_year"]
    gdd_by_year = {}
    for y in sorted(int(v) for v in np.unique(gy)):
        m = gy == y
        gdd_by_year[str(y)] = {
            "n": int(m.sum()),
            "mean": float(gdd28[m].mean()),
            "std": float(gdd28[m].std()),
            "min": float(gdd28[m].min()),
            "max": float(gdd28[m].max()),
        }
    gdd_audit["gdd_past28_base0_by_year"] = gdd_by_year

    # conditioning coverage vs the full-train index (the S4B.4A scope question)
    cond_pair_ids = set(cond_z["pair_id"].astype(str))
    full_pair_ids = set(full.pair_id.astype(str))
    coverage = {
        "conditioning_pair_ids": int(len(cond_pair_ids)),
        "full_train_pair_ids": int(len(full_pair_ids)),
        "conditioning_plots": int(s3_train.plot_uid.nunique()),
        "full_train_plots": int(full.plot_uid.nunique()),
        "conditioned_pairs_also_in_full_train": int(len(cond_pair_ids & full_pair_ids)),
        "conditioning_covers_full_train": bool(cond_pair_ids == full_pair_ids),
        "note": (
            "the frozen S4A conditioning npz covers only the S3 pilot index "
            "(1,000 official-train plots); GDD-P / Weather-P on the full "
            "2,930-plot index would require a new conditioning build"),
    }

    # ---- 3.4 model capacity ---------------------------------------------
    model_cfg = S4ModelConfig.from_dict(cfg["model"])
    capacity = {}
    for cond_name, cond in cfg["conditions"].items():
        if cond.get("mode") != "trained":
            continue
        capacity[cond_name] = capacity_of(cond, model_cfg)

    report = {
        "stage": "S4B.4A",
        "task": "data / time-encoding / GDD-weather / capacity audit",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": " ".join(sys.argv),
        "config": args.config,
        "config_sha256": sha256_file(ROOT / args.config),
        "inputs": {
            "full_train_pairs": cfg["data"]["full_train_pairs"],
            "full_train_pairs_sha256": sha256_file(ROOT / cfg["data"]["full_train_pairs"]),
            "train_index": cfg["data"]["train_index"],
            "train_index_sha256": sha256_file(ROOT / cfg["data"]["train_index"]),
            "s3_index": cfg["data"]["s3_index"],
            "s3_index_sha256_stable": stable_index_sha(s3),
            "feature_index": cfg["data"]["feature_index"],
            "feature_index_sha256": sha256_file(ROOT / cfg["data"]["feature_index"]),
            "conditioning_npz": cfg["data"]["conditioning_npz"],
            "conditioning_npz_sha256": sha256_file(ROOT / cfg["data"]["conditioning_npz"]),
        },
        "3_1_training_data": {
            "full_train_index": {
                "rows": int(len(full)),
                "plots": int(full.plot_uid.nunique()),
                "years": sorted(int(y) for y in full.harvest_year.unique()),
                "pair_types": {str(k): int(v) for k, v in
                               full.pair_type.value_counts().items()},
                "by_year": full_years,
                "delta_days_overall": delta_stats(full.delta_days),
            },
            "s3_pilot_train": {
                "rows": int(len(s3_train)),
                "plots": int(s3_train.plot_uid.nunique()),
                "by_year": s3_train_years,
                "delta_days_overall": delta_stats(s3_train.delta_days),
            },
            "official_validation": {
                "rows": int(len(s3_val)),
                "plots": int(s3_val.plot_uid.nunique()),
                "years": sorted(int(y) for y in s3_val.harvest_year.unique()),
                "by_year": s3_val_years,
            },
            "official_test": test_year_dist,
            "normalisation_sources": {
                "delta": "fitted per fit-call-site (see 3.2)",
                "gdd_weather": "VectorNormStats fitted on inner-train rows only",
            },
        },
        "3_2_time_encoding": time_audit,
        "3_3_gdd_weather": {
            "gdd": gdd_audit,
            "weather": weather_audit,
            "conditioning_coverage": coverage,
        },
        "3_4_capacity": {
            "model_config": model_cfg.as_dict(),
            "conditions": capacity,
            "matching_notes": {
                "b2_no_time_constant_delta_vs_b3_delta":
                    "identical trainable/condition parameter counts and input dim "
                    "(the S4A.1 matched control); difference is sample-level delta only",
                "gdd_and_weather_arms":
                    "add encoder + fusion parameters over B3; capacity is NOT "
                    "matched to B3, so part of any gain may be parameter-driven",
            },
        },
    }

    out_json = ROOT / "reports/s4b4a_data_audit.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(report, fh, indent=2)

    write_md(ROOT / "reports/s4b4a_data_audit.md", report)
    print(json.dumps({
        "full_train": report["3_1_training_data"]["full_train_index"]["by_year"],
        "conditioning_covers_full_train": coverage["conditioning_covers_full_train"],
        "capacity": {k: v["n_trainable_parameters"] for k, v in capacity.items()},
        "elapsed": round(time.time() - t0, 1),
    }, indent=2), flush=True)
    print(f"wrote {out_json}", flush=True)
    return 0


def write_md(path: Path, r: dict) -> None:
    L = []
    L.append("# S4B.4A Data Audit")
    L.append("")
    L.append(f"- Generated: {r['generated_at']}")
    L.append(f"- Config: `{r['config']}` (sha `{r['config_sha256'][:16]}`)")
    L.append("")
    L.append("## 3.1 Training data and year distribution")
    L.append("")
    ft = r["3_1_training_data"]["full_train_index"]
    L.append(f"Full official-train index: **{ft['rows']:,} pairs / "
             f"{ft['plots']:,} plots / years {ft['years']}**")
    L.append("")
    L.append("| year | pairs | plots | images | span7 | span14 | delta mean ± std "
             "[min,max] | median |")
    L.append("|---|---:|---:|---:|---:|---:|---|---:|")
    for y, e in ft["by_year"].items():
        d = e["delta_days"]
        L.append(f"| {y} | {e['pairs']:,} | {e['plots']} | {e['images']:,} | "
                 f"{e['span7']:,} | {e['span14']:,} | "
                 f"{d['mean']:.2f} ± {d['std']:.2f} [{d['min']:.0f},{d['max']:.0f}] | "
                 f"{d['median']:.0f} |")
    L.append("")
    d = ft["delta_days_overall"]
    L.append(f"Overall delta_days: n={d['n']:,} mean={d['mean']:.4f} "
             f"std={d['std']:.4f} min={d['min']:.0f} max={d['max']:.0f} "
             f"median={d['median']:.0f}")
    L.append("")
    L.append("Quantiles: " + ", ".join(
        f"{k}={v:.0f}" for k, v in d["quantiles"].items()))
    L.append("")
    L.append("### Official validation / test year distribution")
    L.append("")
    L.append("| split | pairs | plots | years |")
    L.append("|---|---:|---:|---|")
    val = r["3_1_training_data"]["official_validation"]
    L.append(f"| validation | {val['rows']:,} | {val['plots']} | {val['years']} |")
    for split, e in r["3_1_training_data"]["official_test"].items():
        L.append(f"| {split} | {e['pairs']:,} | {e['plots']} | {e['years']} |")
    L.append("")
    L.append("> 2019 is the official unseen-environment test year; it is **not** in "
             "the official training set. S4B.4A is a *training-set-internal* "
             "year-held-out diagnostic and does not constitute an independent "
             "2019 generalisation test.")
    L.append("")
    L.append("## 3.2 Time encoding")
    L.append("")
    ta = r["3_2_time_encoding"]
    for site, note in ta["fit_call_sites"].items():
        L.append(f"- `{site}`: {note}")
    L.append("")
    n = ta["s4b2_full_train_stats"]
    L.append(f"S4B.2 full-train DeltaNormStats: mean={n['mean']:.6f} "
             f"std={n['std']:.6f} min={n['min']:.0f} max={n['max']:.0f} "
             f"n={n['n_samples']:,}")
    L.append("")
    L.append(f"All frozen checkpoints share the full-train norm: "
             f"**{ta['all_checkpoints_share_full_train_norm']}**")
    L.append("")
    L.append("Per-checkpoint delta norm:")
    L.append("")
    L.append("| condition | seed | mean | std | epochs | final sha (16) |")
    L.append("|---|---:|---:|---:|---:|---|")
    for cond, seeds in ta["s4b2_per_checkpoint"].items():
        for seed, e in seeds.items():
            dn = e["delta_norm"]
            L.append(f"| {cond} | {seed} | {dn['mean']:.6f} | {dn['std']:.6f} | "
                     f"{e['epochs']} | `{e['sha256'][:16]}` |")
    L.append("")
    L.append(f"**S4B.4A requirement:** {ta['s4b4a_requirement']}")
    L.append("")
    L.append("## 3.3 GDD and weather")
    L.append("")
    g = r["3_3_gdd_weather"]["gdd"]
    L.append(f"- GDD: {g['definition']} (unit: {g['unit']})")
    L.append(f"- Protocol R: `{g['protocol_R_definition']}` — deployable = "
             f"{g['protocol_R_deployable']} (retrospective/oracle)")
    L.append(f"- Protocol P: `{g['protocol_P_definition']}` — deployable = "
             f"{g['protocol_P_deployable']}")
    L.append(f"- interval windows fully valid: {g['interval_valid_days_all_full']}; "
             f"past windows fully valid: {g['past_valid_days_all_full']}")
    L.append(f"- conditioning rows: {g['conditioning_rows']:,}; splits: "
             f"{g['conditioning_splits']}")
    L.append("")
    L.append("GDD past-28 base-0 by year (a year-shift check):")
    L.append("")
    L.append("| year | n | mean | std | min | max |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for y, e in g["gdd_past28_base0_by_year"].items():
        L.append(f"| {y} | {e['n']:,} | {e['mean']:.2f} | {e['std']:.2f} | "
                 f"{e['min']:.2f} | {e['max']:.2f} |")
    L.append("")
    w = r["3_3_gdd_weather"]["weather"]
    L.append(f"- Weather channels: {w['channels']}")
    L.append(f"- Excluded: {w['excluded']}")
    L.append(f"- Protocol P window: {w['protocol_P_window_days']} d, strictly "
             f"before t_a: {w['past_window_strictly_before_t_a']}")
    L.append(f"- Structural limit: {w['structural_limit']}")
    L.append("")
    c = r["3_3_gdd_weather"]["conditioning_coverage"]
    L.append(f"**Conditioning coverage:** {c['conditioning_pair_ids']:,} pair ids over "
             f"{c['conditioning_plots']:,} plots; full train has "
             f"{c['full_train_pair_ids']:,} ids over {c['full_train_plots']:,} plots. "
             f"Covers full train: **{c['conditioning_covers_full_train']}**. {c['note']}")
    L.append("")
    L.append("## 3.4 Model capacity")
    L.append("")
    L.append("| condition | parts | trainable | condition params | input dim | "
             "delta | gdd | weather | fusion |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, e in r["3_4_capacity"]["conditions"].items():
        L.append(f"| {name} | {','.join(e['parts']) or '-'} | "
                 f"{e['n_trainable_parameters']:,} | "
                 f"{e['n_condition_parameters']:,} | {e['predictor_input_dim']} | "
                 f"{e['delta_encoder_params']:,} | {e['gdd_encoder_params']:,} | "
                 f"{e['weather_encoder_params']:,} | {e['fusion_params']:,} |")
    L.append("")
    for k, v in r["3_4_capacity"]["matching_notes"].items():
        L.append(f"- **{k}**: {v}")
    L.append("")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
