#!/usr/bin/env python3
"""S2: build FIP 1.0 image pairs and report real pair statistics.

Reads only the frozen S1 manifests; writes a derived pair table outside the
committed manifest directory plus a committed statistics report.

Usage:
    python3 tools/s2_build_pairs.py \
        --config agro_jepa/config/s2_dataset.yaml \
        --out external_data/pairs_cache/fip1_pairs_s2.parquet \
        --report reports/s2_pairs_stats.json \
        --sample-csv reports/s2_pairs_sample.csv
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

from agro_jepa.datasets.fip_pairs import (  # noqa: E402
    OFFICIAL_SPLITS,
    TRAITS,
    TRAIT_NAMES,
    TRAIT_UNITS,
    TRAIT_YEARS,
    PairRule,
    build_pairs,
    load_custom_splits,
    load_observations,
    load_plots,
    pair_manifest_schema,
)
from agro_jepa.datasets.fip_weather import (  # noqa: E402
    DEFAULT_WEATHER_CHANNELS,
    DailyWeatherStore,
    weather_provenance,
)


def load_cfg(path: str | Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_stats(pairs: pd.DataFrame) -> dict:
    """Counts by official split, custom split, year and pair type."""
    out = {}
    for col in ("split", "custom_split", "harvest_year", "pair_type"):
        out[f"by_{col}"] = {
            str(k): int(v) for k, v in pairs[col].value_counts().sort_index().items()
        }
    out["by_split_and_type"] = {
        f"{s}|{t}": int(v)
        for (s, t), v in pairs.groupby(["split", "pair_type"]).size().items()
    }
    return out


def label_stats(pairs: pd.DataFrame) -> dict:
    out = {}
    for t in TRAITS:
        both = pairs[f"{t}_mask_a"] & pairs[f"{t}_mask_b"]
        out[t] = {
            "name": TRAIT_NAMES[t],
            "unit": TRAIT_UNITS[t],
            "years_with_labels": list(TRAIT_YEARS[t]),
            "n_both_valid": int(both.sum()),
            "n_a_valid": int(pairs[f"{t}_mask_a"].sum()),
            "n_b_valid": int(pairs[f"{t}_mask_b"].sum()),
            "n_value_present_a": int(pairs[f"{t}_a"].notna().sum()),
            "n_value_present_b": int(pairs[f"{t}_b"].notna().sum()),
            "n_zero_a": int((pairs[f"{t}_a"] == 0).sum()),
            "n_zero_b": int((pairs[f"{t}_b"] == 0).sum()),
            "n_zero_valid_a": int(((pairs[f"{t}_a"] == 0) & pairs[f"{t}_mask_a"]).sum()),
            "n_zero_valid_b": int(((pairs[f"{t}_b"] == 0) & pairs[f"{t}_mask_b"]).sum()),
            "n_both_valid_by_split": {
                str(k): int(v) for k, v in both.groupby(pairs.split).sum().items()
            },
        }
    out["ph_negative"] = {
        "n_a": int(pairs.ph_negative_a.sum()),
        "n_b": int(pairs.ph_negative_b.sum()),
        "note": "raw negative PH values are preserved (not clipped); validity is "
                "deferred to a dedicated sensitivity analysis",
    }
    return out


def delta_stats(pairs: pd.DataFrame) -> dict:
    d = pairs.delta_days.to_numpy()
    consecutive = pairs.loc[pairs.pair_type == "consecutive", "delta_days"].to_numpy()
    return {
        "min": int(d.min()),
        "max": int(d.max()),
        "median": float(np.median(d)),
        "mean": float(d.mean()),
        "p90": float(np.percentile(d, 90)),
        "hist_0_3": int(((d >= 0) & (d <= 3)).sum()),
        "hist_4_7": int(((d >= 4) & (d <= 7)).sum()),
        "hist_8_14": int(((d >= 8) & (d <= 14)).sum()),
        "hist_15_21": int(((d >= 15) & (d <= 21)).sum()),
        "hist_22_28": int(((d >= 22) & (d <= 28)).sum()),
        "hist_29_35": int(((d >= 29) & (d <= 35)).sum()),
        "hist_36_60": int(((d >= 36) & (d <= 60)).sum()),
        "hist_60plus": int((d > 60).sum()),
        "n_dt_zero": int((d == 0).sum()),
        "consecutive_only": {
            "n": int(consecutive.size),
            "min": int(consecutive.min()),
            "max": int(consecutive.max()),
            "median": float(np.median(consecutive)),
            "mean": float(consecutive.mean()),
            "p90": float(np.percentile(consecutive, 90)),
        } if consecutive.size else {},
    }


def leakage_checks(plots: pd.DataFrame, custom: pd.DataFrame, pairs: pd.DataFrame) -> dict:
    # official split: no plot_uid may be shared
    by_split = {s: set(plots.loc[plots.split == s, "plot_uid"]) for s in OFFICIAL_SPLITS}
    shared = {}
    for i, a in enumerate(OFFICIAL_SPLITS):
        for b in OFFICIAL_SPLITS[i + 1:]:
            n = len(by_split[a] & by_split[b])
            if n:
                shared[f"{a}__{b}"] = n
    # custom split is additive; official_split column must be preserved verbatim
    merged = custom.merge(
        plots[["plot_uid", "split"]], on="plot_uid", how="outer", indicator=True
    )
    official_preserved = bool(
        (merged["_merge"] == "both").all()
        and (merged["official_split"] == merged["split"]).all()
    )

    def gset(name):
        return set(custom.loc[custom.custom_split == name, "genotype_id"])

    cross_train = gset("cross_year_test") & gset("train")
    cross_id = gset("cross_year_test") & gset("id_test")
    # pair ends must carry their own plot_uid in the audited stem
    same_plot = bool(
        all(s.startswith(p) for s, p in zip(pairs.image_stem_a, pairs.plot_uid))
        and all(s.startswith(p) for s, p in zip(pairs.image_stem_b, pairs.plot_uid))
    )
    # image dates must be the dates embedded in the audited filename stem
    stem_date_a = pairs.image_stem_a.str.extract(r"_(\d{8})_", expand=False)
    stem_date_b = pairs.image_stem_b.str.extract(r"_(\d{8})_", expand=False)
    stem_dates_match = bool(
        (stem_date_a == pairs.date_a.str.replace("-", "", regex=False)).all()
        and (stem_date_b == pairs.date_b.str.replace("-", "", regex=False)).all()
    )
    # pair split must inherit the plot split
    plot_split = plots.set_index("plot_uid")["split"]
    inherited = bool((pairs.split.to_numpy() == plot_split.reindex(pairs.plot_uid).to_numpy()).all())

    return {
        "official_split_shared_plot_uid": shared,
        "official_split_disjoint_plots": len(shared) == 0,
        "custom_split_rows_match_plots": official_preserved,
        "custom_cross_year_test_vs_train_shared_genotypes": len(cross_train),
        "custom_cross_year_test_vs_id_test_shared_genotypes": len(cross_id),
        "custom_note": "cross-year test holds out YEARS, not genotypes; genotype "
                       "overlap with train is by construction and is reported, "
                       "not hidden",
        "every_pair_shares_one_plot": same_plot,
        "image_dates_match_filename_stems": stem_dates_match,
        "pair_split_inherited_from_plot": inherited,
        "n_plots": int(pairs.plot_uid.nunique()),
        "n_genotypes": int(pairs.genotype_id.nunique()),
    }


def normalization_stats(pairs: pd.DataFrame, weather: DailyWeatherStore,
                        fit_split: str = "train") -> dict:
    """Fit normalization on the training split only (never on test labels)."""
    tr = pairs[pairs.split == fit_split]
    out = {"fit_split": fit_split, "n_pairs_used": int(len(tr)), "traits": {}}
    for t in ("cc", "ph"):
        vals = np.concatenate([
            tr.loc[tr[f"{t}_mask_a"], f"{t}_a"].to_numpy(dtype=float),
            tr.loc[tr[f"{t}_mask_b"], f"{t}_b"].to_numpy(dtype=float),
        ])
        vals = vals[np.isfinite(vals)]
        out["traits"][t] = {
            "name": TRAIT_NAMES[t],
            "n": int(vals.size),
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=0)),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }
    train_years = sorted(set(tr.harvest_year.tolist()))
    out["weather_years"] = train_years
    out["weather"] = {}
    for ch in DEFAULT_WEATHER_CHANNELS:
        arrs = [weather.channel_values(int(y), ch) for y in train_years]
        v = np.concatenate(arrs)
        v = v[np.isfinite(v)]
        out["weather"][ch] = {
            "n": int(v.size),
            "mean": float(v.mean()),
            "std": float(v.std(ddof=0)),
            "min": float(v.min()),
            "max": float(v.max()),
        }
    out["note"] = ("statistics use train-split labels only; weather uses daily "
                   "series of years that contain train plots (2019 excluded)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s2_dataset.yaml")
    ap.add_argument("--mode", default=None, choices=["consecutive", "span", "both"])
    ap.add_argument("--spans", default=None, help="comma-separated, e.g. 7,14,21,28")
    ap.add_argument("--out", default="external_data/pairs_cache/fip1_pairs_s2.parquet")
    ap.add_argument("--report", default="reports/s2_pairs_stats.json")
    ap.add_argument("--sample-csv", default="reports/s2_pairs_sample.csv")
    ap.add_argument("--sample-rows", type=int, default=200)
    ap.add_argument("--no-parquet", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    cfg = load_cfg(args.config)
    data, pr = cfg["data"], cfg["pair_rule"]

    obs = load_observations(data["observations"])
    plots = load_plots(data["plots"])
    custom = load_custom_splits(data["custom_splits"])

    rule = PairRule(
        mode=args.mode or pr["mode"],
        spans=tuple(int(s) for s in (args.spans.split(",") if args.spans else pr["spans"])),
        span_tolerance_days=int(pr["span_tolerance_days"]),
        min_delta_days=int(pr["min_delta_days"]),
        max_delta_days=pr.get("max_delta_days"),
        label_max_lag_days=int(pr["label_max_lag_days"]),
        dedupe_image_stems=bool(pr["dedupe_image_stems"]),
        manifest_id=str(cfg["manifest_id"]),
    )

    pairs, stats = build_pairs(obs, plots, custom, rule)
    print(f"built {len(pairs):,} pairs in {time.time() - t0:.1f}s")

    # ---- weather window verification (vectorised, all pairs) --------------
    weather = DailyWeatherStore(
        data["weather_daily"], channels=tuple(cfg["weather"]["channels"])
    )
    win_valid = np.zeros(len(pairs), dtype=np.int64)
    for y, g in pairs.groupby("harvest_year"):
        win_valid[g.index.to_numpy()] = weather.window_valid_counts(
            int(y), g.date_a.to_numpy(), g.date_b.to_numpy()
        )
    n_expected = pairs.delta_days.to_numpy()
    weather_check = {
        "n_pairs": int(len(pairs)),
        "n_windows_length_matches_delta": int(
            (pairs.window_n_days.to_numpy() == n_expected).all()
        ),
        "n_windows_fully_covered": int((win_valid == n_expected).sum()),
        "n_windows_with_missing_days": int((win_valid != n_expected).sum()),
        "min_valid_days": int(win_valid.min()),
        "worst_coverage_ratio": float((win_valid / np.maximum(n_expected, 1)).min()),
        "n_days_expected_by_split": {
            str(k): int(v) for k, v in pd.Series(n_expected).groupby(pairs.split).sum().items()
        },
        "weather_years": list(weather.years),
        "weather_manifest_id": weather.manifest_id,
        "weather_daily_sha256": weather.csv_sha256,
    }

    # ---- label / split / delta statistics --------------------------------
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "stage": "S2",
        "manifest_id": cfg["manifest_id"],
        "elapsed_seconds": round(time.time() - t0, 2),
        "rule": rule.as_dict(),
        "input_manifests": {
            "observations": str(data["observations"]),
            "plots": str(data["plots"]),
            "custom_splits": str(data["custom_splits"]),
            "weather_daily": str(data["weather_daily"]),
        },
        "dedupe": stats.get("dedupe", {}),
        "n_pairs": int(len(pairs)),
        "pair_types": {str(k): int(v) for k, v in pairs.pair_type.value_counts().items()},
        "span_pair_counts": {
            str(k): int(v)
            for k, v in pairs.loc[pairs.span_days > 0].span_days.value_counts().sort_index().items()
        },
        "splits": split_stats(pairs),
        "delta_days": delta_stats(pairs),
        "labels": label_stats(pairs),
        "weather_window": weather_check,
        "weather_provenance": weather_provenance(),
        "weather_coverage_by_year": {str(y): weather.coverage(y) for y in weather.years},
        "leakage_checks": leakage_checks(plots, custom, pairs),
        "normalization": normalization_stats(pairs, weather,
                                             fit_split=cfg["normalization"]["fit_split"]),
        "pair_schema": pair_manifest_schema(),
        "columns": list(pairs.columns),
    }

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)
    print(f"wrote {args.report}")

    # ---- derived pair table (not committed; deterministic) ----------------
    if not args.no_parquet:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        pairs.to_parquet(out, index=False)
        report["output_parquet"] = {
            "path": str(out),
            "bytes": out.stat().st_size,
            "sha256": sha256_file(out),
            "committed": False,
            "reason": "derived from committed S1 manifests; rebuild with this tool",
        }
        print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, not committed)")
        with open(args.report, "w") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False, default=str)

    # ---- committed sample for traceability -------------------------------
    if args.sample_rows > 0:
        rng = np.random.RandomState(42)
        if len(pairs) > args.sample_rows:
            idx = np.sort(rng.choice(len(pairs), args.sample_rows, replace=False))
            sample = pairs.iloc[idx]
        else:
            sample = pairs
        Path(args.sample_csv).parent.mkdir(parents=True, exist_ok=True)
        sample.to_csv(args.sample_csv, index=False)
        print(f"wrote {args.sample_csv} ({len(sample)} rows)")

    print(json.dumps({
        "n_pairs": report["n_pairs"],
        "pair_types": report["pair_types"],
        "splits": report["splits"]["by_split"],
        "dedupe": report["dedupe"],
        "weather": {k: v for k, v in weather_check.items()
                    if k in ("n_windows_fully_covered", "n_windows_with_missing_days")},
        "leakage": {k: v for k, v in report["leakage_checks"].items()
                    if k.startswith("n_") or k.endswith("_plots")},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
