#!/usr/bin/env python3
"""S4B test-data preparation: official split audit + frozen test pair index.

This stage performs *data preparation only*.  It never trains, never selects a
model, never touches an official test phenotype label for model choice and never
computes a test metric.  Outputs are (a) an audit of every official split as the
FIP 1.0 publication defines it, and (b) future-image pair indexes built inside
each official test split under the frozen S4A pairing rule.

What is verified here
---------------------
1. **Official split audit** -- per split: plot count, image count, image date
   range, harvest_year coverage, genotype coverage and weather coverage, read
   from the frozen S1 manifests (``fip1_plots.csv`` / ``fip1_observations.csv.gz``)
   and never inferred from the S3 index.
2. **Split semantics** -- the four test splits are cross-checked against the
   definitions in the FIP 1.0 paper (Test P / G / E / G+E): genotype and
   year-site overlap with the training split is computed, so ``test_plot`` is
   shown to be seen-genotype/seen-environment, ``test_genotype`` unseen-genotype
   / seen-environment, ``test_environment`` seen-genotype / unseen-environment and
   ``test_genotype_environment`` both unseen.
3. **Test pair index** -- the same span rule as S3/S4A (span7 / span14 with
   +/-2 d tolerance, dedupe physical pairs, both images in the same plot, split
   inherited from the plot).  Pairs are built per official test split, so no
   pair crosses a split boundary and the official attribution is preserved.
4. **Feature / weather readiness** -- for each pair, whether both images have a
   cached DINOv2 feature, whether the weather store covers ``(t_a, t_b]`` fully,
   and whether the past window ``[t_a-28, t_a-1]`` is fully valid.  This is
   preparation of inputs, not evaluation.

Usage
-----
    python3 tools/s4b_prepare_test_data.py \
        --config agro_jepa/config/s4a1_matched_control.yaml
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
    PairRule,
    dedupe_observations,
    load_observations,
    load_plots,
)
from agro_jepa.datasets.fip_weather import DailyWeatherStore  # noqa: E402

PAIR_TYPES = ("span7", "span14")
SPAN_TOLERANCE_DAYS = 2
MIN_DELTA_DAYS = 1
MAX_DELTA_DAYS = 16
WEATHER_PAST_DAYS = 28
WEATHER_CHANNELS = ("T_mean_C", "T_max_C", "T_min_C", "RH_mean_pct", "Rs_sum_MJ_m2_d")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_index_sha(index: pd.DataFrame) -> str:
    """Same digest rule as ``agro_jepa.datasets.s3_pairs.index_sha256``."""
    cols = sorted(index.columns)
    return hashlib.sha256(index[cols].to_csv(index=False).encode()).hexdigest()


def pair_ids_for(frame: pd.DataFrame) -> np.ndarray:
    """Deterministic ids identical to the S2 rule."""
    import hashlib as _h

    out = np.empty(len(frame), dtype=object)
    for i, (pu, sa, sb, pt, sd) in enumerate(
        zip(frame.plot_uid, frame.image_stem_a, frame.image_stem_b,
            frame.pair_type, frame.span_days)
    ):
        out[i] = _h.sha1(
            f"fip1-s2-v1|{pu}|{sa}|{sb}|{pt}|{sd}".encode()
        ).hexdigest()[:16]
    return out


def build_pairs_for_plot(g: pd.DataFrame, rule: PairRule) -> pd.DataFrame:
    """Span7 / span14 pairs of one plot, using the frozen S2 span rule."""
    epoch = np.datetime64("1970-01-01", "D")
    dates = pd.to_datetime(g["image_date"]).to_numpy(dtype="datetime64[D]")
    dn = (dates - epoch).astype(np.int64)
    positions = np.arange(len(g))
    rows = []
    n = len(g)
    for k in range(n - 1):
        later = dn[k + 1:]
        for span in rule.spans:
            target = dn[k] + int(span)
            rel = int(np.searchsorted(later, target))
            best = None
            for c in (rel - 1, rel):
                if 0 <= c < later.size:
                    cand = (abs(int(later[c]) - target), int(later[c]) - target, c)
                    if best is None or cand < best:
                        best = cand
            if best is None or best[0] > rule.span_tolerance_days:
                continue
            delta = int(later[best[2]] - dn[k])
            if delta < rule.min_delta_days:
                continue
            if rule.max_delta_days is not None and delta > rule.max_delta_days:
                continue
            ib = k + 1 + best[2]
            ia = k
            rows.append({
                "pair_type": f"span{span}",
                "span_days": int(span),
                "image_stem_a": str(g["image_filename_stem"].iloc[ia]).upper(),
                "image_stem_b": str(g["image_filename_stem"].iloc[ib]).upper(),
                "date_a": pd.Timestamp(dates[ia]).strftime("%Y-%m-%d"),
                "date_b": pd.Timestamp(dates[ib]).strftime("%Y-%m-%d"),
                "delta_days": delta,
                "inner_plot_image_a": str(g["inner_plot_image"].iloc[ia]),
                "inner_plot_image_b": str(g["inner_plot_image"].iloc[ib]),
            })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s4a1_matched_control.yaml")
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    s4b = cfg["s4b"]

    t0 = time.time()
    out_dir = Path(s4b["index_out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- frozen manifests (read-only) ------------------------
    plots = load_plots(s4b["plots_manifest"])
    obs = load_observations(s4b["observations_manifest"])
    obs_d, dedupe_stats = dedupe_observations(obs)

    audit: dict = {
        "stage": "S4B",
        "task": "official FIP 1.0 split audit and test pair preparation",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": " ".join(sys.argv),
        "config": args.config,
        "config_sha256": sha256_file(args.config),
        "inputs": {
            "plots": s4b["plots_manifest"],
            "plots_sha256": sha256_file(s4b["plots_manifest"]),
            "observations": s4b["observations_manifest"],
            "observations_sha256": sha256_file(s4b["observations_manifest"]),
            "s2_pairs": s4b["s2_pairs"],
            "s2_pairs_sha256": sha256_file(s4b["s2_pairs"]),
            "images_root": s4b["images_root"],
            "feature_index": s4b["feature_index"],
            "feature_index_sha256": sha256_file(s4b["feature_index"]),
            "weather_daily": cfg["data"]["weather_daily"],
            "weather_daily_sha256": sha256_file(cfg["data"]["weather_daily"]),
        },
        "dedupe": dedupe_stats,
        "official_split_definitions": {
            "source_url": "https://doi.org/10.1093/gigascience/giaf051",
            "quote": (
                "Test (P): Unseen plots with seen genotypes and seen environments "
                "(all years except 2019); Test (G): Unseen genotypes with seen "
                "environments; Test (E): Unseen environment (2019) with seen "
                "genotypes; Test (G and E): Unseen genotypes and unseen "
                "environment (2019). Splitting uses the R package STPGA with the "
                "D-opt algorithm."
            ),
            "column_mapping": {
                "test_plot": "Test (P)",
                "test_genotype": "Test (G)",
                "test_environment": "Test (E)",
                "test_genotype_environment": "Test (G and E)",
            },
        },
        "pair_rule": {
            "pair_types": list(PAIR_TYPES),
            "span_tolerance_days": SPAN_TOLERANCE_DAYS,
            "min_delta_days": MIN_DELTA_DAYS,
            "max_delta_days": MAX_DELTA_DAYS,
            "dedupe_physical_pairs": True,
            "note": (
                "same rule as the frozen S3/S4A index; a pair's two images always "
                "belong to one plot and the split is inherited from that plot"
            ),
        },
        "splits": {},
    }

    train_plots = set(plots.loc[plots.split == "train", "plot_uid"])
    train_genotypes = set(plots.loc[plots.split == "train", "genotype_id"])
    train_yearsites = set(plots.loc[plots.split == "train", "yearsite_uid"])

    rule = PairRule(
        mode="span",
        spans=(7, 14),
        span_tolerance_days=SPAN_TOLERANCE_DAYS,
        min_delta_days=MIN_DELTA_DAYS,
        max_delta_days=MAX_DELTA_DAYS,
        dedupe_image_stems=True,
    )

    weather_store = DailyWeatherStore(
        cfg["data"]["weather_daily"],
        channels=list(WEATHER_CHANNELS),
        add_vpd=False,
        require_mainline=False,
    )
    weather_years = sorted(
        int(y) for y in pd.read_csv(
            cfg["data"]["weather_daily"], usecols=["harvest_year"]
        ).harvest_year.unique()
    )
    _wdf = pd.read_csv(cfg["data"]["weather_daily"], usecols=["harvest_year", "date"])
    _wdf["date"] = pd.to_datetime(_wdf["date"])
    weather_day_counts = {
        str(int(y)): {
            "n_days": int(len(g)),
            "start": str(g.date.min().date()),
            "end": str(g.date.max().date()),
        }
        for y, g in _wdf.groupby("harvest_year")
    }

    features = pd.read_parquet(s4b["feature_index"])
    feat_stems = set(features.image_filename_stem.astype(str).str.upper())

    test_indexes: dict[str, pd.DataFrame] = {}
    for split in s4b["official_splits"]:
        psub = plots[plots.split == split]
        osub = obs_d[obs_d.split == split]
        years = sorted(int(y) for y in psub.harvest_year.unique())
        yearsites = sorted(psub.yearsite_uid.unique())
        genotypes = set(psub.genotype_id)
        g_overlap = len(genotypes & train_genotypes)
        ys_overlap = len(set(yearsites) & train_yearsites)
        dates = pd.to_datetime(osub.image_date)
        audit["splits"][split] = {
            "role": "official test" if split.startswith("test") else (
                "official model-selection split" if split == "validation"
                else "official training split"),
            "paper_label": audit["official_split_definitions"]["column_mapping"].get(
                split),
            "n_plots": int(psub.plot_uid.nunique()),
            "n_images": int(len(osub)),
            "n_image_dates": int(pd.to_datetime(osub.image_date).nunique()),
            "n_inner_plot_images": int(osub.inner_plot_image.nunique()),
            "image_date_range": [str(dates.min().date()), str(dates.max().date())],
            "harvest_years": years,
            "yearsites": yearsites,
            "n_yearsites": len(yearsites),
            "genotypes": int(len(genotypes)),
            "genotypes_seen_in_train": g_overlap,
            "genotypes_unseen_in_train": int(len(genotypes) - g_overlap),
            "yearsites_seen_in_train": ys_overlap,
            "yearsites_unseen_in_train": int(len(set(yearsites)) - ys_overlap),
            "sowing_dates": sorted(psub.sowing_date.unique().tolist()),
            "harvest_dates": sorted(psub.harvest_date.unique().tolist()),
            "weather_years_available": [y for y in years if y in weather_years],
            "weather_years_missing": [y for y in years if y not in weather_years],
            "env_records_per_plot": int(psub.env_records.iloc[0]) if len(psub) else 0,
            "n_images_with_cached_feature": int(
                osub.image_filename_stem.astype(str).str.upper().isin(feat_stems).sum()
            ),
        }

    # ---------------- test pair indexes ----------------------------------
    for split in s4b["test_splits"]:
        osub = obs_d[obs_d.split == split].copy()
        osub["image_filename_stem"] = osub.image_filename_stem.astype(str).str.upper()
        pmeta = plots.set_index("plot_uid")
        parts = []
        for plot_uid, g in osub.groupby("plot_uid", sort=True):
            g = g.sort_values(["image_date", "image_time", "obs_index"],
                              kind="mergesort")
            block = build_pairs_for_plot(g, rule)
            if block.empty:
                continue
            block["plot_uid"] = plot_uid
            block["genotype_id"] = pmeta.at[plot_uid, "genotype_id"]
            block["harvest_year"] = int(pmeta.at[plot_uid, "harvest_year"])
            block["yearsite_uid"] = pmeta.at[plot_uid, "yearsite_uid"]
            block["split"] = split
            parts.append(block)
        if not parts:
            test_indexes[split] = pd.DataFrame()
            audit["splits"][split]["pairs"] = {"n_pairs": 0}
            continue
        df = pd.concat(parts, ignore_index=True)
        # dedupe physical pairs across the two span labels (longest span wins)
        df = df.sort_values(["plot_uid", "image_stem_a", "image_stem_b", "span_days",
                             "pair_type"], kind="mergesort")
        df = df.drop_duplicates(
            subset=["plot_uid", "image_stem_a", "image_stem_b"], keep="last"
        )
        df["pair_id"] = pair_ids_for(df)
        if df.pair_id.duplicated().any():
            raise RuntimeError(f"{split}: duplicate pair_id after dedupe")
        df["window_n_days"] = df.delta_days
        df = df.sort_values(["split", "pair_type", "delta_days", "pair_id"],
                            kind="mergesort").reset_index(drop=True)

        # readiness: cached features and weather coverage (inputs only)
        has_a = df.image_stem_a.isin(feat_stems)
        has_b = df.image_stem_b.isin(feat_stems)
        years = df.harvest_year.to_numpy()
        dates_a = pd.to_datetime(df.date_a).dt.date.to_numpy()
        dates_b = pd.to_datetime(df.date_b).dt.date.to_numpy()
        deltas = df.delta_days.to_numpy()
        int_ok = np.zeros(len(df), dtype=bool)
        past_ok = np.zeros(len(df), dtype=bool)
        for i in range(len(df)):
            if years[i] not in weather_years:
                continue
            w = weather_store.window(int(years[i]), dates_a[i], dates_b[i])
            int_ok[i] = bool(w.n_valid_days == deltas[i])
            pw = weather_store.window(
                int(years[i]),
                pd.Timestamp(dates_a[i]).date() - pd.Timedelta(days=WEATHER_PAST_DAYS + 1),
                pd.Timestamp(dates_a[i]).date() - pd.Timedelta(days=1),
            )
            past_ok[i] = bool(pw.n_valid_days == WEATHER_PAST_DAYS)

        df["feature_ready"] = has_a & has_b
        df["weather_interval_ready"] = int_ok
        df["weather_past_ready"] = past_ok
        df["weather_year_available"] = np.isin(years, weather_years)

        out_path = out_dir / f"{split}_test_pairs.parquet"
        df.to_parquet(out_path, index=False)
        test_indexes[split] = df

        by_type = df.pair_type.value_counts().to_dict()
        audit["splits"][split]["pairs"] = {
            "n_pairs": int(len(df)),
            "n_plots_with_pairs": int(df.plot_uid.nunique()),
            "by_pair_type": {str(k): int(v) for k, v in by_type.items()},
            "n_unique_images": int(
                pd.unique(pd.concat([df.image_stem_a, df.image_stem_b])).size),
            "delta_days": {
                "min": int(df.delta_days.min()) if len(df) else None,
                "max": int(df.delta_days.max()) if len(df) else None,
                "median": float(df.delta_days.median()) if len(df) else None,
            },
            "n_feature_ready": int(df.feature_ready.sum()),
            "n_weather_interval_ready": int(df.weather_interval_ready.sum()),
            "n_weather_past_ready": int(df.weather_past_ready.sum()),
            "n_weather_year_missing": int((~df.weather_year_available).sum()),
            "index_file": str(out_path),
            "index_sha256": sha256_file(out_path),
            "index_bytes": out_path.stat().st_size,
        }
        print(f"[{split}] {len(df)} pairs, "
              f"{int(df.feature_ready.sum())} feature-ready, "
              f"{int(df.weather_interval_ready.sum())} interval-ready, "
              f"{int(df.weather_past_ready.sum())} past-ready", flush=True)

    # ---------------- no-cross-split verification ------------------------
    checks: list[dict] = []

    def record(name: str, ok: bool, detail) -> None:
        checks.append({"id": name, "pass": bool(ok), "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)

    plot_split = plots.set_index("plot_uid")["split"]
    cross = 0
    for split, df in test_indexes.items():
        if df.empty:
            continue
        if not (plot_split.reindex(df.plot_uid).to_numpy() == split).all():
            cross += 1
    record("T1_pairs_inherit_official_split", cross == 0,
           "every test pair's plot keeps its official split attribution")

    stems_ok = True
    for split, df in test_indexes.items():
        if df.empty:
            continue
        if not all(s.startswith(p) for s, p in zip(df.image_stem_a, df.plot_uid)):
            stems_ok = False
        if not all(s.startswith(p) for s, p in zip(df.image_stem_b, df.plot_uid)):
            stems_ok = False
    record("T2_both_images_same_plot", stems_ok,
           "image stems are prefixed with their own plot_uid")

    # pair membership must be disjoint across the four test splits
    sets = [set(df.pair_id) for df in test_indexes.values() if len(df)]
    disjoint = all(not (a & b) for i, a in enumerate(sets) for b in sets[i + 1:])
    record("T3_test_pair_sets_disjoint", disjoint,
           "no pair_id appears in two official test splits")

    # the S3 index (train + validation) must not contain any of these pairs
    s3_index = pd.read_parquet(cfg["data"]["index"], columns=["pair_id", "split"])
    s3_ids = set(s3_index.pair_id)
    leaked = sum(len(set(df.pair_id) & s3_ids) for df in test_indexes.values() if len(df))
    record("T4_no_overlap_with_s3_index", leaked == 0,
           f"{leaked} test pair_ids also occur in the frozen S3 index")

    audit["verification"] = {
        "n_checks": len(checks),
        "n_pass": sum(1 for c in checks if c["pass"]),
        "all_pass": all(c["pass"] for c in checks),
        "checks": checks,
    }
    audit["weather"] = {
        "channels": list(WEATHER_CHANNELS),
        "units": {
            "T_mean_C": "degC", "T_max_C": "degC", "T_min_C": "degC",
            "RH_mean_pct": "%", "Rs_sum_MJ_m2_d": "MJ m-2 d-1",
        },
        "excluded": {
            "P_sum_raw": "unit unresolved (S1); precipitation integrals forbidden by S2",
            "VPD_mean_kPa": "derived channel, opt-in only",
        },
        "years_available": weather_years,
        "year_day_counts": weather_day_counts,
        "past_window_rule": "[t_a-28, ..., t_a-1] (strictly before t_a)",
        "interval_window_rule": "(t_a, t_b]",
        "gdd_definition": "GDD_d = max(T_mean_d - T_base, 0), T_base in {0,5} degC",
        "protocol_P_deployable": True,
        "protocol_R_deployable": False,
        "protocol_R_label": "retrospective / oracle interval conditioning",
        "structural_limit": (
            "S1 verified that all plots of one harvest_year share one weather series, "
            "so weather provides date/year-level environment variation only."
        ),
    }
    audit["scope_limits"] = [
        "No official test phenotype label is used for model selection, "
        "hyper-parameter choice or architecture change in this stage.",
        "No test metric is computed here; evaluation of a frozen model on these "
        "indexes is a separate, explicitly named S4B step.",
        "Protocol R (interval weather) is a retrospective diagnostic only and can "
        "never be reported as deployable forecasting ability.",
        "2019 (test_environment / test_genotype_environment) is an environment the "
        "training years never see; its weather is readable but that fact alone "
        "does not license any model change.",
    ]
    audit["elapsed_seconds"] = round(time.time() - t0, 1)

    out_json = Path("reports/s4b_test_data_audit.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(audit, fh, indent=2)
    print(f"wrote {out_json}")
    print(json.dumps({s: audit["splits"][s].get("pairs", {}) for s in s4b["test_splits"]},
                     indent=1)[:1200])
    return 0 if audit["verification"]["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
