#!/usr/bin/env python3
"""S1 task 1+2: audit the real FIP 1.0 parquet data and build a frozen index.

Reads the 13 parquet shards under external_data/fip1_parquet and emits:

  data/manifests/fip1_plots.csv          one row per plot-year (plot_uid) with
                                         split, year, lot, genotype, dates
  data/manifests/fip1_observations.csv   one row per (plot_uid, image) with the
                                         image date and the matched phenotype
                                         values/dates (per-trait nearest match)
  reports/s1_fip_audit.json              machine-readable audit numbers
  reports/s1_split_statistics.md         human-readable audit numbers

No schema is assumed: every column used below is checked against the parquet
schema first, and the script fails loudly if a field is missing.

Usage:
    python3 tools/s1_audit_fip.py \
        --parquet-dir external_data/fip1_parquet \
        --out-dir data/manifests \
        --report reports/s1_fip_audit.json
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# trait -> (values col, dates col, scalar metadata cols)
TRAITS = {
    "canopy_cover": ("canopy_cover_values", "canopy_cover_dates"),
    "height": ("height_values", "height_dates"),
    "spike_count": ("spike_count_values", "spike_count_dates"),
    "senescence": ("senescence_values", "senescence_dates"),
}
SCALAR_TRAITS = {
    "heading": ("heading_value", "heading_date"),
    "height_final": ("height_final_value", "height_final_date"),
    "yield": ("yield_value", "yield_date"),
    "yield_adjusted": ("yield_adjusted_value", "yield_adjusted_date"),
    "protein": ("protein_value", "protein_date"),
}
ENV_COLS = [
    "temperature_air_200cm_values", "temperature_air_200cm_dates", "temperature_air_200cm_times",
    "humidity_air_200cm_values", "humidity_air_200cm_dates", "humidity_air_200cm_times",
    "precipitation_200cm_values", "precipitation_200cm_dates", "precipitation_200cm_times",
    "irradiance_solar_200cm_values", "irradiance_solar_200cm_dates", "irradiance_solar_200cm_times",
    "temperature_air_10cm_values", "temperature_air_10cm_dates", "temperature_air_10cm_times",
    "temperature_soil_5cm_values", "temperature_soil_5cm_dates", "temperature_soil_5cm_times",
    "humidity_air_10cm_values", "humidity_air_10cm_dates", "humidity_air_10cm_times",
]

PLOT_COLS = ["plot_uid", "yearsite_uid", "crop_type", "experiment_number", "plot_number",
             "range", "row", "lot", "latitude", "longitude", "spatial_check",
             "sowing_date", "harvest_date", "harvest_year", "genotype_id",
             "images", "image_dates", "image_times",
             "inner_plot_images", "alignment_initial_date",
             "alignment_num_steps", "alignment_dates"]
for tr in TRAITS.values():
    PLOT_COLS += list(tr)
for tr in SCALAR_TRAITS.values():
    PLOT_COLS += list(tr)
PLOT_COLS += ENV_COLS
PLOT_COLS += [
    "canopy_cover_trait_id", "canopy_cover_trait_name", "canopy_cover_method_id",
    "canopy_cover_method_name", "canopy_cover_si_unit", "canopy_cover_responsible",
    "height_trait_id", "height_trait_name", "height_method_id", "height_method_name",
    "height_si_unit", "height_responsible",
    "spike_count_trait_id", "spike_count_trait_name", "spike_count_method_id",
    "spike_count_method_name", "spike_count_si_unit", "spike_count_responsible",
    "senescence_trait_id", "senescence_trait_name", "senescence_method_id",
    "senescence_method_name", "senescence_si_unit", "senescence_responsible",
    "yield_si_unit", "yield_method_name", "protein_si_unit", "protein_method_name",
]
PLOT_COLS = list(dict.fromkeys(PLOT_COLS))


def d2s(x):
    if x is None:
        return ""
    try:
        if isinstance(x, float) and math.isnan(x):
            return ""
    except Exception:
        pass
    if hasattr(x, "isoformat"):
        return x.isoformat()
    return str(x)


def as_list(x):
    if x is None:
        return []
    return list(x)


def f2s(x, nd=4):
    if x is None:
        return ""
    try:
        if isinstance(x, float) and math.isnan(x):
            return ""
    except Exception:
        pass
    if x is None:
        return ""
    return f"{float(x):.{nd}f}"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", default="external_data/fip1_parquet")
    ap.add_argument("--out-dir", default="data/manifests")
    ap.add_argument("--report", default="reports/s1_fip_audit.json")
    args = ap.parse_args()

    pqdir = Path(args.parquet_dir)
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    rep = Path(args.report)
    rep.parent.mkdir(parents=True, exist_ok=True)

    shards = sorted(pqdir.glob("*.parquet"))
    if not shards:
        print(f"no parquet shards in {pqdir}", file=sys.stderr)
        return 2

    audit = {
        "generated_at": None,
        "parquet_dir": str(pqdir),
        "shards": [{"file": s.name, "bytes": s.stat().st_size, "sha256": sha256_file(s)}
                   for s in shards],
        "split_of_shard": {},
    }

    plot_rows = []
    obs_rows = []
    split_counts = Counter()
    year_counts = Counter()
    split_year_counts = defaultdict(Counter)
    year_lot = defaultdict(Counter)
    genotype_sets = defaultdict(set)
    uid_split = {}
    plot_uid_all = []
    uid_year = {}
    yearsite_by_split = defaultdict(Counter)

    # metadata (units / methods) collected once
    trait_meta = {}
    all_inner_stats = []
    n_plots_len_mismatch = 0

    for shard in shards:
        split = shard.name.split("-")[0]
        audit["split_of_shard"][shard.name] = split
        table = pq.read_table(shard)
        missing = [c for c in PLOT_COLS if c not in table.column_names]
        if missing:
            raise SystemExit(f"{shard.name}: missing expected columns {missing}")
        n = table.num_rows
        split_counts[split] += n
        cols = {c: table.column(c).to_pylist() for c in PLOT_COLS}
        for i in range(n):
            puid = cols["plot_uid"][i]
            hy = cols["harvest_year"][i]
            uid_split[puid] = split
            uid_year[puid] = hy
            year_counts[hy] += 1
            split_year_counts[split][hy] += 1
            year_lot[hy][cols["lot"][i]] += 1
            genotype_sets[hy].add(cols["genotype_id"][i])
            yearsite_by_split[split][cols["yearsite_uid"][i]] += 1
            if puid not in plot_uid_all:
                plot_uid_all.append(puid)

            for tr in list(TRAITS) + list(SCALAR_TRAITS):
                md = trait_meta.setdefault(tr, {})
                for s in ("trait_id", "trait_name", "method_id", "method_name",
                          "si_unit", "responsible"):
                    k = f"{tr}_{s}"
                    if k in cols and md.get(s) in (None, "") and cols[k][i] not in (None, ""):
                        md[s] = cols[k][i]

            plot_rows.append({
                "plot_uid": puid,
                "split": split,
                "shard": shard.name,
                "yearsite_uid": cols["yearsite_uid"][i],
                "crop_type": cols["crop_type"][i],
                "experiment_number": cols["experiment_number"][i],
                "plot_number": cols["plot_number"][i],
                "range": cols["range"][i],
                "row": cols["row"][i],
                "lot": cols["lot"][i],
                "latitude": f2s(cols["latitude"][i], 6),
                "longitude": f2s(cols["longitude"][i], 6),
                "spatial_check": cols["spatial_check"][i],
                "sowing_date": d2s(cols["sowing_date"][i]),
                "harvest_date": d2s(cols["harvest_date"][i]),
                "harvest_year": hy,
                "genotype_id": cols["genotype_id"][i],
                "n_images": len(as_list(cols["inner_plot_images"][i])),
                "n_full_images": len(as_list(cols["images"][i])),
                "n_image_dates": len(as_list(cols["image_dates"][i])),
                "n_alignment": len(as_list(cols["alignment_num_steps"][i])),
                "alignment_initial_date": d2s(cols["alignment_initial_date"][i]),
                "first_image_date": d2s(as_list(cols["image_dates"][i])[0]) if as_list(cols["image_dates"][i]) else "",
                "last_image_date": d2s(as_list(cols["image_dates"][i])[-1]) if as_list(cols["image_dates"][i]) else "",
                "cc_n": len(as_list(cols["canopy_cover_values"][i])),
                "ph_n": len(as_list(cols["height_values"][i])),
                "whc_n": len(as_list(cols["spike_count_values"][i])),
                "sen_n": len(as_list(cols["senescence_values"][i])),
                "env_records": len(as_list(cols["temperature_air_200cm_values"][i])),
                "yield_value": f2s(cols["yield_value"][i], 3),
                "yield_adjusted_value": f2s(cols["yield_adjusted_value"][i], 3),
                "protein_value": f2s(cols["protein_value"][i], 3),
                "heading_value": f2s(cols["heading_value"][i], 2),
                "height_final_value": f2s(cols["height_final_value"][i], 3),
            })

            # observations: one row per inner-plot image.
            # IMPORTANT: len(inner_plot_images) is not always equal to
            # len(image_dates) (e.g. 2019 test_environment rows). Join by the
            # filename stem embedded in both columns instead of by position.
            imgs = as_list(cols["inner_plot_images"][i])
            orig_imgs = as_list(cols["images"][i])
            idates = [x for x in as_list(cols["image_dates"][i]) if x is not None]
            itimes = as_list(cols["image_times"][i])
            cc_v, cc_d = as_list(cols["canopy_cover_values"][i]), \
                [x for x in as_list(cols["canopy_cover_dates"][i]) if x is not None]
            ph_v, ph_d = as_list(cols["height_values"][i]), \
                [x for x in as_list(cols["height_dates"][i]) if x is not None]
            wh_v, wh_d = as_list(cols["spike_count_values"][i]), \
                [x for x in as_list(cols["spike_count_dates"][i]) if x is not None]
            sn_v, sn_d = as_list(cols["senescence_values"][i]), \
                [x for x in as_list(cols["senescence_dates"][i]) if x is not None]
            for name, arr in (("cc", cc_d), ("ph", ph_d), ("whc", wh_d), ("sen", sn_d)):
                if arr != sorted(arr):
                    audit.setdefault("warnings", []).append(
                        f"{puid}: {name}_dates not sorted; nearest() assumes sorted")
            for name, vals, dts in (("cc", cc_v, cc_d), ("ph", ph_v, ph_d),
                                     ("whc", wh_v, wh_d), ("sen", sn_v, sn_d)):
                if len(vals) != len(dts):
                    audit.setdefault("warnings", []).append(
                        f"{puid}: {name}_values/dates length {len(vals)}!={len(dts)}")
            stem_to_idx = {}
            for j, op in enumerate(orig_imgs):
                if op is None:
                    continue
                stem = op.rsplit("/", 1)[-1].rsplit(".", 1)[0]
                stem_to_idx[stem] = j

            def nearest(vals, dates, target):
                if not dates or target is None:
                    return ("", None, "")
                j = bisect.bisect_left(dates, target)
                cand = [x for x in (j - 1, j) if 0 <= x < len(dates)]
                best_j, best_abs = None, None
                for c in cand:
                    delta = (dates[c] - target).days  # signed: label_date - image_date
                    if best_abs is None or abs(delta) < best_abs:
                        best_abs, best_j = abs(delta), c
                if best_j is None or best_j >= len(vals):
                    return ("", None, "")
                return (f2s(vals[best_j]), (dates[best_j] - target).days,
                        dates[best_j].isoformat())

            n_missing_inner = 0
            n_orphan_inner = 0
            n_date_mismatch = 0
            matched_stems = set()
            for k, img in enumerate(imgs):
                stem = img.rsplit("/", 1)[-1].rsplit(".", 1)[0] if img else ""
                j = stem_to_idx.get(stem)
                if j is None:
                    n_orphan_inner += 1
                    continue
                matched_stems.add(stem)
                dt = idates[j] if j < len(idates) else None
                tm = itimes[j] if j < len(itimes) else None
                fdate = d2s(dt)
                if fdate and dt is not None and stem.rsplit("_", 2)[-2] != dt.strftime("%Y%m%d"):
                    n_date_mismatch += 1
                cc_val, cc_lag, cc_date = nearest(cc_v, cc_d, dt)
                ph_val, ph_lag, ph_date = nearest(ph_v, ph_d, dt)
                wh_val, wh_lag, wh_date = nearest(wh_v, wh_d, dt)
                sn_val, sn_lag, sn_date = nearest(sn_v, sn_d, dt)
                obs_rows.append({
                    "plot_uid": puid,
                    "split": split,
                    "harvest_year": hy,
                    "obs_index": j,
                    "orig_index": j,
                    "image_date": fdate,
                    "image_time": d2s(tm),
                    "image_filename_stem": stem,
                    "inner_plot_image": img,
                    "original_image": orig_imgs[j] if j < len(orig_imgs) else "",
                    "cc_value": cc_val, "cc_lag_days": cc_lag if cc_lag is not None else "",
                    "cc_date": cc_date,
                    "ph_value": ph_val, "ph_lag_days": ph_lag if ph_lag is not None else "",
                    "ph_date": ph_date,
                    "whc_value": wh_val, "whc_lag_days": wh_lag if wh_lag is not None else "",
                    "whc_date": wh_date,
                    "sen_value": sn_val, "sen_lag_days": sn_lag if sn_lag is not None else "",
                    "sen_date": sn_date,
                })
            n_missing_inner += len([s for s in stem_to_idx if s not in matched_stems])
            inner_stats = {
                "n_full_images": len(orig_imgs),
                "n_inner_images": len(imgs),
                "n_missing_inner": n_missing_inner,
                "n_orphan_inner": n_orphan_inner,
                "n_filename_date_mismatch": n_date_mismatch,
            }
            all_inner_stats.append(inner_stats)
            if len(imgs) != len(orig_imgs):
                n_plots_len_mismatch += 1

    # ---- write CSVs
    audit["inner_plot_image_integrity"] = {
        "n_plots": len(all_inner_stats),
        "n_plots_with_length_mismatch": n_plots_len_mismatch,
        "n_full_images": int(sum(s["n_full_images"] for s in all_inner_stats)),
        "n_inner_images": int(sum(s["n_inner_images"] for s in all_inner_stats)),
        "n_full_images_without_inner": int(sum(s["n_missing_inner"] for s in all_inner_stats)),
        "n_orphan_inner_images": int(sum(s["n_orphan_inner"] for s in all_inner_stats)),
        "n_filename_date_mismatch": int(sum(s["n_filename_date_mismatch"] for s in all_inner_stats)),
    }
    def write_csv(path, rows, fields):
        import csv
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    plots_csv = outdir / "fip1_plots.csv"
    obs_csv = outdir / "fip1_observations.csv"
    write_csv(plots_csv, plot_rows, list(plot_rows[0].keys()))
    write_csv(obs_csv, obs_rows, list(obs_rows[0].keys()))

    # ---- audit numbers
    import pandas as pd
    odf = pd.read_csv(obs_csv)
    pdf = pd.read_csv(plots_csv)

    audit["generated_at"] = __import__("time").strftime("%Y-%m-%dT%H:%M:%S%z")
    audit["n_plot_rows"] = len(pdf)
    audit["n_unique_plot_uid"] = pdf.plot_uid.nunique()
    audit["split_counts"] = dict(split_counts)
    audit["split_year_counts"] = {k: dict(v) for k, v in split_year_counts.items()}
    audit["year_counts"] = dict(sorted(year_counts.items()))
    audit["year_lot"] = {str(k): {str(kk): vv for kk, vv in v.items()}
                         for k, v in sorted(year_lot.items())}
    audit["n_observations"] = len(odf)
    audit["trait_metadata"] = trait_meta
    audit["genotype_overlap"] = {
        "n_genotypes_by_year": {str(k): len(v) for k, v in sorted(genotype_sets.items())},
        "pairwise_shared_genotypes": {},
    }
    years_sorted = sorted(genotype_sets)
    for i, y1 in enumerate(years_sorted):
        for y2 in years_sorted[i + 1:]:
            inter = genotype_sets[y1] & genotype_sets[y2]
            audit["genotype_overlap"]["pairwise_shared_genotypes"][f"{y1}_{y2}"] = len(inter)

    # image-date vs phenotype-date offsets
    for trait, col in (("cc", "cc_lag_days"), ("ph", "ph_lag_days"),
                       ("whc", "whc_lag_days"), ("sen", "sen_lag_days")):
        lag = pd.to_numeric(odf[col], errors="coerce").dropna()  # signed
        audit[f"{trait}_lag_days"] = {
            "n_matched": int(lag.size),
            "n_missing": int(len(odf) - lag.size),
            "lag_median_signed": float(lag.median()) if lag.size else None,
            "lag_mean_signed": float(lag.mean()) if lag.size else None,
            "frac_lag_0": float((lag == 0).mean()) if lag.size else None,
            "frac_abs_le_3": float((lag.abs() <= 3).mean()) if lag.size else None,
            "frac_abs_le_7": float((lag.abs() <= 7).mean()) if lag.size else None,
            "frac_positive": float((lag > 0).mean()) if lag.size else None,
            "frac_negative": float((lag < 0).mean()) if lag.size else None,
        }
        vals = pd.to_numeric(odf[col.replace("_lag_days", "_value")], errors="coerce")
        vstats = {
            "n_valid": int(vals.notna().sum()),
            "n_zero": int((vals == 0).sum()),
            "min": float(vals.min()) if vals.notna().any() else None,
            "max": float(vals.max()) if vals.notna().any() else None,
            "median": float(vals.median()) if vals.notna().any() else None,
        }
        for thr in (3, 7):
            m = lag.abs() <= thr
            sub = vals[vals.index.isin(lag[m].index)]
            vstats["within_%dd" % thr] = {
                "n_valid": int(sub.notna().sum()),
                "n_zero": int((sub == 0).sum()),
                "min": float(sub.min()) if sub.notna().any() else None,
                "max": float(sub.max()) if sub.notna().any() else None,
                "median": float(sub.median()) if sub.notna().any() else None,
            }
        audit[f"{trait}_value_stats"] = vstats
    # per-year phenotype availability
    per_year = {}
    for y in sorted(odf.harvest_year.dropna().unique()):
        sub = odf[odf.harvest_year == y]
        per_year[int(y)] = {
            "n_obs": int(len(sub)),
            "cc_valid": int(pd.to_numeric(sub.cc_value, errors="coerce").notna().sum()),
            "ph_valid": int(pd.to_numeric(sub.ph_value, errors="coerce").notna().sum()),
            "whc_valid": int(pd.to_numeric(sub.whc_value, errors="coerce").notna().sum()),
            "sen_valid": int(pd.to_numeric(sub.sen_value, errors="coerce").notna().sum()),
        }
    audit["per_year_observation_traits"] = per_year

    with open(rep, "w") as fh:
        json.dump(audit, fh, indent=2)
    print(json.dumps({k: v for k, v in audit.items()
                      if k in ("n_plot_rows", "n_unique_plot_uid", "split_counts",
                               "year_counts", "n_observations")}, indent=2))
    print(f"wrote {plots_csv} ({len(plot_rows)} rows)")
    print(f"wrote {obs_csv} ({len(obs_rows)} rows)")
    print(f"wrote {rep}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
