#!/usr/bin/env python3
"""S1: audit the raw FIP weather time series and derive a daily table sample.

The parquet stores hourly records per plot-year:
    temperature_air_200cm_{values,dates,times}
    humidity_air_200cm_{values,dates,times}
    irradiance_solar_200cm_{values,dates,times}
    precipitation_200cm_{values,dates,times}
plus optional 10 cm air / 5 cm soil sensors.

This script:
  * verifies hourly cadence, coverage, duplicates and gaps per year;
  * checks whether the weather arrays are identical across plots of a year;
  * derives a daily table for one sample year/plot:
        T_mean/T_max/T_min (degC), RH_mean (%), Rs_sum (MJ m-2 d-1),
        P_sum (mm, raw units), VPD_mean (kPa)
    with explicit aggregation formulas recorded in the report;
  * does NOT decide the physical unit of precipitation (documented as open).

Usage:
    python3 tools/s1_weather_audit.py \
        --parquet-dir external_data/fip1_parquet \
        --report reports/s1_weather_schema.json \
        --daily-out data/manifests/weather_daily_sample_2016.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

SENSORS = {
    "temperature_air_200cm": "degC",
    "humidity_air_200cm": "%",
    "irradiance_solar_200cm": "W m-2 (hourly mean; inferred)",
    "precipitation_200cm": "raw (unit not stated in data card)",
}
OPTIONAL = ["temperature_air_10cm", "humidity_air_10cm", "temperature_soil_5cm"]


def arr_digest(a) -> str:
    a = np.asarray(a, dtype=np.float64)
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", default="external_data/fip1_parquet")
    ap.add_argument("--report", default="reports/s1_weather_schema.json")
    ap.add_argument("--daily-out", default="data/manifests/weather_daily_sample_2016.csv")
    ap.add_argument("--sample-year", type=int, default=2016)
    ap.add_argument("--n-per-year", type=int, default=3,
                    help="plots per year used to test cross-plot equality")
    args = ap.parse_args()

    shards = sorted(Path(args.parquet_dir).glob("*.parquet"))
    base_cols = ["plot_uid", "harvest_year", "sowing_date", "harvest_date"]
    for s in SENSORS:
        base_cols += [f"{s}_values", f"{s}_dates", f"{s}_times"]
    for s in OPTIONAL:
        base_cols += [f"{s}_values"]

    report = {"sensors": SENSORS, "optional_sensors": OPTIONAL, "years": {}}
    daily_by_year = {}
    seen_year_plots = Counter()

    for shard in shards:
        table = pq.read_table(shard, columns=[c for c in base_cols if c in
                                              pq.ParquetFile(shard).schema_arrow.names])
        rows = table.to_pylist()
        for r in rows:
            y = int(r["harvest_year"])
            if seen_year_plots[y] >= args.n_per_year:
                continue
            seen_year_plots[y] += 1
            info = report["years"].setdefault(str(y), {
                "n_plots_checked": 0, "sensors": {},
            })
            info["n_plots_checked"] += 1
            for s in SENSORS:
                dates = r[f"{s}_dates"]
                times = r[f"{s}_times"]
                vals = r[f"{s}_values"]
                days = Counter(dates)
                day_counts = sorted(days.values())
                gaps = []
                for i in range(len(dates) - 1):
                    d0, d1 = dates[i], dates[i + 1]
                    if d1 > d0:
                        delta = (d1 - d0).days
                        if delta > 1:
                            gaps.append({"from": str(d0), "to": str(d1),
                                         "missing_days": delta - 1})
                rec = {
                    "n_records": len(vals),
                    "n_days": len(days),
                    "records_per_day_min": day_counts[0] if day_counts else 0,
                    "records_per_day_max": day_counts[-1] if day_counts else 0,
                    "date_first": str(dates[0]) if dates else None,
                    "date_last": str(dates[-1]) if dates else None,
                    "value_min": float(np.nanmin(vals)) if len(vals) else None,
                    "value_max": float(np.nanmax(vals)) if len(vals) else None,
                    "value_mean": float(np.nanmean(vals)) if len(vals) else None,
                    "n_nan": int(np.isnan(np.asarray(vals, dtype=float)).sum()),
                    "digitizing_gaps": gaps,
                    "array_digest": arr_digest(vals),
                }
                # cross-plot equality check
                prev = info["sensors"].get(s)
                if prev is None:
                    info["sensors"][s] = rec
                else:
                    prev.setdefault("digests", [prev["array_digest"]])
                    prev["digests"].append(rec["array_digest"])
                    prev["identical_across_checked_plots"] = (
                        len(set(prev["digests"])) == 1)

            if seen_year_plots[y] == 1:
                daily_by_year[y] = build_daily(r)

    # finalize cross-plot equality
    for y, info in report["years"].items():
        for s, rec in info["sensors"].items():
            rec.pop("digests", None)
            rec.setdefault("identical_across_checked_plots", True)

    if daily_by_year:
        import csv
        all_rows = []
        for y in sorted(daily_by_year):
            for row in daily_by_year[y]:
                all_rows.append({"harvest_year": y, **row})
        Path(args.daily_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.daily_out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        report["daily_aggregation"] = {
            "output": args.daily_out,
            "years": sorted(daily_by_year),
            "n_days_by_year": {str(y): len(v) for y, v in sorted(daily_by_year.items())},
            "formulas": {
                "T_mean": "mean of hourly temperature_air_200cm over the calendar day",
                "T_max": "max of hourly temperature_air_200cm",
                "T_min": "min of hourly temperature_air_200cm",
                "RH_mean": "mean of hourly humidity_air_200cm",
                "Rs_sum": "sum(hourly irradiance W m-2) * 3600 s / 1e6 -> MJ m-2 d-1",
                "P_sum": "sum(hourly precipitation_200cm) -> raw units, unit unresolved",
                "VPD_mean": "mean(es(T) * (1 - RH/100)), es=0.6108*exp(17.27T/(T+237.3)) kPa",
            },
            "precipitation_unit_status": "unresolved; reported as raw sensor units",
        }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(
        {y: {"n_plots": info["n_plots_checked"],
             "sensors": {s: {k: rec.get(k) for k in
                             ("n_records", "n_days", "records_per_day_min",
                              "records_per_day_max", "n_nan",
                              "identical_across_checked_plots")}
                         for s in ("temperature_air_200cm", "precipitation_200cm")}}
         for y, info in report["years"].items()}, indent=1))
    print(f"wrote {args.report}; daily rows: "
          f"{sum(len(v) for v in daily_by_year.values())} across "
          f"{len(daily_by_year)} years")
    return 0


def build_daily(r) -> list[dict]:
    keys = ["temperature_air_200cm", "humidity_air_200cm",
            "irradiance_solar_200cm", "precipitation_200cm"]
    by_day = {}
    for k in keys:
        dates = r[f"{k}_dates"]
        vals = np.asarray(r[f"{k}_values"], dtype=float)
        for d, v in zip(dates, vals):
            by_day.setdefault(d, {}).setdefault(k, []).append(v)
    rows = []
    for d in sorted(by_day):
        b = by_day[d]
        t = np.asarray(b["temperature_air_200cm"], dtype=float)
        rh = np.asarray(b["humidity_air_200cm"], dtype=float)
        rs = np.asarray(b["irradiance_solar_200cm"], dtype=float)
        p = np.asarray(b["precipitation_200cm"], dtype=float)
        es = 0.6108 * np.exp(17.27 * t / (t + 237.3))
        vpd = es * (1.0 - rh / 100.0)
        rows.append({
            "date": d.isoformat(),
            "T_mean_C": round(float(np.nanmean(t)), 4),
            "T_max_C": round(float(np.nanmax(t)), 4),
            "T_min_C": round(float(np.nanmin(t)), 4),
            "RH_mean_pct": round(float(np.nanmean(rh)), 4),
            "Rs_sum_MJ_m2_d": round(float(np.nansum(rs)) * 3600.0 / 1e6, 4),
            "P_sum_raw": round(float(np.nansum(p)), 4),
            "VPD_mean_kPa": round(float(np.nanmean(vpd)), 4),
            "n_hours_T": int(np.isfinite(t).sum()),
            "n_hours_RH": int(np.isfinite(rh).sum()),
            "n_hours_Rs": int(np.isfinite(rs).sum()),
            "n_hours_P": int(np.isfinite(p).sum()),
        })
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
