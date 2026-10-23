#!/usr/bin/env python3
"""S1 acceptance check: given a real plot_uid and a target image date, print
the image, the label measurements with their real dates and lags, and the
weather records in the left-open/right-closed window (t_a, t_b].

Usage:
    python3 tools/s1_lookup_plot.py --plot FPWW0120667 --date 2016-05-17 --days 7
    python3 tools/s1_lookup_plot.py --plot FPWW0120667 --date 2016-05-17 --json
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

TRAITS = {
    "canopy_cover": ("canopy_cover_values", "canopy_cover_dates"),
    "height": ("height_values", "height_dates"),
    "spike_count": ("spike_count_values", "spike_count_dates"),
    "senescence": ("senescence_values", "senescence_dates"),
}
WEATHER = {
    "temperature_air_200cm": "degC",
    "humidity_air_200cm": "%",
    "irradiance_solar_200cm": "W m-2",
    "precipitation_200cm": "raw",
}


def find_row(parquet_dir: Path, plot_uid: str):
    for shard in sorted(parquet_dir.glob("*.parquet")):
        t = pq.read_table(shard, columns=["plot_uid"])
        uids = t.column("plot_uid").to_pylist()
        if plot_uid in uids:
            full = pq.read_table(shard).to_pylist()
            return shard.name.split("-")[0], full[uids.index(plot_uid)]
    raise SystemExit(f"{plot_uid} not found in {parquet_dir}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", default="external_data/fip1_parquet")
    ap.add_argument("--plot", required=True)
    ap.add_argument("--date", required=True, help="target image date YYYY-MM-DD")
    ap.add_argument("--days", type=int, default=7,
                    help="weather window is (date - days, date]")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    target = datetime.strptime(args.date, "%Y-%m-%d").date()
    win_a = target - timedelta(days=args.days)
    split, r = find_row(Path(args.parquet_dir), args.plot)

    # image record
    stems = [p.rsplit("/", 1)[-1].rsplit(".", 1)[0] for p in r["images"]]
    want = f"{args.plot}_RGB1_{target.strftime('%Y%m%d')}"
    match = [i for i, s in enumerate(stems) if s.startswith(want)]
    img = None
    if match:
        i = match[0]
        img = {
            "index": i,
            "image_date": str(r["image_dates"][i]),
            "image_time": str(r["image_times"][i]),
            "original_image": r["images"][i],
        }
        stem = stems[i]
        inner = [p for p in r["inner_plot_images"] if p.rsplit("/", 1)[-1].startswith(stem)]
        img["inner_plot_image"] = inner[0] if inner else None
        img["has_inner_plot_image"] = bool(inner)

    labels = {}
    for tr, (vc, dc) in TRAITS.items():
        vals = r[vc] or []
        dates = [d for d in (r[dc] or []) if d is not None]
        pairs = list(zip(dates, vals))
        if not pairs:
            labels[tr] = {"n_measurements": 0, "nearest": None}
            continue
        d, v = min(pairs, key=lambda p: abs((p[0] - target).days))
        labels[tr] = {
            "n_measurements": len(pairs),
            "first_date": str(pairs[0][0]),
            "last_date": str(pairs[-1][0]),
            "nearest_date": str(d),
            "nearest_value": float(v),
            "lag_days": (d - target).days,
        }

    weather = {}
    for k, unit in WEATHER.items():
        dates = r[f"{k}_dates"]
        times = r[f"{k}_times"]
        vals = np.asarray(r[f"{k}_values"], dtype=float)
        mask = [i for i, d in enumerate(dates) if win_a < d <= target]
        wv = vals[mask] if mask else np.array([])
        per_day = {}
        for i in mask:
            per_day.setdefault(str(dates[i]), []).append(float(vals[i]))
        weather[k] = {
            "unit": unit,
            "window": f"({win_a.isoformat()}, {target.isoformat()}]",
            "n_hours": int(len(wv)),
            "n_days": len(per_day),
            "hours_per_day": {d: len(v) for d, v in sorted(per_day.items())},
            "sum": float(np.nansum(wv)) if len(wv) else None,
            "mean": float(np.nanmean(wv)) if len(wv) else None,
            "min": float(np.nanmin(wv)) if len(wv) else None,
            "max": float(np.nanmax(wv)) if len(wv) else None,
            "daily_sum": {d: round(float(np.nansum(v)), 4)
                          for d, v in sorted(per_day.items())},
        }

    out = {
        "plot_uid": args.plot,
        "official_split": split,
        "harvest_year": int(r["harvest_year"]),
        "yearsite_uid": r["yearsite_uid"],
        "genotype_id": r["genotype_id"],
        "sowing_date": str(r["sowing_date"]),
        "harvest_date": str(r["harvest_date"]),
        "n_images_total": len(r["images"]),
        "n_inner_images_total": len(r["inner_plot_images"]),
        "query": {"target_date": target.isoformat(), "weather_window_days": args.days},
        "image": img,
        "labels": labels,
        "weather": weather,
    }
    if args.json:
        print(json.dumps(out, indent=2, default=str))
    else:
        print(f"plot_uid      : {out['plot_uid']}  ({split}, {out['harvest_year']})")
        print(f"genotype_id   : {out['genotype_id']}")
        print(f"sowing/harvest: {out['sowing_date']} -> {out['harvest_date']}")
        print(f"images        : {out['n_images_total']} full / "
              f"{out['n_inner_images_total']} inner")
        if img:
            print(f"image @ {target}: #{img['index']} {img['image_date']} "
                  f"{img['image_time']} inner={img['has_inner_plot_image']}")
            print(f"  path: {img['inner_plot_image']}")
        else:
            print(f"image @ {target}: NOT FOUND")
        print("labels (nearest measurement to the image date):")
        for tr, v in labels.items():
            if not v["n_measurements"]:
                print(f"  {tr:14s}: no measurements")
            else:
                n = v
                print(f"  {tr:14s}: {n['nearest_value']:.4f} on {n['nearest_date']} "
                      f"(lag {n['lag_days']:+d} d; {n['n_measurements']} measurements "
                      f"{n['first_date']}..{n['last_date']})")
        print(f"weather window (t_a, t_b] = ({win_a}, {target}]:")
        for k, w in weather.items():
            print(f"  {k:26s} n_hours={w['n_hours']:3d} n_days={w['n_days']} "
                  f"sum={w['sum']:.3f} mean={w['mean']:.3f} [{w['unit']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
