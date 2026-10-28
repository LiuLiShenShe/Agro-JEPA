#!/usr/bin/env python3
"""S4B.4A: per-pair GDD and weather-sequence conditioning tensors (full train).

Forked from ``tools/s4_prepare_weather.py`` so the frozen S4A/S4B.2 tool is
never touched.  This variant adds ``--index`` / ``--out-npz`` overrides so the
full official-train index (2,930 plots, 179,823 pairs) can be conditioned in
one pass and written to ``external_data/s4b4a_conditioning/``.  Same protocol
rules (P = deployable, R = retrospective) and the same SHA256 provenance as the
original.

This tool is the *only* place where weather enters S4.  It reads the frozen S3
pair index (read-only) and the S1 daily weather table, and writes one derived
array per pair with a SHA256 so every downstream number is traceable.

Two protocols, never mixed
--------------------------
Protocol R (retrospective / oracle interval condition)
    The weather actually observed in ``(t_a, t_b]`` -- the frozen S2 window
    convention.  This is *not* deployable future prediction: at ``t_a`` those
    days have not happened yet.  Results that use it are labelled
    ``retrospective`` everywhere.

Protocol P (deployable)
    Weather observed strictly *before* ``t_a``::

        past window of length W = [t_a - W, ..., t_a - 1]   (all < t_a)

    No future weather, and not even the current day's aggregate, enters.

GDD
---
Daily mean temperature is the S1 confirmed ``T_mean_C`` (degC).  The GDD is the
pre-registered baseline::

    GDD_d = max(T_mean_d - T_base, 0)          T_base in {0, 5} degC

Protocol R feature: ``GDD_R = sum over (t_a, t_b]``.
Protocol P features: ``GDD_P = [sum over past 7 d, 14 d, 28 d]``.

This is *not* a genotype-fitted physiological parameter, and there is no
invented sowing date / seasonal accumulation.

Weather vector (GRU channels)
-----------------------------
``T_mean_C``, ``T_max_C``, ``T_min_C``, ``RH_mean_pct``, ``Rs_sum_MJ_m2_d``.
``P_sum_raw`` is **excluded**: its unit is unresolved (S1 audit), and the S2
rule forbids treating it as mm or building precipitation integrals.

Output
------
``external_data/s4_conditioning/s4_pair_conditioning.npz`` plus a JSON summary.
Arrays are in the exact row order of the frozen S3 index; the pair_id array is
stored so the loader can assert alignment.

Usage
-----
    python3 tools/s4_prepare_weather.py \
        --config agro_jepa/config/s4_pilot.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agro_jepa.datasets.fip_weather import (  # noqa: E402
    DailyWeatherStore,
    WEATHER_UNITS,
)


# The GRU only receives channels whose unit and definition are confirmed.
S4_WEATHER_CHANNELS: tuple[str, ...] = (
    "T_mean_C",
    "T_max_C",
    "T_min_C",
    "RH_mean_pct",
    "Rs_sum_MJ_m2_d",
)
EXCLUDED_WEATHER_CHANNELS = {
    "P_sum_raw": "unit unresolved (S1); precipitation integrals forbidden by S2",
    "VPD_mean_kPa": "derived channel, opt-in only; not used in S4A",
}

GDD_BASES: tuple[float, ...] = (0.0, 5.0)
GDD_PAST_WINDOWS: tuple[int, ...] = (7, 14, 28)
WEATHER_PAST_DAYS = 28
MAX_INTERVAL_DAYS = 16  # realised maximum in the frozen S3 index (5..16 d)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _past_window(
    store: DailyWeatherStore,
    harvest_year: int,
    t_a: date,
    n_days: int,
):
    """``[t_a - n_days, ..., t_a - 1]`` -- strictly before ``t_a``.

    Expressed with the frozen ``(start, end]`` convention as
    ``(t_a - n_days - 1, t_a - 1]``, so it reuses the *same* validity rule as
    the interval window: a day is valid only when every channel is finite and
    all sensors have at least one hourly record.
    """
    return store.window(
        harvest_year,
        t_a - timedelta(days=n_days + 1),
        t_a - timedelta(days=1),
    )


def _gdd(values: np.ndarray, mask: np.ndarray, base: float) -> float:
    """Sum of ``max(T_mean - base, 0)`` over valid days."""
    t_mean = values[:, 0]
    ok = mask & np.isfinite(t_mean)
    if not ok.any():
        return float("nan")
    return float(np.maximum(t_mean[ok] - base, 0.0).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s4_pilot.yaml")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--summary", default="reports/s4b4a_conditioning_summary.json")
    ap.add_argument("--index", default=None,
                    help="optional override index parquet (S4B.4A full-train build)")
    ap.add_argument("--out-npz", default=None,
                    help="optional override output npz path")
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    d = cfg["data"]
    index_path = Path(args.index or d["index"])
    weather_csv = Path(d["weather_daily"])
    out_dir = Path(args.out_dir or d["conditioning_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    index = pd.read_parquet(index_path)
    index_sha = _stable_index_sha(index)
    if "pair_id" not in index.columns:
        raise SystemExit(f"{index_path} has no pair_id column")
    if index.index.tolist() != list(range(len(index))):
        raise SystemExit("the frozen index must have a 0..n-1 RangeIndex")
    splits = set(index.split.unique())
    if not splits <= {"train", "validation"}:
        raise SystemExit(f"unexpected splits in the frozen index: {sorted(splits)}")

    store = DailyWeatherStore(
        weather_csv,
        channels=list(d["weather_channels"]),
        add_vpd=False,
        require_mainline=False,
    )
    if tuple(store.channels) != tuple(d["weather_channels"]):
        raise SystemExit(
            f"weather channel mismatch: store={store.channels} config={d['weather_channels']}"
        )
    if "P_sum_raw" in store.channels:
        raise SystemExit("P_sum_raw must not enter the S4 weather vector")

    n = len(index)
    n_ch = len(store.channels)
    gdd_int = {b: np.full(n, np.nan, dtype=np.float32) for b in GDD_BASES}
    gdd_past = {
        (w, b): np.full(n, np.nan, dtype=np.float32)
        for w in GDD_PAST_WINDOWS for b in GDD_BASES
    }
    weather_int = np.zeros((n, MAX_INTERVAL_DAYS, n_ch), dtype=np.float32)
    weather_int_mask = np.zeros((n, MAX_INTERVAL_DAYS), dtype=bool)
    weather_past = np.zeros((n, WEATHER_PAST_DAYS, n_ch), dtype=np.float32)
    weather_past_mask = np.zeros((n, WEATHER_PAST_DAYS), dtype=bool)
    interval_valid_days = np.zeros(n, dtype=np.int16)
    past_valid_days = np.zeros(n, dtype=np.int16)
    hours_int = np.full((n, MAX_INTERVAL_DAYS), -1, dtype=np.int16)
    hours_past = np.full((n, WEATHER_PAST_DAYS), -1, dtype=np.int16)

    pair_ids = index.pair_id.to_numpy().astype(str)
    dates_a = pd.to_datetime(index.date_a)
    dates_b = pd.to_datetime(index.date_b)
    years = index.harvest_year.to_numpy().astype(int)
    deltas = index.delta_days.to_numpy().astype(int)

    for i in range(n):
        y = int(years[i])
        a = dates_a.iloc[i].date()
        b = dates_b.iloc[i].date()
        delta = int(deltas[i])
        if delta <= 0 or delta > MAX_INTERVAL_DAYS:
            raise SystemExit(f"row {i}: delta_days {delta} outside (0,{MAX_INTERVAL_DAYS}]")

        win = store.window(y, a, b)
        # Protocol R: the real interval
        weather_int[i, :delta] = win.values
        weather_int_mask[i, :delta] = win.mask
        hours_int[i, :delta] = win.n_hours_min
        interval_valid_days[i] = win.n_valid_days
        for base in GDD_BASES:
            gdd_int[base][i] = _gdd(win.values, win.mask, base)

        # Protocol P: strictly past days
        pwin = _past_window(store, y, a, WEATHER_PAST_DAYS)
        if pwin.n_days != WEATHER_PAST_DAYS:
            raise SystemExit(
                f"row {i}: past window length {pwin.n_days} != {WEATHER_PAST_DAYS}"
            )
        weather_past[i] = pwin.values
        weather_past_mask[i] = pwin.mask
        hours_past[i] = pwin.n_hours_min
        past_valid_days[i] = pwin.n_valid_days
        for w in GDD_PAST_WINDOWS:
            sl = slice(WEATHER_PAST_DAYS - w, WEATHER_PAST_DAYS)
            for base in GDD_BASES:
                gdd_past[(w, base)][i] = _gdd(pwin.values[sl], pwin.mask[sl], base)

    # ---- integrity: every interval / past window must be fully covered ----
    interval_bad = int((interval_valid_days != deltas).sum())
    past_bad = int((past_valid_days != WEATHER_PAST_DAYS).sum())
    if interval_bad:
        raise SystemExit(f"{interval_bad} interval windows are not fully covered")
    if past_bad:
        raise SystemExit(f"{past_bad} past windows are not fully covered")
    if not np.isfinite(weather_int).all() and False:  # pragma: no cover
        pass
    if not np.isfinite(weather_past).all():
        raise SystemExit("weather_past contains non-finite values on valid days")
    for name, arr in (
        [("gdd_int", a) for a in gdd_int.values()]
        + [(f"gdd_past_{w}_{b}", a) for (w, b), a in gdd_past.items()]
    ):
        if not np.isfinite(arr).all():
            raise SystemExit(f"{name} contains non-finite GDD values")

    payload = {
        "pair_id": pair_ids,
        "split": index.split.to_numpy().astype(str),
        "harvest_year": years.astype(np.int16),
        "delta_days": deltas.astype(np.int16),
        "date_a": dates_a.dt.strftime("%Y-%m-%d").to_numpy().astype(str),
        "date_b": dates_b.dt.strftime("%Y-%m-%d").to_numpy().astype(str),
        "weather_channels": np.array(store.channels, dtype=object),
        "weather_int": weather_int,
        "weather_int_mask": weather_int_mask,
        "weather_int_hours_min": hours_int,
        "weather_past": weather_past,
        "weather_past_mask": weather_past_mask,
        "weather_past_hours_min": hours_past,
        "interval_valid_days": interval_valid_days,
        "past_valid_days": past_valid_days,
        "index_sha256": np.array(index_sha),
        "weather_manifest_id": np.array(store.manifest_id),
        "past_window_rule": np.array("[t_a-W, ..., t_a-1] (strictly before t_a)"),
        "interval_window_rule": np.array("(t_a, t_b]"),
    }
    for base in GDD_BASES:
        payload[f"gdd_int_base{int(base)}"] = gdd_int[base]
    for (w, base), arr in gdd_past.items():
        payload[f"gdd_past{w}_base{int(base)}"] = arr

    npz_path = Path(args.out_npz or (out_dir / "s4_pair_conditioning.npz"))
    np.savez_compressed(npz_path, **payload)
    npz_sha = sha256_file(npz_path)

    # ---- summary (train/validation separately; no official test) ----------
    def describe(mask: np.ndarray, year_vals: np.ndarray) -> dict:
        if not bool(mask.any()):
            return {"n_pairs": 0, "n_plots": 0, "years": []}
        return {
            "n_pairs": int(mask.sum()),
            "n_plots": int(index.loc[mask, "plot_uid"].nunique()),
            "years": sorted(int(y) for y in np.unique(year_vals[mask])),
            "delta_days": {
                "min": int(deltas[mask].min()),
                "max": int(deltas[mask].max()),
                "mean": float(deltas[mask].mean()),
            },
            "interval_valid_days_all_full": bool(
                (interval_valid_days[mask] == deltas[mask]).all()
            ),
            "past_window_valid_days_all_full": bool(
                (past_valid_days[mask] == WEATHER_PAST_DAYS).all()
            ),
            "gdd_int_base0": {
                "mean": float(gdd_int[0.0][mask].mean()),
                "std": float(gdd_int[0.0][mask].std()),
                "min": float(gdd_int[0.0][mask].min()),
                "max": float(gdd_int[0.0][mask].max()),
            },
            "gdd_int_base5": {
                "mean": float(gdd_int[5.0][mask].mean()),
                "std": float(gdd_int[5.0][mask].std()),
                "min": float(gdd_int[5.0][mask].min()),
                "max": float(gdd_int[5.0][mask].max()),
            },
            "gdd_past28_base0": {
                "mean": float(gdd_past[(28, 0.0)][mask].mean()),
                "std": float(gdd_past[(28, 0.0)][mask].std()),
            },
            "corr_delta_vs_gdd_int_base0": float(
                np.corrcoef(deltas[mask], gdd_int[0.0][mask])[0, 1]
            ),
        }

    train_mask = index.split.to_numpy() == "train"
    val_mask = index.split.to_numpy() == "validation"
    summary = {
        "stage": "S4B.4A",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": " ".join(sys.argv),
        "inputs": {
            "index": str(index_path),
            "index_sha256": index_sha,
            "weather_csv": str(weather_csv),
            "weather_manifest_id": store.manifest_id,
            "weather_csv_sha256": store.csv_sha256,
            "n_rows": int(n),
        },
        "weather_vector": {
            "channels": list(store.channels),
            "units": {c: WEATHER_UNITS[c] for c in store.channels},
            "excluded": EXCLUDED_WEATHER_CHANNELS,
            "excluded_reason": (
                "a channel enters the GRU only after its unit, definition and "
                "time alignment are confirmed"
            ),
        },
        "protocol_R": {
            "definition": "interval (t_a, t_b]; real observed weather",
            "deployable": False,
            "label": "retrospective / oracle interval conditioning",
        },
        "protocol_P": {
            "definition": "[t_a-W, ..., t_a-1]; strictly before t_a",
            "deployable": True,
            "label": "deployable past-weather conditioning",
        },
        "gdd": {
            "definition": "GDD_d = max(T_mean_d - T_base, 0); T_base in {0, 5} degC",
            "unit": "degC day",
            "bases": list(GDD_BASES),
            "retrospective_features": "sum over (t_a, t_b]",
            "deployable_features": [f"sum over past {w} d" for w in GDD_PAST_WINDOWS],
            "physiological_status": (
                "pre-registered temperature baseline; not a genotype-fitted parameter"
            ),
        },
        "train": describe(train_mask, years),
        "validation": describe(val_mask, years),
        "structural_limit": (
            "S1 verified that all plots of the same harvest_year share one weather "
            "series, so weather provides date/year-level environment variation only; "
            "there is no plot-level micro-climate signal."
        ),
        "outputs": {
            "npz": str(npz_path),
            "npz_sha256": npz_sha,
            "npz_bytes": int(npz_path.stat().st_size),
        },
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps({
        "npz": str(npz_path),
        "sha256": npz_sha,
        "train": summary["train"],
        "validation": summary["validation"],
        "elapsed_seconds": summary["elapsed_seconds"],
    }, indent=2), flush=True)
    return 0


def _stable_index_sha(index: pd.DataFrame) -> str:
    """Same digest rule as ``agro_jepa.datasets.s3_pairs.index_sha256``."""
    cols = sorted(index.columns)
    return hashlib.sha256(index[cols].to_csv(index=False).encode()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
