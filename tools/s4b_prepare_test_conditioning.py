#!/usr/bin/env python3
"""S4B: conditioning tensors for the prepared official-test pair indexes.

Uses the *same* window and GDD definitions as S4A (``tools/s4_prepare_weather.py``
is imported, not copied)::

    Protocol P (deployable):  [t_a-28, ..., t_a-1]      strictly before t_a
    Protocol R (retrospective): (t_a, t_b]               real observed interval
    GDD_d = max(T_mean_d - T_base, 0), T_base in {0, 5} degC

Normalisation statistics are *not* recomputed here: S4B must reuse the
inner-train statistics stored in the frozen checkpoint, so the arrays are saved
raw and the loader applies the checkpoint's ``DeltaNormStats`` / ``VectorNormStats``.

No phenotype label is read.  The output is an input-only asset for a later,
explicitly labelled test evaluation.

Usage
-----
    python3 tools/s4b_prepare_test_conditioning.py \
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
sys.path.insert(0, str(ROOT / "tools"))

from agro_jepa.datasets.fip_weather import DailyWeatherStore, WEATHER_UNITS  # noqa: E402
from s4_prepare_weather import (  # noqa: E402
    GDD_BASES,
    GDD_PAST_WINDOWS,
    MAX_INTERVAL_DAYS,
    S4_WEATHER_CHANNELS,
    WEATHER_PAST_DAYS,
    _gdd,
    _past_window,
)

INDEX_COLUMNS = [
    "pair_id", "pair_type", "span_days", "plot_uid", "genotype_id", "harvest_year",
    "split", "yearsite_uid", "image_stem_a", "image_stem_b", "date_a", "date_b",
    "delta_days", "window_n_days", "inner_plot_image_a", "inner_plot_image_b",
    "feature_ready", "weather_interval_ready", "weather_past_ready",
    "weather_year_available",
]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s4a1_matched_control.yaml")
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    s4b = cfg["s4b"]
    t0 = time.time()

    index_dir = Path(s4b["index_out_dir"])
    out_dir = Path(s4b["conditioning_out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    store = DailyWeatherStore(
        cfg["data"]["weather_daily"],
        channels=list(S4_WEATHER_CHANNELS),
        add_vpd=False,
        require_mainline=False,
    )
    if tuple(store.channels) != tuple(S4_WEATHER_CHANNELS):
        raise SystemExit("weather channel mismatch against the S4A channel list")

    summary = {
        "stage": "S4B",
        "task": "conditioning tensors for the official test pair indexes (inputs only)",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": " ".join(sys.argv),
        "config": args.config,
        "config_sha256": sha256_file(args.config),
        "definitions": {
            "interval": "(t_a, t_b] -- Protocol R, retrospective / oracle",
            "past": f"[t_a-{WEATHER_PAST_DAYS}, ..., t_a-1] -- Protocol P, deployable",
            "gdd": "GDD_d = max(T_mean_d - T_base, 0); T_base in {0, 5} degC",
            "channels": list(store.channels),
            "units": {c: WEATHER_UNITS[c] for c in store.channels},
            "normalisation": (
                "raw arrays only; the frozen checkpoint's inner-train statistics "
                "must be applied by the evaluator"
            ),
            "phenotype_labels_read": False,
        },
        "weather_manifest_id": store.manifest_id,
        "weather_csv_sha256": store.csv_sha256,
        "splits": {},
    }

    for split in s4b["test_splits"]:
        idx_path = index_dir / f"{split}_test_pairs.parquet"
        if not idx_path.exists():
            raise SystemExit(f"missing pair index {idx_path}; run s4b_prepare_test_data.py")
        index = pd.read_parquet(idx_path, columns=INDEX_COLUMNS)
        if not (index.split == split).all():
            raise SystemExit(f"{idx_path} contains rows outside split {split!r}")
        n = len(index)
        n_ch = len(store.channels)
        gdd_int = {b: np.full(n, np.nan, dtype=np.float32) for b in GDD_BASES}
        gdd_past = {(w, b): np.full(n, np.nan, dtype=np.float32)
                    for w in GDD_PAST_WINDOWS for b in GDD_BASES}
        weather_int = np.zeros((n, MAX_INTERVAL_DAYS, n_ch), dtype=np.float32)
        weather_int_mask = np.zeros((n, MAX_INTERVAL_DAYS), dtype=bool)
        weather_past = np.zeros((n, WEATHER_PAST_DAYS, n_ch), dtype=np.float32)
        weather_past_mask = np.zeros((n, WEATHER_PAST_DAYS), dtype=bool)
        interval_valid = np.zeros(n, dtype=np.int16)
        past_valid = np.zeros(n, dtype=np.int16)

        dates_a = pd.to_datetime(index.date_a)
        dates_b = pd.to_datetime(index.date_b)
        years = index.harvest_year.to_numpy(np.int64)
        deltas = index.delta_days.to_numpy(np.int64)

        for i in range(n):
            y, a, b = int(years[i]), dates_a.iloc[i].date(), dates_b.iloc[i].date()
            delta = int(deltas[i])
            if not (0 < delta <= MAX_INTERVAL_DAYS):
                raise SystemExit(f"row {i}: delta_days {delta} outside (0,{MAX_INTERVAL_DAYS}]")
            win = store.window(y, a, b)
            weather_int[i, :delta] = win.values
            weather_int_mask[i, :delta] = win.mask
            interval_valid[i] = win.n_valid_days
            for base in GDD_BASES:
                gdd_int[base][i] = _gdd(win.values, win.mask, base)
            pwin = _past_window(store, y, a, WEATHER_PAST_DAYS)
            weather_past[i] = pwin.values
            weather_past_mask[i] = pwin.mask
            past_valid[i] = pwin.n_valid_days
            for w in GDD_PAST_WINDOWS:
                sl = slice(WEATHER_PAST_DAYS - w, WEATHER_PAST_DAYS)
                for base in GDD_BASES:
                    gdd_past[(w, base)][i] = _gdd(pwin.values[sl], pwin.mask[sl], base)

        if not np.array_equal(interval_valid, deltas):
            raise SystemExit(f"{split}: interval window not fully covered")
        if not np.all(past_valid == WEATHER_PAST_DAYS):
            raise SystemExit(f"{split}: past window not fully covered")

        payload = {
            "pair_id": index.pair_id.to_numpy().astype(str),
            "split": index.split.to_numpy().astype(str),
            "plot_uid": index.plot_uid.to_numpy().astype(str),
            "genotype_id": index.genotype_id.to_numpy().astype(str),
            "harvest_year": years.astype(np.int16),
            "delta_days": deltas.astype(np.int16),
            "date_a": dates_a.dt.strftime("%Y-%m-%d").to_numpy().astype(str),
            "date_b": dates_b.dt.strftime("%Y-%m-%d").to_numpy().astype(str),
            "weather_channels": np.array(store.channels, dtype=object),
            "weather_int": weather_int,
            "weather_int_mask": weather_int_mask,
            "weather_past": weather_past,
            "weather_past_mask": weather_past_mask,
            "interval_valid_days": interval_valid,
            "past_valid_days": past_valid,
            "index_sha256": np.array(sha256_file(idx_path)),
            "weather_manifest_id": np.array(store.manifest_id),
            "past_window_rule": np.array("[t_a-W, ..., t_a-1] (strictly before t_a)"),
            "interval_window_rule": np.array("(t_a, t_b]"),
            "phenotype_labels_read": np.array(False),
        }
        for base in GDD_BASES:
            payload[f"gdd_int_base{int(base)}"] = gdd_int[base]
        for (w, base), arr in gdd_past.items():
            payload[f"gdd_past{w}_base{int(base)}"] = arr

        path = out_dir / f"{split}_conditioning.npz"
        np.savez_compressed(path, **payload)
        summary["splits"][split] = {
            "n_pairs": int(n),
            "n_plots": int(index.plot_uid.nunique()),
            "delta_days": {"min": int(deltas.min()), "max": int(deltas.max()),
                           "mean": float(deltas.mean())},
            "npz": str(path),
            "npz_sha256": sha256_file(path),
            "npz_bytes": int(path.stat().st_size),
            "interval_valid_days_all_full": True,
            "past_window_valid_days_all_full": True,
            "gdd_int_base0": {"mean": float(gdd_int[0.0].mean()),
                              "min": float(gdd_int[0.0].min()),
                              "max": float(gdd_int[0.0].max())},
            "gdd_past28_base0": {"mean": float(gdd_past[(28, 0.0)].mean())},
        }
        print(f"[{split}] wrote {path} ({n} pairs)", flush=True)

    summary["elapsed_seconds"] = round(time.time() - t0, 1)
    out_json = Path("reports/s4b_test_conditioning.json")
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
