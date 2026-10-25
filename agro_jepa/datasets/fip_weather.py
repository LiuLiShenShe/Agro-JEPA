"""Daily weather store and variable-length (t_a, t_b] window construction.

The daily table is the S1 artifact
``data/manifests/weather_daily_first_plot_per_year.csv`` (2,191 rows: every
calendar day of the six harvest years).  S1 verified that the hourly station
series is *identical across the plots checked for each year* (3 plots/year in
``reports/s1_weather_schema.json``), so the store is indexed by
``(harvest_year, date)`` and shared by all plots of that year.  This is the
reuse mechanism that avoids re-scanning 1.4 GB of parquet per image pair, and
avoids storing one copy of the series per plot.

Window convention (frozen in ``manifest_spec.json``)::

    window = (t_a, t_b]            # left-open, right-closed
    dates  = t_a+1, t_a+2, ..., t_b
    length = (t_b - t_a).days == delta_days

A day is *valid* when every requested channel is finite **and** the daily
aggregation had at least one hourly record for each of temperature, humidity,
radiation and precipitation.  Invalid / absent days keep their slot (so the
sequence length always equals ``delta_days``) but get ``mask=False`` and NaN
values; they are never silently filled with 0.

Units are the S1 ones: degC, %, W m-2 (daily sum in MJ m-2 d-1), and
precipitation in **raw sensor units** -- the unit is unresolved, so this
module deliberately exposes only ``P_sum_raw`` and computes no mm conversion,
no cumulative precipitation and no water-balance quantity.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

DEFAULT_WEATHER_CHANNELS: tuple[str, ...] = (
    "T_mean_C",
    "T_max_C",
    "T_min_C",
    "RH_mean_pct",
    "Rs_sum_MJ_m2_d",
    "P_sum_raw",
)
# VPD is derived from T and RH by S1 (formula recorded below); it is opt-in.
OPTIONAL_DERIVED_CHANNELS: tuple[str, ...] = ("VPD_mean_kPa",)
AVAILABLE_WEATHER_CHANNELS: tuple[str, ...] = (
    DEFAULT_WEATHER_CHANNELS + OPTIONAL_DERIVED_CHANNELS
)

WEATHER_UNITS: dict[str, str] = {
    "T_mean_C": "degC",
    "T_max_C": "degC",
    "T_min_C": "degC",
    "RH_mean_pct": "%",
    "Rs_sum_MJ_m2_d": "MJ m-2 d-1",
    "P_sum_raw": "raw sensor unit (UNRESOLVED - not mm)",
    "VPD_mean_kPa": "kPa",
}

WEATHER_FORMULAS: dict[str, str] = {
    "T_mean_C": "mean of hourly temperature_air_200cm over the calendar day",
    "T_max_C": "max of hourly temperature_air_200cm",
    "T_min_C": "min of hourly temperature_air_200cm",
    "RH_mean_pct": "mean of hourly humidity_air_200cm",
    "Rs_sum_MJ_m2_d": "sum(hourly irradiance_solar_200cm W m-2) * 3600 s / 1e6",
    "P_sum_raw": "sum(hourly precipitation_200cm) in raw sensor units",
    "VPD_mean_kPa": "mean(0.6108*exp(17.27*T/(T+237.3)) * (1 - RH/100))",
}

PRECIPITATION_UNIT_STATUS = "unresolved"
WEATHER_SOURCE = "data/manifests/weather_daily_first_plot_per_year.csv"
_HOURS_COLUMNS = ("n_hours_T", "n_hours_RH", "n_hours_Rs", "n_hours_P")


def _as_date(value: date | datetime | str | np.datetime64) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).date()
    return pd.Timestamp(value).date()


@dataclass(frozen=True)
class WeatherWindow:
    """One variable-length weather sequence for a pair ``(t_a, t_b]``."""

    harvest_year: int
    start_exclusive: date
    end_inclusive: date
    dates: tuple[date, ...]
    values: np.ndarray          # (L, C) float32, NaN where absent
    mask: np.ndarray            # (L,) bool
    n_hours_min: np.ndarray     # (L,) int16, min hourly coverage across sensors
    channels: tuple[str, ...]
    manifest_id: str

    @property
    def n_days(self) -> int:
        return len(self.dates)

    @property
    def n_valid_days(self) -> int:
        return int(self.mask.sum())

    @property
    def n_days_expected(self) -> int:
        return (self.end_inclusive - self.start_exclusive).days

    def as_dict(self) -> dict:
        return {
            "harvest_year": self.harvest_year,
            "window_start_exclusive": self.start_exclusive.isoformat(),
            "window_end_inclusive": self.end_inclusive.isoformat(),
            "window_n_days": self.n_days,
            "window_n_valid_days": self.n_valid_days,
            "weather_manifest_id": self.manifest_id,
            "weather_channels": list(self.channels),
        }


class DailyWeatherStore:
    """Year-indexed daily weather with contiguous, reusable per-year arrays."""

    def __init__(
        self,
        csv_path: str | Path,
        channels: Sequence[str] = DEFAULT_WEATHER_CHANNELS,
        add_vpd: bool = False,
        cache_size: int = 8192,
        require_mainline: bool = True,
    ) -> None:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"weather daily table not found: {csv_path}")
        self.csv_path = csv_path
        self.channels = tuple(
            list(channels) + (["VPD_mean_kPa"] if add_vpd and "VPD_mean_kPa" not in channels else [])
        )
        unknown = set(self.channels) - set(AVAILABLE_WEATHER_CHANNELS)
        if unknown:
            raise ValueError(f"unknown weather channels: {sorted(unknown)}")
        if require_mainline and not {
            "T_mean_C", "RH_mean_pct", "Rs_sum_MJ_m2_d", "P_sum_raw"
        } <= set(self.channels):
            raise ValueError(
                "the S2 mainline weather vector must keep the confirmed "
                "temperature, humidity, radiation and raw-precipitation channels"
            )
        if any("cum" in c or "water" in c for c in self.channels):
            raise ValueError("cumulative precipitation / water balance is out of scope in S2")
        blob = csv_path.read_bytes()
        self.csv_sha256 = hashlib.sha256(blob).hexdigest()
        self.manifest_id = f"s1-weather-daily-v1@{self.csv_sha256[:12]}"

        df = pd.read_csv(csv_path)
        missing = [c for c in self.channels if c not in df.columns]
        if missing:
            raise ValueError(f"{csv_path} lacks channels {missing}")
        df["date"] = pd.to_datetime(df["date"])

        self._tables: dict[int, dict[str, np.ndarray]] = {}
        self._positions: dict[int, dict[date, int]] = {}
        for year, g in df.groupby("harvest_year"):
            g = g.sort_values("date")
            days = [d.date() for d in g["date"]]
            self._tables[int(year)] = {
                "dates": np.array(days, dtype="datetime64[D]"),
                "values": g[list(self.channels)].to_numpy(dtype=np.float32),
                "hours": g[list(_HOURS_COLUMNS)].to_numpy(dtype=np.int16),
            }
            self._positions[int(year)] = {d: i for i, d in enumerate(days)}
        self._cache: dict[tuple, WeatherWindow] = {}
        self._cache_size = int(cache_size)
        self.n_years = len(self._tables)
        self.n_days_total = int(sum(len(t["dates"]) for t in self._tables.values()))

    # -- introspection ------------------------------------------------------
    @property
    def years(self) -> tuple[int, ...]:
        return tuple(sorted(self._tables))

    def date_range(self, harvest_year: int) -> tuple[date, date]:
        tbl = self._tables[int(harvest_year)]
        d = tbl["dates"]
        return pd.Timestamp(d[0]).date(), pd.Timestamp(d[-1]).date()

    def channel_values(self, harvest_year: int, channel: str) -> np.ndarray:
        """All daily values of one channel for one harvest year."""
        if channel not in self.channels:
            raise KeyError(f"{channel!r} not in store channels {self.channels}")
        tbl = self._tables[int(harvest_year)]
        return tbl["values"][:, self.channels.index(channel)]

    # -- window -------------------------------------------------------------
    def window(
        self,
        harvest_year: int,
        start_exclusive: date | datetime | str,
        end_inclusive: date | datetime | str,
    ) -> WeatherWindow:
        """Return the ``(t_a, t_b]`` sequence; length is always delta_days."""
        start = _as_date(start_exclusive)
        end = _as_date(end_inclusive)
        delta = (end - start).days
        if delta < 0:
            raise ValueError(f"end_inclusive {end} precedes start_exclusive {start}")
        year = int(harvest_year)
        key = (year, start, end)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        if year not in self._tables:
            raise KeyError(
                f"harvest_year {year} not in weather store (years: {self.years})"
            )
        values_tbl = self._tables[year]["values"]
        hours_tbl = self._tables[year]["hours"]
        pos = self._positions[year]

        n_ch = len(self.channels)
        values = np.full((delta, n_ch), np.nan, dtype=np.float32)
        mask = np.zeros((delta,), dtype=bool)
        hours_min = np.zeros((delta,), dtype=np.int16)
        dates: list[date] = []
        for k in range(delta):
            d = start + timedelta(days=k + 1)      # left-open: first day is t_a + 1
            dates.append(d)
            p = pos.get(d)
            if p is None:
                continue
            v = values_tbl[p]
            values[k] = v
            h = int(hours_tbl[p].min())
            hours_min[k] = h
            mask[k] = bool(np.isfinite(v).all() and h > 0)

        win = WeatherWindow(
            harvest_year=year,
            start_exclusive=start,
            end_inclusive=end,
            dates=tuple(dates),
            values=values,
            mask=mask,
            n_hours_min=hours_min,
            channels=self.channels,
            manifest_id=self.manifest_id,
        )
        # invariant: the sequence length equals the calendar delta
        assert win.n_days == delta == win.n_days_expected
        if len(self._cache) >= self._cache_size:
            self._cache.clear()
        self._cache[key] = win
        return win

    # -- coverage report ----------------------------------------------------
    def window_valid_counts(
        self,
        harvest_year: int,
        starts_exclusive: Iterable,
        ends_inclusive: Iterable,
    ) -> np.ndarray:
        """Vectorised count of valid days in many ``(t_a, t_b]`` windows.

        Used by the pair builder to verify the whole 10^5-scale pair table
        without constructing each sequence in Python.
        """
        year = int(harvest_year)
        if year not in self._tables:
            raise KeyError(f"harvest_year {year} not in weather store")
        tbl = self._tables[year]
        dates = tbl["dates"]
        valid = np.isfinite(tbl["values"]).all(axis=1) & (tbl["hours"].min(axis=1) > 0)
        prefix = np.concatenate([[0], np.cumsum(valid.astype(np.int64))])
        s = np.searchsorted(
            dates, np.asarray(list(starts_exclusive), dtype="datetime64[D]"), side="right"
        )
        e = np.searchsorted(
            dates, np.asarray(list(ends_inclusive), dtype="datetime64[D]"), side="right"
        )
        return (prefix[e] - prefix[s]).astype(np.int64)

    def coverage(self, harvest_year: int) -> dict:
        tbl = self._tables[int(harvest_year)]
        v = tbl["values"]
        return {
            "year": int(harvest_year),
            "n_days": int(len(tbl["dates"])),
            "date_first": str(pd.Timestamp(tbl["dates"][0]).date()),
            "date_last": str(pd.Timestamp(tbl["dates"][-1]).date()),
            "n_days_all_finite": int(np.isfinite(v).all(axis=1).sum()),
            "n_days_partial_hours": int((tbl["hours"].min(axis=1) < 24).sum()),
            "channel_means": {c: float(np.nanmean(v[:, i])) for i, c in enumerate(self.channels)},
            "channel_mins": {c: float(np.nanmin(v[:, i])) for i, c in enumerate(self.channels)},
            "channel_maxs": {c: float(np.nanmax(v[:, i])) for i, c in enumerate(self.channels)},
        }


def weather_provenance() -> dict:
    """Static description of the weather vector for reports/configs."""
    return {
        "source": WEATHER_SOURCE,
        "manifest_id": "s1-weather-daily-v1",
        "window_convention": "(t_a, t_b] -- left open, right closed; length == delta_days",
        "channels": {
            c: {
                "unit": WEATHER_UNITS[c],
                "formula": WEATHER_FORMULAS[c],
                "derived": c in OPTIONAL_DERIVED_CHANNELS,
            }
            for c in AVAILABLE_WEATHER_CHANNELS
        },
        "precipitation_unit_status": PRECIPITATION_UNIT_STATUS,
        "excluded_derived_quantities": [
            "cumulative precipitation",
            "water balance / evapotranspiration",
            "growing degree days (base temperature not verified for FIP)",
        ],
        "reuse": "one daily table per harvest_year; all plots of a year share it "
                 "(S1 verified identical station series across 3 plots/year)",
    }
