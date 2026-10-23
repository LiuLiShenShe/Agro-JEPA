"""Pair construction for FIP 1.0 (S2).

A *pair* is two valid observations of the **same** ``plot_uid`` at dates
``t_a < t_b``.  Image<->date correspondence always comes from the audited
``image_filename_stem`` column of ``fip1_observations.csv.gz`` (S1 §3: the
``inner_plot_images`` and ``image_dates`` lists are not positionally aligned,
1,773/4,058 plots differ in length).  Nothing here re-derives dates from list
positions.

Pair families
-------------
``consecutive``
    every adjacent pair of distinct images of a plot (irregular ``delta_days``;
    this is the variable-time-interval training signal).
``span{s}``
    for every anchor image, the nearest later image whose ``delta_days`` is
    within ``+/- span_tolerance_days`` of ``s`` (the fixed-horizon controls).

Every sample carries its full provenance: plot, genotype, harvest year,
official split, custom cross-year split, image stems and HF paths, image dates,
``delta_days``, the weather window bounds, and per-trait raw value / actual
measurement date / signed lag / validity mask at both ends.

Label rules (frozen in S1, re-applied here, not re-decided)
----------------------------------------------------------
* a measured ``0`` is data and stays ``0`` with ``mask=True``;
* an absent measurement date is *missing* (NaN, ``mask=False``) -- never 0;
* PH negative values are preserved verbatim and additionally marked with
  ``ph_negative_a/b``; they are not clipped (validity deferred to a dedicated
  sensitivity analysis);
* Sen exists only in 2016-2018 and is never imputed for other years.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .provenance import S2_MANIFEST_ID

TRAITS: tuple[str, ...] = ("cc", "ph", "whc", "sen")
TRAIT_NAMES: dict[str, str] = {
    "cc": "canopy_cover",
    "ph": "plant_height",
    "whc": "wheat_head_count",
    "sen": "senescence",
}
TRAIT_UNITS: dict[str, str] = {
    "cc": "fraction 0-1 (trait metadata says '%', values are fractions)",
    "ph": "m",
    "whc": "count",
    "sen": "0-10 score",
}
TRAIT_YEARS: dict[str, tuple[int, ...]] = {
    "cc": (2016, 2017, 2018, 2019, 2021, 2022),
    "ph": (2016, 2017, 2018, 2019, 2021, 2022),
    "whc": (2016, 2017, 2018, 2019, 2021, 2022),
    "sen": (2016, 2017, 2018),
}

OFFICIAL_SPLITS: tuple[str, ...] = (
    "train",
    "validation",
    "test_plot",
    "test_genotype",
    "test_environment",
    "test_genotype_environment",
)


@dataclass(frozen=True)
class PairRule:
    """Deterministic, serialisable pair-selection rule."""

    mode: str = "consecutive"           # "consecutive" | "span" | "both"
    spans: tuple[int, ...] = (7, 14, 21, 28)
    span_tolerance_days: int = 2
    min_delta_days: int = 1              # exclude same-day / duplicate images
    max_delta_days: int | None = None
    label_max_lag_days: int = 7
    manifest_id: str = S2_MANIFEST_ID
    dedupe_image_stems: bool = True

    def as_dict(self) -> dict:
        return {
            "manifest_id": self.manifest_id,
            "mode": self.mode,
            "spans": list(self.spans),
            "span_tolerance_days": self.span_tolerance_days,
            "min_delta_days": self.min_delta_days,
            "max_delta_days": self.max_delta_days,
            "label_max_lag_days": self.label_max_lag_days,
            "dedupe_image_stems": self.dedupe_image_stems,
        }


# --------------------------------------------------------------------------
# manifest loading
# --------------------------------------------------------------------------
def load_observations(path: str | Path) -> pd.DataFrame:
    """Load the S1 observation manifest and normalise its dtypes.

    Returns one row per (plot_uid, image stem) after removing exact duplicate
    references (S1's manifest re-lists 1,872 images two or three times).
    """
    path = Path(path)
    obs = pd.read_csv(path, dtype={"plot_uid": str, "split": str}, low_memory=False)
    required = {
        "plot_uid", "split", "harvest_year", "obs_index", "image_date",
        "image_time", "image_filename_stem", "inner_plot_image",
    }
    for t in TRAITS:
        required |= {f"{t}_value", f"{t}_lag_days", f"{t}_date"}
    missing = required - set(obs.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")

    obs = obs.copy()
    obs["image_date"] = pd.to_datetime(obs["image_date"], errors="raise")
    for t in TRAITS:
        obs[f"{t}_value"] = pd.to_numeric(obs[f"{t}_value"], errors="coerce").astype("float64")
        obs[f"{t}_lag_days"] = pd.to_numeric(obs[f"{t}_lag_days"], errors="coerce").astype("float64")
        obs[f"{t}_date"] = pd.to_datetime(obs[f"{t}_date"], errors="coerce")
    obs["obs_index"] = pd.to_numeric(obs["obs_index"], errors="raise").astype("int64")
    obs["harvest_year"] = pd.to_numeric(obs["harvest_year"], errors="raise").astype("int64")
    return obs


def dedupe_observations(obs: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop duplicate references to the same physical image of the same plot.

    S1's consecutive-pair manifest contains 1,872 pairs with ``delta_days == 0``
    and ``image_a == image_b``; they come from duplicate rows in the observation
    manifest, not from genuinely repeated same-day imagery.  Those pairs have an
    empty ``(t_a, t_b]`` weather window and are not valid training samples.
    """
    key = ["plot_uid", "image_filename_stem"]
    n_before = len(obs)
    dup_mask = obs.duplicated(subset=key, keep="first")
    out = obs.loc[~dup_mask].copy()
    # stable channel order: date, then acquisition time, then source index
    out = out.sort_values(
        ["plot_uid", "image_date", "image_time", "obs_index"], kind="mergesort"
    ).reset_index(drop=True)
    out["plot_pos"] = out.groupby("plot_uid", sort=False).cumcount()
    stats = {
        "n_observation_rows_in_manifest": int(n_before),
        "n_duplicate_rows_removed": int(dup_mask.sum()),
        "n_unique_plot_image_pairs": int(len(out)),
        "n_plots": int(out.plot_uid.nunique()),
        "n_plots_with_duplicates": int(
            obs.loc[obs.duplicated(subset=key, keep=False), "plot_uid"].nunique()
        ),
    }
    return out, stats


def load_plots(path: str | Path) -> pd.DataFrame:
    plots = pd.read_csv(path, dtype={"plot_uid": str, "genotype_id": str})
    if not plots.plot_uid.is_unique:
        raise ValueError("fip1_plots.csv has duplicate plot_uid")
    return plots


def load_custom_splits(path: str | Path) -> pd.DataFrame:
    cs = pd.read_csv(path, dtype={"plot_uid": str, "genotype_id": str})
    if not cs.plot_uid.is_unique:
        raise ValueError("custom_crossyear_splits.csv has duplicate plot_uid")
    return cs


# --------------------------------------------------------------------------
# pair index construction
# --------------------------------------------------------------------------
def _consecutive_index(dates_daynum: np.ndarray) -> list[tuple[int, int, int]]:
    """(i, j, span=-1) for adjacent images of one plot."""
    n = len(dates_daynum)
    return [(k, k + 1, -1) for k in range(n - 1)]


def _span_index(
    dates_daynum: np.ndarray,
    spans: Sequence[int],
    tolerance: int,
    min_delta: int,
    max_delta: int | None,
) -> list[tuple[int, int, int]]:
    """(i, j, span) with the nearest later image to ``t_i + span``."""
    out: list[tuple[int, int, int]] = []
    n = len(dates_daynum)
    for k in range(n - 1):
        later = dates_daynum[k + 1:]
        for s in spans:
            target = int(dates_daynum[k]) + int(s)
            rel = int(np.searchsorted(later, target))
            best: tuple[int, int, int] | None = None
            for c in (rel - 1, rel):
                if 0 <= c < later.size:
                    d = int(later[c]) - target
                    cand = (abs(d), d, c)
                    if best is None or cand < best:
                        best = cand
            if best is None or best[0] > tolerance:
                continue
            delta = int(later[best[2]] - dates_daynum[k])
            if delta < min_delta:
                continue
            if max_delta is not None and delta > max_delta:
                continue
            out.append((k, k + 1 + best[2], int(s)))
    return out


def _date_str(series: pd.Series) -> np.ndarray:
    """ISO dates with ``None`` (not NaN) for missing label dates."""
    s = pd.to_datetime(series, errors="coerce")
    out = np.empty(len(s), dtype=object)
    ok = s.notna().to_numpy()
    if ok.any():
        out[ok] = s[ok].dt.strftime("%Y-%m-%d").to_numpy(dtype=object)
    out[~ok] = None
    return out


def _assemble(
    obs: pd.DataFrame,
    ia: np.ndarray,
    ib: np.ndarray,
    pair_type: np.ndarray,
    span_days: np.ndarray,
    plot_meta: pd.DataFrame,
    custom_meta: pd.DataFrame,
    rule: PairRule,
) -> pd.DataFrame:
    a = obs.iloc[ia].reset_index(drop=True)
    b = obs.iloc[ib].reset_index(drop=True)
    n = len(a)

    plot_uid = a["plot_uid"].to_numpy(dtype=object)
    stem_a = a["image_filename_stem"].to_numpy(dtype=object)
    stem_b = b["image_filename_stem"].to_numpy(dtype=object)
    date_a = _date_str(a["image_date"])
    date_b = _date_str(b["image_date"])
    delta = (b["image_date"] - a["image_date"]).dt.days.to_numpy(dtype=np.int64)

    pair_ids = np.array([
        hashlib.sha1(
            f"{rule.manifest_id}|{pu}|{sa}|{sb}|{pt}|{sd}".encode()
        ).hexdigest()[:16]
        for pu, sa, sb, pt, sd in zip(plot_uid, stem_a, stem_b, pair_type, span_days)
    ], dtype=object)

    meta = plot_meta.reindex(plot_uid)
    cmeta = custom_meta.reindex(plot_uid)

    out: dict[str, object] = {
        "pair_id": pair_ids,
        "manifest_id": np.full(n, rule.manifest_id, dtype=object),
        "pair_type": pair_type,
        "span_days": span_days.astype(np.int64),
        "plot_uid": plot_uid,
        "genotype_id": meta["genotype_id"].to_numpy(dtype=object),
        "harvest_year": a["harvest_year"].to_numpy(dtype=np.int64),
        "split": a["split"].to_numpy(dtype=object),
        "custom_split": cmeta["custom_split"].to_numpy(dtype=object),
        "i": a["plot_pos"].to_numpy(dtype=np.int64),
        "j": b["plot_pos"].to_numpy(dtype=np.int64),
        "obs_index_a": a["obs_index"].to_numpy(dtype=np.int64),
        "obs_index_b": b["obs_index"].to_numpy(dtype=np.int64),
        "image_stem_a": stem_a,
        "image_stem_b": stem_b,
        "image_a": a["inner_plot_image"].to_numpy(dtype=object),
        "image_b": b["inner_plot_image"].to_numpy(dtype=object),
        "date_a": date_a,
        "date_b": date_b,
        "delta_days": delta,
        "window_start_exclusive": date_a,
        "window_end_inclusive": date_b,
        "window_n_days": delta,
    }

    for t in TRAITS:
        va = a[f"{t}_value"].to_numpy(dtype=np.float64)
        vb = b[f"{t}_value"].to_numpy(dtype=np.float64)
        la = a[f"{t}_lag_days"].to_numpy(dtype=np.float64)
        lb = b[f"{t}_lag_days"].to_numpy(dtype=np.float64)
        da = _date_str(a[f"{t}_date"])
        db = _date_str(b[f"{t}_date"])
        out[f"{t}_a"] = va
        out[f"{t}_b"] = vb
        out[f"{t}_a_lag"] = la
        out[f"{t}_b_lag"] = lb
        out[f"{t}_a_date"] = da
        out[f"{t}_b_date"] = db
        out[f"{t}_mask_a"] = np.isfinite(va) & np.isfinite(la) & (np.abs(la) <= rule.label_max_lag_days)
        out[f"{t}_mask_b"] = np.isfinite(vb) & np.isfinite(lb) & (np.abs(lb) <= rule.label_max_lag_days)

    # PH negatives are kept in the raw value and flagged, never clipped
    out["ph_negative_a"] = np.isfinite(out["ph_a"]) & (np.asarray(out["ph_a"]) < 0)
    out["ph_negative_b"] = np.isfinite(out["ph_b"]) & (np.asarray(out["ph_b"]) < 0)

    return pd.DataFrame(out)


def build_pairs(
    obs: pd.DataFrame,
    plots: pd.DataFrame,
    custom: pd.DataFrame,
    rule: PairRule = PairRule(),
) -> tuple[pd.DataFrame, dict]:
    """Build the pair table.  Deterministic for a given (manifest, rule)."""
    if rule.mode not in {"consecutive", "span", "both"}:
        raise ValueError(f"unknown pair mode {rule.mode!r}")
    obs_d, dedupe_stats = dedupe_observations(obs) if rule.dedupe_image_stems else (obs.copy(), {})
    if "plot_pos" not in obs_d:
        obs_d["plot_pos"] = obs_d.groupby("plot_uid", sort=False).cumcount()

    plot_meta = plots.set_index("plot_uid")
    custom_meta = custom.set_index("plot_uid")
    missing_meta = sorted(set(obs_d.plot_uid) - set(plot_meta.index))
    if missing_meta:
        raise ValueError(f"{len(missing_meta)} plot_uid missing from fip1_plots.csv")
    missing_custom = sorted(set(obs_d.plot_uid) - set(custom_meta.index))
    if missing_custom:
        raise ValueError(f"{len(missing_custom)} plot_uid missing from custom splits")

    epoch = np.datetime64("1970-01-01", "D")
    ia_all: list[np.ndarray] = []
    ib_all: list[np.ndarray] = []
    type_all: list[np.ndarray] = []
    span_all: list[np.ndarray] = []

    for _, g in obs_d.groupby("plot_uid", sort=False):
        positions = g.index.to_numpy()          # positional indices into obs_d
        dn = ((g["image_date"].to_numpy(dtype="datetime64[D]") - epoch).astype(np.int64))

        if rule.mode in {"consecutive", "both"}:
            triples = _consecutive_index(dn)
            triples = [t for t in triples
                       if rule.min_delta_days <= (dn[t[1]] - dn[t[0]])
                       and (rule.max_delta_days is None
                            or (dn[t[1]] - dn[t[0]]) <= rule.max_delta_days)]
            if triples:
                ia, ib, sp = zip(*triples)
                ia_all.append(positions[np.array(ia)])
                ib_all.append(positions[np.array(ib)])
                type_all.append(np.full(len(ia), "consecutive", dtype=object))
                span_all.append(np.array(sp, dtype=np.int64))

        if rule.mode in {"span", "both"}:
            triples = _span_index(dn, rule.spans, rule.span_tolerance_days,
                                  rule.min_delta_days, rule.max_delta_days)
            if triples:
                ia, ib, sp = zip(*triples)
                ia_all.append(positions[np.array(ia)])
                ib_all.append(positions[np.array(ib)])
                type_all.append(np.array([f"span{s}" for s in sp], dtype=object))
                span_all.append(np.array(sp, dtype=np.int64))

    if not ia_all:
        empty = _assemble(
            obs_d, np.array([], int), np.array([], int),
            np.array([], object), np.array([], int),
            plot_meta, custom_meta, rule,
        )
        return empty, {"dedupe": dedupe_stats, "rule": rule.as_dict(), "n_pairs": 0}

    ia = np.concatenate(ia_all)
    ib = np.concatenate(ib_all)
    pair_type = np.concatenate(type_all)
    span_days = np.concatenate(span_all)

    pairs = _assemble(obs_d, ia, ib, pair_type, span_days, plot_meta, custom_meta, rule)
    pairs = pairs.sort_values(["plot_uid", "delta_days", "pair_type", "pair_id"],
                              kind="mergesort").reset_index(drop=True)
    stats = {
        "rule": rule.as_dict(),
        "dedupe": dedupe_stats,
        "n_pairs": int(len(pairs)),
    }
    return pairs, stats


def pair_manifest_schema() -> dict:
    """Human/machine readable description of the pair table columns."""
    cols = {
        "pair_id": "sha1(manifest|plot|stem_a|stem_b|pair_type|span)[:16]",
        "manifest_id": "pair manifest id (fip1-s2-v1)",
        "pair_type": "'consecutive' or 'span{S}'",
        "span_days": "target span for span pairs, -1 for consecutive",
        "plot_uid": "FIP plot-year id; both ends share it",
        "genotype_id": "anonymised 64-hex genotype id from fip1_plots.csv",
        "harvest_year": "int, one of 2016/2017/2018/2019/2021/2022",
        "split": "official split inherited from the plot",
        "custom_split": "custom-crossyear-v1 split, stored separately",
        "i": "position of image A within the deduplicated plot trajectory",
        "j": "position of image B within the deduplicated plot trajectory",
        "obs_index_a/b": "obs_index in fip1_observations.csv.gz",
        "image_stem_a/b": "audited filename stem (authoritative date source)",
        "image_a/b": "inner_plot_image path (HF URL)",
        "date_a/b": "image date YYYY-MM-DD",
        "delta_days": "(date_b - date_a).days == window length",
        "window_start_exclusive": "t_a, open bound of the weather window",
        "window_end_inclusive": "t_b, closed bound of the weather window",
        "window_n_days": "number of daily weather steps == delta_days",
    }
    for t in TRAITS:
        cols[f"{t}_a/b"] = f"raw {TRAIT_NAMES[t]} at t_a / t_b ({TRAIT_UNITS[t]}), NaN if absent"
        cols[f"{t}_a/b_lag"] = "signed lag label_date - image_date, days"
        cols[f"{t}_a/b_date"] = "actual measurement date of the matched label"
        cols[f"{t}_mask_a/b"] = "value present AND |lag| <= label_max_lag_days"
    cols["ph_negative_a/b"] = "true when raw PH is negative (kept, not clipped)"
    return cols
