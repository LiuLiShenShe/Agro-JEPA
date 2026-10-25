"""S4A dataset: frozen DINOv2 features + pre-computed GDD / weather tensors.

The conditioning arrays are built once by ``tools/s4_prepare_weather.py`` and are
loaded read-only.  Every sample carries *both* protocol views (P and R) so a
single dataset can serve every condition; the model config decides which one is
actually consumed, and the trainer records that choice in the checkpoint.

Nothing here reads a future image, a future feature, the future phenotype, or
any official test row.  ``z_b`` is a stop-gradient training target only.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ConditioningSpec:
    """Which conditioning arrays a condition actually consumes."""

    use_delta: bool
    gdd_mode: str          # none | interval | past
    gdd_base: float
    weather_mode: str      # none | interval | past

    def gdd_keys(self) -> tuple[str, ...]:
        if self.gdd_mode == "none":
            return ()
        if self.gdd_mode == "interval":
            return (f"gdd_int_base{int(self.gdd_base)}",)
        if self.gdd_mode == "past":
            return tuple(f"gdd_past{w}_base{int(self.gdd_base)}" for w in (7, 14, 28))
        raise ValueError(f"unknown gdd_mode {self.gdd_mode!r}")

    def weather_keys(self) -> tuple[str, str] | None:
        """(values, mask) array names for the selected protocol."""
        if self.weather_mode == "none":
            return None
        if self.weather_mode == "interval":
            return ("weather_int", "weather_int_mask")
        if self.weather_mode == "past":
            return ("weather_past", "weather_past_mask")
        raise ValueError(f"unknown weather_mode {self.weather_mode!r}")

    def as_dict(self) -> dict:
        return {
            "use_delta": self.use_delta,
            "gdd_mode": self.gdd_mode,
            "gdd_base": self.gdd_base,
            "weather_mode": self.weather_mode,
            "gdd_features": list(self.gdd_keys()),
            "weather_features": list(self.weather_keys() or ()),
        }


def load_conditioning(path: str | Path, index: pd.DataFrame) -> dict[str, np.ndarray]:
    """Load the npz and assert row alignment with the frozen index."""
    npz = np.load(path, allow_pickle=True)
    pair_ids = npz["pair_id"].astype(str)
    if "pair_id" in index.columns:
        expected = index["pair_id"].to_numpy().astype(str)
        if len(expected) != len(pair_ids) or not np.array_equal(expected, pair_ids):
            raise ValueError(
                f"{path} is not aligned with the frozen index row order; "
                "rebuild it with tools/s4_prepare_weather.py"
            )
    return {k: npz[k] for k in npz.files}


class S4PairDataset(Dataset):
    """One row = one image pair; returns features, conditioning and labels."""

    REQUIRED_INDEX_COLUMNS = (
        "pair_id", "plot_uid", "split", "pair_type", "span_days", "delta_days",
        "harvest_year", "image_stem_a", "image_stem_b", "cache_file_a",
        "cache_file_b", "date_a", "date_b", "genotype_id",
    )

    def __init__(
        self,
        index: pd.DataFrame,
        conditioning: dict[str, np.ndarray],
        spec: ConditioningSpec,
        traits: Sequence[str] = ("cc", "ph"),
        load_features: bool = True,
    ) -> None:
        missing = set(self.REQUIRED_INDEX_COLUMNS) - set(index.columns)
        if missing:
            raise ValueError(f"index is missing columns {sorted(missing)}")
        if index.index.tolist() != list(range(len(index))):
            raise ValueError("index must have a 0..n-1 RangeIndex aligned to the npz")
        self.index = index.reset_index(drop=True)
        self.cond = conditioning
        self.spec = spec
        self.traits = tuple(traits)
        self.load_features = bool(load_features)
        for key in spec.gdd_keys():
            if key not in conditioning:
                raise KeyError(f"conditioning array {key!r} is missing from the npz")
        keys = spec.weather_keys()
        if keys is not None:
            for k in keys:
                if k not in conditioning:
                    raise KeyError(f"conditioning array {k!r} is missing from the npz")
            w = conditioning[keys[0]]
            m = conditioning[keys[1]]
            if w.shape[0] != len(index) or m.shape[0] != len(index):
                raise ValueError("weather arrays are not aligned with the index rows")
            if w.shape[1] != m.shape[1]:
                raise ValueError("weather value/mask length mismatch")
        self._rows = self.index.to_dict("records")

    def __len__(self) -> int:
        return len(self.index)

    @staticmethod
    def _load(path: str) -> torch.Tensor:
        arr = np.load(path)
        if arr.ndim != 2:
            raise ValueError(f"feature cache {path} has shape {arr.shape}, expected 2-D")
        return torch.from_numpy(arr.astype(np.float32))

    def __getitem__(self, i: int) -> dict:
        row = self._rows[i]
        sample: dict = {
            "row": i,
            "pair_id": str(row["pair_id"]),
            "plot_uid": str(row["plot_uid"]),
            "genotype_id": str(row["genotype_id"]),
            "split": str(row["split"]),
            "pair_type": str(row["pair_type"]),
            "span_days": int(row["span_days"]),
            "delta_days": torch.tensor(float(row["delta_days"]), dtype=torch.float32),
            "harvest_year": int(row["harvest_year"]),
            "image_stem_a": str(row["image_stem_a"]),
            "image_stem_b": str(row["image_stem_b"]),
            "date_a": str(row["date_a"]),
            "date_b": str(row["date_b"]),
        }
        if self.load_features:
            sample["z_a"] = self._load(row["cache_file_a"])
            sample["z_b"] = self._load(row["cache_file_b"])

        if self.spec.gdd_mode != "none":
            sample["gdd"] = torch.from_numpy(
                np.asarray([self.cond[k][i] for k in self.spec.gdd_keys()],
                           dtype=np.float32)
            )
        if self.spec.weather_mode != "none":
            w_key, m_key = self.spec.weather_keys()
            sample["weather"] = torch.from_numpy(
                np.asarray(self.cond[w_key][i], dtype=np.float32)
            )
            sample["weather_mask"] = torch.from_numpy(
                np.asarray(self.cond[m_key][i], dtype=bool)
            )

        for t in self.traits:
            value = row.get(f"{t}_b", np.nan)
            mask = bool(row.get(f"{t}_mask_b", False))
            sample[f"{t}_b"] = torch.tensor(
                float(value) if pd.notna(value) else float("nan"),
                dtype=torch.float32,
            )
            sample[f"{t}_mask_b"] = torch.tensor(mask, dtype=torch.bool)
            sample[f"{t}_b_lag"] = torch.tensor(
                float(row.get(f"{t}_b_lag", np.nan)), dtype=torch.float32
            )
            sample[f"{t}_b_date"] = (
                "" if pd.isna(row.get(f"{t}_b_date", None)) else str(row[f"{t}_b_date"])
            )
        return sample


def split_rows(
    index: pd.DataFrame,
    role_map: dict[str, str],
    role: str,
) -> pd.DataFrame:
    """Rows of official-train plots with ``role_map[plot_uid] == role``.

    ``_orig_row`` keeps the position in the frozen conditioning arrays; the
    returned frame is otherwise re-indexed to ``0..n-1``.
    """
    train = index[index.split == "train"].copy()
    train["_orig_row"] = np.arange(len(index))[index.split.to_numpy() == "train"]
    train["role"] = train.plot_uid.map(role_map)
    if train.role.isna().any():
        raise RuntimeError("a train plot is missing from the inner plot split")
    return train[train.role == role].drop(columns=["role"]).reset_index(drop=True)


def conditioning_rows(frame: pd.DataFrame) -> np.ndarray:
    """Original conditioning-array positions for a frame from :func:`split_rows`."""
    if "_orig_row" not in frame.columns:
        raise KeyError("frame has no _orig_row column; use split_rows()")
    return frame["_orig_row"].to_numpy(np.int64)


def build_role_map(index: pd.DataFrame, seed: int, fraction: float) -> tuple[dict, dict]:
    """80/20 official-train plot split, stratified by ``harvest_year``.

    Identical rule to the S3.2 probe protocol (``tools/s3_2_probe_eval.py``):
    plots are permuted with ``numpy.random.default_rng(seed)`` inside each year,
    and ``round(n_plots * (1 - fraction))`` plots go to the inner dev set.
    """
    tr = index[index.split == "train"]
    year = tr.groupby("plot_uid").harvest_year.first()
    rng = np.random.default_rng(seed)
    role_map: dict[str, str] = {}
    rows = []
    for y in sorted(year.unique()):
        plots = np.sort(year[year == y].index.to_numpy())
        perm = rng.permutation(len(plots))
        n_dev = max(1, int(round(len(plots) * (1.0 - fraction))))
        dev = set(plots[perm[:n_dev]])
        for p in plots:
            role = "inner_dev" if p in dev else "inner_train"
            role_map[str(p)] = role
            rows.append({"plot_uid": str(p), "harvest_year": int(y), "role": role})
    split = pd.DataFrame(rows).sort_values(["harvest_year", "plot_uid"]).reset_index(drop=True)
    if split.plot_uid.duplicated().any():
        raise RuntimeError("plot split contains duplicated plots")
    stats = {
        "seed": int(seed),
        "inner_train_fraction": float(fraction),
        "strata": "harvest_year",
        "n_plots": int(len(split)),
        "n_inner_train_plots": int((split.role == "inner_train").sum()),
        "n_inner_dev_plots": int((split.role == "inner_dev").sum()),
        "by_year": {
            str(y): {
                "n_plots": int(len(g)),
                "n_inner_train_plots": int((g.role == "inner_train").sum()),
                "n_inner_dev_plots": int((g.role == "inner_dev").sum()),
            }
            for y, g in split.groupby("harvest_year")
        },
        "rule": (
            "official-train plots only; stratified by harvest_year; the same "
            "split is reused by every condition, trait, scope and seed"
        ),
    }
    return role_map, stats
