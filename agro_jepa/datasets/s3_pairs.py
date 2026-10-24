"""S3: frozen train/validation pair index and cached-feature dataset.

The S3 index is a *derived* artifact.  It never modifies the S1 manifests or the
S2 pair table; it filters the S2 table down to the pairs whose two physical
images were actually downloaded, then freezes that row set in a new parquet.

Rules implemented here (S3 protocol §3)
---------------------------------------
* official ``train`` / ``validation`` only (the official test splits are never
  read for model selection);
* initial pair families ``span7`` and ``span14`` (S2 tolerance +/-2 d), with
  ``delta_days`` kept as the real interval -- the span name is only a label;
* one row per ``(plot_uid, image_stem_a, image_stem_b)`` even if the same
  physical pair appears under several ``pair_type`` values;
* both images must exist in the S3 download manifest and in the feature cache;
* phenotype targets at ``t_b`` are attached verbatim (raw value, actual
  measurement date, signed lag, validity mask); negatives/zeros/missing are not
  altered.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

TRAITS: tuple[str, ...] = ("cc", "ph", "whc", "sen")
DEFAULT_PAIR_TYPES: tuple[str, ...] = ("span7", "span14")
ELIGIBLE_SPLITS: tuple[str, ...] = ("train", "validation")


@dataclass(frozen=True)
class S3IndexRule:
    pair_types: tuple[str, ...] = DEFAULT_PAIR_TYPES
    splits: tuple[str, ...] = ELIGIBLE_SPLITS
    dedupe_physical_pairs: bool = True
    require_cache: bool = True
    manifest_id: str = "fip1-s3-v1"

    def as_dict(self) -> dict:
        return {
            "manifest_id": self.manifest_id,
            "pair_types": list(self.pair_types),
            "splits": list(self.splits),
            "dedupe_physical_pairs": self.dedupe_physical_pairs,
            "require_cache": self.require_cache,
        }


# --------------------------------------------------------------------------
# loading helpers
# --------------------------------------------------------------------------
def load_download_selection(path: str | Path) -> pd.DataFrame:
    """Read ``s3_image_selection.json`` into one row per downloaded image."""
    with open(path) as fh:
        meta = json.load(fh)
    rows = meta.get("images", [])
    if not rows:
        raise ValueError(f"{path} contains no images")
    df = pd.DataFrame(rows)
    for col in ("plot_uid", "split", "harvest_year", "image_date",
                "image_filename_stem", "dest", "sha256", "status"):
        if col not in df.columns:
            raise ValueError(f"{path} image records are missing {col!r}")
    df["image_filename_stem"] = df["image_filename_stem"].astype(str).str.upper()
    df["plot_uid"] = df["plot_uid"].astype(str)
    return df


def load_feature_index(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {"image_filename_stem", "cache_file", "sha256", "image_date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    df["image_filename_stem"] = df["image_filename_stem"].astype(str).str.upper()
    return df


def load_s2_pairs(
    path: str | Path,
    pair_types: Sequence[str] = DEFAULT_PAIR_TYPES,
    splits: Sequence[str] = ELIGIBLE_SPLITS,
) -> pd.DataFrame:
    cols = [
        "pair_id", "pair_type", "span_days", "plot_uid", "genotype_id",
        "harvest_year", "split", "image_stem_a", "image_stem_b", "date_a",
        "date_b", "delta_days", "window_n_days",
        *[f"{t}_{end}" for t in TRAITS for end in ("a", "b")],
        *[f"{t}_{end}_lag" for t in TRAITS for end in ("a", "b")],
        *[f"{t}_{end}_date" for t in TRAITS for end in ("a", "b")],
        *[f"{t}_mask_{end}" for t in TRAITS for end in ("a", "b")],
        "ph_negative_a", "ph_negative_b",
    ]
    df = pd.read_parquet(path, columns=cols)
    df = df[df.pair_type.isin(pair_types) & df.split.isin(splits)].copy()
    df["image_stem_a"] = df["image_stem_a"].astype(str).str.upper()
    df["image_stem_b"] = df["image_stem_b"].astype(str).str.upper()
    df["image_a"] = df["image_stem_a"]
    df["image_b"] = df["image_stem_b"]
    df["pair_type"] = df["pair_type"].astype(str)
    df["split"] = df["split"].astype(str)
    return df


# --------------------------------------------------------------------------
# index construction
# --------------------------------------------------------------------------
def build_s3_index(
    pairs: pd.DataFrame,
    selection: pd.DataFrame,
    features: pd.DataFrame,
    rule: S3IndexRule | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Filter and deduplicate the S2 pairs; returns (index, statistics)."""
    rule = rule or S3IndexRule()

    n_s2 = len(pairs)
    # pair-type / split filtering is already applied by load_s2_pairs, but keep
    # the rule explicit so the function is usable with an arbitrary frame
    pairs = pairs[
        pairs.pair_type.isin(rule.pair_types) & pairs.split.isin(rule.splits)
    ].copy()
    n_after_type_split = len(pairs)

    # images actually downloaded, deduplicated by physical stem
    dl = selection.drop_duplicates("image_filename_stem").set_index("image_filename_stem")
    has_a = pairs.image_stem_a.isin(dl.index)
    has_b = pairs.image_stem_b.isin(dl.index)
    n_before_images = len(pairs)
    pairs = pairs[has_a & has_b].copy()
    n_after_images = len(pairs)

    # deduplicate physical image pairs across pair types
    key = ["plot_uid", "image_stem_a", "image_stem_b"]
    n_before_dedupe = len(pairs)
    if rule.dedupe_physical_pairs:
        # keep the pair with the longest span first, then pair_id -- deterministic
        pairs = pairs.sort_values(
            ["plot_uid", "image_stem_a", "image_stem_b", "span_days", "pair_id"],
            kind="mergesort",
        )
        pairs = pairs.drop_duplicates(subset=key, keep="last")
    n_after_dedupe = len(pairs)

    # feature cache paths
    feat = features.drop_duplicates("image_filename_stem").set_index("image_filename_stem")
    cache_a = pairs.image_stem_a.map(feat["cache_file"])
    cache_b = pairs.image_stem_b.map(feat["cache_file"])
    if rule.require_cache:
        ok = cache_a.notna() & cache_b.notna()
        pairs = pairs[ok].copy()
        cache_a = cache_a[ok]
        cache_b = cache_b[ok]
    pairs["cache_file_a"] = cache_a.astype(str)
    pairs["cache_file_b"] = cache_b.astype(str)

    # image-level metadata from the download selection
    for end, col in (("a", "image_stem_a"), ("b", "image_stem_b")):
        stem = pairs[col]
        pairs[f"image_path_{end}"] = stem.map(dl["dest"])
        pairs[f"image_sha256_{end}"] = stem.map(dl["sha256"])
        pairs[f"image_split_{end}"] = stem.map(dl["split"])
        pairs[f"image_year_{end}"] = stem.map(dl["harvest_year"])
        pairs[f"image_date_{end}"] = stem.map(dl["image_date"])

    pairs["delta_days"] = pairs["delta_days"].astype(int)
    pairs["span_days"] = pairs["span_days"].astype(int)
    pairs = pairs.sort_values(["split", "pair_type", "delta_days", "pair_id"],
                              kind="mergesort").reset_index(drop=True)

    stats = index_statistics(pairs, rule)
    stats["filtering"] = {
        "input_rows": int(n_s2),
        "after_pair_type_and_split": int(n_after_type_split),
        "rows_before_image_filter": int(n_before_images),
        "rows_after_image_filter": int(n_after_images),
        "rows_before_dedupe": int(n_before_dedupe),
        "rows_after_dedupe": int(n_after_dedupe),
        "final_rows": int(len(pairs)),
    }
    return pairs, stats


def index_statistics(pairs: pd.DataFrame, rule: S3IndexRule) -> dict:
    out: dict = {"rule": rule.as_dict(), "n_pairs": int(len(pairs))}
    for split in sorted(pairs.split.unique()):
        sub = pairs[pairs.split == split]
        imgs = pd.unique(pd.concat([sub.image_stem_a, sub.image_stem_b]))
        d = sub.delta_days.to_numpy()
        out[split] = {
            "n_pairs": int(len(sub)),
            "n_unique_images": int(len(imgs)),
            "n_unique_physical_pairs": int(
                sub[["plot_uid", "image_stem_a", "image_stem_b"]].drop_duplicates().shape[0]
            ),
            "n_plots": int(sub.plot_uid.nunique()),
            "n_years": int(sub.harvest_year.nunique()),
            "n_by_pair_type": {str(k): int(v) for k, v in sub.pair_type.value_counts().items()},
            "n_by_year": {str(k): int(v) for k, v in sub.harvest_year.value_counts().sort_index().items()},
            "delta_days": {
                "min": int(d.min()) if len(d) else None,
                "max": int(d.max()) if len(d) else None,
                "median": float(np.median(d)) if len(d) else None,
                "mean": float(d.mean()) if len(d) else None,
                "hist_1_3": int(((d >= 1) & (d <= 3)).sum()),
                "hist_4_7": int(((d >= 4) & (d <= 7)).sum()),
                "hist_8_14": int(((d >= 8) & (d <= 14)).sum()),
                "hist_15_21": int(((d >= 15) & (d <= 21)).sum()),
                "hist_22_28": int(((d >= 22) & (d <= 28)).sum()),
                "hist_29_plus": int((d >= 29).sum()),
            },
            "delta_by_pair_type": {
                str(k): {
                    "n": int(len(g)),
                    "min": int(g.delta_days.min()),
                    "max": int(g.delta_days.max()),
                    "median": float(g.delta_days.median()),
                }
                for k, g in sub.groupby("pair_type")
            },
            "labels": {
                t: {
                    "n_valid_b": int(sub[f"{t}_mask_b"].sum()),
                    "n_valid_a": int(sub[f"{t}_mask_a"].sum()),
                    "n_valid_both": int((sub[f"{t}_mask_a"] & sub[f"{t}_mask_b"]).sum()),
                    "n_negative_b": int(
                        ((sub[f"{t}_b"] < 0) & sub[f"{t}_mask_b"]).sum()
                    ),
                }
                for t in TRAITS
            },
        }
    return out


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------
class S3FeaturePairDataset(Dataset):
    """Cached DINOv2 patch-token pairs for V0 training and evaluation.

    Returns the current feature ``z_a``, the future feature ``z_b`` (target,
    stop-gradient), the real interval ``delta_days``, and the future phenotype
    labels used only by the S3 probes.  No future image, future feature, future
    phenotype or target metadata is ever returned in the predictor input.
    """

    def __init__(
        self,
        index: pd.DataFrame,
        traits: Sequence[str] = ("cc", "ph"),
        load_features: bool = True,
    ) -> None:
        required = {
            "pair_id", "plot_uid", "split", "pair_type", "span_days",
            "delta_days", "cache_file_a", "cache_file_b", "image_stem_a",
            "image_stem_b", "date_a", "date_b", "harvest_year",
        }
        missing = required - set(index.columns)
        if missing:
            raise ValueError(f"index is missing columns {sorted(missing)}")
        self.index = index.reset_index(drop=True)
        self.traits = tuple(traits)
        self.load_features = bool(load_features)

    def __len__(self) -> int:
        return len(self.index)

    @staticmethod
    def _load(path: str) -> torch.Tensor:
        arr = np.load(path)
        if arr.ndim != 2:
            raise ValueError(f"feature cache {path} has shape {arr.shape}, expected 2-D")
        return torch.from_numpy(arr.astype(np.float32))

    def __getitem__(self, idx: int) -> dict:
        row = self.index.iloc[idx]
        sample: dict = {
            "pair_id": str(row.pair_id),
            "plot_uid": str(row.plot_uid),
            "genotype_id": str(row.get("genotype_id", "")),
            "split": str(row.split),
            "pair_type": str(row.pair_type),
            "span_days": int(row.span_days),
            "delta_days": torch.tensor(float(row.delta_days), dtype=torch.float32),
            "harvest_year": int(row.harvest_year),
            "image_stem_a": str(row.image_stem_a),
            "image_stem_b": str(row.image_stem_b),
            "date_a": str(row.date_a),
            "date_b": str(row.date_b),
        }
        if self.load_features:
            sample["z_a"] = self._load(row.cache_file_a)
            sample["z_b"] = self._load(row.cache_file_b)

        for t in self.traits:
            value = row.get(f"{t}_b", np.nan)
            mask = bool(row.get(f"{t}_mask_b", False))
            sample[f"{t}_b"] = torch.tensor(
                float(value) if pd.notna(value) else float("nan"), dtype=torch.float32
            )
            sample[f"{t}_mask_b"] = torch.tensor(mask, dtype=torch.bool)
            sample[f"{t}_b_lag"] = torch.tensor(
                float(row.get(f"{t}_b_lag", np.nan)), dtype=torch.float32
            )
            sample[f"{t}_b_date"] = (
                "" if pd.isna(row.get(f"{t}_b_date", None)) else str(row[f"{t}_b_date"])
            )
            sample[f"{t}_a"] = torch.tensor(
                float(row.get(f"{t}_a", np.nan)), dtype=torch.float32
            )
            sample[f"{t}_mask_a"] = torch.tensor(
                bool(row.get(f"{t}_mask_a", False)), dtype=torch.bool
            )
        return sample


def index_sha256(index: pd.DataFrame) -> str:
    """Stable digest of the frozen index (column order independent)."""
    cols = sorted(index.columns)
    payload = index[cols].to_csv(index=False).encode()
    return hashlib.sha256(payload).hexdigest()
