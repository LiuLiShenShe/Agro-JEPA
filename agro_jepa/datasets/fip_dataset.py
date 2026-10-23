"""FIP 1.0 pair dataset (S2).

One sample = one ``(plot_uid, t_a, t_b)`` image pair with its variable-length
daily weather window and the phenotype at both ends.

Two observation paths, both fully traceable:

``visual`` (default)
    decode the real PNG from disk and apply the frozen ``mainline224``
    preprocessing.  Missing files raise a clear error; nothing is zero-filled.
``cached_features``
    read the S1 ``mainline224_s1-v1`` patch-token cache (256x768 float16).
    A missing cache file raises -- a missing cache is *never* treated as a
    valid zero feature vector.

Optional in-sample quality flags (``compute_image_stats``): mean/std of the
decoded luminance and a ``usable`` boolean from a rule that is *reported
before it is applied*.  S1 flagged 1 near-white image out of 97; the rule is
therefore documented and off by default, never silently enforced.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .fip_pairs import TRAITS
from .fip_weather import DailyWeatherStore, WeatherWindow
from .provenance import (
    CACHE_PROTOCOL,
    CACHE_PROTOCOL_VERSION,
    DINOV2_EMBED_DIM,
    MAINLINE_IMAGE_SIZE,
    MAINLINE_N_TOKENS,
    cache_key_for_image,
    build_mainline_transform,
)

IMAGE_MODES: tuple[str, ...] = ("visual", "cached_features")


def build_image_index(images_root: str | Path) -> dict[str, Path]:
    """Map filename stem -> local PNG path (one scan, reused for every sample).

    FIP stems embed the ``plot_uid`` and the acquisition timestamp, so they are
    unique across the dataset.  Scanning once avoids an ``rglob`` per image,
    which would be prohibitive for 10^5 samples.
    """
    root = Path(images_root)
    if not root.exists():
        return {}
    index: dict[str, Path] = {}
    for p in sorted(root.rglob("*.png")):
        index.setdefault(p.stem, p)
    return index


@dataclass
class ImageQualityRule:
    """Documented, reversible image-quality rule (not applied by default).

    S1 observed one near-white frame (FPWW0120667, 2016-02-16, mean luminance
    0.886 / std 0.038) out of 97 downloaded images; the next-brightest frame
    has mean 0.280, so a mean threshold of 0.60 separates it cleanly.  S2
    reports the flag and keeps the sample; filtering is a separate decision.
    """

    near_white_mean: float = 0.60
    near_white_std: float = 0.12
    min_luma_std: float = 0.01
    name: str = "near-white-luma-v1"

    def is_near_white(self, luma_mean: float, luma_std: float) -> bool:
        return bool((luma_mean >= self.near_white_mean)
                    and (luma_std <= self.near_white_std))

    def evaluate(self, luma_mean: float, luma_std: float) -> bool:
        """True when the frame may be used; False only for flagged frames."""
        if self.is_near_white(luma_mean, luma_std):
            return False
        return bool(luma_std >= self.min_luma_std)


class FIPPairDataset(Dataset):
    """Yield one dict per pair; see ``agro_jepa/datasets/fip_pairs.py``."""

    def __init__(
        self,
        pairs: pd.DataFrame,
        weather_store: DailyWeatherStore,
        image_mode: str = "visual",
        images_root: str | Path | None = None,
        feature_cache_dir: str | Path | None = None,
        transform: Callable | None = None,
        source_sha256: Callable[[str], str | None] | None = None,
        compute_image_stats: bool = False,
        quality_rule: ImageQualityRule | None = None,
        traits: Sequence[str] = TRAITS,
    ) -> None:
        if image_mode not in IMAGE_MODES:
            raise ValueError(f"image_mode must be one of {IMAGE_MODES}, got {image_mode!r}")
        if image_mode == "cached_features" and (
            feature_cache_dir is None
            or (source_sha256 is None and images_root is None)
        ):
            raise ValueError(
                "image_mode='cached_features' needs feature_cache_dir and either "
                "images_root (to hash the local file) or a source_sha256 callable"
            )
        self.pairs = pairs.reset_index(drop=True)
        self.weather_store = weather_store
        self.image_mode = image_mode
        self.images_root = Path(images_root) if images_root else None
        self.image_index = build_image_index(self.images_root) if self.images_root else {}
        self.feature_cache_dir = Path(feature_cache_dir) if feature_cache_dir else None
        self.transform = transform if transform is not None else build_mainline_transform()
        self.source_sha256 = source_sha256
        self.compute_image_stats = bool(compute_image_stats)
        self.quality_rule = quality_rule or ImageQualityRule()
        self.traits = tuple(traits)

    # -- sizes --------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.pairs)

    @property
    def image_size(self) -> int:
        return MAINLINE_IMAGE_SIZE

    @property
    def n_tokens(self) -> int:
        return MAINLINE_N_TOKENS

    @property
    def emb_dim(self) -> int:
        return DINOV2_EMBED_DIM

    def splits(self) -> dict[str, int]:
        return {str(k): int(v) for k, v in self.pairs.split.value_counts().items()}

    # -- image helpers ------------------------------------------------------
    @staticmethod
    def _stem(hf_url: str) -> str:
        return str(hf_url).rsplit("/", 1)[-1].rsplit(".", 1)[0]

    def _local_image(self, hf_url: str) -> Path:
        if self.images_root is None:
            raise ValueError("images_root is required for image_mode='visual'")
        p = self.image_index.get(self._stem(hf_url))
        if p is None:
            raise FileNotFoundError(
                f"image not found under {self.images_root}: {hf_url}"
            )
        return p

    def _load_tensor(self, hf_url: str) -> tuple[torch.Tensor, dict]:
        path = self._local_image(hf_url)
        img = Image.open(path).convert("RGB")
        stats: dict = {"image_path": str(path)}
        if self.compute_image_stats:
            arr = np.asarray(img, dtype=np.float32) / 255.0
            luma = arr.mean(axis=2)
            mean = float(luma.mean())
            std = float(luma.std())
            stats.update({
                "luma_mean": mean,
                "luma_std": std,
                "luma_min": float(luma.min()),
                "luma_max": float(luma.max()),
                "width": int(img.width),
                "height": int(img.height),
                "quality_rule": self.quality_rule.name,
                "near_white": self.quality_rule.is_near_white(mean, std),
                "quality_usable": self.quality_rule.evaluate(mean, std),
            })
        return self.transform(img), stats

    def _source_sha(self, hf_url: str) -> str:
        if self.source_sha256 is not None:
            sha = self.source_sha256(hf_url)
        else:
            p = self._local_image(hf_url)
            sha = hashlib.sha256(p.read_bytes()).hexdigest()
        if not sha:
            raise FileNotFoundError(f"cannot resolve the source bytes for {hf_url}")
        return sha

    def _load_cached(self, hf_url: str) -> tuple[torch.Tensor, dict]:
        assert self.feature_cache_dir is not None
        sha = self._source_sha(hf_url)
        key = cache_key_for_image(sha)
        path = self.feature_cache_dir / f"{CACHE_PROTOCOL}_{CACHE_PROTOCOL_VERSION}" / f"{key}.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"missing feature cache for {hf_url} (expected {path}). "
                "A missing cache is not a zero feature vector."
            )
        arr = np.load(path)
        if arr.shape != (MAINLINE_N_TOKENS, DINOV2_EMBED_DIM):
            raise ValueError(
                f"cached features for {hf_url} have shape {arr.shape}, expected "
                f"({MAINLINE_N_TOKENS}, {DINOV2_EMBED_DIM}) for protocol "
                f"{CACHE_PROTOCOL}_{CACHE_PROTOCOL_VERSION}"
            )
        return torch.from_numpy(arr.astype(np.float32)), {
            "feature_cache_file": str(path),
            "feature_cache_key": key,
            "feature_source_sha256": sha,
            "feature_protocol": f"{CACHE_PROTOCOL}_{CACHE_PROTOCOL_VERSION}",
        }

    # -- main ---------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict:
        row = self.pairs.iloc[idx]

        if self.image_mode == "visual":
            img_a, meta_a = self._load_tensor(row.image_a)
            img_b, meta_b = self._load_tensor(row.image_b)
        else:
            img_a, meta_a = self._load_cached(row.image_a)
            img_b, meta_b = self._load_cached(row.image_b)

        win: WeatherWindow = self.weather_store.window(
            row.harvest_year, row.date_a, row.date_b
        )

        sample: dict = {
            "image_a": img_a,
            "image_b": img_b,
            "weather_daily": torch.from_numpy(win.values),        # (L, C) float32
            "weather_mask": torch.from_numpy(win.mask),           # (L,) bool
            "weather_n_hours_min": torch.from_numpy(win.n_hours_min.astype(np.int32)),
            "delta_days": torch.tensor(int(row.delta_days), dtype=torch.int64),
            "plot_uid": str(row.plot_uid),
            "genotype_id": str(row.genotype_id),
            "harvest_year": int(row.harvest_year),
            "split": str(row.split),
            "pair_id": str(row.pair_id),
            "pair_type": str(row.pair_type),
            "image_dates": (str(row.date_a), str(row.date_b)),
            "weather_date_range": (
                win.start_exclusive.isoformat(),
                win.end_inclusive.isoformat(),
            ),
            "weather_dates": tuple(d.isoformat() for d in win.dates),
            "weather_manifest_id": win.manifest_id,
            "image_stem_a": str(row.image_stem_a),
            "image_stem_b": str(row.image_stem_b),
            "trait_names": tuple(self.traits),
        }

        # phenotype / label vectors, ordered exactly as ``self.traits``
        ph_a = np.full(len(self.traits), np.nan, dtype=np.float32)
        ph_b = np.full(len(self.traits), np.nan, dtype=np.float32)
        mask_a = np.zeros(len(self.traits), dtype=bool)
        mask_b = np.zeros(len(self.traits), dtype=bool)
        lag = np.full((2, len(self.traits)), np.nan, dtype=np.float32)
        label_dates = np.empty((2, len(self.traits)), dtype=object)
        label_dates[:] = None
        for k, t in enumerate(self.traits):
            va, vb = row[f"{t}_a"], row[f"{t}_b"]
            if pd.notna(va):
                ph_a[k] = float(va)
            if pd.notna(vb):
                ph_b[k] = float(vb)
            mask_a[k] = bool(row[f"{t}_mask_a"])
            mask_b[k] = bool(row[f"{t}_mask_b"])
            for end, col in ((0, f"{t}_a"), (1, f"{t}_b")):
                if pd.notna(row[f"{col}_lag"]):
                    lag[end, k] = float(row[f"{col}_lag"])
                if pd.notna(row[f"{col}_date"]):
                    label_dates[end, k] = str(row[f"{col}_date"])
        sample.update({
            "phenotype_a": torch.from_numpy(ph_a),            # (T,) float32, NaN = missing
            "phenotype_b": torch.from_numpy(ph_b),
            "phenotype_mask_a": torch.from_numpy(mask_a),     # (T,) bool
            "phenotype_mask_b": torch.from_numpy(mask_b),
            "label_lag_days": torch.from_numpy(lag),          # (2, T) float32
            "label_dates": label_dates,                       # (2, T) object
            "ph_negative_a": bool(row.ph_negative_a),
            "ph_negative_b": bool(row.ph_negative_b),
        })

        if self.compute_image_stats:
            sample["image_stats_a"] = meta_a
            sample["image_stats_b"] = meta_b

        return sample
