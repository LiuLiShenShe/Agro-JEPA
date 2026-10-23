"""Collate function for FIP pairs with variable-length weather windows.

PyTorch's ``default_collate`` rejects unequal-length sequence samples
(``torch/utils/data/_utils/collate.py``: "each element in list of batch should
be of equal size"), and it cannot handle the object arrays used for label
dates.  This collate therefore does the padding explicitly:

* ``weather_daily``  -> ``(B, L_max, C)`` float32, zero-padded
* ``weather_mask``   -> ``(B, L_max)`` bool, ``False`` on padding
* ``weather_dates``  -> ``(B, L_max)`` object, ``None`` on padding
* ``weather_lengths``-> ``(B,)`` int64, the true per-sample length

Padding values are 0.0 *and* masked; no downstream code may treat an unmasked
padded step as an observation.  A sample whose ``weather_lengths`` entry is 0
would mean an empty window and is rejected by the pair rule upstream.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

_META_KEYS = (
    "plot_uid",
    "genotype_id",
    "split",
    "pair_id",
    "pair_type",
    "image_stem_a",
    "image_stem_b",
    "image_dates",
    "weather_date_range",
    "weather_manifest_id",
)


def fip_collate_fn(batch: Sequence[dict]) -> dict:
    if not batch:
        raise ValueError("empty batch")
    out: dict = {}
    b = len(batch)

    image_like = ("image_a", "image_b")
    for k in image_like:
        if all(isinstance(s[k], torch.Tensor) for s in batch):
            out[k] = torch.stack([s[k] for s in batch], dim=0)
        else:
            out[k] = None

    # ---- variable-length weather -----------------------------------------
    lengths = np.array([int(s["weather_daily"].shape[0]) for s in batch], dtype=np.int64)
    l_max = int(lengths.max())
    c = int(batch[0]["weather_daily"].shape[1])
    for s in batch:
        if int(s["weather_daily"].shape[1]) != c:
            raise ValueError("weather channel count differs within a batch")

    weather = torch.zeros((b, l_max, c), dtype=torch.float32)
    wmask = torch.zeros((b, l_max), dtype=torch.bool)
    whours = torch.zeros((b, l_max), dtype=torch.int32)
    wdates = np.full((b, l_max), None, dtype=object)
    for i, s in enumerate(batch):
        n = int(s["weather_daily"].shape[0])
        weather[i, :n] = s["weather_daily"]
        wmask[i, :n] = s["weather_mask"]
        whours[i, :n] = s["weather_n_hours_min"]
        dates = list(s["weather_dates"]) if s.get("weather_dates") else \
            _window_dates(s["weather_date_range"], n)
        if len(dates) != n:
            raise ValueError("weather_dates length disagrees with weather_daily")
        wdates[i, :n] = dates
    out["weather_daily"] = weather
    out["weather_mask"] = wmask
    out["weather_n_hours_min"] = whours
    out["weather_dates"] = wdates
    out["weather_lengths"] = torch.from_numpy(lengths)

    # ---- scalar / vector fields ------------------------------------------
    out["delta_days"] = torch.stack([s["delta_days"] for s in batch], dim=0)
    out["harvest_year"] = torch.tensor([s["harvest_year"] for s in batch], dtype=torch.int64)
    out["phenotype_a"] = torch.stack([s["phenotype_a"] for s in batch], dim=0)
    out["phenotype_b"] = torch.stack([s["phenotype_b"] for s in batch], dim=0)
    out["phenotype_mask_a"] = torch.stack([s["phenotype_mask_a"] for s in batch], dim=0)
    out["phenotype_mask_b"] = torch.stack([s["phenotype_mask_b"] for s in batch], dim=0)
    out["label_lag_days"] = torch.stack([s["label_lag_days"] for s in batch], dim=0)
    out["label_dates"] = [s["label_dates"] for s in batch]
    out["ph_negative_a"] = torch.tensor([bool(s["ph_negative_a"]) for s in batch])
    out["ph_negative_b"] = torch.tensor([bool(s["ph_negative_b"]) for s in batch])
    out["trait_names"] = list(batch[0]["trait_names"])
    for k in _META_KEYS:
        if k in batch[0]:
            out[k] = [s.get(k) for s in batch]

    if "image_stats_a" in batch[0]:
        out["image_stats_a"] = [s["image_stats_a"] for s in batch]
        out["image_stats_b"] = [s["image_stats_b"] for s in batch]
    return out


def _window_dates(date_range: Sequence[str], n_days: int) -> list[str]:
    """Rebuild the ISO dates of a ``(t_a, t_b]`` window (t_a+1 .. t_b)."""
    from datetime import date, timedelta

    start = date.fromisoformat(str(date_range[0]))
    return [(start + timedelta(days=k + 1)).isoformat() for k in range(n_days)]
