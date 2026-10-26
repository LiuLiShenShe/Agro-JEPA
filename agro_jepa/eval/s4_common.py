"""S4A shared evaluation helpers: checkpoint loading and pooled-feature passes.

The feature-collection rule is copied from S3 ``eval_v0.collect_pooled_features``
so the E2 readout sees exactly the S3/S3.2 sample construction: one row per
``(plot_uid, image_stem_b, pair_type)``, keeping the closest current image
(smallest ``delta_days``) deterministically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..datasets.s4_dataset import S4PairDataset, conditioning_rows
from ..models.fip_dino_wm import move_attention_masks
from ..models.s4_conditioning import VectorNormStats
from ..models.s4_predictor import (
    S4ModelConfig,
    build_s4_predictor,
    conditioning_spec,
    has_fixed_delta,
)
from ..models.time_embedding import DeltaNormStats


def _vector_norm(d: dict | None) -> VectorNormStats | None:
    if d is None:
        return None
    return VectorNormStats(
        mean=tuple(d["mean"]), std=tuple(d["std"]), n_samples=int(d.get("n_samples", 0))
    )


def load_s4_model(ckpt_path: str, device: torch.device):
    """Rebuild a condition model from its checkpoint; returns (model, ckpt)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if ckpt.get("format") != "agro-jepa-s4a-state-dict-v1":
        raise ValueError(f"{ckpt_path} is not an S4A state-dict checkpoint")
    model_cfg = S4ModelConfig.from_dict(ckpt["model_config"])
    spec = conditioning_spec({
        "use_delta": ckpt["conditioning_spec"]["use_delta"],
        "delta_mode": ckpt["conditioning_spec"].get("delta_mode", "sample"),
        "fixed_delta_days": ckpt["conditioning_spec"].get("fixed_delta_days"),
        "gdd_mode": ckpt["conditioning_spec"]["gdd_mode"],
        "gdd_base": ckpt["conditioning_spec"]["gdd_base"],
        "weather_mode": ckpt["conditioning_spec"]["weather_mode"],
    })
    dn = ckpt.get("delta_norm")
    delta_norm = DeltaNormStats(
        mean=dn["mean"], std=dn["std"], min=dn["min"], max=dn["max"],
        n_samples=dn["n_samples"],
    ) if dn else None
    model = build_s4_predictor(
        model_cfg, spec,
        delta_norm=delta_norm,
        gdd_norm=_vector_norm(ckpt.get("gdd_norm")),
        weather_norm=_vector_norm(ckpt.get("weather_norm")),
    )
    state = dict(ckpt["model"])
    if "fixed_delta_days" not in state:
        # Checkpoints written before S4A.1 do not carry the (empty) buffer that
        # the constant-interval arm uses.  A *constant* arm without it would
        # silently lose its fixed interval, so that combination is refused.
        if spec.delta_mode == "constant":
            raise ValueError(
                f"{ckpt_path}: delta_mode='constant' but the checkpoint has no "
                "fixed_delta_days buffer"
            )
        state["fixed_delta_days"] = torch.zeros(0, dtype=torch.float32)
    model.load_state_dict(state)
    model.to(device)
    move_attention_masks(model.predictor, device)
    model.eval()
    return model, ckpt


def forward_batch(model, batch: dict, device: torch.device) -> torch.Tensor:
    parts = getattr(model, "parts", [])
    kwargs = {}
    if "delta" in parts and not has_fixed_delta(model):
        kwargs["delta_days"] = batch["delta_days"].to(device, non_blocking=True)
    if "gdd" in parts:
        kwargs["gdd"] = batch["gdd"].to(device, non_blocking=True)
    if "weather" in parts:
        kwargs["weather"] = batch["weather"].to(device, non_blocking=True)
        kwargs["weather_mask"] = batch["weather_mask"].to(device, non_blocking=True)
    return model(batch["z_a"].to(device, non_blocking=True), **kwargs)


def make_dataset(frame: pd.DataFrame, cond: dict, spec, traits, load_features=True):
    rows = (conditioning_rows(frame) if "_orig_row" in frame.columns
            else np.arange(len(frame), dtype=np.int64))
    sliced = {}
    n = cond["delta_days"].shape[0]
    for k, v in cond.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n:
            sliced[k] = v[rows]
        else:
            sliced[k] = v
    return S4PairDataset(frame, sliced, spec, traits, load_features=load_features)


@torch.no_grad()
def mean_future_feature(
    frame: pd.DataFrame, cond: dict, device: torch.device, batch_size: int = 512
) -> torch.Tensor:
    """Mean ``z_b`` over the given rows (used for the B1 baseline)."""
    # load_features only; no conditioning needed
    rows = conditioning_rows(frame)
    ds = S4PairDataset(
        frame, {k: (v[rows] if isinstance(v, np.ndarray) and v.ndim >= 1
                    and v.shape[0] == cond["delta_days"].shape[0] else v)
                for k, v in cond.items()},
        conditioning_spec({}), traits=(), load_features=True,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4,
                        pin_memory=True)
    total, n = None, 0
    for batch in loader:
        z_b = batch["z_b"].to(device, non_blocking=True).float()
        total = z_b.sum(dim=0) if total is None else total + z_b.sum(dim=0)
        n += z_b.shape[0]
    return (total / max(n, 1)).cpu()


@torch.no_grad()
def collect_pooled(
    models: list,
    frame: pd.DataFrame,
    cond: dict,
    spec,
    traits,
    device: torch.device,
    batch_size: int = 512,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """One pass over ``frame``; predictions are averaged over the given seeds.

    Returns ``(metadata, pooled_current (mean z_a), pooled_predicted
    (mean of the seed-averaged z_hat_b))``.  Dedupe is the S3 rule.
    """
    ds = make_dataset(frame, cond, spec, traits)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4,
                        pin_memory=True)
    pool_a, pool_b, meta = [], [], []
    for batch in loader:
        z_a = batch["z_a"].to(device, non_blocking=True)
        z_pred = None
        for m in models:
            out = forward_batch(m, batch, device).float()
            z_pred = out if z_pred is None else z_pred + out
        z_pred = z_pred / len(models)
        pool_a.append(z_a.float().mean(dim=1).cpu().numpy())
        pool_b.append(z_pred.mean(dim=1).cpu().numpy())
        for i in range(z_a.shape[0]):
            row = {
                "pair_id": batch["pair_id"][i],
                "plot_uid": batch["plot_uid"][i],
                "split": batch["split"][i],
                "pair_type": batch["pair_type"][i],
                "span_days": int(batch["span_days"][i]),
                "delta_days": int(float(batch["delta_days"][i])),
                "harvest_year": int(batch["harvest_year"][i]),
                "image_stem_a": batch["image_stem_a"][i],
                "image_stem_b": batch["image_stem_b"][i],
                "date_b": batch["date_b"][i],
            }
            for t in traits:
                row[f"{t}_b"] = float(batch[f"{t}_b"][i])
                row[f"{t}_mask_b"] = bool(batch[f"{t}_mask_b"][i])
                row[f"{t}_b_lag"] = float(batch[f"{t}_b_lag"][i])
                row[f"{t}_b_date"] = str(batch[f"{t}_b_date"][i])
            meta.append(row)
    pooled_a = np.concatenate(pool_a, axis=0).astype(np.float32)
    pooled_b = np.concatenate(pool_b, axis=0).astype(np.float32)
    out = pd.DataFrame(meta)
    out["_orig"] = np.arange(len(out))
    out = out.sort_values(
        ["split", "plot_uid", "image_stem_b", "pair_type", "delta_days", "pair_id"],
        kind="mergesort",
    ).drop_duplicates(["plot_uid", "image_stem_b", "pair_type"], keep="first")
    out = out.reset_index(drop=True)
    keep = out["_orig"].to_numpy()
    out = out.drop(columns=["_orig"])
    out["_pool_index"] = np.arange(len(out))
    return out, pooled_a[keep], pooled_b[keep]
