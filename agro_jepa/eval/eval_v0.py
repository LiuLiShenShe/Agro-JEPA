#!/usr/bin/env python3
"""S3 V0 evaluation: future-latent prediction (E1) and phenotype probes (E2).

E1 -- future latent prediction on the official validation split
    * V0 MSE vs **Persistence** MSE (``z_a`` used as the future feature);
    * mean cosine similarity for both;
    * grouped by nominal span (span7 / span14), harvest year and delta bin;
    * per-sample rows are written to parquet so every number is auditable.

E2 -- future CC / PH linear probes
    Two readouts with the **same architecture and training budget**:
      A. current DINOv2 feature ``mean(Z_a)`` -> future CC/PH
      B. V0 predicted future feature ``mean(Z_hat_b)`` -> future CC/PH
    Targets are the raw phenotype at ``t_b`` with S1/S2's ``|lag|<=7 d`` mask.
    Probe samples are de-duplicated to one row per ``(plot_uid, image_stem_b)``
    (smallest ``delta_days``) so repeated future targets do not re-weight plots.

Usage
-----
    CUDA_VISIBLE_DEVICES=1 python3 agro_jepa/eval/eval_v0.py \
        --config agro_jepa/config/s3_v0.yaml \
        --checkpoint external_data/s3_runs/v0/checkpoints/best.pt \
        --report reports/s3_v0_results.json \
        --predictions reports/s3_v0_val_predictions.parquet
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agro_jepa.datasets.s3_pairs import (  # noqa: E402
    S3FeaturePairDataset,
    index_sha256,
)
from agro_jepa.models.fip_dino_wm import (  # noqa: E402
    FIPDinoWMConfig,
    FIPDinoWM,
    move_attention_masks,
)
from agro_jepa.models.time_embedding import DeltaNormStats  # noqa: E402


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(cfg: dict, ckpt_path: Path, device: torch.device) -> FIPDinoWM:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if ckpt.get("format") != "agro-jepa-s3-v0-state-dict-v1":
        raise ValueError(f"{ckpt_path} is not an S3 V0 state-dict checkpoint")
    dn = ckpt["delta_norm"]
    norm = DeltaNormStats(
        mean=dn["mean"], std=dn["std"], min=dn["min"], max=dn["max"],
        n_samples=dn["n_samples"],
    )
    model_cfg = FIPDinoWMConfig.from_dict(cfg["model"])
    model = FIPDinoWM(model_cfg, norm_stats=norm).to(device)
    model.load_state_dict(ckpt["model"])
    move_attention_masks(model.predictor, device)
    model.eval()
    return model, ckpt


def delta_bin(d: int) -> str:
    if d <= 3:
        return "1-3"
    if d <= 7:
        return "4-7"
    if d <= 14:
        return "8-14"
    if d <= 21:
        return "15-21"
    if d <= 28:
        return "22-28"
    return "29+"


def summarize(df: pd.DataFrame, group_cols: list[str]) -> dict:
    out: dict = {}
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key = "|".join(str(k) for k in keys)
        mse_pred = float(g.mse_pred.mean())
        mse_persist = float(g.mse_persist.mean())
        mse_mean = float(g.mse_train_mean.mean())
        mse_delta_mean = float(g.mse_train_delta_mean.mean())
        out[key] = {
            "n_samples": int(len(g)),
            "n_plots": int(g.plot_uid.nunique()),
            "mse_pred": mse_pred,
            "mse_persist": mse_persist,
            "mse_train_mean": mse_mean,
            "mse_train_delta_mean": mse_delta_mean,
            "relative_improvement": (mse_persist - mse_pred) / mse_persist
            if mse_persist > 0 else None,
            "relative_improvement_vs_train_mean": (
                (mse_mean - mse_pred) / mse_mean if mse_mean > 0 else None
            ),
            "cosine_pred": float(g.cosine_pred.mean()),
            "cosine_persist": float(g.cosine_persist.mean()),
        }
    return out


# --------------------------------------------------------------------------
# E1
# --------------------------------------------------------------------------
@torch.no_grad()
def train_feature_baselines(
    train_df: pd.DataFrame,
    cfg: dict,
    device: torch.device,
) -> tuple[torch.Tensor, dict[int, torch.Tensor], dict]:
    """Train-split mean future feature, global and per real ``delta_days``.

    These are stronger trivial baselines than persistence: a model that only
    learns the average future feature (ignoring ``z_a``) would already beat
    persistence whenever the future distribution is narrower than the
    current->future difference.  Reporting them keeps the V0 result honest.
    """
    ds = S3FeaturePairDataset(train_df, traits=())
    loader = DataLoader(ds, batch_size=int(cfg["eval"]["batch_size"]),
                        shuffle=False, num_workers=4, pin_memory=True)
    total = None
    n = 0
    delta_sums: dict[int, torch.Tensor] = {}
    delta_counts: dict[int, int] = {}
    for batch in loader:
        z_b = batch["z_b"].to(device, non_blocking=True).float()
        bs = z_b.shape[0]
        s = z_b.sum(dim=0)
        total = s if total is None else total + s
        n += bs
        deltas = batch["delta_days"].to(torch.long).cpu().numpy()
        for i, d in enumerate(deltas):
            d = int(d)
            delta_sums[d] = (z_b[i].clone() if d not in delta_sums else delta_sums[d] + z_b[i])
            delta_counts[d] = delta_counts.get(d, 0) + 1
    global_mean = total / max(n, 1)
    delta_means = {d: delta_sums[d] / delta_counts[d] for d in delta_sums}
    info = {
        "n_train_samples": int(n),
        "global_mean_shape": list(global_mean.shape),
        "n_distinct_delta_values": int(len(delta_means)),
        "delta_counts": {str(k): int(v) for k, v in sorted(delta_counts.items())},
        "baseline_rule": (
            "global_mean uses the mean z_b over all train pairs; delta_mean uses "
            "the mean z_b for the exact train delta_days, falling back to the "
            "global mean for unseen deltas"
        ),
    }
    return global_mean, delta_means, info


@torch.no_grad()
def e1_latent_prediction(
    model: FIPDinoWM,
    val_df: pd.DataFrame,
    cfg: dict,
    device: torch.device,
    global_mean: torch.Tensor | None = None,
    delta_means: dict[int, torch.Tensor] | None = None,
    delta_mean_scalar: float | None = None,
) -> tuple[dict, pd.DataFrame, dict]:
    ds = S3FeaturePairDataset(val_df, traits=tuple(cfg["data"]["traits"]))
    loader = DataLoader(ds, batch_size=int(cfg["eval"]["batch_size"]),
                        shuffle=False, num_workers=4, pin_memory=True)
    rows = []
    t0 = time.time()
    for batch in loader:
        z_a = batch["z_a"].to(device, non_blocking=True)
        z_b = batch["z_b"].to(device, non_blocking=True)
        delta = batch["delta_days"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z_pred = model(z_a, delta)
            # inference-time time-channel probes (no retraining):
            # (a) shuffle delta across the batch, (b) replace it by the train mean
            perm = torch.randperm(delta.shape[0], device=delta.device)
            z_pred_shuffled = model(z_a, delta[perm])
            if delta_mean_scalar is not None:
                z_pred_const = model(
                    z_a, torch.full_like(delta, float(delta_mean_scalar))
                )
            else:
                z_pred_const = None
        z_pred = z_pred.float()
        z_pred_shuffled = z_pred_shuffled.float()
        z_pred_const = z_pred_const.float() if z_pred_const is not None else None
        se_pred = (z_pred - z_b).pow(2).mean(dim=(1, 2))
        se_persist = (z_a - z_b).pow(2).mean(dim=(1, 2))
        se_shuffled = (z_pred_shuffled - z_b).pow(2).mean(dim=(1, 2))
        se_const = (
            (z_pred_const - z_b).pow(2).mean(dim=(1, 2))
            if z_pred_const is not None else torch.full_like(se_pred, float("nan"))
        )
        if global_mean is not None:
            se_mean = (global_mean.unsqueeze(0) - z_b).pow(2).mean(dim=(1, 2))
        else:
            se_mean = torch.full_like(se_pred, float("nan"))
        if delta_means:
            preds = []
            for i in range(z_a.shape[0]):
                d = int(float(batch["delta_days"][i]))
                m = delta_means.get(d, global_mean)
                preds.append(m)
            se_delta_mean = torch.stack(
                [(preds[i] - z_b[i]).pow(2).mean() for i in range(z_a.shape[0])]
            )
        else:
            se_delta_mean = torch.full_like(se_pred, float("nan"))
        cos_pred = torch.nn.functional.cosine_similarity(
            z_pred.flatten(1), z_b.flatten(1), dim=-1
        )
        cos_persist = torch.nn.functional.cosine_similarity(
            z_a.flatten(1), z_b.flatten(1), dim=-1
        )
        for i in range(z_a.shape[0]):
            rows.append({
                "pair_id": batch["pair_id"][i],
                "plot_uid": batch["plot_uid"][i],
                "split": batch["split"][i],
                "pair_type": batch["pair_type"][i],
                "span_days": int(batch["span_days"][i]),
                "delta_days": int(float(batch["delta_days"][i])),
                "harvest_year": int(batch["harvest_year"][i]),
                "image_stem_a": batch["image_stem_a"][i],
                "image_stem_b": batch["image_stem_b"][i],
                "date_a": batch["date_a"][i],
                "date_b": batch["date_b"][i],
                "mse_pred": float(se_pred[i]),
                "mse_persist": float(se_persist[i]),
                "mse_train_mean": float(se_mean[i]),
                "mse_train_delta_mean": float(se_delta_mean[i]),
                "mse_pred_delta_shuffled": float(se_shuffled[i]),
                "mse_pred_delta_constant": float(se_const[i]),
                "cosine_pred": float(cos_pred[i]),
                "cosine_persist": float(cos_persist[i]),
            })
    df = pd.DataFrame(rows)
    df["relative_improvement"] = (df.mse_persist - df.mse_pred) / df.mse_persist
    df["delta_bin"] = df.delta_days.map(delta_bin)
    summary = {
        "n_samples": int(len(df)),
        "n_plots": int(df.plot_uid.nunique()),
        "mse_pred": float(df.mse_pred.mean()),
        "mse_persist": float(df.mse_persist.mean()),
        "relative_improvement": float(
            (df.mse_persist.mean() - df.mse_pred.mean()) / df.mse_persist.mean()
        ),
        "cosine_pred": float(df.cosine_pred.mean()),
        "cosine_persist": float(df.cosine_persist.mean()),
        "mse_train_mean": float(df.mse_train_mean.mean()),
        "mse_train_delta_mean": float(df.mse_train_delta_mean.mean()),
        "relative_improvement_vs_train_mean": float(
            (df.mse_train_mean.mean() - df.mse_pred.mean()) / df.mse_train_mean.mean()
        ) if df.mse_train_mean.mean() > 0 else None,
        "relative_improvement_vs_train_delta_mean": float(
            (df.mse_train_delta_mean.mean() - df.mse_pred.mean())
            / df.mse_train_delta_mean.mean()
        ) if df.mse_train_delta_mean.mean() > 0 else None,
        "mse_pred_delta_shuffled": float(df.mse_pred_delta_shuffled.mean()),
        "mse_pred_delta_constant": float(df.mse_pred_delta_constant.mean()),
        "delta_shuffle_mse_increase": float(
            (df.mse_pred_delta_shuffled.mean() - df.mse_pred.mean()) / df.mse_pred.mean()
        ),
        "delta_constant_mse_increase": float(
            (df.mse_pred_delta_constant.mean() - df.mse_pred.mean()) / df.mse_pred.mean()
        ) if df.mse_pred_delta_constant.notna().all() else None,
        "by_span": summarize(df, ["pair_type"]),
        "by_span_and_delta": summarize(df, ["pair_type", "delta_days"]),
        "by_year": summarize(df, ["harvest_year"]),
        "by_delta_bin": summarize(df, ["delta_bin"]),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    return summary, df, {"n_samples": len(df), "n_plots": int(df.plot_uid.nunique())}


# --------------------------------------------------------------------------
# E2 probes
# --------------------------------------------------------------------------
@torch.no_grad()
def collect_pooled_features(
    model: FIPDinoWM,
    df: pd.DataFrame,
    cfg: dict,
    device: torch.device,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """One pass: mean-pooled current/predicted features + probe metadata.

    Returns ``(metadata_df, pooled_current, pooled_predicted)``.  The feature
    matrices are kept out of the dataframe so the parquet stays small and
    portable; ``metadata_df["_pool_index"]`` indexes into both matrices.
    """
    ds = S3FeaturePairDataset(df, traits=tuple(cfg["data"]["traits"]))
    loader = DataLoader(ds, batch_size=int(cfg["eval"]["batch_size"]),
                        shuffle=False, num_workers=4, pin_memory=True)
    pool_a, pool_b, meta = [], [], []
    for batch in loader:
        z_a = batch["z_a"].to(device, non_blocking=True)
        delta = batch["delta_days"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z_hat = model(z_a, delta)
        pool_a.append(z_a.float().mean(dim=1).cpu().numpy())
        pool_b.append(z_hat.float().mean(dim=1).cpu().numpy())
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
            for t in cfg["data"]["traits"]:
                row[f"{t}_b"] = float(batch[f"{t}_b"][i])
                row[f"{t}_mask_b"] = bool(batch[f"{t}_mask_b"][i])
                row[f"{t}_b_lag"] = float(batch[f"{t}_b_lag"][i])
                row[f"{t}_b_date"] = str(batch[f"{t}_b_date"][i])
            meta.append(row)
    pooled_a = np.concatenate(pool_a, axis=0).astype(np.float32)
    pooled_b = np.concatenate(pool_b, axis=0).astype(np.float32)
    out = pd.DataFrame(meta)
    out["_orig"] = np.arange(len(out))
    # one sample per future image per span, closest current image (smallest delta)
    out = out.sort_values(
        ["split", "plot_uid", "image_stem_b", "pair_type", "delta_days", "pair_id"],
        kind="mergesort",
    ).drop_duplicates(["plot_uid", "image_stem_b", "pair_type"], keep="first")
    out = out.reset_index(drop=True)
    idx = out["_orig"].to_numpy()
    pooled_a = pooled_a[idx]
    pooled_b = pooled_b[idx]
    out["_pool_index"] = np.arange(len(out))
    out = out.drop(columns=["_orig"])
    return out, pooled_a, pooled_b


def pool_matrix(df: pd.DataFrame, col: str, pooled_a: np.ndarray,
                pooled_b: np.ndarray) -> np.ndarray:
    idx = df["_pool_index"].to_numpy()
    return (pooled_a if col == "pool_a" else pooled_b)[idx]


class PooledReadout(torch.nn.Module):
    """Linear readout on a mean-pooled DINOv2 feature (same for A and B)."""

    def __init__(self, dim: int = 768) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)


def fit_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: dict,
    seed: int,
    device: torch.device,
) -> dict:
    """Train one linear readout; returns validation metrics in original units."""
    set_seed(seed)
    ecfg = cfg["eval"]
    # standardise inputs with train statistics only
    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True) + 1e-6
    xt = torch.from_numpy((x_train - x_mean) / x_std).to(device)
    xv = torch.from_numpy((x_val - x_mean) / x_std).to(device)
    # standardise targets with train statistics only
    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train)) or 1.0
    yt = torch.from_numpy(((y_train - y_mean) / y_std).astype(np.float32)).to(device)
    yv_t = torch.from_numpy(((y_val - y_mean) / y_std).astype(np.float32)).to(device)

    model = PooledReadout(x_train.shape[1]).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(ecfg["probe_lr"]),
        weight_decay=float(ecfg["probe_weight_decay"]),
    )
    n = len(xt)
    bs = int(ecfg["probe_batch_size"])
    steps_per_epoch = max(math.ceil(n / bs), 1)
    total = int(ecfg["probe_epochs"]) * steps_per_epoch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total, 1))
    rng = np.random.RandomState(seed)

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = -1
    patience = int(ecfg.get("probe_patience", 20))
    for epoch in range(int(ecfg["probe_epochs"])):
        model.train()
        order = rng.permutation(n)
        for start in range(0, n, bs):
            idx = torch.from_numpy(order[start:start + bs]).to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xt[idx])
            loss = torch.nn.functional.mse_loss(pred, yt[idx])
            loss.backward()
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            pred = model(xv)
            val = float(torch.nn.functional.mse_loss(pred, yv_t))
        if val < best_val - 1e-9:
            best_val = val
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_std = model(xv).cpu().numpy()
    pred = pred_std * y_std + y_mean
    y = y_val.astype(np.float64)
    err = pred - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_val_mse_standardised": float(best_val),
        "n_train": int(len(x_train)),
        "n_val": int(len(x_val)),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "target_mean": y_mean,
        "target_std": y_std,
        "pred_mean": float(pred.mean()),
        "pred_std": float(pred.std()),
    }


def aggregate_probe_runs(runs: list[dict]) -> dict:
    keys = ("mae", "rmse", "r2")
    out = {"n_seeds": len(runs), "runs": runs}
    for k in keys:
        vals = np.array([r[k] for r in runs], dtype=float)
        out[k] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "values": vals.tolist(),
        }
    out["n_train"] = runs[0]["n_train"]
    out["n_val"] = runs[0]["n_val"]
    return out


def _run_scope(scope_df: pd.DataFrame, t: str, cfg: dict, device: torch.device,
               pooled_a: np.ndarray, pooled_b: np.ndarray,
               strict_lag3: bool = False) -> dict:
    """Train/evaluate both probe conditions on one sample scope."""
    sub = scope_df[scope_df[f"{t}_mask_b"]].copy()
    if strict_lag3:
        sub = sub[sub[f"{t}_b_lag"].abs() <= 3]
    tr = sub[sub.split == "train"]
    va = sub[sub.split == "validation"]
    out = {
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "n_train_plots": int(tr.plot_uid.nunique()),
        "n_val_plots": int(va.plot_uid.nunique()),
        "n_train_by_pair_type": {
            str(k): int(v) for k, v in tr.pair_type.value_counts().items()
        },
        "n_val_by_pair_type": {
            str(k): int(v) for k, v in va.pair_type.value_counts().items()
        },
        "conditions": {},
    }
    if len(tr) < 100 or len(va) < 30:
        out["skipped"] = True
        return out
    for cond, col in (("A_current", "pool_a"), ("B_predicted", "pool_b")):
        runs = [
            fit_probe(pool_matrix(tr, col, pooled_a, pooled_b),
                      tr[f"{t}_b"].to_numpy(np.float32),
                      pool_matrix(va, col, pooled_a, pooled_b),
                      va[f"{t}_b"].to_numpy(np.float32),
                      cfg, int(seed), device)
            for seed in cfg["eval"]["probe_seeds"]
        ]
        out["conditions"][cond] = aggregate_probe_runs(runs)
    ra = out["conditions"]["A_current"]
    rb = out["conditions"]["B_predicted"]
    out["B_minus_A"] = {
        k: {
            "mean_delta": rb[k]["mean"] - ra[k]["mean"],
            "a_mean": ra[k]["mean"],
            "b_mean": rb[k]["mean"],
        }
        for k in ("mae", "rmse", "r2")
    }
    return out


def e2_probes(
    probe_df: pd.DataFrame,
    cfg: dict,
    device: torch.device,
    pooled_a: np.ndarray,
    pooled_b: np.ndarray,
) -> dict:
    """Overall probe + one probe per nominal span (span7 / span14).

    ``probe_df`` has one row per ``(plot_uid, image_stem_b, pair_type)`` so the
    span-specific tables use independent future targets while the overall table
    de-duplicates to one row per future image.
    """
    overall = probe_df.sort_values(
        ["plot_uid", "image_stem_b", "delta_days", "pair_type", "pair_id"],
        kind="mergesort",
    ).drop_duplicates(["plot_uid", "image_stem_b"], keep="first").reset_index(drop=True)

    results: dict = {
        "sample_construction": {
            "rule": (
                "one row per (plot_uid, image_stem_b, pair_type) for span probes; "
                "overall table keeps the smallest-delta row per (plot_uid, image_stem_b)"
            ),
            "n_rows_before_dedup": int(len(probe_df)),
            "n_overall": int(len(overall)),
            "n_overall_train": int((overall.split == "train").sum()),
            "n_overall_validation": int((overall.split == "validation").sum()),
            "n_overall_train_plots": int(overall[overall.split == "train"].plot_uid.nunique()),
            "n_overall_val_plots": int(overall[overall.split == "validation"].plot_uid.nunique()),
        },
        "scopes": {},
    }
    traits = tuple(cfg["data"]["traits"])
    scopes = {
        "overall": overall,
        "span7": probe_df[probe_df.pair_type == "span7"].reset_index(drop=True),
        "span14": probe_df[probe_df.pair_type == "span14"].reset_index(drop=True),
    }
    for scope_name, scope_df in scopes.items():
        results["scopes"][scope_name] = {"traits": {}}
        for t in traits:
            block = _run_scope(scope_df, t, cfg, device, pooled_a, pooled_b)
            # label timing is a property of the sample set, not the probe
            sub = scope_df[scope_df[f"{t}_mask_b"]]
            tr = sub[sub.split == "train"]
            va = sub[sub.split == "validation"]
            block["label_lag_days"] = {
                "train": {
                    "mean": float(tr[f"{t}_b_lag"].mean()) if len(tr) else None,
                    "median": float(tr[f"{t}_b_lag"].median()) if len(tr) else None,
                    "abs_median": float(tr[f"{t}_b_lag"].abs().median()) if len(tr) else None,
                    "abs_p90": float(tr[f"{t}_b_lag"].abs().quantile(0.9)) if len(tr) else None,
                    "n_abs_le_3": int((tr[f"{t}_b_lag"].abs() <= 3).sum()),
                    "n_abs_le_7": int((tr[f"{t}_b_lag"].abs() <= 7).sum()),
                },
                "validation": {
                    "mean": float(va[f"{t}_b_lag"].mean()) if len(va) else None,
                    "median": float(va[f"{t}_b_lag"].median()) if len(va) else None,
                    "abs_median": float(va[f"{t}_b_lag"].abs().median()) if len(va) else None,
                    "abs_p90": float(va[f"{t}_b_lag"].abs().quantile(0.9)) if len(va) else None,
                    "n_abs_le_3": int((va[f"{t}_b_lag"].abs() <= 3).sum()),
                    "n_abs_le_7": int((va[f"{t}_b_lag"].abs() <= 7).sum()),
                },
            }
            if scope_name == "overall":
                block["strict_lag3"] = _run_scope(
                    scope_df, t, cfg, device, pooled_a, pooled_b, strict_lag3=True
                )
            results["scopes"][scope_name]["traits"][t] = block
    return results


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s3_v0.yaml")
    ap.add_argument("--checkpoint", default="external_data/s3_runs/v0/checkpoints/best.pt")
    ap.add_argument("--report", default="reports/s3_v0_results.json")
    ap.add_argument("--predictions", default="reports/s3_v0_val_predictions.parquet")
    ap.add_argument("--skip-probes", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    set_seed(0)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    index = pd.read_parquet(cfg["data"]["index_out"])
    val_df = index[index.split == "validation"].reset_index(drop=True)
    train_df = index[index.split == "train"].reset_index(drop=True)

    model, ckpt = load_model(cfg, Path(args.checkpoint), device)
    report: dict = {
        "stage": "S3",
        "evaluation": "E1 latent prediction + E2 CC/PH probes",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(ckpt["epoch"]),
        "checkpoint_global_step": int(ckpt["global_step"]),
        "manifest_id": ckpt.get("manifest_id"),
        "index_sha256": index_sha256(index),
        "checkpoint_index_sha256": ckpt.get("index_sha256"),
        "device": str(device),
        "model": "FIP-DINO-WM V0 (frozen DINOv2 ViT-B/14, time-conditioned ViTPredictor)",
        "note": (
            "This is a frozen-DINOv2 feature predictor; it is not an I-JEPA/V-JEPA "
            "EMA teacher and has no pixel decoder."
        ),
    }

    global_mean, delta_means, baseline_info = train_feature_baselines(
        train_df, cfg, device
    )
    e1, pred_df, e1_counts = e1_latent_prediction(
        model, val_df, cfg, device, global_mean, delta_means,
        delta_mean_scalar=float(train_df.delta_days.mean()),
    )
    e1["train_baselines"] = baseline_info
    report["E1_future_latent"] = e1
    Path(args.predictions).parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(args.predictions, index=False)
    csv_path = Path(args.predictions).with_suffix(".csv")
    pred_df.head(2000).to_csv(csv_path, index=False)
    report["E1_predictions_file"] = {
        "parquet": args.predictions,
        "csv_sample": str(csv_path),
        "csv_sample_rows": int(min(len(pred_df), 2000)),
    }

    if not args.skip_probes:
        t0 = time.time()
        probe_df, pooled_a, pooled_b = collect_pooled_features(
            model, pd.concat([train_df, val_df], ignore_index=True), cfg, device
        )
        probe_df.drop(columns=["_pool_index"]).to_parquet(
            Path(args.predictions).with_name("s3_probe_samples.parquet"), index=False
        )
        np.savez_compressed(
            Path(args.predictions).with_name("s3_probe_pooled.npz"),
            pool_a=pooled_a, pool_b=pooled_b,
        )
        report["E2_probes"] = e2_probes(probe_df, cfg, device, pooled_a, pooled_b)
        report["E2_probes"]["elapsed_seconds"] = round(time.time() - t0, 1)

    report["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w") as fh:
        json.dump(report, fh, indent=2)
    e2_summary = "skipped"
    if not args.skip_probes:
        e2_summary = {}
        for t in cfg["data"]["traits"]:
            e2_summary[t] = {}
            for cond in ("A_current", "B_predicted"):
                stats = report["E2_probes"]["scopes"]["overall"]["traits"][t]["conditions"][cond]
                e2_summary[t][cond] = {k: stats[k]["mean"] for k in ("mae", "rmse", "r2")}
    print(json.dumps({
        "E1": {k: e1[k] for k in
               ("n_samples", "n_plots", "mse_pred", "mse_persist",
                "relative_improvement", "cosine_pred", "cosine_persist")},
        "E1_by_span": e1["by_span"],
        "E2": e2_summary,
    }, indent=2), flush=True)
    print(f"wrote {args.report}, {args.predictions}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
