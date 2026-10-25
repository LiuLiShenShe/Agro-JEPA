#!/usr/bin/env python3
"""S3.2 task C -- corrected E2 probe protocol (CC / PH linear readouts).

Why this tool exists
--------------------
The S3/S3.1 probe (``agro_jepa/eval/eval_v0.py:fit_probe``) chooses the best
epoch on the **official validation split** and then reports metrics on that same
split, so the reported E2 numbers are model-selection-biased.  This tool
rebuilds the readout protocol so that

* model selection happens only on an inner dev set carved out of the official
  **train** plots (80 / 20 by plot, stratified by harvest year);
* after selection the hyper-parameters and epoch rule are frozen and the readout
  is refit on **all** official train plots;
* the official validation split is touched only for the final evaluation.

Conditions
----------
A  current DINOv2 feature ``mean(Z_a)``        -> future CC / PH
B  V0 predicted future feature ``mean(Z_hat_b)`` -> future CC / PH

Both use the same mean-pooling, the same linear architecture, the same optimiser,
the same candidate budget, the same plot split and the same seeds.  Each
condition may select its own epoch, using the same rule; a shared-epoch variant
is also reported so the comparison does not depend on B being trained longer.

Caveat stated in the report: the V0 checkpoint itself was selected on this same
official validation split in S3.  This stage removes the *probe-level* early
stopping bias only; it does not turn validation into an untouched test set.

Usage
-----
    CUDA_VISIBLE_DEVICES=1 python3 tools/s3_2_probe_eval.py \
        --config agro_jepa/config/s3_2_diagnostics.yaml
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agro_jepa" / "eval"))
sys.path.insert(0, str(ROOT / "tools"))

from agro_jepa.eval.eval_v0 import (  # noqa: E402
    PooledReadout,
    fit_probe,
    pool_matrix,
    set_seed,
)
from agro_jepa.datasets.s3_pairs import index_sha256  # noqa: E402
from s3_1_diagnostics import cluster_bootstrap, git, sha256_file  # noqa: E402


# --------------------------------------------------------------------------
# artifacts
# --------------------------------------------------------------------------
def sha256_if_exists(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def code_evidence() -> dict:
    """Point at the exact S3 early-stopping lines that this stage corrects."""
    src = ROOT / "agro_jepa" / "eval" / "eval_v0.py"
    lines = src.read_text().splitlines()
    markers = {
        "validation_mse_each_epoch": "val = float(torch.nn.functional.mse_loss(pred, yv_t))",
        "model_selection_on_validation": "if val < best_val - 1e-9:",
        "best_state_from_validation": "best_state = copy.deepcopy(model.state_dict())",
    }
    found = {}
    for key, text in markers.items():
        hits = [i + 1 for i, line in enumerate(lines) if text in line]
        found[key] = {
            "marker": text,
            "lines": hits,
            "code": [lines[i - 1].strip() for i in hits],
        }
    scopes = {}
    for key, text in {
        "train_rows": 'tr = sub[sub.split == "train"]',
        "validation_rows": 'va = sub[sub.split == "validation"]',
        "probe_fit_called_with_validation": "va[f\"{t}_b\"].to_numpy(np.float32),",
    }.items():
        hits = [i + 1 for i, line in enumerate(lines) if text in line]
        scopes[key] = {"marker": text, "lines": hits}
    return {
        "file": str(src.relative_to(ROOT)),
        "function": "fit_probe / _run_scope / e2_probes",
        "evidence": {**found, **scopes},
        "problem": (
            "fit_probe computes the validation MSE after every epoch and keeps the "
            "parameters from the epoch that minimise it; _run_scope passes the "
            "official validation split as x_val/y_val, and e2_probes then reports "
            "metrics on that very split.  The reported S3/S3.1 E2 numbers are "
            "therefore optimistic under model selection."
        ),
        "corrected_here": (
            "selection uses an inner dev set of official-train plots only; the "
            "final readout is refit on all official train plots with a fixed epoch "
            "rule and evaluated once on official validation."
        ),
    }


# --------------------------------------------------------------------------
# scopes (identical sample construction to S3)
# --------------------------------------------------------------------------
def scope_frames(probe_df: pd.DataFrame, trait: str) -> dict[str, pd.DataFrame]:
    scopes: dict[str, pd.DataFrame] = {}
    overall = probe_df.sort_values(
        ["plot_uid", "image_stem_b", "delta_days", "pair_type", "pair_id"],
        kind="mergesort",
    ).drop_duplicates(["plot_uid", "image_stem_b"], keep="first").reset_index(drop=True)
    scopes["overall"] = overall[overall[f"{trait}_mask_b"]].reset_index(drop=True)
    for span in ("span7", "span14"):
        scopes[span] = probe_df[
            (probe_df.pair_type == span) & probe_df[f"{trait}_mask_b"]
        ].reset_index(drop=True)
    return scopes


# --------------------------------------------------------------------------
# plot split
# --------------------------------------------------------------------------
def build_plot_split(
    probe_df: pd.DataFrame, seed: int, fraction: float
) -> tuple[pd.DataFrame, dict]:
    """80 / 20 split of the official-train plots, stratified by harvest_year."""
    tr = probe_df[probe_df.split == "train"]
    year = tr.groupby("plot_uid").harvest_year.first()
    rng = np.random.default_rng(seed)
    rows = []
    for y in sorted(year.unique()):
        plots = np.sort(year[year == y].index.to_numpy())
        perm = rng.permutation(len(plots))
        n_dev = max(1, int(round(len(plots) * (1.0 - fraction))))
        dev = set(plots[perm[:n_dev]])
        for p in plots:
            rows.append({
                "plot_uid": str(p),
                "harvest_year": int(y),
                "role": "inner_dev" if p in dev else "inner_train",
            })
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
    }
    return split, stats


# --------------------------------------------------------------------------
# readout training
# --------------------------------------------------------------------------
def _metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    err = pred.astype(np.float64) - y.astype(np.float64)
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "n": int(len(y)),
    }


def train_readout(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    pcfg: dict,
    seed: int,
    device: torch.device,
    epochs_to_run: int,
    x_dev: np.ndarray | None = None,
    y_dev: np.ndarray | None = None,
    select: bool = False,
) -> tuple[dict, dict]:
    """Train one linear readout.

    ``select=True`` runs the full candidate budget with early stopping on the
    inner dev set.  ``select=False`` runs exactly ``epochs_to_run`` epochs and
    never inspects any held-out target.  In both cases the cosine schedule is
    anchored to the *candidate* budget, so a final run stopped at the selected
    epoch follows exactly the same trajectory as a full-budget run stopped there.
    """
    if select and (x_dev is None or y_dev is None):
        raise ValueError("select=True requires an inner dev set")
    set_seed(seed)
    x_mean = x_tr.mean(axis=0, keepdims=True)
    x_std = x_tr.std(axis=0, keepdims=True) + 1e-6
    y_mean = float(np.mean(y_tr))
    y_std = float(np.std(y_tr)) or 1.0
    xt = torch.from_numpy(((x_tr - x_mean) / x_std).astype(np.float32)).to(device)
    yt = torch.from_numpy(((y_tr - y_mean) / y_std).astype(np.float32)).to(device)
    if select:
        xd = torch.from_numpy(((x_dev - x_mean) / x_std).astype(np.float32)).to(device)
        yd = torch.from_numpy(((y_dev - y_mean) / y_std).astype(np.float32)).to(device)

    model = PooledReadout(x_tr.shape[1]).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(pcfg["lr"]),
        weight_decay=float(pcfg["weight_decay"]),
    )
    n = len(xt)
    bs = int(pcfg["batch_size"])
    steps_per_epoch = max(math.ceil(n / bs), 1)
    candidate_epochs = int(pcfg["candidate_epochs"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(candidate_epochs * steps_per_epoch, 1)
    )
    rng = np.random.RandomState(seed)
    patience = int(pcfg["patience"])

    best_state = copy.deepcopy(model.state_dict())
    best_dev = float("inf")
    best_epoch = -1
    budget = candidate_epochs if select else int(epochs_to_run)
    for epoch in range(budget):
        model.train()
        order = rng.permutation(n)
        for start in range(0, n, bs):
            idx = torch.from_numpy(order[start:start + bs]).to(device)
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            sched.step()
        if select:
            model.eval()
            with torch.no_grad():
                dev = float(torch.nn.functional.mse_loss(model(xd), yd))
            if dev < best_dev - 1e-9:
                best_dev = dev
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
            elif epoch - best_epoch >= patience:
                break
    if select:
        model.load_state_dict(best_state)
    model.eval()
    info = {
        "seed": int(seed),
        "epochs_run": int(epoch + 1),
        "best_epoch": int(best_epoch) if select else int(budget - 1),
        "best_inner_dev_mse_standardised": float(best_dev) if select else None,
        "n_train": int(len(x_tr)),
        "x_mean_sha256": hashlib.sha256(
            np.ascontiguousarray(x_mean, dtype=np.float32).tobytes()
        ).hexdigest(),
        "x_std_sha256": hashlib.sha256(
            np.ascontiguousarray(x_std, dtype=np.float32).tobytes()
        ).hexdigest(),
        "x_mean_overall": float(np.asarray(x_mean, dtype=np.float64).mean()),
        "x_std_overall": float(np.asarray(x_std, dtype=np.float64).mean()),
        "y_mean": y_mean,
        "y_std": y_std,
    }
    return {"model": model, "x_mean": x_mean, "x_std": x_std,
            "y_mean": y_mean, "y_std": y_std}, info


def predict_readout(state: dict, x: np.ndarray, device: torch.device) -> np.ndarray:
    model = state["model"]
    xz = torch.from_numpy(
        ((x - state["x_mean"]) / state["x_std"]).astype(np.float32)
    ).to(device)
    model.eval()
    with torch.no_grad():
        out = model(xz).cpu().numpy().astype(np.float64)
    return out * state["y_std"] + state["y_mean"]


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s3_2_diagnostics.yaml")
    ap.add_argument("--probe-samples", default="reports/s3_probe_samples.parquet")
    ap.add_argument("--pooled", default="reports/s3_probe_pooled.npz")
    ap.add_argument("--out", default="reports/s3_2_probe_results.json")
    ap.add_argument("--protocol-out", default="reports/s3_2_probe_protocol.json")
    ap.add_argument("--predictions", default="reports/s3_2_probe_val_predictions.parquet")
    ap.add_argument("--plot-split", default="reports/s3_2_probe_plot_split.csv")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    pcfg = cfg["probe"]
    device = torch.device(
        args.device or (pcfg["device"] if torch.cuda.is_available() else "cpu")
    )
    set_seed(0)
    t_start = time.time()

    probe_df = pd.read_parquet(args.probe_samples)
    probe_df["_pool_index"] = np.arange(len(probe_df))
    pooled = np.load(args.pooled)
    pool_a, pool_b = pooled["pool_a"], pooled["pool_b"]
    if len(pool_a) != len(probe_df) or len(pool_b) != len(probe_df):
        raise RuntimeError("pooled feature rows do not match the probe sample rows")

    protocol: dict = {
        "stage": "S3.2",
        "task": "C1-C4 -- corrected probe protocol",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": " ".join(sys.argv),
        "config": args.config,
        "config_sha256": sha256_file(Path(args.config)),
        "device": str(device),
        "git_head": git("rev-parse", "HEAD"),
        "C1_s3_protocol_problem": code_evidence(),
        "inputs": {
            "probe_samples": args.probe_samples,
            "probe_samples_sha256": sha256_file(Path(args.probe_samples)),
            "pooled_features": args.pooled,
            "pooled_features_sha256": sha256_if_exists(Path(args.pooled)),
            "pooled_shape": list(pool_a.shape),
            "index_sha256": index_sha256(
                pd.read_parquet(cfg["frozen"]["index"])
            ),
            "checkpoint_sha256": sha256_file(Path(cfg["frozen"]["checkpoint"])),
            "pooling": "mean over the 256 patch tokens (identical for A and B)",
            "architecture": "single linear layer (PooledReadout, 768 -> 1)",
            "candidate_epochs": int(pcfg["candidate_epochs"]),
            "patience": int(pcfg["patience"]),
            "batch_size": int(pcfg["batch_size"]),
            "lr": float(pcfg["lr"]),
            "weight_decay": float(pcfg["weight_decay"]),
            "seeds": [int(s) for s in pcfg["seeds"]],
            "optimiser": "AdamW + cosine annealing anchored to the candidate budget",
        },
        "C2_plot_split": {},
        "C3_inner_selection": {},
        "C4_final_readout": {},
    }

    # ---------------- C2: fixed inner split ------------------------------
    split, split_stats = build_plot_split(
        probe_df, int(pcfg["plot_split_seed"]), float(pcfg["inner_train_fraction"])
    )
    split.to_csv(args.plot_split, index=False)
    role = split.set_index("plot_uid").role
    year_of_plot = split.set_index("plot_uid").harvest_year
    protocol["C2_plot_split"] = {
        **split_stats,
        "file": args.plot_split,
        "rule": (
            "official-train plots only; 80/20 by plot_uid, stratified by "
            "harvest_year; the same split is reused by every trait, scope, "
            "variant and condition"
        ),
        "disjoint": bool(
            not set(split[split.role == "inner_train"].plot_uid)
            & set(split[split.role == "inner_dev"].plot_uid)
        ),
        "official_validation_plots_in_split": int(
            (split.plot_uid.isin(probe_df[probe_df.split == "validation"].plot_uid)).sum()
        ),
    }

    # ---------------- alignment check against S3 aggregates ------------
    def xmat(frame: pd.DataFrame, col: str) -> np.ndarray:
        return pool_matrix(frame, col, pool_a, pool_b)

    alignment = {"tolerance_abs": 2e-3, "values": {}}
    # Exact S3 ``eval`` block, rebuilt from the S3.2 config so the parity check
    # re-runs the historical protocol unchanged.
    old_eval_cfg = {
        "eval": {
            "probe_lr": float(pcfg["lr"]),
            "probe_weight_decay": float(pcfg["weight_decay"]),
            "probe_batch_size": int(pcfg["batch_size"]),
            "probe_epochs": int(pcfg["candidate_epochs"]),
            "probe_patience": int(pcfg["patience"]),
            "probe_seeds": [int(s) for s in pcfg["seeds"]],
        }
    }
    for trait in ("cc", "ph"):
        frame = scope_frames(probe_df, trait)["overall"]
        tr = frame[frame.split == "train"]
        va = frame[frame.split == "validation"].reset_index(drop=True)
        for cond, col in (("A_current", "pool_a"), ("B_predicted", "pool_b")):
            runs = [
                fit_probe(xmat(tr, col), tr[f"{trait}_b"].to_numpy(np.float32),
                          xmat(va, col), va[f"{trait}_b"].to_numpy(np.float32),
                          old_eval_cfg, int(s), device)
                for s in pcfg["seeds"]
            ]
            got = {
                "mae": float(np.mean([r["mae"] for r in runs])),
                "rmse": float(np.mean([r["rmse"] for r in runs])),
                "r2": float(np.mean([r["r2"] for r in runs])),
            }
            ref = cfg["probe"]["reference_s3_1"][trait]["overall"][cond]
            diff = {k: got[k] - float(ref[k]) for k in ("mae", "rmse", "r2")}
            ok = all(abs(v) < alignment["tolerance_abs"] for v in diff.values())
            alignment["values"][f"{trait}|{cond}"] = {
                "recomputed_old_protocol": got,
                "s3_1_reference": ref,
                "diff": diff,
                "within_tolerance": bool(ok),
            }
            if not ok:
                raise SystemExit(
                    f"alignment check failed for {trait}|{cond}: {diff}. "
                    "The pooled-feature rows do not reproduce the S3 aggregates."
                )
    protocol["alignment_check"] = alignment
    alignment_elapsed = round(time.time() - t_start, 1)

    # ---------------- variants -------------------------------------------
    variants = {
        "primary_lag7": int(pcfg["lag_variants"]["primary"]),
        "strict_lag3": int(pcfg["lag_variants"]["strict_lag3"]),
        "same_day_lag0": int(pcfg["lag_variants"]["same_day"]),
    }
    scope_plan = {name: ["overall", "span7", "span14"] for name in variants}
    for name in variants:
        if name != "primary_lag7":
            scope_plan[name] = list(pcfg["sensitivity_scopes"])

    rows: list[dict] = []
    for variant, lag_max in variants.items():
        protocol["C3_inner_selection"][variant] = {}
        protocol["C4_final_readout"][variant] = {}
        for trait in ("cc", "ph"):
            protocol["C3_inner_selection"][variant][trait] = {}
            protocol["C4_final_readout"][variant][trait] = {}
            frames = scope_frames(probe_df, trait)
            for scope in scope_plan[variant]:
                frame = frames[scope]
                frame = frame[frame[f"{trait}_b_lag"].abs() <= lag_max]
                tr = frame[frame.split == "train"].copy()
                va = frame[frame.split == "validation"].copy().reset_index(drop=True)
                tr["role"] = tr.plot_uid.map(role)
                if tr.role.isna().any():
                    raise RuntimeError("official-train plot missing from the split")
                tr_in = tr[tr.role == "inner_train"]
                tr_dev = tr[tr.role == "inner_dev"]
                tag = f"{variant}|{trait}|{scope}"
                if len(tr_in) < 100 or len(tr_dev) < 30 or len(va) < 30:
                    protocol["C3_inner_selection"][variant][trait][scope] = {
                        "skipped": True,
                        "n_inner_train": int(len(tr_in)),
                        "n_inner_dev": int(len(tr_dev)),
                        "n_val": int(len(va)),
                    }
                    continue
                counts = {
                    "lag_max_abs_days": lag_max,
                    "n_train": int(len(tr)),
                    "n_train_plots": int(tr.plot_uid.nunique()),
                    "n_inner_train": int(len(tr_in)),
                    "n_inner_train_plots": int(tr_in.plot_uid.nunique()),
                    "n_inner_dev": int(len(tr_dev)),
                    "n_inner_dev_plots": int(tr_dev.plot_uid.nunique()),
                    "n_val": int(len(va)),
                    "n_val_plots": int(va.plot_uid.nunique()),
                }
                print(f"[{tag}] {counts}", flush=True)

                x_in = {c: xmat(tr_in, c) for c in ("pool_a", "pool_b")}
                x_dev = {c: xmat(tr_dev, c) for c in ("pool_a", "pool_b")}
                x_tr = {c: xmat(tr, c) for c in ("pool_a", "pool_b")}
                x_va = {c: xmat(va, c) for c in ("pool_a", "pool_b")}
                y_in = tr_in[f"{trait}_b"].to_numpy(np.float32)
                y_dev = tr_dev[f"{trait}_b"].to_numpy(np.float32)
                y_tr = tr[f"{trait}_b"].to_numpy(np.float32)
                y_va = va[f"{trait}_b"].to_numpy(np.float32)

                selection: dict = {"counts": counts, "conditions": {}}
                best_epochs = {}
                for cond, col in (("A_current", "pool_a"), ("B_predicted", "pool_b")):
                    runs = []
                    for seed in pcfg["seeds"]:
                        state, info = train_readout(
                            x_in[col], y_in, pcfg, int(seed), device,
                            epochs_to_run=0, x_dev=x_dev[col], y_dev=y_dev,
                            select=True,
                        )
                        dev_pred = predict_readout(state, x_dev[col], device)
                        info["inner_dev_metrics"] = _metrics(y_dev, dev_pred)
                        runs.append(info)
                    best_epochs[cond] = [r["best_epoch"] for r in runs]
                    selection["conditions"][cond] = {
                        "runs": runs,
                        "best_epochs": best_epochs[cond],
                        "selected_epoch_median": int(round(float(np.median(best_epochs[cond])))),
                        "selection_rule": (
                            "minimum inner-dev MSE over the candidate budget "
                            "(same rule for both conditions)"
                        ),
                    }
                shared = int(round(float(np.median(
                    best_epochs["A_current"] + best_epochs["B_predicted"]
                ))))
                selection["shared_epoch_median_over_both_conditions"] = shared
                protocol["C3_inner_selection"][variant][trait][scope] = selection

                final: dict = {"counts": counts, "protocols": {}}
                for proto_name, epoch_rule in (
                    ("per_condition_epochs", "own"),
                    ("shared_epochs", "shared"),
                ):
                    proto_block: dict = {
                        "epoch_rule": (
                            "each condition uses its own median inner-dev best epoch"
                            if epoch_rule == "own"
                            else "both conditions use the median best epoch over all "
                                 "six inner-dev runs"
                        ),
                        "conditions": {},
                    }
                    for cond, col in (("A_current", "pool_a"), ("B_predicted", "pool_b")):
                        epochs = (
                            selection["conditions"][cond]["selected_epoch_median"]
                            if epoch_rule == "own" else shared
                        )
                        runs = []
                        for seed in pcfg["seeds"]:
                            state, info = train_readout(
                                x_tr[col], y_tr, pcfg, int(seed), device,
                                epochs_to_run=int(epochs), select=False,
                            )
                            # official validation is read here for evaluation only
                            pred = predict_readout(state, x_va[col], device)
                            metric = _metrics(y_va, pred)
                            info["metrics"] = metric
                            info["selection_source"] = "inner_dev (official train plots)"
                            info["official_validation_used_for_selection"] = False
                            runs.append(info)
                            if proto_name == "per_condition_epochs":
                                for i in range(len(va)):
                                    rows.append({
                                        "variant": variant,
                                        "trait": trait,
                                        "scope": scope,
                                        "protocol": proto_name,
                                        "condition": cond,
                                        "seed": int(seed),
                                        "pair_id": va.pair_id.iloc[i],
                                        "plot_uid": va.plot_uid.iloc[i],
                                        "pair_type": va.pair_type.iloc[i],
                                        "delta_days": int(va.delta_days.iloc[i]),
                                        "lag_days": float(va[f"{trait}_b_lag"].iloc[i]),
                                        "y_true": float(y_va[i]),
                                        "y_pred": float(pred[i]),
                                    })
                        agg = {}
                        for k in ("mae", "rmse", "r2"):
                            vals = np.array([r["metrics"][k] for r in runs], dtype=float)
                            agg[k] = {
                                "mean": float(vals.mean()),
                                "std": float(vals.std()),
                                "values": vals.tolist(),
                            }
                        agg["n_val"] = int(len(va))
                        agg["n_val_plots"] = int(va.plot_uid.nunique())
                        agg["n_train"] = int(len(tr))
                        agg["n_train_plots"] = int(tr.plot_uid.nunique())
                        agg["epochs_used"] = int(epochs)
                        agg["runs"] = runs
                        proto_block["conditions"][cond] = agg
                    ra = proto_block["conditions"]["A_current"]
                    rb = proto_block["conditions"]["B_predicted"]
                    proto_block["B_minus_A"] = {
                        k: {
                            "mean_delta": rb[k]["mean"] - ra[k]["mean"],
                            "a_mean": ra[k]["mean"],
                            "b_mean": rb[k]["mean"],
                        }
                        for k in ("mae", "rmse", "r2")
                    }
                    final["protocols"][proto_name] = proto_block
                protocol["C4_final_readout"][variant][trait][scope] = final

    # ---------------- C6: plot-cluster bootstrap -------------------------
    preds = pd.DataFrame(rows)
    preds.to_parquet(args.predictions, index=False)
    boot = {
        "unit": "plot_uid (all probe rows of a resampled plot kept together)",
        "resampling": "with replacement",
        "n_resamples": int(pcfg["bootstrap_resamples"]),
        "seed": int(pcfg["bootstrap_seed"]),
        "seed_averaging": "per-sample predictions averaged over the 3 probe seeds",
        "scopes": {},
    }

    def _bmetrics(y, pa, pb) -> dict:
        ea, eb = pa - y, pb - y
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        return {
            "mae_a": float(np.mean(np.abs(ea))),
            "mae_b": float(np.mean(np.abs(eb))),
            "rmse_a": float(np.sqrt(np.mean(ea ** 2))),
            "rmse_b": float(np.sqrt(np.mean(eb ** 2))),
            "r2_a": 1.0 - float(np.sum(ea ** 2)) / ss_tot if ss_tot > 0 else np.nan,
            "r2_b": 1.0 - float(np.sum(eb ** 2)) / ss_tot if ss_tot > 0 else np.nan,
            "delta_mae": float(np.mean(np.abs(eb)) - np.mean(np.abs(ea))),
            "delta_rmse": float(np.sqrt(np.mean(eb ** 2)) - np.sqrt(np.mean(ea ** 2))),
            "delta_r2": 1.0 - float(np.sum(eb ** 2)) / ss_tot
                      - (1.0 - float(np.sum(ea ** 2)) / ss_tot) if ss_tot > 0 else np.nan,
        }

    for variant in variants:
        for trait in ("cc", "ph"):
            for scope in scope_plan[variant]:
                sub = preds[(preds.variant == variant) & (preds.trait == trait)
                            & (preds.scope == scope)]
                if sub.empty:
                    continue
                agg = sub.groupby(["condition", "pair_id", "plot_uid"], as_index=False).agg(
                    y_true=("y_true", "first"), y_pred=("y_pred", "mean")
                )
                a = agg[agg.condition == "A_current"].set_index("pair_id")
                b = agg[agg.condition == "B_predicted"].set_index("pair_id")
                common = a.index.intersection(b.index)
                a, b = a.loc[common], b.loc[common]
                y = a.y_true.to_numpy(np.float64)
                tables = {"y": y, "pa": a.y_pred.to_numpy(np.float64),
                          "pb": b.y_pred.to_numpy(np.float64)}
                plots = a.plot_uid.to_numpy()
                block = {"n_samples": int(len(y)), "n_plots": int(len(np.unique(plots)))}
                for key in ("mae_a", "mae_b", "rmse_a", "rmse_b", "r2_a", "r2_b",
                            "delta_mae", "delta_rmse", "delta_r2"):
                    block[key] = cluster_bootstrap(
                        tables, plots,
                        lambda t, k=key: _bmetrics(t["y"], t["pa"], t["pb"])[k],
                        int(pcfg["bootstrap_resamples"]), int(pcfg["bootstrap_seed"]),
                    )
                boot["scopes"].setdefault(variant, {}).setdefault(trait, {})[scope] = block

    protocol["C6_bootstrap"] = boot
    protocol["artifacts"] = {
        "per_sample_predictions": args.predictions,
        "n_rows": int(len(preds)),
        "plot_split": args.plot_split,
        "alignment_check_seconds": alignment_elapsed,
    }
    protocol["elapsed_seconds"] = round(time.time() - t_start, 1)

    Path(args.protocol_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.protocol_out, "w") as fh:
        json.dump(protocol, fh, indent=2)

    # human/CI-readable headline file
    headline = {
        "stage": "S3.2",
        "C2_plot_split": protocol["C2_plot_split"],
        "C3_selected_epochs": {
            v: {
                t: {
                    s: {
                        "A": blk["conditions"]["A_current"]["selected_epoch_median"],
                        "B": blk["conditions"]["B_predicted"]["selected_epoch_median"],
                        "shared": blk["shared_epoch_median_over_both_conditions"],
                    }
                    for s, blk in scopes.items() if "conditions" in blk
                }
                for t, scopes in traits.items()
            }
            for v, traits in protocol["C3_inner_selection"].items()
        },
        "C4_primary_protocol": {
            v: {
                t: {
                    s: {
                        cond: {
                            k: blk["protocols"]["per_condition_epochs"]["conditions"][cond][k]["mean"]
                            for k in ("mae", "rmse", "r2")
                        }
                        for cond in ("A_current", "B_predicted")
                    } | {
                        "B_minus_A": blk["protocols"]["per_condition_epochs"]["B_minus_A"],
                        "n_val": blk["counts"]["n_val"],
                        "n_val_plots": blk["counts"]["n_val_plots"],
                    }
                    for s, blk in scopes.items() if "protocols" in blk
                }
                for t, scopes in traits.items()
            }
            for v, traits in protocol["C4_final_readout"].items()
        },
    }
    with open(args.out, "w") as fh:
        json.dump(headline, fh, indent=2)
    print(json.dumps(headline["C3_selected_epochs"], indent=2), flush=True)
    print(f"wrote {args.out}, {args.protocol_out}, {args.predictions}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
