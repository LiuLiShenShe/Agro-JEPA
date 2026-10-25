#!/usr/bin/env python3
"""S3.2 task A/B -- controlled ``delta_days`` interventions on the frozen V0 model.

Scientific question
-------------------
The S3.1 global permutation mixes two distinct losses of information:

* the short-span / long-span distinction (span7 realised 5-9 d, span14 12-16 d,
  non-overlapping), and
* the exact per-sample interval inside each span.

This tool separates them by feeding the *same frozen checkpoint* different
interval vectors at inference time.  Nothing is retrained; ``z_a``, ``z_b``,
``pair_id`` and the phenotype labels are identical in every condition.

Conditions
----------
A0  real ``delta_days`` (reference; reproduces the S3.1 baseline)
A1  one permutation over the whole validation set (S3.1 reproduction)
A2  permutation inside each ``pair_type`` only (span7 / span14 kept separate)
A3  every sample in a ``pair_type`` receives that pair_type's **train** mean
A4  permutation inside each ``(pair_type, harvest_year)`` stratum
A5  per-real-interval breakdown of A0 and of A2/A3/A4

All permutation conditions use 20 fixed seeds.  Group-wise permutations use an
independent RNG stream per group, and the tool verifies that the multiset of
``delta_days`` inside every group is unchanged by the permutation.

Uncertainty
-----------
Paired per-sample MSE differences are bootstrapped over ``plot_uid`` (2,000
resamples, fixed seed).  ``delta_* = mse_condition - mse_real``; positive means
the intervention made the prediction worse.

Usage
-----
    CUDA_VISIBLE_DEVICES=1 python3 tools/s3_2_delta_diagnostics.py \
        --config agro_jepa/config/s3_2_diagnostics.yaml
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
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agro_jepa" / "eval"))
sys.path.insert(0, str(ROOT / "tools"))

from agro_jepa.datasets.s3_pairs import index_sha256  # noqa: E402
from agro_jepa.eval.eval_v0 import load_model, set_seed  # noqa: E402
from s3_1_diagnostics import (  # noqa: E402
    cluster_bootstrap,
    e1_forward,
    git,
    sha256_file,
    source_state,
)


# --------------------------------------------------------------------------
# frozen-artifact checks
# --------------------------------------------------------------------------
def check_frozen(cfg: dict) -> dict:
    """Refuse to run on anything other than the S3 checkpoint / index."""
    frozen = cfg["frozen"]
    ckpt = Path(frozen["checkpoint"])
    index_path = Path(frozen["index"])
    ckpt_sha = sha256_file(ckpt)
    index = pd.read_parquet(index_path)
    idx_sha = index_sha256(index)
    out = {
        "checkpoint": str(ckpt),
        "checkpoint_sha256": ckpt_sha,
        "checkpoint_sha256_expected": frozen["checkpoint_sha256"],
        "checkpoint_matches": ckpt_sha == frozen["checkpoint_sha256"],
        "index": str(index_path),
        "index_sha256": idx_sha,
        "index_sha256_expected": frozen["index_sha256"],
        "index_matches": idx_sha == frozen["index_sha256"],
    }
    if not (out["checkpoint_matches"] and out["index_matches"]):
        raise SystemExit(
            "frozen artifact mismatch -- refusing to run on non-S3 inputs: "
            f"{json.dumps(out, indent=2)}"
        )
    return out


# --------------------------------------------------------------------------
# interventions
# --------------------------------------------------------------------------
def group_positions(labels: np.ndarray) -> dict[str, np.ndarray]:
    return {str(g): np.flatnonzero(labels == g) for g in pd.unique(labels)}


def permute_within_groups(
    deltas: np.ndarray,
    labels: np.ndarray,
    seed_base: int,
    seed_index: int,
    stream_offset: int,
) -> tuple[np.ndarray, dict, bool]:
    """Permute ``deltas`` inside each group; one independent RNG per group.

    Returns ``(permuted, group_sizes, frequency_preserved)``.  Groups with fewer
    than two rows cannot be permuted: they keep their original value and are
    recorded, never silently skipped.
    """
    out = deltas.copy()
    groups = group_positions(labels)
    sizes: dict[str, dict] = {}
    preserved = True
    for ordinal, (name, pos) in enumerate(sorted(groups.items())):
        if len(pos) < 2:
            sizes[name] = {
                "n_samples": int(len(pos)),
                "permuted": False,
                "reason": "fewer than two rows in the group; original values kept",
            }
            continue
        rng = np.random.default_rng(
            [int(seed_base) + int(seed_index), int(stream_offset), ordinal]
        )
        perm = rng.permutation(len(pos))
        out[pos] = deltas[pos][perm]
        same_multiset = bool(
            np.array_equal(np.sort(out[pos]), np.sort(deltas[pos]))
        )
        preserved = preserved and same_multiset
        sizes[name] = {
            "n_samples": int(len(pos)),
            "permuted": True,
            "n_changed": int(np.count_nonzero(out[pos] != deltas[pos])),
            "multiset_preserved": same_multiset,
        }
    return out, sizes, preserved


def global_permutation(deltas: np.ndarray, seed: int) -> np.ndarray:
    return deltas[np.random.default_rng(seed).permutation(len(deltas))]


# --------------------------------------------------------------------------
# summaries
# --------------------------------------------------------------------------
def _per_seed_group_mean(values: np.ndarray, pos: np.ndarray) -> tuple[float, float]:
    """Mean across seeds of the per-seed group mean -> (mean, std)."""
    per_seed = values[:, pos].mean(axis=1)
    return float(per_seed.mean()), float(per_seed.std(ddof=1)) if len(per_seed) > 1 else 0.0


def grouped_summary(
    cond: np.ndarray,
    real: np.ndarray,
    labels: np.ndarray,
) -> dict:
    out = {}
    for g in pd.unique(labels):
        pos = np.flatnonzero(labels == g)
        m, s = _per_seed_group_mean(cond, pos)
        base = float(real[pos].mean())
        out[str(g)] = {
            "n_samples": int(len(pos)),
            "mse_real": base,
            "mse_condition_mean": m,
            "mse_condition_std": s,
            "relative_increase_mean": (m - base) / base if base else None,
        }
    return out


def summarize_condition(
    cond: np.ndarray,
    real: np.ndarray,
    plots: np.ndarray,
    span: np.ndarray,
    year: np.ndarray,
    labels_for_delta: np.ndarray,
    n_bootstrap: int,
    boot_seed: int,
) -> dict:
    """Condition-level statistics: per-seed spread + plot-cluster bootstrap."""
    n_seeds = cond.shape[0]
    per_seed = cond.mean(axis=1)
    base = float(real.mean())
    rel = (per_seed - base) / base
    seed_mean = cond.mean(axis=0)

    def metric_diff(t):
        return float(t["cond"].mean() - t["real"].mean())

    def metric_rel(t):
        return float((t["cond"].mean() - t["real"].mean()) / t["real"].mean())

    tables = {"real": real, "cond": seed_mean}
    b_diff = cluster_bootstrap(tables, plots, metric_diff, n_bootstrap, boot_seed)
    b_rel = cluster_bootstrap(tables, plots, metric_rel, n_bootstrap, boot_seed)

    by_span = {}
    for sp in sorted(pd.unique(span)):
        pos = np.flatnonzero(span == sp)
        sub_tables = {"real": real[pos], "cond": seed_mean[pos]}
        by_span[str(sp)] = {
            **grouped_summary(cond, real, span)[str(sp)],
            "paired_diff": cluster_bootstrap(
                sub_tables, plots[pos], metric_diff, n_bootstrap, boot_seed),
            "relative_increase": cluster_bootstrap(
                sub_tables, plots[pos], metric_rel, n_bootstrap, boot_seed),
        }
    return {
        "n_seeds": int(n_seeds),
        "mse_real": base,
        "mse_condition_per_seed": [float(v) for v in per_seed],
        "mse_condition_mean": float(per_seed.mean()),
        "mse_condition_std": float(per_seed.std(ddof=1)) if n_seeds > 1 else 0.0,
        "mse_condition_min": float(per_seed.min()),
        "mse_condition_max": float(per_seed.max()),
        "relative_increase_mean": float(rel.mean()),
        "relative_increase_min": float(rel.min()),
        "relative_increase_max": float(rel.max()),
        "per_sample_worse_fraction": float(
            np.mean([(cond[s] > real).mean() for s in range(n_seeds)])
        ),
        "paired_diff": b_diff,
        "relative_increase_bootstrap": b_rel,
        "by_span": by_span,
        "by_year": grouped_summary(cond, real, year),
        "by_delta": grouped_summary(cond, real, labels_for_delta),
    }


def per_delta_table(
    real: np.ndarray,
    persist: np.ndarray,
    cos_real: np.ndarray,
    cos_persist: np.ndarray,
    conds: dict[str, np.ndarray],
    delta: np.ndarray,
    plots: np.ndarray,
    values: list[int],
) -> dict:
    """A5: per-real-interval breakdown (no unweighted cross-group average)."""
    table = {}
    for d in values:
        pos = np.flatnonzero(delta == d)
        if len(pos) == 0:
            table[str(d)] = {"n_samples": 0, "note": "no validation samples"}
            continue
        row = {
            "n_samples": int(len(pos)),
            "n_plots": int(len(np.unique(plots[pos]))),
            "mse_real": float(real[pos].mean()),
            "mse_persist": float(persist[pos].mean()),
            "cosine_real": float(cos_real[pos].mean()),
            "cosine_persist": float(cos_persist[pos].mean()),
            "diagnostics": {},
        }
        for name, mat in conds.items():
            v = float(mat[:, pos].mean())
            row["diagnostics"][name] = {
                "mse_mean_over_seeds": v,
                "relative_increase": (v - row["mse_real"]) / row["mse_real"],
            }
        table[str(d)] = row
    return table


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s3_2_diagnostics.yaml")
    ap.add_argument("--out", default="reports/s3_2_delta_diagnostics.json")
    ap.add_argument(
        "--predictions", default="reports/s3_2_delta_interventions.parquet"
    )
    ap.add_argument("--permuted-deltas", default="reports/s3_2_permuted_deltas.parquet")
    ap.add_argument("--s3-predictions", default="reports/s3_v0_val_predictions.parquet")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    dcfg = cfg["delta_diagnostics"]
    device = torch.device(
        args.device or (dcfg["device"] if torch.cuda.is_available() else "cpu")
    )
    set_seed(0)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    frozen = check_frozen(cfg)
    index = pd.read_parquet(cfg["frozen"]["index"])
    val_df = index[index.split == "validation"].reset_index(drop=True)
    train_df = index[index.split == "train"].reset_index(drop=True)

    v0_cfg = yaml.safe_load(Path(cfg["frozen"]["config"]).read_text())
    model, ckpt = load_model(v0_cfg, Path(cfg["frozen"]["checkpoint"]), device)
    # e1_forward reads cfg["eval"]["batch_size"]; keep it identical to S3/S3.1
    eval_cfg = {"eval": {"batch_size": int(dcfg["eval_batch_size"])}}

    deltas = val_df.delta_days.to_numpy(np.float64)
    span = val_df.pair_type.to_numpy().astype(str)
    year = val_df.harvest_year.to_numpy().astype(str)
    plots = val_df.plot_uid.to_numpy().astype(str)
    n = len(val_df)
    seed_base = int(dcfg["seed_base"])
    n_seeds = int(dcfg["n_seeds"])

    result: dict = {
        "stage": "S3.2",
        "task": "A/B -- controlled delta_days interventions (no retraining)",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": " ".join(sys.argv),
        "config": args.config,
        "config_sha256": sha256_file(Path(args.config)),
        "device": str(device),
        "frozen": frozen,
        "source_state": source_state(),
        "scope": {
            "split": "validation",
            "n_samples": int(n),
            "n_plots": int(len(np.unique(plots))),
            "pair_types": sorted(pd.unique(span).tolist()),
            "delta_range": [int(deltas.min()), int(deltas.max())],
            "delta_counts": {
                str(k): int(v) for k, v in pd.Series(deltas).value_counts().sort_index().items()
            },
        },
    }

    # ---------------- A0: real intervals --------------------------------
    t0 = time.time()
    base = e1_forward(model, val_df, eval_cfg, device)
    real = base["mse_pred"].astype(np.float64)
    persist = base["mse_persist"].astype(np.float64)
    cos_real = base["cos_pred"].astype(np.float64)
    cos_persist = base["cos_persist"].astype(np.float64)
    a0 = {
        "n_samples": int(n),
        "n_plots": int(len(np.unique(plots))),
        "mse_real": float(real.mean()),
        "mse_persist": float(persist.mean()),
        "cosine_real": float(cos_real.mean()),
        "cosine_persist": float(cos_persist.mean()),
        "relative_improvement_vs_persistence": float(
            (persist.mean() - real.mean()) / persist.mean()
        ),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    ref = dcfg["reference_s3_1"]
    a0["matches_s3_1_reference"] = {
        "mse_real_diff": a0["mse_real"] - float(ref["mse_real"]),
        "abs_diff_lt_1e_4": abs(a0["mse_real"] - float(ref["mse_real"])) < 1e-4,
    }

    # per-sample parity against the stored S3 parquet
    stored_path = Path(args.s3_predictions)
    if stored_path.exists():
        stored = pd.read_parquet(stored_path).set_index("pair_id")
        now = pd.DataFrame({
            "pair_id": val_df.pair_id.to_numpy(),
            "mse_real": real,
            "mse_persist": persist,
        }).set_index("pair_id")
        merged = stored[["mse_pred", "mse_persist"]].join(
            now, how="inner", lsuffix="_s3", rsuffix="_s3_2"
        )
        a0["s3_parquet_parity"] = {
            "n_matched": int(len(merged)),
            "max_abs_diff_mse": float(np.abs(merged.mse_pred - merged.mse_real).max()),
            "max_abs_diff_persist": float(
                np.abs(merged.mse_persist_s3 - merged.mse_persist_s3_2).max()
            ),
        }
    result["A0_real_delta"] = a0

    # ---------------- conditions ----------------------------------------
    conditions: dict[str, np.ndarray] = {}
    condition_meta: dict[str, dict] = {}
    permuted_delta_columns: dict[str, list[np.ndarray]] = {}
    # z_a / z_b must be identical in every condition; the persistence error is
    # the runtime witness.  Record the max deviation from the A0 vector.
    persist_dev: dict[str, float] = {}

    # A1 -- global permutation
    a1 = []
    a1_deltas = []
    dev = []
    t0 = time.time()
    for s in range(n_seeds):
        seed = seed_base + s
        d = global_permutation(deltas, seed)
        a1_deltas.append(d)
        out = e1_forward(model, val_df, eval_cfg, device, delta_override=d)
        a1.append(out["mse_pred"])
        dev.append(float(np.abs(out["mse_persist"] - persist).max()))
    persist_dev["A1_global_permutation"] = max(dev)
    conditions["A1_global_permutation"] = np.stack(a1).astype(np.float64)
    permuted_delta_columns["A1"] = a1_deltas
    condition_meta["A1_global_permutation"] = {
        "description": "one permutation of the real delta_days over the whole validation set",
        "method": "numpy.random.default_rng(seed).permutation(n)",
        "seeds": [seed_base + s for s in range(n_seeds)],
        "n_changed_per_seed": [
            int(np.count_nonzero(x != deltas)) for x in a1_deltas
        ],
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    # A2 -- permutation inside each pair_type
    a2 = []
    a2_deltas = []
    a2_groups: dict | None = None
    a2_preserved: list[bool] = []
    dev = []
    t0 = time.time()
    for s in range(n_seeds):
        seed = seed_base + s
        d, sizes, ok = permute_within_groups(
            deltas, span, dcfg["within_group_seed_base"], seed, stream_offset=1
        )
        a2_deltas.append(d)
        a2_groups = sizes
        a2_preserved.append(ok)
        out = e1_forward(model, val_df, eval_cfg, device, delta_override=d)
        a2.append(out["mse_pred"])
        dev.append(float(np.abs(out["mse_persist"] - persist).max()))
    persist_dev["A2_within_pair_type_permutation"] = max(dev)
    conditions["A2_within_pair_type_permutation"] = np.stack(a2).astype(np.float64)
    permuted_delta_columns["A2"] = a2_deltas
    condition_meta["A2_within_pair_type_permutation"] = {
        "description": (
            "permutation restricted to each pair_type; span7/span14 kept separate, "
            "the short/long class of every sample is preserved"
        ),
        "seeds": [seed_base + s for s in range(n_seeds)],
        "rng": (
            "numpy.random.default_rng([within_group_seed_base + seed, 1, group_ordinal]); "
            "one independent stream per group per seed"
        ),
        "groups": a2_groups,
        "multiset_preserved_all_seeds": bool(all(a2_preserved)),
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    # A3 -- per-pair_type train mean
    t0 = time.time()
    train_means = {
        str(k): float(v)
        for k, v in train_df.groupby("pair_type").delta_days.mean().items()
    }
    a3_delta = np.array([train_means[s] for s in span], dtype=np.float64)
    a3_out = e1_forward(model, val_df, eval_cfg, device, delta_override=a3_delta)
    persist_dev["A3_pair_type_train_mean"] = float(
        np.abs(a3_out["mse_persist"] - persist).max()
    )
    conditions["A3_pair_type_train_mean"] = a3_out["mse_pred"][None, :].astype(np.float64)
    condition_meta["A3_pair_type_train_mean"] = {
        "description": (
            "each sample receives the train-split mean delta_days of its own pair_type"
        ),
        "train_means": train_means,
        "validation_means_for_reference_only": {
            str(k): float(v)
            for k, v in val_df.groupby("pair_type").delta_days.mean().items()
        },
        "rule": "means fitted on the official train split only; validation never re-estimates them",
        "n_distinct_values_used": int(len(np.unique(a3_delta))),
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    # A4 -- permutation inside (pair_type, harvest_year)
    a4 = []
    a4_deltas = []
    a4_groups: dict | None = None
    a4_preserved: list[bool] = []
    strata = np.array([f"{s}|{y}" for s, y in zip(span, year)], dtype=object)
    dev = []
    t0 = time.time()
    for s in range(n_seeds):
        seed = seed_base + s
        d, sizes, ok = permute_within_groups(
            deltas, strata, dcfg["within_group_seed_base"], seed, stream_offset=2
        )
        a4_deltas.append(d)
        a4_groups = sizes
        a4_preserved.append(ok)
        out = e1_forward(model, val_df, eval_cfg, device, delta_override=d)
        a4.append(out["mse_pred"])
        dev.append(float(np.abs(out["mse_persist"] - persist).max()))
    persist_dev["A4_within_pair_type_year_permutation"] = max(dev)
    conditions["A4_within_pair_type_year_permutation"] = np.stack(a4).astype(np.float64)
    permuted_delta_columns["A4"] = a4_deltas
    condition_meta["A4_within_pair_type_year_permutation"] = {
        "description": "permutation inside each (pair_type, harvest_year) stratum",
        "seeds": [seed_base + s for s in range(n_seeds)],
        "rng": (
            "numpy.random.default_rng([within_group_seed_base + seed, 2, group_ordinal])"
        ),
        "groups": a4_groups,
        "multiset_preserved_all_seeds": bool(all(a4_preserved)),
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    # ---------------- summaries -----------------------------------------
    nb = int(dcfg["bootstrap_resamples"])
    bs = int(dcfg["bootstrap_seed"])
    result["conditions"] = {}
    for name, mat in conditions.items():
        block = summarize_condition(mat, real, plots, span, year, deltas.astype(int),
                                    nb, bs)
        block.update(condition_meta[name])
        result["conditions"][name] = block

    result["A5_by_real_delta"] = per_delta_table(
        real, persist, cos_real, cos_persist, conditions, deltas.astype(int), plots,
        [int(v) for v in dcfg["delta_values"]],
    )
    result["A5_note"] = (
        "Groups are reported separately.  Sample counts differ substantially "
        "between interval values, so no unweighted average across delta groups is "
        "taken."
    )

    # z_a / z_b / pair_id invariance: the persistence error can only be
    # unchanged if every condition saw exactly the same visual inputs and the
    # same row order.  A zero deviation is the runtime witness for D6.
    result["invariants"] = {
        "persistence_mse_real": float(persist.mean()),
        "max_abs_persist_deviation_by_condition": persist_dev,
        "z_and_pair_order_unchanged": bool(
            all(v == 0.0 for v in persist_dev.values())
        ),
        "note": (
            "Each condition re-uses the same frozen index order and the same "
            "DataLoader; only the delta_days tensor passed to the predictor is "
            "replaced.  The persistence error depends only on z_a and z_b, so an "
            "exactly unchanged persistence column is the runtime evidence that "
            "z_a, z_b and pair_id were not touched."
        ),
    }

    # ---------------- artifacts -----------------------------------------
    pred = pd.DataFrame({
        "pair_id": val_df.pair_id.to_numpy(),
        "plot_uid": plots,
        "pair_type": span,
        "harvest_year": year,
        "delta_days": deltas.astype(int),
        "mse_real": real,
        "mse_persist": persist,
        "cosine_real": cos_real,
        "cosine_persist": cos_persist,
        "mse_a3": conditions["A3_pair_type_train_mean"][0],
    })
    for name, mat in conditions.items():
        if name == "A3_pair_type_train_mean":
            continue
        tag = name.split("_")[0].lower()
        for s in range(mat.shape[0]):
            pred[f"{tag}_s{seed_base + s}"] = mat[s]
    pred.to_parquet(args.predictions, index=False)

    perm = pd.DataFrame({
        "pair_id": val_df.pair_id.to_numpy(),
        "delta_original": deltas.astype(int),
        "delta_a3": a3_delta,
    })
    for tag, cols in permuted_delta_columns.items():
        for s, arr in enumerate(cols):
            perm[f"{tag.lower()}_s{seed_base + s}"] = arr.astype(int)
    perm.to_parquet(args.permuted_deltas, index=False)

    result["artifacts"] = {
        "per_sample_mse": args.predictions,
        "permuted_deltas": args.permuted_deltas,
        "n_rows_per_sample_mse": int(len(pred)),
        "n_rows_permuted_deltas": int(len(perm)),
    }
    result["elapsed_seconds"] = round(time.time() - t_start, 1)
    result["git_head"] = git("rev-parse", "HEAD")

    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)

    summary = {
        "A0_mse_real": a0["mse_real"],
        "conditions": {
            name: {
                "mse_mean": blk["mse_condition_mean"],
                "relative_increase_mean": blk["relative_increase_mean"],
                "ci95": [
                    blk["relative_increase_bootstrap"]["ci95_low"],
                    blk["relative_increase_bootstrap"]["ci95_high"],
                ],
            }
            for name, blk in result["conditions"].items()
        },
    }
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {out_path}, {args.predictions}, {args.permuted_deltas}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
