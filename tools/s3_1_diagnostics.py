#!/usr/bin/env python3
"""S3.1 diagnostics: provenance re-check, global delta permutation, plot bootstrap.

Three independent checks, all on saved S3 artifacts only (no retraining):

1. **Provenance** -- load the existing ``best.pt`` from a clean ``c1fdb88``
   working tree, recompute the E1 validation predictions and compare them with
   the stored ``s3_v0_val_predictions.parquet`` (max/mean absolute difference).
2. **Global delta permutation** -- S3's original diagnostic permuted
   ``delta_days`` *inside each batch*.  That is not a global permutation.  This
   tool permutes the real ``delta_days`` across the **entire** validation set
   with fixed seeds, keeps ``z_a``/``z_b``/labels fixed, and reports overall and
   per-span MSE.  The old batch-local number is carried through as history.
3. **Plot-level cluster bootstrap** -- 2,000 resamples of ``plot_uid`` (all
   pairs of a sampled plot kept together) for the E1 relative MSE reductions
   and, when the per-sample probe predictions exist, for the E2 A -> B changes.

Usage
-----
    CUDA_VISIBLE_DEVICES=1 python3 tools/s3_1_diagnostics.py \
        --config agro_jepa/config/s3_v0.yaml \
        --checkpoint external_data/s3_runs/v0/checkpoints/best.pt \
        --e1-predictions reports/s3_v0_val_predictions.parquet \
        --out reports/s3_1_diagnostic_audit.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agro_jepa" / "eval"))

from agro_jepa.datasets.s3_pairs import (  # noqa: E402
    S3FeaturePairDataset,
    index_sha256,
)
from agro_jepa.eval.eval_v0 import load_model, set_seed  # noqa: E402

SOURCE_FILES = [
    "agro_jepa/models/fip_dino_wm.py",
    "agro_jepa/models/time_embedding.py",
    "agro_jepa/datasets/s3_pairs.py",
    "agro_jepa/train/train_v0.py",
    "agro_jepa/eval/eval_v0.py",
    "agro_jepa/config/s3_v0.yaml",
]


# --------------------------------------------------------------------------
# provenance helpers
# --------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()


def source_state() -> dict:
    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    diff = git("diff", "--stat", "HEAD")
    files = {}
    for rel in SOURCE_FILES:
        path = ROOT / rel
        blob_sha = git("rev-parse", f"HEAD:{rel}")
        local_sha = git("hash-object", rel)
        files[rel] = {
            "sha256_working_tree": sha256_file(path) if path.exists() else None,
            "git_blob_sha_head": blob_sha,
            "git_blob_sha_working_tree": local_sha,
            "matches_head_blob": bool(blob_sha) and blob_sha == local_sha,
        }
    return {
        "head": head,
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "status_porcelain": status.splitlines(),
        "diff_stat_head": diff,
        "source_files": files,
    }


# --------------------------------------------------------------------------
# E1 recomputation + global delta permutation
# --------------------------------------------------------------------------
@torch.no_grad()
def e1_forward(
    model,
    val_df: pd.DataFrame,
    cfg: dict,
    device: torch.device,
    delta_override: np.ndarray | None = None,
) -> dict:
    """One full validation pass, optionally with globally replaced deltas."""
    ds = S3FeaturePairDataset(val_df, traits=())
    loader = DataLoader(
        ds, batch_size=int(cfg["eval"]["batch_size"]), shuffle=False,
        num_workers=4, pin_memory=True,
    )
    mse_pred, mse_persist, cos_pred, cos_persist = [], [], [], []
    offset = 0
    for batch in loader:
        z_a = batch["z_a"].to(device, non_blocking=True)
        z_b = batch["z_b"].to(device, non_blocking=True)
        bs = z_a.shape[0]
        if delta_override is None:
            delta = batch["delta_days"].to(device, non_blocking=True)
        else:
            delta = torch.as_tensor(
                delta_override[offset:offset + bs], dtype=torch.float32, device=device
            )
        offset += bs
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z_hat = model(z_a, delta)
        z_hat = z_hat.float()
        mse_pred.append((z_hat - z_b).pow(2).mean(dim=(1, 2)).cpu().numpy())
        mse_persist.append((z_a - z_b).pow(2).mean(dim=(1, 2)).cpu().numpy())
        cos_pred.append(torch.nn.functional.cosine_similarity(
            z_hat.flatten(1), z_b.flatten(1), dim=-1).cpu().numpy())
        cos_persist.append(torch.nn.functional.cosine_similarity(
            z_a.flatten(1), z_b.flatten(1), dim=-1).cpu().numpy())
    return {
        "mse_pred": np.concatenate(mse_pred),
        "mse_persist": np.concatenate(mse_persist),
        "cos_pred": np.concatenate(cos_pred),
        "cos_persist": np.concatenate(cos_persist),
    }


def summarize_vec(mse_pred: np.ndarray, mse_persist: np.ndarray) -> dict:
    p = float(mse_pred.mean())
    q = float(mse_persist.mean())
    return {
        "n_samples": int(len(mse_pred)),
        "mse_pred": p,
        "mse_persist": q,
        "relative_improvement": (q - p) / q if q else None,
    }


# --------------------------------------------------------------------------
# cluster bootstrap
# --------------------------------------------------------------------------
def cluster_bootstrap(
    tables: dict[str, np.ndarray],
    plot_ids: np.ndarray,
    metric,
    n_resamples: int,
    seed: int,
) -> dict:
    """Resample ``plot_ids`` with replacement; keep all rows of a plot."""
    uniq = np.unique(plot_ids)
    index_of = {p: np.flatnonzero(plot_ids == p) for p in uniq}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_resamples):
        picks = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([index_of[p] for p in picks])
        values.append(metric({k: v[idx] for k, v in tables.items()}))
    arr = np.asarray(values, dtype=float)
    return {
        "n_resamples": int(n_resamples),
        "n_plots": int(len(uniq)),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)),
        "ci95_low": float(np.percentile(arr, 2.5)),
        "ci95_high": float(np.percentile(arr, 97.5)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def bootstrap_e1(pred_df: pd.DataFrame, n: int, seed: int) -> dict:
    base = {
        "mse_pred": pred_df.mse_pred.to_numpy(np.float64),
        "mse_persist": pred_df.mse_persist.to_numpy(np.float64),
        "mse_train_mean": pred_df.mse_train_mean.to_numpy(np.float64),
    }
    plots = pred_df.plot_uid.to_numpy()

    def rel_persist(t):
        return (t["mse_persist"].mean() - t["mse_pred"].mean()) / t["mse_persist"].mean()

    def rel_mean(t):
        return (t["mse_train_mean"].mean() - t["mse_pred"].mean()) / t["mse_train_mean"].mean()

    out = {
        "unit": "plot_uid (all pairs of a resampled plot are kept together)",
        "resampling": "with replacement",
        "seed": seed,
        "overall": {
            "relative_improvement_vs_persistence": cluster_bootstrap(
                base, plots, rel_persist, n, seed),
            "relative_improvement_vs_train_mean": cluster_bootstrap(
                base, plots, rel_mean, n, seed),
            "mse_pred": cluster_bootstrap(
                base, plots, lambda t: t["mse_pred"].mean(), n, seed),
            "mse_persist": cluster_bootstrap(
                base, plots, lambda t: t["mse_persist"].mean(), n, seed),
            "mse_train_mean": cluster_bootstrap(
                base, plots, lambda t: t["mse_train_mean"].mean(), n, seed),
        },
        "by_span": {},
    }
    for span in ("span7", "span14"):
        sub = pred_df[pred_df.pair_type == span]
        if sub.empty:
            continue
        out["by_span"][span] = {
            "n_samples": int(len(sub)),
            "n_plots": int(sub.plot_uid.nunique()),
            "relative_improvement_vs_persistence": cluster_bootstrap(
                {k: v[pred_df.pair_type.to_numpy() == span] for k, v in base.items()},
                sub.plot_uid.to_numpy(), rel_persist, n, seed),
            "relative_improvement_vs_train_mean": cluster_bootstrap(
                {k: v[pred_df.pair_type.to_numpy() == span] for k, v in base.items()},
                sub.plot_uid.to_numpy(), rel_mean, n, seed),
        }
    return out


def _metrics(y, pa, pb) -> dict:
    ea = pa - y
    eb = pb - y
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2a = 1.0 - np.sum(ea ** 2) / ss_tot if ss_tot > 0 else np.nan
    r2b = 1.0 - np.sum(eb ** 2) / ss_tot if ss_tot > 0 else np.nan
    return {
        "mae_a": np.mean(np.abs(ea)),
        "mae_b": np.mean(np.abs(eb)),
        "rmse_a": np.sqrt(np.mean(ea ** 2)),
        "rmse_b": np.sqrt(np.mean(eb ** 2)),
        "r2_a": r2a,
        "r2_b": r2b,
        "delta_mae": np.mean(np.abs(eb)) - np.mean(np.abs(ea)),
        "delta_rmse": np.sqrt(np.mean(eb ** 2)) - np.sqrt(np.mean(ea ** 2)),
        "delta_r2": r2b - r2a,
    }


def bootstrap_e2(probe_preds: pd.DataFrame, n: int, seed: int) -> dict:
    out = {"unit": "plot_uid (all probe rows of a resampled plot kept together)",
           "resampling": "with replacement", "seed": seed,
           "seed_averaging": "per-sample predictions averaged over the 3 S3 probe seeds",
           "scopes": {}}
    for trait in ("cc", "ph"):
        out["scopes"][trait] = {}
        for scope in ("overall", "span7", "span14"):
            sub = probe_preds[(probe_preds.trait == trait) & (probe_preds.scope == scope)]
            if sub.empty:
                continue
            agg = sub.groupby(["condition", "pair_id", "plot_uid"], as_index=False).agg(
                y_true=("y_true", "first"), y_pred=("y_pred", "mean"))
            a = agg[agg.condition == "A_current"].set_index("pair_id")
            b = agg[agg.condition == "B_predicted"].set_index("pair_id")
            common = a.index.intersection(b.index)
            a = a.loc[common]
            b = b.loc[common]
            y = a.y_true.to_numpy(np.float64)
            tables = {
                "y": y,
                "pa": a.y_pred.to_numpy(np.float64),
                "pb": b.y_pred.to_numpy(np.float64),
            }
            plots = a.plot_uid.to_numpy()
            out["scopes"][trait][scope] = {
                "n_samples": int(len(y)),
                "n_plots": int(len(np.unique(plots))),
            }
            for key in ("mae_a", "mae_b", "rmse_a", "rmse_b", "r2_a", "r2_b",
                        "delta_mae", "delta_rmse", "delta_r2"):
                out["scopes"][trait][scope][key] = cluster_bootstrap(
                    tables, plots, lambda t, k=key: _metrics(t["y"], t["pa"], t["pb"])[k],
                    n, seed)
    return out


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s3_v0.yaml")
    ap.add_argument("--checkpoint", default="external_data/s3_runs/v0/checkpoints/best.pt")
    ap.add_argument("--e1-predictions", default="reports/s3_v0_val_predictions.parquet")
    ap.add_argument("--probe-predictions", default="reports/s3_1_probe_val_predictions.parquet")
    ap.add_argument("--results", default="reports/s3_v0_results.json")
    ap.add_argument("--out", default="reports/s3_1_diagnostic_audit.json")
    ap.add_argument("--permutation-seeds", type=int, default=20)
    ap.add_argument("--bootstrap-resamples", type=int, default=2000)
    ap.add_argument("--bootstrap-seed", type=int, default=20261024)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    set_seed(0)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    index = pd.read_parquet(cfg["data"]["index_out"])
    val_df = index[index.split == "validation"].reset_index(drop=True)
    train_df = index[index.split == "train"].reset_index(drop=True)
    old_pred = pd.read_parquet(args.e1_predictions)
    old_results = json.loads(Path(args.results).read_text())

    audit: dict = {
        "stage": "S3.1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": " ".join(sys.argv),
        "config": args.config,
        "checkpoint": args.checkpoint,
    }
    audit["source_state"] = source_state()

    model, ckpt = load_model(cfg, Path(args.checkpoint), device)
    checkpoint_info = {
        "path": args.checkpoint,
        "sha256": sha256_file(Path(args.checkpoint)),
        "format": ckpt.get("format"),
        "epoch": int(ckpt["epoch"]),
        "global_step": int(ckpt["global_step"]),
        "git_commit_recorded_at_training": ckpt.get("git_commit"),
        "manifest_id": ckpt.get("manifest_id"),
        "index_sha256_recorded": ckpt.get("index_sha256"),
        "index_sha256_recomputed": index_sha256(index),
    }
    checkpoint_info["index_sha256_matches"] = (
        checkpoint_info["index_sha256_recorded"]
        == checkpoint_info["index_sha256_recomputed"]
    )
    audit["checkpoint"] = checkpoint_info

    # ---------------- 1. clean-tree reproduction -------------------------
    t0 = time.time()
    rep = e1_forward(model, val_df, cfg, device)
    merged = old_pred.set_index("pair_id").join(
        pd.DataFrame({
            "pair_id": val_df.pair_id.to_numpy(),
            "mse_pred_recomputed": rep["mse_pred"],
            "mse_persist_recomputed": rep["mse_persist"],
            "cos_pred_recomputed": rep["cos_pred"],
            "cos_persist_recomputed": rep["cos_persist"],
        }).set_index("pair_id"),
        how="inner",
    )
    diffs = {
        "n_matched_pairs": int(len(merged)),
        "n_stored_pairs": int(len(old_pred)),
        "mse_pred_max_abs_diff": float(
            np.abs(merged.mse_pred_recomputed - merged.mse_pred).max()),
        "mse_pred_mean_abs_diff": float(
            np.abs(merged.mse_pred_recomputed - merged.mse_pred).mean()),
        "mse_persist_max_abs_diff": float(
            np.abs(merged.mse_persist_recomputed - merged.mse_persist).max()),
        "cos_pred_max_abs_diff": float(
            np.abs(merged.cos_pred_recomputed - merged.cosine_pred).max()),
        "aggregate": {
            "stored_mse_pred": float(old_pred.mse_pred.mean()),
            "recomputed_mse_pred": float(rep["mse_pred"].mean()),
            "stored_mse_persist": float(old_pred.mse_persist.mean()),
            "recomputed_mse_persist": float(rep["mse_persist"].mean()),
            "stored_cos_pred": float(old_pred.cosine_pred.mean()),
            "recomputed_cos_pred": float(rep["cos_pred"].mean()),
        },
    }
    audit["reproduction"] = {
        **diffs,
        "elapsed_seconds": round(time.time() - t0, 1),
        "note": (
            "Per-sample parity is checked against the stored parquet. The stored "
            "file is authoritative for the S3 report; this pass only re-runs the "
            "committed model code on the committed checkpoint."
        ),
    }

    # ---------------- 2. global delta permutation ------------------------
    t0 = time.time()
    deltas = val_df.delta_days.to_numpy(np.float64)
    n = len(val_df)
    span = val_df.pair_type.to_numpy()
    train_mean = float(train_df.delta_days.mean())
    perm_runs = []
    for s in range(args.permutation_seeds):
        perm_idx = np.random.default_rng(10_000 + s).permutation(n)
        perm = e1_forward(model, val_df, cfg, device, delta_override=deltas[perm_idx])
        rec = {
            "seed": 10_000 + s,
            "overall": summarize_vec(perm["mse_pred"], perm["mse_persist"]),
            "by_span": {
                sp: summarize_vec(perm["mse_pred"][span == sp],
                                  perm["mse_persist"][span == sp])
                for sp in ("span7", "span14")
            },
        }
        perm_runs.append(rec)
    const = e1_forward(
        model, val_df, cfg, device,
        delta_override=np.full(n, train_mean, dtype=np.float64),
    )
    const_by_span = {
        sp: summarize_vec(const["mse_pred"][span == sp],
                          const["mse_persist"][span == sp])
        for sp in ("span7", "span14")
    }
    orig = summarize_vec(rep["mse_pred"], rep["mse_persist"])
    orig_struct = {
        "overall": orig,
        "by_span": {
            sp: summarize_vec(rep["mse_pred"][span == sp],
                              rep["mse_persist"][span == sp])
            for sp in ("span7", "span14")
        },
    }

    def agg_runs(key_path: list[str]) -> dict:
        vals = []
        for r in perm_runs:
            node = r
            for k in key_path:
                node = node[k]
            vals.append(node["mse_pred"])
        arr = np.array(vals)
        base = orig_struct
        for k in key_path:
            base = base[k]
        return {
            "n_seeds": int(len(arr)),
            "mse_pred_mean": float(arr.mean()),
            "mse_pred_std": float(arr.std(ddof=1)),
            "mse_pred_min": float(arr.min()),
            "mse_pred_max": float(arr.max()),
            "mse_increase_mean": float((arr.mean() - base["mse_pred"]) / base["mse_pred"]),
            "mse_increase_min": float((arr.min() - base["mse_pred"]) / base["mse_pred"]),
            "mse_increase_max": float((arr.max() - base["mse_pred"]) / base["mse_pred"]),
            "baseline_mse_pred_original": float(base["mse_pred"]),
        }

    batch_local = old_results.get("E1_future_latent", {})
    audit["delta_permutation"] = {
        "history": {
            "scope": "batch-local (each eval batch permuted independently)",
            "batch_size": int(cfg["eval"]["batch_size"]),
            "mse_pred_original": old_results["E1_future_latent"]["mse_pred"],
            "mse_pred_batch_shuffled": old_results["E1_future_latent"].get(
                "mse_pred_delta_shuffled"),
            "mse_increase": old_results["E1_future_latent"].get(
                "delta_shuffle_mse_increase"),
            "interpretation": (
                "Retained as a historical record only. Because only 256 samples are "
                "permuted with each other inside a batch, most samples keep a delta "
                "close to their own; this number cannot be used to claim the model "
                "is insensitive to the interval."
            ),
        },
        "global_method": {
            "rule": ("one permutation of the real delta_days over the whole "
                     "validation set; z_a, z_b and labels unchanged"),
            "n_samples": int(n),
            "n_plots": int(val_df.plot_uid.nunique()),
            "seeds": [10_000 + s for s in range(args.permutation_seeds)],
            "rng": "numpy.random.default_rng(seed).permutation(n)",
            "batch_alignment": ("the global permutation is sliced by the sequential "
                                "DataLoader offset, so the permutation is not batch-local"),
        },
        "original": orig,
        "original_by_span": orig_struct["by_span"],
        "constant_train_mean": {
            "delta_value": train_mean,
            **summarize_vec(const["mse_pred"], const["mse_persist"]),
            "mse_increase_vs_original": float(
                (const["mse_pred"].mean() - orig["mse_pred"]) / orig["mse_pred"]),
            "by_span": {
                sp: {
                    **const_by_span[sp],
                    "mse_increase_vs_original": float(
                        (const_by_span[sp]["mse_pred"] - orig_struct["by_span"][sp]["mse_pred"])
                        / orig_struct["by_span"][sp]["mse_pred"]),
                }
                for sp in ("span7", "span14")
            },
        },
        "global_permutation": {
            "overall": agg_runs(["overall"]) | {
                "mse_persist": float(orig["mse_persist"]),
            },
            "by_span": {
                sp: agg_runs(["by_span", sp]) for sp in ("span7", "span14")
            },
            "per_seed": perm_runs,
        },
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    # ---------------- 3. plot-level bootstrap ----------------------------
    t0 = time.time()
    audit["bootstrap_e1"] = bootstrap_e1(
        old_pred, args.bootstrap_resamples, args.bootstrap_seed)
    probe_path = Path(args.probe_predictions)
    if probe_path.exists():
        probe_preds = pd.read_parquet(probe_path)
        audit["bootstrap_e2"] = bootstrap_e2(
            probe_preds, args.bootstrap_resamples, args.bootstrap_seed)
    else:
        audit["bootstrap_e2"] = {
            "skipped": True,
            "reason": f"{probe_path} does not exist",
        }
    audit["bootstrap_elapsed_seconds"] = round(time.time() - t0, 1)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(audit, fh, indent=2)
    print(json.dumps({
        "reproduction": {
            k: audit["reproduction"][k] for k in
            ("n_matched_pairs", "mse_pred_max_abs_diff", "mse_pred_mean_abs_diff")
        },
        "global_permutation_overall": audit["delta_permutation"][
            "global_permutation"]["overall"],
        "bootstrap_e1_overall": {
            k: audit["bootstrap_e1"]["overall"][k]["ci95_low"]
            for k in audit["bootstrap_e1"]["overall"]
        },
    }, indent=2), flush=True)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
