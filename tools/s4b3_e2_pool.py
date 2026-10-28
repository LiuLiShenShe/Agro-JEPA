#!/usr/bin/env python3
"""S4B.3B E2 stage 1: pooled feature dump (no labels, no training).

Loads every official-train + test pair's ``z_a`` once and runs each of the
frozen conditions (B2/B2-C/B3) exactly once, writing per-condition pooled
matrices and a combined metadata parquet.  All downstream readout training
happens in ``s4b3_e2_readout.py`` (CPU, re-runnable) so the GPU work is paid
once and the expensive epoch-selection loops do not re-trigger it.

Outputs
-------
* ``external_data/s4b3_runs/e2_pooled/A_current.npz``  (pool_a, [N,768])
* ``external_data/s4b3_runs/e2_pooled/{B2,B2-C,B3}.npz``  (pool_b, [N,768])
* ``external_data/s4b3_runs/e2_pooled/combined_meta.parquet``
* ``external_data/s4b3_runs/e2_pooled/manifest.json``

Usage
-----
    CUDA_VISIBLE_DEVICES=1 python3 tools/s4b3_e2_pool.py \
        --config agro_jepa/config/s4b3_official_test.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from s4b2_validation_e1 import forward_batch, load_s4b2_model  # noqa: E402

COND_KEY = {"B2": "b2_no_time", "B2-C": "b2_no_time_constant_delta",
            "B3": "b3_delta"}
LABEL_COLS = ["cc_b", "ph_b", "cc_b_lag", "ph_b_lag", "cc_mask_b", "ph_mask_b",
              "cc_b_date", "ph_b_date"]


class ZADataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.rows = frame[["cache_file_a", "delta_days"]].to_dict("records")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        row = self.rows[i]
        arr = np.load(ROOT / row["cache_file_a"])
        return {
            "z_a": torch.from_numpy(arr.astype(np.float32)),
            "delta_days": torch.tensor(float(row["delta_days"]), dtype=torch.float32),
        }


@torch.no_grad()
def collect(models_by_cond, frame, device, batch_size, num_workers):
    ds = ZADataset(frame)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    pool_a, pool_b = [], {c: [] for c in models_by_cond}
    for batch in loader:
        z_a = batch["z_a"].to(device, non_blocking=True)
        pool_a.append(z_a.mean(dim=1).cpu().numpy())
        for cond, models in models_by_cond.items():
            pred = None
            for m in models:
                out = forward_batch(m, batch, device).float()
                pred = out if pred is None else pred + out
            pool_b[cond].append((pred / len(models)).mean(dim=1).cpu().numpy())
    return (np.concatenate(pool_a, axis=0).astype(np.float32),
            {c: np.concatenate(v, axis=0).astype(np.float32)
             for c, v in pool_b.items()})


def add_cache_columns(index, feature_index):
    stem2cache = dict(zip(
        feature_index.image_filename_stem.astype(str).str.upper(),
        feature_index.cache_file.astype(str)))
    out = index.copy()
    out["cache_file_a"] = out.image_stem_a.astype(str).str.upper().map(stem2cache)
    out["cache_file_b"] = out.image_stem_b.astype(str).str.upper().map(stem2cache)
    if out.cache_file_a.isna().any() or out.cache_file_b.isna().any():
        raise SystemExit("some endpoints have no cached feature")
    return out.reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s4b3_official_test.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=16)
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    t0 = time.time()
    device = torch.device(args.device or (cfg["device"]["cuda"]
                          if torch.cuda.is_available() else "cpu"))
    seeds = [int(s) for s in cfg["e2"]["seeds"]]

    feature_index = pd.read_parquet(ROOT / cfg["data"]["feature_index"])
    s2 = pd.read_parquet(ROOT / cfg["data"]["s2_pairs"])

    # train frame
    ti = pd.read_parquet(ROOT / cfg["data"]["train_cache_index"])
    ft = pd.read_parquet(ROOT / cfg["data"]["full_train_pairs"],
                         columns=["pair_id", "plot_uid", "harvest_year", "split",
                                  "pair_type", "delta_days", "image_stem_a",
                                  "image_stem_b", "date_a", "date_b",
                                  "genotype_id"] + LABEL_COLS)
    keep = ["pair_id", "plot_uid", "harvest_year", "split", "pair_type",
            "delta_days", "image_stem_a", "image_stem_b", "date_a", "date_b",
            "genotype_id", "cache_file_a", "cache_file_b"]
    train = ti[keep].merge(ft[["pair_id"] + LABEL_COLS], on="pair_id", how="left")
    train = add_cache_columns(train, feature_index)
    print(f"[data] train {len(train)} pairs / {train.plot_uid.nunique()} plots",
          flush=True)
    frames = [train]
    for split in cfg["test_splits"]:
        spec = cfg["test_splits"][split]
        idx = pd.read_parquet(ROOT / spec["index"])
        if sha256_file_path(ROOT / spec["index"]) != spec["index_sha256"]:
            raise SystemExit(f"{split}: index SHA mismatch")
        s2d = (s2[s2.pair_type.isin(["span7", "span14"])]
               .sort_values(["plot_uid", "image_stem_a", "image_stem_b",
                             "span_days", "pair_id"], kind="mergesort")
               .drop_duplicates(["plot_uid", "image_stem_a", "image_stem_b"],
                                keep="last"))
        te = idx.merge(s2d[["pair_id"] + LABEL_COLS], on="pair_id", how="left")
        if len(te) != len(idx):
            raise SystemExit(f"{split}: label merge changed row count")
        te = add_cache_columns(te, feature_index)
        te["split"] = split
        frames.append(te)
        print(f"[data] {split} {len(te)} pairs", flush=True)
    combined = pd.concat(frames, ignore_index=True)
    combined["_pool_index"] = np.arange(len(combined), dtype=np.int64)
    print(f"[data] combined {len(combined)} rows", flush=True)

    models_by_cond = {}
    for cond in ("B2", "B2-C", "B3"):
        key = COND_KEY[cond]
        models_by_cond[cond] = [
            load_s4b2_model(str(ROOT / cfg["frozen"]["checkpoints"][key][f"seed{s}"]),
                            device)[0]
            for s in seeds]
    pool_a, pool_b = collect(models_by_cond, combined, device,
                             args.batch_size, args.num_workers)
    del models_by_cond
    torch.cuda.empty_cache()

    pooled_dir = ROOT / cfg["artifacts"]["e2_pooled_dir"]
    pooled_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "stage": "S4B.3",
        "task": "E2 pooled feature dump",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_head": __import__("subprocess").check_output(
            ["git", "rev-parse", "HEAD"]).decode().strip(),
        "n_rows": int(len(combined)),
        "files": {},
    }
    Path(pooled_dir / "A_current.npz").parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pooled_dir / "A_current.npz", pool_a=pool_a)
    manifest["files"]["A_current"] = {
        "path": "A_current.npz", "shape": list(pool_a.shape),
        "sha256": sha256_file_path(pooled_dir / "A_current.npz")}
    for cond in ("B2", "B2-C", "B3"):
        np.savez_compressed(pooled_dir / f"{cond}.npz", pool_b=pool_b[cond])
        manifest["files"][cond] = {
            "path": f"{cond}.npz", "shape": list(pool_b[cond].shape),
            "sha256": sha256_file_path(pooled_dir / f"{cond}.npz")}
    combined.drop(columns=["cache_file_a", "cache_file_b"]).to_parquet(
        pooled_dir / "combined_meta.parquet", index=False)
    with open(pooled_dir / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[pool] wrote {len(combined)} rows; A_current+3 conditions", flush=True)
    print(f"elapsed {round(time.time() - t0, 1)} s", flush=True)
    return 0


def sha256_file_path(p: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
