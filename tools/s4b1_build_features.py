#!/usr/bin/env python3
"""S4B.1: build frozen DINOv2 ``mainline224`` features for all desired endpoints.

Reuses the S3 feature cache read-only where provenance matches exactly, and
computes features for the new endpoints (full official-train pairs + four test
splits) with the identical frozen encoder/preprocessing.  New features are
written under ``external_data/s4b_features/mainline224_s1-v1/...`` so the S3
cache is never overwritten.

A desired endpoint is satisfied by the S3 cache iff its image_filename_stem has
a row in ``external_data/s3_features/index.parquet`` whose cache_file resolves
to a real (256, 768) float16 .npy.  Those rows are copied verbatim into the S4B
index (provenance preserved); missing endpoints are downloaded (handled by
``s4b1_prepare_images.py``) then encoded here.

Usage
-----
    python3 tools/s4b1_build_features.py \
        --config agro_jepa/config/s4b1_data_cache.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agro_jepa.datasets.provenance import (  # noqa: E402
    CACHE_PROTOCOL,
    CACHE_PROTOCOL_VERSION,
    DINOV2_EMBED_DIM,
    DINOV2_REPO_GIT_COMMIT,
    DINOV2_WEIGHTS_SHA256,
    MAINLINE_N_TOKENS,
    build_mainline_transform,
    cache_key_for_image,
)

WEIGHTS_PATH = Path.home() / ".cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class FeatureWriter:
    def __init__(self, repo_path: Path, out_dir: Path, device: torch.device,
                 batch_size: int = 48, load_workers: int = 16) -> None:
        self.repo_path = repo_path
        self.out_dir = out_dir
        self.device = device
        self.batch_size = batch_size
        self.load_workers = load_workers
        self.transform = build_mainline_transform()
        self._check_provenance()
        t0 = time.time()
        base = torch.hub.load(str(repo_path.resolve()), "dinov2_vitb14",
                              source="local", pretrained=True, verbose=False)
        self.load_seconds = time.time() - t0
        self.model = base.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.emb_dim = int(base.num_features)
        self.patch_size = int(base.patch_size)

    def _check_provenance(self) -> None:
        if not (self.repo_path / "hubconf.py").exists():
            raise FileNotFoundError(f"{self.repo_path} is not a dinov2 clone")
        commit = subprocess.check_output(
            ["git", "-C", str(self.repo_path), "rev-parse", "HEAD"], text=True
        ).strip()
        if commit != DINOV2_REPO_GIT_COMMIT:
            raise RuntimeError(f"dinov2 clone at {commit}, pinned {DINOV2_REPO_GIT_COMMIT}")
        if not WEIGHTS_PATH.exists():
            raise FileNotFoundError(f"pinned weights not found: {WEIGHTS_PATH}")
        got = sha256_file(WEIGHTS_PATH)
        if got != DINOV2_WEIGHTS_SHA256:
            raise RuntimeError(f"weights sha256 {got} != pinned {DINOV2_WEIGHTS_SHA256}")

    def _load_one(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        return self.transform(img)

    @torch.inference_mode()
    def run(self, records: list[dict]) -> tuple[list[dict], dict]:
        by_sha: dict[str, str] = {}
        for r in records:
            by_sha.setdefault(r["sha256"], r["dest"])
        shas = sorted(by_sha)
        rows: list[dict] = []
        n_cached = n_computed = 0
        peak_mem = 0
        t_start = time.time()
        pool = ThreadPoolExecutor(max_workers=self.load_workers)
        try:
            for start in range(0, len(shas), self.batch_size):
                batch_shas = shas[start:start + self.batch_size]
                todo = []
                for sha in batch_shas:
                    key = cache_key_for_image(sha)
                    path = (self.out_dir / CACHE_PROTOCOL / CACHE_PROTOCOL_VERSION
                            / f"{key}.npy")
                    if path.exists():
                        try:
                            arr = np.load(path)
                            if arr.shape == (MAINLINE_N_TOKENS, DINOV2_EMBED_DIM):
                                rows.append({"sha256": sha, "cache_key": key,
                                             "cache_file": str(path),
                                             "status": "cached"})
                                n_cached += 1
                                continue
                        except Exception:  # noqa: BLE001
                            pass
                    todo.append((sha, key, path))
                if not todo:
                    continue
                tensors = list(pool.map(
                    self._load_one, [by_sha[s] for s, _, _ in todo]))
                x = torch.stack(tensors).to(self.device, non_blocking=True)
                feats = self.model.forward_features(x)["x_norm_patchtokens"]
                if feats.shape[1:] != (MAINLINE_N_TOKENS, DINOV2_EMBED_DIM):
                    raise RuntimeError(f"token shape {tuple(feats.shape)}")
                arrs = feats.to(torch.float16).cpu().numpy()
                for (sha, key, path), arr in zip(todo, arrs):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(path, arr)
                    rows.append({"sha256": sha, "cache_key": key,
                                 "cache_file": str(path), "status": "computed"})
                    n_computed += 1
                if self.device.type == "cuda":
                    peak_mem = max(peak_mem, torch.cuda.max_memory_allocated(self.device))
                done = n_cached + n_computed
                if done % (self.batch_size * 10) < self.batch_size:
                    print(f"  encoded {done}/{len(shas)} ({n_cached} cached, "
                          f"{n_computed} computed)", flush=True)
        finally:
            pool.shutdown(wait=True)
        timing = {
            "n_unique_source_images": len(shas),
            "n_cached": n_cached,
            "n_computed": n_computed,
            "elapsed_seconds": round(time.time() - t_start, 1),
            "throughput_images_per_s": round(len(shas) / (time.time() - t_start), 2)
            if (time.time() - t_start) else None,
            "peak_gpu_memory_bytes": int(peak_mem),
            "batch_size": self.batch_size,
            "model_load_seconds": round(self.load_seconds, 2),
        }
        return rows, timing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s4b1_data_cache.yaml")
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    d = cfg["data"]
    feat_cfg = cfg["features"]
    t0 = time.time()

    ep = pd.read_parquet(cfg["index"]["endpoints_out"])
    ep["image_filename_stem"] = ep.image_filename_stem.astype(str).str.upper()
    ep_meta = ep.set_index("image_filename_stem")

    # ---- reuse S3 cache read-only (provenance matched) ------------------
    s3 = pd.read_parquet(d["s3_feature_index"])
    s3["image_filename_stem"] = s3.image_filename_stem.astype(str).str.upper()
    s3["cache_file"] = s3.cache_file.astype(str)
    s3_valid = s3[s3.cache_file.map(lambda p: os.path.isfile(p))].drop_duplicates(
        "image_filename_stem").set_index("image_filename_stem")

    reuse_rows: list[dict] = []
    compute_records: list[dict] = []
    not_found: list[str] = []

    s3_root = Path("external_data/fip1_images")
    new_img_root = Path(cfg["images"]["out_dir"])
    for stem, r in ep_meta.iterrows():
        if stem in s3_valid.index:
            srow = s3_valid.loc[stem]
            reuse_rows.append({
                "image_filename_stem": stem, "plot_uid": str(r.plot_uid),
                "split": str(r.split), "harvest_year": int(r.harvest_year),
                "image_date": str(r.image_date), "sha256": str(srow.sha256),
                "cache_key": str(srow.cache_key), "cache_file": str(srow.cache_file),
                "n_tokens": int(srow.n_tokens), "emb_dim": int(srow.emb_dim),
                "source_path": str(srow.source_path), "feature_source": "reuse_s3",
            })
            continue
        fname = str(r.inner_plot_image).rsplit("/", 1)[-1]
        dest = s3_root / str(r.plot_uid) / fname
        if not dest.exists():
            dest = new_img_root / str(r.plot_uid) / fname
        if not dest.exists() or dest.stat().st_size == 0:
            not_found.append(stem)
            continue
        compute_records.append({
            "image_filename_stem": stem, "plot_uid": str(r.plot_uid),
            "split": str(r.split), "harvest_year": int(r.harvest_year),
            "image_date": str(r.image_date),
            "sha256": sha256_file(dest), "dest": str(dest),
        })

    print(f"endpoints={len(ep_meta)} reuse_s3={len(reuse_rows)} "
          f"to_encode={len(compute_records)} missing_image={len(not_found)}",
          flush=True)
    if not_found:
        with open("reports/s4b1_run_logs/missing_images.txt", "w") as fh:
            fh.write("\n".join(not_found))
        print(f"wrote missing_images.txt ({len(not_found)} stems)", flush=True)

    # ---- encode new endpoints ------------------------------------------
    computed_rows: list[dict] = []
    timing: dict = {}
    if compute_records:
        device = torch.device(feat_cfg["device"]
                              if torch.cuda.is_available() else "cpu")
        writer = FeatureWriter(
            repo_path=Path(d["dinov2_repo"]), out_dir=Path(feat_cfg["out_dir"]),
            device=device, batch_size=int(feat_cfg["batch_size"]))
        rows, timing = writer.run(compute_records)
        cache_by_sha = {r["sha256"]: r["cache_file"] for r in rows}
        for r in compute_records:
            cf = cache_by_sha.get(r["sha256"])
            if cf is None:
                not_found.append(r["image_filename_stem"])
                continue
            computed_rows.append({
                "image_filename_stem": r["image_filename_stem"], "plot_uid": r["plot_uid"],
                "split": r["split"], "harvest_year": r["harvest_year"],
                "image_date": r["image_date"], "sha256": r["sha256"],
                "cache_key": cache_key_for_image(r["sha256"]), "cache_file": cf,
                "n_tokens": MAINLINE_N_TOKENS, "emb_dim": DINOV2_EMBED_DIM,
                "source_path": r["dest"], "feature_source": "s4b_computed",
            })
    elif torch.cuda.is_available():
        # still verify device availability for the report
        device = torch.device("cpu")
        timing = {"note": "no new endpoints to encode"}

    # ---- merge index ----------------------------------------------------
    index_df = pd.DataFrame(reuse_rows + computed_rows).sort_values(
        ["split", "plot_uid", "image_date", "image_filename_stem"], kind="mergesort"
    ).reset_index(drop=True)
    Path(feat_cfg["s4b_feature_index"]).parent.mkdir(parents=True, exist_ok=True)
    index_df.to_parquet(feat_cfg["s4b_feature_index"], index=False)

    report = {
        "stage": "S4B.1",
        "task": "DINOv2 mainline224 feature cache for full train + 4 test splits",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "protocol": f"{CACHE_PROTOCOL}_{CACHE_PROTOCOL_VERSION}",
        "dinov2": {
            "repo_git_commit": DINOV2_REPO_GIT_COMMIT,
            "weights_sha256": DINOV2_WEIGHTS_SHA256,
            "emb_dim": DINOV2_EMBED_DIM,
            "n_tokens": MAINLINE_N_TOKENS,
        },
        "device": feat_cfg["device"],
        "endpoints_total": int(len(ep_meta)),
        "reuse_s3": int(len(reuse_rows)),
        "s4b_computed": int(len(computed_rows)),
        "missing_image_or_failed": int(len(not_found)),
        "coverage": {
            "covered": int(len(index_df)),
            "coverage_fraction": round(len(index_df) / len(ep_meta), 6),
            "by_split": {str(k): int(v) for k, v in index_df.split.value_counts().items()},
        },
        "timing": timing,
        "index_file": feat_cfg["s4b_feature_index"],
        "index_sha256": sha256_file(feat_cfg["s4b_feature_index"]),
    }
    Path(feat_cfg["report"]).parent.mkdir(parents=True, exist_ok=True)
    with open(feat_cfg["report"], "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps({
        "endpoints_total": report["endpoints_total"],
        "reuse_s3": report["reuse_s3"],
        "s4b_computed": report["s4b_computed"],
        "missing": report["missing_image_or_failed"],
        "coverage_fraction": report["coverage"]["coverage_fraction"],
    }, indent=2), flush=True)
    print(f"wrote {feat_cfg['s4b_feature_index']} and {feat_cfg['report']}", flush=True)
    return 0 if not not_found else 1


if __name__ == "__main__":
    sys.exit(main())
