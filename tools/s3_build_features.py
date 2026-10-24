#!/usr/bin/env python3
"""S3: build frozen DINOv2 ``mainline224`` patch-token cache for downloaded FIP images.

Protocol is identical to S1 (``tools/s1_dino_cache.py``):

    PIL RGB -> ToTensor -> Resize(224) -> CenterCrop(224)
            -> Normalize(0.5,0.5) -> DINOv2 ViT-B/14 -> x_norm_patchtokens
            -> float16 .npy, keyed by sha256(image) | protocol | version | weights

Outputs
-------
external_data/dino_cache/mainline224_s1-v1/<cache_key>.npy   (256, 768) float16
external_data/s3_features/index.parquet                      stem -> cache file
reports/s3_feature_cache.json                                provenance + timing

Usage
-----
    python3 tools/s3_build_features.py \
        --selection data/manifests/s3_image_selection.json \
        --out external_data/dino_cache \
        --index external_data/s3_features/index.parquet \
        --report reports/s3_feature_cache.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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
    """Frozen DINOv2 encoder + write-through float16 cache."""

    def __init__(
        self,
        repo_path: Path,
        out_dir: Path,
        device: torch.device,
        batch_size: int = 64,
        load_workers: int = 16,
    ) -> None:
        self.repo_path = repo_path
        self.out_dir = out_dir
        self.device = device
        self.batch_size = batch_size
        self.load_workers = load_workers
        self.transform = build_mainline_transform()
        self._check_provenance()

        t0 = time.time()
        base = torch.hub.load(
            str(repo_path.resolve()), "dinov2_vitb14", source="local",
            pretrained=True, verbose=False,
        )
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
            raise RuntimeError(
                f"dinov2 clone is at {commit}, pinned {DINOV2_REPO_GIT_COMMIT}"
            )
        if not WEIGHTS_PATH.exists():
            raise FileNotFoundError(f"pinned weights not found: {WEIGHTS_PATH}")
        got = sha256_file(WEIGHTS_PATH)
        if got != DINOV2_WEIGHTS_SHA256:
            raise RuntimeError(
                f"dinov2 weights sha256 {got} != pinned {DINOV2_WEIGHTS_SHA256}"
            )

    def _load_one(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        return self.transform(img)

    @torch.inference_mode()
    def run(self, records: list[dict]) -> tuple[list[dict], dict]:
        """Encode every unique image sha; returns per-image rows and timing."""
        # one encode per unique source sha
        by_sha: dict[str, str] = {}
        for r in records:
            by_sha.setdefault(r["sha256"], r["dest"])
        shas = sorted(by_sha)
        rows: list[dict] = []

        n_cached = 0
        n_computed = 0
        peak_mem = 0
        t_start = time.time()
        pool = ThreadPoolExecutor(max_workers=self.load_workers)
        try:
            for start in range(0, len(shas), self.batch_size):
                batch_shas = shas[start:start + self.batch_size]
                todo = []
                for sha in batch_shas:
                    key = cache_key_for_image(sha)
                    path = self.out_dir / f"{CACHE_PROTOCOL}_{CACHE_PROTOCOL_VERSION}" / f"{key}.npy"
                    if path.exists():
                        try:
                            arr = np.load(path)
                            if arr.shape == (MAINLINE_N_TOKENS, DINOV2_EMBED_DIM):
                                rows.append({
                                    "sha256": sha, "cache_key": key,
                                    "cache_file": str(path), "status": "cached",
                                })
                                n_cached += 1
                                continue
                        except Exception:  # noqa: BLE001 - rebuild a bad cache file
                            pass
                    todo.append((sha, key, path))

                if not todo:
                    continue
                tensors = list(pool.map(self._load_one, [by_sha[s] for s, _, _ in todo]))
                x = torch.stack(tensors).to(self.device, non_blocking=True)
                feats = self.model.forward_features(x)["x_norm_patchtokens"]
                if feats.shape[1:] != (MAINLINE_N_TOKENS, DINOV2_EMBED_DIM):
                    raise RuntimeError(
                        f"unexpected token shape {tuple(feats.shape)}; expected "
                        f"(B,{MAINLINE_N_TOKENS},{DINOV2_EMBED_DIM})"
                    )
                arrs = feats.to(torch.float16).cpu().numpy()
                for (sha, key, path), arr in zip(todo, arrs):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(path, arr)
                    rows.append({
                        "sha256": sha, "cache_key": key,
                        "cache_file": str(path), "status": "computed",
                    })
                    n_computed += 1
                if self.device.type == "cuda":
                    peak_mem = max(peak_mem, torch.cuda.max_memory_allocated(self.device))
                done = n_cached + n_computed
                if done % (self.batch_size * 10) < self.batch_size:
                    print(
                        f"  encoded {done}/{len(shas)} unique images "
                        f"({n_cached} cached, {n_computed} computed)",
                        flush=True,
                    )
        finally:
            pool.shutdown(wait=True)
        elapsed = time.time() - t_start
        timing = {
            "n_unique_source_images": len(shas),
            "n_cached": n_cached,
            "n_computed": n_computed,
            "elapsed_seconds": round(elapsed, 1),
            "throughput_images_per_s": round(len(shas) / elapsed, 2) if elapsed else None,
            "peak_gpu_memory_bytes": int(peak_mem),
            "batch_size": self.batch_size,
            "model_load_seconds": round(self.load_seconds, 2),
        }
        return rows, timing


def verify_cache_rows(
    records: list[dict], rows: list[dict], writer: FeatureWriter, n: int = 4
) -> dict:
    """Compare a few cache files with a fresh forward pass."""
    by_sha = {r["sha256"]: r for r in records}
    out = []
    path_by_sha = {r["sha256"]: r["cache_file"] for r in rows}
    rng = np.random.RandomState(0)
    shas = sorted(path_by_sha)
    for sha in rng.choice(shas, size=min(n, len(shas)), replace=False):
        img = writer._load_one(by_sha[sha]["dest"])  # noqa: SLF001 - verification
        with torch.inference_mode():
            live = writer.model.forward_features(img[None].to(writer.device))[
                "x_norm_patchtokens"
            ][0].float().cpu().numpy()
        cached = np.load(path_by_sha[sha]).astype(np.float32)
        diff = np.abs(live - cached)
        cos = float(np.sum(live * cached) /
                    (np.linalg.norm(live) * np.linalg.norm(cached) + 1e-12))
        out.append({
            "sha256": sha,
            "shape": list(cached.shape),
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
            "cosine": cos,
        })
    return {"n_checked": len(out), "checks": out}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default="data/manifests/s3_image_selection.json")
    ap.add_argument("--dinov2-repo", default="external_data/third_party/dinov2")
    ap.add_argument("--out", default="external_data/dino_cache")
    ap.add_argument("--index", default="external_data/s3_features/index.parquet")
    ap.add_argument("--report", default="reports/s3_feature_cache.json")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    with open(args.selection) as fh:
        meta = json.load(fh)
    records = [r for r in meta.get("images", []) if r.get("sha256")]
    if args.limit:
        records = records[: args.limit]
    if not records:
        raise SystemExit(f"{args.selection} contains no downloaded images")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    writer = FeatureWriter(
        repo_path=Path(args.dinov2_repo), out_dir=out_dir, device=device,
        batch_size=args.batch_size,
    )
    rows, timing = writer.run(records)

    # map every image stem to its cache file (duplicate stems/stems sharing a sha
    # all resolve to the same physical cache entry)
    cache_by_sha = {r["sha256"]: r["cache_file"] for r in rows}
    index_rows = []
    for r in records:
        cache_file = cache_by_sha.get(r["sha256"])
        if cache_file is None:
            raise RuntimeError(f"no cache entry for {r['image_filename_stem']}")
        index_rows.append({
            "image_filename_stem": r["image_filename_stem"],
            "plot_uid": r["plot_uid"],
            "split": r["split"],
            "harvest_year": r["harvest_year"],
            "image_date": r["image_date"],
            "sha256": r["sha256"],
            "cache_key": cache_key_for_image(r["sha256"]),
            "cache_file": cache_file,
            "n_tokens": MAINLINE_N_TOKENS,
            "emb_dim": DINOV2_EMBED_DIM,
            "source_path": r["dest"],
        })
    index = pd.DataFrame(index_rows).sort_values(
        ["plot_uid", "image_date", "image_filename_stem"], kind="mergesort"
    )
    Path(args.index).parent.mkdir(parents=True, exist_ok=True)
    index.to_parquet(args.index, index=False)

    verification = verify_cache_rows(records, rows, writer)
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "protocol": f"{CACHE_PROTOCOL}_{CACHE_PROTOCOL_VERSION}",
        "selection_manifest": args.selection,
        "selection_manifest_sha256": sha256_file(args.selection),
        "dinov2": {
            "repo": str(Path(args.dinov2_repo).resolve()),
            "repo_git_commit": DINOV2_REPO_GIT_COMMIT,
            "weights_path": str(WEIGHTS_PATH),
            "weights_sha256": DINOV2_WEIGHTS_SHA256,
            "emb_dim": writer.emb_dim,
            "patch_size": writer.patch_size,
            "n_tokens": MAINLINE_N_TOKENS,
        },
        "cache": {
            "dir": str(out_dir / f"{CACHE_PROTOCOL}_{CACHE_PROTOCOL_VERSION}"),
            "dtype": "float16",
            "shape_per_image": [MAINLINE_N_TOKENS, DINOV2_EMBED_DIM],
            "key_rule": "sha256('<image sha256>|mainline224|s1-v1|<weights sha256>')",
        },
        "images": {
            "n_selection_records": len(records),
            "n_unique_stems": int(index.image_filename_stem.nunique()),
            "n_unique_shas": len({r["sha256"] for r in records}),
            "n_plots": int(index.plot_uid.nunique()),
            "n_by_split": {
                str(k): int(v) for k, v in index.split.value_counts().items()
            },
            "n_by_year": {
                str(k): int(v) for k, v in index.harvest_year.value_counts().sort_index().items()
            },
        },
        "timing": timing,
        "verification": verification,
        "index_file": args.index,
        "index_sha256": sha256_file(args.index),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report["timing"], indent=2), flush=True)
    print(json.dumps(report["verification"], indent=2), flush=True)
    print(f"wrote {args.index} and {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
