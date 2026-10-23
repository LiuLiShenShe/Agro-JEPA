#!/usr/bin/env python3
"""S1: validate the frozen DINOv2 feature-cache protocol on real FIP images.

Two protocols are compared on the *same* real images:

  upstream196  upstream DINO-WM path
               ToTensor -> Resize(224) -> CenterCrop(224) -> Normalize(0.5)
               -> VWorldModel.Resize(196) -> DINOv2 ViT-B/14
               tokens: (1, 196, 768)   [14x14 patches]

  mainline224  Agro-JEPA mainline candidate
               ToTensor -> Resize(224) -> CenterCrop(224) -> Normalize(0.5)
               -> DINOv2 ViT-B/14
               tokens: (1, 256, 768)   [16x16 patches]

The two caches are written to separate directories so they can never be mixed.
Each cache file is keyed by the SHA256 of the source image bytes, the
preprocessing protocol id, and the weights SHA256.

Usage:
    python3 tools/s1_dino_cache.py \
        --images data/manifests/fip1_image_sample.json \
        --dinov2-repo external_data/third_party/dinov2 \
        --out external_data/dino_cache \
        --report reports/s1_feature_cache_smoke.json \
        --limit 12
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

PROTOCOLS = {
    "upstream196": 196,
    "mainline224": 224,
}
PROTOCOL_VERSION = "s1-v1"
WEIGHTS_URL = ("https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/"
               "dinov2_vitb14_pretrain.pth")


def sha256_file(p: str | Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def state_dict_digest(sd) -> str:
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def build_transform() -> T.Compose:
    return T.Compose([
        T.ToTensor(),                                   # uint8 HWC -> float CHW /255
        T.Resize(224),                                  # upstream default_transform(224)
        T.CenterCrop(224),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/manifests/fip1_image_sample.json")
    ap.add_argument("--dinov2-repo", default="external_data/third_party/dinov2")
    ap.add_argument("--out", default="external_data/dino_cache")
    ap.add_argument("--report", default="reports/s1_feature_cache_smoke.json")
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- provenance -------------------------------------------------------
    repo = Path(args.dinov2_repo).resolve()
    if not (repo / "hubconf.py").exists():
        raise SystemExit(f"{repo} is not the dinov2 repo (no hubconf.py)")
    try:
        import subprocess
        commit = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception as e:  # pragma: no cover
        commit = f"unknown ({e})"

    t0 = time.time()
    model = torch.hub.load(str(repo), "dinov2_vitb14", source="local",
                           pretrained=True, verbose=False)
    load_seconds = time.time() - t0
    model.eval().to(device)

    weights_path = Path.home() / ".cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth"
    weights_sha = sha256_file(weights_path) if weights_path.exists() else None
    cfg = {
        "name": "dinov2_vitb14",
        "num_features": int(model.num_features),
        "embed_dim": int(model.embed_dim),
        "patch_size": int(model.patch_size),
        "n_blocks": int(len(model.blocks)) if hasattr(model, "blocks") else None,
        "num_heads": int(getattr(model, "num_heads", -1)),
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "loading": "torch.hub.load(repo, 'dinov2_vitb14', source='local', pretrained=True)",
        "repo_dir": str(repo),
        "repo_git_commit": commit,
        "weights_file": str(weights_path),
        "weights_sha256": weights_sha,
        "weights_url": WEIGHTS_URL,
    }
    cfg["weights_state_dict_sha256"] = state_dict_digest(model.state_dict())

    transform = build_transform()

    # ---- images -----------------------------------------------------------
    with open(args.images) as fh:
        imgs = json.load(fh)["images"]
    imgs = imgs[: args.limit]

    results = []
    timings = {k: [] for k in PROTOCOLS}
    e2e = {k: [] for k in PROTOCOLS}
    with torch.no_grad():
        # untimed warm-up per protocol (cuDNN autotune, allocator)
        warm = Image.open(imgs[0]["file"]).convert("RGB")
        for proto, size in PROTOCOLS.items():
            x = transform(warm)
            if size != 224:
                x = T.Resize(size)(x)
            _ = model.forward_features(x.unsqueeze(0).to(device))
            if device.type == "cuda":
                torch.cuda.synchronize()

        for proto, size in PROTOCOLS.items():
            proto_dir = out / f"{proto}_{PROTOCOL_VERSION}"
            proto_dir.mkdir(parents=True, exist_ok=True)
            for rec in imgs:
                t_start = time.perf_counter()
                src = rec["file"]
                img = Image.open(src).convert("RGB")
                x = transform(img)                              # (3,224,224) float32
                if size != 224:
                    x = T.Resize(size)(x)                      # (3,196,196)
                x = x.unsqueeze(0).to(device)                  # (1,3,H,W)
                t1 = time.perf_counter()
                feats = model.forward_features(x)
                torch.cuda.synchronize() if device.type == "cuda" else None
                secs = time.perf_counter() - t1
                tokens = feats["x_norm_patchtokens"]           # (1,N,D)
                timings[proto].append(secs)
                key = hashlib.sha256(
                    f"{rec['sha256']}|{proto}|{PROTOCOL_VERSION}|{weights_sha}".encode()
                ).hexdigest()
                arr = tokens[0].to(torch.float16).cpu().numpy()
                cache_path = proto_dir / f"{key}.npy"
                np.save(cache_path, arr)
                e2e_secs = time.perf_counter() - t_start
                e2e[proto].append(e2e_secs)
                results.append({
                    "plot_uid": rec["plot_uid"],
                    "image_date": rec["image_date"],
                    "source_file": src,
                    "source_sha256": rec["sha256"],
                    "protocol": proto,
                    "protocol_version": PROTOCOL_VERSION,
                    "input_shape": list(x.shape[1:]),
                    "token_shape": list(tokens.shape),
                    "n_tokens": int(tokens.shape[1]),
                    "output_dtype": str(tokens.dtype),
                    "cache_dtype": "float16",
                    "cache_file": str(cache_path),
                    "cache_bytes": int(cache_path.stat().st_size),
                    "forward_seconds": round(secs, 5),
                    "e2e_seconds": round(e2e_secs, 5),
                    "cache_key": key,
                })

    # ---- protocol summary -------------------------------------------------
    summary = {}
    for proto in PROTOCOLS:
        rows = [r for r in results if r["protocol"] == proto]
        shapes = sorted({tuple(r["token_shape"]) for r in rows})
        n_img = len(rows)
        bytes_per_img = float(np.mean([r["cache_bytes"] for r in rows])) if rows else 0.0
        e2e_mean = float(np.mean(e2e[proto])) if e2e[proto] else 0.0
        summary[proto] = {
            "n_images": n_img,
            "token_shapes": [list(s) for s in shapes],
            "n_tokens": sorted({r["n_tokens"] for r in rows}),
            "dtype": sorted({r["output_dtype"] for r in rows}),
            "cache_dtype": "float16",
            "mean_forward_seconds": round(float(np.mean(timings[proto])), 5),
            "median_forward_seconds": round(float(np.median(timings[proto])), 5),
            "mean_e2e_seconds": round(e2e_mean, 5),
            "throughput_images_per_s": round(1.0 / e2e_mean, 2) if e2e_mean else None,
            "cache_bytes_total": int(sum(r["cache_bytes"] for r in rows)),
            "cache_bytes_per_image_mean": round(bytes_per_img, 1),
            "projected_gib_for_153k_images": round(
                bytes_per_img * 153_022 / (1024 ** 3), 1),
        }

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": str(device),
        "model": cfg,
        "model_load_seconds": round(load_seconds, 2),
        "preprocessing": {
            "upstream196": [
                "PIL RGB", "ToTensor (uint8/255, CHW)",
                "Resize(224)", "CenterCrop(224)",
                "Normalize(mean=0.5,std=0.5) per channel",
                "Resize(196)", "DINOv2 ViT-B/14",
            ],
            "mainline224": [
                "PIL RGB", "ToTensor (uint8/255, CHW)",
                "Resize(224)", "CenterCrop(224)",
                "Normalize(mean=0.5,std=0.5) per channel",
                "DINOv2 ViT-B/14",
            ],
            "note": "DINOv2 does not normalize internally; 0.5 normalization is the only one.",
        },
        "summary": summary,
        "images": results,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps({"model": {k: cfg[k] for k in
                                ("name", "embed_dim", "patch_size", "n_blocks",
                                 "n_params", "repo_git_commit", "weights_sha256")},
                      "summary": summary}, indent=2))
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
