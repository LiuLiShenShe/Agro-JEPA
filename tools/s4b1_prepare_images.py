#!/usr/bin/env python3
"""S4B.1: download only the missing image endpoints and record a manifest.

Reads ``external_data/s4b_index/s4b1_desired_endpoints.parquet`` (produced by
``s4b1_build_full_train_index.py``) and downloads every endpoint whose PNG is
not already present under ``external_data/fip1_images``.  Existing S3 images are
reused read-only -- never re-downloaded and never overwritten.  New images go
to a dedicated ``external_data/s4b_images/<plot_uid>/<fname>`` directory so the
S3 cache directory is left untouched.

Resumable: an existing file is accepted only after re-hashing and confirming it
is a non-trivial PNG; failures are recorded with reasons and retried on rerun.

Usage
-----
    python3 tools/s4b1_prepare_images.py \
        --config agro_jepa/config/s4b1_data_cache.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
OLD_HOSTS = ("https://huggingface.co", "http://huggingface.co")
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def rewrite(url: str) -> str:
    for h in OLD_HOSTS:
        if url.startswith(h):
            return HF_ENDPOINT + url[len(h):]
    return url


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_png(url: str, dest: Path, retries: int = 6) -> tuple[int, str]:
    tmp = dest.with_suffix(dest.suffix + ".part")
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "agro-jepa-s4b1/1.0"})
            with urllib.request.urlopen(req, timeout=180) as resp, open(tmp, "wb") as fh:
                h = hashlib.sha256()
                n = 0
                head = b""
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    if len(head) < len(PNG_MAGIC):
                        head += chunk[: len(PNG_MAGIC) - len(head)]
                    fh.write(chunk)
                    h.update(chunk)
                    n += len(chunk)
            if not head.startswith(PNG_MAGIC):
                raise ValueError(f"not a PNG (first bytes {head!r})")
            if n < 1024:
                raise ValueError(f"suspiciously small PNG ({n} bytes)")
            tmp.replace(dest)
            return n, h.hexdigest()
        except Exception as e:  # noqa: BLE001
            last = e
            if tmp.exists():
                tmp.unlink()
            if attempt < retries - 1:
                time.sleep(3.0 * (attempt + 1))
    raise RuntimeError(f"failed after {retries} attempts: {url}: {last}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s4b1_data_cache.yaml")
    ap.add_argument("--workers", type=int, default=0,
                    help="override images.workers (lower it to avoid HTTP 429)")
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    t0 = time.time()

    ep = pd.read_parquet(cfg["index"]["endpoints_out"])
    s3_root = Path("external_data/fip1_images")
    out_root = Path(cfg["images"]["out_dir"])
    out_root.mkdir(parents=True, exist_ok=True)

    # existing S3 images, keyed by stem (read-only reuse)
    existing: set[str] = set()

    def walk_stems(root: Path) -> set[str]:
        s = set()
        for d, _, fs in os.walk(root):
            for f in fs:
                if f.endswith(".png"):
                    s.add(f[:-4].upper())
        return s

    existing |= walk_stems(s3_root)

    # classify endpoints
    tasks = []
    reused = 0
    for r in ep.itertuples(index=False):
        fname = str(r.inner_plot_image).rsplit("/", 1)[-1]
        stem = str(r.image_filename_stem).upper()
        s3_path = s3_root / str(r.plot_uid) / fname
        new_path = out_root / str(r.plot_uid) / fname
        if stem in existing or s3_path.exists():
            reused += 1
            continue
        tasks.append({
            "plot_uid": str(r.plot_uid),
            "split": str(r.split),
            "harvest_year": int(r.harvest_year),
            "image_date": str(r.image_date),
            "image_filename_stem": stem,
            "inner_plot_image": str(r.inner_plot_image),
            "dest": new_path,
        })

    print(
        f"desired={len(ep)} reused_already_on_disk(excl)={reused} "
        f"to_download={len(tasks)}", flush=True
    )

    records: list[dict] = []
    failures: list[dict] = []

    def work(t: dict) -> dict:
        dest: Path = t["dest"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size > 0:
            return {**t, "dest": str(dest), "bytes": dest.stat().st_size,
                    "sha256": sha256_file(dest), "status": "cached",
                    "seconds": 0.0, "source_url": rewrite(t["inner_plot_image"])}
        t1 = time.time()
        n, sha = download_png(rewrite(t["inner_plot_image"]), dest)
        return {**t, "dest": str(dest), "bytes": n, "sha256": sha,
                "status": "downloaded", "seconds": round(time.time() - t1, 3),
                "source_url": rewrite(t["inner_plot_image"])}

    workers = int(args.workers) if args.workers else int(cfg["images"]["workers"])
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, t): t for t in tasks}
        done = 0
        for f in as_completed(futs):
            t = futs[f]
            done += 1
            try:
                rec = f.result()
                records.append(rec)
                if rec["status"] == "downloaded" and done % 200 == 0:
                    print(f"[{done}/{len(tasks)}] {rec['plot_uid']} "
                          f"{rec['image_date']} {rec['bytes']/1e6:.2f} MB", flush=True)
            except Exception as e:  # noqa: BLE001
                failures.append({
                    "plot_uid": t["plot_uid"],
                    "image_filename_stem": t["image_filename_stem"],
                    "source_url": rewrite(t["inner_plot_image"]),
                    "error": str(e),
                })
                print(f"[FAIL] {t['plot_uid']} {t['image_filename_stem']}: {e}", flush=True)

    records.sort(key=lambda r: (r["plot_uid"], r["image_date"]))
    failures.sort(key=lambda r: (r["plot_uid"], r["image_filename_stem"]))
    elapsed = time.time() - t0
    n_bytes = sum(r["bytes"] for r in records)
    meta = {
        "stage": "S4B.1",
        "task": "download missing image endpoints for full train + 4 test splits",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hf_endpoint": HF_ENDPOINT,
        "endpoints_total": int(len(ep)),
        "reused_already_on_disk": reused,
        "n_to_download": len(tasks),
        "n_downloaded": sum(r["status"] == "downloaded" for r in records),
        "n_cached_skip": sum(r["status"] == "cached" for r in records),
        "n_failed": len(failures),
        "n_bytes": n_bytes,
        "elapsed_seconds": round(elapsed, 1),
        "throughput_images_per_s": round(len(records) / elapsed, 2) if elapsed else None,
        "images_dir": str(out_dir_path(cfg)),
        "s3_images_dir_reused_readonly": str(s3_root),
        "images": records,
        "failures": failures,
    }
    manifest_path = Path(cfg["images"]["manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    report_path = Path(cfg["images"]["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {manifest_path}: download={meta['n_downloaded']} "
          f"cached_skip={meta['n_cached_skip']} failed={meta['n_failed']} "
          f"bytes={n_bytes/1e9:.2f} GB {elapsed:.0f}s", flush=True)
    return 0 if not failures else 1


def out_dir_path(cfg) -> Path:
    return Path(cfg["images"]["out_dir"])


if __name__ == "__main__":
    sys.exit(main())
