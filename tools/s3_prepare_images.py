#!/usr/bin/env python3
"""S3: select and download a distribution-aware FIP 1.0 image subset.

Selection rule (deterministic, recorded in the output manifest)
---------------------------------------------------------------
* only official ``train`` and ``validation`` plots are eligible;
* **all** validation plots are selected;
* train plots are sampled per harvest year with a fixed seed, so every year
  (2016, 2017, 2018, 2021, 2022) is represented in proportion to a fixed
  number of plots rather than to the raw row count;
* for every selected plot **all** of its audited images are fetched.  A plot
  therefore contributes its whole growth curve (autumn -> winter -> stem
  elongation -> heading -> senescence), which is the only way to obtain
  irregular consecutive intervals and fixed spans from the same plant.

Every image is stored once, under its original ``plot_uid`` / filename, and
recorded with source URL, byte size, SHA256, split, year, date and stem.  Re-runs
skip an existing file only after re-hashing it, so a truncated file cannot be
silently accepted.

Usage
-----
    python3 tools/s3_prepare_images.py \
        --train-plots-per-year 200 --workers 32 \
        --out external_data/fip1_images \
        --selection data/manifests/s3_image_selection.json \
        --report external_data/s3_download.json
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

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
OLD_HOSTS = ("https://huggingface.co", "http://huggingface.co")
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

ELIGIBLE_SPLITS = ("train", "validation")


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


def download_png(url: str, dest: Path, retries: int = 5) -> tuple[int, str]:
    """Download one PNG; returns (bytes, sha256).  Raises on HTTP/format error.

    ``retries`` is deliberately generous: the HF mirror answers ``429 Too Many
    Requests`` under 40-way concurrency, and the backoff below absorbs it.
    """
    tmp = dest.with_suffix(dest.suffix + ".part")
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "agro-jepa-s3/1.0"}
            )
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
        except Exception as e:  # noqa: BLE001 - retry any transient failure
            last = e
            if tmp.exists():
                tmp.unlink()
            if attempt < retries - 1:
                time.sleep(3.0 * (attempt + 1))
    raise RuntimeError(f"failed after {retries} attempts: {url}: {last}")


def select_plots(obs: pd.DataFrame, per_year: int, seed: int) -> tuple[list[str], dict]:
    """All validation plots + ``per_year`` random train plots per year."""
    eligible = obs[obs.split.isin(ELIGIBLE_SPLITS)]
    val_plots = sorted(eligible.loc[eligible.split == "validation", "plot_uid"].unique())
    rng = np.random.RandomState(seed)
    train_plots: list[str] = []
    per_year_counts: dict[str, int] = {}
    for year in sorted(eligible.harvest_year.unique()):
        sub = eligible[(eligible.split == "train") & (eligible.harvest_year == year)]
        plots = np.array(sorted(sub.plot_uid.unique()))
        k = min(int(per_year), len(plots))
        chosen = rng.choice(plots, size=k, replace=False)
        train_plots.extend(sorted(str(p) for p in chosen))
        per_year_counts[str(int(year))] = k
    meta = {
        "rule": "all validation plots; per-year random train plots; all images per plot",
        "seed": int(seed),
        "train_plots_per_year_requested": int(per_year),
        "train_plots_per_year_selected": per_year_counts,
        "n_train_plots": len(train_plots),
        "n_validation_plots": len(val_plots),
    }
    return sorted(train_plots + val_plots), meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--observations", default="data/manifests/fip1_observations.csv.gz")
    ap.add_argument("--train-plots-per-year", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", default="external_data/fip1_images")
    ap.add_argument("--selection", default="data/manifests/s3_image_selection.json")
    ap.add_argument("--report", default="external_data/s3_download.json")
    ap.add_argument("--limit", type=int, default=0, help="debug: cap task count")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    obs = pd.read_csv(
        args.observations, dtype={"plot_uid": str, "split": str}, low_memory=False
    )
    obs["image_date"] = pd.to_datetime(obs["image_date"], errors="raise")
    # One row per physical image; S1 re-lists 1,872 image references.
    obs = obs.drop_duplicates("image_filename_stem").copy()
    obs = obs[obs.split.isin(ELIGIBLE_SPLITS)].copy()

    chosen, selection_meta = select_plots(obs, args.train_plots_per_year, args.seed)
    selected = obs[obs.plot_uid.isin(chosen)].copy()
    selected = selected.sort_values(
        ["plot_uid", "image_date", "image_time"], kind="mergesort"
    )
    if args.limit:
        selected = selected.head(args.limit)

    out_root = Path(args.out)
    tasks = []
    for r in selected.itertuples(index=False):
        fname = str(r.inner_plot_image).rsplit("/", 1)[-1]
        tasks.append({
            "plot_uid": str(r.plot_uid),
            "split": str(r.split),
            "harvest_year": int(r.harvest_year),
            "image_date": r.image_date.date().isoformat(),
            "obs_index": int(r.obs_index),
            "image_filename_stem": str(r.image_filename_stem),
            "inner_plot_image": str(r.inner_plot_image),
            "dest": out_root / str(r.plot_uid) / fname,
        })

    n_existing = sum(1 for t in tasks if t["dest"].exists())
    print(
        f"selected {len(chosen)} plots / {len(tasks)} images "
        f"({n_existing} already on disk); {selection_meta}",
        flush=True,
    )
    if args.dry_run:
        return 0

    out_root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    records: list[dict] = []

    def work(t: dict) -> dict:
        dest: Path = t["dest"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size > 0:
            return {
                **t, "dest": str(dest), "bytes": dest.stat().st_size,
                "sha256": sha256_file(dest), "status": "cached",
                "seconds": 0.0, "source_url": rewrite(t["inner_plot_image"]),
            }
        t1 = time.time()
        n, sha = download_png(rewrite(t["inner_plot_image"]), dest)
        return {
            **t, "dest": str(dest), "bytes": n, "sha256": sha,
            "status": "downloaded", "seconds": round(time.time() - t1, 3),
            "source_url": rewrite(t["inner_plot_image"]),
        }

    failures: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, t): t for t in tasks}
        done = 0
        for f in as_completed(futs):
            t = futs[f]
            done += 1
            try:
                rec = f.result()
                records.append(rec)
                if rec["status"] == "downloaded":
                    print(
                        f"[{done}/{len(tasks)}] {rec['plot_uid']} "
                        f"{rec['image_date']} {rec['bytes']/1e6:.2f} MB",
                        flush=True,
                    )
            except Exception as e:  # noqa: BLE001
                failures.append({
                    "plot_uid": t["plot_uid"],
                    "image_filename_stem": t["image_filename_stem"],
                    "source_url": rewrite(t["inner_plot_image"]),
                    "error": str(e),
                })
                print(f"[FAIL] {t['plot_uid']} {t['image_filename_stem']}: {e}", flush=True)

    records.sort(key=lambda r: (r["plot_uid"], r["image_date"], r["obs_index"]))
    failures.sort(key=lambda r: (r["plot_uid"], r["image_filename_stem"]))
    elapsed = time.time() - t0
    n_bytes = sum(r["bytes"] for r in records)
    meta = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hf_endpoint": HF_ENDPOINT,
        "source_manifest": args.observations,
        "selection": selection_meta,
        "n_plots": len({r["plot_uid"] for r in records}),
        "n_images": len(records),
        "n_cached": sum(r["status"] == "cached" for r in records),
        "n_downloaded": sum(r["status"] == "downloaded" for r in records),
        "n_failed": len(failures),
        "n_bytes": n_bytes,
        "elapsed_seconds": round(elapsed, 1),
        "throughput_images_per_s": round(len(records) / elapsed, 2) if elapsed else None,
        "n_images_note": (
            "If n_failed > 0 this is a partial subset; rerun to retry only the "
            "missing files (existing files are re-hashed, never re-downloaded)."
        ),
        "images": records,
        "failures": failures,
    }

    Path(args.selection).parent.mkdir(parents=True, exist_ok=True)
    with open(args.selection, "w") as fh:
        json.dump(meta, fh, indent=2)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w") as fh:
        json.dump(meta, fh, indent=2)

    print(
        f"wrote {args.selection}: {meta['n_images']} images, "
        f"{meta['n_plots']} plots, {n_bytes/1e9:.2f} GB, "
        f"{elapsed:.1f} s, {meta['throughput_images_per_s']} img/s, "
        f"{len(failures)} failures",
        flush=True,
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
