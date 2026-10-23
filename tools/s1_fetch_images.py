#!/usr/bin/env python3
"""S1: download a deterministic subset of real FIP inner-plot images.

Reads data/manifests/fip1_observations.csv (built by tools/s1_audit_fip.py),
selects evenly spaced observations per plot, downloads them through the
HuggingFace mirror, and writes a JSON manifest with a SHA256 per image.

The parquet URLs point at huggingface.co; that host is not reachable from this
machine, so the host is rewritten to $HF_ENDPOINT (default hf-mirror.com).

Usage:
    python3 tools/s1_fetch_images.py \
        --observations data/manifests/fip1_observations.csv \
        --plots FPWW0120001,FPWW... --per-plot 12 \
        --out external_data/fip1_images \
        --manifest data/manifests/fip1_image_sample.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
OLD_HOSTS = ("https://huggingface.co", "http://huggingface.co")


def rewrite(url: str) -> str:
    for h in OLD_HOSTS:
        if url.startswith(h):
            return HF_ENDPOINT + url[len(h):]
    return url


def download(url: str, dest: Path, retries: int = 3) -> tuple[int, str]:
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "agro-jepa-s1/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as fh:
                h = hashlib.sha256()
                n = 0
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    h.update(chunk)
                    n += len(chunk)
            tmp.replace(dest)
            return n, h.hexdigest()
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def pick_even(rows: pd.DataFrame, k: int) -> pd.DataFrame:
    if k <= 0 or len(rows) <= k:
        return rows
    idx = [round(i * (len(rows) - 1) / (k - 1)) for i in range(k)]
    return rows.iloc[sorted(set(idx))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--observations", default="data/manifests/fip1_observations.csv")
    ap.add_argument("--plots", required=True,
                    help="comma-separated plot_uids, or 'auto:<per-year>'")
    ap.add_argument("--per-plot", type=int, default=12, help="0 = all images")
    ap.add_argument("--out", default="external_data/fip1_images")
    ap.add_argument("--manifest", default="data/manifests/fip1_image_sample.json")
    args = ap.parse_args()

    obs = pd.read_csv(args.observations, dtype={"plot_uid": str})
    obs["image_date"] = pd.to_datetime(obs["image_date"])

    if args.plots.startswith("auto:"):
        per_year = int(args.plots.split(":", 1)[1])
        chosen = []
        for y in sorted(obs.harvest_year.unique()):
            sub = obs[obs.harvest_year == y]
            plots = sub.groupby("plot_uid").size().sort_values(ascending=False).index
            chosen += list(plots[:per_year])
    else:
        chosen = [p.strip() for p in args.plots.split(",") if p.strip()]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = []
    for puid in chosen:
        g = obs[obs.plot_uid == puid].sort_values("image_date")
        if g.empty:
            print(f"[warn] {puid} not in observations", flush=True)
            continue
        sel = pick_even(g, args.per_plot)
        for _, r in sel.iterrows():
            if not isinstance(r.inner_plot_image, str):
                continue
            fname = r.inner_plot_image.rsplit("/", 1)[-1]
            dest = out / puid / fname
            tasks.append({
                "plot_uid": puid, "split": r.split, "harvest_year": int(r.harvest_year),
                "image_date": r.image_date.date().isoformat(), "obs_index": int(r.obs_index),
                "url": rewrite(r.inner_plot_image), "dest": dest,
                "inner_plot_image": r.inner_plot_image,
            })

    print(f"selected {len(tasks)} images for {len(chosen)} plots", flush=True)
    results = []
    t0 = time.time()

    def work(t):
        t["dest"].parent.mkdir(parents=True, exist_ok=True)
        if t["dest"].exists():
            data = t["dest"].read_bytes()
            return t, len(data), hashlib.sha256(data).hexdigest(), 0.0, "cached"
        t1 = time.time()
        n, sha = download(t["url"], t["dest"])
        return t, n, sha, time.time() - t1, "downloaded"

    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(work, t) for t in tasks]
        for f in as_completed(futs):
            t, n, sha, secs, status = f.result()
            results.append({
                "plot_uid": t["plot_uid"], "split": t["split"],
                "harvest_year": t["harvest_year"], "image_date": t["image_date"],
                "obs_index": t["obs_index"], "file": str(t["dest"]),
                "bytes": n, "sha256": sha, "seconds": round(secs, 3),
                "status": status, "source_url": t["url"],
            })
            print(f"[{status}] {t['plot_uid']} {t['image_date']} {n/1e6:.2f} MB", flush=True)

    results.sort(key=lambda r: (r["plot_uid"], r["image_date"], r["obs_index"]))
    meta = {
        "hf_endpoint": HF_ENDPOINT,
        "n_images": len(results),
        "n_bytes": sum(r["bytes"] for r in results),
        "elapsed_seconds": round(time.time() - t0, 1),
        "selection": "evenly spaced observations per plot",
        "per_plot": args.per_plot,
        "plots": chosen,
        "images": results,
    }
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {args.manifest}: {len(results)} images, "
          f"{meta['n_bytes']/1e6:.1f} MB, {meta['elapsed_seconds']} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
