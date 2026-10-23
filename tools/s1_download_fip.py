#!/usr/bin/env python3
"""Download the FIP 1.0 tabular parquet shards (HuggingFace: mikeboss/FIP1).

Only the parquet shards are fetched here; the ~153k inner-plot images are
downloaded on demand by tools/s1_fetch_images.py.

The file list (name, byte size, LFS sha256) was read from the dataset's
HuggingFace API tree on 2026-10-23 and is pinned below so the download is
verifiable and reproducible.

Usage:
    python3 tools/s1_download_fip.py --out external_data/fip1_parquet
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
REPO = "mikeboss/FIP1"
REVISION = "11cdd84bc0ba8d5c6641ea7b53550b1658527e1c"

# (path in repo, size, lfs sha256) -- from /api/datasets/mikeboss/FIP1/tree/main/data
FILES = [
    ("data/test_environment-00000-of-00001.parquet", 27529156,
     "0fcdcf36f6d7034aacc58d15857531078ae0255ea825bcf98554484f9ef15349"),
    ("data/test_genotype-00000-of-00001.parquet", 151265501,
     "25b6da2b2d099c7e511bf2740d3d5d2baefe8647b852eefc4a572daa3c169581"),
    ("data/test_genotype_environment-00000-of-00001.parquet", 38305429,
     "e598f0c3cf34592328b18f9ae55dd903c265126d26477f781918f4e71765daef"),
    ("data/test_plot-00000-of-00001.parquet", 153525540,
     "7dfe6570639862eef12c571e2c2a131353048881aa796b6ec0e24ed7608bd5e1"),
    ("data/train-00000-of-00007.parquet", 207898052,
     "2f1a71cd29f30f30418998fe0a0fc06e32180c2445b99db04ee687bb40706eb3"),
    ("data/train-00001-of-00007.parquet", 189211287,
     "a15866e0b9310e0563c8a4c31ea3e91c93fe67e2a41872d8ac3e6ddb0a44b30a"),
    ("data/train-00002-of-00007.parquet", 182674719,
     "67592a119846c97e8284ab0d9cf6a1769cc80cf8aa467196110caae822dd5f00"),
    ("data/train-00003-of-00007.parquet", 114834185,
     "0df0a6337de276482f12fa4f2685e9a03241f610beacb117aecd30a8dc3695cc"),
    ("data/train-00004-of-00007.parquet", 10339036,
     "10cab37b122abc29c31b2cea302fdfd445d69620df21cb7a1d16beb9cdebc67e"),
    ("data/train-00005-of-00007.parquet", 36758958,
     "175b7aaddb036d2ae6b6a761c78c117458425c28435fbf6284436d140d5a8a99"),
    ("data/train-00006-of-00007.parquet", 30366420,
     "04af42894cc71340b970ecc0d78b8e509681d2c79599bea4a6fc852030e086a3"),
    ("data/validation-00000-of-00002.parquet", 116556393,
     "af703adec24f48e1427cec805ceca7662a641720449441d796ba20f3c8205cd5"),
    ("data/validation-00001-of-00002.parquet", 116403834,
     "69369f8f6e65f67d8fd689defce25912640fc2eb17e95783c592d3df91eab2ef"),
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, expected_size: int) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    have = tmp.stat().st_size if tmp.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    req = urllib.request.Request(url, headers=headers)
    mode = "ab" if have else "wb"
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, mode) as fh:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    if tmp.stat().st_size != expected_size:
        raise RuntimeError(f"{dest.name}: size {tmp.stat().st_size} != {expected_size}")
    tmp.replace(dest)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="external_data/fip1_parquet")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = []
    t0 = time.time()
    for repo_path, size, sha in FILES:
        name = repo_path.split("/")[-1]
        dest = out / name
        if dest.exists() and dest.stat().st_size == size and sha256_of(dest) == sha:
            print(f"[skip] {name} (sha256 ok)", flush=True)
            report.append({"file": name, "status": "skip", "size": size, "sha256": sha})
            continue
        if dest.exists() and sha256_of(dest) != sha:
            dest.unlink()
        url = f"{HF_ENDPOINT}/datasets/{REPO}/resolve/{REVISION}/{repo_path}"
        print(f"[get ] {name} ({size/1e6:.1f} MB)", flush=True)
        t1 = time.time()
        download(url, dest, size)
        got = sha256_of(dest)
        if got != sha:
            raise RuntimeError(f"{name}: sha256 {got} != {sha}")
        print(f"[ ok ] {name} {time.time()-t1:.1f}s sha256={got}", flush=True)
        report.append({"file": name, "status": "downloaded", "size": size,
                       "sha256": got, "seconds": round(time.time() - t1, 1)})
    meta = {
        "dataset": REPO,
        "revision": REVISION,
        "hf_endpoint": HF_ENDPOINT,
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "total_seconds": round(time.time() - t0, 1),
        "files": report,
    }
    with open(out / "download_manifest.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(json.dumps(meta, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
