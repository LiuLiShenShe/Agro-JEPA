#!/usr/bin/env python3
"""S1 acceptance verification (read-only, re-derives the key invariants).

Checks:
  A. manifest row counts and dataset totals
  B. official split sizes
  C. every pair is same-plot and Δt matches the (t_a, t_b] window length
  D. image paths belong to the plot and the date in the filename matches
  E. the label dates/lags are internally consistent
  F. the DINOv2 smoke report has the expected token geometry
  G. manifest files match the SHA256 recorded in manifest_spec.json

Exit code 0 = all checks passed.

Usage:
    python3 tools/s1_verify.py --manifests data/manifests \
        --cache-report reports/s1_feature_cache_smoke.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

EXPECTED = {
    "n_plots": 4058,
    "n_observations": 153022,
    "n_pairs": 148964,
    "split_sizes": {"train": 2930, "validation": 380, "test_plot": 250,
                    "test_genotype": 246, "test_environment": 190,
                    "test_genotype_environment": 62},
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", default="data/manifests")
    ap.add_argument("--cache-report", default="reports/s1_feature_cache_smoke.json")
    args = ap.parse_args()
    md = Path(args.manifests)
    failures = []
    checks = []

    def check(name, ok, detail=""):
        checks.append((name, ok, detail))
        if not ok:
            failures.append(f"{name}: {detail}")

    plots = pd.read_csv(md / "fip1_plots.csv", dtype={"plot_uid": str})
    obs = pd.read_csv(md / "fip1_observations.csv.gz", dtype={"plot_uid": str}, low_memory=False)
    pairs = pd.read_parquet(md / "fip1_pairs_consecutive.parquet")

    # A
    check("n_plots", len(plots) == EXPECTED["n_plots"], str(len(plots)))
    check("unique_plot_uid", plots.plot_uid.is_unique, "duplicate plot_uid")
    check("n_observations", len(obs) == EXPECTED["n_observations"], str(len(obs)))
    check("n_inner_images_matches_plots",
          int(obs.groupby("plot_uid").size().sum()) == int(plots.n_images.sum()),
          "sum(obs per plot) != sum(n_images)")
    # B
    for s, n in EXPECTED["split_sizes"].items():
        got = int((plots.split == s).sum())
        check(f"split_size[{s}]", got == n, f"{got} != {n}")
    # C
    check("n_pairs", len(pairs) == EXPECTED["n_pairs"], str(len(pairs)))
    da = pd.to_datetime(pairs.date_a)
    db = pd.to_datetime(pairs.date_b)
    check("pair_dt_matches_dates",
          bool(((db - da).dt.days == pairs.delta_days).all()),
          "delta_days != date_b - date_a")
    check("pair_dt_positive", bool((pairs.delta_days >= 0).all()), "negative dt")
    check("window_matches_dt",
          bool((pairs.window_n_days == pairs.delta_days).all()),
          "window_n_days != delta_days")
    check("window_bounds_match_dates",
          bool(((pairs.window_start_exclusive == pairs.date_a).all()
                and (pairs.window_end_inclusive == pairs.date_b).all())),
          "window bounds != (date_a, date_b]")
    check("pair_manifest_id",
          bool((pairs.manifest_id == "fip1-s1-v1").all()), "missing manifest_id")
    # D: image paths
    fname_a = pairs.image_a.str.rsplit("/", n=1).str[-1]
    fname_b = pairs.image_b.str.rsplit("/", n=1).str[-1]
    check("pair_image_a_plot",
          bool(pd.Series([isinstance(f, str) and f.startswith(str(p))
                          for f, p in zip(fname_a, pairs.plot_uid)]).all()),
          "image_a plot mismatch")
    check("pair_image_b_plot",
          bool(pd.Series([isinstance(f, str) and f.startswith(str(p))
                          for f, p in zip(fname_b, pairs.plot_uid)]).all()),
          "image_b plot mismatch")
    obs_fname = obs.inner_plot_image.str.rsplit("/", n=1).str[-1]
    obs_date = pd.to_datetime(obs.image_date).dt.strftime("%Y%m%d")
    obs_fdate = obs_fname.str.extract(r"_(\d{8})_")[0]
    check("obs_filename_date", bool((obs_fdate == obs_date).all()),
          "filename date != image_date")
    check("obs_filename_plot",
          bool(pd.Series([f.startswith(str(p)) for f, p in
                          zip(obs_fname, obs.plot_uid)]).all()),
          "filename plot prefix mismatch")
    # E: label date/lag consistency
    for tr in ("cc", "ph", "whc", "sen"):
        sub = obs[obs[f"{tr}_date"].notna()].copy()
        if sub.empty:
            continue
        lag = (pd.to_datetime(sub[f"{tr}_date"]) - pd.to_datetime(sub.image_date)).dt.days
        rec = pd.to_numeric(sub[f"{tr}_lag_days"], errors="coerce")
        check(f"{tr}_lag_consistent", bool((lag == rec).all()),
              "stored lag != label_date - image_date")
        check(f"{tr}_value_implies_date",
              bool(sub[f"{tr}_value"].notna().all()),
              "label date present but value missing")
    # F: DINOv2 report
    with open(args.cache_report) as fh:
        cache = json.load(fh)
    for proto, n_tok in (("upstream196", 196), ("mainline224", 256)):
        s = cache["summary"][proto]
        check(f"dino[{proto}]_tokens", s["n_tokens"] == [n_tok], str(s["n_tokens"]))
        check(f"dino[{proto}]_dim",
              all(sh[-1] == 768 for sh in s["token_shapes"]), str(s["token_shapes"]))
        check(f"dino[{proto}]_dtype", s["dtype"] == ["torch.float32"], str(s["dtype"]))
    check("dino_commit", cache["model"]["repo_git_commit"] ==
          "7764ea0f912e53c92e82eb78a2a1631e92725fc8", "commit mismatch")
    check("dino_weights_sha", cache["model"]["weights_sha256"] ==
          "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73",
          "weights sha mismatch")
    # G: hashes
    with open(md / "manifest_spec.json") as fh:
        spec = json.load(fh)
    for name, rec in spec["artifacts"].items():
        p = md / name
        if rec.get("missing"):
            check(f"artifact_present[{name}]", False, "missing")
            continue
        check(f"artifact_sha[{name}]", sha256(p) == rec["sha256"], "sha256 changed")

    for name, ok, detail in checks:
        print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if not ok else ""))
    print(f"\n{sum(1 for _, ok, _ in checks if ok)}/{len(checks)} checks passed")
    if failures:
        print("FAILURES:")
        for f in failures:
            print(" -", f)
        return 1
    print(json.dumps({"status": "ok", "checked_at":
                      datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
