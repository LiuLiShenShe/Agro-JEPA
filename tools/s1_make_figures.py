#!/usr/bin/env python3
"""S1: real-data figures.

  fig1  multi-temporal images of one real plot + canopy-cover / height timeline
  fig2  per-year label availability (CC / PH / WHC / Sen)
  fig3  six-year daily weather distributions (T_mean, Rs, P, VPD)
  fig4  distribution of consecutive-image intervals (the irregular-time fact)

Usage:
    python3 tools/s1_make_figures.py --fig-dir reports/figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

PLOT_UID = "FPWW0120667"  # 2016, official train split, 39 inner images


def fig_timeline(md: Path, img_manifest: Path, out: Path):
    obs = pd.read_csv(md / "fip1_observations.csv", dtype={"plot_uid": str})
    g = obs[obs.plot_uid == PLOT_UID].sort_values("image_date")
    with open(img_manifest) as fh:
        files = {r["image_date"]: r["file"] for r in json.load(fh)["images"]}

    # choose up to 10 dates that have a downloaded image
    have = [r for _, r in g.iterrows() if r.image_date in files]
    if len(have) > 10:
        idx = np.linspace(0, len(have) - 1, 10).round().astype(int)
        have = [have[i] for i in sorted(set(idx))]

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.5, 1])
    for k, r in enumerate(have[:10]):
        ax = fig.add_subplot(gs[k // 5, k % 5])
        ax.imshow(Image.open(files[r.image_date]).convert("RGB"))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{r.image_date}\nCC={r.cc_value:.2f}", fontsize=8)
    ax = fig.add_subplot(gs[1, :])
    dates = pd.to_datetime(g.image_date)
    ax.plot(dates, pd.to_numeric(g.cc_value, errors="coerce"), "o-",
            ms=3, lw=1.2, label="canopy cover (raw, 0-1)")
    ph = pd.to_numeric(g.ph_value, errors="coerce")
    lag = pd.to_numeric(g.ph_lag_days, errors="coerce")
    ax.plot(dates, ph.where(lag <= 7), "s-", ms=3, lw=1.0,
            label="plant height (m, |lag|<=7 d)")
    for r in have:
        ax.axvline(pd.Timestamp(r.image_date), color="grey", alpha=0.25, lw=0.6)
    ax.set_ylabel("value")
    ax.set_title(f"{PLOT_UID} (2016, official train) — image dates and measured traits")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_label_coverage(md: Path, out: Path):
    plots = pd.read_csv(md / "fip1_plots.csv", dtype={"plot_uid": str})
    years = sorted(plots.harvest_year.unique())
    traits = [("cc_n", "CC"), ("ph_n", "PH"), ("whc_n", "WHC"), ("sen_n", "Sen")]
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(years))
    for i, (col, lab) in enumerate(traits):
        vals = [int(plots.loc[plots.harvest_year == y, col].sum()) for y in years]
        ax.bar(x + (i - 1.5) * width, vals, width, label=lab)
    ax.set_xticks(x); ax.set_xticklabels([str(y) for y in years])
    ax.set_ylabel("number of trait observations")
    ax.set_title("FIP 1.0 label availability by harvest year (all official splits)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_weather(md: Path, out: Path):
    df = pd.read_csv(md / "weather_daily_first_plot_per_year.csv")
    cols = [("T_mean_C", "T mean (degC)"), ("Rs_sum_MJ_m2_d", "Rs (MJ m-2 d-1)"),
            ("P_sum_raw", "P (raw sensor units)"), ("VPD_mean_kPa", "VPD (kPa)")]
    years = sorted(df.harvest_year.unique())
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, (col, lab) in zip(axes.ravel(), cols):
        data = [df.loc[df.harvest_year == y, col].dropna().values for y in years]
        ax.boxplot(data, labels=[str(y) for y in years], showfliers=False)
        ax.set_title(lab)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("FIP daily weather distributions per harvest year "
                 "(first plot of each year; identical station series)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_dt(md: Path, out: Path):
    pairs = pd.read_parquet(md / "fip1_pairs_consecutive.parquet")
    fig, ax = plt.subplots(figsize=(9, 4))
    bins = np.arange(0, pairs.delta_days.max() + 2, 1)
    ax.hist(pairs.delta_days, bins=bins)
    ax.set_xlabel("days between consecutive inner-plot images")
    ax.set_ylabel("number of pairs")
    ax.set_title(f"Irregular acquisition intervals, n={len(pairs):,} consecutive pairs")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", default="data/manifests")
    ap.add_argument("--fig-dir", default="reports/figures")
    args = ap.parse_args()
    md = Path(args.manifests)
    outdir = Path(args.fig_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    made = [
        fig_timeline(md, md / "fip1_image_fullday_2016.json",
                     outdir / "s1_plot_timeline_FPWW0120667.png"),
        fig_label_coverage(md, outdir / "s1_label_coverage_by_year.png"),
        fig_weather(md, outdir / "s1_weather_distributions.png"),
        fig_dt(md, outdir / "s1_dt_distribution.png"),
    ]
    for p in made:
        print(p, p.stat().st_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
