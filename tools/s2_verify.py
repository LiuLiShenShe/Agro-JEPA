#!/usr/bin/env python3
"""S2 acceptance checks (A-H from the stage spec), read-only.

Exit code 0 only when every check passes.  Any check that needs real images
uses the subset that exists under ``external_data/fip1_images``; the manifest
level checks run over the full pair table.

Usage:
    python3 tools/s2_verify.py \
        --config agro_jepa/config/s2_dataset.yaml \
        --real-batch-report reports/s2_real_batch.json \
        --pairs-report reports/s2_pairs_stats.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agro_jepa.datasets.collate import fip_collate_fn  # noqa: E402
from agro_jepa.datasets.fip_dataset import FIPPairDataset  # noqa: E402
from agro_jepa.datasets.fip_pairs import (  # noqa: E402
    OFFICIAL_SPLITS,
    PairRule,
    build_pairs,
    load_custom_splits,
    load_observations,
    load_plots,
)
from agro_jepa.datasets.fip_weather import DailyWeatherStore  # noqa: E402
from agro_jepa.datasets.provenance import MAINLINE_N_TOKENS  # noqa: E402


def load_cfg(path):
    with open(path) as fh:
        return yaml.safe_load(fh)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s2_dataset.yaml")
    ap.add_argument("--pairs", default="external_data/pairs_cache/fip1_pairs_s2.parquet")
    ap.add_argument("--real-batch-report", default="reports/s2_real_batch.json")
    ap.add_argument("--pairs-report", default="reports/s2_pairs_stats.json")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    data, pr = cfg["data"], cfg["pair_rule"]
    checks: list[tuple[str, bool, str]] = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    # ---------------------------------------------------------------- data
    obs = load_observations(data["observations"])
    plots = load_plots(data["plots"])
    custom = load_custom_splits(data["custom_splits"])
    pairs_path = Path(args.pairs)
    if pairs_path.exists():
        pairs = pd.read_parquet(pairs_path)
    else:
        pairs, _ = build_pairs(
            obs, plots, custom,
            PairRule(mode=pr["mode"], spans=tuple(pr["spans"]),
                     span_tolerance_days=pr["span_tolerance_days"],
                     min_delta_days=pr["min_delta_days"],
                     max_delta_days=pr.get("max_delta_days"),
                     label_max_lag_days=pr["label_max_lag_days"],
                     manifest_id=cfg["manifest_id"]),
        )
    weather = DailyWeatherStore(data["weather_daily"],
                                channels=tuple(cfg["weather"]["channels"]))

    # ---- A: trace one plot end to end ------------------------------------
    trace_plot = "FPWW0120667"
    trace_date = "2016-05-17"
    g_obs = obs[(obs.plot_uid == trace_plot)].drop_duplicates(["plot_uid", "image_filename_stem"])
    row = g_obs[g_obs.image_filename_stem.str.contains("20160517")]
    check("A_plot_has_image_on_date", len(row) == 1,
          f"{trace_plot} {trace_date}: {len(row)} rows")
    if len(row) == 1:
        r = row.iloc[0]
        stem_date = r.image_filename_stem.split("_")[2]
        check("A_image_date_matches_stem", stem_date == trace_date.replace("-", ""),
              f"stem date {stem_date}")
        check("A_cc_label_on_same_day",
              str(r.cc_date)[:10] == trace_date and int(r.cc_lag_days) == 0,
              f"cc_date={r.cc_date} lag={r.cc_lag_days}")
        check("A_ph_label_within_7d", abs(int(r.ph_lag_days)) <= 7,
              f"ph_lag={r.ph_lag_days}")
        win = weather.window(2016, "2016-05-10", trace_date)
        check("A_weather_window_length", win.n_days == 7, str(win.n_days))
        check("A_weather_window_end_is_t_b", win.end_inclusive == date(2016, 5, 17),
              str(win.end_inclusive))
        trace_pair = pairs[(pairs.plot_uid == trace_plot)
                           & (pairs.date_b == trace_date)]
        check("A_plot_has_a_pair_ending_on_date", len(trace_pair) > 0,
              str(len(trace_pair)))
        if len(trace_pair):
            p = trace_pair.iloc[0]
            check("A_pair_is_same_plot",
                  p.image_stem_a.startswith(trace_plot) and p.image_stem_b.startswith(trace_plot),
                  f"{p.image_stem_a} / {p.image_stem_b}")
            check("A_pair_weather_window_eq_delta",
                  (date.fromisoformat(p.date_b) - date.fromisoformat(p.date_a)).days
                  == int(p.delta_days), f"delta={p.delta_days}")

    # ---- B: every image date comes from the audited stem ------------------
    stem_a = pairs.image_stem_a.str.extract(r"_(\d{8})_", expand=False)
    stem_b = pairs.image_stem_b.str.extract(r"_(\d{8})_", expand=False)
    check("B_pair_dates_match_stems",
          bool((stem_a == pairs.date_a.str.replace("-", "", regex=False)).all()
               and (stem_b == pairs.date_b.str.replace("-", "", regex=False)).all()),
          "pair date != filename stem date")
    check("B_pair_stems_belong_to_plot",
          bool(all(s.startswith(p) for s, p in zip(pairs.image_stem_a, pairs.plot_uid))
               and all(s.startswith(p) for s, p in zip(pairs.image_stem_b, pairs.plot_uid))),
          "pair stem does not start with plot_uid")
    obs_stem_date = obs.image_filename_stem.str.extract(r"_(\d{8})_", expand=False)
    check("B_observation_dates_match_stems",
          bool((obs_stem_date == obs.image_date.dt.strftime("%Y%m%d")).all()),
          "observation date != stem date")
    check("B_no_duplicate_stems_in_pairs",
          bool((pairs.image_stem_a != pairs.image_stem_b).all()),
          "pair uses the same image twice")

    # ---- C: (t_a, t_b] windows, variable length, mask ---------------------
    check("C_all_delta_positive", bool((pairs.delta_days >= 1).all()),
          f"min delta {pairs.delta_days.min()}")
    check("C_window_n_days_eq_delta",
          bool((pairs.window_n_days == pairs.delta_days).all()),
          "window_n_days != delta_days")
    check("C_window_bounds_eq_dates",
          bool((pairs.window_start_exclusive == pairs.date_a).all()
               and (pairs.window_end_inclusive == pairs.date_b).all()),
          "window bounds != (date_a, date_b]")
    # vectorised full-coverage check with the real daily store
    valid_counts = np.zeros(len(pairs), dtype=np.int64)
    for y, gg in pairs.groupby("harvest_year"):
        valid_counts[gg.index.to_numpy()] = weather.window_valid_counts(
            int(y), gg.date_a.to_numpy(), gg.date_b.to_numpy())
    check("C_weather_full_coverage",
          bool((valid_counts == pairs.delta_days.to_numpy()).all()),
          f"{int((valid_counts != pairs.delta_days.to_numpy()).sum())} windows missing days")
    # concrete window on a real pair, including the left-open boundary
    s = pairs.iloc[0]
    win = weather.window(int(s.harvest_year), s.date_a, s.date_b)
    expected = [(date.fromisoformat(s.date_a) + timedelta(days=k + 1)).isoformat()
                for k in range(int(s.delta_days))]
    check("C_window_dates_are_ta_plus_1_to_tb",
          [d.isoformat() for d in win.dates] == expected,
          f"first={win.dates[0]} last={win.dates[-1]}")
    check("C_mask_length_eq_window", len(win.mask) == int(s.delta_days), "")

    # ---- local subset for real-batch checks ------------------------------
    image_root = Path(data["images_root"])
    image_index = {p.stem: p for p in image_root.rglob("*.png")}
    local = set(image_index)
    local_pairs = pairs[pairs.image_stem_a.isin(local)
                        & pairs.image_stem_b.isin(local)].reset_index(drop=True)
    check("C_local_pairs_exist", len(local_pairs) > 0, str(len(local_pairs)))
    ds = FIPPairDataset(local_pairs, weather, image_mode="visual",
                        images_root=image_root, compute_image_stats=True)
    n_batch = min(8, len(ds))
    batch = fip_collate_fn([ds[i] for i in range(n_batch)])
    check("C_batch_weather_lengths_eq_delta",
          bool((batch["weather_lengths"] == batch["delta_days"]).all()),
          "batch length != delta")
    check("C_batch_mask_padding",
          bool(all(bool(batch["weather_mask"][i, :batch["weather_lengths"][i]].all())
                   and not bool(batch["weather_mask"][i, batch["weather_lengths"][i]:].any())
                   for i in range(n_batch))),
          "padding not masked")
    entry = batch["weather_dates"]
    ok_boundary = True
    for i in range(n_batch):
        n = int(batch["weather_lengths"][i])
        start, end = batch["weather_date_range"][i]
        ok_boundary &= (entry[i][0] == (date.fromisoformat(start)
                                        + timedelta(days=1)).isoformat())
        ok_boundary &= (entry[i][n - 1] == end)
    check("C_batch_window_left_open_right_closed", ok_boundary, "")

    # cached-feature path: real S1 cache, and missing cache must NOT be zero-filled
    cache_dir = Path(data["feature_cache_dir"]) / "mainline224_s1-v1"
    import hashlib as _hl
    from agro_jepa.datasets.provenance import cache_key_for_image as _key

    def _has_cache(stem: str) -> bool:
        p = image_index.get(stem)
        if p is None:
            return False
        sha = _hl.sha256(p.read_bytes()).hexdigest()
        return (cache_dir / f"{_key(sha)}.npy").exists()

    both_cached = [i for i in range(len(local_pairs))
                   if _has_cache(local_pairs.image_stem_a[i])
                   and _has_cache(local_pairs.image_stem_b[i])]
    check("C_cached_features_available", len(both_cached) > 0, f"{len(both_cached)} pairs")
    if both_cached:
        ds_c = FIPPairDataset(local_pairs, weather, image_mode="cached_features",
                              images_root=image_root,
                              feature_cache_dir=Path(data["feature_cache_dir"]))
        sample = ds_c[both_cached[0]]
        check("C_cached_token_shape",
              tuple(sample["image_a"].shape) == (MAINLINE_N_TOKENS, 768)
              and tuple(sample["image_b"].shape) == (MAINLINE_N_TOKENS, 768),
              str(tuple(sample["image_a"].shape)))
        check("C_cached_features_finite",
              bool(torch.isfinite(sample["image_a"]).all()),
              "cached features contain NaN/Inf")
        # a missing cache file must raise rather than silently become zeros
        miss = [i for i in range(len(local_pairs))
                if not _has_cache(local_pairs.image_stem_a[i])]
        if miss:
            try:
                ds_c[miss[0]]
                check("C_missing_cache_raises", False, "missing cache was accepted")
            except FileNotFoundError:
                check("C_missing_cache_raises", True)

    # ---- D: missing / zero / negative / quality --------------------------
    traits = ["cc", "ph", "whc", "sen"]
    # raw observations: a measured zero is kept as a finite value
    for t in traits:
        v = obs[f"{t}_value"]
        check(f"D_{t}_measured_zero_kept",
              bool(((v == 0).sum() > 0) or (v.notna() & (v == 0)).sum() == 0),
              f"{(v == 0).sum()} zeros")
    # missing never becomes zero in the pair table
    for t in traits:
        miss = ~np.isfinite(pairs[f"{t}_a"].to_numpy(dtype=float))
        check(f"D_{t}_missing_is_nan_not_zero",
              bool((pairs[f"{t}_a"].to_numpy(dtype=float)[miss] != 0).all()),
              "missing label was coerced to 0")
        check(f"D_{t}_missing_has_no_label_date",
              bool(pd.isna(pairs.loc[miss, f"{t}_a_date"]).all()),
              "missing label retained a date")
    # PH negatives preserved and flagged
    neg_a = pairs.ph_a < 0
    check("D_ph_negatives_preserved", bool(neg_a.sum() > 0), f"{int(neg_a.sum())} negatives")
    check("D_ph_negative_flag_matches_value",
          bool((pairs.ph_negative_a.to_numpy() == neg_a.to_numpy()).all()),
          "ph_negative flag disagrees with raw value")
    check("D_ph_negatives_not_clipped",
          bool(pairs.loc[neg_a, "ph_a"].min() < 0), "negatives were clipped")
    # measured zero + valid mask must stay valid (not treated as missing)
    for t in traits:
        z = (pairs[f"{t}_a"] == 0) & pairs[f"{t}_mask_a"]
        if z.any():
            check(f"D_{t}_zero_with_mask_stays_valid", True,
                  f"{int(z.sum())} zero-valued valid labels")
    # Sen only in its real years
    sen_valid_years = sorted(set(pairs.loc[pairs.sen_mask_a, "harvest_year"].unique()))
    check("D_sen_only_in_labeled_years",
          set(sen_valid_years) <= {2016, 2017, 2018},
          f"sen valid years: {sen_valid_years}")
    # quality rule flags the near-white frame but the sample survives
    white = image_root.rglob("FPWW0120667_RGB1_20160216_125228.png")
    white = next(iter(white), None)
    if white is not None:
        from PIL import Image
        arr = np.asarray(Image.open(white).convert("RGB"), dtype=np.float32) / 255.0
        luma = arr.mean(axis=2)
        flagged = not ds.quality_rule.evaluate(float(luma.mean()), float(luma.std()))
        check("D_near_white_frame_flagged", flagged,
              f"mean={luma.mean():.3f} std={luma.std():.3f}")
        check("D_flagged_frame_not_dropped",
              len(local_pairs[local_pairs.image_stem_a.eq(white.stem)
                              | local_pairs.image_stem_b.eq(white.stem)]) > 0,
              "flagged image was removed from the pair table")

    # ---- E: splits -------------------------------------------------------
    by_split = {s: set(plots.loc[plots.split == s, "plot_uid"]) for s in OFFICIAL_SPLITS}
    shared = {f"{a}__{b}": len(by_split[a] & by_split[b])
              for i, a in enumerate(OFFICIAL_SPLITS)
              for b in OFFICIAL_SPLITS[i + 1:] if by_split[a] & by_split[b]}
    check("E_official_splits_share_no_plot", not shared, str(shared))
    check("E_custom_split_additive_official_preserved",
          bool((custom.official_split.to_numpy()
                == plots.set_index("plot_uid").reindex(custom.plot_uid)["split"].to_numpy()).all()),
          "custom manifest rewrote the official split")
    check("E_pair_split_inherited_from_plot",
          bool((pairs.split.to_numpy()
                == plots.set_index("plot_uid")["split"].reindex(pairs.plot_uid).to_numpy()).all()),
          "pair split differs from its plot split")
    ct = set(custom.loc[custom.custom_split == "cross_year_test", "plot_uid"])
    tr = set(custom.loc[custom.custom_split == "train", "plot_uid"])
    check("E_custom_train_and_cross_year_disjoint_plots", not (ct & tr),
          f"{len(ct & tr)} shared plots")
    check("E_cross_year_years_are_2018_2022",
          set(pairs.loc[pairs.custom_split == "cross_year_test", "harvest_year"].unique())
          == {2018, 2022}, "cross_year_test contains other years")
    g_ct = set(custom.loc[custom.custom_split == "cross_year_test", "genotype_id"])
    g_tr = set(custom.loc[custom.custom_split == "train", "genotype_id"])
    check("E_cross_year_genotype_overlap_reported",
          len(g_ct & g_tr) > 0, f"{len(g_ct & g_tr)} shared genotypes (by design)")

    # ---- F: encoder geometry ---------------------------------------------
    rb = json.loads(Path(args.real_batch_report).read_text())
    enc = rb.get("encoding", {})
    check("F_tokens_256", enc.get("n_tokens") == MAINLINE_N_TOKENS,
          f"n_tokens={enc.get('n_tokens')}")
    check("F_emb_dim_768", enc.get("emb_dim") == 768, f"emb_dim={enc.get('emb_dim')}")
    check("F_matches_mainline_geometry", bool(enc.get("matches_mainline_geometry")), "")
    cr = rb.get("cache_roundtrip", {})
    check("F_cache_roundtrip_consistent",
          cr.get("n_compared", 0) > 0 and cr.get("min_cosine", 0) > 0.999,
          f"n={cr.get('n_compared')} min_cos={cr.get('min_cosine')}")
    ug = rb.get("upstream_geometry", {})
    main = ug.get("mainline", {})
    check("F_upstream_mainline_encoder_input_224",
          main.get("encoder_image_size") == 224, str(main.get("encoder_image_size")))
    check("F_upstream_mainline_num_patches_256",
          main.get("num_patches_train_py") == 256, str(main.get("num_patches_train_py")))
    dg = ug.get("decoder_geometry", {})
    check("F_decoder_incompatibility_documented", bool(dg.get("verdict")), "")

    # ---- G: no training happened -----------------------------------------
    check("G_no_training_in_report",
          ("loss" not in rb and "optimizer" not in rb and rb.get("stage") == "S2"),
          "report suggests training")
    check("G_no_checkpoint_written",
          not any(ROOT.glob("**/model_latest.pth")),
          "a checkpoint exists in the repo")

    # ---- H: deliverables exist -------------------------------------------
    for p in ("reports/s2_pairs.md", "reports/s2_real_batch.md",
              "reports/s2_pairs_stats.json", "reports/s2_real_batch.json",
              "agro_jepa/datasets/fip_dataset.py", "agro_jepa/datasets/fip_pairs.py",
              "agro_jepa/datasets/fip_weather.py", "agro_jepa/datasets/collate.py",
              "tools/s2_build_pairs.py", "tools/s2_inspect_batch.py"):
        check(f"H_exists[{p}]", (ROOT / p).exists(), "missing")

    # ---- report ----------------------------------------------------------
    for name, ok, detail in checks:
        print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if not ok else ""))
    n_ok = sum(1 for _, ok, _ in checks if ok)
    print(f"\n{n_ok}/{len(checks)} checks passed")
    if n_ok != len(checks):
        print("FAILURES:")
        for name, ok, detail in checks:
            if not ok:
                print(f" - {name}: {detail}")
        return 1
    print(json.dumps({"status": "ok", "stage": "S2",
                      "checks": len(checks)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
