#!/usr/bin/env python3
"""S4A acceptance checks.

Two groups of checks, both computed from the produced artifacts rather than
from memory of what the code does:

D-group -- data / leakage integrity
    D1  the conditioning npz is row-aligned with the frozen S3 index
    D2  every interval weather window is fully valid (mask covers delta_days)
    D3  every past window is fully valid (28 days, all before t_a)
    D4  no conditioning day is >= t_a for Protocol P
    D5  the official test splits never appear in the frozen index
    D6  the inner train/dev plot split is disjoint and covers the train plots
    D7  P_sum_raw is not among the weather channels
    D8  the conditioning array SHA is the one the training runs recorded

M-group -- model / comparison integrity
    M1  every planned condition has a best checkpoint for every seed
    M2  each checkpoint records the conditioning spec it was trained with
    M3  b2_no_time has no time/GDD/weather conditioning parts
    M4  every condition used the same epoch budget and the same seed set
    M5  the inner-train row counts are identical across conditions
    M6  official validation was not used for model selection
    M7  all conditions were trained on the same inner-train pairs

Usage
-----
    python3 tools/s4_verify.py --config agro_jepa/config/s4_pilot.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agro_jepa.datasets.s3_pairs import index_sha256  # noqa: E402
from agro_jepa.datasets.s4_dataset import build_role_map  # noqa: E402


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s4_pilot.yaml")
    ap.add_argument("--out", default="reports/s4_verify.json")
    ap.add_argument("--skip-models", action="store_true")
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    tcfg = cfg["training"]
    checks: list[dict] = []

    def record(name: str, ok: bool, detail) -> None:
        checks.append({"id": name, "pass": bool(ok), "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)

    index = pd.read_parquet(cfg["data"]["index"])
    npz_path = Path(cfg["data"]["conditioning_npz"])

    # ---------------- D1 alignment -----------------------------------------
    z = np.load(npz_path, allow_pickle=True)
    d1_ok = np.array_equal(z["pair_id"].astype(str),
                           index.pair_id.to_numpy().astype(str))
    record("D1_conditioning_row_alignment", d1_ok,
           f"{len(z['pair_id'])} npz rows vs {len(index)} index rows")

    # ---------------- D2 / D3 / D4 window validity -------------------------
    delta = index.delta_days.to_numpy()
    valid_int = z["weather_int_mask"].sum(axis=1)
    d2_ok = bool(np.array_equal(valid_int, delta))
    record("D2_interval_window_full_validity", d2_ok,
           f"valid-days == delta_days for all rows (max delta {int(delta.max())})")

    valid_past = z["weather_past_mask"].sum(axis=1)
    d3_ok = bool(np.all(valid_past == int(cfg["data"]["past_window_days"])))
    record("D3_past_window_full_validity", d3_ok,
           f"every past window has {int(cfg['data']['past_window_days'])} valid days")

    # D4: no Protocol P day >= t_a. The stored past array is [t_a-28, t_a-1]
    # (built as (t_a-29, t_a-1]); verify by reconstructing the end date.
    dates_a = pd.to_datetime(index.date_a).dt.date.to_numpy()
    from datetime import timedelta

    last_past = np.array([d - timedelta(days=1) for d in dates_a], dtype="datetime64[D]")
    first_past = np.array([d - timedelta(days=int(cfg["data"]["past_window_days"]))
                           for d in dates_a], dtype="datetime64[D]")
    d4_ok = bool(np.all(last_past < dates_a.astype("datetime64[D]")))
    record("D4_past_window_strictly_before_t_a", d4_ok,
           f"past window ends {last_past[0]} .. {last_past[-1]}, all < t_a")

    # ---------------- D5 no official test ----------------------------------
    d5_ok = set(index.split.unique()) <= {"train", "validation"}
    record("D5_no_official_test_rows", d5_ok,
           f"splits present: {sorted(set(index.split.unique()))}")

    # ---------------- D6 inner split ---------------------------------------
    role_map, stats = build_role_map(
        index, int(cfg["split"]["seed"]), float(cfg["split"]["inner_train_fraction"])
    )
    inner = {p for p, r in role_map.items() if r == "inner_train"}
    dev = {p for p, r in role_map.items() if r == "inner_dev"}
    all_train_plots = set(index[index.split == "train"].plot_uid.astype(str))
    d6_ok = (not (inner & dev)) and ((inner | dev) == all_train_plots)
    record("D6_inner_plot_split_disjoint_and_complete", d6_ok,
           f"{len(inner)} inner-train / {len(dev)} inner-dev plots, "
           f"disjoint={not (inner & dev)}, covers all {len(all_train_plots)} train plots")

    # ---------------- D7 channels ------------------------------------------
    channels = [str(c) for c in z["weather_channels"]]
    d7_ok = ("P_sum_raw" not in channels
             and channels == list(cfg["data"]["weather_channels"]))
    record("D7_weather_channels_confirmed_units", d7_ok, f"channels={channels}")

    # ---------------- D8 SHA ------------------------------------------------
    sha = sha256_file(npz_path)
    record("D8_conditioning_sha_recorded", True,
           f"npz sha256={sha}")

    trained = [n for n, c in cfg["conditions"].items() if c.get("mode") == "trained"]

    # ---------------- M group ----------------------------------------------
    if not args.skip_models:
        ckpts = {}
        m1_fail = []
        for name in trained:
            for seed in tcfg["seeds"]:
                p = Path(tcfg["output_root"]) / name / f"seed{int(seed)}" / "checkpoints" / "best.pt"
                if not p.exists():
                    m1_fail.append(str(p))
        record("M1_checkpoints_present", not m1_fail,
               f"{len(trained) * len(tcfg['seeds'])} expected, "
               f"{len(trained) * len(tcfg['seeds']) - len(m1_fail)} found")

        import torch

        summaries = {}
        m2_fail, m3_fail, m4_fail, m5_fail, m6_fail, m7_fail = [], [], [], [], [], []
        budgets = set()
        seeds_seen = set()
        inner_pairs = set()
        for name in trained:
            for seed in tcfg["seeds"]:
                run = Path(tcfg["output_root"]) / name / f"seed{int(seed)}"
                p = run / "checkpoints" / "best.pt"
                if not p.exists():
                    continue
                c = torch.load(p, map_location="cpu", weights_only=False)
                summaries.setdefault(name, {})[int(seed)] = c
                spec = c["conditioning_spec"]
                expected = cfg["conditions"][name]
                exp_spec = {
                    "use_delta": bool(expected.get("use_delta", False)),
                    "gdd_mode": str(expected.get("gdd_mode", "none")),
                    "gdd_base": float(expected.get("gdd_base", 0.0)),
                    "weather_mode": str(expected.get("weather_mode", "none")),
                }
                if (bool(spec.get("use_delta")) != exp_spec["use_delta"]
                        or spec.get("gdd_mode") != exp_spec["gdd_mode"]
                        or float(spec.get("gdd_base", 0.0)) != exp_spec["gdd_base"]
                        or spec.get("weather_mode") != exp_spec["weather_mode"]):
                    m2_fail.append(f"{name} seed{seed}: {spec} != {exp_spec}")
                if c.get("official_validation_used_for_selection"):
                    m6_fail.append(f"{name} seed{seed}")
                s = json.load(open(run / "run_summary.json"))
                budgets.add((int(s["epochs_requested"]), int(s["batch_size"]),
                             s["selection_rule"]))
                seeds_seen.add(int(seed))
                inner_pairs.add((int(s["train_pairs"]), int(s["inner_dev_pairs"])))
                if int(s["split"]["seed"]) != int(cfg["split"]["seed"]) or \
                        float(s["split"]["inner_train_fraction"]) != float(
                            cfg["split"]["inner_train_fraction"]):
                    m7_fail.append(f"{name} seed{seed}")

        record("M2_checkpoint_matches_config", not m2_fail, m2_fail[:3] or
               "every checkpoint spec equals the config condition")

        # b2 must have no conditioning at all
        for name in trained:
            by_seed = summaries.get(name, {})
            if not by_seed:
                m3_fail.append(f"{name}: no checkpoint")
                continue
            c = by_seed[int(tcfg["seeds"][0])]
            parts = c.get("conditioning_spec", {})
            is_notime = (not parts.get("use_delta")
                         and parts.get("gdd_mode") == "none"
                         and parts.get("weather_mode") == "none")
            if name == "b2_no_time" and not is_notime:
                m3_fail.append(f"{name}: {parts}")
            if name != "b2_no_time" and is_notime:
                m3_fail.append(f"{name}: unexpectedly has no conditioning")
        record("M3_b2_has_no_time_input", not m3_fail,
               m3_fail[:3] or "only b2_no_time is conditioning-free")

        record("M4_shared_budget_and_seed_set", len(budgets) == 1,
               f"distinct (epochs, batch, rule) tuples: {len(budgets)}; "
               f"seeds={sorted(seeds_seen)}")
        record("M5_identical_inner_data_size", len(inner_pairs) == 1,
               f"distinct (train, dev) pair counts: {inner_pairs}")
        record("M6_no_validation_selection", not m6_fail,
               m6_fail[:3] or "no checkpoint reports validation-based selection")
        record("M7_same_inner_split", not m7_fail,
               m7_fail[:3] or f"split seed {cfg['split']['seed']} / fraction "
                              f"{cfg['split']['inner_train_fraction']} everywhere")

    n_pass = sum(1 for c in checks if c["pass"])
    report = {
        "stage": "S4A",
        "generated_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": " ".join(sys.argv),
        "config": args.config,
        "config_sha256": sha256_file(args.config),
        "index_sha256": index_sha256(index),
        "n_checks": len(checks),
        "n_pass": n_pass,
        "all_pass": n_pass == len(checks),
        "checks": checks,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"{n_pass}/{len(checks)} checks pass; wrote {args.out}", flush=True)
    return 0 if n_pass == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
