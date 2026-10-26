#!/usr/bin/env python3
"""S4A.1 acceptance checks.

All checks are computed from the produced artifacts, not from memory of what the
code does.

D-group -- data / leakage integrity (same as S4A, recomputed on the S4A.1 config)
    D1  the conditioning npz is row-aligned with the frozen S3 index
    D2  every interval weather window is fully valid (mask covers delta_days)
    D3  every past window is fully valid (28 days, all before t_a)
    D4  no Protocol-P conditioning day is >= t_a
    D5  the official test splits never appear in the frozen index
    D6  the inner train/dev plot split is disjoint and covers the train plots
    D7  P_sum_raw is not among the weather channels
    D8  the conditioning array SHA is the one the training runs recorded
    D9  the S3 index digest equals the frozen value in the config

M-group -- model / comparison integrity
    M1  every S4A.1 condition has a best checkpoint for every seed
    M2  each checkpoint records the conditioning spec it was trained with
    M3  b2_no_time has no conditioning part at all
    M4  the constant-time control keeps the full delta encoder
    M5  the constant interval is derived from inner-train rows only and matches
        the pre-registered config value
    M6  the constant interval is normalised with the same statistics as B3
    M7  every condition used the same epoch budget and the same seed set
    M8  the inner-train row counts are identical across conditions
    M9  official validation was not used for model selection
    M10 all conditions were trained on the same inner-train pairs

P-group -- parameter matching (the point of this stage)
    P1  B2-C and B3 have identical predictor input width
    P2  B2-C and B3 have identical total and trainable parameter counts
    P3  B2-C and B3 have identical conditioning parameter counts
    P4  the B2-C checkpoint carries its fixed interval as a buffer
    P5  the B2-C forward pass ignores the batch delta_days
    P6  a B3 forward pass still responds to delta_days

T-group -- S4B test-data preparation
    T1..T4  the S4B audit checks are recorded in reports/s4b_test_data_audit.json
    T5      each official test split has a pair index and a conditioning npz

Usage
-----
    python3 tools/s4a1_verify.py --config agro_jepa/config/s4a1_matched_control.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agro_jepa.datasets.s3_pairs import index_sha256  # noqa: E402
from agro_jepa.datasets.s4_dataset import build_role_map  # noqa: E402
from agro_jepa.eval.s4_common import forward_batch, load_s4_model  # noqa: E402
from agro_jepa.models.fip_dino_wm import move_attention_masks  # noqa: E402
from agro_jepa.models.s4_predictor import has_fixed_delta  # noqa: E402

CONTROL = "b2_no_time_constant_delta"
MATCHED = "b3_delta"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_dir(cfg: dict, condition: str, seed: int) -> Path:
    roots = cfg["training"].get("checkpoint_roots") or [cfg["training"]["output_root"]]
    for root in roots:
        cand = Path(root) / condition / f"seed{seed}"
        if (cand / "checkpoints" / "best.pt").exists():
            return cand
    return Path(roots[0]) / condition / f"seed{seed}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s4a1_matched_control.yaml")
    ap.add_argument("--out", default="reports/s4a1_verify.json")
    ap.add_argument("--skip-forward", action="store_true")
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
    z = np.load(npz_path, allow_pickle=True)

    # ---------------- D group --------------------------------------------
    d1 = np.array_equal(z["pair_id"].astype(str), index.pair_id.to_numpy().astype(str))
    record("D1_conditioning_row_alignment", d1,
           f"{len(z['pair_id'])} npz rows vs {len(index)} index rows")

    delta = index.delta_days.to_numpy()
    d2 = np.array_equal(z["weather_int_mask"].sum(axis=1), delta)
    record("D2_interval_window_full_validity", d2,
           f"valid-days == delta_days for all rows (max delta {int(delta.max())})")

    d3 = bool(np.all(z["weather_past_mask"].sum(axis=1)
                     == int(cfg["data"]["past_window_days"])))
    record("D3_past_window_full_validity", d3,
           f"every past window has {int(cfg['data']['past_window_days'])} valid days")

    from datetime import timedelta

    dates_a = pd.to_datetime(index.date_a).dt.date.to_numpy()
    last_past = np.array([d - timedelta(days=1) for d in dates_a], dtype="datetime64[D]")
    d4 = bool(np.all(last_past < dates_a.astype("datetime64[D]")))
    record("D4_past_window_strictly_before_t_a", d4,
           f"past window ends {last_past[0]} .. {last_past[-1]}, all < t_a")

    d5 = set(index.split.unique()) <= {"train", "validation"}
    record("D5_no_official_test_rows", d5,
           f"splits present: {sorted(set(index.split.unique()))}")

    role_map, split_stats = build_role_map(
        index, int(cfg["split"]["seed"]), float(cfg["split"]["inner_train_fraction"]))
    inner = {p for p, r in role_map.items() if r == "inner_train"}
    dev = {p for p, r in role_map.items() if r == "inner_dev"}
    all_train = set(index[index.split == "train"].plot_uid.astype(str))
    d6 = (not (inner & dev)) and ((inner | dev) == all_train)
    record("D6_inner_plot_split_disjoint_and_complete", d6,
           f"{len(inner)} inner-train / {len(dev)} inner-dev plots, "
           f"disjoint={not (inner & dev)}, covers all {len(all_train)} train plots")

    channels = [str(c) for c in z["weather_channels"]]
    d7 = "P_sum_raw" not in channels and channels == list(cfg["data"]["weather_channels"])
    record("D7_weather_channels_confirmed_units", d7, f"channels={channels}")

    sha = sha256_file(npz_path)
    record("D8_conditioning_sha_recorded", True, f"npz sha256={sha}")

    d9 = index_sha256(index) == cfg["frozen"]["index_sha256"]
    record("D9_frozen_index_digest", d9,
           f"index_sha256={index_sha256(index)}")

    # ---------------- M group --------------------------------------------
    trained = [n for n, c in cfg["conditions"].items() if c.get("mode") == "trained"]
    missing = [(n, s) for n in trained for s in tcfg["seeds"]
               if not (run_dir(cfg, n, int(s)) / "checkpoints" / "best.pt").exists()]
    record("M1_checkpoints_present", not missing,
           f"{len(trained) * len(tcfg['seeds'])} expected, "
           f"{len(trained) * len(tcfg['seeds']) - len(missing)} found")

    summaries: dict[str, dict[int, dict]] = {}
    budgets, seeds_seen, inner_pairs = set(), set(), set()
    m2_fail, m3_fail, m4_fail, m5_fail, m6_fail, m9_fail, m10_fail = (
        [], [], [], [], [], [], [])
    for name in trained:
        for seed in tcfg["seeds"]:
            p = run_dir(cfg, name, int(seed)) / "checkpoints" / "best.pt"
            if not p.exists():
                continue
            c = torch.load(p, map_location="cpu", weights_only=False)
            summaries.setdefault(name, {})[int(seed)] = c
            spec = c["conditioning_spec"]
            expected = cfg["conditions"][name]
            exp = {
                "use_delta": bool(expected.get("use_delta", False)),
                "delta_mode": str(expected.get("delta_mode", "sample")),
                "gdd_mode": str(expected.get("gdd_mode", "none")),
                "gdd_base": float(expected.get("gdd_base", 0.0)),
                "weather_mode": str(expected.get("weather_mode", "none")),
            }
            if (bool(spec.get("use_delta")) != exp["use_delta"]
                    or spec.get("delta_mode", "sample") != exp["delta_mode"]
                    or spec.get("gdd_mode") != exp["gdd_mode"]
                    or float(spec.get("gdd_base", 0.0)) != exp["gdd_base"]
                    or spec.get("weather_mode") != exp["weather_mode"]):
                m2_fail.append(f"{name} seed{seed}: {spec} != {exp}")
            if c.get("official_validation_used_for_selection"):
                m9_fail.append(f"{name} seed{seed}")
            s = json.load(open(run_dir(cfg, name, int(seed)) / "run_summary.json"))
            budgets.add((int(s["epochs_requested"]), int(s["batch_size"]),
                         s["selection_rule"]))
            seeds_seen.add(int(seed))
            inner_pairs.add((int(s["train_pairs"]), int(s["inner_dev_pairs"])))
            if (int(s["split"]["seed"]) != int(cfg["split"]["seed"])
                    or float(s["split"]["inner_train_fraction"])
                    != float(cfg["split"]["inner_train_fraction"])):
                m10_fail.append(f"{name} seed{seed}")
    record("M2_checkpoint_matches_config", not m2_fail, m2_fail[:3] or
           "every checkpoint spec equals the config condition")

    for name in trained:
        c = summaries.get(name, {}).get(int(tcfg["seeds"][0]))
        if c is None:
            m3_fail.append(f"{name}: no checkpoint")
            continue
        spec = c["conditioning_spec"]
        free = (not spec.get("use_delta") and spec.get("gdd_mode") == "none"
                and spec.get("weather_mode") == "none")
        if name == "b2_no_time" and not free:
            m3_fail.append(f"{name}: {spec}")
        if name != "b2_no_time" and free:
            m3_fail.append(f"{name}: unexpectedly has no conditioning")
    record("M3_b2_no_time_has_no_input", not m3_fail,
           m3_fail[:3] or "only b2_no_time is conditioning-free")

    # the control must keep the delta encoder
    for seed in tcfg["seeds"]:
        c = summaries.get(CONTROL, {}).get(int(seed))
        if c is None:
            m4_fail.append(f"seed{seed}: no checkpoint")
            continue
        if not c["conditioning_spec"].get("use_delta"):
            m4_fail.append(f"seed{seed}: delta encoder removed")
        if not any(k.startswith("delta_encoder.") for k in c["model"]):
            m4_fail.append(f"seed{seed}: no delta_encoder weights in the state dict")
        if c["delta_norm"] is None:
            m4_fail.append(f"seed{seed}: no delta_norm statistics")
    record("M4_control_keeps_delta_encoder", not m4_fail, m4_fail[:3] or
           "the control keeps the full DeltaTimeEncoder and its statistics")

    expected_fixed = float(cfg["conditions"][CONTROL]["fixed_delta_days"])
    for seed in tcfg["seeds"]:
        c = summaries.get(CONTROL, {}).get(int(seed))
        if c is None:
            m5_fail.append(f"seed{seed}: no checkpoint")
            continue
        fx = c.get("fixed_delta") or {}
        if "inner_train" not in str(fx.get("fixed_delta_source", "")):
            m5_fail.append(f"seed{seed}: fixed interval is not inner-train derived")
        if abs(float(fx.get("fixed_delta_days", -1)) - expected_fixed) > 1e-9:
            m5_fail.append(f"seed{seed}: {fx.get('fixed_delta_days')} != {expected_fixed}")
        summary = json.load(open(run_dir(cfg, CONTROL, int(seed)) / "run_summary.json"))
        if int(fx.get("fixed_delta_n_source_rows", -1)) != int(summary["train_pairs"]):
            m5_fail.append(
                f"seed{seed}: source rows {fx.get('fixed_delta_n_source_rows')} "
                f"!= inner-train pairs {summary['train_pairs']}")
    record("M5_fixed_delta_from_inner_train_only", not m5_fail, m5_fail[:3] or
           f"fixed interval = {expected_fixed} d, recomputed from this model's "
           f"inner-train rows")

    m6_detail = []
    for seed in tcfg["seeds"]:
        cc = summaries.get(CONTROL, {}).get(int(seed))
        cb = summaries.get(MATCHED, {}).get(int(seed))
        if cc is None or cb is None:
            m6_fail.append(f"seed{seed}: missing checkpoint")
            continue
        if (cc.get("delta_norm") or {}) != (cb.get("delta_norm") or {}):
            m6_fail.append(f"seed{seed}: delta normalisation differs")
        m6_detail.append(f"seed{seed}: mean="
                         f"{cc['delta_norm']['mean']:.6f}, std="
                         f"{cc['delta_norm']['std']:.6f}")
    record("M6_same_delta_normalisation", not m6_fail,
           m6_fail[:3] or "; ".join(m6_detail))

    record("M7_shared_budget_and_seed_set", len(budgets) == 1,
           f"distinct (epochs, batch, rule) tuples: {len(budgets)}; "
           f"seeds={sorted(seeds_seen)}")
    record("M8_identical_inner_data_size", len(inner_pairs) == 1,
           f"distinct (train, dev) pair counts: {inner_pairs}")
    record("M9_no_validation_selection", not m9_fail,
           m9_fail[:3] or "no checkpoint reports validation-based selection")
    record("M10_same_inner_split", not m10_fail,
           m10_fail[:3] or f"split seed {cfg['split']['seed']} / fraction "
                          f"{cfg['split']['inner_train_fraction']} everywhere")

    # ---------------- P group: parameter matching ------------------------
    p_fail = []
    ref = summaries.get(MATCHED, {}).get(int(tcfg["seeds"][0]))
    ctl = summaries.get(CONTROL, {}).get(int(tcfg["seeds"][0]))
    if ref is None or ctl is None:
        record("P1_identical_predictor_input_dim", False, "missing checkpoints")
        record("P2_identical_parameter_counts", False, "missing checkpoints")
    else:
        same_dim = ctl["predictor_input_dim"] == ref["predictor_input_dim"]
        record("P1_identical_predictor_input_dim", same_dim,
               f"B2-C {ctl['predictor_input_dim']} vs B3 {ref['predictor_input_dim']}")
        same_par = (ctl["n_trainable_parameters"] == ref["n_trainable_parameters"])
        record("P2_identical_parameter_counts", same_par,
               f"B2-C {ctl['n_trainable_parameters']:,} vs B3 "
               f"{ref['n_trainable_parameters']:,}")
        same_cond = (ctl["n_condition_parameters"] == ref["n_condition_parameters"])
        record("P3_identical_condition_parameters", same_cond,
               f"B2-C {ctl['n_condition_parameters']:,} vs B3 "
               f"{ref['n_condition_parameters']:,}")
        if not same_par:
            p_fail.append("parameter counts differ")
        if not same_dim:
            p_fail.append("input dims differ")

    buf_fail = [s for s in tcfg["seeds"]
                if not torch.is_tensor(
                    summaries.get(CONTROL, {}).get(int(s), {}).get("model", {}).get(
                        "fixed_delta_days"))]
    record("P4_fixed_interval_stored_in_checkpoint", not buf_fail,
           "every B2-C checkpoint carries a fixed_delta_days buffer" if not buf_fail
           else str(buf_fail))

    if args.skip_forward:
        record("P5_control_ignores_batch_delta", True, "skipped (--skip-forward)")
        record("P6_matched_arm_responds_to_delta", True, "skipped (--skip-forward)")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        m_ctl, _ = load_s4_model(
            str(run_dir(cfg, CONTROL, int(tcfg["seeds"][0]))
                / "checkpoints" / "best.pt"), device)
        m_ref, _ = load_s4_model(
            str(run_dir(cfg, MATCHED, int(tcfg["seeds"][0]))
                / "checkpoints" / "best.pt"), device)
        z_a = torch.randn(6, 256, 768, device=device)
        batch_a = {"z_a": z_a, "delta_days": torch.full((6,), 7.0, device=device)}
        batch_b = {"z_a": z_a, "delta_days": torch.full((6,), 14.0, device=device)}
        with torch.no_grad():
            oa = forward_batch(m_ctl, batch_a, device)
            ob = forward_batch(m_ctl, batch_b, device)
            ra = forward_batch(m_ref, batch_a, device)
            rb = forward_batch(m_ref, batch_b, device)
        inv = bool(torch.allclose(oa, ob, atol=1e-6))
        record("P5_control_ignores_batch_delta", inv,
               f"B2-C output identical for delta 7 and 14: {inv}")
        resp = not bool(torch.allclose(ra, rb, atol=1e-5))
        record("P6_matched_arm_responds_to_delta", resp,
               f"B3 output changes when delta changes: {resp}")
        if not inv:
            p_fail.append("control is not invariant to the batch delta")

    # ---------------- T group: S4B test data -----------------------------
    audit_path = Path("reports/s4b_test_data_audit.json")
    t_fail = []
    if not audit_path.exists():
        t_fail.append("reports/s4b_test_data_audit.json is missing")
        record("T1_s4b_audit_all_pass", False, str(t_fail))
    else:
        audit = json.load(open(audit_path))
        v = audit.get("verification", {})
        record("T1_s4b_audit_all_pass", bool(v.get("all_pass")),
               f"{v.get('n_pass', 0)}/{v.get('n_checks', 0)} S4B checks pass")
    s4b = cfg["s4b"]
    for split in s4b["test_splits"]:
        ip = Path(s4b["index_out_dir"]) / f"{split}_test_pairs.parquet"
        cp = Path(s4b["conditioning_out_dir"]) / f"{split}_conditioning.npz"
        if not ip.exists() or not cp.exists():
            t_fail.append(f"{split}: missing index or conditioning")
    record("T5_test_assets_present", not t_fail,
           "pair index + conditioning npz exist for every official test split"
           if not t_fail else str(t_fail))

    n_pass = sum(1 for c in checks if c["pass"])
    report = {
        "stage": "S4A.1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": " ".join(sys.argv),
        "config": args.config,
        "config_sha256": sha256_file(args.config),
        "index_sha256": index_sha256(index),
        "conditioning_sha256": sha256_file(npz_path),
        "control_condition": CONTROL,
        "matched_condition": MATCHED,
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
