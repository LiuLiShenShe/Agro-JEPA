#!/usr/bin/env python3
"""S4B.4A Phase 2: freeze the five-fold year-held-out protocol.

Builds, from the frozen full official-train index only:

* five folds, one held-out harvest year each (2016, 2017, 2018, 2021, 2022);
* inside each fold, the other four years are split 80/20 by *plot*, stratified
  by year, into ``fold_train`` and ``fold_dev`` (epoch selection only);
* the held-out year is entirely excluded from train and dev.

Writes one parquet per fold/role under ``external_data/s4b4a_folds/`` plus
``reports/s4b4a_protocol.{json,md}``.  Every index digest, sample count, plot
count and year list is recorded so the experiment can be reproduced and the
verify tool can detect drift.  No test row and no official-validation row is
read at any point.

Usage
-----
    python3 tools/s4b4a_protocol.py \
        --config agro_jepa/config/s4b4a_year_heldout.yaml
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
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frame_sha(df: pd.DataFrame) -> str:
    cols = sorted(df.columns)
    return hashlib.sha256(df[cols].to_csv(index=False).encode()).hexdigest()


def build_master(cfg: dict) -> pd.DataFrame:
    """train_index (cache columns) + labels from full_train_pairs, row-aligned."""
    tr = pd.read_parquet(ROOT / cfg["data"]["train_index"])
    full = pd.read_parquet(ROOT / cfg["data"]["full_train_pairs"])
    if not np.array_equal(tr.pair_id.astype(str).to_numpy(),
                          full.pair_id.astype(str).to_numpy()):
        raise SystemExit("train_index and full_train_pairs are not row-aligned")
    label_cols = [c for c in full.columns if c not in tr.columns]
    master = tr.copy()
    for c in label_cols:
        master[c] = full[c].to_numpy()
    # conditioning rows are aligned to full/train row order
    master["_orig_row"] = np.arange(len(master), dtype=np.int64)
    return master


def split_plots(plots_by_year: dict[int, np.ndarray], seed: int,
                fraction: float) -> tuple[set, set]:
    """80/20 plot split, stratified by year; same rule as build_role_map.

    Plot uids are strings (e.g. ``FPWW0220027``); they are sorted first and then
    permuted by position, exactly as ``build_role_map`` does.
    """
    rng = np.random.default_rng(seed)
    train, dev = set(), set()
    for y in sorted(plots_by_year):
        plots = np.sort(np.asarray(plots_by_year[y], dtype=str))
        perm = rng.permutation(len(plots))
        n_dev = max(1, int(round(len(plots) * (1.0 - fraction))))
        dev.update(str(plots[p]) for p in perm[:n_dev])
        train.update(str(plots[p]) for p in perm[n_dev:])
    return train, dev


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s4b4a_year_heldout.yaml")
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    t0 = time.time()
    master = build_master(cfg)
    if (ROOT / cfg["data"]["conditioning_npz"]).is_file():
        cond_sha = sha256_file(ROOT / cfg["data"]["conditioning_npz"])
        if cond_sha != cfg["data"]["conditioning_npz_sha256"]:
            raise SystemExit("full-train conditioning digest mismatch")
    years = [int(y) for y in cfg["folds"]["years"]]
    heldout_list = [int(y) for y in cfg["folds"]["heldout_year_per_fold"]]
    if sorted(years) != sorted(heldout_list):
        raise SystemExit("each year must be held out exactly once")

    fold_dir = ROOT / cfg["artifacts"]["fold_dir"]
    fold_dir.mkdir(parents=True, exist_ok=True)

    all_plots = set(master.plot_uid.astype(str).unique())
    fold_records = []
    for fold_year in heldout_list:
        heldout_plots = set(
            master.loc[master.harvest_year == fold_year, "plot_uid"]
            .astype(str).unique())
        remaining = master[master.harvest_year != fold_year]
        plots_by_year = {
            int(y): g.plot_uid.astype(str).unique()
            for y, g in remaining.groupby("harvest_year")
        }
        train_plots, dev_plots = split_plots(
            plots_by_year, int(cfg["folds"]["inner_split_seed"]),
            float(cfg["folds"]["inner_train_fraction"]))

        # strict disjointness / coverage
        if train_plots & dev_plots:
            raise SystemExit(f"fold {fold_year}: train/dev plots overlap")
        if (train_plots | dev_plots) & heldout_plots:
            raise SystemExit(f"fold {fold_year}: heldout leaks into train/dev")
        if (train_plots | dev_plots | heldout_plots) != all_plots:
            raise SystemExit(f"fold {fold_year}: folds do not cover all plots")

        def frame_for(plots: set) -> pd.DataFrame:
            f = master[master.plot_uid.astype(str).isin(plots)].copy()
            return f.reset_index(drop=True)

        roles = {
            "fold_train": frame_for(train_plots),
            "fold_dev": frame_for(dev_plots),
            "heldout": frame_for(heldout_plots),
        }
        entry = {"fold_year": fold_year, "roles": {}}
        for role, f in roles.items():
            if len(f) == 0:
                raise SystemExit(f"fold {fold_year}/{role}: empty")
            out = fold_dir / f"fold{fold_year}_{role}.parquet"
            f.to_parquet(out, index=False)
            entry["roles"][role] = {
                "path": str(out.relative_to(ROOT)),
                "pairs": int(len(f)),
                "plots": int(f.plot_uid.nunique()),
                "images": int(pd.unique(pd.concat(
                    [f.image_stem_a, f.image_stem_b])).size),
                "years": sorted(int(y) for y in f.harvest_year.unique()),
                "span7": int((f.pair_type == "span7").sum()),
                "span14": int((f.pair_type == "span14").sum()),
                "delta_days": {
                    "min": int(f.delta_days.min()),
                    "max": int(f.delta_days.max()),
                    "mean": float(f.delta_days.mean()),
                    "median": float(f.delta_days.median()),
                    "std": float(f.delta_days.std()),
                },
                "file_sha256": sha256_file(out),
                "frame_sha256": frame_sha(f),
                "orig_rows_sha256": hashlib.sha256(
                    f["_orig_row"].to_numpy(np.int64).tobytes()).hexdigest(),
            }
        entry["disjoint_plots"] = True
        entry["covers_all_plots"] = True
        fold_records.append(entry)
        print(f"[fold {fold_year}] train={entry['roles']['fold_train']['pairs']} "
              f"dev={entry['roles']['fold_dev']['pairs']} "
              f"heldout={entry['roles']['heldout']['pairs']} pairs", flush=True)

    protocol = {
        "stage": "S4B.4A",
        "task": "five-fold year-held-out protocol (frozen before any evaluation)",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": " ".join(sys.argv),
        "config": args.config,
        "config_sha256": sha256_file(ROOT / args.config),
        "data_scope": {
            "index": cfg["data"]["full_train_pairs"],
            "index_sha256": sha256_file(ROOT / cfg["data"]["full_train_pairs"]),
            "pairs": int(len(master)),
            "plots": int(master.plot_uid.nunique()),
            "years": sorted(int(y) for y in master.harvest_year.unique()),
            "official_validation_read": False,
            "official_test_read": False,
            "note": (
                "training-set-internal diagnostic; does not use or confirm 2019 "
                "unseen-environment performance"),
        },
        "fold_rule": {
            "heldout": "one whole harvest year (all its plots)",
            "inner_split": "80/20 by plot, stratified by harvest_year",
            "inner_split_seed": int(cfg["folds"]["inner_split_seed"]),
            "inner_train_fraction": float(cfg["folds"]["inner_train_fraction"]),
            "strata": cfg["folds"]["strata"],
            "disjoint_rule": "same plot never spans train/dev/heldout",
            "norm_stats": (
                "DeltaNormStats and GDD/weather VectorNormStats are fitted on "
                "each fold's fold_train rows only; dev/heldout are transformed "
                "with those training statistics"),
            "selection": (
                "epoch = minimum fold-dev future-latent MSE; heldout year is "
                "evaluated once and never used for selection"),
        },
        "folds": fold_records,
        "conditions": {
            name: {
                "description": c.get("description"),
                "use_delta": c.get("use_delta"),
                "delta_mode": c.get("delta_mode", "sample"),
                "fixed_delta_days": c.get("fixed_delta_days"),
                "gdd_mode": c.get("gdd_mode"),
                "gdd_base": c.get("gdd_base"),
                "weather_mode": c.get("weather_mode"),
                "protocol": c.get("protocol"),
            }
            for name, c in cfg["conditions"].items()
        },
        "e1_conditions": cfg["e1_conditions"],
        "e1_contrasts": cfg["e1_contrasts"],
        "e2_conditions": cfg["e2_conditions"],
        "training": {
            "seeds": cfg["training"]["seeds"],
            "candidate_epochs": cfg["training"]["candidate_epochs"],
            "early_stop_patience": cfg["training"]["early_stop_patience"],
            "batch_size": cfg["training"]["batch_size"],
            "lr": cfg["training"]["lr"],
            "selection_metric": cfg["training"]["selection_metric"],
        },
    }
    out_json = ROOT / cfg["artifacts"]["protocol_json"]
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(protocol, fh, indent=2)
    write_md(ROOT / cfg["artifacts"]["protocol_md"], protocol)
    print(f"wrote {out_json} (elapsed {time.time()-t0:.1f}s)", flush=True)
    return 0


def write_md(path: Path, r: dict) -> None:
    L = ["# S4B.4A Five-Fold Year-Held-Out Protocol (frozen before evaluation)", ""]
    L.append(f"- Generated: {r['generated_at']}")
    L.append(f"- Config: `{r['config']}` (sha `{r['config_sha256'][:16]}`)")
    d = r["data_scope"]
    L.append(f"- Data: **{d['pairs']:,} pairs / {d['plots']:,} plots / "
             f"years {d['years']}** (official-train only)")
    L.append(f"- Official validation read: {d['official_validation_read']}; "
             f"official test read: {d['official_test_read']}")
    L.append("")
    L.append("## Fold rule")
    L.append("")
    for k, v in r["fold_rule"].items():
        L.append(f"- **{k}**: {v}")
    L.append("")
    L.append("## Folds")
    L.append("")
    L.append("| fold (held-out year) | role | pairs | plots | years | span7 | span14 | delta mean [min,max] |")
    L.append("|---|---|---:|---:|---|---:|---:|---|")
    for f in r["folds"]:
        for role in ("fold_train", "fold_dev", "heldout"):
            e = f["roles"][role]
            dd = e["delta_days"]
            L.append(f"| {f['fold_year']} | {role} | {e['pairs']:,} | {e['plots']} | "
                     f"{e['years']} | {e['span7']:,} | {e['span14']:,} | "
                     f"{dd['mean']:.2f} [{dd['min']},{dd['max']}] |")
    L.append("")
    L.append("## Conditions")
    L.append("")
    L.append("| condition | delta | gdd | weather | protocol |")
    L.append("|---|---|---|---|---|")
    for name, c in r["conditions"].items():
        delta = ("const %.1f d" % c["fixed_delta_days"]) if c["delta_mode"] == "constant" \
            else ("sample" if c["use_delta"] else "none")
        L.append(f"| {name} | {delta} | {c['gdd_mode']} | {c['weather_mode']} | "
                 f"{c['protocol']} |")
    L.append("")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
