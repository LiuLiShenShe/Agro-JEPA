#!/usr/bin/env python3
"""S4B.2: freeze the fixed epoch budget from real S4A.1 / S4A records.

For each of the three trainable conditions it reads each seed's
``run_summary.json`` (which carries the 0-based ``best_epoch`` recorded by the
actual trainer), computes ``fixed_epochs = median(best_epoch + 1)`` over seeds
{0,1,2}, records the source checkpoint SHA256, and checks the config's
pre-registered value.  A mismatch aborts: a full run must not start from a
guessed budget.

Outputs: reports/s4b2_training_plan.json and .md

Usage
-----
    python3 tools/s4b2_freeze_plan.py --config agro_jepa/config/s4b2_full_train.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CONDITIONS = ("b2_no_time", "b2_no_time_constant_delta", "b3_delta")
# same roots as S4A.1: seeds {0,1} of b2/b3 are the frozen S4A runs
ROOTS = ["external_data/s4a1_runs", "external_data/s4_runs"]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_summary(cond: str, seed: int) -> Path | None:
    for root in ROOTS:
        p = Path(root) / cond / f"seed{seed}" / "run_summary.json"
        if p.is_file():
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s4b2_full_train.yaml")
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    plan: dict = {
        "stage": "S4B.2",
        "task": "frozen fixed-epoch plan for full official-train runs",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": args.config,
        "config_sha256": sha256_file(args.config),
        "rule": cfg["epoch_policy"]["rule"],
        "checkpoint_roots": ROOTS,
        "conditions": {},
        "missing": [],
        "all_complete": True,
    }
    for cond in CONDITIONS:
        seeds = {}
        missing_seeds = []
        for seed in (0, 1, 2):
            sp = find_summary(cond, seed)
            if sp is None:
                missing_seeds.append(seed)
                plan["missing"].append(f"{cond}/seed{seed}")
                continue
            with open(sp) as fh:
                s = json.load(fh)
            ckpt = Path(s.get("checkpoint_best", ""))
            seeds[int(seed)] = {
                "run_summary": str(sp),
                "run_summary_sha256": sha256_file(sp),
                "best_epoch": int(s["best_epoch"]),
                "best_epoch_zero_based": True,
                "epochs_completed": int(s.get("epochs_completed", -1)),
                "checkpoint_best": str(ckpt),
                "checkpoint_best_sha256": sha256_file(ckpt) if ckpt.is_file() else None,
                "n_trainable_parameters": s.get("n_trainable_parameters"),
                "n_condition_parameters": s.get("n_condition_parameters"),
                "predictor_input_dim": s.get("predictor_input_dim"),
                "index_sha256": s.get("index_sha256"),
                "selected_on": "inner_dev_mse_pred",
            }
        if missing_seeds:
            plan["all_complete"] = False
            plan["conditions"][cond] = {"missing_seeds": missing_seeds, "seeds": seeds}
            continue
        raw = [seeds[s]["best_epoch"] + 1 for s in (0, 1, 2)]
        fixed = int(statistics.median(raw))
        expected = int(cfg["epoch_policy"]["conditions"][cond]["fixed_epochs"])
        config_best = cfg["epoch_policy"]["conditions"][cond]["best_epochs"]
        plan["conditions"][cond] = {
            "seeds": seeds,
            "best_epoch_plus_one": raw,
            "fixed_epochs": fixed,
            "config_fixed_epochs": expected,
            "config_best_epochs": config_best,
            "matches_config": fixed == expected,
            "all_best_epochs_match_config": raw == list(config_best),
        }
        if fixed != expected:
            plan["all_complete"] = False

    out = Path(cfg["reports"]["training_plan_json"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(plan, fh, indent=2)

    md = ["# S4B.2 Frozen Training Plan", "",
          f"- Generated: {plan['generated_at']}",
          f"- Rule: {plan['rule']}",
          f"- Source roots: {', '.join(ROOTS)}", "",
          "| condition | seed0 best | seed1 best | seed2 best | +1 values | fixed epochs | matches config |",
          "|---|---:|---:|---:|---|---:|---|"]
    for cond in CONDITIONS:
        c = plan["conditions"][cond]
        if "seeds" not in c:
            md.append(f"| {cond} | MISSING {c['missing_seeds']} | | | | | |")
            continue
        b = [c["seeds"][s]["best_epoch"] for s in (0, 1, 2)]
        md.append(f"| {cond} | {b[0]} | {b[1]} | {b[2]} | {c['best_epoch_plus_one']} | "
                  f"{c['fixed_epochs']} | {c['matches_config']} |")
    md += ["", "Source checkpoint SHA256:", ""]
    for cond in CONDITIONS:
        c = plan["conditions"][cond]
        for s in (0, 1, 2):
            e = c.get("seeds", {}).get(s)
            if e:
                md.append(f"- `{e['checkpoint_best']}` -> `{e['checkpoint_best_sha256']}`")
    md += ["", f"**all_complete = {plan['all_complete']}**"]
    Path(cfg["reports"]["training_plan_md"]).write_text("\n".join(md) + "\n")

    print(json.dumps({"all_complete": plan["all_complete"],
                      "fixed": {c: plan["conditions"][c].get("fixed_epochs")
                                for c in CONDITIONS},
                      "missing": plan["missing"]}, indent=2), flush=True)
    print(f"wrote {out}", flush=True)
    return 0 if plan["all_complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
