#!/usr/bin/env python3
"""Run every S4A trainable condition under one fixed protocol.

This is a thin, sequential launcher around ``agro_jepa/train/train_s4.py``: it
reproduces exactly the command that would be typed by hand, writes a log file
per (condition, seed) under ``reports/s4_run_logs`` and refuses to run a
baseline condition (those have no training at all).

Usage
-----
    CUDA_VISIBLE_DEVICES=1 python3 tools/s4_train_pilot.py \
        --config agro_jepa/config/s4_pilot.yaml
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s4_pilot.yaml")
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these condition names")
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    tcfg = cfg["training"]
    seeds = args.seeds if args.seeds else list(tcfg["seeds"])
    log_dir = ROOT / cfg["artifacts"]["run_logs"]
    log_dir.mkdir(parents=True, exist_ok=True)
    out_root = Path(tcfg["output_root"])

    plan = []
    for name, cond in cfg["conditions"].items():
        if cond.get("mode") != "trained":
            continue
        if args.only and name not in args.only:
            continue
        for seed in seeds:
            plan.append((name, int(seed)))
    print(f"planned runs: {len(plan)}", flush=True)
    for name, seed in plan:
        print(f"  {name} seed{seed}", flush=True)
    if args.dry_run:
        return 0

    index = 0
    for name, seed in plan:
        index += 1
        out_dir = out_root / name / f"seed{seed}"
        log_path = log_dir / f"train_{name}_seed{seed}.log"
        cmd = [
            sys.executable, "agro_jepa/train/train_s4.py",
            "--config", args.config,
            "--condition", name,
            "--seed", str(seed),
            "--output-dir", str(out_dir),
            "--report", str(out_dir / "run_summary.json"),
        ]
        if args.epochs:
            cmd += ["--epochs", str(args.epochs)]
        print(f"[{index}/{len(plan)}] {' '.join(cmd)}", flush=True)
        t0 = time.time()
        with open(log_path, "w") as fh:
            rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT)
        print(f"[{index}/{len(plan)}] {name} seed{seed} -> rc={rc} "
              f"({round(time.time() - t0, 1)} s) log={log_path}", flush=True)
        if rc != 0:
            print(f"FAILED: {name} seed{seed}; see {log_path}", flush=True)
            with open(log_dir / "failed_runs.json", "w") as fh:
                json.dump({"failed": [{"condition": name, "seed": seed, "rc": rc}],
                           "at_index": index}, fh, indent=2)
            return 1
    print("all planned runs finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
