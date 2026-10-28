#!/usr/bin/env python3
"""S4B.4A training grid driver.

Runs ``tools/s4b4a_train.py`` sequentially over the frozen (fold, condition,
seed) grid and appends one log file per job under ``reports/s4b4a_run_logs``.
Jobs whose ``best.pt`` already exists are skipped, so the driver is resumable.

The pilot is the frozen 1-seed grid (``--seeds 0``); the formal grid passes
``--seeds 0,1,2``. Nothing here changes the protocol: every argument is passed
straight through to the trainer, whose budgets come from the frozen config.

Usage
-----
    CUDA_VISIBLE_DEVICES=1 python3 tools/s4b4a_run_grid.py \
        --config agro_jepa/config/s4b4a_year_heldout.yaml \
        --seeds 0 --conditions all --tag pilot1
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s4b4a_year_heldout.yaml")
    ap.add_argument("--folds", default=None)
    ap.add_argument("--conditions", default="all")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--tag", default="grid")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    folds = ([int(x) for x in args.folds.split(",")] if args.folds
             else [int(y) for y in cfg["folds"]["heldout_year_per_fold"]])
    trained = [k for k, v in cfg["conditions"].items() if v.get("mode") == "trained"]
    conds = trained if args.conditions == "all" else args.conditions.split(",")
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else [int(s) for s in cfg["training"]["seeds"]])
    out_root = Path(cfg["training"]["output_root"])
    log_dir = ROOT / cfg["artifacts"]["run_logs_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(f, c, s) for f in folds for c in conds for s in seeds]
    print(f"[grid] {len(jobs)} jobs: folds={folds} conds={conds} seeds={seeds}",
          flush=True)
    t0 = time.time()
    for i, (fold, cond, seed) in enumerate(jobs, 1):
        ck = out_root / f"fold{fold}" / cond / f"seed{seed}" / "checkpoints" / "best.pt"
        log = log_dir / f"{args.tag}_fold{fold}_{cond}_seed{seed}.log"
        if ck.is_file() and not args.force:
            print(f"[grid] {i}/{len(jobs)} skip (exists) {fold}/{cond}/s{seed}",
                  flush=True)
            continue
        cmd = [sys.executable, str(ROOT / "tools/s4b4a_train.py"),
               "--config", args.config, "--fold", str(fold),
               "--condition", cond, "--seed", str(seed)]
        t = time.time()
        print(f"[grid] {i}/{len(jobs)} start {fold}/{cond}/s{seed}", flush=True)
        with open(log, "w") as fh:
            fh.write(" ".join(cmd) + "\n")
            fh.flush()
            rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                 cwd=str(ROOT),
                                 env={**os.environ})
        print(f"[grid] {i}/{len(jobs)} rc={rc} {fold}/{cond}/s{seed} "
              f"{time.time()-t:.0f}s", flush=True)
        if rc != 0:
            print(f"[grid] FAILED; see {log}", flush=True)
            return rc
    print(f"[grid] all jobs done in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
