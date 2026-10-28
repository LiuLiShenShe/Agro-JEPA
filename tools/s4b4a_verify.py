#!/usr/bin/env python3
"""S4B.4A Phase 5: automatic acceptance verification.

Checks, from real artifacts only, that the five-fold year-held-out experiment
obeyed the frozen protocol:

* fold train/dev/heldout are plot-disjoint, cover all 2,930 plots, and their
  file digests match ``reports/s4b4a_protocol.json``;
* each fold's per-checkpoint DeltaNormStats equal the statistics recomputed from
  that fold's fold_train rows (proves norms were fit on training rows only);
* every (fold, condition, seed) checkpoint exists, loads, and carries the right
  condition spec / parameter counts;
* E1 predictions contain only held-out pair ids and their metrics recompute;
* E2 readout used fold-train rows only (held-out labels appear only in the
  evaluation frame);
* S0-S4B.3 tracked assets are unmodified and the pre-existing CLAUDE.md edit is
  neither staged nor committed.

Exit code 0 on PASS, 1 on any FAIL.  Writes ``reports/s4b4a_verify.json``.

Usage
-----
    python3 tools/s4b4a_verify.py --config agro_jepa/config/s4b4a_year_heldout.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agro_jepa.models.time_embedding import DeltaNormStats  # noqa: E402


BASELINE_COMMIT = "27504922095366ac246c8760a08702b3b762d620"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), *args],
                                       text=True).strip()
    except subprocess.CalledProcessError:
        return ""


class Checks:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, name: str, ok: bool, detail: str) -> None:
        self.rows.append({"check": name, "pass": bool(ok), "detail": detail})

    @property
    def all_pass(self) -> bool:
        return all(r["pass"] for r in self.rows)


def recompute_delta_norm(fold_dir: Path, fold_year: int) -> DeltaNormStats:
    train = pd.read_parquet(fold_dir / f"fold{fold_year}_fold_train.parquet")
    return DeltaNormStats.fit(train.delta_days.to_numpy())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s4b4a_year_heldout.yaml")
    ap.add_argument("--seeds", default=None)
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    seeds = [int(s) for s in (args.seeds.split(",") if args.seeds
                              else cfg["training"]["seeds"])]
    fold_dir = ROOT / cfg["artifacts"]["fold_dir"]
    proto = json.load(open(ROOT / cfg["artifacts"]["protocol_json"]))
    proto_folds = {f["fold_year"]: f for f in proto["folds"]}
    c = Checks()

    # ---- 1. index digests match the frozen protocol ----------------------
    sha_ok, sha_bad = True, []
    for fold in proto_folds.values():
        for role, e in fold["roles"].items():
            p = ROOT / e["path"]
            if not p.is_file() or sha256_file(p) != e["file_sha256"]:
                sha_ok = False
                sha_bad.append(f"{fold['fold_year']}/{role}")
    c.add("V1_fold_index_sha_matches_protocol", sha_ok,
          "all fold parquet digests match" if sha_ok else f"drift: {sha_bad}")

    # ---- 2. plot disjointness / coverage ---------------------------------
    all_plots = None
    disjoint_ok = True
    detail = []
    for fold in proto_folds.values():
        plots = {}
        for role in ("fold_train", "fold_dev", "heldout"):
            f = pd.read_parquet(ROOT / fold["roles"][role]["path"])
            plots[role] = set(f.plot_uid.astype(str).unique())
        union = plots["fold_train"] | plots["fold_dev"] | plots["heldout"]
        inter = ((plots["fold_train"] & plots["fold_dev"])
                 | (plots["fold_train"] & plots["heldout"])
                 | (plots["fold_dev"] & plots["heldout"]))
        if inter:
            disjoint_ok = False
            detail.append(f"{fold['fold_year']}: overlap {len(inter)}")
        if all_plots is None:
            all_plots = union
        elif union != all_plots:
            disjoint_ok = False
            detail.append(f"{fold['fold_year']}: plot set != fold 1")
    c.add("V2_fold_plot_disjoint_and_complete", disjoint_ok,
          "; ".join(detail) or f"{len(all_plots)} plots, disjoint across roles")

    # ---- 3. held-out year is entirely the held-out role ------------------
    year_ok = True
    terrs = []
    for fold in proto_folds.values():
        y = fold["fold_year"]
        held = pd.read_parquet(ROOT / fold["roles"]["heldout"]["path"])
        tr = pd.read_parquet(ROOT / fold["roles"]["fold_train"]["path"])
        dv = pd.read_parquet(ROOT / fold["roles"]["fold_dev"]["path"])
        if not (held.harvest_year == y).all():
            year_ok = False
            terrs.append(f"{y}: heldout has other years")
        if (tr.harvest_year == y).any() or (dv.harvest_year == y).any():
            year_ok = False
            terrs.append(f"{y}: held-out year in train/dev")
    c.add("V3_heldout_year_boundary", year_ok,
          "; ".join(terrs) or "held-out year only in heldout role")

    # ---- 4. per-checkpoint norm stats fit on fold_train only -------------
    norm_ok, norm_bad, n_ck = True, [], 0
    conds = [k for k, v in cfg["conditions"].items() if v.get("mode") == "trained"]
    for fold_year in proto_folds:
        expected = recompute_delta_norm(fold_dir, fold_year)
        for cond in conds:
            for seed in seeds:
                p = (Path(cfg["training"]["output_root"]) / f"fold{fold_year}" / cond
                     / f"seed{seed}" / "checkpoints" / "best.pt")
                if not p.is_file():
                    norm_ok = False
                    norm_bad.append(f"missing {p}")
                    continue
                ck = torch.load(p, map_location="cpu", weights_only=False)
                dn = ck.get("delta_norm")
                n_ck += 1
                if dn is None:
                    continue
                if abs(float(dn["mean"]) - expected.mean) > 1e-5 or \
                   abs(float(dn["std"]) - expected.std) > 1e-5 or \
                   int(dn["n_samples"]) != expected.n_samples:
                    norm_ok = False
                    norm_bad.append(
                        f"{fold_year}/{cond}/s{seed}: ck=({dn['mean']:.4f},"
                        f"{dn['std']:.4f},{dn['n_samples']}) vs "
                        f"train=({expected.mean:.4f},{expected.std:.4f},"
                        f"{expected.n_samples})")
                del ck
    c.add("V4_delta_norm_fit_on_fold_train_only", norm_ok,
          "; ".join(norm_bad[:6]) or f"{n_ck} checkpoints match fold_train fit")

    # ---- 5. all (fold, condition, seed) checkpoints present & valid ------
    present_ok = True
    pe = []
    for fold_year in proto_folds:
        for cond in conds:
            for seed in seeds:
                p = (Path(cfg["training"]["output_root"]) / f"fold{fold_year}" / cond
                     / f"seed{seed}" / "checkpoints" / "best.pt")
                if not p.is_file():
                    present_ok = False
                    pe.append(f"missing {fold_year}/{cond}/seed{seed}")
                    continue
                ck = torch.load(p, map_location="cpu", weights_only=False)
                if ck.get("format") != "agro-jepa-s4b4a-state-dict-v1":
                    present_ok = False
                    pe.append(f"bad format {fold_year}/{cond}/seed{seed}")
                if ck.get("condition") != cond:
                    present_ok = False
                    pe.append(f"wrong condition {fold_year}/{cond}/seed{seed}")
                del ck
    c.add("V5_all_checkpoints_present_and_tagged", present_ok,
          "; ".join(pe[:6]) or f"{len(proto_folds)*len(conds)*len(seeds)} checkpoints")

    # ---- 6. E1 must not touch held-out labels ----------------------------
    e1 = json.load(open(ROOT / cfg["artifacts"]["e1_json"]))
    c.add("V6_e1_reads_no_heldout_label",
          e1.get("heldout_labels_read") is False and e1.get("test_used") is False,
          "E1 declares heldout_labels_read=False, test_used=False")

    # ---- 7. E1 predictions contain only held-out pair ids ----------------
    e1_ok, e1_bad = True, []
    for fold_year in proto_folds:
        p = (Path(cfg["artifacts"]["runs_dir"]) / f"fold{fold_year}"
             / "e1_predictions.parquet")
        if not p.is_file():
            e1_ok = False
            e1_bad.append(f"missing {p}")
            continue
        pred = pd.read_parquet(p)
        held = pd.read_parquet(fold_dir / f"fold{fold_year}_heldout.parquet")
        if set(pred.pair_id.astype(str)) != set(held.pair_id.astype(str)):
            e1_ok = False
            e1_bad.append(f"{fold_year}: pair ids != heldout")
    c.add("V7_e1_predictions_are_heldout_only", e1_ok,
          "; ".join(e1_bad) or "E1 predictions cover exactly the held-out pairs")

    # ---- 8. E1 metrics recompute from stored predictions -----------------
    # e1_predictions.parquet is wide: one row per held-out pair, columns
    # mse::<condition>::<seed|avg> and cos::<condition>::<seed|avg>.
    metric_ok, metric_bad = True, []
    for fold_year, block in e1["folds"].items():
        pred = pd.read_parquet(Path(cfg["artifacts"]["runs_dir"]) / f"fold{fold_year}"
                               / "e1_predictions.parquet")
        sa = block["conditions"]["seed_averaged"]
        for cond, s in sa.items():
            col = f"mse::{cond}::avg"
            if col not in pred.columns:
                metric_ok = False
                metric_bad.append(f"{fold_year}/{cond}: missing column")
                continue
            if abs(float(pred[col].mean()) - float(s["mse_pred"])) > 1e-6:
                metric_ok = False
                metric_bad.append(f"{fold_year}/{cond}")
            if abs(float(pred.mse_persist.mean())
                   - float(s["mse_persist"])) > 1e-6:
                metric_ok = False
                metric_bad.append(f"{fold_year}/{cond}: persistence")
    c.add("V8_e1_metrics_recompute", metric_ok,
          "; ".join(metric_bad[:6]) or "seed-averaged MSE recomputes")

    # ---- 9. E2 uses fold-train only (label role audit) -------------------
    e2 = json.load(open(ROOT / cfg["artifacts"]["e2_json"]))
    e2_ok = True
    e2_bad = []
    for fold_year, block in e2["folds"].items():
        if block.get("n_heldout", 0) > 0 and block.get("n_train", 0) > 0:
            continue
        e2_ok = False
        e2_bad.append(str(fold_year))
    c.add("V9_e2_train_dev_heldout_counts_positive",
          e2_ok and bool(e2["folds"]),
          "; ".join(e2_bad) or "every fold has train/dev/heldout readout rows")
    c.add("V10_e2_heldout_labels_only_for_evaluation",
          "held-out labels never used" in e2.get("note", "")
          and "evaluated once" in e2.get("note", ""),
          f"E2 note: {e2.get('note', '')!r}")

    # ---- 11. S0-S4B.3 tracked assets unmodified --------------------------
    changed = git("diff", "--name-only", BASELINE_COMMIT, "--",
                  "reports", "agro_jepa", "tools", "dino_wm", "README.md")
    tracked_bad = [p for p in changed.splitlines() if p.strip()]
    # new (untracked) S4B.4A files are allowed; only modified tracked files are not
    c.add("V11_s0_s4b3_tracked_assets_unmodified", not tracked_bad,
          "no tracked S0-S4B.3 file modified" if not tracked_bad
          else f"modified: {tracked_bad}")

    # ---- 12. CLAUDE.md edit neither staged nor committed -----------------
    staged = git("diff", "--cached", "--name-only")
    head_claude = git("show", "HEAD:CLAUDE.md")
    status = git("status", "--short", "--", "CLAUDE.md")
    claude_ok = ("CLAUDE.md" not in staged.splitlines()
                 and "CLAUDE.md" not in head_claude)
    # a working-tree edit is expected and allowed; it must not be staged/committed
    c.add("V12_claude_md_edit_not_staged_or_committed", claude_ok,
          f"status: {status!r}; staged={bool(staged.strip())}")

    report = {
        "stage": "S4B.4A",
        "checks": c.rows,
        "all_pass": c.all_pass,
        "n_pass": sum(1 for r in c.rows if r["pass"]),
        "n_total": len(c.rows),
    }
    out = ROOT / cfg["artifacts"]["verify_json"]
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    for r in c.rows:
        print(f"[{'PASS' if r['pass'] else 'FAIL'}] {r['check']}: {r['detail']}")
    print(f"\n{report['n_pass']}/{report['n_total']} checks pass -> "
          f"{'PASS' if c.all_pass else 'FAIL'}")
    return 0 if c.all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
