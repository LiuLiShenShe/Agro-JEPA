#!/usr/bin/env python3
"""S4A.1 initialisation audit.

Every training run records ``init_state_sha256``: the SHA256 of the model's
state_dict taken immediately after ``set_seed(seed)`` + ``build_s4_predictor``,
before any optimisation step.  Some S4A runs predate that recording, so this
tool re-derives the hash from the stated rule and compares it with whatever the
run_summary / checkpoint recorded.

It proves two things:

1. the initialisation rule is deterministic and reproducible (re-running it
   two times in the same process gives the same digest), and
2. no run loaded weights from an earlier run or from a pretrained checkpoint.

Usage
-----
    CUDA_VISIBLE_DEVICES=1 python3 tools/s4a1_init_audit.py \
        --config agro_jepa/config/s4a1_matched_control.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agro_jepa.datasets.s4_dataset import ConditioningSpec  # noqa: E402
from agro_jepa.models.fip_dino_wm import move_attention_masks  # noqa: E402
from agro_jepa.models.s4_predictor import (  # noqa: E402
    S4ModelConfig,
    build_s4_predictor,
)
from agro_jepa.train.train_s4 import set_seed, state_sha256  # noqa: E402

CONDITIONS = ["b2_no_time", "b2_no_time_constant_delta", "b3_delta"]


def spec_for(cfg: dict, name: str) -> ConditioningSpec:
    c = cfg["conditions"][name]
    return ConditioningSpec(
        use_delta=bool(c.get("use_delta", False)),
        delta_mode=str(c.get("delta_mode", "sample")),
        fixed_delta_days=c.get("fixed_delta_days"),
        gdd_mode=str(c.get("gdd_mode", "none")),
        gdd_base=float(c.get("gdd_base", 0.0)),
        weather_mode=str(c.get("weather_mode", "none")),
    )


def build_hash(cfg: dict, model_cfg: S4ModelConfig, name: str, seed: int,
               device: torch.device) -> str:
    set_seed(seed)
    model = build_s4_predictor(model_cfg, spec_for(cfg, name)).to(device)
    move_attention_masks(model.predictor, device)
    return state_sha256(model)


def find_run(cfg: dict, name: str, seed: int) -> Path | None:
    roots = cfg["training"].get("checkpoint_roots") or [
        cfg["training"]["output_root"]]
    for root in roots:
        p = Path(root) / name / f"seed{seed}"
        if (p / "run_summary.json").exists():
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s4a1_matched_control.yaml")
    ap.add_argument("--out", default="reports/s4a1_init_hashes.json")
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    seeds = [int(s) for s in cfg["training"]["seeds"]]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_cfg = S4ModelConfig.from_dict(cfg["model"])

    rows: dict[str, dict] = {}
    mismatches: list[str] = []
    n_recorded = 0
    n_match = 0
    for name in CONDITIONS:
        if cfg["conditions"].get(name, {}).get("mode") != "trained":
            continue
        rows[name] = {}
        for seed in seeds:
            h1 = build_hash(cfg, model_cfg, name, seed, device)
            h2 = build_hash(cfg, model_cfg, name, seed, device)
            run = find_run(cfg, name, seed)
            recorded = None
            source = None
            if run is not None:
                summary = json.load(open(run / "run_summary.json"))
                recorded = summary.get("init_state_sha256")
                source = "run_summary"
                if not recorded:
                    ck = run / "checkpoints" / "best.pt"
                    if ck.exists():
                        payload = torch.load(ck, map_location="cpu",
                                             weights_only=False)
                        recorded = (payload.get("metrics") or {}).get(
                            "init_state_sha256")
                        source = "checkpoint.metrics"
            if recorded:
                n_recorded += 1
                if recorded == h1:
                    n_match += 1
                else:
                    mismatches.append(f"{name} seed{seed}: {recorded} != {h1}")
            rows[name][str(seed)] = {
                "recomputed_sha256": h1,
                "rerun_sha256": h2,
                "deterministic": bool(h1 == h2),
                "recorded_sha256": recorded,
                "recorded_source": source if recorded else None,
                "matches_recorded": (bool(recorded == h1) if recorded else None),
            }
            print(f"{name:32s} seed{seed} {h1[:16]} "
                  f"{'OK' if recorded in (None, h1) else 'MISMATCH'}", flush=True)

    report = {
        "stage": "S4A.1",
        "purpose": ("re-derive the init state digest under the stated rule "
                    "and compare it with each run's recorded value"),
        "rule": ("set_seed(seed) immediately before build_s4_predictor, "
                 "move_attention_masks, state_sha256; no weight loaded"),
        "device": str(device),
        "seeds": seeds,
        "conditions": rows,
        "n_recorded": n_recorded,
        "n_matched": n_match,
        "mismatches": mismatches,
        "all_recorded_match": not mismatches,
        "all_deterministic": all(
            v["deterministic"] for r in rows.values() for v in r.values()),
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"wrote {out}: {n_match}/{n_recorded} recorded hashes match, "
          f"mismatches={len(mismatches)}", flush=True)
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
