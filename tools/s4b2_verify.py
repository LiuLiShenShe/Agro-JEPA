#!/usr/bin/env python3
"""S4B.2 acceptance verification over the nine frozen full-train runs.

Reads only real artifacts (run_summary.json, metrics.jsonl, checkpoints, the
derived train index, the data audit and the frozen training plan).  Rebuilds
each predictor from ``final.pt`` and runs one forward pass, re-derives every
digest, and checks the protocol was followed:

* nine runs (3 conditions x seeds 0,1,2) are complete;
* each ran exactly the frozen epoch budget (14 / 13 / 12) for the correct
  number of optimizer steps (fixed epochs x steps_per_epoch), no early stop;
* every recorded loss is finite and the epoch loss decreased;
* checkpoints reload and produce a finite ``[B,256,768]`` prediction;
* parameter counts match the pre-registered matched-control design;
* index / feature-index digests match the frozen S4B.2 config;
* the train index is official-train only, 179,823 pairs / 2,930 plots, holds no
  phenotype label, and no test row was ever read;
* the frozen S4A / S4A.1 source checkpoints are byte-identical to the plan;
* the constant control carries exactly the frozen 9.0-day buffer;
* fresh init: the recorded ``init_state_sha256`` reproduces from
  ``torch.manual_seed(seed)`` with no earlier checkpoint loaded;
* ``CLAUDE.md`` was neither staged nor committed by this stage.

Outputs: reports/s4b2_verify.json

Usage
-----
    python3 tools/s4b2_verify.py --config agro_jepa/config/s4b2_full_train.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
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
from agro_jepa.models.fip_dino_wm import move_attention_masks  # noqa: E402
from agro_jepa.models.s4_predictor import (  # noqa: E402
    S4ModelConfig,
    build_s4_predictor,
    conditioning_spec,
    has_fixed_delta,
)
from agro_jepa.models.time_embedding import DeltaNormStats  # noqa: E402

CONDITIONS = ("b2_no_time", "b2_no_time_constant_delta", "b3_delta")
LABEL_PREFIXES = ("cc", "ph", "whc", "sen")
EXPECTED = {
    "b2_no_time": dict(input_dim=928, trainable=45_898_196, cond=0, epochs=14),
    "b2_no_time_constant_delta": dict(input_dim=1088, trainable=53_882_396,
                                      cond=72_730, epochs=13),
    "b3_delta": dict(input_dim=1088, trainable=53_882_396, cond=72_730, epochs=12),
}
STEPS_PER_EPOCH = 179823 // 128  # 1404


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True, check=False).stdout.strip()


def state_sha256(model) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(np.ascontiguousarray(v.detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def build_model(cfg: dict, condition: str, seed: int, delta_norm, device="cpu"):
    spec = conditioning_spec(cfg["conditions"][condition])
    if spec.delta_mode == "constant":
        from dataclasses import replace
        spec = replace(spec, fixed_delta_days=float(cfg["conditions"][condition]["fixed_delta_days"]))
    model = build_s4_predictor(
        S4ModelConfig.from_dict(cfg["model"]), spec,
        delta_norm=delta_norm if spec.use_delta else None)
    move_attention_masks(model.predictor, device)
    return model, spec


def load_ckpt_exact(ckpt: Path, cfg: dict, device: torch.device):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cs = ck["conditioning_spec"]
    spec = conditioning_spec({
        "use_delta": cs["use_delta"], "delta_mode": cs.get("delta_mode", "sample"),
        "fixed_delta_days": cs.get("fixed_delta_days"), "gdd_mode": cs["gdd_mode"],
        "gdd_base": cs.get("gdd_base", 0.0), "weather_mode": cs["weather_mode"]})
    dn = ck.get("delta_norm")
    delta_norm = DeltaNormStats(mean=dn["mean"], std=dn["std"], min=dn["min"],
                                max=dn["max"], n_samples=dn["n_samples"]) if dn else None
    model = build_s4_predictor(
        S4ModelConfig.from_dict(ck["model_config"]), spec,
        delta_norm=delta_norm if spec.use_delta else None)
    model.load_state_dict(ck["model"])
    model.to(device)
    move_attention_masks(model.predictor, device)
    model.eval()
    return model, spec, ck


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s4b2_full_train.yaml")
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    t_start = time.time()
    root = Path(cfg["training"]["output_root"])
    seeds = [int(s) for s in cfg["training"]["seeds"]]
    checks: dict = {}
    warnings: list[str] = []

    # ---- frozen inputs ---------------------------------------------------
    train_index = pd.read_parquet(cfg["data"]["train_index"])
    idx_sha = index_sha256(train_index)
    feat_sha = sha256_file(cfg["data"]["feature_index"])
    audit = json.load(open(cfg["reports"]["data_audit_json"]))
    plan = json.load(open(cfg["reports"]["training_plan_json"]))

    label_cols = [c for c in train_index.columns
                  if any(c == p or c.startswith(p + "_") for p in LABEL_PREFIXES)]
    checks["train_index"] = {
        "n_pairs": int(len(train_index)),
        "n_plots": int(train_index.plot_uid.nunique()),
        "split_values": sorted(train_index.split.unique().tolist()),
        "train_only": bool((train_index.split == "train").all()),
        "label_columns_present": label_cols,
        "index_sha256": idx_sha,
        "matches_data_audit": idx_sha == audit.get("index_sha256"),
    }
    checks["feature_index"] = {
        "sha256": feat_sha,
        "matches_config": feat_sha == cfg["data"]["feature_index_sha256"],
    }

    # ---- nine runs --------------------------------------------------------
    runs: dict = {}
    all_present = True
    run_checks: dict = {}
    for cond in CONDITIONS:
        expect = EXPECTED[cond]
        for seed in seeds:
            key = f"{cond}/seed{seed}"
            d = root / cond / f"seed{seed}"
            ckpt = d / "checkpoints" / "final.pt"
            last = d / "checkpoints" / "last.pt"
            summ_p = d / "run_summary.json"
            metrics_p = d / "metrics.jsonl"
            present = all(p.is_file() for p in (ckpt, last, summ_p, metrics_p))
            all_present &= present
            entry = {"dir": str(d), "present": present}
            if not present:
                entry["missing"] = [str(p.name) for p in (ckpt, last, summ_p, metrics_p)
                                    if not p.is_file()]
                run_checks[key] = entry
                continue
            s = json.load(open(summ_p))
            epoch_losses, step_losses = [], []
            with open(metrics_p) as fh:
                for line in fh:
                    r = json.loads(line)
                    if r.get("type") == "epoch":
                        epoch_losses.append(float(r["train_loss"]))
                    elif r.get("type") == "step":
                        step_losses.append(float(r["loss"]))
            fin = all(math.isfinite(x) for x in epoch_losses + step_losses) and bool(epoch_losses)
            entry.update({
                "epochs_completed": int(s["epochs_completed"]),
                "epochs_expected": expect["epochs"],
                "epochs_ok": int(s["epochs_completed"]) == expect["epochs"],
                "global_step": int(s["global_step"]),
                "steps_expected": expect["epochs"] * STEPS_PER_EPOCH,
                "steps_ok": int(s["global_step"]) == expect["epochs"] * STEPS_PER_EPOCH,
                "n_epoch_records": len(epoch_losses),
                "first_train_loss": epoch_losses[0] if epoch_losses else None,
                "last_train_loss": epoch_losses[-1] if epoch_losses else None,
                "loss_decreased": bool(epoch_losses and epoch_losses[-1] < epoch_losses[0]),
                "losses_finite": fin,
                "early_stopping": bool(s.get("early_stopping")),
                "official_validation_used_for_selection":
                    bool(s.get("official_validation_used_for_selection")),
                "train_pairs": int(s["train_pairs"]),
                "train_plots": int(s["train_plots"]),
                "n_trainable_parameters": int(s["n_trainable_parameters"]),
                "n_condition_parameters": int(s["n_condition_parameters"]),
                "predictor_input_dim": int(s["predictor_input_dim"]),
                "params_ok": (int(s["n_trainable_parameters"]) == expect["trainable"]
                              and int(s["n_condition_parameters"]) == expect["cond"]
                              and int(s["predictor_input_dim"]) == expect["input_dim"]),
                "index_sha256": s["index_sha256"],
                "index_ok": s["index_sha256"] == audit.get("index_sha256"),
                "feature_index_ok": s["feature_index_sha256"] == cfg["data"]["feature_index_sha256"],
                "conditioning_spec": s["conditioning_spec"],
                "fixed_delta": s.get("fixed_delta"),
                "final_sha": sha256_file(ckpt),
                "final_sha_ok": sha256_file(ckpt) == s.get("checkpoint_final_sha256"),
                "last_sha_ok": sha256_file(last) == s.get("checkpoint_last_sha256"),
                "init_state_sha256": s.get("init_state_sha256"),
                "git_commit": s.get("git_commit"),
                "wall_clock_seconds": s.get("wall_clock_seconds"),
                "peak_gpu_memory_bytes": s.get("peak_gpu_memory_bytes"),
                "samples_per_second": s.get("samples_per_second"),
                "delta_norm": s.get("delta_norm"),
            })
            runs[key] = {"path": str(d), "summary": s}
            run_checks[key] = entry

    checks["runs"] = {"all_present": all_present, "conditions": run_checks}
    checks["all_runs_present"] = all_present

    # core run checks aggregate
    core_ok = all_present
    for key, e in run_checks.items():
        if not e.get("present"):
            continue
        core_ok &= bool(e["epochs_ok"] and e["steps_ok"] and e["losses_finite"]
                        and e["loss_decreased"] and not e["early_stopping"]
                        and not e["official_validation_used_for_selection"]
                        and e["params_ok"] and e["index_ok"] and e["feature_index_ok"]
                        and e["final_sha_ok"] and e["last_sha_ok"]
                        and e["train_pairs"] == 179823 and e["train_plots"] == 2930)
    checks["core_run_checks_pass"] = core_ok

    # ---- delta normaliser shared + constant control ----------------------
    delta_norms = {k: v["summary"].get("delta_norm") for k, v in runs.items()}
    dn_vals = list(delta_norms.values())
    dn_shared = all(v is not None and v == dn_vals[0] for v in dn_vals)
    checks["delta_norm_shared_across_runs"] = {
        "ok": dn_shared, "value": dn_vals[0] if dn_vals else None}
    b2c_fixed = [v["summary"].get("fixed_delta") for k, v in runs.items()
                 if k.startswith("b2_no_time_constant_delta")]
    fixed_ok = all(f is not None and float(f.get("fixed_delta_days", -1)) == 9.0
                   for f in b2c_fixed)
    checks["b2c_fixed_delta_9_0"] = {
        "ok": fixed_ok and len(b2c_fixed) == len(seeds),
        "records": b2c_fixed[:1]}

    # ---- checkpoint reload + forward -------------------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reload_ok = True
    reloads: dict = {}
    # a real validation feature for the forward smoke
    s3 = pd.read_parquet(cfg["validation_e1"]["s3_index"])
    val = s3[s3.split == cfg["validation_e1"]["split"]]
    z_a_cpu = torch.from_numpy(np.load(val.cache_file_a.iloc[0]).astype(np.float32))
    for cond in CONDITIONS:
        for seed in seeds:
            key = f"{cond}/seed{seed}"
            if key not in runs:
                reload_ok = False
                reloads[key] = {"ok": False, "reason": "run missing"}
                continue
            ckpt = Path(runs[key]["path"]) / "checkpoints" / "final.pt"
            try:
                model, spec, ck = load_ckpt_exact(ckpt, cfg, device)
                z = z_a_cpu.unsqueeze(0).to(device)
                with torch.no_grad():
                    if spec.delta_mode == "sample":
                        out = model(z, delta_days=torch.tensor(
                            [float(val.delta_days.iloc[0])], device=device))
                    else:
                        out = model(z)
                finite = bool(torch.isfinite(out).all())
                shape_ok = tuple(out.shape) == (1, 256, 768)
                reloads[key] = {"ok": finite and shape_ok, "shape": list(out.shape),
                                "finite": finite}
                reload_ok &= finite and shape_ok
                del model, out
                torch.cuda.empty_cache()
            except Exception as e:  # noqa: BLE001
                reload_ok = False
                reloads[key] = {"ok": False, "error": repr(e)}
    checks["checkpoint_reload_forward"] = {"ok": reload_ok, "per_run": reloads}

    # ---- fresh-init reproducibility (advisory) ---------------------------
    init_rep = {}
    init_ok = True
    for cond in CONDITIONS:
        for seed in seeds:
            key = f"{cond}/seed{seed}"
            if key not in runs:
                continue
            recorded = runs[key]["summary"].get("init_state_sha256")
            dn = runs[key]["summary"].get("delta_norm")
            dn_obj = DeltaNormStats(mean=dn["mean"], std=dn["std"], min=dn["min"],
                                    max=dn["max"], n_samples=dn["n_samples"]) if dn else None
            torch.manual_seed(int(seed))
            model, _ = build_model(cfg, cond, seed, dn_obj, device="cpu")
            recomputed = state_sha256(model)
            del model
            match = recomputed == recorded
            init_rep[key] = {"recorded": recorded, "recomputed": recomputed,
                             "match": match}
            init_ok &= match
    checks["fresh_init_reproduced"] = {"ok": init_ok, "per_run": init_rep}
    if not init_ok:
        warnings.append("fresh-init digest did not reproduce for at least one run; "
                        "the runs may still be fresh, but this is not proven here")

    # ---- frozen source checkpoints unchanged -----------------------------
    frozen_ok = True
    frozen_records = {}
    for cond, block in plan.get("conditions", {}).items():
        for s, e in (block.get("seeds") or {}).items():
            p = Path(e["checkpoint_best"])
            rec = e.get("checkpoint_best_sha256")
            now = sha256_file(p) if p.is_file() else None
            ok = now is not None and now == rec
            frozen_records[f"{cond}/seed{s}"] = {"path": str(p), "expected": rec,
                                                 "now": now, "ok": ok}
            frozen_ok &= ok
    checks["frozen_source_checkpoints_unchanged"] = {
        "ok": frozen_ok, "per_source": frozen_records}

    # ---- CLAUDE.md untouched by this stage -------------------------------
    staged = git("diff", "--cached", "--name-only").splitlines()
    impl_commit = runs.get("b3_delta/seed0", {}).get("summary", {}).get("git_commit")
    commit_files = (git("show", "--name-only", "--pretty=format:", impl_commit).splitlines()
                    if impl_commit else [])
    md_staged = "CLAUDE.md" in staged
    md_in_commit = "CLAUDE.md" in commit_files
    checks["claude_md_untouched"] = {
        "ok": not md_staged and not md_in_commit,
        "staged": md_staged,
        "in_implementation_commit": md_in_commit,
        "implementation_commit": impl_commit,
        "worktree_status": git("status", "--porcelain", "CLAUDE.md"),
    }
    if checks["claude_md_untouched"]["worktree_status"]:
        warnings.append("CLAUDE.md has a pre-existing worktree modification; it was "
                        "not staged or committed by this stage")

    # ---- aggregate -------------------------------------------------------
    hard_keys = ("core_run_checks_pass", "all_runs_present")
    all_pass = (all(checks[k] for k in hard_keys)
                and checks["train_index"]["train_only"]
                and checks["train_index"]["matches_data_audit"]
                and not checks["train_index"]["label_columns_present"]
                and checks["feature_index"]["matches_config"]
                and checks["delta_norm_shared_across_runs"]["ok"]
                and checks["b2c_fixed_delta_9_0"]["ok"]
                and checks["checkpoint_reload_forward"]["ok"]
                and checks["fresh_init_reproduced"]["ok"]
                and checks["frozen_source_checkpoints_unchanged"]["ok"]
                and checks["claude_md_untouched"]["ok"])

    report = {
        "stage": "S4B.2",
        "task": "acceptance verification over the nine frozen full-train runs",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "config": args.config,
        "config_sha256": sha256_file(args.config),
        "expected": EXPECTED,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "total_steps_all_runs": sum(v["epochs"] * STEPS_PER_EPOCH
                                    for v in EXPECTED.values()) * len(seeds),
        "checks": checks,
        "warnings": warnings,
        "all_pass": bool(all_pass),
        "elapsed_seconds": round(time.time() - t_start, 1),
    }
    out = Path(cfg["reports"]["verify_json"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)

    print(json.dumps({"all_pass": all_pass,
                      "core_run_checks_pass": checks["core_run_checks_pass"],
                      "all_runs_present": checks["all_runs_present"],
                      "checkpoint_reload_forward": checks["checkpoint_reload_forward"]["ok"],
                      "fresh_init_reproduced": checks["fresh_init_reproduced"]["ok"],
                      "b2c_fixed_9_0": checks["b2c_fixed_delta_9_0"]["ok"],
                      "frozen_assets_unchanged": checks["frozen_source_checkpoints_unchanged"]["ok"],
                      "warnings": warnings}, indent=2), flush=True)
    print(f"wrote {out}", flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
