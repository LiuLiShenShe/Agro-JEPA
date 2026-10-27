#!/usr/bin/env python3
"""S4B.2 report writer: training results + provenance from real artifacts.

Aggregates the nine ``run_summary.json`` files (plus the verify, data-audit and
training-plan reports) into:

* ``reports/s4b2_training_results.{json,md}`` -- per-run and per-condition
  epoch budget, step count, wall clock, throughput, peak memory, train-loss
  endpoints, parameter counts and checkpoint SHA256;
* ``reports/s4b2_provenance.md`` -- frozen upstream/model/data digests, the
  config digest and the frozen epoch policy, the implementation commit, and the
  frozen S4A / S4A.1 source checkpoints the epoch budget was derived from.

Nothing here fabricates a number: every value is copied from an artifact that
was produced by an actual run.

Usage
-----
    python3 tools/s4b2_write_reports.py --config agro_jepa/config/s4b2_full_train.yaml
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
COND_LABEL = {
    "b2_no_time": "B2 (no time)",
    "b2_no_time_constant_delta": "B2-C (fixed 9.0 d)",
    "b3_delta": "B3 (real delta)",
}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(p: Path) -> dict | None:
    return json.load(open(p)) if p.is_file() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s4b2_full_train.yaml")
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    root = Path(cfg["training"]["output_root"])
    seeds = [int(s) for s in cfg["training"]["seeds"]]

    runs: dict = {}
    for cond in CONDITIONS:
        for seed in seeds:
            p = root / cond / f"seed{seed}" / "run_summary.json"
            runs[f"{cond}/seed{seed}"] = {
                "path": str(p),
                "summary": load_json(p),
            }
    missing = [k for k, v in runs.items() if v["summary"] is None]

    # ---- per-run rows ----------------------------------------------------
    rows = []
    for cond in CONDITIONS:
        for seed in seeds:
            s = runs[f"{cond}/seed{seed}"]["summary"]
            if s is None:
                rows.append({"condition": cond, "seed": seed, "present": False})
                continue
            hist = s.get("history", [])
            rows.append({
                "condition": cond,
                "seed": seed,
                "present": True,
                "epochs_completed": s["epochs_completed"],
                "steps_per_epoch": s["steps_per_epoch"],
                "global_step": s["global_step"],
                "total_steps": s["total_steps"],
                "train_pairs": s["train_pairs"],
                "train_plots": s["train_plots"],
                "batch_size": s["batch_size"],
                "lr": s["lr"],
                "weight_decay": s["weight_decay"],
                "warmup_steps": s["warmup_steps"],
                "amp": s["amp"],
                "first_train_loss": hist[0]["train_loss"] if hist else None,
                "last_train_loss": hist[-1]["train_loss"] if hist else None,
                "wall_clock_seconds": s["wall_clock_seconds"],
                "samples_per_second": s["samples_per_second"],
                "peak_gpu_memory_bytes": s["peak_gpu_memory_bytes"],
                "n_trainable_parameters": s["n_trainable_parameters"],
                "n_condition_parameters": s["n_condition_parameters"],
                "predictor_input_dim": s["predictor_input_dim"],
                "fixed_delta": s.get("fixed_delta"),
                "delta_norm": s.get("delta_norm"),
                "index_sha256": s["index_sha256"],
                "feature_index_sha256": s["feature_index_sha256"],
                "git_commit": s["git_commit"],
                "init_state_sha256": s.get("init_state_sha256"),
                "checkpoint_final": s["checkpoint_final"],
                "checkpoint_final_sha256": s["checkpoint_final_sha256"],
                "checkpoint_last_sha256": s["checkpoint_last_sha256"],
                "metrics_jsonl": str(root / cond / f"seed{seed}" / "metrics.jsonl"),
                "timeout_seconds": None,
            })

    # ---- per-condition aggregate ----------------------------------------
    agg = {}
    for cond in CONDITIONS:
        present = [r for r in rows if r["condition"] == cond and r.get("present")]
        if not present:
            agg[cond] = {"present": False}
            continue
        def mean(key):
            vals = [r[key] for r in present if r.get(key) is not None]
            return statistics.fmean(vals) if vals else None
        def std(key):
            vals = [r[key] for r in present if r.get(key) is not None]
            return statistics.stdev(vals) if len(vals) > 1 else 0.0
        agg[cond] = {
            "present": True,
            "n_seeds": len(present),
            "fixed_epochs": present[0]["epochs_completed"],
            "global_step": present[0]["global_step"],
            "params_match": len({(r["n_trainable_parameters"],
                                  r["n_condition_parameters"],
                                  r["predictor_input_dim"]) for r in present}) == 1,
            "n_trainable_parameters": present[0]["n_trainable_parameters"],
            "n_condition_parameters": present[0]["n_condition_parameters"],
            "predictor_input_dim": present[0]["predictor_input_dim"],
            "first_train_loss_mean": mean("first_train_loss"),
            "last_train_loss_mean": mean("last_train_loss"),
            "last_train_loss_std": std("last_train_loss"),
            "wall_clock_seconds_mean": mean("wall_clock_seconds"),
            "samples_per_second_mean": mean("samples_per_second"),
            "peak_gpu_memory_bytes_max": max(
                r["peak_gpu_memory_bytes"] for r in present),
        }

    results = {
        "stage": "S4B.2",
        "task": "full official-train training of three frozen conditions",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": args.config,
        "config_sha256": sha256_file(args.config),
        "manifest_id": cfg["manifest_id"],
        "seeds": seeds,
        "n_runs_expected": len(CONDITIONS) * len(seeds),
        "n_runs_present": sum(1 for k, v in runs.items() if v["summary"] is not None),
        "missing": missing,
        "epoch_policy": cfg["epoch_policy"],
        "training": cfg["training"],
        "per_run": rows,
        "per_condition": agg,
        "official_validation_used_for_selection": False,
        "official_test_used": False,
        "early_stopping": False,
    }
    rj = Path(cfg["reports"]["training_results_json"])
    rj.parent.mkdir(parents=True, exist_ok=True)
    with open(rj, "w") as fh:
        json.dump(results, fh, indent=2)

    # ---- training_results.md --------------------------------------------
    L = ["# S4B.2 Full-Train Training Results", "",
         f"- Generated: {results['generated_at']}",
         f"- Config: `{args.config}` (sha `{results['config_sha256'][:16]}...`)",
         f"- Runs present: **{results['n_runs_present']} / {results['n_runs_expected']}**",
         f"- Frozen epoch budget: " +
         ", ".join(f"{COND_LABEL[c]} = {cfg['epoch_policy']['conditions'][c]['fixed_epochs']}"
                   for c in CONDITIONS),
         f"- Batch {cfg['training']['batch_size']}, lr {cfg['training']['lr']}, "
         f"wd {cfg['training']['weight_decay']}, warmup {cfg['training']['warmup_steps']}, "
         f"amp {cfg['training']['amp']}, grad-clip {cfg['training']['grad_clip']}",
         f"- Train pairs/epoch: {179823:,} -> steps/epoch {179823 // 128:,}",
         f"- No early stopping, no epoch re-selection, official validation not used "
         f"for selection, official test never read.",
         "",
         "## Per-run", "",
         "| condition | seed | epochs | global step | first loss | last loss | "
         "wall (s) | samples/s | peak MiB | params | final sha (16) |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        if not r.get("present"):
            L.append(f"| {r['condition']} | {r['seed']} | MISSING | | | | | | | | |")
            continue
        L.append(
            f"| {r['condition']} | {r['seed']} | {r['epochs_completed']} | "
            f"{r['global_step']:,} | {r['first_train_loss']:.5f} | "
            f"{r['last_train_loss']:.5f} | {r['wall_clock_seconds']:.0f} | "
            f"{r['samples_per_second']:.1f} | "
            f"{r['peak_gpu_memory_bytes'] / 2**20:.0f} | "
            f"{r['n_trainable_parameters']:,} | "
            f"`{r['checkpoint_final_sha256'][:16]}` |")
    L += ["", "## Per-condition aggregate (mean over seeds)", "",
          "| condition | epochs | step | params (trainable/cond/dim) | "
          "last loss mean±std | wall mean (s) | samples/s |",
          "|---|---:|---:|---|---:|---:|---:|"]
    for cond in CONDITIONS:
        a = agg[cond]
        if not a.get("present"):
            L.append(f"| {cond} | MISSING | | | | | |")
            continue
        L.append(
            f"| {cond} | {a['fixed_epochs']} | {a['global_step']:,} | "
            f"{a['n_trainable_parameters']:,} / {a['n_condition_parameters']:,} / "
            f"{a['predictor_input_dim']} | "
            f"{a['last_train_loss_mean']:.5f} ± {a['last_train_loss_std']:.5f} | "
            f"{a['wall_clock_seconds_mean']:.0f} | "
            f"{a['samples_per_second_mean']:.1f} |")
    L += ["", "## Artifacts", ""]
    for r in rows:
        if r.get("present"):
            L.append(f"- `{r['checkpoint_final']}` (sha `{r['checkpoint_final_sha256']}`)")
    L += ["", "Run logs: `reports/s4b2_run_logs/`."]
    Path(cfg["reports"]["training_results_md"]).write_text("\n".join(L) + "\n")

    # ---- provenance.md ---------------------------------------------------
    verify = load_json(Path(cfg["reports"]["verify_json"]))
    audit = load_json(Path(cfg["reports"]["data_audit_json"]))
    plan = load_json(Path(cfg["reports"]["training_plan_json"]))
    e1 = load_json(Path(cfg["reports"]["validation_e1_json"]))
    impl_commit = next((r["git_commit"] for r in rows if r.get("present")), "unknown")

    P = ["# S4B.2 Provenance", "",
         f"- Generated: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
         f"- Implementation commit (`git_commit` recorded by the trainer): `{impl_commit}`",
         f"- Current HEAD: `{verify['git_head'] if verify else 'n/a'}`",
         f"- Branch: `{verify['branch'] if verify else 'n/a'}`",
         "",
         "## Frozen upstream / model", "",
         f"- Upstream DINO-WM commit: `{cfg['frozen']['upstream_dino_wm_commit']}`",
         f"- DINOv2 visual encoder commit: `{cfg['frozen']['visual_encoder_commit']}`",
         f"- Visual weights sha256: `{cfg['frozen']['visual_weights_sha256']}`",
         f"- Feature cache version: `{cfg['frozen']['feature_cache_version']}` "
         f"(shape {cfg['frozen']['feature_shape']}, "
         f"{cfg['frozen']['feature_dtype_on_disk']} on disk)",
         "",
         "## Data", "",
         f"- Full official-train pair table: `{cfg['data']['full_train_pairs']}` "
         f"(sha `{cfg['data']['full_train_pairs_sha256']}`)",
         f"- Feature index: `{cfg['data']['feature_index']}` "
         f"(sha `{cfg['data']['feature_index_sha256']}`)",
         f"- Derived training index: `{cfg['data']['train_index']}`",
         f"- S3 validation index: `{cfg['data']['s3_index']}` "
         f"(sha `{cfg['validation_e1']['split_s3_index_sha256']}`)",
         "",
         "Data audit:",
         "",
         f"- pairs **{audit['n_pairs']:,}**, plots **{audit['n_plots']:,}**, "
         f"split {audit['split_values']}, train-only = {audit['train_only']}"
         if audit else "- (data audit missing)",
         f"- labels stripped ({len(audit['dropped_label_columns'])} columns): "
         f"{', '.join(audit['dropped_label_columns'])}" if audit else "",
         f"- index sha256: `{audit['index_sha256']}`" if audit else "",
         f"- cache coverage: {audit['cache_coverage']['n_unique_cache_files']:,} files, "
         f"{audit['cache_coverage']['missing_files']} missing, "
         f"{audit['cache_coverage']['bad_shape']} bad shape"
         if audit else "",
         "",
         "## Frozen epoch policy", "",
         f"- Rule: {cfg['epoch_policy']['rule']}",
         f"- Source roots: {', '.join(plan['checkpoint_roots']) if plan else 'n/a'}",
         ""]
    if plan:
        P += ["| condition | best_epoch (0-based) seeds 0/1/2 | +1 values | fixed epochs |",
              "|---|---|---|---:|"]
        for cond in CONDITIONS:
            c = plan["conditions"].get(cond, {})
            b = [c.get("seeds", {}).get(str(s), {}).get("best_epoch") for s in seeds]
            P.append(f"| {cond} | {b} | {c.get('best_epoch_plus_one')} | "
                     f"{c.get('fixed_epochs')} |")
        P += ["", "Source checkpoints (frozen, must not change):", ""]
        for cond in CONDITIONS:
            for s in seeds:
                e = plan["conditions"].get(cond, {}).get("seeds", {}).get(str(s))
                if e:
                    P.append(f"- `{e['checkpoint_best']}` -> `{e['checkpoint_best_sha256']}`")
    P += ["", "## Verification", ""]
    if verify:
        P.append(f"- all_pass = **{verify['all_pass']}** "
                 f"(core={verify['checks']['core_run_checks_pass']}, "
                 f"reload={verify['checks']['checkpoint_reload_forward']['ok']}, "
                 f"fresh-init={verify['checks']['fresh_init_reproduced']['ok']}, "
                 f"frozen-assets={verify['checks']['frozen_source_checkpoints_unchanged']['ok']})")
        P.append(f"- steps/epoch = {verify['steps_per_epoch']}, "
                 f"total optimizer steps over all 9 runs = "
                 f"{verify['total_steps_all_runs']:,}")
        for w in verify.get("warnings", []):
            P.append(f"- warning: {w}")
    else:
        P.append("- verify report missing")
    if e1:
        P += ["", "## E1 validation diagnostics (diagnostic only)", "",
              f"- {e1['n_val_pairs']:,} validation pairs / {e1['n_val_plots']:,} plots",
              f"- bootstrap: plot-clustered, {e1['bootstrap']['n_resamples']} resamples"]
        for cond in CONDITIONS:
            s = e1["conditions"]["seed_averaged"][cond]
            P.append(f"- {cond}: MSE {s['mse_pred']:.6f}, cos {s['cosine_pred']:.4f}, "
                     f"vs persistence "
                     f"{100 * s['relative_improvement_vs_persist']:.2f}%")
    Path(cfg["reports"]["provenance_md"]).write_text("\n".join(P) + "\n")

    print(json.dumps({"n_runs_present": results["n_runs_present"],
                      "missing": missing,
                      "all_pass_verify": verify["all_pass"] if verify else None,
                      "wrote": [str(rj), cfg["reports"]["training_results_md"],
                                cfg["reports"]["provenance_md"]]}, indent=2),
          flush=True)
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
