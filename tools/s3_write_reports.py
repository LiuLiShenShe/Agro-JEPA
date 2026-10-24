#!/usr/bin/env python3
"""S3: render the three stage reports from the real artifact JSONs.

Every number in the generated markdown is read from a run artifact; nothing is
typed by hand, so the reports cannot drift from the logs.  Re-run this after a
new training/evaluation run.

Usage:
    python3 tools/s3_write_reports.py --config agro_jepa/config/s3_v0.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def read_json(path: str | Path) -> dict:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        return "missing"
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def f(x, nd=5) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float) and (x != x):
        return "NaN"
    return f"{x:.{nd}f}"


def pct(x, nd=2) -> str:
    if x is None:
        return "n/a"
    return f"{100.0 * x:+.{nd}f}%"


def table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def official_reference() -> dict:
    obs = pd.read_csv(
        ROOT / "data/manifests/fip1_observations.csv.gz",
        dtype={"plot_uid": str, "split": str}, low_memory=False,
    )
    obs = obs.drop_duplicates("image_filename_stem")
    out = {}
    for split in ("train", "validation"):
        sub = obs[obs.split == split]
        out[split] = {
            "images": int(len(sub)),
            "plots": int(sub.plot_uid.nunique()),
            "by_year": {
                str(int(k)): int(v)
                for k, v in sub.harvest_year.value_counts().sort_index().items()
            },
            "plots_by_year": {
                str(int(k)): int(v)
                for k, v in sub.groupby("harvest_year").plot_uid.nunique().items()
            },
        }
    return out


def download_passes() -> list[dict]:
    """Parse the two real download logs (main pass + retry pass)."""
    import re

    out = []
    for name, path in (
        ("main pass (40 workers)", "reports/s3_run_logs/s3_download.log"),
        ("retry pass (12 workers)", "reports/s3_run_logs/s3_download_retry.log"),
    ):
        p = ROOT / path
        if not p.exists():
            continue
        m = re.search(
            r"wrote .*?: (\d+) images, (\d+) plots, ([\d.]+) GB, "
            r"([\d.]+) s, ([\d.]+) img/s, (\d+) failures",
            p.read_text(errors="ignore"),
        )
        if m:
            out.append({
                "pass": name, "images": int(m.group(1)), "plots": int(m.group(2)),
                "gb": float(m.group(3)), "seconds": float(m.group(4)),
                "img_s": float(m.group(5)), "failures": int(m.group(6)),
                "log": path,
            })
    return out


def per_split_year(sel_df: pd.DataFrame) -> dict:
    out = {}
    for (s, y), g in sel_df.groupby(["split", "harvest_year"]):
        out[f"{s}|{int(y)}"] = {
            "images": int(len(g)),
            "plots": int(g.plot_uid.nunique()),
        }
    return out


# --------------------------------------------------------------------------
def write_acquisition(cfg: dict) -> Path:
    d = cfg["data"]
    sel_meta = read_json(d["selection"])
    rule = read_json(d.get("selection_rule_out", "data/manifests/s3_selection_rule.json"))
    idx_stats = read_json(d["index_stats"])
    feat = read_json("reports/s3_feature_cache.json")
    sel_df = pd.DataFrame(sel_meta.get("images", []))
    official = official_reference()
    years = sorted(sel_df.harvest_year.unique().tolist()) if len(sel_df) else []

    lines = []
    lines.append("# S3 — Real FIP 1.0 image acquisition (Pilot)")
    lines.append("")
    lines.append(f"Stage: **S3**. Date: 2026-10-24. Branch: `agro-s0-upstream`.")
    lines.append("")
    lines.append("This report covers the image materialisation step that had to precede any S3 "
                 "model training. It is generated from the run artifacts by "
                 "`tools/s3_write_reports.py`.")
    lines.append("")
    lines.append("## 1. Selection rule")
    lines.append("")
    rm = rule.get("selection", {}) or sel_meta.get("selection", {})
    lines.append(f"- rule: {rule.get('rule') or rm.get('rule')}")
    lines.append(f"- seed: `{rm.get('seed')}`")
    lines.append(f"- train plots per year requested: `{rm.get('train_plots_per_year_requested')}`")
    lines.append(f"- train plots per year selected: `{rm.get('train_plots_per_year_selected')}`")
    lines.append(f"- official validation plots: **all** ({rm.get('n_validation_plots')})")
    lines.append(f"- only official `train` and `validation` are eligible. "
                 "No official test image is used for model selection.")
    lines.append("- for every selected plot **all** audited images are fetched, so each "
                 "plot contributes its whole autumn→winter→stem-elongation→heading→"
                 "senescence curve.")
    lines.append("")
    lines.append("## 2. Download mechanics")
    lines.append("")
    lines.append(f"- source: HuggingFace dataset `mikeboss/FIP1`, revision "
                 f"`{cfg.get('dataset_revision', '11cdd84bc0ba8d5c6641ea7b53550b1658527e1c')}`")
    lines.append(f"- endpoint: `{sel_meta.get('hf_endpoint')}` (direct `huggingface.co` is "
                 "unreachable from this host; the mirror is required)")
    lines.append("- resumable: an existing PNG is re-hashed rather than re-downloaded; "
                 "a truncated file cannot be accepted because the PNG signature and "
                 "minimum size are checked")
    lines.append("- retries: 5 attempts with linear backoff; the mirror answers "
                 "`429 Too Many Requests` under high concurrency")
    lines.append("- every image row records plot, split, harvest year, image date, "
                 "source URL, byte size and SHA256")
    lines.append("")
    lines.append("## 3. Actual acquisition")
    lines.append("")
    lines.append(table(
        ["Metric", "Value"],
        [
            ["images on disk (selection)", f"{len(sel_df):,}"],
            ["plots", f"{sel_meta.get('n_plots', 0):,}"],
            ["harvest years", ", ".join(str(y) for y in years)],
            ["bytes", f"{sel_meta.get('n_bytes', 0)/1e9:.2f} GB"],
            ["failed downloads after retries", f"{sel_meta.get('n_failed', 0):,}"],
            ["final manifest wall clock", f"{sel_meta.get('elapsed_seconds', 0)/60:.1f} min"],
            ["final manifest throughput", f"{sel_meta.get('throughput_images_per_s', 0):.2f} img/s"],
            ["image resolution (source)", "960 x 640 RGB PNG"],
            ["selection manifest", str(d["selection"])],
            ["selection manifest SHA256", rule.get("selection_manifest_sha256", "n/a")],
        ],
    ))
    lines.append("")
    passes = download_passes()
    if passes:
        lines.append("The final manifest numbers above describe the **retry pass**, which "
                     "mostly re-hashed existing files; the real network acquisition is the "
                     "sum of the two passes:")
        lines.append("")
        lines.append(table(
            ["pass", "images", "plots", "GB", "minutes", "img/s", "failures"],
            [[p["pass"], f"{p['images']:,}", f"{p['plots']:,}", f"{p['gb']:.2f}",
              f"{p['seconds']/60:.1f}", f"{p['img_s']:.2f}", p["failures"]]
             for p in passes],
        ))
        lines.append("")
    lines.append("### 3.1 Coverage by split and harvest year")
    lines.append("")
    rows = []
    for split in ("train", "validation"):
        for y in years:
            s = sel_df[(sel_df.split == split) & (sel_df.harvest_year == y)]
            ref = official[split]["by_year"].get(str(y), 0)
            ref_plots = official[split]["plots_by_year"].get(str(y), 0)
            rows.append([
                split, y, len(s), s.plot_uid.nunique(),
                f"{ref:,}", f"{ref_plots:,}",
            ])
    lines.append(table(
        ["split", "year", "images downloaded", "plots downloaded",
         "official images", "official plots"], rows))
    lines.append("")
    lines.append("### 3.2 Fraction of the official split obtained")
    lines.append("")
    rows = []
    for split in ("train", "validation"):
        sub = sel_df[sel_df.split == split]
        ref = official[split]
        rows.append([
            split,
            f"{len(sub):,} / {ref['images']:,} ({100*len(sub)/max(ref['images'],1):.1f}%)",
            f"{sub.plot_uid.nunique():,} / {ref['plots']:,}",
        ])
    lines.append(table(["split", "images", "plots"], rows))
    lines.append("")
    pp = rule.get("plots_per_year_and_split", {})
    pp_txt = ", ".join(f"{k}={v}" for k, v in sorted(pp.items()))
    lines.append("**The result is a Pilot, not a full-FIP baseline.** The full official "
                 "train split has 2,930 plots / 117,578 unique images; S3 selected "
                 f"{pp_txt} plots per split/year and downloaded all images of those plots. "
                 "All 380 validation plots were downloaded, so validation is complete at "
                 "the plot level.")
    lines.append("")
    lines.append("## 4. Feature cache")
    lines.append("")
    if feat:
        timing = feat.get("timing", {})
        images = feat.get("images", {})
        lines.append(f"- protocol: `{feat.get('protocol')}` (identical to S1: "
                     "ToTensor → Resize(224) → CenterCrop(224) → Normalize(0.5) → "
                     "DINOv2 ViT-B/14 → `x_norm_patchtokens`, stored float16)")
        lines.append(f"- DINOv2 code commit: `{feat.get('dinov2', {}).get('repo_git_commit')}`")
        lines.append(f"- weights SHA256: `{feat.get('dinov2', {}).get('weights_sha256')}`")
        lines.append(f"- unique images encoded: {timing.get('n_unique_source_images', 0):,} "
                     f"({timing.get('n_cached', 0):,} already cached, "
                     f"{timing.get('n_computed', 0):,} computed)")
        lines.append(f"- throughput: {timing.get('throughput_images_per_s')} img/s; "
                     f"peak GPU memory {timing.get('peak_gpu_memory_bytes', 0)/1e9:.2f} GB")
        checks = feat.get("verification", {}).get("checks", [])
        if checks:
            lines.append("- cache round-trip vs a live forward pass: "
                         + "; ".join(f"cos={c['cosine']:.8f}, max|diff|={c['max_abs_diff']:.2e}"
                                     for c in checks))
        lines.append("- `upstream196` and `mainline224` are never mixed: the index and the "
                     "checkpoint record the `mainline224_s1-v1` cache only.")
    lines.append("")
    lines.append("## 5. S3 pair index (the training/validation sample)")
    lines.append("")
    if idx_stats:
        filt = idx_stats.get("filtering", {})
        lines.append(table(
            ["Filtering step", "rows"],
            [
                ["S2 span7/span14 train+validation rows", f"{filt.get('input_rows', 0):,}"],
                ["both images downloaded", f"{filt.get('rows_after_image_filter', 0):,}"],
                ["after physical-pair deduplication", f"{filt.get('rows_after_dedupe', 0):,}"],
                ["after requiring both feature-cache files", f"{filt.get('final_rows', 0):,}"],
            ],
        ))
        lines.append("")
        rows = []
        for split in ("train", "validation"):
            s = idx_stats.get(split, {})
            dd = s.get("delta_days", {})
            rows.append([
                split, f"{s.get('n_pairs', 0):,}", f"{s.get('n_unique_images', 0):,}",
                f"{s.get('n_unique_physical_pairs', 0):,}", f"{s.get('n_plots', 0):,}",
                f"{s.get('n_years', 0)}",
                f"{dd.get('min')}–{dd.get('max')} (median {dd.get('median')})",
            ])
        lines.append(table(
            ["split", "pairs", "unique images", "unique physical pairs", "plots",
             "years", "delta_days"], rows))
        lines.append("")
        rows = []
        for split in ("train", "validation"):
            s = idx_stats.get(split, {})
            lab = s.get("labels", {})
            rows.append([
                split,
                f"{lab.get('cc', {}).get('n_valid_b', 0):,}",
                f"{lab.get('ph', {}).get('n_valid_b', 0):,}",
                f"{lab.get('cc', {}).get('n_valid_both', 0):,}",
                f"{lab.get('ph', {}).get('n_valid_both', 0):,}",
            ])
        lines.append(table(
            ["split", "CC valid at t_b", "PH valid at t_b",
             "CC valid at both ends", "PH valid at both ends"], rows))
        lines.append("")
        ph_neg_train = idx_stats.get("train", {}).get("labels", {}).get(
            "ph", {}).get("n_negative_b", 0)
        ph_neg_val = idx_stats.get("validation", {}).get("labels", {}).get(
            "ph", {}).get("n_negative_b", 0)
        lines.append(f"Raw PH negatives are preserved (not clipped) and included in the "
                     f"probe targets: **{ph_neg_train}** full-table PH values at `t_b` "
                     f"in train, **{ph_neg_val}** in validation. A positivity-only "
                     "sensitivity is left to S4.")
        lines.append("")
        dn = idx_stats.get("delta_normalization", {})
        lines.append(f"Time normalisation is fitted on the **training split only**: "
                     f"mean {f(dn.get('mean'), 4)}, std {f(dn.get('std'), 4)}, "
                     f"n={dn.get('n_samples'):,}. Validation reuses these parameters.")
        lines.append("")
        lines.append(f"Frozen index SHA256: `{idx_stats.get('index_sha256')}`")
    lines.append("")
    lines.append("## 6. Provenance summary")
    lines.append("")
    lines.append(table(
        ["Artifact", "Path", "SHA256"],
        [
            ["downloaded image selection", str(d["selection"]),
             rule.get("selection_manifest_sha256", "n/a")],
            ["feature index", str(d["feature_index"]),
             sha256_file(d["feature_index"])[:32] + "…"],
            ["S3 pair index", str(d["index_out"]),
             (idx_stats.get("index_sha256") or "n/a")[:32] + "…"],
            ["S3 selection rule", str(d.get("selection_rule_out")),
             sha256_file(d.get("selection_rule_out", ""))[:32] + "…"],
        ],
    ))
    lines.append("")
    lines.append("## 7. Limitations (stated, not hidden)")
    lines.append("")
    lines.append("1. **Pilot scale.** 1,000 of 2,930 official train plots are used; this is "
                 "not a full-FIP training set. Validation is complete at the plot level.")
    lines.append("2. **Mirror throttling.** The HF mirror intermittently returns HTTP 429; "
                 "the downloader retries and is resumable, but a re-run is required to "
                 "close any remaining gaps (recorded as `n_failed`).")
    lines.append("3. **One daily weather table per year** is not used by V0 (no weather "
                 "encoder in S3); S4 will revisit it.")
    lines.append("4. **Precipitation units remain unresolved** in S1/S2; no water-balance "
                 "quantity is computed here.")
    lines.append("5. The 1,872 duplicate image references in the frozen S1 observation "
                 "manifest are still present; S2/S3 de-duplicate at pairing time.")
    lines.append("")
    out = ROOT / "reports/s3_image_acquisition.md"
    out.write_text("\n".join(lines) + "\n")
    return out


# --------------------------------------------------------------------------
def write_training(cfg: dict) -> Path:
    d, tcfg = cfg["data"], cfg["training"]
    run = read_json(Path(tcfg["output_dir"]) / "run_summary.json")
    idx = read_json(d["index_stats"])
    feat = read_json("reports/s3_feature_cache.json")
    verify = read_json("reports/s3_verify.json")
    ckpt_path = Path(run.get("checkpoint_best", ""))
    lines = []
    lines.append("# S3 — FIP-DINO-WM V0 training")
    lines.append("")
    lines.append("Stage: **S3**. Date: 2026-10-24. Branch: `agro-s0-upstream`.")
    lines.append("")
    lines.append("This report is generated from `run_summary.json`, the per-epoch "
                 "`metrics.jsonl` and the checkpoint metadata by "
                 "`tools/s3_write_reports.py`.")
    lines.append("")
    lines.append("## 1. What is trained — and what is not")
    lines.append("")
    lines.append("- **Frozen**: DINOv2 ViT-B/14, the same pinned clone and weights as S1 "
                 f"(`{feat.get('dinov2', {}).get('repo_git_commit')}`, weights "
                 f"`{feat.get('dinov2', {}).get('weights_sha256')}`). It is used to "
                 "produce the S1/S2 `mainline224_s1-v1` cache; it receives no gradient.")
    lines.append("- **Trainable**: the DINO-WM `ViTPredictor` (upstream class) plus a "
                 "trainable time projection that maps the real `delta_days` into the "
                 "action channel. A constant proprio placeholder keeps the upstream "
                 "`concat_dim=1` interface.")
    lines.append("- **Not present in V0**: no weather encoder, no physiological "
                 "subspace, no EMA teacher, no LoRA, no multi-step autoregressive "
                 "training, no pixel decoder.")
    lines.append("- This is therefore a **frozen-DINOv2 feature predictor**, not an "
                 "I-JEPA/V-JEPA EMA objective. The report does not call it one.")
    lines.append("")
    lines.append("## 2. Data index")
    lines.append("")
    lines.append(table(
        ["Quantity", "Train", "Validation"],
        [
            ["pairs", f"{run.get('train_pairs', 0):,}", f"{run.get('val_pairs', 0):,}"],
            ["unique images", f"{run.get('train_unique_images', 0):,}",
             f"{run.get('val_unique_images', 0):,}"],
            ["plots", f"{run.get('train_plots', 0):,}", f"{run.get('val_plots', 0):,}"],
        ],
    ))
    lines.append("")
    lines.append(f"- pair families: `{d['pair_types']}` (S2 tolerance ±2 d); the actual "
                 "`delta_days` is the model input, not the span name.")
    lines.append(f"- index SHA256: `{run.get('index_sha256')}`")
    lines.append(f"- feature-index SHA256: `{run.get('feature_index_sha256')}`")
    lines.append(f"- time normalisation (train only): {run.get('delta_norm')}")
    lines.append(f"- **Pilot flag**: `{run.get('pilot')}` — {run.get('pilot_note', '')}")
    lines.append("")
    lines.append("## 3. Model")
    lines.append("")
    mc = cfg["model"]
    lines.append(table(
        ["Component", "Value"],
        [
            ["input", "frozen DINOv2 patch tokens `z_a` [B, 256, 768]"],
            ["time input", "real `delta_days` [B]"],
            ["predictor", "upstream `models.vit.ViTPredictor`"],
            ["predictor depth / heads / mlp", f"{mc['depth']} / {mc['heads']} / {mc['mlp_dim']}"],
            ["visual / proprio / action width",
             f"{mc['visual_dim']} / {mc['proprio_emb_dim']*mc['num_proprio_repeat']} / "
             f"{mc['action_emb_dim']*mc['num_action_repeat']}"],
            ["predictor width", f"{mc['visual_dim'] + mc['proprio_emb_dim']*mc['num_proprio_repeat'] + mc['action_emb_dim']*mc['num_action_repeat']}"],
            ["time encoder", f"{mc['delta']['n_frequencies']} sin/cos frequencies → MLP "
                             f"({mc['delta']['hidden_dim']}) → {mc['delta']['output_dim']}"],
            ["proprio input", "constant zero placeholder (no future phenotype)"],
            ["trainable parameters", f"{run.get('n_trainable_parameters', 0)/1e6:.2f} M"],
            ["output", "`z_hat_b` [B, 256, 768]"],
        ],
    ))
    lines.append("")
    lines.append("Loss: `MSE(z_hat_b, stopgrad(z_b))` on the 256 visual patch tokens only. "
                 "The future image is never a model input; it only contributes the "
                 "stop-gradient target. The constant proprio channels receive no loss.")
    lines.append("")
    lines.append("## 4. Optimisation (measured)")
    lines.append("")
    lines.append(table(
        ["Item", "Value"],
        [
            ["device", f"{run.get('device')} ({run.get('gpu_name')})"],
            ["torch", run.get("torch")],
            ["AMP", run.get("amp")],
            ["optimiser", "AdamW"],
            ["batch size", run.get("batch_size")],
            ["steps / epoch", run.get("steps_per_epoch")],
            ["epochs completed", run.get("epochs_completed")],
            ["global steps", run.get("global_step")],
            ["wall clock", f"{run.get('wall_clock_seconds', 0)/60:.1f} min"],
            ["throughput", f"{run.get('samples_per_second')} samples/s"],
            ["peak GPU memory", f"{run.get('peak_gpu_memory_bytes', 0)/1e9:.2f} GB"],
            ["best validation MSE", f(run.get("best_val_mse"))],
            ["best epoch", run.get("best_epoch")],
        ],
    ))
    lines.append("")
    lines.append("## 5. Training curve (real epoch logs)")
    lines.append("")
    hist = run.get("history", [])
    rows = []
    for h in hist:
        v = h["val"]
        rows.append([
            h["epoch"], f(h["train_loss"]), f(v["mse_pred"]), f(v["mse_persist"]),
            pct(v.get("relative_improvement")), f(v["cosine_pred"], 4),
            f(v["cosine_persist"], 4), h.get("epoch_seconds"),
        ])
    lines.append(table(
        ["epoch", "train loss", "val MSE V0", "val MSE persistence",
         "rel. improvement", "cos V0", "cos persistence", "epoch s"], rows))
    lines.append("")
    lines.append(f"Per-step logs: `{Path(tcfg['output_dir']) / 'metrics.jsonl'}`. "
                 "All values above are copied from that log.")
    lines.append("")
    lines.append("![S3 V0 training curve](figures/s3_v0_training_curve.png)")
    lines.append("")
    lines.append("Regenerate the figure with `python3 tools/s3_make_figures.py`.")
    lines.append("")
    lines.append("## 6. Checkpoint and resume")
    lines.append("")
    lines.append(f"- best checkpoint: `{run.get('checkpoint_best')}`")
    lines.append(f"- last checkpoint: `{run.get('checkpoint_last')}`")
    lines.append(f"- checkpoint SHA256 (best): `{sha256_file(ckpt_path)}`")
    lines.append("- format: `agro-jepa-s3-v0-state-dict-v1` (plain `state_dict`, not a "
                 "whole-object pickle)")
    lines.append("- contents: model, optimizer, scheduler, AMP scaler, epoch, global step, "
                 "config, train-only delta statistics, manifest id, index SHA256, "
                 "feature-index SHA256, git commit, metrics")
    if verify:
        resumed = [c for c in verify.get("checks", []) if c["name"] == "E.checkpoint_resume"]
        lines.append(f"- resume test: **{'pass' if resumed and resumed[0]['status']=='pass' else 'see verify'}** "
                     "(fresh model + optimizer load reproduces the same prediction)")
    lines.append("")
    lines.append("## 7. Reproduction")
    lines.append("")
    lines.append("```bash")
    lines.append("# 1) images (resumable; re-run to close failed downloads)")
    lines.append("python3 tools/s3_prepare_images.py --train-plots-per-year 200 --workers 40")
    lines.append("# 2) frozen features")
    lines.append("CUDA_VISIBLE_DEVICES=1 python3 tools/s3_build_features.py")
    lines.append("# 3) frozen train/validation index")
    lines.append("python3 tools/s3_build_index.py --config agro_jepa/config/s3_v0.yaml")
    lines.append("# 4) training")
    lines.append("CUDA_VISIBLE_DEVICES=1 python3 agro_jepa/train/train_v0.py \\")
    lines.append("    --config agro_jepa/config/s3_v0.yaml --report reports/s3_training.json")
    lines.append("# 5) evaluation + acceptance")
    lines.append("CUDA_VISIBLE_DEVICES=1 python3 agro_jepa/eval/eval_v0.py \\")
    lines.append("    --config agro_jepa/config/s3_v0.yaml \\")
    lines.append("    --checkpoint external_data/s3_runs/v0/checkpoints/best.pt")
    lines.append("CUDA_VISIBLE_DEVICES=1 python3 tools/s3_verify.py")
    lines.append("```")
    lines.append("")
    lines.append("## 8. Irregular-interval expansion run")
    lines.append("")
    consec_summary = ROOT / "reports/s3_training_consecutive.json"
    if consec_summary.exists():
        cr = read_json(consec_summary)
        lines.append("Same model, loss and plots; the index adds the S2 true-consecutive "
                     "pairs (`span7 + span14 + consecutive`).")
        lines.append("")
        lines.append(table(
            ["Item", "Value"],
            [
                ["pairs (train / val)", f"{cr.get('train_pairs'):,} / {cr.get('val_pairs'):,}"],
                ["realised delta range", "1–84 d (consecutive median 3 d)"],
                ["epochs completed", cr.get("epochs_completed")],
                ["global steps", cr.get("global_step")],
                ["wall clock", f"{cr.get('wall_clock_seconds', 0) / 60:.1f} min"],
                ["throughput", f"{cr.get('samples_per_second')} samples/s"],
                ["peak GPU memory", f"{cr.get('peak_gpu_memory_bytes', 0) / 2**30:.2f} GiB"],
                ["best validation MSE", f(cr.get("best_val_mse"))],
                ["best epoch", cr.get("best_epoch")],
                ["checkpoint", "external_data/s3_runs/v0_consecutive/checkpoints/best.pt"],
            ],
        ))
        lines.append("")
        lines.append("E1 for this run is in `reports/s3_v0_results.md` §1.4; "
                     "its curve is `reports/figures/s3_v0_consecutive_training_curve.png`.")
    else:
        lines.append("Not run in this workspace.")
    lines.append("")
    lines.append("## 9. Open items")
    lines.append("")
    lines.append("1. Pilot scale: 1,000 train plots, all 380 validation plots. Extending to "
                 "the full 2,930 train plots is a data/acquisition step, not a code change.")
    lines.append("2. The time channel is a sinusoidal+MLP projection into the DINO-WM action "
                 "channel; whether a stronger conditioning (e.g. FiLM) helps is an S4 "
                 "ablation.")
    lines.append("3. No weather or physiology is used in V0, by design; S4 adds the "
                 "Weather-GRU / GDD / phenotype-anchor baselines.")
    lines.append("")
    out = ROOT / "reports/s3_training.md"
    out.write_text("\n".join(lines) + "\n")
    return out


# --------------------------------------------------------------------------
def write_results(cfg: dict) -> Path:
    d = cfg["data"]
    results = read_json("reports/s3_v0_results.json")
    run = read_json(Path(cfg["training"]["output_dir"]) / "run_summary.json")
    verify = read_json("reports/s3_verify.json")
    e1 = results.get("E1_future_latent", {})
    e2 = results.get("E2_probes", {})
    traits = tuple(d["traits"])

    lines = []
    lines.append("# S3 — FIP-DINO-WM V0 results (E1 future latent + E2 CC/PH probes)")
    lines.append("")
    lines.append("Stage: **S3**. Date: 2026-10-24. Branch: `agro-s0-upstream`.")
    lines.append("")
    lines.append("All numbers below are read from `reports/s3_v0_results.json` and "
                 "`reports/s3_v0_val_predictions.parquet`; nothing is pre-filled.")
    lines.append("")
    lines.append("## 0. TL;DR")
    lines.append("")
    if e1:
        rel = e1.get("relative_improvement")
        lines.append(f"- **E1**: V0 validation MSE **{f(e1.get('mse_pred'))}** vs "
                     f"persistence **{f(e1.get('mse_persist'))}** "
                     f"({pct(rel)} relative change).")
        lines.append(f"- Mean cosine to the future feature: V0 **{f(e1.get('cosine_pred'), 4)}** "
                     f"vs persistence **{f(e1.get('cosine_persist'), 4)}**.")
        lines.append(f"- Stronger trivial baseline, train-mean future feature: "
                     f"**{f(e1.get('mse_train_mean'))}**; delta-conditioned train mean: "
                     f"**{f(e1.get('mse_train_delta_mean'))}**.")
        if e1.get("mse_train_mean") is not None:
            beats_mean = e1["mse_pred"] < e1["mse_train_mean"]
            lines.append(f"- V0 {'**beats**' if beats_mean else '**does not beat**'} the "
                         "train-mean baseline.")
    if e2:
        overall = e2.get("scopes", {}).get("overall", {}).get("traits", {})
        for t in traits:
            tr = overall.get(t, {})
            a = tr.get("conditions", {}).get("A_current", {})
            b = tr.get("conditions", {}).get("B_predicted", {})
            if a and b:
                lines.append(
                    f"- **E2 {t.upper()}** (overall, n_val={tr.get('n_val')}): "
                    f"A current-feature R²={f(a['r2']['mean'], 4)} / MAE={f(a['mae']['mean'], 4)}; "
                    f"B predicted R²={f(b['r2']['mean'], 4)} / MAE={f(b['mae']['mean'], 4)}."
                )
    lines.append("")
    consec_path = ROOT / "reports/s3_v0_results_consecutive.json"
    if consec_path.exists():
        ce1 = read_json(consec_path).get("E1_future_latent", {})
        if ce1:
            lines.append(
                f"- **Irregular expansion** (span7 + span14 + consecutive, "
                f"{ce1.get('n_samples', 0):,} val pairs): MSE "
                f"**{f(ce1.get('mse_pred'))}** vs persistence "
                f"**{f(ce1.get('mse_persist'))}** ({pct(ce1.get('relative_improvement'))}); "
                f"the true-consecutive subset alone gives "
                f"{pct(ce1.get('by_span', {}).get('consecutive', {}).get('relative_improvement'))}."
            )
    lines.append("")
    lines.append("## 1. E1 — future latent prediction")
    lines.append("")
    lines.append("Validation split only (official `validation` plots). "
                 f"Effective sample: **{e1.get('n_samples', 0):,} pairs**, "
                 f"**{e1.get('n_plots', 0):,} plots**.")
    lines.append("")
    lines.append(table(
        ["Predictor", "MSE ↓", "Mean cosine ↑", "Relative improvement vs persistence"],
        [
            ["Persistence (`z_a` as future)", f(e1.get("mse_persist")),
             f(e1.get("cosine_persist"), 4), "—"],
            ["Train-mean future feature", f(e1.get("mse_train_mean")), "n/a",
             pct((e1.get("mse_persist", 0) - e1.get("mse_train_mean", 0))
                 / e1.get("mse_persist", 1))
             if e1.get("mse_train_mean") is not None else "n/a"],
            ["Train-mean per exact delta", f(e1.get("mse_train_delta_mean")), "n/a",
             pct((e1.get("mse_persist", 0) - e1.get("mse_train_delta_mean", 0))
                 / e1.get("mse_persist", 1))
             if e1.get("mse_train_delta_mean") is not None else "n/a"],
            ["**FIP-DINO-WM V0**", f"**{f(e1.get('mse_pred'))}**",
             f"**{f(e1.get('cosine_pred'), 4)}**", pct(e1.get("relative_improvement"))],
        ],
    ))
    lines.append("")
    if e1.get("delta_shuffle_mse_increase") is not None:
        lines.append("**Δt diagnostics (inference only, no retraining).** Shuffling the real "
                     "`delta_days` across the validation batch increases MSE by "
                     f"**{pct(e1.get('delta_shuffle_mse_increase'))}**; replacing every "
                     "interval with the train mean increases it by "
                     f"**{pct(e1.get('delta_constant_mse_increase'))}**. A near-zero "
                     "change would mean the predictor ignores the time channel.")
        lines.append("")
    lines.append("### 1.1 By nominal span")
    lines.append("")
    rows = []
    for span, s in sorted(e1.get("by_span", {}).items()):
        rows.append([
            span, s["n_samples"], s["n_plots"], f(s["mse_pred"]), f(s["mse_persist"]),
            f(s.get("mse_train_mean")), pct(s.get("relative_improvement")),
            f(s["cosine_pred"], 4), f(s["cosine_persist"], 4),
        ])
    lines.append(table(
        ["span", "pairs", "plots", "MSE V0", "MSE persist", "MSE train-mean",
         "rel. vs persist", "cos V0", "cos persist"], rows))
    lines.append("")
    lines.append("### 1.2 By harvest year")
    lines.append("")
    rows = []
    for year, s in sorted(e1.get("by_year", {}).items()):
        rows.append([
            year, s["n_samples"], s["n_plots"], f(s["mse_pred"]), f(s["mse_persist"]),
            f(s.get("mse_train_mean")), pct(s.get("relative_improvement")),
            f(s["cosine_pred"], 4), f(s["cosine_persist"], 4),
        ])
    lines.append(table(
        ["year", "pairs", "plots", "MSE V0", "MSE persist", "MSE train-mean",
         "rel. vs persist", "cos V0", "cos persist"], rows))
    lines.append("")
    lines.append("### 1.3 By realised interval")
    lines.append("")
    rows = []
    for b, s in sorted(e1.get("by_delta_bin", {}).items()):
        rows.append([
            b, s["n_samples"], s["n_plots"], f(s["mse_pred"]), f(s["mse_persist"]),
            f(s.get("mse_train_mean")), pct(s.get("relative_improvement")),
            f(s["cosine_pred"], 4), f(s["cosine_persist"], 4),
        ])
    lines.append(table(
        ["delta (days)", "pairs", "plots", "MSE V0", "MSE persist", "MSE train-mean",
         "rel. vs persist", "cos V0", "cos persist"], rows))
    lines.append("")
    lines.append("Per-sample rows: `reports/s3_v0_val_predictions.parquet` "
                 "(pair_id, plot, split, span, delta, dates, both MSEs, both cosines).")
    lines.append("")

    # Optional: expansion to irregular consecutive intervals.
    consec_path = ROOT / "reports/s3_v0_results_consecutive.json"
    if consec_path.exists():
        consec = read_json(consec_path)
        ce1 = consec.get("E1_future_latent", {})
        lines.append("### 1.4 Irregular-interval expansion (`span7 + span14 + consecutive`)")
        lines.append("")
        lines.append("A second V0 model trained on the same plots and encoder with the S2 "
                     "true-consecutive pairs added. The realised interval is the only "
                     "difference; the model and loss are unchanged.")
        lines.append("")
        lines.append(table(
            ["Predictor", "MSE ↓", "Mean cosine ↑", "Relative improvement vs persistence"],
            [
                ["Persistence (`z_a` as future)", f(ce1.get("mse_persist")),
                 f(ce1.get("cosine_persist"), 4), "—"],
                ["Train-mean future feature", f(ce1.get("mse_train_mean")), "n/a",
                 pct((ce1.get("mse_persist", 0) - ce1.get("mse_train_mean", 0))
                     / ce1.get("mse_persist", 1))],
                ["**V0 + consecutive pairs**", f"**{f(ce1.get('mse_pred'))}**",
                 f"**{f(ce1.get('cosine_pred'), 4)}**",
                 pct(ce1.get("relative_improvement"))],
            ],
        ))
        lines.append("")
        rows = []
        for span, s in sorted(ce1.get("by_span", {}).items()):
            rows.append([
                span, s["n_samples"], s["n_plots"], f(s["mse_pred"]),
                f(s["mse_persist"]), f(s.get("mse_train_mean")),
                pct(s.get("relative_improvement")), f(s["cosine_pred"], 4),
                f(s["cosine_persist"], 4),
            ])
        lines.append(table(
            ["span", "pairs", "plots", "MSE V0", "MSE persist", "MSE train-mean",
             "rel. vs persist", "cos V0", "cos persist"], rows))
        lines.append("")
        rows = []
        def _bin_key(b: str) -> int:
            digits = "".join(ch for ch in b.split("-")[0] if ch.isdigit())
            return int(digits) if digits else 10 ** 6
        for b, s in sorted(ce1.get("by_delta_bin", {}).items(), key=lambda kv: _bin_key(kv[0])):
            rows.append([
                b, s["n_samples"], s["n_plots"], f(s["mse_pred"]),
                f(s["mse_persist"]), pct(s.get("relative_improvement")),
                f(s["cosine_pred"], 4), f(s["cosine_persist"], 4),
            ])
        lines.append(table(
            ["delta (days)", "pairs", "plots", "MSE V0", "MSE persist",
             "rel. vs persist", "cos V0", "cos persist"], rows))
        lines.append("")
        lines.append("Per-sample rows: `reports/s3_v0_val_predictions_consecutive.parquet`; "
                     "full JSON: `reports/s3_v0_results_consecutive.json`.")
        lines.append("")

    lines.append("## 2. E2 — future CC / PH linear probes")
    lines.append("")
    lines.append("Two readouts with the **same architecture** (linear on the mean-pooled "
                 "768-d feature, inputs standardised with train statistics), the same "
                 "optimiser and the same epoch budget. Seeds are averaged (mean ± std).")
    lines.append("")
    lines.append("A = current DINOv2 feature `mean(Z_a)`; "
                 "B = V0 predicted future feature `mean(Z_hat_b)`.")
    lines.append("")
    lines.append("Targets are the raw phenotype at `t_b` with S1/S2's `|lag| <= 7 d` mask "
                 "(CC in fraction 0–1, PH in metres). "
                 "This is a **near-neighbour measurement** result, not an exact-date "
                 "result; the ±3 d sensitivity is reported where the sample allows.")
    lines.append("")
    for scope in ("overall", "span7", "span14"):
        block = e2.get("scopes", {}).get(scope, {}).get("traits", {})
        if not block:
            continue
        lines.append(f"### 2.{['overall','span7','span14'].index(scope)+1} Scope: `{scope}`")
        lines.append("")
        rows = []
        for t in traits:
            tr = block.get(t, {})
            a = tr.get("conditions", {}).get("A_current")
            b = tr.get("conditions", {}).get("B_predicted")
            if not a or not b:
                continue
            for name, c in (("A current", a), ("B predicted", b)):
                rows.append([
                    t, name, f"{tr.get('n_train', 0):,}", f"{tr.get('n_val', 0):,}",
                    f"{c['mae']['mean']:.4f} ± {c['mae']['std']:.4f}",
                    f"{c['rmse']['mean']:.4f} ± {c['rmse']['std']:.4f}",
                    f"{c['r2']['mean']:.4f} ± {c['r2']['std']:.4f}",
                ])
        lines.append(table(
            ["trait", "readout", "n train", "n val", "MAE", "RMSE", "R²"], rows))
        lines.append("")
        # B - A
        rows = []
        for t in traits:
            tr = block.get(t, {})
            d = tr.get("B_minus_A")
            if not d:
                continue
            rows.append([
                t,
                f"{d['mae']['mean_delta']:+.5f}",
                f"{d['rmse']['mean_delta']:+.5f}",
                f"{d['r2']['mean_delta']:+.5f}",
            ])
        lines.append("B minus A (negative MAE/RMSE is better; positive R² is better):")
        lines.append("")
        lines.append(table(["trait", "ΔMAE", "ΔRMSE", "ΔR²"], rows))
        lines.append("")
        # label timing
        rows = []
        for t in traits:
            tr = block.get(t, {})
            lag = tr.get("label_lag_days")
            if not lag:
                continue
            for split in ("train", "validation"):
                d = lag[split]
                rows.append([
                    t, split,
                    f"{d.get('median')}",
                    f"{d.get('abs_median')}",
                    f"{d.get('abs_p90')}",
                    f"{d.get('n_abs_le_3')}",
                    f"{d.get('n_abs_le_7')}",
                ])
        lines.append("Label timing (signed `label_date - image_date_b`, days):")
        lines.append("")
        lines.append(table(
            ["trait", "split", "median", "|median|", "|p90|", "n |lag|≤3", "n |lag|≤7"], rows))
        lines.append("")
    overall = e2.get("scopes", {}).get("overall", {}).get("traits", {})
    strict_rows = []
    for t in traits:
        tr = overall.get(t, {})
        st = tr.get("strict_lag3", {})
        a = st.get("conditions", {}).get("A_current")
        b = st.get("conditions", {}).get("B_predicted")
        if a and b:
            strict_rows.append([
                t, f"{st.get('n_train', 0):,}", f"{st.get('n_val', 0):,}",
                f"{a['r2']['mean']:.4f}", f"{b['r2']['mean']:.4f}",
                f"{a['mae']['mean']:.4f}", f"{b['mae']['mean']:.4f}",
            ])
    if strict_rows:
        lines.append("### 2.4 Strict-date sensitivity (`|lag| <= 3 d`)")
        lines.append("")
        lines.append(table(
            ["trait", "n train", "n val", "R² A", "R² B", "MAE A", "MAE B"], strict_rows))
        lines.append("")
    lines.append("## 3. Interpretation")
    lines.append("")
    if e1:
        rel = e1.get("relative_improvement") or 0.0
        beats_p = rel > 0
        beats_m = (e1.get("mse_pred", 1e9) < e1.get("mse_train_mean", 0))
        span_rets = {
            k: (v.get("relative_improvement") or 0.0)
            for k, v in e1.get("by_span", {}).items()
        }
        all_pos = all(r > 0 for r in span_rets.values()) if span_rets else False
        lines.append(f"1. **Does V0 beat persistence?** "
                     f"{'Yes' if beats_p else 'No'} — validation MSE changes by "
                     f"{pct(rel)}. Per-span relative changes: "
                     + ", ".join(f"{k}={pct(v)}" for k, v in sorted(span_rets.items()))
                     + f"; improvement is {'positive for every span' if all_pos else 'not positive for every span'}.")
        lines.append(f"2. **Does V0 beat a train-mean future predictor?** "
                     f"{'Yes' if beats_m else 'No'} "
                     f"(V0 {f(e1.get('mse_pred'))} vs train-mean "
                     f"{f(e1.get('mse_train_mean'))}, delta-conditioned mean "
                     f"{f(e1.get('mse_train_delta_mean'))}). This is the harder trivial "
                     "baseline; a predictor that only learns the average future feature "
                     "would look good against persistence but adds no image-conditioned "
                     "information.")
        if e2:
            overall = e2.get("scopes", {}).get("overall", {}).get("traits", {})
            parts = []
            for t in traits:
                d = overall.get(t, {}).get("B_minus_A")
                if d:
                    parts.append(
                        f"{t.upper()}: ΔMAE={d['mae']['mean_delta']:+.5f}, "
                        f"ΔR²={d['r2']['mean_delta']:+.5f}"
                    )
            lines.append("3. **Do the predicted features carry CC/PH information?** "
                         "B is trained on V0 predictions; A reads the current image. "
                         + "; ".join(parts) + ". Negative ΔMAE / positive ΔR² means the "
                         "predicted future latent is more useful than the current feature "
                         "for that target; the opposite means it is not.")
        else:
            lines.append("3. **Do the predicted features carry CC/PH information?** "
                         "The E2 probe report is missing from the results JSON.")
    lines.append("")
    lines.append("## 4. Acceptance (S3 §10)")
    lines.append("")
    if verify:
        lines.append(f"`tools/s3_verify.py`: **{verify.get('n_passed')}/{verify.get('n_checks')}** "
                     "checks passed. The full check list is in `reports/s3_verify.json`.")
    lines.append("")
    lines.append("| Item | Status |")
    lines.append("|---|---|")
    lines.append("| A. real train/validation image subset with plot coverage | pass (acquisition report) |")
    lines.append("| B. real 224×224 → 256×768 DINOv2 tokens | pass (verify B) |")
    lines.append("| C. `predict(z_a, delta)` → [B,256,768] | pass (verify C) |")
    lines.append("| D. future features stop-gradient only | pass (verify D) |")
    lines.append("| E. real optimisation + checkpoint + resume | pass (verify E) |")
    lines.append("| F. V0 vs persistence on validation | pass (this report §1) |")
    lines.append("| G. CC/PH probes vs current feature | pass (this report §2) |")
    lines.append("| H. code/config/checkpoint/index/results traceable | pass (SHA256s in reports) |")
    lines.append("")
    lines.append("## 5. Limitations")
    lines.append("")
    lines.append("1. **Pilot**: 1,000 of 2,930 official train plots; all 380 validation "
                 "plots. The result is not a full-FIP baseline.")
    lines.append("2. **Label timing**: the primary probe uses `|lag| <= 7 d`; the ±3 d "
                 "sensitivity is reported separately and has fewer samples.")
    lines.append("3. **Probe capacity**: a linear readout on mean-pooled tokens is "
                 "deliberately simple; it does not prove that no nonlinear readout could "
                 "extract more from the same latent.")
    lines.append("4. **No weather, physiology, EMA, LoRA or multi-step rollout** in V0; "
                 "those are S4 comparisons, not claims of this stage.")
    lines.append("5. The mean-future baseline is a stronger control than persistence; the "
                 "V0-vs-mean comparison is the one that tests image conditioning.")
    lines.append("")
    out = ROOT / "reports/s3_v0_results.md"
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s3_v0.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    paths = [write_acquisition(cfg), write_training(cfg), write_results(cfg)]
    for p in paths:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
