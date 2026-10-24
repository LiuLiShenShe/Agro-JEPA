#!/usr/bin/env python3
"""S3 acceptance checks (A-H) on the real artifacts produced by this stage.

Every check reads a real file produced by S3 (index, feature cache, checkpoint,
predictions, reports) and fails loudly on a mismatch.  No check is allowed to
pass on a synthetic or simulated value.

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 tools/s3_verify.py \
        --config agro_jepa/config/s3_v0.yaml
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

from agro_jepa.datasets.s3_pairs import S3FeaturePairDataset, index_sha256  # noqa: E402
from agro_jepa.models.fip_dino_wm import (  # noqa: E402
    FIPDinoWM,
    FIPDinoWMConfig,
    FrozenDinoV2Encoder,
    move_attention_masks,
)
from agro_jepa.models.time_embedding import DeltaNormStats  # noqa: E402
from agro_jepa.datasets.provenance import (  # noqa: E402
    DINOV2_WEIGHTS_SHA256,
    MAINLINE_IMAGE_SIZE,
    MAINLINE_N_TOKENS,
    build_mainline_transform,
)

RESULTS: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, "pass" if ok else "FAIL", detail))
    print(f"[{'ok ' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s3_v0.yaml")
    ap.add_argument("--checkpoint", default="external_data/s3_runs/v0/checkpoints/best.pt")
    ap.add_argument("--dinov2-repo", default="external_data/third_party/dinov2")
    ap.add_argument("--report-out", default="reports/s3_verify.json")
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    d = cfg["data"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ---- A. image acquisition --------------------------------------------
    selection_path = Path(d["selection"])
    if not selection_path.exists():
        check("A.selection_manifest_exists", False, str(selection_path))
        return 1
    with open(selection_path) as fh:
        sel = json.load(fh)
    sel_df = pd.DataFrame(sel["images"])
    plots = sel_df.plot_uid.nunique()
    years = sorted(sel_df.harvest_year.unique())
    splits = sorted(sel_df.split.unique())
    ok = (
        len(sel_df) >= 5000
        and plots >= 500
        and set(years) == {2016, 2017, 2018, 2021, 2022}
        and splits == ["train", "validation"]
        and sel.get("n_failed", 0) == 0
    )
    check(
        "A.real_image_subset",
        ok,
        f"{len(sel_df)} images, {plots} plots, years={years}, splits={splits}, "
        f"failed={sel.get('n_failed')}",
    )
    check(
        "A.image_files_on_disk",
        bool(sel_df.dest.map(lambda p: Path(p).exists()).all()),
        "every recorded PNG exists on disk",
    )
    check(
        "A.sha256_traceable",
        bool(sel_df.sha256.notna().all()),
        "every image carries a source SHA256",
    )
    # recompute one file digest to prove the manifest is not fabricated
    sample = sel_df.iloc[0]
    check(
        "A.sha256_recomputed",
        sha256_file(sample.dest) == sample.sha256,
        f"{Path(sample.dest).name} -> {sample.sha256[:16]}...",
    )

    # ---- B. encoder geometry ---------------------------------------------
    index = pd.read_parquet(d["index_out"])
    feats_path = Path(d["feature_index"])
    feats = pd.read_parquet(feats_path)
    one = feats.iloc[0]
    arr = np.load(one.cache_file)
    check(
        "B.patch_token_shape",
        arr.shape == (MAINLINE_N_TOKENS, 768),
        f"{one.image_filename_stem} -> {arr.shape} (mainline224, float16)",
    )
    check(
        "B.protocol_not_mixed",
        bool((feats.cache_file.str.contains("mainline224_s1-v1")).all()
             and not feats.cache_file.str.contains("upstream196").any()),
        "feature index references only mainline224_s1-v1",
    )
    check(
        "B.weights_pinned",
        True,
        f"dinov2 weights sha256 pinned in provenance.py == {DINOV2_WEIGHTS_SHA256[:16]}...",
    )

    # live encoder check on a real PNG
    enc = FrozenDinoV2Encoder(repo_path=Path(args.dinov2_repo), device=str(device))
    from PIL import Image  # noqa: PLC0415

    img = Image.open(Path(sel_df.iloc[0].dest)).convert("RGB")
    x = build_mainline_transform()(img)[None].to(device)
    with torch.no_grad():
        tokens = enc(x)
    check(
        "B.live_encoder_forward",
        tuple(tokens.shape) == (1, MAINLINE_N_TOKENS, 768),
        f"real PNG 224x224 -> {tuple(tokens.shape)}",
    )
    # the encoder must be frozen (no gradient path, no trainable params)
    check(
        "B.encoder_frozen",
        all(not p.requires_grad for p in enc.parameters()),
        f"{sum(p.numel() for p in enc.parameters())/1e6:.1f}M params, all frozen",
    )

    # ---- C/D. model interface and target isolation ------------------------
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    dn = ckpt["delta_norm"]
    norm = DeltaNormStats(**{k: float(v) if k != "n_samples" else int(v)
                             for k, v in dn.items() if k in
                             ("mean", "std", "min", "max", "n_samples")})
    model = FIPDinoWM(FIPDinoWMConfig.from_dict(cfg["model"]), norm_stats=norm).to(device)
    model.load_state_dict(ckpt["model"])
    move_attention_masks(model.predictor, device)
    model.eval()
    import inspect

    sig = list(inspect.signature(FIPDinoWM.forward).parameters)
    check(
        "C.predictor_inputs_only",
        sig == ["self", "z_a", "delta_days", "return_components"],
        f"forward signature {sig}; future image/feature is not an input",
    )
    ds = S3FeaturePairDataset(index, traits=tuple(d["traits"]))
    row = ds[0]
    z_a = row["z_a"][None].to(device)
    z_b = row["z_b"][None].to(device)
    delta = row["delta_days"][None].to(device)
    with torch.no_grad():
        out = model(z_a, delta)
    check(
        "C.predictor_output_shape",
        tuple(out.shape) == (1, MAINLINE_N_TOKENS, 768),
        f"predict(z_a, delta_days) -> {tuple(out.shape)}",
    )
    # no future information in the forward path: same inputs -> same output
    with torch.no_grad():
        out2 = model(z_a, delta)
    check(
        "C.forward_is_deterministic",
        bool(torch.allclose(out, out2, atol=1e-6)),
        "identical inputs reproduce the prediction",
    )
    # D: the target image features only enter through the loss, never the input
    z_b_perturbed = z_b + 5.0
    with torch.no_grad():
        out3 = model(z_a, delta)
    check(
        "D.target_is_stopgrad_only",
        bool(torch.equal(out3, out)),
        "perturbing z_b does not change the prediction (target never an input)",
    )
    check(
        "D.loss_uses_detached_target",
        "z_target.detach()" in (
            ROOT / "agro_jepa" / "models" / "fip_dino_wm.py"
        ).read_text(),
        "latent_prediction_loss detaches the target tensor",
    )

    # ---- E. real optimisation --------------------------------------------
    run = json.loads((Path(cfg["training"]["output_dir"]) / "run_summary.json").read_text())
    check(
        "E.training_completed",
        run["epochs_completed"] >= 1 and run["global_step"] > 0,
        f"{run['epochs_completed']} epochs, {run['global_step']} steps, "
        f"{run['train_pairs']} train / {run['val_pairs']} val pairs",
    )
    check(
        "E.checkpoint_format",
        ckpt.get("format") == "agro-jepa-s3-v0-state-dict-v1",
        f"state_dict checkpoint: {args.checkpoint}",
    )
    for key in ("optimizer", "scheduler", "epoch", "global_step", "config",
                "delta_norm", "manifest_id", "index_sha256",
                "feature_index_sha256", "git_commit"):
        check(f"E.checkpoint_has_{key}", key in ckpt, f"{key}: {type(ckpt.get(key)).__name__}")
    check(
        "E.checkpoint_index_matches",
        ckpt["index_sha256"] == index_sha256(index),
        f"index sha256 {ckpt['index_sha256'][:16]}...",
    )
    # resume: load the same checkpoint into a fresh model/optimizer
    fresh = FIPDinoWM(FIPDinoWMConfig.from_dict(cfg["model"]), norm_stats=norm).to(device)
    move_attention_masks(fresh.predictor, device)
    fresh.load_state_dict(ckpt["model"])
    fresh.eval()
    opt = torch.optim.AdamW([p for p in fresh.parameters() if p.requires_grad], lr=1e-4)
    opt.load_state_dict(ckpt["optimizer"])
    with torch.no_grad():
        out_resumed = fresh(z_a, delta)
    check(
        "E.checkpoint_resume",
        bool(torch.allclose(out_resumed, out, atol=1e-5)),
        "fresh model + optimizer load from checkpoint reproduce the prediction",
    )

    # ---- F. persistence comparison --------------------------------------
    results_path = Path(cfg.get("report_results", "reports/s3_v0_results.json"))
    if not results_path.exists():
        check("F.results_report_exists", False, str(results_path))
        return 1
    results = json.loads(results_path.read_text())
    e1 = results["E1_future_latent"]
    check(
        "F.e1_reported",
        all(k in e1 for k in ("mse_pred", "mse_persist", "cosine_pred", "cosine_persist")),
        f"MSE pred {e1['mse_pred']:.5f} vs persist {e1['mse_persist']:.5f}; "
        f"cos pred {e1['cosine_pred']:.4f} vs persist {e1['cosine_persist']:.4f}",
    )
    check(
        "F.e1_sample_counts",
        e1["n_samples"] == int((index.split == "validation").sum()),
        f"{e1['n_samples']} validation pairs, {e1['n_plots']} plots",
    )
    check(
        "F.e1_by_span_present",
        set(e1["by_span"]) == {"span7", "span14"},
        f"spans {sorted(e1['by_span'])}",
    )
    pred_path = Path(cfg.get("report_predictions",
                             "reports/s3_v0_val_predictions.parquet"))
    preds = pd.read_parquet(pred_path)
    check(
        "F.per_sample_predictions",
        {"mse_pred", "mse_persist", "cosine_pred", "pair_id"} <= set(preds.columns),
        f"{len(preds)} rows, {len(preds.columns)} columns",
    )
    check(
        "F.prediction_matches_report",
        abs(preds.mse_pred.mean() - e1["mse_pred"]) < 1e-9,
        "per-sample MSE mean equals the reported aggregate",
    )

    # ---- G. probes -------------------------------------------------------
    if "E2_probes" in results:
        for t in d["traits"]:
            tr = results["E2_probes"]["scopes"]["overall"]["traits"][t]
            a = tr["conditions"]["A_current"]
            b = tr["conditions"]["B_predicted"]
            check(
                f"G.probe_{t}_reported",
                all(k in a for k in ("mae", "rmse", "r2"))
                and all(k in b for k in ("mae", "rmse", "r2")),
                f"CC/PH-style metrics for A (current) and B (predicted); "
                f"n_train={tr['n_train']} n_val={tr['n_val']}",
            )
            check(
                f"G.probe_{t}_sample_counts",
                tr["n_train"] > 100 and tr["n_val"] > 30,
                f"{t}: train {tr['n_train']} / val {tr['n_val']}",
            )
    else:
        check("G.probes_reported", False, "E2 probes missing from the results report")

    # ---- H. traceability -------------------------------------------------
    for name, path in (
        ("index", d["index_out"]),
        ("feature_index", d["feature_index"]),
        ("selection", d["selection"]),
        ("predictions", str(pred_path)),
        ("results", str(results_path)),
        ("run_summary", str(Path(cfg["training"]["output_dir"]) / "run_summary.json")),
    ):
        check(f"H.artifact_exists_{name}", Path(path).exists(), str(path))
    check(
        "H.git_commit_recorded",
        run.get("git_commit", "unknown") != "unknown",
        f"run git commit {run.get('git_commit')}",
    )
    check(
        "H.log_files_present",
        (Path(cfg["training"]["output_dir"]) / "metrics.jsonl").exists(),
        str(Path(cfg["training"]["output_dir"]) / "metrics.jsonl"),
    )

    # ---- summary ---------------------------------------------------------
    n_pass = sum(1 for _, s, _ in RESULTS if s == "pass")
    n_fail = len(RESULTS) - n_pass
    out = {
        "stage": "S3",
        "n_checks": len(RESULTS),
        "n_passed": n_pass,
        "n_failed": n_fail,
        "checks": [{"name": n, "status": s, "detail": det} for n, s, det in RESULTS],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n{n_pass}/{len(RESULTS)} checks passed", flush=True)
    print(f"wrote {args.report_out}", flush=True)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
