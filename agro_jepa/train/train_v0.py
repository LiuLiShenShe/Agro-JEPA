#!/usr/bin/env python3
"""S3 V0 training: frozen DINOv2 patch tokens -> time-conditioned future predictor.

Only the predictor is trained.  ``image_b`` never enters the forward pass; its
frozen DINOv2 tokens ``z_b`` are the stop-gradient target::

    z_hat_b = model(z_a, delta_days)
    loss    = MSE(z_hat_b, z_b.detach())

Usage
-----
    CUDA_VISIBLE_DEVICES=1 python3 agro_jepa/train/train_v0.py \
        --config agro_jepa/config/s3_v0.yaml \
        --report reports/s3_training.json

    # resume without changing the schedule
    CUDA_VISIBLE_DEVICES=1 python3 agro_jepa/train/train_v0.py \
        --config agro_jepa/config/s3_v0.yaml \
        --resume external_data/s3_runs/v0/checkpoints/last.pt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agro_jepa.datasets.s3_pairs import (  # noqa: E402
    S3FeaturePairDataset,
    index_sha256,
)
from agro_jepa.models.fip_dino_wm import (  # noqa: E402
    FIPDinoWM,
    FIPDinoWMConfig,
    latent_prediction_loss,
    move_attention_masks,
)
from agro_jepa.models.time_embedding import DeltaNormStats  # noqa: E402


# --------------------------------------------------------------------------
# utilities
# --------------------------------------------------------------------------
def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        import subprocess

        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _amp_context(cfg: dict):
    amp = str(cfg.get("amp", "bf16")).lower()
    if amp == "none" or not torch.cuda.is_available():
        return torch.autocast("cuda", enabled=False) if torch.cuda.is_available() else None, amp
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast("cuda", dtype=dtype), amp


def build_model(cfg: dict, norm: DeltaNormStats, device: torch.device) -> FIPDinoWM:
    model = FIPDinoWM(FIPDinoWMConfig.from_dict(cfg["model"]), norm_stats=norm)
    model.to(device)
    move_attention_masks(model.predictor, device)
    return model


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    model: FIPDinoWM,
    loader: DataLoader,
    device: torch.device,
    amp_context,
) -> dict:
    model.eval()
    sums = {
        "sse_pred": 0.0, "sse_persist": 0.0, "n_elements": 0,
        "cos_pred": 0.0, "cos_persist": 0.0, "n_samples": 0,
    }
    by_span: dict[str, dict] = {}
    by_year: dict[str, dict] = {}
    for batch in loader:
        z_a = batch["z_a"].to(device, non_blocking=True)
        z_b = batch["z_b"].to(device, non_blocking=True)
        delta = batch["delta_days"].to(device, non_blocking=True)
        with (amp_context if amp_context is not None else _nullcontext()):
            z_pred = model(z_a, delta)
        z_pred = z_pred.float()
        z_b = z_b.float()
        z_a = z_a.float()

        se_pred = (z_pred - z_b).pow(2)
        se_persist = (z_a - z_b).pow(2)
        n_el = se_pred.numel()
        cos_pred = torch.nn.functional.cosine_similarity(
            z_pred.flatten(1), z_b.flatten(1), dim=-1
        )
        cos_persist = torch.nn.functional.cosine_similarity(
            z_a.flatten(1), z_b.flatten(1), dim=-1
        )
        sums["sse_pred"] += float(se_pred.sum())
        sums["sse_persist"] += float(se_persist.sum())
        sums["n_elements"] += int(n_el)
        sums["cos_pred"] += float(cos_pred.sum())
        sums["cos_persist"] += float(cos_persist.sum())
        sums["n_samples"] += int(z_a.shape[0])

        # per-span / per-year accumulators
        spans = batch["pair_type"] if "pair_type" in batch else [""] * z_a.shape[0]
        years = batch["harvest_year"] if "harvest_year" in batch else [0] * z_a.shape[0]
        per_sample_mse_pred = se_pred.mean(dim=(1, 2))
        per_sample_mse_persist = se_persist.mean(dim=(1, 2))
        for i in range(z_a.shape[0]):
            for key, table in (
                (str(spans[i]), by_span),
                (str(int(years[i])), by_year),
            ):
                row = table.setdefault(key, {
                    "sse_pred": 0.0, "sse_persist": 0.0, "n_elements": 0,
                    "cos_pred": 0.0, "cos_persist": 0.0, "n_samples": 0,
                })
                row["sse_pred"] += float(per_sample_mse_pred[i]) * z_a.shape[1] * z_a.shape[2]
                row["sse_persist"] += float(per_sample_mse_persist[i]) * z_a.shape[1] * z_a.shape[2]
                row["n_elements"] += int(z_a.shape[1] * z_a.shape[2])
                row["cos_pred"] += float(cos_pred[i])
                row["cos_persist"] += float(cos_persist[i])
                row["n_samples"] += 1

    def finalise(d: dict) -> dict:
        n = max(d["n_elements"], 1)
        mse_pred = d["sse_pred"] / n
        mse_persist = d["sse_persist"] / n
        return {
            "n_samples": d["n_samples"],
            "mse_pred": mse_pred,
            "mse_persist": mse_persist,
            "relative_improvement": (mse_persist - mse_pred) / mse_persist
            if mse_persist > 0 else None,
            "cosine_pred": d["cos_pred"] / max(d["n_samples"], 1),
            "cosine_persist": d["cos_persist"] / max(d["n_samples"], 1),
        }

    out = finalise(sums)
    out["by_span"] = {k: finalise(v) for k, v in sorted(by_span.items())}
    out["by_year"] = {k: finalise(v) for k, v in sorted(by_year.items())}
    return out


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


# --------------------------------------------------------------------------
# checkpointing
# --------------------------------------------------------------------------
def save_checkpoint(
    path: Path,
    model: FIPDinoWM,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler,
    epoch: int,
    global_step: int,
    cfg: dict,
    norm: DeltaNormStats,
    index_sha: str,
    feature_index_sha: str,
    metrics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "agro-jepa-s3-v0-state-dict-v1",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "config": cfg,
        "delta_norm": norm.as_dict(),
        "manifest_id": cfg.get("manifest_id"),
        "index_sha256": index_sha,
        "feature_index_sha256": feature_index_sha,
        "git_commit": git_commit(),
        "metrics": metrics,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    torch.save(payload, path)


def load_checkpoint(path: Path, device: torch.device) -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get("format") != "agro-jepa-s3-v0-state-dict-v1":
        raise ValueError(f"{path} is not an S3 V0 state-dict checkpoint")
    return ckpt


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s3_v0.yaml")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--report", default="reports/s3_training.json")
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    d, tcfg = cfg["data"], cfg["training"]
    epochs = args.epochs if args.epochs is not None else int(tcfg["epochs"])
    device = torch.device(tcfg.get("device", "cuda:0"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")
    set_seed(int(tcfg["seed"]))

    # ---- data ------------------------------------------------------------
    index = pd.read_parquet(d["index_out"])
    train_df = index[index.split == "train"].reset_index(drop=True)
    val_df = index[index.split == "validation"].reset_index(drop=True)
    if train_df.empty or val_df.empty:
        raise SystemExit(
            f"need both splits in the index; got train={len(train_df)} val={len(val_df)}"
        )
    norm = DeltaNormStats.fit(train_df.delta_days)
    idx_sha = index_sha256(index)
    feat_sha = sha256_file(d["feature_index"])

    train_ds = S3FeaturePairDataset(train_df, traits=tuple(d["traits"]))
    val_ds = S3FeaturePairDataset(val_df, traits=tuple(d["traits"]))
    loader_kwargs = dict(
        batch_size=int(tcfg["batch_size"]),
        num_workers=int(tcfg["num_workers"]),
        pin_memory=True,
        persistent_workers=int(tcfg["num_workers"]) > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kwargs)

    # ---- model / optimiser ----------------------------------------------
    model = build_model(cfg, norm, device)
    n_params = model.n_trainable_parameters()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"]),
    )
    steps_per_epoch = max(len(train_loader), 1)
    total_steps = epochs * steps_per_epoch
    warmup = int(tcfg.get("warmup_steps", 0))

    def lr_lambda(step: int) -> float:
        if warmup > 0 and step < warmup:
            return float(step + 1) / float(warmup)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    amp_context, amp_name = _amp_context(tcfg)
    scaler = torch.amp.GradScaler("cuda") if amp_name == "fp16" else None

    out_dir = Path(args.output_dir or tcfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)
    log_path = out_dir / "metrics.jsonl"
    log_fh = open(log_path, "a")

    start_epoch = 0
    global_step = 0
    best_val = float("inf")
    best_epoch = -1
    history: list[dict] = []
    if args.resume:
        ckpt = load_checkpoint(Path(args.resume), device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if scaler is not None and ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt["epoch"]) + 1
        global_step = int(ckpt["global_step"])
        norm = DeltaNormStats(
            mean=ckpt["delta_norm"]["mean"], std=ckpt["delta_norm"]["std"],
            min=ckpt["delta_norm"]["min"], max=ckpt["delta_norm"]["max"],
            n_samples=ckpt["delta_norm"]["n_samples"],
        )
        model.set_delta_norm_stats(norm)
        history = list(ckpt.get("metrics", {}).get("history", []))
        best_val = float(ckpt.get("metrics", {}).get("best_val_mse", best_val))
        print(f"resumed {args.resume} at epoch {start_epoch}, step {global_step}", flush=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t_train = time.time()
    n_seen = 0

    for epoch in range(start_epoch, epochs):
        model.train()
        ep_loss = 0.0
        ep_batches = 0
        t0 = time.time()
        for batch in train_loader:
            z_a = batch["z_a"].to(device, non_blocking=True)
            z_b = batch["z_b"].to(device, non_blocking=True)
            delta = batch["delta_days"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with (amp_context if amp_context is not None else _nullcontext()):
                z_pred = model(z_a, delta)
                loss = latent_prediction_loss(z_pred.float(), z_b)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg["grad_clip"]))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg["grad_clip"]))
                optimizer.step()
            scheduler.step()
            global_step += 1
            n_seen += z_a.shape[0]
            ep_loss += float(loss.detach())
            ep_batches += 1

            if global_step % int(tcfg["log_every"]) == 0:
                rec = {
                    "type": "step", "epoch": epoch, "step": global_step,
                    "loss": ep_loss / ep_batches,
                    "lr": scheduler.get_last_lr()[0],
                    "elapsed": round(time.time() - t_train, 1),
                    "samples_per_s": round(n_seen / max(time.time() - t_train, 1e-9), 2),
                }
                log_fh.write(json.dumps(rec) + "\n")
                log_fh.flush()
                print(json.dumps(rec), flush=True)
            if args.max_steps and global_step >= args.max_steps:
                break

        val = evaluate(model, val_loader, device, amp_context)
        epoch_rec = {
            "type": "epoch", "epoch": epoch,
            "train_loss": ep_loss / max(ep_batches, 1),
            "val": val,
            "epoch_seconds": round(time.time() - t0, 1),
            "global_step": global_step,
        }
        history.append(epoch_rec)
        log_fh.write(json.dumps(epoch_rec) + "\n")
        log_fh.flush()
        print(json.dumps({
            "type": "epoch", "epoch": epoch,
            "train_loss": round(epoch_rec["train_loss"], 5),
            "val_mse_pred": round(val["mse_pred"], 5),
            "val_mse_persist": round(val["mse_persist"], 5),
            "val_rel_improvement": round(val["relative_improvement"], 4)
            if val["relative_improvement"] is not None else None,
            "val_cos_pred": round(val["cosine_pred"], 5),
            "val_cos_persist": round(val["cosine_persist"], 5),
        }), flush=True)

        metrics_payload = {
            "history": history,
            "best_val_mse": min(best_val, val["mse_pred"]),
            "best_epoch": epoch if val["mse_pred"] < best_val else best_epoch,
        }
        save_checkpoint(
            out_dir / "checkpoints" / "last.pt", model, optimizer, scheduler,
            scaler, epoch, global_step, cfg, norm, idx_sha, feat_sha, metrics_payload,
        )
        if val["mse_pred"] < best_val:
            best_val = val["mse_pred"]
            best_epoch = epoch
            save_checkpoint(
                out_dir / "checkpoints" / "best.pt", model, optimizer, scheduler,
                scaler, epoch, global_step, cfg, norm, idx_sha, feat_sha, metrics_payload,
            )

        if args.max_steps and global_step >= args.max_steps:
            print(f"stopping at max_steps={args.max_steps}", flush=True)
            break
        if tcfg.get("early_stop_patience") and epoch - best_epoch >= int(tcfg["early_stop_patience"]):
            print(f"early stop at epoch {epoch} (best epoch {best_epoch})", flush=True)
            break

    log_fh.close()
    wall = time.time() - t_train
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    summary = {
        "stage": "S3",
        "model": "FIP-DINO-WM V0",
        "manifest_id": cfg["manifest_id"],
        "pilot": bool(cfg.get("pilot", False)),
        "pilot_note": cfg.get("pilot_note", ""),
        "git_commit": git_commit(),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "amp": amp_name,
        "n_trainable_parameters": n_params,
        "epochs_requested": epochs,
        "epochs_completed": len(history),
        "global_step": global_step,
        "train_pairs": int(len(train_ds)),
        "val_pairs": int(len(val_ds)),
        "train_unique_images": int(pd.unique(pd.concat(
            [train_df.image_stem_a, train_df.image_stem_b])).size),
        "val_unique_images": int(pd.unique(pd.concat(
            [val_df.image_stem_a, val_df.image_stem_b])).size),
        "train_plots": int(train_df.plot_uid.nunique()),
        "val_plots": int(val_df.plot_uid.nunique()),
        "batch_size": int(tcfg["batch_size"]),
        "steps_per_epoch": steps_per_epoch,
        "wall_clock_seconds": round(wall, 1),
        "samples_per_second": round(n_seen / max(wall, 1e-9), 2),
        "peak_gpu_memory_bytes": int(peak),
        "delta_norm": norm.as_dict(),
        "index_sha256": idx_sha,
        "feature_index_sha256": feat_sha,
        "best_val_mse": best_val,
        "best_epoch": best_epoch,
        "history": history,
        "checkpoint_last": str(out_dir / "checkpoints" / "last.pt"),
        "checkpoint_best": str(out_dir / "checkpoints" / "best.pt"),
    }
    with open(out_dir / "run_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps({
        k: summary[k] for k in (
            "epochs_completed", "global_step", "train_pairs", "val_pairs",
            "wall_clock_seconds", "samples_per_second", "peak_gpu_memory_bytes",
            "best_val_mse", "best_epoch",
        )
    }, indent=2), flush=True)
    print(f"wrote {out_dir/'run_summary.json'} and {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
