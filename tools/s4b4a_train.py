#!/usr/bin/env python3
"""S4B.4A: train one condition on one fold for one seed.

Reads a frozen fold parquet pair (``fold_train`` / ``fold_dev``), slices the
full-train conditioning arrays by ``_orig_row``, fits every normalisation
statistic on ``fold_train`` rows ONLY, trains with early stopping on
``fold_dev`` (the held-out year is never touched), and saves a checkpoint under
``external_data/s4b4a_runs/fold<Y>/<condition>/seed<k>/``.

The model, dataset and conditioning code are the S4A / S4B.2 implementations,
reused unchanged.  The only new logic is the per-fold slicing and the per-fold
norm fitting.

Usage
-----
    CUDA_VISIBLE_DEVICES=1 python3 tools/s4b4a_train.py \
        --config agro_jepa/config/s4b4a_year_heldout.yaml \
        --fold 2016 --condition b3_delta --seed 0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agro_jepa.datasets.s4_dataset import (  # noqa: E402
    S4PairDataset,
    load_conditioning,
)
from agro_jepa.models.s4_conditioning import VectorNormStats  # noqa: E402
from agro_jepa.models.s4_predictor import (  # noqa: E402
    S4ModelConfig,
    build_s4_predictor,
    conditioning_spec,
    has_fixed_delta,
)
from agro_jepa.models.time_embedding import DeltaNormStats  # noqa: E402


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
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def amp_context(cfg: dict):
    amp = str(cfg.get("amp", "bf16")).lower()
    if amp == "none" or not torch.cuda.is_available():
        return None, amp
    return torch.autocast("cuda", dtype=torch.bfloat16 if amp == "bf16"
                          else torch.float16), amp


class _Null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def forward_batch(model, batch: dict, device: torch.device) -> torch.Tensor:
    parts = getattr(model, "parts", [])
    kwargs = {}
    if "delta" in parts and not has_fixed_delta(model):
        kwargs["delta_days"] = batch["delta_days"].to(device, non_blocking=True)
    if "gdd" in parts:
        kwargs["gdd"] = batch["gdd"].to(device, non_blocking=True)
    if "weather" in parts:
        kwargs["weather"] = batch["weather"].to(device, non_blocking=True)
        kwargs["weather_mask"] = batch["weather_mask"].to(device, non_blocking=True)
    return model(batch["z_a"].to(device, non_blocking=True), **kwargs)


def state_sha256(model) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(np.ascontiguousarray(v.detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def slice_conditioning(cond: dict, orig_rows: np.ndarray) -> dict:
    """Slice every per-row array by original index position, preserving order."""
    out = {}
    n_ref = cond["delta_days"].shape[0]
    for k, v in cond.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n_ref:
            out[k] = v[orig_rows]
        else:
            out[k] = v
    return out


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device, ctx) -> dict:
    model.eval()
    sse_pred = sse_persist = 0.0
    cos_pred = cos_persist = 0.0
    n_el = n_samples = 0
    for batch in loader:
        z_a = batch["z_a"].to(device, non_blocking=True)
        z_b = batch["z_b"].to(device, non_blocking=True)
        with (ctx if ctx is not None else _Null()):
            z_pred = forward_batch(model, batch, device)
        z_pred, z_b, z_a = z_pred.float(), z_b.float(), z_a.float()
        se_pred = (z_pred - z_b).pow(2).mean(dim=(1, 2))
        se_persist = (z_a - z_b).pow(2).mean(dim=(1, 2))
        cos_p = torch.nn.functional.cosine_similarity(
            z_pred.flatten(1), z_b.flatten(1), dim=-1)
        cos_a = torch.nn.functional.cosine_similarity(
            z_a.flatten(1), z_b.flatten(1), dim=-1)
        n = z_a.shape[0]
        k = int(z_a.shape[1] * z_a.shape[2])
        n_el += k * n
        n_samples += n
        sse_pred += float(se_pred.sum()) * k
        sse_persist += float(se_persist.sum()) * k
        cos_pred += float(cos_p.sum())
        cos_persist += float(cos_a.sum())
    denom = max(n_el, 1)
    return {
        "n_samples": int(n_samples),
        "mse_pred": sse_pred / denom,
        "mse_persist": sse_persist / denom,
        "relative_improvement": (sse_persist - sse_pred) / sse_persist
        if sse_persist > 0 else None,
        "cosine_pred": cos_pred / max(n_samples, 1),
        "cosine_persist": cos_persist / max(n_samples, 1),
    }


def fit_norms(spec, train_df: pd.DataFrame, cond_train: dict, condition: dict):
    """Fit DeltaNormStats / GDD / weather stats on fold_train rows only."""
    delta_norm = DeltaNormStats.fit(train_df.delta_days.to_numpy())
    gdd_norm = weather_norm = None
    if spec.gdd_mode != "none":
        cols = [cond_train[k] for k in spec.gdd_keys()]
        gdd_norm = VectorNormStats.fit(np.stack(cols, axis=1))
    if spec.weather_mode != "none":
        w_key, m_key = spec.weather_keys()
        w = cond_train[w_key]
        m = cond_train[m_key]
        weather_norm = VectorNormStats.fit(w[m])
    fixed = None
    if spec.delta_mode == "constant":
        value = float(condition["fixed_delta_days"])
        spec = replace(spec, fixed_delta_days=value)
        fixed = {
            "fixed_delta_rule": "frozen_s4b4a_constant",
            "fixed_delta_days": value,
            "fixed_delta_source": "config (S4A.1 frozen control value 9.0 d)",
            "fold_train_median": float(train_df.delta_days.median()),
            "fold_train_mean": float(train_df.delta_days.mean()),
            "fold_train_norm_mean": float(delta_norm.mean),
            "fold_train_norm_std": float(delta_norm.std),
            "fixed_delta_normalised_value": float(
                (value - delta_norm.mean) / delta_norm.std),
        }
    return delta_norm, gdd_norm, weather_norm, spec, fixed


def save_checkpoint(path: Path, model, optimizer, scheduler, scaler, epoch, step,
                    cfg, condition_name, seed, spec, norms, metrics,
                    extra: dict | None = None,
                    include_optimizer: bool = False) -> None:
    """Save the evaluation checkpoint.

    ``include_optimizer=False`` (the default) drops the Adam moments: the
    artifact is only ever reloaded for inference by E1/E2, and the moments
    triple the file size (54M-param model -> 550 MB with state, 220 MB without).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    delta_norm, gdd_norm, weather_norm = norms
    payload = {
        "format": "agro-jepa-s4b4a-state-dict-v1",
        "stage": "S4B.4A",
        "manifest_id": cfg.get("manifest_id"),
        "condition": condition_name,
        "conditioning_spec": spec.as_dict(),
        "seed": int(seed),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if include_optimizer else None,
        "scheduler": scheduler.state_dict() if include_optimizer else None,
        "scaler": (scaler.state_dict()
                   if (include_optimizer and scaler is not None) else None),
        "epoch": int(epoch),
        "global_step": int(step),
        "config": cfg,
        "model_config": model.config.as_dict(),
        "delta_norm": delta_norm.as_dict() if delta_norm else None,
        "gdd_norm": gdd_norm.as_dict() if gdd_norm else None,
        "weather_norm": weather_norm.as_dict() if weather_norm else None,
        "fixed_delta": extra,
        "n_trainable_parameters": int(model.n_trainable_parameters()),
        "n_condition_parameters": int(model.n_condition_parameters()),
        "predictor_input_dim": int(model.config.predictor_dim(model.has_condition)),
        "git_commit": git_commit(),
        "metrics": metrics,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    torch.save(payload, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s4b4a_year_heldout.yaml")
    ap.add_argument("--fold", type=int, required=True,
                    help="held-out harvest year, e.g. 2016")
    ap.add_argument("--condition", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None,
                    help="override candidate epochs (pilot/smoke only)")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--limit-train", type=int, default=0,
                    help="pilot/smoke only: cap fold-train rows")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    if args.condition not in cfg["conditions"]:
        raise SystemExit(f"unknown condition {args.condition!r}")
    condition = cfg["conditions"][args.condition]
    if condition.get("mode") != "trained":
        raise SystemExit(f"{args.condition!r} is not trainable")

    tcfg = cfg["training"]
    device = torch.device(tcfg.get("device", "cuda:0"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")
    set_seed(int(args.seed))
    epochs = int(args.epochs if args.epochs is not None
                 else tcfg["candidate_epochs"])

    fold_dir = ROOT / cfg["artifacts"]["fold_dir"]
    tr_path = fold_dir / f"fold{args.fold}_fold_train.parquet"
    dv_path = fold_dir / f"fold{args.fold}_fold_dev.parquet"
    if not tr_path.is_file() or not dv_path.is_file():
        raise SystemExit(f"fold files missing for {args.fold}; run s4b4a_protocol.py")
    train_df = pd.read_parquet(tr_path)
    dev_df = pd.read_parquet(dv_path)
    if args.limit_train:
        train_df = train_df.head(int(args.limit_train)).reset_index(drop=True)

    # conditioning, sliced to the fold rows in this frame's order
    cond_npz = np.load(ROOT / cfg["data"]["conditioning_npz"], allow_pickle=True)
    cond_full = {k: cond_npz[k] for k in cond_npz.files}
    cond_train = slice_conditioning(cond_full, train_df["_orig_row"].to_numpy(np.int64))
    cond_dev = slice_conditioning(cond_full, dev_df["_orig_row"].to_numpy(np.int64))

    spec = conditioning_spec(condition)
    delta_norm, gdd_norm, weather_norm, spec, fixed_delta = fit_norms(
        spec, train_df, cond_train, condition)
    if fixed_delta is not None:
        print(f"[fixed delta] {fixed_delta}", flush=True)

    traits = tuple(cfg["data"].get("traits", [])) or ()
    train_ds = S4PairDataset(train_df, cond_train, spec, traits, load_features=True)
    dev_ds = S4PairDataset(dev_df, cond_dev, spec, traits, load_features=True)

    nw = int(tcfg["num_workers"])
    common = dict(batch_size=int(tcfg["batch_size"]), num_workers=nw,
                  pin_memory=True, persistent_workers=nw > 0)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
    dev_loader = DataLoader(dev_ds, shuffle=False, drop_last=False, **common)

    model_cfg = S4ModelConfig.from_dict(cfg["model"])
    model = build_s4_predictor(
        model_cfg, spec,
        delta_norm=delta_norm if spec.use_delta else None,
        gdd_norm=gdd_norm, weather_norm=weather_norm).to(device)
    from agro_jepa.models.fip_dino_wm import move_attention_masks
    move_attention_masks(model.predictor, device)
    init_state_sha = state_sha256(model)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"]))
    steps_per_epoch = max(len(train_loader), 1)
    total_steps = epochs * steps_per_epoch
    warmup = int(tcfg.get("warmup_steps", 0))

    def lr_lambda(step: int) -> float:
        if warmup > 0 and step < warmup:
            return float(step + 1) / float(warmup)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    ctx, amp_name = amp_context(tcfg)
    scaler = torch.amp.GradScaler("cuda") if amp_name == "fp16" else None

    out_dir = Path(args.output_dir or (
        Path(tcfg["output_root"]) / f"fold{args.fold}" / args.condition
        / f"seed{args.seed}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)
    log_fh = open(out_dir / "metrics.jsonl", "a")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    best_dev = float("inf")
    best_epoch = -1
    best_state = None
    history: list[dict] = []
    n_seen = 0
    global_step = 0
    patience = int(tcfg["early_stop_patience"])

    for epoch in range(epochs):
        model.train()
        ep_loss, ep_batches = 0.0, 0
        te = time.time()
        for batch in train_loader:
            z_a = batch["z_a"].to(device, non_blocking=True)
            z_b = batch["z_b"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with (ctx if ctx is not None else _Null()):
                z_pred = forward_batch(model, batch, device)
                loss = torch.nn.functional.mse_loss(z_pred.float(), z_b.detach())
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               float(tcfg["grad_clip"]))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               float(tcfg["grad_clip"]))
                optimizer.step()
            scheduler.step()
            global_step += 1
            n_seen += z_a.shape[0]
            ep_loss += float(loss.detach())
            ep_batches += 1
            if global_step % int(tcfg["log_every"]) == 0:
                rec = {"type": "step", "condition": args.condition, "seed": args.seed,
                       "fold": args.fold, "epoch": epoch, "step": global_step,
                       "loss": ep_loss / ep_batches,
                       "lr": scheduler.get_last_lr()[0],
                       "elapsed": round(time.time() - t0, 1),
                       "samples_per_s": round(n_seen / max(time.time() - t0, 1e-9), 2)}
                log_fh.write(json.dumps(rec) + "\n")
                log_fh.flush()
            if args.max_steps and global_step >= args.max_steps:
                break

        dev = evaluate(model, dev_loader, device, ctx)
        epoch_rec = {"type": "epoch", "condition": args.condition, "seed": args.seed,
                     "fold": args.fold, "epoch": epoch,
                     "train_loss": ep_loss / max(ep_batches, 1), "fold_dev": dev,
                     "epoch_seconds": round(time.time() - te, 1),
                     "global_step": global_step}
        history.append(epoch_rec)
        log_fh.write(json.dumps(epoch_rec) + "\n")
        log_fh.flush()
        print(json.dumps({"fold": args.fold, "condition": args.condition,
                          "seed": args.seed, "epoch": epoch,
                          "train_loss": round(epoch_rec["train_loss"], 5),
                          "dev_mse": round(dev["mse_pred"], 5),
                          "dev_rel": round(dev["relative_improvement"], 4),
                          "sec": epoch_rec["epoch_seconds"]}), flush=True)
        if dev["mse_pred"] < best_dev - 1e-9:
            best_dev = dev["mse_pred"]
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if args.max_steps and global_step >= args.max_steps:
            break
        if epoch - best_epoch >= patience:
            print(f"early stop at epoch {epoch} (best {best_epoch})", flush=True)
            break

    log_fh.close()
    wall = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    save_checkpoint(out_dir / "checkpoints" / "best.pt", model, optimizer,
                    scheduler, scaler, best_epoch, global_step, cfg, args.condition,
                    args.seed, spec, (delta_norm, gdd_norm, weather_norm),
                    {"history": history, "best_dev_mse": best_dev,
                     "best_epoch": best_epoch, "init_state_sha256": init_state_sha},
                    extra=fixed_delta)
    summary = {
        "stage": "S4B.4A",
        "fold": int(args.fold),
        "condition": args.condition,
        "condition_description": condition.get("description"),
        "protocol": condition.get("protocol"),
        "conditioning_spec": spec.as_dict(),
        "seed": int(args.seed),
        "manifest_id": cfg["manifest_id"],
        "git_commit": git_commit(),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "amp": amp_name,
        "n_trainable_parameters": int(model.n_trainable_parameters()),
        "n_condition_parameters": int(model.n_condition_parameters()),
        "predictor_input_dim": int(model.config.predictor_dim(model.has_condition)),
        "epochs_requested": epochs,
        "epochs_completed": len(history),
        "global_step": global_step,
        "fold_train_pairs": int(len(train_ds)),
        "fold_dev_pairs": int(len(dev_ds)),
        "fold_train_plots": int(train_df.plot_uid.nunique()),
        "fold_dev_plots": int(dev_df.plot_uid.nunique()),
        "batch_size": int(tcfg["batch_size"]),
        "steps_per_epoch": steps_per_epoch,
        "wall_clock_seconds": round(wall, 1),
        "samples_per_second": round(n_seen / max(wall, 1e-9), 2),
        "peak_gpu_memory_bytes": int(peak),
        "selection_metric": "fold_dev_mse_pred",
        "selection_rule": "minimum fold-dev future-latent MSE over the candidate budget",
        "heldout_used_for_selection": False,
        "delta_norm": delta_norm.as_dict(),
        "gdd_norm": gdd_norm.as_dict() if gdd_norm else None,
        "weather_norm": weather_norm.as_dict() if weather_norm else None,
        "best_fold_dev_mse": best_dev,
        "best_epoch": best_epoch,
        "init_state_sha256": init_state_sha,
        "fixed_delta": fixed_delta,
        "history": history,
        "checkpoint_best": str(out_dir / "checkpoints" / "best.pt"),
    }
    with open(out_dir / "run_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps({k: summary[k] for k in (
        "fold", "condition", "seed", "epochs_completed", "global_step",
        "fold_train_pairs", "fold_dev_pairs", "wall_clock_seconds",
        "samples_per_second", "peak_gpu_memory_bytes",
        "n_trainable_parameters", "best_epoch", "best_fold_dev_mse")}, indent=2),
        flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
