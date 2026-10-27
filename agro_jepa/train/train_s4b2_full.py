#!/usr/bin/env python3
"""S4B.2: one condition, one seed, full official-train, fixed epoch budget.

Differences from the S4A pilot trainer (train_s4.py):
* the entire 179,823-pair official-train index is used (no 80/20 inner split);
* no inner-dev set and therefore no early stopping -- training runs for the
  frozen number of epochs and saves ``final.pt`` at the end;
* cosine schedule total steps = full-DataLoader length * fixed epochs;
* no test index, no test label, no weather/GDD is ever read here.

The model/dataset/conditioning building blocks are reused verbatim from
agro_jepa.models.{s4_predictor,time_embedding} and
agro_jepa.datasets.s4_dataset.

Usage
-----
    CUDA_VISIBLE_DEVICES=1 python3 agro_jepa/train/train_s4b2_full.py \
        --config agro_jepa/config/s4b2_full_train.yaml \
        --condition b3_delta --seed 0 --epochs 12
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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

from agro_jepa.datasets.s4_dataset import (  # noqa: E402
    ConditioningSpec,
    S4PairDataset,
)
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
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast("cuda", dtype=dtype), amp


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
    return model(batch["z_a"].to(device, non_blocking=True), **kwargs)


def state_sha256(model) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(np.ascontiguousarray(v.detach().cpu().numpy()).tobytes())
    return h.hexdigest()


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
        cos_p = torch.nn.functional.cosine_similarity(z_pred.flatten(1), z_b.flatten(1), dim=-1)
        cos_a = torch.nn.functional.cosine_similarity(z_a.flatten(1), z_b.flatten(1), dim=-1)
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
        "relative_improvement": (sse_persist - sse_pred) / sse_persist if sse_persist > 0 else None,
        "cosine_pred": cos_pred / max(n_samples, 1),
        "cosine_persist": cos_persist / max(n_samples, 1),
    }


def make_spec(condition: dict) -> ConditioningSpec:
    return conditioning_spec(condition)


def save_checkpoint(path: Path, model, optimizer, scheduler, scaler, epoch, step,
                    cfg, condition_name, seed, spec, delta_norm, metrics,
                    extra=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "agro-jepa-s4b2-state-dict-v1",
        "stage": "S4B.2",
        "manifest_id": cfg.get("manifest_id"),
        "condition": condition_name,
        "conditioning_spec": spec.as_dict(),
        "seed": int(seed),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "global_step": int(step),
        "config": cfg,
        "model_config": model.config.as_dict(),
        "delta_norm": delta_norm.as_dict() if delta_norm else None,
        "n_trainable_parameters": int(model.n_trainable_parameters()),
        "n_condition_parameters": int(model.n_condition_parameters()),
        "predictor_input_dim": int(model.config.predictor_dim(model.has_condition)),
        "fixed_delta": extra,
        "index_sha256": cfg["_index_sha256"],
        "feature_index_sha256": cfg["_feature_index_sha256"],
        "git_commit": git_commit(),
        "metrics": metrics,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    torch.save(payload, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s4b2_full_train.yaml")
    ap.add_argument("--condition", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--resume", action="store_true",
                    help="continue from last.pt (same config/index/seed only)")
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke only: cap index rows (0 = full train index)")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="smoke only: stop after N optimizer steps")
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
                 else cfg["epoch_policy"]["conditions"][args.condition]["fixed_epochs"])

    # ---- frozen inputs ----------------------------------------------------
    index = pd.read_parquet(cfg["data"]["train_index"])
    if args.limit:
        index = index.head(int(args.limit)).reset_index(drop=True)
    idx_sha = hashlib.sha256(
        index[sorted(index.columns)].to_csv(index=False).encode()).hexdigest()
    feat_sha = sha256_file(cfg["data"]["feature_index"])

    spec = make_spec(condition)
    delta_norm = DeltaNormStats.fit(index["delta_days"].to_numpy())
    # B2-C frozen 9.0 d, shared normaliser
    fixed_delta = None
    if spec.delta_mode == "constant":
        requested = float(condition["fixed_delta_days"])
        fixed_delta = {
            "fixed_delta_rule": "frozen_s4a1_constant",
            "fixed_delta_days": requested,
            "fixed_delta_source": "S4A.1 frozen control value (not re-derived)",
            "shared_delta_norm_mean": delta_norm.mean,
            "shared_delta_norm_std": delta_norm.std,
            "fixed_delta_normalised_value": (requested - delta_norm.mean) / delta_norm.std,
        }
        spec = condition_spec_with_fixed(spec, requested)

    traits = tuple(cfg["data"].get("traits", [])) or ()
    ds = S4PairDataset(index, {}, spec, traits, load_features=True)
    nw = int(tcfg["num_workers"])
    common = dict(batch_size=int(tcfg["batch_size"]), num_workers=nw,
                  pin_memory=True, persistent_workers=nw > 0)
    loader = DataLoader(ds, shuffle=bool(tcfg.get("shuffle", True)),
                        drop_last=bool(tcfg.get("drop_last", True)), **common)

    model_cfg = S4ModelConfig.from_dict(cfg["model"])
    model = build_s4_predictor(
        model_cfg, spec,
        delta_norm=delta_norm if spec.use_delta else None).to(device)
    from agro_jepa.models.fip_dino_wm import move_attention_masks
    move_attention_masks(model.predictor, device)
    init_state_sha = state_sha256(model)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"]))
    steps_per_epoch = max(len(loader), 1)
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

    out_dir = Path(args.output_dir or Path(tcfg["output_root"]) / args.condition / f"seed{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)
    log_fh = open(out_dir / "metrics.jsonl", "a")

    cfg = dict(cfg)
    cfg["_index_sha256"] = idx_sha
    cfg["_feature_index_sha256"] = feat_sha

    start_epoch = 0
    global_step = 0
    if args.resume:
        last = out_dir / "checkpoints" / "last.pt"
        if last.is_file():
            ck = torch.load(last, map_location="cpu")
            model.load_state_dict(ck["model"])
            optimizer.load_state_dict(ck["optimizer"])
            scheduler.load_state_dict(ck["scheduler"])
            if scaler is not None and ck.get("scaler"):
                scaler.load_state_dict(ck["scaler"])
            start_epoch = int(ck["epoch"]) + 1
            global_step = int(ck["global_step"])
            print(f"[resume] from epoch {start_epoch}, step {global_step}", flush=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    history: list[dict] = []
    n_seen = 0

    for epoch in range(start_epoch, epochs):
        model.train()
        ep_loss, ep_batches = 0.0, 0
        te = time.time()
        for batch in loader:
            z_a = batch["z_a"].to(device, non_blocking=True)
            z_b = batch["z_b"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with (ctx if ctx is not None else _Null()):
                z_pred = forward_batch(model, batch, device)
                loss = torch.nn.functional.mse_loss(z_pred.float(), z_b.detach())
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
                rec = {"type": "step", "condition": args.condition, "seed": args.seed,
                       "epoch": epoch, "step": global_step,
                       "loss": ep_loss / ep_batches,
                       "lr": scheduler.get_last_lr()[0],
                       "elapsed": round(time.time() - t0, 1),
                       "samples_per_s": round(n_seen / max(time.time() - t0, 1e-9), 2)}
                log_fh.write(json.dumps(rec) + "\n")
                log_fh.flush()
            if args.max_steps and global_step >= args.max_steps:
                break

        epoch_rec = {"type": "epoch", "condition": args.condition, "seed": args.seed,
                     "epoch": epoch, "train_loss": ep_loss / max(ep_batches, 1),
                     "epoch_seconds": round(time.time() - te, 1),
                     "global_step": global_step}
        history.append(epoch_rec)
        log_fh.write(json.dumps(epoch_rec) + "\n")
        log_fh.flush()
        print(json.dumps({"condition": args.condition, "seed": args.seed, "epoch": epoch,
                          "train_loss": round(epoch_rec["train_loss"], 5),
                          "sec": epoch_rec["epoch_seconds"]}), flush=True)
        save_checkpoint(out_dir / "checkpoints" / "last.pt", model, optimizer,
                        scheduler, scaler, epoch, global_step, cfg, args.condition,
                        args.seed, spec, delta_norm,
                        {"history": history, "init_state_sha256": init_state_sha},
                        extra=fixed_delta)
        if args.max_steps and global_step >= args.max_steps:
            print("smoke: stopping after max-steps", flush=True)
            break

    log_fh.close()
    wall = time.time() - t0
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    save_checkpoint(out_dir / "checkpoints" / "final.pt", model, optimizer,
                    scheduler, scaler, epochs - 1, global_step, cfg, args.condition,
                    args.seed, spec, delta_norm,
                    {"history": history, "init_state_sha256": init_state_sha},
                    extra=fixed_delta)
    final_sha = sha256_file(out_dir / "checkpoints" / "final.pt")
    last_sha = sha256_file(out_dir / "checkpoints" / "last.pt")

    summary = {
        "stage": "S4B.2",
        "condition": args.condition,
        "condition_description": condition.get("description"),
        "conditioning_spec": spec.as_dict(),
        "seed": int(args.seed),
        "manifest_id": cfg["manifest_id"],
        "git_commit": git_commit(),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "amp": amp_name,
        "n_trainable_parameters": int(model.n_trainable_parameters()),
        "n_condition_parameters": int(model.n_condition_parameters()),
        "predictor_input_dim": int(model.config.predictor_dim(model.has_condition)),
        "epochs_requested": epochs,
        "epochs_completed": len(history),
        "global_step": global_step,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "train_pairs": int(len(ds)),
        "train_plots": int(index.plot_uid.nunique()),
        "train_unique_images": int(pd.unique(pd.concat(
            [index.image_stem_a, index.image_stem_b])).size),
        "batch_size": int(tcfg["batch_size"]),
        "warmup_steps": warmup,
        "total_steps_schedule": total_steps,
        "lr": float(tcfg["lr"]),
        "weight_decay": float(tcfg["weight_decay"]),
        "grad_clip": float(tcfg["grad_clip"]),
        "wall_clock_seconds": round(wall, 1),
        "samples_per_second": round(n_seen / max(wall, 1e-9), 2),
        "peak_gpu_memory_bytes": int(peak),
        "early_stopping": bool(tcfg.get("early_stop", False)),
        "official_validation_used_for_selection": False,
        "delta_norm": delta_norm.as_dict(),
        "fixed_delta": fixed_delta,
        "index_sha256": idx_sha,
        "feature_index_sha256": feat_sha,
        "init_state_sha256": init_state_sha,
        "init_rule": (
            "torch.manual_seed(seed) (set_seed) immediately before build_s4_predictor; "
            "no weight loaded from any earlier run"
        ),
        "history": history,
        "checkpoint_final": str(out_dir / "checkpoints" / "final.pt"),
        "checkpoint_final_sha256": final_sha,
        "checkpoint_last": str(out_dir / "checkpoints" / "last.pt"),
        "checkpoint_last_sha256": last_sha,
    }
    with open(out_dir / "run_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w") as fh:
            json.dump(summary, fh, indent=2)
    print(json.dumps({k: summary[k] for k in (
        "condition", "seed", "epochs_completed", "global_step", "train_pairs",
        "wall_clock_seconds", "peak_gpu_memory_bytes", "n_trainable_parameters",
        "checkpoint_final_sha256")}, indent=2), flush=True)
    print(f"wrote {out_dir/'run_summary.json'}", flush=True)
    return 0


def condition_spec_with_fixed(spec: ConditioningSpec, value: float) -> ConditioningSpec:
    from dataclasses import replace
    return replace(spec, fixed_delta_days=value)


if __name__ == "__main__":
    sys.exit(main())
