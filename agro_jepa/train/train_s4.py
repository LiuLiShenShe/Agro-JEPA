#!/usr/bin/env python3
"""S4A training: one condition, one seed, one fixed budget.

Every trainable condition uses the *same* protocol:

* official-train plots only, split 80/20 by plot, stratified by harvest_year;
* normalisation statistics fitted on the inner-train plots only;
* model selection = minimum inner-dev future-latent MSE (patience 3);
* the official validation split is *never* read during training;
* identical visual target ``stopgrad(z_b)``, optimiser, LR schedule, AMP,
  batch size, candidate epoch budget and seed set.

Usage
-----
    CUDA_VISIBLE_DEVICES=1 python3 agro_jepa/train/train_s4.py \
        --config agro_jepa/config/s4_pilot.yaml \
        --condition b3_delta --seed 0
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agro_jepa.datasets.s3_pairs import index_sha256  # noqa: E402
from agro_jepa.datasets.s4_dataset import (  # noqa: E402
    S4PairDataset,
    build_role_map,
    conditioning_rows,
    load_conditioning,
    split_rows,
)
from agro_jepa.models.s4_conditioning import VectorNormStats  # noqa: E402
from agro_jepa.models.s4_predictor import (  # noqa: E402
    S4ModelConfig,
    build_s4_predictor,
    conditioning_spec,
    has_fixed_delta,
)
from agro_jepa.models.time_embedding import DeltaNormStats  # noqa: E402


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
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def amp_context(cfg: dict):
    amp = str(cfg.get("amp", "bf16")).lower()
    if amp == "none" or not torch.cuda.is_available():
        return None, amp
    return torch.autocast("cuda", dtype=torch.bfloat16 if amp == "bf16" else torch.float16), amp


class _Null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def forward_batch(model, batch: dict, device: torch.device) -> torch.Tensor:
    """Pass only the inputs the model actually consumes.

    ``delta_days`` is present in every sample as metadata, but it is forwarded
    only when the conditioning spec includes the delta channel -- so
    ``b2_no_time`` genuinely never receives it.  For ``delta_mode="constant"``
    the model supplies its own fixed interval and the batch value is not even
    handed over.
    """
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
    """Digest of a state dict (sorted keys, CPU bytes) for initialization audit."""
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(np.ascontiguousarray(
            v.detach().cpu().numpy()).tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device, ctx) -> dict:
    model.eval()
    sse_pred = sse_persist = 0.0
    cos_pred = cos_persist = 0.0
    n_el = n_samples = 0
    by_span: dict[str, dict] = {}
    by_year: dict[str, dict] = {}
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
        n_el += int(z_a.shape[1] * z_a.shape[2] * n)
        n_samples += n
        sse_pred += float(se_pred.sum()) * int(z_a.shape[1] * z_a.shape[2])
        sse_persist += float(se_persist.sum()) * int(z_a.shape[1] * z_a.shape[2])
        cos_pred += float(cos_p.sum())
        cos_persist += float(cos_a.sum())
        for i in range(n):
            for table, key in (
                (by_span, str(batch["pair_type"][i])),
                (by_year, str(int(batch["harvest_year"][i]))),
            ):
                row = table.setdefault(key, {"sse_pred": 0.0, "sse_persist": 0.0,
                                             "n_el": 0, "n": 0,
                                             "cos_pred": 0.0, "cos_persist": 0.0})
                k = int(z_a.shape[1] * z_a.shape[2])
                row["sse_pred"] += float(se_pred[i]) * k
                row["sse_persist"] += float(se_persist[i]) * k
                row["n_el"] += k
                row["n"] += 1
                row["cos_pred"] += float(cos_p[i])
                row["cos_persist"] += float(cos_a[i])

    def fin(d):
        denom = max(d["n_el"], 1)
        mp = d["sse_pred"] / denom
        ma = d["sse_persist"] / denom
        return {
            "n_samples": d["n"],
            "mse_pred": mp,
            "mse_persist": ma,
            "relative_improvement": (ma - mp) / ma if ma > 0 else None,
            "cosine_pred": d["cos_pred"] / max(d["n"], 1),
            "cosine_persist": d["cos_persist"] / max(d["n"], 1),
        }

    out = fin({"sse_pred": sse_pred, "sse_persist": sse_persist,
               "n_el": n_el, "n": n_samples,
               "cos_pred": cos_pred, "cos_persist": cos_persist})
    out["by_span"] = {k: fin(v) for k, v in sorted(by_span.items())}
    out["by_year"] = {k: fin(v) for k, v in sorted(by_year.items())}
    return out


# --------------------------------------------------------------------------
def fit_norms(condition: dict, index: pd.DataFrame, cond: dict, role_map: dict):
    """Fit every normalisation statistic on inner-train rows only.

    For ``delta_mode="constant"`` the fixed interval is also derived here, from
    the *same* inner-train rows, and must equal the value pre-registered in the
    config.  Nothing from the inner-dev or official-validation rows is used.
    """
    inner_plots = np.array(
        [p for p, r in role_map.items() if r == "inner_train"], dtype=object
    )
    train_rows = index.index[index.split == "train"].to_numpy()
    inner_rows = train_rows[
        np.isin(index.loc[train_rows, "plot_uid"].astype(str).to_numpy(), inner_plots)
    ]
    if len(inner_rows) == 0:
        raise SystemExit("no inner-train rows")
    delta_norm = DeltaNormStats.fit(index.loc[inner_rows, "delta_days"].to_numpy())
    spec = conditioning_spec(condition)

    gdd_norm = None
    if spec.gdd_mode != "none":
        cols = [cond[k][inner_rows] for k in spec.gdd_keys()]
        gdd_norm = VectorNormStats.fit(np.stack(cols, axis=1))

    weather_norm = None
    if spec.weather_mode != "none":
        w_key, m_key = spec.weather_keys()
        w = cond[w_key][inner_rows]
        m = cond[m_key][inner_rows]
        weather_norm = VectorNormStats.fit(w[m])

    fixed = None
    if spec.delta_mode == "constant":
        rule = str(condition.get("fixed_delta_rule", "inner_train_median"))
        vals = index.loc[inner_rows, "delta_days"].to_numpy(np.float64)
        if rule == "inner_train_median":
            value = float(np.median(vals))
        elif rule == "inner_train_mean":
            value = float(vals.mean())
        else:
            raise SystemExit(f"unknown fixed_delta_rule {rule!r}")
        requested = condition.get("fixed_delta_days")
        if requested is not None and abs(float(requested) - value) > 1e-9:
            raise SystemExit(
                f"config fixed_delta_days={requested} but the inner-train {rule} "
                f"is {value}; refusing to train a mis-specified control"
            )
        spec = replace(spec, fixed_delta_days=value)
        fixed = {
            "fixed_delta_rule": rule,
            "fixed_delta_days": value,
            "fixed_delta_source": "inner_train rows of this model's split only",
            "fixed_delta_n_source_rows": int(len(inner_rows)),
            "fixed_delta_inner_train_mean": float(vals.mean()),
            "fixed_delta_inner_train_median": float(np.median(vals)),
            "fixed_delta_inner_train_min": float(vals.min()),
            "fixed_delta_inner_train_max": float(vals.max()),
            "fixed_delta_config_expected": (
                float(requested) if requested is not None else None
            ),
            "fixed_delta_normalised_value": float(
                (value - delta_norm.mean) / delta_norm.std
            ),
        }
    return delta_norm, gdd_norm, weather_norm, spec, int(len(inner_rows)), fixed


def save_checkpoint(path: Path, model, optimizer, scheduler, scaler, epoch, step,
                    cfg, condition_name, seed, spec, norms, metrics,
                    extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    delta_norm, gdd_norm, weather_norm = norms
    payload = {
        "format": "agro-jepa-s4a-state-dict-v1",
        "stage": "S4A",
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
        "gdd_norm": gdd_norm.as_dict() if gdd_norm else None,
        "weather_norm": weather_norm.as_dict() if weather_norm else None,
        "fixed_delta": extra,
        "n_trainable_parameters": int(model.n_trainable_parameters()),
        "n_condition_parameters": int(model.n_condition_parameters()),
        "predictor_input_dim": int(model.config.predictor_dim(model.has_condition)),
        "index_sha256": cfg["_index_sha256"],
        "conditioning_sha256": cfg["_conditioning_sha256"],
        "feature_index_sha256": cfg["_feature_index_sha256"],
        "git_commit": git_commit(),
        "metrics": metrics,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    torch.save(payload, path)


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s4_pilot.yaml")
    ap.add_argument("--condition", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--deterministic-data", action="store_true")
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    if args.condition not in cfg["conditions"]:
        raise SystemExit(f"unknown condition {args.condition!r}")
    condition = cfg["conditions"][args.condition]
    if condition.get("mode") != "trained":
        raise SystemExit(f"condition {args.condition!r} is a baseline, not trainable")

    tcfg = cfg["training"]
    device = torch.device(tcfg.get("device", "cuda:0"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")
    set_seed(int(args.seed))
    epochs = int(args.epochs if args.epochs is not None else tcfg["candidate_epochs"])

    # ---- frozen inputs ----------------------------------------------------
    index = pd.read_parquet(cfg["data"]["index"])
    idx_sha = index_sha256(index)
    if idx_sha != cfg["frozen"]["index_sha256"]:
        raise SystemExit(
            f"frozen index digest {idx_sha} != {cfg['frozen']['index_sha256']}"
        )
    cond_path = Path(cfg["data"]["conditioning_npz"])
    cond_sha = sha256_file(cond_path)
    feat_sha = sha256_file(cfg["frozen"]["feature_index"])
    cond = load_conditioning(cond_path, index)
    if str(cond["index_sha256"]) != idx_sha:
        raise SystemExit("conditioning npz was built from a different index")

    role_map, split_stats = build_role_map(
        index, int(cfg["split"]["seed"]), float(cfg["split"]["inner_train_fraction"])
    )
    delta_norm, gdd_norm, weather_norm, spec, n_inner, fixed_delta = fit_norms(
        condition, index, cond, role_map
    )
    if fixed_delta is not None:
        print(f"[fixed delta] {fixed_delta}", flush=True)

    train_df = split_rows(index, role_map, "inner_train")
    dev_df = split_rows(index, role_map, "inner_dev")
    if len(train_df) == 0 or len(dev_df) == 0:
        raise SystemExit("empty inner train/dev split")

    traits = tuple(cfg["data"]["traits"])
    train_ds = S4PairDataset(train_df, _subset(cond, conditioning_rows(train_df)),
                             spec, traits)
    dev_ds = S4PairDataset(dev_df, _subset(cond, conditioning_rows(dev_df)),
                           spec, traits)
    nw = int(tcfg["num_workers"])
    common = dict(batch_size=int(tcfg["batch_size"]), num_workers=nw,
                  pin_memory=True, persistent_workers=nw > 0)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
    dev_loader = DataLoader(dev_ds, shuffle=False, drop_last=False, **common)

    # ---- model ------------------------------------------------------------
    model_cfg = S4ModelConfig.from_dict(cfg["model"])
    if spec.weather_mode != "none":
        if int(model_cfg.weather.n_channels) != len(cfg["data"]["weather_channels"]):
            raise SystemExit(
                "weather channel count mismatch: "
                f"model {model_cfg.weather.n_channels} vs data "
                f"{len(cfg['data']['weather_channels'])}"
            )
    model = build_s4_predictor(
        model_cfg, spec,
        delta_norm=delta_norm if spec.use_delta else None,
        gdd_norm=gdd_norm,
        weather_norm=weather_norm,
    ).to(device)
    from agro_jepa.models.fip_dino_wm import move_attention_masks

    move_attention_masks(model.predictor, device)
    init_state_sha = state_sha256(model)

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
    ctx, amp_name = amp_context(tcfg)
    scaler = torch.amp.GradScaler("cuda") if amp_name == "fp16" else None

    out_dir = Path(args.output_dir or Path(tcfg["output_root"]) / args.condition / f"seed{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)
    log_fh = open(out_dir / "metrics.jsonl", "a")

    cfg = dict(cfg)
    cfg["_index_sha256"] = idx_sha
    cfg["_conditioning_sha256"] = cond_sha
    cfg["_feature_index_sha256"] = feat_sha

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

        dev = evaluate(model, dev_loader, device, ctx)
        epoch_rec = {"type": "epoch", "condition": args.condition, "seed": args.seed,
                     "epoch": epoch, "train_loss": ep_loss / max(ep_batches, 1),
                     "inner_dev": dev, "epoch_seconds": round(time.time() - te, 1),
                     "global_step": global_step}
        history.append(epoch_rec)
        log_fh.write(json.dumps(epoch_rec) + "\n")
        log_fh.flush()
        print(json.dumps({"condition": args.condition, "seed": args.seed, "epoch": epoch,
                          "train_loss": round(epoch_rec["train_loss"], 5),
                          "dev_mse": round(dev["mse_pred"], 5),
                          "dev_mse_persist": round(dev["mse_persist"], 5),
                          "dev_rel_impr": round(dev["relative_improvement"], 4),
                          "sec": epoch_rec["epoch_seconds"]}), flush=True)
        if dev["mse_pred"] < best_dev - 1e-9:
            best_dev = dev["mse_pred"]
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            save_checkpoint(out_dir / "checkpoints" / "best.pt", model, optimizer,
                            scheduler, scaler, epoch, global_step, cfg, args.condition,
                            args.seed, spec, (delta_norm, gdd_norm, weather_norm),
                            {"history": history, "best_dev_mse": best_dev,
                             "best_epoch": best_epoch,
                             "init_state_sha256": init_state_sha},
                            extra=fixed_delta)
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
    summary = {
        "stage": "S4A",
        "condition": args.condition,
        "condition_description": condition.get("description"),
        "protocol": condition.get("protocol"),
        "conditioning_spec": spec.as_dict(),
        "seed": int(args.seed),
        "manifest_id": cfg["manifest_id"],
        "pilot": bool(cfg.get("pilot", False)),
        "pilot_note": cfg.get("pilot_note", ""),
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
        "train_pairs": int(len(train_ds)),
        "inner_dev_pairs": int(len(dev_ds)),
        "train_plots": int(train_df.plot_uid.nunique()),
        "inner_dev_plots": int(dev_df.plot_uid.nunique()),
        "train_unique_images": int(pd.unique(pd.concat(
            [train_df.image_stem_a, train_df.image_stem_b])).size),
        "batch_size": int(tcfg["batch_size"]),
        "steps_per_epoch": steps_per_epoch,
        "wall_clock_seconds": round(wall, 1),
        "samples_per_second": round(n_seen / max(wall, 1e-9), 2),
        "peak_gpu_memory_bytes": int(peak),
        "selection_metric": "inner_dev_mse_pred",
        "selection_rule": "minimum inner-dev future-latent MSE over the candidate budget",
        "official_validation_used_for_selection": False,
        "delta_norm": delta_norm.as_dict(),
        "gdd_norm": gdd_norm.as_dict() if gdd_norm else None,
        "weather_norm": weather_norm.as_dict() if weather_norm else None,
        "split": split_stats,
        "index_sha256": idx_sha,
        "conditioning_sha256": cond_sha,
        "feature_index_sha256": feat_sha,
        "best_inner_dev_mse": best_dev,
        "best_epoch": best_epoch,
        "init_state_sha256": init_state_sha,
        "init_rule": (
            "torch.manual_seed(seed) (set_seed) immediately before "
            "build_s4_predictor; upstream ViTPredictor default initialisation; "
            "no weight is loaded from any earlier run"
        ),
        "fixed_delta": fixed_delta,
        "history": history,
        "checkpoint_best": str(out_dir / "checkpoints" / "best.pt"),
    }
    with open(out_dir / "run_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w") as fh:
            json.dump(summary, fh, indent=2)
    print(json.dumps({k: summary[k] for k in (
        "condition", "seed", "epochs_completed", "global_step", "train_pairs",
        "inner_dev_pairs", "wall_clock_seconds", "samples_per_second",
        "peak_gpu_memory_bytes", "n_trainable_parameters", "best_epoch",
        "best_inner_dev_mse")}, indent=2), flush=True)
    print(f"wrote {out_dir/'run_summary.json'}", flush=True)
    return 0


def _subset(cond: dict, rows: pd.Index) -> dict:
    """Slice every per-row conditioning array to the requested rows.

    The dataset indexes arrays with the *row position inside its own frame*, so
    the sliced dict must start at position 0 for the frame.
    """
    rows = np.asarray(rows, dtype=np.int64)
    out = {}
    for k, v in cond.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == cond["delta_days"].shape[0]:
            out[k] = v[rows]
        else:
            out[k] = v
    # dataset still validates pair alignment against the frame it is given
    return out


if __name__ == "__main__":
    raise SystemExit(main())
