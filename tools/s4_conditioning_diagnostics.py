#!/usr/bin/env python3
"""S4A inference-only conditioning diagnostics.

Question (7) of the S4A brief: *is the weather-model gain only a year-level
difference?*  A permutation test answers it without retraining:

* **within-year permutation** shuffles the conditioning input among validation
  pairs of the *same* harvest_year.  Year identity is preserved; only the
  date-alignment of the weather/GDD is destroyed.  If the model is only using
  "which year is this", the MSE should not move.
* **global permutation** shuffles across all years, destroying year identity as
  well.  This is the analogue of the S3.1 Δt permutation.

The same test is run for `delta_days` on B3, for GDD on B4, and for the weather
sequence (values and mask permuted together, so a sequence stays a coherent
sequence) on B5.  Only Protocol P and Protocol R inputs are permuted as they are
used; no future target is touched.

Usage
-----
    CUDA_VISIBLE_DEVICES=1 python3 tools/s4_conditioning_diagnostics.py \
        --config agro_jepa/config/s4_pilot.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from agro_jepa.datasets.s4_dataset import load_conditioning  # noqa: E402
from agro_jepa.eval.s4_common import (  # noqa: E402
    load_s4_model,
    make_dataset,
)
from agro_jepa.eval.eval_s4 import run_dir  # noqa: E402
from agro_jepa.models.s4_predictor import conditioning_spec  # noqa: E402
from s3_1_diagnostics import cluster_bootstrap  # noqa: E402


CONDITIONS = [
    "b3_delta",
    "b4_delta_gdd_R",
    "b4_delta_gdd_P",
    "b5_delta_weather_P",
    "b5_delta_weather_R",
]


def _permutations(index: pd.DataFrame, n_seeds: int, seed0: int):
    n = len(index)
    years = index.harvest_year.to_numpy()
    rng = np.random.default_rng(seed0)
    out = {"global": [], "within_year": []}
    for _ in range(n_seeds):
        out["global"].append(rng.permutation(n))
    for _ in range(n_seeds):
        perm = np.arange(n)
        for y in np.unique(years):
            idx = np.flatnonzero(years == y)
            perm[idx] = rng.permutation(idx)
        out["within_year"].append(perm)
    return out


def _forward(models, z_a: torch.Tensor, inputs: dict) -> torch.Tensor:
    out = None
    for m in models:
        p = m(z_a, **inputs).float()
        out = p if out is None else out + p
    return out / len(models)


@torch.no_grad()
def _run(models, ds, real_cond, perm_cond, mode, device, batch_size,
         compute_real: bool = True):
    """Forward passes: real conditioning (optional) and permuted conditioning.

    ``mode`` selects which conditioning tensor is replaced; the others (if any)
    are passed unchanged.  ``z_b`` is only the evaluation target.
    """
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4,
                        pin_memory=True)
    real_mse, perm_mse = [], []
    for batch in loader:
        z_a = batch["z_a"].to(device, non_blocking=True).float()
        z_b = batch["z_b"].to(device, non_blocking=True).float()
        rows = batch["row"].cpu().numpy()
        perm_inputs = {}
        real_inputs = {}
        if mode == "delta":
            perm_inputs = {"delta_days": torch.from_numpy(
                perm_cond["delta"][rows]).to(device)}
            if compute_real:
                real_inputs = {"delta_days": batch["delta_days"].to(device)}
        elif mode == "gdd":
            perm_inputs = {"gdd": torch.from_numpy(perm_cond["gdd"][rows]).to(device)}
            if compute_real:
                real_inputs = {"gdd": torch.from_numpy(
                    real_cond["gdd"][rows]).to(device)}
        elif mode == "weather":
            perm_inputs = {
                "weather": torch.from_numpy(perm_cond["weather"][rows]).to(device),
                "weather_mask": torch.from_numpy(
                    perm_cond["weather_mask"][rows]).to(device),
            }
            if compute_real:
                real_inputs = {
                    "weather": torch.from_numpy(real_cond["weather"][rows]).to(device),
                    "weather_mask": torch.from_numpy(
                        real_cond["weather_mask"][rows]).to(device),
                }
        # Conditions with delta + GDD/weather keep the *real* delta in both
        # passes; only the targeted component is permuted.
        if mode in ("gdd", "weather") and "delta" in getattr(models[0], "parts", []):
            d = batch["delta_days"].to(device)
            perm_inputs["delta_days"] = d
            if compute_real:
                real_inputs["delta_days"] = d
        pp = _forward(models, z_a, perm_inputs)
        perm_mse.append(((pp - z_b).pow(2).mean(dim=(1, 2))).cpu().numpy())
        if compute_real:
            pr = _forward(models, z_a, real_inputs)
            real_mse.append(((pr - z_b).pow(2).mean(dim=(1, 2))).cpu().numpy())
    if not compute_real:
        return None, np.concatenate(perm_mse)
    return np.concatenate(real_mse), np.concatenate(perm_mse)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s4_pilot.yaml")
    ap.add_argument("--out", default="reports/s4_conditioning_diagnostics.json")
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--seed-base", type=int, default=31000)
    ap.add_argument("--bootstrap-resamples", type=int, default=2000)
    ap.add_argument("--bootstrap-seed", type=int, default=20261024)
    ap.add_argument("--smoke-rows", type=int, default=0,
                    help="run on the first N validation rows only (plumbing check)")
    ap.add_argument("--conditions", nargs="*", default=None)
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    device = torch.device(cfg["eval"]["device"] if torch.cuda.is_available() else "cpu")
    index = pd.read_parquet(cfg["data"]["index"])
    cond = load_conditioning(cfg["data"]["conditioning_npz"], index)
    val = index[index.split == "validation"].copy()
    val["_orig_row"] = index.index[index.split == "validation"].to_numpy()
    val = val.reset_index(drop=True)
    if args.smoke_rows:
        val = val.head(int(args.smoke_rows)).reset_index(drop=True)
    rows_idx = val["_orig_row"].to_numpy()
    perms = _permutations(val, args.n_seeds, args.seed_base)

    # cache the conditioning tensors on the validation rows
    gdd_cache, weather_cache = {}, {}
    for name in CONDITIONS:
        spec = conditioning_spec(cfg["conditions"][name])
        if spec.gdd_mode != "none":
            gdd_cache[name] = np.stack(
                [cond[k][rows_idx] for k in spec.gdd_keys()], axis=1
            ).astype(np.float32)
        if spec.weather_mode != "none":
            w_key, m_key = spec.weather_keys()
            weather_cache[name] = (
                cond[w_key][rows_idx].astype(np.float32),
                cond[m_key][rows_idx].astype(bool),
            )

    result = {"stage": "S4A", "task": "conditioning permutation diagnostics",
              "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "command": " ".join(sys.argv),
              "n_val_pairs": int(len(val)), "n_val_plots": int(val.plot_uid.nunique()),
              "n_permutation_seeds": int(args.n_seeds),
              "seed_base": int(args.seed_base),
              "unit": "relative MSE increase = (MSE_permuted - MSE_real) / MSE_real",
              "conditions": {}}
    bootstrap_seed = args.bootstrap_seed

    run_conditions = args.conditions or CONDITIONS
    for name in run_conditions:
        if name not in cfg["conditions"] or cfg["conditions"][name].get("mode") != "trained":
            continue
        spec = conditioning_spec(cfg["conditions"][name])
        models = []
        for seed in cfg["training"]["seeds"]:
            p = run_dir(cfg, name, int(seed)) / "checkpoints" / "best.pt"
            if not p.exists():
                print(f"skip {name}: missing {p}", flush=True)
                break
            m, _ = load_s4_model(str(p), device)
            models.append(m)
        else:
            ds = make_dataset(val, cond, spec, (), load_features=True)
            # rows used by the dataset are 0..n-1; permutations are over the same order
            mode = ("delta" if spec.use_delta and spec.gdd_mode == "none"
                    and spec.weather_mode == "none"
                    else "gdd" if spec.gdd_mode != "none" else "weather")
            block = {"mode": mode, "protocol": cfg["conditions"][name].get("protocol"),
                     "real": {}, "within_year": [], "global": []}
            plots = val.plot_uid.to_numpy()
            identity = np.arange(len(val))

            def build_cond(mode, perm):
                if mode == "delta":
                    return {}, {"delta": cond["delta_days"][rows_idx][perm]}
                if mode == "gdd":
                    return {"gdd": gdd_cache[name]}, {"gdd": gdd_cache[name][perm]}
                w, m = weather_cache[name]
                return ({"weather": w, "weather_mask": m},
                        {"weather": w[perm], "weather_mask": m[perm]})

            _, real_cond0 = build_cond(mode, identity)
            _, real_mse = _run(models, ds, {}, real_cond0, mode, device,
                               int(cfg["eval"]["batch_size"]), compute_real=False)
            block["real"] = {"mse": float(real_mse.mean())}
            plots_bs = plots.copy()
            # a single global permutation is used for the bootstrap reference
            global_perm = perms["global"][0]

            for perm_name in ("within_year", "global"):
                incs = []
                for perm in perms[perm_name]:
                    real_cond, perm_cond = build_cond(mode, perm)
                    _, p = _run(models, ds, real_cond, perm_cond, mode, device,
                                int(cfg["eval"]["batch_size"]), compute_real=False)
                    incs.append(float((p.mean() - block["real"]["mse"])
                                      / block["real"]["mse"]))
                incs = np.array(incs)
                block[perm_name] = {
                    "mse_permuted_mean": float(block["real"]["mse"] * (1 + incs.mean())),
                    "relative_increase_mean": float(incs.mean()),
                    "relative_increase_std": float(incs.std(ddof=1)),
                    "relative_increase_min": float(incs.min()),
                    "relative_increase_max": float(incs.max()),
                    "n_seeds": int(len(incs)),
                    "per_seed": incs.tolist(),
                }
                print(f"[diag] {name} {perm_name}: +{incs.mean()*100:.2f}%", flush=True)

            # plot-cluster bootstrap for both interventions (one permutation each)
            for perm_name, perm in (("within_year", perms["within_year"][0]),
                                    ("global", global_perm)):
                real_cond, perm_cond = build_cond(mode, perm)
                r, p = _run(models, ds, real_cond, perm_cond, mode, device,
                            int(cfg["eval"]["batch_size"]))
                tables = {"real": r, "perm": p}

                def rel(t):
                    return float((t["perm"].mean() - t["real"].mean()) / t["real"].mean())

                block[f"{perm_name}_bootstrap"] = cluster_bootstrap(
                    tables, plots_bs, rel, args.bootstrap_resamples, bootstrap_seed)
            result["conditions"][name] = block
            del models

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
