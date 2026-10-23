#!/usr/bin/env python3
"""S2: inspect a *real* FIP batch end-to-end and verify the encoder geometry.

What this does (no training):

1. builds the S2 pair table from the frozen S1 manifests;
2. builds a ``FIPPairDataset`` over the pairs whose two images exist locally
   (101 real PNGs / 8 plots at the time of writing) and loads a real batch
   through ``fip_collate_fn``;
3. prints every required field with shape / dtype / valid count;
4. verifies the ``(t_a, t_b]`` weather window and mask semantics on real dates;
5. checks that missing labels, measured zeros and negative PH are not conflated;
6. runs the frozen DINOv2 ViT-B/14 at 224 -> asserts 256x768 patch tokens;
7. round-trips the S1 feature cache against a live forward pass;
8. exercises the upstream ``VWorldModel`` / ``ViTPredictor`` geometry to show
   exactly where 196 vs 256 tokens diverge.

Usage:
    python3 tools/s2_inspect_batch.py \
        --config agro_jepa/config/s2_dataset.yaml \
        --batch-size 8 \
        --report reports/s2_real_batch.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agro_jepa.datasets.collate import fip_collate_fn  # noqa: E402
from agro_jepa.datasets.fip_dataset import FIPPairDataset, ImageQualityRule  # noqa: E402
from agro_jepa.datasets.fip_pairs import build_pairs, load_custom_splits, load_observations, load_plots, PairRule  # noqa: E402
from agro_jepa.datasets.fip_weather import DailyWeatherStore, weather_provenance  # noqa: E402
from agro_jepa.datasets.provenance import (  # noqa: E402
    CACHE_PROTOCOL,
    CACHE_PROTOCOL_VERSION,
    DINOV2_EMBED_DIM,
    DINOV2_MODEL,
    DINOV2_REPO_GIT_COMMIT,
    DINOV2_WEIGHTS_SHA256,
    MAINLINE_IMAGE_SIZE,
    MAINLINE_N_TOKENS,
    UPSTREAM_N_TOKENS,
    build_mainline_transform,
    cache_key_for_image,
)

REQUIRED_FIELDS = {
    "image_a": "anchor image tensor / features",
    "image_b": "target image tensor / features",
    "weather_daily": "daily weather (L, C)",
    "weather_mask": "per-day validity (L,)",
    "delta_days": "(date_b - date_a).days",
    "phenotype_a": "phenotype vector at t_a (NaN = missing)",
    "phenotype_b": "phenotype vector at t_b",
    "phenotype_mask_a": "validity of phenotype_a",
    "phenotype_mask_b": "validity of phenotype_b",
    "label_dates": "actual measurement dates per trait and end",
    "label_lag_days": "signed label lag per trait and end",
    "plot_uid": "FIP plot-year id",
    "genotype_id": "anonymised genotype id",
    "harvest_year": "harvest year",
    "image_dates": "(date_a, date_b)",
    "weather_date_range": "(t_a, t_b]",
    "split": "official split",
    "pair_id": "stable pair id",
}


def load_cfg(path: str | Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def local_image_index(root: Path) -> dict[str, Path]:
    return {p.stem: p for p in root.rglob("*.png")}


def select_batch_indices(df: pd.DataFrame, batch_size: int,
                         must_include: list[str] | None = None) -> list[int]:
    """Deterministic mix: every split present in the local subset gets a slot."""
    chosen: list[int] = []
    if must_include:
        for pid in must_include:
            hit = df.index[df.pair_id == pid].tolist()
            if hit:
                chosen.append(int(hit[0]))
    splits = sorted(df.split.unique())
    per_split = max(1, batch_size // max(len(splits), 1))
    for s in splits:
        sub = df[df.split == s].sort_values(["plot_uid", "date_a", "pair_type", "pair_id"])
        for idx in sub.index[:per_split]:
            if int(idx) not in chosen:
                chosen.append(int(idx))
    if len(chosen) < batch_size:
        for idx in df.sort_values(["plot_uid", "date_a", "pair_type", "pair_id"]).index:
            if int(idx) not in chosen:
                chosen.append(int(idx))
            if len(chosen) >= batch_size:
                break
    return chosen[:batch_size]


def field_table(batch: dict) -> list[dict]:
    rows = []
    for name, value in batch.items():
        rec = {"field": name, "required": name in REQUIRED_FIELDS,
               "description": REQUIRED_FIELDS.get(name, "")}
        if torch.is_tensor(value):
            rec.update({
                "kind": "tensor",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "n_elements": int(value.numel()),
            })
            if value.dtype == torch.bool:
                rec["n_true"] = int(value.sum())
                rec["n_false"] = int((~value).sum())
            elif value.dtype.is_floating_point:
                rec["n_nan"] = int(torch.isnan(value).sum())
                rec["n_finite"] = int(torch.isfinite(value).sum())
        elif value is None:
            rec.update({"kind": "none"})
        elif isinstance(value, np.ndarray):
            rec.update({"kind": "ndarray", "shape": list(value.shape),
                        "dtype": str(value.dtype)})
        elif isinstance(value, list):
            rec.update({"kind": "list", "length": len(value),
                        "example": str(value[0])[:200] if value else None})
        else:
            rec.update({"kind": type(value).__name__, "example": str(value)[:200]})
        rows.append(rec)
    return rows


def verify_weather(batch: dict, ds: FIPPairDataset) -> dict:
    lengths = batch["weather_lengths"].tolist()
    deltas = batch["delta_days"].tolist()
    ok_len = all(int(l) == int(d) for l, d in zip(lengths, deltas))
    # first window day is t_a + 1, last is t_b
    ok_bounds = True
    for i, (start, end) in enumerate(batch["weather_date_range"]):
        n = lengths[i]
        dates = [d for d in batch["weather_dates"][i][:n]]
        if dates[0] != (date.fromisoformat(start) + timedelta(days=1)).isoformat():
            ok_bounds = False
        if dates[-1] != end:
            ok_bounds = False
        if len(set(dates)) != n:
            ok_bounds = False
    # mask must be all-True for a fully covered window, and padding must be masked
    pad_mask_ok = True
    for i, n in enumerate(lengths):
        if bool(batch["weather_mask"][i, :n].all()) is not True:
            pad_mask_ok = False
        if bool(batch["weather_mask"][i, n:].any()) is True:
            pad_mask_ok = False
    padded = int(batch["weather_mask"].numel() - batch["weather_mask"].sum())
    return {
        "n_samples": int(len(lengths)),
        "lengths": [int(x) for x in lengths],
        "delta_days": [int(x) for x in deltas],
        "length_equals_delta": bool(ok_len),
        "window_is_left_open_right_closed": bool(ok_bounds),
        "mask_matches_lengths_and_padding": bool(pad_mask_ok),
        "n_padded_weather_steps": padded,
        "n_valid_weather_steps": int(batch["weather_mask"].sum()),
        "weather_channels": list(batch.get("weather_channel_names") or ds.weather_store.channels),
        "examples": [
            {
                "pair_id": batch["pair_id"][i],
                "image_dates": list(batch["image_dates"][i]),
                "weather_window": list(batch["weather_date_range"][i]),
                "weather_n_days": int(lengths[i]),
                "delta_days": int(deltas[i]),
                "weather_dates_head": batch["weather_dates"][i][: min(3, lengths[i])].tolist(),
                "weather_dates_tail": batch["weather_dates"][i][max(0, lengths[i] - 2):lengths[i]].tolist(),
                "weather_first_day": (batch["weather_daily"][i, 0].tolist()),
            }
            for i in range(min(3, len(lengths)))
        ],
    }


def verify_labels(batch: dict) -> dict:
    traits = list(batch["trait_names"])
    ph_idx = traits.index("ph") if "ph" in traits else None
    pa, pb = batch["phenotype_a"], batch["phenotype_b"]
    ma, mb = batch["phenotype_mask_a"], batch["phenotype_mask_b"]

    def zero_and_masked(t: int) -> dict:
        return {
            "measured_zero_a": int(((pa[:, t] == 0) & ma[:, t]).sum()),
            "measured_zero_b": int(((pb[:, t] == 0) & mb[:, t]).sum()),
            "missing_nan_a": int(torch.isnan(pa[:, t]).sum()),
            "missing_nan_b": int(torch.isnan(pb[:, t]).sum()),
            "present_but_outside_lag_a": int(
                (torch.isfinite(pa[:, t]) & ~ma[:, t]).sum()),
            "present_but_outside_lag_b": int(
                (torch.isfinite(pb[:, t]) & ~mb[:, t]).sum()),
        }

    out = {t: zero_and_masked(i) for i, t in enumerate(traits)}
    if ph_idx is not None:
        out["ph_negative"] = {
            "n_flag_a": int(batch["ph_negative_a"].sum()),
            "n_flag_b": int(batch["ph_negative_b"].sum()),
            "n_negative_values_kept_a": int((pa[:, ph_idx] < 0).sum()),
            "n_negative_values_kept_b": int((pb[:, ph_idx] < 0).sum()),
            "note": "negative PH stays in phenotype_*; it is flagged, never clipped",
        }
    # a concrete missing-vs-zero example
    examples = []
    for i in range(len(pa)):
        for t, name in enumerate(traits):
            if torch.isnan(pa[i, t]) and torch.isfinite(pb[i, t]):
                examples.append({
                    "pair_id": batch["pair_id"][i], "trait": name, "end": "a",
                    "value": None, "mask": False,
                    "label_date": batch["label_dates"][i][0, t],
                })
            if torch.isfinite(pa[i, t]) and float(pa[i, t]) == 0.0:
                examples.append({
                    "pair_id": batch["pair_id"][i], "trait": name, "end": "a",
                    "value": 0.0, "mask": bool(ma[i, t]),
                    "label_date": batch["label_dates"][i][0, t],
                })
        if len(examples) >= 4:
            break
    out["examples"] = examples[:4]
    return out


# --------------------------------------------------------------------------
# encoder
# --------------------------------------------------------------------------
def load_local_dinov2(repo: Path, device: torch.device):
    model = torch.hub.load(str(repo), DINOV2_MODEL, source="local",
                           pretrained=True, verbose=False)
    return model.eval().to(device)


def encode_batch(model, images: torch.Tensor, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        return model.forward_features(images.to(device))["x_norm_patchtokens"]


def cache_roundtrip(model, batch: dict, image_index: dict[str, Path],
                    cache_root: Path, device: torch.device) -> dict:
    proto_dir = cache_root / f"{CACHE_PROTOCOL}_{CACHE_PROTOCOL_VERSION}"
    rows = []
    for key in ("image_a", "image_b"):
        stems = batch["image_stem_a"] if key == "image_a" else batch["image_stem_b"]
        for i, stem in enumerate(stems):
            p = image_index.get(stem)
            if p is None:
                continue
            path = proto_dir / f"{cache_key_for_image(sha256_file(p))}.npy"
            if not path.exists():
                continue
            cached = torch.from_numpy(np.load(path).astype(np.float32))
            live = encode_batch(model, batch[key][i: i + 1], device)[0].float().cpu()
            cos = torch.nn.functional.cosine_similarity(
                live.flatten(), cached.flatten(), dim=0).item()
            rows.append({
                "pair_id": batch["pair_id"][i],
                "end": "a" if key == "image_a" else "b",
                "stem": stem,
                "cache_file": str(path),
                "cached_shape": list(cached.shape),
                "live_shape": list(live.shape),
                "max_abs_diff": float((live - cached).abs().max()),
                "mean_cos": float(cos),
            })
    if not rows:
        return {"n_compared": 0}
    return {
        "n_compared": len(rows),
        "max_abs_diff": max(r["max_abs_diff"] for r in rows),
        "min_cosine": min(r["mean_cos"] for r in rows),
        "all_shapes_match": all(r["cached_shape"] == r["live_shape"] for r in rows),
        "examples": rows[:3],
    }


def upstream_geometry_check(model, images: torch.Tensor, device: torch.device) -> dict:
    """Exercise the real upstream classes with the real frozen encoder."""
    upstream_root = ROOT / "dino_wm"
    sys.path.insert(0, str(upstream_root))
    import torch.nn as nn
    from models.visual_world_model import VWorldModel  # noqa: E402
    from models.vit import ViTPredictor  # noqa: E402

    class LocalDino(nn.Module):
        """Same interface as upstream ``DinoV2Encoder``, no hub ref."""
        def __init__(self, base):
            super().__init__()
            self.name = DINOV2_MODEL
            self.base_model = base
            self.emb_dim = int(base.embed_dim)
            self.latent_ndim = 2
            self.patch_size = int(base.patch_size)

        def forward(self, x):
            return self.base_model.forward_features(x)["x_norm_patchtokens"]

    results = {}
    for tag, image_size_cfg in (("mainline", 256), ("upstream_default", MAINLINE_IMAGE_SIZE)):
        enc = LocalDino(model)
        vm = VWorldModel(
            image_size=image_size_cfg,
            num_hist=1,
            num_pred=1,
            encoder=enc,
            proprio_encoder=torch.nn.Identity(),
            action_encoder=torch.nn.Identity(),
            decoder=None,
            predictor=None,
            proprio_dim=1,
            action_dim=1,
            concat_dim=1,
            num_proprio_repeat=1,
            num_action_repeat=1,
            train_encoder=False,
            train_predictor=False,
            train_decoder=False,
        )
        b = images.shape[0]
        obs = {"visual": images.to(device).unsqueeze(1),
               "proprio": torch.zeros(b, 1, 1, device=device)}
        act = torch.zeros(b, 1, 1, device=device)
        with torch.no_grad():
            z = vm.encode(obs, act)
        num_side = image_size_cfg // 16
        results[tag] = {
            "upstream_img_size": image_size_cfg,
            "computed_num_side_patches": num_side,
            "encoder_image_size": int(vm.encoder_image_size),
            "encoder_transform": str(vm.encoder_transform),
            "num_patches_train_py": int(num_side ** 2),
            "encoder_input_shape": [int(x) for x in
                                    (torch.nn.functional.interpolate(
                                        images, size=vm.encoder_image_size).shape)],
            "z_shape": list(z.shape),
            "visual_tokens": int(z.shape[2]),
        }
        if tag == "mainline" and torch.cuda.is_available():
            dim = int(z.shape[-1])
            pred = ViTPredictor(
                num_patches=int(num_side ** 2), num_frames=1, dim=dim,
                depth=1, heads=16, mlp_dim=64, pool="mean", dim_head=64,
                dropout=0.0, emb_dropout=0.0,
            )
            pred.eval().to(device)
            with torch.no_grad():
                out = pred(z.reshape(b, 1 * int(num_side ** 2), dim))
            results[tag]["predictor_num_patches"] = int(num_side ** 2)
            results[tag]["predictor_pos_embedding"] = list(pred.pos_embedding.shape)
            results[tag]["predictor_output"] = list(out.shape)
            attn = pred.transformer.layers[0][0]
            results[tag]["predictor_attention_mask"] = list(attn.bias.shape)
        del vm
    results["note"] = (
        "LocalDino wraps the same frozen local DINOv2 weights used for the S1 "
        "cache; upstream models.dino.DinoV2Encoder resolves a moving ':main' "
        "hub ref and is not used here. With img_size=256 upstream computes "
        "encoder_image_size=(256//16)*14=224 and num_patches=(256//16)^2=256, "
        "i.e. mainline224; with img_size=224 it computes 196/196."
    )
    # the upstream VQVAE decoder infers the grid as sqrt(num_patches) and
    # upsamples by 16, so 196 tokens -> 224 px but 256 tokens -> 256 px.
    from models.vqvae import VQVAE  # noqa: E402
    dec = VQVAE(emb_dim=768, channel=384, n_embed=2048, n_res_block=4,
                n_res_channel=128, quantize=False).eval()
    decoder_geometry = {}
    with torch.no_grad():
        for n in (UPSTREAM_N_TOKENS, MAINLINE_N_TOKENS):
            out, _ = dec(torch.zeros(2, 1, n, 768))
            decoder_geometry[str(n)] = {
                "n_tokens": n,
                "grid": int(n ** 0.5),
                "decoded_shape": [int(x) for x in out.shape],
                "matches_224_visual_target": bool(out.shape[-1] == MAINLINE_IMAGE_SIZE),
            }
    decoder_geometry["verdict"] = (
        "The upstream VQVAE decoder reconstructs 224x224 only from 196 tokens "
        "(14x14 grid). With mainline224's 256 tokens it reconstructs 256x256, "
        "so VWorldModel.forward's decoder MSE against the 224x224 obs tensor "
        "would be a shape error. S3 latent experiments must run with "
        "has_decoder=False (or add a decoder resized to a 16x16 grid); the "
        "predictor path itself is consistent at 256 tokens."
    )
    results["decoder_geometry"] = decoder_geometry
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s2_dataset.yaml")
    ap.add_argument("--pairs", default="external_data/pairs_cache/fip1_pairs_s2.parquet")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--report", default="reports/s2_real_batch.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dinov2-repo", default="external_data/third_party/dinov2")
    ap.add_argument("--no-encode", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    cfg = load_cfg(args.config)
    data, pr = cfg["data"], cfg["pair_rule"]

    # ---- pairs ------------------------------------------------------------
    pairs_path = Path(args.pairs)
    if pairs_path.exists():
        pairs = pd.read_parquet(pairs_path)
        pairs_source = str(pairs_path)
    else:
        rule = PairRule(
            mode=pr["mode"], spans=tuple(pr["spans"]),
            span_tolerance_days=pr["span_tolerance_days"],
            min_delta_days=pr["min_delta_days"],
            max_delta_days=pr.get("max_delta_days"),
            label_max_lag_days=pr["label_max_lag_days"],
            manifest_id=cfg["manifest_id"],
        )
        pairs, _ = build_pairs(
            load_observations(data["observations"]),
            load_plots(data["plots"]),
            load_custom_splits(data["custom_splits"]),
            rule,
        )
        pairs_source = "built in memory"

    image_root = Path(data["images_root"])
    image_index = local_image_index(image_root)
    have_a = pairs.image_stem_a.isin(image_index)
    have_b = pairs.image_stem_b.isin(image_index)
    local_pairs = pairs[have_a & have_b].copy().reset_index(drop=True)
    if local_pairs.empty:
        raise SystemExit(f"no pair has both images under {image_root}")

    weather = DailyWeatherStore(
        data["weather_daily"], channels=tuple(cfg["weather"]["channels"])
    )
    ds = FIPPairDataset(
        local_pairs, weather,
        image_mode="visual",
        images_root=image_root,
        compute_image_stats=True,
        quality_rule=ImageQualityRule(**cfg["image"]["quality_rule"]),
    )

    # include the near-white frame S1 flagged, to show the quality flag is reported
    flagged = local_pairs.loc[
        local_pairs.image_stem_a.eq("FPWW0120667_RGB1_20160216_125228")
        | local_pairs.image_stem_b.eq("FPWW0120667_RGB1_20160216_125228"),
        "pair_id",
    ].tolist()[:1]
    # include a pair with a *measured zero* label, so zero-vs-missing is shown live
    zero_ids = local_pairs.loc[
        ((local_pairs.whc_a == 0) & local_pairs.whc_mask_a)
        | ((local_pairs.whc_b == 0) & local_pairs.whc_mask_b),
        "pair_id",
    ].tolist()[:1]
    idx = select_batch_indices(local_pairs, args.batch_size,
                               must_include=flagged + zero_ids)
    samples = [ds[i] for i in idx]
    batch = fip_collate_fn(samples)

    weather_checks = verify_weather(batch, ds)
    label_checks = verify_labels(batch)

    date_examples = [
        {
            "pair_id": batch["pair_id"][i],
            "plot_uid": batch["plot_uid"][i],
            "split": batch["split"][i],
            "harvest_year": int(batch["harvest_year"][i]),
            "image_stem_a": batch["image_stem_a"][i],
            "image_stem_b": batch["image_stem_b"][i],
            "image_dates": list(batch["image_dates"][i]),
            "weather_window": list(batch["weather_date_range"][i]),
            "delta_days": int(batch["delta_days"][i]),
            "phenotype_a": [round(float(x), 4) if np.isfinite(x) else None
                            for x in batch["phenotype_a"][i].tolist()],
            "phenotype_b": [round(float(x), 4) if np.isfinite(x) else None
                            for x in batch["phenotype_b"][i].tolist()],
            "phenotype_mask_a": batch["phenotype_mask_a"][i].tolist(),
            "phenotype_mask_b": batch["phenotype_mask_b"][i].tolist(),
            "label_lag_days": [[None if not np.isfinite(float(v)) else float(v)
                                for v in row] for row in batch["label_lag_days"][i].tolist()],
            "label_dates": [list(r) for r in batch["label_dates"][i].tolist()],
        }
        for i in range(len(idx))
    ]

    image_quality = []
    for i in range(len(idx)):
        for end in ("a", "b"):
            st = batch[f"image_stats_{end}"][i]
            image_quality.append({
                "pair_id": batch["pair_id"][i], "end": end,
                "stem": st.get("image_path", "").rsplit("/", 1)[-1],
                "luma_mean": round(st["luma_mean"], 4),
                "luma_std": round(st["luma_std"], 4),
                "near_white": st.get("near_white", False),
                "quality_usable": st["quality_usable"],
            })

    # full scan of the local image pool with the documented rule
    from PIL import Image
    rule = ds.quality_rule
    pool = []
    for stem, p in sorted(image_index.items()):
        arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
        luma = arr.mean(axis=2)
        m, s = float(luma.mean()), float(luma.std())
        pool.append({"stem": stem, "path": str(p), "luma_mean": round(m, 4),
                     "luma_std": round(s, 4), "near_white": rule.is_near_white(m, s),
                     "quality_usable": rule.evaluate(m, s)})
    flagged_pool = [r for r in pool if not r["quality_usable"]]

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "stage": "S2",
        "config": str(args.config),
        "pairs_source": pairs_source,
        "n_pairs_total": int(len(pairs)),
        "n_pairs_with_both_images_local": int(len(local_pairs)),
        "n_local_images": int(len(image_index)),
        "n_local_plots": int(local_pairs.plot_uid.nunique()),
        "local_pair_types": {str(k): int(v)
                             for k, v in local_pairs.pair_type.value_counts().items()},
        "local_splits": {str(k): int(v)
                         for k, v in local_pairs.split.value_counts().items()},
        "batch_indices": [int(i) for i in idx],
        "batch_pair_ids": batch["pair_id"],
        "batch_size": int(len(idx)),
        "fields": field_table(batch),
        "date_examples": date_examples,
        "weather_checks": weather_checks,
        "weather_provenance": weather_provenance(),
        "label_checks": label_checks,
        "image_quality": {
            "policy": cfg["image"]["quality_policy"],
            "rule": cfg["image"]["quality_rule"],
            "n_flagged_in_batch": int(sum(1 for q in image_quality if not q["quality_usable"])),
            "flagged": [q for q in image_quality if not q["quality_usable"]],
            "local_pool": {
                "n_images": len(pool),
                "n_flagged": len(flagged_pool),
                "flagged": flagged_pool,
                "luma_mean_range": [min(r["luma_mean"] for r in pool),
                                    max(r["luma_mean"] for r in pool)],
                "luma_std_range": [min(r["luma_std"] for r in pool),
                                   max(r["luma_std"] for r in pool)],
            },
            "note": "flags are reported, never applied; no image is dropped in S2",
        },
    }

    # ---- frozen encoder ---------------------------------------------------
    if not args.no_encode:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        repo = Path(args.dinov2_repo).resolve()
        t1 = time.time()
        model = load_local_dinov2(repo, device)
        load_s = time.time() - t1
        transform = build_mainline_transform()
        # the batch is already transformed; re-encode a slice to keep timing clean
        x = batch["image_a"]
        t2 = time.time()
        tokens_a = encode_batch(model, x, device)
        tokens_b = encode_batch(model, batch["image_b"], device)
        torch.cuda.synchronize() if device.type == "cuda" else None
        enc_s = time.time() - t2
        # the same pipeline applied to a raw PNG must reproduce the batch tensor
        raw = image_index[batch["image_stem_a"][0]]
        from PIL import Image
        t = transform(Image.open(raw).convert("RGB"))
        transform_max_diff = float((t - x[0].cpu()).abs().max())

        report["encoding"] = {
            "device": str(device),
            "torch": torch.__version__,
            "model": DINOV2_MODEL,
            "repo": str(repo),
            "repo_git_commit": DINOV2_REPO_GIT_COMMIT,
            "weights_sha256": DINOV2_WEIGHTS_SHA256,
            "model_load_seconds": round(load_s, 2),
            "batch_forward_seconds": round(enc_s, 3),
            "n_images_encoded": int(2 * len(idx)),
            "input_shape": list(x.shape),
            "tokens_a_shape": list(tokens_a.shape),
            "tokens_b_shape": list(tokens_b.shape),
            "n_tokens": int(tokens_a.shape[1]),
            "emb_dim": int(tokens_a.shape[2]),
            "expected": {"n_tokens": MAINLINE_N_TOKENS, "emb_dim": DINOV2_EMBED_DIM},
            "matches_mainline_geometry": bool(
                tokens_a.shape[1] == MAINLINE_N_TOKENS
                and tokens_a.shape[2] == DINOV2_EMBED_DIM
                and tokens_b.shape[1:] == tokens_a.shape[1:]
            ),
            "sample_id_order_preserved": bool(
                len(tokens_a) == len(batch["pair_id"]) and len(tokens_b) == len(batch["pair_id"])
            ),
            "transform_reproduced_from_raw_png_max_abs_diff": transform_max_diff,
        }
        report["cache_roundtrip"] = cache_roundtrip(
            model, batch, image_index, Path(data["feature_cache_dir"]), device
        )
        report["upstream_geometry"] = upstream_geometry_check(model, x, device)

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)

    # ---- console summary --------------------------------------------------
    print(f"pairs: {len(pairs):,} total, {len(local_pairs):,} with both images local "
          f"({local_pairs.plot_uid.nunique()} plots, {len(image_index)} local PNGs)")
    print(f"batch: {len(idx)} samples from splits {sorted(set(batch['split']))}")
    print(f"{'field':30s} {'shape':22s} {'dtype':14s} extra")
    for row in report["fields"]:
        if row["kind"] == "tensor":
            extra = (f"true={row.get('n_true')}" if row["dtype"] == "torch.bool"
                     else f"nan={row.get('n_nan')}")
            print(f"{row['field']:30s} {str(row['shape']):22s} {row['dtype']:14s} {extra}")
        elif row["kind"] == "list":
            print(f"{row['field']:30s} {'list':22s} {str(row['length']):14s}")
        else:
            print(f"{row['field']:30s} {row['kind']:22s}")
    print("\nweather:", {k: v for k, v in weather_checks.items()
                         if k not in ("examples", "weather_channels")})
    print("labels:", {k: v for k, v in label_checks.items() if k != "examples"})
    if "encoding" in report:
        e = report["encoding"]
        print(f"encode: {e['input_shape']} -> {e['tokens_a_shape']} / {e['tokens_b_shape']} "
              f"matches={e['matches_mainline_geometry']} in {e['batch_forward_seconds']}s")
        print("cache roundtrip:", {k: report["cache_roundtrip"].get(k) for k in
                                   ("n_compared", "max_abs_diff", "min_cosine", "all_shapes_match")})
        print("upstream:", {g: {k: v for k, v in rec.items() if k in
                                ("upstream_img_size", "encoder_image_size", "num_patches_train_py",
                                 "visual_tokens", "predictor_output")}
                            for g, rec in report["upstream_geometry"].items()
                            if isinstance(rec, dict) and "upstream_img_size" in rec})
    print(f"\nwrote {args.report} in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
