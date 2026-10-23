# S1 — DINOv2 feature-cache protocol (validated on real FIP images)

Model: **DINOv2 ViT-B/14**, run on real inner-plot FIP images. Code and weights
are pinned, the two candidate preprocessing protocols are compared on the same
images, and the two caches are written to **separate directories** so they cannot
be mixed.

```bash
# images (real PNGs from the dataset, SHA256 recorded per file)
python3 tools/s1_fetch_images.py \
    --observations data/manifests/fip1_observations.csv \
    --plots FPWW0120667 --per-plot 0 \
    --out external_data/fip1_images \
    --manifest data/manifests/fip1_image_fullday_2016.json

# cache both protocols for a sample
python3 tools/s1_dino_cache.py \
    --images data/manifests/fip1_image_sample.json \
    --dinov2-repo external_data/third_party/dinov2 \
    --out external_data/dino_cache \
    --report reports/s1_feature_cache_smoke.json --limit 60
```

Machine-readable result: `reports/s1_feature_cache_smoke.json`.

## 1. Provenance (pinned, not "current main")

| Item | Value |
|---|---|
| Model | `dinov2_vitb14` |
| Code source | local clone `external_data/third_party/dinov2` |
| **Code commit** | **`7764ea0f912e53c92e82eb78a2a1631e92725fc8`** ("Safely load weights from specified URLs (#598)", 2026-06-03) |
| Loading call | `torch.hub.load(repo, "dinov2_vitb14", source="local", pretrained=True)` |
| Weights file | `~/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth` |
| **Weights SHA256** | **`0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73`** |
| **State-dict SHA256** | `6ba9dd85ee9ac3caec717ec270ee86bb44f57b3b4787679b64e5e1c549af0906` |
| Config | `patch_size=14`, `embed_dim=768`, `num_heads=12`, 12 blocks |
| Params | 86,580,480 |
| Model load time | 1.95 s |
| Device | `cuda:0` (RTX A6000, torch 2.10.0+cu128) |

Why a local clone: upstream `DinoV2Encoder` uses `torch.hub.load(
"facebookresearch/dinov2:main")`, which resolves a moving branch and recorded no
SHA in S0. The local clone makes the cache reproducible; if the hub path must be
used, `repo_dir` + `repo_git_commit` in the report identifies exactly what to
clone. The weights digest is computed over the loaded `state_dict()` so any
change in the `.pth` file invalidates the cache key.

## 2. The two protocols

Both start from the same real PNG (960×640 RGB). Preprocessing follows the
upstream dataset transform (`default_transform(224)`), because S0 verified that
DINOv2 does **no** internal normalization and the 0.5 mean/std is the only one.

| Protocol | Pipeline | Encoder input | Patch tokens |
|---|---|---|---|
| `upstream196` (upstream DINO-WM) | ToTensor → Resize(224) → CenterCrop(224) → Normalize(0.5) → **Resize(196)** | 196×196 | **196 tokens (14×14), dim 768** |
| `mainline224` (Agro-JEPA mainline) | ToTensor → Resize(224) → CenterCrop(224) → Normalize(0.5) → **no extra resize** | 224×224 | **256 tokens (16×16), dim 768** |

## 3. Measured results (60 real images, both protocols)

| Metric | `upstream196` | `mainline224` |
|---|---|---|
| Token shape | `(1, 196, 768)` | `(1, 256, 768)` |
| Tokens per image | 196 | 256 |
| Output dtype | `torch.float32` | `torch.float32` |
| Mean forward time | 12.87 ms | 17.43 ms |
| Mean end-to-end (read+transform+forward+save) | 59.19 ms | 57.23 ms |
| Throughput (single stream) | 16.9 img/s | 17.5 img/s |
| Cache dtype | float16 | float16 |
| Cache size per image | 301,184 B | 393,344 B |
| Projection for all 153,022 images | **42.9 GiB** | **56.1 GiB** |

Both protocols were exercised on the same 60 images (10 plots × 6 years); every
record in the report carries the source file, its SHA256, the protocol id, the
protocol version (`s1-v1`), the weights SHA256, the input shape, the token shape,
the output dtype, the cache path and the cache key.

## 4. Mainline decision

**Mainline = `mainline224` (224×224 input, 256 patch tokens).**

Reasons:

1. 224 is the dataset's native transform size; `upstream196` throws away
   resolution (224 → 196 → 14 patches) before the encoder, which is exactly the
   mismatch flagged in the project handbook;
2. 256 tokens is the standard ViT-B/14 grid at 224 (16×16), so the token layout
   matches the reference DINOv2 benchmark geometry and any published
   feature-space comparison;
3. the cost is modest: +30 % cache bytes (56.1 GiB vs 42.9 GiB projected) and
   +4.6 ms per forward on this GPU.

`upstream196` is retained only as the upstream-compatibility path, in its own
cache directory (`external_data/dino_cache/upstream196_s1-v1/`). Mixing the two
caches in one experiment is forbidden: the token counts differ, so predictor
positional embeddings, causal masks and `num_patches` would be inconsistent.

## 5. Cache format and key

```
external_data/dino_cache/<protocol>_<protocol_version>/<cache_key>.npy
```

* Array: `(n_tokens, 768)` `float16`, the `x_norm_patchtokens` output (no CLS
  token, consistent with `DinoV2Encoder.forward` and `latent_ndim == 2`).
* `cache_key = sha256("<source_sha256>|<protocol>|s1-v1|<weights_sha256>")`
  — a changed image, preprocessing rule or weight file produces a different
  file, so stale entries cannot be silently reused.
* Per-image provenance (source path, SHA256, protocol, input/output shape,
  cache path) is in `reports/s1_feature_cache_smoke.json::images`.

`*.npy` is git-ignored; the cache lives under `external_data/`.

## 6. Scope

This cache serves the **frozen-encoder** stage only. It is *not* a forward pass
for a LoRA-finetuned encoder: if the encoder is ever trained, the cache key must
change (add a fine-tuning run id) and the cache must be regenerated. Nothing in
S1 trains or fine-tunes the encoder.

## 7. Open items

1. Image-quality filter: 1 of 97 sampled images is near-white (mean pixel 225.9).
   Add a brightness/contrast rejection rule before batch caching.
2. Batch caching throughput was measured single-stream; batched extraction is
   expected to be several times faster but was not measured.
3. The 153,022-image cache is **not** written yet (projected 56.1 GiB). Decide
   the exact plot subset first.
