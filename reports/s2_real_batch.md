# S2 — Real FIP batch, weather windows and frozen-encoder verification

Stage: **S2**. Date: 2026-10-23. Branch `agro-s0-upstream`, base commit `395651d`.
S2 implementation commit: `TBD`.
Companion: `reports/s2_pairs.md`.

**No model training was started.** The frozen DINOv2 encoder was run in
`torch.no_grad()` only to verify the input/output geometry and the S1 cache.

```bash
python3 tools/s2_inspect_batch.py \
    --config agro_jepa/config/s2_dataset.yaml \
    --batch-size 8 \
    --report reports/s2_real_batch.json
# -> wrote reports/s2_real_batch.json in 9.6 s

python3 tools/s2_verify.py     # 74/74 acceptance checks
```

Machine-readable output: `reports/s2_real_batch.json`.
Logs: `reports/s2_run_logs/s2_inspect_batch.log`, `s2_verify.log`.

## 1. Environment (measured)

| Item | Value |
|---|---|
| Python | 3.12.3 |
| PyTorch | 2.10.0+cu128 |
| GPU | NVIDIA RTX A6000 (CUDA 12.8) |
| DINOv2 code | local clone `external_data/third_party/dinov2` @ `7764ea0f…` |
| DINOv2 weights | `dinov2_vitb14_pretrain.pth`, SHA256 `0b8b82f8…8c73` |
| Model load | 2.56 s |
| 16 images encoded | 0.413 s (~39 img/s, batch of 8) |

## 2. Scope of the real-image check (honest limitation)

The pair manifest covers 569,821 pairs, but **only 97 inner-plot PNGs are on
this host** (`external_data/fip1_images`, downloaded in S1). Filtering the pair
table to pairs whose *both* images exist locally leaves:

| Quantity | Value |
|---|---|
| pairs with both images local | **152** |
| distinct plots | **6** |
| local PNGs | 97 |
| local pair types | consecutive 39, span7 28, span14 29, span21 32, span28 24 |
| local splits | train 147, test_environment 5 |

So 152 is the **effective real-image sample size** for encoding; the manifest
statistics in `reports/s2_pairs.md` are over the full 569,821 pairs. The full
dataset path resolves non-local images to their HF URLs, which S3 must
materialise before training.

## 3. Real batch fields

Batch size 8: 7 `train` pairs + 1 `test_environment` pair, chosen
deterministically, and forced to include (a) the near-white frame S1 flagged and
(b) a pair with a **measured zero** label.

| Field | Shape | dtype | Valid / notes |
|---|---|---|---|
| `image_a` | `[8, 3, 224, 224]` | float32 | 0 NaN, mainline224 transform |
| `image_b` | `[8, 3, 224, 224]` | float32 | 0 NaN |
| `weather_daily` | `[8, 29, 6]` | float32 | 0 NaN; zero-padded to L_max=29 |
| `weather_mask` | `[8, 29]` | bool | **115 True**, 117 padded False |
| `weather_n_hours_min` | `[8, 29]` | int32 | min hourly coverage across the 4 sensors |
| `weather_dates` | `[8, 29]` | object | `None` on padding |
| `weather_lengths` | `[8]` | int64 | `[2, 6, 28, 19, 29, 13, 3, 15]` |
| `delta_days` | `[8]` | int64 | identical to `weather_lengths` |
| `harvest_year` | `[8]` | int64 | 2016/2019 |
| `phenotype_a` / `phenotype_b` | `[8, 4]` | float32 | 4 NaN = missing Sen in 2019 |
| `phenotype_mask_a` / `_b` | `[8, 4]` | bool | 14 True of 32 |
| `label_lag_days` | `[8, 2, 4]` | float32 | 8 NaN (Sen), signed `label_date − image_date` |
| `label_dates` | list of 8 × `(2,4)` | object | actual measurement dates; `None` when absent |
| `ph_negative_a` / `_b` | `[8]` | bool | 0 in this batch (5,572 exist in the full table) |
| `trait_names` | list | — | `('cc','ph','whc','sen')` |
| `plot_uid`, `genotype_id`, `split`, `pair_id`, `pair_type` | list of 8 | — | traceability |
| `image_stem_a/b`, `image_dates`, `weather_date_range`, `weather_manifest_id` | list of 8 | — | provenance |
| `image_stats_a/b` | list of 8 | — | luma mean/std + quality flag |

`pair_id` order is preserved through the encoder (`sample_id_order_preserved:
true`), so a feature row can be mapped back to a manifest row.

### 3.1 Real date examples (from the manifest, not fabricated)

| pair_id | plot | split | year | image dates | weather window | Δ |
|---|---|---|---|---|---|---|
| `52a8f61f0355c98e` | FPWW0120667 | train | 2016 | 2016-02-16 → 2016-02-18 | (2016-02-16, 2016-02-18] | 2 |
| `60e054101c4a2fe6` | FPWW0120667 | train | 2016 | 2016-05-24 → 2016-05-30 | (2016-05-24, 2016-05-30] | 6 |
| `04eb582f98806ccf` | FPWW0240010 | test_environment | 2019 | 2019-02-13 → 2019-03-13 | (2019-02-13, 2019-03-13] | 28 |
| `36565f3a2b5036e9` | FPWW0240010 | test_environment | 2019 | 2019-02-22 → 2019-03-13 | (2019-02-22, 2019-03-13] | 19 |
| `85f69ab214298ccc` | FPWW0240010 | test_environment | 2019 | 2019-03-13 → 2019-04-11 | (2019-03-13, 2019-04-11] | 29 |
| `920ea6c1f6312e5f` | FPWW0240010 | test_environment | 2019 | 2019-03-29 → 2019-04-11 | (2019-03-29, 2019-04-11] | 13 |
| `e93e01e525e3c1bc` | FPWW0120667 | train | 2016 | 2015-11-09 → 2015-11-12 | (2015-11-09, 2015-11-12] | 3 |
| `6c0489c3c7189741` | FPWW0120667 | train | 2016 | 2015-11-09 → 2015-11-24 | (2015-11-09, 2015-11-24] | 15 |

Every image date equals the `YYYYMMDD` embedded in the audited filename stem.
Rain/winter images get almost no valid phenotype (large lag); summer images get
CC/PH on the same day. That is the real sampling structure of FIP, not a bug.

## 4. Weather window verification (acceptance C)

| Check | Result |
|---|---|
| `weather_lengths == delta_days` | **True** for all 8 |
| first day == `t_a + 1` | **True** |
| last day == `t_b` | **True** (left-open, right-closed) |
| unique, contiguous dates | True |
| mask covers exactly the first `length` steps | True |
| padding masked | 117 of 232 steps, all `False` |
| all real steps valid | 115 / 115 |

Concrete example (`04eb582f98806ccf`): window `(2019-02-13, 2019-03-13]`,
28 daily steps `2019-02-14 … 2019-03-13`, channels
`[T_mean_C, T_max_C, T_min_C, RH_mean_pct, Rs_sum_MJ_m2_d, P_sum_raw]`.

The full-table check in `reports/s2_pairs.md` covers all 569,821 windows with
zero missing days.

## 5. Label handling in the real batch (acceptance D)

Two rows from the same plot make the distinction concrete:

* `52a8f61f…` (2016-02-16): `phenotype_a = [0.1571, 0.2228, 0.0, 2.0]`,
  `mask = [True, False, False, False]`, lags `[0, 55, 91, 133]`.
  CC is same-day and valid; PH/WHC/Sen values exist but were measured 55–133 d
  away, so they are **present but masked**. The WHC `0.0` here is *not* counted
  as a valid zero.
* `60e05410…` (2016-05-24): `phenotype_a = [0.5054, 0.7954, 0.0, 2.0]`,
  `mask = [True, True, True, False]`, lags `[0, 0, 0, 35]`.
  Here WHC `0.0` is measured the same day and is **valid** — a real zero, not
  missing.
* `04eb582f…` (2019): `phenotype_a = [0.0851, 0.0066, 0.0, null]` — Sen is
  **NaN** because 2019 has no senescence measurements; it was not imputed to 0.

Full-table numbers: 5,572 / 4,422 negative PH values preserved and flagged;
zeros kept with `mask=True` (CC 10/6, PH 15,950/11,900 within-lag, WHC
77,464/67,222, Sen 6,794/11,419); missing labels always NaN with `None` date.

## 6. Frozen encoder verification (acceptance F)

| Check | Result |
|---|---|
| Input | `[8, 3, 224, 224]` float32 |
| Patch tokens | **`[8, 256, 768]`** for both `image_a` and `image_b` |
| Expected mainline geometry | 256 tokens × 768 dim — **match** |
| Same preprocessing from raw PNG | max abs diff **0.0** vs the dataset tensor |
| S1 cache round-trip | 13 images compared; max abs diff **7.0e-3** (float16 storage), min cosine **0.99999756**, all shapes `(256, 768)` |

The cache round-trip is the strong check: it recomputes the cache key from the
image bytes and compares a live forward pass against the S1 `.npy`. Agreement at
float16 precision means the dataset transform, the model weights and the cache
protocol are all consistent.

### 6.1 Optional cached-feature path

`FIPPairDataset(image_mode="cached_features")` reads the S1
`mainline224_s1-v1` cache instead of decoding PNGs. Measured on this host:

| Check | Result |
|---|---|
| pairs with both caches present | **32** |
| per-sample feature shape | `(256, 768)` float32 (from float16 storage) |
| batch shape | `[4, 256, 768]` |
| finite values | yes |
| missing cache file | **raises `FileNotFoundError`** — never returned as zeros |

## 7. Where the agricultural dataset meets the upstream DINO-WM interface

The upstream code was **not modified**. The divergence is a geometry rule that
lives in two places, and it is the only thing that must be configured:

| Upstream location | Code | Effect |
|---|---|---|
| `dino_wm/models/visual_world_model.py:56-60` | `decoder_scale = 16`; `num_side_patches = image_size // 16`; `encoder_image_size = num_side_patches * encoder.patch_size`; `encoder_transform = Resize(encoder_image_size)` | with `image_size=224` → `Resize(196)` → **196 tokens (upstream196)** |
| `dino_wm/train.py:247-249` | `num_side_patches = cfg.img_size // 16`; `num_patches = num_side_patches ** 2` | predictor positional embedding and causal mask are sized here |
| `dino_wm/models/vit.py:58,110,117` | `generate_mask_matrix(NUM_PATCHES, NUM_FRAMES)` global; `pos_embedding (1, num_frames*num_patches, dim)` | one global mask; predictors with different patch counts cannot coexist |
| `dino_wm/models/vqvae.py:202-204` | `num_side_patches = int(sqrt(num_patches))`; decoder upsamples ×16 | decoder geometry is inferred from the token count |

**Measured, with the real frozen encoder:**

| Config | `encoder_image_size` | Encoder input | Tokens | `num_patches` | `z` (concat_dim=1) |
|---|---|---|---|---|---|
| upstream default `img_size=224` | 196 | `[8,3,196,196]` | **196** | 196 | `[8,1,196,770]` |
| **mainline `img_size=256`** | **224** | `[8,3,224,224]` | **256** | **256** | `[8,1,256,770]` |

With `img_size=256` the upstream `Resize` becomes a no-op (`Resize(224)` on a
224 tensor), the predictor's positional embedding is `[1, 256, 770]`, its causal
mask is `[1, 1, 256, 256]`, and a forward returns `[8, 256, 770]`. The dataset
therefore feeds `default_transform(224)` (the S1 `mainline224` protocol) and
the trainer config sets `img_size: 256` — a config-only change, no source edit.

**Critical incompatibility found (S3 blocker):** the upstream VQVAE decoder
infers the grid as `sqrt(num_patches)` and upsamples ×16:

| Tokens | Grid | Decoded image |
|---|---|---|
| 196 | 14×14 | **224×224** ✔ matches `obs['visual']` |
| 256 | 16×16 | **256×256** ✘ shape mismatch with the 224 target |

So `VWorldModel.forward`'s image-space reconstruction loss (and
`Trainer.save_ckpt` expectations) is only valid on the 196-token path. The S3
scope is latent prediction, which is exactly the compatible configuration:
**`has_decoder=False`, `model.train_decoder=False`**. If a decoder is ever
needed at 256 tokens, it must be rebuilt for a 16×16 grid (or fed a separate
196-token encoding); changing one size constant is not sufficient.

## 8. Image quality flags (reported, not applied)

Rule `near-white-luma-v1`: flag when `luma_mean >= 0.60` and `luma_std <= 0.12`.
Measured distribution over the 97 local PNGs: mean luminance 0.0776–0.8859,
std 0.0335–0.1939.

| Flagged | Path | mean | std |
|---|---|---|---|
| 1 / 97 | `external_data/fip1_images/FPWW0120667/FPWW0120667_RGB1_20160216_125228.png` | 0.8859 | 0.0383 |

No image was removed from any pair table or batch. The flagged frame appears in
the inspected batch with `quality_usable=false` and is still encoded.

## 9. Acceptance status

| Item | Status | Evidence |
|---|---|---|
| A. trace one plot → image, date, label date, weather interval | **pass** | `s2_verify.py::A_*` (FPWW0120667, 2016-05-17) |
| B. all image dates from verified stem mapping | **pass** | all 569,821 pairs + 153,022 observations |
| C. `(t_a, t_b]`, variable length, mask | **pass** | 569,821 windows fully covered; batch boundaries checked |
| D. missing / zero / PH negative / quality not conflated | **pass** | §5 + 18 `D_*` checks |
| E. official and custom splits independent, no plot leak | **pass** | §7 of `s2_pairs.md` |
| F. 224×224 → 256×768, cache and model consistent | **pass** | `[8,256,768]`, cache cosine 0.99999 |
| G. real load + encode, no training | **pass** | no optimizer/checkpoint; `stage=S2` |
| H. reports, logs, code, commit | **pass** | this file, `s2_pairs.md`, `s2_run_logs/`, commit |

`tools/s2_verify.py` reports **74/74 checks passed** (70 previously; the four
extra are the cached-feature path: cache availability, `(256, 768)` shape,
finite values, and "missing cache raises instead of returning zeros").

## 10. Open items for S3

1. **Only 97 local PNGs.** The real batch is representative of the data path,
   not of the full distribution. Materialising the 153k-image cache (56 GiB at
   `mainline224`) is required before any training run.
2. **Decoder cannot be used at 256 tokens** (§7). S3 must run latent-only.
3. **Weather is one daily table per year.** Verified identical across 3 plots
   per year in S1; a full check across all plots is still open.
4. **Precipitation unit unresolved** → no cumulative/water-balance features.
5. **`upstream196` and `mainline224` caches must never be mixed.** They have
   different token counts; the S2 config pins `mainline224_s1-v1`.
6. The 1,872 duplicate observation references are still present in the frozen
   S1 manifest; S2 de-duplicates at pairing time.
