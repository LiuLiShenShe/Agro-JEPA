# S3 — Real FIP 1.0 image acquisition (Pilot)

Stage: **S3**. Date: 2026-10-24. Branch: `agro-s0-upstream`.

This report covers the image materialisation step that had to precede any S3 model training. It is generated from the run artifacts by `tools/s3_write_reports.py`.

## 1. Selection rule

- rule: all official validation plots + a fixed-seed per-year sample of official train plots; every audited image of every selected plot
- seed: `42`
- train plots per year requested: `200`
- train plots per year selected: `{'2016': 200, '2017': 200, '2018': 200, '2021': 200, '2022': 200}`
- official validation plots: **all** (380)
- only official `train` and `validation` are eligible. No official test image is used for model selection.
- for every selected plot **all** audited images are fetched, so each plot contributes its whole autumn→winter→stem-elongation→heading→senescence curve.

## 2. Download mechanics

- source: HuggingFace dataset `mikeboss/FIP1`, revision `11cdd84bc0ba8d5c6641ea7b53550b1658527e1c`
- endpoint: `https://hf-mirror.com` (direct `huggingface.co` is unreachable from this host; the mirror is required)
- resumable: an existing PNG is re-hashed rather than re-downloaded; a truncated file cannot be accepted because the PNG signature and minimum size are checked
- retries: 5 attempts with linear backoff; the mirror answers `429 Too Many Requests` under high concurrency
- every image row records plot, split, harvest year, image date, source URL, byte size and SHA256

## 3. Actual acquisition

| Metric | Value |
|---|---|
| images on disk (selection) | 49,058 |
| plots | 1,380 |
| harvest years | 2016, 2017, 2018, 2021, 2022 |
| bytes | 53.35 GB |
| failed downloads after retries | 0 |
| final manifest wall clock | 10.1 min |
| final manifest throughput | 81.26 img/s |
| image resolution (source) | 960 x 640 RGB PNG |
| selection manifest | data/manifests/s3_image_selection.json |
| selection manifest SHA256 | b0bac953f33288ce513699e198d72ec720a55302cd5aac20028633fd4da3a63b |

The final manifest numbers above describe the **retry pass**, which mostly re-hashed existing files; the real network acquisition is the sum of the two passes:

| pass | images | plots | GB | minutes | img/s | failures |
|---|---|---|---|---|---|---|
| main pass (40 workers) | 46,037 | 1,374 | 49.94 | 97.7 | 7.85 | 3021 |
| retry pass (12 workers) | 49,058 | 1,380 | 53.35 | 10.1 | 81.26 | 0 |

### 3.1 Coverage by split and harvest year

| split | year | images downloaded | plots downloaded | official images | official plots |
|---|---|---|---|---|---|
| train | 2016 | 7176 | 200 | 16,256 | 452 |
| train | 2017 | 3779 | 200 | 9,285 | 492 |
| train | 2018 | 5635 | 200 | 14,522 | 520 |
| train | 2021 | 11880 | 200 | 44,081 | 742 |
| train | 2022 | 9200 | 200 | 33,434 | 724 |
| validation | 2016 | 4319 | 119 | 4,319 | 119 |
| validation | 2017 | 2572 | 126 | 2,572 | 126 |
| validation | 2018 | 2850 | 102 | 2,850 | 102 |
| validation | 2021 | 652 | 11 | 652 | 11 |
| validation | 2022 | 995 | 22 | 995 | 22 |

### 3.2 Fraction of the official split obtained

| split | images | plots |
|---|---|---|
| train | 37,670 / 117,578 (32.0%) | 1,000 / 2,930 |
| validation | 11,388 / 11,388 (100.0%) | 380 / 380 |

**The result is a Pilot, not a full-FIP baseline.** The full official train split has 2,930 plots / 117,578 unique images; S3 selected train|2016=200, train|2017=200, train|2018=200, train|2021=200, train|2022=200, validation|2016=119, validation|2017=126, validation|2018=102, validation|2021=11, validation|2022=22 plots per split/year and downloaded all images of those plots. All 380 validation plots were downloaded, so validation is complete at the plot level.

## 4. Feature cache

- protocol: `mainline224_s1-v1` (identical to S1: ToTensor → Resize(224) → CenterCrop(224) → Normalize(0.5) → DINOv2 ViT-B/14 → `x_norm_patchtokens`, stored float16)
- DINOv2 code commit: `7764ea0f912e53c92e82eb78a2a1631e92725fc8`
- weights SHA256: `0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73`
- unique images encoded: 49,058 (13,530 already cached, 35,528 computed)
- throughput: 131.48 img/s; peak GPU memory 1.16 GB
- cache round-trip vs a live forward pass: cos=0.99999899, max|diff|=6.37e-03; cos=0.99999923, max|diff|=3.89e-03; cos=0.99999911, max|diff|=4.21e-03; cos=0.99999934, max|diff|=3.91e-03
- `upstream196` and `mainline224` are never mixed: the index and the checkpoint record the `mainline224_s1-v1` cache only.

## 5. S3 pair index (the training/validation sample)

| Filtering step | rows |
|---|---|
| S2 span7/span14 train+validation rows | 194,998 |
| both images downloaded | 71,314 |
| after physical-pair deduplication | 71,314 |
| after requiring both feature-cache files | 71,314 |

| split | pairs | unique images | unique physical pairs | plots | years | delta_days |
|---|---|---|---|---|---|---|
| train | 56,139 | 36,631 | 56,139 | 1,000 | 5 | 5–16 (median 9.0) |
| validation | 15,175 | 10,858 | 15,175 | 380 | 5 | 5–16 (median 9.0) |

| split | CC valid at t_b | PH valid at t_b | CC valid at both ends | PH valid at both ends |
|---|---|---|---|---|
| train | 54,564 | 46,462 | 53,704 | 44,452 |
| validation | 14,306 | 11,033 | 13,959 | 10,550 |

Raw PH negatives are preserved (not clipped) and included in the probe targets: **314** full-table PH values at `t_b` in train, **92** in validation. A positivity-only sensitivity is left to S4.

Time normalisation is fitted on the **training split only**: mean 10.2529, std 3.6748, n=56,139. Validation reuses these parameters.

Frozen index SHA256: `f6ef1871655277549aba2d883b6f76a5831b7976af90fc74d80206591d884933`

## 6. Provenance summary

| Artifact | Path | SHA256 |
|---|---|---|
| downloaded image selection | data/manifests/s3_image_selection.json | b0bac953f33288ce513699e198d72ec720a55302cd5aac20028633fd4da3a63b |
| feature index | external_data/s3_features/index.parquet | 158b31dafad13f09cce46ca32abb695e… |
| S3 pair index | external_data/s3_index/fip1_s3_v0_index.parquet | f6ef1871655277549aba2d883b6f76a5… |
| S3 selection rule | data/manifests/s3_selection_rule.json | b6f3bf1910564b33d56ae9c1d5da610e… |

## 7. Limitations (stated, not hidden)

1. **Pilot scale.** 1,000 of 2,930 official train plots are used; this is not a full-FIP training set. Validation is complete at the plot level.
2. **Mirror throttling.** The HF mirror intermittently returns HTTP 429; the downloader retries and is resumable, but a re-run is required to close any remaining gaps (recorded as `n_failed`).
3. **One daily weather table per year** is not used by V0 (no weather encoder in S3); S4 will revisit it.
4. **Precipitation units remain unresolved** in S1/S2; no water-balance quantity is computed here.
5. The 1,872 duplicate image references in the frozen S1 observation manifest are still present; S2/S3 de-duplicate at pairing time.

