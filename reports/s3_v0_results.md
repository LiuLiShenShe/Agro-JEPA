# S3 — FIP-DINO-WM V0 results (E1 future latent + E2 CC/PH probes)

Stage: **S3**. Date: 2026-10-24. Branch: `agro-s0-upstream`.

All numbers below are read from `reports/s3_v0_results.json` and `reports/s3_v0_val_predictions.parquet`; nothing is pre-filled.

## 0. TL;DR

- **E1**: V0 validation MSE **1.13656** vs persistence **2.19515** (+48.22% relative change).
- Mean cosine to the future feature: V0 **0.7829** vs persistence **0.6279**.
- Stronger trivial baseline, train-mean future feature: **1.83567**; delta-conditioned train mean: **1.81724**.
- V0 **beats** the train-mean baseline.
- **E2 CC** (overall, n_val=8496): A current-feature R²=0.7978 / MAE=0.0821; B predicted R²=0.8458 / MAE=0.0692.
- **E2 PH** (overall, n_val=6302): A current-feature R²=0.9470 / MAE=0.0603; B predicted R²=0.9541 / MAE=0.0558.

- **Irregular expansion** (span7 + span14 + consecutive, 21,617 val pairs): MSE **1.11788** vs persistence **2.10549** (+46.91%); the true-consecutive subset alone gives +42.90%.

## 1. E1 — future latent prediction

Validation split only (official `validation` plots). Effective sample: **15,175 pairs**, **380 plots**.

| Predictor | MSE ↓ | Mean cosine ↑ | Relative improvement vs persistence |
|---|---|---|---|
| Persistence (`z_a` as future) | 2.19515 | 0.6279 | — |
| Train-mean future feature | 1.83567 | n/a | +16.38% |
| Train-mean per exact delta | 1.81724 | n/a | +17.22% |
| **FIP-DINO-WM V0** | **1.13656** | **0.7829** | +48.22% |

**Δt diagnostics (inference only, no retraining).** Shuffling the real `delta_days` across the validation batch increases MSE by **+0.41%**; replacing every interval with the train mean increases it by **+11.44%**. A near-zero change would mean the predictor ignores the time channel.

### 1.1 By nominal span

| span | pairs | plots | MSE V0 | MSE persist | MSE train-mean | rel. vs persist | cos V0 | cos persist |
|---|---|---|---|---|---|---|---|---|
| span14 | 7358 | 380 | 1.16618 | 2.34388 | 1.82830 | +50.25% | 0.7763 | 0.6029 |
| span7 | 7817 | 380 | 1.10868 | 2.05515 | 1.84260 | +46.05% | 0.7890 | 0.6515 |

### 1.2 By harvest year

| year | pairs | plots | MSE V0 | MSE persist | MSE train-mean | rel. vs persist | cos V0 | cos persist |
|---|---|---|---|---|---|---|---|---|
| 2016 | 5201 | 119 | 1.11101 | 2.03972 | 1.87633 | +45.53% | 0.7931 | 0.6609 |
| 2017 | 3485 | 126 | 1.13350 | 2.31043 | 1.80989 | +50.94% | 0.7834 | 0.6090 |
| 2018 | 3831 | 102 | 1.15498 | 2.27081 | 1.81821 | +49.14% | 0.7734 | 0.6066 |
| 2021 | 1131 | 11 | 1.21079 | 2.29983 | 1.80841 | +47.35% | 0.7638 | 0.6059 |
| 2022 | 1527 | 22 | 1.12937 | 2.19410 | 1.81997 | +48.53% | 0.7847 | 0.6287 |

### 1.3 By realised interval

| delta (days) | pairs | plots | MSE V0 | MSE persist | MSE train-mean | rel. vs persist | cos V0 | cos persist |
|---|---|---|---|---|---|---|---|---|
| 15-21 | 2177 | 380 | 1.15853 | 2.38629 | 1.79345 | +51.45% | 0.7795 | 0.5987 |
| 4-7 | 5990 | 380 | 1.10773 | 2.05307 | 1.82984 | +46.05% | 0.7884 | 0.6505 |
| 8-14 | 7008 | 380 | 1.15438 | 2.25721 | 1.85376 | +48.86% | 0.7792 | 0.6177 |

Per-sample rows: `reports/s3_v0_val_predictions.parquet` (pair_id, plot, split, span, delta, dates, both MSEs, both cosines).

### 1.4 Irregular-interval expansion (`span7 + span14 + consecutive`)

A second V0 model trained on the same plots and encoder with the S2 true-consecutive pairs added. The realised interval is the only difference; the model and loss are unchanged.

| Predictor | MSE ↓ | Mean cosine ↑ | Relative improvement vs persistence |
|---|---|---|---|
| Persistence (`z_a` as future) | 2.10549 | 0.6434 | — |
| Train-mean future feature | 1.84369 | n/a | +12.43% |
| **V0 + consecutive pairs** | **1.11788** | **0.7873** | +46.91% |

| span | pairs | plots | MSE V0 | MSE persist | MSE train-mean | rel. vs persist | cos V0 | cos persist |
|---|---|---|---|---|---|---|---|---|
| consecutive | 6442 | 378 | 1.08164 | 1.89428 | 1.86150 | +42.90% | 0.7957 | 0.6796 |
| span14 | 7358 | 380 | 1.16340 | 2.34388 | 1.82873 | +50.36% | 0.7771 | 0.6029 |
| span7 | 7817 | 380 | 1.10491 | 2.05515 | 1.84309 | +46.24% | 0.7900 | 0.6515 |

| delta (days) | pairs | plots | MSE V0 | MSE persist | rel. vs persist | cos V0 | cos persist |
|---|---|---|---|---|---|---|---|
| 1-3 | 3110 | 370 | 1.08827 | 1.84069 | +40.88% | 0.7923 | 0.6856 |
| 4-7 | 8083 | 380 | 1.09090 | 1.99631 | +45.35% | 0.7924 | 0.6606 |
| 8-14 | 7705 | 380 | 1.14644 | 2.23614 | +48.73% | 0.7815 | 0.6218 |
| 15-21 | 2254 | 380 | 1.15896 | 2.38410 | +51.39% | 0.7800 | 0.5996 |
| 22-28 | 138 | 136 | 1.11521 | 2.62487 | +57.51% | 0.7902 | 0.5530 |
| 29+ | 327 | 325 | 1.11153 | 2.10436 | +47.18% | 0.8011 | 0.6627 |

Per-sample rows: `reports/s3_v0_val_predictions_consecutive.parquet`; full JSON: `reports/s3_v0_results_consecutive.json`.

## 2. E2 — future CC / PH linear probes

Two readouts with the **same architecture** (linear on the mean-pooled 768-d feature, inputs standardised with train statistics), the same optimiser and the same epoch budget. Seeds are averaged (mean ± std).

A = current DINOv2 feature `mean(Z_a)`; B = V0 predicted future feature `mean(Z_hat_b)`.

Targets are the raw phenotype at `t_b` with S1/S2's `|lag| <= 7 d` mask (CC in fraction 0–1, PH in metres). This is a **near-neighbour measurement** result, not an exact-date result; the ±3 d sensitivity is reported where the sample allows.

### 2.1 Scope: `overall`

| trait | readout | n train | n val | MAE | RMSE | R² |
|---|---|---|---|---|---|---|
| cc | A current | 31,146 | 8,496 | 0.0821 ± 0.0002 | 0.1075 ± 0.0002 | 0.7978 ± 0.0008 |
| cc | B predicted | 31,146 | 8,496 | 0.0692 ± 0.0002 | 0.0939 ± 0.0002 | 0.8458 ± 0.0008 |
| ph | A current | 25,372 | 6,302 | 0.0603 ± 0.0004 | 0.0816 ± 0.0004 | 0.9470 ± 0.0005 |
| ph | B predicted | 25,372 | 6,302 | 0.0558 ± 0.0005 | 0.0760 ± 0.0004 | 0.9541 ± 0.0005 |

B minus A (negative MAE/RMSE is better; positive R² is better):

| trait | ΔMAE | ΔRMSE | ΔR² |
|---|---|---|---|
| cc | -0.01283 | -0.01363 | +0.04802 |
| ph | -0.00455 | -0.00567 | +0.00712 |

Label timing (signed `label_date - image_date_b`, days):

| trait | split | median | |median| | |p90| | n |lag|≤3 | n |lag|≤7 |
|---|---|---|---|---|---|---|
| cc | train | 0.0 | 0.0 | 0.0 | 29948 | 31146 |
| cc | validation | 0.0 | 0.0 | 0.0 | 8112 | 8496 |
| ph | train | 0.0 | 0.0 | 2.0 | 23851 | 25372 |
| ph | validation | 0.0 | 1.0 | 4.0 | 5401 | 6302 |

### 2.2 Scope: `span7`

| trait | readout | n train | n val | MAE | RMSE | R² |
|---|---|---|---|---|---|---|
| cc | A current | 25,128 | 6,542 | 0.0795 ± 0.0003 | 0.1037 ± 0.0005 | 0.8165 ± 0.0019 |
| cc | B predicted | 25,128 | 6,542 | 0.0688 ± 0.0002 | 0.0932 ± 0.0002 | 0.8517 ± 0.0006 |
| ph | A current | 21,143 | 5,046 | 0.0591 ± 0.0004 | 0.0795 ± 0.0004 | 0.9477 ± 0.0005 |
| ph | B predicted | 21,143 | 5,046 | 0.0530 ± 0.0001 | 0.0729 ± 0.0001 | 0.9560 ± 0.0001 |

B minus A (negative MAE/RMSE is better; positive R² is better):

| trait | ΔMAE | ΔRMSE | ΔR² |
|---|---|---|---|
| cc | -0.01069 | -0.01049 | +0.03526 |
| ph | -0.00604 | -0.00657 | +0.00828 |

Label timing (signed `label_date - image_date_b`, days):

| trait | split | median | |median| | |p90| | n |lag|≤3 | n |lag|≤7 |
|---|---|---|---|---|---|---|
| cc | train | 0.0 | 0.0 | 0.0 | 23937 | 25128 |
| cc | validation | 0.0 | 0.0 | 0.0 | 6159 | 6542 |
| ph | train | 0.0 | 0.0 | 3.0 | 19866 | 21143 |
| ph | validation | 0.0 | 1.0 | 4.0 | 4333 | 5046 |

### 2.3 Scope: `span14`

| trait | readout | n train | n val | MAE | RMSE | R² |
|---|---|---|---|---|---|---|
| cc | A current | 24,675 | 6,322 | 0.0821 ± 0.0001 | 0.1067 ± 0.0001 | 0.8108 ± 0.0005 |
| cc | B predicted | 24,675 | 6,322 | 0.0697 ± 0.0001 | 0.0954 ± 0.0003 | 0.8487 ± 0.0011 |
| ph | A current | 21,230 | 4,972 | 0.0610 ± 0.0003 | 0.0819 ± 0.0001 | 0.9462 ± 0.0002 |
| ph | B predicted | 21,230 | 4,972 | 0.0562 ± 0.0003 | 0.0768 ± 0.0004 | 0.9526 ± 0.0005 |

B minus A (negative MAE/RMSE is better; positive R² is better):

| trait | ΔMAE | ΔRMSE | ΔR² |
|---|---|---|---|
| cc | -0.01243 | -0.01129 | +0.03791 |
| ph | -0.00480 | -0.00504 | +0.00641 |

Label timing (signed `label_date - image_date_b`, days):

| trait | split | median | |median| | |p90| | n |lag|≤3 | n |lag|≤7 |
|---|---|---|---|---|---|---|
| cc | train | 0.0 | 0.0 | 0.0 | 23815 | 24675 |
| cc | validation | 0.0 | 0.0 | 0.0 | 6055 | 6322 |
| ph | train | 0.0 | 0.0 | 2.0 | 20167 | 21230 |
| ph | validation | 0.0 | 1.0 | 5.0 | 4381 | 4972 |

### 2.4 Strict-date sensitivity (`|lag| <= 3 d`)

| trait | n train | n val | R² A | R² B | MAE A | MAE B |
|---|---|---|---|---|---|---|
| cc | 29,948 | 8,112 | 0.8130 | 0.8563 | 0.0794 | 0.0674 |
| ph | 23,851 | 5,401 | 0.9482 | 0.9561 | 0.0585 | 0.0536 |

## 3. Interpretation

1. **Does V0 beat persistence?** Yes — validation MSE changes by +48.22%. Per-span relative changes: span14=+50.25%, span7=+46.05%; improvement is positive for every span.
2. **Does V0 beat a train-mean future predictor?** Yes (V0 1.13656 vs train-mean 1.83567, delta-conditioned mean 1.81724). This is the harder trivial baseline; a predictor that only learns the average future feature would look good against persistence but adds no image-conditioned information.
3. **Do the predicted features carry CC/PH information?** B is trained on V0 predictions; A reads the current image. CC: ΔMAE=-0.01283, ΔR²=+0.04802; PH: ΔMAE=-0.00455, ΔR²=+0.00712. Negative ΔMAE / positive ΔR² means the predicted future latent is more useful than the current feature for that target; the opposite means it is not.

## 4. Acceptance (S3 §10)

`tools/s3_verify.py`: **45/45** checks passed. The full check list is in `reports/s3_verify.json`.

| Item | Status |
|---|---|
| A. real train/validation image subset with plot coverage | pass (acquisition report) |
| B. real 224×224 → 256×768 DINOv2 tokens | pass (verify B) |
| C. `predict(z_a, delta)` → [B,256,768] | pass (verify C) |
| D. future features stop-gradient only | pass (verify D) |
| E. real optimisation + checkpoint + resume | pass (verify E) |
| F. V0 vs persistence on validation | pass (this report §1) |
| G. CC/PH probes vs current feature | pass (this report §2) |
| H. code/config/checkpoint/index/results traceable | pass (SHA256s in reports) |

## 5. Limitations

1. **Pilot**: 1,000 of 2,930 official train plots; all 380 validation plots. The result is not a full-FIP baseline.
2. **Label timing**: the primary probe uses `|lag| <= 7 d`; the ±3 d sensitivity is reported separately and has fewer samples.
3. **Probe capacity**: a linear readout on mean-pooled tokens is deliberately simple; it does not prove that no nonlinear readout could extract more from the same latent.
4. **No weather, physiology, EMA, LoRA or multi-step rollout** in V0; those are S4 comparisons, not claims of this stage.
5. The mean-future baseline is a stronger control than persistence; the V0-vs-mean comparison is the one that tests image conditioning.

