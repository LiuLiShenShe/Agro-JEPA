# S4B.3 E2 -- CC/PH Readout Protocol (frozen before test labels)

- Generated: 2026-10-28T10:42:24+0800
- Git HEAD: `a2856f81cbd529a9ee0fc6c1c9b900c0ef62789f` (agro-s4b3-official-test)
- Config: `agro_jepa/config/s4b3_official_test.yaml` (sha `34ec856035ae66e5`)

## Train scope

- All **2,930 official-train plots** (179,823 pairs).
- Official validation and every test plot are excluded from training and from epoch/model selection.

## Frozen inner split (seed 20261025, 80/20, stratified by harvest_year)

- File: `reports/s4b3_e2_plot_split.csv` (sha `42e8cf9aa32ca2d19cd8f37d4d2b883e9790dd7608f76aae3bb6b99f14744889`)
- inner-train: 2345 plots / inner-dev: 585 plots

| harvest_year | plots | inner-train | inner-dev |
|---|---:|---:|---:|
| 2016 | 452 | 362 | 90 |
| 2017 | 492 | 394 | 98 |
| 2018 | 520 | 416 | 104 |
| 2021 | 742 | 594 | 148 |
| 2022 | 724 | 579 | 145 |

## Conditions

- **A_current**: mean(z_a) current frozen DINOv2 feature
- **B2**: mean(z_hat_b) from b2_no_time
- **B2-C**: mean(z_hat_b) from b2_no_time_constant_delta
- **B3**: mean(z_hat_b) from b3_delta (real delta_days)

## Label matching

- Source: frozen S2 span7/span14 pair table joined on pair_id
- Primary window: `value finite AND date finite AND |label_lag_days| <= lag_max_abs_days; a missing label is INVALID and is never filled with 0` (lag <= 7 d).
- Sensitivity: strict-lag3 (lag <= 3 d) and same-day (lag == 0).
- Missing labels are invalid, never 0.

## Readout

- single linear layer (PooledReadout, 768 -> 1); mean over 256 patch tokens.
- x standardised by inner-train mean/std; y by inner-train mean/std.
- AdamW + cosine annealing anchored to the candidate budget; candidate 150 epochs, patience 20, batch 256, lr 0.001, wd 0.01, seeds [0, 1, 2].
- Selection: minimum inner-dev MSE over the candidate budget; median selected epoch across probe seeds.
- Final: refit on all rows of the full official-train set at the selected epoch; evaluated once per test split.
- Test labels are never used for training, epoch choice or model selection.

## Protocol checks

| check | pass | detail |
|---|---|---|
| E1_all_2930_train_plots | PASS | 2930 train plots -> 2345 inner-train / 585 inner-dev |
| E2_disjoint_and_complete | PASS | inner-train and inner-dev are disjoint and jointly cover all train plots |
| E3_no_validation_or_test_plot | PASS | official validation and every test plot are excluded |
| E4_seed_and_fraction | PASS | seed 20261025, inner-train fraction 0.8 |

**all_pass = True**
