# S4B.2 E1 Validation Diagnostics

- Generated: 2026-10-28T03:46:00+0800
- Git HEAD: `fd2a993236ce776fc0f2f67afeddc3e2a4448884`
- Validation pairs: **15,175** over **380** plots (10,858 images)
- Years: [2016, 2017, 2018, 2021, 2022]
- S3 index sha256: `f6ef1871655277549aba2d883b6f76a5831b7976af90fc74d80206591d884933`
- Bootstrap: plot-clustered, 2000 resamples, seed 20261024

## Frozen checkpoints

| condition | seed | final.pt sha256 (16) | params | input dim |
|---|---:|---|---:|---:|
| b2_no_time | 0 | `08c9dafc450a70a2` | 45,898,196 | 928 |
| b2_no_time | 1 | `7b1b3827c5e6a6d8` | 45,898,196 | 928 |
| b2_no_time | 2 | `540bbb96dbb350ff` | 45,898,196 | 928 |
| b2_no_time_constant_delta | 0 | `d8cebf42d2f97ebe` | 53,882,396 | 1088 |
| b2_no_time_constant_delta | 1 | `61a0d962309d3361` | 53,882,396 | 1088 |
| b2_no_time_constant_delta | 2 | `9391faa54f26c676` | 53,882,396 | 1088 |
| b3_delta | 0 | `f433164646886c69` | 53,882,396 | 1088 |
| b3_delta | 1 | `64ee70f8e4574ba7` | 53,882,396 | 1088 |
| b3_delta | 2 | `37ca926d05f7ac95` | 53,882,396 | 1088 |

## Seed-averaged metrics on validation

| arm | MSE | cos | rel.vs B0(persist) | rel.vs B1(train-mean) |
|---|---:|---:|---:|---:|
| B0 persistence | 2.195148 | 0.6279 | 0 | -- |
| B1 train-mean | 1.839239 | -- | -- | 0 |
| b2_no_time | 1.129680 | 0.7842 | 48.54% | 38.58% |
| b2_no_time_constant_delta | 1.129654 | 0.7842 | 48.54% | 38.58% |
| b3_delta | 1.075503 | 0.7957 | 51.01% | 41.52% |

## Bootstrap 95% CI (plot-clustered)

| arm | rel.vs persist [95% CI] | rel.vs train-mean [95% CI] |
|---|---|---|
| b2_no_time | 48.54% [48.30, 48.79] | 38.59% [38.29, 38.88] |
| b2_no_time_constant_delta | 48.54% [48.29, 48.79] | 38.59% [38.29, 38.88] |
| b3_delta | 51.01% [50.77, 51.23] | 41.53% [41.25, 41.80] |

## Paired B3 - B2-C contrast

- seed-averaged: dMSE = -0.054124 [-0.055900, -0.052246] (negative = B3 better)
- per training seed (training randomness, *not* a plot bootstrap):
  - seed0: dMSE = -0.048660
  - seed1: dMSE = -0.048494
  - seed2: dMSE = -0.049519
  - across seeds: -0.048891 +/- 0.000550 (all same sign = True)

## Stratified paired contrasts (dMSE B3 - B2-C)

| stratum | value | n pairs | dMSE | 95% CI |
|---|---|---:|---:|---|
| by_span | span14 | 7358 | -0.055809 | [-0.058216, -0.053388] |
| by_span | span7 | 7817 | -0.052539 | [-0.054536, -0.050544] |
| by_year | 2016 | 5201 | -0.042326 | [-0.044351, -0.040142] |
| by_year | 2017 | 3485 | -0.076163 | [-0.079372, -0.073081] |
| by_year | 2018 | 3831 | -0.045790 | [-0.047974, -0.043438] |
| by_year | 2021 | 1131 | -0.066698 | [-0.070578, -0.062152] |
| by_year | 2022 | 1527 | -0.055506 | [-0.059979, -0.051018] |

## Notes

- B0 = persistence (z_a), B1 = mean z_b over the full S4B.2 official-train index.
- Seed-averaged = prediction averaged over seeds {0,1,2} before MSE.
- The plot bootstrap captures validation-sample sampling only; it is not a training-seed interval (per-seed spread is reported separately).
- No test row, no test label and no test metric is used anywhere.
