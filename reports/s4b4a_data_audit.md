# S4B.4A Data Audit

- Generated: 2026-10-28T17:38:20+0800
- Config: `agro_jepa/config/s4b4a_year_heldout.yaml` (sha `f46f03e8790a46ab`)

## 3.1 Training data and year distribution

Full official-train index: **179,823 pairs / 2,930 plots / years [2016, 2017, 2018, 2021, 2022]**

| year | pairs | plots | images | span7 | span14 | delta mean ± std [min,max] | median |
|---|---:|---:|---:|---:|---:|---|---:|
| 2016 | 19,537 | 452 | 15,585 | 10,456 | 9,081 | 10.46 ± 3.80 [5,16] | 9 |
| 2017 | 12,529 | 492 | 8,448 | 6,671 | 5,858 | 10.03 ± 3.63 [5,16] | 9 |
| 2018 | 19,515 | 520 | 14,007 | 9,348 | 10,167 | 10.16 ± 3.66 [5,16] | 12 |
| 2021 | 76,246 | 742 | 43,338 | 38,989 | 37,257 | 10.24 ± 3.61 [5,16] | 8 |
| 2022 | 51,996 | 724 | 33,413 | 26,672 | 25,324 | 10.29 ± 3.70 [5,16] | 9 |

Overall delta_days: n=179,823 mean=10.2539 std=3.6668 min=5 max=16 median=9

Quantiles: p0=5, p1=5, p5=5, p25=7, p50=9, p75=14, p95=15, p99=16, p100=16

### Official validation / test year distribution

| split | pairs | plots | years |
|---|---:|---:|---|
| validation | 15,175 | 380 | [2016, 2017, 2018, 2021, 2022] |
| test_plot | 9,851 | 250 | [2016, 2017, 2018, 2021, 2022] |
| test_genotype | 12,059 | 246 | [2016, 2017, 2018, 2021, 2022] |
| test_environment | 6,300 | 190 | [2019] |
| test_genotype_environment | 2,147 | 62 | [2019] |

> 2019 is the official unseen-environment test year; it is **not** in the official training set. S4B.4A is a *training-set-internal* year-held-out diagnostic and does not constitute an independent 2019 generalisation test.

## 3.2 Time encoding

- `tools/train_s4.py:213 (S4A/S4A.1)`: fitted on inner-train rows of this model's split only
- `agro_jepa/train/train_s4b2_full.py:227`: fitted on the ENTIRE official-train index (all 5 years) -- the frozen S4B.2 baseline

S4B.2 full-train DeltaNormStats: mean=10.253922 std=3.666825 min=5 max=16 n=179,823

All frozen checkpoints share the full-train norm: **True**

Per-checkpoint delta norm:

| condition | seed | mean | std | epochs | final sha (16) |
|---|---:|---:|---:|---:|---|
| b2_no_time | seed0 | 10.253922 | 3.666825 | 14 | `08c9dafc450a70a2` |
| b2_no_time | seed1 | 10.253922 | 3.666825 | 14 | `7b1b3827c5e6a6d8` |
| b2_no_time | seed2 | 10.253922 | 3.666825 | 14 | `540bbb96dbb350ff` |
| b2_no_time_constant_delta | seed0 | 10.253922 | 3.666825 | 13 | `d8cebf42d2f97ebe` |
| b2_no_time_constant_delta | seed1 | 10.253922 | 3.666825 | 13 | `61a0d962309d3361` |
| b2_no_time_constant_delta | seed2 | 10.253922 | 3.666825 | 13 | `9391faa54f26c676` |
| b3_delta | seed0 | 10.253922 | 3.666825 | 12 | `f433164646886c69` |
| b3_delta | seed1 | 10.253922 | 3.666825 | 12 | `64ee70f8e4574ba7` |
| b3_delta | seed2 | 10.253922 | 3.666825 | 12 | `37ca926d05f7ac95` |

**S4B.4A requirement:** each fold must refit DeltaNormStats (and GDD/weather VectorNormStats) on that fold's *training plot* rows only

## 3.3 GDD and weather

- GDD: GDD_d = max(T_mean_d - T_base, 0); T_base in {0, 5} degC (unit: degC day)
- Protocol R: `(t_a, t_b]` — deployable = False (retrospective/oracle)
- Protocol P: `[t_a-W, ..., t_a-1] (strictly before t_a)` — deployable = True
- interval windows fully valid: True; past windows fully valid: True
- conditioning rows: 179,823; splits: {'train': 179823}

GDD past-28 base-0 by year (a year-shift check):

| year | n | mean | std | min | max |
|---|---:|---:|---:|---:|---:|
| 2016 | 19,537 | 216.96 | 125.25 | 56.42 | 508.16 |
| 2017 | 12,529 | 250.25 | 138.19 | 64.09 | 572.56 |
| 2018 | 19,515 | 329.53 | 153.75 | 61.08 | 517.41 |
| 2021 | 76,246 | 281.12 | 140.46 | 55.65 | 528.43 |
| 2022 | 51,996 | 287.93 | 162.92 | 24.90 | 561.01 |

- Weather channels: ['T_mean_C', 'T_max_C', 'T_min_C', 'RH_mean_pct', 'Rs_sum_MJ_m2_d']
- Excluded: {'P_sum_raw': 'unit unresolved (S1); precipitation integrals forbidden by S2', 'VPD_mean_kPa': 'derived channel, opt-in only; not used'}
- Protocol P window: 28 d, strictly before t_a: True
- Structural limit: all plots of one harvest_year share a single weather series; weather carries date/year-level environment variation only

**Conditioning coverage:** 179,823 pair ids over 1,000 plots; full train has 179,823 ids over 2,930 plots. Covers full train: **True**. the frozen S4A conditioning npz covers only the S3 pilot index (1,000 official-train plots); GDD-P / Weather-P on the full 2,930-plot index would require a new conditioning build

## 3.4 Model capacity

| condition | parts | trainable | condition params | input dim | delta | gdd | weather | fusion |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| b2_no_time | - | 45,898,196 | 0 | 928 | 0 | 0 | 0 | 0 |
| b2_no_time_constant_delta | delta | 53,882,396 | 72,730 | 1088 | 72,730 | 0 | 0 | 0 |
| b3_delta | delta | 53,882,396 | 72,730 | 1088 | 72,730 | 0 | 0 | 0 |
| gdd_p | delta,gdd | 53,908,480 | 98,814 | 1088 | 72,730 | 24,090 | 0 | 1,994 |
| weather_p | delta,weather | 53,906,928 | 97,262 | 1088 | 72,730 | 0 | 22,538 | 1,994 |

- **b2_no_time_constant_delta_vs_b3_delta**: identical trainable/condition parameter counts and input dim (the S4A.1 matched control); difference is sample-level delta only
- **gdd_and_weather_arms**: add encoder + fusion parameters over B3; capacity is NOT matched to B3, so part of any gain may be parameter-driven

