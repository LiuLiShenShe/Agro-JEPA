# S2 — Real FIP 1.0 pair construction (consecutive + fixed-span)

Stage: **S2 (dataset adaptation, variable time intervals, weather sequences)**.
Date: 2026-10-23.
Branch: `agro-s0-upstream`, base commit `395651d` (S1: `0e8ea40`).
S2 implementation commit: `TBD` (recorded in a follow-up docs commit, as in S1).

Inputs are the **frozen S1 artifacts**; nothing here rewrites
`data/manifests/fip1_pairs_consecutive.parquet`, `manifest_spec.json` or any
SHA256 pinned there.

```bash
python3 tools/s2_build_pairs.py \
    --config agro_jepa/config/s2_dataset.yaml \
    --mode both \
    --out external_data/pairs_cache/fip1_pairs_s2.parquet \
    --report reports/s2_pairs_stats.json \
    --sample-csv reports/s2_pairs_sample.csv \
    --sample-rows 200
# -> 569,821 pairs in 15.8 s
```

Machine-readable output: `reports/s2_pairs_stats.json`.
Committed sample: `reports/s2_pairs_sample.csv` (200 random rows, seed 42).
Run log: `reports/s2_run_logs/s2_build_pairs.log`.

## 1. Inputs (real, frozen)

| Input | Role |
|---|---|
| `data/manifests/fip1_observations.csv.gz` (153,022 rows) | one row per inner-plot image, with matched CC/PH/WHC/Sen value, measurement date and signed lag |
| `data/manifests/fip1_plots.csv` (4,058 plots) | official split, genotype, harvest year |
| `data/manifests/custom_crossyear_splits.csv` | `custom-crossyear-v1`, kept in a separate column |
| `data/manifests/weather_daily_first_plot_per_year.csv` (2,191 days) | S1 daily weather, shared per harvest year |

## 2. Finding: 1,872 duplicate image references in the S1 observation manifest

The S1 observation manifest contains **3,715 rows that re-list an image already
present for the same plot** (1,843 distinct stems, 1,362 plots). They are exact
duplicates of `(plot_uid, image_filename_stem)` with an identical `obs_index`.
Their downstream effect is visible in the frozen S1 pair manifest: **1,872
pairs have `delta_days == 0`, `image_a == image_b`, and an empty `(t_a, t_b]`
weather window**, which contradicts the frozen rule "required: `t_b > t_a`"
(`manifest_spec.json::sample_definition`).

S2 de-duplicates `(plot_uid, image_filename_stem)` before pairing (keeping the
first occurrence, stable order by date/time/obs_index) and enforces
`delta_days >= 1`. This is the only intentional difference from S1's pair
counts:

| Quantity | S1 frozen | S2 deduplicated |
|---|---|---|
| observation rows | 153,022 | 151,150 |
| consecutive pairs | 148,964 | **147,092** (= 148,964 − 1,872) |
| span-7 pairs | 117,403 | **115,702** |

**Reproducibility proof of the span rule:** running the same builder with
de-duplication disabled reproduces S1's span-7 count **exactly** (117,403), so
the difference above is entirely the duplicate references, not a different
pairing rule.

## 3. Pair definition

A pair is two observations of the **same** `plot_uid` at dates `t_a < t_b`.
Image↔date correspondence always uses the audited `image_filename_stem`
(S1 §3: list position is wrong for 1,773/4,058 plots).

| Family | Rule |
|---|---|
| `consecutive` | adjacent distinct images of a plot; irregular Δ (the variable-interval signal) |
| `span7/14/21/28` | nearest later image with `abs(delta − span) <= 2 d` (S1's frozen tolerance) |

Common constraints: `1 <= delta_days <= 120`; both ends from the same plot;
the weather window is exactly `(t_a, t_b]`.

## 4. Real counts

### 4.1 By pair family

| Pair type | Pairs |
|---|---|
| `consecutive` | **147,092** |
| `span7` | 115,702 |
| `span14` | 109,653 |
| `span21` | 100,880 |
| `span28` | 96,494 |
| **total** | **569,821** |

`span` rows can share an `(i, j)` with `consecutive` (different `pair_id`,
different `pair_type`); `pair_id` is unique across the whole table (569,821
distinct).

### 4.2 By official split

| Split | Pairs |
|---|---|
| `train` | 452,855 |
| `validation` | 38,893 |
| `test_plot` | 25,203 |
| `test_genotype` | 30,631 |
| `test_environment` | 16,528 |
| `test_genotype_environment` | 5,711 |

### 4.3 By custom cross-year split (separate column)

| `custom_split` | Pairs |
|---|---|
| `train` | 278,501 |
| `val` | 43,660 |
| `id_test` | 32,816 |
| `cross_year_test` (2018 + 2022) | 214,844 |

### 4.4 By harvest year

| 2016 | 2017 | 2018 | 2019 | 2021 | 2022 |
|---|---|---|---|---|---|
| 80,797 | 50,236 | 71,177 | 22,239 | 201,705 | 143,667 |

### 4.5 Time spans

All pairs: min 1 d, median 13 d, mean 14.08 d, p90 28 d, max 93 d.
`consecutive` only: min 1 d, **median 4 d**, mean 6.20 d, p90 12 d, max 93 d —
the irregular interval distribution that S3 must model explicitly.

| Δ bin | 1–3 | 4–7 | 8–14 | 15–21 | 22–28 | 29–35 | 36–60 | >60 |
|---|---|---|---|---|---|---|---|---|
| pairs | 64,739 | 150,144 | 124,422 | 108,151 | 99,259 | 20,727 | 1,495 | 884 |

## 5. Labels: raw value, actual date, signed lag, validity mask

Validity at both ends of a pair (`value present AND |lag| <= 7 d`):

| Trait | Unit | Both valid | Train | Validation | test_plot | test_genotype | test_env | test_G&E |
|---|---|---|---|---|---|---|---|---|
| CC | fraction 0–1 | 541,770 | 433,969 | 35,697 | 23,443 | 28,671 | 15,069 | 4,921 |
| PH | m | 447,928 | 360,061 | 26,199 | 16,962 | 22,569 | 16,460 | 5,677 |
| WHC | count | 259,773 | 222,170 | 14,131 | 9,444 | 13,167 | 639 | 222 |
| Sen | 0–10 score | 23,205 | 14,950 | 3,881 | 2,389 | 1,985 | **0** | **0** |

Zero / missing / negative handling, verified on the whole table:

| Trait | Zeros at t_a | Zeros that pass the lag mask (t_a / t_b) | Missing (NaN) |
|---|---|---|---|
| CC | 10 (t_b: 6) | 10 / 6 | 0 |
| PH | 42,130 (t_b: 25,408) | 15,950 / 11,900 | 0 |
| WHC | 297,702 (t_b: 259,035) | 77,464 / 67,222 | 2,030 |
| Sen | 78,629 (t_b: 69,204) | 6,794 / 11,419 | 368,326 |

* a measured **0 stays 0 with `mask=True`** — the large WHC/PH zero counts are
  dominated by far-lag matches (early-season images matched to the first
  measurement date), which the lag mask excludes, exactly as S1 warned;
* a **missing** measurement is NaN with `mask=False` **and** `*_date is None`
  (checked in `tools/s2_verify.py`);
* **PH negatives are preserved verbatim**: 5,572 at `t_a`, 4,422 at `t_b`, plus a
  boolean `ph_negative_a/b` quality flag. No clipping, no filling;
* **Sen only exists in 2016–2018**: 0 valid Sen pairs in `test_environment`
  and `test_genotype_environment` (both are 2019-only). No Sen label is imputed.

## 6. Weather windows

| Check | Result |
|---|---|
| Pairs | 569,821 |
| `window_n_days == delta_days` | 569,821 / 569,821 |
| Windows fully covered by the daily table | **569,821 / 569,821** |
| Missing days | **0** |
| Worst coverage ratio | 1.0 |
| Store manifest id | `s1-weather-daily-v1@19aa4ee2f476` |
| Daily table SHA256 | `19aa4ee2…66a4b52e` (matches `manifest_spec.json`) |

Convention `(t_a, t_b]`: the sequence is `t_a+1 … t_b`, length `delta_days`
(verified per pair; the batch check is in §C of `reports/s2_real_batch.md`).

Reuse: the daily table is indexed by `(harvest_year, date)` and shared by every
plot of that year — S1 verified the station series is identical across the
plots checked for each year. No parquet rescanning and no per-plot copy.

Units: degC, %, MJ m⁻² d⁻¹, and **precipitation in raw sensor units**. The unit
is unresolved, so S2 computes no mm conversion, no cumulative precipitation and
no water balance. VPD is available only as an opt-in derived channel, with its
formula recorded in `reports/s2_weather_schema.json` and
`agro_jepa/datasets/fip_weather.py::WEATHER_FORMULAS`. No GDD: the base
temperature is not verified for FIP, so no GDD is computed or named as a
standard.

## 7. Split and leakage checks

| Check | Result |
|---|---|
| Official splits share a `plot_uid` | **none** |
| Pair `split` == its plot's official split | yes (all 569,821) |
| Pair ends carry their own `plot_uid` in the stem | yes |
| Image dates == filename-stem dates | yes |
| Custom split kept as a separate column | yes; `official_split` preserved verbatim |
| Custom `train` ∩ `cross_year_test` plots | **empty** |
| `cross_year_test` years | {2018, 2022} only |
| `cross_year_test` ∩ `train` genotypes | **370** (by construction; reported, not hidden) |
| `cross_year_test` ∩ `id_test` genotypes | 78 |
| Unique plots / genotypes | 4,058 / 904 |

`test_genotype` and `test_genotype_environment` remain the only genotype-unseen
official tests; `test_environment` reuses 43 training genotypes (S1). No
`plot_uid` is shared, so there is no plot-level leakage.

## 8. Normalization statistics (train split only)

Fitted on `train` pairs only (452,855 pairs), never on validation/test labels:

| Trait | n values | mean | std | min | max |
|---|---|---|---|---|---|
| CC | 882,152 | 0.3983 | 0.2573 | 0.0 | 0.9673 |
| PH | 743,211 | 0.4531 | 0.3524 | −0.0137 | 1.3398 |

Weather (daily rows of the years containing train plots — 2019 excluded):
T_mean 9.99 ± 7.60 °C; RH 83.31 ± 12.05 %; Rs 12.95 ± 8.74 MJ m⁻² d⁻¹;
P_sum_raw 2.91 ± 6.53 (raw units). Full table in
`reports/s2_pairs_stats.json::normalization`.

## 9. Artifacts

| Artifact | Committed | Notes |
|---|---|---|
| `reports/s2_pairs_stats.json` | yes | all numbers above + schema + leakage checks |
| `reports/s2_pairs_sample.csv` | yes | 200 real rows, seed 42 |
| `external_data/pairs_cache/fip1_pairs_s2.parquet` | **no** (41.6 MB) | derived; rebuild in ~16 s from committed S1 manifests |
| `agro_jepa/datasets/fip_pairs.py` | yes | builder + rules |
| `agro_jepa/datasets/fip_weather.py` | yes | daily store + `(t_a, t_b]` windows |
| `agro_jepa/datasets/fip_dataset.py` | yes | torch `Dataset` |
| `agro_jepa/datasets/collate.py` | yes | variable-length padding + masks |
| `agro_jepa/config/s2_dataset.yaml` | yes | one config for the whole path |

The pair parquet is deliberately not committed: it is a deterministic function
of the committed manifests, and committing 42 MB of derived rows would blur the
"raw data is not in Git" boundary. Its SHA256 is recorded in the stats JSON.

## 10. Open items

1. The 1,872 duplicate observation references are an S1 manifest issue. S2
   works around it; whether `fip1_observations.csv.gz` should be regenerated is
   left open (it would change a frozen artifact).
2. The weather daily table is one plot per year; S1 verified equality across 3
   plots per year only. A full equality check across all 4,058 plots has not
   been run (it would require re-reading 1.4 GB of parquet).
3. Precipitation units remain unresolved; no water-balance quantity may be
   computed until confirmed.
4. WHC and Sen remain underpowered for the 2019 environment splits (639 and 0
   valid pairs respectively).
