# S1 — Real FIP 1.0 data audit

Stage: **S1 (data access, audit, manifest freeze, frozen-DINOv2 cache validation)**.
Date: 2026-10-23.
Repository: `https://github.com/LiuLiShenShe/Agro-JEPA`, branch `agro-s0-upstream`.
Base commit before this stage: `91c4671`.
Dataset: `mikeboss/FIP1` on HuggingFace, revision **`11cdd84bc0ba8d5c6641ea7b53550b1658527e1c`**,
license **CC-BY-4.0** (verified through the dataset card and Crossref for the paper:
`https://creativecommons.org/licenses/by/4.0/`).

## 0. What was executed on this host

The machine cannot reach `huggingface.co` (TCP connections stall; a 20 s `curl -I`
timed out), while `hf-mirror.com` and `github.com` are reachable. All FIP downloads
therefore used `HF_ENDPOINT=https://hf-mirror.com`; GitHub and
`dl.fbaipublicfiles.com` were used directly for DINOv2 code/weights.

```bash
# 1. tabular shards (13 files, 1,375,668,510 bytes), SHA256 pinned in the script
python3 tools/s1_download_fip.py --out external_data/fip1_parquet
#    -> reports/s1_run_logs/fip1_download.log, external_data/fip1_parquet/download_manifest.json

# 2. full audit: plots + one row per inner-plot image with matched labels
python3 tools/s1_audit_fip.py \
    --parquet-dir external_data/fip1_parquet \
    --out-dir data/manifests \
    --report reports/s1_fip_audit.json

# 3. split / pair / label-timing statistics
python3 tools/s1_dataset_stats.py \
    --manifests data/manifests --report reports/s1_split_statistics.json

# 4. project-defined cross-year split (does not touch official splits)
python3 tools/s1_build_custom_split.py \
    --plots data/manifests/fip1_plots.csv \
    --out data/manifests/custom_crossyear_splits.csv \
    --report reports/s1_custom_split_statistics.json

# 5. weather schema/completeness + daily sample
python3 tools/s1_weather_audit.py \
    --parquet-dir external_data/fip1_parquet \
    --report reports/s1_weather_schema.json \
    --daily-out data/manifests/weather_daily_first_plot_per_year.csv

# 6. acceptance lookup (image -> dates -> labels -> weather window)
python3 tools/s1_lookup_plot.py --plot FPWW0120667 --date 2016-05-17 --days 7
python3 tools/s1_lookup_plot.py --plot FPWW0240010 --date 2019-03-13 --days 7

# 7. figures
python3 tools/s1_make_figures.py --manifests data/manifests --fig-dir reports/figures
```

The raw parquet and the downloaded images live under `external_data/`, which is
git-ignored. Manifests and reports are committed.

## 1. Downloaded inventory

| Artifact | Value |
|---|---|
| Parquet shards | 13 (`train` ×7, `validation` ×2, 1 each for the four test splits) |
| Bytes on disk | 1,375,668,510 |
| SHA256 verified | all 13 match the LFS `oid` from the HuggingFace tree API |
| Rows (plots) | **4,058** |
| Unique `plot_uid` | **4,058** (one row per plot-year, no duplicates) |
| Nested columns | 131 total |
| Inner-plot images referenced | **153,022** |
| Full images referenced | 160,772 (inner crops are a subset; see §5) |

Per-shard hashes are in `reports/s1_fip_audit.json::shards` and
`external_data/fip1_parquet/download_manifest.json`.

## 2. Real schema (not assumed)

Every field used by the audit was checked against the arrow schema of the real
shards. The top-level keys are:

```
plot_uid yearsite_uid crop_type experiment_number plot_number range row lot
latitude longitude spatial_check sowing_date harvest_date harvest_year
images image_dates image_times
alignment_* (soil polygons, steps, dates, times, initial date, transform)
inner_plot_images image_inner_plot_transforms
genotype_id marker_biallelic_codes marker_metadata_strings
canopy_cover_{values,dates,trait_id,trait_name,method_id,method_name,si_unit,responsible}
height_* spike_count_* senescence_*          (same 8-field block, list-valued)
heading_* height_final_* yield_* yield_adjusted_* protein_*
   (same block, scalar value + scalar date)
temperature_air_200cm_* humidity_air_200cm_* irradiance_solar_200cm_*
precipitation_200cm_* temperature_air_10cm_* humidity_air_10cm_*
temperature_soil_5cm_*                        (values, dates, times)
```

Confirmed facts:

* one row = one `plot_uid` = one plot in one harvest year;
* `harvest_year` ∈ {2016, 2017, 2018, 2019, 2021, 2022}; 2015 and 2020 are absent
  (incomplete seasons, as the paper states);
* `crop_type` is `WW` for every row;
* `genotype_id` is an anonymised 64-hex string;
* `image_dates` / `image_times` are parallel to `images` (the full-frame paths).

## 3. Critical finding — the inner-image list is not positionally aligned

`len(inner_plot_images)` is **not** always equal to `len(image_dates)`:

| Quantity | Value |
|---|---|
| plots whose inner list length differs from the full list | **1,773 / 4,058** |
| full images without an inner crop | **7,741** |
| inner images without a matching full image | **0** |

A positional `zip(inner_plot_images, image_dates)` therefore produces wrong dates
(the first audit pass showed 21,966/153,022 filename-date mismatches; e.g. a 2019
plot has 30 inner images and 32 image dates). The audit was rewritten to join by
the **filename stem** (`FPWW0240001_RGB1_20190618_140717`) embedded in both
columns. After the fix: 0 mismatches, 0 orphans.

**Protocol consequence:** never index `inner_plot_images` by position. Always join
on the filename stem (the timestamp in the stem is the authoritative image time).

## 4. Trait inventory, units, and missingness

Units and method names come from the per-trait metadata columns in the real data.
Raw value ranges come from `reports/s1_fip_audit.json`.

| Trait | Data columns | Unit in metadata | Observed range | Plots with ≥1 measurement | Total measurements |
|---|---|---|---|---|---|
| Canopy cover (CC) | `canopy_cover_values/dates` | `%` (metadata) — **values are 0–1 fractions** | 0.0000–0.9673 | 4,058 / 4,058 | 138,699 |
| Plant height (PH) | `height_values/dates` | m | −0.0962–1.6074 | 4,058 / 4,058 | 143,301 |
| Wheat head count (WHC) | `spike_count_values/dates` | count (`1`) | 0–845.5 | 4,033 / 4,058 | 66,829 |
| Senescence (Sen) | `senescence_values/dates` | 0–10 score (`1`) | 0–10 | 2,215 / 4,058 (2016–2018 only) | 22,688 |
| Heading date | `heading_value/date` | date | — | 2,805 / 4,058 | 2,805 |
| Final height | `height_final_value/date` | m | — | 4,058 / 4,058 | 4,058 |
| Grain yield | `yield_value/date` | t ha⁻¹ | — | 3,873 / 4,058 | 3,873 |
| Yield (adjusted) | `yield_adjusted_value/date` | t ha⁻¹ | — | 3,873 / 4,058 | 3,873 |
| Protein content | `protein_value/date` | % | — | 3,076 / 4,058 | 3,076 |

These totals match the paper (Table 4/5): 4,058 plots; CC/PH measured on all;
WHC 4,057 plots; Sen on 2,215 plots (2016–2018); yield 3,873; protein 3,076;
heading 2,805; final height 4,058.

**Zero vs missing.** Zero is a real observation and is kept:

* CC has 2 measured zeros (all other values > 0);
* PH has 10,524 observed zeros (early season); negative values (min −0.096 m)
  are extraction noise and are **not** clipped in the raw manifest;
* WHC has 77,234 zeros among all matched images, but only **17,304 zeros within
  |lag| ≤ 7 d** — the large count is dominated by images outside the May–July
  measurement window being matched to a far-away date;
* Sen has 21,335 zeros overall, 2,211 within |lag| ≤ 7 d.

No ``None`` was converted to 0 anywhere; missing sequences are simply absent in
the parquet and become empty cells in the manifest.

## 5. Image inventory

| Quantity | Value |
|---|---|
| Full-frame image paths | 160,772 |
| Inner-plot image paths (aligned cut-outs) | 153,022 |
| Full images without an inner crop | 7,741 |
| Inner crops without a full-frame path | 0 |
| Plots with a length mismatch between the two lists | 1,773 |
| Filename-vs-date mismatches after stem join | 0 |
| Filename-vs-`plot_uid` mismatches | 0 |

All images are PNG, 960×640 RGB in the 97 downloaded samples. The HuggingFace
repository contains only the aligned inner-plot images; the full-frame paths in
`images` point to `/data/images/...` and require the separate ETH collection
(not downloaded; not needed for V0).

**Example (real plot `FPWW0120667`, 2016):** 39 full images, 39 inner crops,
sowing 2015-10-13, harvest 2016-07-27, 39 CC measurements and 22 PH measurements.
The image at `2016-05-17` has CC = 0.5942 and PH = 0.7061 m measured on the same
date. Full record: `reports/figures/s1_plot_timeline_FPWW0120667.png`.

**Image-quality note.** In the 97 downloaded images, 1 (FPWW0120667,
2016-02-16, mean pixel value 225.9/255) is almost white — snow cover or an
over-exposed/failed crop. A brightness/contrast filter is needed before caching;
it is not applied in S1.

## 6. Weather inventory

Raw records are **hourly**, not daily. Every plot-year row carries the same
station series for its year, and the arrays are identical across the 3 checked
plots per year.

| Year | First day | Last day | Days | Records | Records/day | NaNs |
|---|---|---|---|---|---|---|
| 2016 | 2015-09-01 | 2016-08-31 | 366 | 8,784 | 24 | 0 |
| 2017 | 2016-09-01 | 2017-08-31 | 365 | 8,760 | 24 | 0 |
| 2018 | 2017-09-01 | 2018-08-31 | 365 | 8,760 | 24 | 0 |
| 2019 | 2018-09-01 | 2019-08-31 | 365 | 8,760 | 24 | 0 |
| 2021 | 2020-09-01 | 2021-08-31 | 365 | 8,760 | 24 | 0 |
| 2022 | 2021-09-01 | 2022-08-31 | 365 | 8,760 | 24 | 0 |

Sensors present: air temperature 2 m and 0.1 m, soil temperature 0.05 m,
relative humidity 2 m and 0.1 m, solar irradiance 2 m, precipitation.
Observed ranges (2 m sensors, all years): air temperature −13.8…34.9 °C;
humidity 22.8…100 %; irradiance 0…1044 W m⁻²; precipitation 0…22 (raw units).
Daily aggregation formulas are recorded in `reports/s1_weather_schema.json` and
the sample table `data/manifests/weather_daily_first_plot_per_year.csv`.

**Open issue — precipitation unit.** The data card does not state a unit. The
hourly mean is 0.09–0.17 raw units, which would be ~800–1,500 mm per water year
if the unit were mm. This is high for the Zurich site, so the unit must be
confirmed against the FIP/Agrometeo documentation before any water-balance claim.
No unit is asserted in the manifest.

## 7. Sample realizability

The 153,022 inner images yield **148,964 consecutive same-plot pairs** (Δt 0–93 d;
median 4 d). Candidate counts at fixed spans, selecting for each anchor the
nearest later image with |Δt − span| ≤ 2 d:

| Span | Pairs | CC valid at both ends (|lag|≤7 d) | PH valid at both ends |
|---|---|---|---|
| 7 d | 117,403 | 112,882 | 95,647 |
| 14 d | 110,892 | 105,373 | 88,778 |
| 21 d | 102,183 | 97,238 | 81,377 |
| 28 d | 97,713 | 91,637 | 77,592 |
| 35 d | 88,869 | 83,563 | 71,538 |
| 42 d | 84,115 | 80,139 | 66,126 |

Counts by official split are in `reports/s1_split_statistics.json::span_candidates`.
The CC+PH pair requirement stated in the handbook is therefore comfortably
satisfied at every span; WHC/Sen are much sparser (Sen is 2016–2018 only).

## 8. Deliverables produced by this audit

* `data/manifests/fip1_plots.csv` — 4,058 rows, one per plot-year (official split,
  metadata, counts, scalar traits).
* `data/manifests/fip1_observations.csv.gz` — 153,022 rows, one per inner image
  (image path + date/time, matched CC/PH/WHC/Sen value, signed lag, label date).
  The uncompressed CSV is kept locally but git-ignored.
* `data/manifests/fip1_pairs_consecutive.parquet` — 148,964 same-plot consecutive
  pairs with both image paths, dates, labels, signed lags, and the weather window
  `(t_a, t_b]`.
* `data/manifests/custom_crossyear_splits.csv` — project-defined split.
* `data/manifests/weather_daily_first_plot_per_year.csv` — daily aggregates for
  one plot per year (365/366 days × 6 years).
* `data/manifests/manifest_spec.json` — frozen rules + SHA256 of each artifact.
* `reports/s1_fip_audit.json`, `reports/s1_split_statistics.json`,
  `reports/s1_weather_schema.json`, `reports/s1_custom_split_statistics.json`.
* Figures: `reports/figures/s1_plot_timeline_FPWW0120667.png`,
  `s1_label_coverage_by_year.png`, `s1_weather_distributions.png`,
  `s1_dt_distribution.png`.

## 9. Unresolved issues

1. **Precipitation unit** is not documented in the data card (§6).
2. **CC unit mismatch**: metadata says `%`, values are 0–1 fractions. Downstream
   code must use fractions; reports must not silently multiply by 100.
3. **Inner-image date alignment** must always use the filename stem, never list
   position (§3).
4. **A small fraction of inner crops are unusable** (near-white/over-exposed);
   no quality filter is applied yet.
5. The **full-frame images** (160,772 paths) are not in the HF repo and were not
   downloaded; the project does not need them for V0.
6. **2019 is not in `train`/`validation`** at all; its 252 plots are split between
   `test_environment` (190) and `test_genotype_environment` (62). A model that
   only trains on `train` has no 2019 data.
