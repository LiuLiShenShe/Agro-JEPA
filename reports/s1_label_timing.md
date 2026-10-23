# S1 — Image date vs label measurement date (timing audit)

Source: `data/manifests/fip1_observations.csv.gz` (153,022 inner-plot images,
revision `11cdd84b…`). S1 commit: `0e8ea4093362a30923b89a419c362d1bc71300ff`.
Signed lag is defined as

```
lag_days = label_measurement_date − image_date
```

so `lag = 0` means same-day, `lag > 0` means the label was measured after the
image, `lag < 0` before it. The matching rule is "nearest measurement date for
that plot", implemented in `tools/s1_audit_fip.py`; the actual matched date is
stored per image (`cc_date`, `ph_date`, `whc_date`, `sen_date`).

Reproduce:

```bash
python3 tools/s1_audit_fip.py --parquet-dir external_data/fip1_parquet \
    --out-dir data/manifests --report reports/s1_fip_audit.json
python3 tools/s1_dataset_stats.py --manifests data/manifests \
    --report reports/s1_split_statistics.json
```

## 1. Headline

**Images and phenotypes are not, in general, measured on the same day.** Only
canopy cover is nearly same-day; plant height, head count and senescence are
not, and treating an image date as the label date would be wrong for most
non-CC targets.

| Trait | Images with a measurement in the plot | Same day | \|lag\| ≤ 3 d | \|lag\| ≤ 7 d | Median lag | Mean signed lag |
|---|---|---|---|---|---|---|
| CC | 153,022 / 153,022 (100 %) | 89.5 % | 92.8 % | 96.3 % | 0 d | −0.54 d |
| PH | 153,022 / 153,022 (100 %) | 49.1 % | 70.8 % | 75.8 % | 0 d | +16.64 d |
| WHC | 152,454 / 153,022 (99.6 %) | 42.6 % | 44.2 % | 46.9 % | +13 d | +39.37 d |
| Sen | 61,135 / 153,022 (39.9 %) | 8.9 % | 17.3 % | 17.9 % | +70 d | +80.79 d |

The CC/PH percentages are shares of the *matched* images. For Sen, 17.9 % of
matches correspond to 7.2 % of **all** images (10,960 / 153,022): Sen was only
measured in 2016–2018, near the end of the season, so most images in those years
have no nearby Sen date.

## 2. Signed lag distribution

Counts per lag bin (all images with a match):

| Lag bin | CC | PH | WHC | Sen |
|---|---|---|---|---|
| < −28 d | 1,240 | 0 | 874 | 0 |
| −28 … −9 d | 3,308 | 631 | 1,873 | 37 |
| −8 … −4 d | 3,147 | 4,081 | 1,066 | 18 |
| −3 … −1 d | 2,023 | 13,289 | 636 | 2,615 |
| **0 d** | **137,026** | **75,200** | **64,979** | **5,422** |
| +1 … +3 d | 2,920 | 19,867 | 1,802 | 2,564 |
| +4 … +7 d | 2,366 | 3,590 | 3,084 | 342 |
| +8 … +28 d | 962 | 8,876 | 14,678 | 8,852 |
| ≥ +29 d | 30 | 27,488 | 63,462 | 41,285 |

Reading this table:

* **CC** is almost always computed on the image itself (89.5 % same day); the
  negative tail is small (image after the CC date, e.g. rescheduled acquisition).
* **PH** is measured on fewer dates; images before the first PH date or after
  the last are matched to the nearest end date, producing the +29 d tail
  (27,488 images). Within the PH campaign the same-day share is high.
* **WHC** is measured mostly May–July; images outside that window match the
  first/last WHC date with a large positive lag (63,462 images ≥ +29 d).
* **Sen** starts ~20 days after flowering and only exists in 2016–2018; the
  nearest-date rule produces 41,285 matches ≥ +29 d. Sen should not be used for
  images outside its measurement window without a stricter lag mask.

## 3. Value statistics with a timing mask

Values (not counts) within |lag| ≤ 7 d:

| Trait | Valid images | Zero values | Min | Median | Max |
|---|---|---|---|---|---|
| CC | 147,403 | 2 | 0.0000 | 0.3264 | 0.9673 |
| PH | 115,926 | 3,378 | −0.0962 m | 0.4861 m | 1.6074 m |
| WHC | 71,480 | 17,304 | 0 | 188.25 | 845.5 |
| Sen | 10,960 | 2,211 | 0 | 3.0 | 10 |

Without the mask, WHC shows 77,234 zeros out of 152,454 matches — the extra
zeros are mostly far-lag early-season images matched to the first measurement,
not real zero head counts. **Missing is not zero**: a plot with no measurement
date has no value at all; a measured 0 stays 0.

PH has 10,524 zeros overall (3,378 within 7 d) and a negative minimum
(−0.0962 m). The negatives are extraction noise; they are preserved in the raw
manifest and must be handled explicitly (clip or mask) by the model-stage loader,
not silently dropped.

## 4. Per-year availability (plots with ≥ 1 measurement)

| Year | Plots | CC | PH | WHC | Sen | Inner images |
|---|---|---|---|---|---|---|
| 2016 | 710 | 710 | 710 | 710 | 703 | 25,566 |
| 2017 | 756 | 756 | 756 | 756 | 756 | 14,534 |
| 2018 | 756 | 756 | 756 | 755 | 756 | 21,266 |
| 2019 | 252 | 252 | 252 | 228 | **0** | 6,445 |
| 2021 | 792 | 792 | 792 | 792 | **0** | 48,285 |
| 2022 | 792 | 792 | 792 | 792 | **0** | 36,926 |

This matches the paper's Table 4/5. Sen is unavailable in 2019/2021/2022; any
Sen result is restricted to 2016–2018.

## 5. Rules frozen for downstream use

1. For each image keep the matched label value, the **actual label date**, and
   the **signed lag**; never assume `label_date = image_date`.
2. Primary validity mask: **|lag| ≤ 7 d**; strict variant: **|lag| ≤ 3 d**.
   Report the effective sample size with the chosen threshold.
3. Use **CC + PH** for the first modelling round (both are available for every
   plot; CC is nearly same-day). WHC and Sen are optional targets.
4. A measured 0 is data; an absent measurement date is missing. Do not impute a
   missing trait as 0.
5. The same mask must be applied at training and evaluation time; the pair
   manifests store `*_a_lag`, `*_b_lag`, `*_a_date`, `*_b_date` for both ends.
