# S4B.4A Five-Fold Year-Held-Out Protocol (frozen before evaluation)

- Generated: 2026-10-28T17:40:48+0800
- Config: `agro_jepa/config/s4b4a_year_heldout.yaml` (sha `f46f03e8790a46ab`)
- Data: **179,823 pairs / 2,930 plots / years [2016, 2017, 2018, 2021, 2022]** (official-train only)
- Official validation read: False; official test read: False

## Fold rule

- **heldout**: one whole harvest year (all its plots)
- **inner_split**: 80/20 by plot, stratified by harvest_year
- **inner_split_seed**: 20261025
- **inner_train_fraction**: 0.8
- **strata**: harvest_year
- **disjoint_rule**: same plot never spans train/dev/heldout
- **norm_stats**: DeltaNormStats and GDD/weather VectorNormStats are fitted on each fold's fold_train rows only; dev/heldout are transformed with those training statistics
- **selection**: epoch = minimum fold-dev future-latent MSE; heldout year is evaluated once and never used for selection

## Folds

| fold (held-out year) | role | pairs | plots | years | span7 | span14 | delta mean [min,max] |
|---|---|---:|---:|---|---:|---:|---|
| 2016 | fold_train | 128,147 | 1983 | [2017, 2018, 2021, 2022] | 65,292 | 62,855 | 10.23 [5,16] |
| 2016 | fold_dev | 32,139 | 495 | [2017, 2018, 2021, 2022] | 16,388 | 15,751 | 10.22 [5,16] |
| 2016 | heldout | 19,537 | 452 | [2016] | 10,456 | 9,081 | 10.46 [5,16] |
| 2017 | fold_train | 133,863 | 1951 | [2016, 2018, 2021, 2022] | 68,381 | 65,482 | 10.27 [5,16] |
| 2017 | fold_dev | 33,431 | 487 | [2016, 2018, 2021, 2022] | 17,084 | 16,347 | 10.27 [5,16] |
| 2017 | heldout | 12,529 | 492 | [2017] | 6,671 | 5,858 | 10.03 [5,16] |
| 2018 | fold_train | 128,205 | 1929 | [2016, 2017, 2021, 2022] | 66,212 | 61,993 | 10.27 [5,16] |
| 2018 | fold_dev | 32,103 | 481 | [2016, 2017, 2021, 2022] | 16,576 | 15,527 | 10.26 [5,16] |
| 2018 | heldout | 19,515 | 520 | [2018] | 9,348 | 10,167 | 10.16 [5,16] |
| 2021 | fold_train | 82,796 | 1751 | [2016, 2017, 2018, 2022] | 42,493 | 40,303 | 10.26 [5,16] |
| 2021 | fold_dev | 20,781 | 437 | [2016, 2017, 2018, 2022] | 10,654 | 10,127 | 10.26 [5,16] |
| 2021 | heldout | 76,246 | 742 | [2021] | 38,989 | 37,257 | 10.24 [5,16] |
| 2022 | fold_train | 102,284 | 1766 | [2016, 2017, 2018, 2021] | 52,380 | 49,904 | 10.24 [5,16] |
| 2022 | fold_dev | 25,543 | 440 | [2016, 2017, 2018, 2021] | 13,084 | 12,459 | 10.24 [5,16] |
| 2022 | heldout | 51,996 | 724 | [2022] | 26,672 | 25,324 | 10.29 [5,16] |

## Conditions

| condition | delta | gdd | weather | protocol |
|---|---|---|---|---|
| b2_no_time | none | none | none | none |
| b2_no_time_constant_delta | const 9.0 d | none | none | none |
| b3_delta | sample | none | none | none |
| gdd_p | sample | past | none | P |
| weather_p | sample | none | past | P |

