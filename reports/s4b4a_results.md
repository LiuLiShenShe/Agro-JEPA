# S4B.4A Results -- Cross-Year Diagnostic (training-set internal)

- Folds completed: [2016, 2017, 2018, 2021, 2022]
- Seeds run: [0] (pilot); planned grid: [0, 1, 2]
- Scope: official-train years only; this is **not** an independent 2019 unseen-environment confirmation.

## E1 -- future-latent prediction, per fold

| fold | arm | MSE | cos | vs persistence | vs train-mean |
|---|---|---:|---:|---:|---:|
| 2016 | b2_no_time | 1.292215 | 0.7557 | 36.41% | 32.48% |
| 2016 | b2_no_time_constant_delta | 1.299090 | 0.7546 | 36.07% | 32.12% |
| 2016 | b3_delta | 1.308925 | 0.7533 | 35.58% | 31.61% |
| 2016 | gdd_p | 1.337182 | 0.7479 | 34.19% | 30.13% |
| 2016 | weather_p | 1.322324 | 0.7508 | 34.93% | 30.91% |
| 2017 | b2_no_time | 1.397949 | 0.7303 | 39.35% | 22.86% |
| 2017 | b2_no_time_constant_delta | 1.397095 | 0.7305 | 39.38% | 22.90% |
| 2017 | b3_delta | 1.471804 | 0.7175 | 36.14% | 18.78% |
| 2017 | gdd_p | 1.485419 | 0.7144 | 35.55% | 18.03% |
| 2017 | weather_p | 1.451169 | 0.7218 | 37.04% | 19.92% |
| 2018 | b2_no_time | 1.365389 | 0.7263 | 39.79% | 25.34% |
| 2018 | b2_no_time_constant_delta | 1.364792 | 0.7265 | 39.82% | 25.37% |
| 2018 | b3_delta | 1.390656 | 0.7221 | 38.68% | 23.96% |
| 2018 | gdd_p | 1.416940 | 0.7165 | 37.52% | 22.52% |
| 2018 | weather_p | 1.412738 | 0.7171 | 37.71% | 22.75% |
| 2021 | b2_no_time | 1.368474 | 0.7312 | 39.79% | 26.09% |
| 2021 | b2_no_time_constant_delta | 1.372032 | 0.7307 | 39.64% | 25.90% |
| 2021 | b3_delta | 1.394352 | 0.7271 | 38.65% | 24.70% |
| 2021 | gdd_p | 1.401756 | 0.7258 | 38.33% | 24.30% |
| 2021 | weather_p | 1.396832 | 0.7267 | 38.54% | 24.56% |
| 2022 | b2_no_time | 1.306910 | 0.7478 | 39.88% | 28.16% |
| 2022 | b2_no_time_constant_delta | 1.308234 | 0.7476 | 39.81% | 28.09% |
| 2022 | b3_delta | 1.327538 | 0.7445 | 38.93% | 27.02% |
| 2022 | gdd_p | 1.321425 | 0.7454 | 39.21% | 27.36% |
| 2022 | weather_p | 1.319720 | 0.7460 | 39.29% | 27.45% |

## E1 -- paired contrasts across folds

### b3_delta_minus_b2_no_time_constant_delta (negative = second arm better)

| fold | dMSE | 95% CI | CI excludes 0 | per-seed mean ± sd |
|---|---:|---|---|---|
| 2016 | +0.009813 | [+0.007451, +0.012173] | True | +0.009835 ± 0.000000 |
| 2017 | +0.074714 | [+0.071814, +0.077734] | True | +0.074709 ± 0.000000 |
| 2018 | +0.025906 | [+0.023644, +0.028293] | True | +0.025864 ± 0.000000 |
| 2021 | +0.022292 | [+0.021301, +0.023372] | True | +0.022321 ± 0.000000 |
| 2022 | +0.019276 | [+0.017926, +0.020631] | True | +0.019304 ± 0.000000 |

- folds with CI entirely below 0 (second arm better): 0/5; all CIs exclude 0 on 5/5
- direction: consistent (same sign every fold); stable across folds (majority + all CIs exclude 0): False

### gdd_p_minus_b3_delta (negative = second arm better)

| fold | dMSE | 95% CI | CI excludes 0 | per-seed mean ± sd |
|---|---:|---|---|---|
| 2016 | +0.028247 | [+0.026081, +0.030449] | True | +0.028258 ± 0.000000 |
| 2017 | +0.013597 | [+0.010307, +0.016786] | True | +0.013615 ± 0.000000 |
| 2018 | +0.026238 | [+0.024009, +0.028579] | True | +0.026284 ± 0.000000 |
| 2021 | +0.007415 | [+0.006377, +0.008496] | True | +0.007404 ± 0.000000 |
| 2022 | -0.006114 | [-0.007483, -0.004738] | True | -0.006113 ± 0.000000 |

- folds with CI entirely below 0 (second arm better): 1/5; all CIs exclude 0 on 5/5
- direction: sign flips across folds; stable across folds (majority + all CIs exclude 0): False

### weather_p_minus_b3_delta (negative = second arm better)

| fold | dMSE | 95% CI | CI excludes 0 | per-seed mean ± sd |
|---|---:|---|---|---|
| 2016 | +0.013390 | [+0.011200, +0.015664] | True | +0.013399 ± 0.000000 |
| 2017 | -0.020627 | [-0.023611, -0.017531] | True | -0.020635 ± 0.000000 |
| 2018 | +0.022052 | [+0.019653, +0.024534] | True | +0.022082 ± 0.000000 |
| 2021 | +0.002488 | [+0.001410, +0.003470] | True | +0.002480 ± 0.000000 |
| 2022 | -0.007810 | [-0.009240, -0.006415] | True | -0.007819 ± 0.000000 |

- folds with CI entirely below 0 (second arm better): 2/5; all CIs exclude 0 on 5/5
- direction: sign flips across folds; stable across folds (majority + all CIs exclude 0): False

## E2 -- CC / PH readout (primary |lag| <= 7 d)

### cc

| fold | arm | R2 | MAE |
|---|---|---:|---:|
| 2016 | A_current | 0.70435 | 0.09560 |
| 2016 | b2_no_time | 0.72754 | 0.09190 |
| 2016 | b2_no_time_constant_delta | 0.73532 | 0.09008 |
| 2016 | b3_delta | 0.70743 | 0.09412 |
| 2016 | gdd_p | 0.70856 | 0.09413 |
| 2016 | weather_p | 0.71505 | 0.09192 |
| 2017 | A_current | 0.75875 | 0.08810 |
| 2017 | b2_no_time | 0.76933 | 0.08260 |
| 2017 | b2_no_time_constant_delta | 0.79300 | 0.07831 |
| 2017 | b3_delta | 0.78951 | 0.07943 |
| 2017 | gdd_p | 0.78341 | 0.08040 |
| 2017 | weather_p | 0.77817 | 0.08082 |
| 2018 | A_current | 0.55796 | 0.11008 |
| 2018 | b2_no_time | 0.57413 | 0.10485 |
| 2018 | b2_no_time_constant_delta | 0.59685 | 0.10214 |
| 2018 | b3_delta | 0.55027 | 0.10976 |
| 2018 | gdd_p | 0.55357 | 0.11021 |
| 2018 | weather_p | 0.47580 | 0.11815 |
| 2021 | A_current | 0.73365 | 0.10394 |
| 2021 | b2_no_time | 0.76868 | 0.09349 |
| 2021 | b2_no_time_constant_delta | 0.76340 | 0.09497 |
| 2021 | b3_delta | 0.78807 | 0.08836 |
| 2021 | gdd_p | 0.75810 | 0.09300 |
| 2021 | weather_p | 0.74732 | 0.09692 |
| 2022 | A_current | 0.81874 | 0.08566 |
| 2022 | b2_no_time | 0.82380 | 0.08163 |
| 2022 | b2_no_time_constant_delta | 0.82771 | 0.08173 |
| 2022 | b3_delta | 0.84094 | 0.07939 |
| 2022 | gdd_p | 0.83238 | 0.08086 |
| 2022 | weather_p | 0.83127 | 0.08077 |

| fold | contrast | dR2 | 95% CI |
|---|---|---:|---|
| 2016 | b2_no_time_minus_A_current | +0.02327 | [+0.01545, +0.03120] |
| 2016 | b2_no_time_constant_delta_minus_A_current | +0.03104 | [+0.02322, +0.03869] |
| 2016 | b3_delta_minus_A_current | +0.00315 | [-0.00574, +0.01214] |
| 2016 | b3_delta_minus_b2_no_time_constant_delta | -0.02790 | [-0.03247, -0.02333] |
| 2016 | gdd_p_minus_b3_delta | +0.00113 | [-0.00396, +0.00642] |
| 2016 | weather_p_minus_b3_delta | +0.00773 | [+0.00157, +0.01383] |
| 2017 | b2_no_time_minus_A_current | +0.01053 | [+0.00345, +0.01748] |
| 2017 | b2_no_time_constant_delta_minus_A_current | +0.03414 | [+0.02814, +0.04012] |
| 2017 | b3_delta_minus_A_current | +0.03070 | [+0.02297, +0.03825] |
| 2017 | b3_delta_minus_b2_no_time_constant_delta | -0.00344 | [-0.00920, +0.00246] |
| 2017 | gdd_p_minus_b3_delta | -0.00613 | [-0.01089, -0.00171] |
| 2017 | weather_p_minus_b3_delta | -0.01136 | [-0.01617, -0.00677] |
| 2018 | b2_no_time_minus_A_current | +0.01622 | [+0.00733, +0.02530] |
| 2018 | b2_no_time_constant_delta_minus_A_current | +0.03891 | [+0.02944, +0.04819] |
| 2018 | b3_delta_minus_A_current | -0.00778 | [-0.01956, +0.00386] |
| 2018 | b3_delta_minus_b2_no_time_constant_delta | -0.04669 | [-0.05806, -0.03616] |
| 2018 | gdd_p_minus_b3_delta | +0.00332 | [-0.00551, +0.01205] |
| 2018 | weather_p_minus_b3_delta | -0.07440 | [-0.08660, -0.06207] |
| 2021 | b2_no_time_minus_A_current | +0.03500 | [+0.03197, +0.03820] |
| 2021 | b2_no_time_constant_delta_minus_A_current | +0.02971 | [+0.02603, +0.03324] |
| 2021 | b3_delta_minus_A_current | +0.05440 | [+0.05110, +0.05769] |
| 2021 | b3_delta_minus_b2_no_time_constant_delta | +0.02469 | [+0.02209, +0.02742] |
| 2021 | gdd_p_minus_b3_delta | -0.02996 | [-0.03324, -0.02673] |
| 2021 | weather_p_minus_b3_delta | -0.04074 | [-0.04344, -0.03809] |
| 2022 | b2_no_time_minus_A_current | +0.00506 | [+0.00234, +0.00770] |
| 2022 | b2_no_time_constant_delta_minus_A_current | +0.00895 | [+0.00654, +0.01131] |
| 2022 | b3_delta_minus_A_current | +0.02216 | [+0.01900, +0.02530] |
| 2022 | b3_delta_minus_b2_no_time_constant_delta | +0.01321 | [+0.01096, +0.01537] |
| 2022 | gdd_p_minus_b3_delta | -0.00855 | [-0.01086, -0.00634] |
| 2022 | weather_p_minus_b3_delta | -0.00966 | [-0.01169, -0.00766] |

### ph

| fold | arm | R2 | MAE |
|---|---|---:|---:|
| 2016 | A_current | 0.67676 | 0.15359 |
| 2016 | b2_no_time | 0.68499 | 0.15119 |
| 2016 | b2_no_time_constant_delta | 0.73332 | 0.13795 |
| 2016 | b3_delta | 0.73572 | 0.13533 |
| 2016 | gdd_p | 0.69646 | 0.15040 |
| 2016 | weather_p | 0.63987 | 0.16668 |
| 2017 | A_current | 0.88804 | 0.07446 |
| 2017 | b2_no_time | 0.86555 | 0.07671 |
| 2017 | b2_no_time_constant_delta | 0.87367 | 0.07659 |
| 2017 | b3_delta | 0.89355 | 0.07317 |
| 2017 | gdd_p | 0.85866 | 0.08667 |
| 2017 | weather_p | 0.87771 | 0.08126 |
| 2018 | A_current | 0.89674 | 0.07802 |
| 2018 | b2_no_time | 0.90685 | 0.07389 |
| 2018 | b2_no_time_constant_delta | 0.90658 | 0.07351 |
| 2018 | b3_delta | 0.90818 | 0.07187 |
| 2018 | gdd_p | 0.90503 | 0.07424 |
| 2018 | weather_p | 0.92183 | 0.06697 |
| 2021 | A_current | 0.93258 | 0.07187 |
| 2021 | b2_no_time | 0.93785 | 0.06740 |
| 2021 | b2_no_time_constant_delta | 0.93911 | 0.06580 |
| 2021 | b3_delta | 0.94210 | 0.06440 |
| 2021 | gdd_p | 0.94746 | 0.06045 |
| 2021 | weather_p | 0.95304 | 0.05798 |
| 2022 | A_current | 0.93990 | 0.06856 |
| 2022 | b2_no_time | 0.93504 | 0.06803 |
| 2022 | b2_no_time_constant_delta | 0.93757 | 0.06717 |
| 2022 | b3_delta | 0.94939 | 0.06078 |
| 2022 | gdd_p | 0.95103 | 0.05979 |
| 2022 | weather_p | 0.95183 | 0.06003 |

| fold | contrast | dR2 | 95% CI |
|---|---|---:|---|
| 2016 | b2_no_time_minus_A_current | +0.00823 | [+0.00356, +0.01299] |
| 2016 | b2_no_time_constant_delta_minus_A_current | +0.05659 | [+0.05186, +0.06120] |
| 2016 | b3_delta_minus_A_current | +0.05903 | [+0.05445, +0.06351] |
| 2016 | b3_delta_minus_b2_no_time_constant_delta | +0.00244 | [-0.00077, +0.00573] |
| 2016 | gdd_p_minus_b3_delta | -0.03929 | [-0.04245, -0.03599] |
| 2016 | weather_p_minus_b3_delta | -0.09593 | [-0.10102, -0.09077] |
| 2017 | b2_no_time_minus_A_current | -0.02251 | [-0.02565, -0.01941] |
| 2017 | b2_no_time_constant_delta_minus_A_current | -0.01439 | [-0.01718, -0.01162] |
| 2017 | b3_delta_minus_A_current | +0.00551 | [+0.00289, +0.00819] |
| 2017 | b3_delta_minus_b2_no_time_constant_delta | +0.01990 | [+0.01715, +0.02262] |
| 2017 | gdd_p_minus_b3_delta | -0.03490 | [-0.03736, -0.03242] |
| 2017 | weather_p_minus_b3_delta | -0.01586 | [-0.01822, -0.01348] |
| 2018 | b2_no_time_minus_A_current | +0.01006 | [+0.00657, +0.01354] |
| 2018 | b2_no_time_constant_delta_minus_A_current | +0.00978 | [+0.00660, +0.01314] |
| 2018 | b3_delta_minus_A_current | +0.01142 | [+0.00835, +0.01444] |
| 2018 | b3_delta_minus_b2_no_time_constant_delta | +0.00164 | [-0.00030, +0.00359] |
| 2018 | gdd_p_minus_b3_delta | -0.00317 | [-0.00476, -0.00157] |
| 2018 | weather_p_minus_b3_delta | +0.01364 | [+0.01190, +0.01547] |
| 2021 | b2_no_time_minus_A_current | +0.00528 | [+0.00418, +0.00649] |
| 2021 | b2_no_time_constant_delta_minus_A_current | +0.00653 | [+0.00524, +0.00791] |
| 2021 | b3_delta_minus_A_current | +0.00950 | [+0.00835, +0.01075] |
| 2021 | b3_delta_minus_b2_no_time_constant_delta | +0.00298 | [+0.00227, +0.00371] |
| 2021 | gdd_p_minus_b3_delta | +0.00537 | [+0.00471, +0.00602] |
| 2021 | weather_p_minus_b3_delta | +0.01095 | [+0.01010, +0.01183] |
| 2022 | b2_no_time_minus_A_current | -0.00486 | [-0.00632, -0.00338] |
| 2022 | b2_no_time_constant_delta_minus_A_current | -0.00233 | [-0.00356, -0.00111] |
| 2022 | b3_delta_minus_A_current | +0.00950 | [+0.00823, +0.01077] |
| 2022 | b3_delta_minus_b2_no_time_constant_delta | +0.01183 | [+0.01119, +0.01248] |
| 2022 | gdd_p_minus_b3_delta | +0.00164 | [+0.00092, +0.00237] |
| 2022 | weather_p_minus_b3_delta | +0.00244 | [+0.00158, +0.00334] |

## Decision summary (fixed rules)

- E1 `b3_delta_minus_b2_no_time_constant_delta`: mean dMSE over folds +0.030400; stable = False.
- E1 `gdd_p_minus_b3_delta`: mean dMSE over folds +0.013877; stable = False.
- E1 `weather_p_minus_b3_delta`: mean dMSE over folds +0.001899; stable = False.

- E2 cc `b3_delta_minus_b2_no_time_constant_delta`: mean dR2 over folds -0.00803 (n=5).
- E2 cc `gdd_p_minus_b3_delta`: mean dR2 over folds -0.00804 (n=5).
- E2 cc `weather_p_minus_b3_delta`: mean dR2 over folds -0.02569 (n=5).
- E2 ph `b3_delta_minus_b2_no_time_constant_delta`: mean dR2 over folds +0.00776 (n=5).
- E2 ph `gdd_p_minus_b3_delta`: mean dR2 over folds -0.01407 (n=5).
- E2 ph `weather_p_minus_b3_delta`: mean dR2 over folds -0.01695 (n=5).

## Caveats

- This is a training-set-internal diagnostic, not a 2019 confirmation.
- These numbers are the 1-seed pilot; the formal 3-seed grid (seeds 0,1,2) has NOT run, so no cross-seed CI is reported here.
- Differences are plot-clustered bootstrap CIs.
- GDD-P and Weather-P add parameters over B3 (see data audit §3.4), so part of any gain may be parameter-driven.
- A positive dMSE (E1) means the second arm is WORSE; a positive dR2 (E2) means the second arm is better.

