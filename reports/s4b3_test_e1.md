# S4B.3 E1 -- Future-Latent Prediction on the Official Test Splits

- Generated: 2026-10-28T10:10:20+0800
- Git HEAD: `598e0263dc58f57197ffb5a3e6e5b7e57645261a`
- Protocol: `reports/s4b3_test_protocol.json` (all_pass = True)
- Seeds: [0, 1, 2]; seed rule: seed -1 = prediction averaged over the seed set, then MSE; each other seed = that checkpoint alone
- Bootstrap: plot-clustered, 2000 resamples, seed 20261024
- E1 reads **no** test phenotype label (the test indexes carry none).

## Test splits

| split | pairs | plots | years | span7 | span14 |
|---|---:|---:|---|---:|---:|
| test_plot | 9,851 | 250 | [2016, 2017, 2018, 2021, 2022] | 5067 | 4784 |
| test_genotype | 12,059 | 246 | [2016, 2017, 2018, 2021, 2022] | 6191 | 5868 |
| test_environment | 6,300 | 190 | [2019] | 3356 | 2944 |
| test_genotype_environment | 2,147 | 62 | [2019] | 1135 | 1012 |

## Seed-averaged MSE / cosine / relative improvement

| split | arm | MSE | cos | vs B0 | vs B1 |
|---|---|---:|---:|---:|---:|
| test_plot | B0 persistence | 2.183910 | 0.6303 | 0 | -- |
| | B1 train-mean | 1.829310 | -- | -- | 0 |
| | b2_no_time | 1.126142 | 0.7853 | 48.43% | 38.44% |
| | b2_no_time_constant_delta | 1.125191 | 0.7855 | 48.48% | 38.49% |
| | b3_delta | 1.073037 | 0.7965 | 50.87% | 41.34% |
| test_genotype | B0 persistence | 2.200029 | 0.6267 | 0 | -- |
| | B1 train-mean | 1.835187 | -- | -- | 0 |
| | b2_no_time | 1.137704 | 0.7821 | 48.29% | 38.01% |
| | b2_no_time_constant_delta | 1.137633 | 0.7821 | 48.29% | 38.01% |
| | b3_delta | 1.082086 | 0.7939 | 50.81% | 41.04% |
| test_environment | B0 persistence | 2.077327 | 0.6554 | 0 | -- |
| | B1 train-mean | 1.858036 | -- | -- | 0 |
| | b2_no_time | 1.265049 | 0.7625 | 39.10% | 31.91% |
| | b2_no_time_constant_delta | 1.264772 | 0.7626 | 39.12% | 31.93% |
| | b3_delta | 1.272027 | 0.7621 | 38.77% | 31.54% |
| test_genotype_environment | B0 persistence | 2.067707 | 0.6588 | 0 | -- |
| | B1 train-mean | 1.880009 | -- | -- | 0 |
| | b2_no_time | 1.256012 | 0.7658 | 39.26% | 33.19% |
| | b2_no_time_constant_delta | 1.255677 | 0.7658 | 39.27% | 33.21% |
| | b3_delta | 1.259689 | 0.7661 | 39.08% | 33.00% |

## Bootstrap 95% CI (plot-clustered) -- relative improvement

| split | arm | vs B0 [95% CI] | vs B1 [95% CI] |
|---|---|---|---|
| test_plot | b2_no_time | 48.44% [48.13, 48.74] | 38.44% [38.08, 38.80] |
| test_plot | b2_no_time_constant_delta | 48.48% [48.17, 48.78] | 38.50% [38.12, 38.85] |
| test_plot | b3_delta | 50.87% [50.58, 51.16] | 41.35% [41.00, 41.70] |
| test_genotype | b2_no_time | 48.28% [47.95, 48.63] | 38.00% [37.60, 38.41] |
| test_genotype | b2_no_time_constant_delta | 48.29% [47.95, 48.63] | 38.01% [37.61, 38.42] |
| test_genotype | b3_delta | 50.81% [50.47, 51.14] | 41.03% [40.63, 41.43] |
| test_environment | b2_no_time | 39.10% [38.80, 39.40] | 31.92% [31.58, 32.25] |
| test_environment | b2_no_time_constant_delta | 39.12% [38.81, 39.43] | 31.93% [31.59, 32.28] |
| test_environment | b3_delta | 38.77% [38.39, 39.14] | 31.54% [31.12, 31.96] |
| test_genotype_environment | b2_no_time | 39.25% [38.68, 39.81] | 33.19% [32.56, 33.80] |
| test_genotype_environment | b2_no_time_constant_delta | 39.27% [38.67, 39.86] | 33.21% [32.55, 33.85] |
| test_genotype_environment | b3_delta | 39.07% [38.41, 39.75] | 32.99% [32.26, 33.74] |

## Paired B3 - B2-C and B3 - B2 (seed-averaged, plot bootstrap)

| split | contrast | dMSE | 95% CI |
|---|---|---:|---|
| test_plot | b3_delta_minus_b2_no_time_constant_delta | -0.052152 | [-0.054357, -0.049936] |
| test_plot | b3_delta_minus_b2_no_time | -0.053090 | [-0.055226, -0.050953] |
| test_plot | b2_no_time_constant_delta_minus_b2_no_time | -0.000938 | [-0.001618, -0.000240] |
| test_genotype | b3_delta_minus_b2_no_time_constant_delta | -0.055499 | [-0.057424, -0.053441] |
| test_genotype | b3_delta_minus_b2_no_time | -0.055571 | [-0.057542, -0.053470] |
| test_genotype | b2_no_time_constant_delta_minus_b2_no_time | -0.000072 | [-0.000654, +0.000498] |
| test_environment | b3_delta_minus_b2_no_time_constant_delta | +0.007259 | [+0.003739, +0.010783] |
| test_environment | b3_delta_minus_b2_no_time | +0.006983 | [+0.003465, +0.010489] |
| test_environment | b2_no_time_constant_delta_minus_b2_no_time | -0.000276 | [-0.001862, +0.001453] |
| test_genotype_environment | b3_delta_minus_b2_no_time_constant_delta | +0.004024 | [-0.001978, +0.010090] |
| test_genotype_environment | b3_delta_minus_b2_no_time | +0.003684 | [-0.002656, +0.009905] |
| test_genotype_environment | b2_no_time_constant_delta_minus_b2_no_time | -0.000341 | [-0.003211, +0.002674] |

## Per-seed spread (training randomness, not a plot interval)

| split | contrast | seed0 | seed1 | seed2 | mean +/- std | same sign |
|---|---|---:|---:|---:|---|---|
| test_plot | b3_delta_minus_b2_no_time_constant_delta | -0.045316 | -0.047993 | -0.047006 | -0.046772 +/- 0.001354 | True |
| test_plot | b3_delta_minus_b2_no_time | -0.047115 | -0.049139 | -0.047265 | -0.047840 +/- 0.001128 | True |
| test_plot | b2_no_time_constant_delta_minus_b2_no_time | -0.001799 | -0.001146 | -0.000259 | -0.001068 +/- 0.000773 | True |
| test_genotype | b3_delta_minus_b2_no_time_constant_delta | -0.048963 | -0.049992 | -0.050753 | -0.049903 +/- 0.000898 | True |
| test_genotype | b3_delta_minus_b2_no_time | -0.050162 | -0.049704 | -0.050902 | -0.050256 +/- 0.000605 | True |
| test_genotype | b2_no_time_constant_delta_minus_b2_no_time | -0.001199 | +0.000288 | -0.000149 | -0.000354 +/- 0.000764 | False |
| test_environment | b3_delta_minus_b2_no_time_constant_delta | +0.025323 | +0.015929 | +0.021944 | +0.021065 +/- 0.004758 | True |
| test_environment | b3_delta_minus_b2_no_time | +0.025141 | +0.011087 | +0.023770 | +0.019999 +/- 0.007749 | True |
| test_environment | b2_no_time_constant_delta_minus_b2_no_time | -0.000182 | -0.004842 | +0.001826 | -0.001066 +/- 0.003421 | False |
| test_genotype_environment | b3_delta_minus_b2_no_time_constant_delta | +0.022457 | +0.014339 | +0.010170 | +0.015655 +/- 0.006249 | True |
| test_genotype_environment | b3_delta_minus_b2_no_time | +0.022003 | +0.009178 | +0.011724 | +0.014302 +/- 0.006790 | True |
| test_genotype_environment | b2_no_time_constant_delta_minus_b2_no_time | -0.000454 | -0.005161 | +0.001554 | -0.001354 +/- 0.003447 | False |

## Span stratification (seed-averaged MSE)

| split | arm | span7 MSE | span14 MSE | span7 vs B0 | span14 vs B0 |
|---|---|---:|---:|---:|---:|
| test_plot | b2_no_time | 1.100376 | 1.153431 | 46.21% | 50.50% |
| test_plot | b2_no_time_constant_delta | 1.099543 | 1.152357 | 46.25% | 50.55% |
| test_plot | b3_delta | 1.048166 | 1.099381 | 48.76% | 52.82% |
| test_genotype | b2_no_time | 1.112712 | 1.164072 | 46.15% | 50.28% |
| test_genotype | b2_no_time_constant_delta | 1.112697 | 1.163943 | 46.15% | 50.28% |
| test_genotype | b3_delta | 1.058253 | 1.107231 | 48.79% | 52.70% |
| test_environment | b2_no_time | 1.256441 | 1.274861 | 35.55% | 42.66% |
| test_environment | b2_no_time_constant_delta | 1.256417 | 1.274297 | 35.55% | 42.68% |
| test_environment | b3_delta | 1.267236 | 1.277488 | 34.99% | 42.54% |
| test_genotype_environment | b2_no_time | 1.246994 | 1.266126 | 35.69% | 42.76% |
| test_genotype_environment | b2_no_time_constant_delta | 1.246663 | 1.265787 | 35.70% | 42.78% |
| test_genotype_environment | b3_delta | 1.252098 | 1.268203 | 35.42% | 42.67% |

## Notes

- B0 = persistence (z_a), B1 = mean z_b over the full S4B.2 official-train index.
- Seed-averaged = prediction averaged over seeds {0,1,2} before MSE.
- Plot bootstrap captures test-sample sampling only; per-seed spread captures training randomness and is reported separately.
- B2-C receives one fixed 9.0 d interval; B3 receives each pair's real delta_days.
