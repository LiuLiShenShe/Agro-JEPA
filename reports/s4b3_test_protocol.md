# S4B.3 Official-Test Evaluation Protocol (frozen before E1)

- Generated: 2026-10-28T09:58:53+0800
- Git HEAD: `1df54fb526df77c80a45d44e004999673620651a` (agro-s4b3-official-test)
- S4B.2 baseline commit: `1df54fb526df77c80a45d44e004999673620651a`
- Config: `agro_jepa/config/s4b3_official_test.yaml` (sha `10d435ab8b0e561a`)

## Data: four official test splits (locally re-verified)

| split | pairs | plots | images | span7 | span14 | delta range | years |
|---|---:|---:|---:|---:|---:|---|---|
| test_plot | 9,851 | 250 | 7,050 | 5067 | 4784 | 5-16 d | [2016, 2017, 2018, 2021, 2022] |
| test_genotype | 12,059 | 246 | 8,149 | 6191 | 5868 | 5-16 d | [2016, 2017, 2018, 2021, 2022] |
| test_environment | 6,300 | 190 | 4,281 | 3356 | 2944 | 5-16 d | [2019] |
| test_genotype_environment | 2,147 | 62 | 1,479 | 1135 | 1012 | 5-16 d | [2019] |
| **total** | **30,357** | **748** | -- | -- | -- | -- | -- |

All four index SHA256 values and row/plot counts independently re-produce the frozen S4A.1 audit. Conditioned npz inputs are present and byte-identical.

## Nine frozen checkpoints

| condition | seed | final.pt SHA256 |
|---|---:|---|
| b2_no_time | seed0 | `08c9dafc450a70a2c24a47216f30c0350c600b858e5eb5ef8847c6dbb3f34b04` |
| b2_no_time | seed1 | `7b1b3827c5e6a6d84322dda1b397b0cde9ea4a6ce239f6494a8bf615a5b31503` |
| b2_no_time | seed2 | `540bbb96dbb350ffb80cb0fab715e33ce2f36550843b04b14f11cab5e50a57df` |
| b2_no_time_constant_delta | seed0 | `d8cebf42d2f97ebe27dbf2cfeb231669cbf4f6ff5e60e330649399a3cf292757` |
| b2_no_time_constant_delta | seed1 | `61a0d962309d3361a5bc1291e13af6b3908f40caf6201c7b9f543d1e823d9d20` |
| b2_no_time_constant_delta | seed2 | `9391faa54f26c6760a11d86601bed7e94225b7a5e2d3c4b35cd1c593a1316230` |
| b3_delta | seed0 | `f433164646886c691eb2a5d795924167c0f114c0b95f5d261722df565e8a0b5a` |
| b3_delta | seed1 | `64ee70f8e4574ba7bff47b2206e6d19a1fed03c34a0e9b59fcc63463afe9ab7f` |
| b3_delta | seed2 | `37ca926d05f7ac959710027efc57997173fb4275e2ed67ae5d277e2c55d33e21` |

## E1 -- future-latent prediction (no test label read)

- Arms: **B0** persistence, **B1** train-mean future, **B2** no time, **B2-C** fixed 9.0 d, **B3** real delta_days.
- B2/B2-C/B3 evaluated on seeds 0,1,2 individually and as a three-seed prediction ensemble.
- Bootstrap: plot-clustered, 2000 resamples, seed 20261024; all pairs of a resampled plot stay together.
- Stratified by span7 / span14 and harvest year; paired B3-B2-C and B3-B2 MSE differences reported.
- The four test indexes carry **no** phenotype label column, so E1 cannot read a test label.

## E2 -- CC / PH phenotype readout (frozen before test labels read)

- Linear readout on **all 2,930 official-train plots**, inner 80/20 split by plot_uid stratified by harvest_year, seed 20261025.
- Epoch selected on the inner-dev set, then refit on the full official-train data; official validation and every test plot stay out of training and selection.
- Conditions: **A_current** (mean z_a), **B2**, **B2-C**, **B3** (mean predicted z_hat_b).
- Primary lag window: |label_lag_days| <= 7; strict-lag3 and same-day are sensitivity analyses.
- Missing labels use a real validity mask; they are never filled with 0.

## Protocol checks

| check | pass | detail |
|---|---|---|
| C1_nine_checkpoints_present | PASS | 9 final.pt files |
| C2_feature_index_sha | PASS | 3c0196f8592fe2b4 (expected 3c0196f8592fe2b4) |
| C3_test_counts_and_sha | PASS | row/plot counts and SHA256 match the frozen S4A.1 audit |
| C4_test_feature_coverage | PASS | every test image endpoint has a cached DINOv2 feature |
| C5_test_indexes_label_free | PASS | no test pair index carries a phenotype label column (E1 blind) |
| C6_total_test_pairs | PASS | 30357 pairs over 748 plots across four splits |
| C7_no_pair_id_leak | PASS | 0 test pair_ids also in full-train table |
| C8_no_plot_leak | PASS | 0 test plots also in full-train table |
| C9_full_train_table_sha | PASS | a49b546f57d68881 |
| C10_full_train_counts | PASS | 179823 pairs / 2930 plots / splits ['train'] |
| C11_b2c_vs_b3_distinction | PASS | B2-C fixed 9.0 d (normalised -0.341964); B3 uses each pair's real delta_days in [5,16] |

**all_pass = True**
