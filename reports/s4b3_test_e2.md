# S4B.3 E2 -- CC / PH Readout on the Official Test Splits

- Generated: 2026-10-28T12:57:56+0800
- Git HEAD: `7f0fd9c18ddb0c8cdb99ebd12a63fbae0c242795`
- Protocol: `reports/s4b3_e2_protocol.json` (sha `ee8ffcfad9d0eb41`)
- Inner split: `reports/s4b3_e2_plot_split.csv` (sha `42e8cf9aa32ca2d1`)
- Train scope: all 2,930 official-train plots; official validation and test plots excluded from training/selection.
- Linear readout, mean-pooled 768-d features, epoch selected on inner-dev; test labels read only for evaluation.

## Label matching audit (test splits)

| split | pairs | plots | cc valid (lag<=7) | ph valid (lag<=7) | cc lag | ph lag |
|---|---:|---:|---:|---:|---|---|
| test_plot | 9,851 | 250 | 9,372 | 7,197 | -54..21 d | -19..147 d |
| test_genotype | 12,059 | 246 | 11,544 | 9,543 | -61..25 d | -19..147 d |
| test_environment | 6,300 | 190 | 5,993 | 6,300 | -95..23 d | -5..2 d |
| test_genotype_environment | 2,147 | 62 | 1,923 | 2,147 | -82..14 d | -6..2 d |

## Primary analysis (|lag| <= 7 d): MAE / RMSE / R2

| trait | split | condition | MAE | RMSE | R2 | n | plots | epoch |
|---|---|---|---:|---:|---:|---:|---:|---:|
| cc | test_plot | A_current | 0.08137 | 0.10777 | 0.79022 | 5,549 | 250 | 32 |
| cc | test_plot | B2 | 0.06766 | 0.09279 | 0.84447 | 5,549 | 250 | 24 |
| cc | test_plot | B2-C | 0.06797 | 0.09293 | 0.84401 | 5,549 | 250 | 32 |
| cc | test_plot | B3 | 0.06410 | 0.08675 | 0.86406 | 5,549 | 250 | 57 |
| cc | test_genotype | A_current | 0.07540 | 0.10075 | 0.84260 | 6,685 | 246 | 32 |
| cc | test_genotype | B2 | 0.06147 | 0.08545 | 0.88677 | 6,685 | 246 | 24 |
| cc | test_genotype | B2-C | 0.06143 | 0.08542 | 0.88686 | 6,685 | 246 | 32 |
| cc | test_genotype | B3 | 0.05583 | 0.07637 | 0.90956 | 6,685 | 246 | 57 |
| cc | test_environment | A_current | 0.09598 | 0.12751 | 0.28005 | 3,031 | 190 | 32 |
| cc | test_environment | B2 | 0.10601 | 0.13882 | 0.14657 | 3,031 | 190 | 24 |
| cc | test_environment | B2-C | 0.10406 | 0.13617 | 0.17893 | 3,031 | 190 | 32 |
| cc | test_environment | B3 | 0.09224 | 0.12592 | 0.29790 | 3,031 | 190 | 57 |
| cc | test_genotype_environment | A_current | 0.08396 | 0.11253 | 0.45004 | 962 | 62 | 32 |
| cc | test_genotype_environment | B2 | 0.09605 | 0.12663 | 0.30358 | 962 | 62 | 24 |
| cc | test_genotype_environment | B2-C | 0.09318 | 0.12364 | 0.33609 | 962 | 62 | 32 |
| cc | test_genotype_environment | B3 | 0.08137 | 0.11485 | 0.42715 | 962 | 62 | 57 |
| ph | test_plot | A_current | 0.05916 | 0.07847 | 0.94840 | 4,100 | 250 | 19 |
| ph | test_plot | B2 | 0.04745 | 0.06560 | 0.96394 | 4,100 | 250 | 63 |
| ph | test_plot | B2-C | 0.04700 | 0.06466 | 0.96497 | 4,100 | 250 | 117 |
| ph | test_plot | B3 | 0.04674 | 0.06419 | 0.96548 | 4,100 | 250 | 46 |
| ph | test_genotype | A_current | 0.06064 | 0.08136 | 0.94825 | 5,298 | 246 | 19 |
| ph | test_genotype | B2 | 0.04980 | 0.06852 | 0.96330 | 5,298 | 246 | 63 |
| ph | test_genotype | B2-C | 0.04998 | 0.06873 | 0.96307 | 5,298 | 246 | 117 |
| ph | test_genotype | B3 | 0.04961 | 0.06816 | 0.96368 | 5,298 | 246 | 46 |
| ph | test_environment | A_current | 0.04412 | 0.06199 | 0.93968 | 3,291 | 190 | 19 |
| ph | test_environment | B2 | 0.04474 | 0.05962 | 0.94420 | 3,291 | 190 | 63 |
| ph | test_environment | B2-C | 0.04009 | 0.05701 | 0.94898 | 3,291 | 190 | 117 |
| ph | test_environment | B3 | 0.04325 | 0.05911 | 0.94514 | 3,291 | 190 | 46 |
| ph | test_genotype_environment | A_current | 0.04294 | 0.05936 | 0.95722 | 1,146 | 62 | 19 |
| ph | test_genotype_environment | B2 | 0.04429 | 0.05798 | 0.95919 | 1,146 | 62 | 63 |
| ph | test_genotype_environment | B2-C | 0.03897 | 0.05392 | 0.96470 | 1,146 | 62 | 117 |
| ph | test_genotype_environment | B3 | 0.04366 | 0.05890 | 0.95788 | 1,146 | 62 | 46 |

## Primary analysis: bootstrap 95% CI of R2

| trait | split | condition | R2 | 95% CI |
|---|---|---|---:|---|
| cc | test_plot | A_current | 0.79022 | [0.77223, 0.80623] |
| cc | test_plot | B2 | 0.84447 | [0.82821, 0.86012] |
| cc | test_plot | B2-C | 0.84401 | [0.82770, 0.85983] |
| cc | test_plot | B3 | 0.86406 | [0.84743, 0.87933] |
| cc | test_genotype | A_current | 0.84260 | [0.83020, 0.85295] |
| cc | test_genotype | B2 | 0.88677 | [0.87611, 0.89571] |
| cc | test_genotype | B2-C | 0.88686 | [0.87626, 0.89571] |
| cc | test_genotype | B3 | 0.90956 | [0.89955, 0.91796] |
| cc | test_environment | A_current | 0.28005 | [0.20471, 0.34772] |
| cc | test_environment | B2 | 0.14657 | [0.06216, 0.22655] |
| cc | test_environment | B2-C | 0.17893 | [0.09734, 0.25479] |
| cc | test_environment | B3 | 0.29790 | [0.22730, 0.36219] |
| cc | test_genotype_environment | A_current | 0.45004 | [0.30436, 0.57106] |
| cc | test_genotype_environment | B2 | 0.30358 | [0.15029, 0.43472] |
| cc | test_genotype_environment | B2-C | 0.33609 | [0.19378, 0.45904] |
| cc | test_genotype_environment | B3 | 0.42715 | [0.29197, 0.53954] |
| ph | test_plot | A_current | 0.94840 | [0.94331, 0.95319] |
| ph | test_plot | B2 | 0.96394 | [0.96014, 0.96780] |
| ph | test_plot | B2-C | 0.96497 | [0.96133, 0.96877] |
| ph | test_plot | B3 | 0.96548 | [0.96173, 0.96919] |
| ph | test_genotype | A_current | 0.94825 | [0.94297, 0.95300] |
| ph | test_genotype | B2 | 0.96330 | [0.95883, 0.96697] |
| ph | test_genotype | B2-C | 0.96307 | [0.95870, 0.96680] |
| ph | test_genotype | B3 | 0.96368 | [0.95949, 0.96742] |
| ph | test_environment | A_current | 0.93968 | [0.92909, 0.94839] |
| ph | test_environment | B2 | 0.94420 | [0.93411, 0.95211] |
| ph | test_environment | B2-C | 0.94898 | [0.93877, 0.95698] |
| ph | test_environment | B3 | 0.94514 | [0.93515, 0.95324] |
| ph | test_genotype_environment | A_current | 0.95722 | [0.95117, 0.96251] |
| ph | test_genotype_environment | B2 | 0.95919 | [0.95343, 0.96391] |
| ph | test_genotype_environment | B2-C | 0.96470 | [0.95956, 0.96893] |
| ph | test_genotype_environment | B3 | 0.95788 | [0.95161, 0.96282] |

## Primary analysis: paired contrasts vs A_current (delta R2)

| trait | split | contrast | dR2 | 95% CI |
|---|---|---|---:|---|
| cc | test_plot | B2_minus_A_current | +0.05434 | [+0.04638, +0.06228] |
| cc | test_plot | B2-C_minus_A_current | +0.05387 | [+0.04587, +0.06212] |
| cc | test_plot | B3_minus_A_current | +0.07394 | [+0.06510, +0.08346] |
| cc | test_plot | B3_minus_B2-C | +0.02007 | [+0.01504, +0.02485] |
| cc | test_plot | B2-C_minus_B2 | -0.00047 | [-0.00339, +0.00248] |
| cc | test_genotype | B2_minus_A_current | +0.04419 | [+0.03867, +0.05007] |
| cc | test_genotype | B2-C_minus_A_current | +0.04426 | [+0.03868, +0.04994] |
| cc | test_genotype | B3_minus_A_current | +0.06705 | [+0.06112, +0.07315] |
| cc | test_genotype | B3_minus_B2-C | +0.02279 | [+0.01871, +0.02688] |
| cc | test_genotype | B2-C_minus_B2 | +0.00007 | [-0.00193, +0.00204] |
| cc | test_environment | B2_minus_A_current | -0.13369 | [-0.16430, -0.10308] |
| cc | test_environment | B2-C_minus_A_current | -0.10129 | [-0.12929, -0.07413] |
| cc | test_environment | B3_minus_A_current | +0.01785 | [-0.00764, +0.04417] |
| cc | test_environment | B3_minus_B2-C | +0.11913 | [+0.09518, +0.14300] |
| cc | test_environment | B2-C_minus_B2 | +0.03240 | [+0.01904, +0.04605] |
| cc | test_genotype_environment | B2_minus_A_current | -0.14623 | [-0.18960, -0.10435] |
| cc | test_genotype_environment | B2-C_minus_A_current | -0.11337 | [-0.15242, -0.07133] |
| cc | test_genotype_environment | B3_minus_A_current | -0.02242 | [-0.05555, +0.01059] |
| cc | test_genotype_environment | B3_minus_B2-C | +0.09095 | [+0.06063, +0.12331] |
| cc | test_genotype_environment | B2-C_minus_B2 | +0.03286 | [+0.01696, +0.04984] |
| ph | test_plot | B2_minus_A_current | +0.01552 | [+0.01293, +0.01817] |
| ph | test_plot | B2-C_minus_A_current | +0.01656 | [+0.01399, +0.01927] |
| ph | test_plot | B3_minus_A_current | +0.01705 | [+0.01421, +0.01996] |
| ph | test_plot | B3_minus_B2-C | +0.00049 | [-0.00090, +0.00191] |
| ph | test_plot | B2-C_minus_B2 | +0.00105 | [-0.00004, +0.00222] |
| ph | test_genotype | B2_minus_A_current | +0.01508 | [+0.01252, +0.01761] |
| ph | test_genotype | B2-C_minus_A_current | +0.01488 | [+0.01250, +0.01734] |
| ph | test_genotype | B3_minus_A_current | +0.01548 | [+0.01315, +0.01794] |
| ph | test_genotype | B3_minus_B2-C | +0.00060 | [-0.00046, +0.00171] |
| ph | test_genotype | B2-C_minus_B2 | -0.00020 | [-0.00112, +0.00070] |
| ph | test_environment | B2_minus_A_current | +0.00447 | [+0.00153, +0.00757] |
| ph | test_environment | B2-C_minus_A_current | +0.00928 | [+0.00604, +0.01267] |
| ph | test_environment | B3_minus_A_current | +0.00542 | [+0.00152, +0.00966] |
| ph | test_environment | B3_minus_B2-C | -0.00386 | [-0.00676, -0.00106] |
| ph | test_environment | B2-C_minus_B2 | +0.00482 | [+0.00331, +0.00639] |
| ph | test_genotype_environment | B2_minus_A_current | +0.00197 | [-0.00157, +0.00570] |
| ph | test_genotype_environment | B2-C_minus_A_current | +0.00749 | [+0.00434, +0.01087] |
| ph | test_genotype_environment | B3_minus_A_current | +0.00062 | [-0.00330, +0.00491] |
| ph | test_genotype_environment | B3_minus_B2-C | -0.00687 | [-0.01029, -0.00332] |
| ph | test_genotype_environment | B2-C_minus_B2 | +0.00552 | [+0.00363, +0.00747] |

## Sensitivity: strict-lag3 / same-day mean MAE over splits

| trait | lag variant | condition | mean MAE |
|---|---|---|---:|
| cc | strict_lag3 | A_current | 0.08107 |
| cc | strict_lag3 | B2 | 0.08226 |
| cc | strict_lag3 | B2-C | 0.08189 |
| cc | strict_lag3 | B3 | 0.07220 |
| cc | same_day_lag0 | A_current | 0.08231 |
| cc | same_day_lag0 | B2 | 0.08128 |
| cc | same_day_lag0 | B2-C | 0.08177 |
| cc | same_day_lag0 | B3 | 0.07320 |
| ph | strict_lag3 | A_current | 0.05280 |
| ph | strict_lag3 | B2 | 0.04297 |
| ph | strict_lag3 | B2-C | 0.04265 |
| ph | strict_lag3 | B3 | 0.04687 |
| ph | same_day_lag0 | A_current | 0.04609 |
| ph | same_day_lag0 | B2 | 0.04576 |
| ph | same_day_lag0 | B2-C | 0.04277 |
| ph | same_day_lag0 | B3 | 0.04780 |

## Notes

- A_current = current frozen DINOv2 feature; B2/B2-C/B3 = predicted future latent.
- B2-C receives one fixed 9.0 d interval; B3 receives each pair's real delta_days.
- Missing labels use a validity mask and are never filled with 0.
- Test labels are used only for the final evaluation, never for training, epoch choice or model selection.
