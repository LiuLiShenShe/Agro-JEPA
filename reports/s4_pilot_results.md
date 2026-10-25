# S4A Pilot 结果：时间、积温与天气驱动的未来潜空间预测基线

> 本文件由 `tools/s4_write_reports.py` 从真实运行产物生成。所有数字均来自 `reports/*.json`，未手工填写。

## 1. 运行清单

| 条件 | 协议 | 可训练参数 | 条件参数 | 预测器输入维 | seeds | 候选 epoch | 实际 epoch (seed0) |
|---|---|---|---|---|---|---|---|
| `b2_no_time` | none | 45898196 | 0 | 928 | [0, 1] | 16 | 16 |
| `b3_delta` | none | 53882396 | 72730 | 1088 | [0, 1] | 16 | 15 |
| `b4_gdd_R` | R | 53829660 | 19994 | 1088 | [0, 1] | 16 | 16 |
| `b4_delta_gdd_R` | R | 53904384 | 94718 | 1088 | [0, 1] | 16 | 14 |
| `b4_delta_gdd_R_tbase5` | R | 53904384 | 94718 | 1088 | [0, 1] | 16 | 15 |
| `b4_delta_gdd_P` | P | 53908480 | 98814 | 1088 | [0, 1] | 16 | 14 |
| `b5_delta_weather_P` | P | 53906928 | 97262 | 1088 | [0, 1] | 16 | 14 |
| `b5_delta_weather_R` | R | 53906928 | 97262 | 1088 | [0, 1] | 16 | 14 |

## 2. E1：官方 validation 未来潜空间预测

评价集：15175 对 / 380 个 plot / 10858 张图像；年份 [2016, 2017, 2018, 2021, 2022]。

相对降幅的 95% CI 来自 2,000 次 plot 级 cluster bootstrap。

| 条件 | MSE | vs Persistence | vs train-mean | cosine | n |
|---|---|---|---|---|---|
| `b0_persistence` | 2.19515 | 0.00% | — | — | 15175 |
| `b1_train_mean` | 1.83561 | 16.38% | — | — | 15175 |
| `b2_no_time` | 1.17733 | 46.37% [46.14%, 46.61%] | 35.86% [35.49%, 36.26%] | 0.7740 | 15175 |
| `b3_delta` | 1.13245 | 48.41% [48.17%, 48.67%] | 38.31% [37.93%, 38.70%] | 0.7835 | 15175 |
| `b4_gdd_R` | 1.13510 | 48.29% [48.04%, 48.55%] | 38.16% [37.79%, 38.55%] | 0.7830 | 15175 |
| `b4_delta_gdd_R` | 1.12846 | 48.59% [48.34%, 48.85%] | 38.52% [38.14%, 38.92%] | 0.7844 | 15175 |
| `b4_delta_gdd_R_tbase5` | 1.12703 | 48.66% [48.41%, 48.91%] | 38.60% [38.23%, 38.99%] | 0.7847 | 15175 |
| `b4_delta_gdd_P` | 1.12603 | 48.70% [48.46%, 48.96%] | 38.66% [38.30%, 39.04%] | 0.7849 | 15175 |
| `b5_delta_weather_P` | 1.13029 | 48.51% [48.26%, 48.76%] | 38.42% [38.05%, 38.81%] | 0.7839 | 15175 |
| `b5_delta_weather_R` | 1.12412 | 48.79% [48.54%, 49.05%] | 38.76% [38.41%, 39.13%] | 0.7853 | 15175 |

### 2.1 关键对照（MSE 差值，95% CI）

| 对照 | ΔMSE | 95% CI |
|---|---|---|
| `b3_delta_minus_b2_no_time` | -0.04486 | [-0.04642, -0.04330] |
| `b4_delta_gdd_R_minus_b3_delta` | -0.00399 | [-0.00513, -0.00287] |
| `b4_delta_gdd_P_minus_b3_delta` | -0.00641 | [-0.00750, -0.00532] |
| `b5_delta_weather_P_minus_b3_delta` | -0.00215 | [-0.00308, -0.00116] |
| `b5_delta_weather_R_minus_b3_delta` | -0.00833 | [-0.00957, -0.00710] |
| `b4_delta_gdd_R_tbase5_minus_b4_delta_gdd_R` | -0.00144 | [-0.00219, -0.00068] |
| `b4_delta_gdd_R_minus_b4_delta_gdd_P` | +0.00242 | [+0.00128, +0.00361] |
| `b5_delta_weather_R_minus_b5_delta_weather_P` | -0.00619 | [-0.00719, -0.00515] |

### 2.2 按 pair_type

| 条件 | span7 MSE | span14 MSE | span7 cos | span14 cos |
|---|---|---|---|---|
| `b2_no_time` | 1.14888 | 1.20756 | 0.7804 | 0.7672 |
| `b3_delta` | 1.10474 | 1.16190 | 0.7897 | 0.7769 |
| `b4_gdd_R` | 1.10699 | 1.16497 | 0.7892 | 0.7763 |
| `b4_delta_gdd_R` | 1.10038 | 1.15829 | 0.7906 | 0.7777 |
| `b4_delta_gdd_R_tbase5` | 1.09822 | 1.15763 | 0.7911 | 0.7779 |
| `b4_delta_gdd_P` | 1.09829 | 1.15550 | 0.7910 | 0.7783 |
| `b5_delta_weather_P` | 1.10080 | 1.16162 | 0.7905 | 0.7770 |
| `b5_delta_weather_R` | 1.09589 | 1.15410 | 0.7916 | 0.7787 |

### 2.3 按 harvest_year（相对 persistence 的 MSE 降幅）

| 条件 | 2016 | 2017 | 2018 | 2021 | 2022 |
|---|---|---|---|---|---|
| `b2_no_time` | 43.83% | 48.45% | 47.62% | 45.63% | 46.73% |
| `b3_delta` | 45.78% | 51.01% | 49.32% | 47.56% | 48.81% |
| `b4_gdd_R` | 45.63% | 50.94% | 49.18% | 47.47% | 48.67% |
| `b4_delta_gdd_R` | 45.95% | 51.36% | 49.35% | 47.77% | 48.99% |
| `b4_delta_gdd_R_tbase5` | 46.01% | 51.40% | 49.41% | 47.95% | 49.06% |
| `b4_delta_gdd_P` | 45.95% | 51.37% | 49.62% | 48.25% | 49.00% |
| `b5_delta_weather_P` | 45.83% | 51.26% | 49.30% | 47.80% | 48.87% |
| `b5_delta_weather_R` | 45.99% | 51.46% | 49.69% | 48.37% | 49.24% |

### 2.4 年内对照（年份身份不变的天气/GDD 效应）

| 年份 | 对照 | ΔMSE | 95% CI |
|---|---|---|---|
| 2016 | `b3_delta_minus_b2_no_time` | -0.03982 | [-0.04209, -0.03754] |
| 2016 | `b4_delta_gdd_R_minus_b3_delta` | -0.00343 | [-0.00489, -0.00185] |
| 2016 | `b4_delta_gdd_P_minus_b3_delta` | -0.00348 | [-0.00478, -0.00217] |
| 2016 | `b5_delta_weather_P_minus_b3_delta` | -0.00098 | [-0.00220, +0.00019] |
| 2016 | `b5_delta_weather_R_minus_b3_delta` | -0.00433 | [-0.00581, -0.00291] |
| 2016 | `b4_delta_gdd_R_tbase5_minus_b4_delta_gdd_R` | -0.00127 | [-0.00232, -0.00020] |
| 2016 | `b4_delta_gdd_R_minus_b4_delta_gdd_P` | +0.00005 | [-0.00114, +0.00131] |
| 2016 | `b5_delta_weather_R_minus_b5_delta_weather_P` | -0.00335 | [-0.00457, -0.00219] |
| 2017 | `b3_delta_minus_b2_no_time` | -0.05913 | [-0.06283, -0.05563] |
| 2017 | `b4_delta_gdd_R_minus_b3_delta` | -0.00828 | [-0.01085, -0.00576] |
| 2017 | `b4_delta_gdd_P_minus_b3_delta` | -0.00833 | [-0.01085, -0.00587] |
| 2017 | `b5_delta_weather_P_minus_b3_delta` | -0.00593 | [-0.00829, -0.00357] |
| 2017 | `b5_delta_weather_R_minus_b3_delta` | -0.01044 | [-0.01341, -0.00769] |
| 2017 | `b4_delta_gdd_R_tbase5_minus_b4_delta_gdd_R` | -0.00074 | [-0.00230, +0.00085] |
| 2017 | `b4_delta_gdd_R_minus_b4_delta_gdd_P` | +0.00006 | [-0.00213, +0.00219] |
| 2017 | `b5_delta_weather_R_minus_b5_delta_weather_P` | -0.00450 | [-0.00695, -0.00219] |
| 2018 | `b3_delta_minus_b2_no_time` | -0.03852 | [-0.04104, -0.03597] |
| 2018 | `b4_delta_gdd_R_minus_b3_delta` | -0.00067 | [-0.00311, +0.00168] |
| 2018 | `b4_delta_gdd_P_minus_b3_delta` | -0.00676 | [-0.00868, -0.00492] |
| 2018 | `b5_delta_weather_P_minus_b3_delta` | +0.00035 | [-0.00174, +0.00241] |
| 2018 | `b5_delta_weather_R_minus_b3_delta` | -0.00842 | [-0.01072, -0.00617] |
| 2018 | `b4_delta_gdd_R_tbase5_minus_b4_delta_gdd_R` | -0.00145 | [-0.00301, +0.00016] |
| 2018 | `b4_delta_gdd_R_minus_b4_delta_gdd_P` | +0.00609 | [+0.00362, +0.00857] |
| 2018 | `b5_delta_weather_R_minus_b5_delta_weather_P` | -0.00877 | [-0.01081, -0.00689] |
| 2021 | `b3_delta_minus_b2_no_time` | -0.04439 | [-0.04886, -0.03942] |
| 2021 | `b4_delta_gdd_R_minus_b3_delta` | -0.00483 | [-0.00854, -0.00041] |
| 2021 | `b4_delta_gdd_P_minus_b3_delta` | -0.01603 | [-0.02119, -0.01115] |
| 2021 | `b5_delta_weather_P_minus_b3_delta` | -0.00567 | [-0.00915, -0.00181] |
| 2021 | `b5_delta_weather_R_minus_b3_delta` | -0.01864 | [-0.02370, -0.01340] |
| 2021 | `b4_delta_gdd_R_tbase5_minus_b4_delta_gdd_R` | -0.00412 | [-0.00674, -0.00143] |
| 2021 | `b4_delta_gdd_R_minus_b4_delta_gdd_P` | +0.01120 | [+0.00526, +0.01749] |
| 2021 | `b5_delta_weather_R_minus_b5_delta_weather_P` | -0.01297 | [-0.01737, -0.00886] |
| 2022 | `b3_delta_minus_b2_no_time` | -0.04573 | [-0.04979, -0.04080] |
| 2022 | `b4_delta_gdd_R_minus_b3_delta` | -0.00386 | [-0.00792, +0.00011] |
| 2022 | `b4_delta_gdd_P_minus_b3_delta` | -0.00411 | [-0.00747, -0.00100] |
| 2022 | `b5_delta_weather_P_minus_b3_delta` | -0.00133 | [-0.00491, +0.00198] |
| 2022 | `b5_delta_weather_R_minus_b3_delta` | -0.00927 | [-0.01385, -0.00499] |
| 2022 | `b4_delta_gdd_R_tbase5_minus_b4_delta_gdd_R` | -0.00146 | [-0.00375, +0.00074] |
| 2022 | `b4_delta_gdd_R_minus_b4_delta_gdd_P` | +0.00025 | [-0.00310, +0.00421] |
| 2022 | `b5_delta_weather_R_minus_b5_delta_weather_P` | -0.00794 | [-0.01066, -0.00510] |

### 2.5 基因型泛化切分（官方 validation）

官方 split 只留出 plot、不留出年份：每个 validation 年份在训练集中都出现过。但 183 个 validation 基因型中有 44 个从未出现在任何训练 plot 上（6987 对样本）。按该轴切分已有 E1 预测（`tools/s4_generalization_split.py`，不新增前向）。

| 组 | 条件 | MSE | vs persistence [95% CI] |
|---|---|---|---|
| seen_genotype | `b2_no_time` | 1.17613 | 46.62% [46.30%, 46.95%] |
| seen_genotype | `b3_delta` | 1.13015 | 48.70% [48.36%, 49.06%] |
| seen_genotype | `b4_delta_gdd_P` | 1.12438 | 48.97% [48.61%, 49.33%] |
| seen_genotype | `b4_delta_gdd_R` | 1.12654 | 48.87% [48.51%, 49.23%] |
| seen_genotype | `b4_delta_gdd_R_tbase5` | 1.12486 | 48.95% [48.59%, 49.30%] |
| seen_genotype | `b4_gdd_R` | 1.13332 | 48.56% [48.21%, 48.92%] |
| seen_genotype | `b5_delta_weather_P` | 1.12841 | 48.78% [48.43%, 49.14%] |
| seen_genotype | `b5_delta_weather_R` | 1.12183 | 49.08% [48.72%, 49.45%] |
| unseen_genotype | `b2_no_time` | 1.17874 | 46.07% [45.74%, 46.42%] |
| unseen_genotype | `b3_delta` | 1.13515 | 48.06% [47.71%, 48.42%] |
| unseen_genotype | `b4_delta_gdd_P` | 1.12797 | 48.39% [48.03%, 48.76%] |
| unseen_genotype | `b4_delta_gdd_R` | 1.13072 | 48.27% [47.91%, 48.64%] |
| unseen_genotype | `b4_delta_gdd_R_tbase5` | 1.12957 | 48.32% [47.96%, 48.69%] |
| unseen_genotype | `b4_gdd_R` | 1.13719 | 47.97% [47.61%, 48.34%] |
| unseen_genotype | `b5_delta_weather_P` | 1.13250 | 48.19% [47.83%, 48.55%] |
| unseen_genotype | `b5_delta_weather_R` | 1.12679 | 48.45% [48.09%, 48.82%] |

| 组 | 对照 | ΔMSE | 95% CI |
|---|---|---|---|
| seen_genotype | `b5_delta_weather_P_minus_b3_delta` | -0.00172 | [-0.00314, -0.00032] |
| seen_genotype | `b5_delta_weather_R_minus_b3_delta` | -0.00831 | [-0.00998, -0.00661] |
| seen_genotype | `b4_delta_gdd_P_minus_b3_delta` | -0.00576 | [-0.00729, -0.00423] |
| seen_genotype | `b4_delta_gdd_R_minus_b3_delta` | -0.00359 | [-0.00502, -0.00215] |
| seen_genotype | `b3_delta_minus_b2_no_time` | -0.04595 | [-0.04803, -0.04384] |
| unseen_genotype | `b5_delta_weather_P_minus_b3_delta` | -0.00266 | [-0.00407, -0.00130] |
| unseen_genotype | `b5_delta_weather_R_minus_b3_delta` | -0.00836 | [-0.01033, -0.00634] |
| unseen_genotype | `b4_delta_gdd_P_minus_b3_delta` | -0.00717 | [-0.00881, -0.00547] |
| unseen_genotype | `b4_delta_gdd_R_minus_b3_delta` | -0.00444 | [-0.00621, -0.00263] |
| unseen_genotype | `b3_delta_minus_b2_no_time` | -0.04356 | [-0.04579, -0.04143] |

## 3. 条件输入置换诊断（冻结模型，仅推理）

`within_year`：只在与同一 harvest_year 的样本间置换输入；年份身份保留，只破坏日期对齐。`global`：跨年份置换。

| 条件 | 输入 | real MSE | within-year 增幅 (95% CI) | global 增幅 (95% CI) |
|---|---|---|---|---|
| `b3_delta` | delta | 1.13245 | +9.95% [+9.56, +10.16] | +9.98% [+9.76, +10.36] |
| `b4_delta_gdd_R` | gdd | 1.12846 | +14.78% [+14.39, +15.05] | +16.66% [+16.27, +17.05] |
| `b4_delta_gdd_P` | gdd | 1.12603 | +7.04% [+6.81, +7.32] | +7.49% [+7.24, +7.69] |
| `b5_delta_weather_P` | weather | 1.13029 | +14.60% [+14.28, +15.04] | +16.14% [+15.58, +16.31] |
| `b5_delta_weather_R` | weather | 1.12412 | +18.01% [+17.45, +18.34] | +23.29% [+22.73, +23.82] |

within-year 增幅的 CI 来自单次 within-year 置换的 plot 级 bootstrap；两种置换的均值与标准差见 `reports/s4_conditioning_diagnostics.json`。

## 4. E2：未来 CC / PH 表型读出

协议：S3.2 修正协议（inner-dev 选模 → 全 train 重拟合 → validation 单次评价，2,000 次 plot 级 bootstrap）。A = 当前特征；B = 预测未来特征；C = GDD / 天气基线的预测未来特征（`C_environment_predicted`）。

`A_current` 总体 R²：CC 0.79297 / PH 0.94419（由本次运行的 pooled 特征重算，可与 S3.2 的 A 条件对照）。

### primary_lag7

**CC / overall** — 有效样本 8496，独立 plot 380

| 条件 | 角色 | MAE | RMSE | R² | 选中 epoch |
|---|---|---|---|---|---|
| `A_current` | A_current_feature | 0.08343 ± 0.00123 | 0.10876 ± 0.00102 | 0.79297 ± 0.00388 | 43 |
| `B_b2_no_time` | B_predicted | 0.07484 ± 0.00076 | 0.10113 ± 0.00039 | 0.82102 ± 0.00139 | 40 |
| `B_b3_delta` | B_predicted | 0.06907 ± 0.00019 | 0.09364 ± 0.00033 | 0.84654 ± 0.00109 | 106 |
| `B_b4_gdd_R` | C_environment_predicted | 0.06972 ± 0.00009 | 0.09431 ± 0.00011 | 0.84435 ± 0.00038 | 72 |
| `B_b4_delta_gdd_R` | C_environment_predicted | 0.06878 ± 0.00028 | 0.09354 ± 0.00041 | 0.84687 ± 0.00136 | 87 |
| `B_b4_delta_gdd_R_tbase5` | C_environment_predicted | 0.06933 ± 0.00054 | 0.09376 ± 0.00013 | 0.84616 ± 0.00044 | 97 |
| `B_b4_delta_gdd_P` | C_environment_predicted | 0.06902 ± 0.00043 | 0.09348 ± 0.00055 | 0.84704 ± 0.00181 | 87 |
| `B_b5_delta_weather_P` | C_environment_predicted | 0.06860 ± 0.00019 | 0.09315 ± 0.00020 | 0.84815 ± 0.00064 | 123 |
| `B_b5_delta_weather_R` | C_environment_predicted | 0.06741 ± 0.00015 | 0.09143 ± 0.00025 | 0.85370 ± 0.00081 | 106 |

| B − A | ΔMAE | ΔRMSE | ΔR² |
|---|---|---|---|
| `B_b2_no_time` | -0.00858 | -0.00763 | +0.02804 |
| `B_b3_delta` | -0.01436 | -0.01512 | +0.05357 |
| `B_b4_gdd_R` | -0.01371 | -0.01445 | +0.05138 |
| `B_b4_delta_gdd_R` | -0.01465 | -0.01522 | +0.05390 |
| `B_b4_delta_gdd_R_tbase5` | -0.01410 | -0.01500 | +0.05319 |
| `B_b4_delta_gdd_P` | -0.01441 | -0.01527 | +0.05407 |
| `B_b5_delta_weather_P` | -0.01483 | -0.01561 | +0.05518 |
| `B_b5_delta_weather_R` | -0.01602 | -0.01733 | +0.06073 |

**CC / span7** — 有效样本 6542，独立 plot 380

| 条件 | 角色 | MAE | RMSE | R² | 选中 epoch |
|---|---|---|---|---|---|
| `A_current` | A_current_feature | 0.08083 ± 0.00135 | 0.10510 ± 0.00173 | 0.81135 ± 0.00622 | 37 |
| `B_b2_no_time` | B_predicted | 0.07140 ± 0.00069 | 0.09492 ± 0.00027 | 0.84616 ± 0.00088 | 45 |
| `B_b3_delta` | B_predicted | 0.06918 ± 0.00057 | 0.09323 ± 0.00043 | 0.85159 ± 0.00138 | 78 |
| `B_b4_gdd_R` | C_environment_predicted | 0.06887 ± 0.00058 | 0.09266 ± 0.00056 | 0.85340 ± 0.00176 | 78 |
| `B_b4_delta_gdd_R` | C_environment_predicted | 0.06796 ± 0.00083 | 0.09198 ± 0.00103 | 0.85552 ± 0.00325 | 103 |
| `B_b4_delta_gdd_R_tbase5` | C_environment_predicted | 0.06819 ± 0.00071 | 0.09181 ± 0.00055 | 0.85607 ± 0.00173 | 78 |
| `B_b4_delta_gdd_P` | C_environment_predicted | 0.06861 ± 0.00061 | 0.09248 ± 0.00057 | 0.85394 ± 0.00179 | 78 |
| `B_b5_delta_weather_P` | C_environment_predicted | 0.06853 ± 0.00029 | 0.09255 ± 0.00024 | 0.85372 ± 0.00075 | 76 |
| `B_b5_delta_weather_R` | C_environment_predicted | 0.06621 ± 0.00014 | 0.08980 ± 0.00014 | 0.86231 ± 0.00043 | 110 |

| B − A | ΔMAE | ΔRMSE | ΔR² |
|---|---|---|---|
| `B_b2_no_time` | -0.00943 | -0.01018 | +0.03481 |
| `B_b3_delta` | -0.01165 | -0.01187 | +0.04024 |
| `B_b4_gdd_R` | -0.01196 | -0.01244 | +0.04205 |
| `B_b4_delta_gdd_R` | -0.01287 | -0.01312 | +0.04417 |
| `B_b4_delta_gdd_R_tbase5` | -0.01264 | -0.01329 | +0.04473 |
| `B_b4_delta_gdd_P` | -0.01222 | -0.01261 | +0.04259 |
| `B_b5_delta_weather_P` | -0.01230 | -0.01254 | +0.04238 |
| `B_b5_delta_weather_R` | -0.01462 | -0.01530 | +0.05096 |

**CC / span14** — 有效样本 6322，独立 plot 380

| 条件 | 角色 | MAE | RMSE | R² | 选中 epoch |
|---|---|---|---|---|---|
| `A_current` | A_current_feature | 0.08336 ± 0.00005 | 0.10819 ± 0.00008 | 0.80553 ± 0.00029 | 52 |
| `B_b2_no_time` | B_predicted | 0.07251 ± 0.00011 | 0.09707 ± 0.00014 | 0.84345 ± 0.00045 | 111 |
| `B_b3_delta` | B_predicted | 0.07084 ± 0.00038 | 0.09597 ± 0.00031 | 0.84699 ± 0.00098 | 71 |
| `B_b4_gdd_R` | C_environment_predicted | 0.07194 ± 0.00055 | 0.09767 ± 0.00041 | 0.84152 ± 0.00135 | 29 |
| `B_b4_delta_gdd_R` | C_environment_predicted | 0.07078 ± 0.00046 | 0.09629 ± 0.00038 | 0.84598 ± 0.00121 | 61 |
| `B_b4_delta_gdd_R_tbase5` | C_environment_predicted | 0.07246 ± 0.00123 | 0.09797 ± 0.00095 | 0.84052 ± 0.00310 | 31 |
| `B_b4_delta_gdd_P` | C_environment_predicted | 0.06914 ± 0.00013 | 0.09375 ± 0.00016 | 0.85398 ± 0.00049 | 96 |
| `B_b5_delta_weather_P` | C_environment_predicted | 0.07049 ± 0.00020 | 0.09611 ± 0.00016 | 0.84654 ± 0.00050 | 96 |
| `B_b5_delta_weather_R` | C_environment_predicted | 0.06906 ± 0.00055 | 0.09333 ± 0.00037 | 0.85529 ± 0.00114 | 61 |

| B − A | ΔMAE | ΔRMSE | ΔR² |
|---|---|---|---|
| `B_b2_no_time` | -0.01084 | -0.01112 | +0.03792 |
| `B_b3_delta` | -0.01251 | -0.01222 | +0.04145 |
| `B_b4_gdd_R` | -0.01142 | -0.01052 | +0.03599 |
| `B_b4_delta_gdd_R` | -0.01258 | -0.01191 | +0.04045 |
| `B_b4_delta_gdd_R_tbase5` | -0.01089 | -0.01022 | +0.03499 |
| `B_b4_delta_gdd_P` | -0.01422 | -0.01444 | +0.04845 |
| `B_b5_delta_weather_P` | -0.01287 | -0.01208 | +0.04101 |
| `B_b5_delta_weather_R` | -0.01430 | -0.01486 | +0.04976 |

**PH / overall** — 有效样本 6302，独立 plot 380

| 条件 | 角色 | MAE | RMSE | R² | 选中 epoch |
|---|---|---|---|---|---|
| `A_current` | A_current_feature | 0.06183 ± 0.00034 | 0.08373 ± 0.00063 | 0.94419 ± 0.00084 | 51 |
| `B_b2_no_time` | B_predicted | 0.05684 ± 0.00057 | 0.07821 ± 0.00045 | 0.95130 ± 0.00056 | 67 |
| `B_b3_delta` | B_predicted | 0.05449 ± 0.00030 | 0.07483 ± 0.00018 | 0.95542 ± 0.00022 | 117 |
| `B_b4_gdd_R` | C_environment_predicted | 0.05480 ± 0.00056 | 0.07580 ± 0.00096 | 0.95426 ± 0.00116 | 62 |
| `B_b4_delta_gdd_R` | C_environment_predicted | 0.05383 ± 0.00047 | 0.07456 ± 0.00017 | 0.95575 ± 0.00020 | 80 |
| `B_b4_delta_gdd_R_tbase5` | C_environment_predicted | 0.05252 ± 0.00029 | 0.07309 ± 0.00013 | 0.95747 ± 0.00015 | 117 |
| `B_b4_delta_gdd_P` | C_environment_predicted | 0.05501 ± 0.00075 | 0.07542 ± 0.00020 | 0.95471 ± 0.00024 | 85 |
| `B_b5_delta_weather_P` | C_environment_predicted | 0.05270 ± 0.00009 | 0.07347 ± 0.00022 | 0.95703 ± 0.00026 | 97 |
| `B_b5_delta_weather_R` | C_environment_predicted | 0.05262 ± 0.00013 | 0.07299 ± 0.00020 | 0.95759 ± 0.00023 | 122 |

| B − A | ΔMAE | ΔRMSE | ΔR² |
|---|---|---|---|
| `B_b2_no_time` | -0.00498 | -0.00551 | +0.00711 |
| `B_b3_delta` | -0.00734 | -0.00889 | +0.01123 |
| `B_b4_gdd_R` | -0.00702 | -0.00793 | +0.01006 |
| `B_b4_delta_gdd_R` | -0.00800 | -0.00917 | +0.01156 |
| `B_b4_delta_gdd_R_tbase5` | -0.00930 | -0.01064 | +0.01328 |
| `B_b4_delta_gdd_P` | -0.00682 | -0.00830 | +0.01052 |
| `B_b5_delta_weather_P` | -0.00912 | -0.01026 | +0.01284 |
| `B_b5_delta_weather_R` | -0.00921 | -0.01074 | +0.01340 |

**PH / span7** — 有效样本 5046，独立 plot 380

| 条件 | 角色 | MAE | RMSE | R² | 选中 epoch |
|---|---|---|---|---|---|
| `A_current` | A_current_feature | 0.06041 ± 0.00040 | 0.08101 ± 0.00018 | 0.94571 ± 0.00024 | 35 |
| `B_b2_no_time` | B_predicted | 0.05335 ± 0.00005 | 0.07381 ± 0.00003 | 0.95493 ± 0.00003 | 139 |
| `B_b3_delta` | B_predicted | 0.05622 ± 0.00142 | 0.07615 ± 0.00123 | 0.95201 ± 0.00155 | 36 |
| `B_b4_gdd_R` | C_environment_predicted | 0.05320 ± 0.00064 | 0.07368 ± 0.00089 | 0.95509 ± 0.00108 | 58 |
| `B_b4_delta_gdd_R` | C_environment_predicted | 0.05249 ± 0.00051 | 0.07303 ± 0.00072 | 0.95587 ± 0.00087 | 58 |
| `B_b4_delta_gdd_R_tbase5` | C_environment_predicted | 0.05445 ± 0.00148 | 0.07414 ± 0.00097 | 0.95452 ± 0.00119 | 22 |
| `B_b4_delta_gdd_P` | C_environment_predicted | 0.05550 ± 0.00177 | 0.07514 ± 0.00088 | 0.95328 ± 0.00110 | 22 |
| `B_b5_delta_weather_P` | C_environment_predicted | 0.05216 ± 0.00058 | 0.07234 ± 0.00069 | 0.95670 ± 0.00083 | 58 |
| `B_b5_delta_weather_R` | C_environment_predicted | 0.05315 ± 0.00085 | 0.07270 ± 0.00091 | 0.95627 ± 0.00110 | 28 |

| B − A | ΔMAE | ΔRMSE | ΔR² |
|---|---|---|---|
| `B_b2_no_time` | -0.00706 | -0.00720 | +0.00923 |
| `B_b3_delta` | -0.00419 | -0.00486 | +0.00630 |
| `B_b4_gdd_R` | -0.00721 | -0.00734 | +0.00938 |
| `B_b4_delta_gdd_R` | -0.00792 | -0.00798 | +0.01017 |
| `B_b4_delta_gdd_R_tbase5` | -0.00596 | -0.00687 | +0.00882 |
| `B_b4_delta_gdd_P` | -0.00491 | -0.00587 | +0.00758 |
| `B_b5_delta_weather_P` | -0.00825 | -0.00867 | +0.01100 |
| `B_b5_delta_weather_R` | -0.00726 | -0.00831 | +0.01056 |

**PH / span14** — 有效样本 4972，独立 plot 380

| 条件 | 角色 | MAE | RMSE | R² | 选中 epoch |
|---|---|---|---|---|---|
| `A_current` | A_current_feature | 0.06192 ± 0.00020 | 0.08355 ± 0.00045 | 0.94403 ± 0.00060 | 44 |
| `B_b2_no_time` | B_predicted | 0.05646 ± 0.00035 | 0.07818 ± 0.00039 | 0.95099 ± 0.00049 | 76 |
| `B_b3_delta` | B_predicted | 0.05545 ± 0.00016 | 0.07744 ± 0.00026 | 0.95192 ± 0.00032 | 64 |
| `B_b4_gdd_R` | C_environment_predicted | 0.05454 ± 0.00026 | 0.07562 ± 0.00015 | 0.95415 ± 0.00018 | 123 |
| `B_b4_delta_gdd_R` | C_environment_predicted | 0.05482 ± 0.00026 | 0.07648 ± 0.00004 | 0.95310 ± 0.00005 | 64 |
| `B_b4_delta_gdd_R_tbase5` | C_environment_predicted | 0.05322 ± 0.00016 | 0.07438 ± 0.00023 | 0.95563 ± 0.00028 | 81 |
| `B_b4_delta_gdd_P` | C_environment_predicted | 0.05457 ± 0.00029 | 0.07576 ± 0.00013 | 0.95398 ± 0.00016 | 101 |
| `B_b5_delta_weather_P` | C_environment_predicted | 0.05557 ± 0.00108 | 0.07636 ± 0.00040 | 0.95325 ± 0.00049 | 47 |
| `B_b5_delta_weather_R` | C_environment_predicted | 0.05400 ± 0.00027 | 0.07473 ± 0.00026 | 0.95522 ± 0.00031 | 81 |

| B − A | ΔMAE | ΔRMSE | ΔR² |
|---|---|---|---|
| `B_b2_no_time` | -0.00546 | -0.00537 | +0.00696 |
| `B_b3_delta` | -0.00647 | -0.00611 | +0.00789 |
| `B_b4_gdd_R` | -0.00738 | -0.00793 | +0.01012 |
| `B_b4_delta_gdd_R` | -0.00710 | -0.00707 | +0.00908 |
| `B_b4_delta_gdd_R_tbase5` | -0.00870 | -0.00917 | +0.01161 |
| `B_b4_delta_gdd_P` | -0.00735 | -0.00779 | +0.00995 |
| `B_b5_delta_weather_P` | -0.00635 | -0.00719 | +0.00922 |
| `B_b5_delta_weather_R` | -0.00793 | -0.00882 | +0.01119 |

### strict_lag3

**CC / overall** — 有效样本 8112，独立 plot 380

| 条件 | 角色 | MAE | RMSE | R² | 选中 epoch |
|---|---|---|---|---|---|
| `A_current` | A_current_feature | 0.08064 ± 0.00067 | 0.10536 ± 0.00121 | 0.80635 ± 0.00445 | 66 |
| `B_b2_no_time` | B_predicted | 0.07214 ± 0.00083 | 0.09698 ± 0.00076 | 0.83594 ± 0.00258 | 58 |
| `B_b3_delta` | B_predicted | 0.06684 ± 0.00011 | 0.08993 ± 0.00022 | 0.85894 ± 0.00068 | 112 |
| `B_b4_gdd_R` | C_environment_predicted | 0.06691 ± 0.00025 | 0.08992 ± 0.00032 | 0.85896 ± 0.00101 | 92 |
| `B_b4_delta_gdd_R` | C_environment_predicted | 0.06633 ± 0.00020 | 0.08957 ± 0.00023 | 0.86006 ± 0.00073 | 108 |
| `B_b4_delta_gdd_R_tbase5` | C_environment_predicted | 0.06705 ± 0.00049 | 0.09039 ± 0.00076 | 0.85748 ± 0.00238 | 73 |
| `B_b4_delta_gdd_P` | C_environment_predicted | 0.06668 ± 0.00017 | 0.08981 ± 0.00007 | 0.85930 ± 0.00022 | 97 |
| `B_b5_delta_weather_P` | C_environment_predicted | 0.06648 ± 0.00034 | 0.08966 ± 0.00025 | 0.85977 ± 0.00077 | 92 |
| `B_b5_delta_weather_R` | C_environment_predicted | 0.06521 ± 0.00016 | 0.08770 ± 0.00006 | 0.86585 ± 0.00018 | 72 |

| B − A | ΔMAE | ΔRMSE | ΔR² |
|---|---|---|---|
| `B_b2_no_time` | -0.00850 | -0.00838 | +0.02959 |
| `B_b3_delta` | -0.01379 | -0.01543 | +0.05259 |
| `B_b4_gdd_R` | -0.01373 | -0.01544 | +0.05261 |
| `B_b4_delta_gdd_R` | -0.01431 | -0.01579 | +0.05371 |
| `B_b4_delta_gdd_R_tbase5` | -0.01358 | -0.01497 | +0.05113 |
| `B_b4_delta_gdd_P` | -0.01396 | -0.01555 | +0.05295 |
| `B_b5_delta_weather_P` | -0.01416 | -0.01570 | +0.05342 |
| `B_b5_delta_weather_R` | -0.01543 | -0.01766 | +0.05950 |

**PH / overall** — 有效样本 5401，独立 plot 380

| 条件 | 角色 | MAE | RMSE | R² | 选中 epoch |
|---|---|---|---|---|---|
| `A_current` | A_current_feature | 0.06056 ± 0.00117 | 0.08144 ± 0.00167 | 0.94418 ± 0.00231 | 54 |
| `B_b2_no_time` | B_predicted | 0.05589 ± 0.00070 | 0.07602 ± 0.00022 | 0.95137 ± 0.00029 | 66 |
| `B_b3_delta` | B_predicted | 0.05266 ± 0.00013 | 0.07150 ± 0.00015 | 0.95699 ± 0.00019 | 125 |
| `B_b4_gdd_R` | C_environment_predicted | 0.05334 ± 0.00017 | 0.07236 ± 0.00040 | 0.95595 ± 0.00049 | 66 |
| `B_b4_delta_gdd_R` | C_environment_predicted | 0.05235 ± 0.00057 | 0.07144 ± 0.00080 | 0.95705 ± 0.00097 | 62 |
| `B_b4_delta_gdd_R_tbase5` | C_environment_predicted | 0.05215 ± 0.00033 | 0.07109 ± 0.00071 | 0.95748 ± 0.00085 | 66 |
| `B_b4_delta_gdd_P` | C_environment_predicted | 0.05156 ± 0.00007 | 0.07067 ± 0.00008 | 0.95798 ± 0.00009 | 134 |
| `B_b5_delta_weather_P` | C_environment_predicted | 0.05134 ± 0.00050 | 0.07032 ± 0.00079 | 0.95838 ± 0.00094 | 79 |
| `B_b5_delta_weather_R` | C_environment_predicted | 0.05086 ± 0.00028 | 0.06930 ± 0.00028 | 0.95959 ± 0.00033 | 124 |

| B − A | ΔMAE | ΔRMSE | ΔR² |
|---|---|---|---|
| `B_b2_no_time` | -0.00467 | -0.00541 | +0.00720 |
| `B_b3_delta` | -0.00790 | -0.00994 | +0.01281 |
| `B_b4_gdd_R` | -0.00722 | -0.00908 | +0.01177 |
| `B_b4_delta_gdd_R` | -0.00821 | -0.01000 | +0.01287 |
| `B_b4_delta_gdd_R_tbase5` | -0.00842 | -0.01035 | +0.01330 |
| `B_b4_delta_gdd_P` | -0.00900 | -0.01077 | +0.01380 |
| `B_b5_delta_weather_P` | -0.00923 | -0.01111 | +0.01421 |
| `B_b5_delta_weather_R` | -0.00970 | -0.01214 | +0.01542 |

### same_day_lag0

**CC / overall** — 有效样本 7890，独立 plot 380

| 条件 | 角色 | MAE | RMSE | R² | 选中 epoch |
|---|---|---|---|---|---|
| `A_current` | A_current_feature | 0.07972 ± 0.00119 | 0.10392 ± 0.00100 | 0.81061 ± 0.00365 | 32 |
| `B_b2_no_time` | B_predicted | 0.07039 ± 0.00003 | 0.09456 ± 0.00016 | 0.84320 ± 0.00054 | 96 |
| `B_b3_delta` | B_predicted | 0.06779 ± 0.00047 | 0.09072 ± 0.00036 | 0.85569 ± 0.00113 | 32 |
| `B_b4_gdd_R` | C_environment_predicted | 0.06752 ± 0.00090 | 0.09048 ± 0.00096 | 0.85644 ± 0.00306 | 63 |
| `B_b4_delta_gdd_R` | C_environment_predicted | 0.06667 ± 0.00084 | 0.08992 ± 0.00082 | 0.85820 ± 0.00257 | 58 |
| `B_b4_delta_gdd_R_tbase5` | C_environment_predicted | 0.06661 ± 0.00038 | 0.08965 ± 0.00032 | 0.85905 ± 0.00102 | 64 |
| `B_b4_delta_gdd_P` | C_environment_predicted | 0.06614 ± 0.00035 | 0.08899 ± 0.00022 | 0.86115 ± 0.00068 | 75 |
| `B_b5_delta_weather_P` | C_environment_predicted | 0.06748 ± 0.00048 | 0.09067 ± 0.00061 | 0.85583 ± 0.00194 | 28 |
| `B_b5_delta_weather_R` | C_environment_predicted | 0.06505 ± 0.00023 | 0.08700 ± 0.00026 | 0.86728 ± 0.00080 | 77 |

| B − A | ΔMAE | ΔRMSE | ΔR² |
|---|---|---|---|
| `B_b2_no_time` | -0.00933 | -0.00936 | +0.03259 |
| `B_b3_delta` | -0.01193 | -0.01320 | +0.04507 |
| `B_b4_gdd_R` | -0.01220 | -0.01344 | +0.04582 |
| `B_b4_delta_gdd_R` | -0.01305 | -0.01400 | +0.04758 |
| `B_b4_delta_gdd_R_tbase5` | -0.01311 | -0.01427 | +0.04844 |
| `B_b4_delta_gdd_P` | -0.01358 | -0.01493 | +0.05053 |
| `B_b5_delta_weather_P` | -0.01223 | -0.01325 | +0.04521 |
| `B_b5_delta_weather_R` | -0.01467 | -0.01692 | +0.05667 |

**PH / overall** — 有效样本 2908，独立 plot 374

| 条件 | 角色 | MAE | RMSE | R² | 选中 epoch |
|---|---|---|---|---|---|
| `A_current` | A_current_feature | 0.05788 ± 0.00179 | 0.07513 ± 0.00153 | 0.94388 ± 0.00230 | 57 |
| `B_b2_no_time` | B_predicted | 0.05298 ± 0.00017 | 0.06967 ± 0.00011 | 0.95176 ± 0.00015 | 94 |
| `B_b3_delta` | B_predicted | 0.05285 ± 0.00028 | 0.06976 ± 0.00023 | 0.95164 ± 0.00031 | 110 |
| `B_b4_gdd_R` | C_environment_predicted | 0.05371 ± 0.00276 | 0.07024 ± 0.00225 | 0.95091 ± 0.00318 | 57 |
| `B_b4_delta_gdd_R` | C_environment_predicted | 0.05350 ± 0.00076 | 0.07095 ± 0.00094 | 0.94997 ± 0.00132 | 33 |
| `B_b4_delta_gdd_R_tbase5` | C_environment_predicted | 0.05299 ± 0.00076 | 0.07047 ± 0.00135 | 0.95062 ± 0.00190 | 33 |
| `B_b4_delta_gdd_P` | C_environment_predicted | 0.05386 ± 0.00293 | 0.07046 ± 0.00243 | 0.95061 ± 0.00344 | 57 |
| `B_b5_delta_weather_P` | C_environment_predicted | 0.05317 ± 0.00245 | 0.06944 ± 0.00186 | 0.95205 ± 0.00258 | 57 |
| `B_b5_delta_weather_R` | C_environment_predicted | 0.05257 ± 0.00053 | 0.06930 ± 0.00056 | 0.95227 ± 0.00077 | 38 |

| B − A | ΔMAE | ΔRMSE | ΔR² |
|---|---|---|---|
| `B_b2_no_time` | -0.00490 | -0.00546 | +0.00788 |
| `B_b3_delta` | -0.00503 | -0.00537 | +0.00776 |
| `B_b4_gdd_R` | -0.00417 | -0.00489 | +0.00703 |
| `B_b4_delta_gdd_R` | -0.00439 | -0.00418 | +0.00609 |
| `B_b4_delta_gdd_R_tbase5` | -0.00490 | -0.00466 | +0.00675 |
| `B_b4_delta_gdd_P` | -0.00402 | -0.00467 | +0.00673 |
| `B_b5_delta_weather_P` | -0.00471 | -0.00569 | +0.00817 |
| `B_b5_delta_weather_R` | -0.00532 | -0.00584 | +0.00840 |

### 4.1 B − A 的 plot 级 bootstrap（primary_lag7 / overall）

| trait | 条件 | ΔMAE [95% CI] | ΔRMSE [95% CI] | ΔR² [95% CI] |
|---|---|---|---|---|
| cc | `B_b2_no_time` | -0.00822 [-0.00940, -0.00711] | -0.00712 [-0.00860, -0.00562] | +0.02582 [+0.02058, +0.03111] |
| cc | `B_b3_delta` | -0.01291 [-0.01418, -0.01163] | -0.01378 [-0.01555, -0.01202] | +0.04838 [+0.04286, +0.05421] |
| cc | `B_b4_delta_gdd_P` | -0.01312 [-0.01452, -0.01175] | -0.01398 [-0.01581, -0.01215] | +0.04904 [+0.04320, +0.05511] |
| cc | `B_b4_delta_gdd_R` | -0.01323 [-0.01460, -0.01188] | -0.01382 [-0.01575, -0.01200] | +0.04850 [+0.04244, +0.05442] |
| cc | `B_b4_delta_gdd_R_tbase5` | -0.01319 [-0.01458, -0.01186] | -0.01399 [-0.01586, -0.01218] | +0.04907 [+0.04322, +0.05506] |
| cc | `B_b4_gdd_R` | -0.01271 [-0.01402, -0.01141] | -0.01337 [-0.01521, -0.01157] | +0.04702 [+0.04129, +0.05289] |
| cc | `B_b5_delta_weather_P` | -0.01326 [-0.01463, -0.01193] | -0.01407 [-0.01594, -0.01236] | +0.04932 [+0.04361, +0.05526] |
| cc | `B_b5_delta_weather_R` | -0.01455 [-0.01596, -0.01320] | -0.01593 [-0.01796, -0.01392] | +0.05534 [+0.04872, +0.06194] |
| ph | `B_b2_no_time` | -0.00453 [-0.00560, -0.00341] | -0.00491 [-0.00632, -0.00352] | +0.00625 [+0.00446, +0.00800] |
| ph | `B_b3_delta` | -0.00641 [-0.00753, -0.00536] | -0.00774 [-0.00905, -0.00645] | +0.00968 [+0.00808, +0.01132] |
| ph | `B_b4_delta_gdd_P` | -0.00658 [-0.00764, -0.00554] | -0.00768 [-0.00895, -0.00636] | +0.00960 [+0.00795, +0.01114] |
| ph | `B_b4_delta_gdd_R` | -0.00757 [-0.00868, -0.00647] | -0.00856 [-0.00987, -0.00729] | +0.01065 [+0.00903, +0.01229] |
| ph | `B_b4_delta_gdd_R_tbase5` | -0.00833 [-0.00947, -0.00721] | -0.00944 [-0.01087, -0.00808] | +0.01167 [+0.01000, +0.01348] |
| ph | `B_b4_gdd_R` | -0.00680 [-0.00792, -0.00568] | -0.00742 [-0.00879, -0.00614] | +0.00929 [+0.00769, +0.01098] |
| ph | `B_b5_delta_weather_P` | -0.00833 [-0.00946, -0.00723] | -0.00927 [-0.01061, -0.00799] | +0.01147 [+0.00993, +0.01312] |
| ph | `B_b5_delta_weather_R` | -0.00817 [-0.00926, -0.00709] | -0.00950 [-0.01082, -0.00822] | +0.01174 [+0.01014, +0.01340] |

### 4.2 条件间对照（E2 预测特征，plot 级 bootstrap）

与 4.1 的 B−A 不同，这里比较的是两个**预测特征**条件；每个条件的 seed 预测先取平均，再计算 R² 并对 plot 做 bootstrap。因此绝对 R² 与 4.1 的“逐 seed 再取均值”略有差异。

**CC** — n=8496，独立 plot 380

| 对照 | ΔR² [95% CI] | ΔMAE [95% CI] |
|---|---|---|
| `B_b3_delta_minus_B_b2_no_time` | +0.02255 [+0.01809, +0.02715] | -0.00469 [-0.00562, -0.00376] |
| `B_b5_delta_weather_P_minus_B_b3_delta` | +0.00095 [-0.00175, +0.00358] | -0.00035 [-0.00099, +0.00028] |
| `B_b5_delta_weather_R_minus_B_b3_delta` | +0.00697 [+0.00365, +0.01028] | -0.00164 [-0.00238, -0.00090] |
| `B_b4_delta_gdd_P_minus_B_b3_delta` | +0.00066 [-0.00227, +0.00353] | -0.00021 [-0.00092, +0.00051] |
| `B_b4_delta_gdd_R_minus_B_b3_delta` | +0.00012 [-0.00279, +0.00283] | -0.00032 [-0.00098, +0.00040] |
| `B_b4_delta_gdd_R_tbase5_minus_B_b4_delta_gdd_R` | +0.00057 [-0.00160, +0.00272] | +0.00004 [-0.00049, +0.00056] |
| `B_b4_delta_gdd_P_minus_B_b2_no_time` | +0.02322 [+0.01873, +0.02783] | -0.00491 [-0.00596, -0.00388] |
| `B_b5_delta_weather_P_minus_B_b2_no_time` | +0.02350 [+0.01900, +0.02803] | -0.00504 [-0.00602, -0.00407] |
| `B_b5_delta_weather_R_minus_B_b5_delta_weather_P` | +0.00602 [+0.00294, +0.00943] | -0.00129 [-0.00206, -0.00059] |

**PH** — n=6302，独立 plot 380

| 对照 | ΔR² [95% CI] | ΔMAE [95% CI] |
|---|---|---|
| `B_b3_delta_minus_B_b2_no_time` | +0.00343 [+0.00217, +0.00479] | -0.00188 [-0.00272, -0.00107] |
| `B_b5_delta_weather_P_minus_B_b3_delta` | +0.00179 [+0.00081, +0.00276] | -0.00191 [-0.00262, -0.00115] |
| `B_b5_delta_weather_R_minus_B_b3_delta` | +0.00207 [+0.00106, +0.00303] | -0.00176 [-0.00246, -0.00105] |
| `B_b4_delta_gdd_P_minus_B_b3_delta` | -0.00007 [-0.00092, +0.00081] | -0.00017 [-0.00077, +0.00046] |
| `B_b4_delta_gdd_R_minus_B_b3_delta` | +0.00097 [+0.00001, +0.00195] | -0.00116 [-0.00185, -0.00046] |
| `B_b4_delta_gdd_R_tbase5_minus_B_b4_delta_gdd_R` | +0.00103 [+0.00029, +0.00174] | -0.00076 [-0.00129, -0.00024] |
| `B_b4_delta_gdd_P_minus_B_b2_no_time` | +0.00335 [+0.00191, +0.00479] | -0.00204 [-0.00296, -0.00118] |
| `B_b5_delta_weather_P_minus_B_b2_no_time` | +0.00522 [+0.00406, +0.00643] | -0.00379 [-0.00465, -0.00296] |
| `B_b5_delta_weather_R_minus_B_b5_delta_weather_P` | +0.00028 [-0.00063, +0.00118] | +0.00016 [-0.00050, +0.00083] |

### 4.3 共享 epoch 对照（S3.2 第二种读出 regime）

每个条件都使用同一个 epoch（所有 A/B/C 条件与 probe seed 的 inner-dev 选中 epoch 中位数），以排除“B 只是训练更久”的解释。

**CC / overall** — shared epoch 91

| 条件 | R² | ΔR² vs A [95% CI] |
|---|---|---|
| `A_current` | 0.79499 | — |
| `B_b2_no_time` | 0.82421 | +0.02812 [+0.02308, +0.03327] |
| `B_b3_delta` | 0.84597 | +0.04954 [+0.04405, +0.05541] |
| `B_b4_gdd_R` | 0.84512 | +0.04864 [+0.04303, +0.05432] |
| `B_b4_delta_gdd_R` | 0.84607 | +0.04998 [+0.04393, +0.05592] |
| `B_b4_delta_gdd_R_tbase5` | 0.84668 | +0.05033 [+0.04461, +0.05634] |
| `B_b4_delta_gdd_P` | 0.84730 | +0.05113 [+0.04552, +0.05719] |
| `B_b5_delta_weather_P` | 0.84673 | +0.05053 [+0.04513, +0.05638] |
| `B_b5_delta_weather_R` | 0.85169 | +0.05573 [+0.04928, +0.06241] |

**PH / overall** — shared epoch 90

| 条件 | R² | ΔR² vs A [95% CI] |
|---|---|---|
| `A_current` | 0.94670 | — |
| `B_b2_no_time` | 0.95146 | +0.00515 [+0.00342, +0.00695] |
| `B_b3_delta` | 0.95428 | +0.00806 [+0.00639, +0.00979] |
| `B_b4_gdd_R` | 0.95513 | +0.00877 [+0.00707, +0.01053] |
| `B_b4_delta_gdd_R` | 0.95594 | +0.00958 [+0.00781, +0.01140] |
| `B_b4_delta_gdd_R_tbase5` | 0.95611 | +0.00981 [+0.00804, +0.01167] |
| `B_b4_delta_gdd_P` | 0.95505 | +0.00862 [+0.00699, +0.01023] |
| `B_b5_delta_weather_P` | 0.95650 | +0.01019 [+0.00854, +0.01186] |
| `B_b5_delta_weather_R` | 0.95674 | +0.01057 [+0.00890, +0.01239] |

## 5. 验收检查

`15/15` 通过。

| 检查 | 结果 | 说明 |
|---|---|---|
| D1_conditioning_row_alignment | PASS | 71314 npz rows vs 71314 index rows |
| D2_interval_window_full_validity | PASS | valid-days == delta_days for all rows (max delta 16) |
| D3_past_window_full_validity | PASS | every past window has 28 valid days |
| D4_past_window_strictly_before_t_a | PASS | past window ends 2018-01-11 .. 2017-02-21, all < t_a |
| D5_no_official_test_rows | PASS | splits present: ['train', 'validation'] |
| D6_inner_plot_split_disjoint_and_complete | PASS | 800 inner-train / 200 inner-dev plots, disjoint=True, covers all 1000 train plots |
| D7_weather_channels_confirmed_units | PASS | channels=['T_mean_C', 'T_max_C', 'T_min_C', 'RH_mean_pct', 'Rs_sum_MJ_m2_d'] |
| D8_conditioning_sha_recorded | PASS | npz sha256=ce4f71b688a2010f1637c55c6e56878b986cd2ed6474b83ab53bbeebb3743215 |
| M1_checkpoints_present | PASS | 16 expected, 16 found |
| M2_checkpoint_matches_config | PASS | every checkpoint spec equals the config condition |
| M3_b2_has_no_time_input | PASS | only b2_no_time is conditioning-free |
| M4_shared_budget_and_seed_set | PASS | distinct (epochs, batch, rule) tuples: 1; seeds=[0, 1] |
| M5_identical_inner_data_size | PASS | distinct (train, dev) pair counts: {(44925, 11214)} |
| M6_no_validation_selection | PASS | no checkpoint reports validation-based selection |
| M7_same_inner_split | PASS | split seed 20261025 / fraction 0.8 everywhere |

## 6. 天气 / GDD 定义（摘要）

- 天气通道：['T_mean_C', 'T_max_C', 'T_min_C', 'RH_mean_pct', 'Rs_sum_MJ_m2_d']
- 单位：{'T_mean_C': 'degC', 'T_max_C': 'degC', 'T_min_C': 'degC', 'RH_mean_pct': '%', 'Rs_sum_MJ_m2_d': 'MJ m-2 d-1'}
- 排除：{'P_sum_raw': 'unit unresolved (S1); precipitation integrals forbidden by S2', 'VPD_mean_kPa': 'derived channel, opt-in only; not used in S4A'}
- Protocol P：[t_a-W, ..., t_a-1]; strictly before t_a
- Protocol R：interval (t_a, t_b]; real observed weather
- GDD：GDD_d = max(T_mean_d - T_base, 0); T_base in {0, 5} degC
- 结构性限制：S1 verified that all plots of the same harvest_year share one weather series, so weather provides date/year-level environment variation only; there is no plot-level micro-climate signal.

## 7. 产物路径

- `results_json`: `reports/s4_pilot_results.json`
- `e1_predictions`: `reports/s4_e1_val_predictions.parquet`
- `e1_bootstrap`: `reports/s4_e1_bootstrap.json`
- `e2_metrics`: `reports/s4_e2_probe_metrics.json`
- `e2_predictions`: `reports/s4_e2_probe_val_predictions.parquet`
- `e2_pooled_dir`: `external_data/s4_runs/e2_pooled`
- `run_logs`: `reports/s4_run_logs`

完整协议定义见 `reports/s4_data_protocol.md`。
