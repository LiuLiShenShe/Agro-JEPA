# S4B 官方测试数据准备核查

> 本文件由 `tools/s4a1_write_reports.py` 从 `reports/s4b_test_data_audit.json` 与 `reports/s4b_test_conditioning.json` 生成。本阶段只做数据核查与配对准备，不评价任何 official test 指标。

## 1. 官方 split 定义（来源与本地核对）

- 论文：https://doi.org/10.1093/gigascience/giaf051
- 原文：Test (P): Unseen plots with seen genotypes and seen environments (all years except 2019); Test (G): Unseen genotypes with seen environments; Test (E): Unseen environment (2019) with seen genotypes; Test (G and E): Unseen genotypes and unseen environment (2019). Splitting uses the R package STPGA with the D-opt algorithm.
- 本地 6 个 split 与论文 4 个 test 集一一对应（见下表 paper_label）。

## 2. 各 split 覆盖

| split | 角色 | 论文 | plot | 图像 | 日期范围 | harvest_year | 基因型(总/训练见过) | yearsite(未见) | 天气年份缺失 | 已缓存特征 |
|---|---|---|---|---|---|---|---|---|---|---|
| `train` | official training split | — | 2930 | 117,578 | 2015-11-09 … 2022-07-18 | [2016, 2017, 2018, 2021, 2022] | 849/849 | 0 | — | 37,670 |
| `validation` | official model-selection split | — | 380 | 11,388 | 2015-11-09 … 2022-07-18 | [2016, 2017, 2018, 2021, 2022] | 183/162 | 0 | — | 11,388 |
| `test_plot` | official test | Test (P) | 250 | 7,381 | 2015-11-09 … 2022-07-18 | [2016, 2017, 2018, 2021, 2022] | 164/164 | 0 | — | 0 |
| `test_genotype` | official test | Test (G) | 246 | 8,441 | 2015-11-09 … 2022-07-18 | [2016, 2017, 2018, 2021, 2022] | 30/0 | 0 | — | 0 |
| `test_environment` | official test | Test (E) | 190 | 4,701 | 2018-10-31 … 2019-07-18 | [2019] | 48/43 | 2 | — | 0 |
| `test_genotype_environment` | official test | Test (G and E) | 62 | 1,661 | 2018-10-31 … 2019-07-18 | [2019] | 30/0 | 2 | — | 0 |

## 3. 测试配对索引（冻结 span7/span14 规则）

| split | 配对数 | 有配对 plot | span7 | span14 | 唯一图像 | delta 范围/中位 | 特征就绪 | 区间天气就绪 | 过去天气就绪 |
|---|---|---|---|---|---|---|---|---|---|
| `test_plot` | 9851 | 250 | 5067 | 4784 | 7050 | 5–16 / 9.0 | 0 | 9851 | 9851 |
| `test_genotype` | 12059 | 246 | 6191 | 5868 | 8149 | 5–16 / 9.0 | 0 | 12059 | 12059 |
| `test_environment` | 6300 | 190 | 3356 | 2944 | 4281 | 5–16 / 8.0 | 0 | 6300 | 6300 |
| `test_genotype_environment` | 2147 | 62 | 1135 | 1012 | 1479 | 5–16 / 8.0 | 0 | 2147 | 2147 |

每个 split 的配对索引文件与 SHA256：

- `external_data/s4b_index/test_plot_test_pairs.parquet` — `76ef410784562e0b744ceef0e3af2c8345cee4902202e878ff8586f7853bafb8`
- `external_data/s4b_index/test_genotype_test_pairs.parquet` — `f260423ff94f95bf9d43e77ae0bb94284a5c78871ea3a0d5b3c84576dda6cf5e`
- `external_data/s4b_index/test_environment_test_pairs.parquet` — `66dcd5c4f531dab0d1f82559cf68661df491e2b926362a5e4d44b7e75bbf11b2`
- `external_data/s4b_index/test_genotype_environment_test_pairs.parquet` — `0df8e0cfa59768a1a7c31f62a032e23908da34595c4732382f6eccae9ad97373`

## 4. 核查检查

4/4 通过。

| 检查 | 结果 | 说明 |
|---|---|---|
| `T1_pairs_inherit_official_split` | PASS | every test pair's plot keeps its official split attribution |
| `T2_both_images_same_plot` | PASS | image stems are prefixed with their own plot_uid |
| `T3_test_pair_sets_disjoint` | PASS | no pair_id appears in two official test splits |
| `T4_no_overlap_with_s3_index` | PASS | 0 test pair_ids also occur in the frozen S3 index |

## 5. 天气与 GDD 覆盖

- 通道：['T_mean_C', 'T_max_C', 'T_min_C', 'RH_mean_pct', 'Rs_sum_MJ_m2_d']，单位 {'T_mean_C': 'degC', 'T_max_C': 'degC', 'T_min_C': 'degC', 'RH_mean_pct': '%', 'Rs_sum_MJ_m2_d': 'MJ m-2 d-1'}
- 排除：['P_sum_raw', 'VPD_mean_kPa']（P_sum_raw 单位未定，禁止做降水积分）
- 逐日天气可用年份：[2016, 2017, 2018, 2019, 2021, 2022]

| harvest_year | 天数 | 覆盖区间 |
|---|---|---|
| 2016 | 366 | 2015-09-01 … 2016-08-31 |
| 2017 | 365 | 2016-09-01 … 2017-08-31 |
| 2018 | 365 | 2017-09-01 … 2018-08-31 |
| 2019 | 365 | 2018-09-01 … 2019-08-31 |
| 2021 | 365 | 2020-09-01 … 2021-08-31 |
| 2022 | 365 | 2021-09-01 … 2022-08-31 |

每个 harvest_year 的天气序列覆盖 9 月 1 日至次年 8 月 31 日；2016 与 2021 为闰年（366 天）。四个 test split 的全部 t_a / t_b 日期都落在覆盖区间内，因此天气/GDD 输入完备。

- Protocol P（可部署）：`[t_a-28, ..., t_a-1] (strictly before t_a)`
- Protocol R（回顾性/oracle）：`(t_a, t_b]`，不可写成预报能力
- GDD：GDD_d = max(T_mean_d - T_base, 0), T_base in {0,5} degC

| split | 条件配对数 | 区间窗口全覆盖 | 过去窗口全覆盖 | GDD_R(Tb0) 均值 |
|---|---|---|---|---|
| `test_plot` | 9851 | True | True | 115.34 |
| `test_genotype` | 12059 | True | True | 119.17 |
| `test_environment` | 6300 | True | True | 78.18 |
| `test_genotype_environment` | 2147 | True | True | 83.77 |

条件张量文件（只含输入，不含表型标签）：

- `external_data/s4b_conditioning/test_plot_conditioning.npz` — `9744ec89103343c4d812f21137586aa0f22dd919d1933fc74fa60b39508606ad`
- `external_data/s4b_conditioning/test_genotype_conditioning.npz` — `b7fc22f79666a1df01701289abf0458d6204e365332b6fb3fcdb4d12b6cf5468`
- `external_data/s4b_conditioning/test_environment_conditioning.npz` — `19db51177498cb511898148f7ca92148defb148845ca8679095f30dbeb46a76a`
- `external_data/s4b_conditioning/test_genotype_environment_conditioning.npz` — `21d6197e2682eda56439e602759edf1a591c1c73acde5968df968fa4228db62d`

## 6. 仍然阻止正式 S4B 评价的问题

1. **视觉特征覆盖为 0**：四个 official test split 的全部图像都还没有下载与 DINOv2 编码；必须先按 S4B 的下载+编码流程生成特征缓存，才能对冻结模型做 test 评价。
2. **test 标签未读取**：本阶段刻意不读任何 test 表型列；正式评价前需要一次单独、预注册的标签附加步骤（只附加、不选模）。
3. **环境外推**：2019 是训练年份完全未见的环境，且天气是年份级序列，test_environment 的独立环境数只有 1；任何 test_environment 结论都必须按“单环境”表述。
4. **test_environment 的基因型并非全部见过**：5 个基因型在训练集中从未出现（论文定义为 seen genotypes，实际有例外），报告时必须如实说明。
5. **B2-C 的固定间隔依赖训练子集**：S4B 的正式评价必须使用 checkpoint 内记录的固定值与归一化统计，不能重新拟合。

## 7. 复现命令

```bash
python3 tools/s4b_prepare_test_data.py --config agro_jepa/config/s4a1_matched_control.yaml
python3 tools/s4b_prepare_test_conditioning.py --config agro_jepa/config/s4a1_matched_control.yaml
```
