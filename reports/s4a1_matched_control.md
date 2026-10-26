# S4A.1 参数量匹配的时间条件对照与 S4B 测试数据准备

> 本文件由 `tools/s4a1_write_reports.py` 从真实运行产物生成；数字全部来自 `reports/*.json`，未手工填写。结论由脚本里的判定规则从这些数字重新推导。

## 0. 资产与不可变输入

- 配置：`agro_jepa/config/s4a1_matched_control.yaml`，SHA256 `166c466539ce81b159798c1dc2781511ca3ed38456b930536ba0a47e763da0b2`
- 冻结 S3 索引 SHA256：`f6ef1871655277549aba2d883b6f76a5831b7976af90fc74d80206591d884933`
- 冻结条件张量 SHA256：`ce4f71b688a2010f1637c55c6e56878b986cd2ed6474b83ab53bbeebb3743215`
- 独立验收：**27/27 项通过**（`reports/s4a1_verify.json`）
- 训练 seed 集合：[0, 1, 2]

## 1. 运行清单与参数量

| 条件 | 可训练参数 | 条件参数 | 输入维 | seeds | 实际 epoch | 训练耗时 (s) |
|---|---|---|---|---|---|---|
| `b2_no_time` | 45,898,196 | 0 | 928 | [0, 1, 2] | 16@13, 16@13, 16@15 | 2378, 2634, 4085 |
| `b2_no_time_constant_delta` | 53,882,396 | 72,730 | 1088 | [0, 1, 2] | 16@12, 16@12, 16@12 | 4457, 4222, 2332 |
| `b3_delta` | 53,882,396 | 72,730 | 1088 | [0, 1, 2] | 15@11, 15@11, 15@11 | 2439, 2195, 4317 |

**参数量匹配判定**：B2-C 与 B3 的可训练参数 完全一致（B2-C 53,882,396 vs B3 53,882,396）。

**固定时间间隔**：9.0 d（规则 `inner_train_median`，来源 inner_train rows of this model's split only，n=44925；该训练子集的均值 10.2518、范围 [5.0, 16.0]）。归一化后取值 -0.3406，与 B3 共用同一套 `DeltaNormStats`。（seed0 的 `run_summary.json` 早于该字段的记录版本，此处取自其 checkpoint，二者语义相同。）

**随机初始化可复现性**：每条 run 在 `set_seed(seed)` 后立即 `build_s4_predictor`，再对 state_dict 求 SHA256。`tools/s4a1_init_audit.py` 按同一规则重新构建两次并与记录值比对（`reports/s4a1_init_hashes.json`）。

- 重复构建全部一致（确定性）：**True**
- 有记录值的 run：3 个，全部匹配：**3/3**，不一致 0 个。其余 run 早于该字段的记录版本，其哈希按同一规则重算且确定。

| 条件 | seed0 | seed1 | seed2 |
|---|---|---|---|
| `b2_no_time` | ad5c4c78d84a5297* | 548faec646b27e5e* | cbc4661d0e79c1bd |
| `b2_no_time_constant_delta` | c432b6e5bc1581af* | a229571efed4297e | 265ace5bef56c6f9 |
| `b3_delta` | eaf346fe0220081f* | 70e85406961bcc5c* | ae45772340865b7d* |

（`*` = 该 run 早于字段记录，显示的是按规则重算的确定性哈希；无标记 = 与 run 内记录值一致。）

## 2. 独立验收检查

| 检查 | 结果 | 说明 |
|---|---|---|
| `D1_conditioning_row_alignment` | PASS | 71314 npz rows vs 71314 index rows |
| `D2_interval_window_full_validity` | PASS | valid-days == delta_days for all rows (max delta 16) |
| `D3_past_window_full_validity` | PASS | every past window has 28 valid days |
| `D4_past_window_strictly_before_t_a` | PASS | past window ends 2018-01-11 .. 2017-02-21, all < t_a |
| `D5_no_official_test_rows` | PASS | splits present: ['train', 'validation'] |
| `D6_inner_plot_split_disjoint_and_complete` | PASS | 800 inner-train / 200 inner-dev plots, disjoint=True, covers all 1000 train plots |
| `D7_weather_channels_confirmed_units` | PASS | channels=['T_mean_C', 'T_max_C', 'T_min_C', 'RH_mean_pct', 'Rs_sum_MJ_m2_d'] |
| `D8_conditioning_sha_recorded` | PASS | npz sha256=ce4f71b688a2010f1637c55c6e56878b986cd2ed6474b83ab53bbeebb3743215 |
| `D9_frozen_index_digest` | PASS | index_sha256=f6ef1871655277549aba2d883b6f76a5831b7976af90fc74d80206591d884933 |
| `M1_checkpoints_present` | PASS | 9 expected, 9 found |
| `M2_checkpoint_matches_config` | PASS | every checkpoint spec equals the config condition |
| `M3_b2_no_time_has_no_input` | PASS | only b2_no_time is conditioning-free |
| `M4_control_keeps_delta_encoder` | PASS | the control keeps the full DeltaTimeEncoder and its statistics |
| `M5_fixed_delta_from_inner_train_only` | PASS | fixed interval = 9.0 d, recomputed from this model's inner-train rows |
| `M6_same_delta_normalisation` | PASS | seed0: mean=10.251753, std=3.675335; seed1: mean=10.251753, std=3.675335; seed2: mean=10.251753, std=3.675335 |
| `M7_shared_budget_and_seed_set` | PASS | distinct (epochs, batch, rule) tuples: 1; seeds=[0, 1, 2] |
| `M8_identical_inner_data_size` | PASS | distinct (train, dev) pair counts: {(44925, 11214)} |
| `M9_no_validation_selection` | PASS | no checkpoint reports validation-based selection |
| `M10_same_inner_split` | PASS | split seed 20261025 / fraction 0.8 everywhere |
| `P1_identical_predictor_input_dim` | PASS | B2-C 1088 vs B3 1088 |
| `P2_identical_parameter_counts` | PASS | B2-C 53,882,396 vs B3 53,882,396 |
| `P3_identical_condition_parameters` | PASS | B2-C 72,730 vs B3 72,730 |
| `P4_fixed_interval_stored_in_checkpoint` | PASS | every B2-C checkpoint carries a fixed_delta_days buffer |
| `P5_control_ignores_batch_delta` | PASS | B2-C output identical for delta 7 and 14: True |
| `P6_matched_arm_responds_to_delta` | PASS | B3 output changes when delta changes: True |
| `T1_s4b_audit_all_pass` | PASS | 4/4 S4B checks pass |
| `T5_test_assets_present` | PASS | pair index + conditioning npz exist for every official test split |

## 3. E1：官方 validation 未来潜空间预测

评价集：15,175 对 / 380 个 plot / 10,858 张图；年份 [2016, 2017, 2018, 2021, 2022]。official validation 从不参与选模。

### 3.1 seed 平均预测（与 S4A E1 同口径）

| 条件 | MSE | vs Persistence | vs train-mean | cosine | n |
|---|---|---|---|---|---|
| `b2_no_time` | 1.17370 | 46.53% +46.54% [+46.27%, +46.80%] | 36.06% +36.07% [+35.74%, +36.38%] | 0.7748 | 15,175 |
| `b2_no_time_constant_delta` | 1.17189 | 46.61% +46.62% [+46.35%, +46.88%] | 36.16% +36.17% [+35.84%, +36.48%] | 0.7751 | 15,175 |
| `b3_delta` | 1.12628 | 48.69% +48.70% [+48.43%, +48.96%] | 38.64% +38.65% [+38.32%, +38.96%] | 0.7848 | 15,175 |

### 3.2 每个训练 seed 的指标（训练随机性，独立于 plot bootstrap）

| 条件 | seed0 MSE | seed1 MSE | seed2 MSE | 跨 seed 均值±SD |
|---|---|---|---|---|
| `b2_no_time` | 1.18851 | 1.18830 | 1.18876 | 1.18852 ± 0.00023 |
| `b2_no_time_constant_delta` | 1.18768 | 1.18734 | 1.18675 | 1.18726 ± 0.00048 |
| `b3_delta` | 1.15098 | 1.14920 | 1.14873 | 1.14964 ± 0.00119 |

### 3.3 配对差值

| 对照 | plot bootstrap ΔMSE（95% CI） | 逐 seed ΔMSE（均值±SD，符号一致） |
|---|---|---|
| `b3_delta_minus_b2_no_time_constant_delta` | -0.04559 [-0.04705, -0.04406] | -0.03762 ± 0.00079，3 seeds，同号=True |
| `b3_delta_minus_b2_no_time` | -0.04740 [-0.04888, -0.04589] | -0.03889 ± 0.00126，3 seeds，同号=True |
| `b2_no_time_constant_delta_minus_b2_no_time` | -0.00181 [-0.00245, -0.00119] | -0.00126 ± 0.00065，3 seeds，同号=True |

### 3.4 分层（seed 平均预测）

| 条件 | span7 MSE | span14 MSE | span7 cos | span14 cos |
|---|---|---|---|---|
| `b2_no_time` | 1.14529 | 1.20387 | 0.7811 | 0.7680 |
| `b2_no_time_constant_delta` | 1.14237 | 1.20325 | 0.7818 | 0.7681 |
| `b3_delta` | 1.09866 | 1.15563 | 0.7909 | 0.7783 |

| 条件 | 2016 | 2017 | 2018 | 2021 | 2022 |
|---|---|---|---|---|---|
| `b2_no_time` | 43.97% | 48.62% | 47.79% | 45.83% | 46.89% |
| `b2_no_time_constant_delta` | 44.04% | 48.72% | 47.89% | 45.86% | 47.00% |
| `b3_delta` | 46.03% | 51.33% | 49.61% | 47.81% | 49.06% |

### 3.5 分层配对差值（plot bootstrap 95% CI）

**by_span**

| 层 | b3_delta_minus_b2_no_time_constant_delta | b3_delta_minus_b2_no_time | b2_no_time_constant_delta_minus_b2_no_time |
|---|---|---|---|
| `span14` (n=7,358) | -0.04759 [-0.04976, -0.04542] | -0.04822 [-0.05046, -0.04601] | -0.00063 [-0.00151, +0.00023] |
| `span7` (n=7,817) | -0.04370 [-0.04537, -0.04201] | -0.04663 [-0.04827, -0.04491] | -0.00292 [-0.00366, -0.00214] |

**by_year**

| 层 | b3_delta_minus_b2_no_time_constant_delta | b3_delta_minus_b2_no_time | b2_no_time_constant_delta_minus_b2_no_time |
|---|---|---|---|
| `2016` (n=5,201) | -0.04074 [-0.04303, -0.03843] | -0.04204 [-0.04424, -0.03983] | -0.00130 [-0.00211, -0.00044] |
| `2017` (n=3,485) | -0.06038 [-0.06357, -0.05721] | -0.06263 [-0.06614, -0.05929] | -0.00225 [-0.00366, -0.00074] |
| `2018` (n=3,831) | -0.03908 [-0.04140, -0.03664] | -0.04125 [-0.04352, -0.03900] | -0.00218 [-0.00371, -0.00069] |
| `2021` (n=1,131) | -0.04472 [-0.04866, -0.04084] | -0.04542 [-0.04930, -0.04132] | -0.00071 [-0.00249, +0.00144] |
| `2022` (n=1,527) | -0.04527 [-0.04939, -0.04049] | -0.04777 [-0.05207, -0.04304] | -0.00250 [-0.00473, -0.00038] |

## 4. 判定（由脚本按预注册规则从上面的数字推导）

**Q1/Q2（E1）**：B3 优于 B2-C：plot bootstrap 的 95% CI 完全低于 0，且每个训练 seed 的配对 ΔMSE 都为负。样本级 Δt 在移除全局时间先验后仍有稳定增益。

- `b3_delta − b2_no_time_constant_delta`：plot bootstrap -0.04559 [-0.04705, -0.04406]；逐 seed -0.03762 ± 0.00079。
- `b2_no_time_constant_delta − b2_no_time`：plot bootstrap -0.00181 [-0.00245, -0.00119]。 该项衡量“同宽 + 常量时间通道”相对“完全无时间通道”的作用；若其 CI 含 0，说明 S4A 的 B3−B2 增益主要来自样本级时间信息，而不是通道变宽本身；若显著为负，则 B3−B2 中混有宽度/参数量效应。

## 5. E2：CC / PH 表型读出（辅助证据）

协议：S3.2 修正协议（inner-dev 选模 → 全 official-train 重拟合 → official validation 单次评价 → 2,000 次 plot 级 bootstrap）。有效样本：CC 8,496 / 380 plots；PH 6,302 / 380 plots。

### CC（primary_lag7 / overall）

| 条件 | MAE | RMSE | R² | 选中 epoch |
|---|---|---|---|---|
| `A_current` | 0.08343 ± 0.00123 | 0.10876 ± 0.00102 | 0.79297 ± 0.00388 | 43 |
| `B_b2_no_time` | 0.07468 ± 0.00097 | 0.10097 ± 0.00044 | 0.82156 ± 0.00155 | 40 |
| `B_b2_no_time_constant_delta` | 0.07424 ± 0.00115 | 0.10034 ± 0.00067 | 0.82379 ± 0.00236 | 40 |
| `B_b3_delta` | 0.06872 ± 0.00018 | 0.09332 ± 0.00021 | 0.84757 ± 0.00069 | 123 |

| 条件 − A | ΔMAE | ΔRMSE | ΔR² |
|---|---|---|---|
| `B_b2_no_time` | -0.00875 | -0.00778 | +0.02859 |
| `B_b2_no_time_constant_delta` | -0.00919 | -0.00842 | +0.03082 |
| `B_b3_delta` | -0.01471 | -0.01543 | +0.05460 |

| 配对对照 | ΔR²（plot bootstrap 95% CI） |
|---|---|
| `B_b3_delta_minus_B_b2_no_time_constant_delta` | +0.01898 [+0.01514, +0.02290] |
| `B_b3_delta_minus_B_b2_no_time` | +0.02220 [+0.01830, +0.02637] |
| `B_b2_no_time_constant_delta_minus_B_b2_no_time` | +0.00322 [+0.00099, +0.00539] |

### PH（primary_lag7 / overall）

| 条件 | MAE | RMSE | R² | 选中 epoch |
|---|---|---|---|---|
| `A_current` | 0.06183 ± 0.00034 | 0.08373 ± 0.00063 | 0.94419 ± 0.00084 | 51 |
| `B_b2_no_time` | 0.05601 ± 0.00058 | 0.07758 ± 0.00041 | 0.95209 ± 0.00050 | 67 |
| `B_b2_no_time_constant_delta` | 0.05607 ± 0.00037 | 0.07750 ± 0.00030 | 0.95218 ± 0.00037 | 67 |
| `B_b3_delta` | 0.05361 ± 0.00037 | 0.07408 ± 0.00022 | 0.95631 ± 0.00026 | 117 |

| 条件 − A | ΔMAE | ΔRMSE | ΔR² |
|---|---|---|---|
| `B_b2_no_time` | -0.00582 | -0.00615 | +0.00790 |
| `B_b2_no_time_constant_delta` | -0.00576 | -0.00622 | +0.00799 |
| `B_b3_delta` | -0.00821 | -0.00964 | +0.01212 |

| 配对对照 | ΔR²（plot bootstrap 95% CI） |
|---|---|
| `B_b3_delta_minus_B_b2_no_time_constant_delta` | +0.00340 [+0.00225, +0.00457] |
| `B_b3_delta_minus_B_b2_no_time` | +0.00350 [+0.00223, +0.00476] |
| `B_b2_no_time_constant_delta_minus_B_b2_no_time` | +0.00010 [-0.00082, +0.00107] |

**Q3（E2 是否与 E1 一致）**：判定规则为“B3 − B2-C 的 ΔR² plot bootstrap 95% CI 是否完全在 0 的一侧，且方向与 E1 相同”。

- CC：ΔR² +0.01898 [+0.01514, +0.02290] → B3 更好
- PH：ΔR² +0.00340 [+0.00225, +0.00457] → B3 更好

E2 只作辅助判断：即使 E1 显示样本级 Δt 有增益，也不能据此断言未来表型预测必然改善；反之亦然。

## 5b. 八个验收问题的直接回答

- **Q1**：是。B2-C 与 B3 的可训练参数均为 53,882,396，条件参数均为 72,730，predictor 输入维均为 1088；B2-C 保留完整 DeltaTimeEncoder，只把样本级间隔替换为固定值。
- **Q2**：B3 优于 B2-C：plot bootstrap 的 95% CI 完全低于 0，且每个训练 seed 的配对 ΔMSE 都为负。样本级 Δt 在移除全局时间先验后仍有稳定增益。
- **Q3**：E2 与 E1 方向一致（两者都指向 B3 更好）。（CC ΔR² +0.01898 [+0.01514, +0.02290]；PH ΔR² +0.00340 [+0.00225, +0.00457]）
- **Q4**：3 个配对 seed 的逐 seed 配对 ΔMSE 为 -0.03762 ± 0.00079，范围 [-0.03814, -0.03671]，全部同号=True。在 seed 集合内结论方向不因训练随机性翻转。
- **Q5**：是。已按论文定义逐一核到本地官方划分 4 个 test split（test_plot, test_genotype, test_environment, test_genotype_environment），共 30,357 对预注册配对。
- **Q6**：是。每个配对都继承其 plot 的官方 split，配对两端同一 plot，各 test split 的配对集合互不相交，且与冻结 S3 索引零重叠。
- **Q7**：天气：是，四个 test split 的区间窗口与过去窗口全部覆盖且 GDD 可计算。视觉特征：否，当前就绪 0 对（测试图像尚未下载与编码）。
- **Q8**：视觉特征缓存为 0、test 标签尚未按预注册流程附加、test_environment 只有单一未见环境（2019）且含 5 个训练未见基因型。这三项必须在正式 S4B 训练/评价前解决，本阶段不越界执行。

## 6. 不可声称的结论

1. 不得把本阶段的 B2-C 称为“完全不包含时间先验的模型”：它保留完整的 `DeltaTimeEncoder` 与一个由训练子集决定的全局固定间隔，只是没有样本级时间信息。
2. 不得把 plot bootstrap 置信区间当作训练随机性置信区间；训练随机性只看§3.2/§3.3 的逐 seed 结果。
3. 不得把本阶段结果外推到 official test 或 2,930-plot 全量训练。
4. 不得声称跨年份/跨环境泛化：validation 的年份全部在训练集中出现过。

## 7. 复现命令

```bash
# 1) 新增 seed（seed 0/1 复用 S4A 的冻结 checkpoint，不重写）
CUDA_VISIBLE_DEVICES=1 python3 agro_jepa/train/train_s4.py \
  --config agro_jepa/config/s4a1_matched_control.yaml \
  --condition b2_no_time_constant_delta --seed 0
# 2) E1 / E2
CUDA_VISIBLE_DEVICES=1 python3 tools/s4a1_e1_eval.py --config agro_jepa/config/s4a1_matched_control.yaml
CUDA_VISIBLE_DEVICES=1 python3 tools/s4a1_e2_eval.py --config agro_jepa/config/s4a1_matched_control.yaml
# 3) 验收与报告
python3 tools/s4a1_verify.py --config agro_jepa/config/s4a1_matched_control.yaml
python3 tools/s4a1_write_reports.py --config agro_jepa/config/s4a1_matched_control.yaml
```
