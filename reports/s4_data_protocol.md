# S4 数据与评价协议（S4A Pilot）

本文件是 S4A 的协议卡：定义进入模型的每个变量、它的单位、它的时间窗口、
它是否可在 t_a 时刻获得，以及它在什么意义上**不是**什么。所有数字来自
`tools/s4_prepare_weather.py` 与 `tools/s4_verify.py` 的实际运行产物。

## 1. 冻结资产（只读输入）

| 资产 | 路径 | 标识 |
|---|---|---|
| S3 配对索引 | `external_data/s3_index/fip1_s3_v0_index.parquet` | `f6ef1871655277549aba2d883b6f76a5831b7976af90fc74d80206591d884933` |
| 冻结 DINOv2 特征索引 | `external_data/s3_features/index.parquet` | `mainline224_s1-v1` 缓存协议 |
| 视觉主干 | DINOv2 ViT-B/14 | `7764ea0f912e53c92e82eb78a2a1631e92725fc8` |
| 上游 DINO-WM |  vendored clone | `0a9492fa12044b852ae9e001cc74604b79c8bb0c` |
| 逐日天气表 | `data/manifests/weather_daily_first_plot_per_year.csv` | S1 产物，2,191 行 / 6 个 harvest year |
| V0 checkpoint（历史参考） | `external_data/s3_runs/v0/checkpoints/best.pt` | `c337fca506e6a7e31fb90fca1f9197033992c2f8054287d76ec69b4c37426414` |

S4A 只读这些文件，不写回。配对索引的 71,314 行 = 1,000 个官方 train plot
（56,139 对）+ 380 个官方 validation plot（15,175 对）。官方 test split
从未被读取（`tools/s4_verify.py` D5）。

## 2. 视觉目标

* 输入：224×224 RGB → 冻结 DINOv2 ViT-B/14 → `z_a`，`[B, 256, 768]`。
* 目标：真实未来图像经过同一冻结编码器的 patch tokens `z_b`，`[B, 256, 768]`，
  只在损失函数中作为 stop-gradient 目标出现。
* 所有可训练条件预测同一个目标；E1 报告 future-feature MSE、平均 cosine、
  相对 persistence 与 train-mean 的降幅，并按 pair_type / delta_days /
  harvest_year 分别统计。

## 3. 内部划分（早停与选模）

官方 train plot 内部按 `plot_uid` 分层（strata = `harvest_year`）划分：

* inner-train plot：800（80%）
* inner-dev plot：200（20%）
* 随机种子 `20261025`，与 S3.2 probe 协议完全相同的规则和种子。
  规则：每年内用 `numpy.random.default_rng(seed)` 置换 plot，取
  `round(n_plots * 0.2)` 个进入 inner-dev。

* inner-dev 只用于早停与 epoch 选择；
* 官方 validation 只做最终 Pilot 评价，不参与任何选模；
* 官方 test 在 S4A 完全不出现。

`tools/s4_verify.py` D6 / M7 检查该划分的互斥、完备和跨条件一致性。

## 4. 时间输入

`delta_days` = 配对索引中真实记录的 `date_b - date_a` 天数，S3 冻结值域
5–16 d（span7 与 span14 两个 pair_type，各带 ±2 d 容差）。

* B3：输入 `z_a + delta_days`，编码方式与 V0 相同（正弦频率阶梯 + MLP → 10 维
  action 通道）。
* B2：**网络层面移除时间输入**——预测器输入宽度从 1,088 降到 928，
  而不是推理时把 Δt 设 0。可训练参数量从 53.88 M 降到 45.90 M，差异如实报告。

## 5. GDD（积温）

定义（预先设定，不是按 genotype 拟合的生理参数）：

```
T_mean_d   = 逐日平均温度（S1 已确认单位 degC，channel T_mean_C）
GDD_d      = max(T_mean_d - T_base, 0)
T_base     ∈ {0, 5} degC
单位       : degC·day
```

Protocol R（事后 / oracle 区间条件）：
`GDD_R = Σ_{d ∈ (t_a, t_b]} GDD_d`，即配对索引自带的 S2 冻结天气窗口。

Protocol P（可部署）：
`GDD_P(w) = Σ_{d ∈ [t_a-w, t_a-1]} GDD_d`，w ∈ {7, 14, 28} 天，
所有日期都严格早于 t_a。

实测分布（71,314 对全量）：

| 量 | mean | std | min | max |
|---|---|---|---|---|
| train `GDD_R (T_base=0)` | 119.88 | 77.48 | 3.75 | 314.06 |
| val `GDD_R (T_base=0)` | 116.40 | 76.67 | 3.75 | 334.97 |
| train `GDD_R (T_base=5)` | 74.78 | 62.61 | 0.00 | 236.16 |
| train `GDD_P(28) (T_base=0)` | 277.04 | 149.52 | – | – |

`delta_days` 与 `GDD_R` 的相关系数在 train 为 **0.523**、validation 为
**0.497**。这说明 GDD 与 Δt 高度相关但不等价：B4 单独使用 GDD（不给 Δt）与
B3 单独使用 Δt 构成一组可解释的对照，而不是同一个输入的两种写法。

**没有**使用播种日期、没有使用季节性累积起点：S4A 不编造作物生育期起点，
`GDD_P` 只用明确的过去 7/14/28 天固定窗口。

## 6. 天气向量（GRU）

进入模型的逐日天气通道，及其 S1 已确认的单位与定义：

| 通道 | 单位 | 定义 |
|---|---|---|
| `T_mean_C` | degC | 当日逐时 `temperature_air_200cm` 均值 |
| `T_max_C` | degC | 当日最大值 |
| `T_min_C` | degC | 当日最小值 |
| `RH_mean_pct` | % | 当日逐时 `humidity_air_200cm` 均值 |
| `Rs_sum_MJ_m2_d` | MJ m⁻² d⁻¹ | `Σ hourly irradiance_solar_200cm (W m⁻²) × 3600 / 1e6` |

**排除**的通道及原因：

* `P_sum_raw`：S1 审计结论为「单位未确认（raw sensor unit）」。S2 规则禁止把它
  当作 mm 使用，也禁止构造降水积分。因此 S4A 的天气向量**不含降水**。
* `VPD_mean_kPa`：派生通道，S2 中为 opt-in；S4A 未启用（未参与任何模型）。

Protocol P：过去 28 天 `[t_a-28, t_a-1]`（严格早于 t_a）。
Protocol R：区间 `(t_a, t_b]`（真实发生天气）。

变长序列用 padding mask 处理：被 mask 的步在送入 GRU 前置零，masked mean 与
last-valid 都只统计有效天。窗口内任何一天缺测都会让该天 `mask=False`；
实测 71,314 个区间窗口与 71,314 个过去窗口**全部 100% 有效**
（`tools/s4_verify.py` D2 / D3），因此 S4A 中不存在被误当真实天气的 padding。

## 7. 天气数据的结构性限制

S1 审计发现：同一 harvest_year 内不同 plot 的逐时天气序列完全一致。因此
FIP 天气序列只提供**日期 / 年份层面的环境变化**，不构成 plot 级微气候信号。

S4A 的相应措施：

1. 报告每个模型的按 `harvest_year` 性能（E1 `by_year`）；
2. 明确区分「模型学到了年份级差异」与「模型学到了逐日天气细节」：
   Protocol R 的 `GDD_R` 与 Δt 的相关性 0.52 已经说明 GDD 通道携带的是
   区间长度之外的额外信息，但它仍然只是年份内同一序列上的日期函数；
3. 官方 `test_environment`（2019）只留给 S4B 作独立环境测试，S4A 不读取，
   不用它选择模型、窗口或超参数。

## 8. Protocol P 与 Protocol R 的强制区分

| | 输入 | 可部署性 | 用途 |
|---|---|---|---|
| **Protocol P** | t_a 前 7/14/28 天真实天气 + Δt | 是 | 检验未来天气未知时模型能否预测未来潜状态 |
| **Protocol R** | `(t_a, t_b]` 真实天气 | 否 | 分析实际天气条件与视觉状态变化的关系 |

任何使用 Protocol R 的结果在报告里一律标注为
`retrospective / oracle interval conditioning`，不与 Protocol P 合并成单一
「未来预报」指标。

## 9. 训练预算与选模（所有可训练条件一致）

| 项 | 值 |
|---|---|
| 训练 plot | 官方 train 内 800 个 plot，44,925 对 |
| 早停 / 选模 plot | 官方 train 内 200 个 plot，11,214 对 |
| 评价集 | 官方 validation 380 plot / 15,175 对 |
| 候选 epoch 预算 | 16 |
| early stopping patience | 3（按 inner-dev future-latent MSE） |
| 种子集合 | {0, 1}（每个条件都跑） |
| batch size / 优化器 | 128 / AdamW(lr 3e-4, wd 0.05) |
| 学习率计划 | 200 步 warmup + cosine（锚定 16 epoch 预算） |
| AMP | bf16 |
| 归一化统计 | 只用 inner-train 行拟合 |
| 选模指标 | inner-dev future-latent MSE（单一规则） |

官方 validation 在训练期间**不被读取**（`tools/s4_verify.py` M6）。
若某个条件无法在相同预算下收敛，记录实际结果与原因，不修改其预算。

## 10. E1 指标

每个条件（seed 平均后的预测）在官方 validation 上报告：

* future-feature MSE；
* 平均 cosine similarity；
* 相对 Latent Persistence（`z_a`）的 MSE 降幅；
* 相对 train-mean future feature（inner-train 上 `z_b` 的均值）的 MSE 降幅；
* 按 pair_type（span7 / span14）、按 delta_days、按 harvest_year 的分解。

B0 / B1 无需训练，直接由同一批 validation 行计算。

## 11. E2 指标（沿用 S3.2 修正协议）

对 B2–B5 的预测未来潜特征，用 S3.2 的 probe 协议：

* A：当前 DINOv2 特征 `mean(z_a)` → 未来 CC / PH；
* B：各条件预测未来特征 `mean(ẑ_b)` → 未来 CC / PH；
* C：GDD / 天气基线的预测未来特征（B4 / B5 系列）→ 未来 CC / PH。

1. 仅在官方 train 内部（inner-dev）选择 epoch；
2. 选择完成后在全部对应训练 plot 上重拟合；
3. 官方 validation 只做最终评价；
4. 2,000 次 plot 级 cluster bootstrap 报告 B−A 的差值与 95% CI。

指标：CC 与 PH 的 MAE / RMSE / R²，附有效样本数与独立 plot 数，
lag 变体：主 `|lag| ≤ 7` d、严格 `|lag| ≤ 3` d、样本量允许时 `lag = 0`。

E1 与 E2 分别判断：潜空间 MSE 更好**不自动**等价于未来表型预测更好。

## 12. S4A 产出文件

| 文件 | 内容 |
|---|---|
| `external_data/s4_conditioning/s4_pair_conditioning.npz` | 逐对 GDD / 天气张量（P 与 R 两套） |
| `reports/s4_conditioning_summary.json` | 通道、单位、协议定义、实测分布 |
| `external_data/s4_runs/<condition>/seed<k>/checkpoints/best.pt` | 完整 state_dict |
| `reports/s4_pilot_results.json` | E1 + E2 汇总 |
| `reports/s4_e1_val_predictions.parquet` | E1 逐样本行 |
| `reports/s4_e1_bootstrap.json` | E1 plot 级 bootstrap |
| `reports/s4_e2_probe_metrics.json` | E2 读出头指标与 bootstrap |
| `reports/s4_e2_probe_val_predictions.parquet` | E2 逐样本预测 |
| `reports/s4_verify.json` | D1–D8 / M1–M7 验收检查 |
| `reports/s4_run_logs/` | 每次训练与评测的原始日志 |
