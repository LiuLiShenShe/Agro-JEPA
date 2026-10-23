# Agro-JEPA：基于 DINO-WM 的环境驱动作物生长世界模型

> **科研立项、模型设计、可复现实验与 Coding Agent 分阶段执行手册**  
> 版本：2026-10-20 · 研究对象：冬小麦多时相表型预测 · 推荐开发底座：DINO-WM  
> 硬件假定：2 × NVIDIA RTX A6000（各 48 GB），128 GB RAM  
> **本文由原《Agro-JEPA 农业世界模型顶刊立项与可复现实验启动方案》改写。** 保留原方案的研究问题、三项核心设计、四类主实验、主要消融、论文结构和图表规划；将“从零搭建网络”调整为“基于 DINO-WM 渐进式修改”，并修订与官方数据字段不符的设定。文中尚未完成的实验用“拟验证”表达，不预填结果。

---

## 0. 一页决策：到底从什么代码开始？

**选用 [DINO-WM 官方仓库](https://github.com/gaoyuezhou/dino_wm) 作为唯一主工程代码底座。**

- **核心论文**：[DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning](https://proceedings.mlr.press/v267/zhou25t.html)，ICML 2025。
- **视觉权重来源**：[DINOv2](https://github.com/facebookresearch/dinov2)，使用 `dinov2_vitb14`。
- **JEPA/EMA 机制参考**：[I-JEPA](https://github.com/facebookresearch/ijepa)。
- **动作条件 JEPA 参考**：[V-JEPA 2 / V-JEPA 2-AC](https://github.com/facebookresearch/vjepa2)。
- **农业数据**：[FIP 1.0（Hugging Face）](https://huggingface.co/datasets/mikeboss/FIP1)、[原始论文](https://doi.org/10.1093/gigascience/giaf051)。

### 0.1 选型理由

本项目学习的是一个**环境条件化的生长状态转移函数**：

$$
\hat{z}_{t+\Delta t}=P_\phi\bigl(z_t,\,F_\psi(M_{(t,t+\Delta t]}),\,\Delta t\bigr).
$$

DINO-WM 已经提供了预训练视觉特征编码、动作条件化特征预测、潜空间损失、多步 rollout，以及训练、配置、checkpoint 的主体框架。它比原版 I-JEPA（单图空间掩码表征预测）更接近本项目的动态预测形式，也比直接移植依赖机器人视频的 V-JEPA 2-AC 更轻量。

**必须准确界定：DINO-WM 不是原生的 EMA 双编码器 JEPA。** 原始 DINO-WM 用固定的预训练 DINOv2 特征作为预测目标。本文采取两步路线：先以**冻结 DINOv2 + 特征预测**建立可信的世界模型基线，再在后续版本引入 **EMA Teacher + 可训练投影层/LoRA**。不能把尚未加入的 EMA、Cross-Attention、AdaLN-Zero 称为上游已实现功能。

### 0.2 开发底座 ≠ 论文对比基线

| 角色 | 方法 | 如何使用 |
|---|---|---|
| **工程底座** | **DINO-WM** | Fork/Clone 后修改数据、动作编码、预测器、损失和下游任务 |
| 原始潜空间基线 | Frozen DINOv2 + DINO-WM Predictor | 在 FIP 上适配，固定配置保留不添加农业模块 |
| 强任务基线 | 直接表型回归、Weather-LSTM/GRU、DINOv2 + MLP、GDD 曲线 | 检验世界模型是否超过简单且合理的预测器 |
| 架构参照 | I-JEPA、V-JEPA 2-AC | 借鉴机制；只有确实完成任务适配并同协议训练，才纳入定量性能表 |

### 0.3 上游代码中要动的文件

| DINO-WM 上游路径 | 现在的职责 | 本项目修改方向 |
|---|---|---|
| [`models/dino.py`](https://github.com/gaoyuezhou/dino_wm/blob/main/models/dino.py) | 读取 DINOv2 patch/CLS 表征 | ViT-S/14 → ViT-B/14；后续投影、LoRA、EMA |
| [`models/visual_world_model.py`](https://github.com/gaoyuezhou/dino_wm/blob/main/models/visual_world_model.py) | `encode`、`predict`、`forward`、`rollout`、latent loss | 农业前向接口、生理表型头、目标特征与 rollout |
| [`models/vit.py`](https://github.com/gaoyuezhou/dino_wm/blob/main/models/vit.py) | 带时间掩码的特征 Transformer | 第一步沿用；后续替换/扩展为天气 Cross-Attention + AdaLN-Zero |
| [`datasets/traj_dset.py`](https://github.com/gaoyuezhou/dino_wm/blob/main/datasets/traj_dset.py) | 固定时序窗口的轨迹切片 | 保留原文件；新建 FIP 不规则间隔 sampler |
| [`train.py`](https://github.com/gaoyuezhou/dino_wm/blob/main/train.py) | Hydra、Accelerate、多卡训练及存档 | 扩充农业 Dataset、损失日志、阶段配置 |
| [`conf/train.yaml`](https://github.com/gaoyuezhou/dino_wm/blob/main/conf/train.yaml) | 原训练配置 | 新建 `conf/train_agro.yaml`，不覆盖原版 |

**两个实现细节：** 原始 `conf/encoder/dino.yaml` 默认是 `dinov2_vits14`；原始 `visual_world_model.py` 在 DINO 配置下有依赖解码器尺度的图像 resize 逻辑。目标采用 `224×224` 和 ViT-B/14 时，应让图像真正以 224 输入 DINOv2，得到 `16×16=256` 个 patch token，而不是无意中被转为 196 像素。原来的 `ViTPredictor` 也不是现成的天气 Cross-Attention。

---

## 1. 项目总览与科学问题

### 1.1 基本信息

- **暂定英文题目**：*Agro-JEPA: An Environment-Driven Joint-Embedding Predictive World Model for Crop Growth Dynamics and Phenotypic Forecasting*
- **暂定中文题目**：Agro-JEPA：一种用于作物生长动态推演与表型预测的环境驱动联合嵌入世界模型
- **拟投稿方向**：*Plant Phenomics*、*Computers and Electronics in Agriculture*；具体目标随实验贡献和结果调整。
- **首个研究对象**：FIP 1.0 的冬小麦试验小区图像、环境时间序列与表型。

### 1.2 科学问题

普通静态视觉表型模型拟合 $I_t\rightarrow y_t$，但农事预测真正关心的是：**如果当前作物状态已知，并给定后续一段时间的环境过程，那么它将如何随时间变化？**

$$
S_{t_b}=f\bigl(S_{t_a}, M_{(t_a,t_b]}, t_b-t_a\bigr),\qquad t_b>t_a.
$$

FIP 的图像观测与农学表型不必每天同步：不同图像间隔不同，部分表型缺失，而气象观测更密集。因而需要将**不规则观测间隔、连续环境积分与视觉状态转移**作为整体问题来建模。

### 1.3 三项拟研究贡献

1. **农学环境积分驱动预测**（Agronomic Environment-Driven Transition）：将两次观测期间的气温、辐射、湿度、降水等序列及有效积温、热胁迫等派生量编码为条件动作，使世界模型支持不规则时间间隔的预测。
2. **生理锚定双子空间**（Phenotype-Anchored Latent Factorization）：将潜空间划分为能解码真实表型的生理子空间和其余视觉子空间，探索其对可解释性和长时 rollout 稳定性的收益。
3. **长周期生长推演与气象情景敏感性分析**：评价未来 7/14/28/42 天的表型预测、季中产量预测以及替换天气轨迹后的输出响应。

创新性论证不应写成“首次使用动作条件世界模型”——DINO-WM、V-JEPA 2-AC 已有相关机制。论文真正需要证明的是：**农业环境积分 + 物候时间尺度 + 生理监督能否带来相对于已有动作条件潜空间模型的额外价值，特别是在跨年度与多步预测中。**

### 1.4 三条待检验假设

- **H1 环境有效性**：真实逐日天气及农业积分特征比仅给定 `Δt`、播种后天数（DAS）或日期更能预测跨年度生长变化。
- **H2 生理锚定有效性**：分离并监督生理子空间能降低多步 rollout 的表型误差累积，且不明显削弱视觉特征预测。
- **H3 潜空间预测有效性**：在可比资源预算下，潜空间未来表型预测相对于视觉直接回归/像素帧预测有精度或效率优势。

**关键判别：** 如果天气增强仅优于“没有时间信息”的弱基线，却不能优于 `DAS + Δt` 或 `GDD-only`，则不能声称模型学到了独立于季节时钟的气象响应。

---

## 2. FIP 1.0：数据、字段与实验协议

### 2.1 一手数据事实

论文报告 **4,058 个小麦小区、六个收获年份（2016、2017、2018、2019、2021、2022）和超过 153,000 张对齐图像**。小区是天然的时序实体，最小样本应从**同一 `plot_uid` 的两次或多次观测**构造，而不是随机拼接不同小区图像。

| 类型 | 官方字段或构造方式 | 使用方法 |
|---|---|---|
| 小区与品种 | `plot_uid`, `genotype_id`, `harvest_year`, `yearsite_uid` | 分组、避免图像跨小区混配、构造跨年划分 |
| 图像序列 | `inner_plot_images`, `images`, `image_dates`，另有内区/对齐变换元数据 | 优先使用已对齐 `inner_plot_images`；检查其日期映射 |
| 冠层覆盖度 | `canopy_cover_values`, `canopy_cover_dates` | CC 监督，范围约 0–1 |
| 株高 | `height_values`, `height_dates` | PH 监督，单位 m；与拍照可能不同日 |
| 麦穗数量 | `spike_count_values`, `spike_count_dates` | WHC 监督，仅有观测才作有效标签 |
| 衰老评分 | `senescence_values`, `senescence_dates` | Sen 监督，范围 0–10；仅 2016–2018 有 |
| 产量/蛋白质 | `yield_value`, `protein_value` | 季末目标；2017 年蛋白质没有测量 |
| 天气传感器 | `temperature_air_200cm_*`, `humidity_air_200cm_*`, `irradiance_solar_200cm_*`, `precipitation_200cm_*` | 原始记录按时间聚合为逐日天气 |
| 可选天气 | `temperature_air_10cm_*`, `humidity_air_10cm_*`, `temperature_soil_5cm_*` | 后续环境变量消融 |

这些字段来自 [FIP 官方论文 Table 6](https://doi.org/10.1093/gigascience/giaf051)。**原始数据不是已经预先整理好的 `[T_mean, T_max, T_min, Rs, VPD, P]` 六列日表。** 先从时间戳记录计算每日统计量：平均/最高/最低气温、日辐射累计或平均、日降水总量、相对湿度、VPD；每列需注明聚合方式与单位。

> 原文件将所有年份的 CC、PH、WHC、Sen 视为同样可用。官方 Table 4 证明 Sen 只在 2016–2018 年采集；Table 5 证明 2017 年没有蛋白质标签。**缺测与真实观测到的 0 必须区分**，不要把没有测量的麦穗数或衰老评分硬填成 0。

### 2.2 下载与存放

- [Hugging Face：`mikeboss/FIP1`](https://huggingface.co/datasets/mikeboss/FIP1)。
- [ETH Research Collection 原始数据入口](https://doi.org/20.500.11850/697773)。
- Hugging Face 数据卡说明：该仓库包含已对齐的内区图像；原始整幅图像路径需要另行从 ETH 数据源获取。**先检查实际下载产物和磁盘需求，再决定是否需要原始大图**，本项目 V0 不依赖原始大图。

建议将只读原始数据与程序生成的索引/缓存分开：

```text
Agro-JEPA/
├── third_party/dino_wm/          # 可选：保留上游代码快照/说明
├── configs/                     # 项目新配置
├── data/
│   ├── fip_dataset.py
│   ├── fip_pairs.py
│   ├── weather_daily.py
│   └── split_protocol.py
├── models/
│   ├── vision_encoder.py
│   ├── agronomic_action_encoder.py
│   ├── world_predictor.py
│   ├── phenotype_heads.py
│   └── agro_jepa.py
├── losses/agro_loss.py
├── scripts/
│   ├── audit_fip.py
│   ├── build_fip_index.py
│   ├── train_world_model.py
│   ├── eval_horizons.py
│   ├── eval_early_yield.py
│   └── eval_weather_scenarios.py
├── reports/                     # 有数据、命令、指标、图表的研究记录
└── external_data/               # .gitignore，不把数据/大权重加入 Git
```

上述是目标结构，**从 DINO-WM 的原目录增量调整即可**，不要求第一天重构成完全不同的项目。

### 2.3 数据划分：优先官方协议，补充研究自定义 OOD

**主报告协议**：采用 FIP 发布者提供的训练、验证及不同泛化测试子集（包括未见小区、未见基因型、未见环境、未见基因型与环境）。这便于横向对比。Agent 首先输出官方 split 名称、样本量、年份与基因型重叠统计，然后冻结 manifest。

**补充协议（项目自定义）**：保留原设想的跨年外推实验：

| 角色 | 年份 | 规则 |
|---|---|---|
| Train/Val/ID-Test | 2016、2017、2019、2021 | 年内按 `genotype_id` 划分 80/10/10，固定种子；样本量由实际索引统计 |
| Cross-Year Test | 2018、2022 | 这些年份整年留出；报告不同年份单独结果和合并结果 |

需要注意：**上表的 ID-Test 实际是“同训练年份、未见基因型”测试**，不要把它描述为“已见品种的常规 ID”。此外，2018/2022 是否在当地数据中构成特定程度的干旱/热浪，应根据 FIP 站点真实天气分布给出定量描述；不能仅凭欧洲当年的宏观事件赋予每个小区相同胁迫标签。

归一化统计只在当次协议的训练子集上计算，测试集复用训练统计。官方划分与自定义划分分别生成独立 manifest，不能混用 checkpoint 做对比。

### 2.4 样本定义：图像、天气与表型的严格时间连接

对小区 $i$ 的实际图像日 $t_a<t_b$：

$$
\mathcal{T}_i=\left(I_{i,t_a},\,M_{(t_a,t_b]},\,\Delta t,\,I_{i,t_b},\,y^{dyn}_{i,t_a},m_{i,t_a},\,y^{dyn}_{i,t_b},m_{i,t_b}\right).
$$

- `Δt=(t_b-t_a).days`，首轮采样范围 `3–45` 天；候选必须来自同一 `plot_uid`。
- 天气采用**左开右闭** `(t_a,t_b]` 的每日记录，正好 `Δt` 个日步；训练与评估必须同一约定。
- 表型随各自真实测量日期匹配图像。先报告日期偏移分布，再选匹配窗口（如 `±3d` 初始候选）；有效性掩码按真实测量构造。
- 图像日期与表型日期不重合时，保存 `label_lag_days`；**评估表型时不能声称是当天真实值**，除非同日测量。
- WHC 和 Sen 没有测量的时间点为 `mask=0`；真实测得 `0` 才是数值 `0`。
- 本任务未来天气是真实历史观测，适用于**事后已知天气的条件推演**。当要声称“季中真实可部署预测”时，需要区分历史已知天气、未来天气预报和直接使用事后真实天气的 oracle 情况。

### 2.5 数据阶段产出与完成条件

必须形成 `reports/data_inventory.md`、`reports/weather_schema.md`、`reports/split_statistics.md`、`data/manifests/*.parquet`（或 csv）和几张真实样例可视化，清楚标注每个任务在各年份/划分上的有效样本数。首轮只要能构造真实的 **CC+PH** 配对样本，就可以进入模型阶段；其余标签随后接入。

---

## 3. 网络设计与从 DINO-WM 到 Agro-JEPA 的版本演进

### 3.1 不变的骨架

```text
当前小麦冠层图 I_t ──→ DINOv2 ViT-B/14 ──→ Patch Tokens / 全局状态
                                                 │
逐日环境 M_(t,t+Δt] ──→ Agronomic Action Encoder ──┤
                                                 ▼
                                       Conditioned Predictor
                                                 │
                                                 ▼
                            未来 Patch / Global 潜状态 ẑ_(t+Δt)
                                                 │
                             ┌───────────────────┴─────────────────┐
                             ▼                                     ▼
                     Future Phenotype Head                  下一步 Rollout
                   CC / PH / WHC / Sen                     t_2 → t_3 ...
                                                                   │
                                                                   ▼
                                                         Season-End Head
                                                           Yield / Protein

真实未来图 I_(t+Δt) ──→ 冻结 DINO 特征（早期）/ EMA Teacher（后期）
                                             └─→ Stop-grad target for latent loss
```

### 3.2 严格的版本定义

| 版本 | 做什么 | 与上游相比的最小变化 | 关键对照 |
|---|---|---|---|
| **V0：FIP-DINO-WM** | 固定 DINOv2，单步未来 patch 表征预测 | 农业 Dataset、`Δt` 条件、关闭 decoder、未来表型 probe | 当前图直接预测未来表型、仅时间输入 |
| **V1：Weather-DINO-WM** | 引入逐日天气编码及 GDD 等统计 | 替换机器人动作、可变长天气编码 | `Δt`、DAS、GDD-only、Raw Weather |
| **V2：Agro-JEPA-Anchor** | 引入 `512=64+448` 生理锚定潜空间 | state projector、表型 head、anchor/orth loss | 无生理锚定、无正交项 |
| **V3：Agro-JEPA-Rollout** | 多步自回归、季末 yield/protein | teacher forcing 与自由 rollout、终局 head | 直接回归、不 rollout、逐步预测 |
| **V4：Agro-JEPA-Full** | EMA Teacher、最后四层 LoRA、Cross-Attention + AdaLN | 端到端改造与精调 | V3 固定特征版本、无 EMA、无 LoRA |

> V0–V4 是**增量模型版本**，不代表每阶段都必须使用全部复杂模块。V0 已足以回答：DINO-WM 在 FIP 上能否学到未来潜状态、能否改善表型预测？

### 3.3 张量尺寸及对齐方式

| 张量 | 尺寸（目标） | 说明 |
|---|---|---|
| `I_t` | `[B,3,224,224]` | 已对齐的内区小麦图 |
| `DINO patch tokens` | `[B,256,768]` | ViT-B/14，224/14=16 |
| `Projected patch tokens` | `[B,256,512]`（可选） | V4 可选的 patch 投影；需与 EMA Teacher 目标维度一致 |
| `z_t` | `[B,512]` | V2 起对 768 维 patch 池化后应用可训练 `768→512` 全局投影 |
| `z_phy`, `z_vis` | `[B,64]`, `[B,448]` | 对**全局状态**拆分，不把维度拆分误称自动正交 |
| `M` | `[B,L,C]`, `L≤45` | 按天聚合，可变长，附有效天数 mask |
| `Action tokens` | `[B,8,512]` | 农业动作的固定 token 表示，V1 可先简单实现 |
| `Predicted future patches` | `[B,256,768]`（V0–V3） | 继续匹配冻结 DINOv2 的原始 patch 目标；V4 如改为 512 维须同步改 Teacher 与 Predictor |
| `Predicted global state` | `[B,512]` | 池化未来 patch tokens 后用于下游 |
| `ŷ_dyn` | `[B,4]` | CC、PH、WHC、Sen；按标签有效性计算损失 |
| `ŷ_end` | `[B,2]` | GY、PC；按标签有效性计算损失 |

V0–V3 的**未来 patch 动力学始终保持 768 维**，可直接与冻结 DINOv2 的目标做 latent loss。V2 新增的 `768→512` 仅作用于 `mean/attention-pool(patches)` 后的**全局生理状态**：当前真实 patch 与预测未来 patch 共享这个全局投影/表型头；因此生理监督可以反向影响 Predictor，同时不改变固定 DINO patch 目标。V4 才根据实测收益决定是否把 patch 预测空间也改为 512，并同步引入一致的 EMA Teacher。不要将“global state 是 512 维”误读成“所有 patch token 也已变成 512 维”。

### 3.4 环境动作编码器

从气象原始序列推导：

$$
\operatorname{GDD}_d=\max\bigl(0,\min(T_{mean,d},T_{upper})-T_{base}\bigr),
$$
$$
\Delta\operatorname{GDD}_{a\to b}=\sum_{d=a+1}^{b}\operatorname{GDD}_d.
$$

- 初始候选：`T_base=0°C`、`T_upper=30°C`；通过训练集与农学文献核实后作为固定配置，必要时做敏感性分析。
- 另外计算区间温度统计、热胁迫日数/超阈值积分、累积太阳辐射、累积降雨、VPD 积分、DAS 与 `Δt`。
- V1 初版使用 `DailyWeather → 小型 GRU/1D-CNN → Action Tokens`；V1 先不引入 Mamba，便于在固定预算下说明收益。
- V4 再使用 Cross-Attention 让图像 patch query 读取天气 action tokens，并可用 AdaLN-Zero 提供全局条件。

### 3.5 生理锚定与多步推演

$$
z_t=[z_t^{phy}\Vert z_t^{vis}],\quad d_{phy}=64,\ d_{vis}=448.
$$

- 对 `z_phy` 使用监督 head 预测具有有效标签的动态表型。
- `z_vis` 通过潜空间预测目标学习，其是否携带重复的表型信息由表征分析检验。
- 子空间只靠通道切分并**不**自动具备独立性；用跨 batch 交叉协方差正则化并与“只切分、不加正交项”作消融。

多步 rollout：

$$
\hat z_{t_k}=P_\phi\bigl(\hat z_{t_{k-1}}, a_{(t_{k-1},t_k]}\bigr),\quad k=1,\dots,K.
$$

分别评估**单步直接跨越 28 天**与**4 个约 7 天的多步自由 rollout**。两者的性能、输入条件和梯度路径并不相同，不应混写为一个数值。

### 3.6 Loss 的渐进式引入

**V0（冻结特征目标）：**

$$
\mathcal L_{latent}=\operatorname{SmoothL1}(\hat Z_{t_b},\operatorname{sg}(Z_{t_b}))+\alpha\bigl(1-\cos(\hat Z_{t_b},Z_{t_b})\bigr).
$$

**V2（加入表型监督和正交）：**

$$
\mathcal L=\mathcal L_{latent}+\lambda_{phy}\mathcal L_{phy}+\lambda_{orth}\mathcal L_{orth}+\lambda_{var}\mathcal L_{var}.
$$

$$
\mathcal L_{phy}=\sum_{q\in\{t_a,t_b\}}\frac{\sum_j m_{q,j}\,\ell(\hat y_{q,j},y_{q,j})}{\max(1,\sum_j m_{q,j})}.
$$

- `L_orth` 可采用批内去均值后的两子空间交叉协方差 Frobenius 范数；应控制归一化与 batch size 的影响。
- `L_var` 可采用 VICReg 风格的标准差下界项；它并不等价于完整 VICReg（后者还具有协方差/不变性项）。
- 初始搜索权重可从原方案 `λ_phy=2.0`、`λ_orth=0.1`、`λ_var=0.05` 出发，**不是已证实的最优参数**。
- V4 EMA：`θ_teacher ← τ θ_teacher + (1−τ) θ_student`；Teacher 只作停止梯度目标，需让投影/可训练层也遵循清晰的一致更新策略。

### 3.7 训练配置：实测替代臆测

- 首轮从冻结 DINOv2、输入 224、单步预测、混合精度开始，复用原仓库 Accelerate/Hydra 框架。
- 在 A6000 上用一次真实 batch benchmark 选择 micro-batch、梯度累积与总 batch；`2×64` 不应预先视为一定可用。
- 初始优化器 `AdamW`，Predictor 学习率候选 `3e-4`；后续 LoRA 候选 `1e-4`，配合余弦调度和 warmup。
- 缓存特征适合 V0–V3 **冻结编码器**阶段；V4 若启用 LoRA，不能拿旧缓存代替当前编码器前向更新。
- 初版可测试 15–30 epoch 是否稳定，再延伸到 60 epoch；训练耗时、缓存大小、速度提升全部以实测为准。

---

## 4. Baselines：先证明问题成立，再证明方法有贡献

### 4.1 第一梯队：必须完成

| 方法 | 输入 | 测什么 |
|---|---|---|
| Persistence（保持当前表型） | `y_t` | 最简单未来表型下界 |
| Time/DAS-MLP | `y_t`、DAS、`Δt` | 预测收益是否只是物候时间信息 |
| GDD-Thermal Curve | 初始表型、热时间积分 | 农学时间基线 |
| Weather-GRU / Weather-LSTM | 初始表型、逐日天气 | 环境输入是否不依赖视觉也足够 |
| DINOv2 + Direct MLP | 当前图、`Δt`、天气统计 | 世界模型 vs 直接监督回归 |
| FIP-DINO-WM (V0) | 图像、`Δt` | **核心原始潜空间世界模型基线** |
| Weather-DINO-WM (V1) | 图像、天气、`Δt` | 环境动作编码器增益 |
| Agro-JEPA V2/V3/V4 | 上述基础 + 生理锚定/rollout/EMA | 核心方法贡献 |

**公平比较规则**：同一组小区与时间窗口、同一可见历史信息、同一未来天气可用性、同一归一化和指标。直接预测模型可使用未来气象条件，不要人为削弱以制造世界模型优势。

### 4.2 第二梯队：资源允许后加入

- ResNet50 + Temporal-GRU，或 DINOv2 + Temporal-GRU：传统视觉/时序模型。
- PatchTST：天气时序基线。
- ConvLSTM Future-Frame Predictor：真正训练成**未来帧预测任务**的像素基线。
- VideoMAE / 视频扩散：仅在确实完成等价的**条件未来预测适配**后报告，不直接把无条件视频自编码器当成未来预测模型。
- V-JEPA 2-AC：用作条件潜空间预测的概念与组件参照，若进行定量对比，需明确如何适配低频农业图像及天气条件。

---

## 5. 主实验、评价标准与消融设计

### 实验 E1：多预测跨度的动态表型推演

**输入**：当前可见图像、可见历史表型（如果对应基线允许）、指定预测跨度的气象序列。**输出**：目标日期 CC、PH，数据允许时加 WHC、Sen。

- 目标跨度：7/14/28/42 天。目标日若无图像/实测标签，使用明确的容许窗口并报告 `actual_gap_days` 与标签日期差，不假定所有小区均有严格等距图像。
- 报告 `MAE`、`RMSE`、`R²`、`nRMSE`（明确分母，例如训练集 IQR 或均值）、Pearson `r`；按年份/生长期分组。
- 在官方 test split 和自定义 2018/2022 跨年 test 上分别评价，不混成一个“统一 OOD 成绩”。

**主要判断**：是否超过 `DINOv2+Direct MLP`、`Weather-GRU`、`Time/DAS-MLP`；以及 E1 的提升来自天气内容还是仅来自日期。

### 实验 E2：多步自由 Rollout 与误差传播

对同一时间总跨度，比较：直接跨度预测、Teacher-forced 多步预测、自由自回归多步预测。设置 K=1/2/3/4，在 28–42 天窗口中比较误差随步数增长的曲线。

**主要判断**：生理锚定能否降低 rollout 表型漂移，并给出相对 V1、V2 的变化。

### 实验 E3：季中超前产量和蛋白质预测

- 按相对收获日的可见截止日期定义任务：收获前 60/45/30/15 天；采用图像时只能使用截止日期前已获得的观测。
- 分别比较 `Direct Regression`、`Historical Encoder + Weather`、`Agro-JEPA Rollout → Terminal Head`。
- 对未来天气输入单独报告：**事后真实天气 oracle** vs **截止时可用的季节气候统计/预报方案**。如缺少真实预报历史，至少保留前者并将其准确表述为 oracle 条件实验。
- Yield 为主任务；Protein 由于 2017 无标签，按有效样本独立评估。

**主要判断**：增加 rollout 后，是否比使用相同信息的直接终局回归器更准确。

### 实验 E4：天气情景替换与跨年外推

原设想保留为**气象情景模拟（weather-swap scenario analysis）**：固定相同的小区初始图像，分别输入真实年份气象、另一年份的气象或构造的高温/低降雨情景，观察未来 PH/CC/Sen 等变化。

- 对每个情景报告天气变量改变了多少、模型输出改变多少、与跨年同品种真实观测相比偏差多少。
- 重点做**站点天气年差异与 GDD 基线比较**，而不是仅展示好看的两条分叉曲线。
- 跨年天气替换属于模型反事实敏感性实验；跨年实测表型同时还受播期、土壤、品种与管理影响。**它不能单独证明因果规律**。如果要升级为因果结论，需要更强的识别设计或干预数据。
- Sen 只在 2016–2018 有标签，不能拿 2022 年 Sen 曲线作为真实验证目标。

### 实验 E5：鲁棒性与效率

固定同等输入尺寸、硬件、训练样本和日志统计，测量：`sec/epoch`、峰值显存、推理耗时。进行 ColorJitter、光照扰动、轻微几何变化等测试，并报告干扰前后表型预测的误差差值。

### 必做消融矩阵

| ID | Weather | GDD 先验 | Physiology Anchor | Orth / Var | EMA / LoRA | 目的 |
|---|---|---|---|---|---|---|
| A0 | 无，仅时间 | 无 | 无 | 无 | 无 | 时间时钟对照 |
| A1 | 原始天气 | 无 | 无 | 无 | 无 | 环境序列自身贡献 |
| A2 | 原始天气 | 有 | 无 | 无 | 无 | 农学积分贡献 |
| A3 | 原始天气 | 有 | 有 | 无 | 无 | 生理监督贡献 |
| A4 | 原始天气 | 有 | 有 | 有 | 无 | 解耦/方差项贡献 |
| A5 | 原始天气 | 有 | 有 | 有 | 有 | 完整模型增益 |

额外敏感性：`phy:vis` = `16:496, 32:480, 64:448, 128:384`；预测跨度与自由 rollout 步数；weather shuffled / weather swapped、DAS-only、GDD-only。**要特别检查天气输入是否只是年份/生长季的代理变量**：同一站点同一天的天气对不同小区通常相同，因此在同年内靠天气区分小区是不可行的，必须依赖初始视觉和品种/地块差异；跨年天气收益需在独立年份上报告。

### 统计报告规则

至少使用 3 个训练随机种子报告 mean ± std（建议 42、2025、3407）；若测试小区之间时间窗口高度重叠，统计区间采用**按 `plot_uid` 分组 bootstrap**，不要把每个重叠图像对当作独立实验重复。只用验证集挑选配置，测试集完成最终比较。

---

## 6. 论文与图表设计（结果产生后填入）

### 6.1 论文结构

1. **Introduction**：农业从静态表型提取到环境条件动态预测；不规则时间、环境信息、潜空间生理可解释性问题；界定相对 DINO-WM/V-JEPA 的增量。
2. **Materials and Methods**：FIP 数据结构、独立测试划分、日期/标签对齐、DINO-WM 底座、天气积分 Action Encoder、生理锚定、Rollout、Loss。
3. **Experimental Setup**：统一输入信息、训练预算、baselines、OOD 协议、度量指标。
4. **Results**：单步跨度、自由 rollout、季中产量、天气情景、消融、效率。
5. **Discussion**：何时真正学到天气响应、何时只是季节时钟、天气替换的可解释边界、农业多站点拓展。
6. **Conclusion**：仅总结实验已证实的内容。

### 6.2 六张建议核心图

| 图 | 内容 | 数据来源 |
|---|---|---|
| Fig. 1 | FIP 多年度真实天气分布、时间覆盖、样例小区真实图像序列与标签日期 | 数据审计 + 真实抽样 |
| Fig. 2 | Agro-JEPA 全架构图（DINO-WM 底座与新增模块分色） | 最终实现的代码结构 |
| Fig. 3 | CC/PH/WHC/Sen 轨迹：真实点、直接预测、自由 rollout、基线 | E1/E2，可用标签分别作图 |
| Fig. 4 | 收获前 60/45/30/15 天产量预测曲线与散点 | E3 |
| Fig. 5 | 气象 swap 前后天气过程及模型响应、与可比真实年份曲线 | E4 |
| Fig. 6 | 潜空间与生理变量相关性、不同生长期 embedding 及消融对比 | E2/E5 + 表征分析 |

**Fig. 5 不事先标注“矮化 12 cm”“提前衰老 14 天”，Fig. 3 不事先绘制领先曲线**；所有数值和形状来自真实推理结果。

### 6.3 四张建议核心表

- **Table 1**：不同跨度 E1，按官方划分与跨年测试分组的 CC/PH 主结果；缺测的 WHC/Sen 单列有效样本数与适用年份。
- **Table 2**：季中产量/蛋白质 E3，四个可见截止时间，区分 oracle 未来天气与可部署信息。
- **Table 3**：世界模型/直接回归/像素预测的速度、显存、鲁棒性，在等价任务和预算下比较。
- **Table 4**：A0–A5 + 子空间比例 + rollout horizon 的消融表。

---

## 7. 8 周推进建议与每阶段交付物

| 周次 | 阶段 | 技术目标 | 可进入下一阶段的事实标准 |
|---|---|---|---|
| Week 1 | S0 / S1 | 固化上游 DINO-WM、审计 FIP | 列出真实字段/缺失、可读取真实图像与天气，生成官方 split 清单 |
| Week 2 | S2 | 不规则时间采样和 daily weather | 真实的 `(I_a,weather,I_b)` batch、目标间隔分布和日期匹配记录 |
| Week 3 | S3 / S4 | 冻结 DINO-WM V0、任务强基线 | 跑通真实训练/恢复/单步指标；与 Time/DAS、Direct MLP 比较 |
| Week 4 | S5 | Raw Weather、GDD Action Encoder | 实现 `Δt/DAS/Raw/GDD` 对照，在同一协议输出数值 |
| Week 5 | S6 | 生理锚定与消融 | Anchor/Orth/Var 对照 + 真实未来表型结果 |
| Week 6 | S7 | 多步 rollout、Yield/Protein | 单步与自由 rollout 误差曲线，季中终局预测 |
| Week 7 | S8 | EMA/LoRA、Cross-Attention/AdaLN | 完整模型与 V3 等预算对照，记录实测资源占用 |
| Week 8 | S9 / S10 | 跨年/情景分析、图表与论文 | 真实可复核 4 表 6 图、方法/结论与证据一一对应 |

**进度原则**：每一阶段先出一个小规模真实数据可复现结果，再升级模型；如 V0/V1 对强基线没有实质性改善，优先分析样本定义、未来标签目标与时间条件，而不是立即堆叠更复杂模块。

---

# 8. Coding Agent 执行区：可以逐次复制的 Prompt

使用方式：先把 **P0 通用上下文**发送给 Agent（或作为 `AGENTS.md`/任务背景），然后按 P1 → P10 的顺序单独执行。每次只允许 Agent 完成当前阶段；新阶段从上一阶段的报告和 Git commit 接续。Prompt 中的路径可以按服务器现状调整。

## P0｜通用背景：每次任务可附在开头

```text
你是 Agro-JEPA 项目的代码研究助手。科研目标：在 FIP 1.0 冬小麦多年度图像与环境数据上，从官方 DINO-WM 代码演进出环境驱动潜空间生长预测模型，并检验不规则时间、天气积分、生理表型锚定、多步 rollout 的真实贡献。

主上游：https://github.com/gaoyuezhou/dino_wm
数据：https://huggingface.co/datasets/mikeboss/FIP1
原论文：https://doi.org/10.1093/gigascience/giaf051
参考：https://github.com/facebookresearch/ijepa 和 https://github.com/facebookresearch/vjepa2

务必区分上游已存在的代码与本项目新增代码；不把冻结 DINO 特征预测冒称为原版 I-JEPA EMA。优先沿用 DINO-WM 的训练骨架，增量实现。用真实数据、真实运行命令、实际代码位置和实测结果汇报；不预填论文数字。每完成一个阶段给我：①变更文件；②执行命令；③真实实验产物/指标；④未解决的技术问题；⑤对应 Git commit。不要改动与本项目无关的其他研究仓库或已有数据资产。
```

## P1｜S0：获取 DINO-WM、梳理代码、锁定初始版本

```text
本轮只做基线落地和代码阅读，不引入 Agro-JEPA 新模块。
1. 在独立 Agro-JEPA 工作目录 clone 官方 gaoyuezhou/dino_wm，记录上游 HEAD SHA、分支、许可证、依赖与运行入口，创建新工作分支（例如 agro-s0-upstream）。
2. 逐个阅读 README、environment.yaml、conf/train.yaml、conf/encoder/dino.yaml、conf/predictor/vit.yaml、models/dino.py、models/visual_world_model.py、models/vit.py、datasets/traj_dset.py、train.py；画出真实函数调用链，从 Dataset 到编码、Predictor、latent loss、optimizer、checkpoint、rollout。
3. 给出准确的需修改接口清单：数据字段/形状、num_hist/num_pred、机器人 proprio/action 如何替换、has_decoder=false 的真实行为、ViT-S→ViT-B 后的 shape 修改、224 图像的预处理路径。
4. 只在确认无碍原版可复现的前提下提供一个最小执行命令。不要把原机器人环境的 Mujoco/规划流程作为农业任务的必备依赖。
5. 输出 reports/s0_upstream_audit.md、reports/s0_api_map.md，提交 commit。
验收：我能根据报告知道每个 Agro-JEPA 模块应该接入上游哪一个类和函数；代码版本有精确 commit SHA。
```

## P2｜S1：下载和审计 FIP 1.0，确定真实数据协议

```text
本轮只处理 FIP 数据和数据说明，不训练模型。
1. 阅读 FIP 官方论文的 Data/Methods、Table 4/5/6、官方划分说明与 Hugging Face 数据卡；审计数据目录并获取必要的对齐内区图像和 tabular 数据。
2. 列出实际 DatasetDict split 名称、每 split 小区数量、年份分布、genotype_id 重叠；记录 plot_uid、inner_plot_images、image_dates、CC/PH/WHC/Sen/GY/PC（按官方真实字段名）及 weather_* 日期/时间/value 的存在情况、有效率、单位。
3. 对比论文发表的 Table 4/5：Sen 只在 2016–2018；2017 protein 缺失。对图像日期与 height_dates 的偏移做统计，并标注每个动态任务的真实观测次数。
4. 从真实数据绘制：单一小区多时相图像与时间轴、六年环境变量分布、标签有效样本数图。先使用官方已对齐 inner_plot_images，不额外构建昂贵配准系统。
5. 形成官方 split manifest 与研究补充跨年 split manifest；所有产物标明构造规则，保留原始数据原状。
输出 reports/data_inventory.md、reports/split_statistics.md、reports/label_timing.md、必要的 CSV/Parquet 索引与图片，提交 commit。
验收：能从某个实际 plot_uid 列出真实的图像路径、图像日期、标签测量日期和天气记录，而不是估算/伪造示例。
```

## P3｜S2：实现 daily weather 和可变间隔的世界模型 Dataset

```text
本轮实现农业数据管线，不改网络结构。
1. 基于已审计的 FIP 实际 schema，将温度、相对湿度、辐射和降水的带时间戳原始观测聚合为逐日表，生成 T_mean/T_max/T_min/RH/辐射/降雨；VPD、GDD 是派生量，写出公式与单位。
2. 对每个 plot_uid 的可用图像产生 (t_a,t_b) 配对，Δt 取 3–45 天；天气窗严格对应 (t_a,t_b] 共 Δt 天；支持变长天气 padded batch 和 length mask。
3. 按真实测量时间对齐 CC 与 PH，并保留 label_lag_days、label_mask。WHC、Sen 接口预留，但绝不将“没有测量”写成观测 0。
4. 输出 fip_pairs.py、weather_daily.py、collate_fn 和一个样例可执行脚本：加载来自训练 split 的真实 8 个样本，打印各 tensor shape、plot_uid、真实日期、有效标签与天气统计。
5. 写出数据覆盖表：每个预测跨度候选数、每年有效图像对数、CC/PH 目标标签覆盖率。
验收：真实 batch 至少包含 image_a、image_b、weather_daily、weather_mask、delta_days、phenotype_a/b、phenotype_mask_a/b、plot_uid/date；日期约定与统计报告一致。生成 reports/s2_pairs.md 并提交 commit。
```

## P4｜S3：训练最小可运行 FIP-DINO-WM（V0）

```text
本轮构建最小潜空间世界模型，不加入生理正交、EMA、LoRA 或复杂气象 encoder。
1. 沿用 DINO-WM 的 DINOv2 编码器、ViT Predictor、训练与 checkpoint 框架；视觉骨干使用 dinov2_vitb14 并完全冻结。显式确认 224×224 输入产生 256 个 patch tokens、各通道维度一致。
2. 以 Δt 编码替代机器人动作输入；必要时用与 DINO-WM 兼容的独立 time embedding，而不是原样复用机器人的 proprio 字段。关闭图像 decoder，以未来真实图像 DINO patch 特征为 stop-gradient 训练目标。
3. 先使用少量真实小区完成多轮 loss 下降、checkpoint 保存/恢复与单步预测，再运行完整训练 split 的首个稳定实验；记录 batch、epoch、学习率、wall time 和 GPU 占用。
4. 用冻结的潜空间表征训练同一协议下的 future CC/PH 读出头；同时做不使用 predictor 的直接预测对照。
5. 实现独立预测及评估脚本，输出 7/14 天有效标签的 RMSE/MAE/R²、每个跨度真实样本数，并保存预测对照 CSV。
验收：在 FIP 真实图像上完成可恢复的训练与评估；报告不能只停留在 random tensor/smoke test。提交 reports/s3_v0.md 和 commit。
```

## P5｜S4：建立强基线，检查“只是时间时钟”问题

```text
本轮不添加新颖模块，只建立公平的 FIP 未来表型任务基线。
1. 在与 V0 完全相同的 split / 时间窗口 / 目标标签条件下，实现 Persistence、DAS+Δt MLP、GDD Thermal Curve、Weather-GRU、DINOv2+Direct Regression。
2. 精确对齐模型可见信息：当 V0 有当前图像与未来真实天气，直接回归模型也应有对应信息；不存在该信息时拆开实验。
3. 输出按 Δt=7/14/28/42 天、按年份、按 CC/PH 分组的样本数、MAE、RMSE、R²；将天气和时间输入替换/乱序以判别时间代理效应。
4. 特别报告“只给 Δt”和“给 DAS+Δt”差异，判断 V0 是否真的优于静态表征直接回归。失败也据实记录，定位标签稀疏、输入信息和预测目标等原因。
验收：reports/s4_baselines.md 具有可复现命令、正式基线对照表、误差分布和所用 checkpoint；提交 commit。
```

## P6｜S5：天气动作编码器 + GDD，得到 V1

```text
本轮只验证第一项研究贡献：逐日环境与农学积分的作用。
1. 在 V0 基础上新增 models/agronomic_action_encoder.py，输入 [B,L,C] daily weather + time mask + Δt + 累积 GDD，输出固定长度动作 token 或与原 Predictor 兼容的动作向量。
2. 首先接入最简单的 GRU/1D-CNN 聚合动作条件，保持 V0 predictor、图像 encoder、训练时长尽可能一致。
3. 用配置提供四个严格对照：仅 Δt、DAS+Δt、原始逐日天气、原始逐日天气+GDD/热害积分；可增加天气序列 shuffle 作为信息真实性检验。
4. 在官方 test 与 2018/2022 跨年 test 计算 E1，逐年份报告结果。保留天气派生量计算方式和消融配置。
验收：产出 V1 与 V0 同协议对照表，能够具体说清环境积分是否提高跨年性能，而不只证明模型会接收天气 tensor。提交 reports/s5_weather_action.md 和 commit。
```

## P7｜S6：生理锚定双子空间，得到 V2

```text
本轮验证第二项贡献：生理锚定潜空间，不开展大规模终局产量实验。
1. 保持 V1 的 768 维 patch-to-patch latent 预测与冻结目标不变；新增对 patch 池化后的 `768→512` 全局状态投影，将此全局状态拆成 phy=64 与 vis=448。共享这一投影以从当前真实 patch 和未来预测 patch 得到对应全局表型状态。
2. 用 phy head 预测 CC/PH；WHC、Sen 仅在真实存在标签的样本上计算 mask loss。当前帧与预测未来帧分别建立监督，正确处理无标签样本。
3. 将 V1 latent loss 与 phenotype loss 相结合；另用开关控制 orth cross-covariance loss、VICReg-style variance loss。实现 orth/variance 时考虑 batch size 和无标签情况下的行为。
4. 对照 V1、V2-with-anchor、V2-with-anchor+orth、V2-full；优先评价 7/14/28 天未来真实表型和多步预测的可行性。
5. 提供 latent 标准差/交叉协方差和 phenotype-head 实测结果，明确“通道切分”与“实际可分离表征”不是一回事。
验收：每个新增模块都可通过配置关掉；实验结果可归因于独立改动。输出 reports/s6_anchor.md 并提交 commit。
```

## P8｜S7：多步自由 rollout + 季中产量预测，得到 V3

```text
本轮验证动态世界模型的长期预测能力与季末下游价值。
1. 基于已训练 V2 predictor，实现多段气象条件下的自由潜空间 rollout，接口允许输入初始图像与时间段列表，后续阶段不再读取真实未来图像特征。
2. 对 28/42 天跨度，分别跑一次直接跨距预测、teacher-forced 多步、完全自由 rollout；记录每一步的 latent error、CC/PH 误差及可用样本数。
3. 新增 yield/protein 终局 head；构造收获前 60/45/30/15 天的可见信息截止任务，训练 direct regression 与 rollout→head 两组公平对照。
4. 明确标记未来天气是 history-only、climatology/forecast 还是 hindsight-oracle。若只能使用历史真实天气，按照 oracle 条件结果单独汇报，不能写成可部署预测精度。
5. GY 与 PC 分开做有效标签 mask，2017 PC 不进入 PC 指标。
验收：真实多步轨迹 CSV、step-wise error 曲线、不同截断日期产量对照表和 checkpoints 均可复现。输出 reports/s7_rollout_yield.md 并提交 commit。
```

## P9｜S8：EMA Teacher、LoRA、Cross-Attention，得到 V4

```text
本轮在已跑通的 V3 上引入完整版训练机制，不要破坏 V3 checkpoint 与配置。
1. 添加 Context Encoder / EMA Target Encoder；DINOv2 使用预训练 ViT-B/14，冻结前 8 blocks，在后 4 blocks 的指定 attention projection 注入 r=16 LoRA（具体层路径基于实际模型类确认）。
2. 正确处理 student 与 EMA teacher 的可训练参数/投影层对应关系、stop-gradient、EMA 动量 schedule；明确哪些参数进 optimizer。
3. 扩展预测器为以当前 768 维 patch tokens 为 Query、天气 action tokens 为 Key/Value 的 Cross-Attention（对 Q/K/V 分别作适配投影，支持天气 512 维输入），增加可开关 AdaLN-Zero 调制；保留旧 Predictor 配置用于同预算比较。
4. 先冻结特征验证新的 Cross-Attention 模块，再启用 LoRA/EMA 端到端训练。启用 LoRA 后禁止用过时的预提取特征代替动态图像编码梯度。
5. 对照 V3 固定特征、V3+Cross-Attn、V3+Cross-Attn+EMA/LoRA（具体拆分以可执行性为准），记录训练显存、每 epoch 时间和 CC/PH 与 rollout 结果。
验收：每组有独立 config、精确 checkpoint 与真实评估指标；如果 full model 不如冻结底座，也要客观解释并保留最强版本。提交 reports/s8_full_model.md 和 commit。
```

## P10｜S9：完成跨年、消融与气象情景模拟

```text
本轮只做定量分析与出图，不再随意修改主方法。
1. 在固定的官方与自定义跨年 test manifest 上运行 A0–A5 消融，覆盖 7/14/28/42 天跨度，并将结果按真实有效样本数记录。
2. 进行 2018/2022 逐年结果与真实天气分布分析；必要时分别报告同年未见品种与跨年未见年份，不合并解释。
3. 在共同品种/相近初始阶段上实现固定图像+不同年度天气的 scenario swap，对比真实记录，标注年间播期/管理/位置混杂来源；不要宣称已证明气象因果效应。
4. 制作候选 Fig. 1、3、4、5、6 和 Table 1–4 的草稿，所有绘图脚本从 CSV/Parquet 原始结果读取，不硬编码理想曲线或任意百分比。
5. 使用 3 个随机种子报告 mean±std，并尝试 plot_uid 级 bootstrap 区间；生成每个结果对应的 config/checkpoint/manifest/command 清单。
验收：reports/s9_final_eval.md 包含全部模型对照、强基线和消融；图表可以由脚本重新生成；提交 commit。
```

## P11｜S10：论文写作与 Novelty 复核

```text
本轮基于真实完成的实验结果整理论文，不将尚未获得的结果写成既定结论。
1. 重点深读并横向分析 DINO-WM、I-JEPA、V-JEPA 2-AC，以及最接近的农业时序表型/基于天气的生长预测论文；逐篇整理问题设定、输入模态、未来预测目标、条件化方式、时序跨度、训练目标、数据规模和真正与本工作重叠之处。使用论文正文的方法和实验细节，而不仅是摘要。
2. 对比我们的代码新增项及经过消融证实的收益，形成一段具体的 novelty statement：从通用机器人动作条件潜空间预测，推进到不规则农学天气积分与可监督生理状态的联合生长推演；不得将相同结构换领域直接说成方法首创。
3. 按本文 Introduction/Methods/Results/Discussion/Conclusion 提纲撰写。所有方法描述与最终代码实现保持一致；所有数值、图和表须有实验产物出处。
4. 若关键贡献在 E1/E2/E3 及强基线比较中不成立，提出缩减/替代的论文主张或下一轮针对性的技术修改，而不是制造优势。
输出 reports/related_work_deep_review.md、reports/novelty_claim.md、paper/draft.md、paper/figures_and_tables.md，提交 commit。
验收：每条主要科研结论都能定位到真实实验、可执行配置及原始证据。
```

---

## 9. Agent 阶段交接模板（每轮完成后填）

```markdown
# Agro-JEPA Stage <编号> 交接
- 当前分支与 commit：
- 上游 DINO-WM commit：
- 本阶段问题与方法：
- 修改的关键文件：
- 数据 split/manifest/checkpoint：
- 可复现命令：
- 实际运行环境与训练步数：
- 核心数值（含有效样本量）：
- 对照基线与结论：
- 下一阶段可复用资产：
- 未解决的具体问题：
```

下一次给 Agent 发任务时，同时附上上一轮交接文本和当前新阶段的 Prompt，使模型能够依据 **真实的 Git 状态、已有数据与实验产物**继续，而不是重新猜测进度。

---

## 10. 关键决策点：什么情况下应该调整研究路线？

| 事实 | 下一步决策 |
|---|---|
| V0 不如 DAS+Δt/直接回归 | 先分析未来状态目标与图像观测的信息量，重新设计 latent/prediction head；不急着加 EMA |
| V1 天气增益只在同年有效，跨年无增益 | 分析季节时钟、真实天气分布差异与 OOD 泛化；增加天气内容 shuffle、year holdout 与 GDD-only |
| V2 单步有优势但 rollout 漂移仍大 | 研究多步损失、分段预测和生理状态读出，而不是只提高 orth 权重 |
| V3 rollout 不能超过同输入的直接 yield regression | 暂时将“表型生长推演”设为主贡献，产量预测降为应用任务 |
| V4 LoRA/EMA 没有净提升 | 保留冻结 DINOv2 的更优版本，定量解释微调收益与成本 |
| 气象 swap 输出变化但无法对应真实观测 | 报告为条件敏感性分析；不升级为经过实证确认的因果反事实 |

---

## 11. 原方案的关键修订记录

| 原先表述 | 现在采用的表述/实现 |
|---|---|
| “从零搭建 DINOv2+LoRA+EMA+JEPA 主网络” | 首选 **DINO-WM 官方代码**；先 V0 冻结潜空间，再逐步添加 EMA/LoRA |
| “FIP 自带 6 维逐日气象矩阵” | 从官方 `*_values/dates/times` 传感器序列按天聚合、计算派生变量 |
| “所有年份都有四项动态表型” | Sen 只在 2016–2018；Protein 2017 缺失；WHC/PH/CC 依真实日期与掩码使用 |
| “每次照片都能得到当天四项表型标签” | 保留原始测量日期与图像日期差，按任务指定匹配窗口 |
| “2018/2022 可全面验证真实早衰曲线” | 2018 可验证有标签的 Sen；2022 无该标签，不作为真实 Sen 评估 |
| “跨年天气替换证明气候因果律” | 定义为天气情景敏感性与跨年外推；因果解释需更多识别依据 |
| “训练 6.5 小时、速度 6 倍、误差改进 X% 已确定” | 全部转为**实测指标或待验证假设**；论文表格不事先填最优值 |
| “V-JEPA/I-JEPA 即可直接作为同任务对照” | 需实际适配且同协议评估；先完成 DINO-WM V0 和强任务基线 |
| “固定 80/10/10 是唯一协议” | 官方 FIP split 为主，原自定义 80/10/10 + 2018/2022 作为补充 |
| “JEPA 子空间切分自动正交防坍塌” | 正交性与稳定性由损失/测量和消融证实，不由拆分维度自行产生 |

---

## 12. 一手来源与实现入口

1. [Zhou et al., DINO-WM, ICML 2025 (PMLR 正式论文)](https://proceedings.mlr.press/v267/zhou25t.html)
2. [DINO-WM 官方 GitHub](https://github.com/gaoyuezhou/dino_wm)
3. [Meta DINOv2 官方 GitHub](https://github.com/facebookresearch/dinov2)
4. [Meta I-JEPA 官方 GitHub](https://github.com/facebookresearch/ijepa)
5. [Meta V-JEPA 2 与 V-JEPA 2-AC 官方 GitHub](https://github.com/facebookresearch/vjepa2)
6. [FIP 1.0 原始论文，GigaScience 2025](https://doi.org/10.1093/gigascience/giaf051)，重点读取 Table 4、Table 5、Table 6、官方 split 方法。
7. [FIP 1.0 Hugging Face 数据集与数据卡](https://huggingface.co/datasets/mikeboss/FIP1)
8. [FIP 1.0 ETH Research Collection 数据资源](https://doi.org/20.500.11850/697773)

**最终实施顺序：** `P1 上游梳理 → P2 数据审计 → P3 样本构造 → P4 V0 → P5 强基线 → P6 V1 天气 → P7 V2 生理 → P8 V3 Rollout → P9 V4 EMA/LoRA → P10 实验 → P11 论文`。每一步只有在产出真实、可复核的结果后才进入下一步。
