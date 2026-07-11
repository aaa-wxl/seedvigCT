# 基于脑电与眼电跨模态门控融合的驾驶疲劳检测方法

## A Driving Fatigue Detection Method Based on Cross-Modal Gated Fusion of EEG and EOG

> **作者**：[待补充]
> **单位**：[待补充]
> **通讯作者**：[待补充]
>
> 本文按当前模型状态撰写，记录 Raw EEG-EOG Delta-Gate 模型的架构原理与数据流。本文为初稿，实验结果部分待补充，不随后续代码变更更新。

---

## 摘要

驾驶疲劳检测是保障交通安全的重要课题。脑电（EEG）信号能够直接反映大脑生理状态，是疲劳检测的金标准信号；眼电（EOG）信号则能捕捉眨眼、闭眼、眼球运动等与困意强相关的行为特征。然而，EOG 信号易受自主性动作（如揉眼、光照反射）干扰，直接融合会引入噪声。针对这一问题，本文提出一种基于脑电与眼电跨模态门控融合的驾驶疲劳检测模型（Raw EEG-EOG Delta-Gate）。该模型以 EEG 为主干提取时序表征，通过跨模态注意力（Cross-Attention）使 EEG 按需从 EOG 中读取佐证信息，并引入门控机制（EOG Gate）自适应控制每个时间窗口中眼电信息的进入强度；同时设计时间差分模块（Temporal Delta）显式建模最后两个窗口之间的状态变化趋势。模型同时输出二分类结果与连续 PERCLOS 回归值，支持多任务、纯分类、纯回归三种训练目标。本文系统阐述该模型的网络结构、维度演算、融合机制与损失函数设计。

**关键词**：驾驶疲劳检测；脑电；眼电；跨模态注意力；门控融合；时间差分

## Abstract

Driving fatigue detection is critical for traffic safety. Electroencephalogram (EEG) signals directly reflect brain physiological states and serve as the gold standard for fatigue monitoring, while electrooculogram (EOG) signals capture drowsiness-related behaviors such as blinking and eye closure. However, EOG is easily contaminated by voluntary artifacts (eye rubbing, light reflex), and naive fusion introduces noise. To address this, this paper proposes a Raw EEG-EOG Delta-Gate model based on cross-modal gated fusion. The model extracts temporal representations from EEG as the backbone, employs cross-attention for EEG to read corroborating evidence from EOG on demand, and introduces an EOG Gate to adaptively control how much EOG information enters each time window. A Temporal Delta module is further designed to explicitly model the state-change trend between the last two windows. The model jointly outputs a binary classification and a continuous PERCLOS regression, supporting multitask, classification-only, and regression-only training objectives. This paper systematically describes the network architecture, dimensional computation, fusion mechanism, and loss design of the model.

**Keywords**: driving fatigue detection; EEG; EOG; cross-modal attention; gated fusion; temporal delta

---

## 0 引言

驾驶疲劳是引发交通事故的重要原因之一。在长时间驾驶过程中，驾驶员的注意力、反应能力与警觉性会随疲劳累积而下降，因此实时、可靠的疲劳检测对交通安全具有重要意义。

在各类生理信号中，脑电（EEG）直接反映大脑皮层的神经活动，被认为是疲劳检测的金标准信号；眼电（EOG）则记录眼球运动与眼睑状态，能够捕捉眨眼频率上升、眼睑下垂、慢速眼动等典型的疲劳行为特征。两者具有较强的互补性，多模态融合已成为疲劳检测的主流方向。

然而，EEG-EOG 多模态融合仍面临两个挑战：

1. **EOG 信号噪声问题**。EOG 不仅包含与疲劳相关的眨眼、闭眼信息，也包含自主性动作（如揉眼、打哈欠、光照反射）带来的非疲劳性扰动。若将 EOG 信息无差别地融入 EEG，噪声会干扰判别。需要一种机制能够根据当前 EEG 上下文，自适应判断 EOG 信息在各个时间窗口中的可信程度。

2. **疲劳的时序变化趋势建模不足**。疲劳是一个渐进过程，"状态的变化趋势"往往比"状态的绝对值"更具判别力。例如，最后一段时间内状态明显下滑，即使绝对值尚未达到疲劳阈值，也已强烈暗示正在进入疲劳。现有方法多依赖最终状态表征，未显式建模相邻窗口间的变化趋势。

针对上述问题，本文提出 Raw EEG-EOG Delta-Gate 模型，主要贡献包括：

- 设计 **EEG-EOG 跨模态注意力融合**机制：EEG 作为查询（Query），EOG 作为键值（Key/Value），使 EEG 按需从 EOG 中读取相关佐证，而非无差别接收；
- 引入 **EOG 门控（EOG Gate）**机制：结合 EEG 上下文与 EOG 融合结果，为每个时间窗口输出一个 0~1 的可信度，自适应控制眼电信息进入强度；
- 设计 **时间差分（Temporal Delta）**模块：显式计算最后两个窗口的状态差分，并与最终状态拼接，增强对疲劳变化趋势的建模能力；
- 模型同时输出二分类与 PERCLOS 回归，并支持多任务、纯分类、纯回归三种训练目标。

## 1 模型总体架构

模型以 8 个连续 8 秒窗口（共约 64 秒）为一个样本，输入 EEG 与 EOG 两路原始信号，分别经各自的 CNN 编码器提取 token 序列；随后通过跨模态注意力将 EOG 信息按需融入 EEG；融合后的窗口序列经位置编码与时序 Transformer 建模窗口间演变；最后取末窗口状态并拼接末窗口差分，分别送入分类头与回归头。整体流程如图 1 所示。

**图 1** Raw EEG-EOG Delta-Gate 模型总体流程

默认配置如表 1 所示。

**表 1** 模型默认配置

| 符号 | 含义 | 默认值 |
|---|---|---|
| $B$ | 批大小（batch size） | 4 |
| $T$ | 序列长度（窗口数） | 8 |
| $D$ | 嵌入维度（embedding dim） | 64 |
| $K$ | 分类类别数 | 2（binary） |

总输入如表 2 所示。

**表 2** 模型输入

| 模态 | 维度 | 实际意义 |
|---|---|---|
| EEG | $[B, 8, 17, 1600]$ | $B$ 个样本；每个样本 8 个连续窗口；17 个 EEG 通道；每窗口 1600 采样点（8s × 200Hz） |
| EOG | $[B, 8, 7, 1000]$ | $B$ 个样本；8 个连续窗口；7 个 EOG 通道；每窗口 1000 采样点（8s × 125Hz） |

## 2 EEG 分支

EEG 分支负责将每段 8 秒的原始脑电波"读懂"，浓缩为一个 64 维窗口表征。其流程如表 3 所示。

**表 3** EEG 分支维度演算

| 模块 | 输入维度 | 输出维度 | 实际意义 |
|---|---|---|---|
| Reshape | $[B, 8, 17, 1600]$ | $[B\cdot8, 1, 17, 1600]$ | 合并 batch 与时间窗口，逐窗口做 CNN 编码 |
| CNN Patch Encoder | $[B\cdot8, 1, 17, 1600]$ | $[B\cdot8, 64, 1, 200]$ | 提取局部时序/空间特征 |
| EEG Tokens | $[B\cdot8, 64, 1, 200]$ | $[B\cdot8, 200, 64]$ | 每窗口编码为 200 个 token，每个 64 维 |
| Window Transformer | $[B\cdot8, 200, 64]$ | $[B\cdot8, 200, 64]$ | 单窗口内建模 token 间关系 |
| Mean Pool | $[B\cdot8, 200, 64]$ | $[B\cdot8, 64]$ | 窗口内 200 个 token 汇总为一个窗口表示 |
| Window Embedding | $[B\cdot8, 64]$ | $[B, 8, 64]$ | 每个样本 8 个窗口，每窗口一个 64 维表征 |

### 2.1 CNN Patch Encoder：1600 如何变为 200

CNN 的核心动作为**用一个小模板在数据上滑动**：以一个宽度为 64 的"放大镜"在长度为 1600 的时间轴上从左滑到右，每停留一次即将所见 64 个采样点总结为一个输出值。卷积输出尺寸满足

$$
L_{out} = \left\lfloor \frac{L_{in} + 2 \cdot \text{padding} - \text{kernel}}{\text{stride}} \right\rfloor + 1 \tag{1}
$$

Patch Encoder 共 12 层，可分为"改变尺寸"与"不改变尺寸"两类。逐层维度演算如表 4 所示。

**表 4** EEG Patch Encoder 逐层演算（$N = B\cdot 8$）

| 层 | 操作 | 输出 $[N, C, H, W]$ | 说明 |
|---|---|---|---|
| 输入 | — | $[N, 1, 17, 1600]$ | 原始 |
| 1 | Conv2d(k=1×64, p=0×32) | $[N, 16, 17, 1601]$ | 时间方向扫描，通道 $1\to16$，same 卷积 |
| 2 | BatchNorm2d | $[N, 16, 17, 1601]$ | 归一化，不改变形状 |
| 3 | GELU | $[N, 16, 17, 1601]$ | 激活，不改变形状 |
| 4 | Conv2d(k=**17**×1, groups=16) | $[N, 32, \mathbf{1}, 1601]$ | ⭐ $H$ 由 17 压为 1，通道 $16\to32$ |
| 5 | BatchNorm2d | $[N, 32, 1, 1601]$ | 不改变 |
| 6 | GELU | $[N, 32, 1, 1601]$ | 不改变 |
| 7 | Conv2d(k=1×16, p=0×8, groups=32) | $[N, 32, 1, 1602]$ | 时间方向再扫，same 卷积 |
| 8 | Conv2d(k=1×1) | $[N, 64, 1, 1602]$ | $1\times1$ 卷积：纯通道变换 $32\to64$ |
| 9 | BatchNorm2d | $[N, 64, 1, 1602]$ | 不改变 |
| 10 | GELU | $[N, 64, 1, 1602]$ | 不改变 |
| 11 | AvgPool2d(k=1×8, s=1×8) | $[N, 64, 1, \mathbf{200}]$ | ⭐ 时间 $1602\to200$ |
| 12 | Dropout | $[N, 64, 1, 200]$ | 随机丢弃，防过拟合 |

两个关键操作决定了最终形状：

**（1）17 个通道的"消失"（第 4 层）**。第 4 层使用高度为 17 的卷积核，恰好等于 EEG 通道数，即将 17 个电极位置在空间维度上一次性"竖向压扁"为 1，实现跨通道空间融合。

**（2）1600 到 200 的压缩（第 11 层）**。平均池化以核大小 8、步长 8 在时间轴上每 8 个相邻点取均值、输出 1 个点。由式 (1)：

$$
L_{out} = \left\lfloor \frac{1602 - 8}{8} \right\rfloor + 1 = 199 + 1 = 200 \tag{2}
$$

故 $200 \approx L_{in}/8$。池化步长是控制 token 数量的关键旋钮：步长为 16 得约 100 个 token，步长为 4 得约 400 个 token。

最终经 `.squeeze(2).transpose(1,2)` 将 $[N, 64, 1, 200]$ 转为 $[N, 200, 64]$，即 **200 个 token，每个 token 为 64 维向量**，作为 Transformer 的输入。

### 2.2 Window Transformer 与窗口池化

200 个 token 经 Window Transformer 进行自注意力建模，使窗口内相关的脑电节律相互增强；再经平均池化将 200 个 token 汇总为 1 个 64 维向量，代表该 8 秒窗口的整体脑电表征。8 个窗口对应 8 个向量，排成 $[B, 8, 64]$。

## 3 EOG 分支

EOG 分支与 EEG 分支结构相似，但有两个关键区别，其流程如表 5 所示。

**表 5** EOG 分支维度演算

| 模块 | 输入维度 | 输出维度 | 实际意义 |
|---|---|---|---|
| Reshape | $[B, 8, 7, 1000]$ | $[B\cdot8, 1, 7, 1000]$ | 逐窗口做 EOG CNN 编码 |
| EOG CNN Patch Encoder | $[B\cdot8, 1, 7, 1000]$ | $[B\cdot8, 64, 1, 125]$ | 提取眼动/眨眼/闭眼特征 |
| EOG Tokens | $[B\cdot8, 64, 1, 125]$ | $[B\cdot8, 125, 64]$ | 每窗口编码为 125 个 token，每个 64 维 |

**区别一**：EOG 使用独立的 CNN 编码器，因其采样率（125Hz）与通道数（7）均与 EEG 不同。其 token 数 125 同样由 $1000/8 \approx 125$ 得到。

**区别二**：EOG 分支不使用 Transformer，也不进行平均池化，而是保留为 125 张卡片形式 $[B\cdot8, 125, 64]$，作为后续跨模态注意力的键值。

### 3.1 为何 EOG 不使用 Transformer

EOG 的任务不是将自身总结为单一表征，而是作为"可翻阅的字典"供 EEG 查询。若对 EOG 施加 Transformer 与平均池化，125 张卡片将坍缩为 1 个向量。此时跨模态注意力的查询长度为 1、键值长度为 1，注意力退化为标量缩放，"按需挑选"的能力丧失，与门控机制功能重复。因此保留 125 张卡片是使跨模态注意力具有意义的前提。

从注意力预算角度，EEG 分支通过 Window Transformer 完成窗口内自交互并形成整体表征；EOG 分支无需 token 间自交互，其信息流动由跨模态注意力承担。如此可节省参数与计算量，并避免功能重复。

需说明，给 EOG 增加不含池化的 Transformer 亦是合理的替代设计；本文未采用该方案，主要出于参数效率与过拟合控制的考量，其优劣需由消融实验验证。

## 4 EEG-EOG 跨模态注意力融合

### 4.1 跨模态注意力

跨模态注意力使 EEG 主动从 EOG 中"按需读取"佐证信息，而非无差别接收。以 EEG 窗口表征为查询，EOG token 为键值，其流程如表 6 所示。

**表 6** 跨模态注意力维度演算

| 模块 | 输入维度 | 输出维度 | 实际意义 |
|---|---|---|---|
| Query (EEG) | $[B, 8, 64]$ | $[B\cdot8, 1, 64]$ | 每个 EEG 窗口表征作为查询 |
| Key/Value (EOG) | $[B\cdot8, 125, 64]$ | $[B\cdot8, 125, 64]$ | EOG token 作为键值 |
| Cross-Attention | Q:$[B\cdot8,1,64]$; K,V:$[B\cdot8,125,64]$ | $[B\cdot8,1,64]$ | EEG 按需从 EOG 读取信息 |
| Attended EOG | $[B\cdot8,1,64]$ | $[B,8,64]$ | 每个 EEG 窗口对应一个注意力加权的 EOG 表征 |

注意力计算可表述为

$$
\text{Attended} = \text{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V \tag{3}
$$

其中 $Q$ 来自 EEG，$K, V$ 来自 EOG。直观地，EEG 以自身"问题"在 EOG"书架"上比对标签、择相关者加权综合。若某段脑电已足以判别疲劳，其对 EOG 的注意力权重趋于均匀或偏低；若需要佐证，则精准锁定若干眼动 token。

### 4.2 EOG 门控机制

眼电有时为噪声（如自主性眨眼）。模型需一个"信任度旋钮"判断该 8 秒 EOG 是否可信。门控结构为

$$
g = \sigma\!\left( W_2 \cdot \text{GELU}(W_1 [\,e_{\text{EEG}}; e_{\text{EOG}}\,]) \right),\quad g \in (0,1) \tag{4}
$$

其中 $[\,e_{\text{EEG}}; e_{\text{EOG}}\,]$ 为 EEG 原始表征与跨模态注意力输出拼接的 128 维向量，$W_1\in\mathbb{R}^{64\times128}$，$W_2\in\mathbb{R}^{1\times64}$，$\sigma$ 为 Sigmoid。门控为每个窗口输出一个 0~1 的标量，作为该窗口 EOG 信息的可信度。其流程如表 7 所示。

**表 7** 门控融合维度演算

| 模块 | 输入维度 | 输出维度 | 实际意义 |
|---|---|---|---|
| Gate 输入 | EEG $[B,8,64]$ + Attended $[B,8,64]$ | $[B,8,128]$ | 拼接 EEG 与 EOG 注意力表征 |
| EOG Gate | $[B,8,128]$ | $[B,8,1]$ | 每窗口学习一个 EOG 使用权重 |
| Gated EOG | $g\,[B,8,1]\times$ Attended $[B,8,64]$ | $[B,8,64]$ | 控制每窗口 EOG 信息进入量 |
| Add & Norm | EEG $[B,8,64]$ + Gated EOG $[B,8,64]$ | $[B,8,64]$ | 残差相加后 LayerNorm |

### 4.3 为何门控位于跨模态注意力之后

门控要判断的是"该 8 秒 EOG 是否可信"，而此判断依赖于 EOG 实际内容及其与 EEG 的匹配程度，这些信息只有在跨模态注意力计算完成后才存在。门控的输入为 EEG 表征与注意力输出 $e_{\text{EOG}}$ 的拼接，二者缺一不可，故顺序因果锁死：

$$
\text{Cross-Attention} \rightarrow \text{Gate} \rightarrow \text{Add} \tag{5}
$$

进一步地，单看 EOG 无法区分"疲劳性闭眼"与"自主性揉眼"，可信度判断必须结合 EEG 上下文：若 EEG 亦显示状态异常，则 EOG 本次可信；若 EEG 一切正常，则 EOG 多为干扰。门控正是结合两边判断"该信几分"的机制。

跨模态注意力与门控分工不同，并不冗余：前者为细粒度（在 125 个 token 间分配权重，依据内容相似度），后者为粗粒度（对整个窗口输出一个 0~1 标量，依据整体可信度）。前者是"精挑"，后者是"把关"，先精挑再把关。

此外，门控作用于注意力输出而非 EEG 主干，且位于残差相加之前：EEG 作为主信息始终全量保留，门控仅决定辅助 EOG 信息的进入量；若先相加后过滤，噪声已污染 EEG，难以挽回。

## 5 时间建模

融合后的 8 个窗口表征经位置编码与时序 Transformer 建模窗口间演变，并引入时间差分模块，其流程如表 8 所示。

**表 8** 时间建模维度演算

| 模块 | 输入维度 | 输出维度 | 实际意义 |
|---|---|---|---|
| Fused Sequence | $[B,8,64]$ | $[B,8,64]$ | 8 个连续窗口的融合表征 |
| Positional Encoding | $[B,8,64]$ + $[1,8,64]$ | $[B,8,64]$ | 加入窗口顺序信息 |
| Temporal Transformer | $[B,8,64]$ | $[B,8,64]$ | 建模 8 窗口间疲劳变化 |
| Last Window | $[B,8,64]$ | $[B,64]$ | 取最后窗口作为当前状态 |
| Temporal Delta | $h_{-1} - h_{-2}$ | $[B,64]$ | 最后两窗口的状态变化趋势 |
| Concat | $[B,64]$ + $[B,64]$ | $[B,128]$ | 同时保留当前状态与变化趋势 |

### 5.1 位置编码与时序 Transformer

Transformer 本身不具备顺序感知能力，故先加入可学习的位置编码以告知模型 8 个窗口的先后顺序，再经时序 Transformer 进行窗口间自注意力建模，刻画 64 秒内疲劳的渐进演变。时序 Transformer 与 EEG 分支的 Window Transformer 原理相同、作用对象不同：后者建模单窗口内 200 个 token，前者建模 8 个窗口。

### 5.2 时间差分模块

疲劳的"变化趋势"往往比"绝对值"更具判别力。模型取末窗口表征 $h_{-1}$ 与前一窗口表征 $h_{-2}$ 之差作为状态变化趋势：

$$
\Delta h = h_{-1} - h_{-2},\quad h = [\,h_{-1};\, \Delta h\,] \in \mathbb{R}^{128} \tag{6}
$$

将当前状态与变化趋势拼接为 128 维向量，显式地将"变化方向"提供给分类与回归头，避免模型自行猜测趋势。当序列长度为 1 时，差分退化为零向量。

## 6 输出与损失函数

### 6.1 双头输出

模型设有分类头与回归头，分别输出二分类 logits 与连续 PERCLOS 值，如表 9 所示。

**表 9** 输出头维度演算

| 输出头 | 输入维度 | 输出维度 | 实际意义 |
|---|---|---|---|
| Classification Linear | $[B,128]$ | $[B,2]$ | 二分类 logits |
| Regression Linear + Sigmoid | $[B,128]$ | $[B,1]$ | 连续 PERCLOS，范围 $[0,1]$ |

最终输出为

$$
\text{class\_logits} \in \mathbb{R}^{B\times2},\quad \text{perclos} \in [0,1]^{B\times1} \tag{7}
$$

并附带可观测的 `eog_gate` $[B,8,1]$ 与 `eog_attention_weights` $[B,8,1,125]$。需说明，模型内未显式实现 Softmax，`class_logits` 为原始分数；Softmax 隐式发生于交叉熵损失内部，推理时若需概率需自行施加。

### 6.2 损失函数

损失函数由分类损失与回归损失两部分构成：

$$
\mathcal{L}_{\text{cls}} = \text{CrossEntropy}(\text{class\_logits},\, y_{\text{cls}}) \tag{8}
$$

$$
\mathcal{L}_{\text{reg}} = \text{SmoothL1}(\text{perclos},\, y_{\text{perclos}}) \tag{9}
$$

其中分类损失内部含 Softmax，对未正确分类的罚分随正确类别概率降低而升高；回归损失 SmoothL1 在误差较大时按线性惩罚、误差较小时按平方惩罚，兼顾稳健性与收敛速度。

依据训练目标 $\text{objective}$，总损失为

$$
\mathcal{L} =
\begin{cases}
\mathcal{L}_{\text{cls}}, & \text{objective} = \text{classification} \\
\mathcal{L}_{\text{reg}}, & \text{objective} = \text{regression} \\
\mathcal{L}_{\text{cls}} + \lambda\, \mathcal{L}_{\text{reg}}, & \text{objective} = \text{multitask}
\end{cases} \tag{10}
$$

其中 $\lambda$ 为回归权重（默认 0.5）。多任务模式下两任务共享底层特征、相互借力，通常优于单独训练；单任务模式则可使模型全力优化单一目标。$\lambda$ 用于平衡两损失项的量级与相对重要程度。

训练采用 AdamW 优化器，单步流程为：前向预测 $\rightarrow$ 损失计算 $\rightarrow$ 梯度清零 $\rightarrow$ 反向传播 $\rightarrow$ 参数更新。

## 7 实验设置

### 7.1 数据集

实验基于 SEED-VIG 公开疲劳数据集。EEG 采样率为 200Hz、17 通道；EOG 采样率为 125Hz、7 通道。每 8 秒为一个窗口，连续 8 个窗口构成一个 64 秒样本。标签为 PERCLOS 连续值，并按阈值 0.35 映射为二分类（awake / fatigue），按 0.35 与 0.70 映射为三分类。

### 7.2 训练设置

主要超参数如表 10 所示。

**表 10** 训练超参数

| 超参数 | 取值 |
|---|---|
| 序列长度 $T$ | 8 |
| 批大小 $B$ | 4 |
| 嵌入维度 $D$ | 64 |
| 注意力头数 | 4 |
| Window Transformer 层数 | 1 |
| Temporal Transformer 层数 | 1 |
| 学习率 | $1\times10^{-3}$ |
| 训练轮数 | 20 |
| 优化器 | AdamW |

### 7.3 评估指标与划分策略

划分策略包括实验内 5 折交叉验证（within_experiment_5fold）与按被试分组（group_subject）。评估指标包括准确率（accuracy）、宏 F1（macro F1）、平衡准确率（balanced accuracy）用于分类，平均绝对误差（MAE）、均方根误差（RMSE）、皮尔逊相关系数（Pearson）用于回归。

### 7.4 消融设计

为验证各模块贡献，设计三组对照实验：

- **EEG-only**：仅使用 EEG，不使用跨模态注意力；
- **EOG-only**：仅使用 EOG，不使用跨模态注意力；
- **EEG-EOG Delta-Gate**：同时使用跨模态注意力、时间差分与门控（EOG dropout = 0.25）。

### 7.5 实验结果与分析

> **待补充**。本文为初稿，实验结果（各模型在各指标上的具体数值、消融对比、统计显著性分析）将在正式版本中补充，本文不包含任何未经实际运行得出的结果数据。

## 8 结论

本文提出一种基于脑电与眼电跨模态门控融合的驾驶疲劳检测模型。该模型以 EEG 为主干提取时序表征，通过跨模态注意力使 EEG 按需从 EOG 中读取佐证信息，并引入门控机制自适应控制眼电信息进入强度，有效缓解了 EOG 噪声问题；时间差分模块显式建模末窗口状态变化趋势，增强了对疲劳渐进过程的刻画能力。模型同时输出二分类与连续 PERCLOS，并支持多任务、纯分类、纯回归三种训练目标。后续工作将补充完整实验结果与消融分析，以验证各模块的有效性。

## 参考文献

> **待补充**。初稿参考文献将在正式版本中补全，预计包括：SEED-VIG 数据集文献、EEG/EOG 疲劳检测相关研究、Conformer 与 Transformer 相关文献、跨模态注意力与门控融合相关方法、多任务学习相关方法等。
