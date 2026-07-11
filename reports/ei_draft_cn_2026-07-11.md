# 面向跨被试警觉度估计的原始 EEG/EOG 可靠性门控融合方法

## 摘要

驾驶疲劳与警觉度下降是交通安全中的重要风险因素。SEED-VIG 数据集提供了同步采集的脑电（EEG）、眼电（EOG）以及基于眼部闭合程度的 PERCLOS 连续警觉度标签，为多模态警觉度估计提供了公开基准。已有研究通常关注分类准确率或特征级融合，但在严格跨被试设置下，模型是否真正学习到可泛化的生理表征仍需进一步分析。本文基于原始 EEG/EOG 信号构建轻量级 Conformer 框架，将 PERCLOS 连续值回归作为主任务，并将二分类指标作为回归输出阈值化后的工程评估结果。针对 SEED-VIG 中 EOG 与 PERCLOS 标签高度相关的问题，本文进一步分析 EOG-only 强基线，并提出可靠性门控的 EEG/EOG 融合策略，使 EEG 残差信息在 EOG-only 高误差样本和 EOG 严重失效场景中提供补偿。实验结果表明，在严格 group-subject 设置下，原始 EOG-only 回归模型达到 RMSE 0.1432、Pearson 0.8536；本文 no-delta EEG/EOG 融合模型达到 RMSE 0.1433、Pearson 0.8508，并在 F1 和平衡准确率上略优于 EOG-only。在 EOG 全零退化场景中，融合模型 RMSE 由 EOG-only 的 0.4939 降至 0.3394；在 EOG-only 误差最大的样本中，融合模型在所有退化设置下均降低 RMSE。结果说明，EOG 是 SEED-VIG PERCLOS 预测中的强代理信号，而 EEG/EOG 可靠性门控融合的价值主要体现在困难样本和 EOG 不可靠场景下的鲁棒补偿。

**关键词：** 警觉度估计；SEED-VIG；PERCLOS；脑电；眼电；跨被试评估；多模态融合

## 1 引言

驾驶员警觉度下降会显著增加交通事故风险，因此基于生理信号的警觉度估计具有重要研究价值。脑电信号能够反映神经活动状态，眼电信号能够刻画眼动和眨眼行为，两者常被用于疲劳检测、警觉度估计和人机交互监测任务。SEED-VIG 数据集同步提供 EEG、EOG 与 PERCLOS 标签，是该方向常用的公开数据集。

然而，SEED-VIG 的标签本身由眼部闭合程度定义，这使 EOG 与标签之间存在天然相关性。在这种情况下，如果只报告多模态模型相对于 EEG-only 或传统特征模型的提升，可能无法回答一个关键问题：模型是否真正利用了 EEG 与 EOG 的互补信息，还是主要依赖 EOG 重构 PERCLOS 标签。因此，本文首先将 EOG-only 作为强基线进行分析，并在严格跨被试设置下比较原始 EEG/EOG 融合方法的实际收益。

本文不将任务主线设定为二分类准确率最大化，而是采用连续 PERCLOS 回归作为主任务。原因是 PERCLOS 本身为连续警觉度指标，直接回归能够保留更多标签信息；二分类准确率、F1 和平衡准确率则通过阈值化回归输出得到，用作工程场景下的辅助评估指标。实验也表明，CE-only 分类训练虽然能够获得较高分类指标，但会破坏连续 PERCLOS 估计能力，导致 RMSE 明显升高且 Pearson 相关系数为负。

本文的主要贡献如下：

1. 构建了一个基于原始 EEG/EOG 的轻量级 Conformer 警觉度估计框架，在严格 group-subject 协议下评估跨被试泛化能力。
2. 提出并验证了以 PERCLOS 回归为主任务、分类指标由回归输出阈值化得到的训练与评估范式。
3. 系统分析了 SEED-VIG 中 EOG-only 强基线现象，指出 EOG 是 PERCLOS 标签的强代理信号，多模态方法必须与 EOG-only 进行公平比较。
4. 通过 EOG 退化实验与误差分层分析表明，EEG/EOG 可靠性门控融合能够在 EOG-only 高误差样本和 EOG 严重失效场景中提供补偿。

## 2 相关工作

### 2.1 基于 EEG/EOG 的警觉度估计

早期警觉度估计方法通常依赖人工设计的频域、时域或时频域特征，再使用传统机器学习模型进行分类或回归。SEED-VIG 数据集的提出推动了 EEG/EOG 多模态警觉度估计研究。原始 SEED-VIG 工作指出，EEG、EOG 与时间依赖关系均对警觉度估计具有价值。

近年来，深度学习方法逐渐用于该任务。例如，VIGNet 等方法使用卷积神经网络从 EEG 中学习表征，并分别进行分类和回归实验；HMS-TENet 等工作引入多分支特征提取、拓扑注意力和多任务学习，用于提升疲劳或警觉度识别性能。这些工作证明了深度模型在 SEED-VIG 上的有效性，但不同论文之间常存在分类/回归任务、被试划分方式和标签阈值设置差异，因此不能仅凭准确率进行直接横向比较。

### 2.2 多模态融合与可靠性建模

EEG 与 EOG 具有互补性：EEG 更接近神经活动状态，EOG 则直接反映眼动行为。已有研究中常见的融合方式包括特征拼接、注意力融合、门控融合和多任务学习。近期方法进一步关注跨模态可靠性、模态缺失和跨被试泛化问题。对于 SEED-VIG 这类 PERCLOS 标签与 EOG 高度相关的数据集，可靠性建模尤其重要，因为模型需要区分“EOG 作为强代理信号”与“EEG/EOG 互补融合”这两类贡献。

### 2.3 本文定位

与单纯追求 clean average 准确率不同，本文重点关注严格跨被试设置下的连续 PERCLOS 估计、EOG-only 强基线分析以及 EOG 不可靠场景下的融合鲁棒性。本文不声称多模态融合在干净测试条件下全面超过 EOG-only，而是强调 EEG 残差信息在困难样本和严重 EOG 失效条件下的补偿作用。

## 3 方法

### 3.1 问题定义

给定长度为 \(T\) 的原始 EEG 序列和 EOG 序列，目标是预测对应窗口的连续 PERCLOS 值：

```text
y = f(EEG, EOG),  y in [0, 1]
```

其中 \(y\) 越大表示警觉度越低或疲劳程度越高。本文以连续 PERCLOS 回归作为主任务，二分类指标通过阈值化预测值计算。

### 3.2 原始信号编码

模型首先将每个 8 秒窗口的原始 EEG 和 EOG 分别输入轻量级 CNN patch encoder。该模块由时域卷积、通道卷积、逐点卷积、批归一化和 GELU 激活组成，用于从原始信号中提取局部时序模式。得到的窗口级 token 进一步通过轻量级自注意力模块形成窗口表征。

### 3.3 EEG/EOG 可靠性门控融合

对于 EEG/EOG 融合，本文采用 EEG-query/EOG-key-value 的跨模态注意力结构。EEG 表征作为 Query，EOG 表征作为 Key 和 Value，使模型能够根据 EEG 当前状态选择性查询 EOG 信息。随后引入 EOG 可靠性门控，对跨模态信息进行加权融合。

在 anchor residual 版本中，模型分别预测 EEG 分支和 EOG 分支的 PERCLOS 值，并通过门控系数学习两者之间的残差修正：

```text
y_eog = h_eog(EOG)
y_eeg = h_eeg(EEG)
g = sigmoid(W[EEG, EOG])
y_fused = y_eog + g * (y_eeg - y_eog)
```

该设计的动机不是让 EEG 完全取代 EOG，而是让 EEG 在 EOG-only 预测不可靠时提供修正。

### 3.4 训练目标

本文主方法采用 SmoothL1 损失进行 PERCLOS 回归：

```text
L_main = SmoothL1(y_fused, y_true)
```

CE-only 和 CE+SmoothL1 多任务损失仅作为消融实验保留。原因是实验显示 CE-only 能优化阈值化分类标签，但会显著损害连续 PERCLOS 预测能力。

## 4 实验设置

### 4.1 数据集与协议

实验使用 SEED-VIG 数据集。该数据集包含原始 EEG、EOG 以及 PERCLOS 标签。本文采用两类评估协议：

1. within-experiment：同一实验内划分训练与测试。
2. group-subject：按被试划分训练、验证和测试集，保证测试被试不出现在训练集中。

本文主结果采用 group-subject 设置，因为该设置更能反映跨被试泛化能力。

### 4.2 评价指标

连续 PERCLOS 估计使用 RMSE、MAE 和 Pearson 相关系数评价。工程二分类指标包括 Accuracy、Macro-F1 和 Balanced Accuracy，它们由连续 PERCLOS 输出阈值化后计算。

### 4.3 对比方法

本文对比以下模型：

- `D:\eeg-eog` 中已有的传统特征拼接基线。
- 原始 EEG+EOG cross-attention 模型。
- 原始 EOG-only 回归模型。
- 原始 EEG+EOG delta/gate 模型。
- 原始 EEG+EOG anchor residual 模型。
- 原始 EEG+EOG anchor residual no-delta 模型。

## 5 实验结果与分析

### 5.1 主结果

表 1 给出了严格跨被试设置下的主要结果。

| 方法 | 训练目标 | Acc | F1 | BalAcc | RMSE | Pearson |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 特征拼接基线 | 回归 | 0.4912 | - | - | 0.2563 | 0.5746 |
| 原始 EEG+EOG cross-attention | 回归 | 0.7233 | 0.7241 | 0.7332 | 0.1652 | 0.8145 |
| 原始 EOG-only | 回归 | 0.8250 | 0.7883 | 0.7820 | 0.1432 | 0.8536 |
| 原始 EEG+EOG delta/gate | 回归 | 0.8233 | 0.7995 | 0.7977 | 0.1483 | 0.8406 |
| 原始 EEG+EOG anchor residual | 回归 | 0.8027 | 0.7894 | 0.7993 | 0.1471 | 0.8398 |
| 原始 EEG+EOG anchor residual, no-delta | 回归 | 0.8263 | 0.8043 | 0.8017 | 0.1433 | 0.8508 |

可以看到，原始 EOG-only 在 clean average RMSE 和 Pearson 上仍然是强基线。no-delta 融合模型与 EOG-only 在 RMSE 上基本持平，并在 F1 和 Balanced Accuracy 上略有提升。这说明在 SEED-VIG 上，EOG 对 PERCLOS 标签具有强代理作用；多模态融合不能简单声称在平均性能上全面超过 EOG-only。

### 5.2 损失函数消融

表 2 展示了分类目标与回归目标的差异。

| 设置 | Acc | F1 | BalAcc | RMSE | Pearson |
| --- | ---: | ---: | ---: | ---: | ---: |
| EOG-only CE-only | 0.8261 | 0.8118 | 0.8191 | 0.3070 | -0.4279 |
| EEG+EOG delta/gate CE-only | 0.8155 | 0.7984 | 0.8013 | 0.2949 | -0.6462 |
| EOG-only regression | 0.8250 | 0.7883 | 0.7820 | 0.1432 | 0.8536 |
| EEG+EOG no-delta regression | 0.8263 | 0.8043 | 0.8017 | 0.1433 | 0.8508 |

CE-only 训练能够获得较高分类指标，但连续 PERCLOS 预测性能显著下降。因此，本文将 PERCLOS regression-only 作为主任务，将 CE-only 和多任务损失作为消融。

### 5.3 EOG 退化鲁棒性分析

为了验证 EEG 残差信息是否在 EOG 不可靠时发挥作用，本文对测试集 EOG 输入进行退化，包括加噪、随机 mask、通道 mask、片段 mask 和全零输入。

| 场景 | EOG RMSE | 融合 RMSE | 差值 |
| --- | ---: | ---: | ---: |
| clean | 0.1432 | 0.1433 | +0.0001 |
| EOG 加噪 0.5 | 0.1928 | 0.1926 | -0.0002 |
| 随机 mask 25% | 0.2059 | 0.2627 | +0.0568 |
| 通道 mask 25% | 0.1735 | 0.2068 | +0.0333 |
| 片段 mask 25% | 0.1713 | 0.2110 | +0.0397 |
| EOG 全零 | 0.4939 | 0.3394 | -0.1545 |

结果表明，在部分 mask 场景下，当前融合模型仍不够稳定；但在 EOG 全零场景中，融合模型明显优于 EOG-only，说明 EEG 分支确实提供了退化保护。

进一步地，在 EOG-only 误差最大的样本上，融合模型在所有退化场景中均降低 RMSE：

| 场景 | worst samples 上融合相对 EOG-only 的 RMSE 差值 |
| --- | ---: |
| clean | -0.0221 |
| EOG 加噪 0.5 | -0.0721 |
| 随机 mask 25% | -0.0269 |
| 通道 mask 25% | -0.0149 |
| 片段 mask 25% | -0.0494 |
| EOG 全零 | -0.5061 |

这说明 EEG 残差信息的价值主要体现在困难样本和 EOG-only 预测失效的情况下。

## 6 讨论

本文结果提示，在 SEED-VIG 上评价多模态警觉度估计方法时，必须谨慎处理 EOG 与 PERCLOS 标签之间的强相关性。如果不加入 EOG-only 强基线，可能会高估多模态融合的贡献。本文发现 EOG-only 在 clean average 上非常强，而 EEG/EOG 融合模型的价值更多体现在鲁棒性和困难样本补偿方面。

这也解释了为什么一些分类论文能够报告较高准确率：一方面，分类阈值会简化连续 PERCLOS 估计问题；另一方面，不同论文可能采用不同的被试划分协议。如果只比较 Accuracy，而不比较 RMSE、Pearson 和严格跨被试协议，就容易得到不公平结论。

## 7 结论

本文围绕 SEED-VIG 数据集构建了原始 EEG/EOG 可靠性门控 Conformer 框架，并在严格跨被试设置下分析连续 PERCLOS 警觉度估计问题。实验表明，EOG-only 是该数据集上的强基线，说明 EOG 与 PERCLOS 标签之间存在显著代理关系。本文提出的 no-delta EEG/EOG 融合模型在 clean average 上与 EOG-only 基本持平，并在分类工程指标、EOG 全零退化场景以及 EOG-only 高误差样本上表现出补偿作用。未来工作将进一步改进部分 EOG 缺失场景下的融合稳定性，并探索跨被试表征对齐方法。

## 参考文献与资料来源（投稿前需统一格式）

1. SEED-VIG 数据集官方页面：https://bcmi.sjtu.edu.cn/home/seed/seed-vig.html
2. SEED-VIG 原始论文：https://arxiv.org/abs/1606.07790
3. VIGNet: A Deep Convolutional Neural Network for EEG-based Driver Vigilance Estimation. 需投稿前核验正式出版信息。
4. HMS-TENet 相关工作。需投稿前核验论文题名、期刊、年份和 DOI。
5. TMU-Net / 可靠性门控多模态融合相关工作。需投稿前核验正式引用格式。
6. E2CF / EEG-EOG cross-modal fusion 相关工作。需投稿前核验正式引用格式。

## 图表清单

- 图 1：原始 EEG/EOG 可靠性门控 Conformer 架构。文件：`reports/raw_eeg_eog_reliability_gated_conformer_cn.svg`
- 表 1：group-subject 主结果。
- 表 2：分类损失与回归损失消融。
- 表 3：EOG 退化鲁棒性分析。
- 表 4：EOG-only 高误差样本分层分析。
