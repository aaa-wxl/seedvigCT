# SEED-VIG 原始 EEG/EOG 警觉度估计导师汇报简版

日期：2026-07-11

## 一句话主线

本文建议定位为：面向 SEED-VIG 数据集的原始 EEG/EOG 连续警觉值估计方法，采用严格跨被试评估协议，以 PERCLOS 回归作为主任务，并将二分类准确率、F1 和平衡准确率作为由回归输出阈值化得到的工程评估指标。

## 给导师汇报时的核心说法

当前不建议继续堆叠更大的模型。SEED-VIG 的标签是 PERCLOS，而 PERCLOS 本身由眼部闭合程度定义，因此 EOG-only 在干净测试条件下表现很强是合理现象。这个现象不是失败，而是论文中需要主动解释的关键发现。

本文更稳妥的叙事是：

```text
EOG 在干净条件下是 PERCLOS 的强代理信号；
原始 EEG/EOG 门控融合不能声称在平均 RMSE 上全面超过 EOG-only；
但是 EEG 残差信息能够在 EOG-only 容易出错的样本和 EOG 严重失效场景中提供补偿。
```

因此，论文主线不应写成“多模态融合全面优于 EOG-only”，而应写成“严格评估下的 EOG 代理效应分析与可靠性门控融合”。

## 推荐方法名称

建议使用：

```text
原始 EEG/EOG 可靠性门控 Conformer
```

英文稿中可写为：

```text
Raw EEG-EOG Reliability-Gated Conformer
```

不建议继续把主方法命名为 `Delta-Gate Conformer`，因为当前实验显示 no-delta anchor residual 版本优于 temporal-delta 版本。Temporal delta 更适合作为消融实验，而不是主创新点。

## 架构图修改建议

原图可以继续使用布局，但文字和逻辑需要改：

| 原图标注 | 建议修改 |
| --- | --- |
| `Softmax` | `阈值化分类评估` |
| `Output Probabilities` | `二分类工程指标` |
| `PERCLOS Output` | `PERCLOS 连续警觉值输出` |
| `Temporal Delta` | `可选时间差分（消融）` |
| `Raw EEG-EOG Delta-Gate Conformer Architecture` | `原始 EEG/EOG 可靠性门控 Conformer 架构` |

图注建议：

```text
图 1 原始 EEG/EOG 可靠性门控 Conformer 架构。模型分别使用轻量级 CNN patch encoder 从原始 EEG 和 EOG 窗口中提取时序 token，通过 EEG-query/EOG-key-value 的跨模态注意力与可靠性门控完成融合。主输出为连续 PERCLOS 回归值；二分类准确率、F1 和平衡准确率由 PERCLOS 预测值阈值化后计算。时间差分和分类损失仅作为消融设置。
```

已新增同风格架构图：`reports/raw_eeg_eog_reliability_gated_conformer_cn.svg`，并生成 PNG 预览：`reports/raw_eeg_eog_reliability_gated_conformer_cn.png`。

## 当前主要结果

除特别说明外，以下结果均为 group-subject 跨被试设置。

| 方法 | 训练目标 | Acc | F1 | BalAcc | RMSE | Pearson | 说明 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `D:\eeg-eog` 特征拼接基线 | 回归 | 0.4912 | - | - | 0.2563 | 0.5746 | 旧特征基线 |
| 原始 EEG+EOG cross-attention | 回归 | 0.7233 | 0.7241 | 0.7332 | 0.1652 | 0.8145 | 第一版原始多模态模型 |
| 原始 EOG-only | 回归 | 0.8250 | 0.7883 | 0.7820 | 0.1432 | 0.8536 | 同评估脚本 clean 对比 |
| 原始 EEG+EOG delta/gate | 回归 | 0.8233 | 0.7995 | 0.7977 | 0.1483 | 0.8406 | 时间差分/门控消融 |
| 原始 EEG+EOG anchor residual | 回归 | 0.8027 | 0.7894 | 0.7993 | 0.1471 | 0.8398 | 带时间差分 |
| 原始 EEG+EOG anchor residual, no-delta | 回归 | 0.8263 | 0.8043 | 0.8017 | 0.1433 | 0.8508 | 当前最合适的融合版本 |

结论：

```text
干净测试条件下，EOG-only 和 no-delta 融合模型在 RMSE 上基本打平；
no-delta 融合模型在 F1 和 BalAcc 上略有优势；
EOG-only 在 Pearson 上仍略强。
```

注意：`runs/auto_raw_eog_only_binary_group_subject_reg/summary.json` 当前只统计了 fold1-4。表中 EOG-only clean 结果采用 EOG corruption 评估脚本对 fold0-4 重新评估得到，因此更适合与 no-delta 融合模型公平比较。

## 损失函数选择

主论文建议只使用 PERCLOS 回归损失：

```text
L_main = SmoothL1(y_pred, PERCLOS)
```

二分类 Acc、F1、BalAcc 不参与主训练，只由连续 PERCLOS 预测值阈值化后计算。

原因如下：

| 目标设置 | Acc | F1 | BalAcc | RMSE | Pearson | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| EOG-only CE-only | 0.8261 | 0.8118 | 0.8191 | 0.3070 | -0.4279 | 分类可用，连续估计失败 |
| EEG+EOG delta/gate CE-only | 0.8155 | 0.7984 | 0.8013 | 0.2949 | -0.6462 | 连续 PERCLOS 估计失败 |
| EOG-only regression | 0.8250 | 0.7883 | 0.7820 | 0.1432 | 0.8536 | 主基线 |
| EEG+EOG no-delta regression | 0.8263 | 0.8043 | 0.8017 | 0.1433 | 0.8508 | 主融合模型 |

因此，CE-only 或 CE+SmoothL1 多任务损失可以保留为消融，但不建议作为主方法。这样写更清楚，也更容易解释为什么本文不是单纯追求分类准确率。

## 鲁棒性证据

多模态融合当前最有价值的证据不是 clean average 全面超过 EOG-only，而是在 EOG 失效或 EOG-only 高误差样本上的补偿能力。

| 场景 | EOG RMSE | 融合 RMSE | 差值 | 解释 |
| --- | ---: | ---: | ---: | --- |
| clean | 0.1432 | 0.1433 | +0.0001 | 基本打平 |
| EOG 加噪 0.5 | 0.1928 | 0.1926 | -0.0002 | 基本打平 |
| 随机 mask 25% | 0.2059 | 0.2627 | +0.0568 | 融合更差 |
| 通道 mask 25% | 0.1735 | 0.2068 | +0.0333 | 融合更差 |
| 片段 mask 25% | 0.1713 | 0.2110 | +0.0397 | 融合更差 |
| EOG 全零 | 0.4939 | 0.3394 | -0.1545 | 融合明显更稳 |

在 EOG-only 误差最大的样本上，融合模型的 RMSE 差值均为负：

| 场景 | worst samples 上融合相对 EOG-only 的 RMSE 差值 |
| --- | ---: |
| clean | -0.0221 |
| EOG 加噪 0.5 | -0.0721 |
| 随机 mask 25% | -0.0269 |
| 通道 mask 25% | -0.0149 |
| 片段 mask 25% | -0.0494 |
| EOG 全零 | -0.5061 |

可以写成：

```text
EEG 并不能在干净 SEED-VIG PERCLOS 预测中替代 EOG；
但 EEG 残差信息能够修正 EOG-only 的困难样本，并在严重 EOG 缺失时提供退化保护。
```

## 建议论文贡献点

1. 构建了一个面向 SEED-VIG 连续警觉值估计的原始 EEG/EOG Conformer 框架，并采用严格跨被试协议进行评估。
2. 提出了可靠性门控的 EEG/EOG 融合方式，将 PERCLOS 回归作为主任务，将二分类指标作为阈值化工程评估。
3. 系统分析了 SEED-VIG 中 EOG 对 PERCLOS 标签的强代理效应，指出多模态方法必须与 EOG-only 强基线比较。
4. 通过 EOG 退化实验和误差分层分析说明，EEG 残差信息在 EOG-only 高误差样本和严重 EOG 缺失场景中具有补偿价值。

## 不建议写的内容

- 不要声称融合模型在 clean average 上全面超过 EOG-only。
- 不要把 temporal delta 写成主创新点。
- 不要把 Softmax 分类头写成主训练目标。
- 不要直接拿 HMS-TENet 的高分类准确率做简单横向比较，必须说明标签定义、评估协议、分类/回归任务差异。

## 下一步

1. 使用同风格新图替换原始 Delta-Gate 图。
2. 用 `reports/ei_draft_cn_2026-07-11.md` 作为 EI 初稿基础。
3. 后续如果要投稿，再补充正式参考文献格式、图表编号和会议模板。
