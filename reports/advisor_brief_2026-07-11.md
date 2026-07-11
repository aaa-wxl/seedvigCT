# SEED-VIG Raw EEG/EOG Advisor Brief

Date: 2026-07-11

## One-Sentence Position

This project should be framed as raw EEG/EOG continuous vigilance estimation under strict group-subject evaluation, with regression-only PERCLOS prediction as the main task and classification metrics reported by thresholding the regression output.

## 中文导师口头版

当前建议不要继续堆模型。SEED-VIG 的标签是 PERCLOS，本身来自眼部信息，所以 EOG-only 表现很强是合理现象，也正是需要在论文里解释清楚的地方。我们的主线应改成：在严格跨被试设置下，先证明 EOG 是强代理基线，再说明 raw EEG/EOG 可靠性门控融合在困难样本和 EOG 严重失效时有补偿价值。主任务使用 PERCLOS 回归，分类准确率只作为阈值化后的工程指标。这样比强行说多模态 clean average 超过 EOG-only 更稳，也更容易回应审稿人。

## What To Tell The Advisor

1. The current strongest clean baseline is raw EOG-only regression because SEED-VIG uses PERCLOS as the vigilance label, and PERCLOS is ocular by definition.
2. This does not make multimodal learning meaningless. It means the paper must not claim that EEG+EOG fusion always beats EOG-only on clean average RMSE.
3. The defensible contribution is narrower and stronger: strict evaluation reveals the EOG proxy effect, while EEG/EOG gated fusion gives complementary robustness on difficult or unreliable EOG samples.
4. The main loss should be PERCLOS regression-only: `SmoothL1(y_pred, PERCLOS)`. CE-only and CE+regression multitask loss are ablations, not the main method.

## Recommended Method Name

Use:

```text
Raw EEG-EOG Reliability-Gated Conformer
```

Avoid using `Delta-Gate` as the main method name because the no-delta anchor residual run was better than the temporal-delta version.

## Architecture Figure Edits

The current architecture figure can be used for advisor discussion after these small text edits:

| Current label | Recommended label |
| --- | --- |
| `Softmax` | `Thresholded Class Output / Auxiliary Evaluation` |
| `Output Probabilities` | `Binary Vigilance Metrics` |
| `Temporal Delta` | `Optional Temporal Delta (Ablation)` |
| `Raw EEG-EOG Delta-Gate Conformer Architecture` | `Raw EEG-EOG Reliability-Gated Conformer Architecture` |

Caption draft:

```text
Figure 1. Raw EEG-EOG reliability-gated Conformer. Raw EEG and EOG windows are encoded by lightweight CNN patch encoders. The main output is continuous PERCLOS estimated with a regression head. Binary vigilance metrics are computed by thresholding the PERCLOS prediction. Temporal delta and classification losses are retained only as ablations.
```

## Main Clean Results

All results below are group-subject unless otherwise stated.

| Method | Objective | Acc | F1 | BalAcc | RMSE | Pearson | Note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Feature concat baseline from `D:\eeg-eog` | regression | 0.4912 | - | - | 0.2563 | 0.5746 | old feature baseline |
| Raw EEG+EOG cross-attention | regression | 0.7233 | 0.7241 | 0.7332 | 0.1652 | 0.8145 | first raw fusion result |
| Raw EOG-only | regression | 0.8250 | 0.7883 | 0.7820 | 0.1432 | 0.8536 | same-script clean comparison |
| Raw EEG+EOG delta/gate | regression | 0.8233 | 0.7995 | 0.7977 | 0.1483 | 0.8406 | delta/gate ablation |
| Raw EEG+EOG anchor residual | regression | 0.8027 | 0.7894 | 0.7993 | 0.1471 | 0.8398 | with temporal delta |
| Raw EEG+EOG anchor residual, no-delta | regression | 0.8263 | 0.8043 | 0.8017 | 0.1433 | 0.8508 | best fusion variant |

Interpretation:

```text
Clean average RMSE: EOG-only and no-delta fusion are essentially tied.
Classification operating metrics: no-delta fusion is slightly better on F1/BalAcc.
Regression correlation: EOG-only remains slightly stronger.
```

Important note: `runs/auto_raw_eog_only_binary_group_subject_reg/summary.json` currently summarizes folds 1-4 only. The EOG-only clean row above uses the EOG corruption evaluation script over folds 0-4, which is the fair comparison against no-delta fusion.

## Loss Function Decision

Use regression-only for the main paper:

```text
L_main = SmoothL1(y_perclos, PERCLOS)
```

Report binary accuracy, F1, and balanced accuracy by thresholding the predicted PERCLOS value. Do not train the main model with CE.

Why:

| Objective | Acc | F1 | BalAcc | RMSE | Pearson | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| EOG-only CE-only | 0.8261 | 0.8118 | 0.8191 | 0.3070 | -0.4279 | good classification, bad PERCLOS |
| EEG+EOG delta/gate CE-only | 0.8155 | 0.7984 | 0.8013 | 0.2949 | -0.6462 | bad PERCLOS |
| EOG-only regression | 0.8250 | 0.7883 | 0.7820 | 0.1432 | 0.8536 | main baseline |
| EEG+EOG no-delta regression | 0.8263 | 0.8043 | 0.8017 | 0.1433 | 0.8508 | main fusion model |

Conclusion:

```text
CE-only can optimize thresholded labels but destroys continuous PERCLOS estimation. Multitask loss has literature support, but the local evidence does not justify using it as the main objective.
```

## Robustness Evidence

The useful fusion claim is not clean-average superiority. The useful claim is compensation under unreliable EOG evidence.

| Scenario | EOG RMSE | Fusion RMSE | Delta RMSE | Interpretation |
| --- | ---: | ---: | ---: | --- |
| clean | 0.1432 | 0.1433 | +0.0001 | tied |
| EOG noise 0.5 | 0.1928 | 0.1926 | -0.0002 | tied |
| random mask 25% | 0.2059 | 0.2627 | +0.0568 | fusion worse |
| channel mask 25% | 0.1735 | 0.2068 | +0.0333 | fusion worse |
| segment mask 25% | 0.1713 | 0.2110 | +0.0397 | fusion worse |
| EOG zero | 0.4939 | 0.3394 | -0.1545 | fusion more robust |

Worst EOG-error samples:

| Scenario | Fusion minus EOG RMSE on worst samples |
| --- | ---: |
| clean | -0.0221 |
| EOG noise 0.5 | -0.0721 |
| random mask 25% | -0.0269 |
| channel mask 25% | -0.0149 |
| segment mask 25% | -0.0494 |
| EOG zero | -0.5061 |

Paper claim:

```text
EEG does not replace EOG on clean SEED-VIG PERCLOS prediction. Instead, EEG residual information improves difficult samples where EOG-only prediction is unreliable, and provides a fallback under severe EOG loss.
```

## Suggested Paper Contributions

1. A raw EEG/EOG Conformer-style framework for continuous SEED-VIG PERCLOS estimation under strict group-subject evaluation.
2. A reliability-gated EEG/EOG fusion variant that keeps regression as the main task and reports classification only as an operational metric.
3. An EOG proxy analysis showing that EOG-only is a strong baseline on SEED-VIG, so multimodal methods must be evaluated against EOG-only rather than only EEG-only or feature fusion.
4. A robustness and error-stratification analysis showing that EEG residual information helps on the hardest EOG-only samples and under severe EOG corruption.

## Do Not Claim

- Do not claim the fusion model is clean-average SOTA over EOG-only.
- Do not claim temporal delta is the main innovation.
- Do not present Softmax classification as the primary training objective.
- Do not compare directly against HMS-TENet accuracy without explaining split protocol, label definition, and regression-vs-classification differences.

## Immediate Next Steps

1. Update the architecture figure labels using the edits above.
2. Prepare a 5-page advisor slide deck:
   - Problem and dataset
   - EOG proxy finding
   - Reliability-gated raw model
   - Clean and robustness results
   - EI paper plan and limitations
3. Draft the paper around regression-only PERCLOS estimation.
4. Keep CE-only, multitask loss, temporal delta, and EOG corruption as ablations.

## Reproducibility Commands

Main test suite:

```powershell
& 'C:\Users\ASUS\miniconda3\envs\torch\python.exe' -m unittest tests.test_raw_conformer_pipeline
```

EOG corruption analysis:

```powershell
& 'C:\Users\ASUS\miniconda3\envs\torch\python.exe' -m experiments.evaluate_eog_corruption `
  --run-root runs `
  --out-dir runs\eog_corruption_analysis `
  --folds 0 1 2 3 4 `
  --batch-size 32 `
  --device cuda
```
