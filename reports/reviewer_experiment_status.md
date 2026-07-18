# Reviewer Experiment Status

Generated: 2026-07-13T18:01:22

## Current Interpretation

The minimal EOG-residual-correction experiment is the best regression-oriented fusion result so far. It is numerically better than EOG-only on all averaged metrics, but paired fold tests are still not significant. Use it to support a careful claim: multimodal fusion improves the mean performance over a strong EOG proxy and offers complementary correction/robustness evidence, not a strong statistical-dominance claim.

Paired residual-correction vs EOG-only deltas: Acc +0.0118 (p=0.5785), Macro-F1 +0.0218 (p=0.4792), BalAcc +0.0213 (p=0.4916), RMSE -0.0046 (p=0.4037), Pearson +0.0088 (p=0.4099). Residual correction is also close to EOG-Q: similar Acc, lower RMSE, higher Pearson, but lower F1/BalAcc.

## Local Ablations

| Setting | Folds | Acc | Macro-F1 | BalAcc | RMSE | Pearson | MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| eeg_only_regression | 5 | 0.6378 +/- 0.0728 | 0.5711 +/- 0.0837 | 0.6002 +/- 0.0977 | 0.2345 +/- 0.0336 | 0.5428 +/- 0.1483 | 0.1883 +/- 0.0341 |
| eog_only_regression | 5 | 0.8250 +/- 0.0526 | 0.7883 +/- 0.0787 | 0.7820 +/- 0.0837 | 0.1432 +/- 0.0212 | 0.8536 +/- 0.0343 | 0.1078 +/- 0.0155 |
| eog_only_classification | 5 | 0.8261 +/- 0.0357 | 0.8118 +/- 0.0388 | 0.8191 +/- 0.0360 | 0.3070 +/- 0.0699 | -0.4279 +/- 0.4979 | 0.2513 +/- 0.0613 |
| cross_attention_regression | 5 | 0.7976 +/- 0.0367 | 0.7768 +/- 0.0484 | 0.7828 +/- 0.0523 | 0.1615 +/- 0.0222 | 0.8213 +/- 0.0441 | 0.1286 +/- 0.0224 |
| cross_attention_gate_regression | 5 | 0.8152 +/- 0.0109 | 0.7903 +/- 0.0250 | 0.7902 +/- 0.0368 | 0.1460 +/- 0.0091 | 0.8472 +/- 0.0365 | 0.1101 +/- 0.0123 |
| cross_attention_eog_query_no_gate_regression | 5 | 0.8178 +/- 0.0330 | 0.7942 +/- 0.0464 | 0.7915 +/- 0.0520 | 0.1503 +/- 0.0082 | 0.8249 +/- 0.0501 | 0.1116 +/- 0.0104 |
| cross_attention_eog_query_regression | 5 | 0.8367 +/- 0.0322 | 0.8165 +/- 0.0353 | 0.8104 +/- 0.0330 | 0.1423 +/- 0.0123 | 0.8454 +/- 0.0276 | 0.1054 +/- 0.0099 |
| eog_residual_correction_regression | 5 | 0.8368 +/- 0.0356 | 0.8101 +/- 0.0509 | 0.8033 +/- 0.0585 | 0.1386 +/- 0.0118 | 0.8624 +/- 0.0380 | 0.1032 +/- 0.0141 |
| cross_attention_bidirectional_regression | 5 | 0.8180 +/- 0.0167 | 0.7964 +/- 0.0257 | 0.7947 +/- 0.0325 | 0.1519 +/- 0.0143 | 0.8324 +/- 0.0219 | 0.1132 +/- 0.0130 |
| delta_gate_regression | 5 | 0.8233 +/- 0.0216 | 0.7995 +/- 0.0383 | 0.7977 +/- 0.0487 | 0.1483 +/- 0.0143 | 0.8406 +/- 0.0264 | 0.1096 +/- 0.0136 |
| delta_gate_classification | 5 | 0.8080 +/- 0.0515 | 0.7926 +/- 0.0619 | 0.8020 +/- 0.0649 | 0.3105 +/- 0.0486 | -0.7082 +/- 0.1261 | 0.2530 +/- 0.0445 |
| anchor_residual_delta_regression | 5 | 0.8027 +/- 0.0373 | 0.7894 +/- 0.0349 | 0.7993 +/- 0.0370 | 0.1471 +/- 0.0171 | 0.8398 +/- 0.0508 | 0.1098 +/- 0.0187 |
| anchor_residual_nodelta_regression | 5 | 0.8263 +/- 0.0484 | 0.8043 +/- 0.0572 | 0.8017 +/- 0.0584 | 0.1433 +/- 0.0097 | 0.8508 +/- 0.0340 | 0.1079 +/- 0.0140 |

## Robustness Sweep

Summary: `D:\seedvig-CT\runs\reviewer_eog_corruption_sweep\summary.md`

## Public SOTA Table To Fill With Verified Protocols

| Method | Public code | Protocol | Task | Modalities | Reported metric | Comparable here? |
| --- | --- | --- | --- | --- | --- | --- |
| HMS-TENet | yes | verify | classification/regression | EEG+EOG features | verify | protocol-sensitive |
| MHCL | yes | cross-subject reported | classification | EEG+EOG | verify | compare only by protocol/task |
| E2CF | verify | verify | vigilance estimation | EEG+EOG | verify | citation/protocol needed |
| DeltaGateNet | verify | inter-subject reported | classification | EEG | verify | not direct regression match |
