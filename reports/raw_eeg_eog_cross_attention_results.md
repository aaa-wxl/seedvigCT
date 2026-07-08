# Raw EEG+EOG Cross-Attention Results

Date: 2026-07-08

Model: raw EEG CNN/Conformer tokens with raw EOG cross-attention.

## Within-Experiment 5-Fold

Run prefix: `runs/raw_eeg_eog_cross_f*`

| Metric | Mean +/- Std |
| --- | --- |
| Accuracy | 0.7732 +/- 0.0260 |
| Macro F1 | 0.7700 +/- 0.0330 |
| Balanced Accuracy | 0.7802 +/- 0.0311 |
| MAE | 0.1005 +/- 0.0087 |
| RMSE | 0.1320 +/- 0.0113 |
| Pearson | 0.8637 +/- 0.0433 |

Compared with the previous feature concat model: accuracy +0.0037, RMSE -0.0062, Pearson +0.0140.

## Group-Subject 5-Fold

Run prefix: `runs/raw_eeg_eog_cross_group_subject_f*`

| Metric | Mean +/- Std |
| --- | --- |
| Accuracy | 0.7233 +/- 0.0549 |
| Macro F1 | 0.7241 +/- 0.0591 |
| Balanced Accuracy | 0.7332 +/- 0.0431 |
| MAE | 0.1293 +/- 0.0297 |
| RMSE | 0.1652 +/- 0.0292 |
| Pearson | 0.8145 +/- 0.0530 |

Compared with the previous feature concat group-subject model: accuracy +0.2321, RMSE -0.0911, Pearson +0.2399.

## Split Audit

`runs/auto_raw_eeg_eog_cross_group_subject/split_audit.json` confirms train/validation/test subjects are disjoint for all group-subject folds.

Next minimal experiment: repeat group-subject once with a different seed before treating the gain as final.
