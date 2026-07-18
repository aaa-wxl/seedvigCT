# Minimal EyeQueryNet Reviewer Experiments

This report only uses the paper-facing EyeQueryNet model family.

## Mean over five subject-wise folds

| Method | Folds | Acc | Macro-F1 | BalAcc | RMSE | Pearson |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EyeQueryNet | 5 | 0.8367 $\pm$ 0.0322 $\uparrow$ | 0.8165 $\pm$ 0.0353 $\uparrow$ | 0.8104 $\pm$ 0.0330 $\uparrow$ | 0.1423 $\pm$ 0.0123 $\downarrow$ | 0.8454 $\pm$ 0.0276 $\uparrow$ |
| EOG-only | 5 | 0.8250 $\pm$ 0.0526 $\uparrow$ | 0.7883 $\pm$ 0.0787 $\uparrow$ | 0.7820 $\pm$ 0.0837 $\uparrow$ | 0.1432 $\pm$ 0.0212 $\downarrow$ | 0.8536 $\pm$ 0.0343 $\uparrow$ |
| w/o Cross-modal Gate | 5 | 0.8178 $\pm$ 0.0330 $\uparrow$ | 0.7942 $\pm$ 0.0464 $\uparrow$ | 0.7915 $\pm$ 0.0520 $\uparrow$ | 0.1503 $\pm$ 0.0082 $\downarrow$ | 0.8249 $\pm$ 0.0501 $\uparrow$ |
| w/o Temporal Transformer | 5 | 0.7196 $\pm$ 0.0400 $\uparrow$ | 0.6753 $\pm$ 0.0663 $\uparrow$ | 0.6805 $\pm$ 0.0644 $\uparrow$ | 0.1956 $\pm$ 0.0189 $\downarrow$ | 0.7044 $\pm$ 0.0523 $\uparrow$ |
| Window length = 4 | 5 | 0.8013 $\pm$ 0.0261 $\uparrow$ | 0.7744 $\pm$ 0.0272 $\uparrow$ | 0.7678 $\pm$ 0.0244 $\uparrow$ | 0.1639 $\pm$ 0.0109 $\downarrow$ | 0.7963 $\pm$ 0.0394 $\uparrow$ |
| Window length = 16 | 5 | 0.8364 $\pm$ 0.0430 $\uparrow$ | 0.8132 $\pm$ 0.0571 $\uparrow$ | 0.8123 $\pm$ 0.0648 $\uparrow$ | 0.1463 $\pm$ 0.0255 $\downarrow$ | 0.8438 $\pm$ 0.0407 $\uparrow$ |
| Window length = 1 | 5 | 0.7686 $\pm$ 0.0451 $\uparrow$ | 0.7268 $\pm$ 0.0726 $\uparrow$ | 0.7238 $\pm$ 0.0741 $\uparrow$ | 0.1834 $\pm$ 0.0131 $\downarrow$ | 0.7259 $\pm$ 0.0784 $\uparrow$ |
| EEG-only | 5 | 0.6378 $\pm$ 0.0728 $\uparrow$ | 0.5711 $\pm$ 0.0837 $\uparrow$ | 0.6002 $\pm$ 0.0977 $\uparrow$ | 0.2345 $\pm$ 0.0336 $\downarrow$ | 0.5428 $\pm$ 0.1483 $\uparrow$ |

## Per-fold table

| Method | Fold | Acc | Macro-F1 | BalAcc | RMSE | Pearson |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EyeQueryNet | 0 | 0.7885 | 0.7678 | 0.7670 | 0.1545 | 0.8624 |
| EyeQueryNet | 1 | 0.8765 | 0.8584 | 0.8480 | 0.1447 | 0.8413 |
| EyeQueryNet | 2 | 0.8682 | 0.8456 | 0.8299 | 0.1382 | 0.8852 |
| EyeQueryNet | 3 | 0.8221 | 0.7829 | 0.7746 | 0.1207 | 0.8030 |
| EyeQueryNet | 4 | 0.8280 | 0.8280 | 0.8325 | 0.1533 | 0.8349 |
| EOG-only | 0 | 0.7464 | 0.6749 | 0.6661 | 0.1828 | 0.8507 |
| EOG-only | 1 | 0.9026 | 0.8907 | 0.8866 | 0.1270 | 0.8789 |
| EOG-only | 2 | 0.7974 | 0.7305 | 0.7111 | 0.1459 | 0.8858 |
| EOG-only | 3 | 0.8245 | 0.7913 | 0.7895 | 0.1240 | 0.7895 |
| EOG-only | 4 | 0.8541 | 0.8539 | 0.8567 | 0.1364 | 0.8632 |
| w/o Cross-modal Gate | 0 | 0.7882 | 0.7525 | 0.7406 | 0.1556 | 0.8632 |
| w/o Cross-modal Gate | 1 | 0.8676 | 0.8562 | 0.8648 | 0.1390 | 0.8575 |
| w/o Cross-modal Gate | 2 | 0.8365 | 0.8126 | 0.8037 | 0.1598 | 0.8477 |
| w/o Cross-modal Gate | 3 | 0.7762 | 0.7297 | 0.7250 | 0.1421 | 0.7276 |
| w/o Cross-modal Gate | 4 | 0.8202 | 0.8201 | 0.8232 | 0.1552 | 0.8285 |
| w/o Temporal Transformer | 0 | 0.6727 | 0.5591 | 0.5730 | 0.2187 | 0.7234 |
| w/o Temporal Transformer | 1 | 0.7441 | 0.7327 | 0.7538 | 0.1940 | 0.7611 |
| w/o Temporal Transformer | 2 | 0.6724 | 0.6460 | 0.6509 | 0.2041 | 0.7422 |
| w/o Temporal Transformer | 3 | 0.7708 | 0.7041 | 0.6909 | 0.1615 | 0.6143 |
| w/o Temporal Transformer | 4 | 0.7379 | 0.7347 | 0.7338 | 0.1996 | 0.6810 |
| Window length = 4 | 0 | 0.7708 | 0.7350 | 0.7253 | 0.1802 | 0.8239 |
| Window length = 4 | 1 | 0.8259 | 0.7949 | 0.7810 | 0.1643 | 0.8011 |
| Window length = 4 | 2 | 0.8382 | 0.8028 | 0.7822 | 0.1600 | 0.8425 |
| Window length = 4 | 3 | 0.7814 | 0.7488 | 0.7567 | 0.1468 | 0.7274 |
| Window length = 4 | 4 | 0.7903 | 0.7903 | 0.7936 | 0.1683 | 0.7864 |
| Window length = 16 | 0 | 0.7854 | 0.7259 | 0.7099 | 0.1927 | 0.8332 |
| Window length = 16 | 1 | 0.8620 | 0.8507 | 0.8595 | 0.1346 | 0.8660 |
| Window length = 16 | 2 | 0.9049 | 0.8953 | 0.8996 | 0.1237 | 0.9099 |
| Window length = 16 | 3 | 0.8279 | 0.7931 | 0.7890 | 0.1265 | 0.7926 |
| Window length = 16 | 4 | 0.8015 | 0.8009 | 0.8036 | 0.1541 | 0.8173 |
| Window length = 1 | 0 | 0.7037 | 0.6165 | 0.6173 | 0.2038 | 0.7479 |
| Window length = 1 | 1 | 0.7757 | 0.7431 | 0.7365 | 0.1905 | 0.7178 |
| Window length = 1 | 2 | 0.8429 | 0.8273 | 0.8299 | 0.1763 | 0.8131 |
| Window length = 1 | 3 | 0.7494 | 0.6797 | 0.6696 | 0.1649 | 0.5818 |
| Window length = 1 | 4 | 0.7713 | 0.7671 | 0.7658 | 0.1813 | 0.7688 |
| EEG-only | 0 | 0.6988 | 0.6756 | 0.6789 | 0.2301 | 0.6638 |
| EEG-only | 1 | 0.6538 | 0.6538 | 0.7306 | 0.2455 | 0.6925 |
| EEG-only | 2 | 0.5132 | 0.4480 | 0.4496 | 0.2380 | 0.6018 |
| EEG-only | 3 | 0.7158 | 0.5458 | 0.5624 | 0.1774 | 0.4651 |
| EEG-only | 4 | 0.6075 | 0.5324 | 0.5797 | 0.2817 | 0.2910 |

## EOG corruption robustness

| Scenario | EOG RMSE | EyeQueryNet RMSE | Delta RMSE | EOG BalAcc | EyeQueryNet BalAcc | Delta BalAcc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 0.1432 | 0.1423 | -0.0009 | 0.7820 | 0.8104 | +0.0284 |
| noise_0.5 | 0.1928 | 0.1664 | -0.0265 | 0.7768 | 0.8103 | +0.0335 |
| channel_mask_0.25 | 0.1735 | 0.2314 | +0.0579 | 0.7542 | 0.7364 | -0.0178 |
| zero | 0.4939 | 0.3656 | -0.1283 | 0.5000 | 0.5000 | +0.0000 |

## Late fusion sanity baseline

| Method | Acc | Macro-F1 | BalAcc | RMSE | Pearson | Mean EOG weight |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Val-selected EEG/EOG output fusion | 0.8290 | 0.7930 | 0.7877 | 0.1411 | 0.8591 | 0.90 |

## EEG branch perturbation

| Scenario | Acc | BalAcc | RMSE | Pearson | Delta RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| none | 0.8367 | 0.8104 | 0.1423 | 0.8454 | +0.0000 |
| zero | 0.8366 | 0.8103 | 0.1423 | 0.8454 | +0.0000 |
| paired_shuffle | 0.8366 | 0.8103 | 0.1423 | 0.8454 | +0.0000 |

Internal note: near-zero deltas mean this checkpoint is almost insensitive to EEG perturbation; do not use this as positive evidence for complementary neural EEG information.
