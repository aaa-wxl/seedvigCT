# Reviewer Evidence Addendum

## Paired Significance Tests

| Comparison | Metric | Ours | Baseline | Delta | paired t p | Wilcoxon p |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| final_vs_eog | test_rmse | 0.1433 | 0.1432 | 8.36e-05 | 0.9922 | 0.6250 |
| final_vs_eog | test_pearson | 0.8508 | 0.8536 | -0.0028 | 0.8136 | 1.0000 |
| final_vs_eog | test_accuracy | 0.8263 | 0.8250 | 0.0013 | 0.9433 | 1.0000 |
| final_vs_eog | test_macro_f1 | 0.8043 | 0.7883 | 0.0160 | 0.5382 | 0.6250 |
| final_vs_eog | test_balanced_accuracy | 0.8017 | 0.7820 | 0.0197 | 0.4342 | 0.4375 |
| final_vs_cross_attention | test_rmse | 0.1433 | 0.1615 | -0.0182 | 0.1293 | 0.1250 |
| final_vs_cross_attention | test_pearson | 0.8508 | 0.8213 | 0.0295 | 0.1587 | 0.1250 |
| final_vs_cross_attention | test_accuracy | 0.8263 | 0.7976 | 0.0287 | 0.2401 | 0.4375 |
| final_vs_cross_attention | test_macro_f1 | 0.8043 | 0.7768 | 0.0276 | 0.2950 | 0.4375 |
| final_vs_cross_attention | test_balanced_accuracy | 0.8017 | 0.7828 | 0.0188 | 0.4352 | 0.8125 |
| final_vs_cross_attention_gate | test_rmse | 0.1433 | 0.1460 | -0.0027 | 0.3213 | 0.3125 |
| final_vs_cross_attention_gate | test_pearson | 0.8508 | 0.8472 | 0.0036 | 0.4429 | 0.4375 |
| final_vs_cross_attention_gate | test_accuracy | 0.8263 | 0.8152 | 0.0111 | 0.6995 | 1.0000 |
| final_vs_cross_attention_gate | test_macro_f1 | 0.8043 | 0.7903 | 0.0141 | 0.6414 | 0.8125 |
| final_vs_cross_attention_gate | test_balanced_accuracy | 0.8017 | 0.7902 | 0.0115 | 0.6468 | 0.8125 |

## Model Efficiency

| Model | Params | Latency ms/sample | Device |
| --- | ---: | ---: | --- |
| EEG-only | 73539 | 0.849 | cuda |
| EOG-only | 73219 | 0.925 | cuda |
| Cross-attention | 93315 | 1.706 | cuda |
| Cross-attention+gate | 101636 | 2.227 | cuda |
| Final anchor residual | 84998 | 2.098 | cuda |
