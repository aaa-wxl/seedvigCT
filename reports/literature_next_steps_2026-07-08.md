# Literature-Based Next Steps

Date: 2026-07-08

## Local Position

Current raw model:

| Protocol | Accuracy | RMSE | Pearson |
| --- | ---: | ---: | ---: |
| Within-experiment 5-fold, raw EEG+EOG cross-attention | 0.7732 +/- 0.0260 | 0.1320 +/- 0.0113 | 0.8637 +/- 0.0433 |
| Group-subject 5-fold, raw EEG+EOG cross-attention | 0.7233 +/- 0.0549 | 0.1652 +/- 0.0292 | 0.8145 +/- 0.0530 |

Previous feature baselines from `D:\eeg-eog`:

| Protocol | Accuracy | RMSE | Pearson |
| --- | ---: | ---: | ---: |
| Within-experiment, concat feature fusion | 0.7695 | 0.1382 | 0.8497 |
| Group-subject, concat feature fusion | 0.4912 | 0.2563 | 0.5746 |

Conclusion: raw EEG+raw EOG cross-attention is already the main line. The group-subject gain is large, so the next step is stability and leakage-proof evidence, not a bigger model.

## Recent Literature Signals

- SEED-VIG is built around EEG, forehead EOG, and continuous PERCLOS vigilance labels. Source: https://bcmi.sjtu.edu.cn/home/seed/seed-vig.html
- The original SEED-VIG work already argues that EEG/EOG fusion and temporal dependency are useful for vigilance estimation. Source: https://arxiv.org/abs/1606.07790
- HMS-TENet uses split-band fusion, topological attention, and multi-task learning. It is still a useful feature-based baseline, especially for binary/cross-subject classification. Source: https://www.sciencedirect.com/science/article/pii/S174680942400073X
- TMU-Net uses gated multimodal fusion and uncertainty-weighted multitask learning, supporting reliability-aware EEG/EOG fusion rather than simple concatenation. Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC10891521/
- MHCNN-STF and Modified TSception point to multi-scale temporal/spatial/frequency CNNs as strong lightweight baselines. Sources: https://pmc.ncbi.nlm.nih.gov/articles/PMC11770883/ and https://arxiv.org/abs/2512.21747
- DeltaGateNet emphasizes temporal change/delta modeling and reports a large subject-dependent to subject-independent gap on SEED-VIG. Source: https://arxiv.org/abs/2602.14071
- E2CF focuses on EEG-EOG cross-modal fusion with asymmetric attention and causal feature alignment. Source: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=11106604
- MHCL reports strong subject-dependent and cross-subject results on SEED-VIG, pointing to subject-invariant representation learning as the next hard target. Sources: https://pubmed.ncbi.nlm.nih.gov/40942793/ and https://github.com/xuexiba233/MHCL
- Delay-aware cross-modal knowledge distillation targets deployable EEG/EOG vigilance estimation and is useful later, after the main model is stable. Source: https://pubmed.ncbi.nlm.nih.gov/40675771/

## Recommended Next Order

1. Finish the seed=1 group-subject repeat that is already running.

Acceptance: group-subject remains clearly above the old feature concat baseline, especially Pearson and RMSE.

2. Add the smallest required ablations for the raw model.

Run:

```text
raw EEG-only group_subject
raw EOG-only group_subject
raw EEG+EOG cross-attention group_subject
raw EEG+EOG cross-attention within_experiment
```

This answers the key reviewer question: whether EOG is merely reconstructing PERCLOS or genuinely helping EEG.

3. Add temporal delta modeling before adding a larger Transformer.

Minimal change:

```text
window_embedding[t]
delta_embedding[t] = window_embedding[t] - window_embedding[t-1]
prediction = head([window_embedding[t], delta_embedding[t]])
```

This is the cheapest architecture move aligned with DeltaGateNet and fatigue dynamics.

4. Add EOG reliability gating after delta modeling.

Minimal change:

```text
fused = eeg + gate(eeg, eog) * cross_attention(eeg, eog)
```

This follows the direction of TMU-Net and E2CF without turning the codebase into a new framework.

5. Only after 1-4, try subject-invariant learning.

First candidate: supervised contrastive loss over subject-disjoint batches. Domain adversarial training can wait unless the repeat seed shows unstable group-subject results.

## Paper Story

The clean story is not "we used a bigger CNN+Transformer." It is:

```text
Raw EEG learns the physiological temporal pattern.
Raw EOG provides complementary eye-movement evidence.
Cross-attention lets EEG query EOG only when useful.
Temporal delta/reliability gating improves cross-subject robustness.
```

Do not spend time on standard ViT as the next step. For SEED-VIG, the literature and local results both point to raw signal Conformer-style encoding, cross-modal attention, temporal dynamics, and subject generalization.
