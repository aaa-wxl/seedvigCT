# SEED-VIG Raw EEG Conformer

Minimal raw EEG-only CNN+Transformer pipeline for SEED-VIG vigilance estimation.

The code reads SEED-VIG raw EEG windows directly from `Raw_Data/*.mat`, aligns
them with `perclos_labels/*.mat`, and reports metrics against the existing
feature-based experiments from `D:\eeg-eog`.

```powershell
& 'C:\Users\ASUS\miniconda3\envs\torch\python.exe' -m experiments.train_raw_conformer `
  --data-root 'D:\eeg-eog\data\SEED-VIG' `
  --split-strategy within_experiment_5fold `
  --fold 0 `
  --run-dir runs\raw_conformer_within_f0 `
  --epochs 20 `
  --batch-size 4 `
  --device cuda
```

Run the stricter cross-subject protocol by changing:

```powershell
--split-strategy group_subject --run-dir runs\raw_conformer_group_f0
```

Smoke test:

```powershell
& 'C:\Users\ASUS\miniconda3\envs\torch\python.exe' -m unittest tests.test_raw_conformer_pipeline -v
```

Build raw EEG cache before full runs:

```powershell
& 'C:\Users\ASUS\miniconda3\envs\torch\python.exe' -m experiments.cache_raw_eeg `
  --data-root 'D:\eeg-eog\data\SEED-VIG' `
  --cache-dir cache\seedvig_raw_eeg
```

Then pass `--cache-dir cache\seedvig_raw_eeg` to training.

EEG+EOG cross-attention run:

```powershell
& 'C:\Users\ASUS\miniconda3\envs\torch\python.exe' -m experiments.train_raw_conformer `
  --data-root 'D:\eeg-eog\data\SEED-VIG' `
  --cache-dir cache\seedvig_raw_eeg `
  --split-strategy within_experiment_5fold `
  --fold 0 `
  --run-dir runs\raw_eeg_eog_cross_f0 `
  --epochs 20 `
  --batch-size 4 `
  --device cuda `
  --use-eog-cross-attention
```

Useful raw binary group-subject ablations:

```powershell
# raw EEG-only
& 'C:\Users\ASUS\miniconda3\envs\torch\python.exe' -m experiments.auto_raw_experiments `
  --input-mode eeg `
  --label-mode binary `
  --prefix raw_eeg_only_binary_group_subject_f `
  --state-dir runs\auto_raw_eeg_only_binary_group_subject `
  --no-use-eog-cross-attention

# raw EOG-only
& 'C:\Users\ASUS\miniconda3\envs\torch\python.exe' -m experiments.auto_raw_experiments `
  --input-mode eog `
  --label-mode binary `
  --prefix raw_eog_only_binary_group_subject_f `
  --state-dir runs\auto_raw_eog_only_binary_group_subject `
  --no-use-eog-cross-attention

# raw EOG-only, CE-only classification objective
& 'C:\Users\ASUS\miniconda3\envs\torch\python.exe' -m experiments.auto_raw_experiments `
  --input-mode eog `
  --label-mode binary `
  --training-objective classification `
  --prefix raw_eog_only_binary_group_subject_cls_f `
  --state-dir runs\auto_raw_eog_only_binary_group_subject_cls `
  --no-use-eog-cross-attention

# raw EOG-only, PERCLOS regression objective; binary metrics use thresholded predictions
& 'C:\Users\ASUS\miniconda3\envs\torch\python.exe' -m experiments.auto_raw_experiments `
  --input-mode eog `
  --label-mode binary `
  --training-objective regression `
  --prefix raw_eog_only_binary_group_subject_reg_f `
  --state-dir runs\auto_raw_eog_only_binary_group_subject_reg `
  --no-use-eog-cross-attention

# raw EEG+EOG cross-attention with temporal delta, EOG gate, and modality dropout
& 'C:\Users\ASUS\miniconda3\envs\torch\python.exe' -m experiments.auto_raw_experiments `
  --input-mode eeg_eog `
  --label-mode binary `
  --prefix raw_eeg_eog_delta_gate_binary_group_subject_f `
  --state-dir runs\auto_raw_eeg_eog_delta_gate_binary_group_subject `
  --use-eog-cross-attention `
  --use-temporal-delta `
  --use-eog-gate `
  --eog-dropout 0.25
```

Auto-run the full raw binary ablation plan after the GPU becomes available:

```powershell
& 'C:\Users\ASUS\miniconda3\envs\torch\python.exe' -m experiments.run_raw_ablation_plan
```

Auto-run and monitor the original group-subject EEG+EOG queue:

```powershell
& 'C:\Users\ASUS\miniconda3\envs\torch\python.exe' -m experiments.auto_raw_experiments
```
