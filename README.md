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
