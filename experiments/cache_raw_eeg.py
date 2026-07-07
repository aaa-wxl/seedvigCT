import argparse
import json
from pathlib import Path

import numpy as np

from seedvig.raw_dataset import (
    DEFAULT_CACHE_DTYPE,
    WINDOW_SECONDS,
    _load_perclos,
    _load_raw_eeg,
    _load_raw_eog,
    discover_raw_seed_vig_files,
    window_and_normalize_eeg,
    window_and_normalize_signal,
)


def build_cache(data_root, cache_dir, dtype=DEFAULT_CACHE_DTYPE):
    data_root = Path(data_root)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for pair in discover_raw_seed_vig_files(data_root):
        eeg, sample_rate = _load_raw_eeg(pair.raw_path)
        perclos = _load_perclos(pair.label_path)
        samples_per_window = int(sample_rate * WINDOW_SECONDS)
        windows = window_and_normalize_eeg(eeg, int(perclos.shape[0]), samples_per_window)
        cache_path = cache_dir / f"{pair.experiment_id}.eeg.npy"
        np.save(cache_path, windows.astype(dtype, copy=False))
        eog, eog_sample_rate = _load_raw_eog(pair.raw_path)
        eog_windows = window_and_normalize_signal(
            eog,
            int(perclos.shape[0]),
            int(eog_sample_rate * WINDOW_SECONDS),
        )
        eog_cache_path = cache_dir / f"{pair.experiment_id}.eog.npy"
        np.save(eog_cache_path, eog_windows.astype(dtype, copy=False))
        written.append(
            {
                "experiment_id": pair.experiment_id,
                "path": cache_path.name,
                "eog_path": eog_cache_path.name,
                "shape": list(windows.shape),
                "eog_shape": list(eog_windows.shape),
                "sample_rate": sample_rate,
                "eog_sample_rate": eog_sample_rate,
                "dtype": str(np.dtype(dtype)),
            }
        )

    manifest = {"data_root": str(data_root), "files": written}
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return written


def main():
    parser = argparse.ArgumentParser(description="Pre-window SEED-VIG raw EEG into .npy cache.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--dtype", default=DEFAULT_CACHE_DTYPE)
    args = parser.parse_args()

    written = build_cache(args.data_root, args.cache_dir, dtype=args.dtype)
    for item in written:
        print(f"{item['experiment_id']}: {item['shape']} -> {item['path']}")
    print(f"cached_files: {len(written)}")


if __name__ == "__main__":
    main()
