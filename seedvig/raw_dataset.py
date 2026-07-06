from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset


WINDOW_SECONDS = 8
LABEL_MODE_CHOICES = ("three_class", "binary")
DEFAULT_CACHE_DTYPE = "float32"


def perclos_to_class(perclos, label_mode="three_class"):
    value = torch.as_tensor(perclos, dtype=torch.float32)
    if label_mode == "binary":
        return (value > 0.35).long()
    if label_mode == "three_class":
        label = torch.zeros_like(value, dtype=torch.long)
        label = torch.where(value > 0.35, torch.ones_like(label), label)
        return torch.where(value > 0.70, torch.full_like(label, 2), label)
    raise ValueError(f"label_mode must be one of: {LABEL_MODE_CHOICES}")


def label_mode_num_classes(label_mode):
    if label_mode == "binary":
        return 2
    if label_mode == "three_class":
        return 3
    raise ValueError(f"label_mode must be one of: {LABEL_MODE_CHOICES}")


@dataclass(frozen=True)
class RawSeedVIGFilePair:
    experiment_id: str
    subject_id: int
    raw_path: Path
    label_path: Path


@dataclass(frozen=True)
class RawSequenceIndex:
    experiment_index: int
    start: int
    session_fold: int


def _subject_id_from_stem(stem):
    return int(stem.split("_", 1)[0])


def discover_raw_seed_vig_files(root_path):
    root = Path(root_path)
    raw_dir = root / "Raw_Data"
    label_dir = root / "perclos_labels"
    for directory in (raw_dir, label_dir):
        if not directory.exists():
            raise FileNotFoundError(f"Required SEED-VIG directory not found: {directory}")

    pairs = []
    for raw_path in sorted(raw_dir.glob("*.mat"), key=lambda item: item.name):
        label_path = label_dir / raw_path.name
        if not label_path.exists():
            raise FileNotFoundError(f"Missing paired PERCLOS file for {raw_path.name}")
        pairs.append(
            RawSeedVIGFilePair(
                experiment_id=raw_path.stem,
                subject_id=_subject_id_from_stem(raw_path.stem),
                raw_path=raw_path,
                label_path=label_path,
            )
        )
    return pairs


def _load_perclos(path):
    mat = loadmat(path)
    if "perclos" not in mat:
        raise KeyError(f"{path} does not contain `perclos`")
    return np.asarray(mat["perclos"], dtype=np.float32).reshape(-1)


def _struct_field(mat_struct, field):
    value = mat_struct[0, 0][field]
    return np.asarray(value)


def _load_raw_eeg(path):
    mat = loadmat(path)
    if "EEG" not in mat:
        raise KeyError(f"{path} does not contain `EEG`")
    eeg_struct = mat["EEG"]
    data = _struct_field(eeg_struct, "data").astype(np.float32, copy=False)
    sample_rate = int(_struct_field(eeg_struct, "sample_rate").squeeze())
    if data.ndim != 2:
        raise ValueError(f"{path}: EEG.data must have shape [samples, channels], got {data.shape}")
    return data, sample_rate


def window_and_normalize_eeg(eeg, window_count, samples_per_window):
    expected_samples = int(window_count * samples_per_window)
    if eeg.shape[0] != expected_samples:
        raise ValueError(
            f"EEG samples {eeg.shape[0]} do not match {window_count} windows "
            f"with {samples_per_window} samples per window"
        )
    windows = eeg.reshape(window_count, samples_per_window, -1).transpose(0, 2, 1)
    mean = windows.mean(axis=-1, keepdims=True)
    std = windows.std(axis=-1, keepdims=True)
    return ((windows - mean) / (std + 1e-6)).astype(np.float32, copy=False)


def _resolve_validation_fold(fold, validation_fold):
    if validation_fold is None:
        validation_fold = (fold + 1) % 5
    validation_fold = int(validation_fold)
    if validation_fold < 0 or validation_fold > 4:
        raise ValueError("validation_fold must be in [0, 4]")
    if validation_fold == fold:
        raise ValueError("validation_fold must differ from test fold")
    return validation_fold


def _allowed(subject_id, session_fold, split, split_strategy, fold, subjects, validation_fold):
    if split == "all":
        return True
    if split_strategy == "within_experiment_5fold":
        is_test = session_fold == fold
        is_val = session_fold == validation_fold
    elif split_strategy == "group_subject":
        is_test = subject_id in set(subjects[fold::5])
        is_val = subject_id in set(subjects[validation_fold::5])
    else:
        raise ValueError("split_strategy must be within_experiment_5fold or group_subject")

    if split == "test":
        return is_test
    if split == "val":
        return is_val
    if split == "train":
        return not is_test and not is_val
    raise ValueError("split must be all, train, val, or test")


class RawSeedVIGSequenceDataset(Dataset):
    """Raw EEG sequences aligned to SEED-VIG PERCLOS windows."""

    def __init__(
        self,
        root_path,
        cache_dir=None,
        sequence_length=8,
        split="train",
        split_strategy="within_experiment_5fold",
        fold=0,
        validation_fold=None,
        label_mode="three_class",
        cache_size=32,
    ):
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        if fold < 0 or fold > 4:
            raise ValueError("fold must be in [0, 4]")

        self.root_path = Path(root_path)
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self.sequence_length = int(sequence_length)
        self.split = split
        self.split_strategy = split_strategy
        self.fold = int(fold)
        self.validation_fold = _resolve_validation_fold(self.fold, validation_fold)
        self.label_mode = label_mode
        self.num_classes = label_mode_num_classes(label_mode)
        self.cache_size = max(1, int(cache_size))
        self.file_pairs = discover_raw_seed_vig_files(self.root_path)
        self.subjects = sorted({pair.subject_id for pair in self.file_pairs})
        self.perclos_by_experiment = [_load_perclos(pair.label_path) for pair in self.file_pairs]
        self.sequence_index = self._build_sequence_index()
        self._cache = OrderedDict()

    def _build_sequence_index(self):
        sequence_index = []
        for experiment_index, pair in enumerate(self.file_pairs):
            window_count = int(self.perclos_by_experiment[experiment_index].shape[0])
            if window_count % 5 != 0:
                raise ValueError(f"{pair.experiment_id}: expected window count divisible by 5")
            session_length = window_count // 5
            for session_fold in range(5):
                if not _allowed(
                    pair.subject_id,
                    session_fold,
                    self.split,
                    self.split_strategy,
                    self.fold,
                    self.subjects,
                    self.validation_fold,
                ):
                    continue
                session_start = session_fold * session_length
                session_stop = session_start + session_length
                last_start = session_stop - self.sequence_length
                for start in range(session_start, last_start + 1):
                    sequence_index.append(RawSequenceIndex(experiment_index, start, session_fold))
        return sequence_index

    def __len__(self):
        return len(self.sequence_index)

    def _experiment(self, experiment_index):
        if experiment_index in self._cache:
            self._cache.move_to_end(experiment_index)
            return self._cache[experiment_index]

        pair = self.file_pairs[experiment_index]
        perclos = np.clip(self.perclos_by_experiment[experiment_index].astype(np.float32), 0.0, 1.0)
        if self.cache_dir is not None:
            cache_path = self.cache_dir / f"{pair.experiment_id}.eeg.npy"
            if cache_path.exists():
                eeg_windows = np.load(cache_path)
                if eeg_windows.shape[0] != perclos.shape[0]:
                    raise ValueError(
                        f"{pair.experiment_id}: cached windows {eeg_windows.shape[0]} "
                        f"do not match PERCLOS windows {perclos.shape[0]}"
                    )
                value = {
                    "eeg_windows": eeg_windows,
                    "perclos": perclos,
                }
                self._cache[experiment_index] = value
                while len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
                return value

        eeg, sample_rate = _load_raw_eeg(pair.raw_path)
        samples_per_window = int(sample_rate * WINDOW_SECONDS)
        value = {
            "eeg_windows": window_and_normalize_eeg(eeg, int(perclos.shape[0]), samples_per_window),
            "perclos": perclos,
        }
        self._cache[experiment_index] = value
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return value

    def __getitem__(self, index):
        item = self.sequence_index[index]
        pair = self.file_pairs[item.experiment_index]
        experiment = self._experiment(item.experiment_index)
        eeg = np.asarray(experiment["eeg_windows"][item.start : item.start + self.sequence_length], dtype=np.float32)

        perclos_sequence = experiment["perclos"][item.start : item.start + self.sequence_length]
        perclos = torch.tensor(float(perclos_sequence[-1]), dtype=torch.float32)
        class_label = perclos_to_class(perclos, label_mode=self.label_mode)

        return {
            "subject_id": torch.tensor(pair.subject_id, dtype=torch.long),
            "session_fold": torch.tensor(item.session_fold, dtype=torch.long),
            "window_index": torch.tensor(item.start, dtype=torch.long),
            "experiment_id": pair.experiment_id,
            "eeg": torch.from_numpy(eeg.copy()).float(),
            "targets": {
                "perclos": perclos,
                "perclos_sequence": torch.from_numpy(perclos_sequence.copy()).float(),
                "class": class_label.long(),
            },
        }
