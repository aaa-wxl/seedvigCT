import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy.io import savemat


def _write_seed_vig_fixture(root, name="1_20200101_noon.mat", windows=10):
    raw_dir = root / "Raw_Data"
    label_dir = root / "perclos_labels"
    raw_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    eeg_sample_rate = 200
    samples_per_window = eeg_sample_rate * 8
    eeg = np.arange(windows * samples_per_window * 17, dtype=np.float32).reshape(
        windows * samples_per_window,
        17,
    )
    perclos = np.linspace(0.1, 0.9, windows, dtype=np.float32).reshape(-1, 1)
    savemat(
        raw_dir / name,
        {
            "EEG": {
                "data": eeg,
                "sample_rate": np.array([[eeg_sample_rate]], dtype=np.uint16),
                "chn": np.array([["C"] * 17], dtype=object),
                "node_number": np.array([[17]], dtype=np.uint8),
            }
            ,
            "EOG": {
                "eog": np.arange(windows * 1000 * 7, dtype=np.float32).reshape(windows * 1000, 7),
                "eog_h": np.zeros((windows * 1000, 1), dtype=np.float32),
                "eog_v": np.zeros((windows * 1000, 1), dtype=np.float32),
                "eog_config": {
                    "current_sample_rate": np.array([[125]], dtype=np.uint16),
                    "segment_duration": np.array([[8]], dtype=np.uint8),
                    "segment_number": np.array([[windows]], dtype=np.uint16),
                },
                "eog_number": np.array([[windows * 1000]], dtype=np.int32),
            },
        },
    )
    savemat(label_dir / name, {"perclos": perclos})


class RawConformerPipelineTests(unittest.TestCase):
    def test_raw_dataset_returns_session_safe_eeg_sequences(self):
        from seedvig.raw_dataset import RawSeedVIGSequenceDataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_seed_vig_fixture(root)

            dataset = RawSeedVIGSequenceDataset(root, sequence_length=2, split="all")
            sample = dataset[0]

        self.assertEqual(len(dataset), 5)
        self.assertEqual(tuple(sample["eeg"].shape), (2, 17, 1600))
        self.assertEqual(sample["targets"]["class"].item(), 0)
        self.assertAlmostEqual(sample["targets"]["perclos"].item(), 0.18888889, places=6)
        self.assertEqual(sample["experiment_id"], "1_20200101_noon")

    def test_raw_cache_builder_writes_windowed_eeg_and_eog_used_by_dataset(self):
        from experiments.cache_raw_eeg import build_cache
        from seedvig.raw_dataset import RawSeedVIGSequenceDataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            cache_root = root / "cache"
            _write_seed_vig_fixture(data_root)

            written = build_cache(data_root, cache_root)
            uncached = RawSeedVIGSequenceDataset(data_root, sequence_length=2, split="all")
            cached = RawSeedVIGSequenceDataset(
                data_root,
                sequence_length=2,
                split="all",
                cache_dir=cache_root,
                include_eog=True,
            )

            self.assertEqual(len(written), 1)
            self.assertTrue((cache_root / "1_20200101_noon.eeg.npy").exists())
            self.assertTrue((cache_root / "1_20200101_noon.eog.npy").exists())
            self.assertTrue((cache_root / "manifest.json").exists())
            self.assertTrue(torch.allclose(cached[0]["eeg"], uncached[0]["eeg"]))
            self.assertEqual(tuple(cached[0]["eog"].shape), (2, 7, 1000))

    def test_raw_eeg_conformer_outputs_class_and_perclos_predictions(self):
        from seedvig.models import RawEEGConformer

        model = RawEEGConformer(
            eeg_channels=17,
            embedding_dim=32,
            attention_heads=4,
            window_transformer_layers=1,
            temporal_layers=1,
            num_classes=3,
        )
        outputs = model(torch.randn(2, 3, 17, 1600))

        self.assertEqual(tuple(outputs["class_logits"].shape), (2, 3))
        self.assertEqual(tuple(outputs["perclos"].shape), (2, 1))
        self.assertTrue(torch.all(outputs["perclos"] >= 0.0))
        self.assertTrue(torch.all(outputs["perclos"] <= 1.0))

    def test_raw_eeg_conformer_cross_attends_to_eog(self):
        from seedvig.models import RawEEGConformer

        model = RawEEGConformer(
            eeg_channels=17,
            eog_channels=7,
            embedding_dim=32,
            attention_heads=4,
            window_transformer_layers=0,
            temporal_layers=1,
            use_eog_cross_attention=True,
            num_classes=3,
        )
        outputs = model(torch.randn(2, 3, 17, 1600), eog=torch.randn(2, 3, 7, 1000))

        self.assertEqual(tuple(outputs["class_logits"].shape), (2, 3))
        self.assertEqual(tuple(outputs["perclos"].shape), (2, 1))
        self.assertEqual(tuple(outputs["eog_attention_weights"].shape[:2]), (2, 3))

    def test_reference_comparison_reports_deltas_against_existing_experiments(self):
        from seedvig.reference_results import compare_to_references

        comparisons = compare_to_references(
            {
                "test_accuracy": 0.75,
                "test_rmse": 0.14,
                "test_pearson": 0.86,
            },
            split_strategy="within_experiment_5fold",
        )

        concat = next(item for item in comparisons if item["reference"] == "concat_fusion")
        self.assertAlmostEqual(concat["delta_accuracy"], -0.0195, places=4)
        self.assertAlmostEqual(concat["delta_rmse"], 0.0018, places=4)
        self.assertAlmostEqual(concat["delta_pearson"], 0.0103, places=4)

    def test_training_smoke_writes_metrics_with_reference_comparison(self):
        from experiments.train_raw_conformer import run_training

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_root = tmp_path / "data"
            run_dir = tmp_path / "run"
            _write_seed_vig_fixture(data_root)

            metrics = run_training(
                data_root=data_root,
                cache_dir=tmp_path / "cache",
                run_dir=run_dir,
                epochs=1,
                sequence_length=2,
                batch_size=1,
                embedding_dim=16,
                attention_heads=4,
                window_transformer_layers=0,
                temporal_layers=0,
                max_batches=1,
                eval_max_batches=1,
                include_eog=True,
                use_eog_cross_attention=True,
                device="cpu",
            )

            self.assertTrue((run_dir / "final_metrics.json").exists())
            self.assertIn("test_accuracy", metrics)
            self.assertIn("reference_comparisons", metrics)
            self.assertEqual(metrics["reference_comparisons"][0]["reference"], "concat_fusion")

    def test_auto_raw_experiment_builds_group_subject_command_and_summary(self):
        from argparse import Namespace

        from experiments.auto_raw_experiments import build_command, summarize_folds

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            args = Namespace(
                python="python",
                data_root=tmp_path / "data",
                cache_dir=tmp_path / "cache",
                split_strategy="group_subject",
                run_root=tmp_path / "runs",
                prefix="raw_eeg_eog_cross_group_subject_f",
                epochs=20,
                batch_size=4,
                device="cuda",
            )

            command = build_command(args, fold=2)
            self.assertIn("--split-strategy", command)
            self.assertIn("group_subject", command)
            self.assertIn("--use-eog-cross-attention", command)
            self.assertIn(str(tmp_path / "runs" / "raw_eeg_eog_cross_group_subject_f2"), command)

            for fold, accuracy in enumerate((0.5, 0.6)):
                run_dir = args.run_root / f"{args.prefix}{fold}"
                run_dir.mkdir(parents=True)
                (run_dir / "final_metrics.json").write_text(
                    json.dumps(
                        {
                            "test_accuracy": accuracy,
                            "test_macro_f1": accuracy - 0.1,
                            "test_balanced_accuracy": accuracy,
                            "test_mae": 0.2,
                            "test_rmse": 0.25,
                            "test_pearson": 0.6,
                        }
                    ),
                    encoding="utf-8",
                )

            summary = summarize_folds(args.run_root, args.prefix, "group_subject", tmp_path / "state", folds=(0, 1))

            self.assertEqual(summary["fold_count"], 2)
            self.assertAlmostEqual(summary["metrics"]["test_accuracy"]["mean"], 0.55)
            self.assertTrue((tmp_path / "state" / "summary.md").exists())


if __name__ == "__main__":
    unittest.main()
