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

    def test_raw_eeg_conformer_supports_temporal_delta_and_eog_gate(self):
        from seedvig.models import RawEEGConformer

        model = RawEEGConformer(
            eeg_channels=17,
            eog_channels=7,
            embedding_dim=32,
            attention_heads=4,
            window_transformer_layers=0,
            temporal_layers=0,
            use_eog_cross_attention=True,
            use_temporal_delta=True,
            use_eog_gate=True,
            eog_dropout=1.0,
            num_classes=2,
        )
        model.train()
        outputs = model(torch.randn(2, 3, 17, 1600), eog=torch.randn(2, 3, 7, 1000))

        self.assertEqual(tuple(outputs["class_logits"].shape), (2, 2))
        self.assertEqual(tuple(outputs["perclos"].shape), (2, 1))
        self.assertEqual(tuple(outputs["eog_gate"].shape), (2, 3, 1))

    def test_raw_eeg_conformer_supports_eog_anchor_residual_fusion(self):
        from seedvig.models import RawEEGConformer

        model = RawEEGConformer(
            eeg_channels=17,
            eog_channels=7,
            embedding_dim=32,
            attention_heads=4,
            window_transformer_layers=0,
            temporal_layers=0,
            use_temporal_delta=True,
            use_eog_anchor_residual=True,
            num_classes=2,
        )
        outputs = model(torch.randn(2, 3, 17, 1600), eog=torch.randn(2, 3, 7, 1000))

        expected = outputs["perclos_eog"] + outputs["eog_residual_gate"] * (
            outputs["perclos_eeg"] - outputs["perclos_eog"]
        )
        self.assertEqual(tuple(outputs["class_logits"].shape), (2, 2))
        self.assertEqual(tuple(outputs["perclos"].shape), (2, 1))
        self.assertEqual(tuple(outputs["perclos_eeg"].shape), (2, 1))
        self.assertEqual(tuple(outputs["perclos_eog"].shape), (2, 1))
        self.assertEqual(tuple(outputs["eog_residual_gate"].shape), (2, 1))
        self.assertTrue(torch.allclose(outputs["perclos"], expected))
        self.assertTrue(torch.all(outputs["perclos"] >= 0.0))
        self.assertTrue(torch.all(outputs["perclos"] <= 1.0))

    def test_eog_corruption_preserves_eeg_and_masks_eog(self):
        from experiments.evaluate_eog_corruption import corrupt_batch

        batch = {
            "eeg": torch.ones(2, 3, 17, 4),
            "eog": torch.ones(2, 3, 7, 4),
        }
        corrupted = corrupt_batch(batch, kind="mask", value=0.5, seed=0)

        self.assertTrue(torch.equal(corrupted["eeg"], batch["eeg"]))
        self.assertFalse(torch.equal(corrupted["eog"], batch["eog"]))
        self.assertEqual(corrupted["eog"].shape, batch["eog"].shape)

    def test_worst_error_strata_reports_fusion_delta(self):
        from experiments.evaluate_eog_corruption import worst_error_strata

        rows = worst_error_strata(
            eog_predictions=torch.tensor([0.0, 0.2, 0.9, 0.9]),
            fusion_predictions=torch.tensor([0.0, 0.2, 0.6, 0.7]),
            targets=torch.tensor([0.0, 0.0, 0.5, 0.5]),
            bins=2,
        )

        self.assertEqual(rows[-1]["stratum"], "worst_50%")
        self.assertLess(rows[-1]["fusion_rmse"], rows[-1]["eog_rmse"])
        self.assertLess(rows[-1]["delta_rmse"], 0.0)

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

    def test_training_smoke_supports_raw_eog_only_input_mode(self):
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
                input_mode="eog",
                label_mode="binary",
                device="cpu",
            )

            self.assertTrue((run_dir / "final_metrics.json").exists())
            self.assertEqual(metrics["input_mode"], "eog")
            self.assertEqual(metrics["training_objective"], "regression")
            self.assertEqual(metrics["selection_metric"], "val_rmse")

    def test_training_smoke_supports_classification_objective_auto_selection(self):
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
                input_mode="eog",
                label_mode="binary",
                training_objective="classification",
                device="cpu",
            )

            self.assertEqual(metrics["training_objective"], "classification")
            self.assertEqual(metrics["selection_metric"], "val_balanced_accuracy")
            self.assertIn("best_selection_score", metrics)

    def test_training_smoke_supports_regression_objective_auto_selection(self):
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
                input_mode="eog",
                label_mode="binary",
                training_objective="regression",
                device="cpu",
            )

            self.assertEqual(metrics["training_objective"], "regression")
            self.assertEqual(metrics["selection_metric"], "val_rmse")
            self.assertIn("test_accuracy", metrics)

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
                label_mode="binary",
                input_mode="eeg",
                run_root=tmp_path / "runs",
                prefix="raw_eeg_eog_cross_group_subject_f",
                epochs=20,
                batch_size=4,
                seed=1,
                device="cuda",
                training_objective="classification",
                regression_weight=0.5,
                selection_metric="val_balanced_accuracy",
                use_eog_cross_attention=False,
                use_temporal_delta=True,
                use_eog_gate=False,
                use_eog_anchor_residual=True,
                eog_dropout=0.0,
            )

            command = build_command(args, fold=2)
            self.assertIn("--split-strategy", command)
            self.assertIn("group_subject", command)
            self.assertIn("--label-mode", command)
            self.assertIn("binary", command)
            self.assertIn("--input-mode", command)
            self.assertIn("eeg", command)
            self.assertIn("--seed", command)
            self.assertIn("1", command)
            self.assertIn("--training-objective", command)
            self.assertIn("classification", command)
            self.assertIn("--selection-metric", command)
            self.assertIn("val_balanced_accuracy", command)
            self.assertIn("--use-temporal-delta", command)
            self.assertIn("--use-eog-anchor-residual", command)
            self.assertNotIn("--use-eog-cross-attention", command)
            self.assertIn(str(tmp_path / "runs" / "raw_eeg_eog_cross_group_subject_f2"), command)

            args.training_objective = "multitask"
            args.selection_metric = "auto"
            command = build_command(args, fold=2)
            self.assertIn("--training-objective", command)
            self.assertIn("multitask", command)

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

    def test_raw_ablation_plan_builds_eeg_eog_and_delta_gate_commands(self):
        from argparse import Namespace

        from experiments.run_raw_ablation_plan import EXPERIMENTS, build_auto_command

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            args = Namespace(
                python="python",
                cwd=tmp_path,
                data_root=tmp_path / "data",
                cache_dir=tmp_path / "cache",
                run_root=tmp_path / "runs",
                epochs=20,
                batch_size=4,
                seed=0,
                device="cuda",
                poll_seconds=30,
                training_objective="classification",
                regression_weight=0.5,
                selection_metric="val_balanced_accuracy",
            )

            eeg_command = build_auto_command(args, EXPERIMENTS[0])
            eog_command = build_auto_command(args, EXPERIMENTS[1])
            delta_gate_command = build_auto_command(args, EXPERIMENTS[2])

            self.assertIn("raw_eeg_only_binary_group_subject_f", eeg_command)
            self.assertIn("eeg", eeg_command)
            self.assertIn("--training-objective", eeg_command)
            self.assertIn("classification", eeg_command)
            self.assertIn("--selection-metric", eeg_command)
            self.assertIn("val_balanced_accuracy", eeg_command)
            self.assertIn("--no-use-eog-cross-attention", eeg_command)
            self.assertIn("raw_eog_only_binary_group_subject_f", eog_command)
            self.assertIn("eog", eog_command)
            self.assertIn("--no-use-eog-cross-attention", eog_command)
            self.assertIn("raw_eeg_eog_delta_gate_binary_group_subject_f", delta_gate_command)
            self.assertIn("eeg_eog", delta_gate_command)
            self.assertIn("--use-eog-cross-attention", delta_gate_command)
            self.assertIn("--use-temporal-delta", delta_gate_command)
            self.assertIn("--use-eog-gate", delta_gate_command)
            self.assertIn("--eog-dropout", delta_gate_command)

            args.training_objective = "multitask"
            args.selection_metric = "auto"
            eeg_command = build_auto_command(args, EXPERIMENTS[0])
            self.assertIn("--training-objective", eeg_command)
            self.assertIn("multitask", eeg_command)


if __name__ == "__main__":
    unittest.main()
