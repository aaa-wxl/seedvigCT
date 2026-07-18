import argparse
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

METRICS = (
    ("test_accuracy", "Acc"),
    ("test_macro_f1", "Macro-F1"),
    ("test_balanced_accuracy", "BalAcc"),
    ("test_rmse", "RMSE"),
    ("test_pearson", "Pearson"),
)

VARIANTS = (
    {
        "name": "EyeQueryNet",
        "prefix": "raw_eeg_eog_cross_eog_query_binary_group_subject_reg_f",
        "state_dir": Path(r"runs\auto_raw_eeg_eog_cross_eog_query_binary_group_subject_reg"),
        "input_mode": "eeg_eog",
        "flags": ("--use-eog-cross-attention", "--cross-attention-direction", "eog_queries_eeg", "--use-eog-gate"),
    },
    {
        "name": "EOG-only",
        "prefix": "raw_eog_only_binary_group_subject_reg_f",
        "state_dir": Path(r"runs\auto_raw_eog_only_binary_group_subject_reg"),
        "input_mode": "eog",
        "flags": ("--no-use-eog-cross-attention",),
    },
    {
        "name": "w/o Cross-modal Gate",
        "prefix": "raw_eeg_eog_cross_eog_query_nogate_binary_group_subject_reg_f",
        "state_dir": Path(r"runs\auto_raw_eeg_eog_cross_eog_query_nogate_binary_group_subject_reg"),
        "input_mode": "eeg_eog",
        "flags": ("--use-eog-cross-attention", "--cross-attention-direction", "eog_queries_eeg"),
    },
    {
        "name": "w/o Temporal Transformer",
        "prefix": "raw_eeg_eog_cross_eog_query_notemporal_binary_group_subject_reg_f",
        "state_dir": Path(r"runs\auto_raw_eeg_eog_cross_eog_query_notemporal_binary_group_subject_reg"),
        "input_mode": "eeg_eog",
        "temporal_layers": 0,
        "flags": ("--use-eog-cross-attention", "--cross-attention-direction", "eog_queries_eeg", "--use-eog-gate"),
    },
    {
        "name": "Window length = 4",
        "prefix": "raw_eeg_eog_cross_eog_query_seq4_binary_group_subject_reg_f",
        "state_dir": Path(r"runs\auto_raw_eeg_eog_cross_eog_query_seq4_binary_group_subject_reg"),
        "input_mode": "eeg_eog",
        "sequence_length": 4,
        "flags": ("--use-eog-cross-attention", "--cross-attention-direction", "eog_queries_eeg", "--use-eog-gate"),
    },
    {
        "name": "Window length = 16",
        "prefix": "raw_eeg_eog_cross_eog_query_seq16_binary_group_subject_reg_f",
        "state_dir": Path(r"runs\auto_raw_eeg_eog_cross_eog_query_seq16_binary_group_subject_reg"),
        "input_mode": "eeg_eog",
        "sequence_length": 16,
        "flags": ("--use-eog-cross-attention", "--cross-attention-direction", "eog_queries_eeg", "--use-eog-gate"),
    },
    {
        "name": "Window length = 1",
        "prefix": "raw_eeg_eog_cross_eog_query_seq1_binary_group_subject_reg_f",
        "state_dir": Path(r"runs\auto_raw_eeg_eog_cross_eog_query_seq1_binary_group_subject_reg"),
        "input_mode": "eeg_eog",
        "sequence_length": 1,
        "flags": ("--use-eog-cross-attention", "--cross-attention-direction", "eog_queries_eeg", "--use-eog-gate"),
    },
    {
        "name": "EEG-only",
        "prefix": "raw_eeg_only_binary_group_subject_f",
        "state_dir": Path(r"runs\auto_raw_eeg_only_binary_group_subject"),
        "input_mode": "eeg",
        "flags": ("--no-use-eog-cross-attention",),
    },
)


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_rows(run_root, variant, folds):
    rows = []
    missing = []
    for fold in folds:
        path = run_root / f"{variant['prefix']}{fold}" / "final_metrics.json"
        if path.exists():
            rows.append({"method": variant["name"], "fold": fold, **_read_json(path)})
        else:
            missing.append(fold)
    return rows, missing


def _mean_std(rows, metric):
    values = [float(row[metric]) for row in rows if metric in row]
    if not values:
        return None
    return {"mean": statistics.mean(values), "std": statistics.pstdev(values)}


def _summarize(run_root, folds):
    methods = []
    missing = {}
    for variant in VARIANTS:
        rows, missing_folds = _load_rows(run_root, variant, folds)
        missing[variant["name"]] = missing_folds
        methods.append(
            {
                "method": variant["name"],
                "fold_count": len(rows),
                "folds": rows,
                "summary": {label: _mean_std(rows, key) for key, label in METRICS},
            }
        )
    return {"methods": methods, "missing": missing}


def _fmt(value, lower_is_better=False):
    if value is None:
        return "-"
    arrow = r"$\downarrow$" if lower_is_better else r"$\uparrow$"
    return f"{value['mean']:.4f} $\\pm$ {value['std']:.4f} {arrow}"


def _metric_summary(rows, keys):
    return {key: _mean_std(rows, key) for key in keys}


def _write_report(path, result, robustness_summary, supplement_summary):
    lines = [
        "# Minimal EyeQueryNet Reviewer Experiments",
        "",
        "This report only uses the paper-facing EyeQueryNet model family.",
        "",
        "## Mean over five subject-wise folds",
        "",
        "| Method | Folds | Acc | Macro-F1 | BalAcc | RMSE | Pearson |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in result["methods"]:
        summary = method["summary"]
        lines.append(
            f"| {method['method']} | {method['fold_count']} | "
            f"{_fmt(summary['Acc'])} | {_fmt(summary['Macro-F1'])} | "
            f"{_fmt(summary['BalAcc'])} | {_fmt(summary['RMSE'], True)} | {_fmt(summary['Pearson'])} |"
        )

    lines.extend(
        [
            "",
            "## Per-fold table",
            "",
            "| Method | Fold | Acc | Macro-F1 | BalAcc | RMSE | Pearson |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for method in result["methods"]:
        for row in method["folds"]:
            lines.append(
                f"| {method['method']} | {row['fold']} | "
                f"{row.get('test_accuracy', 0):.4f} | {row.get('test_macro_f1', 0):.4f} | "
                f"{row.get('test_balanced_accuracy', 0):.4f} | {row.get('test_rmse', 0):.4f} | "
                f"{row.get('test_pearson', 0):.4f} |"
            )

    if robustness_summary:
        lines.extend(
            [
                "",
                "## EOG corruption robustness",
                "",
                "| Scenario | EOG RMSE | EyeQueryNet RMSE | Delta RMSE | EOG BalAcc | EyeQueryNet BalAcc | Delta BalAcc |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for scenario, metrics in robustness_summary.items():
            lines.append(
                f"| {scenario} | {metrics['eog_rmse']['mean']:.4f} | {metrics['fusion_rmse']['mean']:.4f} | "
                f"{metrics['delta_rmse']['mean']:+.4f} | {metrics['eog_balanced_accuracy']['mean']:.4f} | "
                f"{metrics['fusion_balanced_accuracy']['mean']:.4f} | {metrics['delta_balanced_accuracy']['mean']:+.4f} |"
            )

    if supplement_summary:
        late = supplement_summary.get("late_fusion", {})
        if late:
            summary = late["summary"]
            lines.extend(
                [
                    "",
                    "## Late fusion sanity baseline",
                    "",
                    "| Method | Acc | Macro-F1 | BalAcc | RMSE | Pearson | Mean EOG weight |",
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                    (
                        f"| Val-selected EEG/EOG output fusion | {summary['accuracy']['mean']:.4f} | "
                        f"{summary['macro_f1']['mean']:.4f} | {summary['balanced_accuracy']['mean']:.4f} | "
                        f"{summary['rmse']['mean']:.4f} | {summary['pearson']['mean']:.4f} | "
                        f"{summary['eog_weight']['mean']:.2f} |"
                    ),
                ]
            )

        eeg = supplement_summary.get("eeg_perturbation", {})
        if eeg:
            lines.extend(
                [
                    "",
                    "## EEG branch perturbation",
                    "",
                    "| Scenario | Acc | BalAcc | RMSE | Pearson | Delta RMSE |",
                    "| --- | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for scenario, summary in eeg["summary"].items():
                lines.append(
                    f"| {scenario} | {summary['accuracy']['mean']:.4f} | "
                    f"{summary['balanced_accuracy']['mean']:.4f} | {summary['rmse']['mean']:.4f} | "
                    f"{summary['pearson']['mean']:.4f} | {summary.get('delta_rmse', {'mean': 0.0})['mean']:+.4f} |"
                )
            lines.append("")
            lines.append(
                "Internal note: near-zero deltas mean this checkpoint is almost insensitive to EEG perturbation; "
                "do not use this as positive evidence for complementary neural EEG information."
            )

    missing = {name: folds for name, folds in result["missing"].items() if folds}
    if missing:
        lines.extend(["", "## Missing folds", "", "```json", json.dumps(missing, indent=2), "```"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_train_command(args, variant):
    return [
        args.python,
        "-m",
        "experiments.auto_raw_experiments",
        "--cwd",
        str(args.cwd),
        "--data-root",
        str(args.data_root),
        "--cache-dir",
        str(args.cache_dir),
        "--run-root",
        str(args.run_root),
        "--state-dir",
        str(variant["state_dir"]),
        "--prefix",
        variant["prefix"],
        "--split-strategy",
        "group_subject",
        "--label-mode",
        "binary",
        "--input-mode",
        variant["input_mode"],
        "--folds",
        *[str(fold) for fold in args.folds],
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--sequence-length",
        str(variant.get("sequence_length", args.sequence_length)),
        "--temporal-layers",
        str(variant.get("temporal_layers", args.temporal_layers)),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--training-objective",
        "regression",
        *variant["flags"],
    ]


def _run_missing(args, result):
    missing_by_name = {name: folds for name, folds in result["missing"].items() if folds}
    if not missing_by_name:
        return
    for variant in VARIANTS:
        if variant["name"] not in missing_by_name:
            continue
        command = _build_train_command(args, variant)
        variant["state_dir"].mkdir(parents=True, exist_ok=True)
        (variant["state_dir"] / "minimal_command.json").write_text(
            json.dumps(command, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        subprocess.run(command, cwd=args.cwd, check=True)


def _run_robustness(args):
    if not args.run_robustness:
        path = args.robustness_out_dir / "summary.json"
        return _read_json(path) if path.exists() else None
    from experiments.evaluate_eog_corruption import evaluate_pair

    result = evaluate_pair(
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        run_root=args.run_root,
        eog_prefix="raw_eog_only_binary_group_subject_reg_f",
        fusion_prefix="raw_eeg_eog_cross_eog_query_binary_group_subject_reg_f",
        out_dir=args.robustness_out_dir,
        folds=args.folds,
        corruptions=[("none", 0.0), ("noise", 0.5), ("channel_mask", 0.25), ("zero", 0.0)],
        batch_size=args.robustness_batch_size,
        device=args.device,
        seed=args.seed,
        seeds=args.robustness_seeds,
        max_batches=args.max_batches,
    )
    return result["summary"]


def _load_model_and_metrics(run_root, prefix, fold, input_mode, device):
    from experiments.evaluate_eog_corruption import _load_model

    run_dir = run_root / f"{prefix}{fold}"
    metrics = _read_json(run_dir / "final_metrics.json")
    return _load_model(run_dir, metrics, input_mode, device), metrics


def _dataset(args, metrics, split, fold):
    from seedvig.raw_dataset import RawSeedVIGSequenceDataset

    return RawSeedVIGSequenceDataset(
        args.data_root,
        cache_dir=args.cache_dir,
        include_eog=True,
        sequence_length=int(metrics.get("sequence_length", 8)),
        split=split,
        split_strategy=metrics.get("split_strategy", "group_subject"),
        fold=int(fold),
        label_mode=metrics.get("label_mode", "binary"),
    )


def _collect(model, loader, device, input_mode, eeg_scenario="none"):
    import torch
    from experiments.evaluate_eog_corruption import _metrics
    from experiments.train_raw_conformer import _move_to_device

    predictions = []
    targets = []
    with torch.no_grad():
        for batch in loader:
            if eeg_scenario == "zero":
                batch["eeg"] = torch.zeros_like(batch["eeg"])
            elif eeg_scenario == "paired_shuffle" and batch["eeg"].shape[0] > 1:
                batch["eeg"] = torch.roll(batch["eeg"], shifts=1, dims=0)
            batch = _move_to_device(batch, device)
            if input_mode == "eog":
                outputs = model(batch["eog"])
            elif input_mode == "eeg":
                outputs = model(batch["eeg"])
            else:
                outputs = model(batch["eeg"], eog=batch["eog"])
            predictions.append(outputs["perclos"].squeeze(-1).detach().cpu())
            targets.append(batch["targets"]["perclos"].float().detach().cpu())
    predictions = torch.cat(predictions)
    targets = torch.cat(targets)
    return predictions, targets, _metrics(predictions, targets, loader.dataset.label_mode)


def _summarize_scenarios(rows):
    scenarios = {}
    for row in rows:
        scenarios.setdefault(row["scenario"], []).append(row)
    return {
        scenario: _metric_summary(items, ("accuracy", "macro_f1", "balanced_accuracy", "rmse", "pearson", "delta_rmse"))
        for scenario, items in scenarios.items()
    }


def _run_supplement(args):
    path = args.supplement_dir / "summary.json"
    if not args.run_supplement and path.exists():
        return _read_json(path)
    if not args.run_supplement:
        return None

    import torch
    from torch.utils.data import DataLoader
    from experiments.evaluate_eog_corruption import _metrics

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    args.supplement_dir.mkdir(parents=True, exist_ok=True)

    late_rows = []
    eeg_rows = []
    for fold in args.folds:
        eeg_model, eeg_metrics = _load_model_and_metrics(
            args.run_root, "raw_eeg_only_binary_group_subject_f", fold, "eeg", device
        )
        eog_model, eog_metrics = _load_model_and_metrics(
            args.run_root, "raw_eog_only_binary_group_subject_reg_f", fold, "eog", device
        )
        fusion_model, fusion_metrics = _load_model_and_metrics(
            args.run_root, "raw_eeg_eog_cross_eog_query_binary_group_subject_reg_f", fold, "eeg_eog", device
        )
        val_loader = DataLoader(_dataset(args, eog_metrics, "val", fold), batch_size=args.robustness_batch_size)
        test_loader = DataLoader(_dataset(args, eog_metrics, "test", fold), batch_size=args.robustness_batch_size)

        eeg_val, val_targets, _ = _collect(eeg_model, val_loader, device, "eeg")
        eog_val, _, _ = _collect(eog_model, val_loader, device, "eog")
        weights = [0.0, 0.25, 0.5, 0.75, 1.0]
        best_weight = min(
            weights,
            key=lambda weight: _metrics(weight * eog_val + (1.0 - weight) * eeg_val, val_targets, "binary")["rmse"],
        )
        eeg_test, test_targets, _ = _collect(eeg_model, test_loader, device, "eeg")
        eog_test, _, _ = _collect(eog_model, test_loader, device, "eog")
        late_metrics = _metrics(best_weight * eog_test + (1.0 - best_weight) * eeg_test, test_targets, "binary")
        late_rows.append({"fold": int(fold), "eog_weight": best_weight, **late_metrics})

        base = None
        for scenario in ("none", "zero", "paired_shuffle"):
            _, _, metrics = _collect(fusion_model, test_loader, device, "eeg_eog", eeg_scenario=scenario)
            if scenario == "none":
                base = metrics
            eeg_rows.append(
                {
                    "fold": int(fold),
                    "scenario": scenario,
                    **metrics,
                    "delta_rmse": metrics["rmse"] - base["rmse"],
                }
            )

    result = {
        "late_fusion": {
            "folds": late_rows,
            "summary": _metric_summary(
                late_rows, ("accuracy", "macro_f1", "balanced_accuracy", "rmse", "pearson", "eog_weight")
            ),
        },
        "eeg_perturbation": {
            "folds": eeg_rows,
            "summary": _summarize_scenarios(eeg_rows),
        },
    }
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _self_check():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for variant in VARIANTS[:2]:
            for fold in (0, 1):
                run_dir = root / f"{variant['prefix']}{fold}"
                run_dir.mkdir(parents=True)
                (run_dir / "final_metrics.json").write_text(
                    json.dumps(
                        {
                            "test_accuracy": 0.8 + fold * 0.1,
                            "test_macro_f1": 0.7,
                            "test_balanced_accuracy": 0.75,
                            "test_rmse": 0.1 + fold * 0.02,
                            "test_pearson": 0.8,
                        }
                    ),
                    encoding="utf-8",
                )
        result = _summarize(root, (0, 1))
        assert abs(result["methods"][0]["summary"]["Acc"]["mean"] - 0.85) < 1e-12
        assert result["methods"][2]["fold_count"] == 0


def main():
    parser = argparse.ArgumentParser(description="Run the minimal paper-facing EyeQueryNet evidence pack.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\eeg-eog\data\SEED-VIG"))
    parser.add_argument("--cache-dir", type=Path, default=Path(r"D:\seedvig-CT\cache\seedvig_raw_eeg"))
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--report-path", type=Path, default=Path(r"reports\minimal_eogq_experiments.md"))
    parser.add_argument("--json-path", type=Path, default=Path(r"reports\minimal_eogq_experiments.json"))
    parser.add_argument("--robustness-out-dir", type=Path, default=Path(r"runs\eogq_corruption_analysis_current"))
    parser.add_argument("--supplement-dir", type=Path, default=Path(r"runs\eogq_supplement_current"))
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--temporal-layers", type=int, default=1)
    parser.add_argument("--robustness-batch-size", type=int, default=32)
    parser.add_argument("--robustness-seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-missing", action="store_true")
    parser.add_argument("--run-robustness", action="store_true")
    parser.add_argument("--run-supplement", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()

    if args.self_check:
        _self_check()
        return

    result = _summarize(args.run_root, args.folds)
    if args.run_missing:
        _run_missing(args, result)
        result = _summarize(args.run_root, args.folds)
    robustness_summary = _run_robustness(args)
    supplement_summary = _run_supplement(args)
    payload = {"fold_results": result, "robustness": robustness_summary, "supplement": supplement_summary}
    args.json_path.parent.mkdir(parents=True, exist_ok=True)
    args.json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(args.report_path, result, robustness_summary, supplement_summary)


if __name__ == "__main__":
    main()
