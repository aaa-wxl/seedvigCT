import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from seedvig.reference_results import REFERENCE_RESULTS


METRICS = (
    "test_accuracy",
    "test_macro_f1",
    "test_balanced_accuracy",
    "test_mae",
    "test_rmse",
    "test_pearson",
)


def _now():
    return datetime.now().isoformat(timespec="seconds")


def _append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"time": _now(), **row}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _last_epoch(run_dir):
    path = run_dir / "metrics.csv"
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else None


def build_command(args, fold):
    return [
        args.python,
        "-m",
        "experiments.train_raw_conformer",
        "--data-root",
        str(args.data_root),
        "--cache-dir",
        str(args.cache_dir),
        "--split-strategy",
        args.split_strategy,
        "--fold",
        str(fold),
        "--label-mode",
        args.label_mode,
        "--run-dir",
        str(args.run_root / f"{args.prefix}{fold}"),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--use-eog-cross-attention",
    ]


def summarize_folds(run_root, prefix, split_strategy, state_dir, folds=(0, 1, 2, 3, 4)):
    rows = []
    for fold in folds:
        path = run_root / f"{prefix}{fold}" / "final_metrics.json"
        if not path.exists():
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"fold": fold, **metrics})

    summary = {"fold_count": len(rows), "folds": [row["fold"] for row in rows], "metrics": {}}
    for metric in METRICS:
        values = [float(row[metric]) for row in rows if metric in row]
        if values:
            summary["metrics"][metric] = {
                "mean": statistics.mean(values),
                "std": statistics.pstdev(values),
            }

    comparisons = []
    references = REFERENCE_RESULTS.get(split_strategy, {})
    metric_map = {
        "test_accuracy": "accuracy",
        "test_rmse": "rmse",
        "test_pearson": "pearson",
    }
    for name, ref in references.items():
        item = {"reference": name}
        for metric, ref_key in metric_map.items():
            if metric in summary["metrics"]:
                item[f"delta_{ref_key}"] = summary["metrics"][metric]["mean"] - ref[ref_key]
                item[f"reference_{ref_key}"] = ref[ref_key]
        comparisons.append(item)
    summary["reference_comparisons"] = comparisons

    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [f"# {prefix} summary", "", f"fold_count: {summary['fold_count']}", ""]
    for metric, values in summary["metrics"].items():
        lines.append(f"- {metric}: {values['mean']:.4f} +/- {values['std']:.4f}")
    if comparisons:
        lines.extend(["", "## reference deltas"])
        for item in comparisons:
            bits = [item["reference"]]
            for key in ("delta_accuracy", "delta_rmse", "delta_pearson"):
                if key in item:
                    bits.append(f"{key}={item[key]:+.4f}")
            lines.append("- " + ", ".join(bits))
    (state_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def run_queue(args):
    event_log = args.state_dir / "events.jsonl"
    _append_jsonl(
        event_log,
        {
            "event": "queue_start",
            "prefix": args.prefix,
            "split_strategy": args.split_strategy,
            "label_mode": args.label_mode,
            "seed": args.seed,
        },
    )
    for fold in args.folds:
        run_dir = args.run_root / f"{args.prefix}{fold}"
        if (run_dir / "final_metrics.json").exists():
            _append_jsonl(event_log, {"event": "skip_done", "fold": fold, "run_dir": str(run_dir)})
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        command = build_command(args, fold)
        (run_dir / "command.json").write_text(json.dumps(command, ensure_ascii=False, indent=2), encoding="utf-8")
        _append_jsonl(event_log, {"event": "fold_start", "fold": fold, "run_dir": str(run_dir), "command": command})
        with (run_dir / "stdout.log").open("ab") as stdout, (run_dir / "stderr.log").open("ab") as stderr:
            process = subprocess.Popen(command, cwd=args.cwd, stdout=stdout, stderr=stderr)
            last_seen = None
            while process.poll() is None:
                row = _last_epoch(run_dir)
                if row and row.get("epoch") != last_seen:
                    last_seen = row.get("epoch")
                    _append_jsonl(
                        event_log,
                        {
                            "event": "epoch",
                            "fold": fold,
                            "epoch": row.get("epoch"),
                            "best_epoch": row.get("best_epoch"),
                            "val_rmse": row.get("val_rmse"),
                            "val_pearson": row.get("val_pearson"),
                        },
                    )
                time.sleep(args.poll_seconds)
        if process.returncode != 0:
            _append_jsonl(event_log, {"event": "fold_failed", "fold": fold, "returncode": process.returncode})
            return process.returncode
        final_metrics = run_dir / "final_metrics.json"
        if not final_metrics.exists():
            _append_jsonl(event_log, {"event": "fold_failed", "fold": fold, "reason": "missing_final_metrics"})
            return 1
        metrics = json.loads(final_metrics.read_text(encoding="utf-8"))
        _append_jsonl(
            event_log,
            {
                "event": "fold_done",
                "fold": fold,
                "best_epoch": metrics.get("best_epoch"),
                "test_accuracy": metrics.get("test_accuracy"),
                "test_rmse": metrics.get("test_rmse"),
                "test_pearson": metrics.get("test_pearson"),
            },
        )

    summary = summarize_folds(args.run_root, args.prefix, args.split_strategy, args.state_dir, args.folds)
    _append_jsonl(event_log, {"event": "queue_done", "summary": summary})
    return 0


def main():
    parser = argparse.ArgumentParser(description="Run and monitor raw EEG+EOG cross-attention fold queues.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\eeg-eog\data\SEED-VIG"))
    parser.add_argument("--cache-dir", type=Path, default=Path(r"D:\seedvig-CT\cache\seedvig_raw_eeg"))
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--state-dir", type=Path, default=Path(r"runs\auto_raw_eeg_eog_cross_group_subject"))
    parser.add_argument("--prefix", default="raw_eeg_eog_cross_group_subject_f")
    parser.add_argument("--split-strategy", choices=("group_subject", "within_experiment_5fold"), default="group_subject")
    parser.add_argument("--label-mode", choices=("three_class", "binary"), default="three_class")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--poll-seconds", type=int, default=30)
    raise SystemExit(run_queue(parser.parse_args()))


if __name__ == "__main__":
    main()
