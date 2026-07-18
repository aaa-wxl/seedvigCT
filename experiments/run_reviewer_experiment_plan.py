import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from experiments.evaluate_eog_corruption import default_corruption_sweep
from seedvig.models import CROSS_ATTENTION_DIRECTION_CHOICES


DEFAULT_DATA_ROOT = Path(r"D:\eeg-eog\data\SEED-VIG")
DEFAULT_CACHE_DIR = Path(r"D:\seedvig-CT\cache\seedvig_raw_eeg")

REVIEW_ABLATIONS = (
    {
        "name": "eeg_only_regression",
        "state_dir": Path(r"runs\auto_raw_eeg_only_binary_group_subject"),
        "prefix": "raw_eeg_only_binary_group_subject_f",
        "input_mode": "eeg",
        "training_objective": "regression",
        "use_eog_cross_attention": False,
    },
    {
        "name": "eog_only_regression",
        "state_dir": Path(r"runs\auto_raw_eog_only_binary_group_subject_reg"),
        "prefix": "raw_eog_only_binary_group_subject_reg_f",
        "input_mode": "eog",
        "training_objective": "regression",
        "use_eog_cross_attention": False,
    },
    {
        "name": "eog_only_classification",
        "state_dir": Path(r"runs\auto_raw_eog_only_binary_group_subject_cls"),
        "prefix": "raw_eog_only_binary_group_subject_cls_f",
        "input_mode": "eog",
        "training_objective": "classification",
        "use_eog_cross_attention": False,
    },
    {
        "name": "cross_attention_regression",
        "state_dir": Path(r"runs\auto_raw_eeg_eog_cross_binary_group_subject"),
        "prefix": "raw_eeg_eog_cross_binary_group_subject_f",
        "input_mode": "eeg_eog",
        "training_objective": "regression",
        "use_eog_cross_attention": True,
    },
    {
        "name": "cross_attention_gate_regression",
        "state_dir": Path(r"runs\auto_raw_eeg_eog_cross_gate_binary_group_subject_reg"),
        "prefix": "raw_eeg_eog_cross_gate_binary_group_subject_reg_f",
        "input_mode": "eeg_eog",
        "training_objective": "regression",
        "use_eog_cross_attention": True,
        "use_eog_gate": True,
    },
    {
        "name": "cross_attention_eog_query_regression",
        "state_dir": Path(r"runs\auto_raw_eeg_eog_cross_eog_query_binary_group_subject_reg"),
        "prefix": "raw_eeg_eog_cross_eog_query_binary_group_subject_reg_f",
        "input_mode": "eeg_eog",
        "training_objective": "regression",
        "use_eog_cross_attention": True,
        "use_eog_gate": True,
        "cross_attention_direction": "eog_queries_eeg",
    },
    {
        "name": "cross_attention_bidirectional_regression",
        "state_dir": Path(r"runs\auto_raw_eeg_eog_cross_bidir_binary_group_subject_reg"),
        "prefix": "raw_eeg_eog_cross_bidir_binary_group_subject_reg_f",
        "input_mode": "eeg_eog",
        "training_objective": "regression",
        "use_eog_cross_attention": True,
        "use_eog_gate": True,
        "cross_attention_direction": "bidirectional",
    },
    {
        "name": "eog_residual_correction_regression",
        "state_dir": Path(r"runs\auto_raw_eeg_eog_residual_correction_binary_group_subject_reg"),
        "prefix": "raw_eeg_eog_residual_correction_binary_group_subject_reg_f",
        "input_mode": "eeg_eog",
        "training_objective": "regression",
        "use_eog_residual_correction": True,
        "use_eog_cross_attention": False,
    },
    {
        "name": "delta_gate_regression",
        "state_dir": Path(r"runs\auto_raw_eeg_eog_delta_gate_binary_group_subject_reg"),
        "prefix": "raw_eeg_eog_delta_gate_binary_group_subject_reg_f",
        "input_mode": "eeg_eog",
        "training_objective": "regression",
        "use_eog_cross_attention": True,
        "use_temporal_delta": True,
        "use_eog_gate": True,
        "eog_dropout": 0.25,
    },
    {
        "name": "delta_gate_classification",
        "state_dir": Path(r"runs\auto_raw_eeg_eog_delta_gate_binary_group_subject_cls"),
        "prefix": "raw_eeg_eog_delta_gate_binary_group_subject_cls_f",
        "input_mode": "eeg_eog",
        "training_objective": "classification",
        "use_eog_cross_attention": True,
        "use_temporal_delta": True,
        "use_eog_gate": True,
        "eog_dropout": 0.25,
    },
    {
        "name": "anchor_residual_delta_regression",
        "state_dir": Path(r"runs\auto_raw_eeg_eog_anchor_residual_binary_group_subject_reg"),
        "prefix": "raw_eeg_eog_anchor_residual_binary_group_subject_reg_f",
        "input_mode": "eeg_eog",
        "training_objective": "regression",
        "use_eog_anchor_residual": True,
        "use_temporal_delta": True,
        "use_eog_cross_attention": False,
    },
    {
        "name": "anchor_residual_nodelta_regression",
        "state_dir": Path(r"runs\auto_raw_eeg_eog_anchor_residual_binary_group_subject_reg_nodelta"),
        "prefix": "raw_eeg_eog_anchor_residual_binary_group_subject_reg_nodelta_f",
        "input_mode": "eeg_eog",
        "training_objective": "regression",
        "use_eog_anchor_residual": True,
        "use_eog_cross_attention": False,
    },
)


def _now():
    return datetime.now().isoformat(timespec="seconds")


def _append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": _now(), **row}, ensure_ascii=False) + "\n")


def _format_corruption(item):
    kind, value = item
    if kind in ("none", "zero"):
        return kind
    return f"{kind}:{value:g}"


def build_ablation_command(args, ablation):
    command = [
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
        str(ablation["state_dir"]),
        "--prefix",
        ablation["prefix"],
        "--split-strategy",
        "group_subject",
        "--label-mode",
        "binary",
        "--input-mode",
        ablation["input_mode"],
        "--folds",
        *[str(fold) for fold in args.folds],
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--poll-seconds",
        str(args.poll_seconds),
        "--training-objective",
        ablation["training_objective"],
    ]
    if ablation.get("use_eog_residual_correction"):
        command.append("--use-eog-residual-correction")
    elif ablation.get("use_eog_anchor_residual"):
        command.append("--use-eog-anchor-residual")
    elif ablation.get("use_eog_cross_attention"):
        command.append("--use-eog-cross-attention")
        if ablation.get("cross_attention_direction"):
            command.extend(["--cross-attention-direction", ablation["cross_attention_direction"]])
    else:
        command.append("--no-use-eog-cross-attention")
    if ablation.get("use_temporal_delta"):
        command.append("--use-temporal-delta")
    if ablation.get("use_eog_gate"):
        command.append("--use-eog-gate")
    if ablation.get("eog_dropout"):
        command.extend(["--eog-dropout", str(ablation["eog_dropout"])])
    return command


def build_robustness_command(args):
    command = [
        args.python,
        "-m",
        "experiments.evaluate_eog_corruption",
        "--data-root",
        str(args.data_root),
        "--cache-dir",
        str(args.cache_dir),
        "--run-root",
        str(args.run_root),
        "--eog-prefix",
        "raw_eog_only_binary_group_subject_reg_f",
        "--fusion-prefix",
        "raw_eeg_eog_anchor_residual_binary_group_subject_reg_nodelta_f",
        "--out-dir",
        str(args.robustness_out_dir),
        "--folds",
        *[str(fold) for fold in args.folds],
        "--batch-size",
        str(getattr(args, "robustness_batch_size", 32)),
        "--device",
        args.device,
        "--seeds",
        *[str(seed) for seed in args.robustness_seeds],
    ]
    for corruption in default_corruption_sweep():
        command.extend(["--corruption", _format_corruption(corruption)])
    return command


def _gpu_memory_used_mb():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
        stderr=subprocess.STDOUT,
    )
    return int(output.strip().splitlines()[0].strip())


def _gpu_process_rows():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        text=True,
        stderr=subprocess.STDOUT,
    )
    rows = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) == 3 and parts[0].isdigit():
            rows.append({"pid": int(parts[0]), "name": parts[1], "memory": parts[2]})
    return rows


def stop_gpu_python_processes(event_log):
    current_pid = os.getpid()
    for row in _gpu_process_rows():
        if row["pid"] == current_pid or not row["name"].lower().endswith("python.exe"):
            continue
        _append_jsonl(event_log, {"event": "stop_gpu_python", **row})
        subprocess.run(["powershell", "-NoProfile", "-Command", f"Stop-Process -Id {row['pid']} -Force"], check=False)


def wait_for_gpu(max_used_mb, poll_seconds, event_log):
    while True:
        try:
            used_mb = _gpu_memory_used_mb()
        except Exception as exc:
            _append_jsonl(event_log, {"event": "gpu_check_unavailable", "error": str(exc)})
            return
        if used_mb <= max_used_mb:
            _append_jsonl(event_log, {"event": "gpu_ready", "used_mb": used_mb, "max_used_mb": max_used_mb})
            return
        _append_jsonl(event_log, {"event": "gpu_wait", "used_mb": used_mb, "max_used_mb": max_used_mb})
        time.sleep(poll_seconds)


def run_logged(command, cwd, stdout_path, stderr_path, event_log, name):
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    _append_jsonl(event_log, {"event": "command_start", "name": name, "command": command})
    with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
        process = subprocess.Popen(command, cwd=cwd, stdout=stdout, stderr=stderr)
        returncode = process.wait()
    _append_jsonl(event_log, {"event": "command_done", "name": name, "returncode": returncode})
    return returncode


def _summary_line(name, summary_path):
    if not summary_path.exists():
        return f"| {name} | missing | - | - | - | - | - | - |"
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = data.get("metrics", {})
    def value(metric):
        item = metrics.get(metric)
        if not item:
            return "-"
        return f"{item['mean']:.4f} +/- {item['std']:.4f}"
    return (
        f"| {name} | {data.get('fold_count', 0)} | {value('test_accuracy')} | "
        f"{value('test_macro_f1')} | {value('test_balanced_accuracy')} | "
        f"{value('test_rmse')} | {value('test_pearson')} | {value('test_mae')} |"
    )


def write_status_report(args):
    report_path = args.report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Reviewer Experiment Status",
        "",
        f"Generated: {_now()}",
        "",
        "## Local Ablations",
        "",
        "| Setting | Folds | Acc | Macro-F1 | BalAcc | RMSE | Pearson | MAE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for ablation in REVIEW_ABLATIONS:
        lines.append(_summary_line(ablation["name"], ablation["state_dir"] / "summary.json"))
    robustness = args.robustness_out_dir / "summary.md"
    lines.extend(
        [
            "",
            "## Robustness Sweep",
            "",
            f"Summary: `{robustness}`",
            "",
            "## Public SOTA Table To Fill With Verified Protocols",
            "",
            "| Method | Public code | Protocol | Task | Modalities | Reported metric | Comparable here? |",
            "| --- | --- | --- | --- | --- | --- | --- |",
            "| HMS-TENet | yes | verify | classification/regression | EEG+EOG features | verify | protocol-sensitive |",
            "| MHCL | yes | cross-subject reported | classification | EEG+EOG | verify | compare only by protocol/task |",
            "| E2CF | verify | verify | vigilance estimation | EEG+EOG | verify | citation/protocol needed |",
            "| DeltaGateNet | verify | inter-subject reported | classification | EEG | verify | not direct regression match |",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def run_plan(args):
    args.state_dir.mkdir(parents=True, exist_ok=True)
    event_log = args.state_dir / "events.jsonl"
    _append_jsonl(event_log, {"event": "reviewer_plan_start"})

    for ablation in REVIEW_ABLATIONS:
        if args.device.startswith("cuda"):
            if args.stop_gpu_python_processes:
                stop_gpu_python_processes(event_log)
            wait_for_gpu(args.gpu_max_used_mb, args.gpu_poll_seconds, event_log)
        command = build_ablation_command(args, ablation)
        command_path = args.state_dir / f"{ablation['name']}_command.json"
        command_path.write_text(json.dumps(command, ensure_ascii=False, indent=2), encoding="utf-8")
        returncode = run_logged(
            command,
            args.cwd,
            args.state_dir / f"{ablation['name']}.stdout.log",
            args.state_dir / f"{ablation['name']}.stderr.log",
            event_log,
            ablation["name"],
        )
        if returncode != 0:
            write_status_report(args)
            return returncode

    if args.device.startswith("cuda"):
        if args.stop_gpu_python_processes:
            stop_gpu_python_processes(event_log)
        wait_for_gpu(args.gpu_max_used_mb, args.gpu_poll_seconds, event_log)
    robustness_command = build_robustness_command(args)
    (args.state_dir / "robustness_command.json").write_text(
        json.dumps(robustness_command, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    returncode = run_logged(
        robustness_command,
        args.cwd,
        args.state_dir / "robustness.stdout.log",
        args.state_dir / "robustness.stderr.log",
        event_log,
        "robustness_sweep",
    )
    write_status_report(args)
    _append_jsonl(event_log, {"event": "reviewer_plan_done", "returncode": returncode})
    return returncode


def main():
    parser = argparse.ArgumentParser(description="Run reviewer-requested ablations and EOG robustness sweeps.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--state-dir", type=Path, default=Path(r"runs\reviewer_experiment_plan"))
    parser.add_argument("--report-path", type=Path, default=Path(r"reports\reviewer_experiment_status.md"))
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--robustness-batch-size", type=int, default=32)
    parser.add_argument("--robustness-out-dir", type=Path, default=Path(r"runs\reviewer_eog_corruption_sweep"))
    parser.add_argument("--robustness-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--gpu-max-used-mb", type=int, default=1024)
    parser.add_argument("--gpu-poll-seconds", type=int, default=120)
    parser.add_argument("--stop-gpu-python-processes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--cross-attention-direction",
        choices=CROSS_ATTENTION_DIRECTION_CHOICES,
        default="eeg_queries_eog",
        help=argparse.SUPPRESS,
    )
    raise SystemExit(run_plan(parser.parse_args()))


if __name__ == "__main__":
    main()
