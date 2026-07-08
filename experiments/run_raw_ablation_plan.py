import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


DEFAULT_DATA_ROOT = Path(r"D:\eeg-eog\data\SEED-VIG")
DEFAULT_CACHE_DIR = Path(r"D:\seedvig-CT\cache\seedvig_raw_eeg")

EXPERIMENTS = (
    {
        "name": "raw_eeg_only_binary_group_subject",
        "state_dir": Path(r"runs\auto_raw_eeg_only_binary_group_subject"),
        "prefix": "raw_eeg_only_binary_group_subject_f",
        "input_mode": "eeg",
        "use_eog_cross_attention": False,
    },
    {
        "name": "raw_eog_only_binary_group_subject",
        "state_dir": Path(r"runs\auto_raw_eog_only_binary_group_subject"),
        "prefix": "raw_eog_only_binary_group_subject_f",
        "input_mode": "eog",
        "use_eog_cross_attention": False,
    },
    {
        "name": "raw_eeg_eog_delta_gate_binary_group_subject",
        "state_dir": Path(r"runs\auto_raw_eeg_eog_delta_gate_binary_group_subject"),
        "prefix": "raw_eeg_eog_delta_gate_binary_group_subject_f",
        "input_mode": "eeg_eog",
        "use_eog_cross_attention": True,
        "use_temporal_delta": True,
        "use_eog_gate": True,
        "eog_dropout": 0.25,
    },
)


def _now():
    return datetime.now().isoformat(timespec="seconds")


def _append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": _now(), **row}, ensure_ascii=False) + "\n")


def gpu_memory_used_mb(nvidia_smi="nvidia-smi"):
    command = [
        nvidia_smi,
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
    first_line = output.strip().splitlines()[0]
    return int(first_line.strip())


def wait_for_gpu(max_used_mb, poll_seconds, event_log):
    while True:
        try:
            used_mb = gpu_memory_used_mb()
        except Exception as exc:  # pragma: no cover - only for machines without nvidia-smi.
            _append_jsonl(event_log, {"event": "gpu_check_unavailable", "error": str(exc)})
            return
        if used_mb <= max_used_mb:
            _append_jsonl(event_log, {"event": "gpu_ready", "used_mb": used_mb, "max_used_mb": max_used_mb})
            return
        _append_jsonl(event_log, {"event": "gpu_wait", "used_mb": used_mb, "max_used_mb": max_used_mb})
        time.sleep(poll_seconds)


def build_auto_command(args, experiment):
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
        str(experiment["state_dir"]),
        "--prefix",
        experiment["prefix"],
        "--split-strategy",
        "group_subject",
        "--label-mode",
        "binary",
        "--input-mode",
        experiment["input_mode"],
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
    ]
    if experiment.get("use_eog_cross_attention"):
        command.append("--use-eog-cross-attention")
    else:
        command.append("--no-use-eog-cross-attention")
    if experiment.get("use_temporal_delta"):
        command.append("--use-temporal-delta")
    if experiment.get("use_eog_gate"):
        command.append("--use-eog-gate")
    if experiment.get("eog_dropout"):
        command.extend(["--eog-dropout", str(experiment["eog_dropout"])])
    return command


def run_plan(args):
    plan_log = args.state_dir / "events.jsonl"
    _append_jsonl(
        plan_log,
        {
            "event": "plan_start",
            "experiments": [experiment["name"] for experiment in EXPERIMENTS],
            "gpu_max_used_mb": args.gpu_max_used_mb,
        },
    )
    for experiment in EXPERIMENTS:
        if args.device.startswith("cuda"):
            wait_for_gpu(args.gpu_max_used_mb, args.gpu_poll_seconds, plan_log)
        command = build_auto_command(args, experiment)
        experiment["state_dir"].mkdir(parents=True, exist_ok=True)
        (experiment["state_dir"] / "plan_command.json").write_text(
            json.dumps(command, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _append_jsonl(plan_log, {"event": "experiment_start", "name": experiment["name"], "command": command})
        with (experiment["state_dir"] / "plan_stdout.log").open("ab") as stdout, (
            experiment["state_dir"] / "plan_stderr.log"
        ).open("ab") as stderr:
            process = subprocess.Popen(command, cwd=args.cwd, stdout=stdout, stderr=stderr)
            returncode = process.wait()
        if returncode != 0:
            _append_jsonl(plan_log, {"event": "experiment_failed", "name": experiment["name"], "returncode": returncode})
            return returncode
        _append_jsonl(plan_log, {"event": "experiment_done", "name": experiment["name"]})
    _append_jsonl(plan_log, {"event": "plan_done"})
    return 0


def main():
    parser = argparse.ArgumentParser(description="Wait for GPU and run the raw binary group-subject ablation plan.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--state-dir", type=Path, default=Path(r"runs\auto_raw_ablation_plan"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--gpu-max-used-mb", type=int, default=1024)
    parser.add_argument("--gpu-poll-seconds", type=int, default=120)
    raise SystemExit(run_plan(parser.parse_args()))


if __name__ == "__main__":
    main()
