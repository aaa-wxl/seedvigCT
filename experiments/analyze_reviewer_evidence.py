import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from scipy import stats as scipy_stats

from seedvig.models import RawEEGConformer


DEFAULT_COMPARISONS = (
    ("final_vs_eog", "raw_eeg_eog_anchor_residual_binary_group_subject_reg_nodelta_f", "raw_eog_only_binary_group_subject_reg_f"),
    ("final_vs_cross_attention", "raw_eeg_eog_anchor_residual_binary_group_subject_reg_nodelta_f", "raw_eeg_eog_cross_binary_group_subject_f"),
    ("final_vs_cross_attention_gate", "raw_eeg_eog_anchor_residual_binary_group_subject_reg_nodelta_f", "raw_eeg_eog_cross_gate_binary_group_subject_reg_f"),
)

METRICS = ("test_rmse", "test_pearson", "test_accuracy", "test_macro_f1", "test_balanced_accuracy")

PROFILE_MODELS = (
    ("EEG-only", {"input_mode": "eeg"}),
    ("EOG-only", {"input_mode": "eog"}),
    ("Cross-attention", {"input_mode": "eeg_eog", "use_eog_cross_attention": True}),
    ("Cross-attention+gate", {"input_mode": "eeg_eog", "use_eog_cross_attention": True, "use_eog_gate": True}),
    (
        "Cross-attention EOG-query",
        {
            "input_mode": "eeg_eog",
            "use_eog_cross_attention": True,
            "use_eog_gate": True,
            "cross_attention_direction": "eog_queries_eeg",
        },
    ),
    (
        "Cross-attention bidirectional",
        {
            "input_mode": "eeg_eog",
            "use_eog_cross_attention": True,
            "use_eog_gate": True,
            "cross_attention_direction": "bidirectional",
        },
    ),
    ("EOG residual correction", {"input_mode": "eeg_eog", "use_eog_residual_correction": True}),
    ("Final anchor residual", {"input_mode": "eeg_eog", "use_eog_anchor_residual": True}),
)


def _read_metrics(run_root, prefix, fold):
    path = Path(run_root) / f"{prefix}{fold}" / "final_metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))


def paired_fold_values(run_root, ours_prefix, baseline_prefix, metric, folds=(0, 1, 2, 3, 4)):
    ours = []
    baseline = []
    for fold in folds:
        ours.append(float(_read_metrics(run_root, ours_prefix, fold)[metric]))
        baseline.append(float(_read_metrics(run_root, baseline_prefix, fold)[metric]))
    return ours, baseline


def paired_statistics(ours, baseline):
    deltas = [left - right for left, right in zip(ours, baseline)]
    if len(set(round(delta, 12) for delta in deltas)) == 1:
        paired_t_p = 1.0 if abs(deltas[0]) < 1e-12 else 0.0
    else:
        paired_t_p = float(scipy_stats.ttest_rel(ours, baseline).pvalue)
    if all(abs(delta) < 1e-12 for delta in deltas):
        wilcoxon_p = 1.0
    else:
        wilcoxon_p = float(scipy_stats.wilcoxon(ours, baseline, zero_method="zsplit").pvalue)
    return {
        "ours_mean": statistics.mean(ours),
        "baseline_mean": statistics.mean(baseline),
        "mean_delta": statistics.mean(deltas),
        "paired_t_p": paired_t_p,
        "wilcoxon_p": wilcoxon_p,
    }


def _model_from_config(config):
    input_mode = config.get("input_mode", "eeg")
    return RawEEGConformer(
        eeg_channels=7 if input_mode == "eog" else 17,
        eog_channels=7,
        embedding_dim=64,
        attention_heads=4,
        window_transformer_layers=1,
        temporal_layers=1,
        num_classes=2,
        max_sequence_length=32,
        use_eog_cross_attention=bool(config.get("use_eog_cross_attention", False)),
        use_eog_gate=bool(config.get("use_eog_gate", False)),
        use_temporal_delta=bool(config.get("use_temporal_delta", False)),
        use_eog_anchor_residual=bool(config.get("use_eog_anchor_residual", False)),
        use_eog_residual_correction=bool(config.get("use_eog_residual_correction", False)),
        eog_dropout=float(config.get("eog_dropout", 0.0)),
        cross_attention_direction=config.get("cross_attention_direction", "eeg_queries_eog"),
    )


def profile_model(name, config, device="cpu", repeats=50, warmup=10):
    device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    input_mode = config.get("input_mode", "eeg")
    model = _model_from_config(config).to(device).eval()
    params = sum(parameter.numel() for parameter in model.parameters())
    eeg = torch.randn(1, 8, 17 if input_mode != "eog" else 7, 1600 if input_mode != "eog" else 1000, device=device)
    eog = torch.randn(1, 8, 7, 1000, device=device) if input_mode == "eeg_eog" else None
    repeats = max(1, int(repeats))
    warmup = max(0, int(warmup))
    with torch.no_grad():
        for _ in range(warmup):
            model(eeg, eog=eog)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            model(eeg, eog=eog)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    return {
        "model": name,
        "params": int(params),
        "latency_ms": elapsed * 1000.0 / repeats,
        "device": str(device),
    }


def analyze(run_root=Path("runs"), folds=(0, 1, 2, 3, 4), device="cpu", repeats=50, warmup=10):
    significance = {}
    for name, ours_prefix, baseline_prefix in DEFAULT_COMPARISONS:
        significance[name] = {}
        for metric in METRICS:
            ours, baseline = paired_fold_values(run_root, ours_prefix, baseline_prefix, metric, folds)
            significance[name][metric] = paired_statistics(ours, baseline)
    efficiency = [profile_model(name, config, device=device, repeats=repeats, warmup=warmup) for name, config in PROFILE_MODELS]
    return {"significance": significance, "efficiency": efficiency}


def _fmt(value):
    return f"{value:.4g}" if abs(value) < 0.001 else f"{value:.4f}"


def write_markdown(path, result):
    lines = [
        "# Reviewer Evidence Addendum",
        "",
        "## Paired Significance Tests",
        "",
        "| Comparison | Metric | Ours | Baseline | Delta | paired t p | Wilcoxon p |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for comparison, metrics in result["significance"].items():
        for metric, row in metrics.items():
            lines.append(
                f"| {comparison} | {metric} | {_fmt(row['ours_mean'])} | {_fmt(row['baseline_mean'])} | "
                f"{_fmt(row['mean_delta'])} | {_fmt(row['paired_t_p'])} | {_fmt(row['wilcoxon_p'])} |"
            )
    lines.extend(
        [
            "",
            "## Model Efficiency",
            "",
            "| Model | Params | Latency ms/sample | Device |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for row in result["efficiency"]:
        lines.append(f"| {row['model']} | {row['params']} | {row['latency_ms']:.3f} | {row['device']} |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate reviewer evidence tables from completed SEED-VIG runs.")
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--out-json", type=Path, default=Path(r"reports\reviewer_evidence_addendum.json"))
    parser.add_argument("--out-md", type=Path, default=Path(r"reports\reviewer_evidence_addendum.md"))
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    result = analyze(args.run_root, args.folds, args.device, args.repeats, args.warmup)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(args.out_md, result)


if __name__ == "__main__":
    main()
