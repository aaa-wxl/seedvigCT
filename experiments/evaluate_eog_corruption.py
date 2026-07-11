import argparse
import json
import math
import statistics
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from experiments.train_raw_conformer import _move_to_device
from seedvig.metrics import classification_metrics, regression_metrics
from seedvig.models import RawEEGConformer
from seedvig.raw_dataset import RawSeedVIGSequenceDataset, label_mode_num_classes, perclos_to_class


DEFAULT_DATA_ROOT = Path(r"D:\eeg-eog\data\SEED-VIG")
DEFAULT_CACHE_DIR = Path(r"D:\seedvig-CT\cache\seedvig_raw_eeg")
CORRUPTION_CHOICES = ("none", "noise", "mask", "channel_mask", "segment_mask", "zero")


def _clone_batch(batch):
    if torch.is_tensor(batch):
        return batch.clone()
    if isinstance(batch, dict):
        return {key: _clone_batch(value) for key, value in batch.items()}
    if isinstance(batch, list):
        return list(batch)
    return batch


def corrupt_batch(batch, kind="none", value=0.0, seed=0):
    corrupted = _clone_batch(batch)
    if kind == "none" or "eog" not in corrupted:
        return corrupted
    eog = corrupted["eog"]
    if kind == "zero":
        corrupted["eog"] = torch.zeros_like(eog)
        return corrupted

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    if kind == "noise":
        corrupted["eog"] = eog + torch.randn(eog.shape, dtype=eog.dtype, generator=generator) * float(value)
    elif kind == "mask":
        corrupted["eog"] = eog * (torch.rand(eog.shape, dtype=eog.dtype, generator=generator) >= float(value))
    elif kind == "channel_mask":
        mask = torch.rand(eog.shape[:3] + (1,), dtype=eog.dtype, generator=generator) >= float(value)
        corrupted["eog"] = eog * mask
    elif kind == "segment_mask":
        mask = torch.rand(eog.shape[:2] + (1, 1), dtype=eog.dtype, generator=generator) >= float(value)
        corrupted["eog"] = eog * mask
    else:
        raise ValueError(f"kind must be one of: {CORRUPTION_CHOICES}")
    return corrupted


def _metrics(predictions, targets, label_mode):
    cls = classification_metrics(
        perclos_to_class(predictions, label_mode),
        perclos_to_class(targets, label_mode),
        label_mode_num_classes(label_mode),
    )
    reg = regression_metrics(predictions, targets)
    return {
        "accuracy": cls["accuracy"],
        "macro_f1": cls["macro_f1"],
        "balanced_accuracy": cls["balanced_accuracy"],
        "mae": reg["mae"],
        "rmse": reg["rmse"],
        "pearson": reg["pearson"],
    }


def _rmse(values):
    if not values:
        return 0.0
    return math.sqrt(sum(value * value for value in values) / len(values))


def worst_error_strata(eog_predictions, fusion_predictions, targets, bins=4):
    eog_predictions = torch.as_tensor(eog_predictions, dtype=torch.float32)
    fusion_predictions = torch.as_tensor(fusion_predictions, dtype=torch.float32)
    targets = torch.as_tensor(targets, dtype=torch.float32)
    order = torch.argsort(torch.abs(eog_predictions - targets))
    chunks = torch.chunk(order, int(bins))
    rows = []
    for index, chunk in enumerate(chunks, start=1):
        if chunk.numel() == 0:
            continue
        eog_errors = (eog_predictions[chunk] - targets[chunk]).tolist()
        fusion_errors = (fusion_predictions[chunk] - targets[chunk]).tolist()
        name = f"q{index}"
        if index == len(chunks):
            name = f"worst_{100 // len(chunks)}%"
        eog_rmse = _rmse(eog_errors)
        fusion_rmse = _rmse(fusion_errors)
        rows.append(
            {
                "stratum": name,
                "count": int(chunk.numel()),
                "eog_rmse": eog_rmse,
                "fusion_rmse": fusion_rmse,
                "delta_rmse": fusion_rmse - eog_rmse,
            }
        )
    return rows


def _load_model(run_dir, metrics, input_mode, device):
    model = RawEEGConformer(
        eeg_channels=7 if input_mode == "eog" else 17,
        embedding_dim=int(metrics.get("embedding_dim", 64)),
        attention_heads=int(metrics.get("attention_heads", 4)),
        window_transformer_layers=int(metrics.get("window_transformer_layers", 1)),
        temporal_layers=int(metrics.get("temporal_layers", 1)),
        num_classes=label_mode_num_classes(metrics.get("label_mode", "binary")),
        max_sequence_length=max(int(metrics.get("sequence_length", 8)), 32),
        use_eog_cross_attention=bool(metrics.get("use_eog_cross_attention", False)),
        use_temporal_delta=bool(metrics.get("use_temporal_delta", False)),
        use_eog_gate=bool(metrics.get("use_eog_gate", False)),
        use_eog_anchor_residual=bool(metrics.get("use_eog_anchor_residual", False)),
        eog_dropout=float(metrics.get("eog_dropout", 0.0)),
    ).to(device)
    checkpoint = torch.load(run_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _collect_predictions(model, loader, device, input_mode, corruption, value, seed, max_batches=None):
    predictions = []
    targets = []
    with torch.no_grad():
        for step, batch in enumerate(loader):
            batch = corrupt_batch(batch, kind=corruption, value=value, seed=seed + step)
            batch = _move_to_device(batch, device)
            if input_mode == "eog":
                outputs = model(batch["eog"])
            else:
                outputs = model(batch["eeg"], eog=batch.get("eog"))
            predictions.append(outputs["perclos"].squeeze(-1).detach().cpu())
            targets.append(batch["targets"]["perclos"].float().detach().cpu())
            if max_batches is not None and step + 1 >= max_batches:
                break
    if not predictions:
        raise RuntimeError("evaluation produced no batches")
    return torch.cat(predictions), torch.cat(targets)


def _scenario_name(kind, value):
    if kind in ("none", "zero"):
        return kind
    return f"{kind}_{value:g}"


def evaluate_pair(
    data_root=DEFAULT_DATA_ROOT,
    cache_dir=DEFAULT_CACHE_DIR,
    run_root=Path("runs"),
    eog_prefix="raw_eog_only_binary_group_subject_reg_f",
    fusion_prefix="raw_eeg_eog_anchor_residual_binary_group_subject_reg_nodelta_f",
    out_dir=Path("runs/eog_corruption_analysis"),
    folds=(0, 1, 2, 3, 4),
    corruptions=(("none", 0.0), ("noise", 0.5), ("mask", 0.25), ("channel_mask", 0.25), ("segment_mask", 0.25), ("zero", 0.0)),
    batch_size=32,
    device="cuda",
    seed=0,
    max_batches=None,
):
    device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    strata_rows = []

    for fold in folds:
        eog_run = Path(run_root) / f"{eog_prefix}{fold}"
        fusion_run = Path(run_root) / f"{fusion_prefix}{fold}"
        eog_metrics = json.loads((eog_run / "final_metrics.json").read_text(encoding="utf-8"))
        fusion_metrics = json.loads((fusion_run / "final_metrics.json").read_text(encoding="utf-8"))
        label_mode = fusion_metrics.get("label_mode", eog_metrics.get("label_mode", "binary"))
        sequence_length = int(fusion_metrics.get("sequence_length", eog_metrics.get("sequence_length", 8)))
        dataset = RawSeedVIGSequenceDataset(
            data_root,
            cache_dir=cache_dir,
            include_eog=True,
            sequence_length=sequence_length,
            split="test",
            split_strategy=fusion_metrics.get("split_strategy", "group_subject"),
            fold=int(fold),
            label_mode=label_mode,
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        eog_model = _load_model(eog_run, eog_metrics, "eog", device)
        fusion_model = _load_model(fusion_run, fusion_metrics, "eeg_eog", device)

        for kind, value in corruptions:
            scenario = _scenario_name(kind, value)
            eog_pred, targets = _collect_predictions(
                eog_model,
                loader,
                device,
                "eog",
                kind,
                float(value),
                seed + fold * 1000,
                max_batches,
            )
            fusion_pred, _ = _collect_predictions(
                fusion_model,
                loader,
                device,
                "eeg_eog",
                kind,
                float(value),
                seed + fold * 1000,
                max_batches,
            )
            eog_result = _metrics(eog_pred, targets, label_mode)
            fusion_result = _metrics(fusion_pred, targets, label_mode)
            all_rows.append(
                {
                    "fold": int(fold),
                    "scenario": scenario,
                    **{f"eog_{key}": value for key, value in eog_result.items()},
                    **{f"fusion_{key}": value for key, value in fusion_result.items()},
                    "delta_rmse": fusion_result["rmse"] - eog_result["rmse"],
                    "delta_pearson": fusion_result["pearson"] - eog_result["pearson"],
                    "delta_balanced_accuracy": fusion_result["balanced_accuracy"] - eog_result["balanced_accuracy"],
                }
            )
            for row in worst_error_strata(eog_pred, fusion_pred, targets):
                strata_rows.append({"fold": int(fold), "scenario": scenario, **row})

    summary = _summarize_rows(all_rows)
    (out_dir / "per_fold.json").write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "strata.json").write_text(json.dumps(strata_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(out_dir / "summary.md", summary, strata_rows)
    return {"per_fold": all_rows, "strata": strata_rows, "summary": summary}


def _summarize_rows(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["scenario"], []).append(row)
    summary = {}
    for scenario, items in grouped.items():
        summary[scenario] = {}
        for key in items[0]:
            if key in ("fold", "scenario"):
                continue
            values = [float(item[key]) for item in items]
            summary[scenario][key] = {"mean": statistics.mean(values), "std": statistics.pstdev(values)}
    return summary


def _write_markdown(path, summary, strata_rows):
    lines = ["# EOG Corruption Analysis", ""]
    for scenario, metrics in summary.items():
        lines.append(f"## {scenario}")
        for key in ("eog_rmse", "fusion_rmse", "delta_rmse", "eog_pearson", "fusion_pearson", "delta_pearson", "delta_balanced_accuracy"):
            if key in metrics:
                value = metrics[key]
                lines.append(f"- {key}: {value['mean']:.4f} +/- {value['std']:.4f}")
        lines.append("")
    worst_rows = [row for row in strata_rows if row["stratum"].startswith("worst")]
    if worst_rows:
        lines.append("## worst-error strata")
        for scenario in sorted({row["scenario"] for row in worst_rows}):
            values = [row["delta_rmse"] for row in worst_rows if row["scenario"] == scenario]
            lines.append(f"- {scenario}: delta_rmse={statistics.mean(values):+.4f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_corruption(value):
    if ":" not in value:
        return value, 0.0
    kind, amount = value.split(":", 1)
    return kind, float(amount)


def main():
    parser = argparse.ArgumentParser(description="Evaluate whether EEG+EOG fusion is robust under EOG corruption.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--eog-prefix", default="raw_eog_only_binary_group_subject_reg_f")
    parser.add_argument("--fusion-prefix", default="raw_eeg_eog_anchor_residual_binary_group_subject_reg_nodelta_f")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/eog_corruption_analysis"))
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--corruption",
        action="append",
        default=None,
        help="Use kind or kind:value. Kinds: none, noise, mask, channel_mask, segment_mask, zero.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()
    evaluate_pair(
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        run_root=args.run_root,
        eog_prefix=args.eog_prefix,
        fusion_prefix=args.fusion_prefix,
        out_dir=args.out_dir,
        folds=args.folds,
        corruptions=[_parse_corruption(item) for item in (args.corruption or ["none", "noise:0.5", "mask:0.25", "channel_mask:0.25", "segment_mask:0.25", "zero"])],
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        max_batches=args.max_batches,
    )


if __name__ == "__main__":
    main()
