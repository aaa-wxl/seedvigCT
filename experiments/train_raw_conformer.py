import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from seedvig.metrics import classification_metrics, regression_metrics
from seedvig.models import RawEEGConformer
from seedvig.raw_dataset import LABEL_MODE_CHOICES, RawSeedVIGSequenceDataset, label_mode_num_classes, perclos_to_class
from seedvig.reference_results import compare_to_references


DEFAULT_DATA_ROOT = Path(r"D:\eeg-eog\data\SEED-VIG")
INPUT_MODE_CHOICES = ("eeg", "eog", "eeg_eog")
TRAINING_OBJECTIVE_CHOICES = ("multitask", "classification", "regression")
SELECTION_METRIC_CHOICES = (
    "auto",
    "val_loss",
    "val_classification",
    "val_regression",
    "val_accuracy",
    "val_macro_f1",
    "val_balanced_accuracy",
    "val_mae",
    "val_rmse",
    "val_pearson",
)
LOWER_IS_BETTER_SELECTIONS = {"val_loss", "val_classification", "val_regression", "val_mae", "val_rmse"}


def _move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _loss(outputs, targets, objective="regression", regression_weight=0.5):
    class_loss = F.cross_entropy(outputs["class_logits"], targets["class"].long())
    regression_loss = F.smooth_l1_loss(outputs["perclos"].squeeze(-1), targets["perclos"].float())
    if objective == "classification":
        loss = class_loss
    elif objective == "regression":
        loss = regression_loss
    elif objective == "multitask":
        loss = class_loss + regression_weight * regression_loss
    else:
        raise ValueError(f"objective must be one of: {TRAINING_OBJECTIVE_CHOICES}")
    return loss, {
        "classification": class_loss.detach(),
        "regression": regression_loss.detach(),
    }


def _resolve_selection_metric(selection_metric, objective):
    if selection_metric != "auto":
        return selection_metric
    if objective == "classification":
        return "val_balanced_accuracy"
    if objective == "regression":
        return "val_rmse"
    return "val_loss"


def _is_improved(row, selection_metric, best_score):
    score = float(row[selection_metric])
    if best_score is None:
        return True, score
    if selection_metric in LOWER_IS_BETTER_SELECTIONS:
        return score < best_score, score
    return score > best_score, score


def _model_inputs(batch, input_mode):
    if input_mode == "eog":
        return batch["eog"], None
    if input_mode == "eeg_eog":
        return batch["eeg"], batch.get("eog")
    return batch["eeg"], None


def _evaluate(
    model,
    loader,
    device,
    num_classes,
    max_batches=None,
    prefix="test",
    input_mode="eeg",
    training_objective="regression",
    regression_weight=0.5,
    label_mode="three_class",
):
    was_training = model.training
    model.eval()
    totals = {"loss": 0.0, "classification": 0.0, "regression": 0.0}
    class_predictions = []
    class_targets = []
    perclos_predictions = []
    perclos_targets = []
    steps = 0
    with torch.no_grad():
        for batch in loader:
            batch = _move_to_device(batch, device)
            signal, eog = _model_inputs(batch, input_mode)
            outputs = model(signal, eog=eog)
            loss, components = _loss(outputs, batch["targets"], training_objective, regression_weight)
            perclos_prediction = outputs["perclos"].squeeze(-1).detach().cpu()
            totals["loss"] += float(loss.detach().cpu())
            totals["classification"] += float(components["classification"].cpu())
            totals["regression"] += float(components["regression"].cpu())
            if training_objective == "regression":
                class_predictions.append(perclos_to_class(perclos_prediction, label_mode))
            else:
                class_predictions.append(outputs["class_logits"].argmax(dim=1).detach().cpu())
            class_targets.append(batch["targets"]["class"].long().detach().cpu())
            perclos_predictions.append(perclos_prediction)
            perclos_targets.append(batch["targets"]["perclos"].float().detach().cpu())
            steps += 1
            if max_batches is not None and steps >= max_batches:
                break
    if was_training:
        model.train()
    if steps == 0:
        raise RuntimeError("evaluation produced no batches")

    cls = classification_metrics(torch.cat(class_predictions), torch.cat(class_targets), num_classes)
    reg = regression_metrics(torch.cat(perclos_predictions), torch.cat(perclos_targets))
    return {
        f"{prefix}_loss": totals["loss"] / steps,
        f"{prefix}_classification": totals["classification"] / steps,
        f"{prefix}_regression": totals["regression"] / steps,
        f"{prefix}_accuracy": cls["accuracy"],
        f"{prefix}_macro_f1": cls["macro_f1"],
        f"{prefix}_balanced_accuracy": cls["balanced_accuracy"],
        f"{prefix}_mae": reg["mae"],
        f"{prefix}_rmse": reg["rmse"],
        f"{prefix}_pearson": reg["pearson"],
        f"{prefix}_confusion_matrix": json.dumps(cls["confusion_matrix"]),
        f"{prefix}_steps": steps,
    }


def _write_epoch_log(run_dir, row, write_header):
    csv_path = run_dir / "metrics.csv"
    jsonl_path = run_dir / "metrics.jsonl"
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_training(
    data_root=DEFAULT_DATA_ROOT,
    cache_dir=None,
    run_dir="runs/raw_eeg_conformer",
    epochs=20,
    sequence_length=8,
    batch_size=4,
    embedding_dim=64,
    attention_heads=4,
    window_transformer_layers=1,
    temporal_layers=1,
    learning_rate=1e-3,
    split_strategy="within_experiment_5fold",
    fold=0,
    validation_fold=None,
    label_mode="three_class",
    max_batches=None,
    eval_max_batches=None,
    seed=0,
    device="cpu",
    include_eog=False,
    use_eog_cross_attention=False,
    input_mode="eeg",
    use_temporal_delta=False,
    use_eog_gate=False,
    use_eog_anchor_residual=False,
    eog_dropout=0.0,
    training_objective="regression",
    regression_weight=0.5,
    selection_metric="auto",
):
    torch.manual_seed(seed)
    data_root = Path(data_root)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    num_classes = label_mode_num_classes(label_mode)
    if training_objective not in TRAINING_OBJECTIVE_CHOICES:
        raise ValueError(f"training_objective must be one of: {TRAINING_OBJECTIVE_CHOICES}")
    selection_metric = _resolve_selection_metric(selection_metric, training_objective)
    if selection_metric not in SELECTION_METRIC_CHOICES:
        raise ValueError(f"selection_metric must be one of: {SELECTION_METRIC_CHOICES}")
    if input_mode not in INPUT_MODE_CHOICES:
        raise ValueError(f"input_mode must be one of: {INPUT_MODE_CHOICES}")
    if use_eog_anchor_residual:
        use_eog_cross_attention = False
    if (use_eog_cross_attention or use_eog_anchor_residual) and input_mode == "eeg":
        input_mode = "eeg_eog"
    if input_mode == "eog" and (use_eog_cross_attention or use_eog_anchor_residual):
        raise ValueError("EOG-only mode cannot use EEG/EOG fusion")

    dataset_args = {
        "root_path": data_root,
        "cache_dir": cache_dir,
        "include_eog": include_eog or input_mode in ("eog", "eeg_eog") or use_eog_cross_attention or use_eog_anchor_residual,
        "sequence_length": sequence_length,
        "split_strategy": split_strategy,
        "fold": fold,
        "validation_fold": validation_fold,
        "label_mode": label_mode,
    }
    train_dataset = RawSeedVIGSequenceDataset(split="train", **dataset_args)
    val_dataset = RawSeedVIGSequenceDataset(split="val", **dataset_args)
    test_dataset = RawSeedVIGSequenceDataset(split="test", **dataset_args)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = RawEEGConformer(
        eeg_channels=7 if input_mode == "eog" else 17,
        embedding_dim=embedding_dim,
        attention_heads=attention_heads,
        window_transformer_layers=window_transformer_layers,
        temporal_layers=temporal_layers,
        num_classes=num_classes,
        max_sequence_length=max(sequence_length, 32),
        use_eog_cross_attention=use_eog_cross_attention,
        use_temporal_delta=use_temporal_delta,
        use_eog_gate=use_eog_gate,
        use_eog_anchor_residual=use_eog_anchor_residual,
        eog_dropout=eog_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    best_score = None
    best_val_loss = None
    best_epoch = 0
    best_model_path = run_dir / "best_model.pt"
    total_steps = 0
    last_metrics = {}

    for epoch in range(1, epochs + 1):
        model.train()
        totals = {"loss": 0.0, "classification": 0.0, "regression": 0.0}
        steps = 0
        for batch in train_loader:
            batch = _move_to_device(batch, device)
            signal, eog = _model_inputs(batch, input_mode)
            outputs = model(signal, eog=eog)
            loss, components = _loss(outputs, batch["targets"], training_objective, regression_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            totals["loss"] += float(loss.detach().cpu())
            totals["classification"] += float(components["classification"].cpu())
            totals["regression"] += float(components["regression"].cpu())
            steps += 1
            total_steps += 1
            if max_batches is not None and steps >= max_batches:
                break
        if steps == 0:
            raise RuntimeError("training produced no batches")

        row = {
            "epoch": epoch,
            "loss": totals["loss"] / steps,
            "classification": totals["classification"] / steps,
            "regression": totals["regression"] / steps,
            "steps": steps,
            "split_strategy": split_strategy,
            "fold": fold,
            "label_mode": label_mode,
            "sequence_length": sequence_length,
            "embedding_dim": embedding_dim,
            "window_transformer_layers": window_transformer_layers,
            "temporal_layers": temporal_layers,
            "input_mode": input_mode,
            "training_objective": training_objective,
            "regression_weight": regression_weight,
            "selection_metric": selection_metric,
            "use_temporal_delta": use_temporal_delta,
            "use_eog_gate": use_eog_gate,
            "use_eog_anchor_residual": use_eog_anchor_residual,
            "eog_dropout": eog_dropout,
        }
        row.update(
            _evaluate(
                model,
                val_loader,
                device,
                num_classes,
                eval_max_batches,
                prefix="val",
                input_mode=input_mode,
                training_objective=training_objective,
                regression_weight=regression_weight,
                label_mode=label_mode,
            )
        )
        improved, score = _is_improved(row, selection_metric, best_score)
        row["improved"] = improved
        if improved:
            best_score = score
            best_val_loss = row["val_loss"]
            best_epoch = epoch
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "metrics": row}, best_model_path)
        row["best_epoch"] = best_epoch
        row["best_val_loss"] = best_val_loss
        row["best_selection_score"] = best_score
        _write_epoch_log(run_dir, row, write_header=(epoch == 1))
        last_metrics = row

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    last_metrics.update(
        _evaluate(
            model,
            test_loader,
            device,
            num_classes,
            eval_max_batches,
            prefix="test",
            input_mode=input_mode,
            training_objective=training_objective,
            regression_weight=regression_weight,
            label_mode=label_mode,
        )
    )
    last_metrics.update(
        {
            "epochs_ran": epochs,
            "total_steps": total_steps,
            "best_model_path": str(best_model_path),
            "run_dir": str(run_dir),
            "selection_metric": selection_metric,
            "training_objective": training_objective,
            "regression_weight": regression_weight,
        }
    )
    last_metrics["reference_comparisons"] = compare_to_references(last_metrics, split_strategy)
    (run_dir / "final_metrics.json").write_text(
        json.dumps(last_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return last_metrics


def main():
    parser = argparse.ArgumentParser(description="Train raw EEG-only Conformer on SEED-VIG.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--run-dir", default="runs/raw_eeg_conformer")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--window-transformer-layers", type=int, default=1)
    parser.add_argument("--temporal-layers", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--split-strategy",
        choices=("within_experiment_5fold", "group_subject"),
        default="within_experiment_5fold",
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--validation-fold", type=int, default=None)
    parser.add_argument("--label-mode", choices=LABEL_MODE_CHOICES, default="three_class")
    parser.add_argument("--input-mode", choices=INPUT_MODE_CHOICES, default="eeg")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--eval-max-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--include-eog", action="store_true")
    parser.add_argument("--use-eog-cross-attention", action="store_true")
    parser.add_argument("--use-temporal-delta", action="store_true")
    parser.add_argument("--use-eog-gate", action="store_true")
    parser.add_argument("--use-eog-anchor-residual", action="store_true")
    parser.add_argument("--eog-dropout", type=float, default=0.0)
    parser.add_argument("--training-objective", choices=TRAINING_OBJECTIVE_CHOICES, default="regression")
    parser.add_argument("--regression-weight", type=float, default=0.5)
    parser.add_argument("--selection-metric", choices=SELECTION_METRIC_CHOICES, default="auto")
    args = parser.parse_args()

    metrics = run_training(
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        run_dir=args.run_dir,
        epochs=args.epochs,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim,
        attention_heads=args.attention_heads,
        window_transformer_layers=args.window_transformer_layers,
        temporal_layers=args.temporal_layers,
        learning_rate=args.learning_rate,
        split_strategy=args.split_strategy,
        fold=args.fold,
        validation_fold=args.validation_fold,
        label_mode=args.label_mode,
        input_mode=args.input_mode,
        max_batches=args.max_batches,
        eval_max_batches=args.eval_max_batches,
        seed=args.seed,
        device=args.device,
        include_eog=args.include_eog,
        use_eog_cross_attention=args.use_eog_cross_attention,
        use_temporal_delta=args.use_temporal_delta,
        use_eog_gate=args.use_eog_gate,
        use_eog_anchor_residual=args.use_eog_anchor_residual,
        eog_dropout=args.eog_dropout,
        training_objective=args.training_objective,
        regression_weight=args.regression_weight,
        selection_metric=args.selection_metric,
    )
    for key, value in metrics.items():
        if isinstance(value, (int, float, bool, str)):
            print(f"{key}: {value}")
    print("reference_comparisons:")
    for item in metrics["reference_comparisons"]:
        print(json.dumps(item, ensure_ascii=False))


if __name__ == "__main__":
    main()
