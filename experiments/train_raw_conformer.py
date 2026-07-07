import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from seedvig.metrics import classification_metrics, regression_metrics
from seedvig.models import RawEEGConformer
from seedvig.raw_dataset import LABEL_MODE_CHOICES, RawSeedVIGSequenceDataset, label_mode_num_classes
from seedvig.reference_results import compare_to_references


DEFAULT_DATA_ROOT = Path(r"D:\eeg-eog\data\SEED-VIG")


def _move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _loss(outputs, targets):
    class_loss = F.cross_entropy(outputs["class_logits"], targets["class"].long())
    regression_loss = F.smooth_l1_loss(outputs["perclos"].squeeze(-1), targets["perclos"].float())
    return class_loss + 0.5 * regression_loss, {
        "classification": class_loss.detach(),
        "regression": regression_loss.detach(),
    }


def _evaluate(model, loader, device, num_classes, max_batches=None, prefix="test"):
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
            outputs = model(batch["eeg"], eog=batch.get("eog"))
            loss, components = _loss(outputs, batch["targets"])
            totals["loss"] += float(loss.detach().cpu())
            totals["classification"] += float(components["classification"].cpu())
            totals["regression"] += float(components["regression"].cpu())
            class_predictions.append(outputs["class_logits"].argmax(dim=1).detach().cpu())
            class_targets.append(batch["targets"]["class"].long().detach().cpu())
            perclos_predictions.append(outputs["perclos"].squeeze(-1).detach().cpu())
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
):
    torch.manual_seed(seed)
    data_root = Path(data_root)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    num_classes = label_mode_num_classes(label_mode)

    dataset_args = {
        "root_path": data_root,
        "cache_dir": cache_dir,
        "include_eog": include_eog or use_eog_cross_attention,
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
        eeg_channels=17,
        embedding_dim=embedding_dim,
        attention_heads=attention_heads,
        window_transformer_layers=window_transformer_layers,
        temporal_layers=temporal_layers,
        num_classes=num_classes,
        max_sequence_length=max(sequence_length, 32),
        use_eog_cross_attention=use_eog_cross_attention,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
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
            outputs = model(batch["eeg"], eog=batch.get("eog"))
            loss, components = _loss(outputs, batch["targets"])
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
        }
        row.update(_evaluate(model, val_loader, device, num_classes, eval_max_batches, prefix="val"))
        improved = best_val_loss is None or row["val_loss"] < best_val_loss
        row["improved"] = improved
        if improved:
            best_val_loss = row["val_loss"]
            best_epoch = epoch
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "metrics": row}, best_model_path)
        row["best_epoch"] = best_epoch
        row["best_val_loss"] = best_val_loss
        _write_epoch_log(run_dir, row, write_header=(epoch == 1))
        last_metrics = row

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    last_metrics.update(_evaluate(model, test_loader, device, num_classes, eval_max_batches, prefix="test"))
    last_metrics.update(
        {
            "epochs_ran": epochs,
            "total_steps": total_steps,
            "best_model_path": str(best_model_path),
            "run_dir": str(run_dir),
            "selection_metric": "val_loss",
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
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--eval-max-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--include-eog", action="store_true")
    parser.add_argument("--use-eog-cross-attention", action="store_true")
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
        max_batches=args.max_batches,
        eval_max_batches=args.eval_max_batches,
        seed=args.seed,
        device=args.device,
        include_eog=args.include_eog,
        use_eog_cross_attention=args.use_eog_cross_attention,
    )
    for key, value in metrics.items():
        if isinstance(value, (int, float, bool, str)):
            print(f"{key}: {value}")
    print("reference_comparisons:")
    for item in metrics["reference_comparisons"]:
        print(json.dumps(item, ensure_ascii=False))


if __name__ == "__main__":
    main()
