import math

import torch


def _safe_divide(numerator, denominator):
    return 0.0 if denominator == 0 else numerator / denominator


def classification_metrics(predictions, targets, num_classes):
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for target, prediction in zip(targets.tolist(), predictions.tolist()):
        confusion[int(target), int(prediction)] += 1

    total = int(confusion.sum().item())
    correct = int(torch.diag(confusion).sum().item())
    accuracy = _safe_divide(correct, total)
    recalls = []
    f1_scores = []
    for class_index in range(num_classes):
        true_positive = int(confusion[class_index, class_index].item())
        false_positive = int(confusion[:, class_index].sum().item()) - true_positive
        false_negative = int(confusion[class_index, :].sum().item()) - true_positive
        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        recalls.append(recall)
        f1_scores.append(_safe_divide(2.0 * precision * recall, precision + recall))

    return {
        "accuracy": accuracy,
        "macro_f1": sum(f1_scores) / num_classes,
        "balanced_accuracy": sum(recalls) / num_classes,
        "confusion_matrix": confusion.tolist(),
    }


def regression_metrics(predictions, targets):
    errors = predictions - targets
    mae = torch.mean(torch.abs(errors)).item()
    rmse = math.sqrt(torch.mean(errors * errors).item())
    centered_predictions = predictions - predictions.mean()
    centered_targets = targets - targets.mean()
    denominator = torch.linalg.vector_norm(centered_predictions) * torch.linalg.vector_norm(centered_targets)
    pearson = 0.0
    if denominator.item() != 0.0:
        pearson = torch.sum(centered_predictions * centered_targets).item() / denominator.item()
    return {"mae": mae, "rmse": rmse, "pearson": pearson}
