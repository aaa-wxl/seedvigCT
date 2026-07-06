REFERENCE_RESULTS = {
    "within_experiment_5fold": {
        "concat_fusion": {"accuracy": 0.7695, "rmse": 0.1382, "pearson": 0.8497},
        "random_forest": {"accuracy": 0.7749, "rmse": 0.2030, "pearson": 0.7963},
        "repaired_hmstenet": {"accuracy": 0.6637, "rmse": 0.3097, "pearson": 0.5706},
    },
    "group_subject": {
        "concat_group_subject": {"accuracy": 0.4912, "rmse": 0.2563, "pearson": 0.5746},
        "random_forest": {"accuracy": 0.5202, "rmse": 0.2930, "pearson": 0.5195},
        "repaired_hmstenet": {"accuracy": 0.6168, "rmse": 0.3182, "pearson": 0.5123},
    },
}


def compare_to_references(metrics, split_strategy):
    references = REFERENCE_RESULTS.get(split_strategy, {})
    comparisons = []
    for name, reference in references.items():
        item = {"reference": name}
        if "test_accuracy" in metrics:
            item["delta_accuracy"] = float(metrics["test_accuracy"]) - reference["accuracy"]
        if "test_rmse" in metrics:
            item["delta_rmse"] = float(metrics["test_rmse"]) - reference["rmse"]
        if "test_pearson" in metrics:
            item["delta_pearson"] = float(metrics["test_pearson"]) - reference["pearson"]
        item.update({f"reference_{key}": value for key, value in reference.items()})
        comparisons.append(item)
    return comparisons
