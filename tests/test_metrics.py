from src.metrics import compute_classification_metrics


def test_compute_classification_metrics_returns_expected_keys():
    metrics = compute_classification_metrics(
        ["a", "b", "c"],
        ["a", "a", "c"],
    )

    assert set(metrics) == {
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
    }


def test_compute_classification_metrics_accuracy_is_correct():
    metrics = compute_classification_metrics(
        ["a", "b", "c", "d"],
        ["a", "b", "x", "x"],
    )

    assert metrics["accuracy"] == 0.5
