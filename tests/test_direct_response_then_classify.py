import numpy as np

from waferpulse.experiments.direct_response_then_classify_benchmark import (
    evaluate_predictions,
    response_to_bad,
)


def test_response_class_is_derived_with_strict_threshold() -> None:
    response = np.array([0.40, 0.75, 0.7501, 1.02])

    assert response_to_bad(response, threshold=0.75).tolist() == [0, 0, 1, 1]


def test_metrics_categorize_predicted_response_not_true_class() -> None:
    actual = np.array([0.40, 1.00, 0.42, 1.10])
    predicted = np.array([0.39, 0.70, 0.43, 0.90])

    metrics = evaluate_predictions(actual, predicted, threshold=0.75)

    assert metrics["tn"] == 2
    assert metrics["fp"] == 0
    assert metrics["fn"] == 1
    assert metrics["tp"] == 1
    assert metrics["recall_bad"] == 0.5
