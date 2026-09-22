import numpy as np

from waferpulse.experiments.lot_ranking_diagnostic import (
    cross_fitted_class_mean_prediction,
    oracle_count_selection,
    within_lot_rank_average,
)


def test_oracle_count_selection_preserves_each_lots_true_bad_count() -> None:
    lots = np.array(["a", "a", "a", "b", "b"])
    actual = np.array([1, 0, 0, 1, 1])
    probability = np.array([0.7, 0.9, 0.1, 0.2, 0.8])

    selected = oracle_count_selection(lots, actual, probability)

    assert selected.tolist() == [0, 1, 0, 1, 1]
    for lot in np.unique(lots):
        mask = lots == lot
        assert selected[mask].sum() == actual[mask].sum()


def test_class_mean_prediction_uses_training_folds_only() -> None:
    response = np.array([0.0, 10.0, 2.0, 20.0])
    actual = np.array([0, 1, 0, 1])
    fold = np.array([1, 1, 2, 2])

    result = cross_fitted_class_mean_prediction(
        response, actual, fold, predicted_bad=actual
    )

    assert result.tolist() == [2.0, 20.0, 0.0, 10.0]


def test_within_lot_rank_average_is_scale_free() -> None:
    import pandas as pd

    reference = pd.DataFrame({"lot": ["a", "a", "b", "b"]})
    first = np.array([0.2, 0.8, 10.0, 20.0])
    same_order_different_scale = np.array([2.0, 80.0, -4.0, 7.0])

    result = within_lot_rank_average(
        reference, [first, same_order_different_scale]
    )

    assert result.tolist() == [0.5, 1.0, 0.5, 1.0]
