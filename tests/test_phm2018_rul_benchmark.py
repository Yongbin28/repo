import numpy as np
import pandas as pd

from waferpulse.experiments.phm2018_rul_benchmark import (
    expanding_cycle_splits,
    failure_cycle_ids,
)


def test_failure_cycle_ids_increment_only_after_observed_reset() -> None:
    labels = pd.Series([12.0, 8.0, 4.0, 0.0, 20.0, 16.0, np.nan, np.nan])

    cycles = failure_cycle_ids(labels)

    assert cycles.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]


def test_expanding_splits_never_train_on_future_cycle() -> None:
    cycles = np.repeat(np.arange(5), 2)

    splits = expanding_cycle_splits(cycles)

    assert len(splits) == 2
    for train_index, test_index in splits:
        assert cycles[train_index].max() < cycles[test_index].min()


def test_single_cycle_is_not_evaluable() -> None:
    assert expanding_cycle_splits(np.zeros(10, dtype=int)) == []
