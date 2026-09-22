import numpy as np
import pandas as pd

from waferpulse.experiments.bosch_plasma_etch_benchmark import (
    _group_oof_predictions,
    coordinate_features,
    group_to_experiment_key,
    model_candidates,
)


def test_group_name_maps_to_measurement_key() -> None:
    assert (
        group_to_experiment_key("Day_2024_07_02_Wafer_01")
        == "2024-07-02_01"
    )


def test_coordinate_features_do_not_include_metrology() -> None:
    frame = pd.DataFrame(
        {
            "X": [0.0, 30_000.0],
            "Y": [0.0, 40_000.0],
            "si_etch": [999.0, 999.0],
            "preox_thickness": [999.0, 999.0],
        }
    )
    features = coordinate_features(frame)

    assert "si_etch" not in features
    assert "preox_thickness" not in features
    assert np.isclose(features.loc[1, "coordinate__radius"], 0.5)


def test_group_oof_never_fits_the_held_out_group() -> None:
    features = pd.DataFrame({"value": [0.0, 0.0, 10.0, 10.0]})
    target = np.array([0.0, 0.0, 10.0, 10.0])
    groups = np.array([1, 1, 2, 2])
    dummy = model_candidates(1)["dummy_mean"]

    predicted = _group_oof_predictions(dummy, features, target, groups)

    assert np.allclose(predicted[:2], 10.0)
    assert np.allclose(predicted[2:], 0.0)
