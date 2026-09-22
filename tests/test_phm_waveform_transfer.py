import numpy as np
import pandas as pd

from waferpulse.experiments.phm_waveform_transfer_benchmark import (
    build_fold_features,
    extract_external_waveforms,
    normalize_waveforms,
)


def test_external_waveforms_use_only_requested_upstream_signals() -> None:
    frame = pd.DataFrame(
        {
            "WAFER_ID": [1, 1, 1],
            "STAGE": ["A", "A", "A"],
            "TIMESTAMP": [0, 1, 2],
            "_source_index": [0, 0, 0],
            "PRESSURIZED_CHAMBER_PRESSURE": [1.0, 2.0, 3.0],
            "AVG_REMOVAL_RATE": [999.0, 999.0, 999.0],
        }
    )

    waveforms, signals = extract_external_waveforms(
        frame, points=5, signals=["PRESSURIZED_CHAMBER_PRESSURE"]
    )

    assert signals == ("PRESSURIZED_CHAMBER_PRESSURE",)
    assert waveforms.shape == (1, 5)
    assert np.isclose(waveforms.mean(), 0.0)
    assert np.isclose(waveforms.std(), 1.0)


def test_fold_feature_basis_does_not_depend_on_validation_values() -> None:
    rng = np.random.default_rng(42)
    train = rng.normal(size=(8, 3, 12))
    validation = rng.normal(size=(2, 3, 12))
    external = normalize_waveforms(rng.normal(size=(20, 12)))

    train_a, validation_a, _ = build_fold_features(
        train, validation, external, components=4
    )
    changed_validation = validation + 1000.0
    train_b, validation_b, _ = build_fold_features(
        train, changed_validation, external, components=4
    )

    assert np.allclose(train_a, train_b)
    assert not np.allclose(validation_a, validation_b)
