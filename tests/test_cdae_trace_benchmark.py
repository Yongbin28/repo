import numpy as np
import pytest

from waferpulse.experiments.cdae_trace_benchmark import (
    FoldTraceScaler,
    _build_autoencoder,
    _encode,
    _train_autoencoder,
)


torch = pytest.importorskip("torch")


def test_fold_trace_scaler_uses_training_statistics_and_fills_missing_values() -> None:
    training = np.asarray(
        [
            [[1.0, np.nan, 3.0, 4.0], [10.0, 10.0, 10.0, 10.0]],
            [[2.0, 2.0, 2.0, 2.0], [12.0, 12.0, 12.0, 12.0]],
        ],
        dtype=np.float32,
    )
    validation = np.asarray([[[np.nan, 5.0, 6.0, 7.0], [14.0] * 4]], dtype=np.float32)

    scaler = FoldTraceScaler().fit(training)
    transformed = scaler.transform(validation)

    assert transformed.shape == validation.shape
    assert np.isfinite(transformed).all()
    assert np.isclose(scaler.median_[0], 2.0)


def test_cdae_reconstructs_original_shape_and_encodes_one_eighth_time_axis() -> None:
    rng = np.random.default_rng(42)
    traces = rng.normal(size=(8, 3, 176)).astype(np.float32)
    model, history = _train_autoencoder(
        traces,
        epochs=1,
        batch_size=4,
        learning_rate=0.001,
        random_state=42,
        architecture="paper",
    )
    with torch.no_grad():
        reconstructed = model(torch.from_numpy(traces))
    embedding = _encode(model, traces)

    assert reconstructed.shape == torch.Size([8, 3, 176])
    assert embedding.shape == (8, 64 * 22)
    assert len(history) == 1
    assert np.isfinite(history[0])
