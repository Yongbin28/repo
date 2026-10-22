from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from waferpulse.core.explainability import explain_prediction


@pytest.mark.parametrize("wrapped", [False, True])
def test_linear_shap_without_saved_explainer_matches_selected_prediction(wrapped):
    background = pd.DataFrame({"probe_a": [1., 2., 4., 8.],
                               "probe_b": [2., np.nan, 3., 7.]})
    targets = np.array([[10., 2.], [20., 4.], [40., 8.], [80., 16.]])
    estimator = MultiOutputRegressor(Lasso(alpha=.01)) if wrapped else Lasso(alpha=.01)
    pipe = make_pipeline(SimpleImputer(), StandardScaler(), estimator)
    pipe.fit(background, targets)
    row = pd.DataFrame({"probe_a": [5.], "probe_b": [np.nan]})
    result = explain_prediction(pipe, row, background, Path("missing.joblib"), 1)
    assert result["feature_names"] == ["probe_a", "probe_b"]
    np.testing.assert_allclose(
        result["base_values"] + sum(result["values"]), pipe.predict(row)[0, 1]
    )
    baseline = explain_prediction(pipe, background, background, Path("missing.joblib"), 1)
    np.testing.assert_allclose(baseline["base_values"], pipe.predict(background)[:, 1].mean())


@pytest.mark.parametrize("whiten", [False, True])
@pytest.mark.parametrize("wrapped", [False, True])
def test_pca_explanations_use_original_measurements_and_preserve_prediction(whiten, wrapped):
    rng = np.random.default_rng(42)
    background = pd.DataFrame(rng.normal(size=(30, 4)), columns=[
        "TP_t3_0__mean", "TP_t3_0__std", "TP_t4_0__mean", "TP_t5_0__mean"
    ])
    target = np.column_stack([background.iloc[:, 0] * 2, background.iloc[:, 1] * 5])
    background.iloc[0, 2] = np.nan
    estimator = MultiOutputRegressor(Lasso(alpha=.01)) if wrapped else Lasso(alpha=.01)
    # Match production's nested preprocessing and dimension reduction.
    pipe = make_pipeline(make_pipeline(SimpleImputer(), StandardScaler(),
                                       PCA(n_components=3, whiten=whiten)), estimator)
    pipe.fit(background, target)
    row = background.iloc[[0]]
    result = explain_prediction(pipe, row, background, Path("missing.joblib"), 1)
    assert result["feature_names"] == list(background.columns)
    assert len(result["values"]) == 4
    np.testing.assert_allclose(result["base_values"] + sum(result["values"]),
                               pipe.predict(row)[0, 1], atol=1e-10)
    # Attribute a one-parameter change to that parameter, not to a PCA label.
    changed = row.copy()
    changed.iloc[0, 0] += 1
    other = explain_prediction(pipe, changed, background, Path("missing.joblib"), 1)
    np.testing.assert_allclose(np.array(other["values"])[1:], np.array(result["values"])[1:])
    np.testing.assert_allclose(sum(other["values"]) - sum(result["values"]),
                               pipe.predict(changed)[0, 1] - pipe.predict(row)[0, 1], atol=1e-10)
