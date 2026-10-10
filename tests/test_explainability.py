from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
