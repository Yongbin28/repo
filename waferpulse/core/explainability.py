"""Explain fitted pipelines without relying on version-specific SHAP pickles."""

import joblib
import numpy as np
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.multioutput import MultiOutputRegressor


def explain_prediction(pipe, row, background, explainer_path, target_index=0):
    transformed = pipe[:-1].transform(row)
    names = list(pipe[:-1].get_feature_names_out())
    estimator = pipe[-1]
    if isinstance(estimator, MultiOutputRegressor):
        estimator = estimator.estimators_[target_index]

    if isinstance(estimator, (Lasso, Ridge, ElasticNet)):
        if background is None or background.empty:
            raise ValueError("Training reference data is required for linear SHAP explanations")
        reference = pipe[:-1].transform(background)
        mean = np.asarray(reference.mean(axis=0)).ravel()
        coefficients = np.asarray(estimator.coef_)
        intercept = np.asarray(estimator.intercept_)
        if coefficients.ndim > 1:
            coefficients = coefficients[target_index]
            intercept = intercept[target_index]
        # Exact interventional SHAP for a linear model: beta * (x - E[X]).
        # Compute directly from the fitted model, avoiding SHAP/Numba imports
        # and serialized explainer compatibility across deployment versions.
        data = np.asarray(transformed).reshape(len(row), -1)[0]
        values = coefficients * (data - mean)
        base = float(intercept + coefficients @ mean)
    else:
        import shap

        if not explainer_path.exists():
            raise FileNotFoundError(f"No SHAP explainer for {explainer_path.name}")
        explanation = joblib.load(explainer_path)(transformed)
        values = np.asarray(explanation.values)
        base_values = np.asarray(explanation.base_values)
        values = values[0, :, target_index] if values.ndim == 3 else values[0]
        base = float(base_values[0, target_index] if base_values.ndim > 1 else base_values.reshape(-1)[0])
        data = np.asarray(transformed)[0]

    if len(names) != len(values):
        raise ValueError("SHAP feature names do not match transformed model features")
    return {"values": values.tolist(), "base_values": base,
            "data": data.tolist(), "feature_names": names}
