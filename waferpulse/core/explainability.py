"""Explain fitted pipelines without relying on version-specific SHAP pickles."""

import joblib
import numpy as np
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def _steps(transformer):
    if isinstance(transformer, Pipeline):
        for _, step in transformer.steps:
            yield from _steps(step)
    elif transformer is not None and transformer != "passthrough":
        yield transformer


def _original_linear_inputs(pipe, row, background, coefficients):
    """Fold affine preprocessing into coefficients, keeping imputed probe inputs."""
    steps = list(_steps(pipe[:-1]))
    weights = np.asarray(coefficients, dtype=float).copy()
    for step in reversed(steps):
        if isinstance(step, PCA):
            if step.whiten:
                weights = weights / np.sqrt(step.explained_variance_)
            weights = weights @ step.components_
        elif isinstance(step, StandardScaler):
            if step.with_std:
                weights = weights / step.scale_
        elif isinstance(step, SimpleImputer):
            break
        else:
            raise ValueError("Original-feature explanations require supported affine preprocessing")

    data, reference = row, background
    names = list(row.columns)
    for step in steps:
        if isinstance(step, SimpleImputer):
            names = list(step.get_feature_names_out(names))
            data, reference = step.transform(data), step.transform(reference)
        else:
            # Scaling and PCA have already been folded into the coefficients.
            break
    data = np.asarray(data, dtype=float)[0]
    reference = np.asarray(reference, dtype=float)
    if len(names) != len(weights):
        raise ValueError("Original feature names do not match effective linear coefficients")
    return names, data, reference.mean(axis=0), weights


def explain_prediction(pipe, row, background, explainer_path, target_index=0):
    transformed = pipe[:-1].transform(row)
    names = list(pipe[:-1].get_feature_names_out())
    estimator = pipe[-1]
    if isinstance(estimator, MultiOutputRegressor):
        estimator = estimator.estimators_[target_index]

    if isinstance(estimator, (Lasso, Ridge, ElasticNet)):
        if background is None or background.empty:
            raise ValueError("Training reference data is required for linear SHAP explanations")
        coefficients = np.asarray(estimator.coef_)
        if coefficients.ndim > 1:
            coefficients = coefficients[target_index]
        # Exact interventional SHAP for a linear model: beta * (x - E[X]).
        # Compute directly from the fitted model, avoiding SHAP/Numba imports
        # and serialized explainer compatibility across deployment versions.
        names, data, mean, coefficients = _original_linear_inputs(
            pipe, row, background, coefficients
        )
        values = coefficients * (data - mean)
        predictions = np.asarray(pipe.predict(background))
        base = float(predictions[:, target_index].mean() if predictions.ndim > 1 else predictions.mean())
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
