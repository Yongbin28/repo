"""Leakage-safe convolutional-autoencoder benchmark for EquipmentData.

This experiment adapts the published Jebril/Pleschberger/Susto pipeline:
raw multistage traces -> convolutional autoencoder -> RBF-SVM. Unlike the
paper's reported subject split, evaluation here uses unseen manufacturing lots.
The same fold-trained embeddings are also tested for continuous response
regression; those regression variants are an extension, not a paper result.

PyTorch and imbalanced-learn are optional experiment dependencies. Install
``requirements-experiments.txt`` before running this module.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from waferpulse.tools.equipment_data import (
    load_equipment_dataset,
    load_equipment_trace_dataset,
)
from waferpulse.tools.equipment_modeling import (
    QualityHurdleForestBlendRegressor,
    _choose_recall_guardrail_threshold,
)
from waferpulse.experiments.stacked_hurdle_benchmark import (
    _cross_fitted_probability_features as _cross_fitted_engineered_probabilities,
)


SEED = 42


def _require_experiment_dependencies(require_sampling: bool = True) -> Tuple[Any, Any, Any, Any, Any]:
    try:
        import torch
        extras = None
        if require_sampling:
            from imblearn.over_sampling import KMeansSMOTE
            from imblearn.pipeline import Pipeline as ImbalancedPipeline
            extras = (KMeansSMOTE, ImbalancedPipeline)
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError(
            "CDAE dependencies are missing. Install requirements-experiments.txt first."
        ) from exc
    return torch, nn, DataLoader, TensorDataset, extras


@dataclass
class FoldTraceScaler:
    """Channel-wise imputation and scaling learned from training lots only."""

    median_: np.ndarray | None = None
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, traces: np.ndarray) -> "FoldTraceScaler":
        values = np.asarray(traces, dtype=np.float32)
        self.median_ = np.nanmedian(values, axis=(0, 2)).astype(np.float32)
        self.median_ = np.nan_to_num(self.median_, nan=0.0)
        filled = self._fill(values)
        self.mean_ = filled.mean(axis=(0, 2), dtype=np.float64).astype(np.float32)
        scale = filled.std(axis=(0, 2), dtype=np.float64).astype(np.float32)
        self.scale_ = np.where(scale < 1e-6, 1.0, scale).astype(np.float32)
        return self

    def _fill(self, traces: np.ndarray) -> np.ndarray:
        if self.median_ is None:
            raise RuntimeError("FoldTraceScaler must be fitted before transform")
        values = np.asarray(traces, dtype=np.float32).copy()
        missing = ~np.isfinite(values)
        if missing.any():
            replacement = np.broadcast_to(self.median_[None, :, None], values.shape)
            values[missing] = replacement[missing]
        return values

    def transform(self, traces: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("FoldTraceScaler must be fitted before transform")
        filled = self._fill(traces)
        return (
            (filled - self.mean_[None, :, None]) / self.scale_[None, :, None]
        ).astype(np.float32)


def _build_autoencoder(input_channels: int, architecture: str = "paper") -> Any:
    _, nn, _, _, _ = _require_experiment_dependencies(require_sampling=False)
    import torch.nn.functional as functional

    if architecture not in {"paper", "adapted"}:
        raise ValueError("architecture must be 'paper' or 'adapted'")

    class CausalConv1d(nn.Module):
        """Keras-compatible causal Conv1D for kernel size three."""

        def __init__(self, input_count: int, output_count: int) -> None:
            super().__init__()
            self.convolution = nn.Conv1d(input_count, output_count, kernel_size=3)

        def forward(self, values: Any) -> Any:
            return self.convolution(functional.pad(values, (2, 0)))

    class ConvolutionalAutoencoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            convolution = CausalConv1d if architecture == "paper" else (
                lambda input_count, output_count: nn.Conv1d(
                    input_count, output_count, kernel_size=3, padding=1
                )
            )
            self.encoder = nn.Sequential(
                convolution(input_channels, 64),
                nn.ReLU(),
                nn.AvgPool1d(kernel_size=2),
                convolution(64, 64),
                nn.ReLU(),
                nn.AvgPool1d(kernel_size=2),
                convolution(64, 64),
                nn.ReLU(),
                nn.AvgPool1d(kernel_size=2),
            )
            if architecture == "paper":
                # Faithful to deepAE.py: Dense(22*7), reshape to (22, 7),
                # then three causal convolution/upsampling blocks.
                self.decoder_dense = nn.Sequential(nn.Linear(64 * 22, 22 * 7), nn.ReLU())
                self.decoder = nn.Sequential(
                    CausalConv1d(7, 64),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    CausalConv1d(64, 64),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    CausalConv1d(64, 64),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    CausalConv1d(64, input_channels),
                )
            else:
                self.decoder_dense = None
                self.decoder = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv1d(64, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv1d(64, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv1d(64, input_channels, kernel_size=3, padding=1),
                )

        def encode(self, values: Any) -> Any:
            return self.encoder(values)

        def forward(self, values: Any) -> Any:
            latent = self.encode(values)
            if self.decoder_dense is not None:
                compressed = self.decoder_dense(latent.flatten(start_dim=1))
                latent = compressed.reshape(len(values), 7, 22)
            return self.decoder(latent)

    return ConvolutionalAutoencoder()


def _train_autoencoder(
    traces: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    random_state: int,
    architecture: str = "paper",
) -> Tuple[Any, List[float]]:
    torch, nn, DataLoader, TensorDataset, _ = _require_experiment_dependencies(require_sampling=False)
    torch.manual_seed(random_state)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    generator = torch.Generator().manual_seed(random_state)
    model = _build_autoencoder(traces.shape[1], architecture=architecture).to("cpu")
    loader = DataLoader(
        TensorDataset(torch.from_numpy(traces)),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    history: List[float] = []
    model.train()
    for _ in range(epochs):
        total_loss = 0.0
        total_items = 0
        for (batch,) in loader:
            optimizer.zero_grad(set_to_none=True)
            reconstructed = model(batch)
            loss = loss_function(reconstructed, batch)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch)
            total_items += len(batch)
        history.append(total_loss / max(total_items, 1))
    return model, history


def _encode(model: Any, traces: np.ndarray, batch_size: int = 64) -> np.ndarray:
    torch, _, DataLoader, TensorDataset, _ = _require_experiment_dependencies(require_sampling=False)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(traces)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    rows: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            latent = model.encode(batch).flatten(start_dim=1)
            rows.append(latent.cpu().numpy())
    return np.concatenate(rows).astype(np.float32)


def _make_smote_svm(
    random_state: int,
    *,
    svm_c: float = 1.0,
    svm_gamma: str | float = "scale",
) -> Any:
    _, _, _, _, extras = _require_experiment_dependencies()
    KMeansSMOTE, ImbalancedPipeline = extras
    return ImbalancedPipeline(
        [
            ("scale", StandardScaler()),
            (
                "smote",
                KMeansSMOTE(
                    k_neighbors=3,
                    cluster_balance_threshold=0.0,
                    kmeans_estimator=MiniBatchKMeans(
                        n_clusters=8,
                        batch_size=128,
                        n_init=5,
                        random_state=random_state,
                    ),
                    random_state=random_state,
                ),
            ),
            ("svm", SVC(C=svm_c, gamma=svm_gamma, kernel="rbf")),
        ]
    )


def _cross_fitted_cdae_probability(
    train_embedding: np.ndarray,
    train_bad: np.ndarray,
    train_groups: np.ndarray,
    validation_embedding: np.ndarray,
    *,
    random_state: int,
    response: np.ndarray | None = None,
    tune_svm: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    folds = list(
        StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=random_state).split(
            train_embedding, train_bad, train_groups
        )
    )
    candidates: List[Tuple[float, str | float]] = [(1.0, "scale")]
    if tune_svm:
        candidates = [
            (c_value, gamma_value)
            for c_value in (0.1, 0.3, 1.0, 3.0, 10.0)
            for gamma_value in ("scale", 1e-4, 3e-4, 1e-3)
        ]

    # Scaling and KMeans-SMOTE do not depend on C/gamma. Reusing one resampled
    # inner-training set per fold both makes candidate comparisons fair and
    # avoids repeating the expensive clustering twenty times.
    prepared_folds: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    if tune_svm:
        _, _, _, _, extras = _require_experiment_dependencies()
        KMeansSMOTE, _ = extras
        for fold, (inner_train, inner_validation) in enumerate(folds, start=1):
            scaler = StandardScaler().fit(train_embedding[inner_train])
            scaled_train = scaler.transform(train_embedding[inner_train])
            scaled_validation = scaler.transform(train_embedding[inner_validation])
            sampler = KMeansSMOTE(
                k_neighbors=3,
                cluster_balance_threshold=0.0,
                kmeans_estimator=MiniBatchKMeans(
                    n_clusters=8,
                    batch_size=128,
                    n_init=5,
                    random_state=random_state + fold,
                ),
                random_state=random_state + fold,
            )
            resampled_embedding, resampled_bad = sampler.fit_resample(
                scaled_train, train_bad[inner_train]
            )
            prepared_folds.append(
                (
                    resampled_embedding,
                    resampled_bad,
                    scaled_validation,
                    inner_validation,
                )
            )

    candidate_rows: List[Dict[str, Any]] = []
    candidate_outputs: List[Tuple[np.ndarray, Any]] = []
    for candidate_number, (svm_c, svm_gamma) in enumerate(candidates, start=1):
        cross_score = np.full(len(train_embedding), np.nan, dtype=float)
        for fold, (inner_train, inner_validation) in enumerate(folds, start=1):
            if tune_svm:
                (
                    resampled_embedding,
                    resampled_bad,
                    scaled_validation,
                    validation_positions,
                ) = prepared_folds[fold - 1]
                classifier = SVC(C=svm_c, gamma=svm_gamma, kernel="rbf")
                classifier.fit(resampled_embedding, resampled_bad)
                cross_score[validation_positions] = classifier.decision_function(
                    scaled_validation
                )
            else:
                classifier = _make_smote_svm(
                    random_state + candidate_number * 100 + fold,
                    svm_c=svm_c,
                    svm_gamma=svm_gamma,
                )
                classifier.fit(train_embedding[inner_train], train_bad[inner_train])
                cross_score[inner_validation] = classifier.decision_function(
                    train_embedding[inner_validation]
                )
        if not np.isfinite(cross_score).all():
            raise ValueError("Cross-fitted CDAE classifier scores contain non-finite values")

        calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
        calibrator.fit(cross_score.reshape(-1, 1), train_bad)
        probability = calibrator.predict_proba(cross_score.reshape(-1, 1))[:, 1]
        row: Dict[str, Any] = {
            "svm_c": float(svm_c),
            "svm_gamma": str(svm_gamma),
            "inner_roc_auc": float(roc_auc_score(train_bad, probability)),
            "inner_pr_auc": float(average_precision_score(train_bad, probability)),
        }
        if response is not None:
            good_mean = float(response[train_bad == 0].mean())
            bad_mean = float(response[train_bad == 1].mean())
            hurdle = good_mean + probability * (bad_mean - good_mean)
            row["inner_hurdle_r2"] = float(r2_score(response, hurdle))
        candidate_rows.append(row)
        candidate_outputs.append((cross_score, calibrator))

    ranking_metric = "inner_hurdle_r2" if response is not None else "inner_roc_auc"
    best_index = max(
        range(len(candidate_rows)), key=lambda index: candidate_rows[index][ranking_metric]
    )
    selected = candidate_rows[best_index]
    cross_score, calibrator = candidate_outputs[best_index]
    train_probability = calibrator.predict_proba(cross_score.reshape(-1, 1))[:, 1]

    classifier = _make_smote_svm(
        random_state,
        svm_c=selected["svm_c"],
        svm_gamma=(
            selected["svm_gamma"]
            if selected["svm_gamma"] == "scale"
            else float(selected["svm_gamma"])
        ),
    )
    classifier.fit(train_embedding, train_bad)
    validation_score = classifier.decision_function(validation_embedding)
    validation_probability = calibrator.predict_proba(validation_score.reshape(-1, 1))[:, 1]
    diagnostics = {
        "ranking_metric": ranking_metric,
        "selected": selected,
        "candidates": candidate_rows,
    }
    return train_probability, validation_probability, diagnostics


def _extra_trees_regressor(random_state: int, selected_features: int = 256) -> Any:
    return SklearnPipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("variance", VarianceThreshold()),
            ("selector", SelectKBest(f_regression, k=selected_features)),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=250,
                    max_depth=10,
                    min_samples_leaf=3,
                    max_features="sqrt",
                    n_jobs=1,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _regression_metrics(actual: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    return {
        "r2": float(r2_score(actual, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(actual, prediction))),
        "mae": float(mean_absolute_error(actual, prediction)),
    }


def _classification_metrics(actual: np.ndarray, probability: np.ndarray) -> Dict[str, float]:
    prediction = (probability >= 0.5).astype(int)
    guardrail_threshold = _choose_recall_guardrail_threshold(actual, probability, 0.80)
    guardrail_prediction = (probability >= guardrail_threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(actual, probability)),
        "pr_auc": float(average_precision_score(actual, probability)),
        "f1_bad": float(f1_score(actual, prediction, zero_division=0)),
        "mcc": float(matthews_corrcoef(actual, prediction)),
        "recall_bad": float(recall_score(actual, prediction, zero_division=0)),
        "precision_bad": float(precision_score(actual, prediction, zero_division=0)),
        "recall_guardrail_threshold": float(guardrail_threshold),
        "guardrail_recall_bad": float(
            recall_score(actual, guardrail_prediction, zero_division=0)
        ),
        "guardrail_precision_bad": float(
            precision_score(actual, guardrail_prediction, zero_division=0)
        ),
        "guardrail_f1_bad": float(f1_score(actual, guardrail_prediction, zero_division=0)),
        "guardrail_mcc": float(matthews_corrcoef(actual, guardrail_prediction)),
    }


def _probability_lot_context(probability: np.ndarray, groups: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame({"probability": probability, "lot": groups})
    grouped = frame.groupby("lot", sort=False)["probability"]
    return np.column_stack(
        [
            probability,
            grouped.transform("mean"),
            grouped.transform("std").fillna(0.0),
            grouped.transform("min"),
            grouped.transform("max"),
            grouped.transform(lambda values: values.quantile(0.25)),
            grouped.transform(lambda values: values.quantile(0.75)),
        ]
    )


def _lot_severity_prediction(
    train_context: np.ndarray,
    train_response: np.ndarray,
    train_bad: np.ndarray,
    train_groups: np.ndarray,
    validation_context: np.ndarray,
    validation_groups: np.ndarray,
) -> np.ndarray:
    context_columns = ["p", "p_mean", "p_std", "p_min", "p_max", "p_q25", "p_q75"]
    train_frame = pd.DataFrame(train_context, columns=context_columns)
    train_frame["lot"] = train_groups
    train_frame["response"] = train_response
    train_frame["bad"] = train_bad
    lot_features = train_frame.groupby("lot", sort=False)[context_columns[1:]].first()
    bad_mean = (
        train_frame.loc[train_frame["bad"].eq(1)]
        .groupby("lot", sort=False)["response"]
        .mean()
    )
    eligible = lot_features.index.intersection(bad_mean.index)
    severity_model = SklearnPipeline(
        [("scale", StandardScaler()), ("model", Ridge(alpha=10.0))]
    ).fit(lot_features.loc[eligible], bad_mean.loc[eligible])

    validation_frame = pd.DataFrame(validation_context, columns=context_columns)
    validation_frame["lot"] = validation_groups
    validation_lot_features = validation_frame.groupby("lot", sort=False)[
        context_columns[1:]
    ].first()
    predicted_by_lot = pd.Series(
        severity_model.predict(validation_lot_features),
        index=validation_lot_features.index,
    ).clip(0.75, 2.1)
    return pd.Series(validation_groups).map(predicted_by_lot).to_numpy(float)


def _align_traces(
    data_dir: Path,
    max_invalid_rate: float,
    stage_mode: str = "both",
) -> Tuple[Any, np.ndarray]:
    engineered = load_equipment_dataset(data_dir, stage_mode=stage_mode)
    trace = load_equipment_trace_dataset(
        data_dir,
        stage_mode=stage_mode,
        max_invalid_rate=max_invalid_rate,
    )
    trace_index = pd.MultiIndex.from_frame(trace.identity[["lot", "wafer"]])
    engineered_index = pd.MultiIndex.from_frame(engineered.identity[["lot", "wafer"]])
    positions = trace_index.get_indexer(engineered_index)
    if (positions < 0).any():
        raise ValueError("Raw traces and engineered features do not contain the same wafers")
    return engineered, trace.traces[positions]


def run_benchmark(
    data_dir: Path,
    output_dir: Path,
    *,
    epochs: int = 80,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    random_state: int = SEED,
    architecture: str = "paper",
    max_invalid_rate: float = 1.0,
    include_engineered_stack: bool = False,
    ensemble_seeds: int = 1,
    tune_svm: bool = False,
    stage_mode: str = "both",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if ensemble_seeds < 1:
        raise ValueError("ensemble_seeds must be at least one")
    engineered, traces = _align_traces(
        data_dir,
        max_invalid_rate=max_invalid_rate,
        stage_mode=stage_mode,
    )
    X = engineered.features.reset_index(drop=True)
    response = engineered.response.to_numpy(float)
    bad = engineered.bad_label.to_numpy(int)
    groups = engineered.groups.to_numpy(str)
    folds = list(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state).split(
            X, bad, groups
        )
    )

    prediction_names = [
        "production_baseline",
        "cdae_latent_ridge",
        "cdae_latent_extra_trees",
        "engineered_plus_cdae_extra_trees",
        "engineered_plus_cdae_probability_extra_trees",
        "cdae_probability_hurdle",
        "baseline_75_cdae_hurdle_25",
        "baseline_50_cdae_hurdle_50",
            "baseline_25_cdae_hurdle_75",
            "lot_context_cdae_hurdle",
            "baseline_75_lot_context_25",
            "baseline_50_lot_context_50",
            "baseline_25_lot_context_75",
    ]
    if include_engineered_stack:
        prediction_names.extend(
            [
                "fused_probability_hurdle",
                "baseline_75_fused_hurdle_25",
                "baseline_50_fused_hurdle_50",
                "baseline_25_fused_hurdle_75",
            ]
        )
    predictions = {name: np.full(len(response), np.nan) for name in prediction_names}
    cdae_probability = np.full(len(response), np.nan)
    lot_context_probability = np.full(len(response), np.nan)
    fused_probability = np.full(len(response), np.nan) if include_engineered_stack else None
    engineered_probability_sources: Tuple[str, ...] = ()
    fold_rows: List[Dict[str, Any]] = []
    tuning_rows: List[Dict[str, Any]] = []
    started = time.perf_counter()

    for fold_number, (train, validation) in enumerate(folds, start=1):
        fold_started = time.perf_counter()
        scaler = FoldTraceScaler().fit(traces[train])
        train_traces = scaler.transform(traces[train])
        validation_traces = scaler.transform(traces[validation])
        seed_train_probabilities: List[np.ndarray] = []
        seed_validation_probabilities: List[np.ndarray] = []
        history: List[float] = []
        train_embedding: np.ndarray | None = None
        validation_embedding: np.ndarray | None = None
        for seed_index in range(ensemble_seeds):
            seed = random_state + fold_number + seed_index * 1000
            autoencoder, seed_history = _train_autoencoder(
                train_traces,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                random_state=seed,
                architecture=architecture,
            )
            seed_train_embedding = _encode(autoencoder, train_traces)
            seed_validation_embedding = _encode(autoencoder, validation_traces)
            (
                seed_train_probability,
                seed_validation_probability,
                tuning,
            ) = _cross_fitted_cdae_probability(
                seed_train_embedding,
                bad[train],
                groups[train],
                seed_validation_embedding,
                random_state=random_state + fold_number * 10 + seed_index * 1000,
                response=response[train],
                tune_svm=tune_svm,
            )
            seed_train_probabilities.append(seed_train_probability)
            seed_validation_probabilities.append(seed_validation_probability)
            tuning_rows.append(
                {
                    "fold": fold_number,
                    "ensemble_seed": seed_index + 1,
                    **tuning["selected"],
                    "ranking_metric": tuning["ranking_metric"],
                }
            )
            if seed_index == 0:
                history = seed_history
                train_embedding = seed_train_embedding
                validation_embedding = seed_validation_embedding

        train_probability = np.mean(seed_train_probabilities, axis=0)
        validation_probability = np.mean(seed_validation_probabilities, axis=0)
        assert train_embedding is not None and validation_embedding is not None
        cdae_probability[validation] = validation_probability

        train_lot_context = _probability_lot_context(
            train_probability, groups[train]
        )
        validation_lot_context = _probability_lot_context(
            validation_probability, groups[validation]
        )
        lot_probability_model = LogisticRegression(
            C=0.3,
            solver="lbfgs",
            max_iter=1000,
        ).fit(train_lot_context, bad[train])
        validation_lot_probability = lot_probability_model.predict_proba(
            validation_lot_context
        )[:, 1]
        lot_context_probability[validation] = validation_lot_probability
        validation_bad_severity = _lot_severity_prediction(
            train_lot_context,
            response[train],
            bad[train],
            groups[train],
            validation_lot_context,
            groups[validation],
        )

        validation_fused_probability: np.ndarray | None = None
        if include_engineered_stack:
            (
                train_engineered_meta,
                validation_engineered_meta,
                engineered_probability_sources,
            ) = _cross_fitted_engineered_probabilities(
                X.iloc[train],
                bad[train],
                groups[train],
                X.iloc[validation],
                random_state=random_state + fold_number * 100,
            )
            fusion = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
            fusion.fit(
                np.column_stack([train_probability, train_engineered_meta]),
                bad[train],
            )
            validation_fused_probability = fusion.predict_proba(
                np.column_stack([validation_probability, validation_engineered_meta])
            )[:, 1]
            assert fused_probability is not None
            fused_probability[validation] = validation_fused_probability

        baseline = QualityHurdleForestBlendRegressor(random_state=random_state + fold_number)
        baseline.fit(X.iloc[train], response[train])
        baseline_prediction = baseline.predict(X.iloc[validation])
        predictions["production_baseline"][validation] = baseline_prediction

        ridge = SklearnPipeline(
            [("scale", StandardScaler()), ("model", Ridge(alpha=100.0))]
        ).fit(train_embedding, response[train])
        predictions["cdae_latent_ridge"][validation] = ridge.predict(validation_embedding)

        latent_regressor = _extra_trees_regressor(
            random_state + fold_number, selected_features=256
        ).fit(train_embedding, response[train])
        predictions["cdae_latent_extra_trees"][validation] = latent_regressor.predict(
            validation_embedding
        )

        train_combined = np.column_stack([X.iloc[train].to_numpy(), train_embedding])
        validation_combined = np.column_stack(
            [X.iloc[validation].to_numpy(), validation_embedding]
        )
        combined_regressor = _extra_trees_regressor(random_state + fold_number).fit(
            train_combined, response[train]
        )
        predictions["engineered_plus_cdae_extra_trees"][validation] = (
            combined_regressor.predict(validation_combined)
        )

        probability_regressor = _extra_trees_regressor(random_state + fold_number).fit(
            np.column_stack([train_combined, train_probability]), response[train]
        )
        predictions["engineered_plus_cdae_probability_extra_trees"][validation] = (
            probability_regressor.predict(
                np.column_stack([validation_combined, validation_probability])
            )
        )

        good_mean = float(response[train][bad[train] == 0].mean())
        bad_mean = float(response[train][bad[train] == 1].mean())
        hurdle = good_mean + validation_probability * (bad_mean - good_mean)
        predictions["cdae_probability_hurdle"][validation] = hurdle
        predictions["baseline_75_cdae_hurdle_25"][validation] = (
            0.75 * baseline_prediction + 0.25 * hurdle
        )
        predictions["baseline_50_cdae_hurdle_50"][validation] = (
            0.50 * baseline_prediction + 0.50 * hurdle
        )
        predictions["baseline_25_cdae_hurdle_75"][validation] = (
            0.25 * baseline_prediction + 0.75 * hurdle
        )
        lot_hurdle = good_mean + validation_lot_probability * (
            validation_bad_severity - good_mean
        )
        predictions["lot_context_cdae_hurdle"][validation] = lot_hurdle
        predictions["baseline_75_lot_context_25"][validation] = (
            0.75 * baseline_prediction + 0.25 * lot_hurdle
        )
        predictions["baseline_50_lot_context_50"][validation] = (
            0.50 * baseline_prediction + 0.50 * lot_hurdle
        )
        predictions["baseline_25_lot_context_75"][validation] = (
            0.25 * baseline_prediction + 0.75 * lot_hurdle
        )
        if validation_fused_probability is not None:
            fused_hurdle = good_mean + validation_fused_probability * (bad_mean - good_mean)
            predictions["fused_probability_hurdle"][validation] = fused_hurdle
            predictions["baseline_75_fused_hurdle_25"][validation] = (
                0.75 * baseline_prediction + 0.25 * fused_hurdle
            )
            predictions["baseline_50_fused_hurdle_50"][validation] = (
                0.50 * baseline_prediction + 0.50 * fused_hurdle
            )
            predictions["baseline_25_fused_hurdle_75"][validation] = (
                0.25 * baseline_prediction + 0.75 * fused_hurdle
            )
        fold_rows.append(
            {
                "fold": fold_number,
                "train_wafers": int(len(train)),
                "validation_wafers": int(len(validation)),
                "train_lots": int(pd.Series(groups[train]).nunique()),
                "validation_lots": int(pd.Series(groups[validation]).nunique()),
                "latent_features": int(train_embedding.shape[1]),
                "reconstruction_mse_first": float(history[0]),
                "reconstruction_mse_final": float(history[-1]),
                "classification_roc_auc": float(
                    roc_auc_score(bad[validation], validation_probability)
                ),
                "seconds": time.perf_counter() - fold_started,
            }
        )
        print(json.dumps(fold_rows[-1]), flush=True)

    if not all(np.isfinite(values).all() for values in predictions.values()):
        raise ValueError("One or more regression variants contain incomplete predictions")
    if not np.isfinite(cdae_probability).all():
        raise ValueError("CDAE classification probability is incomplete")

    results = pd.DataFrame(
        [
            {"experiment": name, **_regression_metrics(response, prediction)}
            for name, prediction in predictions.items()
        ]
    ).sort_values("r2", ascending=False)
    classification = _classification_metrics(bad, cdae_probability)
    lot_context_classification = _classification_metrics(bad, lot_context_probability)
    fused_classification = (
        _classification_metrics(bad, fused_probability)
        if fused_probability is not None
        else None
    )
    metadata = {
        "method": (
            f"{architecture} E64-E64-E64 CDAE + KMeans-SMOTE RBF-SVM"
        ),
        "validation": "5-fold StratifiedGroupKFold by manufacturing lot",
        "probability_calibration": "4-fold inner StratifiedGroupKFold sigmoid calibration",
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "random_state": int(random_state),
        "ensemble_seeds": int(ensemble_seeds),
        "svm_tuning": bool(tune_svm),
        "wafers": int(len(response)),
        "lots": int(pd.Series(groups).nunique()),
        "trace_shape": [int(value) for value in traces.shape],
        "engineered_feature_count": int(X.shape[1]),
        "architecture": architecture,
        "max_invalid_rate": float(max_invalid_rate),
        "stage_mode": stage_mode,
        "classification": classification,
        "lot_context_classification": lot_context_classification,
        "fused_classification": fused_classification,
        "engineered_probability_sources": list(engineered_probability_sources),
        "seconds": time.perf_counter() - started,
        "paper_reference": "https://doi.org/10.1109/TSM.2022.3146988",
        "claim_limit": (
            "Classification adapts the published method under stricter lot grouping. "
            "Continuous-response variants are new experiments, not published results."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    variant = f"{architecture}{traces.shape[1]}"
    if stage_mode != "both":
        variant += f"_{stage_mode}"
    if include_engineered_stack:
        variant += "_fused"
    if ensemble_seeds > 1:
        variant += f"_s{ensemble_seeds}"
    if tune_svm:
        variant += "_tuned"
    metadata["variant"] = variant
    fold_frame = pd.DataFrame(fold_rows)
    prediction_ledger = pd.DataFrame(
        {
            "lot": engineered.identity["lot"],
            "wafer": engineered.identity["wafer"],
            "response": response,
            "actual_bad": bad,
            "cdae_bad_probability": cdae_probability,
            "lot_context_bad_probability": lot_context_probability,
            **(
                {"fused_bad_probability": fused_probability}
                if fused_probability is not None
                else {}
            ),
            **{f"prediction__{name}": value for name, value in predictions.items()},
        }
    )
    results.to_csv(output_dir / "cdae_trace_regression_benchmark.csv", index=False)
    fold_frame.to_csv(output_dir / "cdae_trace_fold_metrics.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(
        output_dir / "cdae_trace_svm_tuning.csv", index=False
    )
    prediction_ledger.to_csv(output_dir / "cdae_trace_oof_predictions.csv", index=False)
    (output_dir / "cdae_trace_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    results.to_csv(
        output_dir / f"cdae_trace_regression_benchmark_{variant}.csv", index=False
    )
    fold_frame.to_csv(output_dir / f"cdae_trace_fold_metrics_{variant}.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(
        output_dir / f"cdae_trace_svm_tuning_{variant}.csv", index=False
    )
    prediction_ledger.to_csv(
        output_dir / f"cdae_trace_oof_predictions_{variant}.csv", index=False
    )
    (output_dir / f"cdae_trace_metadata_{variant}.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return results, metadata


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--architecture", choices=("paper", "adapted"), default="paper")
    parser.add_argument(
        "--stage",
        choices=("both", "equipment1", "equipment2"),
        default="both",
        help="Use the two-stage intersection or every labeled wafer available at one stage.",
    )
    parser.add_argument(
        "--reliable-sensors-only",
        action="store_true",
        help="Use the production 49-sensor lane instead of the paper-like 56-sensor lane.",
    )
    parser.add_argument(
        "--include-engineered-stack",
        action="store_true",
        help="Fuse CDAE probability with cross-fitted engineered-model probabilities.",
    )
    parser.add_argument(
        "--ensemble-seeds",
        type=int,
        default=1,
        help="Average probabilities from independently trained CDAEs in every outer fold.",
    )
    parser.add_argument(
        "--tune-svm",
        action="store_true",
        help="Select RBF-SVM C/gamma using grouped inner-fold response R-squared.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    root = Path(__file__).resolve().parents[2]
    results, metadata = run_benchmark(
        root / "EquipmentData",
        root / "output" / "equipment_quality",
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        architecture=args.architecture,
        max_invalid_rate=0.01 if args.reliable_sensors_only else 1.0,
        include_engineered_stack=args.include_engineered_stack,
        ensemble_seeds=args.ensemble_seeds,
        tune_svm=args.tune_svm,
        stage_mode=args.stage,
    )
    print("\nCDAE REGRESSION RESULTS\n" + results.to_string(index=False), flush=True)
    print("\nCDAE CLASSIFICATION\n" + json.dumps(metadata["classification"], indent=2))


if __name__ == "__main__":
    main()
