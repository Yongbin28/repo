"""CDAE-pretrained supervised temporal mixture-of-experts benchmark.

The network learns class probability and class-conditional response experts
jointly. All trace scaling, CDAE pretraining, supervised early stopping, and
response normalisation are fitted inside each unseen-lot outer training fold.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from waferpulse.experiments.cdae_trace_benchmark import (
    FoldTraceScaler,
    _require_experiment_dependencies,
    _train_autoencoder,
)
from waferpulse.tools.equipment_data import (
    load_equipment_dataset,
    load_equipment_trace_dataset,
)
from waferpulse.tools.equipment_modeling import QualityHurdleForestBlendRegressor


SEED = 42


def _align(data_dir: Path) -> Tuple[Any, np.ndarray]:
    engineered = load_equipment_dataset(data_dir, stage_mode="both")
    traces = load_equipment_trace_dataset(data_dir, stage_mode="both", max_invalid_rate=1.0)
    trace_index = pd.MultiIndex.from_frame(traces.identity[["lot", "wafer"]])
    target_index = pd.MultiIndex.from_frame(engineered.identity[["lot", "wafer"]])
    positions = trace_index.get_indexer(target_index)
    if (positions < 0).any():
        raise ValueError("Trace and response identities do not align")
    return engineered, traces.traces[positions]


def _build_model(encoder: Any, good_mean_z: float, bad_mean_z: float) -> Any:
    torch, nn, _, _, _ = _require_experiment_dependencies()

    class TemporalHurdleNetwork(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = encoder
            self.shared = nn.Sequential(
                nn.Linear(64 * 22, 128),
                nn.ReLU(),
                nn.Dropout(0.20),
                nn.Linear(128, 64),
                nn.ReLU(),
            )
            self.class_head = nn.Linear(64, 1)
            self.good_head = nn.Linear(64, 1)
            self.bad_head = nn.Linear(64, 1)
            nn.init.constant_(self.good_head.bias, good_mean_z)
            nn.init.constant_(self.bad_head.bias, bad_mean_z)

        def forward(self, values: Any) -> Tuple[Any, Any, Any, Any]:
            latent = self.encoder(values).flatten(start_dim=1)
            shared = self.shared(latent)
            logit = self.class_head(shared).squeeze(1)
            probability = torch.sigmoid(logit)
            good_response = self.good_head(shared).squeeze(1)
            bad_response = self.bad_head(shared).squeeze(1)
            mixture = (1.0 - probability) * good_response + probability * bad_response
            return logit, good_response, bad_response, mixture

    return TemporalHurdleNetwork()


def _loss(
    outputs: Tuple[Any, Any, Any, Any],
    response_z: Any,
    bad: Any,
    *,
    positive_weight: Any,
) -> Any:
    torch, _, _, _, _ = _require_experiment_dependencies()
    import torch.nn.functional as functional

    logit, good_response, bad_response, mixture = outputs
    class_loss = functional.binary_cross_entropy_with_logits(
        logit, bad, pos_weight=positive_weight
    )
    mixture_loss = functional.smooth_l1_loss(mixture, response_z, beta=0.25)
    good_mask = bad < 0.5
    bad_mask = ~good_mask
    good_loss = functional.smooth_l1_loss(
        good_response[good_mask], response_z[good_mask], beta=0.10
    )
    bad_loss = functional.smooth_l1_loss(
        bad_response[bad_mask], response_z[bad_mask], beta=0.25
    )
    return class_loss + 2.0 * mixture_loss + 0.5 * (good_loss + bad_loss)


def _fit_supervised(
    model: Any,
    train_traces: np.ndarray,
    train_response: np.ndarray,
    train_bad: np.ndarray,
    validation_traces: np.ndarray,
    validation_response: np.ndarray,
    validation_bad: np.ndarray,
    *,
    response_mean: float,
    response_scale: float,
    random_state: int,
    max_epochs: int = 120,
    patience: int = 20,
) -> Tuple[Any, Dict[str, Any]]:
    torch, _, DataLoader, TensorDataset, _ = _require_experiment_dependencies()
    torch.manual_seed(random_state)
    generator = torch.Generator().manual_seed(random_state)
    train_response_z = ((train_response - response_mean) / response_scale).astype(np.float32)
    validation_response_z = (
        (validation_response - response_mean) / response_scale
    ).astype(np.float32)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_traces),
            torch.from_numpy(train_response_z),
            torch.from_numpy(train_bad.astype(np.float32)),
        ),
        batch_size=32,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_x = torch.from_numpy(validation_traces)
    validation_y = torch.from_numpy(validation_response_z)
    validation_class = torch.from_numpy(validation_bad.astype(np.float32))
    positive_weight = torch.tensor(
        float((train_bad == 0).sum() / max((train_bad == 1).sum(), 1)),
        dtype=torch.float32,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    best_epoch = 0
    wait = 0
    history: List[float] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch_x, batch_y, batch_class in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(
                model(batch_x), batch_y, batch_class, positive_weight=positive_weight
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                _loss(
                    model(validation_x),
                    validation_y,
                    validation_class,
                    positive_weight=positive_weight,
                )
            )
        history.append(validation_loss)
        if validation_loss < best_loss - 1e-4:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "best_inner_loss": best_loss,
    }


def _predict(model: Any, traces: np.ndarray, mean: float, scale: float) -> Tuple[np.ndarray, np.ndarray]:
    torch, _, DataLoader, TensorDataset, _ = _require_experiment_dependencies()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(traces)),
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )
    probability: List[np.ndarray] = []
    response: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            logit, _, _, mixture = model(batch)
            probability.append(torch.sigmoid(logit).numpy())
            response.append((mixture.numpy() * scale) + mean)
    return np.concatenate(probability), np.concatenate(response)


def _regression_metrics(actual: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    return {
        "r2": float(r2_score(actual, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(actual, prediction))),
        "mae": float(mean_absolute_error(actual, prediction)),
    }


def run_benchmark(data_dir: Path, output_dir: Path) -> pd.DataFrame:
    dataset, traces = _align(data_dir)
    X = dataset.features.reset_index(drop=True)
    response = dataset.response.to_numpy(float)
    bad = dataset.bad_label.to_numpy(int)
    groups = dataset.groups.to_numpy(str)
    outer_folds = list(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(
            X, bad, groups
        )
    )
    predictions = {
        name: np.full(len(response), np.nan)
        for name in (
            "supervised_temporal_hurdle",
            "production_baseline",
            "baseline_75_temporal_25",
            "baseline_50_temporal_50",
            "baseline_25_temporal_75",
        )
    }
    probability = np.full(len(response), np.nan)
    fold_rows: List[Dict[str, Any]] = []

    for fold_number, (train, validation) in enumerate(outer_folds, start=1):
        started = time.perf_counter()
        scaler = FoldTraceScaler().fit(traces[train])
        train_traces = scaler.transform(traces[train])
        validation_traces = scaler.transform(traces[validation])
        inner_train_relative, inner_validation_relative = next(
            StratifiedGroupKFold(
                n_splits=5,
                shuffle=True,
                random_state=SEED + fold_number,
            ).split(train_traces, bad[train], groups[train])
        )
        response_mean = float(response[train][inner_train_relative].mean())
        response_scale = float(response[train][inner_train_relative].std(ddof=1))
        autoencoder, pretrain_history = _train_autoencoder(
            train_traces[inner_train_relative],
            epochs=40,
            batch_size=32,
            learning_rate=0.001,
            random_state=SEED + fold_number,
            architecture="paper",
        )
        good_mean_z = float(
            (
                response[train][inner_train_relative][bad[train][inner_train_relative] == 0].mean()
                - response_mean
            )
            / response_scale
        )
        bad_mean_z = float(
            (
                response[train][inner_train_relative][bad[train][inner_train_relative] == 1].mean()
                - response_mean
            )
            / response_scale
        )
        model = _build_model(autoencoder.encoder, good_mean_z, bad_mean_z)
        model, training_summary = _fit_supervised(
            model,
            train_traces[inner_train_relative],
            response[train][inner_train_relative],
            bad[train][inner_train_relative],
            train_traces[inner_validation_relative],
            response[train][inner_validation_relative],
            bad[train][inner_validation_relative],
            response_mean=response_mean,
            response_scale=response_scale,
            random_state=SEED + fold_number,
        )
        fold_probability, fold_prediction = _predict(
            model, validation_traces, response_mean, response_scale
        )
        probability[validation] = fold_probability
        predictions["supervised_temporal_hurdle"][validation] = fold_prediction

        baseline = QualityHurdleForestBlendRegressor(random_state=SEED + fold_number)
        baseline.fit(X.iloc[train], response[train])
        baseline_prediction = baseline.predict(X.iloc[validation])
        predictions["production_baseline"][validation] = baseline_prediction
        for baseline_weight in (0.75, 0.50, 0.25):
            name = f"baseline_{int(100 * baseline_weight)}_temporal_{int(100 * (1-baseline_weight))}"
            predictions[name][validation] = (
                baseline_weight * baseline_prediction
                + (1.0 - baseline_weight) * fold_prediction
            )
        fold_rows.append(
            {
                "fold": fold_number,
                "pretrain_final_mse": float(pretrain_history[-1]),
                **training_summary,
                "fold_r2": float(r2_score(response[validation], fold_prediction)),
                "fold_roc_auc": float(roc_auc_score(bad[validation], fold_probability)),
                "seconds": time.perf_counter() - started,
            }
        )
        print(json.dumps(fold_rows[-1]), flush=True)

    rows = [
        {"experiment": name, **_regression_metrics(response, prediction)}
        for name, prediction in predictions.items()
    ]
    results = pd.DataFrame(rows).sort_values("r2", ascending=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "supervised_temporal_hurdle_benchmark.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(
        output_dir / "supervised_temporal_hurdle_folds.csv", index=False
    )
    pd.DataFrame(
        {
            "lot": dataset.identity["lot"],
            "wafer": dataset.identity["wafer"],
            "response": response,
            "actual_bad": bad,
            "bad_probability": probability,
            **{f"prediction__{name}": value for name, value in predictions.items()},
        }
    ).to_csv(output_dir / "supervised_temporal_hurdle_oof.csv", index=False)
    metadata = {
        "validation": "5-fold StratifiedGroupKFold by manufacturing lot",
        "trace_shape": [int(value) for value in traces.shape],
        "classification": {
            "roc_auc": float(roc_auc_score(bad, probability)),
            "pr_auc": float(average_precision_score(bad, probability)),
        },
    }
    (output_dir / "supervised_temporal_hurdle_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return results


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    results = run_benchmark(root / "EquipmentData", root / "output" / "equipment_quality")
    print("\nSUPERVISED TEMPORAL HURDLE\n" + results.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
