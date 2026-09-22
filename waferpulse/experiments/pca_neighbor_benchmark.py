"""Cross-lot nearest-neighbour audit in fold-trained raw-trace PCA space."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from waferpulse.tools.equipment_data import load_equipment_trace_dataset


SEED = 42


def _metrics(actual: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    return {
        "r2": float(r2_score(actual, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(actual, prediction))),
        "mae": float(mean_absolute_error(actual, prediction)),
    }


def run_benchmark(data_dir: Path, output_dir: Path) -> pd.DataFrame:
    dataset = load_equipment_trace_dataset(data_dir, max_invalid_rate=1.0)
    raw = dataset.traces.reshape(len(dataset.response), -1).astype(np.float64)
    response = dataset.response.to_numpy(float)
    bad = dataset.bad_label.to_numpy(int)
    groups = dataset.groups.to_numpy(str)
    folds = list(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(
            raw, bad, groups
        )
    )
    keys = [(components, neighbours) for components in (16, 32, 64, 128) for neighbours in (5, 10, 20, 40)]
    class_probability = {key: np.full(len(response), np.nan) for key in keys}
    direct_prediction = {key: np.full(len(response), np.nan) for key in keys}
    hurdle_prediction = {
        (components, neighbours, scale): np.full(len(response), np.nan)
        for components, neighbours in keys
        for scale in (1.0, 1.5, 2.0, 3.0)
    }
    started = time.perf_counter()
    for fold, (train, validation) in enumerate(folds, start=1):
        scaler = StandardScaler().fit(raw[train])
        train_scaled = scaler.transform(raw[train])
        validation_scaled = scaler.transform(raw[validation])
        pca = PCA(n_components=128, svd_solver="randomized", random_state=SEED + fold)
        train_embedding = pca.fit_transform(train_scaled)
        validation_embedding = pca.transform(validation_scaled)
        good_mean = float(response[train][bad[train] == 0].mean())
        bad_mean = float(response[train][bad[train] == 1].mean())
        for components, neighbours in keys:
            classifier = KNeighborsClassifier(
                n_neighbors=neighbours,
                weights="distance",
                metric="euclidean",
            ).fit(train_embedding[:, :components], bad[train])
            probability = classifier.predict_proba(
                validation_embedding[:, :components]
            )[:, 1]
            class_probability[(components, neighbours)][validation] = probability
            regressor = KNeighborsRegressor(
                n_neighbors=neighbours,
                weights="distance",
                metric="euclidean",
            ).fit(train_embedding[:, :components], response[train])
            direct_prediction[(components, neighbours)][validation] = regressor.predict(
                validation_embedding[:, :components]
            )
            for scale in (1.0, 1.5, 2.0, 3.0):
                calibrated = np.clip(probability * scale, 0.0, 1.0)
                hurdle_prediction[(components, neighbours, scale)][validation] = (
                    good_mean + calibrated * (bad_mean - good_mean)
                )
        print(json.dumps({"fold": fold, "pca_variance": float(pca.explained_variance_ratio_.sum())}), flush=True)

    rows = []
    for key, probability in class_probability.items():
        components, neighbours = key
        rows.append(
            {
                "experiment": f"pca{components}_knn{neighbours}_direct",
                **_metrics(response, direct_prediction[key]),
                "roc_auc": float(roc_auc_score(bad, probability)),
                "pr_auc": float(average_precision_score(bad, probability)),
            }
        )
        for scale in (1.0, 1.5, 2.0, 3.0):
            rows.append(
                {
                    "experiment": f"pca{components}_knn{neighbours}_hurdle_scale{scale}",
                    **_metrics(response, hurdle_prediction[(components, neighbours, scale)]),
                    "roc_auc": float(roc_auc_score(bad, probability)),
                    "pr_auc": float(average_precision_score(bad, probability)),
                }
            )
    result = pd.DataFrame(rows).sort_values("r2", ascending=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "pca_neighbor_benchmark.csv", index=False)
    (output_dir / "pca_neighbor_metadata.json").write_text(
        json.dumps(
            {
                "validation": "5-fold StratifiedGroupKFold by manufacturing lot",
                "seconds": time.perf_counter() - started,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    result = run_benchmark(root / "EquipmentData", root / "output" / "equipment_quality")
    print("\nPCA NEIGHBOUR RESULTS\n" + result.head(20).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
