"""Paper-like classification reproduction for the released CDAE strategy.

This module intentionally uses repeated stratified wafer holdouts so its result
can be compared with the published 30% validation protocol. It is not the
reportable production score because manufacturing lots can cross the split.
Use ``cdae_trace_benchmark`` for the strict unseen-lot evaluation and response
regression extension.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedShuffleSplit

from waferpulse.experiments.cdae_trace_benchmark import (
    FoldTraceScaler,
    _encode,
    _make_smote_svm,
    _train_autoencoder,
)
from waferpulse.tools.equipment_data import load_equipment_trace_dataset


SEED = 42


def _metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(actual, prediction)),
        "f1_bad": float(f1_score(actual, prediction, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(actual, prediction, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(actual, prediction, average="weighted", zero_division=0)
        ),
        "mcc": float(matthews_corrcoef(actual, prediction)),
        "recall_bad": float(recall_score(actual, prediction, pos_label=1, zero_division=0)),
        "precision_bad": float(
            precision_score(actual, prediction, pos_label=1, zero_division=0)
        ),
    }


def run_reproduction(
    data_dir: Path,
    output_dir: Path,
    *,
    repeats: int = 20,
    epochs: int = 80,
    batch_size: int = 32,
    random_state: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dataset = load_equipment_trace_dataset(
        data_dir,
        stage_mode="both",
        max_invalid_rate=1.0,
    )
    traces = dataset.traces
    bad = dataset.bad_label.to_numpy(int)
    splitter = StratifiedShuffleSplit(
        n_splits=repeats,
        test_size=0.30,
        random_state=random_state,
    )
    rows: List[dict[str, float | int]] = []
    started = time.perf_counter()

    for repeat, (train, validation) in enumerate(splitter.split(traces, bad), start=1):
        repeat_started = time.perf_counter()
        scaler = FoldTraceScaler().fit(traces[train])
        train_traces = scaler.transform(traces[train])
        validation_traces = scaler.transform(traces[validation])
        autoencoder, history = _train_autoencoder(
            train_traces,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=0.001,
            random_state=random_state + repeat,
            architecture="paper",
        )
        train_embedding = _encode(autoencoder, train_traces)
        validation_embedding = _encode(autoencoder, validation_traces)
        classifier = _make_smote_svm(random_state + repeat)
        classifier.fit(train_embedding, bad[train])
        prediction = classifier.predict(validation_embedding)
        row = {
            "repeat": repeat,
            "train_wafers": int(len(train)),
            "validation_wafers": int(len(validation)),
            "train_lots": int(dataset.groups.iloc[train].nunique()),
            "validation_lots": int(dataset.groups.iloc[validation].nunique()),
            "shared_lots": int(
                len(set(dataset.groups.iloc[train]) & set(dataset.groups.iloc[validation]))
            ),
            "reconstruction_mse": float(history[-1]),
            **_metrics(bad[validation], prediction),
            "seconds": time.perf_counter() - repeat_started,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    results = pd.DataFrame(rows)
    metric_columns = [
        "accuracy",
        "f1_bad",
        "f1_macro",
        "f1_weighted",
        "mcc",
        "recall_bad",
        "precision_bad",
        "reconstruction_mse",
    ]
    summary = pd.DataFrame(
        [
            {
                "metric": metric,
                "mean": float(results[metric].mean()),
                "std": float(results[metric].std(ddof=1)),
                "minimum": float(results[metric].min()),
                "maximum": float(results[metric].max()),
            }
            for metric in metric_columns
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "cdae_paper_reproduction_folds.csv", index=False)
    summary.to_csv(output_dir / "cdae_paper_reproduction_summary.csv", index=False)
    metadata = {
        "architecture_source": (
            "https://github.com/mpleschberger/DeepAutoEncoder/blob/main/deepAE.py"
        ),
        "paper": "https://doi.org/10.1109/TSM.2022.3146988",
        "protocol": f"{repeats} repeated stratified wafer splits with 30% validation",
        "warning": (
            "Paper-comparison diagnostic only: lots overlap train and validation, so this is "
            "not a deployable unseen-lot estimate."
        ),
        "epochs": epochs,
        "wafers": int(len(bad)),
        "bad_wafers": int(bad.sum()),
        "trace_shape": [int(value) for value in traces.shape],
        "seconds": time.perf_counter() - started,
    }
    (output_dir / "cdae_paper_reproduction_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return results, summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(list(argv) if argv is not None else None)
    root = Path(__file__).resolve().parents[2]
    _, summary = run_reproduction(
        root / "EquipmentData",
        root / "output" / "equipment_quality",
        repeats=args.repeats,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    print("\nPAPER-LIKE CDAE SUMMARY\n" + summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
