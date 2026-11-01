"""Audit EquipmentData and compare cleaning variants on identical held-out lots.

Run: python verify_training_quality.py
Outputs are written separately from production model artifacts.
"""
from pathlib import Path
import json
import joblib
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

from waferpulse.tools import equipment_data as data
from waferpulse.experiments.bosch_plasma_etch_benchmark import load_dataset


def numeric_only(raw):
    numeric = pd.to_numeric(raw, errors="coerce")
    return numeric, pd.Series(False, index=raw.index), raw.notna() & numeric.isna()


def main():
    root = Path(__file__).resolve().parent
    output = root / "output" / "training_quality_verification"
    output.mkdir(parents=True, exist_ok=True)
    variants = {}
    for name, threshold in [("current_cleaning", 0.01), ("recovery_without_sensor_exclusion", 1.0)]:
        print(f"Loading {name}", flush=True)
        variants[name] = data.load_equipment_dataset(root / "EquipmentData", max_invalid_rate=threshold)
    # The comparator keeps mandatory key/label validation and deduplication;
    # raw string tables cannot fairly be passed directly to numeric regressors.
    with patch.object(data, "_coerce_sensor_numeric", numeric_only):
        print("Loading numeric_coercion_only", flush=True)
        variants["numeric_coercion_only"] = data.load_equipment_dataset(root / "EquipmentData", max_invalid_rate=1.0)
    joblib.dump(variants, output / "equipment_audit_inputs.joblib")
    reference = variants["current_cleaning"]
    splits = list(GroupKFold(n_splits=5).split(reference.features, reference.response, reference.groups))
    audits, scores, folds, ledgers = {}, [], [], []
    for name, dataset in variants.items():
        pd.testing.assert_frame_equal(reference.identity, dataset.identity)
        pd.testing.assert_series_equal(reference.response, dataset.response)
        X, y = dataset.features, dataset.response
        audits[name] = {
            "quality_summary": dataset.quality_summary,
            "missing_feature_cells": int(X.isna().sum().sum()),
            "missing_feature_rate": float(X.isna().to_numpy().mean()),
            "infinite_feature_cells": int(np.isinf(X.to_numpy()).sum()),
            "all_missing_features": X.columns[X.isna().all()].tolist(),
            "constant_features": X.columns[X.nunique(dropna=True).le(1)].tolist(),
            "duplicate_wafer_keys": int(dataset.identity.duplicated().sum()),
            "finite_targets": bool(np.isfinite(y).all()),
        }
        dataset.sensor_quality.to_csv(output / f"{name}_sensor_quality.csv", index=False)
        models = {"mean_baseline": DummyRegressor(), "extra_trees": ExtraTreesRegressor(n_estimators=100, min_samples_leaf=3, max_features=0.7, random_state=42, n_jobs=1)}
        for model_name, estimator in models.items():
            print(f"Evaluating {name}: {model_name}", flush=True)
            prediction = np.full(len(y), np.nan)
            for fold, (train, test) in enumerate(splits, 1):
                assert set(dataset.groups.iloc[train]).isdisjoint(dataset.groups.iloc[test])
                model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), clone(estimator))
                model.fit(X.iloc[train], y.iloc[train])
                prediction[test] = model.predict(X.iloc[test])
                folds.append({"variant": name, "model": model_name, "fold": fold, "rmse": float(np.sqrt(mean_squared_error(y.iloc[test], prediction[test]))), "r2": float(r2_score(y.iloc[test], prediction[test]))})
            scores.append({"variant": name, "model": model_name, "features": X.shape[1], "r2": float(r2_score(y, prediction)), "rmse": float(np.sqrt(mean_squared_error(y, prediction))), "mae": float(mean_absolute_error(y, prediction))})
            ledgers.append(dataset.identity.assign(variant=name, model=model_name, actual=y, predicted=prediction))
    # Check losses at joins rather than interpreting every exclusion as corruption.
    response = data._read_semicolon_csv(root / "EquipmentData" / "response.csv")
    response = data._normalise_keys(response).drop_duplicates(["lot", "wafer"])
    matched = set(reference.identity.itertuples(index=False, name=None))
    unmatched = response.loc[[key not in matched for key in response[["lot", "wafer"]].itertuples(index=False, name=None)]]
    unmatched.to_csv(output / "unmatched_response_wafers.csv", index=False)
    report = {"scope": "EquipmentData only; not Bosch or local STDF", "validation": "Identical 5-fold GroupKFold by lot; fold-local median imputation; fixed ExtraTrees parameters; no tuning", "comparison_limit": "All variants retain identity validation, label checks, response deduplication and temporal aggregation. This isolates sensor recovery/exclusion; it does not estimate the effect of every cleaning step.", "unmatched_response_wafers": len(unmatched), "source_sha256": reference.provenance["source_sha256"], "audit": audits}
    (output / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(scores).to_csv(output / "comparison.csv", index=False)
    pd.DataFrame(folds).to_csv(output / "fold_metrics.csv", index=False)
    pd.concat(ledgers, ignore_index=True).to_csv(output / "predictions.csv", index=False)
    print(pd.DataFrame(scores).to_string(index=False), flush=True)
    print(f"Report: {output}", flush=True)


def verify_bosch():
    root = Path(__file__).resolve().parent
    source = root / "dataset" / "bosch_plasma_etch"
    output = root / "output" / "training_quality_verification" / "bosch"
    output.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(source)
    process = pd.read_csv(source / "bosch_process_wafer_features.csv")
    measurements = dataset.measurements
    targets = measurements.groupby(["experiment_key", "lot_number"], as_index=False)["si_etch"].mean()
    joined = targets.merge(process, on="experiment_key", how="inner", validate="one_to_one")
    y, groups = joined.si_etch, joined.lot_number
    columns = [c for c in dataset.process if c != "experiment_key"]
    all_columns = [c for c in process if c != "experiment_key"]
    current = joined[columns]
    usable = current.columns[current.nunique(dropna=True).gt(1) & current.notna().any()]
    variants = {"current_cleaning": current, "without_sparse_column_exclusion": joined[all_columns], "remove_constant_columns": current[usable]}
    audit = {"metadata": dataset.metadata, "scope": "Wafer-average silicon etch; process inputs only; existing feature cache", "duplicate_process_keys": int(process.experiment_key.duplicated().sum()), "duplicate_measurement_coordinates": int(measurements.duplicated(["experiment_key", "X", "Y"]).sum()), "missing_measurement_cells": int(measurements.isna().sum().sum()), "finite_target": bool(np.isfinite(y).all()), "unmatched_process_keys": sorted(set(process.experiment_key) - set(targets.experiment_key)), "unmatched_measurement_keys": sorted(set(targets.experiment_key) - set(process.experiment_key)), "source_sha256": {p.name: data._sha256(p) for p in [source / "Process_data.nc", source / "Dictionary_process.nc", source / "bosch_process_wafer_features.csv", source / "Si_Oxide_etch_89_points.csv"]}, "variants": {}}
    splits = list(GroupKFold(n_splits=5).split(current, y, groups))
    scores, folds = [], []
    from sklearn.ensemble import GradientBoostingRegressor
    for name, X in variants.items():
        audit["variants"][name] = {"features": X.shape[1], "missing_cells": int(X.isna().sum().sum()), "infinite_cells": int(np.isinf(X.to_numpy()).sum()), "constant_columns": X.columns[X.nunique(dropna=True).le(1)].tolist(), "all_missing_columns": X.columns[X.isna().all()].tolist()}
        for model_name, estimator in {"mean_baseline": DummyRegressor(), "gradient_boosting": GradientBoostingRegressor(random_state=42)}.items():
            print(f"Bosch {name}: {model_name}", flush=True)
            prediction = np.full(len(y), np.nan)
            for fold, (train, test) in enumerate(splits, 1):
                assert set(groups.iloc[train]).isdisjoint(groups.iloc[test])
                model = make_pipeline(SimpleImputer(strategy="median"), clone(estimator))
                model.fit(X.iloc[train], y.iloc[train])
                prediction[test] = model.predict(X.iloc[test])
                folds.append({"variant": name, "model": model_name, "fold": fold, "rmse": float(np.sqrt(mean_squared_error(y.iloc[test], prediction[test])))})
            scores.append({"variant": name, "model": model_name, "r2": float(r2_score(y, prediction)), "rmse": float(np.sqrt(mean_squared_error(y, prediction))), "mae": float(mean_absolute_error(y, prediction))})
    (output / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    pd.DataFrame(scores).to_csv(output / "comparison.csv", index=False)
    pd.DataFrame(folds).to_csv(output / "fold_metrics.csv", index=False)
    print(pd.DataFrame(scores).to_string(index=False), flush=True)


def verify_sources():
    """Independently inspect source keys and decoded means against the cache."""
    from netCDF4 import Dataset
    from waferpulse.experiments.bosch_plasma_etch_benchmark import group_to_experiment_key, _safe_name
    root = Path(__file__).resolve().parent
    output = root / "output" / "training_quality_verification"
    checks = {}
    for name in ("equipment1", "equipment2"):
        frame = data._read_semicolon_csv(root / "EquipmentData" / f"{name}.csv")
        timestamps = data._timestamp_index(frame.timestamp)
        bad_keys = frame[["lot", "wafer"]].isna().any(axis=1) | frame[["lot", "wafer"]].fillna("").apply(lambda c: c.str.strip().eq("")).any(axis=1)
        checks[name] = {"missing_or_blank_key_rows": int(bad_keys.sum()), "timestamp_min": int(timestamps.min()), "timestamp_max": int(timestamps.max()), "out_of_expected_range": int((~timestamps.between(0, 175)).sum())}
    (output / "source_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    source = root / "dataset" / "bosch_plasma_etch"
    cache = pd.read_csv(source / "bosch_process_wafer_features.csv").set_index("experiment_key")
    checks = {"checked_means": 0, "mean_mismatches": 0, "raw_nonfinite_readings": 0, "channels_with_nonfinite": 0, "missing_rate_mismatches": 0, "missing_rate_not_in_cache": 0, "cache_missing_mean_columns": 0}
    with Dataset(source / "Dictionary_process.nc") as file:
        decoder = np.asarray(file.variables["data"][:], dtype=float)
    with Dataset(source / "Process_data.nc") as file:
        for name, group in file.groups.items():
            key = group_to_experiment_key(name)
            values = decoder[np.asarray(group.variables["data"][:], dtype=np.uint16)]
            names = np.asarray(group.variables["feature"][:], dtype=str)
            for index, sensor in enumerate(names):
                prefix = "signal__" + _safe_name(sensor)
                signal = values[:, index]
                finite = np.isfinite(signal)
                count = int((~finite).sum())
                checks["raw_nonfinite_readings"] += count
                checks["channels_with_nonfinite"] += int(count > 0)
                if prefix + "__mean" not in cache:
                    checks["cache_missing_mean_columns"] += 1
                    continue
                expected = np.mean(signal[finite]) if finite.any() else np.nan
                checks["checked_means"] += 1
                checks["mean_mismatches"] += int(not np.isclose(expected, cache.loc[key, prefix + "__mean"], rtol=1e-8, atol=1e-8, equal_nan=True))
                if prefix + "__missing_rate" in cache:
                    checks["missing_rate_mismatches"] += int(not np.isclose((~finite).mean(), cache.loc[key, prefix + "__missing_rate"], equal_nan=True))
                else:
                    checks["missing_rate_not_in_cache"] += 1
    (output / "bosch" / "source_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
    verify_bosch()
    verify_sources()
