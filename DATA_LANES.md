# WaferPulse Data Lanes and Evidence Boundary

WaferPulse separates real-data model evidence from local functional demonstrations. The separation is enforced by `waferpulse/core/data_lanes.py` and independent application workspaces. The overall package structure is documented in [ARCHITECTURE.md](ARCHITECTURE.md).

## Lane A — Real open-source prediction

Source files:

- `EquipmentData/equipment1.csv`
- `EquipmentData/equipment2.csv`
- `EquipmentData/response.csv`

Permitted functions:

- source-data validation and provenance;
- wafer-level temporal feature engineering;
- continuous wafer-response regression;
- bad-wafer classification (`response > 0.75`);
- stratified lot-grouped validation;
- champion-model feature importance and prediction ledger.

This lane validates **upstream equipment sensors → wafer-test response**. It does not validate direct wafer-probe to final-test prediction, field reliability, or MTTF.

Reliability safeguards:

- identical response duplicates are removed and conflicting labels fail the run;
- each equipment sequence must contain 176 unique ordered timestamps;
- spreadsheet-rendered decimal values are deterministically recovered and counted in the sensor audit;
- sensors above the configured source-format rate are excluded from the default reliable feature set;
- all missing-value imputation and target-based feature selection occur inside each validation fold;
- folds are stratified while keeping each manufacturing lot entirely within one fold;
- reported predictions are out-of-fold predictions, not fitted training predictions;
- a single regression and classification champion is persisted only after model comparison.
- random-wafer, true-class oracle, and within-class scores are labelled diagnostics and are never substituted for the reportable lot-grouped score.

Evidence is written to `output/equipment_quality/`:

- `metrics.json`
- `fold_metrics.csv`
- `prediction_ledger_oof.csv`
- `feature_importance.csv`
- `sensor_quality.csv`
- `provenance.json`
- `equipment_quality_models.joblib`
- `quality_summary.json`
- `r2_experiment_benchmark.csv`
- `r2_engineered_nine_algorithm_summary.csv`
- `r2_calibration_blend.csv`
- `r2_repeated_group_validation.csv`
- `r2_stacked_hurdle_benchmark.csv`
- `r2_stacked_hurdle_summary.csv`

The nine-algorithm and parameter ledgers are experiment evidence. `metrics.json` is the authoritative deployed-model score after the selected candidate is rerun through the production lot-grouped pipeline.

## Lane B — Bosch plasma-etch quality prediction

Source files:

- `dataset/bosch_plasma_etch/Process_data.nc` (or `data/bosch_plasma_etch/Process_data.nc`)
- `dataset/bosch_plasma_etch/Dictionary_process.nc`
- `dataset/bosch_plasma_etch/Si_Oxide_etch_89_points.csv`

Permitted functions:

- upstream plasma-etch trace validation and feature engineering via `ml_compute_statistic.col_stats`;
- continuous wafer-average silicon-etch regression using Report Table 3-5 algorithms;
- response-first research high-etch screening;
- coordinate-assisted 89-point spatial virtual metrology;
- five-fold manufacturing-lot-held-out validation.

This lane validates **upstream Bosch process traces → downstream wafer-average
silicon etch**. It also evaluates coordinate-assisted silicon-etch maps. It does
not validate final electrical wafer test, yield, field reliability, or MTTF.

Reliability safeguards:

- manufacturing lots remain intact in each validation fold;
- dates, lot/wafer identifiers and downstream metrology are excluded from the
  sensor-only wafer-average predictors;
- the high-etch category is applied only after continuous response prediction;
- the default 44.0 threshold is labelled as a research rule because the public
  source supplies no official production limit;
- map scores are disclosed as coordinate-assisted because a coordinate-only
  baseline already explains most of the spatial pattern;
- oxide-etch generalization is poor and is not promoted as a supported result.

Evidence is written to `output/bosch_plasma_etch/` and
`output/bosch_response_then_classify/`:

- `bosch_benchmark_report.xlsx` (multi-tab Excel report: Summary, Top_3_Models, Wafer_Average_Si_Etch, Map_Point_Si_Etch, All_Metrics)
- `model_benchmark_summary.csv`
- `top_3_models.csv`
- `metrics.csv`
- `oof_predictions.csv`
- `summary.json`
- `oof_response_then_class.csv`

Using the 9 algorithms proposed in Report Table 3-5 and unified features from `ml_compute_statistic`, Lasso and ElasticNet predict wafer-average silicon etch at R² **0.8837**, RMSE **0.1363**, and MAE **0.1091**, followed by LightGBM at R² **0.8324** and XGBoost at R² **0.8323**. Lasso also gives the strongest research high-etch F1 of **0.9000** at the default 44.0 boundary. The process-plus-coordinate silicon map reaches R² **0.9925** (LightGBM).

## Lane C — Local/STDF demonstration

Permitted functions:

- STDF parsing verification;
- historical/golden baseline demonstration;
- SPC, Cpk and PSI;
- specification-proximity monitoring;
- GDBN spatial-pattern analysis;
- alerts, grading and dashboard integration.

This lane demonstrates that the functions operate correctly on local or specification-generated data. Results from this lane must not be reported as real-world predictive accuracy.

## Running the system

Start the Streamlit application and choose **Model Prediction & Validation**.
Select **EquipmentData CSV benchmark** for Lane A, **Zenodo BOSCH plasma-etch
benchmark** for Lane B, or **Local dataset analysis** for a read-only Lane C
exploration. **Model Preparation Pipeline** and **Wafer Analytics &
Prediction** also remain Lane C workspaces and display a demonstration-data notice.

```powershell
.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```
