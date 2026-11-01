# Training quality verification — 7 October 2026

## Conclusion

Bosch produces usable inputs and a useful wafer-average silicon-etch predictor under manufacturing-lot-held-out validation. Its sparse-column exclusion improves the tested model slightly, but the improvement is too small and inconsistent across folds to establish a strong general performance benefit from cleaning.

EquipmentData passes structural checks, but this experiment does **not** establish that its sensor recovery/exclusion improves prediction. Clean numeric inputs alone are insufficient evidence of a high-quality predictive dataset.

## Experiment

Run `python verify_training_quality.py` from the project root. Outputs are separate from existing production model artifacts in `output/training_quality_verification`.

Each comparison uses the same wafer identities, targets, five GroupKFold splits by manufacturing lot, model parameters and random seed. Median imputation is fitted inside each training fold. No lot appears in both training and validation. Model parameters were fixed rather than tuned against these results.

EquipmentData compares current cleaning, decimal recovery with all sensor columns retained, and numeric coercion without decimal recovery or sensor exclusions. Every variant still validates labels/keys, removes duplicate response records, and uses identical temporal aggregation. This comparison isolates sensor handling; it does not measure the benefit of every cleaning step.

Bosch compares current cached features with all cached columns and with constant columns removed. It predicts wafer-average silicon etch from process features only, without X/Y coordinates. It evaluates existing feature preparation rather than comparing undecoded raw strings with a numeric model.

## EquipmentData findings

- 1,319 exact duplicate response rows removed; 1,319 unique labelled wafers remain, with zero conflicting labels or threshold mismatches.
- Equipment 1 has 971 wafers; equipment 2 has 1,319. Requiring both stages produces 971 matched wafers across 40 lots. The other 348 labelled wafers are excluded by stage availability, not sensor cleaning. Their identities are exported for review.
- All source lot/wafer IDs are present; all timestamps are within 0–175. The loader confirms 176 observations per wafer and rejects duplicate timestamps.
- Seven of 56 sensor columns are excluded by the configured 1% source-format rule. Retained sensor columns contain 8,366 recovered readings.
- Current training matrix: 971 rows, 686 numeric features, zero missing or infinite feature cells, zero duplicate wafer identities, and finite targets. However, 190 features are constant.
- Classes: 801 good and 170 bad (17.5% bad); 22 of 40 lots contain only one class. Large trace-row counts do not mean equally large independent training sample counts.

Fixed ExtraTrees comparison (lower RMSE is better):

| Sensor handling | R² | RMSE | MAE |
|---|---:|---:|---:|
| Current cleaning | -0.1051 | 0.4444 | 0.2828 |
| Recovery, no sensor exclusions | -0.1307 | 0.4495 | 0.2825 |
| Numeric coercion only | -0.0972 | 0.4428 | 0.2748 |
| Training-fold mean baseline | -0.0690 | 0.4371 | 0.3002 |

Current cleaning improves RMSE over retaining all recovered sensors, but worsens RMSE relative to numeric coercion only. It beats numeric coercion in only two of five folds. All three ExtraTrees variants have worse pooled RMSE than the mean baseline. This result concerns this fixed estimator and split; it does not establish that all models fail.

Existing saved production metrics use a different estimator and stratified lot splits: R² 0.1597, RMSE 0.3875. These are historical artifacts rather than newly reproduced scores. They also show limited predictive strength and are not a cleaning ablation.

Date-like decimal recovery follows the code's documented source-format assumption. Tests confirm the conversion algorithm, but original pre-spreadsheet measurements are unavailable here to verify every recovered value independently.

## Bosch findings

- 96 process traces; 88 measured and matched wafers across 10 lots. Eight traces lack downstream labels and are excluded from supervised training.
- 7,832 measurement rows: exactly 89 coordinates per wafer, no duplicate wafer/coordinate records, and one lot identity per wafer.
- Silicon-etch targets are complete at all 89 points per wafer. The 157 missing measurement cells belong to `postox_thickness_nan`; this report does not verify that separate target.
- Independently decoded 3,106 wafer/channel means from the NetCDF source and compared them with cached features: zero mismatches at numerical tolerance. Decoded readings contain no nonfinite values. This verifies cached means, not every cached statistic.
- Current training matrix: 88 rows, 312 features, zero missing or infinite cells; 97 features are constant.
- Keeping all 442 cached columns creates 10,270 missing cells in the matched matrix. Some channels exist only for selected wafers; the fold imputer skips columns with no observed training values.

Fixed GradientBoosting comparison:

| Feature handling | R² | RMSE | MAE |
|---|---:|---:|---:|
| Current cleaning | 0.6676 | 0.2304 | 0.1826 |
| No sparse-column exclusions | 0.6598 | 0.2331 | 0.1844 |
| Constant columns removed | 0.6699 | 0.2296 | 0.1821 |
| Training-fold mean baseline | -0.0326 | 0.4061 | 0.3498 |

Current cleaning lowers RMSE by approximately 1.15% versus retaining sparse columns, and improves only two of five folds. It lowers RMSE by approximately 43.3% versus the mean baseline, which supports the usefulness of the process features and tested model together. That baseline comparison does not isolate cleaning's contribution. Removing constant columns yields a further small improvement, not a demonstrated general gain.

## Practical implications and limits

1. Use Bosch for the stronger current training evidence. Preserve lot-held-out validation and training-fold preprocessing.
2. For EquipmentData, investigate signal/target relationships, stage coverage and lot shift before claiming that cleaning produces strong predictive training data. Keep the full-stage and equipment-2-only populations separate when reporting results.
3. Remove constant/all-missing columns using a transformer fitted on training folds. The constant-column Bosch comparison here identifies constants across the supplied matrix and is exploratory.
4. Current sensor-quality and sparse-column eligibility rules inspect the complete supplied input before cross-validation. They do not use target values, but strictly inductive deployment validation should learn data-dependent eligibility from training lots or freeze a sensor schema from an independent source audit.
5. These are fixed-model, single-split-plan comparisons, with 40 EquipmentData lots and only 10 Bosch lots. Small differences across cleaning variants should not be presented as statistically established improvements or factory-readiness.
6. Source hashes, sensor-quality records, missingness audits, fold scores and EquipmentData prediction ledgers are saved with the results. No production cleaning rules or trained models were changed.
