# PHM16 CMP upstream-to-downstream validation

## Decision

The PHM Society 2016 CMP virtual-metrology lane passes the requested aggregate
R² > 0.8 gate on an untouched chronological manufacturing-campaign holdout:

- selected model: LightGBM;
- development campaigns 0–14 grouped OOF R²: 0.979660;
- untouched latest campaigns 15–18 R²: 0.961533;
- untouched RMSE: 5.546461;
- untouched MAE: 4.249775;
- untouched samples: 349 wafer-stage records;
- physical-wafer overlap between development and final holdout: 0.

This verifies prediction of downstream **average material-removal-rate (MRR)
metrology** from upstream CMP process traces. It does not verify final
electrical wafer test, yield, field reliability, or the original EquipmentData
response target.

## Dataset and quality contract

- Authority: PHM Society 2016 CMP Data Challenge specification.
- Transport mirror: Zenodo DOI `10.5281/zenodo.19803296`.
- Raw scope: 185 source files, 672,744 process rows, 1,981 wafer-stage labels,
  and 1,699 physical wafers.
- Inputs: equipment/chamber context, consumable usage, pressures, slurry flows,
  rotations, water status, duration, phase statistics, and a Preston-style
  pressure-speed proxy.
- Excluded predictors: `WAFER_ID`, source-file identity, absolute timestamp,
  and `AVG_REMOVAL_RATE`.
- Published quality rule: four MRR values above 4,000 are documented label
  outliers while normal values are below 170. A fixed ceiling of 300 removes
exactly those four records before modeling, leaving 1,977 records. This rule
is declared before model selection and final evaluation.

As a sensitivity check, retaining all four documented outliers and fitting the
same selected LightGBM method gives final-campaign R² 0.099153, RMSE 216.126,
and MAE 25.726. The campaign-15 outlier has actual MRR 4,326.154 while the
model predicts 367.949. The report therefore depends on the published data-
quality correction and does not claim robustness to corrupted MRR labels.

The SHA-256 of the source label file used for validation is
`5b22c120de02adb950b9162a4a40d2cb14ad4458751b3f9747da708ad1c78178`.

## Validation contract

PHM16 does not expose a literal semiconductor lot identifier. Each physical
wafer is assigned to the chronological campaign containing its earliest source
file; one campaign spans ten consecutive source files. Both stages from one
physical wafer therefore remain in the same campaign.

LightGBM was chosen strictly from five-fold `GroupKFold` predictions on the
first 15 campaigns. Only after model selection was complete was it fitted to
all development campaigns and evaluated once on the last four campaigns.

| Final campaign | Samples | R² | RMSE | MAE |
|---:|---:|---:|---:|---:|
| 15 | 134 | 0.959363 | 5.388421 | 4.353215 |
| 16 | 102 | 0.963643 | 6.704537 | 4.892275 |
| 17 | 86 | 0.958400 | 4.378839 | 3.359935 |
| 18 | 27 | 0.584253 | 4.759786 | 4.143494 |
| **Aggregate 15–18** | **349** | **0.961533** | **5.546461** | **4.249775** |

Campaign 18 has a narrow target range and does not individually exceed R² 0.8;
the reportable gate is the predeclared aggregate latest-campaign holdout. This
limitation must remain visible in research reporting.

## Integrated artifacts

- Benchmark: `waferpulse/experiments/phm2016_cmp_benchmark.py`
- Inference contract: `waferpulse/tools/cmp_virtual_metrology.py`
- Dashboard: `waferpulse/dashboard/cmp_virtual_metrology_page.py`
- Bundled verified model:
  `waferpulse/artifacts/phm2016_cmp_verified_model.joblib`
- Bundled model SHA-256:
  `b0956f28bd2cb6a2ec6c4a1d520f35817713fcf290c2aae894d0688f512b7331`

Rebuild the ignored evidence and model artifact with:

```powershell
.\.venv\Scripts\python.exe -m waferpulse.experiments.phm2016_cmp_benchmark
```

## References

- PHM Society, *2016 PHM Data Challenge CMP specification*:
  https://phmsociety.org/wp-content/uploads/2016/05/PHM16DataChallengeCFP.pdf
- Zenodo transport mirror: https://doi.org/10.5281/zenodo.19803296
- Chen et al., *Predicting the Wafer Material Removal Rate for Semiconductor
  Chemical Mechanical Polishing Using a Fusion Network*:
  https://doi.org/10.3390/app122211478
