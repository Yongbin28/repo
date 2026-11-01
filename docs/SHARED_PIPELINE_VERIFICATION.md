# Shared prediction pipeline verification — 26 October 2026

Bosch and EquipmentData now use the same preparation-to-modelling coordinator and planning stage. Fresh runs against the original tracked implementation show that the coordinator preserves the tested results when model settings are held constant. Bosch screening changes when it adopts the selected conservative profile instead of its previous hard-coded balanced profile.

## Measured results

| Evaluation | Before | After | Finding |
|---|---:|---:|---|
| Equipment regression R² | 0.089236 | 0.089236 | Identical |
| Equipment bad-wafer recall | 80.59% | 80.59% | Identical |
| Bosch wafer-average silicon-etch R² | 0.787404 | 0.787404 | Identical |
| Bosch spatial silicon-etch R² | 0.990043 | 0.990043 | Identical |
| Bosch screening accuracy | 88.64% | 89.77% | Profile changed |
| Bosch high-etch recall | 76.19% | 71.43% | Profile changed |
| Bosch screening regression R² | 0.786768 | 0.787404 | Profile changed |

Equipment regression and classification both select ExtraTrees in these fresh runs. Bosch selects LightGBM for wafer-average silicon etch and its response-based screening, and Random Forest for spatial silicon etch.

## Prediction-level checks

- Equipment: all 971 held-out wafer records match exactly, including regression predictions, bad-wafer probabilities, fold assignments and predicted classes. All candidate scores, six champion fold-metric records and 120 feature-relevance records match. Maximum numeric difference: zero.
- Bosch regression: all 38 evaluation rows and all 196,944 prediction records match exactly across targets, model candidates and feature sets. Maximum numeric difference: zero. These are prediction records across multiple evaluations, not distinct wafers.
- Bosch screening with the shared conservative profile: predictions differ for the five model families whose budgets depend on the profile. For the LightGBM champion, three of 88 wafer categories change. False positives decrease from five to three, while false negatives increase from five to six. Higher overall accuracy therefore accompanies lower high-etch recall; this is not an improvement on every metric.
- Bosch screening control: when the new screening function uses the original balanced profile, all 880 prediction records across the ten candidate names match exactly. One name is the backward-compatible Ridge alias. Maximum numeric difference: zero. This control identifies the profile change as the cause of the screening differences.

## Data and settings

Equipment uses all 971 matched wafers from 40 lots and both equipment stages. The comparison uses the current dashboard defaults: three outer stratified lot-grouped folds, 60 selected features, 60 trees/boosting iterations, seed 42 and balanced search. All nine current regression algorithms and both current classifier candidates are evaluated, including the existing inner grouped tuning and diagnostic calculations.

Bosch uses 88 matched wafers from ten lots and all 7,832 spatial measurements. Both versions use five manufacturing-lot-held-out folds, seed 42 and the conservative regression profile recorded in the existing Bosch summary. The research screening threshold is 44.0. Before the refactor, screening always used balanced settings; afterward, it uses the same conservative profile as regression. A further new-code run with balanced screening provides the control comparison.

Groq is disabled or its selected profile is fixed during this audit so that a new language-model choice cannot change the experiment. Plot generation is skipped; model fitting, preprocessing, validation, prediction, scoring and numerical/report exports run normally. Automated tests separately check planner success, rate-limit fallback, dataset handover, preparation failure and the dashboard entry point.

The original tracked source is taken from commit `c193146d3922bd85fb97566647205a9b9b513e3c`. Exact source snapshots and hashes are saved with the comparison outputs. Source-specific model and preparation implementations are retained. The observations establish equivalence for the tested settings and data.

## Existing dashboard evidence

Existing dashboard output folders are retained. Comparison outputs are saved separately under `output/shared_pipeline_verification`.

The historical equipment dashboard artifact reports the older QualityHurdleForestBlend model with R² 0.1597, five folds and 120 selected features. The current trainer already compared nine standard regressors before this refactor. That historical score is therefore not the baseline for the fresh three-fold/60-feature comparison above. Differences from historical artifacts should not be attributed to the coordinator alone.

The historical Bosch regression and LightGBM screening scores are reproduced by the original-code runs. A future dashboard rerun with the conservative profile will produce the new screening results shown above; the refactor does not silently replace previously saved evidence.

## Reproduce

Run these commands from the repository root. The explicit commit keeps the original baseline fixed after further changes are committed. Set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` to `1` to match the audit's numerical execution settings.

```powershell
python -X utf8 -m waferpulse.experiments.shared_pipeline_verification --dataset equipment --baseline-ref c193146d3922bd85fb97566647205a9b9b513e3c
python -X utf8 -m waferpulse.experiments.shared_pipeline_verification --dataset bosch --baseline-ref c193146d3922bd85fb97566647205a9b9b513e3c
```

Machine-readable comparisons are in:

- [Equipment comparison](../output/shared_pipeline_verification/equipment/comparison.json)
- [Bosch comparison](../output/shared_pipeline_verification/bosch/comparison.json)

The updated report section and common flowchart are in [Agent-coordinated model training and prediction](AGENT_COORDINATED_PREDICTION.md).
