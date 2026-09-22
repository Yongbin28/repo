# PHM 2016 CMP Dataset Selection and Phase 1 Alignment

## Selection decision

The PHM Society 2016 Chemical-Mechanical Planarization (CMP) dataset is used as
a **separate external virtual-metrology validation dataset**. It is selected
because it contains the research structure needed to demonstrate continuous
upstream-to-downstream wafer-performance prediction:

> upstream time-series process measurements → downstream wafer metrology

It is not merged with EquipmentData and its score is never assigned to the
EquipmentData or electrical wafer-test model.

## What the dataset contains

The official PHM challenge describes a CMP tool that presses and rotates a
wafer against a polishing pad while applying slurry. Every trace row records a
time instance of process variables. The separate label file provides
`AVG_REMOVAL_RATE` for each `WAFER_ID × STAGE`, calculated from wafer-thickness
measurements before and after polishing.

WaferPulse's verified clean-label cohort contains:

| Item | Verified value |
|---|---:|
| Raw upstream trace rows | 672,744 |
| Raw labelled wafer-stage samples | 1,981 |
| Published extreme-label exclusions | 4 |
| Modelled wafer-stage samples | 1,977 |
| Physical wafers | 1,695 |
| Chronological source-file campaigns | 19 |
| Response-free engineered predictors | 348 |

The upstream signals cover process pressure, air-bag pressure, slurry flow,
wafer/table/head rotation, chamber state, and consumable usage. The downstream
target is continuous material-removal rate rather than a pass/fail label.

## Leakage-safe validation contract

1. `WAFER_ID`, absolute `TIMESTAMP`, and source-file identity are never model
   predictors.
2. `AVG_REMOVAL_RATE` is never used by feature engineering or inference.
3. Four labels above 4,000 are removed using a fixed published data-quality rule
   established before model comparison; normal MRR values are below 170.
4. Chronological file groups 0–14 form development data.
5. Algorithm selection uses five-fold GroupKFold within development campaigns.
6. Latest campaigns 15–18 remain untouched until the selected model is tested.
7. Development and final partitions contain no overlapping physical wafer IDs.

LightGBM achieved:

| Evaluation | R² | RMSE | MAE |
|---|---:|---:|---:|
| Development campaign-grouped OOF | 0.9797 | 4.3324 | 3.4404 |
| Untouched latest campaigns | 0.9615 | 5.5465 | 4.2498 |

The untouched result honestly exceeds the project's R² 0.8 benchmark for this
specific CMP virtual-metrology population.

## Alignment with the Phase 1 report

| Phase 1 element | PHM implementation | Alignment |
|---|---|---|
| Upstream manufacturing inputs | CMP equipment time-series traces | Direct |
| Downstream continuous performance | Post-process average material-removal rate | Direct virtual-metrology analogue |
| Feature engineering | Fold-safe statistical, temporal, phase, chamber and physics-proxy features | Direct |
| Multiple algorithms | Ridge, PLS, SVR, Random Forest, Extra Trees, boosting, XGBoost, LightGBM and CatBoost | Direct |
| Honest manufacturing separation | Chronological source-file campaigns | Partial: official lot IDs are unavailable |
| R²/RMSE/MAE evaluation | Grouped development plus untouched final campaigns | Direct |
| Wafer-pulse/electrical test target | Not present | Not aligned |
| SPC, PSI and historical monitoring | Remains in the separated local/STDF lane | Architectural alignment only |

This supports the Phase 1 methodology claim that upstream equipment behaviour
can be used for downstream continuous virtual metrology. It does **not** support
a claim that the same R² applies to wafer-pulse current, probe-to-final-test
yield, or the original EquipmentData response.

## How it should be used in the report

Use the datasets as two complementary evidence lanes:

- **EquipmentData:** closer to the intended electrical/wafer-response domain;
  strict unseen-lot R² remains 0.2655, showing the real information limitation.
- **PHM 2016 CMP:** strong external proof that the proposed leakage-safe
  upstream-to-downstream pipeline works when the process sensors contain a
  predictable continuous metrology signal; untouched-campaign R² is 0.9615.

SPC, PSI, Cpk and historical comparison may continue to use local or generated
STDF-style data, but those functions must remain labelled as demonstrations.
The three lanes must not be pooled into one reported accuracy value.

## Reproducibility and sources

- Run the benchmark:
  `.venv\Scripts\python.exe -m waferpulse.experiments.phm2016_cmp_benchmark`
- Run the verification tests:
  `.venv\Scripts\python.exe -m pytest -q tests/test_cmp_virtual_metrology.py`
- Open the application and select **CMP Upstream → Downstream Metrology**.

Primary sources:

- [PHM Society 2016 challenge description](https://phmsociety.org/conference/annual-conference-of-the-phm-society/annual-conference-of-the-prognostics-and-health-management-society-2016/phm-data-challenge-4/)
- [Official challenge specification](https://phmsociety.org/wp-content/uploads/2016/05/PHM16DataChallengeCFP.pdf)
- [Published PHM16 outlier and fusion-model study](https://doi.org/10.3390/app122211478)
