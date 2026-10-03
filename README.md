# WaferPulse

## Explainable AI for Proactive Semiconductor Quality Risk Detection

WaferPulse is a research and demonstration platform for detecting semiconductor
quality risks before they propagate into downstream manufacturing losses. It
combines equipment-sensor modelling, virtual metrology, STDF processing,
statistical process monitoring, spatial wafer analysis, and explainable machine
learning in one Streamlit application.

The project deliberately separates validated public-data results from local or
synthetic demonstrations. A model score is reported only for the dataset and
target on which it was evaluated.

> **Scope:** WaferPulse supports process and quality-risk analysis. It does not
> claim to predict field lifetime or prove that probe-mark damage directly
> creates interfacial voids.

## Key capabilities

- Validate and transform time-series equipment data into wafer-level features.
- Predict a continuous wafer-response value and identify elevated-risk wafers.
- Predict downstream Bosch plasma-etch measurements from upstream process traces.
- Parse STDF V4 files and combine repeated wafer-test records.
- Monitor process drift with SPC, Cpk, PSI, and specification-proximity signals.
- Compare incoming wafers against historical and virtual-golden baselines.
- Detect Good-Die–Bad-Neighbour (GDBN) and other spatial risk patterns.
- Explain predictions with feature importance, SHAP where supported, and a
  permutation-importance fallback.
- Save reproducible metrics, prediction ledgers, feature importance, provenance,
  and trained-model artifacts.

## Application workspaces

| Workspace | Data | Supported use | Evidence boundary |
|---|---|---|---|
| **Model Prediction & Validation — EquipmentData** *(optional)* | Historical `equipment1.csv`, `equipment2.csv`, and `response.csv` benchmark | Two equipment stages → continuous wafer response and elevated-risk classification; includes local source-table analysis | Not the current WaferPulse benchmark and not used for the Bosch results below |
| **Model Prediction & Validation — Zenodo BOSCH** | [Zenodo record 17122442](https://zenodo.org/records/17122442): Bosch 5 Hz process traces and 89-point etch maps | Process traces → wafer-average silicon etch and coordinate-assisted spatial virtual metrology; includes local source-table analysis | Validated with manufacturing-lot-held-out folds; the high-etch boundary is a research rule, not a factory specification |
| **Model Prediction & Validation — Local upload** | User-selected CSV, TSV, Excel, or Parquet table | Read-only exploration plus optional in-memory nine-model numeric regression AutoML with a user-selected target and optional group boundary | Local evidence only; the upload is never merged into either public benchmark or saved by the dashboard |
| **Model Preparation Pipeline** | Local STDF/CSV data | Decryption, wafer merging, feature extraction, and local model preparation | Functional demonstration; results are not public-data accuracy evidence |
| **Wafer Analytics & Prediction** | Incoming local DLOG/STDF/CSV data | Prediction, SPC/PSI, golden comparison, SHAP, and spatial analysis | Functional demonstration dependent on locally trained artifacts |

See [DATA_LANES.md](DATA_LANES.md) for the enforced claim boundary and
[ARCHITECTURE.md](ARCHITECTURE.md) for the package design.

## Current validated benchmark: Bosch plasma etch

The Bosch benchmark joins 96 upstream 5 Hz equipment traces to 88 measured
wafers across 10 manufacturing lots. All reported scores use five-fold
manufacturing-lot-held-out validation.

| Task | Inputs | Best result |
|---|---|---|
| Wafer-average silicon etch | Upstream process summaries only | Gradient Boosting: R² **0.8006**, RMSE **0.1784**, MAE **0.1508** |
| Research high-etch screening | Predicted average etch, threshold 44.0 | F1 **0.7442**, recall **0.7619** with LightGBM |
| 89-point silicon-etch map | Process summaries + known X/Y | LightGBM: R² **0.9917** |

The public dataset does not provide an official production pass/fail limit, so
44.0 is a documented research threshold only. The spatial score is also
coordinate-assisted: a coordinate-only Extra Trees baseline already reaches R²
**0.9823**. It must therefore be described as spatial virtual metrology, not
pure equipment-sensor prediction.

## How WaferPulse works

```text
Equipment traces / STDF / CSV
              │
              ▼
      Validation and provenance
              │
              ▼
   Wafer-level feature engineering
              │
       ┌──────┴────────┐
       ▼               ▼
 Model prediction   SPC / PSI / Cpk
       │               │
       └──────┬────────┘
              ▼
 Explainability and spatial analysis
              │
              ▼
 Quality-risk evidence and dashboard
```

For the local STDF workflow, the detailed processing sequence is:

1. Decrypt STDF records into structured CSV data.
2. Combine repeated tests for the same physical wafer by die coordinates.
3. Extract distribution, limit-proximity, SPC, and PSI features.
4. Train and compare regression models using group-aware validation.
5. Persist the leading models and generate an Excel pipeline summary.
6. Score incoming lots and display explanations and process-shift evidence.

## Project structure

```text
.
├── streamlit_app.py              # Main Streamlit application
├── run_app.py                    # Development launcher
├── desktop_app.py                # Windows desktop wrapper
├── stdf_decryptor.py             # STDF V4 parser
├── wafer_data_combiner.py        # Wafer-level merge and deduplication
├── ml_compute_statistic.py       # Statistical feature engineering
├── ml_train_model.py             # Local model training and explainers
├── ml_yield_prediction.py        # Incoming-lot prediction workflow
├── spatial_analysis.py           # Wafer-map and GDBN analysis
├── reliability_grading.py        # Local demonstration scoring only
├── waferpulse/
│   ├── agents/                   # Analytical responsibilities
│   ├── crews/                    # Workflow orchestration
│   ├── core/                     # Contracts and data-lane policy
│   ├── dashboard/                # Modular Streamlit pages
│   ├── experiments/              # Reproducible benchmark studies
│   ├── tools/                    # Deterministic data/ML functions
│   └── artifacts/                # Optional verified research artifacts
├── tests/                        # Automated tests
├── data/                         # Optional external benchmark datasets
├── output/                       # Generated evidence and model output
├── DATA_LANES.md                 # Claim and evidence boundaries
└── ARCHITECTURE.md               # Agent/crew architecture
```

Raw manufacturing data, generated outputs, trained local models, presentations,
and reports are intentionally excluded by `.gitignore`.

## Installation

### Prerequisites

- Python 3.8 or later; Python 3.10 or 3.11 is recommended.
- Windows is recommended for the desktop build and local STDF workflow.
- Sufficient memory for tree ensembles and large trace datasets.

### Set up a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For optional research experiments such as the CDAE reproduction:

```powershell
pip install -r requirements-experiments.txt
```

For development and testing tools:

```powershell
pip install -r requirements-dev.txt
```

## Running the application

Start the dashboard with either command:

```powershell
python run_app.py
```

```powershell
python -m streamlit run streamlit_app.py
```

Streamlit normally opens the application at `http://localhost:8501`.

Use the sidebar to select one of the three workspaces:

1. **Model Preparation Pipeline**
2. **Model Prediction & Validation** — then select EquipmentData, Zenodo BOSCH, or a local upload
3. **Wafer Analytics & Prediction**

## Reproducing the Bosch plasma-etch benchmark

Download the public source from [Zenodo record 17122442](https://zenodo.org/records/17122442)
and place the required Bosch files under `data/bosch_plasma_etch/`:

```text
data/bosch_plasma_etch/
├── Process_data.nc
├── Dictionary_process.nc
└── Si_Oxide_etch_89_points.csv
```

Install the experiment dependencies and run:

```powershell
pip install -r requirements-experiments.txt
python -m waferpulse.experiments.bosch_plasma_etch_benchmark --data-root data/bosch_plasma_etch
python -m waferpulse.experiments.bosch_response_then_classify_benchmark --data-root data/bosch_plasma_etch --threshold 44.0
```

Evidence is written to `output/bosch_plasma_etch/` and
`output/bosch_response_then_classify/`. The same workflows can be rebuilt from
the Zenodo BOSCH selection in **Model Prediction & Validation**.

## Local STDF pipeline

Common commands for the local demonstration workflow are:

```powershell
# Decrypt one STDF file
python stdf_decryptor.py input.stdf output.csv

# Auto-discover supported STDF files
python stdf_decryptor.py

# Combine repeated wafer CSV records
python wafer_data_combiner.py "path\to\T&P_Decrypted"

# Extract features
python ml_compute_statistic.py --root "path\to\T&P_Decrypted" --out features.csv

# Train local models
python ml_train_model.py --data features.csv

# Generate the pipeline summary
python app_summary.py
```

Supported STDF inputs include `.stdf`, `.std`, `.std_1`, gzip-compressed
variants, and nested ZIP archives.

### Wafer merge policy

Repeated records are keyed by die coordinates:

1. A passing result has priority over a failing result for the same die.
2. When status is equal, the newer record has priority.
3. Missing test values may be backfilled from the older record.

Review this policy before using the pipeline with a different factory or retest
convention.

## Explainability and monitoring

WaferPulse provides several complementary views:

- **Feature importance:** identifies influential equipment or test features.
- **SHAP analysis:** explains model output when a compatible explainer exists.
- **Permutation importance:** provides a fallback for unsupported estimators.
- **SPC:** detects normalized median and mean shifts against historical data.
- **PSI:** measures changes in parameter distributions.
- **Cpk and limit proximity:** show movement toward specification boundaries.
- **Golden comparison:** compares candidate distributions with reference wafers.
- **GDBN analysis:** highlights probe-good dies surrounded by bad neighbours.
- **Prediction ledger:** retains row-level out-of-fold evidence for auditability.

Explanations indicate associations learned by a model; they do not establish a
physical causal mechanism.

## Testing

Run the automated test suite from the project root:

```powershell
pytest -q
```

The tests cover data-lane enforcement, equipment-data contracts, crew
orchestration, Bosch feature contracts, and selected benchmark workflows.

## Windows desktop build

```batch
build_full_system.bat
```

The script creates a packaging environment and builds the executable with
PyInstaller. It also terminates stale application and Python processes before
building, so save other Python work first.

## Data governance and limitations

- Do not commit confidential manufacturing data, customer identifiers, or
  proprietary model artifacts.
- Dataset identifiers, timestamps, and target columns are excluded where they
  would cause leakage.
- Manufacturing lots remain grouped during reported equipment-data validation.
- Local and synthetic demonstrations must not be reported as real-data model
  accuracy.
- The Bosch wafer-average score applies only to silicon-etch prediction from
  upstream process summaries under lot-held-out validation.
- The Bosch high-etch category uses a research threshold, and the spatial-map
  score uses known X/Y coordinates.
- Probe-mark damage, interfacial voids, current crowding, Joule heating, and
  electromigration are related engineering mechanisms, but the software does
  not prove a single causal chain among them.

## Documentation

- [Data lanes and evidence boundary](DATA_LANES.md)
- [Software architecture](ARCHITECTURE.md)

## Project status

WaferPulse is an academic/research prototype. Any production deployment requires
site-specific data validation, security review, calibration, change control,
and approval from qualified semiconductor process and quality engineers.
