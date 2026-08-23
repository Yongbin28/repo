# T&P Wafer Quality Gate

An end-to-end machine learning pipeline for semiconductor wafer yield prediction. The system ingests raw STDF (Standard Test Data Format) binary files from Trim & Probe testing, decrypts them into structured CSV data, extracts statistical and process-control features, trains ensemble regression models, and serves real-time yield predictions through an interactive Streamlit dashboard.

---

## Table of Contents

- [System Overview](#system-overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Dataset Conventions](#dataset-conventions)
- [Pipeline Stages](#pipeline-stages)
  - [Stage 1 — STDF Decryption](#stage-1--stdf-decryption)
  - [Stage 2 — Wafer Data Combiner](#stage-2--wafer-data-combiner)
  - [Stage 3 — Feature Extraction](#stage-3--feature-extraction)
  - [Stage 4 — Model Training](#stage-4--model-training)
  - [Stage 5 — Pipeline Summary](#stage-5--pipeline-summary)
- [Machine Learning Features](#machine-learning-features)
  - [Base Statistical Features](#base-statistical-features)
  - [Limit Proximity (PPM) Features](#limit-proximity-ppm-features)
  - [Historical SPC & PSI Features](#historical-spc--psi-features)
- [Single Lot Yield Predictor](#single-lot-yield-predictor)
- [Distribution Shift Analysis](#distribution-shift-analysis)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Desktop Application](#desktop-application)
- [Installation](#installation)
- [Usage](#usage)
  - [Development Mode](#development-mode)
  - [Command-Line Interface](#command-line-interface)
  - [Building the Desktop App](#building-the-desktop-app)
- [Configuration](#configuration)
- [Module Reference](#module-reference)
- [Trained Model Algorithms](#trained-model-algorithms)
- [Requirements](#requirements)

---

## System Overview

The T&P Wafer Quality Gate automates the full lifecycle of semiconductor yield analysis:

1. **Ingest** raw binary test data (STDF V4) from automated test equipment (ATE).
2. **Decrypt** binary records into human-readable CSV with per-device parametric results, test limits, and die coordinates.
3. **Combine** multi-source wafer runs into unified, de-duplicated datasets per physical wafer (handling re-tests, multi-site insertions, and overlapping data).
4. **Extract** 100+ manufacturing-oriented statistical features per lot, including quantile distributions, outlier rates, limit proximity metrics, and historical SPC/PSI stability signals.
5. **Train** an ensemble of 11 regression models (linear, tree-based, and gradient-boosted), automatically selecting and persisting the top 3 by R² score.
6. **Predict** yield on new incoming lots in real time, with model-confidence scoring and distribution shift alerts.
7. **Visualize** results through an interactive dashboard with per-test distribution overlays, SPC alerts, PSI scoring, Cpk analysis, and multi-lot historical comparison.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          Streamlit Dashboard                              │
│                         (streamlit_app.py)                                │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────────────┐   │
│  │  Pipeline Mode   │  │  Yield Predictor │  │  Distribution Shift   │   │
│  │  (Batch Process) │  │  (Single Lot)    │  │  Analysis (SPC/PSI)   │   │
│  └────────┬─────────┘  └────────┬─────────┘  └───────────┬───────────┘   │
└───────────┼──────────────────────┼────────────────────────┼──────────────┘
            │                      │                        │
   ┌────────▼────────┐    ┌───────▼────────┐      ┌────────▼──────┐
   │  stdf_decryptor │    │  ml_yield_     │      │  Historical   │
   │  (.stdf → .csv) │    │  prediction    │      │  Inventory    │
   └────────┬────────┘    └───────┬────────┘      │  Scanner      │
            │                     │               └───────────────┘
   ┌────────▼────────┐    ┌───────▼────────┐
   │  wafer_data_    │    │  ml_compute_   │ ◄── Feature Extraction
   │  combiner       │    │  statistic     │     + SPC/PSI Features
   └────────┬────────┘    └───────┬────────┘
            │                     │
            └──────────┬──────────┘
                       │
              ┌────────▼────────┐
              │  ml_train_model │ ◄── 11 Algorithms
              │  (.joblib)      │     Top 3 Saved
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │  app_summary    │ ◄── Pipeline Status Summary
              │  (.xlsx)        │     (Excel Report)
              └─────────────────┘
```

### Supporting Modules

| Module | Purpose |
|--------|---------|
| `utils.py` | Logging, path resolution, tester family mapping, file explosion (zip/gz) |
| `partname_mapping.py` | PartName normalization and canonical grouping |
| `desktop_app.py` | PyInstaller-compatible wrapper with Tkinter control window |
| `run_app.py` | Minimal Streamlit launcher for development |

---

## Project Structure

```
code-Micron/
├── streamlit_app.py         # Main application — dashboard, pipeline orchestration, UI
├── stdf_decryptor.py        # Universal STDF V4 binary parser → CSV converter
├── wafer_data_combiner.py   # Multi-source wafer CSV merger and de-duplicator
├── ml_compute_statistic.py  # Feature extraction engine (stats, PPM, SPC, PSI)
├── ml_train_model.py        # Multi-algorithm model trainer and evaluator
├── ml_yield_prediction.py   # Real-time yield prediction from raw DLOG files
├── app_summary.py           # Post-pipeline summary and Excel report generator
├── utils.py                 # Shared utilities (logging, tester map, path helpers)
├── partname_mapping.py      # PartName normalization for product variant grouping
├── desktop_app.py           # Desktop launcher (PyInstaller + Tkinter)
├── run_app.py               # Development Streamlit launcher
├── build_full_system.bat    # One-click build script for Windows executable
├── desktop_app.spec         # PyInstaller build specification
├── hook-streamlit_app.py    # PyInstaller hidden import hook
├── requirements.txt         # Python dependencies
├── pyproject.toml           # Project metadata and build configuration
├── setup.py                 # Legacy setuptools configuration
├── TesterFamilyMap.xlsx     # Tester ↔ Generic product mapping reference
├── icon.jpg                 # Application icon
├── .gitignore
│
├── dataset/                 # Primary data storage root
│   └── J750/             # Tester family directory
│       └── LT3942_8VL3942XV/           # Generic_PartName directory
│           ├── T&P_Decrypted/          # Decrypted CSV files
│           │   ├── HB025401_[13]_97.9%/    # Lot folder (LotID_[Wafers]_Yield%)
│           │   │   ├── *.csv               # Per-wafer test data
│           │   │   └── limit/              # Test limits
│           │   │       └── *_limits.csv
│           │   └── ...
│           └── Model/                  # Trained model artifacts
│               ├── merged_features_LT3942.csv      # Extracted feature matrix
│               ├── model_CatBoost_LT3942.joblib     # Top model #1
│               ├── model_XGBoost_LT3942.joblib      # Top model #2
│               ├── model_HistGBR_LT3942.joblib      # Top model #3
│               └── model_details_LT3942.xlsx        # Validation report
│
├── J750/                 # Yield mapping data (Result_*.xlsx)
│   └── LT3942/
│       └── Result_LT3942.xlsx
│
├── pipeline/                # Runtime pipeline logs
├── cfc/                     # CFC (Configuration) staging area
└── catboost_info/           # CatBoost internal training logs
```

---

## Dataset Conventions

### Folder Naming

Lot-level folders within `T&P_Decrypted/` follow a strict naming convention:

```
{LotID}_{[WaferList]}_{Yield%}
```

Examples:
- `HB025401_[13]_97.9%` — Lot HB025401, Wafer 13, 97.9% yield
- `W2512024_[01-25]_92.3%` — Lot W2512024, Wafers 01 through 25, 92.3% yield
- `P070238_[03,05,07]_88.1%` — Lot P070238, Wafers 3, 5, and 7, 88.1% yield

### CSV Header Format

Decrypted CSVs use a 6-row header structure:

| Row | Content | Example |
|-----|---------|---------|
| 1 | DLOG source filename | `Created from DLOG: LT3942WS_00_...` |
| 2 | Test program name | `Test Program: FFU033` |
| 3 | File creation timestamp | `File Creation Date: 05/05/2025 15:38:45` |
| 4 | T-number codes (T1.0, T1.1, ...) | `,,,,,,,T1.0,T1.1,T2.0,...` |
| 5 | Measurement units | `,,,,,,,V,mA,mV,...` |
| 6 | Column headers | `Device #,Bin,Site,X,Y,Fails,Alarms,VCC_3P3,...` |

### Limits Files

Each lot folder contains a `limit/` subdirectory with a `*_limits.csv`:

| Column | Description |
|--------|-------------|
| `Test Number` | T-code (e.g., T1.0) |
| `Test Description` | Human-readable test name |
| `Units` | Measurement unit |
| `STDF Min` | Lower specification limit (LSL) |
| `STDF Max` | Upper specification limit (USL) |

---

## Pipeline Stages

### Stage 1 — STDF Decryption

**Module:** `stdf_decryptor.py`

Converts raw STDF V4 binary test data into structured CSV.

**Key capabilities:**
- **High-performance binary parser** using direct `struct` unpacking with pre-compiled format objects
- **Automatic endianness detection** from FAR record
- **Record types handled:** PIR (Part Information), PTR (Parametric Test Results), PRR (Part Results), MIR (Master Information)
- **Support for:** `.stdf`, `.std`, `.std_1`, `.stdf.gz`, `.std.gz`, `.std_1.gz`, and nested ZIP archives
- **Die coordinate resolution** from XY fields with fallback to X_DIE/Y_DIE test parameters
- **Automatic limit extraction** generating a companion `_limits.csv` with min/max specification values
- **Parallel processing** via `ProcessPoolExecutor` in auto-discovery mode

**Outputs:**
- Main CSV with per-device test results (Device #, Bin, Site, X, Y, Fails, Alarms, test columns...)
- Limits CSV in `limit/` subdirectory

### Stage 2 — Wafer Data Combiner

**Module:** `wafer_data_combiner.py`

Merges multiple CSV files for the same physical wafer into a single de-duplicated dataset.

**Merge strategy:**
1. Files are sorted by creation date (newest first).
2. Rows are keyed by (X, Y) die coordinates.
3. **Pass priority:** A passing device always overwrites a failing device at the same coordinate.
4. **Recency priority:** Among same-status devices, the newer record wins.
5. **Partial fill:** Missing test values from older records backfill gaps in the winning record.

**Additional operations:**
- T-number de-aliasing across files (canonical mapping of test columns)
- Limit file cleanup (retains only the newest limits file)
- Original source file removal after successful combination

### Stage 3 — Feature Extraction

**Module:** `ml_compute_statistic.py`

Transforms raw per-device test data into a single feature row per lot for ML training.

**Processing flow:**
1. **Dynamic header parsing** — Locates `Device #` / `Bin` row automatically regardless of header position
2. **Site filtration** — Retains only Bin 1 (passing) devices; excludes rows with Fails or Alarms
3. **Numeric conversion** — Coerces all columns to numeric, filtering out metadata columns
4. **Per-column statistics** — Computes 15+ features per test parameter (see [Machine Learning Features](#machine-learning-features))
5. **Limit proximity features** — PPM-style nearness metrics to LSL/USL boundaries
6. **Lot-level aggregation** — Weighted average across all wafers in a lot (weighted by pass count)
7. **Historical SPC/PSI features** — Leave-one-out stability signals computed across the entire dataset (see [Historical SPC & PSI Features](#historical-spc--psi-features))
8. **Yield mapping** — Target variable (y) parsed from folder name or loaded from external `Result_*.xlsx`

**Output:** `merged_features_{Generic}.csv` — One row per lot, hundreds of feature columns, target column `y`.

### Stage 4 — Model Training

**Module:** `ml_train_model.py`

Trains and evaluates 11 regression algorithms, saves the top 3 models.

**Training protocol:**
1. **Group-aware splitting:** `GroupShuffleSplit` ensures no wafers from the same lot appear in both train and validation sets (80/20 split).
2. **Pipeline architecture:** Each model is wrapped in a `sklearn.Pipeline` with appropriate preprocessing (imputation + optional scaling).
3. **Evaluation metrics:** MAE, RMSE, R² on the held-out validation set.
4. **Model persistence:** Top 3 models saved as `.joblib` files. Old models for the same generic are automatically cleaned up.
5. **Validation report:** Excel workbook with per-sample predictions and model comparison summary.

See [Trained Model Algorithms](#trained-model-algorithms) for the full list of models.

### Stage 5 — Pipeline Summary

**Module:** `app_summary.py`

Generates a consolidated Excel report (`Pipeline_Status_Summary.xlsx`) after all generics are processed.

**Report sheets:**
| Sheet | Content |
|-------|---------|
| `Status` | Top-3 models per generic with R², RMSE, MAE, tester family, and pipeline warnings |
| `Summary` | Aggregate statistics (total devices, models with R² > 0.5, success rate) |
| `All_Validation` | Consolidated validation predictions with actual vs. predicted yield, color-coded by model rank |

---

## Machine Learning Features

### Base Statistical Features

For each numeric test column that passes exclusion filters, the following features are extracted (prefixed by normalized test name, e.g., `t1_0__mean`):

| Feature | Description |
|---------|-------------|
| `__mean` | Arithmetic mean |
| `__std` | Standard deviation (ddof=1) |
| `__median` | 50th percentile |
| `__iqr` | Interquartile range (Q75 − Q25) |
| `__min` | Minimum value |
| `__max` | Maximum value |
| `__missing_rate` | Fraction of NaN values |
| `__outlier_rate` | Fraction of values outside median ± 3.0 × IQR |
| `__p01` ... `__p99` | Quantile values at 1%, 5%, 25%, 50%, 75%, 95%, 99% |

### Limit Proximity (PPM) Features

When LSL/USL limits are available, four additional features measure how close the distribution sits to specification boundaries:

| Feature | Description |
|---------|-------------|
| `__ppm_near_LSL_pct` | Fraction of values within 5% of the LSL-to-USL range from LSL |
| `__ppm_near_USL_pct` | Fraction of values within 5% of the LSL-to-USL range from USL |
| `__ppm_near_LSL_sigma` | Fraction of values within 0.5σ of LSL |
| `__ppm_near_USL_sigma` | Fraction of values within 0.5σ of USL |

### Historical SPC & PSI Features

When **5 or more lots** are available in the training dataset, the system computes leave-one-out process stability features for each lot. For each test parameter:

**SPC (Statistical Process Control) Features:**

| Feature | Description |
|---------|-------------|
| `__spc__median_shift` | \|current_median − baseline_median\| / baseline_IQR — normalized median drift |
| `__spc__sigma_dist` | \|current_mean − baseline_mean\| / baseline_std — distance in sigma units |
| `__spc__is_shift` | Binary flag: 1.0 if median_shift > 1.5, else 0.0 |

**PSI (Population Stability Index) Features:**

| Feature | Description |
|---------|-------------|
| `__psi__score` | Raw PSI value (10-bin histogram comparison) |
| `__psi__is_minor` | Binary flag: 1.0 if PSI ≥ 0.10 |
| `__psi__is_major` | Binary flag: 1.0 if PSI > 0.25 |

**Baseline construction:** For each lot, the baseline is the aggregate of **all other lots** in the same dataset scan (leave-one-out). This prevents data leakage while giving the model process stability context. Test parameters that appear in fewer than half the lots are excluded from SPC/PSI computation.

---

## Single Lot Yield Predictor

**Module:** `ml_yield_prediction.py`

Provides real-time yield prediction for a new incoming lot.

**Prediction flow:**
1. **Upload** a raw DLOG file (`.stdf`, `.std`, `.csv`, `.gz`, or `.zip`)
2. **Decrypt** binary → CSV (skipped if already CSV)
3. **Extract features** using the same `ml_compute_statistic` pipeline
4. **Align** feature columns to the training set's schema (`merged_features_*.csv`)
5. **Predict** using all available top-3 models for the generic
6. **Return** individual model predictions, ensemble average, raw data for shift analysis

**Model discovery order:**
1. `dataset/{Family}/{Generic}_{PartName}/Model/model_*_{Generic}.joblib`
2. `dataset/{Family}/{Generic}/Model/model_*_{Generic}.joblib`
3. Root-level fallback: `model_*_{Generic}.joblib`

---

## Distribution Shift Analysis

The dashboard includes a distribution shift analysis module that compares the current incoming lot against selected historical lots.

### SPC Analysis
- Compares current lot's median for each test against the historical IQR
- Flags a **shift** when `|current_median − historical_median| > 1.5 × IQR`

### PSI Analysis
- Computes the Population Stability Index using 10-bin histogram comparison
- **Normal**: PSI < 0.10
- **Minor Shift**: 0.10 ≤ PSI < 0.25
- **Major Shift**: PSI ≥ 0.25

### Visualization
- Multi-lot overlay histograms (up to 10 historical lots) with per-lot color coding
- LSL/USL specification limit lines (red dashed)
- 99% data distribution view (default) and full-range view with limits
- Summary table with sortable PSI status, SPC alerts, and Cpk values for both current and historical data

---

## Streamlit Dashboard

The web interface operates in two modes:

### Pipeline Data Prep Mode
- **Tester Family Filter** — Filter generics by tester family (J750, Eagle, Catalyst, J750, HP94K)
- **Batch Processing** — Select multiple generics for sequential pipeline execution
- **Per-Generic Tracker** — Expandable progress cards with real-time logs, progress bars, and ETA
- **Configurable Steps** — Toggle Feature Extraction, Model Training, OneDrive cleanup independently
- **Live Dashboard** — Batch progress, elapsed time, and estimated completion

### Single Yield Predictor Mode
- **DLOG Upload** — Drag-and-drop support for STDF/CSV/ZIP files
- **Yield Metrics** — Predicted yield, historical average, yield grade (Z/A/H/I/C/F), model confidence
- **Per-Model Comparison** — Bar chart showing individual predictions from each trained model
- **Distribution Shift Analysis** — Interactive test-by-test comparison with historical lots
- **Historical Lot Selector** — Filter by fiscal year, select up to 10 reference lots

---

## Desktop Application

The system can be packaged as a standalone Windows executable using PyInstaller.

**Module:** `desktop_app.py`

**Architecture:**
- A Tkinter control window launches a Streamlit server as a subprocess
- The default browser opens automatically to `http://localhost:8501`
- The control window provides "Open in Browser" and "Stop Server" buttons
- Bundled modules can be invoked as pseudo-scripts via the frozen executable

**Build command:**
```batch
build_full_system.bat
```

This script:
1. Kills stale processes
2. Copies Playwright browser binaries
3. Creates a virtual environment
4. Installs dependencies
5. Runs PyInstaller with `desktop_app.spec`

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd "code -Micron"

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### Optional: Install Playwright browsers (for any web scraping features)
```bash
playwright install chromium
```

---

## Usage

### Development Mode

```bash
# Run the Streamlit dashboard
python -m streamlit run streamlit_app.py

# Or use the launcher script
python run_app.py
```

The dashboard opens at `http://localhost:8501`.

### Command-Line Interface

Every pipeline module supports standalone CLI execution:

```bash
# Decrypt a single STDF file
python stdf_decryptor.py input.stdf output.csv

# Auto-discover and decrypt all STDF files in dataset/
python stdf_decryptor.py

# Combine wafer data in a folder
python wafer_data_combiner.py /path/to/T&P_Decrypted

# Extract features from decrypted data
python ml_compute_statistic.py --root /path/to/T&P_Decrypted --out features.csv

# Extract features with external yield mapping and auto-train
python ml_compute_statistic.py --root /path/to/T&P_Decrypted --yield-data Result_LT3942.xlsx --train

# Train models on extracted features
python ml_train_model.py --data merged_features_LT3942.csv

# Predict yield from a DLOG file
python ml_yield_prediction.py --generic LT3942 --dlog input_file.stdf

# Normalize a part name
python partname_mapping.py -g LT3942 -p 8VL3942XV

# Generate pipeline summary report
python app_summary.py
```

### Building the Desktop App

```batch
# Run the full build (Windows only)
build_full_system.bat

# Output: dist/T&P_Wafer_Quality_Gate/
```

---

## Configuration

### Key Constants

| Location | Constant | Default | Description |
|----------|----------|---------|-------------|
| `ml_compute_statistic.py` | `OUTLIER_K` | 3.0 | IQR multiplier for outlier detection |
| `ml_compute_statistic.py` | `NEAR_MARGIN_RATIO` | 0.05 | PPM margin as fraction of LSL-to-USL range |
| `ml_compute_statistic.py` | `NEAR_SIGMA_K` | 0.5 | Sigma multiplier for PPM limit proximity |
| `ml_compute_statistic.py` | `MIN_BASELINE_LOTS` | 5 | Minimum lots required for SPC/PSI feature computation |
| `ml_train_model.py` | `SEED` | 42 | Random seed for reproducibility |
| `ml_train_model.py` | `TEST_SIZE` | 0.2 | Validation split fraction |
| `stdf_decryptor.py` | `CHUNK_SIZE` | 1 MB | Read buffer size |

### Exclusion Filters

Columns matching these patterns are excluded from feature extraction:

**Metadata columns:** `device_`, `bin`, `site`, `x`, `y`, `fails`, `alarms`, `device_bin_site`, `test_number`

**Test name patterns:** `x_die_location`, `y_die_location`, `timeline`, `uph`, `test_time_prior_`, `index_time`, `down_time_total_`

---

## Module Reference

| Module | Lines | Description |
|--------|-------|-------------|
| `streamlit_app.py` | ~1750 | Main application: UI layout, pipeline orchestration, distribution analysis, prediction display |
| `ml_compute_statistic.py` | ~570 | Feature extraction engine with SPC/PSI stability signals and lot-level aggregation |
| `stdf_decryptor.py` | ~430 | Universal STDF V4 parser with struct-based binary decoding |
| `wafer_data_combiner.py` | ~385 | Wafer-level CSV merger with pass-priority and date-aware conflict resolution |
| `app_summary.py` | ~445 | Post-pipeline Excel report generator with styled validation output |
| `utils.py` | ~310 | Shared utilities: logging, tester family resolution, zip/gz explosion |
| `ml_train_model.py` | ~220 | Multi-algorithm training loop with group-aware cross-validation |
| `ml_yield_prediction.py` | ~230 | End-to-end prediction pipeline (decrypt → extract → align → infer) |
| `desktop_app.py` | ~200 | PyInstaller-compatible desktop launcher with Tkinter control window |
| `partname_mapping.py` | ~140 | PartName normalization with regex rules and alias overrides |
| `run_app.py` | ~30 | Minimal development Streamlit launcher |

---

## Trained Model Algorithms

The training pipeline evaluates the following algorithms. All are wrapped in `sklearn.Pipeline` with appropriate preprocessing:

| Model | Preprocessing | Key Hyperparameters |
|-------|---------------|---------------------|
| **Ridge** | Impute (median) + Scale | α = 5.0 |
| **Lasso** | Impute (median) + Scale | α = 0.01 |
| **ElasticNet** | Impute (median) + Scale | α = 0.05, l1_ratio = 0.3 |
| **SVR** | Impute (median) + Scale | C = 10.0, ε = 0.2 |
| **KNN** | Impute (median) + Scale | k = min(7, n_train), distance-weighted |
| **RandomForest** | Impute (median) | 500 trees, all cores |
| **ExtraTrees** | Impute (median) | 500 trees, all cores |
| **HistGradientBoosting** | Impute (median) | 500 iterations |
| **XGBoost*** | Impute (median) | 1000 trees, lr = 0.05 |
| **CatBoost*** | Impute (median) | 1000 iterations, silent |
| **LightGBM*** | Impute (median) | 1000 trees, verbosity = -1 |

\* Optional — only included if the corresponding package is installed.

---

## Requirements

```
streamlit>=1.30.0       # Web dashboard framework
pandas>=2.0.0           # Data manipulation
numpy>=1.24.0           # Numerical computation
scikit-learn>=1.3.0     # ML algorithms and pipelines
joblib>=1.3.0           # Model serialization
xgboost>=2.0.0          # Gradient boosting (optional)
catboost>=1.2.0         # Gradient boosting (optional)
lightgbm>=4.1.0         # Gradient boosting (optional)
plotly>=5.18.0          # Interactive charting
openpyxl>=3.1.2         # Excel I/O
selenium>=4.1.0         # Browser automation (legacy)
playwright>=1.40.0      # Browser automation (legacy)
```

**Development extras:**
```
pytest>=7.0.0
black>=23.0
isort>=5.10
mypy>=1.0
```
