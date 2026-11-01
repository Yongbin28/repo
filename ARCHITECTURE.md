# WaferPulse Code Architecture

Bosch and EquipmentData use one shared prediction coordinator: agents perform one responsibility, a crew controls execution order, tools contain deterministic implementation, and the dashboard renders state and starts the selected crew.

```text
waferpulse/
├── agents/
│   ├── equipment_data_agent.py       # source validation + temporal features
│   ├── equipment_model_agent.py      # equipment model adapter
│   ├── bosch_prediction_agents.py    # Bosch data + model adapters
│   └── prediction_model_agent.py    # shared planning → training stage
├── crews/
│   ├── prediction_crew.py            # shared Agent 1 → Agent 2 orchestration
│   ├── equipment_prediction_crew.py  # equipment settings and compatibility API
│   └── bosch_prediction_crew.py      # Bosch settings
├── tools/
│   ├── equipment_data.py             # CSV quality gate + temporal features
│   ├── equipment_modeling.py         # grouped CV + hurdle/forest ensemble + diagnostics
│   └── equipment_pipeline.py         # stable public tool facade
├── experiments/
│   ├── equipment_r2_benchmark.py     # raw vs engineered + nine algorithms/parameters
│   ├── stacked_hurdle_benchmark.py   # nested probability-stack experiment
│   ├── cdae_paper_reproduction.py    # authors' wafer-split classification protocol
│   ├── cdae_trace_benchmark.py       # unseen-lot raw-trace CDAE + response strategy
│   ├── bosch_plasma_etch_benchmark.py            # lot-held-out etch metrology
│   ├── bosch_response_then_classify_benchmark.py # response-first screening
│   └── phm2016_cmp_benchmark.py                   # legacy CMP experiment
├── core/
│   ├── data_lanes.py                 # evidence policy and provenance
│   └── equipment_contracts.py        # shared constants and result contracts
├── dashboard/
│   ├── components.py                 # reusable Streamlit components
│   ├── equipment_prediction_page.py  # equipment-response UI page
│   └── bosch_plasma_etch_page.py     # Bosch evidence workspace
├── config.py                         # application paths
└── __init__.py
```

## Dependency direction

```text
Streamlit app
    ↓
dashboard page
    ↓
prediction crew
    ↓
data agent → model agent
    ↓
deterministic tools + core policy
```

Lower layers do not import the dashboard. This keeps data processing testable without starting Streamlit and prevents UI state from changing model behavior.

Both dashboard routes execute `PredictionCrew`: prepare the data once, stop if preparation fails, then pass that same dataset object to the model agent. Both model agents execute `PredictionModelAgent` to plan from the current prepared data, use the bounded balanced fallback when needed, and invoke deterministic training and evidence export. `validate()` runs preparation alone. Source formats, candidate models, targets and validation settings remain in the dataset adapters and modelling functions; sharing a coordinator does not require identical models.

Bosch's model agent passes the same `BoschDataset` and search decision to wafer-average/spatial regression and response-based screening. Screening no longer reloads the source dataset or silently uses a different search profile. The standalone benchmark `run()` functions remain available as deterministic tool entry points. Their prepared-data counterparts are `train_bosch_models()` and `train_bosch_screening()`.

Production model selection remains in `tools/equipment_modeling.py`. It benchmarks the report's nine regression algorithms (Ridge, Lasso, ElasticNet, KNN, Random Forest, Extra Trees, HistGBR, XGBoost and LightGBM). Hyperparameters are selected in inner lot-grouped folds, including tree depth, leaf constraints and cost-complexity pruning where supported; the regression champion is the highest-R² model on outer unseen-lot predictions. Broader research sweeps remain isolated in `experiments/`.

An optional bounded lightweight Groq controller can select a reviewed conservative, balanced, or extensive search profile for EquipmentData, Bosch, and local-upload regression. It is capped at eight agent decisions and 45 seconds. Missing credentials, timeouts, rate limits, invalid output, or disallowed actions automatically select the existing balanced deterministic workflow. The LLM never calculates metrics or approves the champion. The controller calls Groq directly so the Streamlit Cloud Python runtime does not need CrewAI's incompatible ChromaDB dependency chain.

The CDAE study is also experiment-only. It consumes validated `(wafer, sensor, timestamp)` tensors from `tools/equipment_data.py`, then learns imputation, scaling, convolutional embeddings, KMeans-SMOTE and sigmoid calibration inside the active lot fold. `cdae_paper_reproduction.py` reproduces the authors' released causal-convolution and 22×7 decoder architecture under their easier repeated wafer-split protocol; its result is diagnostic only because lots overlap. `cdae_trace_benchmark.py` applies the same architecture under unseen-lot validation and tests the current response-hurdle strategy on top. It supports grouped inner-fold SVM tuning, multi-seed probability averaging, and `both` / `equipment1` / `equipment2` stage ablations so the 1,319 labeled equipment2 wafers are explicitly audited rather than silently excluded by the two-stage intersection. Its optional PyTorch dependencies are listed in `requirements-experiments.txt`; they are not required by the desktop application.

The active second real-data workspace is the Bosch plasma-etch lane.
Its dashboard uses `BoschPredictionCrew`, which runs the same coordinator as
`EquipmentPredictionCrew`. `experiments/bosch_plasma_etch_benchmark.py` decodes the 5 Hz equipment traces,
builds response-free process summaries, and evaluates wafer-average and spatial
etch prediction with complete manufacturing lots held out. The separate
`bosch_response_then_classify_benchmark.py` applies a configurable research
threshold only after continuous silicon-etch prediction. The dashboard exposes
both evidence sets and clearly distinguishes sensor-only wafer-average prediction
from coordinate-assisted spatial virtual metrology. The older PHM16 CMP modules
remain available as legacy experiments but are no longer an application
workspace.

## Compatibility layer

The root files `equipment_data_pipeline.py` and `data_provenance.py` are intentionally small import facades. Existing scripts continue to work, while new development should import from `waferpulse.agents`, `waferpulse.crews`, `waferpulse.tools`, or `waferpulse.core`.

## Adding another analytical workflow

1. Put deterministic calculations in `waferpulse/tools/`.
2. Wrap each single responsibility in an agent under `waferpulse/agents/`.
3. Supply the dataset agents and settings to the shared `PredictionCrew` in `waferpulse/crews/`.
4. Put all Streamlit elements in `waferpulse/dashboard/`.
5. Add its permitted data functions to `waferpulse/core/data_lanes.py`.
6. Test the tool and crew independently.

The older STDF modules remain at repository root as a compatibility boundary. They can be migrated one workflow at a time without risking the working application.

The report-ready explanation and common flowchart are in
[`docs/AGENT_COORDINATED_PREDICTION.md`](docs/AGENT_COORDINATED_PREDICTION.md).
