# WaferPulse Code Architecture

The active real-data workflow follows the same clean separation used by HydroSight: agents perform one responsibility, a crew controls execution order, tools contain deterministic implementation, and the dashboard only renders state.

```text
waferpulse/
├── agents/
│   ├── equipment_data_agent.py       # source validation + temporal features
│   └── equipment_model_agent.py      # grouped validation + model evidence
├── crews/
│   └── equipment_prediction_crew.py  # Agent 1 → Agent 2 orchestration
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

The production model selection remains in `tools/equipment_modeling.py`. The larger nine-algorithm and parameter sweep is deliberately isolated in `experiments/` so optional CatBoost, XGBoost and LightGBM dependencies do not leak into the desktop runtime.

The CDAE study is also experiment-only. It consumes validated `(wafer, sensor, timestamp)` tensors from `tools/equipment_data.py`, then learns imputation, scaling, convolutional embeddings, KMeans-SMOTE and sigmoid calibration inside the active lot fold. `cdae_paper_reproduction.py` reproduces the authors' released causal-convolution and 22×7 decoder architecture under their easier repeated wafer-split protocol; its result is diagnostic only because lots overlap. `cdae_trace_benchmark.py` applies the same architecture under unseen-lot validation and tests the current response-hurdle strategy on top. It supports grouped inner-fold SVM tuning, multi-seed probability averaging, and `both` / `equipment1` / `equipment2` stage ablations so the 1,319 labeled equipment2 wafers are explicitly audited rather than silently excluded by the two-stage intersection. Its optional PyTorch dependencies are listed in `requirements-experiments.txt`; they are not required by the desktop application.

The active second real-data workspace is the Bosch plasma-etch lane.
`experiments/bosch_plasma_etch_benchmark.py` decodes the 5 Hz equipment traces,
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
3. Define execution order in `waferpulse/crews/`.
4. Put all Streamlit elements in `waferpulse/dashboard/`.
5. Add its permitted data functions to `waferpulse/core/data_lanes.py`.
6. Test the tool and crew independently.

The older STDF modules remain at repository root as a compatibility boundary. They can be migrated one workflow at a time without risking the working application.
