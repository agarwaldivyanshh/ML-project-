# Graph-Based Fraud Detection in Banking Transactions

A machine learning and network analytics project that detects organized money laundering and multi-hop mule rings in banking transaction logs using Graph Neural Networks (GraphSAGE and GAT) compared against tabular baselines (XGBoost and Random Forest).

## Overview

Conventional row-level classifiers evaluate each bank transfer in isolation. Coordinated fraud rings exploit this by splitting stolen funds across multi-hop "sleeper" mule accounts or structuring transfers below reporting thresholds. By modeling accounts as nodes and directed fund transfers as edges, Graph Neural Networks aggregate neighborhood risk across 1-hop and 2-hop connections to uncover laundering topologies.

### Key Components
- **Dataset:** PaySim mobile money simulator (6.36M transactions, 0.13% fraud prevalence), filtered to `TRANSFER` and `CASH_OUT` operations.
- **Graph Construction:** 115,836 account nodes and 61,182 directed transfer edges engineered with balance drift, velocity, and pass-through ratio features (`data/prepare_data.py`).
- **Models:**
  - **GAT (Graph Attention Network):** Multi-head attention over neighbor transfers to weight high-risk laundering conduits.
  - **GraphSAGE:** Inductive neighbor sampling and aggregation across 2-hop neighborhoods.
  - **XGBoost & Random Forest:** Tabular baselines trained on isolated transaction row features.
- **Interactive Dashboard:** Built with Streamlit and Apache ECharts (`app.py`) for subgraph inspection, benchmark comparison, and custom transaction stress-testing.

## Benchmark Results (PaySim Test Split)

| Model | PR-AUC | ROC-AUC | Precision | Recall |
|---|---:|---:|---:|---:|
| **GAT (Graph Attention Network)** | **0.947** | **0.988** | **0.932** | **0.951** |
| **GraphSAGE (Neighbor Sampling)** | 0.918 | 0.976 | 0.904 | 0.928 |
| **XGBoost (Tabular Baseline)** | 0.841 | 0.945 | 0.852 | 0.830 |
| **Random Forest (Tabular Control)** | 0.762 | 0.912 | 0.781 | 0.745 |

*PR-AUC is used as the primary metric due to severe class imbalance (0.13% fraud), where accuracy is misleading.*

## Running Locally

### 1. Launch the Web App
```bash
pip install -r requirements.txt
streamlit run app.py
```
Then open `http://localhost:8501` in your browser (or run `./run.sh` on macOS).

### 2. Re-run Training & Tests (Optional)
```bash
pip install -r requirements-train.txt
python3 data/prepare_data.py
python3 models/tabular_baselines.py
python3 pipeline/train_gnn.py
pytest tests/
```

## Repository Structure
```text
├── app.py                     # Streamlit dashboard application
├── data/
│   ├── prepare_data.py        # PaySim preprocessing & PyG graph builder
│   └── demo_scenarios.pkl     # Pre-extracted investigation subgraphs
├── models/
│   ├── gnn_models.py          # GraphSAGE, GAT, and FocalLoss definitions
│   ├── tabular_baselines.py   # Random Forest & XGBoost training
│   ├── explainer.py           # Subgraph explanation & risk summary generator
│   └── saved/                 # Trained checkpoints & benchmark metrics
├── pipeline/
│   └── train_gnn.py           # PyTorch Geometric training & evaluation script
├── utils/
│   ├── echarts_component.py   # ECharts iframe renderer with zoom controls
│   ├── echarts_visualizer.py  # Transaction network graph layouts
│   └── echarts_benchmarks.py  # Benchmark & temporal chart configs
└── tests/
    └── test_robustness.py     # Model & edge-case verification suite
```
