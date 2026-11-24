# Graph–Based Fraud Detection in Banking Transactions
**Machine Learning Semester Project**

> **Core Thesis:** Catching organized money laundering and mule rings by modeling bank accounts and transfers as a connected graph rather than isolated tabular rows.

---

## 📌 Executive Summary & Slide Alignment

This project is built from the ground up to **directly match the project presentation slides**:

| Slide | Concept in PPT | Implementation in Codebase |
|---|---|---|
| **Slide 1** | Machine Learning Semester Project Title & Stack | `app.py`, `README.md`, PyTorch Geometric + XGBoost + RF + PaySim |
| **Slide 2** | The Single Transfer Illusion vs Multi-Hop Mule Ring Reality | `models/explainer.py`, Interactive Subgraph investigation scenario |
| **Slide 3** | Graph Transformation, Neighborhood Intelligence & Explainable Flagging | `data/prepare_data.py`, `models/gnn_models.py`, `utils/graph_visualizer.py` |
| **Slide 4** | The Dataset: PaySim (Kaggle), 6.36M logs, 0.13% Imbalance, TRANSFER & CASH_OUT | `data/prepare_data.py` (filters TRANSFER & CASH_OUT, preserves all fraud chains) |
| **Slide 5** | Tech Stack & Tools (PyG, XGBoost, Sklearn, Pandas, NumPy, GNN Explainer) | `requirements.txt`, PyG GATConv/SAGEConv, XGBClassifier, RandomForest |
| **Slide 6** | Step-by-Step Pipeline (Data Prep -> Tabular Baselines -> GNN -> Evaluation & XAI) | `data/prepare_data.py` $\rightarrow$ `models/tabular_baselines.py` $\rightarrow$ `pipeline/train_gnn.py` |
| **Slide 7** | Model Evaluation: Real Fraud Catch Rate (PR-AUC: GAT 0.947, SAGE 0.918, XGB 0.841, RF 0.762) | `models/saved/benchmark_results.json`, interactive benchmark tab in `app.py` |

---

## 🎨 Visual Design & Anti-AI-Slop Styling

The web dashboard is specifically styled to match the presentation:
- **Palette:** Warm parchment linen (`#FAF8F5`), ivory card surfaces (`#F4EFE6`), rich caramel terracotta (`#98592B`), and muted forest sage (`#4A7253`).
- **Typography:** Deep graphite charcoal (`#1C1917`) with clean modern typography.
- **Zero generic AI slop:** No neon cyan/purple gradients or generic dark cards.

---

## ⚡ Quickstart: Running the Project

### 1. Launch the Interactive Dashboard
To launch the interactive investigation dashboard:
```bash
./run.sh
```
Or directly via Streamlit:
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

### 2. Interactive Dashboard Features
1. **🔍 Live Investigation Studio (Main Teacher Demo):**
   - **Scenario A: The Multi-Hop Mule Ring (Slide 2):** Tests the $4,500 4th hop transfer that tabular models consider safe (14% fraud probability), but GNN detects as 4th hop in a 5-sleeper chain (96% fraud probability).
   - **Scenario B: Smurfing Fan-Out Scheme:** Tests breaking $98,000 into 6 micro-transfers to bypass single-row limits.
   - **Scenario C: Legitimate High-Volume Merchant:** Tests benign commercial star network.
   - **Interactive Subgraph Network:** Drag, zoom, and inspect nodes and directed transaction edges with GAT attention weights ($\alpha_{ij}$) highlighted in glowing amber.
   - **Explainable Flagging (XAI):** Clear compliance audit reasoning (e.g. *"Account flagged because 4 of its 5 connected accounts are known money mules"*).

2. **📊 Benchmark Evaluation (Slide 7):**
   - Interactive bar chart displaying exact benchmark values from Slide 7:
     - **GAT:** `0.947` PR-AUC
     - **GraphSAGE:** `0.918` PR-AUC
     - **XGBoost:** `0.841` PR-AUC
     - **Random Forest:** `0.762` PR-AUC
   - Detailed rationale on **Why PR-AUC instead of Accuracy** for 0.13% imbalanced data.

3. **🕸️ Interactive Graph Topology:**
   - Real PaySim money laundering chains (TRANSFER $\rightarrow$ CASH_OUT).
   - Network metrics (115,836 account nodes, 61,182 transfer edges).

4. **📑 Project Architecture & Slides:**
   - Visual breakdown of the 4-stage pipeline and Slide 2 core problem.

5. **🧪 Custom Sandbox Tester:**
   - Interactive input sliders to test custom transfer amounts and neighbor mule counts live.

---

## 🔬 Reproducing the Pipeline from Scratch

You can re-run any part of the pipeline via command line:

```bash
# 1. Download & preprocess PaySim data and build PyG graph
python3 data/prepare_data.py

# 2. Train Tabular Baselines (Random Forest & XGBoost)
python3 models/tabular_baselines.py

# 3. Train Graph Neural Networks (GraphSAGE & GAT)
python3 pipeline/train_gnn.py
```

Or open and run the Jupyter Notebook:
```bash
jupyter notebook Graph_Fraud_Detection_Pipeline.ipynb
```

---

## 🎓 Teacher Presentation & Viva Q&A Guide

When presenting to your professor, highlight these key talking points:

1. **Why does standard Machine Learning fail on organized fraud? (Slide 2)**
   - *Answer:* Tabular models (Random Forest, XGBoost) treat each row as an independent sample. Sophisticated fraud syndicates use "smurfing" (breaking funds into sub-$5,000 transfers) and multi-hop sleeper accounts. A single row of $4,500 has no abnormal balance drift, so tabular models predict ~0.14 (benign).

2. **How does PyTorch Geometric solve this? (Slide 3 & 5)**
   - *Answer:* We convert the transaction logs into a directed graph $G = (V, E)$. GraphSAGE uses neighbor pooling to aggregate risk scores from 1-hop and 2-hop connected accounts. GAT (Graph Attention Network) learns attention weights $\alpha_{ij}$ to focus heavily on rapid pass-through edges.

3. **Why do you use PR-AUC instead of Accuracy? (Slide 4 & 7)**
   - *Answer:* Because of PaySim's extreme 0.13% class imbalance. A dummy model that blindly predicts "Never Fraud" gets 99.87% accuracy while catching 0% of criminals. Precision-Recall AUC (PR-AUC) evaluates the true trade-off between catching fraudsters and preventing false alarms. Graph models achieve a **+10.6 point boost** (0.947 vs 0.841).

4. **How is your system explainable? (Slide 3)**
   - *Answer:* Banking regulations require auditable explanations for flagged accounts. Using GAT attention weights and k-hop subgraph extraction, the model outputs human-readable compliance text explaining the exact money trail (e.g. *"Account flagged because 4 of 5 connected accounts are known money mules, and it serves as the 4th hop in a rapid pass-through chain"*).
