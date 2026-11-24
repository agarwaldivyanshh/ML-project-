#!/usr/bin/env bash
# Startup script for Graph-Based Fraud Detection Web Dashboard

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1

echo "=========================================================="
echo "  Graph-Based Fraud Detection in Banking Transactions"
echo "  Machine Learning Semester Project"
echo "=========================================================="

# Check if data exists, if not generate it
if [ ! -f "data/pyg_graph_data.pt" ]; then
    echo "[+] Preparing graph data and baseline splits..."
    python3 data/prepare_data.py
fi

# Check if models exist
if [ ! -f "models/saved/gat.pt" ]; then
    echo "[+] Training tabular baselines..."
    python3 models/tabular_baselines.py
    echo "[+] Training GraphSAGE and GAT models..."
    python3 pipeline/train_gnn.py
fi

echo "[+] Starting aesthetic Streamlit Dashboard..."
streamlit run app.py --server.port 8501
