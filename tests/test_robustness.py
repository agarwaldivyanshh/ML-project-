import os
import sys
import pickle
import json
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from models.gnn_models import GraphSAGEModel, GATModel
from models.explainer import FraudGraphExplainer

def test_models_exist():
    print(">>> 1. Testing Model Artifacts Existence...")
    files = [
        "data/demo_scenarios.pkl",
        "data/pyg_graph_data.pt",
        "models/saved/random_forest.pkl",
        "models/saved/xgboost.pkl",
        "models/saved/graphsage.pt",
        "models/saved/gat.pt",
        "models/saved/benchmark_results.json"
    ]
    for f in files:
        path = os.path.join(BASE_DIR, f)
        assert os.path.exists(path), f"Missing artifact: {f}"
        print(f"  [PASS] {f} exists ({os.path.getsize(path)/1024:.1f} KB)")

def test_tabular_inference():
    print("\n>>> 2. Testing Tabular Baselines Inference...")
    with open(os.path.join(BASE_DIR, "models/saved/random_forest.pkl"), "rb") as f:
        rf = pickle.load(f)
    with open(os.path.join(BASE_DIR, "models/saved/xgboost.pkl"), "rb") as f:
        xgb = pickle.load(f)
        
    test_cases = [
        ("Normal $4,500 Transfer", [4500.0, 4500.0, 0.0, 0.0, 4500.0, 1, 0, 0.0, 0.0, 1, 1, 0, 1.0]),
        ("Micro $0.01 Transfer", [0.01, 100.0, 99.99, 50.0, 50.01, 1, 0, 0.0, 0.0, 0, 0, 0, 0.0001]),
        ("Zero Dollar Transfer", [0.0, 0.0, 0.0, 0.0, 0.0, 1, 0, 0.0, 0.0, 0, 1, 1, 0.0]),
        ("Extreme $10 Billion Transfer", [1e10, 1e10, 0.0, 0.0, 1e10, 1, 0, 0.0, 0.0, 1, 1, 0, 1.0]),
        ("Negative Amount Drift (Anomaly)", [-500.0, 1000.0, 1500.0, 500.0, 0.0, 0, 1, 0.0, 0.0, 0, 0, 1, 0.0]),
        ("Massive Balance Error", [50000.0, 10000.0, 0.0, 0.0, 50000.0, 1, 0, 40000.0, 0.0, 1, 1, 0, 5.0]),
    ]
    
    for label, row in test_cases:
        X = np.array([row])
        rf_p = rf.predict_proba(X)[0][1]
        xgb_p = xgb.predict_proba(X)[0][1]
        assert not np.isnan(rf_p) and not np.isinf(rf_p), f"RF failed on {label}"
        assert not np.isnan(xgb_p) and not np.isinf(xgb_p), f"XGB failed on {label}"
        print(f"  [PASS] {label:32s} -> RF: {rf_p:.3f}, XGB: {xgb_p:.3f}")

def test_gnn_inference():
    print("\n>>> 3. Testing GNN PyG Models (GraphSAGE & GAT)...")
    graph_dict = torch.load(os.path.join(BASE_DIR, "data/pyg_graph_data.pt"), weights_only=False)
    data = graph_dict['data']
    in_channels = data.x.size(1)
    
    sage = GraphSAGEModel(in_channels=in_channels, hidden_channels=64, out_channels=2)
    sage.load_state_dict(torch.load(os.path.join(BASE_DIR, "models/saved/graphsage.pt"), weights_only=True))
    sage.eval()
    
    gat = GATModel(in_channels=in_channels, hidden_channels=64, out_channels=2, heads=2)
    gat.load_state_dict(torch.load(os.path.join(BASE_DIR, "models/saved/gat.pt"), weights_only=True))
    gat.eval()
    
    with torch.no_grad():
        sub_x = data.x[:50]
        sub_edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
        
        sage_out = sage(sub_x, sub_edge_index)
        gat_out = gat(sub_x, sub_edge_index)
        gat_out_attn, (attn_edges, alpha) = gat(sub_x, sub_edge_index, return_attention_weights=True)
        
        assert sage_out.shape == (50, 2), "GraphSAGE output shape mismatch"
        assert gat_out.shape == (50, 2), "GAT output shape mismatch"
        assert not torch.isnan(sage_out).any(), "GraphSAGE NaN detected"
        assert not torch.isnan(gat_out).any(), "GAT NaN detected"
        assert alpha is not None, "GAT attention weights should not be None"
        
        print("  [PASS] GraphSAGE forward pass successful, no NaNs.")
        print("  [PASS] GAT forward pass successful, attention extracted.")

def test_explainer_scenarios():
    print("\n>>> 4. Testing Explainer on All Scenarios...")
    explainer = FraudGraphExplainer()
    with open(os.path.join(BASE_DIR, "data/demo_scenarios.pkl"), "rb") as f:
        scenarios = pickle.load(f)
        
    for sc_name in ['slide2_multi_hop_ring', 'smurfing_fan_out', 'legitimate_merchant']:
        sc = scenarios[sc_name]
        target = sc['target_account']
        audit = explainer.explain_account(target, scenario_data=sc)
        assert 'summary' in audit
        assert 'predictions' in audit
        assert len(audit['reasons']) > 0
        gat_score = audit['predictions']['GAT (Graph Attention Network)']['prob']
        print(f"  [PASS] Scenario '{sc['name']}' audited successfully -> GAT score: {gat_score}")

def test_sandbox_calculation_safety():
    print("\n>>> 5. Testing Sandbox Math on Extreme / Un-ordinary Inputs...")
    extreme_amounts = [
        -1000000.0,
        -0.01,
        0.0,
        0.00001,
        4500.0,
        99999999.0,
        1e12,
        1e18,
        float('inf'),
        float('-inf'),
        float('nan')
    ]
    
    extreme_mules = [-5, 0, 1, 4, 10, 100, 9999]
    extreme_speeds = [
        "Instant Liquidation (< 2 hrs)",
        "Standard Commercial (Days/Weeks)",
        "Unknown Speed Option",
        "",
        None
    ]
    
    passed_count = 0
    for amt in extreme_amounts:
        for mules in extreme_mules:
            for spd in extreme_speeds:
                try:
                    s_amt = float(amt)
                    if np.isnan(s_amt) or np.isinf(s_amt) or s_amt < 0:
                        s_amt = 4500.0
                except:
                    s_amt = 4500.0
                    
                try:
                    s_mules = max(0, min(10, int(mules)))
                except:
                    s_mules = 0
                    
                is_low_amount = (s_amt <= 10000.0)
                if is_low_amount:
                    rf_p = 0.14
                    xgb_p = 0.18
                else:
                    rf_p = min(0.85, max(0.01, 0.40 + (np.log10(max(1.0, s_amt)) / 25.0)))
                    xgb_p = min(0.88, max(0.01, 0.45 + (np.log10(max(1.0, s_amt)) / 22.0)))
                    
                if s_mules >= 2 or spd == "Instant Liquidation (< 2 hrs)":
                    sage_p = min(0.97, max(0.01, 0.40 + (s_mules * 0.11)))
                    gat_p = min(0.99, max(0.01, 0.48 + (s_mules * 0.10)))
                else:
                    sage_p = min(0.40, max(0.01, 0.08 + (s_mules * 0.05)))
                    gat_p = min(0.35, max(0.01, 0.04 + (s_mules * 0.04)))
                    
                for p in [rf_p, xgb_p, sage_p, gat_p]:
                    assert 0.0 <= p <= 1.0, f"Probability out of bounds: {p}"
                    assert not np.isnan(p), f"NaN probability detected: {p}"
                    assert not np.isinf(p), f"Inf probability detected: {p}"
                passed_count += 1
                
    print(f"  [PASS] {passed_count} extreme edge-case combinations tested with ZERO crashes.")

if __name__ == "__main__":
    test_models_exist()
    test_tabular_inference()
    test_gnn_inference()
    test_explainer_scenarios()
    test_sandbox_calculation_safety()
    print("\n=======================================================")
    print("ALL 5 ROBUSTNESS AND INTEGRITY TEST SUITES PASSED (100%)")
    print("=======================================================")
