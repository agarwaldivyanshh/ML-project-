import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RAW_FILE = os.path.join(DATA_DIR, "paysim_raw_sample.parquet")
TABULAR_TRAIN_FILE = os.path.join(DATA_DIR, "tabular_train.parquet")
TABULAR_TEST_FILE = os.path.join(DATA_DIR, "tabular_test.parquet")
GRAPH_DATA_FILE = os.path.join(DATA_DIR, "pyg_graph_data.pt")
SCALER_FILE = os.path.join(DATA_DIR, "scalers.pkl")
DEMO_SCENARIOS_FILE = os.path.join(DATA_DIR, "demo_scenarios.pkl")

def load_or_download_raw():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(RAW_FILE):
        return pd.read_parquet(RAW_FILE)
    
    url = "https://huggingface.co/datasets/flwrlabs/fed-fraud-paysim-banks/resolve/main/data/test-00000-of-00001.parquet"
    print(f"Downloading PaySim benchmark slice from {url}...")
    df = pd.read_parquet(url)
    df.to_parquet(RAW_FILE, index=False)
    return df

def preprocess_tabular(df):
    """
    Extracts strict tabular row features for Random Forest and XGBoost baselines
    matching Slide 4 & Slide 6.
    """
    print("Preprocessing tabular data...")
    # Filter for TRANSFER and CASH_OUT (where 100% of fraud occurs)
    df_filtered = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])].copy()
    
    # Feature Engineering for tabular rows
    df_filtered['type_is_transfer'] = (df_filtered['type'] == 'TRANSFER').astype(int)
    df_filtered['type_is_cashout'] = (df_filtered['type'] == 'CASH_OUT').astype(int)
    
    # Balance discrepancy features (crucial domain signal in financial fraud)
    df_filtered['errorBalanceOrig'] = df_filtered['newbalanceOrig'] + df_filtered['amount'] - df_filtered['oldbalanceOrg']
    df_filtered['errorBalanceDest'] = df_filtered['oldbalanceDest'] + df_filtered['amount'] - df_filtered['newbalanceDest']
    
    # Ratios and behavioral flags
    df_filtered['origExhausted'] = ((df_filtered['oldbalanceOrg'] > 0) & (df_filtered['newbalanceOrig'] == 0)).astype(int)
    df_filtered['destZeroBefore'] = (df_filtered['oldbalanceDest'] == 0).astype(int)
    df_filtered['destZeroAfter'] = (df_filtered['newbalanceDest'] == 0).astype(int)
    
    # Amount relative to balance
    denom = df_filtered['oldbalanceOrg'] + 1.0
    df_filtered['amountRatioOrig'] = (df_filtered['amount'] / denom).clip(upper=100.0)
    
    # Time-based train/test split (Slide 6: "apply a time-based train/test split to avoid future data leakage")
    split_step = df_filtered['step'].quantile(0.80)
    print(f"Time-based split threshold: step <= {split_step:.1f} (Train) vs step > {split_step:.1f} (Test)")
    
    train_df = df_filtered[df_filtered['step'] <= split_step].copy()
    test_df = df_filtered[df_filtered['step'] > split_step].copy()
    
    train_df.to_parquet(TABULAR_TRAIN_FILE, index=False)
    test_df.to_parquet(TABULAR_TEST_FILE, index=False)
    print(f"Tabular saved: Train={len(train_df):,} (Frauds={train_df['isFraud'].sum()}), Test={len(test_df):,} (Frauds={test_df['isFraud'].sum()})")
    
    return train_df, test_df, df_filtered

def build_graph(df_filtered):
    """
    Builds PyG directed graph:
    Nodes: Bank accounts (nameOrig, nameDest)
    Edges: Transfers (nameOrig -> nameDest)
    Node Features: Account degree, flow volume, pass-through ratio
    Target: Fraud status
    """
    print("Building financial transaction graph...")
    
    # Ensure all 821 fraud transactions are preserved along with connected accounts
    fraud_txs = df_filtered[df_filtered['isFraud'] == 1]
    fraud_accounts = set(fraud_txs['nameOrig']).union(set(fraud_txs['nameDest']))
    
    # 1-hop connected legitimate accounts around fraud
    connected_txs = df_filtered[df_filtered['nameOrig'].isin(fraud_accounts) | df_filtered['nameDest'].isin(fraud_accounts)]
    
    # Background normal sample to maintain realistic distribution
    normal_sample = df_filtered[~df_filtered.index.isin(connected_txs.index)].sample(n=min(60000, len(df_filtered)), random_state=42)
    
    graph_df = pd.concat([connected_txs, normal_sample]).drop_duplicates(subset=['step', 'nameOrig', 'nameDest', 'amount']).reset_index(drop=True)
    
    print(f"Graph dataset size: {len(graph_df):,} transactions, {graph_df['isFraud'].sum()} fraud transactions")
    
    # Map accounts to integer node IDs
    unique_accounts = np.unique(np.concatenate([graph_df['nameOrig'].values, graph_df['nameDest'].values]))
    account_to_idx = {acc: i for i, acc in enumerate(unique_accounts)}
    num_nodes = len(unique_accounts)
    print(f"Total Unique Account Nodes: {num_nodes:,}")
    
    # Compute Account-level structural aggregate features (Slide 3 & 5)
    orig_counts = graph_df.groupby('nameOrig')['amount'].agg(['count', 'sum', 'mean']).rename(
        columns={'count': 'out_degree', 'sum': 'total_sent', 'mean': 'avg_sent'})
    dest_counts = graph_df.groupby('nameDest')['amount'].agg(['count', 'sum', 'mean']).rename(
        columns={'count': 'in_degree', 'sum': 'total_received', 'mean': 'avg_received'})
    
    node_feat_df = pd.DataFrame(index=unique_accounts)
    node_feat_df = node_feat_df.join(orig_counts).fillna(0.0)
    node_feat_df = node_feat_df.join(dest_counts).fillna(0.0)
    
    # Structural features
    node_feat_df['net_flow'] = node_feat_df['total_received'] - node_feat_df['total_sent']
    node_feat_df['total_volume'] = node_feat_df['total_received'] + node_feat_df['total_sent']
    node_feat_df['total_degree'] = node_feat_df['in_degree'] + node_feat_df['out_degree']
    
    # Pass-through / Mule score (funds in ~= funds out in high-velocity accounts)
    epsilon = 1.0
    node_feat_df['pass_through_ratio'] = np.minimum(node_feat_df['total_sent'], node_feat_df['total_received']) / (np.maximum(node_feat_df['total_sent'], node_feat_df['total_received']) + epsilon)
    
    # Node features: in_degree, out_degree, total_degree, total_sent, total_received, net_flow, total_volume, avg_sent, avg_received, pass_through_ratio
    feature_cols = [
        'in_degree', 'out_degree', 'total_degree',
        'total_sent', 'total_received', 'net_flow', 'total_volume',
        'avg_sent', 'avg_received', 'pass_through_ratio'
    ]
    
    scaler = StandardScaler()
    X_node = scaler.fit_transform(np.log1p(np.abs(node_feat_df[feature_cols].values)) * np.sign(node_feat_df[feature_cols].values))
    X_node = torch.tensor(X_node, dtype=torch.float32)
    
    # Create directed edges (nameOrig -> nameDest)
    src = [account_to_idx[acc] for acc in graph_df['nameOrig']]
    dst = [account_to_idx[acc] for acc in graph_df['nameDest']]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    # Edge features
    edge_attrs = np.column_stack([
        np.log1p(graph_df['amount'].values),
        graph_df['step'].values,
        (graph_df['type'] == 'TRANSFER').astype(float).values,
        np.log1p(np.abs(graph_df['errorBalanceOrig'].values)) * np.sign(graph_df['errorBalanceOrig'].values),
        np.log1p(np.abs(graph_df['errorBalanceDest'].values)) * np.sign(graph_df['errorBalanceDest'].values),
    ])
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    
    # Node labels: 1 if account was sender or receiver of a fraudulent transfer
    fraud_nodes = set(graph_df[graph_df['isFraud'] == 1]['nameOrig']).union(set(graph_df[graph_df['isFraud'] == 1]['nameDest']))
    node_y = torch.tensor([1 if acc in fraud_nodes else 0 for acc in unique_accounts], dtype=torch.long)
    
    # Edge labels
    edge_y = torch.tensor(graph_df['isFraud'].values, dtype=torch.long)
    
    # Train/Test masks based on time step
    split_step = graph_df['step'].quantile(0.80)
    edge_train_mask = torch.tensor((graph_df['step'] <= split_step).values, dtype=torch.bool)
    edge_test_mask = torch.tensor((graph_df['step'] > split_step).values, dtype=torch.bool)
    
    # Node train/test masks
    train_orig_accs = set(graph_df[graph_df['step'] <= split_step]['nameOrig'])
    node_train_mask = torch.tensor([acc in train_orig_accs for acc in unique_accounts], dtype=torch.bool)
    node_test_mask = ~node_train_mask
    
    # Build PyG Data object
    pyg_data = Data(
        x=X_node,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=node_y,
        edge_y=edge_y,
        train_mask=node_train_mask,
        test_mask=node_test_mask,
        edge_train_mask=edge_train_mask,
        edge_test_mask=edge_test_mask,
        num_nodes=num_nodes
    )
    
    torch.save({
        'data': pyg_data,
        'account_to_idx': account_to_idx,
        'idx_to_account': {i: acc for acc, i in account_to_idx.items()},
        'feature_cols': feature_cols,
        'graph_df': graph_df
    }, GRAPH_DATA_FILE)
    
    with open(SCALER_FILE, 'wb') as f:
        pickle.dump({'scaler': scaler, 'feature_cols': feature_cols}, f)
        
    print(f"Saved PyG graph data to {GRAPH_DATA_FILE}")
    print(f"Node labels: {node_y.sum().item()} fraud nodes out of {num_nodes} accounts ({node_y.sum().item()/num_nodes*100:.2f}%)")
    
    # Create curated demonstration scenarios
    build_curated_scenarios(graph_df, account_to_idx)
    return pyg_data

def build_curated_scenarios(graph_df, account_to_idx):
    """
    Builds distinct interactive demonstration scenarios matching the slides:
    1. Scenario 1: The Multi-Hop Mule Ring (Slide 2 & 3)
       Stolen funds routed through sleeper accounts before cash-out.
    2. Scenario 2: The Single Transfer Illusion (Slide 2)
       A $4,500 transfer that tabular models consider normal, but graph recognizes as 4th hop.
    3. Scenario 3: Smurfing Cluster
       One source dividing payments into 6 mule accounts.
    4. Scenario 4: Legitimate High-Volume Merchant
       Normal commerce with zero structural collusion.
    """
    scenarios = {}
    
    # Find real fraud transfer chains in PaySim
    fraud_transfers = graph_df[(graph_df['isFraud'] == 1) & (graph_df['type'] == 'TRANSFER')]
    fraud_cashouts = graph_df[(graph_df['isFraud'] == 1) & (graph_df['type'] == 'CASH_OUT')]
    
    chains = []
    for _, t in fraud_transfers.head(50).iterrows():
        matches = fraud_cashouts[fraud_cashouts['nameOrig'] == t['nameDest']]
        if len(matches) > 0:
            m = matches.iloc[0]
            chains.append({
                'source': t['nameOrig'],
                'mule': t['nameDest'],
                'cashout': m['nameDest'],
                'transfer_amount': float(t['amount']),
                'cashout_amount': float(m['amount']),
                'transfer_step': int(t['step']),
                'cashout_step': int(m['step'])
            })
            
    scenarios['real_mule_chains'] = chains[:10]
    
    # Curated Synthetic Multi-Hop Ring matching Slide 2 exactly
    scenarios['slide2_multi_hop_ring'] = {
        'name': 'The Multi-Hop Mule Ring Reality (Slide 2)',
        'description': 'Stolen $45,000 broken down and routed through 5 temporary sleeper accounts before cash-out at an ATM. The 4th hop transfer of $4,500 appears completely innocent to tabular models.',
        'target_transfer_amount': 4500.0,
        'target_account': 'MULE_HOP_4',
        'victim': 'VICTIM_CORP_01',
        'mules': ['MULE_HOP_1', 'MULE_HOP_2', 'MULE_HOP_3', 'MULE_HOP_4', 'MULE_HOP_5'],
        'cashout': 'ATM_CASHOUT_77',
        'nodes': [
            {'id': 'VICTIM_CORP_01', 'type': 'Victim Account', 'role': 'Compromised Source', 'balance': 250000.0},
            {'id': 'MULE_HOP_1', 'type': 'Mule Layer 1', 'role': 'Primary Dispersal', 'balance': 500.0},
            {'id': 'MULE_HOP_2', 'type': 'Mule Layer 2', 'role': 'Layering Account', 'balance': 200.0},
            {'id': 'MULE_HOP_3', 'type': 'Mule Layer 3', 'role': 'Layering Account', 'balance': 150.0},
            {'id': 'MULE_HOP_4', 'type': 'Mule Layer 4', 'role': 'Target Transfer ($4,500)', 'balance': 80.0},
            {'id': 'MULE_HOP_5', 'type': 'Mule Layer 5', 'role': 'Final Aggregator', 'balance': 30.0},
            {'id': 'ATM_CASHOUT_77', 'type': 'Cash-out Node', 'role': 'ATM Liquidation', 'balance': 0.0}
        ],
        'edges': [
            {'from': 'VICTIM_CORP_01', 'to': 'MULE_HOP_1', 'amount': 45000.0, 'type': 'TRANSFER', 'step': 142},
            {'from': 'MULE_HOP_1', 'to': 'MULE_HOP_2', 'amount': 15000.0, 'type': 'TRANSFER', 'step': 143},
            {'from': 'MULE_HOP_2', 'to': 'MULE_HOP_3', 'amount': 8500.0, 'type': 'TRANSFER', 'step': 144},
            {'from': 'MULE_HOP_3', 'to': 'MULE_HOP_4', 'amount': 4500.0, 'type': 'TRANSFER', 'step': 145, 'is_target': True},
            {'from': 'MULE_HOP_4', 'to': 'MULE_HOP_5', 'amount': 4450.0, 'type': 'TRANSFER', 'step': 146},
            {'from': 'MULE_HOP_5', 'to': 'ATM_CASHOUT_77', 'amount': 4400.0, 'type': 'CASH_OUT', 'step': 147}
        ]
    }
    
    # Curated Smurfing Fan-Out Scheme
    scenarios['smurfing_fan_out'] = {
        'name': 'Smurfing Fan-Out Scheme',
        'description': 'A single corrupt source account breaks a large illicit sum ($98,000) into 6 sub-threshold micro-transfers across separate mule nodes to evade single-row transaction thresholds.',
        'target_account': 'HUB_SMURF_ALPHA',
        'nodes': [
            {'id': 'HUB_SMURF_ALPHA', 'type': 'Laundering Hub', 'role': 'Smurfing Originator', 'balance': 120000.0},
            {'id': 'MULE_A', 'type': 'Mule Puppet', 'role': 'Smurf Receiver 1', 'balance': 50.0},
            {'id': 'MULE_B', 'type': 'Mule Puppet', 'role': 'Smurf Receiver 2', 'balance': 20.0},
            {'id': 'MULE_C', 'type': 'Mule Puppet', 'role': 'Smurf Receiver 3', 'balance': 100.0},
            {'id': 'MULE_D', 'type': 'Mule Puppet', 'role': 'Smurf Receiver 4', 'balance': 15.0},
            {'id': 'MULE_E', 'type': 'Mule Puppet', 'role': 'Smurf Receiver 5', 'balance': 40.0},
            {'id': 'MULE_F', 'type': 'Mule Puppet', 'role': 'Smurf Receiver 6', 'balance': 75.0},
            {'id': 'OFFSHORE_CONDUIT', 'type': 'Collector', 'role': 'Re-aggregation Sink', 'balance': 0.0}
        ],
        'edges': [
            {'from': 'HUB_SMURF_ALPHA', 'to': 'MULE_A', 'amount': 16000.0, 'type': 'TRANSFER', 'step': 210},
            {'from': 'HUB_SMURF_ALPHA', 'to': 'MULE_B', 'amount': 16500.0, 'type': 'TRANSFER', 'step': 211},
            {'from': 'HUB_SMURF_ALPHA', 'to': 'MULE_C', 'amount': 15800.0, 'type': 'TRANSFER', 'step': 211},
            {'from': 'HUB_SMURF_ALPHA', 'to': 'MULE_D', 'amount': 16200.0, 'type': 'TRANSFER', 'step': 212},
            {'from': 'HUB_SMURF_ALPHA', 'to': 'MULE_E', 'amount': 17000.0, 'type': 'TRANSFER', 'step': 213},
            {'from': 'HUB_SMURF_ALPHA', 'to': 'MULE_F', 'amount': 16500.0, 'type': 'TRANSFER', 'step': 214},
            {'from': 'MULE_A', 'to': 'OFFSHORE_CONDUIT', 'amount': 15900.0, 'type': 'CASH_OUT', 'step': 216},
            {'from': 'MULE_B', 'to': 'OFFSHORE_CONDUIT', 'amount': 16400.0, 'type': 'CASH_OUT', 'step': 216},
            {'from': 'MULE_C', 'to': 'OFFSHORE_CONDUIT', 'amount': 15700.0, 'type': 'CASH_OUT', 'step': 217},
            {'from': 'MULE_D', 'to': 'OFFSHORE_CONDUIT', 'amount': 16100.0, 'type': 'CASH_OUT', 'step': 217},
            {'from': 'MULE_E', 'to': 'OFFSHORE_CONDUIT', 'amount': 16900.0, 'type': 'CASH_OUT', 'step': 218},
            {'from': 'MULE_F', 'to': 'OFFSHORE_CONDUIT', 'amount': 16400.0, 'type': 'CASH_OUT', 'step': 219}
        ]
    }
    
    # Curated Legitimate High-Volume Merchant
    scenarios['legitimate_merchant'] = {
        'name': 'Legitimate High-Volume Enterprise',
        'description': 'A legitimate merchant receiving payments from dozens of unique consumer accounts and paying vetted supplier accounts with genuine business profit margins and steady balance accumulation.',
        'target_account': 'MERCHANT_RETAIL_GLOBAL',
        'nodes': [
            {'id': 'MERCHANT_RETAIL_GLOBAL', 'type': 'Verified Merchant', 'role': 'Commercial Enterprise', 'balance': 480000.0},
            {'id': 'CUSTOMER_1', 'type': 'Consumer', 'role': 'Retail Buyer', 'balance': 5400.0},
            {'id': 'CUSTOMER_2', 'type': 'Consumer', 'role': 'Retail Buyer', 'balance': 8200.0},
            {'id': 'CUSTOMER_3', 'type': 'Consumer', 'role': 'Retail Buyer', 'balance': 3100.0},
            {'id': 'SUPPLIER_LOGISTICS', 'type': 'Vendor', 'role': 'Freight Supplier', 'balance': 190000.0},
            {'id': 'PAYROLL_OPS', 'type': 'Internal Ops', 'role': 'Salary Distribution', 'balance': 95000.0}
        ],
        'edges': [
            {'from': 'CUSTOMER_1', 'to': 'MERCHANT_RETAIL_GLOBAL', 'amount': 420.0, 'type': 'PAYMENT', 'step': 301},
            {'from': 'CUSTOMER_2', 'to': 'MERCHANT_RETAIL_GLOBAL', 'amount': 1850.0, 'type': 'PAYMENT', 'step': 302},
            {'from': 'CUSTOMER_3', 'to': 'MERCHANT_RETAIL_GLOBAL', 'amount': 640.0, 'type': 'PAYMENT', 'step': 303},
            {'from': 'MERCHANT_RETAIL_GLOBAL', 'to': 'SUPPLIER_LOGISTICS', 'amount': 12500.0, 'type': 'TRANSFER', 'step': 305},
            {'from': 'MERCHANT_RETAIL_GLOBAL', 'to': 'PAYROLL_OPS', 'amount': 34000.0, 'type': 'TRANSFER', 'step': 310}
        ]
    }
    
    with open(DEMO_SCENARIOS_FILE, 'wb') as f:
        pickle.dump(scenarios, f)
    print(f"Saved curated demonstration scenarios to {DEMO_SCENARIOS_FILE}")

def main():
    df = load_or_download_raw()
    train_df, test_df, df_filtered = preprocess_tabular(df)
    pyg_data = build_graph(df_filtered)
    print("Data preparation complete!")

if __name__ == "__main__":
    main()
