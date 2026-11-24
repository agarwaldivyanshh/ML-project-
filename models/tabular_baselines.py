import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "saved")
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_COLS = [
    'amount',
    'oldbalanceOrg',
    'newbalanceOrig',
    'oldbalanceDest',
    'newbalanceDest',
    'type_is_transfer',
    'type_is_cashout',
    'errorBalanceOrig',
    'errorBalanceDest',
    'origExhausted',
    'destZeroBefore',
    'destZeroAfter',
    'amountRatioOrig'
]

def load_tabular_data():
    train_df = pd.read_parquet(os.path.join(DATA_DIR, "tabular_train.parquet"))
    test_df = pd.read_parquet(os.path.join(DATA_DIR, "tabular_test.parquet"))
    
    X_train = train_df[FEATURE_COLS].fillna(0).values
    y_train = train_df['isFraud'].values
    
    X_test = test_df[FEATURE_COLS].fillna(0).values
    y_test = test_df['isFraud'].values
    
    return X_train, y_train, X_test, y_test, train_df, test_df

def compute_metrics(y_true, y_prob, threshold=0.5):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred).tolist()
    
    return {
        'pr_auc': float(pr_auc),
        'roc_auc': float(roc_auc),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'confusion_matrix': cm,
        'precision_curve': precision.tolist()[:100],  # sample points for plotting
        'recall_curve': recall.tolist()[:100]
    }

def train_tabular_models():
    print("Loading tabular train & test sets...")
    X_train, y_train, X_test, y_test, _, _ = load_tabular_data()
    print(f"X_train shape: {X_train.shape}, Frauds: {y_train.sum()} ({y_train.sum()/len(y_train)*100:.3f}%)")
    print(f"X_test shape: {X_test.shape}, Frauds: {y_test.sum()} ({y_test.sum()/len(y_test)*100:.3f}%)")
    
    # 1. Random Forest (Tabular Control - Slide 5 & 7)
    print("\n--- Training Random Forest (Tabular Control) ---")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    rf_metrics = compute_metrics(y_test, rf_probs)
    print(f"Random Forest Results -> PR-AUC: {rf_metrics['pr_auc']:.4f}, ROC-AUC: {rf_metrics['roc_auc']:.4f}, Recall: {rf_metrics['recall']:.4f}, F1: {rf_metrics['f1']:.4f}")
    
    # 2. XGBoost (Tabular Baseline - Slide 5 & 7)
    print("\n--- Training XGBoost (Tabular Baseline) ---")
    pos_weight = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-5)
    xgb = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.08,
        scale_pos_weight=pos_weight * 0.5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='aucpr',
        random_state=42
    )
    xgb.fit(X_train, y_train)
    xgb_probs = xgb.predict_proba(X_test)[:, 1]
    xgb_metrics = compute_metrics(y_test, xgb_probs)
    print(f"XGBoost Results -> PR-AUC: {xgb_metrics['pr_auc']:.4f}, ROC-AUC: {xgb_metrics['roc_auc']:.4f}, Recall: {xgb_metrics['recall']:.4f}, F1: {xgb_metrics['f1']:.4f}")
    
    # Save models and baseline evaluation
    with open(os.path.join(MODELS_DIR, "random_forest.pkl"), "wb") as f:
        pickle.dump(rf, f)
    with open(os.path.join(MODELS_DIR, "xgboost.pkl"), "wb") as f:
        pickle.dump(xgb, f)
        
    tabular_results = {
        'random_forest': rf_metrics,
        'xgboost': xgb_metrics,
        'feature_cols': FEATURE_COLS,
        'feature_importances_rf': dict(zip(FEATURE_COLS, rf.feature_importances_.tolist())),
        'feature_importances_xgb': dict(zip(FEATURE_COLS, xgb.feature_importances_.tolist()))
    }
    
    with open(os.path.join(MODELS_DIR, "tabular_results.pkl"), "wb") as f:
        pickle.dump(tabular_results, f)
        
    print(f"\nSaved models and tabular results to {MODELS_DIR}")
    return tabular_results

if __name__ == "__main__":
    train_tabular_models()
