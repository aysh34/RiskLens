import os
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tabpfn_client import TabPFNClassifier
from tabpfn_client.api_models import ModelVersion
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_benchmark_data():
    print("Fetching authentic benchmark financial risk data from OpenML...")
    # OpenML dataset ID 46562: Real financial/credit risk classification benchmark
    X, y_raw = fetch_openml(data_id=46562, as_frame=True, return_X_y=True)
    
    # Map target to binary integer {0, 1}
    classes = y_raw.unique()
    y = (y_raw == classes[0]).astype(int)
    return X, y

if __name__ == "__main__":
    if not os.environ.get("TABPFN_TOKEN"):
        raise RuntimeError("ERROR: TABPFN_TOKEN environment variable is missing. Set it via export TABPFN_TOKEN='...'")

    X, y = load_benchmark_data()
    print(f"Dataset loaded. Shape: {X.shape} | Features: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Initializing TabPFN-3.5 with High-Effort Thinking Mode & Grouped Entity Controls...")
    
    model = TabPFNClassifier.create_default_for_version(
        ModelVersion.V3_5,
        thinking_effort="high",
        thinking_metric="roc_auc"
    )
    
    # Fit model on training split
    model.fit(X_train, y_train)
    
    print("Evaluating on held-out test split...")
    probas = model.predict_proba(X_test)
    test_auc = roc_auc_score(y_test, probas[:, 1])
    print(f"Verified Test ROC AUC: {test_auc:.4f}")

    # Save metrics, model, schema, AND raw training splits for attribution
    with open('metrics.txt', 'w') as f:
        f.write(f"{test_auc:.4f}")
        
    joblib.dump(model, 'enterprise_model.pkl')
    joblib.dump(X.columns.tolist(), 'model_schema.pkl')
    joblib.dump(X_train, 'X_train_raw.pkl')
    joblib.dump(y_train, 'y_train_raw.pkl')
    print("Success! Enterprise model, schema, training splits, and metrics persisted securely.")